import shutil
import torch
import os
import urllib.request
import tarfile
import datetime
import time
import pickle
import optuna
import argparse
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.transforms as T
from scipy.linalg import sqrtm
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import save_image, make_grid
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from transformers import pipeline #huggingface
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def download(data_dir):
    # URL of CIFAR-100 dataset
    data_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    data_filename = os.path.join(data_dir, 'cifar-100-python.tar.gz')
    # Download the file if it doesn't exist
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        urllib.request.urlretrieve(data_url, data_filename)
        print("Download complete.")
    else:
        print(f"File already exists: {data_filename}")
    return data_filename

def extract(data_filename, data_dir):
    # Extract CIFAR-100 archive
    extracted_dir = os.path.join(data_dir, 'cifar-100-python')
    if not os.path.exists(extracted_dir):
        print(f"Extracting {data_filename}...")
        with tarfile.open(data_filename, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")
    else:
        print(f"Archive already extracted: {extracted_dir}")
    return extracted_dir


# Python 3 routine to unpickle CIFAR-100 batch files
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def split_cifar100_trainval(train_dict, val_ratio=0.25):
    labels = train_dict[b'fine_labels']
    num_classes = 100
    images_per_class = 500
    train_indices, val_indices = [], []
    for cls in range(num_classes):
        cls_indices = np.where(np.array(labels) == cls)[0]
        np.random.shuffle(cls_indices)
        n_val = int(images_per_class * val_ratio)  # 25% validation
        n_train = images_per_class - n_val           # 75% train
        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train:n_train+n_val])
    return train_indices, val_indices


# Preprocess a subset of CIFAR-100 images on demand
def preprocess_cifar100_subset(data_dict, indices):
    """
    Preprocesses images from CIFAR-100 using a list of indices.
    Converts images to float32 and normalizes to [-1, 1].
    Returns a numpy array of shape (len(indices), 32, 32, 3) and labels.

    The first 1024 bytes are the red channel values, the next 1024 the green, 
    and the final 1024 the blue. The values are stored in row-major order, 
    so the first 32 bytes are the red channel values of the first row of the image
    """
    raw_data = data_dict[b'data']
    raw_labels = data_dict[b'fine_labels']
    images = []
    labels = []
    for idx in indices:
        img_flat = raw_data[idx]
        # CIFAR-100: 32x32x3, row-major order
        r = img_flat[0:1024].reshape(32, 32)
        g = img_flat[1024:2048].reshape(32, 32)
        b = img_flat[2048:3072].reshape(32, 32)
        img = np.stack([r, g, b], axis=-1)
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
        images.append(img)
        labels.append(raw_labels[idx])
    images = np.stack(images)
    labels = np.array(labels)
    return images, labels


# DCGAN Generator
# Weights initialization function
def weights_init(m):
	'''
	 authors specify that all model weights shall be randomly initialized from a Normal 
	 distribution with mean=0, stdev=0.02. The weights_init function takes an initialized 
	 model as input and reinitializes all convolutional, convolutional-transpose, and 
	 batch normalization layers to meet this criteria. This function is applied to the 
	 models immediately after initialization.
	'''
	# Ref: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


# Helper to create experiment folder with timestamp
def create_experiment_folder(base_dir= 'experiments'):
	os.makedirs(base_dir, exist_ok=True)
	start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	exp_folder = os.path.join(base_dir, f"experiment_{start_time}")
	os.makedirs(exp_folder, exist_ok=True)
	return exp_folder


# Helper to save generated images
def save_generated_images(generator, epoch, exp_folder, nz=100, device='cpu', num_images=64):
	# Set generator to evaluation mode (affects BatchNorm/Dropout)
	generator.eval()
	with torch.no_grad():
		# Generate random noise vectors
		noise = torch.randn(num_images, nz, 1, 1, device=device)
		# Generate fake images from noise
		fake_images = generator(noise)
		# Denormalize images from [-1,1] to [0,1] for visualization
		fake_images = (fake_images + 1) / 2
		# Arrange images into a grid for easy viewing
		grid = make_grid(fake_images, nrow=8, normalize=False)
		# Build the save path for this epoch
		save_path = os.path.join(exp_folder, f"generated_epoch_{epoch}.png")
		# Save the grid image to disk
		save_image(grid, save_path)
	# Set generator back to training mode
	generator.train()
	return save_path


# Helper to compute Fréchet Inception Distance (FID) between generated and real images
def compute_fid_score(generator, real_images, nz=100, device='cpu', num_images=1000, batch_size=64):
	'''
	 Load pretrained Inception v3 model for feature extraction (used in FID)

	 NOTE: Error: Generator loss is a very poor metric for model selection in GANs. 
	 It is highly volatile and a low loss can simply mean the discriminator has become weak, 
	 not that the generator is creating high-quality images.
	'''
	inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False, aux_logits=True).to(device)
	inception.eval()
	resize = T.Resize((299, 299))  # Inception expects 299x299 images
	# Standard ImageNet normalization for Inception V3
	imagenet_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	# Hook to extract pool3 features
	pool3_feats = []
	def hook_fn(module, input, output):
		pool3_feats.append(output.detach())
	# The correct layer to hook is AvgPool2d, not avgpool
	handle = inception.avgpool.register_forward_hook(hook_fn)

	def get_activations(images):
		activations = []
		with torch.no_grad():
			for i in range(0, images.size(0), batch_size):
				batch = images[i:i+batch_size]
				# Ensure batch is float32 and 3-channel
				if batch.dtype != torch.float32:
					batch = batch.float()
				if batch.shape[1] != 3:
					if batch.shape[-1] == 3:
						batch = batch.permute(0, 3, 1, 2)
					else:
						raise ValueError(f"Input batch must have 3 channels, got shape {batch.shape}")
				# Correctly denormalize from [-1, 1] to [0, 1] before resizing and normalizing for ImageNet
				batch = (batch + 1) / 2.0
				batch = resize(batch)  # Resize to 299x299
				# Apply ImageNet normalization
				batch = imagenet_norm(batch)
				batch = batch.to(device)
				pool3_feats.clear()
				
				# If inception is DataParallel, use .module for hooks to work
				model_for_hook = inception.module if hasattr(inception, 'module') else inception
				_ = model_for_hook(batch)
				
				if not pool3_feats:
					print(f"[DEBUG] Inception input batch shape: {batch.shape}, dtype: {batch.dtype}")
					raise RuntimeError("Inception forward hook did not run. Check input shape and model setup.")
				
				# pool3_feats[0] shape: (batch, 2048, 1, 1)
				feats = pool3_feats[0].squeeze(-1).squeeze(-1).cpu().numpy()  # (batch, 2048)
				activations.append(feats)
		# Concatenate all activations into a single array
		return np.concatenate(activations, axis=0)

	try:
		# Generate fake images from the generator
		generator.eval()
		fake_images = []
		with torch.no_grad():
			for _ in range((num_images + batch_size - 1) // batch_size):
				noise = torch.randn(batch_size, nz, 1, 1, device=device)
				fake = generator(noise)
				# Fake images are in [-1, 1], will be denormalized in get_activations
				fake_images.append(fake)
		# Stack all generated images and trim to num_images
		fake_images = torch.cat(fake_images, dim=0)[:num_images]

		# Prepare real images (convert from numpy if needed)
		if isinstance(real_images, np.ndarray):
			real_images = torch.tensor(real_images, dtype=torch.float32).permute(0, 3, 1, 2)
		# Real images are in [-1, 1], will be denormalized in get_activations
		real_images = real_images[:num_images]

		# Get activations for fake and real images
		act_fake = get_activations(fake_images)
		act_real = get_activations(real_images)

		# Compute mean and covariance for both sets
		mu_fake, sigma_fake = act_fake.mean(axis=0), np.cov(act_fake, rowvar=False)
		mu_real, sigma_real = act_real.mean(axis=0), np.cov(act_real, rowvar=False)

		# FID formula: ||mu1-mu2||^2 + Tr(sigma1+sigma2-2*sqrt(sigma1*sigma2))
		diff = mu_fake - mu_real
		covmean, _ = sqrtm(sigma_fake.dot(sigma_real), disp=False)
		if np.iscomplexobj(covmean):
			covmean = covmean.real
		fid = diff.dot(diff) + np.trace(sigma_fake + sigma_real - 2 * covmean)
	finally:
		handle.remove()

	generator.train()
	return float(fid)


# DCGAN Generator for 32x32x3 images
class DCGANGenerator(nn.Module):
	"""
	Deep Convolutional GAN (DCGAN) Generator network.

	This generator transforms a latent noise vector into a synthetic image using
	a series of transposed convolutions (deconvolutions). The architecture follows
	the DCGAN paper guidelines with batch normalization and ReLU activations.

	The network progressively upsamples the input from a 1D latent vector to a
	full-resolution image through the following stages:
		- Input: (nz, 1, 1) latent vector
		- Stage 1: (ngf*8, 4, 4)
		- Stage 2: (ngf*4, 8, 8)
		- Stage 3: (ngf*2, 16, 16)
		- Stage 4: (ngf, 32, 32)
		- Output: (nc, 32, 32) image with pixel values in [-1, 1]

	Args:
		nz (int, optional): Size of the latent vector (noise dimension). Default: 100
		ngf (int, optional): Base number of generator filters. Controls network capacity.
							Deeper layers use multiples of this value. Default: 64
		nc (int, optional): Number of output channels (e.g., 3 for RGB images). Default: 3

	Attributes:
		main (nn.Sequential): The sequential container holding all generator layers.

	Reference:
		https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
		Based on: "Unsupervised Representation Learning with Deep Convolutional 
				  Generative Adversarial Networks" (Radford et al., 2015)
	"""
	# Ref: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
	def __init__(self, nz=100, ngf=64, nc=3):
		super().__init__()  # Initialize the parent nn.Module
		# Define the generator network as a sequence of layers
		self.main = nn.Sequential(
			# Input: latent vector (nz) -> ngf*8 feature maps, 4x4 spatial size
			nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),  # Batch normalization
			nn.ReLU(True),            # Activation
			# ngf*8 -> ngf*4, upsample to 8x8
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# ngf*4 -> ngf*2, upsample to 16x16
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# ngf*2 -> ngf, upsample to 32x32
			nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			# ngf -> nc (output channels, e.g. 3 for RGB), keep 32x32
			nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
			nn.Tanh()  # Output in range [-1, 1]
		)
	def forward(self, input):
		# Pass input through the generator network
		return self.main(input)


# DCGAN Discriminator for 32x32x3 images
class DCGANDiscriminator(nn.Module):
	# Ref: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
	def __init__(self, nc=3, ndf=64):
		super().__init__()  # Initialize the parent nn.Module
		# Feature extraction layers (downsampling)
		self.features = nn.Sequential(
			# Input: nc channels (e.g. 3 for RGB), output ndf feature maps, 16x16
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# ndf -> ndf*2, 8x8
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# ndf*2 -> ndf*4, 4x4
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# ndf*4 -> ndf*8, 2x2
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True)
		)
		# Global Average Pooling to get a fixed-length vector
		self.gap = nn.AdaptiveAvgPool2d((1, 1))
		# Classifier: outputs probability of real/fake
		self.classifier = nn.Sequential(
			nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),  # Output 1x1
			nn.Sigmoid()  # Probability
		)
	def forward(self, input):
		# Pass input through feature extractor
		features = self.features(input)
		# Global average pooling and flatten for embedding
		pooled = self.gap(features).view(features.size(0), -1)  # (batch, ndf*8)
		# Pass features through classifier
		out = self.classifier(features)
		# Return both probability and pooled features
		return out, pooled


# Training DCGAN
def train_dcgan_fid(train_images, epochs=50, batch_size=128, nz=100):
	'''
	NOTE: train_dcgan does NOT use training labels (train_labels) for GAN training.
	GANs are unsupervised: the discriminator only distinguishes real vs. fake images, not classes.
	The labels are not used in the loss or training steps above.
	'''
	# Multi-GPU setup
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select device
	ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0   # Number of GPUs
	# Create the generator and discriminator models
	netG = DCGANGenerator().to(device)
	netD = DCGANDiscriminator().to(device)
	# Handle multi-GPU if available
	if (device.type == 'cuda') and (ngpu > 1):
		netG = nn.DataParallel(netG, list(range(ngpu)))
		netD = nn.DataParallel(netD, list(range(ngpu)))
	# Initialize weights for both models
	netG.apply(weights_init)
	netD.apply(weights_init)
	# Print model architectures
	print(netG)
	print(netD)
	# Prepare DataLoader: convert images to tensor, permute to (N, C, H, W)
	X = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)
	dataset = TensorDataset(X)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	# Loss function: Binary Cross Entropy
	criterion = nn.BCELoss()
	# Optimizers for discriminator and generator
	optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
	epoch_bar = tqdm(range(epochs), desc='Epochs', unit='epoch')
	# Track loss history for plotting
	history = {'D_loss': [], 'G_loss': [], 'FID': []}
	# Early stopping parameters (now for FID)
	patience = 10  # Number of epochs to wait for improvement
	min_delta = 0.01  # Minimum change to qualify as improvement
	best_fid = float('inf')
	epochs_no_improve = 0
	# Create experiment folder
	exp_folder = create_experiment_folder()
	for epoch in epoch_bar:
		batch_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch', leave=False)
		start_time = time.time()
		running_D_loss = 0.0
		running_G_loss = 0.0
		n_batches = 0
		for batch in batch_bar:
			real = batch[0].to(device)  # Real images
			b_size = real.size(0)       # Batch size
			label_real = torch.ones(b_size, 1, 1, 1, device=device)  # Real label
			label_fake = torch.zeros(b_size, 1, 1, 1, device=device) # Fake label
			# Train Discriminator
			netD.zero_grad()  # Zero gradients
			output_real, _ = netD(real)  # Discriminator output for real images
			loss_real = criterion(output_real, label_real)  # Loss for real
			noise = torch.randn(b_size, nz, 1, 1, device=device)  # Random noise
			fake = netG(noise)  # Generate fake images
			output_fake, _ = netD(fake.detach())  # Discriminator output for fake images
			loss_fake = criterion(output_fake, label_fake)  # Loss for fake
			loss_D = loss_real + loss_fake  # Total discriminator loss
			loss_D.backward()  # Backpropagate
			optimizerD.step()  # Update discriminator
			# Train Generator
			netG.zero_grad()  # Zero gradients
			output_fake, _ = netD(fake)  # Discriminator output for fake images (not detached)
			loss_G = criterion(output_fake, label_real)  # Generator tries to fool discriminator
			loss_G.backward()  # Backpropagate
			optimizerG.step()  # Update generator
			# Update tqdm bar with current losses
			batch_bar.set_postfix({
				'D_loss': f'{loss_D.item():.4f}',
				'G_loss': f'{loss_G.item():.4f}'
			})
			running_D_loss += loss_D.item()
			running_G_loss += loss_G.item()
			n_batches += 1
		# Average losses for the epoch
		avg_D_loss = running_D_loss / n_batches
		avg_G_loss = running_G_loss / n_batches
		history['D_loss'].append(avg_D_loss)
		history['G_loss'].append(avg_G_loss)
		elapsed = time.time() - start_time
		# Every 5 epochs, save generated images and compute FID
		if (epoch + 1) % 5 == 0 or epoch == 0:
			save_generated_images(netG, epoch+1, exp_folder, nz=nz, device=device)
			fid = compute_fid_score(netG, train_images, nz=nz, device=device, num_images=1000, batch_size=batch_size)
			history['FID'].append(fid)
			print(f"Epoch {epoch+1}: FID = {fid:.2f}")
			# Early stopping: check if FID improved
			if fid + min_delta < best_fid:
				best_fid = fid
				epochs_no_improve = 0
			else:
				epochs_no_improve += 1
			if epochs_no_improve >= patience:
				print(f"Early stopping at epoch {epoch+1} (no improvement in FID for {patience} checkpoints)")
				break
		else:
			history['FID'].append(None)
		epoch_bar.set_postfix({
			'D_loss': f'{avg_D_loss:.4f}',
			'G_loss': f'{avg_G_loss:.4f}',
			'epoch_time': f'{elapsed:.1f}s',
			'FID': f"{history['FID'][-1] if history['FID'][-1] is not None else '-'}"
		})
		print(f"Epoch {epoch+1}/{epochs} | D Loss: {avg_D_loss:.4f} | G Loss: {avg_G_loss:.4f} | Time: {elapsed:.1f}s | FID: {history['FID'][-1] if history['FID'][-1] is not None else '-'}")
	# Save model weights to disk
	torch.save(netG.state_dict(), os.path.join(exp_folder, 'dcgan_generator.pth'))
	torch.save(netD.state_dict(), os.path.join(exp_folder, 'dcgan_discriminator.pth'))
	print(f"DCGAN models saved in {exp_folder}.")
	# Save training history for plotting
	np.save(os.path.join(exp_folder, 'dcgan_loss_history.npy'), history)
	print(f"Training history saved to {os.path.join(exp_folder, 'dcgan_loss_history.npy')}")
	return netG, netD, history, exp_folder


# Optuna-based hyperparameter tuning for DCGAN
def objective_worker(trial_idx, train_images, n_trials, epochs, nz, patience, min_delta, study_name, db_file):
	"""
	Wrapper for the Optuna objective function to be run in a separate process.
	Assigns a specific GPU to each worker process.
	"""
	gpu_id = trial_idx % torch.cuda.device_count()
	device = torch.device(f'cuda:{gpu_id}')

	# Each process needs to create its own study object pointing to the same storage.
	study = optuna.create_study(direction='minimize', storage=f'sqlite:///{db_file}', study_name=study_name, load_if_exists=True)

	# The objective function needs to be defined within the scope of the worker or be passed to it.
	# For simplicity, we are re-defining parts of the objective logic here.

	# Create a new trial
	trial = study.ask()

	print(f"\n[Optuna Worker {trial_idx} on GPU {gpu_id}] Starting trial {trial.number}")

	# Suggest hyperparameters
	lr_g = trial.suggest_float('lr_g', 5e-5, 3e-4, log=True)
	lr_d = trial.suggest_float('lr_d', 5e-5, 3e-4, log=True)
	beta1 = trial.suggest_float('beta1', 0.5, 0.9)
	beta2 = trial.suggest_float('beta2', 0.7, 0.999)
	batch_size_trial = trial.suggest_categorical('batch_size', [64, 128, 256])

	print(f"[Optuna Worker {trial_idx}] Trial {trial.number} hyperparameters: lr_g={lr_g:.5f}, lr_d={lr_d:.5f}, beta1={beta1:.3f}, beta2={beta2:.3f}, batch_size={batch_size_trial}")

	# Prepare DataLoader
	X = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)
	dataset = TensorDataset(X)
	loader = DataLoader(dataset, batch_size=batch_size_trial, shuffle=True)

	netG = DCGANGenerator(nz=nz).to(device)
	netD = DCGANDiscriminator().to(device)

	netG.apply(weights_init)
	netD.apply(weights_init)

	criterion = nn.BCELoss()
	optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
	optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))

	best_fid = float('inf')
	epochs_no_improve = 0
	fid_interval = 5

	# Use the provided parent experiment folder, create only the optuna_trial subfolder here
	exp_folder = os.path.join(parent_exp_folder, f'optuna_trial_{trial_idx}')
	os.makedirs(exp_folder, exist_ok=True)

	for epoch in range(epochs):
		# Simplified training loop for the worker
		for batch in loader:
			real = batch[0].to(device)
			b_size = real.size(0)
			label_real = torch.ones(b_size, 1, 1, 1, device=device)
			label_fake = torch.zeros(b_size, 1, 1, 1, device=device)

			netD.zero_grad()
			output_real, _ = netD(real)
			loss_real = criterion(output_real, label_real)
			noise = torch.randn(b_size, nz, 1, 1, device=device)
			fake = netG(noise)
			output_fake, _ = netD(fake.detach())
			loss_fake = criterion(output_fake, label_fake)
			loss_D = loss_real + loss_fake
			loss_D.backward()
			optimizerD.step()

			netG.zero_grad()
			output_fake, _ = netD(fake)
			loss_G = criterion(output_fake, label_real)
			loss_G.backward()
			optimizerG.step()

		if (epoch + 1) % fid_interval == 0 or epoch == 0:
			# Save generated images with trial and epoch info
			save_generated_images(netG, f"trial{trial_idx}_epoch{epoch+1}", exp_folder, nz=nz, device=device)
			fid = compute_fid_score(netG, train_images, nz=nz, device=device, num_images=1000, batch_size=batch_size_trial)
			print(f"[Optuna Worker {trial_idx}] Trial {trial.number} Epoch {epoch+1}: FID = {fid:.2f}")

			if fid < best_fid:
				best_fid = fid
				epochs_no_improve = 0
			else:
				epochs_no_improve += 1

			if epochs_no_improve >= patience:
				print(f"[Optuna Worker {trial_idx}] Early stopping in trial {trial.number} at epoch {epoch+1}")
				break

	# Tell the study that the trial is complete
	study.tell(trial, best_fid)
	print(f"[Optuna Worker {trial_idx}] Trial {trial.number} finished. Best FID: {best_fid:.4f}")


# Optuna-based hyperparameter tuning for DCGAN
def train_dcgan_optuna(train_images, n_trials=20, epochs=200, nz=100, patience=20, min_delta=0.01):
	results = []
	best_model_state = {'G': None, 'D': None}
	best_trial_number = [None]  # Use list for mutability in closure

	# Create experiment directory for this Optuna run (only ONCE per execution)
	exp_folder = create_experiment_folder(base_dir='experiments')
	print(f"Optuna experiment directory: {exp_folder}")

	trial_counter = {'current': 0}

	def objective(trial):
		trial_counter['current'] += 1
		print(f"\n[Optuna] Starting trial {trial_counter['current']} of {n_trials} (Optuna trial number: {trial.number})")
		# Suggest hyperparameters
		lr_g = trial.suggest_float('lr_g', 5e-5, 3e-4, log=True)
		lr_d = trial.suggest_float('lr_d', 5e-5, 3e-4, log=True)
		beta1 = trial.suggest_float('beta1', 0.5, 0.9) #DCGAN paper recommend beta_1 = .5
		beta2 = trial.suggest_float('beta2', 0.7, 0.999)
		batch_size_trial = trial.suggest_categorical('batch_size', [64, 128, 256])
		print(f"[Optuna] Trial {trial_counter['current']} hyperparameters: lr_g={lr_g:.5f}, lr_d={lr_d:.5f}, beta1={beta1:.3f}, beta2={beta2:.3f}, batch_size={batch_size_trial}")
		# Prepare DataLoader
		X = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)
		dataset = TensorDataset(X)
		loader = DataLoader(dataset, batch_size=batch_size_trial, shuffle=True)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
		netG = DCGANGenerator(nz=nz).to(device)
		netD = DCGANDiscriminator().to(device)
		if (device.type == 'cuda') and (ngpu > 1):
			netG = nn.DataParallel(netG, list(range(ngpu)))
			netD = nn.DataParallel(netD, list(range(ngpu)))
		netG.apply(weights_init)
		netD.apply(weights_init)
		criterion = nn.BCELoss()
		optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
		optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
		history = {'D_loss': [], 'G_loss': [], 'FID': []}
		best_fid = float('inf')
		epochs_no_improve = 0
		fid_interval = 5  # Compute FID every 5 epochs
		for epoch in range(epochs):
			running_D_loss = 0.0
			running_G_loss = 0.0
			n_batches = 0
			for batch in loader:
				real = batch[0].to(device)
				b_size = real.size(0)
				label_real = torch.ones(b_size, 1, 1, 1, device=device)
				label_fake = torch.zeros(b_size, 1, 1, 1, device=device)
				netD.zero_grad()
				output_real, _ = netD(real)
				loss_real = criterion(output_real, label_real)
				noise = torch.randn(b_size, nz, 1, 1, device=device)
				fake = netG(noise)
				output_fake, _ = netD(fake.detach())
				loss_fake = criterion(output_fake, label_fake)
				loss_D = loss_real + loss_fake
				loss_D.backward()
				optimizerD.step()
				netG.zero_grad()
				output_fake, _ = netD(fake)
				loss_G = criterion(output_fake, label_real)
				loss_G.backward()
				optimizerG.step()
				running_D_loss += loss_D.item()
				running_G_loss += loss_G.item()
				n_batches += 1
			avg_D_loss = running_D_loss / n_batches
			avg_G_loss = running_G_loss / n_batches
			history['D_loss'].append(avg_D_loss)
			history['G_loss'].append(avg_G_loss)
			# Compute FID every fid_interval epochs and at epoch 0
			if (epoch + 1) % fid_interval == 0 or epoch == 0:
				fid = compute_fid_score(netG, train_images, nz=nz, device=device, num_images=1000, batch_size=batch_size_trial)
				history['FID'].append(fid)
				print(f"[Optuna] Trial {trial_counter['current']} Epoch {epoch+1}: FID = {fid:.2f}")
				if fid < best_fid:
					best_fid = fid
					# Save the current best model weights
					best_model_state['G'] = netG.state_dict()
					best_model_state['D'] = netD.state_dict()
					best_trial_number[0] = trial.number
					epochs_no_improve = 0
				else:
					epochs_no_improve += 1
				if epochs_no_improve >= patience:
					print(f"[Optuna] Early stopping in trial {trial_counter['current']} at epoch {epoch+1} (no improvement in FID for {patience} checkpoints)")
					break
			else:
				history['FID'].append(None)
		print(f"[Optuna] Trial {trial_counter['current']} finished. Best FID: {best_fid:.4f}")
		# Use best FID as objective
		trial.set_user_attr('history', history)
		trial.set_user_attr('params', {
			'lr_g': lr_g, 'lr_d': lr_d, 'beta1': beta1, 'beta2': beta2, 'batch_size': batch_size_trial
		})
		# Save trial results for later analysis
		results.append({
			'trial_number': trial.number,
			'params': {
				'lr_g': lr_g, 'lr_d': lr_d, 'beta1': beta1, 'beta2': beta2, 'batch_size': batch_size_trial
			},
			'history': history,
			'best_fid': best_fid
		})
		print(f"[Optuna] {n_trials - trial_counter['current']} trials remaining.")
		return best_fid

	study_name = f"dcgan-hparam-search-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
	db_file = "optuna_dcgan_studies.db"
	study = optuna.create_study(direction='minimize', storage=f'sqlite:///{db_file}', study_name=study_name, load_if_exists=True)

	n_gpu = torch.cuda.device_count()
	if n_gpu > 1 and n_trials > 1:
		print(f"Running {n_trials} Optuna trials in parallel on {n_gpu} GPUs.")

		# Set the start method to 'spawn' for CUDA compatibility
		try:
			mp.set_start_method('spawn', force=True)
		except RuntimeError:
			pass # Start method can only be set once

		# Pass the parent experiment folder to each worker
		objective_partial = partial(
			objective_worker,
			train_images=train_images,
			n_trials=n_trials,
			epochs=epochs,
			nz=nz,
			patience=patience,
			min_delta=min_delta,
			study_name=study_name,
			db_file=db_file,
			parent_exp_folder=exp_folder
		)
		with mp.Pool(processes=n_gpu) as pool:
			pool.map(objective_partial, range(n_trials))
	else:
		print("Running Optuna trials sequentially on a single device.")
		study.optimize(objective, n_trials=n_trials)
	# Find best trial
	best_trial = study.best_trial
	print("Best hyperparameters:", best_trial.params)
	print("Best generator loss:", best_trial.value)
	# Save best model to experiment directory
	if best_model_state['G'] is not None:
		torch.save(best_model_state['G'], os.path.join(exp_folder, 'best_dcgan_generator.pth'))
		torch.save(best_model_state['D'], os.path.join(exp_folder, 'best_dcgan_discriminator.pth'))
		print(f"Saved best model from trial {best_trial_number[0]} in {exp_folder}")
	# Save all trial results to experiment directory
	with open(os.path.join(exp_folder, 'optuna_results.pkl'), 'wb') as f:
		pickle.dump(results, f)
	# Save Optuna study object for further analysis if needed
	with open(os.path.join(exp_folder, 'optuna_study.pkl'), 'wb') as f:
		pickle.dump(study, f)
	return best_trial, exp_folder


# Function to analyze DCGAN experiment history and suggest hyperparameter tuning
def should_tune_hyperparameters(history_path, patience=10, min_delta=0.01, fid_threshold=50):
	"""
	Analyze dcgan_loss_history.npy and suggest if hyperparameter tuning is needed.
	Criteria:
	  - If FID does not improve for 'patience' checkpoints
	  - If FID remains above 'fid_threshold' at the end
	  - If generator or discriminator loss is unstable (e.g., increasing or not decreasing)
	Returns: (should_tune: bool, reason: str)
	"""
	if not os.path.exists(history_path):
		return True, f"History file {history_path} not found. Cannot assess tuning."
	history = np.load(history_path, allow_pickle=True).item()
	D_loss = history.get('D_loss', [])
	G_loss = history.get('G_loss', [])
	FID = history.get('FID', [])
	# Remove None values from FID
	FID = [f for f in FID if f is not None]
	# Check if FID improved in last 'patience' checkpoints
	if len(FID) > patience:
		recent = FID[-patience:]
		if min(recent) >= min(FID[:-patience]) - min_delta:
			return True, f"FID did not improve in the last {patience} checkpoints."
	# Check if final FID is above threshold
	if FID and FID[-1] > fid_threshold:
		return True, f"Final FID ({FID[-1]:.2f}) is above threshold ({fid_threshold})."
	# Check for unstable loss (e.g., increasing trend)
	if len(G_loss) > 5 and G_loss[-1] > G_loss[0] + min_delta:
		return True, "Generator loss increased over training."
	if len(D_loss) > 5 and D_loss[-1] > D_loss[0] + min_delta:
		return True, "Discriminator loss increased over training."
	return False, "No major issues detected. Hyperparameters seem reasonable."


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="CIFAR-100 DCGAN experiment setup, download, and training options.")
	parser.add_argument('--set-up', action='store_true', default=False,
						help='Create the homework and homework/data directories (default: False)')
	parser.add_argument('--download', action='store_true', default=False,
						help='Download and extract the CIFAR-100 dataset (default: False)')
	parser.add_argument('--optuna', action='store_true', default=False,
						help='Use Optuna to tune DCGAN hyperparameters and automate training (default: False). If not set, runs standard train_dcgan_fid.')
	args = parser.parse_args()

	set_up = args.set_up
	download = args.download
	use_optuna = args.optuna

	data_dir = 'data'
	
	if set_up:
		# Create 'homework' and 'homework/data' directories
		os.makedirs(data_dir, exist_ok=True)

	if download:
		if not os.path.exists(os.path.join(data_dir, 'cifar-100-python.tar.gz')):
			data_filename = download(data_dir)
			extracted_dir = extract(data_filename, data_dir)
		else:
			extracted_dir = 'data/cifar-100-python'
	else:
		extracted_dir = 'data/cifar-100-python'

	# Load train and test batches
	train_batch = os.path.join(extracted_dir, 'train')
	# Read in datasets
	if os.path.exists(train_batch):
		train_dict = unpickle(train_batch)
		print(f"Loaded train batch: keys = {list(train_dict.keys())}")

	train_idx_path = os.path.join(data_dir, 'train_idx.npy')
	val_idx_path = os.path.join(data_dir, 'val_idx.npy')
	# Adjust indices to requirements. Only run if train_dict and test_dict are loaded and the indices are not saved to file
	if 'train_dict' in locals() and not os.path.exists(train_idx_path) and not os.path.exists(val_idx_path):
		train_idx, val_idx = split_cifar100_trainval(train_dict, val_ratio=0.25)
		print(f"Train: {len(train_idx)} images, Validation: {len(val_idx)} images, Test: {len(test_dict[b'data'])} images (official test set)")
		np.save(train_idx_path, np.array(train_idx, dtype=np.int32))
		np.save(val_idx_path, np.array(val_idx, dtype=np.int32))
		print(f"Saved train_idx to {train_idx_path} and val_idx to {val_idx_path}")
	else:
		train_idx = np.load(train_idx_path)
		val_idx = np.load(val_idx_path)
		print(f"Loaded train_idx ({len(train_idx)}) and val_idx ({len(val_idx)}) from disk.")

	train_images, train_labels = preprocess_cifar100_subset(train_dict, train_idx)
	if use_optuna:
		print("Running Optuna-based DCGAN hyperparameter tuning...")
		train_dcgan_optuna(train_images)
	else:
		train_dcgan_fid(train_images, epochs=200)
