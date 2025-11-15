import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dcgan_CIFAR100 import (
    DCGANGenerator,
    DCGANDiscriminator,
    compute_fid_score,
    weights_init,
    train_dcgan_fid as orig_train_dcgan_fid,
    unpickle,
    preprocess_cifar100_subset,
)


def train_dcgan_fid(train_images, optimizerG, optimizerD, exp_folder = 'experiments/experiment_20251112_210616', epochs=200, batch_size=64, nz=100):
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
	epoch_bar = tqdm(range(epochs), desc='Epochs', unit='epoch')
	# Track loss history for plotting
	history = {'D_loss': [], 'G_loss': [], 'FID': []}
	# Early stopping parameters (now for FID)
	patience = 10  # Number of epochs to wait for improvement
	min_delta = 0.01  # Minimum change to qualify as improvement
	best_fid = float('inf')
	epochs_no_improve = 0
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


best_params = {
    'lr_g': 0.0002899839218668599,
    'lr_d': 9.720978420923859e-05,
    'beta1': 0.7142109833058737,
    'beta2': 0.9867233256125157,
    'batch_size': 64
}

extracted_dir = 'data/cifar-100-python'
data_dir = 'data'
# Load train and test batches
train_batch = os.path.join(extracted_dir, 'train')
# Read in datasets
if os.path.exists(train_batch):
    train_dict = unpickle(train_batch)
    print(f"Loaded train batch: keys = {list(train_dict.keys())}")

train_idx_path = os.path.join(data_dir, 'train_idx.npy')
train_idx = np.load(train_idx_path)
train_images, train_labels = preprocess_cifar100_subset(train_dict, train_idx)

# Use the best hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = DCGANGenerator().to(device)
netD = DCGANDiscriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)

optimizerG = torch.optim.Adam(netG.parameters(), lr=best_params['lr_g'], betas=(best_params['beta1'], best_params['beta2']))
optimizerD = torch.optim.Adam(netD.parameters(), lr=best_params['lr_d'], betas=(best_params['beta1'], best_params['beta2']))

# Use your existing training loop, but set batch_size and optimizer as above
train_dcgan_fid(
    train_images,
    optimizerG, 
    optimizerD,
    epochs=200, 
    batch_size=best_params['batch_size'],
    nz=100
)
