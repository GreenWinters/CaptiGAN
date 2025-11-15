import os
import datetime
import csv
import argparse
import torch
import numpy as np
import json
import pickle
from tqdm import tqdm
from collections import Counter
from transformers import CLIPProcessor, CLIPModel, pipeline
from torchvision import transforms
from dcgan_CIFAR100 import DCGANDiscriminator, preprocess_cifar100_subset, create_experiment_folder, unpickle
import torch.nn as nn
import torch.optim as optim

# --- Load DCGAN Discriminators from experiments directory ---
def load_dcgan_discriminator(model_path, model_class, device='cpu'):
    """
    The output of the last convolutional layer (or the last feature map before the classification head)
    in the truncated Discriminator is a raw image feature vector.
    This function removes the final classification layer and adds a trainable projection head to map
    DCGAN features to the CLIP embedding dimension (512 for ViT-B/16).
    """
    class DCGANEncoderWithProj(nn.Module):
        def __init__(self, base_model, embed_dim=512):
            super().__init__()
            # Remove classifier (assume classifier is named 'classifier')
            self.features = base_model.features
            # Determine output feature dim
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 32, 32)
                feat = self.features(dummy)
                feat_dim = feat.view(1, -1).shape[1]
            # Projection head: Linear + LayerNorm
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, embed_dim),
                nn.ReLU(),
                nn.LayerNorm(embed_dim)
            )
        def forward(self, x):
            feats = self.features(x)
            feats = feats.reshape(feats.size(0), -1)
            embed = self.proj(feats)
            embed = nn.functional.normalize(embed, dim=-1)
            return embed
    base_model = model_class()
    state_dict = torch.load(model_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    base_model.load_state_dict(state_dict)
    encoder = DCGANEncoderWithProj(base_model, embed_dim=512).to(device)
    encoder.eval()
    return encoder

# --- Load HuggingFace Diffusion Model (CLIP) ---
def load_diffusion_CLIP_encoder(device='cpu'):
    '''
    Load CLIP 
    Ref https://huggingface.co/openai/clip-vit-base-patch16
    '''
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=False)
    return clip_model, clip_processor


# --- LSTM Decoder for Captioning ---
class LSTMDecoder(nn.Module):
    """
    Image-conditioned LSTM decoder for sequence generation (e.g., captioning).

    This module projects a fixed-dimensional image embedding to initialize the hidden
    state of an LSTM, embeds input token indices, and autoregressively produces
    per-token vocabulary logits.

    Args:
        embed_dim (int): Dimensionality of the input image embedding that conditions the decoder.
        vocab_size (int): Size of the vocabulary (number of target classes/tokens).
        hidden_dim (int, optional): Hidden size used for token embeddings and the LSTM.
            Defaults to 512.
        num_layers (int, optional): Number of stacked LSTM layers. Defaults to 2.
        dropout (float, optional): Dropout probability applied between LSTM layers during
            training. Note: PyTorch only applies dropout when num_layers > 1. Defaults to 0.0.

    Attributes:
        embed_dim (int): Image embedding dimensionality expected by img_proj.
        hidden_dim (int): Hidden size used across embeddings, LSTM, and classifier.
        vocab_size (int): Number of output classes/tokens.
        word_embed (nn.Embedding): Token embedding layer producing hidden_dim features.
        lstm (nn.LSTM): LSTM with batch_first=True, input_size=hidden_dim, hidden_size=hidden_dim.
        img_proj (nn.Linear): Linear projection from embed_dim to hidden_dim used to
            initialize the first LSTM layer's hidden state.
        classifier (nn.Linear): Linear classifier mapping hidden states to vocabulary logits.

    Shape conventions:
        - img_embed: (batch, embed_dim)
        - captions: (batch, seq_len) of dtype torch.long (token indices)
        - logits: (batch, seq_len, vocab_size)

    Notes:
        - The initial hidden state h0 for layer 0 is derived from img_embed via img_proj;
          remaining layers (if any) are zero-initialized. The initial cell state c0 is
          zero-initialized for all layers.
        - Dropout within nn.LSTM is active only if num_layers > 1 (PyTorch behavior).
    """
    def __init__(self, embed_dim, vocab_size, hidden_dim=512, num_layers=2, dropout=0.0):
        '''
        set dropout=0.1 with num_layers=1. PyTorch warns that dropout is ignored unless num_layers > 1.
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word_embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.img_proj = nn.Linear(embed_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img_embed, captions):
        batch_size = img_embed.size(0)
        seq_len = captions.size(1)
        num_layers = self.lstm.num_layers
        h0_img = self.img_proj(img_embed).unsqueeze(0)  # (1, batch, hidden_dim)
        if num_layers > 1:
            h0_rest = torch.zeros(num_layers - 1, batch_size, self.hidden_dim, device=img_embed.device, dtype=img_embed.dtype)
            h0 = torch.cat([h0_img, h0_rest], dim=0)
        else:
            h0 = h0_img
        c0 = torch.zeros_like(h0)
        x = self.word_embed(captions)  # (batch, seq_len, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        logits = self.classifier(out)  # (batch, seq_len, vocab_size)
        return logits

# --- Dumbed-down Decoder Option ---
class SimpleLSTMDecoder(nn.Module):
    """
    A simple image-conditioned LSTM decoder for sequence generation (e.g., captioning).

    This module conditions an LSTM on a fixed image embedding by projecting the image
    features into the LSTM hidden space and using that projection to initialize the
    hidden state h0. The cell state c0 is initialized to zeros. Token IDs are embedded,
    processed by the LSTM (batch_first=True), and projected to vocabulary logits.

    Architecture:
    - img_proj: Linear(embed_dim -> hidden_dim) used to initialize h0
    - word_embed: Embedding(vocab_size, hidden_dim) for caption tokens
    - lstm: LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
    - classifier: Linear(hidden_dim -> vocab_size) to produce per-step logits

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the upstream image embedding.
    vocab_size : int
        Size of the target vocabulary.
    hidden_dim : int, default=128
        LSTM hidden size and token embedding size.
    num_layers : int, default=1
        Number of stacked LSTM layers.
    dropout : float, default=0.0
        Dropout probability applied between LSTM layers (effective only if num_layers > 1).

    Inputs (forward)
    ----------------
    img_embed : torch.Tensor
        Image features of shape (batch_size, embed_dim).
    captions : torch.LongTensor
        Token IDs of shape (batch_size, seq_len). Typically includes a BOS token at
        the start and may include an EOS token.

    Returns
    -------
    logits : torch.FloatTensor
        Unnormalized token logits of shape (batch_size, seq_len, vocab_size). These
        can be passed to CrossEntropyLoss (with targets aligned to captions) or to
        softmax for probabilities.

    Shape notes
    -----------
    - batch_size: B, sequence length: T, vocab size: V, image embedding dim: E, hidden dim: H
    - img_embed: (B, E) -> img_proj -> h0: (1, B, H); c0: zeros_like(h0)
    - captions: (B, T) -> word_embed -> (B, T, H)
    - LSTM output: (B, T, H) -> classifier -> logits: (B, T, V)

    Usage notes
    -----------
    - Teacher forcing: This forward assumes teacher forcing (it consumes the full
      caption sequence in one pass). It does not perform autoregressive sampling.
    - num_layers: The provided initialization creates h0 with a single layer
      (shape (1, B, H)). For num_layers > 1, you must adapt/expand the initial
      state to match shape (num_layers, B, H), otherwise PyTorch will raise a
      shape mismatch error.
    - Dropout: LSTM dropout is active only when num_layers > 1 (per PyTorch semantics).

    Example
    -------
    >>> B, T, E, V, H = 8, 16, 512, 30522, 256
    >>> img = torch.randn(B, E)
    >>> caps = torch.randint(0, V, (B, T))
    >>> dec = SimpleLSTMDecoder(embed_dim=E, vocab_size=V, hidden_dim=H, num_layers=1)
    >>> logits = dec(img, caps)  # (B, T, V)
    """
    def __init__(self, embed_dim, vocab_size, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word_embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.img_proj = nn.Linear(embed_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    def forward(self, img_embed, captions):
        batch_size = img_embed.size(0)
        seq_len = captions.size(1)
        num_layers = self.lstm.num_layers
        h0_img = self.img_proj(img_embed).unsqueeze(0)
        h0 = h0_img
        c0 = torch.zeros_like(h0)
        x = self.word_embed(captions)
        out, _ = self.lstm(x, (h0, c0))
        logits = self.classifier(out)
        return logits

# --- Caption Enrichment (Flush Out Captions) ---
def enrich_captions_with_transformer(captions, model_name="google/flan-t5-small", batch_size=32):
    """
    Use a text2text-generation transformer to expand class names into descriptive sentences.
    """
    generator = pipeline("text2text-generation", model=model_name)
    enriched = []
    prompts = [f"Describe a {cap} in a sentence." for cap in captions]
    for i in tqdm(range(0, len(prompts), batch_size), desc="Enriching captions"):
        batch_prompts = prompts[i:i+batch_size]
        batch_outputs = generator(batch_prompts, max_length=20)
        batch_texts = [out['generated_text'] for out in batch_outputs]
        enriched.extend(batch_texts)
    return enriched

# --- CLIP-style Contrastive Loss ---
def clip_contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    Compute the symmetric CLIP-style contrastive loss for a batch of image and text embeddings.

    This loss encourages matching image–text pairs within the same batch to have high similarity and non-matching pairs to have low similarity. It averages the image-to-text and text-to-image cross-entropy losses computed from temperature-scaled similarity logits.

    Args:
        image_embeds (torch.Tensor): L2-normalized image embeddings of shape (batch_size, embed_dim).
        text_embeds (torch.Tensor): L2-normalized text embeddings of shape (batch_size, embed_dim).
        temperature (float, optional): Positive temperature used to scale similarity logits (smaller values sharpen the distribution). Default: 0.07.

    Returns:
        torch.Tensor: A scalar tensor containing the symmetric contrastive loss.

    Notes:
        - Assumes the i-th image in the batch matches the i-th text (identity labels).
        - Similarity logits are computed as image_embeds @ text_embeds.T / temperature.
        - image_embeds and text_embeds must have the same batch_size.
    """
    # image_embeds, text_embeds: (batch, embed_dim), both normalized
    logits = image_embeds @ text_embeds.t() / temperature
    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)
    loss_i2t = nn.CrossEntropyLoss()(logits, labels)
    loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2

# --- Vocabulary and Tokenization Utilities ---
def build_vocab_and_tokenizer(captions, vocab_size=1000, max_len=45):
    """
    Builds a vocabulary and tokenizer for a list of text captions.

    Args:
        captions (list of str): List of text captions to build the vocabulary from.
        vocab_size (int, optional): Maximum size of the vocabulary including special tokens. Default is 1000.
        max_len (int, optional): Maximum length of tokenized captions (including special tokens). Default is 45.

    Returns:
        vocab (list of str): List of vocabulary words, including special tokens ['<pad>', '<bos>', '<eos>', '<unk>'].
        word2idx (dict): Mapping from word to index in the vocabulary.
        idx2word (dict): Mapping from index to word in the vocabulary.
        encode_caption (function): Function that encodes a caption string into a tensor of token indices,
                                   adding <bos> and <eos> tokens, padding to max_len, and using <unk> for unknown words.
    """
    all_tokens = [w for cap in captions for w in cap.lower().split()]
    vocab = ['<pad>', '<bos>', '<eos>', '<unk>'] + [w for w, _ in Counter(all_tokens).most_common(vocab_size-4)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    def encode_caption(caption, max_len=max_len):
        tokens = caption.lower().split()
        ids = [word2idx.get('<bos>')] + [word2idx.get(w, word2idx['<unk>']) for w in tokens][:max_len-2] + [word2idx.get('<eos>')]
        ids += [word2idx.get('<pad>')] * (max_len - len(ids))
        return torch.tensor(ids)
    return vocab, word2idx, idx2word, encode_caption

def train_captioning_model(
    img_encoder, decoder,
    train_images, train_caption_ids, train_clip_text_embeds,
    val_images, val_caption_ids, val_clip_text_embeds,
    word2idx, vocab,
    lr, contrastive_loss_weight,
    epochs, batch_size, device, exp_dir):
    """
    Main function to train the image captioning model.
    """
    # --- Training loop ---
    # Use different learning rates for encoder and decoder
    optimizer = torch.optim.Adam([
        {'params': img_encoder.parameters(), 'lr': lr * 0.1},
        {'params': decoder.parameters(), 'lr': lr}])
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_state = None
    train_losses = []
    val_losses = []

    def get_batches(images, caption_ids, batch_size):
        for i in range(0, len(images), batch_size):
            yield images[i:i+batch_size], caption_ids[i:i+batch_size]

    for epoch in range(epochs):
        img_encoder.train()
        decoder.train()
        total_loss = 0
        train_batches = int(np.ceil(len(train_images) / batch_size))
        with tqdm(total=train_batches, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=100) as pbar:
            for i, (img_batch, cap_batch) in enumerate(get_batches(train_images, train_caption_ids, batch_size)):
                img_batch = torch.tensor(img_batch, dtype=torch.float32, device=device)
                if img_batch.shape[1] == 32 and img_batch.shape[3] == 3:
                    img_batch = img_batch.permute(0, 3, 1, 2)  # NHWC -> NCHW
                cap_batch = cap_batch.to(device)
                # Get image embedding
                img_embed = img_encoder(img_batch)
                # Get precomputed text embedding for this batch
                text_embed = train_clip_text_embeds[i*batch_size:i*batch_size+img_batch.size(0)].to(device)
                # LSTM decoder output
                logits = decoder(img_embed, cap_batch[:, :-1])
                # Cross-entropy loss (caption prediction)
                ce_loss = ce_loss_fn(logits.reshape(-1, decoder.vocab_size), cap_batch[:, 1:].reshape(-1))
                # Contrastive loss
                contrast_loss = clip_contrastive_loss(img_embed, text_embed)
                loss = ce_loss + contrastive_loss_weight * contrast_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * img_batch.size(0)
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                pbar.update(1)
        avg_train_loss = total_loss / len(train_images)
        train_losses.append(avg_train_loss)

        # Validation
        img_encoder.eval()
        decoder.eval()
        val_loss = 0
        val_batches = int(np.ceil(len(val_images) / batch_size))
        with torch.no_grad():
            with tqdm(total=val_batches, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", ncols=100) as pbar_val:
                for i, (img_batch, cap_batch) in enumerate(get_batches(val_images, val_caption_ids, batch_size)):
                    img_batch = torch.tensor(img_batch, dtype=torch.float32, device=device)
                    if img_batch.shape[1] == 32 and img_batch.shape[3] == 3:
                        img_batch = img_batch.permute(0, 3, 1, 2)  # NHWC -> NCHW
                    cap_batch = cap_batch.to(device)
                    img_embed = img_encoder(img_batch)
                    text_embed = val_clip_text_embeds[i*batch_size:i*batch_size+img_batch.size(0)].to(device)
                    logits = decoder(img_embed, cap_batch[:, :-1])
                    ce_loss = ce_loss_fn(logits.reshape(-1, decoder.vocab_size), cap_batch[:, 1:].reshape(-1))
                    contrast_loss = clip_contrastive_loss(img_embed, text_embed)
                    loss = ce_loss + contrastive_loss_weight * contrast_loss
                    val_loss += loss.item() * img_batch.size(0)
                    pbar_val.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                    pbar_val.update(1)
        avg_val_loss = val_loss / len(val_images)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {
                'img_encoder': img_encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'vocab': vocab,
                'word2idx': word2idx,
                'idx2word': idx2word
            }
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save losses
    with open(os.path.join(exp_dir, 'train_val_losses.json'), 'w') as f:
        json.dump({'train_loss': train_losses, 'val_loss': val_losses}, f, indent=2)

    # Save best model
    if best_state:
        save_path = os.path.join(exp_dir, 'captioning_model.pth')
        torch.save(best_state, save_path)
        print(f"Trained model saved to {save_path}")
    
    return best_val_loss


def get_clip_text_embeds(captions, batch_size=128):
    # Precompute CLIP text embeddings for all captions (train and val)
    all_embeds = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Precomputing CLIP text embeddings"):
        batch_caps = captions[i:i+batch_size]
        text_inputs = clip_processor(text=batch_caps, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(args.device)
        with torch.no_grad():
            text_embed = text_encoder(**text_inputs).pooler_output
            text_embed = nn.functional.normalize(text_embed, dim=-1)
        all_embeds.append(text_embed.cpu())
    return torch.cat(all_embeds, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP with DCGAN+CLIP features")
    parser.add_argument('--discriminator', type=str, choices=['early_stop', 'hp0', 'hp1'], default='hp1',
                        help="Which DCGAN discriminator to use: 'early_stop' (20251108_193221), 'hp0' (Optuna Search - 20251108_21474), 'hp1' (Optuna Search - 20251112_210616) ")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=45, help='Max caption length')
    parser.add_argument('--caption_mode', type=str, choices=['simple', 'enriched'], default='simple',
                        help="Use 'simple' for class names, 'enriched' to flush out captions with a transformer.")
    parser.add_argument('--decoder_type', type=str, choices=['default', 'simple'], default='default',
                        help="Use 'default' for the original LSTM decoder, 'simple' for a dumbed-down decoder.")
    args = parser.parse_args()

    # Select discriminator
    if args.discriminator == 'early_stop':
        disc_path = 'experiments/experiment_20251108_193221/dcgan_discriminator.pth'
    if args.discriminator == "hp0":
        disc_path = 'experiments/experiment_20251108_214747/best_dcgan_discriminator.pth'
    if args.discriminator == "hp1":
        disc_path = 'experiments/experiment_20251112_210616/dcgan_discriminator.pth'

    # Load base encoders and data
    clip_model, clip_processor = load_diffusion_CLIP_encoder(device=args.device)

    # Load Data     
    extracted_dir = 'data/cifar-100-python'
    train_batch = os.path.join(extracted_dir, 'train')
    meta_path = os.path.join(extracted_dir, 'meta')
    data_dir = "data"

    if os.path.exists(train_batch):
        train_dict = unpickle(train_batch)
        print(f"Loaded train batch: keys = {list(train_dict.keys())}")
    else:
        raise FileNotFoundError(f"Train batch file not found: {train_batch}")

    # Map integer labels to class names using CIFAR-100 meta file
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        label_names = meta['fine_label_names']
    else:
        raise FileNotFoundError(f"CIFAR-100 meta file not found: {meta_path}")

    # Validate train/val index and dicts
    train_idx_path = os.path.join(data_dir, 'train_idx.npy')
    val_idx_path = os.path.join(data_dir, 'val_idx.npy')
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)

    train_images, train_labels = preprocess_cifar100_subset(train_dict, train_idx)
    val_images, val_labels = preprocess_cifar100_subset(train_dict, val_idx)
    print(f"Loaded train_idx ({len(train_idx)}) and val_idx ({len(val_idx)}) from disk.")    

    # --- Caption selection ---
    train_captions = [label_names[int(lbl)] for lbl in train_labels]
    val_captions = [label_names[int(lbl)] for lbl in val_labels]

    if args.caption_mode == 'enriched':
        print("Enriching captions using a transformer...")
        train_captions = enrich_captions_with_transformer(train_captions)
        val_captions = enrich_captions_with_transformer(val_captions)
        vocab_size = 1000
        max_len = args.max_length
    else:
        # For simple/class name captions, use a small vocab and short max_len
        vocab_size = 120
        max_len = 5

    vocab, word2idx, idx2word, encode_caption = build_vocab_and_tokenizer(train_captions, vocab_size=vocab_size, max_len=max_len)
    train_caption_ids = torch.stack([encode_caption(c, max_len=max_len) for c in train_captions])
    val_caption_ids = torch.stack([encode_caption(c, max_len=max_len) for c in val_captions])

    # CLIP text encoder (frozen)
    text_encoder = clip_model.text_model
    for p in text_encoder.parameters():
        p.requires_grad = False

    train_clip_text_embeds = get_clip_text_embeds(train_captions)
    val_clip_text_embeds = get_clip_text_embeds(val_captions)

    # --- Manual Hyperparameter Search ---
    lr_options = [2e-6] #[5e-5, 2e-6]
    contrastive_weight_options = [0.01] #, 0.05, 0.2, 0.5]
    results = {}

    # Create a unique experiment directory for this grid search run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    grid_exp_dir = os.path.join('experiments', 'captioning_grid_search', f'experiment_{timestamp}')
    os.makedirs(grid_exp_dir, exist_ok=True)

    grid_results = []

    for lr in lr_options:
        for weight in contrastive_weight_options:
            print("\n" + "="*50)
            print(f"STARTING EXPERIMENT: lr={lr}, contrastive_weight={weight}")
            print("="*50)

            # Subdirectory for this hyperparameter combination
            run_name = f"lr_{lr}_weight_{weight}"
            exp_dir = os.path.join(grid_exp_dir, run_name)
            os.makedirs(exp_dir, exist_ok=True)
            print(f"Experiment folder: {exp_dir}")

            # Re-initialize models for a clean run
            img_encoder = load_dcgan_discriminator(disc_path, DCGANDiscriminator, device=args.device)
            if args.decoder_type == 'simple':
                decoder = SimpleLSTMDecoder(embed_dim=512, vocab_size=len(vocab), hidden_dim=128, num_layers=1).to(args.device)
            else:
                decoder = LSTMDecoder(embed_dim=512, vocab_size=len(vocab), hidden_dim=512, num_layers=2).to(args.device)

            best_val_loss = train_captioning_model(
                img_encoder=img_encoder,
                decoder=decoder,
                train_images=train_images,
                train_caption_ids=train_caption_ids,
                train_clip_text_embeds=train_clip_text_embeds,
                val_images=val_images,
                val_caption_ids=val_caption_ids,
                val_clip_text_embeds=val_clip_text_embeds,
                word2idx=word2idx,
                vocab=vocab,
                lr=lr,
                contrastive_loss_weight=weight,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                exp_dir=exp_dir
            )

            # Save best model with caption mode in filename
            pth_path = os.path.join(exp_dir, f'captioning_model_{args.caption_mode}.pth')
            train_val_json_path = os.path.join(exp_dir, 'train_val_losses.json')
            # Move files if needed (train_captioning_model already saves them, but we ensure correct naming)
            if os.path.exists(os.path.join(exp_dir, 'captioning_model.pth')):
                os.rename(os.path.join(exp_dir, 'captioning_model.pth'), pth_path)
            # train_val_losses.json is already saved in exp_dir

            grid_results.append({
                'run_name': run_name,
                'lr': lr,
                'contrastive_weight': weight,
                'caption_mode': args.caption_mode,
                'decoder_type': args.decoder_type,
                'best_val_loss': best_val_loss,
                'model_path': pth_path,
                'train_val_json': train_val_json_path
            })
            print(f"FINISHED EXPERIMENT: lr={lr}, weight={weight} | Best Val Loss: {best_val_loss:.4f}")

    # Save grid search results as CSV and JSON in the grid experiment directory
    csv_path = os.path.join(grid_exp_dir, 'grid_search_results.csv')
    json_path = os.path.join(grid_exp_dir, 'grid_search_results.json')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=grid_results[0].keys())
        writer.writeheader()
        for row in grid_results:
            writer.writerow(row)
    with open(json_path, 'w') as f:
        json.dump(grid_results, f, indent=2)

    print("\n" + "="*50)
    print("GRID SEARCH COMPLETE")
    print("="*50)
    for row in sorted(grid_results, key=lambda item: item['best_val_loss']):
        print(f"Run: {row['run_name']:<25} | Best Validation Loss: {row['best_val_loss']:.4f} | Model: {row['model_path']}")