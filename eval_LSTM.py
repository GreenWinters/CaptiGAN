import os
import torch
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from dcgan_CIFAR100 import DCGANDiscriminator, preprocess_cifar100_subset, unpickle
from ques2 import (
    load_dcgan_discriminator,
    load_diffusion_CLIP_encoder,
    LSTMDecoder,
    SimpleLSTMDecoder,
)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge


def plot_losses(loss_json_path, out_path=None):
    with open(loss_json_path, 'r') as f:
        losses = json.load(f)
    train_loss = losses['train_loss']
    val_loss = losses['val_loss']
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()


def evaluate_lstm_on_val(model, img_encoder, val_images, val_caption_ids, idx2word, word2idx, device='cuda', max_length=45):
    model.eval()
    img_encoder.eval()
    references = []
    hypotheses = []
    results_pairs = []
    images_to_save = []
    with torch.no_grad():
        batch_size = 32
        def get_batches(images, caption_ids, batch_size):
            for i in range(0, len(images), batch_size):
                yield images[i:i+batch_size], caption_ids[i:i+batch_size], range(i, min(i+batch_size, len(images)))
        for img_batch, cap_batch, idxs in get_batches(val_images, val_caption_ids, batch_size):
            img_batch_tensor = torch.tensor(img_batch, dtype=torch.float32, device=device)
            if img_batch_tensor.shape[1] == 32 and img_batch_tensor.shape[3] == 3:
                img_batch_tensor = img_batch_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
            cap_batch = cap_batch.to(device)
            img_embed = img_encoder(img_batch_tensor)
            # Greedy decoding
            inputs = cap_batch[:, :1]  # <bos>
            outputs = []
            for _ in range(max_length-1):
                logits = model(img_embed, inputs)
                next_token = logits[:, -1, :].argmax(-1, keepdim=True)
                inputs = torch.cat([inputs, next_token], dim=1)
                outputs.append(next_token)
            outputs = torch.cat(outputs, dim=1)  # (batch, seq_len)
            for i, (ref, hyp, img_idx) in enumerate(zip(cap_batch, outputs, idxs)):
                # Remove special tokens
                ref_tokens = [idx2word[j.item()] for j in ref if j.item() not in [word2idx['<pad>'], word2idx['<bos>'], word2idx['<eos>']]]
                hyp_tokens = [idx2word[j.item()] for j in hyp if j.item() not in [word2idx['<pad>'], word2idx['<bos>'], word2idx['<eos>']]]
                references.append([ref_tokens])
                hypotheses.append(' '.join(hyp_tokens))
                if len(results_pairs) < 10:
                    # Save image and captions for first 10
                    images_to_save.append((img_batch[i], ' '.join(ref_tokens), ' '.join(hyp_tokens)))
                results_pairs.append({'true_label': ' '.join(ref_tokens), 'generated_label': ' '.join(hyp_tokens)})
    # BLEU-4
    bleu4 = corpus_bleu(references, [h.split() for h in hypotheses], smoothing_function=SmoothingFunction().method1)
    # ROUGE-L
    rouge = Rouge()
    rouge_l = rouge.get_scores(hypotheses, [' '.join(r[0]) for r in references], avg=True)['rouge-l']['f']
    return {'BLEU-4': bleu4, 'ROUGE-L': rouge_l, 'results_pairs': results_pairs, 'images_to_save': images_to_save}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LSTM captioning model on validation set")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--max_length', type=int, default=None, help='Max caption length (overrides checkpoint if set)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to LSTM model .pth file')
    parser.add_argument('--decoder_type', type=str, choices=['default', 'simple'], default='default',
                        help="Use 'default' for the original LSTM decoder, 'simple' for a dumbed-down decoder.")
    parser.add_argument('--discriminator', type=str, default=None,
                        help="Path to DCGAN discriminator .pth file (overrides checkpoint if set)")
    args = parser.parse_args()

    # Load model checkpoint (contains vocab, word2idx, idx2word, decoder, img_encoder)
    model_state = torch.load(args.model_path, map_location=args.device)
    vocab = model_state['vocab']
    word2idx = model_state['word2idx']
    idx2word = model_state['idx2word']

    # Load validation data
    extracted_dir = 'data/cifar-100-python'
    train_batch = os.path.join(extracted_dir, 'train')
    meta_path = os.path.join(extracted_dir, 'meta')
    data_dir = "data"
    val_idx_path = os.path.join(data_dir, 'val_idx.npy')
    if os.path.exists(train_batch):
        train_dict = unpickle(train_batch)
    else:
        raise FileNotFoundError(f"Train batch file not found: {train_batch}")
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        label_names = meta['fine_label_names']
    else:
        raise FileNotFoundError(f"CIFAR-100 meta file not found: {meta_path}")
    val_idx = np.load(val_idx_path)
    val_images, val_labels = preprocess_cifar100_subset(train_dict, val_idx)
    val_captions = [label_names[int(lbl)] for lbl in val_labels]

    # Use tokenizer from checkpoint
    max_length = args.max_length if args.max_length is not None else len(model_state['vocab'])
    encode_caption = lambda c, max_len=max_length: torch.tensor(
        [word2idx.get('<bos>')] +
        [word2idx.get(w, word2idx['<unk>']) for w in c.lower().split()][:max_len-2] +
        [word2idx.get('<eos>')] +
        [word2idx.get('<pad>')] * (max_len - (len(c.lower().split()) + 2))
    )
    val_caption_ids = torch.stack([encode_caption(c, max_len=max_length) for c in val_captions])

    # Load image encoder
    if args.discriminator:
        img_encoder = load_dcgan_discriminator(args.discriminator, DCGANDiscriminator, device=args.device)
    else:
        # Use encoder from checkpoint
        img_encoder = load_dcgan_discriminator(None, DCGANDiscriminator, device=args.device)
        img_encoder.load_state_dict(model_state['img_encoder'])

    # Load decoder
    if args.decoder_type == 'simple':
        decoder = SimpleLSTMDecoder(embed_dim=512, vocab_size=len(vocab), hidden_dim=128, num_layers=1).to(args.device)
    else:
        decoder = LSTMDecoder(embed_dim=512, vocab_size=len(vocab), hidden_dim=512, num_layers=2).to(args.device)
    decoder.load_state_dict(model_state['decoder'])

    # Plot training/validation loss if available
    exp_dir = os.path.dirname(args.model_path)
    loss_json_path = os.path.join(exp_dir, 'train_val_losses.json')
    if os.path.exists(loss_json_path):
        plot_path = os.path.join(exp_dir, 'loss_curve.png')
        plot_losses(loss_json_path, out_path=plot_path)
        print(f"Loss curve saved to {plot_path}")
    else:
        print(f"Loss file not found at {loss_json_path}")

    # Evaluate
    results = evaluate_lstm_on_val(decoder, img_encoder, val_images, val_caption_ids, idx2word, word2idx, device=args.device, max_length=max_length)
    print("Evaluation Results:")
    print(f"BLEU-4: {results['BLEU-4']:.4f}")
    print(f"ROUGE-L: {results['ROUGE-L']:.4f}")

    # Save results to experiment directory
    results_path = os.path.join(exp_dir, 'val_caption_results.json')
    with open(results_path, 'w') as f:
        json.dump(results['results_pairs'], f, indent=2)
    print(f"Saved validation caption results to {results_path}")

    # Save and display 10 image-caption pairs
    for i, (img, true_cap, gen_cap) in enumerate(results['images_to_save']):
        plt.figure()
        # Denormalize from [-1, 1] to [0, 255]
        img_disp = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if img_disp.shape[-1] == 3:
            plt.imshow(img_disp)
        else:
            plt.imshow(img_disp.transpose(1,2,0))
        plt.title(f"True: {true_cap}\nGen: {gen_cap}")
        plt.axis('off')
        img_path = os.path.join(exp_dir, f'val_img_{i+1}.png')
        plt.savefig(img_path)
        plt.close()
        print(f"Saved image {i+1} to {img_path}")
