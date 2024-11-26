#!/usr/bin/env python3
# train.py

import os
import math
import logging
import argparse
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import set_seed
from transformers import GPT2Tokenizer, BertTokenizer, T5Tokenizer, RobertaTokenizer

from autoencoder import (
    GPT2Encoder,
    T5Encoder,
    RoBERTaDecoder,
    AutoEncoder,
    FeatureDataset,
    BertNoiser,
    NoNoise,
    GPT2Noiser,
    evaluate_model,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train an AutoEncoder model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--encoder_model', type=str, default='gpt2', help='Encoder model name')
    parser.add_argument('--decoder_model', type=str, default='roberta-base', help='Decoder model name')
    parser.add_argument('--hidden_size', type=int, default=None, help='Hidden size for encoder and decoder')    
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data file (.pt format)')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data file (.pt format)')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the model and logs')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--noise_type', type=str, default='bert', choices=['bert', 'gpt2', 'none'], help='Type of noise to add to input sequences')
    parser.add_argument('--mlm_probability', type=float, default=0.15, help='Masking probability for MLM noise')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training and evaluation')
    args = parser.parse_args()
    return args

def load_features(features_path, encoder_tokenizer, decoder_tokenizer):
    data = torch.load(features_path)
    features = []
    for item in data:
        # Get raw text
        raw_text = item.get('raw_text', '')
        if not raw_text:
            continue  # Skip if raw_text is empty

        # Encode using encoder tokenizer
        encoder_ids = encoder_tokenizer.encode(raw_text, add_special_tokens=True)
        # Encode using decoder tokenizer
        decoder_ids = decoder_tokenizer.encode(raw_text, add_special_tokens=True)
        # Create feature
        feature = {
            'input_ids_enc': encoder_ids,
            'lm_labels': decoder_ids,
            'raw_text': raw_text,
            'cond': None,
        }
        features.append(feature)
    return features


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Save the arguments to a file
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Set random seed
    set_seed(args.seed)

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'training.log'),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info("Training parameters %s", args)

    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Initialize encoder
    if args.encoder_model.startswith('gpt2'):
        encoder = GPT2Encoder(model_name=args.encoder_model, hidden_size=args.hidden_size)
    elif args.encoder_model.startswith('t5'):
        encoder = T5Encoder(model_name=args.encoder_model, hidden_size=args.hidden_size)
    else:
        raise ValueError(f"Unsupported encoder model: {args.encoder_model}")

    # Initialize decoder
    if args.decoder_model.startswith('roberta'):
        decoder = RoBERTaDecoder(model_name=args.decoder_model, hidden_size=args.hidden_size)
    else:
        raise ValueError(f"Unsupported decoder model: {args.decoder_model}")

    # Create autoencoder
    autoencoder = AutoEncoder(encoder, decoder, hidden_size=args.hidden_size)
    autoencoder.to(device)

    # Get tokenizer names
    encoder_tokenizer = encoder.tokenizer
    decoder_tokenizer = decoder.tokenizer

    # Load features
    train_features = load_features(args.train_data, encoder_tokenizer, decoder_tokenizer)
    val_features = load_features(args.val_data, encoder_tokenizer, decoder_tokenizer)


    # Prepare datasets
    train_dataset = FeatureDataset(features=train_features)
    val_dataset = FeatureDataset(features=val_features)

    # Prepare data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=FeatureDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=FeatureDataset.collate_fn,
    )

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(autoencoder.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    # Set up noise function
    if args.noise_type == 'bert':
        if isinstance(encoder.tokenizer, (BertTokenizer, RobertaTokenizer)):
            noiser = BertNoiser(encoder.tokenizer, mlm_probability=args.mlm_probability)
        elif isinstance(encoder.tokenizer, GPT2Tokenizer):
            noiser = GPT2Noiser(encoder.tokenizer, mlm_probability=args.mlm_probability)
        else:
            raise ValueError("Unsupported tokenizer for noise type 'bert'.")
    elif args.noise_type == 'gpt2':
        noiser = GPT2Noiser(encoder.tokenizer, mlm_probability=args.mlm_probability)
    else:
        noiser = NoNoise()

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        autoencoder.train()
        total_loss = 0.0
        total_tokens = 0
        for step, batch in enumerate(train_loader):
            input_ids_enc, _, lm_labels, _ = batch
            input_ids_enc = input_ids_enc.to(device)
            lm_labels = lm_labels.to(device)

            # Add noise to encoder input
            input_ids_enc_noised = noiser.noise(input_ids_enc)
            input_ids_enc_noised = input_ids_enc_noised.to(device)

            # Forward pass
            loss, _, _ = autoencoder(
                input_ids_enc=input_ids_enc_noised,
                attention_mask_enc=None,
                labels=lm_labels,
            )
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            mask = lm_labels.ne(-100)
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

            if global_step % 100 == 0:
                avg_loss = total_loss / total_tokens
                perplexity = math.exp(avg_loss)
                logger.info(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
                total_loss = 0.0
                total_tokens = 0

        # Validation
        val_loss, val_accuracy, val_perplexity = evaluate_model(autoencoder, val_loader, device)
        logger.info(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Perplexity: {val_perplexity:.2f}")
        scheduler.step(val_loss)

        # Save the model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            autoencoder.save(args.output_dir, f'checkpoint_epoch_{epoch}')

if __name__ == '__main__':
    main()