import argparse
import json
import os
import torch
import re
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial
from multiprocessing import Pool

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=os.path.join('logs', 'pre_processing.log'),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

class Feature:
    def __init__(self, encoder_ids=None, decoder_ids=None, raw_text=None):
        self.encoder_ids = encoder_ids
        self.decoder_ids = decoder_ids
        self.raw_text = raw_text

def get_tokenizers(encoder_model, decoder_model):
    """Load tokenizers for the specified encoder and decoder models."""
    try:
        encoder_toker = AutoTokenizer.from_pretrained(encoder_model)
        decoder_toker = AutoTokenizer.from_pretrained(decoder_model)
        logging.info(f"Loaded tokenizers: Encoder ({encoder_model}), Decoder ({decoder_model})")
    except Exception as e:
        logging.error(f"Error loading tokenizers: {e}")
        raise
    return encoder_toker, decoder_toker

def get_feature(line, maxlen, encoder_toker, decoder_toker):
    """Tokenize text using encoder and decoder tokenizers."""
    try:
        raw_text = line.strip()
        encoder_ids = encoder_toker.encode(raw_text, truncation=True, max_length=maxlen)
        decoder_ids = decoder_toker.encode(raw_text, truncation=True, max_length=maxlen)
        return vars(Feature(encoder_ids, decoder_ids, raw_text))
    except Exception as e:
        logging.error(f"Error tokenizing text: {e}")
        return None

def distributed_main(args):
    """Process dataset files, split into train/dev/test sets, and save as PyTorch tensors."""
    files = [os.path.join(args.corpus, fname) for fname in os.listdir(args.corpus) if fname.endswith('.json')]
    pool = Pool()

    encoder_toker, decoder_toker = get_tokenizers(args.encoder_model, args.decoder_model)

    for file in tqdm(files, total=len(files)):
        try:
            logging.info(f"Processing file: {file}")
            with open(file, "r", encoding="utf-8") as reader:
                data = json.load(reader)  # Load entire JSON as a list of dicts
                texts = [entry['text'].strip() for entry in data if entry.get('text', '').strip()]

            if not texts:
                logging.warning(f"Skipping empty file: {file}")
                continue  

            np.random.shuffle(texts)
            train_split = int(args.train_ratio * len(texts))
            dev_split = train_split + int(args.dev_ratio * len(texts))

            dataset = {
                "train": texts[:train_split],
                "dev": texts[train_split:dev_split],
                "test": texts[dev_split:]
            }

            for split_name, split_data in dataset.items():
                if split_data:
                    logging.info(f"Processing {len(split_data)} samples for {split_name} split in {file}")
                    features = pool.map(partial(get_feature, maxlen=args.maxlen, encoder_toker=encoder_toker, decoder_toker=decoder_toker), split_data)
                    features = [f for f in features if f is not None]  # Remove failed samples

                    save_path = os.path.join(os.path.dirname(file), 'parsed', split_name)
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(features, os.path.join(save_path, f'{os.path.basename(file)[:-5]}_{split_name}.pt'))

        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")

    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True, help='Directory of JSON corpus')
    parser.add_argument('--encoder_model', required=True, help='Hugging Face model for encoding (e.g., "bert-base-uncased")')
    parser.add_argument('--decoder_model', required=True, help='Hugging Face model for decoding (e.g., "gpt2")')
    parser.add_argument('--maxlen', type=int, default=256, help='Maximum token length')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--dev_ratio', type=float, default=0.1, help='Dev split ratio')
    args = parser.parse_args()

    if os.path.isdir(args.corpus):
        distributed_main(args)
    else:
        logging.error("Corpus directory does not exist.")
