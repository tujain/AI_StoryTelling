import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from transformers import AutoTokenizer
import logging

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=os.path.join('logs', 'pre_processing.log'),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

class Feature:
    def __init__(self, encoder_ids=None, decoder_ids=None, raw_text=None, cond=None):
        self.encoder_ids = encoder_ids
        self.decoder_ids = decoder_ids
        self.raw_text = raw_text
        self.cond = cond

# Define the data split ratio
split_ratio = {"train": 0.8, "dev": 0.1, "test": 0.1}

def load_tokenizer(model_name, hf_token=None):
    """Load tokenizer safely with error handling."""
    try:
        return AutoTokenizer.from_pretrained(model_name, token=hf_token if hf_token else None)
    except Exception as e:
        logging.error(f"Error loading tokenizer {model_name}: {e}")
        raise RuntimeError(f"Failed to load tokenizer {model_name}")

def get_feature(text, maxlen, encoder_toker, decoder_toker):
    """
    Tokenizes a single line of text using the provided encoder and decoder tokenizers.
    """
    try:
        raw_text = text.strip()
        if not raw_text:
            return None

        # Tokenize
        enc_full = encoder_toker.encode(raw_text, add_special_tokens=True)[:maxlen]
        dec_full = decoder_toker.encode(raw_text, add_special_tokens=True)[:maxlen]

        return vars(Feature(encoder_ids=enc_full, decoder_ids=dec_full, raw_text=raw_text))
    except Exception as e:
        logging.error(f"Error processing text: {e}")
        return None

def load_texts_from_ndjson(file):
    """
    Reads NDJSON file and extracts the "text" field from each line,
    skipping empty lines or invalid JSON entries.
    """
    texts = []
    try:
        with open(file, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()  # Remove leading/trailing whitespace
                if not line:  # Skip empty lines
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "text" in data:
                        texts.append(data["text"])
                except json.JSONDecodeError as e:
                    logging.error(f"Skipping invalid JSON line in {file}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error reading {file}: {e}")

    return texts if texts else None

def distributed_main(args):
    """Main function to process text and save as `.pt` files."""
    
    # Load Tokenizers
    encoder_toker = load_tokenizer(args.encoder_model, args.hf_token)
    decoder_toker = load_tokenizer(args.decoder_model, args.hf_token)

    # Get all NDJSON files from the corpus directory
    files = [os.path.join(args.corpus, fname) for fname in os.listdir(args.corpus) if fname.endswith('.json')]
    logging.info(f"Found {len(files)} JSON files in corpus.")

    # Optimize multiprocessing: Use available CPU cores but limit to 8 max
    num_cores = min(cpu_count(), 8)
    logging.info(f"Using {num_cores} CPU cores for multiprocessing.")

    # Use multiprocessing for efficiency
    with Pool(processes=num_cores) as pool:
        for file in tqdm(files, total=len(files)):
            logging.info(f"Processing {file}")

            texts = load_texts_from_ndjson(file)
            if texts is None:
                continue  # Skip to next file if parsing failed

            np.random.shuffle(texts)
            total = len(texts)
            logging.info(f"Total texts in {file}: {total}")

            # Data splits
            train_split = int(split_ratio["train"] * total)
            dev_split = train_split + int(split_ratio["dev"] * total)
            dataset = {
                "train": texts[:train_split],
                "dev": texts[train_split:dev_split],
                "test": texts[dev_split:]
            }

            for split_name, split_texts in dataset.items():
                if not split_texts:
                    logging.info(f"No data for split: {split_name} in {file}.")
                    continue

                logging.info(f"Processing {len(split_texts)} texts for split: {split_name}.")
                
                # Apply multiprocessing tokenization
                features = pool.map(
                    partial(get_feature, maxlen=args.maxlen, encoder_toker=encoder_toker, decoder_toker=decoder_toker),
                    split_texts,
                    chunksize=max(1, len(split_texts) // num_cores)
                )

                # Remove None values
                features = [f for f in features if f is not None]
                output_dir = os.path.join(os.path.dirname(file), 'parsed', split_name)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{os.path.basename(file)[:-5]}_{split_name}.pt")
                torch.save(features, output_file)
                logging.info(f"Saved {len(features)} features for split '{split_name}' to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True, help='Directory containing training corpus JSON files')
    parser.add_argument('--maxlen', type=int, default=256, help='Maximum token length for each sequence')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face access token')
    parser.add_argument('--encoder_model', required=True, help='Name of the encoder model')
    parser.add_argument('--decoder_model', required=True, help='Name of the decoder model')
    args = parser.parse_args()

    if os.path.isdir(args.corpus):
        distributed_main(args)
    else:
        logging.error("Corpus directory does not exist.")
