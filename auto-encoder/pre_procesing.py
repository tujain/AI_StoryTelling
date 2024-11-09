import argparse
import json
import os
import logging
from dataclasses import dataclass, asdict
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Dict

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    DistilBertTokenizer,
    DistilGPT2Tokenizer,
    RobertaTokenizer,
    T5Tokenizer,
    AutoModel,
    AutoTokenizer,
)

# Global variables for embedding model and tokenizer
embedding_model = None
embedding_tokenizer = None

def pool_initializer(embedding_model_name: str):
    """
    Initializer for each worker in the multiprocessing pool.
    Loads the NVIDIA Embed v2 model and tokenizer.
    
    Args:
        embedding_model_name (str): The name/path of the NVIDIA Embed v2 model.
    """
    global embedding_model
    global embedding_tokenizer
    try:
        embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        embedding_model = AutoModel.from_pretrained(embedding_model_name)
        embedding_model.eval()
        embedding_model.to('cpu')  # Change to 'cuda' if GPU is available and desired
        logging.info(f"NVIDIA Embed v2 model '{embedding_model_name}' loaded successfully in worker.")
    except Exception as e:
        logging.error(f"Failed to initialize NVIDIA Embed v2 model '{embedding_model_name}': {e}")
        embedding_model = None
        embedding_tokenizer = None

@dataclass
class Feature:
    bert_ids: Optional[List[int]] = None
    gpt2_ids: Optional[List[int]] = None
    roberta_ids: Optional[List[int]] = None
    t5_ids: Optional[List[int]] = None
    nvidia_embed: Optional[List[float]] = None
    raw_text: Optional[str] = None
    cond: Optional[str] = None  # Placeholder for any conditional data or labels

def initialize_tokenizers() -> Dict[str, AutoTokenizer]:
    """
    Initialize and return all required tokenizers except for NVIDIA Embed v2.

    Returns:
        Dict[str, AutoTokenizer]: Dictionary of initialized tokenizers.
    """
    try:
        bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        gpt2_tokenizer = DistilGPT2Tokenizer.from_pretrained('distilgpt2')
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')  # Lightweight T5 model
        logging.info("All tokenizers initialized successfully.")
        return {
            'bert': bert_tokenizer,
            'gpt2': gpt2_tokenizer,
            'roberta': roberta_tokenizer,
            't5': t5_tokenizer,
        }
    except Exception as e:
        logging.error(f"Failed to initialize tokenizers: {e}")
        raise e

def get_feature(line: str, maxlen: int, tokenizers: Dict[str, AutoTokenizer]) -> Dict:
    """
    Process a single line of text and convert it into token IDs and embeddings using various tokenizers and models.
    Includes validation checks to ensure feature integrity.

    Args:
        line (str): The input text line.
        maxlen (int): The maximum length for token sequences.
        tokenizers (Dict[str, AutoTokenizer]): Dictionary containing all tokenizers.

    Returns:
        Dict: A dictionary containing token IDs, embeddings, and the raw text.
    """
    global embedding_model
    global embedding_tokenizer

    raw_text = line.strip()
    
    # Initialize empty feature dictionary
    feature = asdict(Feature(raw_text=raw_text))

    if not raw_text:
        logging.warning("Encountered empty raw_text. Skipping feature extraction.")
        return feature  # Returning with raw_text only

    # Tokenize and encode with DistilBERT
    try:
        bert_ids = tokenizers['bert'].encode(raw_text, truncation=True, max_length=maxlen)
        if not bert_ids:
            logging.warning("DistilBERT tokenization resulted in empty token IDs.")
        elif len(bert_ids) > maxlen:
            logging.warning(f"DistilBERT token IDs exceed maxlen: {len(bert_ids)} > {maxlen}")
        feature['bert_ids'] = bert_ids
    except Exception as e:
        logging.error(f"DistilBERT tokenization failed for text: {raw_text[:50]}... Error: {e}")

    # Tokenize and encode with DistilGPT2
    try:
        gpt2_ids = tokenizers['gpt2'].encode(raw_text, truncation=True, max_length=maxlen)
        if not gpt2_ids:
            logging.warning("DistilGPT2 tokenization resulted in empty token IDs.")
        elif len(gpt2_ids) > maxlen:
            logging.warning(f"DistilGPT2 token IDs exceed maxlen: {len(gpt2_ids)} > {maxlen}")
        feature['gpt2_ids'] = gpt2_ids
    except Exception as e:
        logging.error(f"DistilGPT2 tokenization failed for text: {raw_text[:50]}... Error: {e}")

    # Tokenize and encode with RoBERTa
    try:
        roberta_ids = tokenizers['roberta'].encode(raw_text, truncation=True, max_length=maxlen)
        if not roberta_ids:
            logging.warning("RoBERTa tokenization resulted in empty token IDs.")
        elif len(roberta_ids) > maxlen:
            logging.warning(f"RoBERTa token IDs exceed maxlen: {len(roberta_ids)} > {maxlen}")
        feature['roberta_ids'] = roberta_ids
    except Exception as e:
        logging.error(f"RoBERTa tokenization failed for text: {raw_text[:50]}... Error: {e}")

    # Tokenize and encode with T5
    try:
        t5_ids = tokenizers['t5'].encode(raw_text, truncation=True, max_length=maxlen)
        if not t5_ids:
            logging.warning("T5 tokenization resulted in empty token IDs.")
        elif len(t5_ids) > maxlen:
            logging.warning(f"T5 token IDs exceed maxlen: {len(t5_ids)} > {maxlen}")
        feature['t5_ids'] = t5_ids
    except Exception as e:
        logging.error(f"T5 tokenization failed for text: {raw_text[:50]}... Error: {e}")

    # Generate NVIDIA-Embed v2 embeddings
    if embedding_model and embedding_tokenizer:
        try:
            inputs = embedding_tokenizer(raw_text, return_tensors='pt', truncation=True, max_length=maxlen)
            with torch.no_grad():
                embeddings = embedding_model(**inputs)
                # Assuming the model outputs last_hidden_state
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                # Average pooling to get a fixed-size embedding
                embed = torch.mean(embeddings, dim=1).squeeze().cpu().numpy().tolist()
                feature['nvidia_embed'] = embed
        except Exception as e:
            logging.warning(f"NVIDIA Embed v2 embedding failed for text: {raw_text[:50]}... Error: {e}")
    else:
        logging.warning("NVIDIA Embed v2 model or tokenizer not initialized. Skipping embeddings.")

    # Additional check: Ensure raw_text is not excessively long
    if len(raw_text) > maxlen * 4:  # Arbitrary threshold
        logging.warning(f"Raw text length {len(raw_text)} exceeds threshold.")

    return feature

# Define the split ratio
SPLIT_RATIO = {"train": 0.8, "dev": 0.1, "test": 0.1}

def split_data(data: List[str]) -> Dict[str, List[str]]:
    """
    Shuffle and split the data into train, dev, and test sets based on SPLIT_RATIO.

    Args:
        data (List[str]): List of text entries.

    Returns:
        Dict[str, List[str]]: A dictionary containing split datasets.
    """
    np.random.shuffle(data)
    total = len(data)
    train_end = int(SPLIT_RATIO["train"] * total)
    dev_end = train_end + int(SPLIT_RATIO["dev"] * total)

    return {
        "train": data[:train_end],
        "dev": data[train_end:dev_end],
        "test": data[dev_end:]
    }

def validate_feature(feature: Dict, maxlen: int, file_path: str, split_name: str, index: int) -> bool:
    """
    Validate the extracted feature to ensure correctness.

    Args:
        feature (Dict): The feature dictionary to validate.
        maxlen (int): The maximum allowed length for token IDs.
        file_path (str): Path to the source file (for logging purposes).
        split_name (str): Name of the data split (train/dev/test).
        index (int): Index of the feature in the dataset (for logging).

    Returns:
        bool: True if the feature is valid, False otherwise.
    """
    is_valid = True
    raw_text = feature.get('raw_text', '').strip()

    if not raw_text:
        logging.error(f"[{file_path}][{split_name}][{index}] Raw text is empty.")
        is_valid = False

    # Validate DistilBERT IDs
    bert_ids = feature.get('bert_ids', [])
    if not bert_ids:
        logging.error(f"[{file_path}][{split_name}][{index}] DistilBERT IDs are empty.")
        is_valid = False
    elif len(bert_ids) > maxlen:
        logging.error(f"[{file_path}][{split_name}][{index}] DistilBERT IDs exceed maxlen: {len(bert_ids)} > {maxlen}")
        is_valid = False

    # Validate DistilGPT2 IDs
    gpt2_ids = feature.get('gpt2_ids', [])
    if not gpt2_ids:
        logging.error(f"[{file_path}][{split_name}][{index}] DistilGPT2 IDs are empty.")
        is_valid = False
    elif len(gpt2_ids) > maxlen:
        logging.error(f"[{file_path}][{split_name}][{index}] DistilGPT2 IDs exceed maxlen: {len(gpt2_ids)} > {maxlen}")
        is_valid = False

    # Validate RoBERTa IDs
    roberta_ids = feature.get('roberta_ids', [])
    if not roberta_ids:
        logging.error(f"[{file_path}][{split_name}][{index}] RoBERTa IDs are empty.")
        is_valid = False
    elif len(roberta_ids) > maxlen:
        logging.error(f"[{file_path}][{split_name}][{index}] RoBERTa IDs exceed maxlen: {len(roberta_ids)} > {maxlen}")
        is_valid = False

    # Validate T5 IDs
    t5_ids = feature.get('t5_ids', [])
    if not t5_ids:
        logging.error(f"[{file_path}][{split_name}][{index}] T5 IDs are empty.")
        is_valid = False
    elif len(t5_ids) > maxlen:
        logging.error(f"[{file_path}][{split_name}][{index}] T5 IDs exceed maxlen: {len(t5_ids)} > {maxlen}")
        is_valid = False

    # Validate NVIDIA-Embed v2 Embedding
    nvidia_embed = feature.get('nvidia_embed', [])
    if nvidia_embed:
        if not isinstance(nvidia_embed, list) or not all(isinstance(x, float) for x in nvidia_embed):
            logging.error(f"[{file_path}][{split_name}][{index}] NVIDIA Embed v2 embedding is invalid.")
            is_valid = False
    else:
        logging.warning(f"[{file_path}][{split_name}][{index}] NVIDIA Embed v2 embedding is missing.")
        # Depending on requirements, you might set is_valid = False here
        # is_valid = False

    return is_valid

def process_file(file_path: str, maxlen: int, tokenizers: Dict[str, AutoTokenizer], pool: Pool):
    """
    Process a single JSON file: tokenize texts, split data, validate features, and save features.

    Args:
        file_path (str): Path to the JSON file.
        maxlen (int): Maximum token length.
        tokenizers (Dict[str, AutoTokenizer]): Dictionary containing all tokenizers.
        pool (Pool): Multiprocessing pool for parallel processing.
    """
    logging.info(f"Processing file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as reader:
            data = []
            for line in reader:
                try:
                    parsed_line = json.loads(line)
                    text = parsed_line.get('text', '').strip()
                    if text:
                        data.append(text)
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON decode error in file {file_path}: {e}")
                    continue

        if not data:
            logging.warning(f"No valid text data found in file {file_path}. Skipping.")
            return

        logging.info(f"Total text entries in {file_path}: {len(data)}")
        split_datasets = split_data(data)

        for split_name, split_data_list in split_datasets.items():
            if not split_data_list:
                logging.warning(f"No data for split '{split_name}' in file {file_path}. Skipping this split.")
                continue

            logging.info(f"Processing split '{split_name}' with {len(split_data_list)} entries.")

            # Partial function with fixed arguments
            feature_func = partial(get_feature, maxlen=maxlen, tokenizers=tokenizers)

            # Determine an optimal chunksize
            chunksize = max(1, len(split_data_list) // (cpu_count() * 4))

            # Tokenize in parallel with tqdm progress bar
            features = list(tqdm(pool.imap(feature_func, split_data_list, chunksize=chunksize),
                                 total=len(split_data_list),
                                 desc=f"Tokenizing {split_name}"))

            # Validate features
            valid_features = []
            for idx, feature in enumerate(features):
                if validate_feature(feature, maxlen, file_path, split_name, idx):
                    valid_features.append(feature)
                else:
                    logging.warning(f"Invalid feature at index {idx} in split '{split_name}' of file {file_path}. Skipping.")

            if not valid_features:
                logging.warning(f"All features in split '{split_name}' of file {file_path} are invalid. Skipping saving.")
                continue

            # Define the output directory
            parsed_dir = os.path.join(os.path.dirname(file_path), 'parsed', split_name)
            os.makedirs(parsed_dir, exist_ok=True)

            # Define the output file path
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(parsed_dir, f'{base_filename}_{split_name}.pt')

            # Save the valid features using torch
            torch.save(valid_features, output_path)
            logging.info(f"Saved {len(valid_features)} valid {split_name} features to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred while processing file {file_path}: {e}")

def distributed_main(args):
    """
    Main function to process all JSON files in the corpus directory.

    Args:
        args: Command-line arguments containing 'corpus' and 'maxlen'.
    """
    corpus_dir = args.corpus
    maxlen = args.maxlen
    embedding_model_name = args.embedding_model  # New argument for embedding model

    # Initialize tokenizers
    try:
        tokenizers = initialize_tokenizers()
    except Exception as e:
        logging.error(f"Initialization of tokenizers failed: {e}")
        return

    # Gather all JSON files in the corpus directory
    json_files = [os.path.join(corpus_dir, fname) for fname in os.listdir(corpus_dir) if fname.endswith('.json')]

    if not json_files:
        logging.error(f"No JSON files found in the directory {corpus_dir}. Exiting.")
        return

    logging.info(f"Found {len(json_files)} JSON files in the directory {corpus_dir}.")

    # Initialize multiprocessing pool with initializer to load embedding model
    with Pool(processes=cpu_count(), initializer=pool_initializer, initargs=(embedding_model_name,)) as pool:
        for file_path in tqdm(json_files, desc="Processing files"):
            process_file(file_path, maxlen, tokenizers, pool)

    logging.info("All files have been processed successfully.")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process and tokenize text data from JSON files.")
    parser.add_argument(
        '--corpus',
        required=True,
        type=str,
        help='Path to the directory containing JSON corpus files.'
    )
    parser.add_argument(
        '--maxlen',
        type=int,
        default=256,
        help='Maximum token length for tokenizers (default: 256).'
    )
    parser.add_argument(
        '--embedding_model',
        required=True,
        type=str,
        help='Name or path of the NVIDIA Embed v2 model.'
    )
    return parser.parse_args()

def main():
    # Configure logging with file handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("processing.log"),
            logging.StreamHandler()
        ]
    )

    args = parse_arguments()

    if not os.path.isdir(args.corpus):
        logging.error(f"The provided corpus path '{args.corpus}' is not a directory or does not exist.")
        return

    distributed_main(args)

if __name__ == '__main__':
    main()
