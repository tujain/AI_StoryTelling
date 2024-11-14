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
    GPT2Tokenizer,
    RobertaTokenizer,
    T5Tokenizer,
)
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK's Punkt tokenizer if not already downloaded
nltk.download('punkt', quiet=True)

@dataclass
class Feature:
    bert_ids: Optional[List[int]] = None
    gpt2_ids: Optional[List[int]] = None
    roberta_ids: Optional[List[int]] = None
    t5_ids: Optional[List[int]] = None
    raw_text: Optional[str] = None
    chunk_id: Optional[int] = None  # Identifier for the chunk

def initialize_tokenizers() -> Dict[str, 'AutoTokenizer']:
    """
    Initialize and return all required tokenizers.

    Returns:
        Dict[str, 'AutoTokenizer']: Dictionary of initialized tokenizers.
    """
    try:
        bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')  # Requires SentencePiece
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

def get_max_token_limit(tokenizers: Dict[str, 'AutoTokenizer']) -> int:
    """
    Determine the minimum maximum token limit across all tokenizers.

    Args:
        tokenizers (Dict[str, 'AutoTokenizer']): Dictionary of tokenizers.

    Returns:
        int: The minimum maximum token limit.
    """
    # Define maximum tokens for each tokenizer
    # DistilBERT and RoBERTa: 512 tokens
    # GPT-2: 1024 tokens
    # T5: 512 tokens
    tokenizer_max_limits = {
        'bert': 512,
        'gpt2': 1024,
        'roberta': 512,
        't5': 512,
    }
    min_limit = min(tokenizer_max_limits.values())
    logging.info(f"Using a maximum token limit of {min_limit} tokens per chunk.")
    return min_limit

def split_text_into_chunks(raw_text: str, tokenizers: Dict[str, 'AutoTokenizer'], max_tokens: int) -> List[Dict]:
    """
    Split raw_text into chunks without exceeding max_tokens per chunk.

    Args:
        raw_text (str): The text to split.
        tokenizers (Dict[str, 'AutoTokenizer']): Dictionary of tokenizers.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        List[Dict]: List of dictionaries, each containing a chunk of text and its chunk_id.
    """
    sentences = sent_tokenize(raw_text)
    chunks = []
    current_chunk = ""
    chunk_id = 0

    for sentence in sentences:
        # Estimate the number of tokens in the sentence using the most restrictive tokenizer (e.g., BERT)
        num_tokens = len(tokenizers['bert'].encode(sentence, add_special_tokens=False))
        
        if num_tokens > max_tokens:
            # If a single sentence exceeds max_tokens, split the sentence further
            words = sentence.split()
            sub_sentence = ""
            for word in words:
                if sub_sentence:
                    temp_sentence = sub_sentence + ' ' + word
                else:
                    temp_sentence = word
                sub_num_tokens = len(tokenizers['bert'].encode(temp_sentence, add_special_tokens=False))
                if sub_num_tokens > max_tokens:
                    if sub_sentence:
                        chunks.append({'text': sub_sentence, 'chunk_id': chunk_id})
                        chunk_id += 1
                        sub_sentence = word
                    else:
                        # Single word exceeds max_tokens, unlikely but handle gracefully
                        logging.warning(f"A single word exceeds the max token limit: '{word}'. Skipping this word.")
                        sub_sentence = ""
                else:
                    sub_sentence = temp_sentence
            if sub_sentence:
                chunks.append({'text': sub_sentence, 'chunk_id': chunk_id})
                chunk_id += 1
        else:
            # Check if adding the sentence would exceed the current chunk's max_tokens
            if current_chunk:
                temp_chunk = current_chunk + ' ' + sentence
            else:
                temp_chunk = sentence
            estimated_tokens = len(tokenizers['bert'].encode(temp_chunk, add_special_tokens=False))
            if estimated_tokens > max_tokens:
                # Save the current chunk and start a new one
                if current_chunk:
                    chunks.append({'text': current_chunk, 'chunk_id': chunk_id})
                    chunk_id += 1
                current_chunk = sentence
            else:
                # Add sentence to the current chunk
                current_chunk = temp_chunk

    # Add the last chunk
    if current_chunk:
        chunks.append({'text': current_chunk, 'chunk_id': chunk_id})
    
    return chunks

def get_feature(chunk: Dict, max_tokens: int, tokenizers: Dict[str, 'AutoTokenizer']) -> Dict:
    """
    Process a single text chunk and convert it into token IDs using various tokenizers.
    Includes validation checks to ensure feature integrity.

    Args:
        chunk (Dict): Dictionary containing 'text' and 'chunk_id'.
        max_tokens (int): The maximum number of tokens per chunk.
        tokenizers (Dict[str, 'AutoTokenizer']): Dictionary containing all tokenizers.

    Returns:
        Dict: A dictionary containing token IDs, the raw text, and chunk_id.
    """
    raw_text = chunk['text'].strip()
    chunk_id = chunk['chunk_id']
    
    # Initialize empty feature dictionary
    feature = asdict(Feature(raw_text=raw_text, chunk_id=chunk_id))

    if not raw_text:
        logging.warning(f"[Chunk {chunk_id}] Encountered empty raw_text. Skipping feature extraction.")
        return feature  # Returning with raw_text only

    # Tokenize and encode with DistilBERT
    try:
        bert_ids = tokenizers['bert'].encode(raw_text, truncation=True, max_length=max_tokens)
        if not bert_ids:
            logging.warning(f"[Chunk {chunk_id}] DistilBERT tokenization resulted in empty token IDs.")
        feature['bert_ids'] = bert_ids
    except Exception as e:
        logging.error(f"[Chunk {chunk_id}] DistilBERT tokenization failed for text: {raw_text[:50]}... Error: {e}")

    # Tokenize and encode with GPT2
    try:
        gpt2_ids = tokenizers['gpt2'].encode(raw_text, truncation=True, max_length=max_tokens)
        if not gpt2_ids:
            logging.warning(f"[Chunk {chunk_id}] GPT2 tokenization resulted in empty token IDs.")
        feature['gpt2_ids'] = gpt2_ids
    except Exception as e:
        logging.error(f"[Chunk {chunk_id}] GPT2 tokenization failed for text: {raw_text[:50]}... Error: {e}")

    # Tokenize and encode with RoBERTa
    try:
        roberta_ids = tokenizers['roberta'].encode(raw_text, truncation=True, max_length=max_tokens)
        if not roberta_ids:
            logging.warning(f"[Chunk {chunk_id}] RoBERTa tokenization resulted in empty token IDs.")
        feature['roberta_ids'] = roberta_ids
    except Exception as e:
        logging.error(f"[Chunk {chunk_id}] RoBERTa tokenization failed for text: {raw_text[:50]}... Error: {e}")

    # Tokenize and encode with T5
    try:
        t5_ids = tokenizers['t5'].encode(raw_text, truncation=True, max_length=max_tokens)
        if not t5_ids:
            logging.warning(f"[Chunk {chunk_id}] T5 tokenization resulted in empty token IDs.")
        feature['t5_ids'] = t5_ids
    except Exception as e:
        logging.error(f"[Chunk {chunk_id}] T5 tokenization failed for text: {raw_text[:50]}... Error: {e}")

    return feature

def split_data(data: List[str], tokenizers: Dict[str, 'AutoTokenizer'], max_tokens: int) -> Dict[str, List[Dict]]:
    """
    Shuffle, split the data into train, dev, and test sets, and further split texts into chunks.

    Args:
        data (List[str]): List of text entries.
        tokenizers (Dict[str, 'AutoTokenizer']): Dictionary of tokenizers.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        Dict[str, List[Dict]]: A dictionary containing split datasets with chunks.
    """
    np.random.shuffle(data)
    total = len(data)
    train_end = int(SPLIT_RATIO["train"] * total)
    dev_end = train_end + int(SPLIT_RATIO["dev"] * total)

    splits = {
        "train": data[:train_end],
        "dev": data[train_end:dev_end],
        "test": data[dev_end:]
    }

    split_chunks = {}
    for split_name, split_data_list in splits.items():
        chunks = []
        for text in split_data_list:
            split = split_text_into_chunks(text, tokenizers, max_tokens)
            chunks.extend(split)
        split_chunks[split_name] = chunks

    return split_chunks

def validate_feature(feature: Dict, file_path: str, split_name: str, index: int) -> bool:
    """
    Validate the extracted feature to ensure correctness.

    Args:
        feature (Dict): The feature dictionary to validate.
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
    elif len(bert_ids) > 512:
        logging.error(f"[{file_path}][{split_name}][{index}] DistilBERT IDs exceed 512 tokens: {len(bert_ids)} > 512")
        is_valid = False

    # Validate GPT2 IDs
    gpt2_ids = feature.get('gpt2_ids', [])
    if not gpt2_ids:
        logging.error(f"[{file_path}][{split_name}][{index}] GPT2 IDs are empty.")
        is_valid = False
    elif len(gpt2_ids) > 1024:
        logging.error(f"[{file_path}][{split_name}][{index}] GPT2 IDs exceed 1024 tokens: {len(gpt2_ids)} > 1024")
        is_valid = False

    # Validate RoBERTa IDs
    roberta_ids = feature.get('roberta_ids', [])
    if not roberta_ids:
        logging.error(f"[{file_path}][{split_name}][{index}] RoBERTa IDs are empty.")
        is_valid = False
    elif len(roberta_ids) > 512:
        logging.error(f"[{file_path}][{split_name}][{index}] RoBERTa IDs exceed 512 tokens: {len(roberta_ids)} > 512")
        is_valid = False

    # Validate T5 IDs
    t5_ids = feature.get('t5_ids', [])
    if not t5_ids:
        logging.error(f"[{file_path}][{split_name}][{index}] T5 IDs are empty.")
        is_valid = False
    elif len(t5_ids) > 512:
        logging.error(f"[{file_path}][{split_name}][{index}] T5 IDs exceed 512 tokens: {len(t5_ids)} > 512")
        is_valid = False

    return is_valid

def process_file(file_path: str, tokenizers: Dict[str, 'AutoTokenizer'], max_tokens: int):
    """
    Process a single JSON file: tokenize texts, split data, validate features, and save features.

    Args:
        file_path (str): Path to the JSON file.
        tokenizers (Dict[str, 'AutoTokenizer']): Dictionary containing all tokenizers.
        max_tokens (int): Maximum number of tokens per chunk.
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
        split_chunks = split_data(data, tokenizers, max_tokens)

        for split_name, chunks in split_chunks.items():
            if not chunks:
                logging.warning(f"No data for split '{split_name}' in file {file_path}. Skipping this split.")
                continue

            logging.info(f"Processing split '{split_name}' with {len(chunks)} chunks.")

            # Partial function with fixed arguments
            feature_func = partial(get_feature, max_tokens=max_tokens, tokenizers=tokenizers)

            # Determine an optimal chunksize
            chunksize = max(1, len(chunks) // (cpu_count() * 4))

            # Tokenize in parallel with tqdm progress bar
            with Pool(processes=cpu_count()) as pool:
                features = list(tqdm(pool.imap(feature_func, chunks, chunksize=chunksize),
                                     total=len(chunks),
                                     desc=f"Tokenizing {split_name}"))

            # Validate features
            valid_features = []
            for idx, feature in enumerate(features):
                if validate_feature(feature, file_path, split_name, idx):
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

    # Initialize tokenizers
    try:
        tokenizers = initialize_tokenizers()
    except Exception as e:
        logging.error(f"Initialization of tokenizers failed: {e}")
        return

    # Determine the maximum token limit
    max_tokens = get_max_token_limit(tokenizers)

    # Gather all JSON files in the corpus directory
    json_files = [os.path.join(corpus_dir, fname) for fname in os.listdir(corpus_dir) if fname.endswith('.json')]

    if not json_files:
        logging.error(f"No JSON files found in the directory {corpus_dir}. Exiting.")
        return

    logging.info(f"Found {len(json_files)} JSON files in the directory {corpus_dir}.")

    # Process each file sequentially
    for file_path in tqdm(json_files, desc="Processing files"):
        process_file(file_path, tokenizers, max_tokens)

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
    return parser.parse_args()

# Define the split ratio
SPLIT_RATIO = {"train": 0.8, "dev": 0.1, "test": 0.1}

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
