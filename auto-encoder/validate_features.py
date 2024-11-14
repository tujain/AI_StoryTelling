import argparse
import os
import logging
import torch
from typing import List, Dict, Any
from transformers import (
    DistilBertTokenizer,
    GPT2Tokenizer,
    RobertaTokenizer,
    T5Tokenizer,
)
from tqdm import tqdm

# Define maximum token limits for each tokenizer
TOKENIZER_MAX_LIMITS = {
    'bert': 512,
    'gpt2': 1024,
    'roberta': 512,
    't5': 512,
}

def initialize_tokenizers() -> Dict[str, Any]:
    """
    Initialize all required tokenizers.

    Returns:
        Dict[str, Any]: Dictionary of initialized tokenizers.
    """
    try:
        bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
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

def validate_feature(feature: Dict[str, Any], split_name: str, index: int) -> List[str]:
    """
    Validate a single feature.

    Args:
        feature (Dict[str, Any]): The feature dictionary to validate.
        split_name (str): Name of the data split (train/dev/test).
        index (int): Index of the feature in the dataset.

    Returns:
        List[str]: List of validation error messages. Empty if valid.
    """
    errors = []
    required_fields = ['bert_ids', 'gpt2_ids', 'roberta_ids', 't5_ids', 'raw_text', 'chunk_id']

    # Check for missing fields
    for field in required_fields:
        if field not in feature:
            errors.append(f"Missing field '{field}'.")
    
    if errors:
        return errors  # No need to proceed further if fields are missing

    # Validate raw_text
    raw_text = feature['raw_text']
    if not isinstance(raw_text, str) or not raw_text.strip():
        errors.append("Field 'raw_text' is empty or not a string.")

    # Validate chunk_id
    chunk_id = feature['chunk_id']
    if not isinstance(chunk_id, int):
        errors.append("Field 'chunk_id' is not an integer.")

    # Validate token IDs
    token_fields = ['bert_ids', 'gpt2_ids', 'roberta_ids', 't5_ids']
    for tokenizer, field in zip(['bert', 'gpt2', 'roberta', 't5'], token_fields):
        token_ids = feature[field]
        max_limit = TOKENIZER_MAX_LIMITS[tokenizer]
        if not isinstance(token_ids, list):
            errors.append(f"Field '{field}' is not a list.")
            continue
        if not all(isinstance(token, int) for token in token_ids):
            errors.append(f"Field '{field}' contains non-integer tokens.")
        if len(token_ids) > max_limit:
            errors.append(f"Field '{field}' exceeds the maximum token limit of {max_limit}: {len(token_ids)} tokens.")
    
    return errors

def validate_pt_file(pt_file_path: str, tokenizers: Dict[str, Any], split_name: str) -> Dict[str, Any]:
    """
    Validate all features in a .pt file.

    Args:
        pt_file_path (str): Path to the .pt file.
        tokenizers (Dict[str, Any]): Initialized tokenizers.
        split_name (str): Name of the data split.

    Returns:
        Dict[str, Any]: Summary of validation results.
    """
    try:
        features = torch.load(pt_file_path)
        if not isinstance(features, list):
            logging.error(f"The file {pt_file_path} does not contain a list of features.")
            return {"total": 0, "valid": 0, "invalid": 1, "errors": ["File does not contain a list of features."]}
    except Exception as e:
        logging.error(f"Failed to load {pt_file_path}: {e}")
        return {"total": 0, "valid": 0, "invalid": 1, "errors": [f"Failed to load file: {e}"]}
    
    total = len(features)
    valid = 0
    invalid = 0
    error_messages = []

    for idx, feature in enumerate(tqdm(features, desc=f"Validating {os.path.basename(pt_file_path)}")):
        errors = validate_feature(feature, split_name, idx)
        if not errors:
            valid += 1
        else:
            invalid += 1
            error_messages.append(f"Feature {idx} errors: " + "; ".join(errors))
    
    return {"total": total, "valid": valid, "invalid": invalid, "errors": error_messages}

def traverse_parsed_directory(parsed_dir: str) -> List[Dict[str, str]]:
    """
    Traverse the parsed directory and collect all .pt files.

    Args:
        parsed_dir (str): Path to the parsed directory.

    Returns:
        List[Dict[str, str]]: List of dictionaries with 'split' and 'file_path'.
    """
    splits = ['train', 'dev', 'test']
    pt_files = []
    for split in splits:
        split_dir = os.path.join(parsed_dir, split)
        if not os.path.isdir(split_dir):
            logging.warning(f"Split directory '{split_dir}' does not exist. Skipping.")
            continue
        for fname in os.listdir(split_dir):
            if fname.endswith('.pt'):
                pt_files.append({"split": split, "file_path": os.path.join(split_dir, fname)})
    return pt_files

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate preprocessed .pt feature files.")
    parser.add_argument(
        '--parsed_dir',
        required=True,
        type=str,
        help='Path to the parsed directory containing train/dev/test subdirectories with .pt files.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("validation.log"),
            logging.StreamHandler()
        ]
    )

    # Initialize tokenizers (optional: can be removed if not needed for deeper validation)
    tokenizers = initialize_tokenizers()

    # Traverse parsed directory to find all .pt files
    pt_files = traverse_parsed_directory(args.parsed_dir)
    if not pt_files:
        logging.error(f"No .pt files found in the parsed directory '{args.parsed_dir}'. Exiting.")
        return

    logging.info(f"Found {len(pt_files)} .pt files for validation.")

    # Initialize summary statistics
    summary = {
        "train": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
        "dev": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
        "test": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
    }

    # Validate each .pt file
    for file_info in pt_files:
        split = file_info["split"]
        file_path = file_info["file_path"]
        logging.info(f"Validating file: {file_path} (Split: {split})")
        result = validate_pt_file(file_path, tokenizers, split)
        summary[split]["total"] += result["total"]
        summary[split]["valid"] += result["valid"]
        summary[split]["invalid"] += result["invalid"]
        summary[split]["errors"].extend(result["errors"])
    
    # Print summary report
    print("\n=== Validation Summary ===")
    for split in ['train', 'dev', 'test']:
        print(f"\nSplit: {split}")
        print(f"Total Features: {summary[split]['total']}")
        print(f"Valid Features: {summary[split]['valid']}")
        print(f"Invalid Features: {summary[split]['invalid']}")
        if summary[split]['errors']:
            print("Errors:")
            for error in summary[split]['errors']:
                print(f"  - {error}")
        else:
            print("No errors found.")
    
    # Save summary to log
    logging.info("=== Validation Summary ===")
    for split in ['train', 'dev', 'test']:
        logging.info(f"\nSplit: {split}")
        logging.info(f"Total Features: {summary[split]['total']}")
        logging.info(f"Valid Features: {summary[split]['valid']}")
        logging.info(f"Invalid Features: {summary[split]['invalid']}")
        if summary[split]['errors']:
            logging.info("Errors:")
            for error in summary[split]['errors']:
                logging.info(f"  - {error}")
        else:
            logging.info("No errors found.")

if __name__ == "__main__":
    main()
