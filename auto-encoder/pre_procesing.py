import argparse
import pdb
import json
import subprocess as sp
import os
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
from transformers import BertTokenizer
from functools import partial
import random
import numpy as np
from multiprocessing import Pool
import re

class Feature:
    def __init__(self, bert_ids=None, gpt2_ids=None, raw_text=None, cond=None):
        self.bert_ids = bert_ids
        self.gpt2_ids = gpt2_ids
        self.raw_text = raw_text
        self.cond = cond

bert_toker = BertTokenizer.from_pretrained('bert-base-uncased')
gpt2_toker = GPT2Tokenizer.from_pretrained('gpt2')

split_ratio = {"train": 0.8, "dev": 0.1, "test": 0.1}

def get_sliding_features(text, maxlen, stride=None):
    """
    Given a long text input, this function:
    1. Encodes it using BERT and GPT-2 tokenizers.
    2. Splits the tokenized sequences into multiple windows of size maxlen.
    3. Returns a list of Feature objects.

    stride: how far to move the window each step. If stride = maxlen, no overlap.
    If stride < maxlen, you'll get overlapping windows.
    """
    if stride is None:
        stride = maxlen  # no overlap by default

    raw_text = text.strip()
    bert_ids_full = bert_toker.encode(raw_text)
    gpt2_ids_full = gpt2_toker.encode(raw_text)

    features = []
    start = 0
    # Slide over the tokenized inputs
    while start < len(bert_ids_full) and start < len(gpt2_ids_full):
        # End index for this window
        end = start + maxlen
        b_chunk = bert_ids_full[start:end]
        g_chunk = gpt2_ids_full[start:end]

        if len(b_chunk) == 0 or len(g_chunk) == 0:
            break

        feature = vars(Feature(b_chunk, g_chunk, raw_text))
        features.append(feature)
        start += stride

    return features

def distributed_main(args):
    files = [os.path.join(args.corpus, fname) for fname in os.listdir(args.corpus) if fname.endswith('.json')]
    pool = Pool()

    for file in tqdm(files, total=len(files)):
        print(f"Processing {file}")
        with open(file, "r", encoding="utf-8") as reader:
            chunk = []
            block = []
            for line in reader:
                parsed_line = json.loads(line)
                text = parsed_line['text']

                if not text.strip():
                    if block:
                        chunk.append(block)
                        block = []
                else:
                    block.append(text)
            # save last chunk
            if block:
                chunk.append(block)
            
            if chunk:
                # Taking only the first chunk as in original code
                data = chunk[0]
                np.random.shuffle(data)

                # Calculate the split indices
                train_split = int(split_ratio["train"] * len(data))
                dev_split = train_split + int(split_ratio["dev"] * len(data))

                dataset = {}
                dataset["train"] = data[:train_split]
                dataset["dev"] = data[train_split:dev_split]
                dataset["test"] = data[dev_split:]
                
                for split_name in split_ratio.keys():
                    if len(dataset[split_name]) != 0:
                        # Instead of directly calling get_feature on each line,
                        # we now first convert each line to multiple features via sliding window
                        all_features = []
                        for text_line in dataset[split_name]:
                            # Generate multiple features for each long line
                            sliding_feats = get_sliding_features(text_line, args.maxlen)
                            all_features.extend(sliding_feats)

                        # Now we have a flattened list of features from all lines
                        # pool.map is not needed for tokenization if we already did it above,
                        # but if we wanted to still use parallelization for some reason, we can.
                        # Here, all_features are already processed, so we can directly save them.
                        
                        # Make sure directories exist
                        parsed_dir = os.path.join(os.path.dirname(file), 'parsed', split_name)
                        if not os.path.exists(parsed_dir):
                            os.makedirs(parsed_dir)
                        
                        # Save the preprocessed features
                        torch.save(all_features, os.path.join(parsed_dir, f'{os.path.basename(file)[:-5]}_{split_name}.pt'))

    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='directory of training corpus')
    parser.add_argument('--maxlen', type=int, default=256,
                        help='maximum length of the sequence for each window')
    args = parser.parse_args()

    if os.path.isdir(args.corpus):
        distributed_main(args)
