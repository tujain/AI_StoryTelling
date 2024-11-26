#!/usr/bin/env python3
# autoencoder.py

import os
import logging
from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    GPT2Tokenizer,
    GPT2Model,
    T5Tokenizer,
    T5EncoderModel,
    RobertaTokenizer,
    RobertaForCausalLM,
    BertTokenizer,
    BertModel,
    BertConfig,
    RobertaConfig,
    set_seed,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Model Definitions
# -------------------------------

class GPT2Encoder(nn.Module):
    def __init__(self, model_name='gpt2', hidden_size: Optional[int] = None):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        if hidden_size and hidden_size != self.model.config.hidden_size:
            logger.info(f"Adjusting GPT-2 hidden size from {self.model.config.hidden_size} to {hidden_size}")
            self.model.config.hidden_size = hidden_size
            self.model.wte = nn.Embedding(self.model.config.vocab_size, hidden_size)
            self.model.wpe = nn.Embedding(self.model.config.n_positions, hidden_size)
            self.model.ln_f = nn.LayerNorm(hidden_size)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]

class T5Encoder(nn.Module):
    def __init__(self, model_name='t5-small', hidden_size: Optional[int] = None):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        if hidden_size and hidden_size != self.model.config.d_model:
            logger.info(f"Adjusting T5 hidden size from {self.model.config.d_model} to {hidden_size}")
            self.model.config.d_model = hidden_size
            # Note: Adjusting T5's hidden size is non-trivial and may require re-initialization
        self.hidden_size = self.model.config.d_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]

class RoBERTaDecoder(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_size: Optional[int] = None):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        config = RobertaConfig.from_pretrained(model_name, is_decoder=True)
        self.model = RobertaForCausalLM.from_pretrained(model_name, config=config)
        if hidden_size and hidden_size != self.model.config.hidden_size:
            logger.info(f"Adjusting RoBERTa hidden size from {self.model.config.hidden_size} to {hidden_size}")
            self.model.config.hidden_size = hidden_size
            # Note: Adjusting hidden sizes may require re-initialization
        self.hidden_size = self.model.config.hidden_size

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

class AutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, hidden_size: Optional[int] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Adjust hidden sizes if necessary
        if self.encoder.hidden_size != self.decoder.hidden_size:
            self.hidden_size = hidden_size or self.decoder.hidden_size
            self.projection = nn.Linear(self.encoder.hidden_size, self.decoder.hidden_size)
            logger.info(f"Added projection layer from encoder hidden size {self.encoder.hidden_size} to decoder hidden size {self.decoder.hidden_size}")
        else:
            self.hidden_size = self.encoder.hidden_size
            self.projection = None

    def forward(self, input_ids_enc, attention_mask_enc=None, labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids_enc, attention_mask=attention_mask_enc)
        # Optionally project encoder outputs to match decoder hidden size
        if self.projection:
            encoder_outputs = self.projection(encoder_outputs)
        # Prepare decoder attention mask
        attention_mask_dec = (labels != -100).long() if labels is not None else None
        # The decoder expects inputs_embeds
        loss, logits = self.decoder(
            inputs_embeds=encoder_outputs,
            attention_mask=attention_mask_dec,
            labels=labels
        )
        return loss, logits, encoder_outputs


    def save(self, output_dir, prefix):
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, f"{prefix}_autoencoder.pth")
        torch.save(self.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(0)))
        logger.info(f"Model loaded from {checkpoint_path}")

# -------------------------------
# Utility Functions and Classes
# -------------------------------

class Feature:
    def __init__(self, input_ids_enc, lm_labels, raw_text=None, cond=None):
        self.input_ids_enc = input_ids_enc
        self.lm_labels = lm_labels
        self.raw_text = raw_text
        self.cond = cond


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, idx):
        feat_dict = self.features[idx]
        feature = Feature(
            input_ids_enc=feat_dict['input_ids_enc'],
            lm_labels=feat_dict['lm_labels'],
            raw_text=feat_dict.get('raw_text', ''),
            cond=feat_dict.get('cond', None)
        )
        return feature

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate_fn(features):
        input_ids_enc_list = [torch.tensor(f.input_ids_enc) for f in features]
        lm_labels_list = [torch.tensor(f.lm_labels) for f in features]

        # Determine the maximum sequence length
        max_length = max(
            max(len(seq) for seq in input_ids_enc_list),
            max(len(seq) for seq in lm_labels_list)
        )

        # Pad sequences to the same length
        input_ids_enc = pad_sequence(
            input_ids_enc_list,
            batch_first=True,
            padding_value=0,
        )

        lm_labels = pad_sequence(
            lm_labels_list,
            batch_first=True,
            padding_value=-100,
        )

        # Ensure both tensors are of the same length
        if input_ids_enc.size(1) != max_length:
            padding = (0, max_length - input_ids_enc.size(1))
            input_ids_enc = nn.functional.pad(input_ids_enc, padding, value=0)

        if lm_labels.size(1) != max_length:
            padding = (0, max_length - lm_labels.size(1))
            lm_labels = nn.functional.pad(lm_labels, padding, value=-100)

        cond = None
        if features[0].cond is not None:
            if isinstance(features[0].cond, (int, str)):
                cond = [f.cond for f in features]
            else:
                cond_list = [torch.tensor(f.cond) for f in features]
                cond_max_length = max([seq.size(0) for seq in cond_list])
                cond = pad_sequence(
                    cond_list,
                    batch_first=True,
                    padding_value=0,
                )
        return input_ids_enc, None, lm_labels, cond


def evaluate_model(model, data_loader, device):
    """
    Evaluates the model on the given data loader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids_enc, _, lm_labels, _ = (t.to(device) if t is not None else None for t in batch)
            loss, logits, _ = model(input_ids_enc=input_ids_enc, labels=lm_labels)
            mask = lm_labels.ne(-100)
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
            predictions = logits.argmax(dim=-1)
            correct = (predictions == lm_labels).masked_select(mask).sum().item()
            total_correct += correct
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    logger.info(f"Evaluation results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Perplexity: {perplexity:.2f}")
    return avg_loss, accuracy, perplexity

# -------------------------------
# Noise Functions
# -------------------------------

class NoiseBase:
    """
    Base class for adding noise to token sequences.
    """
    def __init__(self):
        pass

    def noise(self, batch_tokens):
        """
        Applies noise to a batch of token sequences.
        """
        noised_sequences = []
        for tokens in batch_tokens:
            noised_sequence = self._apply_noise(tokens)
            noised_sequences.append(noised_sequence)
        return pad_sequence(noised_sequences, batch_first=True, padding_value=0)

    def _apply_noise(self, tokens):
        """
        Applies noise to a single token sequence.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class GPT2Noiser(NoiseBase):
    """
    Adds noise to sequences for GPT-2 tokenizer by replacing tokens with random tokens.
    """
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__()
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.vocab_size = len(tokenizer)
    
    def _apply_noise(self, tokens):
        tokens = tokens.clone()
        # Ensure probability_matrix is on the same device as tokens
        probability_matrix = torch.full(tokens.shape, self.mlm_probability, device=tokens.device)
        
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(tokens.tolist(), already_has_special_tokens=True)
        # Ensure special_tokens_mask tensor is on the same device as tokens
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=tokens.device)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Replace masked tokens with random tokens
        random_tokens = torch.randint(0, self.vocab_size, tokens.shape, dtype=torch.long, device=tokens.device)
        tokens[masked_indices] = random_tokens[masked_indices]
        
        return tokens

class BertNoiser(NoiseBase):
    """
    Adds noise to sequences similar to the method used in BERT.
    """
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__()
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)
    
    def _apply_noise(self, tokens):
        tokens = tokens.clone()
        probability_matrix = torch.full(tokens.shape, self.mlm_probability, device=tokens.device)
        
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(tokens.tolist(), already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=tokens.device)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(tokens.shape, 0.8, device=tokens.device)).bool() & masked_indices
        tokens[indices_replaced] = self.mask_token_id
        
        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(tokens.shape, 0.5, device=tokens.device)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(0, self.vocab_size, tokens.shape, dtype=torch.long, device=tokens.device)
        tokens[indices_random] = random_tokens[indices_random]
        
        # The rest of the time, keep the original token (do nothing)
        return tokens


class NoNoise(NoiseBase):
    """
    Does not alter the tokens.
    """
    def __init__(self):
        super().__init__()

    def _apply_noise(self, tokens):
        return tokens

# -------------------------------
# Main Function (Optional)
# -------------------------------

def main():
    # Example usage (not executed when imported)
    pass

if __name__ == '__main__':
    main()