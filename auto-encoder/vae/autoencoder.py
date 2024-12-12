import os
import torch 
import json
import math
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, GPT2Tokenizer
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
from transformers import MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer, BertModel, BertTokenizer
import argparse
import logging
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import BertModel, BertTokenizer, BertConfig, BertLayer, T5ForConditionalGeneration, T5Config, T5Tokenizer, T5EncoderModel, T5Model
from transformers.models.bert.modeling_bert import BertLMPredictionHead
# from generation_utils import beam_search_naive, prepare_for_bleu, generate_sequence
import math
from os.path import join
import copy
import itertools
import tqdm
from tqdm import trange
import pdb

import torch
from tqdm import trange
import torch.nn.functional as F
import numpy as np
import logging
import pdb

# GPT
EOS_ID = 50256
# Bert
SEP_ID = 102
PAD_ID= 0
# T5
PAD_ID_T5 = 0
SEP_ID_T5 = 1
def prepare_for_bleu(sentence, tokenizer, skip_special_tokens = False):
    sent=[]
    tokenizer_name = tokenizer.__class__.__name__
    if skip_special_tokens:
        end_of_sentence = {'BertTokenizer': [], 'GPT2Tokenizer': [], 'T5Tokenizer': []}
    else:
        end_of_sentence = {'BertTokenizer': [SEP_ID, PAD_ID], 'GPT2Tokenizer': [EOS_ID], 'T5Tokenizer': [SEP_ID_T5, PAD_ID_T5],}
    for s in sentence[1:]:
        if s not in end_of_sentence[tokenizer_name]:
            sent.append(s)
        else:
            break
    return sent


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)



def generate_next_token(model_gpt, prev, temperature=1, top_k = 0, top_p=1.0, sample=False, past=None):

    with torch.no_grad():
        #pdb.set_trace()
        gpt_output = model_gpt.transformer(prev, position_ids=None, token_type_ids=None, past_key_values=past)
        hidden_states, past = gpt_output['last_hidden_state'], gpt_output['past_key_values']
        logits = model_gpt.lm_head(hidden_states)
        logits = logits[:, -1, :] / temperature
        if top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        else:
            logits = top_k_logits(logits, k=top_k)
        probs = F.softmax(logits, dim=-1)


        if sample:
            prev = torch.multinomial(probs, num_samples=1)
            return prev, probs[0][prev], past
        else:
            probs_sel, prev = torch.topk(probs, k=top_k, dim=-1)
            return prev, probs_sel, past

###########################################################################
# Beam search based on ottokart/beam_search
###########################################################################
class Node(object):
    def __init__(self, parent, state, value, cost):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent  # parent Node, None for root
        self.state = state
        self.length = 1 if parent is None else parent.length + 1
        self.cum_cost = parent.cum_cost*(self.length-1)/self.length + cost/self.length if parent else cost
        # self.cum_cost = parent.cum_cost + cost if parent else cost
        self._sequence = None

    # def __repr__(self):
    #    return f'value = {self.value}, parent = {self.parent.value}, cost = {self.cum_cost}'

def beam_search_naive(model_gpt, bs, length=48, beam_width=3, beam_examples=1, past=None):
    """
    currently it does NOT support batch parallel
    """

    all_decode, all_decode_losses = [], []
    for b in range(bs):
        next_fringe = [Node(parent=None, state=past, value=EOS_ID, cost=0.0)]
        results = []
        for i in range(length):
            fringe, all_prev, all_probs, all_past = [], torch.Tensor(0).long().cuda(), [], []
            for nn in next_fringe:
                if (nn.value == EOS_ID) and (i>0):
                    results.append(nn)
                    continue
                else:
                    fringe.extend([nn]*beam_width)

                if not fringe:
                    break

                prev, probs, past = generate_next_token(model_gpt, torch.Tensor([[nn.value]]).long().cuda(), temperature=1, top_k=beam_width, sample=False, past=nn.state)
                # pdb.set_trace()

                log_probs = torch.log(probs)[0]
                all_prev = torch.cat((all_prev, prev[0]))
                all_probs.extend(log_probs.tolist())
                all_past.extend([past]*len(log_probs))


            next_fringe = []
            for prev, log_probs, past, nn in zip(all_prev, all_probs, all_past, fringe):
                new_node = Node(parent=nn, state=past, value=prev.item(), cost=log_probs)
                next_fringe.append(new_node)

            next_fringe = sorted(next_fringe, key=lambda nn: nn.cum_cost, reverse=True)[:beam_width]

        results.extend(next_fringe)

        results.sort(key=lambda nn : nn.cum_cost, reverse=True)

        if beam_examples == 1:
            # Single response version
            best_result = results[0].parent
            decode, decode_loss = [], []
            while best_result.value != EOS_ID:
                decode.append(best_result.value)
                decode_loss.append(best_result.cum_cost)
                best_result = best_result.parent
            decode.append(best_result.value)
            decode_loss.append(best_result.cum_cost)
            decode, decode_loss = decode[::-1], decode_loss[::-1]
            all_decode.append(decode)
            all_decode_losses.append(decode_loss)
        else:
            # Top beam_n_examples 
            best_results = results[:beam_examples]
            sent_all_decode, sent_all_decode_losses = [],[]
            for best_result in best_results:
                decode, decode_loss = [], []
                while best_result.value != -1:
                    decode.append(best_result.value)
                    decode_loss.append(best_result.cum_cost)
                    best_result = best_result.parent
                decode, decode_loss = decode[::-1], decode_loss[::-1]
                sent_all_decode.append(decode)
                sent_all_decode_losses.append(decode_loss)
            all_decode.append(sent_all_decode)
            all_decode_losses.append(sent_all_decode_losses)

    if beam_examples == 1:
        output = torch.nn.utils.rnn.pad_sequence([torch.tensor(f, dtype=torch.long) for f in all_decode], batch_first=True, padding_value=EOS_ID)
    else:
        output = torch.nn.utils.rnn.pad_sequence([torch.tensor(s, dtype=torch.long) for s in all_decode[0]], batch_first=True, padding_value=EOS_ID)

    return output




def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



def generate_sequence(model, temperature=1, top_k=1, top_p = 1.0, length=20, sample=False, past=None, device='cuda'):
    output = past[0][0].new_zeros([past[0][0].size(0),0])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    bsz = past[0][0].size(0)
    prev = torch.Tensor([EOS_ID]*bsz).long().cuda().unsqueeze(1)
    output = torch.cat((output, prev), dim=1)
    for i in range(length):
        prev, probs, past = generate_next_token(model, prev, temperature=temperature, top_k=top_k, top_p=top_p, sample=sample, past=past)
        output = torch.cat((output, prev), dim=1)
    return output

class Feature:
    def __init__(self, bert_ids, gpt2_ids, raw_text, cond=None):
        self.input_ids_bert = bert_ids
        self.input_ids_dec = [EOS_ID] + gpt2_ids
        self.lm_labels = gpt2_ids + [EOS_ID]
        if cond is not None:
            self.cond = cond

class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
class FeatureDataset(Dataset):
    """ pytorch dataset for GPT2 training """

    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        feat_dict = self.features[i]
        feat = Feature(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        input_ids_bert = pad_sequence([torch.tensor(f.input_ids_bert)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        input_ids_dec = pad_sequence([torch.tensor(f.input_ids_dec, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=0)
        lm_labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        if not hasattr(features[0], 'cond'):
            cond = [None for f in features]
        else:
            if isinstance(features[0].cond, int) or isinstance(features[0].cond, str):
                cond = [f.cond for f in features]
            else: #cont feature
                cond = pad_sequence([torch.tensor(f.cond)
                               for f in features],
                              batch_first=True, padding_value=0)

        return (input_ids_bert, input_ids_dec, lm_labels, cond)


class BucketingDataLoader(object):
    """ this loads pt chunks and then convert to mini-batch loader"""
    def __init__(self, pt_name, batch_size, max_seq_len,
                 bucket=100, shuffle=True,
                 rank=0, num_replica=1):

        self.pt_name = pt_name
        self.batch_size = batch_size
        self.max_len = max_seq_len
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.rank = rank
        self.num_replica = num_replica


    def __iter__(self):
        chunk = torch.load(self.pt_name)[self.rank::self.num_replica]
        # discard long examples
        trunc_chunk = []
        lens = []
        total = len(chunk)
        for feat in chunk:
            if len(feat['gpt2_ids'])+2 > tokenizer_gpt2.max_len_single_sentence or len(feat['bert_ids']) > tokenizer_bert.max_len_single_sentence:
                continue
            tot_len = len(feat['gpt2_ids'])
            if tot_len > self.max_len:
                feat['gpt2_ids'] = feat['gpt2_ids'][:self.max_len]
            if len(feat['bert_ids']) > self.max_len:
                feat['bert_ids'] = feat['bert_ids'][:self.max_len]
            trunc_chunk.append(feat)
            lens.append(tot_len)
        print (f"{self.pt_name}: rank {self.rank} has chunks {len(trunc_chunk)}/{total}, batch_size: {self.batch_size}")
        print (f"{self.pt_name}: rank {self.rank} has chunks {len(trunc_chunk)}/{total}, total iterations: {len(trunc_chunk)//self.batch_size}")
        dataset = FeatureDataset(trunc_chunk)
        sampler = BucketSampler(lens, self.bucket_size, self.batch_size,
                                droplast=True, shuffle=self.shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=0,  # can test multi-worker
                            collate_fn=FeatureDataset.collate)
        yield from loader

    def __len__(self):
        raise NotImplementedError()

class DistributedBucketingDataLoader(object):
    """ distributed version """
    def __init__(self, db_dir, *args, **kwargs):
        self.db_dir = db_dir
        self.args = args
        self.kwargs = kwargs

    def _get_files(self):
        files = [os.path.join(self.db_dir, fname) for fname in os.listdir(self.db_dir) if fname.endswith('.pt')]
        files.sort()
        if not ('shuffle' in self.kwargs and self.kwargs['shuffle'] == False):
            random.shuffle(files)
        return files

    def __iter__(self):
        for db_name in self._get_files():
            loader = BucketingDataLoader(db_name, *self.args, **self.kwargs)
            yield from loader

class InfiniteDistributedBucketingDataLoader(DistributedBucketingDataLoader):
    def __init__(self, db_dir, *args, **kwargs):
        super().__init__(db_dir, *args, **kwargs)
        self.iterator = iter(self)

    def __iter__(self):
        while True:
            for db_name in self._get_files():
                loader = BucketingDataLoader(db_name, *self.args, **self.kwargs)
                yield from loader

    def __next__(self):
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self)

class TextDataset(Dataset):
    def __init__(self, prefix_path, tokenizer, max_length=16, device=torch.device("cpu")):
        # Load the prefixes from a file
        with open(prefix_path, "r") as f:
            prefixes = [line.strip() for line in f]
        self.input_ids = [tokenizer.encode(prefix, padding='max_length', max_length=max_length, return_tensors="pt").to(device) for prefix in prefixes]

    def __getitem__(self, idx):
        return 0, 0, 0, self.input_ids[idx]

    def __len__(self):
        return len(self.input_ids)

class TextDataLoader(object):
    def __init__(self, prefix_path, tokenizer, batch_size, max_length=256, device=torch.device("cpu")):
        self.dataset = TextDataset(prefix_path, tokenizer, max_length, device)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

    def __iter__(self):
        return iter(self.loader)


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    rep_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score


def cal_most_freq(generated, k=5):
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    top_k_counter = [defaultdict(int), defaultdict(int),
                    defaultdict(int), defaultdict(int)]
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        sorted_counter = sorted(counter[n].items(), key=lambda item: item[1], reverse=True)
        for key, value in sorted_counter[:k]:
            top_k_counter[n][key] = float(value/total)
    return top_k_counter

def eval_model_loss(model, gpt_model, gpt_tokenizer, ae_step, eval_dataloader, noiser, device, logger, max_valid_size = None, onlypart=False, ground=False):
    if onlypart:
        print ('WARNING! Only part of the dev data evaluated!')
    tot_loss, tot_correct, tot_tokens = 0., 0, 0
    tot_gpt_tokens = 0
    tot_nll_mid = 0
    input_sentence, input_sentence_corrupted, predict_sentence = [], [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_dataloader, desc=f"validation_epoch{ae_step}"):
            if ground:
                input_ids_bert, input_ids_dec = batch[3], batch[1]
            else:
                input_ids_bert, input_ids_dec = batch[0], batch[1]
            input_ids_enc = noiser.noise(input_ids_bert)
            tokenizer_name = model.decoder.tokenizer.__class__.__name__
            input_ids_dec_cls = {'BertTokenizer': input_ids_bert, 'GPT2Tokenizer': input_ids_dec}
            batch = (input_ids_enc, input_ids_dec_cls[tokenizer_name], ) + batch[2:]
            batch = tuple(t.to(device) for t in batch[:3])
            input_ids_enc, input_ids_dec, lm_labels = batch

            loss, correct, ntokens, h = model(input_ids_enc, input_ids_dec, lm_labels)
            tot_loss += loss.item() * ntokens
            tot_correct += correct.item()
            tot_tokens += ntokens.item()

            h_all = model.encoder_mean(input_ids_bert[:2,:].to(device))
            h_0, h_N = h_all[0], h_all[1]
            h_mid = (h_0+h_N)/2
            resp = model.generate_from(h_mid.unsqueeze(0))[0]
            nll_mid, n_gpt_tokens = eval_gpt2(gpt_model, gpt_tokenizer, resp)
            tot_nll_mid += nll_mid.item() * n_gpt_tokens
            tot_gpt_tokens += n_gpt_tokens


           

            if tot_tokens > 128 * 256 * 1024 and onlypart:
                break

            input_sentence += model.decode(input_ids_bert)
            input_sentence_corrupted += model.decode(input_ids_enc)
            predict_sentence += [x.lower() for x in model.generate_from(h)]
            if max_valid_size and len(input_sentence) >= max_valid_size:
                break

        if tot_tokens == 0 or tot_gpt_tokens == 0:
            logger.warning("Evaluation skipped due to no tokens processed!")
            return float('inf'), float('inf'), float('inf'), 0, 0, 0            

        loss = tot_loss/tot_tokens
        nll_mid_avg = tot_nll_mid/tot_gpt_tokens
        ppl = torch.exp(torch.tensor(loss))
        mid_ppl = torch.exp(torch.tensor(nll_mid_avg))
        acc = tot_correct / tot_tokens
        input_sentence = [t.strip() for t in input_sentence]
        predict_sentence = [t.strip() for t in predict_sentence]



        _, _, rouge_l = calc_rouge(input_sentence, predict_sentence)
        bleu = calc_bleu(input_sentence, predict_sentence)
        # self_bleu = calc_self_bleu(predict_sentence)
    batch_size = input_ids_bert.shape[0]
    logger.info('Validation:')
    logger.info('Steps: {}, '
                'Loss: {}, '
                'PPL: {}, '
                'Acc: {}, '
                'Int_PPL: {}, '
                'Rouge: {}, '
                'Robust BLEU: {}, '.format(ae_step, loss.item(), ppl.item(), acc, mid_ppl.item(), rouge_l, bleu))
    
    rand_id = torch.randint(batch_size, (1,))[0]
    logger.info("Input Sentence:")
    logger.info(input_sentence[rand_id].strip())
    logger.info("Corrupted Sentence:")
    logger.info(input_sentence_corrupted[rand_id].strip())
    logger.info("Output Sentence:")
    logger.info(predict_sentence[rand_id].strip())

        
    return loss.item(), ppl.item(), mid_ppl.item(), acc, rouge_l, bleu

def load_cond_model(model):
    if model is None or model == "None":
        return None
    elif model.startswith('Helsinki-NLP'):
        assert model in ['Helsinki-NLP/opus-mt-zh-en', 'Helsinki-NLP/opus-mt-en-de']
        cond_model = MarianMTModel.from_pretrained(model).encoder
        cond_tokenizer = MarianTokenizer.from_pretrained(model)
    elif model.startswith("t5"):
        # check if this is a valid t5 model. 
        assert model in ['t5-small', 't5-base', 't5-large']
        cond_model = T5ForConditionalGeneration.from_pretrained(model).encoder
        cond_tokenizer = T5Tokenizer.from_pretrained(model)
    elif model.startswith("bert"):
        assert model in ['bert-base-uncased', 'bert-medium-uncased', 'bert-large-uncased']
        cond_model = BertModel.from_pretrained(model)
        cond_tokenizer = BertTokenizer.from_pretrained(model)
    else:
        raise NotImplementedError

    return cond_model, cond_tokenizer

def generate_hidden(model, dataloader, noiser, device, cond_model = None, max_size = None, ground = False, h_noiser='none'):
    if max_size==-1:
        max_size = None
    hiddens = []
    cond = []
    cond_model = load_cond_model(cond_model)
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc=f"generate hidden"):
            if ground:
                input_ids_bert = batch[3]
            else:
                input_ids_bert = batch[0]
            input_ids_enc = noiser.noise(input_ids_bert)
            input_ids_enc = input_ids_enc.to(device)
            # if gpu:
            if h_noiser == 'vae':  
                mean = model.encoder_mean(input_ids_enc)
                log_var = model.encoder_log_var(input_ids_enc)
                sampled_h = model.reparameterize(mean, log_var)
                hiddens.extend([h for h in sampled_h])
            else:
                hiddens.extend([h for h in model.encoder_mean(input_ids_enc)])
            # else:
            #     hiddens.extend([h.cpu().numpy().astype(np.float16) for h in model.encoder_mean(input_ids_enc)])
            if len(batch)>3:
                cond_feature = (not isinstance(batch[3], list)) and (not any(np.array(batch[3].view(-1)) == None)) and batch[3].ndim > 1
                if cond_feature:
                    input_ids_bert_cond = noiser.noise(batch[3]).to(device)
                    if cond_model is None:
                        cond.extend([c.view(c.shape[0], -1) for c in model.encoder_mean(input_ids_bert_cond)])
                    else:
                        cond.extend([c.view(c.shape[0], -1) for c in cond_model(input_ids_bert_cond).last_hidden_state])
                else:
                    cond.extend(batch[3])
            if max_size and len(hiddens) >= max_size:
                break        
    return hiddens, cond


def calc_rouge(original_sentences, predict_sentences, default= "None"):
    rouge_1 = 0.0
    rouge_2 = 0.0
    rouge_l = 0.0
    num_sample = len(original_sentences)
    for original, predict in zip(original_sentences, predict_sentences):
        # Remove padding
        original, predict = original.replace("<PAD>", "").strip(), predict.replace("<PAD>", "").strip()
        if original == "": original = default
        if predict == "": predict = default
        rouge = RougeCalculator(stopwords=True, lang="en")
        r1 = rouge.rouge_1(summary=predict, references=original)
        r2 = rouge.rouge_2(summary=predict, references=original)
        rl = rouge.rouge_l(summary=predict, references=original)
        rouge_1 += r1
        rouge_2 += r2
        rouge_l += rl
    return rouge_1/num_sample, rouge_2/num_sample, rouge_l/num_sample

def calc_bleu(original_sentences, predict_sentences, default= "None"):
    bleu = 0.0
    num_sample = len(original_sentences)
    for original, predict in zip(original_sentences, predict_sentences):
        # Remove padding
        
        original, predict = original.replace("<PAD>", "").strip(), predict.replace("<PAD>", "").strip()
        if original == "": original = default
        if predict == "": predict = default
        b = BLEUCalculator(lang="en").bleu(summary=predict, references=original)
        bleu += b
    return bleu/num_sample

def calc_self_bleu(predict_sentences, default= "None"):
    bleu = 0.0
    cnt = 0
    for i, p1 in enumerate(predict_sentences):
        for p2 in predict_sentences[i+1:]:
            # Remove padding
            if p2 == "": p2 = default
            if p1 == "": p1 = default

            b = BLEUCalculator(lang="en").bleu(summary=p1.replace("<PAD>", "").strip(), references=p2.replace("<PAD>", "").strip())
            bleu += b
            cnt += 1
    return bleu/cnt

def compute_last_kernel_shape(args):
    t = args.sentence_len + 2 * (args.filter_shape - 1)
    for _ in range(args.num_layer-1):
        t = int(math.floor((t - args.filter_shape) / 2) + 1)
    return t

    # last_kernel_shape = compute_last_kernel_shape(args)

def eval_gpt2(model, tokenizer, text, max_len = 512):
    model = model.cuda()
    if len(text) == 0:
        text = "No text"
    encoded_input = tokenizer(text, return_tensors='pt')
    # encoded_input = tokenizer.batch_encode_plus(text, padding=True, truncation=True)
    input_ids = encoded_input.input_ids
    # input_ids = torch.LongTensor(input_ids)
    input_ids = input_ids[:,:max_len].cuda()
    n_token = input_ids.size(1)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return outputs[0], n_token

def prepare_for_bleu(sentence, tokenizer, skip_special_tokens = False):
    sent=[]
    tokenizer_name = tokenizer.__class__.__name__
    if skip_special_tokens:
        end_of_sentence = {'BertTokenizer': [], 'GPT2Tokenizer': [], 'T5Tokenizer': []}
    else:
        end_of_sentence = {'BertTokenizer': [SEP_ID, PAD_ID], 'GPT2Tokenizer': [EOS_ID], 'T5Tokenizer': [SEP_ID_T5, PAD_ID_T5],}
    for s in sentence[1:]:
        if s not in end_of_sentence[tokenizer_name]:
            sent.append(s)
        else:
            break
    return sent


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)



def generate_next_token(model_gpt, prev, temperature=1, top_k = 0, top_p=1.0, sample=False, past=None):

    with torch.no_grad():
        #pdb.set_trace()
        gpt_output = model_gpt.transformer(prev, position_ids=None, token_type_ids=None, past_key_values=past)
        hidden_states, past = gpt_output['last_hidden_state'], gpt_output['past_key_values']
        logits = model_gpt.lm_head(hidden_states)
        logits = logits[:, -1, :] / temperature
        if top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        else:
            logits = top_k_logits(logits, k=top_k)
        probs = F.softmax(logits, dim=-1)


        if sample:
            prev = torch.multinomial(probs, num_samples=1)
            return prev, probs[0][prev], past
        else:
            probs_sel, prev = torch.topk(probs, k=top_k, dim=-1)
            return prev, probs_sel, past

###########################################################################
# Beam search based on ottokart/beam_search
###########################################################################
class Node(object):
    def __init__(self, parent, state, value, cost):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent  # parent Node, None for root
        self.state = state
        self.length = 1 if parent is None else parent.length + 1
        self.cum_cost = parent.cum_cost*(self.length-1)/self.length + cost/self.length if parent else cost
        # self.cum_cost = parent.cum_cost + cost if parent else cost
        self._sequence = None

    # def __repr__(self):
    #    return f'value = {self.value}, parent = {self.parent.value}, cost = {self.cum_cost}'

def beam_search_naive(model_gpt, bs, length=48, beam_width=3, beam_examples=1, past=None):
    """
    currently it does NOT support batch parallel
    """

    all_decode, all_decode_losses = [], []
    for b in range(bs):
        next_fringe = [Node(parent=None, state=past, value=EOS_ID, cost=0.0)]
        results = []
        for i in range(length):
            fringe, all_prev, all_probs, all_past = [], torch.Tensor(0).long().cuda(), [], []
            for nn in next_fringe:
                if (nn.value == EOS_ID) and (i>0):
                    results.append(nn)
                    continue
                else:
                    fringe.extend([nn]*beam_width)

                if not fringe:
                    break

                prev, probs, past = generate_next_token(model_gpt, torch.Tensor([[nn.value]]).long().cuda(), temperature=1, top_k=beam_width, sample=False, past=nn.state)
                # pdb.set_trace()

                log_probs = torch.log(probs)[0]
                all_prev = torch.cat((all_prev, prev[0]))
                all_probs.extend(log_probs.tolist())
                all_past.extend([past]*len(log_probs))


            next_fringe = []
            for prev, log_probs, past, nn in zip(all_prev, all_probs, all_past, fringe):
                new_node = Node(parent=nn, state=past, value=prev.item(), cost=log_probs)
                next_fringe.append(new_node)

            next_fringe = sorted(next_fringe, key=lambda nn: nn.cum_cost, reverse=True)[:beam_width]

        results.extend(next_fringe)

        results.sort(key=lambda nn : nn.cum_cost, reverse=True)

        if beam_examples == 1:
            # Single response version
            best_result = results[0].parent
            decode, decode_loss = [], []
            while best_result.value != EOS_ID:
                decode.append(best_result.value)
                decode_loss.append(best_result.cum_cost)
                best_result = best_result.parent
            decode.append(best_result.value)
            decode_loss.append(best_result.cum_cost)
            decode, decode_loss = decode[::-1], decode_loss[::-1]
            all_decode.append(decode)
            all_decode_losses.append(decode_loss)
        else:
            # Top beam_n_examples 
            best_results = results[:beam_examples]
            sent_all_decode, sent_all_decode_losses = [],[]
            for best_result in best_results:
                decode, decode_loss = [], []
                while best_result.value != -1:
                    decode.append(best_result.value)
                    decode_loss.append(best_result.cum_cost)
                    best_result = best_result.parent
                decode, decode_loss = decode[::-1], decode_loss[::-1]
                sent_all_decode.append(decode)
                sent_all_decode_losses.append(decode_loss)
            all_decode.append(sent_all_decode)
            all_decode_losses.append(sent_all_decode_losses)

    if beam_examples == 1:
        output = torch.nn.utils.rnn.pad_sequence([torch.tensor(f, dtype=torch.long) for f in all_decode], batch_first=True, padding_value=EOS_ID)
    else:
        output = torch.nn.utils.rnn.pad_sequence([torch.tensor(s, dtype=torch.long) for s in all_decode[0]], batch_first=True, padding_value=EOS_ID)

    return output




def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



def generate_sequence(model, temperature=1, top_k=1, top_p = 1.0, length=20, sample=False, past=None, device='cuda'):
    output = past[0][0].new_zeros([past[0][0].size(0),0])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    bsz = past[0][0].size(0)
    prev = torch.Tensor([EOS_ID]*bsz).long().cuda().unsqueeze(1)
    output = torch.cat((output, prev), dim=1)
    for i in range(length):
        prev, probs, past = generate_next_token(model, prev, temperature=temperature, top_k=top_k, top_p=top_p, sample=sample, past=past)
        output = torch.cat((output, prev), dim=1)
    return output

# current_path = os.path.dirname(os.path.abspath(__file__))

def print_chkpt_info(loading_info, chkpt_state_dict = None, model = None):
    missing_keys, unexpected_keys, mismatched_keys = loading_info["missing_keys"], loading_info["unexpected_keys"], loading_info["mismatched_keys"]

    if chkpt_state_dict is not None and model is not None:
        # Get all keys from the state dictionary
        all_state_dict_keys = set(chkpt_state_dict.keys())

        # Get all keys from the model's state dictionary
        all_model_keys = set(model.state_dict().keys())

        # Properly loaded keys are the intersection of model keys and state dictionary keys
        properly_loaded_keys = all_state_dict_keys.intersection(all_model_keys)

        print("Properly loaded:", properly_loaded_keys)

    # Any missing or unexpected keys indicate a problem
    if missing_keys:
        print("Warning: Missing keys in state_dict:", missing_keys)

    if unexpected_keys:
        print("Warning: Unexpected keys in state_dict:", unexpected_keys)

    if mismatched_keys:
        print("Warning: Mismatched keys in state_dict:", mismatched_keys)

    # If there are no missing and unexpected keys, the weights are properly loaded.
    if not missing_keys and not unexpected_keys and not mismatched_keys:
        print("All weights are properly loaded.")

class ConvModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.sentence_len = args.sentence_len
        self.padding = lambda x, filter_s: nn.ConstantPad1d((filter_s-1, self.sentence_len-x.shape[1]+filter_s-1), self.tokenizer.pad_token_id)(x)

    def _BN(self, shape):
        return nn.BatchNorm2d(shape)

    def _LN(self, shape, device = 'cuda'):
        # nasty way to save encoder checkpoint
        # global _permute
        f = nn.LayerNorm(shape, device = device) 
        def _permute(x):
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return _permute

    def _InitWeight(self, module):        
        # weight initialize for conv_transpose layer
        for m in self.modules():
            if isinstance(m, module):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def load(self, state_dict):
        self.load_state_dict(state_dict)

    def save(self, output_dir, prefix):
        torch.save(self.state_dict(), os.path.join(output_dir, f"{prefix}-{self.file_name}"))

    @classmethod
    def from_pretrained(cls, encoder, input_dir, prefix, *unused):
        ckpt = torch.load(os.path.join(input_dir, f"{prefix}-{cls.file_name}"), map_location='cpu')
        encoder['model_args']['state_dict'] = ckpt
        return encoder

    @classmethod
    def compute_last_filter_shape(cls, args):
        t = args.sentence_len + 2 * (args.filter_shape - 1)
        for _ in range(args.num_layer-1):
            t = int(math.floor((t - args.filter_shape) / 2) + 1)
        return t - args.num_feature * 2 + 2



class ConvolutionEncoder(ConvModel):
    file_name = 'CNN.pkl'
    def __init__(self, args, state_dict = None):
        super().__init__(args)
        last_filter_shape = ConvModel.compute_last_filter_shape(args)
        # assert(args.num_feature == final_len)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embed = nn.Embedding(self.tokenizer.vocab_size, args.embed_dim)
        self.filter_shape = args.filter_shape
        self.file_name = 'CNN.pkl'
        
        embed_size = self.embed.weight.size()[1]
        if args.reg_layer == 'none':
            self.reg_layer = nn.ModuleList([nn.Identity() for l in range(args.num_layer)])
        elif args.reg_layer == 'bn':
            self.reg_layer = nn.ModuleList([self._BN(embed_size)] + [self._BN(args.filter_size * (2 ** l)) for l in range(0,args.num_layer-1)])
        elif args.reg_layer == 'ln':
            self.reg_layer = [self._LN(embed_size)] + [self._LN(args.filter_size * (2 ** l)) for l in range(0,args.num_layer-1)]
        else:
            raise NotImplementedError

        conv_shapes = [(embed_size, args.filter_size, args.filter_shape, 1)]
        conv_shapes += [(args.filter_size * (2 ** l), args.filter_size * (2 ** (l+1)), args.filter_shape, 1) for l in range(0,args.num_layer-2)]
        conv_shapes += [(args.filter_size * (2 ** (args.num_layer-2)), args.latent_size, last_filter_shape, 1)]
        self.conv_layer = nn.ModuleList([self._CONV(*conv_shapes[l]) for l in range(0,args.num_layer)])

        self.model_size = sum(t.numel() for t in self.parameters())

        if state_dict:
            self.load(state_dict)
        else:
            self._InitWeight(nn.Conv2d)

    def _CONV(self, *shape):
        return nn.Conv2d(shape[0], shape[1], (shape[2], shape[3]), stride=2)
    
    def forward(self, x):
        x = self.padding(x[:, :self.sentence_len], self.filter_shape)
        x = self.embed(x)
        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x.size()) < 3:
            x = x.view(1, *x.size())
        # reshape for convolution layer
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        x = self.dropout(x)
        # N 1 L emb => N emb L 1
        h = x.transpose_(1, 3)
        for l in range(len(self.reg_layer)):
            h = self.conv_layer[l](self.dropout(F.relu(self.reg_layer[l](h))))

        return h.squeeze(-1) #[bsz, latent_size, final_len]
        
    
class DeconvolutionDecoder(ConvModel):
    file_name = 'DCNN.pkl'
    def __init__(self, args, state_dict = None):
        super().__init__(args)
        last_filter_shape = ConvModel.compute_last_filter_shape(args)
        # assert(args.num_feature == final_len)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embed = nn.Embedding(self.tokenizer.vocab_size, args.embed_dim)
        self.tau = args.tau
        self.filter_shape = args.filter_shape
        self.out_layer = args.out_layer
        self.file_name = 'DCNN.pkl'
        if args.reg_layer == 'none':
            self.reg_layer = nn.ModuleList([nn.Identity() for l in range(args.num_layer)])
        elif args.reg_layer == 'bn':
            self.reg_layer = nn.ModuleList([self._BN(args.latent_size)] + [self._BN(args.filter_size * (2 ** (l-1))) for l in range(args.num_layer-1, 0, -1)])
        elif args.reg_layer == 'ln':
            self.reg_layer = [self._LN(args.latent_size)] + [self._LN(args.filter_size * (2 ** (l-1))) for l in range(args.num_layer-1, 0, -1)]
        else:
            raise NotImplementedError        
        deconv_shapes = [(args.latent_size, args.filter_size * (2 ** (args.num_layer-2)), last_filter_shape, 1)]
        deconv_shapes += [(args.filter_size * (2 ** l), args.filter_size * (2 ** (l-1)), args.filter_shape, 1) for l in range(args.num_layer-2, 0, -1)]
        
        ## last layer
        if args.out_layer == 'pred_emb':
            deconv_shapes += [(args.filter_size , self.embed.weight.size()[1], args.filter_shape, 1)]
        elif args.out_layer == 'pred_token':
            deconv_shapes += [(args.filter_size , self.embed.weight.size()[0], args.filter_shape, 1)]
        elif args.out_layer == 'lm_head':
            lm_head_dim = self.embed.weight.size()[1]
            config = BertConfig.from_pretrained('bert-base-uncased')
            config.hidden_size = lm_head_dim
            self.lm_head = BertLMPredictionHead(config)
            self.final_ln = nn.LayerNorm(lm_head_dim)
            deconv_shapes += [(args.filter_size, lm_head_dim, args.filter_shape, 1)]
        else:
            raise NotImplementedError


        self.deconv_layer = nn.ModuleList([self._DECONV(*deconv_shapes[l]) for l in range(0,args.num_layer)])

        self.model_size = sum(t.numel() for t in self.parameters())

        if state_dict:
            self.load(state_dict)
        else:
            self._InitWeight(nn.ConvTranspose2d)

    def _DECONV(self, *shape):
        return nn.ConvTranspose2d(shape[0], shape[1], (shape[2], shape[3]), stride=2)

    def forward(self, hidden_state, input_ids_dec=None):
        h = self.dropout(hidden_state)
        log_prob = self.compute_prob(h)
        input_ids_dec = self.padding(input_ids_dec[:, :self.sentence_len], 1)
        assert(log_prob.shape[1] == input_ids_dec.shape[1])
        loss = [F.nll_loss(sentence_emb_matrix, word_ids, size_average=False) for sentence_emb_matrix, word_ids in zip(log_prob, input_ids_dec)]
        average_loss = sum([torch.sum(l) for l in loss]) / log_prob.size()[0]/ input_ids_dec.shape[1]

        # total = torch.ne(input_ids_dec, -1).float().sum()
        # Not comparable with GPT2 decoder, as the pad_tokens are calculated in total loss
        total = torch.ne(input_ids_dec, -1).float().sum()
        correct = (log_prob.max(dim=-1)[1] == input_ids_dec).sum()

        return average_loss, correct, total, hidden_state

    def compute_prob(self, hidden_state):
        h = hidden_state.unsqueeze(-1) #[bsz, latent_size, final_len, 1]
        for l in range(len(self.reg_layer)):
            h = self.deconv_layer[l](self.dropout(F.relu(self.reg_layer[l](h))))
        x_hat = h.transpose_(1, 3).squeeze()
        return self.compute_logp(x_hat)


    def compute_logp(self, x_hat):
        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x_hat.size()) < 3:
            x_hat = x_hat.view(1, *x_hat.size())
        #[bsz, L, emb_dim]

        ######   orginal implementation  ######    
        if self.out_layer == 'pred_emb':
            # normalize
            norm_x_hat = torch.norm(x_hat, 2, dim=2, keepdim=True)
            rec_x_hat = x_hat / norm_x_hat
            # compute probability
            w = Variable(self.embed.weight.data).t()
            norm_w = w/torch.norm(w, 2, dim=0, keepdim=True)
            cos_sim = torch.einsum("ble,ev->blv", rec_x_hat, norm_w) / self.tau
            log_prob = F.log_softmax(cos_sim, dim=2)
        elif self.out_layer == 'pred_token':
            log_prob = F.log_softmax(x_hat, dim=2)
        elif self.out_layer == 'lm_head':
            x_hat = self.lm_head(self.final_ln(x_hat))  #[bsz, L, emb_dim] => [bsz, L, vocab_dim]
            log_prob = F.log_softmax(x_hat, dim=2)
        else:
            raise NotImplementedError

        log_prob = log_prob[:,(self.filter_shape-1):-(self.filter_shape -1),:]

        return log_prob

    def generate_from(self, hidden_state):
        log_prob = self.compute_prob(hidden_state)
        out = log_prob.max(dim=-1)[1]
        out = out.tolist()
        gen = [self.tokenizer.decode(prepare_for_bleu(s, self.tokenizer)) for s in out]
        resps = [g.encode('ascii','ignore').decode('ascii') for g in gen]
        return resps
      
    



class DeconformerDecoder(DeconvolutionDecoder):
    file_name = 'DCF.pkl'
    def __init__(self, args, state_dict = None):
        super().__init__(args)
        self.reg_layer2 = copy.deepcopy(self.reg_layer)
        self.file_name = 'DCF.pkl'
        config = BertConfig.from_pretrained('bert-base-uncased')
        configs = {}
        bert_modules = []
        for l in range(args.num_layer):
            configs[l] = copy.deepcopy(config)
            configs[l].hidden_size = args.latent_size if l==0 else args.filter_size * (2 ** (args.num_layer-l-1))    
            bert_modules.append(BertLayer(configs[l])) 
        self.bert_layer = nn.ModuleList(bert_modules)

        if state_dict:
            self.load(state_dict)
        self.model_size = sum(t.numel() for t in self.parameters())


    def compute_prob(self, hidden_state):
        h = hidden_state.unsqueeze(-1) #[bsz, latent_size, final_len, 1]
        for l in range(len(self.reg_layer)):
            # BERT block
            h = self.reg_layer2[l](h)
            h = h.squeeze(-1).permute(0, 2, 1)
            h = self.bert_layer[l](h)[0]
            h = self.dropout(h)
            h = h.permute(0, 2, 1).unsqueeze(-1)
            # Deconv block
            # h = self.deconv_layer[l](self.dropout(F.relu(self.reg_layer[l](h))))   #[bsz, latent_size, cur_len, 1]
            h = self.deconv_layer[l](self.dropout(F.relu(h)))   #[bsz, latent_size, cur_len, 1] This sometimes leads to unstability issue
        x_hat = h.transpose_(1, 3).squeeze()
        return self.compute_logp(x_hat)
    

class BertEncoder(nn.Module):
    def __init__(self, args, model_enc=None, hidden_size = None):
        super().__init__()
        self.name = args.enc_model
        if not hidden_size: hidden_size = args.latent_size
        self.model_enc = model_enc
        self.tokenizer = BertTokenizer.from_pretrained(self.name)
        self.num_feature = args.num_feature
        if model_enc is None:
            if hasattr(args, 'load_enc') and args.load_enc:
                # load pretrained bert 
                self.model_enc = BertModel.from_pretrained(self.name)
                args.latent_size = self.model_enc.config.hidden_size
            else: 
                # from scratch
                config = BertConfig.from_pretrained(self.name)
                config.hidden_size = hidden_size          
                self.model_enc = BertModel(config)  

        self.model_size = sum(t.numel() for t in self.model_enc.parameters())
    
    def forward(self, input_ids_bert=None, attention_mask = None):
        if attention_mask == None:
            attention_mask =torch.ne(input_ids_bert, 0)
        encoded_output = self.model_enc(input_ids_bert, attention_mask)
        hidden_state = encoded_output['last_hidden_state'][:, :self.num_feature, :]
        return hidden_state.permute(0, 2, 1) # bsz x latent x feature num

    def named_parameters(self):
        return self.model_enc.named_parameters()

    def save(self, output_dir, prefix):
        torch.save(self.model_enc.state_dict(), os.path.join(output_dir, prefix+'-BERT.pkl'))

    def from_pretrained(encoder, input_dir, prefix, name = 'bert-base-uncased'):
        model_enc = BertModel.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix), map_location='cpu'))
        encoder['model_args']['model_enc'] = model_enc 
        return encoder





class T5Encoder(nn.Module):
    def __init__(self, args, model_enc=None, hidden_size = None):
        super().__init__()
        self.name = args.enc_model
        if not hidden_size: hidden_size = args.latent_size
        self.model_enc = model_enc
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.t5_tokenizer = T5Tokenizer.from_pretrained(self.name)
        self.num_feature = args.num_feature
        if model_enc is None:
            if hasattr(args, 'load_enc') and args.load_enc:
                # load pretrained t5
                self.model_enc, loading_info = T5EncoderModel.from_pretrained(self.name, output_loading_info = True)
                print_chkpt_info(loading_info)
                args.latent_size = self.model_enc.config.hidden_size
            else: 
                # from scratch
                config = T5Config.from_pretrained(self.name)
                config.hidden_size = hidden_size          
                self.model_enc = T5EncoderModel(config)

        self.model_size = sum(t.numel() for t in self.model_enc.parameters())
    
    def forward(self, input_ids=None):
        text_batch = [self.tokenizer.decode(t, skip_special_tokens=True) for t in input_ids] # BERT tokenizer
        tokenized = self.t5_tokenizer.batch_encode_plus(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids_t5, att_msk_t5 = tokenized.input_ids.to(input_ids.device), tokenized.attention_mask.to(input_ids.device)
        encoded_output = self.model_enc(input_ids_t5, attention_mask= att_msk_t5)
        hidden_state = encoded_output['last_hidden_state'][:, :self.num_feature, :]
        # hidden_state = hidden_state * self.mw + self.mb # TODO: check
        # torch.mean(hidden_state, -1), torch.std(hidden_state, -1)
        return hidden_state.permute(0, 2, 1) # bsz x latent x feature num

    def named_parameters(self):
        return self.model_enc.named_parameters()

    def save(self, output_dir, prefix):
        torch.save(self.model_enc.state_dict(), os.path.join(output_dir, prefix+'-T5_enc.pkl'))

    def from_pretrained(encoder, input_dir, prefix, name = 't5-large'):
        chkpt_state_dict = torch.load(os.path.join(input_dir, prefix+'-T5_enc.pkl'), map_location='cpu')
        model_enc, loading_info = T5EncoderModel.from_pretrained(name, state_dict = chkpt_state_dict, output_loading_info = True)

        print_chkpt_info(loading_info, chkpt_state_dict, model_enc)

        encoder['model_args']['model_enc'] = model_enc 
        return encoder

## Does not work for now. 
class T5Decoder(nn.Module):
    def __init__(self, args, model_dec=None, hidden_size=None):
        super().__init__()
        self.name = args.dec_model
        if not hidden_size: hidden_size = args.latent_size
        self.model_dec = model_dec
        self.tokenizer = T5Tokenizer.from_pretrained(self.name)
        self.num_feature = args.num_feature
        if model_dec is None:
            if hasattr(args, 'load_dec') and args.load_dec:
                # load pretrained t5
                self.model_dec = T5Model.from_pretrained(self.name)
                args.latent_size = self.model_dec.config.hidden_size
            else: 
                # from scratch
                config = T5Config.from_pretrained(self.name)
                config.hidden_size = hidden_size          
                self.model_dec = T5Model(config)
                self.model_size = sum(t.numel() for t in self.model_dec.parameters())
    
    def forward(self, input_ids_t5, att_msk_t5):
        output = self.model_dec(input_ids=input_ids_t5, attention_mask=att_msk_t5)
        return output
    
    def named_parameters(self):
        return self.model_dec.named_parameters()

    def save(self, output_dir, prefix):
        torch.save(self.model_dec.state_dict(), os.path.join(output_dir, prefix+'-T5_dec.pkl'))
    
    @staticmethod
    def from_pretrained(decoder, input_dir, prefix, name='t5-large'):
        model_dec = T5Model.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix+'-T5_dec.pkl'), map_location='cpu'))
        decoder['model_args']['model_dec'] = model_dec 
        return decoder

class BertConvEncoder(BertEncoder):
    def __init__(self, args, state_dict = None, model_enc=None, hidden_size=768):
        super().__init__(args, model_enc=model_enc, hidden_size=hidden_size)
        last_filter_shape = args.sentence_len - 2*args.num_feature + 2
        self.final_layer = nn.Conv2d(hidden_size, args.latent_size, (last_filter_shape, 1), stride=2)
        self.model_size = sum(t.numel() for t in self.model_enc.parameters())
        self.padding = lambda x: nn.ConstantPad1d((0, args.sentence_len-x.shape[2]), self.tokenizer.pad_token_id)(x)
        if state_dict:
            self.load_state_dict(state_dict)
    
    def forward(self, input_ids_bert=None):
        encoded_output = self.model_enc(input_ids_bert, attention_mask=torch.ne(input_ids_bert, 0))
        bert_output = encoded_output['last_hidden_state']
        bert_output = F.relu(bert_output)

        bert_output = bert_output.permute(0, 2, 1)
        bert_output = self.padding(bert_output)
        bert_output = bert_output.unsqueeze(-1)
        hidden_state = self.final_layer(bert_output)
        hidden_state = hidden_state.squeeze(-1)
        return hidden_state # bsz x latent x feature num


    def save(self, output_dir, prefix):
        torch.save(self.state_dict(), os.path.join(output_dir, prefix+'-BERTCNN.pkl'))

    def from_pretrained(encoder, input_dir, prefix, *unused):
        enc_ckpt = torch.load(os.path.join(input_dir, prefix+'-BERTCNN.pkl'), map_location='cpu')
        encoder['model_args']['state_dict'] = enc_ckpt
        return encoder

class GPT2Decoder(nn.Module):
    def __init__(self, args, model_gpt=None, model_pre=None):
        super().__init__()
        self.name = args.dec_model
        self.model_gpt = model_gpt
        self.model_pre = model_pre
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.name) 
        self.num_feature = args.num_feature
        self.max_len = args.sentence_len
        if model_gpt is None:
            if hasattr(args, 'load_dec') and args.load_dec:
                # load pretrained bert 
                self.model_gpt = GPT2LMHeadModel.from_pretrained(self.name)
                config = GPT2Config.from_pretrained(self.name)
                if config.n_embd == args.latent_size:
                    self.embd_adapter = nn.Identity()
                else:
                    self.embd_adapter = nn.Linear(args.latent_size, config.n_embd) 
            else: 
                # from scratch
                config = GPT2Config.from_pretrained(self.name)
                config.n_embd = args.latent_size
                config.n_head = args.n_head               
                self.model_gpt = GPT2LMHeadModel(config) 
        if model_pre is None:
            if args.share_gpts:
                self.model_pre = self.model_gpt
                self.model_size = sum(t.numel() for t in self.model_gpt.parameters()) 
                return
            else:
                config = GPT2Config.from_pretrained(self.name)
                config.n_embd = args.latent_size           
                self.model_pre = GPT2LMHeadModel(config)
        self.model_size = sum(t.numel() for t in self.model_gpt.parameters()) + sum(t.numel() for t in self.model_pre.parameters())

    def forward(self, hidden_state, input_ids_dec=None, lm_labels=None):
        # TODO: if input length is shorter than num_feature, there will be issue with BERT-DECONV 
        # assert(hidden_state.shape[2] == self.num_feature)  
        # bsz x latent x feature num

        hidden_state = hidden_state.permute(0, 2, 1).contiguous()
        hidden_state = self.embd_adapter(hidden_state)
        hidden_state = hidden_state.permute(0, 2, 1).contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        context = self.model_pre(inputs_embeds=hidden_state.permute(0, 2, 1))[-1] #list
        lm_logits = self.model_gpt(input_ids=input_ids_dec, past_key_values=context)[0]
        bsz, seq_len, vocab_size = lm_logits.size()
        loss = loss_fct(lm_logits.view(-1, vocab_size), lm_labels.view(-1))
        loss = loss.view(bsz, seq_len)
        total = torch.ne(lm_labels, -1).float().sum()
        loss = torch.sum(loss) / total
        correct = (lm_logits.max(dim=-1)[1] == lm_labels).sum()

        # if hidden_state is None:
        #     correct, total = correct.item(), total.item()
        return loss, correct, total, hidden_state

    def save(self, output_dir, prefix):
        torch.save(self.model_gpt.state_dict(), os.path.join(output_dir, prefix+'-GPT2.pkl'))
        torch.save(self.model_pre.state_dict(), os.path.join(output_dir, prefix+'-PRE.pkl'))

    
    def from_pretrained(decoder, input_dir, prefix, name = 'gpt'):
        model_gpt = GPT2LMHeadModel.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix+'-GPT2.pkl'), map_location='cpu'))
        model_pre = GPT2LMHeadModel.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix+'-PRE.pkl'), map_location='cpu'))
        decoder['model_args']['model_gpt'] = model_gpt
        decoder['model_args']['model_pre'] = model_pre
        return decoder

    def named_parameters(self):
        return list(self.model_pre.named_parameters()) + list(self.model_gpt.named_parameters())

    def generate_from(self, hidden_states, sample=False, beam_width = -1, top_k = 1, skip_special_tokens = False):
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        hidden_states = self.embd_adapter(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        with torch.no_grad():
            if beam_width == -1:
                #greedy/sample
                hidden_states = hidden_states.permute(0, 2, 1)

                batch_size = 64
                num_batches = (hidden_states.shape[0] - 1)  // batch_size + 1
                batches = [hidden_states[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
                resps = []
                for b in tqdm.tqdm(batches):
                    context = self.model_pre(inputs_embeds=b)[-1]
                    out = generate_sequence(self.model_gpt, temperature=1, top_k=top_k, length=self.max_len, sample=sample, past=context, device='cuda')
                    out = out.tolist()
                    gen = [self.tokenizer.decode(prepare_for_bleu(s, self.tokenizer, skip_special_tokens = skip_special_tokens), skip_special_tokens = skip_special_tokens) for s in out]
                    resps.extend([g.encode('ascii','ignore').decode('ascii') for g in gen])
            else:
                # beam
                resps = []
                for hidden_state in hidden_states.permute(0, 2, 1):
                    # hidden_state: 1 x dim
                    context = self.model_pre(inputs_embeds=hidden_state)[-1]
                    out = beam_search_naive(self.model_gpt, 1, length=self.max_len+10, beam_width=beam_width, beam_examples=1, past=context)
                    out = out.tolist()
                    gen = [self.tokenizer.decode(prepare_for_bleu(s, self.tokenizer, skip_special_tokens = skip_special_tokens), skip_special_tokens = skip_special_tokens) for s in out]
                    resps.append(gen[-1].encode('ascii','ignore').decode('ascii'))
        return resps



        
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, args = None):
        # E.g. encoder = {'model_cls':'BertDecoder','model_args':{'encoder':encoder_object} }
        super().__init__()
        self.encoder = encoder['model_cls'](**encoder['model_args'])
        self.decoder = decoder['model_cls'](**decoder['model_args'])
        self.h_noiser = args.h_noiser
        self.h_noiser_ratio = args.h_noiser_ratio
        self.h_tanh = args.h_tanh 


    def forward(self, input_ids_enc, input_ids_dec=None, lm_labels=None):
        hidden_state = self.encoder_mean(input_ids_enc)
        if self.h_noiser == 'normal':
            hidden_state = hidden_state + self.h_noiser_ratio*torch.randn_like(hidden_state)
        elif self.h_noiser == 'none':
            hidden_state = hidden_state
        else:
            NotImplementedError
        if isinstance(self.decoder, GPT2Decoder):
            return self.decoder(hidden_state, input_ids_dec=input_ids_dec, lm_labels=lm_labels)
        elif isinstance(self.decoder, DeconvolutionDecoder):
            return self.decoder(hidden_state, input_ids_dec=input_ids_dec)
        else:
            NotImplementedError

    def encoder_mean(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        if self.h_tanh:
            hidden_state = torch.tanh(hidden_state)
        return hidden_state

    def save(self, output_dir, prefix):
        self.encoder.save(output_dir, prefix)
        self.decoder.save(output_dir, prefix)
              
    @classmethod
    def from_pretrained(cls, encoder, decoder, input_dir, args):
        prefix = args.resume_ckpt
        encoder_new = encoder['model_cls'].from_pretrained(encoder, input_dir, prefix, name=args.enc_model)
        decoder_new = decoder['model_cls'].from_pretrained(decoder, input_dir, prefix, name=args.dec_model)
        model = cls(encoder_new, decoder_new, args)
        return model

    def named_enc_parameters(self):
        return self.encoder.named_parameters()

    def named_dec_parameters(self):
        return self.decoder.named_parameters()

    # def named_pretrained_parameters(self):
    #     return list(self.model_enc.named_parameters()) + list(self.model_gpt.named_parameters())

    def generate_from(self, *kargs, **kwargs):
        return self.decoder.generate_from(*kargs, **kwargs)

    # def generate_from_beam(self, *kargs):
    #     return self.decoder.generate_from_beam(*kargs)

    def decode(self, outs, tokenizer = 'enc'):
        resps = []
        self.tokenizers = {'enc': self.encoder.tokenizer, 'dec': self.decoder.tokenizer}
        for out in outs:
            out = out.tolist()
            gen = self.tokenizers[tokenizer].decode(prepare_for_bleu(out, tokenizer=self.tokenizers[tokenizer]))
            resps.append(gen.encode('ascii','ignore').decode('ascii'))
        return resps

    def encode(self, text):
        input_ids = self.encoder.tokenizer.encode(text)
        input_ids = torch.tensor([input_ids]).cuda()
        return self.encoder_mean(input_ids)

    # def encode_batch(self, *kargs):    
    #     return self.encoder.encode(*kargs)

class VAE(AutoEncoder):
    def __init__(self, encoder, decoder, args = None):
        super().__init__(encoder, decoder, args = args)
        args.latent_size = self.encoder.model_enc.config.hidden_size
        self.fc_mean = torch.nn.Linear(args.latent_size, args.latent_size)
        self.fc_log_var = torch.nn.Linear(args.latent_size, args.latent_size)
        self.beta = args.h_noiser_ratio

    def forward(self, input_ids_enc, input_ids_dec=None, lm_labels=None):
        # bsz x latent x feature num
        mean = self.encoder_mean(input_ids_enc)
        log_var = self.encoder_log_var(input_ids_enc)
        sampled_h = self.reparameterize(mean, log_var)
        if isinstance(self.decoder, GPT2Decoder):
            BCE, correct, total, _ = self.decoder(sampled_h, input_ids_dec=input_ids_dec, lm_labels=lm_labels)
        elif isinstance(self.decoder, DeconvolutionDecoder):
            BCE, correct, total, _ = self.decoder(sampled_h, input_ids_dec=input_ids_dec)
        else:
            NotImplementedError
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) 
        loss = BCE + self.beta * KLD  
        return loss, correct, total, mean

    def encoder_mean(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        shapes = hidden_state.shape
        mean = self.fc_mean(hidden_state.permute(0, 2, 1)).view(-1, shapes[2], shapes[1]).permute(0, 2, 1)
        if self.h_tanh:
            mean = torch.tanh(mean)
        return mean

    def encoder_log_var(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        shapes = hidden_state.shape
        log_var = self.fc_log_var(hidden_state.permute(0, 2, 1)).view(-1, shapes[2], shapes[1]).permute(0, 2, 1) # offset with -5
        return log_var

    def save(self, output_dir, prefix):
        torch.save(self.state_dict(), os.path.join(output_dir, prefix+'-VAE.pkl'))
              
    @classmethod
    def from_pretrained(cls, encoder, decoder, input_dir, args):
        prefix = args.resume_ckpt
        model = cls(encoder, decoder, args =args)
        model.load_state_dict(torch.load(os.path.join(input_dir, prefix+'-VAE.pkl'), map_location='cpu'))
        return model

    def named_enc_parameters(self):
        return itertools.chain(self.encoder.named_parameters(), self.fc_mean.named_parameters(), self.fc_log_var.named_parameters())

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    



def encoderModels(args):
    encoder = {}
    encoder['model_args'] = {'args':args}
    if args.enc_model.startswith('bert'):
        encoder['model_cls'] = BertEncoder 
    elif 't5' in args.enc_model:
        encoder['model_cls'] = T5Encoder
    elif args.enc_model == 'conv':
        encoder['model_cls'] = ConvolutionEncoder
    elif args.enc_model == 'bertconv':
        encoder['model_cls'] = BertConvEncoder
    else:
        raise NotImplementedError
    return encoder

def decoderModels(args):
    decoder = {}
    decoder['model_args'] = {'args':args}
    if args.dec_model.startswith('gpt2'):
        decoder['model_cls'] = GPT2Decoder
    elif 't5' in args.enc_model:
        decoder['model_cls'] = T5Decoder
    elif args.dec_model == 'deconv':
        decoder['model_cls'] = DeconvolutionDecoder
    elif args.dec_model == 'deconformer':
        decoder['model_cls'] = DeconformerDecoder
    else:
        raise NotImplementedError
    return decoder

def add_noise(sents, args):
    if not args.noise_type:
        return sents
    else:
        sents = sents.cpu().numpy()
        num_corrupted = floor(args.noise_ratio * args.sentence_len)
        if args.noise_type == 's':
            sents_permutated= substitute_sent(sents, num_corrupted, args)
        elif args.noise_type == 'pw':
            sents_permutated= permutate_sent(sents, num_corrupted)
        elif args.noise_type == 'ps':
            sents_permutated= permutate_sent_whole(sents)
        elif args.noise_type == 'a':
            sents_permutated= add_sent(sents, num_corrupted, args)   
        elif args.noise_type == 'd':
            sents_permutated= delete_sent(sents, num_corrupted) 
        elif args.noise_type == 'm':
            sents_permutated= mixed_noise_sent(sents, num_corrupted, args)
        else:
            raise NotImplementedError
        
        return torch.LongTensor(sents_permutated).cuda()


def permutate_sent(sents, num_corrupted):
    # permutate the words in sentence
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s= np.random.choice(len(sent_temp)-1, size=num_corrupted, replace=True)
        temp = sent_temp[idx_s[0]]
        for ii in range(num_corrupted-1):
            sent_temp[idx_s[ii]] = sent_temp[idx_s[ii+1]]
        sent_temp[idx_s[num_corrupted-1]] = temp
        sents_p.append(sent_temp)
    return sents_p  

def permutate_sent_whole(sents):
    # permutate the whole sentence
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        sents_p.append(np.random.permutation(sent_temp))
    return sents_p  
    
    
def substitute_sent(sents, num_corrupted, args):
    # substitute single word 
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s= np.random.choice(len(sent_temp)-1, size=num_corrupted, replace=True)   
        for ii in range(num_corrupted):
            sent_temp[idx_s[ii]] = np.random.choice(args.v)
        sents_p.append(sent_temp)
    return sents_p       

def delete_sent(sents, num_corrupted):
    # substitute single word 
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s= np.random.choice(len(sent_temp)-1, size=num_corrupted, replace=True)   
        for ii in range(num_corrupted):
            sent_temp[idx_s[ii]] = -1
        sents_p.append([s for s in sent_temp if s!=-1])
    return sents_p 
    
def add_sent(sents, num_corrupted, args):
    # substitute single word 
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s= np.random.choice(len(sent_temp)-1, size=num_corrupted, replace=True)   
        for ii in range(num_corrupted):
            sent_temp.insert(idx_s[ii], np.random.choice(args.v))
        sents_p.append(sent_temp[:args.sentence_len])
    return sents_p  


def mixed_noise_sent(sents, num_corrupted, args):
    sents = delete_sent(sents, num_corrupted)
    sents = add_sent(sents, num_corrupted, args)
    sents = substitute_sent(sents, num_corrupted, args)
    return sents

def get_whole_word_mask(tokenizer):
    def is_beginning_of_word(i):
        x = tokenizer.convert_ids_to_tokens(i)
        if x in tokenizer.all_special_tokens:
            return True
        return not x.startswith('##')
    mask_whole_words = torch.ByteTensor(list(
    map(is_beginning_of_word, range(len(tokenizer)))))
    return mask_whole_words

class noise:
    def __init__(self):
        pass

    def _noise(self):
        pass

    def noise(self, tensor, mlm_probability = 0.3):
        # tensor: LongTensor bsz x seq_len
        noised_tensor = []
        to_keep = torch.ne(tensor, 0)
        for sent, keep in zip(tensor.split(1, dim=0), to_keep.split(1, dim=0)):
            noised_tensor.append(self._noise(sent[keep]))
        noised_tensor = pad_sequence(noised_tensor, batch_first=True, padding_value=0)
        return noised_tensor


class noise_bart(noise):
    def __init__(self,
                 tokenizer,
                 mlm_probability=None,
                 poisson_lambda=3.0,#randomly shuffle sentences for this proportion of inputs
                 permute_ratio=0.1, #take this proportion of words and permute them
                 mask_ratio=0.3, #fraction of words/subwords that will be masked
                 random_ratio=0.2, #instead of using [MASK], use random token this often
                 replace_length=1 #when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)
                  ):
        _lambda = poisson_lambda
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= (k + 1)
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        self.mask_span_distribution = torch.distributions.Categorical(ps)
        self.mask_whole_word = get_whole_word_mask(tokenizer)
        self.permute_ratio = permute_ratio
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.replace_length = replace_length
        self.mask_idx = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
    
    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        is_word_end = is_word_start.clone()
        
        is_word_end[:-1] = is_word_start[1:]
        is_word_end[-1] = 0
        is_word_end[-2] = 1
        return is_word_start, is_word_end
    
    def _noise(self, source):
        if not torch.is_tensor(source):
            source = torch.tensor(source, dtype=torch.long)
        is_word_start, is_word_end = self.word_starts(source)
        if self.permute_ratio > 0.0:
            source = self.permute_words(source, is_word_end, self.permute_ratio)

        if self.mask_ratio > 0:
            source = self.add_whole_word_mask(source, is_word_start, self.mask_ratio)
        return source

    def permute_words(self, source, is_word_end, p=1.0):
        result = source.clone()
        is_word_end = is_word_end.bool()
        word_ends = (is_word_end[1:] * ~is_word_end[:-1]).nonzero() + 2
        num_words = word_ends.size(0)
        num_to_permute = math.ceil((num_words * 2 * p) / 2.0)
        substitutions = torch.randperm(num_words)[:num_to_permute]
        ordering = torch.arange(0, num_words)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            word = source[(word_ends[i - 1] if i > 0 else 1):word_ends[i]]
            result[index:index + word.size(0)] = word
            index += word.size(0)
        return result

    def add_whole_word_mask(self, source, is_word_start, p):
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source


        lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat([lengths, self.mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return self.add_insertion_noise(source, num_inserts / source.size(0))

        assert (lengths > 0).all()

        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero()
        indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[-1] = 255 # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(1, self.vocab_size, size=(mask_random.sum(),))

        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if self.replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = self.mask_idx
                source[indices[mask_random]] = torch.randint(1, self.vocab_size, size=(mask_random.sum(),))

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(low=1, high=self.vocab_size, size=(num_random,))

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

class noise_bert(noise):
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def _noise(self, inputs):
        """ Prepare masked tokens for masked language modeling: 80% MASK, 20% random. """

        # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        probability = torch.full(inputs.shape, self.mlm_probability)

        special_tokens_mask = [ 1 if x in [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, 0] else 0 for x in inputs.tolist()]
        probability.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability).bool()
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 20% of the time, we replace masked input tokens with random word
        indices_random = masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs


class noise_sub(noise):
    def __init__(self, tokenizer, mlm_probability=0.3):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
    def _noise(self, inputs ):
        """ Prepare masked tokens for masked language modeling: 30% random. """

        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        probability = torch.full(inputs.shape, self.mlm_probability)

        special_tokens_mask = [ 1 if x in [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, 0] else 0 for x in inputs.tolist()]
        probability.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability).bool()
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[masked_indices] = random_words[masked_indices]

        return inputs
    

class noise_sub_uniform(noise):
    def __init__(self, tokenizer, mlm_probability=0.3):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
    def _noise(self, inputs ):
        """ Prepare masked tokens for masked language modeling: uniformly sample from [0, mlm_prob] """
        cur_mlm_prob = random.uniform(0, self.mlm_probability)
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        probability = torch.full(inputs.shape, cur_mlm_prob)

        special_tokens_mask = [ 1 if x in [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, 0] else 0 for x in inputs.tolist()]
        probability.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability).bool()
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[masked_indices] = random_words[masked_indices]

        return inputs


class noise_none(noise):
    def __init__(self, tokenizer, mlm_probability=None):
        self.tokenizer = tokenizer

    def _noise(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        return inputs

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return (1.0 - x)/(1.0 - warmup)

def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)))


def noamwd_decay(step, warmup_steps,
                 model_size, rate=0.5, decay_steps=1000, start_step=500):
    """Learning rate schedule optimized for huge batches
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)) *
        rate ** (max(step - start_step + decay_steps, 0) // decay_steps))


def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate`
    every :obj:`decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))



SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}


class Adam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(Adam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(grad, alpha=1 - beta1)
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss

class Adamax(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), eps=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, eps=eps, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(Adamax, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_inf'] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                update = exp_avg / (exp_inf + eps)

                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']


                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss

class Lamb(Optimizer):
    """ Implements Lamb algorithm.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True, max_grad_norm=1.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                step_size = group["lr"]
                if group["correct_bias"]:
                    exp_avg = exp_avg / (1.0 - beta1 ** state["step"])
                    exp_avg_sq = exp_avg_sq / (1.0 - beta2 ** state["step"])

                update = exp_avg / exp_avg_sq.sqrt().add_(group["eps"])
                if group["weight_decay"] > 0.0:
                    update.add_(p.data * group["weight_decay"])

                w_norm = torch.norm(p.data)
                g_norm = torch.norm(update)
                if w_norm == 0 or g_norm == 0:
                    ratio = 1.
                else:
                    ratio = (w_norm / g_norm) 

                p.data.add_(-group["lr"] * ratio * update)

        return loss
