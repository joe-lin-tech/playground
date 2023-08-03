import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk import word_tokenize
from collections import Counter
from typing import Literal
from params import *


class TranslationDataset(Dataset):
    def __init__(self, path: str, mode: Literal['train', 'val', 'test'], dicts = None):
        self.path = path
        self.mode = mode
        self.src_data = []
        self.tgt_data = []
        with open(path) as f:
            for line in f:
                line = line.strip().split('\t')
                self.src_data.append(self.token_transform(line[0]))
                self.tgt_data.append(self.token_transform(" ".join(list(line[1]))))
        
        if dicts is None:
            self.src_enc_dict, self.src_dec_dict = self.build_dict(self.src_data)
            self.tgt_enc_dict, self.tgt_dec_dict = self.build_dict(self.tgt_data)
        else:
            self.src_enc_dict, self.src_dec_dict = dicts['src_enc_dict'], dicts['src_dec_dict']
            self.tgt_enc_dict, self.tgt_dec_dict = dicts['tgt_enc_dict'], dicts['tgt_dec_dict']

        idxs = sorted(range(len(self.src_data)), key=lambda i: len(self.src_data[i]))
        self.src = [[self.src_enc_dict.get(t, UNK_IDX) for t in self.src_data[i]] for i in idxs]
        self.tgt = [[self.tgt_enc_dict.get(t, UNK_IDX) for t in self.tgt_data[i]] for i in idxs]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
    
    # function to tokenize input sentence
    def token_transform(self, sentence: str):
        return ["BOS"] + word_tokenize(sentence.lower()) + ["EOS"]

    def build_dict(self, sentences, max_words = 50000):
        word_count = Counter()
        for s in sentences:
            for t in s:
                word_count[t] += 1
        ls = word_count.most_common(max_words)

        enc_dict = { w[0]: i + 4 for i, w in enumerate(ls) }
        enc_dict['UNK'] = UNK_IDX
        enc_dict['PAD'] = PAD_IDX
        enc_dict['BOS'] = BOS_IDX
        enc_dict['EOS'] = EOS_IDX
        dec_dict = { v: k for k, v in enc_dict.items() }

        return enc_dict, dec_dict

    # function to collate data samples into batch tensors
    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(torch.tensor(src_sample))
            tgt_batch.append(torch.tensor(tgt_sample))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        return src_batch, tgt_batch
    