import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk import word_tokenize
import torch.nn.functional as F
import numpy as np
from transformers import SqueezeBertTokenizer
from typing import Literal
from params import *
import json


class EmotionDataset(Dataset):
    def __init__(self, path: str, mode: Literal['train', 'val', 'test']):
        self.path = path
        self.mode = mode
        self.comments = []
        self.emotions = []

        self.tokenizer = SqueezeBertTokenizer.from_pretrained("squeezebert/squeezebert-uncased")

        with open(path) as f:
            for line in f:
                line = line.split('\t')
                # comment = self.token_transform(line[0])
                comment = line[0]
                self.comments.append(comment)
                self.emotions.append([int(l) for l in line[1].split(',')])

        self.inputs = self.comments
        self.labels = self.emotions

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.inputs[idx], None, add_special_tokens=True, max_length=MAX_LENGTH, padding="max_length", truncation=True)
        labels = [F.one_hot(torch.tensor(c), len(EMOTIONS)) for c in self.labels[idx]]
        return { "ids": inputs['input_ids'], "mask": inputs['attention_mask'], "labels": torch.tensor(np.logical_or.reduce(labels)).long() if len(labels) > 1 else labels[0] }

    # function to collate data samples into batch tensors
    def collate_fn(self, batch):
        ids_batch = []
        mask_batch = []
        label_batch = []
        for sample in batch:
            ids_batch.append(sample['ids'])
            mask_batch.append(sample['mask'])
            label_batch.append(sample['labels'])
        
        return { 'ids': torch.tensor(ids_batch), 'mask': torch.tensor(mask_batch), 'labels': torch.stack(label_batch).float() }
    