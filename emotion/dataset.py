import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk import word_tokenize
import torch.nn.functional as F
import numpy as np
from transformers import SqueezeBertTokenizer
from typing import Literal
from datasets import load_dataset
from params import *
import json


class EmotionDataset(Dataset):
    def __init__(self, inputs, labels, mode: Literal['train', 'val', 'test']):
        self.mode = mode

        self.tokenizer = SqueezeBertTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.inputs[idx], None, add_special_tokens=True, max_length=MAX_LENGTH, padding="max_length", truncation=True)
        labels = [0] * len(EMOTIONS)
        for c in self.labels[idx]:
            labels[c] = 1
        return {
            "ids": torch.tensor(inputs['input_ids']).long(),
            "mask": torch.tensor(inputs['attention_mask']).long(),
            "labels": torch.tensor(labels).float()
        }

    # function to collate data samples into batch tensors
    # def collate_fn(self, batch):
    #     ids_batch = []
    #     mask_batch = []
    #     label_batch = []
    #     for sample in batch:
    #         ids_batch.append(sample['ids'])
    #         mask_batch.append(sample['mask'])
    #         label_batch.append(sample['labels'])
        
    #     return { 'ids': torch.tensor(ids_batch), 'mask': torch.tensor(mask_batch), 'labels': torch.stack(label_batch).float() }