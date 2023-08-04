import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk import word_tokenize
from typing import Literal
from params import *
import json


class EmotionDataset(Dataset):
    def __init__(self, path: str, mode: Literal['train', 'val', 'test']):
        self.path = path
        self.mode = mode
        self.comments = []
        self.emotions = []
        self.vocab = set()

        # with open(path) as f:
        #     for line in f:
        #         line = line.split('\t')
        #         comment = self.token_transform(line[0])
        #         self.comments.append(comment)
        #         for c in comment:
        #             self.vocab.add(c)
        #         self.emotions.append([int(l) for l in line[1].split(',')])

        with open(path) as f:
            for line in f:
                line = json.loads(line)
                comment = self.token_transform(line['text'])
                self.comments.append(comment)
                for c in comment:
                    self.vocab.add(c)
                self.emotions.append([int(line['label'])])
        
        self.vocab = { w: i + 2 for i, w in enumerate(sorted(self.vocab)) }
        self.vocab['<unk>'] = 0
        self.vocab['<pad>'] = 1
        self.index = { v: k for k, v in self.vocab.items() }

        # idxs = sorted(range(len(self.comments)), key=lambda i: len(self.comments[i]))
        # self.inputs = [[self.vocab.get(c, 0) for c in self.comments[i]] for i in idxs]
        # self.labels = [self.emotions[i] for i in idxs]
        self.inputs = [[self.vocab.get(c, 0) for c in comments] for comments in self.comments]
        self.labels = self.emotions

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    
    # function to tokenize input sentence
    def token_transform(self, sentence: str):
        return word_tokenize(sentence.lower())

    # function to collate data samples into batch tensors
    def collate_fn(self, batch):
        input_batch = []
        label_batch = torch.zeros((len(batch), len(EMOTIONS)))
        for i, (input, label) in enumerate(batch):
            # fix the following to be more adaptive
            input_batch.append(torch.tensor((input + [PAD_IDX] * (TOKEN_LENGTH - len(input)))[:TOKEN_LENGTH]))
            label_batch[i, label] = 1

        input_batch = pad_sequence(input_batch, padding_value=PAD_IDX, batch_first=True)
        return input_batch, label_batch
    