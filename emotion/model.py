import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from params import *

class WordEmbedding(nn.Module):
    def __init__(self, vocab_dict: dict, emb_size):
        super(WordEmbedding, self).__init__()
        self.vocab_dict = vocab_dict
        self.emb_size = emb_size
        self.build_dict('../data/emotion/glove.6B/glove.6B.300d.txt')
        self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=True)

    def build_dict(self, path: str):
        embedding_dict = {}
        with open(path) as f:
            for line in f:
                line = line.split()
                embedding_dict[line[0]] = np.array(line[1:], dtype=np.float32)
        
        self.embedding_matrix = []
        self.embedding_matrix.append(np.zeros(self.emb_size, dtype=np.float32)) # UNK EMBED
        self.embedding_matrix.append(np.zeros(self.emb_size, dtype=np.float32)) # PAD EMBED
        for word in self.vocab_dict.keys():
            if word in embedding_dict:
                self.embedding_matrix.append(embedding_dict[word])
            else:
                self.embedding_matrix.append(np.zeros(self.emb_size, dtype=np.float32))
        self.embedding_matrix = np.vstack(self.embedding_matrix)
        self.embedding_matrix[0] = np.mean(self.embedding_matrix, axis=0)
        self.embedding_matrix = torch.tensor(self.embedding_matrix)

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long())
        

class EmotionClassifier(nn.Module):
    def __init__(self, vocab_dict: dict, emb_size: int, dim_lstm: int, dim_linear: int):
        super(EmotionClassifier, self).__init__()
        self.word_embedding = WordEmbedding(vocab_dict, emb_size)
        # self.word_embedding = nn.Embedding(len(vocab_dict), emb_size)
        # self.lstm = nn.LSTM(input_size=emb_size, hidden_size=dim_lstm, num_layers=1)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=dim_lstm, num_layers=2, dropout=0.5)
        # self.lstm = nn.LSTM(input_size=emb_size, hidden_size=dim_lstm, num_layers=5, dropout=0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(dim_lstm * TOKEN_LENGTH, dim_linear)
        self.out = nn.Linear(dim_linear, len(EMOTIONS))
    
    def forward(self, input: Tensor):
        embeddings = self.word_embedding(input)
        lstm, (hidden, cell) = self.lstm(embeddings)
        return self.out(self.linear(self.flatten(lstm)))
    