import torch

UNK_IDX, PAD_IDX = 0, 1

EMOTIONS = ('admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral')

# EMOTIONS = ('sadness', 'joy', 'love', 'anger', 'fear', 'surprise')

DEVICE = torch.device('mps')

BATCH_SIZE = 64
EPOCHS = 25
EMBED_SIZE = 300
LSTM_HID_DIM = 128
LINEAR_DIM = 64
TOKEN_LENGTH = 30
TRAIN_FILE = '../data/emotion/train.tsv'
DEV_FILE = '../data/emotion/dev.tsv'
SAVE_FILE = '../models/emotion.pt'
