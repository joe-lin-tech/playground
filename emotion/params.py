import torch

UNK_IDX, PAD_IDX = 0, 1

# EMOTIONS = [
#     'admiration',
#     'amusement',
#     'anger',
#     'annoyance',
#     'approval',
#     'caring',
#     'confusion',
#     'curiosity',
#     'desire',
#     'disappointment',
#     'disapproval',
#     'disgust',
#     'embarrassment',
#     'excitement',
#     'fear',
#     'gratitude',
#     'grief',
#     'joy',
#     'love',
#     'nervousness',
#     'optimism',
#     'pride',
#     'realization',
#     'relief',
#     'remorse',
#     'sadness',
#     'surprise',
#     'neutral',
# ]

EMOTIONS = [
    'sadness',
    'joy',
    'love',
    'anger',
    'fear',
    'surprise'
]

BATCH_SIZE = 64

DEVICE = torch.device('mps')

EPOCHS = 50
EMBED_SIZE = 100
LSTM_HID_DIM = 128
LINEAR_DIM = 64
TOKEN_LENGTH = 30
# TRAIN_FILE = '../data/emotion/train.tsv'
# DEV_FILE = '../data/emotion/dev.tsv'
TRAIN_FILE = '../data/emotion/train-emotion.txt'
DEV_FILE = '../data/emotion/dev-emotion.txt'
SAVE_FILE = '../models/emotion4.pt'
