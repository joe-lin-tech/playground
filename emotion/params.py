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
EPOCHS = 5
MAX_LENGTH = 40
LEARNING_RATE = 5e-6
SEED = 5
LOG_INTERVAL = 8
TRAIN_FILE = '../data/emotion/train.tsv'
DEV_FILE = '../data/emotion/dev.tsv'
SAVE_FILE = '../models/emotion_bert.pt'
