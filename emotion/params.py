import torch

UNK_IDX, PAD_IDX = 0, 1

EMOTIONS = ('admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral')

# EMOTIONS = ('sadness', 'joy', 'love', 'anger', 'fear', 'surprise')

DEVICE = torch.device('cuda') # NOTE: poor training performance on mps

BATCH_SIZE = 32
EPOCHS = 10
MAX_LENGTH = 40
LEARNING_RATE = 5e-5
SEED = 1234
LOG_INTERVAL = 16
TRAIN_FILE = '../data/emotion/train.tsv'
DEV_FILE = '../data/emotion/dev.tsv'
SAVE_FILE = '../models/emotion_bert.pt'
