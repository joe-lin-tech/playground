import torch

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

DEBUG = False
DEVICE = torch.device('mps')

BATCH_SIZE = 64
EPOCHS = 2 if DEBUG else 10
NUM_ENCODER_LAYERS = 3 if DEBUG else 6
NUM_DECODER_LAYERS = 3 if DEBUG else 6
NHEAD = 8
EMB_SIZE = 128 if DEBUG else 512
FFN_HID_DIM = 256 if DEBUG else 2048
DROPOUT = 0.1
TRAIN_FILE = '../data/en-cn/train_mini.txt' if DEBUG else '../data/en-cn/train.txt'
DEV_FILE = '../data/en-cn/dev_mini.txt' if DEBUG else '../data/en-cn/dev.txt'
SAVE_FILE = '../models/translate_debug.pt' if DEBUG else '../models/translate.pt'
