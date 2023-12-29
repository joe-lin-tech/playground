import torch

DEVICE = torch.device('mps')
BATCH_SIZE = 64
LOG_INTERVAL = 8
EPOCHS = 10

N_CLASSES = 1000

SEED = 0
LEARNING_RATE = 0.01

PRETRAIN_DIRECTORY = '/Volumes/SSD/image-net/ILSVRC/Data/CLS-LOC'
ROOT_DIRECTORY = '/Volumes/SSD/voc'

PRETRAIN_SAVE_FILE = '../models/pretrain.pt'
SAVE_FILE = '../models/yolo.pt'