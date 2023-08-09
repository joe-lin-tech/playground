import torch

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

DEVICE = torch.device('mps')

BATCH_SIZE = 64
EPOCHS = 25

NUM_CHANNELS = [6, 16]
NUM_FEATURES = [128, 64]

SAVE_FILE = '../models/image.pt'