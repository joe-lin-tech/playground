import torchvision
import torch
import torch.nn as nn
from model import YOLO
from params import *
from timeit import default_timer as timer

train_iter = torchvision.datasets.ImageFolder(root=PRETRAIN_DIRECTORY,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Resize((448, 448))
                                                ]))
train_dataloader = torch.utils.data.DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True)

model = YOLO(mode='pretrain')
model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_epoch(model, optimizer):
    model.train()

    for inputs, labels in train_dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        print(inputs.shape)
        logits = model(inputs)
        print(logits.shape)

        optimizer.zero_grad()

        loss = loss_fn(logits, labels)
    
    return None

for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    # print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))