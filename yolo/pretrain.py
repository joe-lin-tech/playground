import torchvision
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)

writer = SummaryWriter()

def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(inputs)

        optimizer.zero_grad()

        loss = loss_fn(logits, labels)
        loss.backward()

        optimizer.step()
        losses += loss.item()
        writer.add_scalar("Loss/train", loss, i)
    
    return losses / len(train_iter)

train_losses = []
for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    train_losses.append(train_loss)
    
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    # print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

torch.save(model.state_dict(), PRETRAIN_SAVE_FILE)