import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import YOLO
from tqdm import tqdm
import wandb
import os
from params import *
from timeit import default_timer as timer

torch.manual_seed(SEED)

wandb.init(
    # set the wandb project where this run will be logged
    project="yolo-pretrain",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "seed": SEED
    }
)

train_iter = torchvision.datasets.ImageFolder(root=os.path.join(PRETRAIN_DIRECTORY, 'train'),
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Resize((224, 224))
                                                ]))
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True)

model = YOLO(mode='pretrain')
model.to(DEVICE)
wandb.watch(model, log_freq=LOG_INTERVAL)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(inputs)
        print(logits.shape, labels.shape)

        optimizer.zero_grad()

        loss = loss_fn(logits, labels)
        loss.backward()

        optimizer.step()
        losses += loss.item()

        if (i + 1) % LOG_INTERVAL == 0:
            wandb.log({ "loss": loss.item() })
    
    return losses / len(train_dataloader)

val_iter = torchvision.datasets.ImageFolder(root=os.path.join(PRETRAIN_DIRECTORY, 'val'),
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Resize((224, 224))
                                                ]))
val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, shuffle=True)

def evaluate(model):
    model.eval()
    losses = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(val_dataloader)):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)

            loss = loss_fn(logits, labels)
            losses += loss.item()
    
    return losses / len(val_dataloader)

train_losses = []
for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    val_loss = evaluate(model)
    train_losses.append(train_loss)
    
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

torch.save(model.state_dict(), PRETRAIN_SAVE_FILE)