from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from params import *
from model import EmotionClassifier
from dataset import EmotionDataset
from timeit import default_timer as timer


train_iter = EmotionDataset(TRAIN_FILE, 'train')
train_dataloader = DataLoader(
    train_iter, batch_size=BATCH_SIZE, collate_fn=train_iter.collate_fn)

classifier = EmotionClassifier(train_iter.vocab, EMBED_SIZE, LSTM_HID_DIM, LINEAR_DIM)

classifier = classifier.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(classifier.parameters(), lr=0.02, weight_decay=1e-3)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for inputs, labels in train_dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        logits = model(inputs)
                
        optimizer.zero_grad()
        
        loss = loss_fn(logits, labels)
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_iter)

val_iter = EmotionDataset(DEV_FILE, 'val')
val_dataloader = DataLoader(
    val_iter, batch_size=BATCH_SIZE, collate_fn=val_iter.collate_fn)

def evaluate(model):
    model.eval()
    losses = 0

    for inputs, labels in val_dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(inputs)

        loss = loss_fn(logits, labels)
        losses += loss.item()

    return losses / len(val_iter)


for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(classifier, optimizer)
    end_time = timer()
    val_loss = evaluate(classifier)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

torch.save(classifier.state_dict(), SAVE_FILE)
