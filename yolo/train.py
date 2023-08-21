import torchvision
import torch
import torch.nn as nn
from model import YOLO
from params import *
from timeit import default_timer as timer

def collate_fn(batch):
    input_batch = []
    label_batch = []
    for input, label in batch:
        input_batch.append(input)
        label_batch.append(label)
    return torch.stack(input_batch), label_batch

train_iter = torchvision.datasets.VOCDetection(root=ROOT_DIRECTORY,
                                               image_set="train",
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Resize((448, 448))
                                                ]))
train_dataloader = torch.utils.data.DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = YOLO()
model.to(DEVICE)

# loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_epoch(model, optimizer):
    model.train()

    for inputs, labels in train_dataloader:
        inputs = inputs.to(DEVICE)
        detections = model(inputs)
        print(detections.shape)

        optimizer.zero_grad()
    
    return None

for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    # print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))