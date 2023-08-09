import torch
import torchvision
from model import Classifier, transform
from timeit import default_timer as timer
from params import *

train_iter = torchvision.datasets.CIFAR10(root='../data', train=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True)

classifier = Classifier(num_channels=NUM_CHANNELS, num_features=NUM_FEATURES)
classifier.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for images, labels in train_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        logits = model(images)
                
        optimizer.zero_grad()
        
        loss = loss_fn(logits, labels)
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_iter)

train_losses = []
for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(classifier, optimizer)
    end_time = timer()
    train_losses.append(train_loss)
    
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

torch.save(classifier.state_dict(), SAVE_FILE)