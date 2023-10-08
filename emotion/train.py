from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from params import *
from model import EmotionClassifier
from dataset import EmotionDataset
from tqdm import tqdm
from datasets import load_dataset
import wandb
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from timeit import default_timer as timer
import matplotlib.pyplot as plt

torch.manual_seed(SEED)

wandb.init(
    # set the wandb project where this run will be logged
    project="emotion-analysis",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "seed": SEED
    }
)

data = load_dataset('go_emotions')
train, val = data['train'].to_pandas(), data["validation"].to_pandas()

train_iter = EmotionDataset(train.text.tolist(), train.labels.tolist(), 'train')
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True)

classifier = EmotionClassifier()
classifier = classifier.to(DEVICE)
wandb.watch(classifier, log_freq=LOG_INTERVAL)

# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

# optimizer = torch.optim.Adam(classifier.parameters(), lr=0.02, weight_decay=1e-3)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_dataloader) * EPOCHS)


def train_epoch(model, optimizer, scheduler):
    model.train()
    losses = 0

    for i, inputs in enumerate(tqdm(train_dataloader)):
        ids = inputs['ids'].to(DEVICE)
        mask = inputs['mask'].to(DEVICE)
        labels = inputs['labels'].to(DEVICE)
        
        logits = model(ids, mask)
                
        optimizer.zero_grad()
        
        loss = loss_fn(logits, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()
        losses += loss.item()

        if (i + 1) % LOG_INTERVAL == 0:
            wandb.log({ "loss": loss.item() })
    
    print(scheduler.get_last_lr())

    return losses / len(train_dataloader)

val_iter = EmotionDataset(val.text.tolist(), val.labels.tolist(), 'val')
val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, shuffle=True)

def evaluate(model):
    model.eval()
    losses = 0

    eval_labels = []
    eval_pred = []
    with torch.no_grad():
        for inputs in tqdm(val_dataloader):
            ids = inputs['ids'].to(DEVICE)
            mask = inputs['mask'].to(DEVICE)
            labels = inputs['labels'].to(DEVICE)

            logits = model(ids, mask)

            loss = loss_fn(logits, labels)
            losses += loss.item()

            eval_labels.extend(labels)
            eval_pred.extend(torch.sigmoid(logits))

    eval_labels = torch.stack(eval_labels).cpu().detach().numpy()
    eval_pred = torch.stack(eval_pred).cpu().detach().numpy()
    
    fpr, tpr, _ = metrics.roc_curve(eval_labels.ravel(), eval_pred.ravel())
    auc_micro = metrics.auc(fpr, tpr)

    return losses / len(val_dataloader), auc_micro

train_losses = []
val_losses = []
for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(classifier, optimizer, scheduler)
    end_time = timer()
    val_loss, auc_micro = evaluate(classifier)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, AUC: {auc_micro:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save(classifier.state_dict(), SAVE_FILE)

plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.legend()
plt.show()