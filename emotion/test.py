from datasets import load_dataset
from dataset import EmotionDataset
from model import EmotionClassifier
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
from params import *

data = load_dataset('go_emotions')
test = data['test'].to_pandas()

test_iter = EmotionDataset(test.text.tolist(), test.labels.tolist(), 'val')
test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('mps')

model = EmotionClassifier()
model = model.to(device)
model.load_state_dict(torch.load(SAVE_FILE, map_location=device))

model.eval()
losses = 0

eval_labels = []
eval_pred = []
with torch.no_grad():
    for inputs in tqdm(test_dataloader):
        ids = inputs['ids'].to(device)
        mask = inputs['mask'].to(device)
        labels = inputs['labels'].to(device)

        logits = model(ids, mask)

        eval_labels.extend(labels)
        eval_pred.extend(torch.sigmoid(logits))

eval_labels = torch.stack(eval_labels).cpu().detach().numpy()
eval_pred = torch.stack(eval_pred).cpu().detach().numpy()

print(eval_labels.shape)
for i, c in enumerate(EMOTIONS):
    fpr, tpr, _ = metrics.roc_curve(eval_labels[:, i], eval_pred[:, i])
    auc_micro = metrics.auc(fpr, tpr)
    print(c, auc_micro)

# fpr, tpr, _ = metrics.roc_curve(eval_labels.ravel(), eval_pred.ravel())
# auc_micro = metrics.auc(fpr, tpr)

print(auc_micro)