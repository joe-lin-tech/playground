from torch.utils.data import DataLoader
import torch
from params import *
from dataset import EmotionDataset
from model import EmotionClassifier

train_iter = EmotionDataset(TRAIN_FILE, 'train')
train_dataloader = DataLoader(
    train_iter, batch_size=BATCH_SIZE, collate_fn=train_iter.collate_fn)

classifier = EmotionClassifier(train_iter.vocab, EMBED_SIZE, LSTM_HID_DIM, LINEAR_DIM)

classifier = classifier.to(DEVICE)

classifier.load_state_dict(torch.load(SAVE_FILE))

# actual function to classify input sentence
def classify(model: torch.nn.Module, input_sentence: str):
    model.eval()
    input = [train_iter.vocab.get(t, UNK_IDX) for t in train_iter.token_transform(input_sentence)]

    input = torch.tensor((input + [PAD_IDX] * (TOKEN_LENGTH - len(input)))[:TOKEN_LENGTH]).long().unsqueeze(0).to(DEVICE)
    logits = model(input)
    
    _, output = torch.exp(logits).topk(1)

    return output, EMOTIONS[output]


import json
count = 0
total = 0
with open('../data/emotion/test-emotion.txt') as f:
    for line in f:
        line = json.loads(line)
        comment = line['text']
        result, _ = classify(classifier, comment)
        if result in [int(line['label'])]:
            total += 1
        count += 1
print(total / count)