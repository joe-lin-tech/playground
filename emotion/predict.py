from torch.utils.data import DataLoader
import torch
from params import *
from dataset import EmotionDataset
from model import EmotionClassifier

train_iter = EmotionDataset(TRAIN_FILE, 'train')
# train_dataloader = DataLoader(
#     train_iter, batch_size=BATCH_SIZE, collate_fn=train_iter.collate_fn)

classifier = EmotionClassifier()

classifier = classifier.to(DEVICE)

classifier.load_state_dict(torch.load(SAVE_FILE))

# actual function to classify input sentence
def classify(model: torch.nn.Module, input_sentence: str):
    model.eval()
    input = train_iter.tokenizer(input_sentence, None, add_special_tokens=True, max_length=MAX_LENGTH, padding="max_length", truncation=True)

    logits = model(torch.tensor([input['input_ids']]).to(DEVICE), torch.tensor([input['attention_mask']]).to(DEVICE))
    
    _, output = torch.exp(logits).topk(1)

    return output, EMOTIONS[output]


# import json
# count = 0
# total = 0
# with open('../data/emotion/test-emotion.txt') as f:
#     for line in f:
#         line = json.loads(line)
#         comment = line['text']
#         result, _ = classify(classifier, comment)
#         if result in [int(line['label'])]:
#             total += 1
#         count += 1
# print(total / count)
# count = 0
# total = 0
# with open('../data/emotion/train.tsv') as f:
#     for line in f:
#         line = line.split('\t')
#         result, _ = classify(classifier, line[0])
#         if result in [int(l) for l in line[1].split(',')]:
#             total += 1
#         count += 1
# print(total / count)
print(classify(classifier, "Thank you friend"))