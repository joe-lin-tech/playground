from torch.utils.data import DataLoader
import torch
from params import *
from tqdm import tqdm
from dataset import EmotionDataset
from transformers import SqueezeBertTokenizer
from model import EmotionClassifier

# train_iter = EmotionDataset(TRAIN_FILE, 'train')
# train_dataloader = DataLoader(
#     train_iter, batch_size=BATCH_SIZE, collate_fn=train_iter.collate_fn)

device = torch.device('mps')

classifier = EmotionClassifier()
classifier = classifier.to(device)
classifier.load_state_dict(torch.load(SAVE_FILE, map_location=device))

tokenizer = SqueezeBertTokenizer.from_pretrained("squeezebert/squeezebert-uncased")

# actual function to classify input sentence
def classify(model: torch.nn.Module, input_sentence: str):
    model.eval()
    input = tokenizer(input_sentence, None, add_special_tokens=True, max_length=MAX_LENGTH, padding="max_length", truncation=True)

    logits = model(torch.tensor([input['input_ids']]).to(device), torch.tensor([input['attention_mask']]).to(device))
    _, output = torch.exp(logits).topk(1)

    return output, EMOTIONS[output]


count = 0
total = 0
with open('../data/emotion/test.tsv') as f:
    for line in tqdm(f, total=5427):
        line = line.split('\t')
        result, _ = classify(classifier, line[0])
        if result in [int(l) for l in line[1].split(',')]:
            total += 1
        count += 1
print(total / count)