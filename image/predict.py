import torch
import torchvision
from model import Classifier, transform
from params import *

test_iter = torchvision.datasets.CIFAR10(root='../data', train=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_iter, batch_size=BATCH_SIZE, shuffle=False)

classifier = Classifier(num_channels=NUM_CHANNELS, num_features=NUM_FEATURES)
classifier.to(DEVICE)

classifier.load_state_dict(torch.load(SAVE_FILE))

def predict(model):
    model.eval()
    acc = 0
    for images, labels in test_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        _, predicted = torch.max(logits, 1)

        acc += (predicted == labels).sum().item()
    return acc

acc = predict(classifier)
print(acc / len(test_iter))