import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import SqueezeBertModel
import numpy as np
from params import *
        

class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.bert_model = SqueezeBertModel.from_pretrained("squeezebert/squeezebert-uncased")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, len(EMOTIONS))
    
    def forward(self, ids: Tensor, mask: Tensor):
        return self.out(self.dropout(self.bert_model(ids, attention_mask=mask)['pooler_output']))
    