from torch.utils.data import DataLoader
import torch
from params import *
from dataset import TranslationDataset
from model import Seq2SeqTransformer

train_iter = TranslationDataset(TRAIN_FILE, 'train')
train_dataloader = DataLoader(
    train_iter, batch_size=BATCH_SIZE, collate_fn=train_iter.collate_fn)

SRC_VOCAB_SIZE = len(train_iter.src_enc_dict)
TGT_VOCAB_SIZE = len(train_iter.tgt_enc_dict)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

transformer = transformer.to(DEVICE)

transformer.load_state_dict(torch.load(SAVE_FILE))

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    print(src_sentence)
    src = [train_iter.src_enc_dict.get(t, UNK_IDX) for t in train_iter.token_transform(src_sentence)]

    src = torch.transpose(torch.tensor(src).long().unsqueeze(0), 0, 1).to(DEVICE)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 10, start_symbol=BOS_IDX).flatten()
    return " ".join([train_iter.tgt_dec_dict.get(t, 'UNK') for t in list(tgt_tokens.cpu().numpy())])

print(translate(transformer, "Anyone can do that."))
print(translate(transformer, "We are a family."))
print(translate(transformer, "I don't feel like taking a walk this morning."))
print(translate(transformer, "Hello everyone."))
print(translate(transformer, "This is my name."))