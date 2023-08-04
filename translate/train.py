from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from params import *
from model import Seq2SeqTransformer
from dataset import TranslationDataset
from timeit import default_timer as timer


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


train_iter = TranslationDataset(TRAIN_FILE, 'train')
train_dataloader = DataLoader(
    train_iter, batch_size=BATCH_SIZE, collate_fn=train_iter.collate_fn)

SRC_VOCAB_SIZE = len(train_iter.src_enc_dict)
TGT_VOCAB_SIZE = len(train_iter.tgt_enc_dict)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for src, tgt in train_dataloader:
        src = torch.transpose(src, 0, 1).to(DEVICE)
        tgt = torch.transpose(tgt, 0, 1).to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        
    return losses / len(train_iter)

dicts = {}
dicts['src_enc_dict'] = train_iter.src_enc_dict
dicts['src_dec_dict'] = train_iter.src_dec_dict
dicts['tgt_enc_dict'] = train_iter.tgt_enc_dict
dicts['tgt_dec_dict'] = train_iter.tgt_dec_dict
val_iter = TranslationDataset(DEV_FILE, 'val', dicts)
val_dataloader = DataLoader(
    val_iter, batch_size=BATCH_SIZE, collate_fn=val_iter.collate_fn)

def evaluate(model):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = torch.transpose(src, 0, 1).to(DEVICE)
        tgt = torch.transpose(tgt, 0, 1).to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_iter)


for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

torch.save(transformer.state_dict(), SAVE_FILE)