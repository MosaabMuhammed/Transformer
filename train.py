import torch
import torch.nn as nn
from config import Config
from torch.utils.data import DataLoader
from dataset import train_de, train_en, en_encode, de_encode, en_cnter, de_cnter
from dataset import TranslationDataset
from dataset import en_stoi, de_stoi
from dataset import valid_en, valid_de, pad_collate_fn
from src.Transformer import Transformer

train_dataset = TranslationDataset(source_samples=train_en, 
                                   target_samples=train_de, 
                                   source_encode=en_encode, 
                                   target_encode=de_encode)
valid_dataset = TranslationDataset(source_samples=valid_en, 
                                   target_samples=valid_de, 
                                   source_encode=en_encode, 
                                   target_encode=de_encode)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, drop_last=True, pin_memory=True, collate_fn=pad_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, drop_last=False, pin_memory=True, collate_fn=pad_collate_fn)

model = Transformer(n_src_tokens=len(en_cnter), 
                    n_trg_tokens=len(de_cnter), 
                    n_layers=Config.n_layers, 
                    n_heads=Config.n_heads, 
                    d_model=Config.d_model, 
                    src_pad_idx=en_stoi['<pad>'],
                    trg_pad_idx=de_stoi['<pad>'],
                    device=Config.device).to(Config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
loss_module = nn.CrossEntropyLoss(ignore_index=de_stoi['<pad>'])

src_batch, trg_batch = next(iter(train_loader))

for epoch in range(Config.n_epochs):
    for src_batch, trg_batch in train_loader:
        src_batch, trg_batch = src_batch.to(Config.device), trg_batch.to(Config.device)

        optimizer.zero_grad(set_to_none=True)

        out_batch = model(src_batch, trg_batch[:, :-1])
        out = out_batch.reshape(-1, out_batch.shape[-1])
        loss = loss_module(out, trg_batch[:, 1:].reshape(-1))

        loss.backward()
        optimizer.step()
        print(loss.item())