import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, n_src_tokens: int, n_trg_tokens: int, n_layers: int, n_heads: int, d_model: int, src_pad_idx: int, trg_pad_idx: int, device: str):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.encoder = Encoder(n_layers, n_src_tokens, n_heads, d_model, device)
        self.decoder = Decoder(n_layers, n_trg_tokens, n_heads, d_model, device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        trg_mask = self.make_future_token_mask(trg, trg) * self.make_pad_mask(trg, trg)
        src_trg_mask = self.make_pad_mask(trg, src)
        context = self.encoder(src, mask=src_mask)
        out = self.decoder(trg, context, trg_mask, src_trg_mask)
        return out

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        q = (q != self.src_pad_idx).view(-1, 1, len_q, 1)
        k = (k != self.trg_pad_idx).view(-1, 1, 1, len_k)
        mask = q & k
        return mask.long()

    def make_future_token_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.triu(torch.ones((len_q, len_k)), diagonal=1) == 0
        return mask.type(torch.BoolTensor).to(self.device)