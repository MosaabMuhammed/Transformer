import torch.nn as nn
from Attention import MultiHeadAttention
from Norm import LayerNorm
from FeedForward import PositionWiseFeedForward
from Embeddings import InputEmbedding

class DecoderLayer(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super().__init__()
        self.masked_multi_att = MultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.multi_att = MultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.norm1 = LayerNorm(d_model)

        self.ff = PositionWiseFeedForward(n_embd=d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, context, trg_mask=None, src_trg_mask=None):
        _x = x
        out = self.masked_multi_att(x, x, x, trg_mask)
        out += _x
        out = self.norm1(out)
        _x = out
        out = self.multi_att(out, context, context, src_trg_mask)
        out += _x
        out = self.norm2(out)
        _x = out
        out = self.ff(out)
        out += _x
        out = self.norm3(out)
        return out

class Decoder(nn.Module):
    def __init__(self, n_layers, n_tokens: int, n_heads: int, d_model: int, device: str):
        super().__init__()
        self.input_embedding = InputEmbedding(n_tokens=n_tokens, n_embd=d_model, device=device)
        self.decoder_layers = nn.ModuleList([DecoderLayer(n_heads, d_model) for _ in range(n_layers)])

        self.fc = nn.Linear(in_features=d_model, out_features=n_tokens)

    def forward(self, x, context, trg_mask=None, src_trg_mask=None):
        x = self.input_embedding(x)

        for layer in self.decoder_layers:
            x = layer(x, context, trg_mask, src_trg_mask)

        out = self.fc(x)
        return out