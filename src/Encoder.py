import torch.nn as nn
from Attention import MultiHeadAttention
from Norm import LayerNorm
from FeedForward import PositionWiseFeedForward
from Embeddings import InputEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super().__init__()
        self.multi_att = MultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.norm1 = LayerNorm(d_model)

        self.ff = PositionWiseFeedForward(n_embd=d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask):
        _x = x
        out = self.multi_att(x, x, x, mask)
        _x = self.norm1(out + _x)
        out = self.ff(out)
        out = self.norm2(out + _x)
        return out

class Encoder(nn.Module):
    def __init__(self, n_layers, n_tokens: int, n_heads: int, d_model: int, device):
        super().__init__()
        self.input_embedding = InputEmbedding(n_tokens=n_tokens, n_embd=d_model, device=device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, d_model) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        x = self.input_embedding(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x