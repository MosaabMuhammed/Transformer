import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        att = q @ k.transpose(-2, -1)
        att /= torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            att = att.masked_fill(mask == 0, -10000)
        
        att = torch.softmax(att, dim=-1)
        out = att @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int=8, d_model: int=512):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be modulos of n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.self_att = ScaledDotProductAttention()
        self.W_out = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.query(q), self.key(k), self.value(v)
        q, k, v = self.split_for_heads(q), self.split_for_heads(k), self.split_for_heads(v)
        x = self.self_att(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.head_dim)
        out = self.W_out(x)
        return out

    def split_for_heads(self, x):
        B, T, C = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return x