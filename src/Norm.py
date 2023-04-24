import torch
import torcn.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, out_features: int, e: float=1e-05):
        super().__init__()
        self.e = e
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta  = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True) 
        var = x.var(dim=-1, keepdims=True)
        x = x - mean / torch.sqrt(var + self.e)
        out = self.gamma * x + self.beta
        return out