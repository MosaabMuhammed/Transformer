import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]

class InputEmbedding(nn.Module):
    def __init__(self, n_tokens: int, n_embd: int, device: str):
        super().__init__()
        self.n_embd = n_embd
        self.embd = nn.Embedding(n_tokens, n_embd)
        self.pe   = PositionalEncoding(d_model=n_embd, max_len=n_embd, device=device)
        self.linear = nn.Linear(in_features=n_embd, out_features=n_embd)

    def forward(self, x):
        out = self.embd(x)
        out += self.pe(x)
        out = self.linear(out)
        out *= torch.sqrt(torch.tensor(self.n_embd))
        return out