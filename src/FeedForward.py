import torch
import torcn.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, n_embd: int=512):
        super().__init__()
        self.fc1 = nn.Linear(in_features=n_embd, out_features=4*n_embd)
        self.fc2 = nn.Linear(in_features=4*n_embd, out_features=n_embd)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x