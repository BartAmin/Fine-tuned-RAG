import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, df):
        self.q         = torch.tensor(np.stack(df['question_embedding'].values), dtype=torch.float32)
        self.t         = torch.tensor(np.stack(df['text_embedding'].values),     dtype=torch.float32)
        self.chunk_ids = df['chunk_id'].tolist()  

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        return self.q[idx], self.t[idx], self.chunk_ids[idx] 


class NeuralNet(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, dropout):
        super().__init__()

        layers, in_dim = [], dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, dim))
        self.net            = nn.Sequential(*layers)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(self.net(x) + self.residual_scale * x, dim=1)


class MNRLossWithMasking(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, t, chunk_ids):
        q = F.normalize(q, dim=1)
        t = F.normalize(t, dim=1)
        logits = (q @ t.T) / self.temperature

        mask = torch.zeros_like(logits, dtype=torch.bool)
        for i, cid_i in enumerate(chunk_ids):
            for j, cid_j in enumerate(chunk_ids):
                if i != j and cid_i == cid_j:
                    mask[i, j] = True
        logits[mask] = -1e9

        labels = torch.arange(len(q), device=q.device)
        return F.cross_entropy(logits, labels)