import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_network import EmbeddingGenerator

from .utils import parse_cols_config


class EmbedderNN(nn.Module):
    def __init__(self, cols_config_path, hidden_size=128, dropout=0.3):
        super().__init__()
        input_dim, cat_dims, cat_idxs, cat_emb_dim = parse_cols_config(
            cols_config_path)
        self.hidden_size = hidden_size
        self.embedder = EmbeddingGenerator(
            input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.nn = nn.Sequential(
            nn.Linear(self.post_embed_dim, self.hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        bs, ms, n_rows_per_month, hidden_size = x.size()
        x = self.embedder(x.view(-1, hidden_size))
        x = x.view(bs, ms, n_rows_per_month, -1)
        return self.nn(x)
