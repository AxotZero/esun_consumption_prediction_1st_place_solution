from pdb import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_network import EmbeddingGenerator

from .utils import parse_cols_config


class FixedEmbedderNN(nn.Module):
    def __init__(self, cols_config_path, emb_feat_dim=32, hidden_size=128, dropout=0.3, num_layers=1):
        super().__init__()
        input_dim, num_idxs, cat_dims, cat_idxs, _ = parse_cols_config(cols_config_path)
        self.hidden_size = hidden_size
        self.embedder = FixedEmbedder(input_dim, emb_feat_dim, num_idxs, cat_idxs, cat_dims)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.input_layer = nn.Linear(self.post_embed_dim, self.hidden_size)
        self.num_layers = num_layers
        self.nn_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_size, 2*self.hidden_size),
                    nn.Dropout(dropout),
                    nn.Linear(2*self.hidden_size, self.hidden_size),
                    nn.LayerNorm(self.hidden_size)
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        bs, ms, n_rows_per_month, n_features = x.size()
        x = self.embedder(x.view(-1, n_features))
        x = self.input_layer(x)
        if self.num_layers > 0:
            x = self.nn_layers(x)
        x = x.view(bs, ms, n_rows_per_month, -1)
        return x


class EmbedderNN(nn.Module):
    def __init__(self, cols_config_path, hidden_size=128, dropout=0.3):
        super().__init__()
        input_dim, _, cat_dims, cat_idxs, cat_emb_dim = parse_cols_config(
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


class FixedEmbedder(torch.nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(self, input_dim, emb_dim=32, num_idxs=[], cat_idxs=[], cat_dims=[]):
        """This is an embedding module for an entire set of features
        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        """
        super().__init__()

        self.skip_embedding = False
        self.post_embed_dim = emb_dim*(len(num_idxs)+len(cat_idxs))

        self.embeddings = torch.nn.ModuleList()
        self.nns = torch.nn.ModuleList()

        for cat_dim in cat_dims:
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))
        for _ in num_idxs:
            self.nns.append(nn.Linear(1, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        num_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(
                    self.nns[num_feat_counter](x[:, feat_init_idx].view(-1, 1).float())
                )
                num_feat_counter += 1
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        # set_trace()
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

