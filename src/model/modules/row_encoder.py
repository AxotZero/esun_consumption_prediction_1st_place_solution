from pdb import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_network import EmbeddingGenerator

from .utils import parse_cols_config





class TransformerRowEncoder(nn.Module):
    def __init__(self, cols_config_path, emb_feat_dim=64, output_size=128, nhead=4, dropout=0.3, num_layers=1):
        super().__init__()

        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.emb_feat_dim = emb_feat_dim

        input_dim, num_idxs, cat_dims, cat_idxs, _ = parse_cols_config(cols_config_path)
        self.embedder = FixedEmbedder(input_dim, emb_feat_dim, num_idxs, cat_idxs, cat_dims)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.special_token = nn.Parameter(
            torch.ones(self.emb_feat_dim), requires_grad=False)
        self.attn_layer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.emb_feat_dim, dim_feedforward=4*self.emb_feat_dim, nhead=nhead, dropout=dropout),
            num_layers=self.num_layers
        )
        self.output_layer = nn.Linear(self.emb_feat_dim, self.output_size)

    def forward(self, x, mask=None):
        bs, ms, n_rows_per_month, n_features = x.size()
        x = self.embedder(x.view(-1, n_features))
        x = x.view(-1, n_features, self.emb_feat_dim)

        mask = mask.view(-1)
        special_token = torch.cat(
            [self.special_token]*(torch.sum(~mask)), dim=0).view(-1, 1, self.emb_feat_dim)
        ret = torch.zeros(bs*ms*n_rows_per_month, self.output_size).to(x.get_device())
        _x = torch.cat([special_token, x[~mask]], dim=1).permute(1, 0, 2)
        _x = self.output_layer(self.attn_layer(_x).permute(1, 0, 2)[:, 0])
        ret[~mask] = _x
        return ret.view(bs, ms, n_rows_per_month, -1)


class FixedEmbedder1DCNN(nn.Module):
    def __init__(self, cols_config_path, emb_feat_dim=32, num_targets=128, hidden_size=128, 
                 dropout=0.3, mask_feat_ratio=0, mask_row_ratio=0, 
                 apply_bn=True, preserve_mask=False):
        super().__init__()
        input_dim, num_idxs, cat_dims, cat_idxs, _ = parse_cols_config(cols_config_path)
        self.hidden_size = hidden_size
        self.num_targets = num_targets
        self.embedder = FixedEmbedder(input_dim, emb_feat_dim, num_idxs, cat_idxs, cat_dims, mask_feat_ratio)

        self.cnn_encoder = (CnnEncoder if apply_bn else CnnEncoderNoBN)(self.embedder.post_embed_dim, 
                                      num_targets=num_targets, 
                                      hidden_size=hidden_size,
                                      dropout=dropout)
        self.mask_row_ratio = mask_row_ratio
        self.preserve_mask = preserve_mask

    def forward(self, x, mask=None):
        bs, ms, n_rows_per_month, n_features = x.size()
        x = self.embedder(x.view(-1, n_features))
        if mask is not None:
            mask = mask.view(-1)
            if self.training and self.mask_row_ratio > 0:
                ratio_mask = (torch.rand(len(mask)) < self.mask_row_ratio).to(x.get_device())
                mask = torch.logical_or(mask, ratio_mask)
            if self.preserve_mask:
                ratio_mask = (torch.rand(len(mask)) < 0.01).to(x.get_device())
                mask[ratio_mask] = 0
            ret = torch.zeros(bs*ms*n_rows_per_month, self.num_targets).to(x.get_device())
            ret[~mask] = self.cnn_encoder(x[~mask])
            x = ret
        else:
            x = self.cnn_encoder(x)
        x = x.view(bs, ms, n_rows_per_month, -1)
        return x


class CnnEncoder(nn.Module):
    """
    src: https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
    """
    def __init__(self, num_features, num_targets=128, hidden_size=512, dropout=0.3):
        super().__init__()
        cha_1 = 64
        cha_2 = 128
        cha_3 = 128

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(dropout*0.9)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(dropout*0.8)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(dropout*0.6)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(dropout*0.5)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()
        
        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(dropout)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                        self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class CnnEncoderNoBN(nn.Module):
    """
    src: https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
    """
    def __init__(self, num_features, num_targets=128, hidden_size=512, dropout=0.3):
        super().__init__()
        cha_1 = 64
        cha_2 = 128
        cha_3 = 128

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.dropout_c1 = nn.Dropout(dropout)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.dropout_c2 = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.dropout_c2_1 = nn.Dropout(dropout)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.dropout_c2_2 = nn.Dropout(dropout)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()
        
        self.dropout3 = nn.Dropout(dropout)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                        self.cha_1_reshape)

        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.dropout3(x)
        x = self.dense3(x)

        return x



class FixedEmbedderNN(nn.Module):
    def __init__(self, cols_config_path, emb_feat_dim=32, hidden_size=128, output_size=-1, dropout=0.3, num_layers=1):
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
        self.output_layer = nn.Linear(self.hidden_size, output_size) if output_size != -1 else nn.Identity()
        
    def forward(self, x, mask=None):
        bs, ms, n_rows_per_month, n_features = x.size()
        x = self.embedder(x.view(-1, n_features))
        x = self.input_layer(x)
        # if mask is not None:
        #     mask = mask.view(-1)
        #     ret = torch.zeros(bs*ms*n_rows_per_month, self.hidden_size).to(x.get_device())
        #     ret[~mask] = self.nn_layers(x[~mask])
        #     x = ret
        # else:
        #     x = self.nn_layers(x)
        x = self.nn_layers(x)
        x = self.output_layer(x)
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
    src: tabnet
    """

    def __init__(self, input_dim, emb_dim=32, num_idxs=[], cat_idxs=[], cat_dims=[], mask_feat_ratio=0):
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

        self.mask_feat_ratio = mask_feat_ratio
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
        
        cols = []
        cat_feat_counter = 0
        num_feat_counter = 0
        batch_size = x.size()[0]
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

            # mask features(exclueds shoptag, txn_cnt, txn_amt)
            if self.training and self.mask_feat_ratio > 0 and feat_init_idx>2:
                mask = torch.rand(batch_size) < self.mask_feat_ratio
                cols[-1][mask] = 0
        # concat
        # set_trace()
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

