from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange


class BatchNormLastDim(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x


class RowsTransformerAggregator(nn.Module):
    def __init__(self, hidden_size=128, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.special_token = nn.Parameter(
            torch.randn(self.hidden_size), requires_grad=False)
        self.AttenLayer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hidden_size, dim_feedforward=4*self.hidden_size, nhead=nhead, dropout=dropout),
            num_layers=self.num_layers
        )

    def forward(self, x, rows_per_month_mask=None):
        bs, ms, n_rows_per_month, hidden_size = x.size()
        # add special token, out_size = (bs, ms, nrows+1, hidden)
        special_token = torch.cat(
            [self.special_token]*bs*ms, dim=0).view(bs, ms, 1, -1)
        x = torch.cat([special_token, x], dim=2)
        # reshape to (bs*ms, nrows+1, hidden)
        x = x.view(-1, n_rows_per_month+1, hidden_size)
        if rows_per_month_mask is not None:
            rows_per_month_mask = F.pad(rows_per_month_mask, (1,0,0,0,0,0), value=False)
            rows_per_month_mask = rows_per_month_mask.view(-1, n_rows_per_month+1)
        # aggregate
        x = x.permute(1, 0, 2)
        x = self.AttenLayer(x, src_key_padding_mask=rows_per_month_mask)
        x = x.permute(1, 0, 2)
        # reshape to bs, ms, hidden
        x = x[:, 0].view(bs, ms, hidden_size)
        return x


class RowsFastformerAggregator(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.special_token = nn.Parameter(
            torch.randn(self.hidden_size), requires_grad=False)
        self.AttenLayer = nn.ModuleList(
            [  
                Fastformer(hidden_size, hidden_size)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x, rows_per_month_mask=None):
        bs, ms, n_rows_per_month, hidden_size = x.size()
        # add special token, out_size = (bs, ms, nrows+1, hidden)
        special_token = torch.cat(
            [self.special_token]*bs*ms, dim=0).view(bs, ms, 1, -1)
        x = torch.cat([special_token, x], dim=2)
        # reshape to (bs*ms, nrows+1, hidden)
        x = x.view(-1, n_rows_per_month+1, hidden_size)
        if rows_per_month_mask is not None:
            rows_per_month_mask = F.pad(rows_per_month_mask, (1,0,0,0,0,0), value=False)
            rows_per_month_mask = rows_per_month_mask.view(-1, n_rows_per_month+1)
        # aggregate
        for layer in self.AttenLayer:
            x = layer(x, mask=rows_per_month_mask)
        # reshape to bs, ms, hidden
        x = x[:, 0].view(bs, ms, hidden_size)
        return x


class Fastformer(nn.Module):
    def __init__(self, dim=128, decode_dim=128):
        super(Fastformer, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        # self.to_qkv = nn.Linear(dim, decode_dim * 3, bias = False)
        self.weight_q = nn.Linear(dim, decode_dim, bias = False)
        self.weight_k = nn.Linear(dim, decode_dim, bias = False)
        self.weight_v = nn.Linear(dim, decode_dim, bias = False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias = False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, mask = None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b n ()')

        # Caculate the global query
        alpha_weight = (torch.mul(query, self.weight_alpha) * self.scale_factor).masked_fill(mask, mask_value)
        alpha_weight = torch.softmax(alpha_weight, dim = -1)
        global_query = query * alpha_weight
        global_query = torch.einsum('b n d -> b d', global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b d -> b copy d', copy = n)
        p = repeat_global_query * key
        beta_weight = (torch.mul(p, self.weight_beta) * self.scale_factor).masked_fill(mask, mask_value)
        beta_weight = torch.softmax(beta_weight, dim = -1)
        global_key = p * beta_weight
        global_key = torch.einsum('b n d -> b d', global_key)

        # key-value
        key_value_interaction = torch.einsum('b j, b n j -> b n j', global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        return result
