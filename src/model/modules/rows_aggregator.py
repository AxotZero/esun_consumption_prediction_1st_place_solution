from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F


class RowsTransformerAggregator(nn.Module):
    def __init__(self, hidden_size=128, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.special_token = nn.Parameter(
            torch.randn(self.hidden_size), requires_grad=True)
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
