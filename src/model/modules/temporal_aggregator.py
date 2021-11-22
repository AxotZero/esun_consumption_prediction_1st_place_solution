from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGruAggregator(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers, 
            dropout=dropout,
            batch_first = True
        )
    def forward(self, x, mask):
        x = x * ~(mask.unsqueeze(2))
        x, _ = self.gru(x)
        last_indices = -torch.argmin(torch.flip(mask.type(torch.float), dims=(1,)), dim=1)-1
        return x[range(x.size()[0]), last_indices]


class TemporalTransformerAggregator(nn.Module):
    def __init__(self, hidden_size=128, nhead=8, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.special_token = nn.Parameter(
            torch.randn(self.hidden_size), requires_grad=True)
        self.AttenLayer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=nhead),
            num_layers=self.num_layers
        )

    def forward(self, x, month_mask=None):
        bs, ms, hidden_size = x.size()
        # add special token, out_size = (bs, ms, nrows+1, hidden)
        special_token = torch.cat(
            [self.special_token]*bs, dim=0).view(bs, 1, -1)
        x = torch.cat([special_token, x], dim=1)
        if month_mask is not None:
            month_mask = F.pad(month_mask, (1, 0, 0, 0), value=False)
        # aggregate
        x = x.permute(1, 0, 2)
        x = self.AttenLayer(x, src_key_padding_mask=month_mask)
        x = x.permute(1, 0, 2)
        return x[:, 0]
