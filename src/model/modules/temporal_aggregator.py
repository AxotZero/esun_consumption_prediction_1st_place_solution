from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F



def TimeCnn(input_size=128, output_size=128, time_window=3):
    return nn.Conv1d(input_size, output_size, time_window, padding=(time_window-1, 0))


class Seq2SeqCnnAggregator(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, dropout=0.3):
        super().__init__()
        self.agg = nn.Sequential(
            TimeCnn(input_size, hidden_size, 1),
            nn.SiLU(),
            nn.Dropout(dropout),

            TimeCnn(hidden_size, hidden_size, 5),
            nn.SiLU(),
            nn.Dropout(dropout),

            TimeCnn(hidden_size, hidden_size, 3),
            nn.SiLU(),
            nn.Dropout(dropout),

            TimeCnn(hidden_size, hidden_size, 2),
            nn.SiLU(),
        )

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1) # to (b, emb_dim, 24)
        x = self.agg(x)
        x = x.permute(0, 2, 1)
        return x


class Seq2SeqGruAggregator(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2, dropout=0.3, add_time=False):
        super().__init__()
        
        self.add_time = add_time
        if self.add_time:
            self.time_emb = nn.Parameter(torch.randn(12,16), requires_grad=False)

        self.gru = nn.GRU(
            input_size = input_size + (16 if add_time else 0),
            hidden_size = hidden_size,
            num_layers = num_layers, 
            dropout=dropout,
            batch_first = True
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * ~(mask.unsqueeze(2))
        if self.add_time:
            time_emb = torch.cat([self.time_emb, self.time_emb], dim=0)
            time_emb = torch.cat([torch.unsqueeze(time_emb, dim=0)]*x.size()[0], dim=0)
            x = torch.cat([time_emb, x], dim=-1)
        x, _ = self.gru(x)
        return x



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
        return x
