from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_network import EmbeddingGenerator

from base import BaseModel
from .modules.utils import parse_cols_config
from .modules.row_encoder import *
from .modules.rows_aggregator import *
from .modules.temporal_aggregator import *
from constant import target_indices


class MultiIndexModelBase(BaseModel):
    def __init__(self, 
                 cols_config_path="", emb_feat_dim=32, mask_feat_ratio=0, dropout=0.3, hidden_size=128,
                 temporal_aggregator_type="TemporalTransformerAggregator", temporal_aggregator_args={}, 
                 ):
        super().__init__()

        input_dim, num_idxs, cat_dims, cat_idxs, _ = parse_cols_config(cols_config_path)
        self.hidden_size = hidden_size
        self.embedder = FixedEmbedder(input_dim, emb_feat_dim, num_idxs, cat_idxs, cat_dims, mask_feat_ratio)
        self.row_encoder = nn.Sequential(
            nn.Linear(self.embedder.post_embed_dim, hidden_size*4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*4, hidden_size*2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2, hidden_size),
        )
        self.rows_aggregator = nn.Sequential(
            nn.Linear(hidden_size*49, hidden_size*4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*4, hidden_size*2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2, hidden_size),
        )
        self.temporal_aggregator = eval(f"{temporal_aggregator_type}")(**temporal_aggregator_args)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_aggregator_args["hidden_size"], 49),
        )

    def forward(self, x):
        batch_indices, x = x
        dts, shoptags, txn_amt = x[:, 0], x[:, 1], x[:, -1]
        batch_size = int(torch.max(batch_indices) + 1)

        x = self.embedder(x[:, 1:])
        rows_emb = self.row_encoder(x) * txn_amt.view(-1, 1)
        _x = torch.zeros((batch_size, 24, 49, self.hidden_size)).to(x.get_device())
        _x[batch_indices.long(), dts.long(), shoptags.long()] = rows_emb
        x = self.rows_aggregator(_x.view(batch_size, 24, -1))
        x = self.temporal_aggregator(x)
        x = self.classifier(x)
        return x


class BigArch(BaseModel):
    def __init__(self, 
                 row_encoder_type="EmbedderNN", row_encoder_args={}, 
                 rows_aggregator_type="RowsTransformerAggregator", rows_aggregator_args={}, 
                 temporal_aggregator_type="TemporalTransformerAggregator", temporal_aggregator_args={}, 
                 hidden_size=128, num_classes=49):
        super().__init__()
        self.row_encoder = eval(f"{row_encoder_type}")(**row_encoder_args)
        self.rows_aggregator = eval(f"{rows_aggregator_type}")(**rows_aggregator_args)
        self.temporal_aggregator = eval(f"{temporal_aggregator_type}")(**temporal_aggregator_args)
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 49),
        )

    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x, rows_per_month_mask, month_mask = x
        else:
            rows_per_month_mask, month_mask = None, None
        x = self.row_encoder(x, rows_per_month_mask)
        x = self.rows_aggregator(x, rows_per_month_mask)
        x = self.temporal_aggregator(x, month_mask)
        x = self.classifier(x)
        return x



class BigArchBaseLine(BaseModel):
    def __init__(self, cols_config_path, hidden_size=128, num_layers=2, num_classes=49):
        super().__init__()
        self.row_encoder = EmbedderNN(cols_config_path, hidden_size=hidden_size)
        self.rows_aggregator = RowsTransformerAggregator(hidden_size=hidden_size, num_layers=num_layers)
        self.temporal_aggregator = TemporalTransformerAggregator(hidden_size=hidden_size, num_layers=num_layers)
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size*2, 49),
        )

    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x, rows_per_month_mask, month_mask = x
        else:
            rows_per_month_mask, month_mask = None, None
        x = self.row_encoder(x)
        x = self.rows_aggregator(x, rows_per_month_mask)
        x = self.temporal_aggregator(x, month_mask)
        x = self.classifier(x)
        if self.num_classes == 16:
            return x[:, target_indices]
        return x


class SelfAttenNN(BaseModel):
    def __init__(self, cols_config_path, cat_emb=-1, num_classes=49):
        super().__init__()

        input_dim, cat_dims, cat_idxs, cat_emb_dim = parse_cols_config(cols_config_path)
        if cat_emb != -1:
            cat_emb_dim = cat_emb
        self.hidden_size = 128
        self.num_layers = 2

        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.input_nn = nn.Linear(self.embedder.post_embed_dim, self.hidden_size)
        self.special_token = nn.Parameter(torch.randn(self.hidden_size), requires_grad=True)
        self.SelfAttenLayer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8),
            num_layers=self.num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.Dropout(0.3),
            nn.Linear(2*self.hidden_size, num_classes)
        )

    def forward(self, x):
        x_shape = x.size()
        x = self.embedder(torch.reshape(x, (-1, x_shape[-1])))
        x = x.view(x_shape[0], x_shape[1], self.embedder.post_embed_dim)
        x = self.input_nn(x)
        # add special token
        # generate special token to batch_size
        # print(self.special_token)
        special_token = torch.cat([self.special_token]*x_shape[0], dim=0).view(x_shape[0], 1, -1)
        x = torch.cat([special_token, x], dim=1)
        # reshape to S,B,E
        x = x.permute(1, 0, 2)
        x = self.SelfAttenLayer(x)
        x = x.permute(1, 0, 2)
        # run classifier
        x = self.classifier(x[:, 0])
        return x


class GruNN(BaseModel):
    def __init__(self, cols_config_path, cat_emb=-1, num_classes=49):
        super().__init__()

        input_dim, cat_dims, cat_idxs, cat_emb_dim = parse_cols_config(cols_config_path)
        if cat_emb != -1:
            cat_emb_dim = cat_emb
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.hidden_size = 128
        self.num_layers = 5
        self.gru = nn.GRU(
            input_size = self.embedder.post_embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers, 
            batch_first = True
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.Dropout(),
            nn.Linear(2*self.hidden_size, num_classes)
        )

    def forward(self, x):
        x_shape = x.size()
        x = self.embedder(torch.reshape(x, (-1, x_shape[-1])))
        x = torch.reshape(x, (x_shape[0], x_shape[1], self.embedder.post_embed_dim))
        x, _ = self.gru(x)
        x = self.classifier(x[:, -1, :])
        return x

    
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    
    model = SelfAttenNN('../data/preprocessed/v1/column_config_generated.yml')
    data = torch.zeros((32, 200, 52))
    print(model(data))

