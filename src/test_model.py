from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from pytorch_tabnet.tab_network import EmbeddingGenerator

from model.model import *

model = BigArchBaseLine('../data/preprocessed/v1/column_config_wo_dt.yml')
data = torch.zeros((2, 2, 10, 51))
print(model(data).size())
