import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(x)

model = Model(dropout=0.3)
torch.save(model.state_dict(), 'test.pth')
print(model)
model = Model(dropout=0.5)
model.load_state_dict(torch.load('test.pth'))
print(model)

