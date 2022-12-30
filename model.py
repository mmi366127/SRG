import torch.nn.functional as F
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.w = nn.Linear(dim, 1, bias = False)
        with torch.no_grad():
            self.w.weight.data = torch.zeros(self.w.weight.size()) 

    def forward(self, x):
        return torch.sigmoid(self.w(x))

class LinearRegression(nn.Module):
    def __init__(self, dim: int):
        super(LinearRegression, self).__init__()
        self.w = nn.Linear(dim, 1, bias = False)
        with torch.no_grad():
            self.w.weight.data = torch.zeros(self.w.weight.size())
        
    def forward(self, x):
        return self.w(x)

