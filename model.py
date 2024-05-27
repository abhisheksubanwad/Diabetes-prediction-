import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=8, h1=12, h2=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

def load_model(path):
    model = Model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
