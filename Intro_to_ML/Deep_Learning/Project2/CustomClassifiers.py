import torch
from torch import nn
import torch.nn.functional as F

# Define custom network architecture
class Classifier(nn.Module):
    def __init__(self,hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        for idx,layer in enumerate(hidden_units):
            i = str(idx+1)
            if idx == 0:
                setattr(self, 'fc'+i, nn.Linear(25088,layer))
            else:
                setattr(self, 'fc'+str(idx+1), nn.Linear(hidden_units[idx-1],layer))
        setattr(self, 'fc'+str(len(hidden_units)+1), nn.Linear(hidden_units[-1],102))

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        for idx,layer in enumerate(self.hidden_units):
            i = str(idx+1)
            tmp = getattr(self,'fc'+i)
            x = F.relu(tmp(x))
        tmp = getattr(self,'fc'+str(len(self.hidden_units)+1))
        x = F.log_softmax(tmp(x), dim=1)

        return x
