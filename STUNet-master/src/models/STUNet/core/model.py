from .STUNet import STUNet
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, *args, **config):
        super().__init__()
        self.model = STUNet(**config)
    
    def forward(self, batch):
        return self.model(**batch)