import numpy as np
import torch
from torch import nn as nn
import torch.optim as optim
from torch.nn import functional as F

class Metrics(nn.Module):
    def __init__(self, ):
        return
    def forward(self, predict, target):
        corr = 0
        for finger in range(5):
            if finger == 3:
                continue
            data = troch.stack((predict[:,finger], target[:,finger]), dim=0)
            corr += torch.corrcoef(x)[0,1]
        return corr / 4