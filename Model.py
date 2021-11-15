from torch import nn
import torch
from torch.nn.modules import linear
import torchvision.models as models
import torch.nn.functional as F


class DenseNet121(nn.Module):
    
    def __init__(self, dropout = 0.5, momentum = 0.9):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(pretrained = True, progress = False)
        self.model.classifier = nn.Linear(1024, 1) #changing last layer
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)