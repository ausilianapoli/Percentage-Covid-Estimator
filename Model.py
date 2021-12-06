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
        return x#self.sigmoid(x)

class InceptionV3B(nn.Module):

    def __init__(self, dropout = 0.5, momentum = 0.9):
        super(InceptionV3B, self).__init__()
        self.model = models.inception_v3(pretrained = True, progress = False, aux_logits = False)
        self.model.fc = nn.Linear(98304, 1) #changing last layer
        modules = list(self.model.children())
        self.branches = nn.Sequential(*modules[:7])
        self.body = nn.Sequential(*modules[7:])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #x = self.model(x)
        features = []
        for item in x:
            features.append(self.branches(item))
            #print(self.branches(item).shape)
        features = torch.cat(features, 0)
        #print(features.shape)
        #mean = torch.mean(features, 0)
        #print(mean)
        #features_no = self.branches(x[0])
        #features_he = self.branches(x[1])
        #features_clahe = self.branches(x[2])
        #mean = torch.mean(features_no, features_he, features_clahe)
        x = self.body(features)
        return x

class ResNext(nn.Module):

    def __init__(self, dropout = 0.5, momentum = 0.9):
        super(ResNext, self).__init__()
        self.model = models.resnext50_32x4d(pretrained = True, progress = False)
        self.model.fc = nn.Linear(2048, 1) #changing last layer
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.model(x)
        return x#self.sigmoid(x)
    
class InceptionV3Branches(nn.Module):

    def __init__(self, dropout = 0.5, momentum = 0.9):
        super(InceptionV3Branches, self).__init__()
        self.model = models.inception_v3(pretrained = True, progress = False, aux_logits = False)
        self.model.fc = nn.Linear(2048, 1) #changing last layer
        modules = list(self.model.children())
        self.head = nn.Sequential(*modules[:7])
        self.body = nn.Sequential(*modules[7:])
        
    def forward(self, x, x_he, x_clahe):
        x = self.head(x)
        x_he = self.head(x_he)
        x_clahe = self.head(x_clahe)
        x_mixed = torch.cat(x, x_he, x_clahe)
        x_mixed = self.body(x_mixed)
        return x_mixed