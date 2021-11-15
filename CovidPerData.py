from torch.utils.data.dataset import Dataset as Dataset
from torchvision import transforms
import os
import glob
import numpy as np
import random
import pandas as pd
from PIL import Image

SLICE = 0
PERCENTAGE = 1
SUBJECT = 2

class CovidPerData(Dataset):
    
    def __init__(self, data_path = None, mode = 'training'):
        self.__data_path = data_path
        self.__mode = mode
        self.__labels_path = None
        self.__X = None
        self.__Y = None
        np.random.seed(1702)
        random.seed(1702)
        self.__adapter_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        self.__data_augmentation = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomResizedCrop(220), transforms.RandomRotation((-10, +10))])
        self.__assign_data()
        
    def __read_labels(self):
        if os.path.exists(self.__labels_path):
            self.__Y = pd.read_csv(self.__labels_path, header = None) #it's a dataframe 
                    
    def __assign_data(self):
        if self.__mode == 'training':
            self.__X = glob.glob(os.path.join(self.__data_path, 'Train', '*.png'))
            self.__labels_path = os.path.join(self.__data_path, 'Train.csv')
            self.__read_labels()
        elif self.__mode == 'validation':
            self.__X = glob.glob(os.path.join(self.__data_path, 'Val', '*.png'))
            self.__labels_path = os.path.join(self.__data_path, 'Val.csv')
            self.__read_labels()
        elif self.__mode == 'testing':
            self.__X = glob.glob(os.path.join(self.__data_path, 'Test', '*.png'))
            self.__labels_path = os.path.join(self.__data_path, 'Test.csv')
            self.__read_labels()

    def run(self):
        self.__assign_data()
                
    def __getitem__(self, index):
        x = Image.open(self.__X[index]).convert('RGB')
        x = self.__adapter_transform(x)
        if self.__mode == 'training':
            x = self.__data_augmentation(x)
        image_name = os.path.basename(self.__X[index])
        y = float(self.__Y[self.__Y[SLICE] == image_name][PERCENTAGE])

        '''
        if self.__mode == 'training':
            if random.random() > .5:
                x = self.__transform_flip(x)
            if random.random() > .5:
                x = self.__transform_zoom(x)
            if random.random() > .5:
                x = self.__transform_interpolate(x)
            if random.random() > .5:
                x = self.__transform_rotate(x)
            if random.random() > .5:
                x = self.__transform_contrast(x)
            if random.random() > .5:
                x = self.__transform_translate(x)
        
        for t in x:
            self.__transform_normalize(t)
        '''    
        return x, y
    
    def __len__(self):
        return len(self.__X)
        
        
