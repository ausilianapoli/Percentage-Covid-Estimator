from torch.utils.data.dataset import Dataset as Dataset
from torchvision import transforms
import os
import glob
import numpy as np
import random
import pandas as pd
from PIL import Image
import cv2

SLICE = 0
PERCENTAGE = 1
SUBJECT = 2

class CovidPerData(Dataset):
    
    def __init__(self, data_path = None, mode = 'training', inception = False, predict = False):
        self.__data_path = data_path
        self.__mode = mode
        self.__labels_path = None
        self.__X = None
        self.__Y = None
        np.random.seed(1702)
        random.seed(1702)
        #resize = 299 if inception else 224
        self.__predict = predict
        self.__adapter_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        self.__data_augmentation = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomResizedCrop(512-20), transforms.Resize(512), transforms.RandomRotation((-10, +10))])
        self.__read_data()
        if self.__mode == 'training' or self.__mode == 'test':
            self.__extract_data()
            self.__splitting_data()
            self.__assign_data()
        
    def __read_data(self):
        if self.__mode == 'training' or self.__mode == 'test':
            self.__X = glob.glob(os.path.join(self.__data_path, 'Train', '*.png'))
            self.__labels_path = os.path.join(self.__data_path, 'Train.csv')
        elif self.__mode == 'evaluate':
            self.__X = glob.glob(os.path.join(self.__data_path, 'Val', '*.png'))
            self.__labels_path = os.path.join(self.__data_path, 'Val.csv')
        if os.path.exists(self.__labels_path):
            self.__Y = pd.read_csv(self.__labels_path, header = None) #it's a dataframe

    def __extract_data(self):
        slices = self.__Y[SLICE].values
        percentages = self.__Y[PERCENTAGE].values
        data = {k:v for k,v in zip(slices, percentages)}
        self.__Y = data
                    
    def __assign_data(self):
        if self.__mode == 'training':
            self.__X = self.__X_training
        elif self.__mode == 'test':
            self.__X = self.__X_validation

    def __splitting_data(self):
        validation_len = (len(self.__X) * 10) // 100
        self.__X_training, self.__X_validation = None, None
        idx = np.random.permutation(len(self.__X))
        self.__X_training = self.__X[validation_len:]
        self.__X_validation = self.__X[:validation_len]
                
    def __getitem__(self, index):
        x = Image.open(self.__X[index]).convert('RGB')
        #x = cv2.imread(self.__X[index])
        #x = np.uint8(cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)) #histogram equalization
        #x = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #x = Image.fromarray(x)
        x = self.__adapter_transform(x)
        if self.__mode == 'training':
            x = self.__data_augmentation(x)
        image_name = os.path.basename(self.__X[index])
        if self.__predict:
            y = image_name
        else:
            y = self.__Y[image_name]
        
        return x, y
    
    def __len__(self):
        return len(self.__X)