from torch.utils.data.dataset import Dataset as Dataset
from torchvision import transforms
import os
import glob
import numpy as np
import random
import pandas as pd
from PIL import Image
import cv2
import copy
import torch

SLICE = 0
PERCENTAGE = 1
SUBJECT = 2

class CovidPerData(Dataset):
    
    def __init__(self, data_path = None, mode = 'training', predict = False, he_processing = False, clahe_processing = False):
        self.__data_path = data_path
        self.__mode = mode
        self.__labels_path = None
        self.__X = None
        self.__Y = None
        self.__subjects = None
        self.__he_processing = he_processing
        self.__clahe_processing = clahe_processing
        np.random.seed(1702)
        random.seed(1702)
        #resize = 299 if inception else 224
        self.__predict = predict
        self.__adapter_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        self.__data_augmentation = transforms.Compose([transforms.GaussianBlur(kernel_size = (5, 5)), transforms.ColorJitter(), transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(), transforms.RandomResizedCrop(512-20), transforms.Resize(512), transforms.RandomRotation((-10, +10)), transforms.RandomAffine(degrees = 0, shear = (10), fillcolor = 0)])
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
        ''' ---splitting based on subjects---
        subjects = self.__Y[SUBJECT].values
        subject_dict = {k:v for k,v in zip(slices, subjects)}
        self.__Y = data_dict
        self.__subjects = subject_dict
        '''
                    
    def __assign_data(self):
        if self.__mode == 'training':
            self.__X = self.__X_training
        elif self.__mode == 'test':
            self.__X = self.__X_validation

    def __splitting_data(self): #to have all slices of one subject in only set
        validation_len = (len(self.__X) * 10) // 100
        self.__X_training, self.__X_validation = None, None
        self.__X_training = self.__X[validation_len:]
        self.__X_validation = self.__X[:validation_len]
        '''
        dirname = os.path.dirname(self.__X[0])
        subjects = np.unique(list(self.__subjects.values()))
        self.__X_validation = []
        subjects_copy = copy.deepcopy(self.__subjects)
        while len(self.__X_validation) < validation_len:
            random_subject = subjects[np.random.randint(0, len(subjects))]
            for k,v in subjects_copy.items():
                if v == random_subject:
                    self.__X_validation.append(os.path.join(dirname,k))
                    self.__subjects.pop(k)
        self.__X_training = list(map(lambda x: os.path.join(dirname, x), list(self.__subjects.keys())))
        '''

    def __HE(self, img):
        new_img = np.zeros(img.shape)
        for i in range(img.shape[2]):
            new_img[:,:,i] = cv2.equalizeHist(img[:,:,i])
        return np.uint8(new_img)

    def __CLAHE(self, img):
        new_img = np.zeros(img.shape)
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
        for i in range(img.shape[2]):
            new_img[:,:,i] = clahe.apply(img[:,:,i])
        return np.uint8(new_img)

    def __mixup(self, x, y):
        random_idx = random.randint(0, self.__len__() - 1)
        random_image_name = os.path.basename(self.__X[random_idx])
        random_x = self.__adapter_transform(Image.open(self.__X[random_idx]).convert('RGB'))
        random_y = self.__Y[random_image_name]
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x + (1 - lam) * random_x
        mixed_y = lam * y + (1 - lam) * random_y
        return mixed_x, mixed_y
                
    def __getitem__(self, index):
        image_name = os.path.basename(self.__X[index])
        if self.__predict:
            y = image_name
        else:
            y = self.__Y[image_name]

        x = cv2.imread(self.__X[index])
        if self.__he_processing:
            x = self.__HE(x)
        elif self.__clahe_processing:
            x = self.__CLAHE(x)
        x = Image.fromarray(x)
        x = self.__adapter_transform(x)
        if self.__mode == 'training':
            x = self.__data_augmentation(x)
            if random.random() > 0.5:
                x, y = self.__mixup(x, y)
        return x, y
                
    
    def __len__(self):
        return len(self.__X)