from CovidPerData import CovidPerData
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
import numpy as np
from Model import DenseNet121
from torch import nn
import os
from tqdm import tqdm
import argparse
import time

class AverageValueMeter():
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.__sum = 0
        self.__num = 0
        
    def add(self, value, num):
        self.__sum += value * num
        self.__num += num
        
    def value(self):
        return self.__sum / self.__num


def MAE(predictions, gt):
    assert predictions.shape == gt.shape
    return ((predictions-gt).abs()).mean()


def RMSE(predictions, gt):
    assert predictions.shape == gt.shape
    return ((predictions-gt)**2).mean()**(1/2)


def pearson_correlation(predictions, gt):
    assert predictions.shape == gt.shape
    v_predictions = predictions - torch.mean(predictions)
    v_gt = gt - torch.mean(gt)      
    coefficient = torch.sum(v_predictions * v_gt) / (torch.sqrt(torch.sum(v_predictions ** 2)) * torch.sqrt(torch.sum(v_gt ** 2)))  
    

def train(network, loader, criterion, lr, weight_decay, epochs, exp_name, logdir, weights, save):
    epochs_done = 0
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    if weights != '':
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epochs_done = checkpoint['epochs'] + 1 #epochs done in previous training phases     
    
    try:
        os.makedirs(os.path.join(logdir, exp_name))
    except:
        pass
    
    logging_file = open(os.path.join(logdir, exp_name, 'console_logs.txt'), 'a')
    title = '\n============== EPOCHS {} LR {} WD {} NOTE {} ==============\n'.format(epochs, lr, weight_decay, note)
    logging_file.writelines(title)
    logging_file.close()
    
    for e in tqdm(range(epochs_done, epochs)):
        for mode in ['train', 'test']:
            loss_meter.reset()
            metric_meter.reset()
            network.train() if mode == 'train' else network.eval()
            
            with torch.set_grad_enabled(mode == 'train'):
                for i, batch in enumerate(loader[mode]):
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    
                    output = network(x.float())
                    loss = criterion(output.float(), y.float().unsqueeze(dim = 1))
                    
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    metric = MAE(output.view(-1), y.view(-1))
                    loss_meter.add(loss.item(), x.shape[0])
                    metric_meter.add(metric, x.shape[0])
                
            logging_file = open(os.path.join(logdir, exp_name, 'console_logs.txt'), 'a')
            logging = 'Epoch {}/{} batches {}/{} mode {} loss {:.4f} metric {:.4f}\n'.format(e + 1, epochs, i + 1, len(loader[mode]), mode, loss_meter.value(), metric_meter.value())
            logging_file.writelines(logging)
            logging_file.close()
            print('\n',logging)

        scheduler.step()
            
        if save:
            torch.save({
                    'epochs': e,
                    'weights': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(logdir, exp_name) + '/%s-%d.tar'%(exp_name, e + 1))
            

def evaluate(network, loader):
    
    network.eval()
    
    predictions, labels = [], []
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(loader):
            print('Processing batch: {}/{}'.format(i + 1, len(loader)))
            x = batch[0].to(device)
            y = batch[1].to(device)
            
            output = network(x.float())
            preds = output.view(-1)
            labs = y.view(-1)
                        
            predictions.extend(list(preds))
            labels.extend(list(labs))
            
    predictions = np.array(predictions)
    labels = np.array(labels)

    print('MAE: ', MAE(output, y))
    print('RMSE: ', RMSE(output, y))
    print('Pearson Correlation: ', pearson_correlation(output, y))


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Data dir')  
#parser.add_argument('--mode', type=str, help='Network mode (training, validation, testing)')  
parser.add_argument('--network', type=str, help='Network model (densenet121, inceptionv3, resnet50)')
parser.add_argument('--logs', type=str, default='logs', help='Logs dir')
parser.add_argument('--weights', type=str, default='', help='Checkpoints dir')  
parser.add_argument('--batch', type=int, default=16, help='Batch size')  
parser.add_argument('--epochs', type=int, default=50, help='Epochs') 
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='Weight decay')
parser.add_argument('--workers', type=int, default=8, help='Workers')
parser.add_argument('--expname', type=str, default='exp', help='Experiment name')
parser.add_argument('--note', type=str, default='lr 0001 wd 0 ADAM', help='Notes for logging file')

opt = parser.parse_args()

if opt.network == 'densenet121':
    network = DenseNet121()
else:
    print('TODO')
criterion = nn.SmoothL1Loss()
lr = opt.lr
weight_decay = opt.wd
optimizer = Adam(network.parameters(), lr = lr, weight_decay = weight_decay)
epochs = opt.epochs
exp_name = opt.expname
logdir = opt.logs
weights = opt.weights
if weights != '':
    print('Use model from checkpoint: ', os.path.basename(weights))
    checkpoint = torch.load(weights)
    network.load_state_dict(checkpoint['weights'])
device = "cuda" if torch.cuda.is_available() else "cpu"
network.to(device)
loss_meter = AverageValueMeter()
metric_meter = AverageValueMeter()
save = True
note = opt.note
#mode = opt.mode
dataset_train = CovidPerData(opt.data, 'training')
loader_train = DataLoader(dataset_train, batch_size = 16, num_workers = opt.workers, shuffle = True)
dataset_test = CovidPerData(opt.data, 'test')
loader_test = DataLoader(dataset_test, batch_size = 16, num_workers = opt.workers, shuffle = True)
loader = {
        'train' : loader_train,
        'test': loader_test
        }


#if mode == 'training':  
train(network, loader, criterion, lr, weight_decay, epochs, exp_name, logdir, weights, save)
#elif mode == 'validation':
    #evaluate(network, loader)     
#    print('TODO')
#elif mode == 'testing':
#    print('TODO')         
                
