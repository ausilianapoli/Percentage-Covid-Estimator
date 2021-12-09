from CovidPerData import CovidPerData
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import torch
import numpy as np
from Model import DenseNet121, InceptionV3, ResNext, InceptionV3Branches
from torch import nn
import os
from tqdm import tqdm
import argparse
from matplotlib import pyplot as plt
import csv

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
    

def train(network, loader, criterion, epochs, exp_name, logdir, weights, save):
    loss_meter = AverageValueMeter()
    metric_meter = AverageValueMeter()
    
    epochs_done = 0
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    if weights != '':
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epochs_done = checkpoint['epochs'] + 1 #epochs done in previous training phases
        criterion_parameter = checkpoint['criterion']
        try:
            criterion.delta = criterion_parameter
        except:
            criterion.beta = criterion_parameter
    
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
                    if len(batch) == 2:
                        x = batch[0].to(device)
                        y = batch[1].to(device)
                        output = network(x.float())
                    else:
                        x = batch[0].to(device)
                        x_he = batch[1].to(device)
                        x_clahe = batch[2].to(device)
                        y = batch[3].to(device)
                        output = network(x.float(), x_he.float(), x_clahe.float())
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
        try:
            if criterion.delta > 1:
                criterion.delta -= 0.5
                criterion_parameter = criterion.delta
        except:
            if criterion.beta > 1:
                criterion.beta -= 0.5
                criterion_parameter = criterion.beta
            
        if save:
            torch.save({
                    'epochs': e,
                    'weights': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'criterion': criterion_parameter
                }, os.path.join(logdir, exp_name) + '/%s-%d.tar'%(exp_name, e + 1))
            

def predict(network, loader, logdir, exp_name):
    network.eval()
    results = dict()
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(loader):
            print('Processing batch: {}/{}'.format(i + 1, len(loader)))
            x = batch[0].to(device)
            y = batch[1]
            output = network(x.float())
            results[y[0].replace("'", "")] = output.item()

    csv.register_dialect('myDialect', delimiter = ',', quoting=csv.QUOTE_NONE)
    with open(os.path.join(logdir, exp_name, 'predictions.csv'), 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for key in results.keys():
           writer.writerow((key, str(results[key])))


def sort_logs(logs):
    logs = sorted(logs.items())
    return [t[1] for t in logs]

def plot_learning_curve(filename, title):
    logs_file = open(filename, 'r')
    logs_list = logs_file.readlines()
    train_loss = dict()
    train_accuracy = dict()
    test_loss = dict()
    test_accuracy = dict()
    for row in logs_list:
        epoch = int(row.split(' ')[1].split('/')[0])
        loss = float(row.split(' ')[7])
        accuracy = float(row.split(' ')[-1])
        if 'train' in row: 
            train_loss[epoch] = loss
            train_accuracy[epoch] = accuracy
        else:
            test_loss[epoch] = loss
            test_accuracy[epoch] = accuracy
    train_loss = sort_logs(train_loss)
    train_accuracy = sort_logs(train_accuracy)
    test_loss = sort_logs(test_loss)
    test_accuracy = sort_logs(test_accuracy)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize = (20,8)) 
    plt.plot(train_loss, label = 'training loss')
    plt.plot(test_loss, label = 'validation loss')
    plt.legend(loc = 'upper right')
    plt.title(title)
    plt.xticks(np.arange(0, len(train_loss) + 1, step = 5))
    plt.grid()
    plt.show()
    plt.figure(figsize = (20,8)) 
    plt.plot(train_accuracy, label = 'training metric')
    plt.plot(test_accuracy, label = 'validation metric')
    plt.legend(loc = 'upper right')
    plt.title(title)
    plt.grid()
    #plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, len(train_loss) + 1, step = 5))
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Data dir')  
parser.add_argument('--action', type=str, help='Action to run (train, predict, evaluate, plot)')  
parser.add_argument('--network', type=str, help='Network model (densenet121, inceptionv3, resnext50, inceptionv3b)')
parser.add_argument('--logs', type=str, default='logs', help='Logs dir')
parser.add_argument('--weights', type=str, default='', help='Checkpoints dir')  
parser.add_argument('--batch', type=int, default=16, help='Batch size')  
parser.add_argument('--epochs', type=int, default=50, help='Epochs') 
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='Weight decay')
parser.add_argument('--workers', type=int, default=8, help='Workers')
parser.add_argument('--expname', type=str, default='exp', help='Experiment name')
parser.add_argument('--note', type=str, default='lr 0001 wd 0 ADAM', help='Notes for logging file')
parser.add_argument('--he', action = 'store_true', help = 'If True, apply HE preprocessing')
parser.add_argument('--clahe', action = 'store_true', help = 'If True, apply CLAHE preprocessing')

opt = parser.parse_args()

if opt.action == 'plot':
    plot_learning_curve(opt.data, 'NN')
else:
    branches = False
    if opt.network == 'densenet121':
        network = DenseNet121()
    elif opt.network == 'inceptionv3':
        network = InceptionV3()
        inception = True
    elif opt.network == 'resnext50':
        network = ResNext()
    elif opt.network == 'inceptionv3b':
        network = InceptionV3Branches()
        branches = True
    weights = opt.weights
    if weights != '':
        print('Use model from checkpoint: ', os.path.basename(weights))
        checkpoint = torch.load(weights)
        network.load_state_dict(checkpoint['weights'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network.to(device)
    exp_name = opt.expname
    logdir = opt.logs
    if opt.action == 'training':
        criterion_parameter = 15
        try:
            criterion = nn.HuberLoss(delta = criterion_parameter)
        except:
            criterion = nn.SmoothL1Loss(beta = criterion_parameter)
        #criterion = nn.L1Loss()
        lr = opt.lr
        weight_decay = opt.wd
        #optimizer = Adam(network.parameters(), lr = lr, weight_decay = weight_decay)
        optimizer = SGD(network.parameters(), lr = lr, weight_decay = weight_decay)
        epochs = opt.epochs
        note = opt.note
        save = True
        dataset_train = CovidPerData(opt.data, mode = 'training', branches = branches, he_processing = opt.he, clahe_processing = opt.clahe)
        loader_train = DataLoader(dataset_train, batch_size = opt.batch, num_workers = opt.workers, shuffle = True)
        dataset_test = CovidPerData(opt.data, mode = 'test', branches = branches, he_processing = opt.he, clahe_processing = opt.clahe)
        loader_test = DataLoader(dataset_test, batch_size = opt.batch, num_workers = opt.workers, shuffle = True)
        loader = {
                'train' : loader_train,
                'test': loader_test
                }
        train(network, loader, criterion, epochs, exp_name, logdir, weights, save)
    elif opt.action == 'predict':
        dataset = CovidPerData(opt.data, mode = 'evaluate', branches = branches, predict = True, he_processing = opt.he, clahe_processing = opt.clahe)
        loader = DataLoader(dataset, batch_size = 1, num_workers = opt.workers)
        predict(network, loader, logdir, exp_name)
    elif opt.action == 'evaluate':
        print('TODO')