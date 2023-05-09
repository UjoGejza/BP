# training_detection.py
# Author: Sebastián Chupáč
# This is a training script for typo detection position dependent. 
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
from numpy import random

from dataset import MyDataset
#import class of which model you want to train
from models import ConvLSTMDetectionBigger
import ansi_print

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', type=int, default=50)#batch size
    parser.add_argument('-mi', type=int, default=200_000)#maximum iterations
    parser.add_argument('-lr', type=float, default=0.001)#learning rate
    parser.add_argument('-lr_scale', type=float, default=0.9)#learning rate scale
    parser.add_argument('-lr_scaleiter', type=int, default=10_000)#scale learning rate every lr_scaleiter iterations
    parser.add_argument('-online', type=int, default=1)#generate typos during training
    parser.add_argument('-centre', type=int, default=6)#centre value of gaussan distribution of number of typos per sample 
    parser.add_argument('-spread', type=int, default=2)#spread value of gaussan distribution of number of typos per sample
    parser.add_argument('-load_model', type=str, default='_')#file from to load parameters to continue training
    parser.add_argument('-save_model', type=str, default='ConvLSTMDetection.pt')#file to save trained model to 
    parser.add_argument('-train_file', type=str, default='data/wiki_2M.txt')#input training file, if online = 0 make sure training file contains typos, and vice versa
    parser.add_argument('-test_train_file', type=str, default='data/wiki_train_test_1k_typos.txt')#input validation file
    parser.add_argument('-test_test_file', type=str, default='data/wiki_test_test_1k_typos2.txt')#input testing file
    return parser.parse_args()

args = parseargs()

print(args)
#if GPU with cuda is available it will be use for training to boost performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USING: {device}')

batch_size = args.bs
max_iterations = args.mi
learning_rate = args.lr
learning_rate_scale = args.lr_scale
learning_rate_scale_iter = args.lr_scaleiter
online = args.online
centre = args.centre
spread = args.spread
load_model = args.load_model
save_model = args.save_model
train_file = args.train_file
test_test_file = args.test_test_file
test_train_file = args.test_train_file

#crate MyDataset class and DataLoader class for all input files
training_data = MyDataset(train_file)
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
testing_test_data = MyDataset(test_test_file)
testing_test_data_loader = DataLoader(testing_test_data, shuffle=True)
testing_train_data = MyDataset(test_train_file)
testing_train_data_loader = DataLoader(testing_train_data, shuffle=True)

#load charset from dataset
alphabet = training_data.charlist_base

channels = len(alphabet)
#create the model class
model = ConvLSTMDetectionBigger()
print('model class: ConvLSTMDetectionBigger')
#load parameters if file path provided
if load_model !='_': model = torch.load(load_model)
model.to(device)
model.train()
#set the loss function here, detection uses BCELoss
loss_fn = nn.BCELoss()
print(f'MODEL ARCHITECTURE: ')
for name, param in model.state_dict().items():
    print(name, param.size())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

green = 'green'
yellow = 'yellow'
red = 'red'
white = 'white'
id = 'id'

#generates "constant" number of typos for each sample 
def add_typos(item):
    error_index = random.randint(50, size=(4*50))
    error_char = random.randint(low=97, high=123, size=(4*50))
    extra_char_one_hot = torch.zeros(1, channels)
    extra_char_one_hot[0][alphabet.index('#')] = 1
    for i in range(4*50):
        if (i%4)<3:
            #swap char for another char
            if chr(error_char[i]) != item['label_text'][i//4][error_index[i]]:
                item['bi_class'][i//4][error_index[i]] = 0
                item['sample_one_hot'][i//4][error_index[i]] = torch.zeros(channels)#training_data.channels
                item['sample_one_hot'][i//4][error_index[i]][alphabet.index(chr(error_char[i]))] = 1
        else:
            #insert extra char
            base_one_hot = torch.zeros(1, channels)
            base_one_hot[0][alphabet.index(chr(error_char[i]))] = 1
            label_text = list(item['label_text'][i//4])
            bi_class = list(item['bi_class'][i//4])
            label_text.insert(error_index[i], '#')
            bi_class.insert(error_index[i], 0)
            item['sample_one_hot'][i//4] = torch.cat((item['sample_one_hot'][i//4][:error_index[i]], base_one_hot, item['sample_one_hot'][i//4][error_index[i]:-1]), 0)

            label_text.pop(len(label_text)-1)
            bi_class.pop(len(bi_class)-1)

            item['label_text'][i//4] = ''.join(label_text)
            item['bi_class'][i//4] = torch.tensor(bi_class)

    return item
# generates typos with random frequency
def add_typos_RF(item):
    typo_count = np.clip(np.round(random.normal(centre, spread, batch_size)).astype(int), 0, 8 )
    typo_index = random.randint(50, size=(10*batch_size))
    typo_char = random.randint(low=97, high=123, size=(10*batch_size))
    base_one_hot = torch.zeros(1,  channels)
    base_one_hot_space =base_one_hot.detach().clone()
    base_one_hot_space[0][alphabet.index(' ')] = 1
    typo_i = 0

    for batch_i in range(batch_size):
        #typo type generation: 1=swap, 2=swap+insert, 3=swap+insert+delete
        typo_type = random.choice(3, typo_count[batch_i])
        typo_type.sort()
        typo_type = typo_type[::-1]#deleting first loses less information from end of samples
        for i in range(typo_count[batch_i]):
            typo_i +=1
            if typo_type[i] == 0:
                #swap char for another char
                if chr(typo_char[typo_i]) != item['label_text'][batch_i][typo_index[typo_i]]:
                    sample_text = list(item['sample_text'][batch_i])                
                    sample_text[typo_index[typo_i]] = chr(typo_char[typo_i])
                    item['sample_text'][batch_i] = ''.join(sample_text)
                    item['bi_class'][batch_i][typo_index[typo_i]] = 0
                    item['sample_one_hot'][batch_i][typo_index[typo_i]] = torch.zeros( channels)#training_data.channels
                    item['sample_one_hot'][batch_i][typo_index[typo_i]][alphabet.index(chr(typo_char[typo_i]))] = 1

            if typo_type[i] == 1:
                #insert extra char
                insert_one_hot = base_one_hot.detach().clone()
                insert_one_hot[0][alphabet.index(chr(typo_char[typo_i]))] = 1
                sample_text = list(item['sample_text'][batch_i])
                label_text = list(item['label_text'][batch_i])
                bi_class = list(item['bi_class'][batch_i])
            
                sample_text.insert(typo_index[typo_i], chr(typo_char[typo_i]))
                label_text.insert(typo_index[i], '#')
                bi_class.insert(typo_index[i], 0)
                item['sample_one_hot'][batch_i] = torch.cat((item['sample_one_hot'][batch_i][:typo_index[typo_i]], insert_one_hot, item['sample_one_hot'][batch_i][typo_index[typo_i]:-1]), 0)
                
            
                sample_text.pop(len(sample_text)-1)
                label_text.pop(len(label_text)-1)
                bi_class.pop(len(bi_class)-1)

                item['sample_text'][batch_i] = ''.join(sample_text)
                item['label_text'][i//5] = ''.join(label_text)
                item['bi_class'][i//4] = torch.tensor(bi_class)

    return item

#training loop function
def train():
    iteration = 0
    while (iteration < max_iterations):
        for item in training_data_loader:
            iteration += 1
            #generates typos during training, prevents overfitting
            if online: item = add_typos_RF(item)

            item['sample_one_hot'] = item['sample_one_hot'].transpose(1, 2)
            item['sample_one_hot'] = item['sample_one_hot'].to(device)
            item['bi_class'] = item['bi_class'].to(device)
            outputs = model(item['sample_one_hot'])
            outputs = torch.squeeze(outputs)
            loss = loss_fn(outputs, item['bi_class'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #print training info
            if (iteration)%100 == 0:
                lr = 'lr'
                print(f'Iteration {iteration}/{max_iterations}, loss = {loss.item():.4f}, lr = {optimizer.param_groups[0][lr]:.8f}') 
            
            #scale learning rate
            if iteration%learning_rate_scale_iter == 0:
                optimizer.param_groups[0]['lr'] *= learning_rate_scale
            
            #test on validation and test file
            if (iteration)%500 == 0:
                model.eval()
                with torch.no_grad():
                    print('Train data test:')
                    test(testing_train_data_loader)
                    print('\033[0;34mTest data test:\033[0;37m')
                    test(testing_test_data_loader)
                    model.train()

            #when max iteration is reached, saves the model and ends training     
            if iteration >= max_iterations:
                torch.save(model, save_model)
                break

#for testimg the model during training, prints some examples of inputs and outputs, and other statistics
def test(data_loader):
    TP, FP, TN, FN, TPR, PPV, F1, ACC_CM, TNR, BA = 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    confusion_matrix = torch.rand(2,2)
    
    for i,item in enumerate(data_loader):
        item['sample_one_hot'] = item['sample_one_hot'].transpose(1, 2)
        item['sample_one_hot'] = item['sample_one_hot'].to(device)
        item['bi_class'] = item['bi_class'].to(device)
        outputs = model(item['sample_one_hot'])
        outputs = torch.squeeze(outputs)
        item['bi_class'] = item['bi_class'][0]
        
        #turns the probability into binary classification
        outputs = [1 if out>0.6 else 0 for out in outputs]
        
        #counts confusion matrix stats
        for index, out in enumerate(outputs):
            if item['bi_class'][index] == 1 and out == 1: TP +=1
            if item['bi_class'][index] == 1 and out == 0: FN +=1
            if item['bi_class'][index] == 0 and out == 1: FP +=1
            if item['bi_class'][index] == 0 and out == 0: TN +=1
        
        #prints examples
        if i>data_loader.__len__()-6:
            print(f'ID: {item[id][0]}')
            print(item['label_text'][0])
            ansi_print.a_print(item['sample_text'][0], item['label_text'][0], white, yellow)
            ansi_print.a_print(outputs, item['bi_class'], green, red)
    
    confusion_matrix[0][0] = TP
    confusion_matrix[0][1] = FN
    confusion_matrix[1][0] = FP
    confusion_matrix[1][1] = TN
    print(confusion_matrix.numpy())
    
    #compute and print statistics
    TPR = TP/(TP+FN) #sensitivity, recall, hit rate, or true positive rate (TPR)
    TNR = TN/(TN+FP) #specificity, selectivity or true negative rate (TNR)
    PPV = TP/(TP+FP) #precision or positive predictive value (PPV)
    F1 = 2 * (PPV * TPR)/(PPV + TPR) #F1 score is the harmonic mean of precision and sensitivity:
    ACC_CM = (TP + TN)/(TP + TN + FP + FN) #accuracy
    BA = (TPR + TNR)/2 #balanced accuracy
    print(f'Accuracy: {ACC_CM*100:.2f}%')
    print(f'Balanced accuracy: {BA*100:.2f}%')
    print(f'Recall: {TPR:.4f}, TNR: {ansi_print.colors[green]}{TNR:.4f}{ansi_print.colors[white]}, Precision: {PPV:.4f}, F1: {F1:.4f}')


train()