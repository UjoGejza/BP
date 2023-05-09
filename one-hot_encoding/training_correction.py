# training_correction.py
# Author: Sebastián Chupáč
# This is a training script for typo correction position dependent. 
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
from numpy import random

from dataset import MyDataset
#import class of which model you want to train
from models import ConvLSTMCorrectionBigger
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
    parser.add_argument('-save_model', type=str, default='ConvLSTMCorrection.pt')#file to save trained model to 
    parser.add_argument('-train_file', type=str, default='one-hot_encoding/data/wiki-20k.txt')#input training file, if online = 0 make sure training file contains typos, and vice versa
    parser.add_argument('-test_train_file', type=str, default='one-hot_encoding/data/wiki-20k_typos_train_1k.txt')#input validation file
    parser.add_argument('-test_test_file', type=str, default='one-hot_encoding/data/wiki_test_test_1k_typos2.txt')#input testing file
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
model = ConvLSTMCorrectionBigger()
print('model class: ConvLSTMCorrectionBigger')
#load parameters if file path provided
if load_model !='_': model = torch.load(load_model)
model.to(device)
model.train()
#set the loss function here, correction uses CrossEntropyLos
loss_fn = nn.CrossEntropyLoss()
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
    for i in range(4*50):
        if (i%4)<3:
            #swap char for another char
            sample_text = list(item['sample_text'][i//4])
            
            sample_text[error_index[i]] = chr(error_char[i])
            item['sample_text'][i//4] = ''.join(sample_text)
            if chr(error_char[i]) != item['label_text'][i//4][error_index[i]]:
                
                item['sample_one_hot'][i//4][error_index[i]] = torch.zeros(channels)#training_data.channels
                item['sample_one_hot'][i//4][error_index[i]][alphabet.index(chr(error_char[i]))] = 1
                
        else:
            #insert extra char
            base_one_hot = torch.zeros(1, channels)
            base_one_hot[0][alphabet.index(chr(error_char[i]))] = 1
            sample_text = list(item['sample_text'][i//4])
            label_text = list(item['label_text'][i//4])
            
            label = list(item['label'][i//4])
            
            sample_text.insert(error_index[i], chr(error_char[i]))
            label_text.insert(error_index[i], '#')
            label.insert(error_index[i], alphabet.index('#'))
           
            item['sample_one_hot'][i//4] = torch.cat((item['sample_one_hot'][i//4][:error_index[i]], base_one_hot, item['sample_one_hot'][i//4][error_index[i]:-1]), 0)
            
            sample_text.pop(len(sample_text)-1)
            label_text.pop(len(label_text)-1)
            label.pop(len(label)-1)

            item['sample_text'][i//4] = ''.join(sample_text)
            item['label_text'][i//4] = ''.join(label_text)
            item['label'][i//4] = torch.tensor(label)

    return item

# generates typos with random frequency
def add_typos_RF(item):
    typo_count = np.clip(np.round(random.normal(centre, spread, batch_size)).astype(int), 0, 8 )#set typo frequency
    typo_index = random.randint(50, size=(10*batch_size))
    typo_char = random.randint(low=97, high=123, size=(10*batch_size))
    base_one_hot = torch.zeros(1, channels)
    base_one_hot_space =base_one_hot.detach().clone()
    base_one_hot_space[0][alphabet.index(' ')] = 1
    typo_i = 0

    for batch_i in range(batch_size):
        #typo type generation: 1=swap, 2=swap+insert, 3=swap+insert+delete
        typo_type = random.choice(2, typo_count[batch_i])
        typo_type.sort()
        typo_type = typo_type[::-1]#deleting first loses less information from end of samples
        for i in range(typo_count[batch_i]):
            typo_i +=1
            if typo_type[i] == 0:
                #swap char for another char
                sample_text = list(item['sample_text'][batch_i])
                sample_text[typo_index[typo_i]] = chr(typo_char[typo_i])
                item['sample_text'][batch_i] = ''.join(sample_text)

                item['sample_one_hot'][batch_i][typo_index[typo_i]] = torch.zeros(channels)#training_data.channels
                item['sample_one_hot'][batch_i][typo_index[typo_i]][alphabet.index(chr(typo_char[typo_i]))] = 1

            if typo_type[i] == 1:
                #insert extra char
                insert_one_hot = base_one_hot.detach().clone()
                insert_one_hot[0][alphabet.index(chr(typo_char[typo_i]))] = 1
                sample_text = list(item['sample_text'][batch_i])
                label = list(item['label'][batch_i])
                
                sample_text.insert(typo_index[typo_i], chr(typo_char[typo_i]))
                
                label.insert(typo_index[typo_i], alphabet.index('#'))
                item['sample_one_hot'][batch_i] = torch.cat((item['sample_one_hot'][batch_i][:typo_index[typo_i]], insert_one_hot, item['sample_one_hot'][batch_i][typo_index[typo_i]:-1]), 0)
                sample_text.pop(len(sample_text)-1)               
                label.pop(len(label)-1) 

                item['sample_text'][batch_i] = ''.join(sample_text)
                item['label'][batch_i] = torch.tensor(label)

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
            item['label'] = item['label'].to(device)
            outputs = model(item['sample_one_hot'])
            item['label'] = item['label'].long()
            loss = loss_fn(outputs, item['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print training info
            if iteration%100 == 0:
                lr = 'lr'
                print(f'Iteration {iteration}/{max_iterations}, loss = {loss.item():.4f}, lr = {optimizer.param_groups[0][lr]:.8f}')             
            
            #scale learning rate
            if iteration%learning_rate_scale_iter == 0:
                optimizer.param_groups[0]['lr'] *= learning_rate_scale
            
            #test on validation and test file
            if iteration%200 == 0:
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
    correct = 0
    all= 0
    corrected_typos = 0
    all_typos = 0
    created_typos = 0


    for i, item in enumerate(data_loader):
        item['sample_one_hot'] = item['sample_one_hot'].transpose(1, 2)
        item['sample_one_hot'] = item['sample_one_hot'].to(device)
        item['label'] = item['label'].to(device)
        outputs = model(item['sample_one_hot'])
        outputs = outputs[0]
        
        #from outpu matrix takes class indices with the highest value -> indeces in charset
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        
        item['bi_class'] = item['bi_class'][0]
        item['label'] = item['label'][0]
        output_text_list = list(pred)

        #counts stats
        for index, p in enumerate(pred):
            if item['label'][index] == p: correct+=1
            if item['bi_class'][index] == 0:
                all_typos += 1
                if item['label'][index] == p: corrected_typos+=1
            else:
                if item['label'][index] != p: created_typos+=1
            if i>data_loader.__len__()-6:
                output_text_list[index] = alphabet[p]
            all +=1

        #prints examples
        if i>data_loader.__len__()-6:
            try:
                output_text = ''.join(output_text_list)
                print(f'ID: {item[id][0]}')
                print(item['label_text'][0])
                ansi_print.a_print(item['sample_text'][0], item['label_text'][0],white, yellow)
                ansi_print.a_print(output_text, item['label_text'][0],green, red)
            except:
                print('error printing example - prob encoding')

    #compute and print statistics
    acc = correct/all
    acc_corrected = corrected_typos/all_typos
    acc_corrected_created = corrected_typos/(all_typos+created_typos)
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Corrected typos: {corrected_typos} / {all_typos}, {ansi_print.colors[green]}{acc_corrected*100:.2f}%{ansi_print.colors[white]}')
    print(f'Typos created: {created_typos}, final acc: {ansi_print.colors[green]}{acc_corrected_created*100:.2f}%{ansi_print.colors[white]}')

train()