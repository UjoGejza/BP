# training_correction_CTC.py
# Author: Sebastián Chupáč
# This is a training script for typo correction utilizing CTC to get rid off position dependency. 
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import Levenshtein
import numpy as np
from numpy import random

#dataset for files with padding
from dataset_pad import MyDataset
#import class of which model you want to train
from models import ConvLSTMCorrectionCTC
import ansi_print

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', type=int, default=50)#batch size
    parser.add_argument('-mi', type=int, default=600_000)#maximum iterations
    parser.add_argument('-lr', type=float, default=0.001)#learning rate
    parser.add_argument('-lr_scale', type=float, default=0.9)#learning rate scale
    parser.add_argument('-lr_scaleiter', type=int, default=10_000)#scale learning rate every lr_scaleiter iterations
    parser.add_argument('-online', type=int, default=1)#generate typos during training
    parser.add_argument('-centre', type=int, default=6)#centre value of gaussan distribution of number of typos per sample
    parser.add_argument('-spread', type=int, default=2)#spread value of gaussan distribution of number of typos per sample
    parser.add_argument('-load_model', type=str, default='_')#file from to load parameters to continue training
    parser.add_argument('-save_model', type=str, default='ConvLSTMCorrectionCTC.pt')#file to save trained model to
    parser.add_argument('-train_file', type=str, default='datawiki_RLOAWP2_2M.txt')#input training file, if online = 0 make sure training file contains typos, and vice versa
    parser.add_argument('-test_train_file', type=str, default='data_pad/wiki_RLOAWP2_train_1k_typosRF3_CTC.txt')#input validation file
    parser.add_argument('-test_test_file', type=str, default='data_pad/wiki_RLOAWP2_test_1k_typosRF3_CTC.txt')#input testing file
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
alphabet = training_data.charlist_extra_ctc

channels = len(alphabet)
#create the model class
model = ConvLSTMCorrectionCTC()
print('model class: UNetCorrectionCTC')
#load parameters if file path provided
if load_model !='_': model = torch.load(load_model)
model.to(device)
model.train()
#set the loss function here, correction with CTC uses CTCLoss
loss_fn = nn.CTCLoss(zero_infinity=True)

sample_length = 73
#set lengths of network outputs(CTCLOss inputs) and ground truth(targets)
#set this based on the used architecture
input_lengths = torch.full(size=(batch_size, ), fill_value=sample_length+10, dtype=torch.long)
target_lengths = torch.full(size=(batch_size, ), fill_value=60, dtype=torch.long)

print(f'MODEL ARCHITECTURE: ')
for name, param in model.state_dict().items():
    print(name, param.size())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

green = 'green'
yellow = 'yellow'
red = 'red'
white = 'white'
id = 'id'
blank = 'ѧ'
pad = 'Є'
unknown = 'є'

#generates typos with random frequency, "constant" typos were not used with CTC
def add_typos_RF(item):
    typo_count = np.clip(np.round(random.normal(centre, spread, batch_size)).astype(int), 0, 8 )
    typo_index = random.randint(low=3, high=63, size=(10*batch_size))
    typo_char = random.randint(low=97, high=123, size=(10*batch_size))
    base_one_hot = torch.zeros(1, 90)
    base_one_hot_pad =base_one_hot.detach().clone()
    base_one_hot_pad[0][alphabet.index(pad)] = 1
    typo_i = 0

    for batch_i in range(batch_size):
        #typo type generation: 1=swap, 2=swap+insert, 3=swap+insert+delete
        typo_type = random.choice(3, typo_count[batch_i])
        
        sample_text = list(item['sample_text'][batch_i])
        for i in range(typo_count[batch_i]):
            typo_i +=1
            if sample_text[typo_index[typo_i]] == pad: typo_index[typo_i] = random.randint(low=3, high=42)
            if typo_type[i] == 0:
                #swap char for another char
                sample_text[typo_index[typo_i]] = chr(typo_char[typo_i])

                item['sample_one_hot'][batch_i][typo_index[typo_i]] = torch.zeros(channels)#training_data.channels
                item['sample_one_hot'][batch_i][typo_index[typo_i]][alphabet.index(chr(typo_char[typo_i]))] = 1

            if typo_type[i] == 1:
                #insert extra char
                insert_one_hot = base_one_hot.detach().clone()
                insert_one_hot[0][alphabet.index(chr(typo_char[typo_i]))] = 1
    
            
                sample_text.insert(typo_index[typo_i], chr(typo_char[typo_i]))
    
                item['sample_one_hot'][batch_i] = torch.cat((item['sample_one_hot'][batch_i][:typo_index[typo_i]], insert_one_hot, item['sample_one_hot'][batch_i][typo_index[typo_i]:-1]), 0)
    
            
                sample_text.pop(len(sample_text)-1)

            if typo_type[i] == 2:
                #delete char
   
                del sample_text[typo_index[typo_i]]
                sample_text.insert(len(sample_text), pad)
                item['sample_one_hot'][batch_i] = torch.cat((item['sample_one_hot'][batch_i][:typo_index[typo_i]], item['sample_one_hot'][batch_i][typo_index[typo_i]+1:], base_one_hot_pad), 0)
        item['sample_text'][batch_i] = ''.join(sample_text)
    return item

#training loop function
def train():
    iteration = 0
    while (iteration < max_iterations):
        for item in training_data_loader:
            iteration += 1
            #generates typos during training, prevents overfitting
            if online: item = add_typos_RF(item)
            target_lengths = item['label_length']
            item['sample_one_hot'] = item['sample_one_hot'].transpose(1, 2)
            item['sample_one_hot'] = item['sample_one_hot'].to(device)
            outputs = model(item['sample_one_hot'])
            outputs = outputs.permute(2, 0, 1)
            
            #CTCLoss requires probability matrix -> logsoftmax
            outputs = torch.nn.functional.log_softmax(outputs, 2)
            loss = loss_fn(outputs, item['label'], input_lengths, target_lengths)

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
    sum_distance = 0
    sum_ratio = 0

    for i, item in enumerate(data_loader):
        item['sample_one_hot'] = item['sample_one_hot'].transpose(1, 2)
        item['sample_one_hot'] = item['sample_one_hot'].to(device)
        outputs = model(item['sample_one_hot'])
        outputs = outputs[0]
        #from outpu matrix takes class indices with the highest value -> indeces in charset (best path decoding)
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        output_list = list(pred)

        #remove all chains of the same character longer than 1 (aa -> a)
        trimmed_output_list_str = []
        for out in output_list:
            if len(trimmed_output_list_str) == 0 or alphabet.index(trimmed_output_list_str[-1]) != out:
                trimmed_output_list_str.append(alphabet[out])

        #remove "blank" 
        trimmed_output_list_txt_no_blank = [x for x in trimmed_output_list_str if x!= blank]
        
        #converts decoded output to string
        final_str = ''.join(trimmed_output_list_txt_no_blank)

        #count stats
        edit_distance = Levenshtein.distance(final_str, item['label_text'][0][:item['label_length']])
        indel_ratio = Levenshtein.ratio(final_str, item['label_text'][0][:item['label_length']])
        sum_distance += edit_distance
        sum_ratio += indel_ratio

        #converts raw output to string, subs blank for printable char    
        if i>data_loader.__len__()-6:
            raw_output = []
            for e in output_list:
                c = alphabet[e]
                if c == blank or c == pad: c = '='
                raw_output.append(c)
            raw_output_str = ''.join(raw_output)
            
            #prints examples
            try:
                print(f'ID: {item[id][0]}')
                print('GT :',item['label_text'][0][:item['label_length']])
                print('BAD:',item['sample_text'][0][3:item['sample_text'][0].find(pad, 35)])
                print('OUT:',final_str)
                ansi_print.a_print(final_str, item['label_text'][0], green, red )
                #ansi_print.a_print(item['sample_text'][0], item['label_text'][0],white, yellow)
                print('RAW:',raw_output_str)
                #print(f'edit distance: {edit_distance}')
            except:
                print('error printing example - prob encoding')
    #prints stats
    print(f'Average edit distance: {ansi_print.colors[green]}{sum_distance/2000:.2f}{ansi_print.colors[white]}')
    print(f'Average indel similarity: {ansi_print.colors[green]}{sum_ratio/2000:.4f}{ansi_print.colors[white]}') #1 - normalized_distance

train()