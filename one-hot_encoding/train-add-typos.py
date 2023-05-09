# train_add_typos.py
# Author: Sebastián Chupáč
# only for testing different online typo generating functions, no real use
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from numpy import random

from dataset import MyDataset
from models import NeuralNetworkOneHot, NeuralNetwork, NeuralNetworkOneHotConv2
import ansi_print



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USING: {device}')


batch_size = 50
epochs = 5
learning_rate = 0.001

training_data = MyDataset('one-hot_encoding/data/examplerandom_length.txt')
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle = False)

alphabet = training_data.charlist_extra_ctc


def typos(item):#old very slow
    for i_batch, samples in enumerate(item['label']):
        bad_sample = []
        for i_sample, character in enumerate(item['ok_text'][i_batch]):
            if ((character>='A'<='Z') or (character>='a'<='z') ):
                if torch.rand(1).item()<=0.1:
                    character = chr(int((torch.rand(1).item()*26)) + 97)
                    if character != item['ok_text'][i_batch][i_sample]: 
                        item['label'][i_batch][i_sample] = 0
                        item['bad_sample_one_hot'][i_batch][i_sample] = torch.zeros(69)#training_data.channels
                        item['bad_sample_one_hot'][i_batch][i_sample][alphabet.index(character)] = 1
                        item['bad_sample'][i_batch][i_sample] = alphabet.index(character)
            bad_sample.append(character)
        item['bad_text'][i_batch] = ''.join(bad_sample)
    return item


def add_typos(item):
    for i_batch, _ in enumerate(item['label']):
        error_index = random.randint(50, size=(5))
        error_char = random.randint(low=97, high=123, size=(5))
        for i in range(5):
            #item['bad_text'][i_batch][error_index[i]] = chr(error_char[i])
            #'bad_text' is not updated with typos
            if chr(error_char[i]) != item['ok_text'][i_batch][error_index[i]]:
                item['label'][i_batch][error_index[i]] = 0
                item['bad_sample_one_hot'][i_batch][error_index[i]] = torch.zeros(69)#training_data.channels
                item['bad_sample_one_hot'][i_batch][error_index[i]][alphabet.index(chr(error_char[i]))] = 1
                #item['bad_sample'][i_batch][error_index[i]] = alphabet.index(chr(error_char[i]))
    return item


def new_add_typos(item):
    error_index = random.randint(50, size=(4*batch_size))
    error_char = random.randint(low=97, high=123, size=(4*batch_size))
    #print(item['bad_sample'][[0, 1, 2]])
    #print(item['bad_sample'][[0, 0, 0, 1, 2], [2, 1, 0, 0, 0]])
    #item['bad_sample'][[0, 0, 0, 1, 2], [2, 1, 0, 0, 0]] = torch.tensor([501, 502, 503, 504, 505], dtype=torch.float)
    #print(item['bad_sample'][[0, 1, 2]])
    #extra_char_one_hot = torch.zeros(1, 69)
    #extra_char_one_hot[0][alphabet.index('#')] = 1
    for i in range(4*batch_size):
        if (i%4)<3:
            #swap char for another char
            bad_text = list(item['bad_text'][i//4])
            #item['bad_sample'][i//4][error_index[i]] = alphabet.index(chr(error_char[i]))
            bad_text[error_index[i]] = chr(error_char[i])
            item['bad_text'][i//4] = ''.join(bad_text)
            if chr(error_char[i]) != item['ok_text'][i//4][error_index[i]]:
                #item['label'][i//4][error_index[i]] = 0
                item['bad_sample_one_hot'][i//4][error_index[i]] = torch.zeros(69)#training_data.channels
                item['bad_sample_one_hot'][i//4][error_index[i]][alphabet.index(chr(error_char[i]))] = 1
                #item['bad_sample'][i//4][error_index[i]] = training_data.charlist.index(chr(error_char[i]))
        else:
            #insert extra char
            base_one_hot = torch.zeros(1, 69)
            base_one_hot[0][alphabet.index(chr(error_char[i]))] = 1
            bad_text = list(item['bad_text'][i//4])
            ok_text = list(item['ok_text'][i//4])
            #label = list(item['label'][i//4])
            ok_sample = list(item['ok_sample'][i//4])
            #bad_sample = list(item['bad_sample'][i//4])
            
            bad_text.insert(error_index[i], chr(error_char[i]))
            ok_text.insert(error_index[i], '#')
            #label.insert(error_index[i], 0)
            ok_sample.insert(error_index[i], alphabet.index('#'))
            #bad_sample.insert(error_index[i], alphabet.index(chr(error_char[i])))
            item['bad_sample_one_hot'][i//4] = torch.cat((item['bad_sample_one_hot'][i//4][:error_index[i]], base_one_hot, item['bad_sample_one_hot'][i//4][error_index[i]:-1]), 0)
            #item['ok_sample_one_hot'][i//4] = torch.cat((item['ok_sample_one_hot'][i//4][:error_index[i]], extra_char_one_hot, item['ok_sample_one_hot'][i//4][error_index[i]:-1]), 0)
            
            bad_text.pop(len(bad_text)-1)
            ok_text.pop(len(ok_text)-1)
            #label.pop(len(label)-1)
            ok_sample.pop(len(ok_sample)-1)
            #bad_sample.pop(len(bad_sample)-1)

            item['bad_text'][i//4] = ''.join(bad_text)
            item['ok_text'][i//4] = ''.join(ok_text)
            #item['label'][i//4] = torch.tensor(label)
            item['ok_sample'][i//4] = torch.tensor(ok_sample)
            #item['bad_sample'][i//4] = torch.tensor(bad_sample)

    return item

def new_add_typos_delete(item):
    error_index = random.randint(50, size=(5*batch_size))
    error_char = random.randint(low=97, high=123, size=(5*batch_size))
    base_one_hot = torch.zeros(1, 90)
    base_one_hot_space =base_one_hot.detach().clone()
    base_one_hot_space[0][alphabet.index(' ')] = 1
    #print(item['bad_sample'][[0, 1, 2]])
    #print(item['bad_sample'][[0, 0, 0, 1, 2], [2, 1, 0, 0, 0]])
    #item['bad_sample'][[0, 0, 0, 1, 2], [2, 1, 0, 0, 0]] = torch.tensor([501, 502, 503, 504, 505], dtype=torch.float)
    #print(item['bad_sample'][[0, 1, 2]])
    #extra_char_one_hot = torch.zeros(1, 69)
    #extra_char_one_hot[0][alphabet.index('#')] = 1
    for i in range(5*batch_size):
        if (i%5)<3:
            #swap char for another char
            bad_text = list(item['bad_text'][i//5])
            #item['bad_sample'][i//4][error_index[i]] = alphabet.index(chr(error_char[i]))
            bad_text[error_index[i]] = chr(error_char[i])
            item['bad_text'][i//5] = ''.join(bad_text)
            if chr(error_char[i]) != item['ok_text'][i//5][error_index[i]]:
                #item['label'][i//4][error_index[i]] = 0
                item['bad_sample_one_hot'][i//5][error_index[i]] = torch.zeros(90)#training_data.channels
                item['bad_sample_one_hot'][i//5][error_index[i]][alphabet.index(chr(error_char[i]))] = 1
                #item['bad_sample'][i//4][error_index[i]] = training_data.charlist.index(chr(error_char[i]))
        if (i%5)==4:
            #insert extra char
            insert_one_hot = base_one_hot.detach().clone()
            insert_one_hot[0][alphabet.index(chr(error_char[i]))] = 1
            bad_text = list(item['bad_text'][i//5])
            #ok_text = list(item['ok_text'][i//5])
            #label = list(item['label'][i//4])
            #ok_sample = list(item['ok_sample'][i//5])
            #bad_sample = list(item['bad_sample'][i//4])
            
            bad_text.insert(error_index[i], chr(error_char[i]))
            #ok_text.insert(error_index[i], '#')
            #label.insert(error_index[i], 0)
            #ok_sample.insert(error_index[i], alphabet.index('#'))
            #bad_sample.insert(error_index[i], alphabet.index(chr(error_char[i])))
            item['bad_sample_one_hot'][i//5] = torch.cat((item['bad_sample_one_hot'][i//5][:error_index[i]], insert_one_hot, item['bad_sample_one_hot'][i//5][error_index[i]:-1]), 0)
            #item['ok_sample_one_hot'][i//4] = torch.cat((item['ok_sample_one_hot'][i//4][:error_index[i]], extra_char_one_hot, item['ok_sample_one_hot'][i//4][error_index[i]:-1]), 0)
            
            bad_text.pop(len(bad_text)-1)
            #ok_text.pop(len(ok_text)-1)
            #label.pop(len(label)-1)
            #ok_sample.pop(len(ok_sample)-1)
            #bad_sample.pop(len(bad_sample)-1)

            item['bad_text'][i//5] = ''.join(bad_text)
            #item['ok_text'][i//5] = ''.join(ok_text)
            #item['label'][i//4] = torch.tensor(label)
            #item['ok_sample'][i//5] = torch.tensor(ok_sample)
            #item['bad_sample'][i//4] = torch.tensor(bad_sample)

        if (i%5)==3:
            #delete char
            bad_text = list(item['bad_text'][i//5])
            del bad_text[error_index[i]]
            bad_text.insert(len(bad_text), ' ')
            item['bad_text'][i//5] = ''.join(bad_text)
            item['bad_sample_one_hot'][i//5] = torch.cat((item['bad_sample_one_hot'][i//5][:error_index[i]], item['bad_sample_one_hot'][i//5][error_index[i]+1:], base_one_hot_space), 0)
    return item

def new_add_typos_RF(item):
    typo_count = np.clip(np.round(random.normal(5, 2, batch_size)).astype(int), 0, 8 )#set typo frequency
    typo_index = random.randint(50, size=(10*batch_size))
    typo_char = random.randint(low=97, high=123, size=(10*batch_size))
    base_one_hot = torch.zeros(1, 90)
    base_one_hot_space =base_one_hot.detach().clone()
    base_one_hot_space[0][alphabet.index(' ')] = 1
    typo_i = -1
    #print(item['bad_sample'][[0, 1, 2]])
    #print(item['bad_sample'][[0, 0, 0, 1, 2], [2, 1, 0, 0, 0]])
    #item['bad_sample'][[0, 0, 0, 1, 2], [2, 1, 0, 0, 0]] = torch.tensor([501, 502, 503, 504, 505], dtype=torch.float)
    #print(item['bad_sample'][[0, 1, 2]])
    #extra_char_one_hot = torch.zeros(1, 69)
    #extra_char_one_hot[0][alphabet.index('#')] = 1
    for batch_i in range(batch_size):
        #typo type generation: 1=swap, 2=swap+insert, 3=swap+insert+delete
        typo_type = random.choice(3, typo_count[batch_i])
        typo_type.sort()
        typo_type = typo_type[::-1]#deleting first loses less information from end of samples
        for i in range(typo_count[batch_i]):
            typo_i +=1
            if typo_index[typo_i]>=len(item['bad_text'][batch_i]): continue
            if typo_type[i] == 0:
                #swap char for another char
                bad_text = list(item['bad_text'][batch_i])
                #item['bad_sample'][i//4][error_index[i]] = alphabet.index(chr(error_char[i]))
                bad_text[typo_index[typo_i]] = chr(typo_char[typo_i])
                item['bad_text'][batch_i] = ''.join(bad_text)

                item['bad_sample_one_hot'][batch_i][typo_index[typo_i]] = torch.zeros(90)#training_data.channels
                item['bad_sample_one_hot'][batch_i][typo_index[typo_i]][alphabet.index(chr(typo_char[typo_i]))] = 1
                    #item['bad_sample'][i//4][error_index[i]] = training_data.charlist.index(chr(error_char[i]))

            if typo_type[i] == 1:
                #insert extra char
                insert_one_hot = base_one_hot.detach().clone()
                insert_one_hot[0][alphabet.index(chr(typo_char[typo_i]))] = 1
                bad_text = list(item['bad_text'][batch_i])
                #ok_text = list(item['ok_text'][batch_i])
                #label = list(item['label'][i//4])
                #ok_sample = list(item['ok_sample'][batch_i])
                #bad_sample = list(item['bad_sample'][i//4])
            
                bad_text.insert(typo_index[typo_i], chr(typo_char[typo_i]))
                #ok_text.insert(error_index[i], '#')
                #label.insert(error_index[i], 0)
                #ok_sample.insert(error_index[i], alphabet.index('#'))
                #bad_sample.insert(error_index[i], alphabet.index(chr(error_char[i])))
                item['bad_sample_one_hot'][batch_i] = torch.cat((item['bad_sample_one_hot'][batch_i][:typo_index[typo_i]], insert_one_hot, item['bad_sample_one_hot'][batch_i][typo_index[typo_i]:-1]), 0)
                #item['ok_sample_one_hot'][i//4] = torch.cat((item['ok_sample_one_hot'][i//4][:error_index[i]], extra_char_one_hot, item['ok_sample_one_hot'][i//4][error_index[i]:-1]), 0)
            
                bad_text.pop(len(bad_text)-1)
                #ok_text.pop(len(ok_text)-1)
                #label.pop(len(label)-1)
                #ok_sample.pop(len(ok_sample)-1)
                #bad_sample.pop(len(bad_sample)-1)

                item['bad_text'][batch_i] = ''.join(bad_text)
                #item['ok_text'][i//5] = ''.join(ok_text)
                #item['label'][i//4] = torch.tensor(label)
                #item['ok_sample'][i//5] = torch.tensor(ok_sample)
                #item['bad_sample'][i//4] = torch.tensor(bad_sample)

            if typo_type[i] == 2:
                #delete char
                bad_text = list(item['bad_text'][batch_i])
                del bad_text[typo_index[typo_i]]
                bad_text.insert(len(bad_text), ' ')
                item['bad_text'][batch_i] = ''.join(bad_text)
                item['bad_sample_one_hot'][batch_i] = torch.cat((item['bad_sample_one_hot'][batch_i][:typo_index[typo_i]], item['bad_sample_one_hot'][batch_i][typo_index[typo_i]+1:], base_one_hot_space), 0)
    return item

def train():
    #n_total_steps = len(training_data_loader)
    for epoch in range(epochs):
        print(epoch)
        for i, item in enumerate(training_data_loader):
            #item = new_add_typos_delete(item)
            item = new_add_typos_RF(item)
            print(i)
            #item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
            #print(item['id'][49])
            #print(item['ok_text'][49], len(item['ok_text'][49]))
            #print(item['bad_text'][49], len(item['bad_text'][49]))
            #print(item['label'][49])
            #print(item['ok_sample'][49])
            #print(item['bad_sample'][49])
            

train()
            
