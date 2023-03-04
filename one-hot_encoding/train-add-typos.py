
import torch
from torch import nn
from torch.utils.data import DataLoader

from numpy import random

from dataset import MyDataset
from models import NeuralNetworkOneHot, NeuralNetwork, NeuralNetworkOneHotConv2
import ansi_print



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USING: {device}')


batch_size = 50
epochs = 5
learning_rate = 0.001

training_data = MyDataset('one-hot_encoding/data/wiki-20k.txt')
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle = True)

alphabet = training_data.charlist


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
    print(item['bad_sample'][[0, 1, 2]])
    print(item['bad_sample'][[0, 0, 0, 1, 2], [2, 1, 0, 0, 0]])
    item['bad_sample'][[0, 0, 0, 1, 2], [2, 1, 0, 0, 0]] = torch.tensor([501, 502, 503, 504, 505], dtype=torch.float)
    print(item['bad_sample'][[0, 1, 2]])
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

def train():
    #n_total_steps = len(training_data_loader)
    for epoch in range(epochs):
        print(epoch)
        for i, item in enumerate(training_data_loader):
            #item = new_add_typos(item)
            print(i)
            #item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
            #print(item['id'][49])
            #print(item['ok_text'][49])
            #print(item['bad_text'][49])
            #print(item['label'][49])
            #print(item['ok_sample'][49])
            #print(item['bad_sample'][49])
            

train()
            
