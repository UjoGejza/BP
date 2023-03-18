
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse

from numpy import random

from dataset import MyDataset
from models import ConvLSTMCorrectionBigger
import ansi_print

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', type=int, default=50)
    parser.add_argument('-mi', type=int, default=200_000)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-lr_scale', type=float, default=0.9)
    parser.add_argument('-lr_scaleiter', type=int, default=10_000)
    parser.add_argument('-online', type=int, default=1)
    parser.add_argument('-load_model', type=str, default='_')
    parser.add_argument('-save_model', type=str, default='ConvLSTMCorrection.pt')
    parser.add_argument('-train_file', type=str, default='one-hot_encoding/data/wiki-20k.txt')
    parser.add_argument('-test_train_file', type=str, default='one-hot_encoding/data/wiki-20k_typos_train_1k.txt')
    parser.add_argument('-test_test_file', type=str, default='one-hot_encoding/data/wiki_test_test_1k_typos2.txt')
    return parser.parse_args()

args = parseargs()

print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USING: {device}')

batch_size = args.bs
max_iterations = args.mi
learning_rate = args.lr
learning_rate_scale = args.lr_scale
learning_rate_scale_iter = args.lr_scaleiter
online = args.online
load_model = args.load_model
save_model = args.save_model
train_file = args.train_file
test_test_file = args.test_test_file
test_train_file = args.test_train_file

training_data = MyDataset(train_file)
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
testing_test_data = MyDataset(test_test_file)
testing_test_data_loader = DataLoader(testing_test_data, shuffle=True)
testing_train_data = MyDataset(test_train_file)
testing_train_data_loader = DataLoader(testing_train_data, shuffle=True)


alphabet = training_data.charlist_base

model = ConvLSTMCorrectionBigger()
print('model class: ConvLSTMCorrectionBigger')
if load_model !='_': model = torch.load(load_model)
model.to(device)
model.train()
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


def add_typos(item):
    error_index = random.randint(50, size=(4*50))
    error_char = random.randint(low=97, high=123, size=(4*50))
    #extra_char_one_hot = torch.zeros(1, 69)
    #extra_char_one_hot[0][alphabet.index('#')] = 1
    for i in range(4*50):
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
                #item['bad_sample'][i_batch][error_index[i]] = training_data.charlist.index(chr(error_char[i]))
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
    iteration = 0
    while (iteration < max_iterations):
        for item in training_data_loader:
            iteration += 1
            if online: item = add_typos(item)
            #print(iteration)
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
            item['ok_sample'] = item['ok_sample'].to(device)
            outputs = model(item['bad_sample_one_hot'])
            item['ok_sample'] = item['ok_sample'].long()
            loss = loss_fn(outputs, item['ok_sample'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration%400 == 0:
                lr = 'lr'
                print(f'Iteration {iteration}/{max_iterations}, loss = {loss.item():.4f}, lr = {optimizer.param_groups[0][lr]:.8f}')             
            if iteration%learning_rate_scale_iter == 0:
                optimizer.param_groups[0]['lr'] *= learning_rate_scale
            if iteration%500 == 0:
                model.eval()
                with torch.no_grad():
                    print('Train data test:')
                    test(testing_train_data_loader)
                    print('\033[0;34mTest data test:\033[0;37m')
                    test(testing_test_data_loader)
                    model.train()
            if iteration >= max_iterations:
                torch.save(model, save_model)
                break

def test(data_loader):
    correct = 0
    all= 0
    corrected_typos = 0
    all_typos = 0
    created_typos = 0


    for i, item in enumerate(data_loader):
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        item['ok_sample'] = item['ok_sample'].to(device)
        outputs = model(item['bad_sample_one_hot'])
        outputs = outputs[0]
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        item['label'] = item['label'][0]
        item['ok_sample'] = item['ok_sample'][0]
        output_text_list = list(pred)

        for index, p in enumerate(pred):
            if item['ok_sample'][index] == p: correct+=1
            if item['label'][index] == 0:
                all_typos += 1
                if item['ok_sample'][index] == p: corrected_typos+=1
            else:
                if item['ok_sample'][index] != p: created_typos+=1
            if i>data_loader.__len__()-6:
                output_text_list[index] = alphabet[p]
            all +=1

        if i>data_loader.__len__()-6:
            output_text = ''.join(output_text_list)
            print(f'ID: {item[id][0]}')
            print(item['ok_text'][0])
            ansi_print.a_print(item['bad_text'][0], item['ok_text'][0],white, yellow)
            ansi_print.a_print(output_text, item['ok_text'][0],green, red)

    acc = correct/all
    acc_corrected = corrected_typos/all_typos
    acc_corrected_created = corrected_typos/(all_typos+created_typos)
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Corrected typos: {corrected_typos} / {all_typos}, {ansi_print.colors[green]}{acc_corrected*100:.2f}%{ansi_print.colors[white]}')
    print(f'Typos created: {created_typos}, final acc: {ansi_print.colors[green]}{acc_corrected_created*100:.2f}%{ansi_print.colors[white]}')

train()