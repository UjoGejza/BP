
import torch
from torch import nn
from torch.utils.data import DataLoader

from numpy import random

from dataset import MyDataset
from models import ConvLSTMCorrection
import ansi_print

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USING: {device}')

batch_size = 50
max_iterations = 40_000
learning_rate = 0.001

training_data = MyDataset('one-hot_encoding/data/wiki-1k-train-insert-swap.txt')
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
testing_test_data = MyDataset('one-hot_encoding/data/wiki-1k-test-insert-swap.txt')
testing_test_data_loader = DataLoader(testing_test_data, shuffle=True)
testing_train_data = MyDataset('one-hot_encoding/data/wiki-1k-train-insert-swap.txt')
testing_train_data_loader = DataLoader(testing_train_data, shuffle=True)

alphabet = training_data.charlist

model = ConvLSTMCorrection()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
print(f'MODEL ARCHITECTURE: ')
for name, param in model.state_dict().items():
    print(name, param.size())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


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
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
            item['ok_sample'] = item['ok_sample'].to(device)
            outputs = model(item['bad_sample_one_hot'])
            item['ok_sample'] = item['ok_sample'].long()
            loss = loss_fn(outputs, item['ok_sample'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration%500 == 0:
                lr = 'lr'
                print(f'Iteration {iteration}/{max_iterations}, loss = {loss.item():.4f}, lr = {optimizer.param_groups[0][lr]:.8f}')             
            if iteration%500 == 0:
                optimizer.param_groups[0]['lr'] *= 0.85
                with torch.no_grad():
                    print('Train data test:')
                    test(testing_train_data_loader)
                    print('\033[0;34mTest data test:\033[0;37m')
                    test(testing_test_data_loader)
            if iteration >= max_iterations:
                torch.save(model, 'ConvLSTMCorrection.pt')
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
            ansi_print.a_print(item['bad_text'][0], item['ok_text'][0], 'yellow')
            ansi_print.a_print(output_text, item['ok_text'][0], 'red')

    acc = correct/all
    acc_corrected = corrected_typos/all_typos
    acc_corrected_created = corrected_typos/(all_typos+created_typos)
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Corrected typos: {corrected_typos} / {all_typos}, {ansi_print.colors.GREEN}{acc_corrected*100:.2f}%{ansi_print.colors.RESET}')
    print(f'Typos created: {created_typos}, final acc: {ansi_print.colors.GREEN}{acc_corrected_created*100:.2f}%{ansi_print.colors.RESET}')

train()