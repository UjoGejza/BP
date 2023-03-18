
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import Levenshtein

from numpy import random

from dataset import MyDataset
from models import ConvLSTMCorrectionCTCBigger
import ansi_print

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', type=int, default=50)
    parser.add_argument('-mi', type=int, default=600_000)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-lr_scale', type=float, default=0.9)
    parser.add_argument('-lr_scaleiter', type=int, default=10_000)
    parser.add_argument('-online', type=int, default=1)
    parser.add_argument('-load_model', type=str, default='_')
    parser.add_argument('-save_model', type=str, default='ConvLSTMCorrectionCTC.pt')
    parser.add_argument('-train_file', type=str, default='one-hot_encoding/data/wiki-20k.txt')
    parser.add_argument('-test_train_file', type=str, default='one-hot_encoding/data/scifi_train_test_1k_typos_CTC.txt')
    parser.add_argument('-test_test_file', type=str, default='one-hot_encoding/data/scifi_test_test_1k_typos_CTC.txt')
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


alphabet = training_data.charlist_extra_ctc

channels = len(alphabet)

model = ConvLSTMCorrectionCTCBigger()
print('model class: ConvLSTMCorrectionCTCBigger')
if load_model !='_': model = torch.load(load_model)
model.to(device)
model.train()
loss_fn = nn.CTCLoss(zero_infinity=True)

sample_length = 50

lengths = torch.full(size=(batch_size, ), fill_value=60, dtype=torch.long)
target_lengths = torch.full(size=(batch_size, ), fill_value=50, dtype=torch.long)

print(f'MODEL ARCHITECTURE: ')
for name, param in model.state_dict().items():
    print(name, param.size())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

green = 'green'
yellow = 'yellow'
red = 'red'
white = 'white'
id = 'id'


def add_typos(item):#old
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

def new_add_typos_delete(item):#for ctc
    error_index = random.randint(sample_length, size=(5*batch_size))
    error_char = random.randint(low=97, high=123, size=(5*batch_size))
    base_one_hot = torch.zeros(1, channels)
    base_one_hot_space =base_one_hot.detach().clone()
    base_one_hot_space[0][alphabet.index(' ')] = 1
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
                item['bad_sample_one_hot'][i//5][error_index[i]] = torch.zeros(channels)#training_data.channels
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

def train():
    iteration = 0
    while (iteration < max_iterations):
        for item in training_data_loader:
            iteration += 1
            if online: item = new_add_typos_delete(item)
            #print(iteration)
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
            outputs = model(item['bad_sample_one_hot'])
            outputs = outputs.permute(2, 0, 1)
            outputs = torch.nn.functional.log_softmax(outputs, 2)
            #item['ok_sample'] = item['ok_sample'].long()
            loss = loss_fn(outputs, item['ok_sample'], lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration%100 == 0:
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

#https://stackoverflow.com/questions/2460177/edit-distance-in-python #replaced with Levenshtein lib


def test(data_loader):
    sum_distance = 0
    sum_ratio = 0

    for i, item in enumerate(data_loader):
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        item['ok_sample'] = item['ok_sample'].to(device)
        outputs = model(item['bad_sample_one_hot'])
        outputs = outputs[0]
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        item['label'] = item['label'][0]
        item['ok_sample'] = item['ok_sample'][0]
        output_list = list(pred)

        #remove all chains of the same character longer than 1 (aa -> a)
        trimmed_output_list_str = []
        for out in output_list:
            if len(trimmed_output_list_str) == 0 or alphabet.index(trimmed_output_list_str[-1]) != out:
                trimmed_output_list_str.append(alphabet[out])

        #remove "blank" (~) 
        trimmed_output_list_txt_no_blank = [x for x in trimmed_output_list_str if x!= '~']
        final_str = ''.join(trimmed_output_list_txt_no_blank)

        edit_distance = Levenshtein.distance(final_str, item['ok_text'][0])
        indel_ratio = Levenshtein.ratio(final_str, item['ok_text'][0])
        sum_distance += edit_distance
        sum_ratio += indel_ratio

        if i>data_loader.__len__()-6:
            raw_output = []
            for e in output_list:
                raw_output.append(alphabet[e])
            raw_output_str = ''.join(raw_output)
            
            
            try:
              print(f'ID: {item[id][0]}')
              print(item['ok_text'][0])
              ansi_print.a_print(item['bad_text'][0], item['ok_text'][0],white, yellow)
              print(raw_output_str)
              print(final_str)
              ansi_print.a_print(final_str, item['ok_text'][0], green, red )
              #print(f'edit distance: {edit_distance}')
            except:
              print('error printing example - prob encoding')

    print(f'Average edit distance: {ansi_print.colors[green]}{sum_distance/1000:.2f}{ansi_print.colors[white]}')
    print(f'Average indel similarity: {ansi_print.colors[green]}{sum_ratio/1000:.4f}{ansi_print.colors[white]}') #1 - normalized_distance

train()