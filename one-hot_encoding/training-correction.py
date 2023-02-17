
import torch
from torch import nn
from torch.utils.data import DataLoader

from numpy import random

from dataset import MyDataset
from models import Conv2RecurrentCorrection
import ansi_print

#import wandb

#wandb.init(project="my-test-project-BP")

#wandb.config = {
#  "learning_rate": 0.0006,
#  "epochs": 150,
#  "batch_size": 30
#}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USING: {device}')
#batch_size = wandb.config['batch_size']
#epochs = wandb.config['epochs']
#learning_rate = wandb.config['learning_rate']

batch_size = 50
epochs = 200
learning_rate = 0.001

training_data = MyDataset('one-hot_encoding/data/wiki-20k.txt')
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
testing_test_data = MyDataset('one-hot_encoding/data/wiki-1k-test-insert-swap.txt')
testing_test_data_loader = DataLoader(testing_test_data, shuffle=True)
testing_train_data = MyDataset('one-hot_encoding/data/wiki-1k-train-insert-swap.txt')
testing_train_data_loader = DataLoader(testing_train_data, shuffle=True)

alphabet = training_data.charlist

#for x in training_data_loader:
#    print(x['bad_sample'].shape)
#    print(x['label'].shape)
#    print(x['ok_sample_one_hot'].shape)
#    print(x['bad_sample_one_hot'].shape)
#    break

model = Conv2RecurrentCorrection()
model.to(device)
#nn.BCEWithLogitsLoss
loss_fn = nn.MSELoss()
print(f'MODEL ARCHITECTURE: ')
for name, param in model.state_dict().items():
    print(name, param.size())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def typos(item):#old very slow
    for i_batch, samples in enumerate(item['label']):
        bad_sample = []
        for i_sample, character in enumerate(item['ok_text'][i_batch]):
            if ((character>='A'<='Z') or (character>='a'<='z') ):
                if torch.rand(1).item()<=0.1:
                    character = chr(int((torch.rand(1).item()*26)) + 97)
                    if character != item['ok_text'][i_batch][i_sample]: 
                        item['label'][i_batch][i_sample] = 0
                        item['bad_sample_one_hot'][i_batch][i_sample] = torch.zeros(162)#training_data.channels
                        item['bad_sample_one_hot'][i_batch][i_sample][alphabet.index(character)] = 1
                        item['bad_sample'][i_batch][i_sample] = alphabet.index(character)
            bad_sample.append(character)
        item['bad_text'][i_batch] = ''.join(bad_sample)
    return item


def add_typos(item):
    error_index = random.randint(50, size=(4*50))
    error_char = random.randint(low=97, high=123, size=(4*50))
    extra_char_one_hot = torch.zeros(1, 162)
    extra_char_one_hot[0][alphabet.index('#')] = 1
    for i in range(4*50):
        if (i%4)>0:
            #swap char for another char
            bad_text = list(item['bad_text'][i//4])
            item['bad_sample'][i//4][error_index[i]] = alphabet.index(chr(error_char[i]))
            bad_text[error_index[i]] = chr(error_char[i])
            item['bad_text'][i//4] = ''.join(bad_text)
            if chr(error_char[i]) != item['ok_text'][i//4][error_index[i]]:
                item['label'][i//4][error_index[i]] = 0
                item['bad_sample_one_hot'][i//4][error_index[i]] = torch.zeros(162)#training_data.channels
                item['bad_sample_one_hot'][i//4][error_index[i]][alphabet.index(chr(error_char[i]))] = 1
                #item['bad_sample'][i_batch][error_index[i]] = training_data.charlist.index(chr(error_char[i]))
        else:
            #insert extra char
            base_one_hot = torch.zeros(1, 162)
            base_one_hot[0][alphabet.index(chr(error_char[i]))] = 1
            bad_text = list(item['bad_text'][i//4])
            ok_text = list(item['ok_text'][i//4])
            label = list(item['label'][i//4])
            ok_sample = list(item['ok_sample'][i//4])
            bad_sample = list(item['bad_sample'][i//4])
            
            bad_text.insert(error_index[i], chr(error_char[i]))
            ok_text.insert(error_index[i], '#')
            label.insert(error_index[i], 0)
            ok_sample.insert(error_index[i], alphabet.index('#'))
            bad_sample.insert(error_index[i], alphabet.index(chr(error_char[i])))
            item['bad_sample_one_hot'][i//4] = torch.cat((item['bad_sample_one_hot'][i//4][:error_index[i]], base_one_hot, item['bad_sample_one_hot'][i//4][error_index[i]:-1]), 0)
            item['ok_sample_one_hot'][i//4] = torch.cat((item['ok_sample_one_hot'][i//4][:error_index[i]], extra_char_one_hot, item['ok_sample_one_hot'][i//4][error_index[i]:-1]), 0)
            
            bad_text.pop(len(bad_text)-1)
            ok_text.pop(len(ok_text)-1)
            label.pop(len(label)-1)
            ok_sample.pop(len(ok_sample)-1)
            bad_sample.pop(len(bad_sample)-1)

            item['bad_text'][i//4] = ''.join(bad_text)
            item['ok_text'][i//4] = ''.join(ok_text)
            item['label'][i//4] = torch.tensor(label)
            item['ok_sample'][i//4] = torch.tensor(ok_sample)
            item['bad_sample'][i//4] = torch.tensor(bad_sample)

    return item

def train():
    #n_total_steps = len(training_data_loader)
    for epoch in range(epochs):
        for i, item in enumerate(training_data_loader):
            #item = add_typos(item)
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
            #print(item['bad_sample_one_hot'].shape)
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
            item['ok_sample'] = item['ok_sample'].to(device)
            outputs = model(item['bad_sample_one_hot'])
            loss = loss_fn(outputs, item['ok_sample'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            print(f'epoch {epoch+1}, loss = {loss.item():.4f}')
            #wandb.log({"loss": loss})
            # Optional
            #wandb.watch(model)
            #print('Train data test:')
            #test(testing_train_data_loader)
            if (epoch+1)%2 == 0:
              print('Train data test:')
              test(testing_train_data_loader)
              print('\033[0;34mTest data test:\033[0;37m')
              test(testing_test_data_loader)

def test(data_loader):
    #TP, FP, TN, FN, TPR, PPV, F1, ACC_CM, TNR, BA = 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    #confusion_matrix = torch.rand(2,2)
    correct = 0
    all= 0
    for item in data_loader:
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        item['ok_sample'] = item['ok_sample'].to(device)
        #print(item['bad_sample_one_hot'].shape)
        outputs = model(item['bad_sample_one_hot'])
        outputs = outputs[0]
        item['ok_sample'] = item['ok_sample'][0]
        outputs = outputs.round()
        output_text_list = list(outputs)
        for index, out in enumerate(outputs):
            if item['ok_sample'][index] == out: correct+=1
            output_text_list[index] = alphabet[out.int()]
            all +=1
    #TPR = TP/(TP+FN) #sensitivity, recall, hit rate, or true positive rate (TPR)
    #TNR = TN/(TN+FP) #specificity, selectivity or true negative rate (TNR)
    #PPV = TP/(TP+FP) #precision or positive predictive value (PPV)
    #F1 = 2 * (PPV * TPR)/(PPV + TPR) #F1 score is the harmonic mean of precision and sensitivity:
    #ACC_CM = (TP + TN)/(TP + TN + FP + FN) #accuracy
    #BA = (TPR + TNR)/2 #balanced accuracy
    output_text = ''.join(output_text_list)
    acc = correct/all
    ansi_print.a_print(item['bad_text'][0], item['ok_text'][0], 'yellow')
    ansi_print.a_print(output_text, item['ok_text'][0], 'red')
    print(f'Accuracy: {acc*100:.2f}%')
    #print(f'Balanced accuracy: {BA*100:.2f}%')
    #print(f'Recall: {TPR:.4f}, TNR: {TNR:.4f}, Precision: {PPV:.4f}, F1: {F1:.4f}')


train()