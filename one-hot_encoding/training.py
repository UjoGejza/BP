
import torch
from torch import nn
from torch.utils.data import DataLoader

from numpy import random

from dataset import MyDataset
from process_corpus import add_typos
from models import NeuralNetworkOneHot, NeuralNetwork, NeuralNetworkOneHotConv1
import ansi_print

import wandb

#wandb.init(project="my-test-project-BP")

#wandb.config = {
#  "learning_rate": 0.0006,
#  "epochs": 150,
#  "batch_size": 30
#}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#batch_size = wandb.config['batch_size']
#epochs = wandb.config['epochs']
#learning_rate = wandb.config['learning_rate']

batch_size = 30
epochs = 150
learning_rate = 0.0006

training_data = MyDataset('one-hot_encoding/data/corpus_processed.txt')
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
testing_test_data = MyDataset('one-hot_encoding/data/corpus_test_processed_with_typos.txt')
testing_test_data_loader = DataLoader(testing_test_data, shuffle=True)
testing_train_data = MyDataset('one-hot_encoding/data/corpus_train_test_processed_with_typos.txt')
testing_train_data_loader = DataLoader(testing_train_data, shuffle=True)

for x in training_data_loader:
    print(x['bad_sample'].shape)
    print(x['label'].shape)
    print(x['ok_sample_one_hot'].shape)
    print(x['bad_sample_one_hot'].shape)
    break

model = NeuralNetworkOneHotConv1()
model.to(device)
#nn.BCEWithLogitsLoss
loss_fn = nn.BCELoss()
print(model.parameters())
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
                        item['bad_sample_one_hot'][i_batch][i_sample][training_data.charlist.index(character)] = 1
                        item['bad_sample'][i_batch][i_sample] = training_data.charlist.index(character)
            bad_sample.append(character)
        item['bad_text'][i_batch] = ''.join(bad_sample)
    return item

def add_typos(item):#used only during training
    for i_batch, _ in enumerate(item['label']):
        error_index = random.randint(50, size=(5))
        error_char = random.randint(low=97, high=123, size=(5))
        for i in range(5):
            #item['bad_text'][i_batch][error_index[i]] = chr(error_char[i])
            #'bad_text' is not updated with typos
            if chr(error_char[i]) != item['ok_text'][i_batch][error_index[i]]:
                item['label'][i_batch][error_index[i]] = 0
                item['bad_sample_one_hot'][i_batch][error_index[i]] = torch.zeros(162)#training_data.channels
                item['bad_sample_one_hot'][i_batch][error_index[i]][training_data.charlist.index(chr(error_char[i]))] = 1
                #item['bad_sample'][i_batch][error_index[i]] = training_data.charlist.index(chr(error_char[i]))
    return item
        

        

def train():
    #n_total_steps = len(training_data_loader)
    for epoch in range(epochs):
        for i, item in enumerate(training_data_loader):
            item = typos(item)
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
            #print(item['bad_sample_one_hot'].shape)
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
            item['label'] = item['label'].to(device)
            outputs = model(item['bad_sample_one_hot'])
            loss = loss_fn(outputs, item['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            print(f'epoch {epoch+1}, loss = {loss.item():.4f}')
            #wandb.log({"loss": loss})
            # Optional
            #wandb.watch(model)
            print('Train data test:')
            test(testing_train_data_loader)
            #print('\033[0;34mTest data test:\033[0;37m')
            #test(testing_test_data_loader)

def test(data_loader):
    TP, FP, TN, FN, TPR, PPV, F1, ACC_CM, TNR, BA = 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    confusion_matrix = torch.rand(2,2)
    for item in data_loader:
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        item['label'] = item['label'].to(device)
        #print(item['bad_sample_one_hot'].shape)
        outputs = model(item['bad_sample_one_hot'])
        outputs = outputs[0]
        item['label'] = item['label'][0]
        outputs = [1 if out>0.5 else 0 for out in outputs]
        for index, out in enumerate(outputs):
            if item['label'][index] == 1 and out>0.5: TP +=1
            if item['label'][index] == 1 and out<=0.5: FN +=1
            if item['label'][index] == 0 and out>0.5: FP +=1
            if item['label'][index] == 0 and out<=0.5: TN +=1
    confusion_matrix[0][0] = TP
    confusion_matrix[0][1] = FN
    confusion_matrix[1][0] = FP
    confusion_matrix[1][1] = TN
    print(confusion_matrix)
    TPR = TP/(TP+FN) #sensitivity, recall, hit rate, or true positive rate (TPR)
    TNR = TN/(TN+FP) #specificity, selectivity or true negative rate (TNR)
    PPV = TP/(TP+FP) #precision or positive predictive value (PPV)
    F1 = 2 * (PPV * TPR)/(PPV + TPR) #F1 score is the harmonic mean of precision and sensitivity:
    ACC_CM = (TP + TN)/(TP + TN + FP + FN) #accuracy
    BA = (TPR + TNR)/2 #balanced accuracy
    ansi_print.a_print(item['bad_text'][0], item['ok_text'][0], 'yellow')
    ansi_print.a_print(outputs, item['label'], 'red')
    print(f'Accuracy: {ACC_CM*100:.2f}%')
    print(f'Balanced accuracy: {BA*100:.2f}%')
    print(f'Recall: {TPR:.4f}, Precision: {PPV:.4f}, F1: {F1:.4f}')

train()