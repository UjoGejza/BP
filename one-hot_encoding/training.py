
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import MyDataset
from process_corpus import add_typos
from models import NeuralNetworkOneHot, NeuralNetwork, NeuralNetworkOneHotConv
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

training_data = MyDataset('one-hot_encoding/data/corpus_processed_with_typos.txt')
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

model = NeuralNetworkOneHotConv()
model.to(device)
#nn.BCEWithLogitsLoss
loss_fn = nn.BCELoss()
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    #n_total_steps = len(training_data_loader)
    for epoch in range(epochs):
        for i, item in enumerate(training_data_loader):
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
    correct = 0
    all = 0
    for item in data_loader:
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        item['label'] = item['label'].to(device)
        #print(item['bad_sample_one_hot'].shape)
        outputs = model(item['bad_sample_one_hot'])
        outputs = outputs[0]
        item['label'] = item['label'][0]
        for index, out in enumerate(outputs):
            if abs(out - item['label'][index])<0.1:
                correct += 1
            all +=1
    ansi_print.a_print(item['bad_text'][0], item['ok_text'][0], 'yellow')
    outputs = [1 if out>0.9 else 0 for out in outputs]
    ansi_print.a_print(outputs, item['label'], 'red')
    acc = correct/all
    print(f'TEST: acc: {acc*100:.4f}%')


train()