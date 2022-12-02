
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import MyDataset
from models import NeuralNetworkOneHot, NeuralNetwork
import ansi_print

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_data = MyDataset('data/corpus_processed_with_typos.txt')
training_data_loader = DataLoader(training_data, batch_size=32, shuffle = True)
testing_test_data = MyDataset('data/corpus_test_processed_with_typos.txt')
testing_test_data_loader = DataLoader(testing_test_data, shuffle=True)
testing_train_data = MyDataset('data/corpus_train_test_processed_with_typos.txt')
testing_train_data_loader = DataLoader(testing_train_data, shuffle=True)
#shape: batch = 32, C = 1, H = 1, W(T) = 50.

for x in training_data_loader:
    print(x['bad_sample'].shape)
    print(x['label'].shape)
    print(x['ok_sample_one_hot'].shape)
    print(x['bad_sample_one_hot'].shape)
    break

model = NeuralNetworkOneHot()
model.to(device)
#nn.BCEWithLogitsLoss
loss_fn = nn.BCELoss()
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    n_total_steps = len(training_data_loader)
    for epoch in range(200):
        for i, item in enumerate(training_data_loader):
            item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
            item['label'] = item['label'].to(device)
            outputs = model(item['bad_sample_one_hot'])
            loss = loss_fn(outputs, item['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'epoch {epoch+1}, loss = {loss.item():.4f}')
        print('Train data test:')
        test(testing_train_data_loader)
        print('\033[0;34mTest data test:\033[0;37m')
        test(testing_test_data_loader)

def test(data_loader):
    correct = 0
    all = 0
    for item in data_loader:
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        item['label'] = item['label'].to(device)
        outputs = model(item['bad_sample_one_hot'])
        outputs = outputs[0]
        item['label'] = item['label'][0]
        for index, out in enumerate(outputs):
            if abs(out - item['label'][index])<0.1:
                correct += 1
            all +=1
    ansi_print.a_print(item['bad_text'][0], item['ok_text'][0])
    outputs = [1 if out>0.9 else 0 for out in outputs]
    ansi_print.a_print(outputs, item['label'])
    acc = correct/all
    print(f'TEST: acc: {acc*100:.4f}%')


train()