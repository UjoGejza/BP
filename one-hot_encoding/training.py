
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import MyDataset
from models import NeuralNetwork

MD = MyDataset('one-hot_encoding/data/corpus_processed_with_typos.txt')
data_loader = DataLoader(MD, batch_size=32)
#shape: batch = 32, C = 1, H = 1, W(T) = 50.

for x in data_loader:
    print(x['bad_sample'].shape)
    print(x['label'].shape)
    print(x['ok_sample_one_hot'].shape)
    print(x['bad_sample_one_hot'].shape)
    break

model = NeuralNetwork()
#nn.BCEWithLogitsLoss
loss_fn = nn.BCELoss()
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    n_total_steps = len(data_loader)
    for epoch in range(100):
        for i, item in enumerate(data_loader):

            outputs = model(item['bad_sample'])
            loss = loss_fn(outputs, item['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)%10==0:
            print(f'epoch {epoch+1}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')


train()