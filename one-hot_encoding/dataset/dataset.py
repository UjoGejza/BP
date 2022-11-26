import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class MyDataset(Dataset):
    def __init__(self, file:str):
        self.f = open(file, "r", encoding="cp1252") #0x92 is a smart quote(’) of Windows-1252. It simply doesn't exist in unicode, therefore it can't be decoded.
        self.charlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
                        'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', ',', '.', '!', 
                        '?', '\'', '\"', '’', '-', '+', '…', '“', '”', '(', ')', ':', '/', ';']#add uppercase
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        latin = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
         's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        symbols = ['!', '¡', '?', '¿', ',', '—', '.', '·', ':', ';', '\\', '_', '&', '#', '@', '(', ')', '[', ']', '{', '}', '+',
           '-', '*', '/', '±', '=', '≠', '<', '>', '≤', '≥', 'ϵ', '∞', '%', '‰', '£', '€', '$', '§', '©', '®', '℥', "'",
           '‘', '’', '`', '„', '“', '"', '»', '«', '›', '‹', '☞', '☜', '^', '~', '°', '˛', '†', '|', '⁂', '⊥', '¬', '¤']

        czech_special = ['Á', 'Č', 'Ď', 'É', 'Ě', 'Í', 'Ň', 'Ó', 'Ř', 'Š', 'Ť', 'Ú', 'Ů', 'Ý', 'Ž',
                 'á', 'č', 'ď', 'é', 'ě', 'í', 'ň', 'ó', 'ř', 'š', 'ť', 'ú', 'ů', 'ý', 'ž']
        #todo refactor
        self.clear_samples = torch.zeros(1750, 50)#correct line of text
        self.dirty_samples = torch.zeros(1750, 50)#same line with typos
        self.labels = torch.zeros(1750, 50)
        self.IDs = torch.zeros(1750)
        lnum = 0

        for line in self.f:
            line = line.lower()
            line = line[:-1]

            if lnum % 2 == 0:#correct line
                #get ID
                id_idx = line.find(' ')
                self.IDs[lnum//2] = int(line[2:id_idx])

                line = line[id_idx+1:]
                sample = []
                for character in line:
                    sample.append(self.charlist.index(character))
                sample_tensor = torch.tensor(sample)
                self.clear_samples[lnum//2] = sample_tensor
               
            else:#dirty line
                 #add typos
                '''for i,character in enumerate(line):
                    if (character>='a'<='z'):
                        if torch.rand(1).item()<=0.1:
                            line = line.replace(character, self.charlist[int((torch.rand(1).item()*26))], 1)'''
                
                sample = []
                for character in line:
                    sample.append(self.charlist.index(character))
                sample_tensor = torch.tensor(sample)
                self.dirty_samples[lnum//2] = sample_tensor

                mask = self.clear_samples[lnum//2] == self.dirty_samples[lnum//2]
                label = torch.ones_like(mask, dtype=float)
                for i,value in enumerate(mask):
                    if value == False: 
                        label[i] = 0
                self.labels[lnum//2] = label
            lnum += 1

    
    def __len__(self):
        return len(self.clear_samples)

    def __getitem__(self, idx):
        #TODO prevod na one-hot
        return {'id': self.IDs[idx],
                'clear_sample': self.clear_samples[idx],
                'dirty_sample': self.dirty_samples[idx],
                'label': self.labels[idx]}

MD = MyDataset('one-hot_encoding/dataset/corpus_processed_with_typos.txt')
item = MD.__getitem__(500)
print(MD.__len__())
print(item)

data_loader = DataLoader(MD, batch_size=32)
#shape: batch = 32, C = 1, H = 1, W(T) = 50.

for x in data_loader:
    print(x['dirty_sample'].shape)
    print(x['label'].shape)
    break

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(50, 256)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(256, 50)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        #Binary Activation Function
        #estimator = StraightThroughEstimator()
        #out = estimator(out)
        
        return out

model = NeuralNetwork()
#nn.BCEWithLogitsLoss
loss_fn = nn.BCELoss()
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    n_total_steps = len(data_loader)
    for epoch in range(100):
        for i, item in enumerate(data_loader):

            outputs = model(item['dirty_sample'])
            loss = loss_fn(outputs, item['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)%10==0:
            print(f'epoch {epoch+1}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')


train()

