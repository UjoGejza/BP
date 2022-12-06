import torch
from torch import nn

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
        
        return out

class NeuralNetworkOneHot(nn.Module):
    def __init__(self):
        super(NeuralNetworkOneHot, self).__init__()
        self.f = nn.Flatten()
        self.linear1 = nn.Linear(50 * 162, 2048)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 50)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.f(x)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        
        return out

#conv1d model