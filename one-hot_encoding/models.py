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

class NeuralNetworkOneHotConv1(nn.Module):
    def __init__(self):
        super(NeuralNetworkOneHotConv1, self).__init__()
        self.conv1 = nn.Conv1d(162, 81, 3)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(81, 60, 3)
        self.conv3 = nn.Conv1d(60, 50, 3)
        self.f = nn.Flatten()
        self.linear1 = nn.Linear(50 * 44, 1000)
        self.linear2 = nn.Linear(1000, 256)
        self.linear3 = nn.Linear(256, 50)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.f(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        
        return out

class NeuralNetworkOneHotConv2(nn.Module):
    def __init__(self):
        super(NeuralNetworkOneHotConv2, self).__init__()
        self.conv1 = nn.Conv1d(162, 90, 3)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(90, 60, 3)
        self.conv3 = nn.Conv1d(60, 30, 3)
        self.f = nn.Flatten()
        self.linear1 = nn.Linear(30 * 44, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 50)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.f(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        
        return out

class Conv2BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [nn.Conv1d(162, 90, 3),
                    nn.BatchNorm1d(90),
                    nn.LeakyReLU(),
                    nn.Conv1d(90, 60, 3),
                    nn.BatchNorm1d(60),
                    nn.LeakyReLU(),
                    nn.Conv1d(60, 30, 3),
                    nn.BatchNorm1d(30),
                    nn.LeakyReLU(),
                    nn.Flatten(),
                    nn.Linear(30*44, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 50),
                    nn.Sigmoid()]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)



class Conv2Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = [nn.Conv1d(162, 90, 3),
                    nn.LeakyReLU(),
                    nn.Conv1d(90, 60, 3),
                    nn.LeakyReLU(),
                    nn.Conv1d(60, 30, 3),
                    nn.LeakyReLU()]
        self.conv = nn.Sequential(*self.conv)
        self.rec = nn.LSTM(30, 30, 2)
        self.lin = [nn.Flatten(),
                    nn.Linear(30*44, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 50),
                    nn.Sigmoid()]
        self.lin = nn.Sequential(*self.lin)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.lin(x)

class Conv2RecurrentCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = [nn.Conv1d(162, 90, 3),
                    nn.LeakyReLU(),
                    nn.Conv1d(90, 60, 3),
                    nn.LeakyReLU(),
                    nn.Conv1d(60, 30, 3),
                    nn.LeakyReLU()]
        self.conv = nn.Sequential(*self.conv)
        self.rec = nn.LSTM(30, 30, 2)
        self.lin = [nn.Flatten(),
                    nn.Linear(30*44, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 50)]
        self.lin = nn.Sequential(*self.lin)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.lin(x)


class NeuralNetworkCorrection2(nn.Module):
    def __init__(self):
        super(NeuralNetworkCorrection2, self).__init__()
        self.conv1 = nn.Conv1d(162, 90, 3)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(90, 60, 3)
        self.conv3 = nn.Conv1d(60, 30, 3)
        self.f = nn.Flatten()
        self.linear1 = nn.Linear(30 * 44, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 50)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.f(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        
        return out