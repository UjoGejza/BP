# models.py
# Author: Sebastián Chupáč
# This file contains classes for all proposed architectures. 
import torch
from torch import nn

#most of the models were just experimental

#currently used: ConvLSTMCorrection/Detection(Bigger)(CTC)


#V========== OLD MODELS ==========V

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
        self.conv = [nn.Conv1d(69, 128, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 5, padding=2),
                    nn.LeakyReLU()]
        self.conv = nn.Sequential(*self.conv)
        self.rec = nn.LSTM(512, 256, 2, bidirectional=True)
        self.lin = [nn.Flatten(),
                    nn.Linear(512*50, 4096),
                    nn.LeakyReLU(),
                    nn.Linear(4096, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 50)]
        self.lin = nn.Sequential(*self.lin)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.lin(x)
    
class Conv2BiggerKernelRecurrentCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = [nn.Conv1d(69, 128, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.LeakyReLU()]
        self.conv = nn.Sequential(*self.conv)
        self.rec = nn.LSTM(512, 256, 2, bidirectional=True)
        self.lin = [nn.Flatten(),
                    nn.Linear(512*50, 4096),
                    nn.LeakyReLU(),
                    nn.Linear(4096, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 50)]
        self.lin = nn.Sequential(*self.lin)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.lin(x)
    
class Conv2BiggerKernelAggRecurrentCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = [nn.Conv1d(69, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv = nn.Sequential(*self.conv)
        self.rec = nn.LSTM(512, 256, 2, bidirectional=True)
        self.agg = [nn.Conv1d(512, 50, 5, padding=2),
                    nn.BatchNorm1d(50),
                    nn.LeakyReLU()]
        self.agg = nn.Sequential(*self.agg)
        self.lin = [nn.Flatten(),
                    nn.Linear(50*50, 2048),
                    nn.LeakyReLU(),
                    nn.Linear(2048, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 50)]
        self.lin = nn.Sequential(*self.lin)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        x = self.agg(x)
        return self.lin(x)

class Conv2BiggerKernelAggRecurrentDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = [nn.Conv1d(69, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv = nn.Sequential(*self.conv)
        self.rec = nn.LSTM(512, 256, 2, bidirectional=True)
        self.agg = [nn.Conv1d(512, 50, 5, padding=2),
                    nn.BatchNorm1d(50),
                    nn.LeakyReLU()]
        self.agg = nn.Sequential(*self.agg)
        self.lin = [nn.Flatten(),
                    nn.Linear(50*50, 2048),
                    nn.LeakyReLU(),
                    nn.Linear(2048, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 50),
                    nn.Sigmoid()]
        self.lin = nn.Sequential(*self.lin)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        x = self.agg(x)
        return self.lin(x)
    
class LSTMRecurrentCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.rec = nn.LSTM(69, 256, 2, bidirectional=True)
        self.lin = [nn.Flatten(),
                    nn.Linear(512*50, 4096),
                    nn.LeakyReLU(),
                    nn.Linear(4096, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 50)]
        self.lin = nn.Sequential(*self.lin)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.lin(x)

class Conv2RecurrentDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = [nn.Conv1d(69, 128, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 5, padding=2),
                    nn.LeakyReLU()]
        self.conv = nn.Sequential(*self.conv)
        self.rec = nn.LSTM(512, 256, 2, bidirectional=True)
        self.lin = [nn.Flatten(),
                    nn.Linear(512*50, 4096),
                    nn.LeakyReLU(),
                    nn.Linear(4096, 256),
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

#----------------------------------------

#V========== DETECTION MODELS ==========V
#                              input size -> output size
#    text_length(50) x charset_length(69) -> text_length x 1
class ConvLSTMDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = [nn.Conv1d(69, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv1 = nn.Sequential(*self.conv1)
        self.rec = nn.LSTM(512, 256, 2, bidirectional=True)
        self.conv2 = [nn.Conv1d(512, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 64, 9, padding=4),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(),
                    nn.Conv1d(64, 32, 5, padding=2),
                    nn.BatchNorm1d(32),
                    nn.LeakyReLU(),
                    nn.Conv1d(32, 1, 5, padding=2),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid()]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.conv2(x)
    
#                              input size -> output size
#    text_length(50) x charset_length(69) -> text_length x 1
class ConvLSTMDetectionBigger(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = [nn.Conv1d(69, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv1 = nn.Sequential(*self.conv1)
        self.rec = nn.LSTM(512, 256, 4, bidirectional=True)
        self.conv2 = [nn.Conv1d(512, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 64, 5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(),
                    nn.Conv1d(64, 64, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(64, 32, 9, padding=4),
                    nn.BatchNorm1d(32),
                    nn.LeakyReLU(),
                    nn.Conv1d(32, 32, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(32, 1, 5, padding=2),
                    nn.BatchNorm1d(1),
                    nn.LeakyReLU(),
                    nn.Sigmoid()]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.conv2(x)

#-----------------------------------------

#V========== CORRECTION MODELS ==========V
#                              input size -> output size
#    text_length(50) x charset_length(69) -> text_length x 69
class ConvLSTMCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = [nn.Conv1d(69, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv1 = nn.Sequential(*self.conv1)
        self.rec = nn.LSTM(512, 256, 2, bidirectional=True)
        self.conv2 = [nn.Conv1d(512, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 69, 5, padding=2),
                    nn.BatchNorm1d(69),
                    nn.LeakyReLU()]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.conv2(x)
    
#                              input size -> output size
#    text_length(50) x charset_length(69) -> text_length x 69
class ConvLSTMCorrectionBigger(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = [nn.Conv1d(69, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv1 = nn.Sequential(*self.conv1)
        self.rec = nn.LSTM(512, 256, 4, bidirectional=True)
        self.conv2 = [nn.Conv1d(512, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9, padding=4),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 69, 5, padding=2),
                    nn.BatchNorm1d(69),
                    nn.LeakyReLU()]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.conv2(x)
    

#                              input size -> output size
#    text_length(73) x charset_length(90) -> text_length+10(83) x 90
class ConvLSTMCorrectionCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = [nn.Conv1d(90, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv1 = nn.Sequential(*self.conv1)
        self.rec = nn.LSTM(512, 256, 2, bidirectional=True)
        self.conv2 = [nn.ConvTranspose1d(512, 256, 9, padding=2),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(256, 128, 9, padding=2),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(128, 90, 5, padding=1),
                    nn.BatchNorm1d(90),
                    nn.LeakyReLU()]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.conv2(x)
    
#                              input size -> output size
#    text_length(73) x charset_length(90) -> text_length+10(83) x 90
class ConvLSTMCorrectionCTCBigger(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = [nn.Conv1d(90, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv1 = nn.Sequential(*self.conv1)
        self.rec = nn.LSTM(512, 256, 4, bidirectional=True)
        self.conv2 = [nn.ConvTranspose1d(512, 256, 9, padding=3),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(256, 256, 9,  padding=3),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(256, 128, 9, padding=3),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(128, 128, 9, padding=3),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(128, 90, 5, padding=1),
                    nn.BatchNorm1d(90),
                    nn.LeakyReLU()]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.conv2(x)
    
#                              input size -> output size
#    text_length(73) x charset_length(90) -> text_length+2(75) x 90
class ConvLSTMCorrectionCTCBiggerPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = [nn.Conv1d(90, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv1 = nn.Sequential(*self.conv1)
        self.rec = nn.LSTM(512, 256, 4, bidirectional=True)
        self.conv2 = [nn.ConvTranspose1d(512, 256, 9, padding=4),#conv
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(256, 256, 9,  padding=4),#conv
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),#deconv(2,2)
                    nn.ConvTranspose1d(256, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(128, 128, 9, padding=4),#conv
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(128, 90, 5, padding=1),#conv
                    nn.BatchNorm1d(90),
                    nn.LeakyReLU()]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.conv2(x)
    
#                              input size -> output size
#    text_length(73) x charset_length(90) -> text_lengthx2(146) x 90
class ConvLSTMCorrectionCTCBiggerPad2x(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = [nn.Conv1d(90, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU()]
        self.conv1 = nn.Sequential(*self.conv1)
        self.rec = nn.LSTM(512, 256, 4, bidirectional=True)
        self.conv2 = [nn.Conv1d(512, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(256, 256, 2, 2),#doubles size, (256, 256, 8, 2, padding=3)
                    nn.Conv1d(256, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 90, 9, padding=4),
                    nn.BatchNorm1d(90),
                    nn.LeakyReLU()]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.conv2(x)
    
class UNetCorrectionCTCBiggerPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = [nn.Conv1d(90, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(2, 2),
                    nn.Conv1d(128, 256, 9, padding=3),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=4),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(2, 2),
                    nn.Conv1d(256, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(),
                    nn.Conv1d(512, 512, 9, padding=4),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(),]
        self.conv1 = nn.Sequential(*self.conv1)
        self.rec = nn.LSTM(512, 256, 4, bidirectional=True)
        self.conv2 = [nn.Conv1d(512, 256, 9, padding=5),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Conv1d(256, 256, 9, padding=5),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(256, 256, 2, 2),
                    nn.Conv1d(256, 128, 9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 9,  padding=4),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(128, 128, 2, 2),
                    nn.Conv1d(128, 90, 9, padding=4),
                    nn.BatchNorm1d(90),
                    nn.LeakyReLU()]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x,_ = self.rec(x)
        x = x.permute(1, 2, 0)
        return self.conv2(x)
    
#---------------------------------------------
