import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m = nn.Conv1d(16, 33, 3, stride=2)
m.to(device)
input = torch.randn(20, 16, 50)
input = input.to(device)
output = m(input)
print('sucess')