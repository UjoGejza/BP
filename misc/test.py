#misc
import torch
from numpy import random
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    RESET = '\e[0m'
    WHITE = '\033[0;37m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

samples = torch.zeros(4, 5, dtype=float)
batch_index = torch.tensor(np.arange(0, 50, 0.25), dtype=int)
samples[0] = torch.tensor([1, 2, 3, 4, 5])
samples[1] = torch.tensor([6, 7, 8, 9, 10])
samples[2] = torch.tensor([11, 12, 13, 14, 15])
samples[3] = torch.tensor([16, 17, 18, 19, 20])
error_index = random.randint(5, size=(2*4))
error_char = random.randint(low=97, high=123, size=(2*4))
print(samples[[0, 0, 1]])
samples[[0, 0, 1], [0, 2, 3]] = torch.tensor([50, 60, 70], dtype=float)
print(samples)


pismeno1 = torch.tensor([0, 0, 1, 0, 0, 0])
pismeno2 = torch.tensor([1, 0, 0, 0, 0, 0])
pismeno3 = torch.tensor([0, 1, 0, 0, 0, 0])

slovo = torch.zeros(3, 6)
slovo[0] = pismeno1
slovo[1] = pismeno2
slovo[2] = pismeno3

print(slovo)
print(slovo.shape)

slovo = slovo.transpose(0, 1)
print(slovo)
print(slovo.shape)
print(pismeno1)
pismeno1 = pismeno1.reshape(1, -1)

print(pismeno1)

a = [4, 2, 5, 7]
b = [4, 2, 1, 7]
print("test")
o = open('out.txt', 'w', encoding='ansi')
for i,value_a in enumerate(a):
    if value_a == b[i] : color = bcolors.OKGREEN
    else: color = bcolors.FAIL
    print(f"{color}{b[i]}{bcolors.WHITE}reset", end='', file=o)

torch2d = torch.rand((3, 4))

print(torch2d)

torch2d = torch.reshape(torch2d, (6, -1))

print(torch2d)

at = torch.tensor(a)
bt = torch.tensor(b)

for jedna, dva in zip(at,bt):
    print(jedna)
    print(dva)

print(at)
print(bt)
mask = []
mask = at == bt

print(mask)
label = [1 if m == True else 0 for m in mask]
decimal = [0.4, 0.75, 0.51, 0.01, 0.033]
decimal = [1 if d >0.5 else 0 for d in decimal]
print(decimal)

print(label)