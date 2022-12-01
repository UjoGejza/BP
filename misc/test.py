import torch
from datasets import load_dataset

#load_dataset("wikipedia", "20220301.simple")

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