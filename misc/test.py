import torch

a = [4, 2, 5, 7]
b = [4, 2, 1, 7]

at = torch.tensor(a)
bt = torch.tensor(b)

print(at)
print(bt)

mask = at == bt

print(mask)

label = torch.ones_like(mask, dtype=float)
for i,value in enumerate(mask):
    if value == False: 
        label[i] = 0
print(label)