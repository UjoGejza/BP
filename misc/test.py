import torch

a = [4, 2, 5, 7]
b = [4, 2, 1, 7]

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