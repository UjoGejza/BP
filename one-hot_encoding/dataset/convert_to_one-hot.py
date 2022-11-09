import torch
import numpy as np
from torch.utils.data import Dataset

def add_typos(file:str, prob:float):
    f = open(file, "r")
    o = open('one-hot_encoding/dataset/corpus_with_typos.txt', "w")
    position = 0
    for line in f:
        for i,character in enumerate(line):
            if ((character>='A'<='Z') or (character>='a'<='z') ):
                if torch.rand(1).item()<=prob:
                    character = chr(int((torch.rand(1).item()*26)) + 97)
            o.write(character)
        position = position+i+1                    
    f.close()
    o.close()

def convert_to_one_hot(file:str) -> torch.Tensor:
    f = open(file, "r")
    file_len = 0
    for line in f.readlines():
        file_len += len(line)

    one_hot = torch.zeros(26)
    print(one_hot)

    data = torch.randn(file_len, 26)
    position = 0

    f = open(file, "r")
    for line in f:
        line = line.upper()
        for i,character in enumerate(line):
            if (character>='A'<='Z'):
                one_hot[(ord(character))-65] = 1.
                data[position+i] = one_hot
                one_hot = torch.zeros(26)
            else:
                data[position+i] = torch.zeros(26)
        position = position + i+1

    o = open("one-hot_encoding/dataset/output.txt", 'w')
    np.set_printoptions(threshold=np.inf)
    print(data.numpy(), file=o)
    o.close()
    f.close()
    return data

filepath = "one-hot_encoding/dataset/corpus.txt"
one_hot_tensor = convert_to_one_hot(filepath)
add_typos(filepath, 0.03)
