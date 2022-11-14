import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, file:str):
        self.f = open(file, "r", encoding="cp1252") #0x92 is a smart quote(’) of Windows-1252. It simply doesn't exist in unicode, therefore it can't be decoded.
        self.charlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
                        'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', ',', '.', '!', 
                        '?', '\'', '\"', '’', '-', '+', '…', '“', '”', '(', ')', ':', '/', ';']
        self.clear_samples = torch.zeros(1750, 50)
        self.dirty_samples = torch.zeros(1750, 50)
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

    
    def __len___(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        return {'id': self.IDs[idx],
                'clear_sample': self.clear_samples[idx],
                'dirty_sample': self.dirty_samples[idx],
                'label': self.labels[idx]}

MD = MyDataset('one-hot_encoding/dataset/corpus_processed_with_typos.txt')
item = MD.__getitem__(5)
print(item)
