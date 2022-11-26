import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, file:str):
        self.charlist = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u', 
         'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '¡', '?', '¿', ',', '—', '…', '”', '“',
         '.', '·', ':', ';', '\\', '_', '&', '#', '@', '(', ')', '[', ']', '{', '}', '+', '-', '*', '/', '±', '=', '≠', '<', '>', '≤', 
         '≥', 'ϵ', '∞', '%', '‰', '£', '€', '$', '§', '©', '®', '℥', "'", '‘', '’', '`', '„', '“', '"', '»', '«', '›', '‹', '☞', '☜', 
         '^', '~', '°', '˛', '†', '|', '⁂', '⊥', '¬', '¤', 'Á', 'Č', 'Ď', 'É', 'Ě', 'Í', 'Ň', 'Ó', 'Ř', 'Š', 'Ť', 'Ú', 'Ů', 'Ý', 'Ž',
        'á', 'č', 'ď', 'é', 'ě', 'í', 'ň', 'ó', 'ř', 'š', 'ť', 'ú', 'ů', 'ý', 'ž']


        with open(file, "r", encoding="cp1252") as f: #0x92 is a smart quote(’) of Windows-1252. It simply doesn't exist in unicode, therefore it can't be decoded.
            lines = f.readlines()
        sample_count = len(lines)//3
        sample_size = len(lines[1])-1#ignore '\n'?
        channels = len(self.charlist)
        
        self.IDs = torch.zeros(sample_count)
        self.ok_samples = torch.zeros(sample_count, sample_size)
        self.bad_samples = torch.zeros(sample_count, sample_size)
        self.labels = torch.zeros(sample_count, sample_size)
        self.ok_samples_one_hot = torch.zeros(sample_count, sample_size, channels)
        self.bad_samples_one_hot = torch.zeros(sample_count, sample_size, channels)
        self.ok_text = np.ndarray(shape=(sample_count, sample_size))
        self.bad_text = np.ndarray(shape=(sample_count, sample_size))

        self.IDs = lines[::3]
        self.ok_text = lines[1::3]
        self.bad_text = lines[2::3]

        for index,id in enumerate(self.IDs):
            id = int(id[2:])
            self.IDs[index] = id

            self.ok_text[index] = self.ok_text[index][:-1]#ignore '\n'
            self.bad_text[index] = self.bad_text[index][:-1]#ignore '\n'
            
            sample = []
            for i,character in enumerate(self.ok_text[index]):
                sample.append(self.charlist.index(character))
                self.ok_samples_one_hot[index][i][self.charlist.index(character)] = 1
            self.ok_samples[index] = torch.tensor(sample)
            
            sample = []            
            for i,character in enumerate(self.bad_text[index]):
                sample.append(self.charlist.index(character))
                self.bad_samples_one_hot[index][i][self.charlist.index(character)] = 1
            self.bad_samples[index] = torch.tensor(sample)

            self.labels[index] = torch.tensor([1 if ok == bad else 0 for ok, bad in zip(self.ok_samples[index], self.bad_samples[index])])


    def __len__(self):
        return len(self.ok_samples)


    def __getitem__(self, idx):
        #maybe move convert to one-hot here, idk
        return {'id': self.IDs[idx],
                'ok_sample': self.ok_samples[idx],
                'bad_sample': self.bad_samples[idx],
                'ok_sample_one_hot': self.ok_samples_one_hot[idx],
                'bad_sample_one_hot': self.bad_samples_one_hot[idx],
                'ok_text': self.ok_text[idx],
                'bad_text': self.bad_text[idx],
                'label': self.labels[idx],}



MD = MyDataset('one-hot_encoding/data/corpus_processed_with_typos.txt')
item = MD.__getitem__(1637)
o = open("one-hot_encoding/data/output.txt", 'w')
np.set_printoptions(threshold=np.inf)
    
print(MD.__len__(), file = o)
print(item['id'], file = o)
print(item['ok_text'], file = o)
print(item['bad_text'], file = o)
print(item['ok_sample'], file = o)
print(item['bad_sample'], file = o)
print(item['ok_sample_one_hot'].numpy(), file = o)
print(item['bad_sample_one_hot'].numpy(), file = o)

o.close()



