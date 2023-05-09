# dataset.py
# Author: Sebastián Chupáč
# This file implements PyTorch class dataset for my data. This version for unpadded files with constant length
import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    #initializes class, reads prepared items file, creates all required data like one-hot representations, and stores them
    def __init__(self, file:str):
        #list of characters the network recognizes, anything thats not in this list will result in error, choose one
        self.charlist_base = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u', 
         'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '?', ',', '§','.', '#']#size: 69
        
        self.charlist_extra = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u', 
         'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '?', ',', '.','\'','"', '-',':',';' ,
         '(', ')', '%','/', '—', '–', '”', '“', '+', '=', '§', '[', ']', '’', '&', '~', '*', '#']#size: 90

        self.charlist_base_ctc = ['~', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u', 
         'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '?', ',', '.', '§']#size: 69
        
        self.charlist_extra_ctc = ['~', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u', 
         'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '?', ',', '.','\'','"', '-',':',';' ,
         '(', ')', '%','/', '—', '–', '”', '“', '+', '=', '§', '[', ']', '’', '&', '*', '#']#blank at 0, size: 90

        alphabet = self.charlist_base

        with open(file, "r", encoding="utf-8") as f: #0x92 is a smart quote(’) of Windows-1252. It simply doesn't exist in unicode, therefore it can't be decoded.
            lines = f.readlines()
        sample_count = len(lines)//3
        #length of input and gt text
        item_size = 50
        channels = len(alphabet)
        

        self.IDs = torch.zeros(sample_count)
        self.labels = torch.zeros(sample_count, item_size)
        self.samples = torch.zeros(sample_count, item_size)
        self.bi_classes = torch.zeros(sample_count, item_size)
        self.samples_one_hot = torch.zeros(sample_count, item_size, channels)
        self.label_text = np.array(lines[1::3])
        self.sample_text = np.array(lines[2::3])

        self.IDs = lines[::3]


        for index,id in enumerate(self.IDs):
            id = int(id[2:])
            self.IDs[index] = id

            self.label_text[index] = self.label_text[index][:-1]#ignore '\n'
            self.sample_text[index] = self.sample_text[index][:-1]#ignore '\n'
            
            sample = []
            sample_txt = []
            for i,character in enumerate(self.label_text[index]):
                if (character not in alphabet):
                    character = ' '
                sample_txt.append(character)
                sample.append(alphabet.index(character))
            self.labels[index][:len(sample)] = torch.tensor(sample)
            self.label_text[index] = ''.join(sample_txt)

            
            sample = [] 
            sample_txt = []           
            for i,character in enumerate(self.sample_text[index]):
                if (character not in alphabet): 
                    character = ' '
                sample_txt.append(character)
                sample.append(alphabet.index(character))
                self.samples_one_hot[index][i][alphabet.index(character)] = 1
            self.samples[index][:len(sample)] = torch.tensor(sample)
            self.sample_text[index] = ''.join(sample_txt)

            self.bi_classes[index] = torch.tensor([1 if ok == bad else 0 for ok, bad in zip(self.labels[index], self.samples[index])])

    #length of dataset
    def __len__(self):
        return len(self.labels)
    
    #returns one item from dataset
    def __getitem__(self, idx):
        return {'id': self.IDs[idx],
                'label': self.labels[idx],
                'sample': self.samples[idx],
                'sample_one_hot': self.samples_one_hot[idx],
                'label_text': self.label_text[idx],
                'sample_text': self.sample_text[idx],
                'bi_class': self.bi_classes[idx]}



