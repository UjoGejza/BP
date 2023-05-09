# dataset_pad.py
# Author: Sebastián Chupáč
# This file implements PyTorch class dataset for my data. This version for padded input files 
import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    #initializes class, reads prepared items file, creates all required data like one-hot representations, and stores them
    def __init__(self, file:str):
        blank = 'ѧ'
        pad = 'Є'
        #unknown = 'є'
        
        #list of characters the network recognizes, anything thats not in this list will result in error
        self.charlist_extra_ctc = [blank, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u', 
         'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '?', ',', '.','\'','"', '-',':',';' ,
         '(', ')', '%','/', '—', '–', '”', '“', '+', '=', '[', ']', '’', '&', '*', '#', pad]#blank at 0, size: 90

        alphabet = self.charlist_extra_ctc

        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        item_count = len(lines)//3
        #GT and input text are padded to these values
        label_size = 60
        sample_size = 73
        channels = len(alphabet)
        
        self.IDs = torch.zeros(item_count)
        self.labels = torch.zeros(item_count, label_size)
        self.samples = torch.zeros(item_count, sample_size)
        self.samples_one_hot = torch.zeros(item_count, sample_size, channels)
        self.label_text = np.array(lines[1::3])
        self.sample_text = np.array(lines[2::3])
        self.label_length = torch.zeros(item_count, dtype=torch.long)

        self.IDs = lines[::3]

        for index,id in enumerate(self.IDs):
            id = int(id[2:])
            self.IDs[index] = id

            self.label_text[index] = self.label_text[index][:-1]#ignore '\n'
            self.sample_text[index] = self.sample_text[index][:-1]#ignore '\n'
            
            sample = []
            for i,character in enumerate(self.label_text[index]):
                sample.append(alphabet.index(character))
                
            self.labels[index] = torch.tensor(sample)
            self.label_length[index] = self.label_text[index].find(pad, 49, label_size)
            if self.label_length[index] == -1: self.label_length[index] = 60

            
            sample = []          
            for i,character in enumerate(self.sample_text[index]):

                sample.append(alphabet.index(character))
                self.samples_one_hot[index][i][alphabet.index(character)] = 1
            self.samples[index][:len(sample)] = torch.tensor(sample)


    #length of dataset
    def __len__(self):
        return len(self.samples)
 
    #returns one item from dataset
    def __getitem__(self, idx):
        return {'id': self.IDs[idx],
                'label': self.labels[idx],
                'sample': self.samples[idx],
                'sample_one_hot': self.samples_one_hot[idx],
                'label_text': self.label_text[idx],
                'sample_text': self.sample_text[idx],
                'label_length': self.label_length[idx]}




