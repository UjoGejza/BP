import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, file:str):
        blank = 'ѧ'
        pad = 'Є'
        #unknown = 'є'
        
        self.charlist_extra_ctc = [blank, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u', 
         'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '?', ',', '.','\'','"', '-',':',';' ,
         '(', ')', '%','/', '—', '–', '”', '“', '+', '=', '[', ']', '’', '&', '*', '#', pad]#blank at 0, size: 90

        alphabet = self.charlist_extra_ctc

        with open(file, "r", encoding="utf-8") as f: #0x92 is a smart quote(’) of Windows-1252. It simply doesn't exist in unicode, therefore it can't be decoded.
            lines = f.readlines()
        sample_count = len(lines)//3
        #sample_size = len(lines[1])-1#ignore '\n'?
        ok_sample_size = 60
        bad_sample_size = 73
        channels = len(alphabet)
        
        self.IDs = torch.zeros(sample_count)
        self.ok_samples = torch.zeros(sample_count, ok_sample_size)
        self.bad_samples = torch.zeros(sample_count, bad_sample_size)
        #self.labels = torch.zeros(sample_count, sample_size)
        #self.ok_samples_one_hot = torch.zeros(sample_count, sample_size, channels)
        self.bad_samples_one_hot = torch.zeros(sample_count, bad_sample_size, channels)
        self.ok_text = np.array(lines[1::3])
        self.bad_text = np.array(lines[2::3])
        self.ok_sample_index = torch.zeros(sample_count, dtype=torch.long)

        self.IDs = lines[::3]
        #self.ok_text = lines[1::3]
        #self.bad_text = lines[2::3]

        for index,id in enumerate(self.IDs):
            id = int(id[2:])
            self.IDs[index] = id

            self.ok_text[index] = self.ok_text[index][:-1]#ignore '\n'
            self.bad_text[index] = self.bad_text[index][:-1]#ignore '\n'
            
            sample = []
            sample_txt = []
            for i,character in enumerate(self.ok_text[index]):
                #if (character not in alphabet): #change this to $ change it also lower
                    #character = unknown
                #sample_txt.append(character)
                sample.append(alphabet.index(character))
                #self.ok_samples_one_hot[index][i][alphabet.index(character)] = 1
            self.ok_samples[index] = torch.tensor(sample)
            #self.ok_text[index] = ''.join(sample_txt)
            self.ok_sample_index[index] = self.ok_text[index].find(pad, 49, ok_sample_size)
            if self.ok_sample_index[index] == -1: self.ok_sample_index[index] = 60

            
            sample = [] 
            #sample_txt = []           
            for i,character in enumerate(self.bad_text[index]):
                #if (character not in alphabet): 
                    #character = unknown
                #sample_txt.append(character)
                sample.append(alphabet.index(character))
                self.bad_samples_one_hot[index][i][alphabet.index(character)] = 1
            self.bad_samples[index][:len(sample)] = torch.tensor(sample)
            #self.bad_text[index] = ''.join(sample_txt)

            #self.labels[index] = torch.tensor([1 if ok == bad else 0 for ok, bad in zip(self.ok_samples[index], self.bad_samples[index])])


    def __len__(self):
        return len(self.bad_samples)

 
    def __getitem__(self, idx):
        #maybe move convert to one-hot here, idk
        return {'id': self.IDs[idx],
                'ok_sample': self.ok_samples[idx],
                'bad_sample': self.bad_samples[idx],
                #'ok_sample_one_hot': self.ok_samples_one_hot[idx],
                'bad_sample_one_hot': self.bad_samples_one_hot[idx],
                'ok_text': self.ok_text[idx],
                'bad_text': self.bad_text[idx],
                'ok_sample_index': self.ok_sample_index[idx]}
                #'label': self.labels[idx]}



