import torch
from torch import nn
from torch.utils.data import DataLoader

from numpy import random

from dataset import MyDataset
from models import ConvLSTMCorrection
import ansi_print

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USING: {device}')

testing_test_data = MyDataset('one-hot_encoding/data/wiki_train_test_1k_typos.txt')
testing_test_data_loader = DataLoader(testing_test_data, shuffle=True)

alphabet = testing_test_data.charlist

model = ConvLSTMCorrection()
model = torch.load('one-hot_encoding/data/ConvLSTMCorrectionWiki.pt')
model.to(device)
model.eval()

def test(data_loader):
    correct = 0
    all= 0
    corrected_typos = 0
    all_typos = 0
    created_typos = 0

    for i, item in enumerate(data_loader):
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        item['ok_sample'] = item['ok_sample'].to(device)
        outputs = model(item['bad_sample_one_hot'])
        outputs = outputs[0]
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        item['label'] = item['label'][0]
        item['ok_sample'] = item['ok_sample'][0]
        output_text_list = list(pred)

        for index, p in enumerate(pred):
            if item['ok_sample'][index] == p: correct+=1
            if item['label'][index] == 0:
                all_typos += 1
                if item['ok_sample'][index] == p: corrected_typos+=1
            else:
                if item['ok_sample'][index] != p: created_typos+=1
            if i>data_loader.__len__()-6:
                output_text_list[index] = alphabet[p]
            all +=1

        if i>data_loader.__len__()-6:
            output_text = ''.join(output_text_list)
            ansi_print.a_print(item['bad_text'][0], item['ok_text'][0], 'yellow')
            ansi_print.a_print(output_text, item['ok_text'][0], 'red')

    acc = correct/all
    acc_corrected = corrected_typos/all_typos
    acc_corrected_created = corrected_typos/(all_typos+created_typos)
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Corrected typos: {corrected_typos} / {all_typos}, {ansi_print.colors.GREEN}{acc_corrected*100:.2f}%{ansi_print.colors.RESET}')
    print(f'Typos created: {created_typos}, final acc: {ansi_print.colors.GREEN}{acc_corrected_created*100:.2f}%{ansi_print.colors.RESET}')

test(testing_test_data_loader)