import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse

from numpy import random

from dataset import MyDataset
from models import ConvLSTMCorrectionBigger, ConvLSTMCorrection, ConvLSTMCorrectionCTC, ConvLSTMCorrectionCTCBigger, ConvLSTMDetection, ConvLSTMDetectionBigger
import ansi_print

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='detection')
    parser.add_argument('-model_file', type=str, default='one-hot_encoding/results/ConvLSTMDetectionBigger2_3/ConvLSTMDetectionBigger2_3.pt')
    parser.add_argument('-test_file', type=str, default='one-hot_encoding/data/examplerandom_length.txt')
    parser.add_argument('-output_file', type=str, default='test_output.txt')
    return parser.parse_args()

args = parseargs()
mode = args.mode
model_file = args.model_file
test_file = args.test_file
output_file = args.output_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USING: {device}')

test_data = MyDataset(test_file)
test_data_loader = DataLoader(test_data, shuffle=False)

def correction(data_loader):
    model = ConvLSTMCorrectionBigger()
    model = torch.load(model_file)
    model.to(device)
    model.eval()
    #for name, param in model.state_dict().items():
    #    print(name, param.size())
    alphabet = test_data.charlist_base
    
    for item in data_loader:
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        #item['ok_sample'] = item['ok_sample'].to(device)
        outputs = model(item['bad_sample_one_hot'])
        outputs = outputs[0]
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        #item['label'] = item['label'][0]
        output_text_list = list(pred)

        for index, p in enumerate(pred):
            output_text_list[index] = alphabet[p]

        output_text = ''.join(output_text_list)
        try:
            f.write(str(item['id'][0].item())+'\n')#id
            f.write(item['ok_text'][0]+'\n')#ground truth
            f.write(item['bad_text'][0]+'\n')#input
            f.write(output_text+'\n')#output
        except:
            print('error printing example - prob encoding')

def CTC(data_loader):
    model = ConvLSTMCorrectionCTC()
    model = torch.load(model_file)
    model.to(device)
    model.eval()
    #for name, param in model.state_dict().items():
    #    print(name, param.size())
    alphabet = test_data.charlist_extra_ctc

    for item in data_loader:
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        #item['ok_sample'] = item['ok_sample'].to(device)
        outputs = model(item['bad_sample_one_hot'])
        outputs = outputs[0]
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        output_list = list(pred)

        #remove all chains of the same character longer than 1 (aa -> a)
        trimmed_output_list_str = []
        for out in output_list:
            if len(trimmed_output_list_str) == 0 or alphabet.index(trimmed_output_list_str[-1]) != out:
                trimmed_output_list_str.append(alphabet[out])

        #remove "blank" (~) 
        trimmed_output_list_txt_no_blank = [x for x in trimmed_output_list_str if x!= '~']
        final_str = ''.join(trimmed_output_list_txt_no_blank)
   
        try:
            f.write(str(item['id'].item())+'\n')#id
            f.write(item['ok_text'][0]+'\n')#ground truth
            f.write(item['bad_text'][0]+'\n')#input
            f.write(final_str+'\n')#output
        except:
            print('error printing example - prob encoding')

def detection(data_loader):
    model = ConvLSTMDetectionBigger()
    model = torch.load(model_file)
    model.to(device)
    model.eval()
    #for name, param in model.state_dict().items():
    #    print(name, param.size())

    for item in data_loader:
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].transpose(1, 2)
        item['bad_sample_one_hot'] = item['bad_sample_one_hot'].to(device)
        #item['label'] = item['label'].to(device)
        outputs = model(item['bad_sample_one_hot'])
        outputs = torch.squeeze(outputs)
        #outputs = outputs[0]
        item['label'] = item['label'][0]
        outputs = [1 if out>0.6 else 0 for out in outputs]

        f.write(str(item['id'].item())+'\n')#id
        f.write(item['ok_text'][0]+'\n')
        f.write(item['bad_text'][0]+'\n')#input
        label = ['1' if l == 1 else '0' for l in item['label']]
        out = ['1' if o == 1 else '0' for o in outputs]
        f.write(''.join(label)+'\n')#ground truth
        f.write(''.join(out)+'\n')#output


 
with open(output_file, "w", encoding="UTF-8", errors='ignore') as f:
    if mode == 'correction':
        correction(test_data_loader)
    if mode == 'ctc' or mode == 'CTC':
        CTC(test_data_loader)
    if mode == 'detection':
        detection(test_data_loader)
