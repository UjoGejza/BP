# inference.py
# Author: Sebastián Chupáč
# This script is used to perform inference and save results to text file
#which will be processed by the eval.py script to compute statistics

import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse

from numpy import random

#import from dataset_pad if using CTC
from dataset import MyDataset
from models import ConvLSTMCorrectionBigger, ConvLSTMCorrection, ConvLSTMCorrectionCTC, ConvLSTMCorrectionCTCBigger, ConvLSTMDetection, ConvLSTMDetectionBigger, ConvLSTMCorrectionCTCBiggerPad, UNetCorrectionCTCBiggerPad, ConvLSTMCorrectionCTCBiggerPad2x

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='detection')#choose mode: correction, detection, ctc_pad, basically which function will be called
    parser.add_argument('-model_file', type=str, default='models/ConvLSTMDetection3/ConvLSTMDetection3.pt')#model to perform inference on
    parser.add_argument('-test_file', type=str, default='data/scifi_test_test_1k_typos_2M.txt')#input test file
    parser.add_argument('-output_file', type=str, default='models/ConvLSTMDetection3/ConvLSTMDetection3.txt')#inference results will be saved here
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

#test text correction
def correction(data_loader):
    model = ConvLSTMCorrectionBigger()
    model = torch.load(model_file)
    model.to(device)
    model.eval()
    #for name, param in model.state_dict().items():
    #    print(name, param.size())
    alphabet = test_data.charlist_base

    for item in data_loader:
        item['sample_one_hot'] = item['sample_one_hot'].transpose(1, 2)
        item['sample_one_hot'] = item['sample_one_hot'].to(device)
        outputs = model(item['sample_one_hot'])
        outputs = outputs[0]
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        output_text_list = list(pred)

        for index, p in enumerate(pred):
            output_text_list[index] = alphabet[p]

        output_text = ''.join(output_text_list)
        try:
            f.write(str(item['id'][0].item())+'\n')#id
            f.write(item['label_text'][0]+'\n')#ground truth
            f.write(item['sample_text'][0]+'\n')#input
            f.write(output_text+'\n')#output
        except:
            print('error printing example - prob encoding')

#not recommended to use CTC without padding
def CTC(data_loader):
    model = ConvLSTMCorrectionCTC()
    model = torch.load(model_file)
    model.to(device)
    model.eval()
    blank = 'ѧ'#old CTC models without pad used ~ as blank
    alphabet = test_data.charlist_extra_ctc

    for item in data_loader:
        item['sample_one_hot'] = item['sample_one_hot'].transpose(1, 2)
        item['sample_one_hot'] = item['sample_one_hot'].to(device)
        outputs = model(item['sample_one_hot'])
        outputs = outputs[0]
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        output_list = list(pred)

        #remove all chains of the same character longer than 1 (aa -> a)
        trimmed_output_list_str = []
        for out in output_list:
            if len(trimmed_output_list_str) == 0 or alphabet.index(trimmed_output_list_str[-1]) != out:
                trimmed_output_list_str.append(alphabet[out])

        #remove "blank"
        trimmed_output_list_txt_no_blank = [x for x in trimmed_output_list_str if x!= blank]
        final_str = ''.join(trimmed_output_list_txt_no_blank)
   
        try:
            f.write(str(item['id'].item())+'\n')#id
            f.write(item['label_text'][0]+'\n')#ground truth
            f.write(item['sample_text'][0]+'\n')#input
            f.write(final_str+'\n')#output
        except:
            print('error printing example - prob encoding')

#test text correction with CTC 
def CTC_pad(data_loader):
    pad = 'Є'
    blank = 'ѧ'
    model = ConvLSTMCorrectionCTCBigger()
    model = torch.load(model_file)
    model.to(device)
    model.eval()
    alphabet = test_data.charlist_extra_ctc

    for item in data_loader:
        item['sample_one_hot'] = item['sample_one_hot'].transpose(1, 2)
        item['sample_one_hot'] = item['sample_one_hot'].to(device)
        outputs = model(item['sample_one_hot'])
        outputs = outputs[0]
        pred = torch.squeeze(torch.topk(outputs, 1, dim=0, sorted=False).indices)
        output_list = list(pred)

        #remove all chains of the same character longer than 1 (aa -> a)
        trimmed_output_list_str = []
        raw_output = []
        for out in output_list:
            raw_output.append(alphabet[out])
            if len(trimmed_output_list_str) == 0 or alphabet.index(trimmed_output_list_str[-1]) != out:
                trimmed_output_list_str.append(alphabet[out])

        #remove "blank"
        trimmed_output_list_txt_no_blank = [x for x in trimmed_output_list_str if x!= blank]
        final_str = ''.join(trimmed_output_list_txt_no_blank)
        raw_output_str = ''.join(raw_output)
        try:
            f.write(str(item['id'].item())+'\n')#id
            f.write(item['label_text'][0][:item['label_length']]+'\n')#ground truth
            f.write(item['sample_text'][0][3:item['sample_text'][0].find(pad, 33)]+'\n')#input
            f.write(raw_output_str+'\n')
            f.write(final_str+'\n')#output
        except:
            print('error printing example - prob encoding')

#test typo detection
def detection(data_loader):
    model = ConvLSTMDetection()
    model = torch.load(model_file)
    model.to(device)
    model.eval()
    #for name, param in model.state_dict().items():
    #    print(name, param.size())

    for item in data_loader:
        item['sample_one_hot'] = item['sample_one_hot'].transpose(1, 2)
        item['sample_one_hot'] = item['sample_one_hot'].to(device)
        #item['bi_class'] = item['bi_class'].to(device)
        outputs = model(item['sample_one_hot'])
        outputs = torch.squeeze(outputs)
        #outputs = outputs[0]
        item['bi_class'] = item['bi_class'][0]
        outputs = [1 if out>0.6 else 0 for out in outputs]

        f.write(str(item['id'].item())+'\n')#id
        f.write(item['label_text'][0]+'\n')
        f.write(item['sample_text'][0]+'\n')#input
        bi_class = ['1' if l == 1 else '0' for l in item['bi_class']]
        out = ['1' if o == 1 else '0' for o in outputs]
        f.write(''.join(bi_class)+'\n')#ground truth
        f.write(''.join(out)+'\n')#output


 
with open(output_file, "w", encoding="UTF-8", errors='ignore') as f:
    if mode == 'correction':
        correction(test_data_loader)
    if mode == 'ctc' or mode == 'CTC':
        CTC(test_data_loader)
    if mode == 'ctc_pad' or mode == 'CTC_pad':
        CTC_pad(test_data_loader)
    if mode == 'detection':
        detection(test_data_loader)
