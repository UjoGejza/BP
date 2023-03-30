import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import Levenshtein
from numpy import random

from dataset import MyDataset
from models import ConvLSTMCorrectionBigger, ConvLSTMCorrection, ConvLSTMCorrectionCTC, ConvLSTMCorrectionCTCBigger, ConvLSTMDetection, ConvLSTMDetectionBigger
import ansi_print

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='ctc')
    parser.add_argument('-file', type=str, default='one-hot_encoding/eval/ConvLSTMCorrectionCTC_Bigger_3_wiki.txt')
    return parser.parse_args()

args = parseargs()
mode = args.mode
file = args.file

def eval_correction():
    f = open(file, "r", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::4]
    ground_truths = lines[1::4]
    inputs = lines[2::4]
    outputs = lines[3::4]

    f.close()

    correct = 0
    all= 0
    corrected_typos = 0
    all_typos = 0
    created_typos = 0
    sum_distance = 0
    sum_ratio = 0

    for sample_i,_ in enumerate(IDs):
        edit_distance = Levenshtein.distance(outputs[sample_i][:-1], ground_truths[sample_i][:-1])
        indel_ratio = Levenshtein.ratio(outputs[sample_i][:-1], ground_truths[sample_i][:-1])
        sum_distance += edit_distance
        sum_ratio += indel_ratio
        for index, out in enumerate(outputs[sample_i][:-1]):
            all += 1
            if ground_truths[sample_i][index] == out: correct += 1
            if ground_truths[sample_i][index]!=inputs[sample_i][index]:
                if out == ground_truths[sample_i][index]: corrected_typos += 1
                all_typos += 1
            else:
                if out != ground_truths[sample_i][index]: created_typos += 1
    acc = correct/all
    acc_corrected = corrected_typos/all_typos
    acc_abs = corrected_typos/(all_typos+created_typos)
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Corrected typos: {corrected_typos} / {all_typos}, {acc_corrected*100:.2f}%')
    print(f'Typos created: {created_typos}, final acc: {acc_abs*100:.2f}%')
    print(f'Average edit distance: {sum_distance/len(IDs):.2f}')
    print(f'Average indel similarity: {sum_ratio/len(IDs):.4f}') #1 - normalized_distance



def eval_CTC():
    f = open(file, "r", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::4]
    ground_truths = lines[1::4]
    inputs = lines[2::4]
    outputs = lines[3::4]

    f.close()
    
    correct = 0
    all= 0
    fixes = 0
    all_typos = 0
    #created_typos = 0
    sum_distance = 0
    sum_ratio = 0

    for sample_i,_ in enumerate(IDs):
        edit_distance_input = Levenshtein.distance(inputs[sample_i][:-1], ground_truths[sample_i][:-1])
        edit_distance_output = Levenshtein.distance(outputs[sample_i][:-1], ground_truths[sample_i][:-1])
        fixes_sample = edit_distance_input - edit_distance_output
        fixes += fixes_sample
        all_typos += edit_distance_input
        indel_ratio = Levenshtein.ratio(outputs[sample_i][:-1], ground_truths[sample_i][:-1])
        sum_distance += edit_distance_output
        sum_ratio += indel_ratio
        all += len(outputs[sample_i][:-1])
        correct += (len(outputs[sample_i][:-1]) - edit_distance_output)
        
    acc = correct/all
    #acc_corrected = corrected_typos/all_typos
    acc_abs = fixes/(all_typos)
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Fixed: {fixes}/{all}')
    print(f'Corrected typos: _____ / _____, _____%')
    print(f'Typos created: _____, final acc: {acc_abs*100:.2f}%')
    print(f'Average edit distance: {sum_distance/len(IDs):.2f}')
    print(f'Average indel similarity: {sum_ratio/len(IDs):.4f}') #1 - normalized_distance

def eval_detection():
    f = open(file, "r", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::5]
    ok_lines = lines[1::5]
    inputs = lines[2::5]
    ground_truths = lines[3::5]
    outputs = lines[4::5]

    f.close()

    TP, FP, TN, FN, TPR, PPV, F1, ACC_CM, TNR, BA, ABS = 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    confusion_matrix = torch.rand(2,2)
    for sample_i,_ in enumerate(IDs):
        for index, out in enumerate(outputs[sample_i][:-1]):
            if ground_truths[sample_i][index] == '1' and out == '1': TP +=1
            elif ground_truths[sample_i][index] == '0' and out == '0': TN +=1
            elif ground_truths[sample_i][index] == '1' and out == '0': FN +=1
            elif ground_truths[sample_i][index] == '0' and out == '1': FP +=1


    confusion_matrix[0][0] = TP
    confusion_matrix[0][1] = FN
    confusion_matrix[1][0] = FP
    confusion_matrix[1][1] = TN

    print(confusion_matrix.numpy())
    TPR = TP/(TP+FN) #sensitivity, recall, hit rate, or true positive rate (TPR)
    TNR = TN/(TN+FP) #specificity, selectivity or true negative rate (TNR)
    PPV = TP/(TP+FP) #precision or positive predictive value (PPV)
    F1 = 2 * (PPV * TPR)/(PPV + TPR) #F1 score is the harmonic mean of precision and sensitivity:
    ACC_CM = (TP + TN)/(TP + TN + FP + FN) #accuracy
    BA = (TPR + TNR)/2 #balanced accuracy
    ABS = TN/(TN+FP+FN) #absolute acc
    print(f'Accuracy: {ACC_CM*100:.2f}%')
    print(f'Balanced accuracy: {BA*100:.2f}%')
    print(f'Recall: {TPR:.4f}, TNR: {TNR:.4f}, Precision: {PPV:.4f}, F1: {F1:.4f}')
    print(f'Corrected typos: {TNR*100:.2f}%')
    print(f'Absolute acc (corrected/(all+created)): {ABS*100:.2f}%')

if mode == 'correction':
    eval_correction()
if mode == 'ctc' or mode == 'CTC':
    eval_CTC()
if mode == 'detection':
    eval_detection()