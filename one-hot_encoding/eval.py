# eval.py
# Author: Sebastián Chupáč
# This script is used to compute stats from inference files

import torch
import argparse
import Levenshtein


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='file')#choose mode: correction, detection, CTC, file, basically which function will be called
    parser.add_argument('-file', type=str, default='one-hot_encoding/data_news/news_test_RLOAWP2_2k_typosRF(6,2)_CTC.txt')#inference file to compute stats from
    return parser.parse_args()

args = parseargs()
mode = args.mode
file = args.file

#compute stats for text correction
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
    acc_abs = (corrected_typos-created_typos)/(all_typos)
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Corrected typos: {corrected_typos} / {all_typos}, {acc_corrected*100:.2f}%')
    print(f'Typos created: {created_typos}, absolute acc: {acc_abs*100:.2f}%')
    print(f'Average edit distance: {sum_distance/len(IDs):.2f}')
    print(f'Average indel similarity: {sum_ratio/len(IDs):.4f}') #1 - normalized_distance


#compute stats for correction with CTC
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
    swap_count = 0
    delete_count = 0
    insert_count = 0
    input_swap_count = 0
    input_delete_count = 0
    input_insert_count = 0

    for sample_i,_ in enumerate(IDs):
        edit_distance_input = Levenshtein.distance(inputs[sample_i][:-1], ground_truths[sample_i][:-1])
        edit_distance_output = Levenshtein.distance(outputs[sample_i][:-1], ground_truths[sample_i][:-1])
        operations = Levenshtein.editops(ground_truths[sample_i][:-1], outputs[sample_i][:-1])
        input_operations = Levenshtein.editops(ground_truths[sample_i][:-1], inputs[sample_i][:-1])
        for op in operations:
            if op[0]=='replace': swap_count+=1
            if op[0]=='delete': delete_count+=1
            if op[0]=='insert': insert_count+=1
        for input_op in input_operations:
            if input_op[0]=='replace': input_swap_count+=1
            if input_op[0]=='delete': input_delete_count+=1
            if input_op[0]=='insert': input_insert_count+=1
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
    acc_abs = fixes/(all_typos)#(IvG-PvG)/IvG
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Fixed: {fixes}/{all}')
    print(f'Corrected typos: _____ / _____, _____%')
    print(f'Typos created: _____, absolute acc: {acc_abs*100:.2f}%')
    print(f'Average edit distance: {sum_distance/len(IDs):.2f}')
    print(f'Average indel similarity: {sum_ratio/len(IDs):.4f}') #1 - normalized_distance
    
    print(f'fixed swaps: {(1-(swap_count/input_swap_count))*100:.2f}%')
    print(f'fixed inserts: {(1-(insert_count/input_insert_count))*100:.2f}%')
    print(f'fixed deletes: {(1-(delete_count/input_delete_count))*100:.2f}%')
    
    print(f'number of swaps: {swap_count}, ratio: {swap_count/sum_distance}')
    print(f'number of inserts: {insert_count}, ratio: {insert_count/sum_distance}')
    print(f'number of deletes: {delete_count}, ratio: {delete_count/sum_distance}')

    print(f'number of original swaps: {input_swap_count}, ratio: {input_swap_count/all_typos}')
    print(f'number of original inserts: {input_insert_count}, ratio: {input_insert_count/all_typos}')
    print(f'number of original deletes: {input_delete_count}, ratio: {input_delete_count/all_typos}')
    
#compute stats for typo detection
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
    ABS = (TN-FN)/(TN+FP) #absolute acc
    print(f'Accuracy: {ACC_CM*100:.2f}%')
    print(f'Balanced accuracy: {BA*100:.2f}%')
    print(f'Recall: {TPR:.4f}, TNR: {TNR:.4f}, Precision: {PPV:.4f}, F1: {F1:.4f}')
    print(f'Corrected typos: {TNR*100:.2f}%')
    print(f'Absolute acc ((corrected-created)/(all)): {ABS*100:.2f}%')

#compute stats for input file
def eval_file():
    f = open(file, "r", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::3]
    ground_truths = lines[1::3]
    inputs = lines[2::3]

    f.close()
    
    correct_chars = 0
    all_chars= 0
    all_typos = 0
    #created_typos = 0
    sum_ratio = 0
    sample_length_sum = 0
    pad = 'Є'
    swap_count = 0
    delete_count = 0
    insert_count = 0

    for sample_i,_ in enumerate(IDs):
        input = inputs[sample_i][3:inputs[sample_i].find(pad, 30)]
        gt = ground_truths[sample_i][:ground_truths[sample_i].find(pad, 30)]
        edit_distance_input = Levenshtein.distance(input, gt)
        operations = Levenshtein.editops(input, gt)
        for op in operations:
            if op[0]=='replace': swap_count+=1
            if op[0]=='delete': delete_count+=1
            if op[0]=='insert': insert_count+=1
        all_typos += edit_distance_input
        sample_length_sum += len(input)
        indel_ratio = Levenshtein.ratio(input, gt)
        sum_ratio += indel_ratio
        all_chars += len(input)
        correct_chars += (len(input) - edit_distance_input)
        
    acc = correct_chars/all_chars
    print(f'Samples: {len(IDs)}')
    print(f'Average sample length: {sample_length_sum/len(IDs):.2f}')
    print(f'Accuracy: {acc*100:.2f}%, correct/bad: {correct_chars}/{all_chars}')
    print(f'Typos: {all_typos}')
    print(f'Average edit distance (typos per sample): {all_typos/len(IDs):.2f}')
    print(f'Average indel similarity: {sum_ratio/len(IDs):.4f}') #1 - normalized_distance
    print(f'number of swaps: {swap_count}, ratio: {swap_count/all_typos}')
    print(f'number of inserts: {insert_count}, ratio: {insert_count/all_typos}')
    print(f'number of deletes: {delete_count}, ratio: {delete_count/all_typos}')

if mode == 'correction':
    eval_correction()
if mode == 'ctc' or mode == 'CTC':
    eval_CTC()
if mode == 'detection':
    eval_detection()
if mode == 'file':
    eval_file()