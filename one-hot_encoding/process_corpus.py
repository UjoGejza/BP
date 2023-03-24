import torch
import numpy as np
from numpy import random

#from txt data creates structured data in format:
#ID<index> 
#<sample>
#<sample>
def process_corpus(file:str):
    f = open(file, "r", encoding="UTF-8", errors='ignore')
    o = open(file[:-4]+'random_length.txt', "w", encoding="UTF-8", errors='ignore')
    id = 0
    sample_length = 50
    old_sample_length = 50
    rest_of_line = ''
    start_sample = 0
    for line in f:
        sample = line[start_sample:sample_length-len(rest_of_line)]
        sample = rest_of_line + sample
        #if len(sample) != sample_length: break
        while sample.find('\n')==-1:
            o.write(f'ID{id}\n')
            o.write(sample+'\n')
            o.write(sample+'\n')
            old_sample_length = sample_length
            sample_length = random.randint(low=40, high=60)#this generates random length of line/sample/input
            id += 1
            if start_sample==0:
                start_sample = old_sample_length - len(rest_of_line)
                if start_sample == 0 and sample!=rest_of_line: start_sample = old_sample_length
            else: start_sample += old_sample_length
            sample = line[start_sample:(start_sample+sample_length)]
            if len(sample) != sample_length: break
            if id > 2_000_000:
                break
        rest_of_line = sample[:sample.find('\n')]+' '
        start_sample = 0
        if id > 500_000:
            break
    f.close()
    o.close()


#=====these are outdated functions, keep for reference=======
def insert_chars(file:str, prob:float):
    f = open(file, "r", encoding="UTF-8")
    o = open(file[:-4]+'-insert.txt', "w", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::3]
    ok = lines[1::3]
    bad = lines[2::3]
    for idx, id in enumerate(IDs):
        o.write(id)
        ok[idx] = list(ok[idx])
        bad[idx] = list(bad[idx])
        for i,character in enumerate(bad[idx][0:-1]):
            if torch.rand(1).item()<=prob:
                character = chr(int((torch.rand(1).item()*26)) + 97)
                ok[idx].insert(i,'#')
                bad[idx].insert(i,character)
                ok[idx].pop(len(ok[idx])-2)
                bad[idx].pop(len(bad[idx])-2)
        o.write(''.join(ok[idx]))
        o.write(''.join(bad[idx]))
    f.close()
    o.close()

def add_typos(file:str, prob:float):
    f = open(file, "r", encoding="UTF-8")
    o = open(file[:-4]+'-swap.txt', "w", encoding="UTF-8")
    position = 0
    lines = f.readlines()
    IDs = lines[0::3]
    ok = lines[1::3]
    bad = lines[2::3]

    for idx, id in enumerate(IDs):
        o.write(id)
        o.write(ok[idx])
        for i,character in enumerate(bad[idx]):
            if character !='\n':
                if torch.rand(1).item()<=prob:
                    character = chr(int((torch.rand(1).item()*26)) + 97)
            o.write(character)
        position = position+i+1                    
    f.close()
    o.close()

def new_add_typos_and_insert_chars(file:str):
    f = open(file, "r", encoding="UTF-8")
    o = open(file[:-4]+'_typos.txt', "w", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::3]
    ok = lines[1::3]
    bad = lines[2::3]
    for idx, id in enumerate(IDs):
        o.write(id)
        ok[idx] = list(ok[idx])
        bad[idx] = list(bad[idx])
        
        error_index = random.randint(50, size=(4))
        error_char = random.randint(low=97, high=123, size=(4))

        for i in range(4):
            if i%4<3:
                bad[idx][error_index[i]] = chr(error_char[i])
            else:
                bad[idx].insert(error_index[i], chr(error_char[i]))
                ok[idx].insert(error_index[i], '#')
                ok[idx].pop(len(ok[idx])-2)
                bad[idx].pop(len(bad[idx])-2)
        o.write(''.join(ok[idx]))
        o.write(''.join(bad[idx]))
    f.close()
    o.close()

def new_add_typos_ctc(file:str):#for ctc
    f = open(file, "r", encoding="UTF-8")
    o = open(file[:-4]+'_typos_CTC.txt', "w", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::3]
    ok = lines[1::3]
    bad = lines[2::3]
    for idx, id in enumerate(IDs):
        o.write(id)
        ok[idx] = list(ok[idx])
        bad[idx] = list(bad[idx])
        
        error_index = random.randint(50, size=(5))
        error_char = random.randint(low=97, high=123, size=(5))

        for i in range(5):
            if i%5<3:
                bad[idx][error_index[i]] = chr(error_char[i])
            if i%5==4:
                bad[idx].insert(error_index[i], chr(error_char[i]))
                #ok[idx].insert(error_index[i], '#')
                #ok[idx].pop(len(ok[idx])-2)
                bad[idx].pop(len(bad[idx])-2)
            if i%5==3:
                bad[idx].pop(error_index[i])
                bad[idx].insert(49, ' ')
        o.write(''.join(ok[idx]))
        o.write(''.join(bad[idx]))
    f.close()
    o.close()
#-------------------------------------------------

#this is the final function for generating typos allowing random frequency and all types
#all above typos functions are outdated
def new_add_typos_RF(file:str):
    f = open(file, "r", encoding="UTF-8")
    o = open(file[:-4]+'_typosRF3_CTC.txt', "w", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::3]
    ok = lines[1::3]
    bad = lines[2::3]
    for idx, id in enumerate(IDs):
        o.write(id)
        ok[idx] = list(ok[idx])
        bad[idx] = list(bad[idx])

        #typo frquency:
        #generate normal(5,2), round, to int, clip between 0 - 8
        typo_count = np.clip(int(np.round(random.normal(5, 2))), 0, 8 )
        
        typo_index = random.randint(50, size=(typo_count))
        typo_char = random.randint(low=97, high=123, size=(typo_count))
        
        #typo type generation: 1=swap, 2=swap+insert, 3=swap+insert+delete
        typo_type = random.choice(3, typo_count)
        typo_type.sort()
        typo_type = typo_type[::-1]#deleting first loses less information from end of samples

        for i in range(typo_count):
            if typo_index[i]>=len(bad[idx])-1: continue
            if typo_type[i] == 0:#swap
                bad[idx][typo_index[i]] = chr(typo_char[i])
            if typo_type[i] == 1:#insert
                bad[idx].insert(typo_index[i], chr(typo_char[i]))
                #if using insert with no CTC, uncomment these 2 lines
                #ok[idx].insert(typo_index[i], '#')
                #ok[idx].pop(len(ok[idx])-2)
                bad[idx].pop(len(bad[idx])-2)#keeps bad the same and original length
            if typo_type[i] == 2:#delete (only use with CTC models)
                bad[idx].pop(typo_index[i])
                bad[idx].insert(len(bad[idx])-1, ' ')#keeps bad the same and original length
        o.write(''.join(ok[idx]))
        o.write(''.join(bad[idx]))
    f.close()
    o.close()


#process_corpus('one-hot_encoding/data/example.txt')
new_add_typos_RF('one-hot_encoding/data/scifi_all.txt')




