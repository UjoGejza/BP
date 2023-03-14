import torch
from numpy import random

#from txt data creates structured data in format:
#ID<index> 
#<sample>
#<sample>
def process_corpus(file:str):
    f = open(file, "r", encoding="UTF-8", errors='ignore')
    o = open(file[:-4]+'_processed_500k.txt', "w", encoding="UTF-8", errors='ignore')
    id = 0
    sample_length = 50
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
            id += 1
            if start_sample==0:
                start_sample = sample_length - len(rest_of_line)
                if start_sample == 0 and sample!=rest_of_line: start_sample = sample_length
            else: start_sample += sample_length
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

def insert_chars(file:str, prob:float):
    f = open(file, "r", encoding="UTF-8")
    o = open(file[:-4]+'-insert.txt', "w", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::3]
    clear = lines[1::3]
    dirty = lines[2::3]
    for idx, id in enumerate(IDs):
        o.write(id)
        clear[idx] = list(clear[idx])
        dirty[idx] = list(dirty[idx])
        for i,character in enumerate(dirty[idx][0:-1]):
            if torch.rand(1).item()<=prob:
                character = chr(int((torch.rand(1).item()*26)) + 97)
                clear[idx].insert(i,'#')
                dirty[idx].insert(i,character)
                clear[idx].pop(len(clear[idx])-2)
                dirty[idx].pop(len(dirty[idx])-2)
        o.write(''.join(clear[idx]))
        o.write(''.join(dirty[idx]))
    f.close()
    o.close()

def add_typos(file:str, prob:float):
    f = open(file, "r", encoding="UTF-8")
    o = open(file[:-4]+'-swap.txt', "w", encoding="UTF-8")
    position = 0
    lines = f.readlines()
    IDs = lines[0::3]
    clear = lines[1::3]
    dirty = lines[2::3]

    for idx, id in enumerate(IDs):
        o.write(id)
        o.write(clear[idx])
        for i,character in enumerate(dirty[idx]):
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
    clear = lines[1::3]
    dirty = lines[2::3]
    for idx, id in enumerate(IDs):
        o.write(id)
        clear[idx] = list(clear[idx])
        dirty[idx] = list(dirty[idx])
        
        error_index = random.randint(50, size=(4))
        error_char = random.randint(low=97, high=123, size=(4))

        for i in range(4):
            if i%4<3:
                dirty[idx][error_index[i]] = chr(error_char[i])
            else:
                dirty[idx].insert(error_index[i], chr(error_char[i]))
                clear[idx].insert(error_index[i], '#')
                clear[idx].pop(len(clear[idx])-2)
                dirty[idx].pop(len(dirty[idx])-2)
        o.write(''.join(clear[idx]))
        o.write(''.join(dirty[idx]))
    f.close()
    o.close()

def new_add_typos_and_insert_chars_and_delete(file:str):#for ctc
    f = open(file, "r", encoding="UTF-8")
    o = open(file[:-4]+'_typos_CTC.txt', "w", encoding="UTF-8")
    lines = f.readlines()
    IDs = lines[0::3]
    clear = lines[1::3]
    dirty = lines[2::3]
    for idx, id in enumerate(IDs):
        o.write(id)
        clear[idx] = list(clear[idx])
        dirty[idx] = list(dirty[idx])
        
        error_index = random.randint(50, size=(5))
        error_char = random.randint(low=97, high=123, size=(5))

        for i in range(5):
            if i%5<3:
                dirty[idx][error_index[i]] = chr(error_char[i])
            if i%5==4:
                dirty[idx].insert(error_index[i], chr(error_char[i]))
                #clear[idx].insert(error_index[i], '#')
                #clear[idx].pop(len(clear[idx])-2)
                dirty[idx].pop(len(dirty[idx])-2)
            if i%5==3:
                dirty[idx].pop(error_index[i])
                dirty[idx].insert(49, ' ')
        o.write(''.join(clear[idx]))
        o.write(''.join(dirty[idx]))
    f.close()
    o.close()


#process_corpus('one-hot_encoding/data/scifi_smaller.txt')
new_add_typos_and_insert_chars_and_delete('one-hot_encoding/data/wiki_all.txt')
#add_typos('one-hot_encoding/data/wiki-1k-train.txt', 0.1)
#add_typos('one-hot_encoding/data/wiki-1k-test.txt', 0.1)
#insert_chars('one-hot_encoding/data/wiki-1k-test.txt', 0.025)
#add_typos('one-hot_encoding/data/wiki-1k-test-insert.txt', 0.05)
#new_add_typos_and_insert_chars('one-hot_encoding/data/wiki_test_30k.txt')
#new_add_typos_and_insert_chars('one-hot_encoding/data/wiki_2M.txt')
#new_add_typos_and_insert_chars('one-hot_encoding/data/wiki_4M.txt')
#new_add_typos_and_insert_chars('one-hot_encoding/data/wiki_all.txt')
#new_add_typos_and_insert_chars('one-hot_encoding/data/scifi_test_30k.txt')
#new_add_typos_and_insert_chars('one-hot_encoding/data/scifi_2M.txt')
#new_add_typos_and_insert_chars('one-hot_encoding/data/scifi_all.txt')



