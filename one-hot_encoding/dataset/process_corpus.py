import torch

#from txt data creates structured data in format:
#ID<index> <sample>
#<sample>
def process_corpus(file:str):
    f = open(file, "r", encoding="utf8")
    o = open('one-hot_encoding/dataset/corpus_processed.txt', "w")
    id = 0
    sample_length = 50
    rest_of_line = ''
    start_sample = 0
    for line in f:
        sample = line[start_sample:sample_length-len(rest_of_line)]
        sample = rest_of_line + sample
        if len(sample) != sample_length: break
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
        rest_of_line = sample[:sample.find('\n')]+' '
        start_sample = 0
    f.close()
    o.close()

def add_typos(file:str, prob:float):
    f = open(file, "r")
    o = open('one-hot_encoding/dataset/corpus_processed_with_typos.txt', "w")
    position = 0
    lines = f.readlines()
    IDs = lines[0::3]
    clear = lines[1::3]
    dirty = lines[2::3]

    for idx, id in enumerate(IDs):
        o.write(id)
        o.write(clear[idx])
        for i,character in enumerate(dirty[idx]):
            if ((character>='A'<='Z') or (character>='a'<='z') ):
                if torch.rand(1).item()<=prob:
                    character = chr(int((torch.rand(1).item()*26)) + 97)
            o.write(character)
        position = position+i+1                    
    f.close()
    o.close()

process_corpus('one-hot_encoding/dataset/corpus.txt')
add_typos('one-hot_encoding/dataset/corpus_processed.txt', 0.05)



