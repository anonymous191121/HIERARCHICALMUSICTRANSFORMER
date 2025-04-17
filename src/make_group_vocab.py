import os
import json
import operator

def load_txt(path):
    raw_data = []
    for path, directories, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                with open(f'{path}/{file}', 'r') as f:
                    for line in f:
                        raw_data.append(line.strip())                 
    return raw_data

def get_inst_group(raw_data, book_size=8):
    
    inst_group = [[] for _ in range(book_size)]
    
    with open('data/vocabs/inst.json', 'r') as file:
        inst_vocab = json.load(file)
    
    inst_sum = [0] * 133
    for text_seq in raw_data:
        if isinstance(text_seq, str):
            toks = text_seq.split()
        l_toks = len(toks)

        for idx in range(0, l_toks, 4):
            t1, t2, t3, t4 = toks[idx : idx + 4]
    
            if t4[0] == 'x' or t4[0] == 'X' or t4[0] == 'y' or t4 == '<unk>':
                inst_sum[int(inst_vocab[t4])] += 1
                
    sum_dict = {}

    for idx, value in enumerate(inst_sum):
        sum_dict[idx] = value
    
    sum_dict = sorted(sum_dict.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(0, 133, book_size):
        for offset in range(book_size):
            try:
                inst_group[offset].append(sum_dict[i+offset][0])
            except:
                break
    # print(inst_group)
    return inst_group

def get_onset_group(raw_data, book_size=8):
    
    onset_group = [[] for _ in range(book_size)]
    
    with open('data/vocabs/onset.json', 'r') as file:
        onset_vocab = json.load(file)
    
    onset_sum = [0] * 100
    for text_seq in raw_data:
        if isinstance(text_seq, str):
            toks = text_seq.split()
        l_toks = len(toks)

        for idx in range(0, l_toks, 4):
            t1, t2, t3, t4 = toks[idx : idx + 4]
    
            if t1[0] == 'p' or t1[0] == 'P':
                try:
                    onset_sum[int(onset_vocab[t4])] += 1
                except:
                    onset_sum[3] += 1
                
    sum_dict = {}

    for idx, value in enumerate(onset_sum):
        sum_dict[idx] = value
    
    sum_dict = sorted(sum_dict.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(0, 100, book_size):
        for offset in range(book_size):
            try:
                onset_group[offset].append(sum_dict[i+offset][0])
            except:
                break
    # print(onset_group)
    return onset_group
    

def make_inst(raw_data, book_size=8):
    with open('data/vocabs/inst.json', 'r') as file:
        inst_vocab = json.load(file)
        
    inst_group = get_inst_group(raw_data)
    
    
    for B in range(book_size):
        
        inst_book_vocab = dict()
        inst_book_vocab['<pad>'] = 0
        inst_book_vocab['<eos>'] = 1
        inst_book_vocab['<bos>'] = 2
        inst_book_vocab['<unk>'] = 3

        v_idx = 4
        vocab = ''
        
        for text_seq in raw_data:
            if isinstance(text_seq, str):
                toks = text_seq.split()
            l_toks = len(toks)

            for idx in range(0, l_toks, 4):
                t1, t2, t3, t4 = toks[idx : idx + 4]
                
                if (t1[0] == 'm' or t1[0] == 'M'):
                    if vocab not in inst_book_vocab:
                        inst_book_vocab[vocab] = v_idx
                        v_idx += 1
                    vocab = ''
        
                if (t4[0] == 'x' or t4[0] == 'X' or t4[0] == 'y' or t4 == '<unk>') and (int(inst_vocab[t4]) in inst_group[B]):
                    if t4 not in vocab:
                        vocab += t4
        
        tmp = dict()
        for idx, key in enumerate(inst_book_vocab):
            tmp[idx] = key
            
        inst_book_vocab.update(tmp)
        
        with open(f'data/vocabs/book/inst_B{B}.json', 'w') as file:
            json.dump(inst_book_vocab, file, indent=4)
        with open(f'data/vocabs/book/inst_Group.json', 'w') as file:
            json.dump(inst_group, file, indent=4)
            
def make_onset(raw_data, book_size=8):
    with open('data/vocabs/onset.json', 'r') as file:
        onset_vocab = json.load(file)
        
    onset_group = get_onset_group(raw_data)
    
    
    for B in range(book_size):
        
        book_vocab = dict()
        book_vocab['<pad>'] = 0
        book_vocab['<eos>'] = 1
        book_vocab['<bos>'] = 2
        book_vocab['<unk>'] = 3

        v_idx = 4
        vocab = ''
        
        for text_seq in raw_data:
            if isinstance(text_seq, str):
                toks = text_seq.split()
            l_toks = len(toks)

            for idx in range(0, l_toks, 4):
                t1, t2, t3, t4 = toks[idx : idx + 4]
                
                if t1 == 'NT' or (t1[0] == 'm' or t1[0] == 'M'):
                    if vocab not in book_vocab:
                        book_vocab[vocab] = v_idx
                        v_idx += 1
                    vocab = ''
        
                if (t1[0] == 'p' or t1[0] == 'P') and (int(onset_vocab[t1]) in onset_group[B]):
                    if t1 not in vocab:
                        vocab += t1
        
        tmp = dict()
        for idx, key in enumerate(book_vocab):
            tmp[idx] = key
            
        book_vocab.update(tmp)
        
        with open(f'data/vocabs/book/onset_B{B}.json', 'w') as file:
            json.dump(book_vocab, file, indent=4)
        with open(f'data/vocabs/book/onset_Group.json', 'w') as file:
            json.dump(onset_group, file, indent=4)

def make_dur(raw_data):
    DDic = dict()
    DDic['<pad>'] = 0
    DDic['<eos>'] = 1
    DDic['<bos>'] = 2
    DDic['<unk>'] = 3
    ratio = 4
    d_idx = 4
    for text_seq in raw_data:
        if isinstance(text_seq, str):
            toks = text_seq.split()
        l_toks = len(toks)
        D = ''
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]

            if t1[0] == 'p' or t1[0] == 'P':
                if D in DDic.keys():
                    pass
                else:
                    DDic[D] = d_idx
                    d_idx += 1
                D = ''
                
                
            elif t2[0] == 'r':
                D += t2

    tmp = dict()
    for idx, key in enumerate(DDic):
        tmp[idx] = key
        
    DDic.update(tmp)
    with open(f'data/vocabs/dur_group.json', 'w') as file:
        json.dump(DDic, file, indent=4)

def make_group(book_size=8, path='data/raw'):
    raw_data = load_txt(path)
    
    make_inst(raw_data)
    make_onset(raw_data)
    make_dur(raw_data)

if __name__ == '__main__':

    # TODO
    txt_data_path = "data/raw"
    
    make_group(path=txt_data_path)