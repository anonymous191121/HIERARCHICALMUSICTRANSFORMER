import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class OnsetBook(Dataset):
    def __init__(self, data, booksize=8):
        super().__init__()
        self.data = data
        self.booksize = booksize
        self.onset_book = []
        
        chord_vocab_path = 'data/vocabs/chord.json'
        onset_vocab_path = 'data/vocabs/onset.json'
        onset_group_path = 'data/vocabs/book/onset_Group.json'
        inst_vocab_path = 'data/vocabs/inst.json'
        pitch_vocab_path = 'data/vocabs/pitch.json'
        dur_vocab_path = 'data/vocabs/dur.json'
        with open(chord_vocab_path, 'r') as file:
            self.chord_vocab = json.load(file)
        with open(onset_group_path, 'r') as file:
            self.onset_group = json.load(file)
        with open(onset_vocab_path, 'r') as file:
            self.onset_vocab = json.load(file)
            
        for i in range(booksize):
            with open(f'data/vocabs/book/onset_B{i}.json', 'r') as file:
                vocab = json.load(file)
            self.onset_book.append(vocab)
            
        with open(inst_vocab_path, 'r') as file:
            self.inst_vocab = json.load(file)
        with open(pitch_vocab_path, 'r') as file:
            self.pitch_vocab = json.load(file)
        with open(dur_vocab_path, 'r') as file:
            self.dur_vocab = json.load(file)
        pit2alphabet = ['C', 'd', 'D', 'e', 'E', 'F', 'g', 'G', 'a', 'A', 'b', 'B']
        self.chord_proj = {'C':1, 'd':2, 'D':3, 'e':4, 'E':5, 'F':6, 'g':7, 'G':8, 'a':9, 'A':10, 'b':11, 'B':12, 'N':13}
        self.char2pit = {x: id for id, x in enumerate(pit2alphabet)}
        
    def __len__(self):
        return len(self.data)
    
    def ispitch(self, x):  # judge if a event str is a pitch (CO - B9)
        return len(x) == 2 and x[0] in self.char2pit and (x[1] == 'O' or x[1].isdigit())

    def __getitem__(self, idx):
        text_seq = self.data[idx]
        
        if isinstance(text_seq, str):
            toks = text_seq.split()
            
        l_toks = len(toks)
        ratio = 4
        
        P = ['', '', '', '', '', '', '', '']
        
        chord_list = []
        measure = 0
        total_len = 1

        cur_inst = -1
        
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
                
            if (t1[0] == 'm' or t1[0] == 'M'):
                total_len += 1
            elif t1[0] == 'H':
                chord_list.append(self.chord_vocab[t1])
        
        chord_tensor = chord_list
        target_chord_tensor = [0,2] + chord_tensor[:766] + [1,0]

        measure_book_container = []
        book_container = []
        book_onset_info = [0]*16
        
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
            
            if measure == 766:
                break
                
            if (t1[0] == 'm' or t1[0] == 'M'):
                if cur_inst != -1:
                    for i in range(self.booksize):
                        try:
                            book_onset_info[i] = self.onset_book[i][P[i]]
                        except:
                            book_onset_info[i] = 3
                    book_onset_info[8] = cur_inst
                    book_onset_info[9] = int((measure/total_len)*24)
                    book_onset_info[10] = measure
                    
                    book_onset_info[11] = target_chord_tensor[measure-1]
                    book_onset_info[12] = target_chord_tensor[measure]
                    book_onset_info[13] = target_chord_tensor[measure+1]
                    book_onset_info[14] = target_chord_tensor[measure+2]
                    book_onset_info[15] = target_chord_tensor[measure+3]
                    
                    book_container.append(book_onset_info)
                
                measure += 1
                book_onset_info = [0]*16
                cur_inst = -1
                P = ['', '', '', '', '', '', '', '']
                
                measure_book_container.append(book_container)
                book_container = []
                
            elif t1 == 'NT':
                if cur_inst != -1:
                    for i in range(self.booksize):
                        try:
                            book_onset_info[i] = self.onset_book[i][P[i]]
                        except:
                            book_onset_info[i] = 3
                    book_onset_info[8] = cur_inst
                    book_onset_info[9] = int((measure/total_len)*24)
                    book_onset_info[10] = measure
                    
                    book_onset_info[11] = target_chord_tensor[measure-1]
                    book_onset_info[12] = target_chord_tensor[measure]
                    book_onset_info[13] = target_chord_tensor[measure+1]
                    book_onset_info[14] = target_chord_tensor[measure+2]
                    book_onset_info[15] = target_chord_tensor[measure+3]
                    
                    book_container.append(book_onset_info)
                
                book_onset_info = [0]*16
                cur_inst = -1
                P = ['', '', '', '', '', '', '', '']

                    
            elif t1[0] == 'p' or t1[0] == 'P':
                if t1 == 'PA':
                    continue
                
                if int(self.onset_vocab[t1]) in self.onset_group[0]:
                    P[0] += t1
                elif int(self.onset_vocab[t1]) in self.onset_group[1]:
                    P[1] += t1
                elif int(self.onset_vocab[t1]) in self.onset_group[2]:
                    P[2] += t1
                elif int(self.onset_vocab[t1]) in self.onset_group[3]:
                    P[3] += t1
                elif int(self.onset_vocab[t1]) in self.onset_group[4]:
                    P[4] += t1
                elif int(self.onset_vocab[t1]) in self.onset_group[5]:
                    P[5] += t1
                elif int(self.onset_vocab[t1]) in self.onset_group[6]:
                    P[6] += t1
                elif int(self.onset_vocab[t1]) in self.onset_group[7]:
                    P[7] += t1

                
            elif t4[0] == 'x' or t4[0] == 'X' or t4[0] == 'y' or t4 == '<unk>':
                cur_inst = self.inst_vocab[t4]
                pass
            
        if cur_inst != -1:
            for i in range(self.booksize):
                try:
                    book_onset_info[i] = self.onset_book[i][P[i]]
                except:
                    book_onset_info[i] = 3
            book_onset_info[8] = cur_inst
            book_onset_info[9] = int((measure/total_len)*24)
            book_onset_info[10] = measure
            book_onset_info[11] = target_chord_tensor[measure-1]
            book_onset_info[12] = target_chord_tensor[measure]
            book_onset_info[13] = target_chord_tensor[measure+1]
            book_onset_info[14] = target_chord_tensor[measure+2]
            book_onset_info[15] = target_chord_tensor[measure+3]
            
            book_container.append(book_onset_info)
        
        book_onset_info = [0]*16
        cur_inst = -1
        P = ['', '', '', '', '', '', '', '']
        
        measure_book_container.append(book_container)
        book_container = []
        
        init_inst = set()
                  
        for first in measure_book_container[1]:
            init_inst.add(first[8])
            
        for inst in init_inst:
            book_container.append([2,2,2,2,2,2,2,2,inst, 0, 0, 0, 0, 2, target_chord_tensor[2], target_chord_tensor[3]])
                
        measure_book_container[0] = book_container

        return target_chord_tensor, measure_book_container
    
    
def create_Obook(gen=None):
    if gen is not None:
        raw_data = gen
                    
        train_dataset = OnsetBook(raw_data)
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_Obook)
        return train_loader
    else:
        raw_data = []
    
        folder_path = "data/raw"
        file_paths = []
        for path, directories, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = path + "/" + file
                    file_paths.append(file_path)
        
        for path in file_paths:
            with open(path, 'r') as f:
                for line in tqdm(f):
                    raw_data.append(line.strip())
        
        train, val = train_test_split(raw_data, test_size=0.3, random_state=5)
          
        train_dataset = OnsetBook(train)
        val_dataset = OnsetBook(val)
        
        batch_size = 1
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_Obook)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_Obook)
        
        # return train_loader, True, True
        return train_loader, val_loader

def handle_2D_batch(batch):
    padded_samples = []
    # print("[[[[[[[[[[[]]]]]]]]]]]")
    # print(batch)
    for sample in batch:
        # 각 내부 리스트를 텐서로 변환
        # [] print(sample)
        if len(sample) == 0:
            continue
        sub_tensors = [torch.tensor(seq, dtype=torch.long) for seq in sample]
        # 내부 리스트들에 대해 패딩 (결과: (내부 리스트 개수, 최대 길이))
        sample_padded = pad_sequence(sub_tensors, batch_first=True, padding_value=0)
        padded_samples.append(sample_padded)
    
    # 여러 샘플을 하나의 배치로 패딩 (결과: (배치 크기, 최대 내부 리스트 개수, 최대 길이))
    batch_tensor = pad_sequence(padded_samples, batch_first=True, padding_value=0)
    return batch_tensor

def collate_Obook(batch):
    chord, onset = zip(*batch)
    onset = onset[0]
    chord = [torch.tensor(c, dtype=torch.long) for c in chord]  
    # padding_value = <eos>
    chord_pad = pad_sequence(chord, padding_value=0, batch_first=True)
    onset_pad = handle_2D_batch(onset)
    
    return chord_pad, onset_pad

