import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class Dur(Dataset):
    def __init__(self, data, booksize=8):
        super().__init__()
        self.data = data

        onset_vocab_path = 'data/vocabs/onset.json'
        dur_vocab_path = 'data/vocabs/dur_group.json'
        measure_vocab_path = 'data/vocabs/measure.json'

        with open(onset_vocab_path, 'r') as file:
            self.onset_vocab = json.load(file)
            
        with open(measure_vocab_path, 'r') as file:
            self.measure_vocab = json.load(file)

        with open(dur_vocab_path, 'r') as file:
            self.dur_vocab = json.load(file)
            
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
        
        measure_dur_container = []
        onset_container = []
        
        measure = 0
        total_len = 1
        
        
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
                
            if (t1[0] == 'm' or t1[0] == 'M'):
                total_len += 1
                
        
        measure_dur_list = []
        onset_list = []
        D = ''
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
            
            if measure == 766:
                break
                
            if (t1[0] == 'm' or t1[0] == 'M'):
                if len(measure_dur_list) > 0:
                    try:
                        measure_dur_list.append(self.dur_vocab[D])
                    except:
                        measure_dur_list.append(3)
                        
                    measure_dur_container.append([m_idx] + measure_dur_list)
                    measure_dur_list = []
                    
                    onset_container.append([int((measure/total_len)*24)] + onset_list)
                    onset_list = []
                    D = ''
                    
                m_idx = self.measure_vocab[t1]
                measure += 1
                
            elif t1 == 'NT':
                if len(measure_dur_list) > 0:
                    try:
                        measure_dur_list.append(self.dur_vocab[D])
                    except:
                        measure_dur_list.append(3)
                    measure_dur_container.append([m_idx] + measure_dur_list)
                    measure_dur_list = []
                    
                    onset_container.append([int((measure/total_len)*24)] + onset_list)
                    onset_list = []
                    D = ''
                    
            elif t1[0] == 'p' or t1[0] == 'P':
                if D != '':
                    try:
                        measure_dur_list.append(self.dur_vocab[D])
                    except:
                        measure_dur_list.append(3)
                    D = ''
                onset_list.append(self.onset_vocab[t1])
                
            elif t2[0] == 'r':
                D += t2
            
        if len(measure_dur_list) > 0:
            try:
                measure_dur_list.append(self.dur_vocab[D])
            except:
                measure_dur_list.append(3)
            measure_dur_container.append([m_idx] + measure_dur_list)
            measure_dur_list = []
            
            onset_container.append([int((measure/total_len)*24)] + onset_list)
            onset_list = []

        return onset_container, measure_dur_container
    
    
def create_Dur(gen=None):
    if gen is not None:
        raw_data = gen
                    
        train_dataset = Dur(raw_data)
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_dur)
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
        
        train_dataset = Dur(train)
        val_dataset = Dur(val)
        
        batch_size = 1
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_dur)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_dur)
        
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

def collate_dur(batch):
    onset, measure = zip(*batch)
    # print("[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]")
    # print(onset)
    # print("[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]")
    # print(measure)
    # onset = onset[0]
    # measure = measure[0]
    onset_pad = handle_2D_batch(onset)
    measure_pad = handle_2D_batch(measure)
    
    return onset_pad, measure_pad