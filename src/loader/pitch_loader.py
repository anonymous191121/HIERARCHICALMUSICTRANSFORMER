import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class Pitch(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        chord_vocab_path = 'data/vocabs/chord.json'
        inst_vocab_path = 'data/vocabs/inst.json'
        pitch_vocab_path = 'data/vocabs/pitch.json'
        dur_vocab_path = 'data/vocabs/dur.json'
        with open(chord_vocab_path, 'r') as file:
            self.chord_vocab = json.load(file)
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
        
        chord_list = []
        measure = 0
        total_len = 1
        
        input_dur = []
        output_pitch = []
        
        cur_inst = -1
        
        input_dur_container = []
        output_pitch_container = []
        measure_container = []
        
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
                
            if (t1[0] == 'm' or t1[0] == 'M'):
                total_len += 1
            elif t1[0] == 'H':
                # chord_list.append(self.chord_vocab[t1])
                chord_list.append(self.chord_proj[t1[1]])
        
        chord_tensor = chord_list
        # target_chord_tensor = [0,14] + chord_tensor[:766] + [15,0]
        target_chord_tensor = [0,2] + chord_tensor[:766] + [1,0]
        
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
            
            if measure == 767:
                break
                
            if (t1[0] == 'm' or t1[0] == 'M'):
                if len(output_pitch) == 0:
                    pass
                else:
                    # adj_chord = [target_chord_tensor[measure-1], target_chord_tensor[measure], target_chord_tensor[measure+1]]
                    adj_chord = [target_chord_tensor[measure-1], target_chord_tensor[measure], target_chord_tensor[measure+1], target_chord_tensor[measure+2], target_chord_tensor[measure+3]]
                    
                    input_dur = [cur_inst] + input_dur
                    input_dur = [int((measure/total_len)*24)] + input_dur
                    input_dur = adj_chord + input_dur
                    # input_dur.append(measure)
                    # input_dur.append()
                    # input_dur.append(self.inst_vocab[t4])
                
                    output_pitch_container.append(output_pitch)
                    input_dur_container.append(input_dur)
                
                output_pitch = []
                input_dur = []
                cur_inst = -1
                measure += 1
                
            elif t1 == 'NT':
                if len(output_pitch) == 0:
                    pass
                else:
                    # adj_chord = [target_chord_tensor[measure-1], target_chord_tensor[measure], target_chord_tensor[measure+1]]
                    adj_chord = [target_chord_tensor[measure-1], target_chord_tensor[measure], target_chord_tensor[measure+1], target_chord_tensor[measure+2], target_chord_tensor[measure+3]]
                    input_dur = [cur_inst] + input_dur
                    input_dur = [int((measure/total_len)*24)] + input_dur
                    input_dur = adj_chord + input_dur
                    # input_dur.append(measure)
                    # input_dur.append()
                    # input_dur.append(self.inst_vocab[t4])
                
                    output_pitch_container.append(output_pitch)
                    input_dur_container.append(input_dur)
                
                output_pitch = []
                input_dur = []
                cur_inst = -1
                    
            elif t1[0] == 'p' or t1[0] == 'P':
                pass
                        
            elif self.ispitch(t1[0:2]):
                cur_inst = self.inst_vocab[t4]
                dur_idx = self.dur_vocab[t2]
                pitch_idx = self.pitch_vocab[t1]
    
                input_dur.append(dur_idx)
                output_pitch.append(pitch_idx)
            
        
        measure_container.append(measure)
        # target_chord_tensor = torch.tensor(target_chord_tensor)

        return target_chord_tensor, input_dur_container, output_pitch_container

def create_pitch(batch_size, gen=None):
    if gen is not None:
        raw_data = gen
                    
        train_dataset = Pitch(raw_data)
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_Pitch)
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
        
        train_dataset = Pitch(train)
        val_dataset = Pitch(val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_Pitch)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_Pitch)
        
        return train_loader, val_loader

def handle_2D_batch(batch):
    padded_samples = []
    for sample in batch:
        # 각 내부 리스트를 텐서로 변환
        sub_tensors = [torch.tensor(seq, dtype=torch.long) for seq in sample]
        # 내부 리스트들에 대해 패딩 (결과: (내부 리스트 개수, 최대 길이))
        sample_padded = pad_sequence(sub_tensors, batch_first=True, padding_value=0)
        padded_samples.append(sample_padded)
    
    # 여러 샘플을 하나의 배치로 패딩 (결과: (배치 크기, 최대 내부 리스트 개수, 최대 길이))
    batch_tensor = pad_sequence(padded_samples, batch_first=True, padding_value=0)
    return batch_tensor

def collate_Pitch(batch):
    chord, dur, pitch = zip(*batch)
    chord = [torch.tensor(c, dtype=torch.long) for c in chord]  
    # padding_value = <eos>
    chord_pad = pad_sequence(chord, padding_value=0, batch_first=True)
    
    dur_pad = handle_2D_batch(dur)
    pitch_pad = handle_2D_batch(pitch)

    return chord_pad, dur_pad, pitch_pad
