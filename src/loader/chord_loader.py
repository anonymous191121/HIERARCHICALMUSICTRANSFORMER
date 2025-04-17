import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class ChordDataset(Dataset):
    def __init__(self, data, base_model=''):
        super().__init__()
        self.data = data
        self.base_model = base_model
        vocab_path = 'data/vocabs/chord.json'

        with open(vocab_path, 'r') as file:
            self.vocab = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_seq = self.data[idx]
        
        if isinstance(text_seq, str):
            toks = text_seq.split()
        l_toks = len(toks)
        ratio = 4
        chord_list = []
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
            if t1[0] == 'h' or t1[0] == 'H':
                chord_list.append(t1)
                
        target_chord_seq = self.get_chord_seq(chord_list)
            
        target_chord_tensor = [self.vocab[chd] for chd in target_chord_seq]
        target_chord_tensor = [2] + target_chord_tensor[:766] + [1]
        target_chord_tensor = torch.tensor(target_chord_tensor)

        return target_chord_tensor
    
    def get_chord_seq(self, chord_list):
        group_list = []
        for idx in range(0, len(chord_list)):
            group_list.append(chord_list[idx])
            
        return group_list
    
def create_dataloaders(batch_size, base_model=''):
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
            
    train, val_test = train_test_split(raw_data, test_size=0.3, random_state=5)
    
    train_dataset = ChordDataset(train, base_model)
    val_dataset = ChordDataset(val_test, base_model)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)

    return train_loader, val_loader

def collate_batch(batch):
    padded = pad_sequence(batch, padding_value=0, batch_first=True)
    return padded