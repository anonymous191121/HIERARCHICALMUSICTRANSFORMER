import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class InstBookDataset(Dataset):
    def __init__(self, data, booksize=8):
        super().__init__()
        self.data = data
        self.booksize = booksize
        self.inst_book = []
        
        for i in range(booksize):
            with open(f'data/vocabs/book/inst_B{i}.json', 'r') as file:
                vocab = json.load(file)
            self.inst_book.append(vocab)
        
        chord_vocab_path = 'data/vocabs/chord.json'
        with open(chord_vocab_path, 'r') as file:
            self.chord_vocab = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_seq = self.data[idx]
        
        if isinstance(text_seq, str):
            toks = text_seq.split()
            
        l_toks = len(toks)
        ratio = 4
        chord_list = []
        
        inst_in_measure = []
        inst_list = []
        
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
            if t1[0] == 'H':
                chord_list.append(t1)

            if t4[0] == 'x' or t4[0] == 'X' or t4[0] == 'y' or t4 == '<unk>':
                inst_in_measure.append(t4)
                
            if (t1[0] == 'm' or t1[0] == 'M') and len(chord_list) > 0:
                inst_list.append(inst_in_measure)
                inst_in_measure = []
        inst_list.append(inst_in_measure)
        
        chord_tensor = [self.chord_vocab[chd] for chd in chord_list]
        target_chord_tensor = [2] + chord_tensor[:766] + [1]
        target_chord_tensor = torch.tensor(target_chord_tensor)
        
        inst_book_string = self.make_inst_book_string(inst_list)
        target_group_inst_tensor = torch.tensor(inst_book_string)

        return target_chord_tensor, target_group_inst_tensor
    
    def make_inst_book_string(self, inst_list):
        book_inst = []
        for book_num in range(self.booksize):
            tmp_book_inst = [2]
            
            for measure_inst in inst_list:
                group_inst = ''
                
                for inst in measure_inst:
                    if inst in group_inst:
                        pass
                    elif inst in self.inst_book[book_num].keys():
                        group_inst += inst
                try:
                    tmp_book_inst.append(self.inst_book[book_num][group_inst])
                except:
                    tmp_book_inst.append(3)
            tmp_book_inst = tmp_book_inst[:766] + [1]
            book_inst.append(tmp_book_inst)
            
        return book_inst

def create_book(batch_size, book_size=8, gen=''):
    if gen == '':
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
        
        train_dataset = InstBookDataset(train, booksize=book_size)
        val_dataset = InstBookDataset(val, booksize=book_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch_book)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch_book)
        
        return train_loader, val_loader
    else:
        raw_data = gen
                
        train_dataset = InstBookDataset(raw_data, booksize=book_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch_book)
        return train_loader

def collate_batch_book(batch):
    target_chord_tensor, target_inst_tensor = zip(*batch)
    
    # Convert chords to tensors if needed.
    target_chord_tensor = [
        torch.tensor(chord) if not isinstance(chord, torch.Tensor) else chord
        for chord in target_chord_tensor
    ]
    
    # For inst data: keep each "book" as a list of tensors,
    # converting each sequence individually.
    target_inst_tensor = [
        [torch.tensor(seq) if not isinstance(seq, torch.Tensor) else seq
         for seq in inst]
        for inst in target_inst_tensor
    ]
    
    # Pad chord sequences.
    chord_padded = pad_sequence(target_chord_tensor, padding_value=0, batch_first=True)
    
    # Each book is now a list of tensors.
    inst_padded_books = [
        pad_sequence(book, batch_first=True, padding_value=0)
        for book in target_inst_tensor
    ]
    
    # Determine the maximum sequence length from the padded books.
    max_seq_len = max(book.shape[1] for book in inst_padded_books)
    
    # Pad each book along the sequence dimension to have the same max_seq_len.
    inst_padded_books = [
        torch.nn.functional.pad(book, (0, max_seq_len - book.shape[1]), value=0)
        for book in inst_padded_books
    ]
    
    # Stack the padded books to form a tensor of shape [batch_size, book_size, max_seq_len].
    inst_padded = torch.stack(inst_padded_books)
    
    return chord_padded, inst_padded
