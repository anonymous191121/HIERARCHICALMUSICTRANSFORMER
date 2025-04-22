import sys
import json
import torch
from ...model.chord_model import ChordDecoder
from ...model.inst_model import InstTransformer

def make_chord(chord_model_path, temperature, device, top_k=5):
    vocab_size = 136  # size of the vocabulary
    d_model = 512  # dimension of model
    num_layers = 6  # number of decoder layers
    num_heads = 8  # number of attention heads
    d_ff = d_model*4  # dimension of feed-forward network
    max_seq_len = 768
    dropout = 0.1  # dropout rate
    model = ChordDecoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout)
    model.load_state_dict(torch.load(chord_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    vocab_path = 'data/vocabs/chord.json'
    with open(vocab_path, 'r') as file:
        chord_vocab = json.load(file)

    chord_idx = torch.tensor([2]).unsqueeze(0).to(device)
    out = model.sampling(chord_idx, temperature=temperature, top_k=top_k).squeeze(0)
    out_txt = [chord_vocab[str(c.item())] for c in out]
    return out, out_txt

def book_2_inst(input_token, single_inst_vocab, inst_book):            
    output = [0] * 133
    for idx, book_token in enumerate(input_token):
        try:
            book_str = inst_book[idx][str(int(book_token.item()))]
            if len(book_str) == 5:
                output[3] == 1
                continue
        except:
            output[3] = 1
            continue
        
        for mark in range(0, len(book_str), 2):
            single_inst_str = book_str[mark:mark+2]
            output[single_inst_vocab[single_inst_str]] = 1
            
    return output

def make_inst(inst_model_path, chord_seq, temperature, device, top_k=5):
    book_size=8
    book_vocab_size = []
    inst_book = []
    for i in range(book_size):
        with open(f'data/vocabs/book/inst_B{i}.json', 'r') as file:
            vocab = json.load(file)
        inst_book.append(vocab)
        book_vocab_size.append(len(vocab)//2)
    with open(f'data/vocabs/inst.json', 'r') as file:
        inst_single = json.load(file)
    d_model=512
    num_heads=8
    d_ff=2048
    num_layers=3
    max_len=5000
    dropout=0.1
    model = InstTransformer(book_size=book_size, book_vocab_size=book_vocab_size, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, max_len=max_len, dropout=dropout)
    model.load_state_dict(torch.load(inst_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    chord_seq = chord_seq.unsqueeze(0).to(device)
    inst = torch.tensor([[[2,2,2,2,2,2,2,2]]]).to(device)
    pred = model.generate(chord_seq, inst, temperature=temperature, k=top_k, prompt_len=0)
    
    # bos 제거
    pred = pred[:, 1:, :]
    
    output_len = chord_seq.shape[1] - 2

    pred_hitmap = torch.zeros(output_len, 133)

    for i in range(output_len):
        inst_map = torch.tensor(book_2_inst(pred[0,i,:], inst_single, inst_book))
        pred_hitmap[i,:] = inst_map.T
    pred_hitmap = pred_hitmap[:output_len,:]
    
    inst_out_list = []
    inst_out_list.append(pred_hitmap.tolist())
    
    inst_seq = []
    for i in range(len(inst_out_list)):
        inst_idx_seq = []
        for p in inst_out_list[i]:
            i_list = []
            for idx, bit in enumerate(p):
                if bit == 1:
                    i_list.append(idx)
            inst_idx_seq.append(i_list)
        inst_seq.append(inst_idx_seq)
    inst_seq = inst_seq[0]
    
    inst_txt = []
    for insts in inst_seq:
        inner = []
        for i in insts:
            inner.append(inst_single[str(i)])
        inst_txt.append(inner)
    
    return inst_seq, inst_txt

def make_onset(onset_model_path, chord_idx, inst_idx, temperature, device, top_k=5):
    
    
    
    
    
    return onset_idx, onset_txt



def make_music():
    return None


if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO
    chord_model_path = "src/train/out/chord/chord_8_1.7804_0.6566.pt"
    inst_model_path = "src/train/out/inst/inst_15_0.9675_0.6808.pt"
    onset_model_path = "data/raw"
    dur_model_path = "data/raw"
    pitch_model_path = "data/raw"

    temperature = float(sys.argv[1])
    
    chord_idx, chord_txt = make_chord(chord_model_path, temperature, device=device)
    print(chord_txt)
    print(len(chord_txt))
    # chord_idx : [  2,   6,   6,   6]
    # chord_txt : ['<bos>', 'HCM', '<eos>']
    
    inst_idx, inst_txt = make_inst(inst_model_path, chord_idx, temperature, device=device)
    # inst_idx : [[52, 60, 76, 77], [48, 52, 77], [52]]
    # inst_txt : [['xM', 'xU', 'Xa', 'Xb'], ['xI', 'xM', 'Xb'], ['xM']]

    onset_idx, onset_txt = make_onset(onset_model_path, chord_idx, inst_idx, temperature, device=device)
    