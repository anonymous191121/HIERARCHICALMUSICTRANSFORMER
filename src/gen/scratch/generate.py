import sys
import json
import torch
import warnings
from tqdm import tqdm
from datetime import datetime
from ...model.chord_model import ChordDecoder
from ...model.inst_model import InstTransformer
from ...model.onset_model import OnsetTransformer
from ...model.dur_model import DurEncoder
from ...model.pitch_model import PitchEncoder

# ignore only the PyTorch UserWarning about tensor.T deprecation
warnings.filterwarnings(
    "ignore",
    message=r".*The use of `x\.T` on tensors of dimension other than 2.*",
    category=UserWarning,
)

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

def onsetbook2idx(book):
    onset_book_vocab = []
    booksize = 8

    for i in range(booksize):
        with open(f'data/vocabs/book/onset_B{i}.json', 'r') as file:
            vocab = json.load(file)
        onset_book_vocab.append(vocab)

    onset_vocab_path = 'data/vocabs/onset.json'
    with open(onset_vocab_path, 'r') as file:
        onset_vocab = json.load(file)
    mix = ''
    for i in range(8):
        try:
            if book[i] < 4:
                continue
            mix += onset_book_vocab[i][str(book[i])]
        except:
            continue
        
    idx_list = []
    for i in range(0, len(mix), 2):
        idx_list.append(onset_vocab[mix[i:i+2]])
    return sorted(idx_list)


def make_onset(onset_model_path, chord_idx, inst_idx, temperature, device, top_k=5):
    book_size=8
    book_vocab_size = []
    for i in range(book_size):
        with open(f'data/vocabs/book/onset_B{i}.json', 'r') as file:
            vocab = json.load(file)
        book_vocab_size.append((len(vocab)//2))
    d_model = 512
    num_heads = 8
    d_ff = d_model*4
    model = OnsetTransformer(d_model=d_model,book_size=book_size, book_vocab_size=book_vocab_size, num_heads=num_heads, d_ff=d_ff)
    model.load_state_dict(torch.load(onset_model_path, map_location=device, weights_only=True)["model_state_dict"])
    model.to(device)
    model.eval()
    
    chord_idx = torch.cat((torch.tensor([0]).to(device), chord_idx, torch.tensor([0]).to(device)),dim=0).to(device)
    l = chord_idx.shape[0] - 4
    
    onset_output_container = [[[2,2,2,2,2,2,2,2]]*len(inst_idx[0])]
    
    for measure_idx in range(l):
        if measure_idx == 0:
            inst_cnt = len(inst_idx[measure_idx])
            
            if inst_cnt == 0:
                onset_output_container.append([[]])
                continue
            
            init_en_input = torch.zeros(1, inst_cnt, 16)
            for i in range(inst_cnt):
                init_en_input[:, i, 8] = inst_idx[measure_idx][i]
            init_en_input[:, :, :8] = 2
            init_en_input[:, :, 9] = 0
            init_en_input[:, :, 10] = 0
            init_en_input[:, :, 11] = 0
            init_en_input[:, :, 12] = 0
            init_en_input[:, :, 13] = 2
            init_en_input[:, :, 14] = chord_idx[2]
            init_en_input[:, :, 15] = chord_idx[3]
            
            
                
            init_de_input = torch.zeros(1, inst_cnt, 16)
            for i in range(inst_cnt):
                init_de_input[:, i, 8] = inst_idx[measure_idx][i]
            init_de_input[:, :, 9] = 0
            init_de_input[:, :, 10] = 0
            init_de_input[:, :, 11] = 0
            init_de_input[:, :, 12] = 2
            init_de_input[:, :, 13] = chord_idx[2]
            init_de_input[:, :, 14] = chord_idx[3]
            init_de_input[:, :, 15] = chord_idx[4]
            
            output_logit_container = model(encoder_in=init_en_input.long().to(device), decoder_in=init_de_input.long().to(device))
            pred = model.top_k_sampling(output_logit_container, k=top_k, temperature=temperature).to(device)
            onset_output_container.append(pred[0].int().tolist())
        else:
            inst_cnt = len(inst_idx[measure_idx-1])
            de_inst_cnt = len(inst_idx[measure_idx])
            
            if de_inst_cnt == 0:
                onset_output_container.append([[]])
                continue

            else:
                if inst_cnt == 0:
                    init_en_input = torch.zeros(1, de_inst_cnt, 16)
                    for i in range(de_inst_cnt):
                        init_en_input[:, i, 8] = inst_idx[measure_idx][i]
                    init_en_input[:, :, :8] = 2
                    init_en_input[:, :, 9] = int((measure_idx/l)*24)
                    init_en_input[:, :, 10] = measure_idx
                    init_en_input[:, :, 11] = chord_idx[measure_idx-1]
                    init_en_input[:, :, 12] = chord_idx[measure_idx]
                    init_en_input[:, :, 13] = chord_idx[measure_idx+1]
                    init_en_input[:, :, 14] = chord_idx[measure_idx+2]
                    init_en_input[:, :, 15] = chord_idx[measure_idx+3]
                    
                    init_de_input = torch.zeros(1, de_inst_cnt, 16)
                    for i in range(de_inst_cnt):
                        init_de_input[:, i, 8] = inst_idx[measure_idx][i]
                    init_de_input[:, :, 9] = int((measure_idx/l)*24)
                    init_de_input[:, :, 10] = measure_idx
                    init_de_input[:, :, 11] = chord_idx[measure_idx]
                    init_de_input[:, :, 12] = chord_idx[measure_idx+1]
                    init_de_input[:, :, 13] = chord_idx[measure_idx+2]
                    init_de_input[:, :, 14] = chord_idx[measure_idx+3]
                    init_de_input[:, :, 15] = chord_idx[measure_idx+4]
                    
                    output_logit_container = model(encoder_in=init_en_input.long().to(device), decoder_in=init_de_input.long().to(device))
                    pred = model.top_k_sampling(output_logit_container, k=top_k, temperature=temperature).to(device)
                    onset_output_container.append(pred[0].int().tolist())
                
                else:
                    init_en_input = torch.zeros(1, inst_cnt, 16)
                    for i in range(min(inst_cnt, len(onset_output_container[measure_idx]))):
                        # print("[[[[[]]]]]")
                        # print(torch.tensor(onset_output_container[measure_idx-1][i]))
                        init_en_input[:, i, :8] = torch.tensor(onset_output_container[measure_idx][i])
                    for i in range(inst_cnt):
                        # print("[[[[[[[[[[[[]]]]]]]]]]]]")
                        # print(inst_idx_seq[measure_idx-1][i])
                        init_en_input[:, i, 8] = inst_idx[measure_idx-1][i]
                    init_en_input[:, :, 9] = int((measure_idx/l)*24)
                    init_en_input[:, :, 10] = measure_idx
                    init_en_input[:, :, 11] = chord_idx[measure_idx-1]
                    init_en_input[:, :, 12] = chord_idx[measure_idx]
                    init_en_input[:, :, 13] = chord_idx[measure_idx+1]
                    init_en_input[:, :, 14] = chord_idx[measure_idx+2]
                    init_en_input[:, :, 15] = chord_idx[measure_idx+3]
                    
                    init_de_input = torch.zeros(1, de_inst_cnt, 16)
                    for i in range(de_inst_cnt):
                        init_de_input[:, i, 8] = inst_idx[measure_idx][i]
                    init_de_input[:, :, 9] = int((measure_idx/l)*24)
                    init_de_input[:, :, 10] = measure_idx
                    init_de_input[:, :, 11] = chord_idx[measure_idx]
                    init_de_input[:, :, 12] = chord_idx[measure_idx+1]
                    init_de_input[:, :, 13] = chord_idx[measure_idx+2]
                    init_de_input[:, :, 14] = chord_idx[measure_idx+3]
                    init_de_input[:, :, 15] = chord_idx[measure_idx+4]
                    # print(inst_idx_seq)
                    # print(init_en_input.shape)
                    # print(init_de_input.shape)
                    # print(de_inst_cnt)
                    output_logit_container = model(encoder_in=init_en_input.long().to(device), decoder_in=init_de_input.long().to(device))
                    pred = model.top_k_sampling(output_logit_container, k=top_k, temperature=temperature).to(device)
                    onset_output_container.append(pred[0].int().tolist())
    
    onset_book = onset_output_container[1:]
    onset_idx = []
    
    for idx, in_measure in enumerate(onset_book):
        store = []
        if len(in_measure) == 0:
            store.append([])
            continue
        
        for i in range(len(inst_idx[idx])):
            store.append(onsetbook2idx(in_measure[i]))
        
        onset_idx.append(store)
    
    onset_vocab_path = 'data/vocabs/onset.json'
    with open(onset_vocab_path, 'r') as file:
        onset_vocab = json.load(file)
        
    onset_txt = []
    for onset_per_measuer in onset_idx:
        per_meauser = []
        for in_measeur in onset_per_measuer:
            inner = []
            for o in in_measeur:
                inner.append(onset_vocab[str(o)])
            per_meauser.append(inner)
        onset_txt.append(per_meauser)
    
    return onset_book, onset_idx, onset_txt


def make_dur(dur_model_path, onset_idx, temperature, device, top_k=5):
    with open(f'data/vocabs/dur_group.json', 'r') as file:
        vocab = json.load(file)
    vocab_size = len(vocab)//2
    
    onset_vocab_path = 'data/vocabs/dur_group.json'
    with open(onset_vocab_path, 'r') as file:
        mix_dur_vocab = json.load(file)

    onset_vocab_path = 'data/vocabs/dur.json'
    with open(onset_vocab_path, 'r') as file:
        pure_dur_vocab = json.load(file)
        
    # TODO
    d_model = 128  # dimension of model
    num_layers = 4  # number of decoder layers
    d_ff = d_model*4  # dimension of feed-forward network
    model = DurEncoder(dur_size=vocab_size, d_model=d_model, num_layers=num_layers, d_ff=d_ff)
    model.load_state_dict(torch.load(dur_model_path, map_location=device, weights_only=True)["model_state_dict"])
    model.to(device)
    model.eval()
    
    l = len(onset_idx)
    
    dur_output_container = []
    for idx, in_measure in enumerate(onset_idx):
        if len(in_measure) == 0:
            dur_output_container.append([[]])
            continue
        
        tmp_out_onset = []
        for in_onset in in_measure:
        
            in_onset = [int((idx/l)*24)] + in_onset
            in_onset = torch.tensor(in_onset).unsqueeze(0).to(device)
            
            m_outputs, d_outputs = model(in_onset)
            dur_output = torch.argmax(d_outputs, dim=2)
            tmp_out_onset.append(dur_output[0].int().tolist())
        dur_output_container.append(tmp_out_onset)

    pure_dur_output_container = []
    for in_measure in dur_output_container:
        tmp = []
        if len(in_measure) == 0:
            pure_dur_output_container.append([[]])
            continue
        for in_dur in in_measure:
            # in_dur = [int]
            in_tmp = ''
            for dur in in_dur:
                if dur < 4:
                    continue
                in_tmp += mix_dur_vocab[str(dur)]
            in_tmp_idx = []
            for i in range(0, len(in_tmp), 2):
                in_tmp_idx.append(pure_dur_vocab[in_tmp[i:i+2]])
            tmp.append(in_tmp_idx)
        pure_dur_output_container.append(tmp)
        
    dur_group = dur_output_container
    dur_pure = pure_dur_output_container
    
    dur_txt = []
    for onset_per_measuer in dur_pure:
        per_meauser = []
        for in_measeur in onset_per_measuer:
            inner = []
            for o in in_measeur:
                inner.append(pure_dur_vocab[str(o)])
            per_meauser.append(inner)
        dur_txt.append(per_meauser)
    
    return dur_group, dur_pure, dur_txt

def make_pitch(pitch_model_path, chord_idx, inst_idx_seq, dur_pure, temperature, device, top_k=5):
    vocab_size = 132  # size of the vocabulary not 832
    d_model = 64  # dimension of model
    num_layers = 12  # number of decoder layers
    num_heads = 8  # number of attention heads
    d_ff = d_model*4  # dimension of feed-forward network
    max_seq_len = 512

    model = PitchEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len)
    model.load_state_dict(torch.load(pitch_model_path, map_location=device, weights_only=True)["model_state_dict"])
    model.to(device)
    
    onset_vocab_path = 'data/vocabs/dur.json'
    with open(onset_vocab_path, 'r') as file:
        pure_dur_vocab = json.load(file)
    vocab_path = 'data/vocabs/chord.json'
    with open(vocab_path, 'r') as file:
        chord_vocab = json.load(file)
        
    l = len(chord_idx) - 2
    chord_proj = {'C':1, 'd':2, 'D':3, 'e':4, 'E':5, 'F':6, 'g':7, 'G':8, 'a':9, 'A':10, 'b':11, 'B':12, 'N':13}

    proj_chord = [chord_vocab[str(c.item())][1] for c in chord_idx[1:-1]]
    proj_expand_chord = [0,14] + [chord_proj[c] for c in proj_chord] + [15,0]
    pitch_output_container = []

    for m_idx, in_measure in enumerate(dur_pure):
        r_idx = int((m_idx/l)*24)

        
        if len(in_measure) == 0:
            pitch_output_container.append([[]])
            continue
        
        tmp_out = []
        for i_idx, in_inst in enumerate(in_measure):
            inst_idx = inst_idx_seq[m_idx][i_idx]
            pitch_input = torch.tensor([c for c in proj_expand_chord[m_idx:m_idx+5]] + [r_idx, inst_idx] + in_inst).to(device)
            pitch_input = pitch_input.unsqueeze(0)
            # print("INPUT")
            # print(pitch_input)
            outputs = model(pitch_input)
            outputs = outputs[:,7:]
            output_ids = model.sampling(outputs, k=top_k, temperature=temperature)
            tmp_out.append(output_ids[0].int().tolist())
        pitch_output_container.append(tmp_out)
            
    pitch_idx = pitch_output_container
    
    pitch_vocab_path = 'data/vocabs/pitch.json'
    with open(pitch_vocab_path, 'r') as file:
        pitch_vocab = json.load(file)
    
    pitch_txt = []
    for onset_per_measuer in pitch_idx:
        per_meauser = []
        for in_measeur in onset_per_measuer:
            inner = []
            for o in in_measeur:
                inner.append(pitch_vocab[str(o)])
            per_meauser.append(inner)
        pitch_txt.append(per_meauser)
    
    return pitch_idx, pitch_txt

def int_to_letter(num):
    if num < 10:
        return str(num)
    return chr(ord('a') + (num - 10))

def make_music(chord_txt, inst_txt, onset_txt, dur_group, dur_txt, pitch_txt):
    # TODO 
    # You can adjust this (4/4, 3/4, 2/2 6/8 etc...)
    TIME_SIGNATURE = 'mw'
    chord_txt = chord_txt[1:-1] # remove bos, eos
    l = len(chord_txt)
    output_txt = ''
    mapping = 1
    inst_track_map = {}
    
    onset_vocab_path = 'data/vocabs/dur_group.json'
    with open(onset_vocab_path, 'r') as file:
        mix_dur_vocab = json.load(file)
    
    for m_idx in range(l):
        # print("M")
        # print(m_idx)
        output_txt += f'{TIME_SIGNATURE} RZ TZ YZ '
        output_txt += f'{chord_txt[m_idx]} RZ TZ YZ '
        
        for i_idx, inst in enumerate(inst_txt[m_idx]):
            # print("I")
            # print(i_idx)
            output_txt += 'NT RZ TZ YZ '
            inst_str = inst
            
            onset_len = len(onset_txt[m_idx][i_idx])
            o_idx = 0
            d_idx = 0
            p_idx = 0
            for _ in range(onset_len):
                output_txt += f'{onset_txt[m_idx][i_idx][o_idx]} RZ TZ YZ '
                
                if inst_str not in inst_track_map.keys():
                    if mapping > 0:
                        track = int_to_letter(mapping)
                    else:
                        track = mapping
                    inst_track_map[inst_str] = track
                    mapping += 1

                # if str(dur_output_container[m_idx][i_idx][_]) == '0' or pitch_vocab[str(pitch_output_container[m_idx][i_idx][p_idx])] == '<pad>' or pitch_output_container[m_idx][i_idx][p_idx] < 4 or str(pitch_output_container[m_idx][i_idx][p_idx]) == '0':
                #     break
                
                if len(mix_dur_vocab[str(dur_group[m_idx][i_idx][_])]) > 2:
                    output_txt += f'{pitch_txt[m_idx][i_idx][p_idx]} {dur_txt[m_idx][i_idx][d_idx]} t{inst_track_map[inst_str]} {inst_str} '
                    d_idx += 1
                    p_idx += 1
                    if str(dur_txt[m_idx][i_idx][_]) == '0' or pitch_txt[m_idx][i_idx][p_idx] == '<pad>' or pitch_txt[m_idx][i_idx][p_idx] == '0':
                        break
                    output_txt += f'{pitch_txt[m_idx][i_idx][p_idx]} {dur_txt[m_idx][i_idx][d_idx]} t{inst_track_map[inst_str]} {inst_str} '
                    o_idx += 1
                    d_idx += 1
                    p_idx += 1
                else:    
                    output_txt += f'{pitch_txt[m_idx][i_idx][p_idx]} {dur_txt[m_idx][i_idx][d_idx]} t{inst_track_map[inst_str]} {inst_str} '
                    o_idx += 1
                    d_idx += 1
                    p_idx += 1
    
    return output_txt


if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO
    chord_model_path = "src/train/out/chord/chord_8_1.7804_0.6566.pt"
    inst_model_path = "src/train/out/inst/inst_15_0.9675_0.6808.pt"
    onset_model_path = "src/train/out/onset/onset_38_0.6973_0.1639.pt"
    dur_model_path = "src/train/out/dur/dur_72_0.5537_0.4769.pt"
    pitch_model_path = "src/train/out/pitch/pitch_19_2.9024_0.1776.pt"

    num_generate = int(sys.argv[1])
    temperature = float(sys.argv[2])
    
    for i in tqdm(range(num_generate)):
        while True:
            try:
                chord_idx, chord_txt = make_chord(chord_model_path, temperature, device=device)
                # chord_idx : [  2,   6,   6,   6]
                # chord_txt : ['<bos>', 'HCM', '<eos>']
                
                inst_idx, inst_txt = make_inst(inst_model_path, chord_idx, temperature, device=device)
                # inst_idx : [[52, 60, 76, 77], [48, 52, 77], [52]]
                # inst_txt : [['xM', 'xU', 'Xa', 'Xb'], ['xI', 'xM', 'Xb'], ['xM']]
                
                onset_book, onset_idx, onset_txt = make_onset(onset_model_path, chord_idx, inst_idx, temperature, device=device)
                # onset_book : [[[4, 4, 4, 4, 8, 4, 4, 4], [4, 4, 4, 4, 8, 4, 4, 4]], [[4, 4, 4, 4, 8, 4, 9, 4]]]
                # onset_idx : [[[8, 16, 24, 28, 32, 40, 48], []], [[12, 30, 34], [20]]]

                dur_group, dur_pure, dur_txt = make_dur(dur_model_path, onset_idx, temperature, device=device)
                
                pitch_idx, pitch_txt = make_pitch(pitch_model_path, chord_idx, inst_idx, dur_pure, temperature, device=device)
                
                output_txt = make_music(chord_txt, inst_txt, onset_txt, dur_group, dur_txt, pitch_txt)

                now = datetime.now()  
                timestamp = now.strftime("%m%d-%H%M%S")

                with open(f'demo/txt/{timestamp}_{len(chord_idx)-2}M_{i}.txt', 'w') as file:
                    file.write(output_txt)
                print(f"FINISH GENERATION With {len(chord_idx)-2} Measures")
                break
            except:
                pass
                
        
    
    