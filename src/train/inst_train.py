import os
import pickle
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from ..model.inst_model import InstTransformer
from ..loader.inst_loader import create_book
from ..utils import calculate_accuracy

import torch
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

#############################################################
#############################################################
#############################################################
batch_size = 16
num_epochs = 500
#############################################################
#############################################################
#############################################################
    
book_size=8
book_vocab_size = []
for i in range(book_size):
    with open(f'data/vocabs/book/inst_B{i}.json', 'r') as file:
        vocab = json.load(file)
    book_vocab_size.append(len(vocab)//2)
d_model=512
num_heads=8
d_ff=2048
num_layers=3
max_len=5000
dropout=0.1
model = InstTransformer(book_size=book_size, book_vocab_size=book_vocab_size, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, max_len=max_len, dropout=dropout)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)


train_loader, val_loader = create_book(batch_size, book_size=book_size)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}')


folder_path = f'src/train/out/inst'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def run(model=model, train_loader=train_loader, val_loader=val_loader, epochs=num_epochs, device=device):

    model.to(device)
    model.train()

    log = ''
    best_val_loss = 100000000
    best_val_acc = -1

    train_loss = []
    val_loss = []
    
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        
        total_train_loss = [0 for _ in range(model.book_size)]
        total_acc = 0
        
        total_val_loss = [0 for _ in range(model.book_size)]
        total_val_acc = 0
        
        for (chords, insts) in tqdm(train_loader, ncols=60):
            optimizer.zero_grad()

            chords = chords.to(device)
            insts = insts.to(device)

            input_inst = (insts[:,:,:-1]).permute(0,2,1)
            target_inst = (insts[:,:,1:]).permute(0,2,1)

            output_logit_container = model(chords, input_inst)
            
            loss_container, loss = model.ce_loss(output_logit_container, target_inst)
            loss.backward()

            pred = model.inference_sampling(chords, input_inst)
            accuracy, train_book_acc = calculate_accuracy(pred, target_inst, pad_token=0, book_size=book_size)
            total_acc += accuracy/8
            
            for i in range(model.book_size):
                total_train_loss[i] += loss_container[i].item()
            
            optimizer.step()

        scheduler.step()
        train_loss.append(total_train_loss)
        train_acc.append(total_acc)
        
        with torch.no_grad():
            for (chords, insts) in tqdm(val_loader, ncols=60):
                chords = chords.to(device)
                insts = insts.to(device)

                input_inst = (insts[:,:,:-1]).permute(0,2,1)
                target_inst = (insts[:,:,1:]).permute(0,2,1)

                output_logit_container = model(chords, input_inst)

                loss_container, loss = model.ce_loss(output_logit_container, target_inst)
                
                pred = model.inference_sampling(chords, input_inst)
                accuracy, val_book_acc = calculate_accuracy(pred, target_inst, pad_token=0, book_size=book_size)
                total_val_acc += accuracy/8
                
                for i in range(model.book_size):
                    total_val_loss[i] += loss_container[i].item()

        val_loss.append(total_val_loss)
        val_acc.append(total_val_acc)

        log += f'Epoch {epoch+1} | T ACC : {train_acc[-1]:.4f} V ACC : {val_acc[-1]:.4f} |  LR: {scheduler.get_last_lr()[0]:.8f}\n'
        save_log(log)
        print(f'Epoch {epoch+1} | T ACC : {train_acc[-1]:.4f} V ACC : {val_acc[-1]:.4f} |  LR: {scheduler.get_last_lr()[0]:.8f}')
        for i in range(model.book_size):
            log +=f'BOOK {i} | Train Loss: {train_loss[-1][i]:.6f} Train Acc : {train_book_acc[i]:.8f} | Val Loss: {val_loss[-1][i]:.6f} | V ACC : {val_book_acc[i]:.4f}\n' 
            print(f'BOOK {i} | Train Loss: {train_loss[-1][i]:.6f} Train Acc : {train_book_acc[i]:.8f} | Val Loss: {val_loss[-1][i]:.6f} | V ACC : {val_book_acc[i]:.4f}')
        save_log(log)
            
        with open(f'{folder_path}/train_loss.pkl', 'wb') as file:
            pickle.dump(train_loss, file)
        with open(f'{folder_path}/val_loss.pkl', 'wb') as file:
            pickle.dump(val_loss, file)
        with open(f'{folder_path}/train_acc.pkl', 'wb') as file:
            pickle.dump(train_acc, file)
        with open(f'{folder_path}/val_acc.pkl', 'wb') as file:
            pickle.dump(val_acc, file)

        if (val_loss[-1][0] < best_val_loss or val_acc[-1] > best_val_acc):
            if val_loss[-1][0] < best_val_loss:
                best_val_loss = val_loss[-1][0]
            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
            
            torch.save(model.state_dict(), f'{folder_path}/inst_{epoch+1}_{val_loss[-1][0]:.4f}_{val_acc[-1]:.4f}.pt')
            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1][0]:.6f}, Val Acc: {val_acc[-1]:.6f}')
            log += f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1][0]:.6f}, Val Acc: {val_acc[-1]:.6f}\n'
            save_log(log)

def save_log(log):
    with open(f'{folder_path}/log.txt', "w") as file:
        file.write(log)
    return

run()
