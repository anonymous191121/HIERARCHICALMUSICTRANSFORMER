import os
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from ..utils import noneAR_token_accuracy
from tqdm import tqdm
from ..model.dur_model import DurEncoder
from ..loader.dur_loader import create_Dur

# TODO
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_loader, val_loader = create_Dur()
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}')

with open(f'data/vocabs/dur_group.json', 'r') as file:
    vocab = json.load(file)
vocab_size = len(vocab)//2
print(vocab_size)
# TODO
num_epochs = 100
inner_batch = 128
d_model = 128  # dimension of model
num_layers = 4  # number of decoder layers
num_heads = 8  # number of attention heads
d_ff = d_model*4  # dimension of feed-forward network
max_seq_len = 768

model = DurEncoder(dur_size=vocab_size, d_model=d_model, num_layers=num_layers, d_ff=d_ff)
model.to(device)

lr = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
criterion = nn.CrossEntropyLoss(ignore_index = 0)
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

folder_path = f'src/train/out/dur'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def run(model=model, train_loader=train_loader, val_loader=val_loader, epochs=num_epochs, device=device):
    log =''
    model.to(device)
    model.train()

    best_val_loss = 1000000
    best_val_acc = 0

    train_loss = []
    val_loss = []
    
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        inner_cnt = 0
        
        epoch_loss = []
        
        for batch_idx, (onset, dur) in enumerate(tqdm(train_loader, ncols=60, desc="Training")):
            
            dur = dur.squeeze(0).to(device)
            onset = onset.squeeze(0).to(device)
            
            inner = dur.shape[0]
                
            for i in range(0, inner, inner_batch):
                optimizer.zero_grad()
                
                inner_on = onset[i:i+inner_batch, :]
                inner_dur = dur[i:i+inner_batch, :]

                m_outputs, d_outputs = model(inner_on)

                m_output_ids, d_output_ids = torch.argmax(m_outputs, dim=2), torch.argmax(d_outputs, dim=2)
                
                m_loss = criterion(m_outputs.reshape(-1, 65), inner_dur[:, 0].unsqueeze(1).reshape(-1))
                d_loss = criterion(d_outputs.reshape(-1, model.dur_size), inner_dur[:, 1:].reshape(-1))
                
                loss = m_loss + d_loss

                loss.backward()

                optimizer.step()
                # scheduler.step()
                total, correct, acc = noneAR_token_accuracy(torch.cat([m_output_ids, d_output_ids], dim=1), inner_dur)
                
                total_cnt += total
                total_correct += correct
                total_acc += acc
                total_train_loss += loss.item()
                epoch_loss.append(loss.item())
            
            inner_cnt += inner
            
            if batch_idx % 1000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch}], Step [{batch_idx}], LR: {current_lr:.6f}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
                log += f"Epoch [{epoch}], Step [{batch_idx}], LR: {current_lr:.6f}, Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
                log += f'{d_output_ids[:3,:10]}\n'
                log += f'{inner_dur[:3,:10]}\n'
                save_log(log)

        train_loss.append(total_train_loss / inner_cnt)
        train_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: {scheduler.get_last_lr()[0]:.12f}')
        print(f"SAMPLE TRAIN OUTPUT : ")
        print(d_output_ids[:5,:20])
        print(inner_dur[:5,:20])
        scheduler.step()
        log += f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: {scheduler.get_last_lr()[0]:.12f}\n'
        log += f'{d_output_ids[:5,:20]}\n'
        log += f'{inner_dur[:5,:20]}\n'
        save_log(log)
        scheduler.step()
    
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        inner_cnt = 0
        with torch.no_grad():
            for batch_idx, (onset, dur) in enumerate(tqdm(val_loader, ncols=60, desc="Validating")):

                dur = dur.squeeze(0).to(device)
                onset = onset.squeeze(0).to(device)
                
                inner = dur.shape[0]
                
                for i in range(0, inner, inner_batch):
                    inner_on = onset[i:i+inner_batch, :]
                    inner_dur = dur[i:i+inner_batch, :]
                        
                    m_outputs, d_outputs = model(inner_on)

                    m_output_ids, d_output_ids = torch.argmax(m_outputs, dim=2), torch.argmax(d_outputs, dim=2)
                    
                    m_loss = criterion(m_outputs.reshape(-1, 65), inner_dur[:, 0].unsqueeze(1).reshape(-1))
                    d_loss = criterion(d_outputs.reshape(-1, model.dur_size), inner_dur[:, 1:].reshape(-1))
                    
                    loss = m_loss + d_loss
                    total, correct, acc = noneAR_token_accuracy(torch.cat([m_output_ids, d_output_ids], dim=1), inner_dur)
                    
                    total_cnt += total
                    total_correct += correct
                    total_acc += acc
                    
                    total_val_loss += loss.item()
                
                inner_cnt += inner

        val_loss.append(total_val_loss / inner_cnt)
        val_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}') 
        log += f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}\n'
        save_log(log)
        
        
        with open(folder_path + '/train_loss.pkl', 'wb') as file:
            pickle.dump(train_loss, file)
        with open(folder_path + '/val_loss.pkl', 'wb') as file:
            pickle.dump(val_loss, file)
        with open(folder_path + '/train_acc.pkl', 'wb') as file:
            pickle.dump(train_acc, file)
        with open(folder_path + '/val_acc.pkl', 'wb') as file:
            pickle.dump(val_acc, file)

        if val_loss[-1] < best_val_loss or best_val_acc < val_acc[-1]:
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]

            # torch.save(model.state_dict(), folder_path + f'/model_{epoch+1}_{val_acc[-1]:.4f}_{test_acc[-1]:.4f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # if applicable
                # add any other states you need
            }, folder_path + f'/dur_{epoch+1}_{train_acc[-1]:.4f}_{val_acc[-1]:.4f}.pt')

            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}') 
            log += f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}\n'
            save_log(log)
            
def save_log(log):
    with open(f'{folder_path}/log.txt', "w") as file:
        file.write(log)
    return

run()
