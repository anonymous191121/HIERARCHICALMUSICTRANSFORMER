import os
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from ..utils import noneAR_token_accuracy, pitch_prob
from tqdm import tqdm
from ..model.pitch_model import PitchEncoder
from ..loader.pitch_loader import create_pitch

# TODO
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

#############################################################
#############################################################
#############################################################

num_epochs = 100
inner_batch = 60
vocab_size = 132  # size of the vocabulary not 832
d_model = 64  # dimension of model
num_layers = 12  # number of decoder layers
num_heads = 8  # number of attention heads
d_ff = d_model*4  # dimension of feed-forward network
max_seq_len = 512

# Optimizer % Define warmup/decay scheduler
train_loader, val_loader = create_pitch(1)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}')

model = PitchEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len)

model.load_state_dict(torch.load(
    'src/model/pretrain/pitch_pretrain.pt', 
    map_location=device)["model_state_dict"])

model.to(device)

lr = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
#############################################################
#############################################################
#############################################################

criterion = nn.CrossEntropyLoss(ignore_index = 0)

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

folder_path = f'src/train/out/pitch'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def run(model=model, train_loader=train_loader, val_loader=val_loader, epochs=num_epochs, device=device):
    log =''
    model.to(device)
    model.train()

    best_val_acc = -1
    best_val_loss = 1000000

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
        
        for batch_idx, (_, dur, pitch) in enumerate(tqdm(train_loader, ncols=60, desc="Training")):
            
            dur = dur.to(device)
            sq_dur = dur.squeeze(0)
            pitch = pitch.to(device)
            sq_pitch = pitch.squeeze(0)
            
            inner = pitch.shape[0]
                
            for i in range(0, inner, inner_batch):
                optimizer.zero_grad()
                dur = sq_dur[i:i+inner_batch, :]
                pitch = sq_pitch[i:i+inner_batch, :]
                
                outputs = model(dur)
                outputs = outputs[:,7:]
                output_ids = torch.argmax(outputs, dim=2)
                pitch_districution = pitch_prob(pitch, vocab_size)

                loss = criterion(outputs.reshape(-1, vocab_size), pitch.reshape(-1))
                
                loss.backward()

                optimizer.step()
                # scheduler.step()
                total, correct, acc = noneAR_token_accuracy(output_ids, pitch)
                
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
                log += f'{output_ids[:3,:10]}\n'
                log += f'{pitch[:3,:10]}\n'
                save_log(log)

        train_loss.append(total_train_loss / inner_cnt)
        train_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: {scheduler.get_last_lr()[0]:.12f}')
        print(f"SAMPLE TRAIN OUTPUT : ")
        print(output_ids[:5,:20])
        print(pitch[:5,:20])
        scheduler.step()
        log += f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: {scheduler.get_last_lr()[0]:.12f}\n'
        log += f'{output_ids[:5,:20]}\n'
        log += f'{pitch[:5,:20]}\n'
        save_log(log)
        scheduler.step()
        # continue
    
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        inner_cnt = 0
        with torch.no_grad():
            for batch_idx, (_, dur, pitch) in enumerate(tqdm(val_loader, ncols=60, desc="Validating")):

                dur = dur.to(device)
                sq_dur = dur.squeeze(0)
                pitch = pitch.to(device)
                sq_pitch = pitch.squeeze(0)
                
                inner = pitch.shape[0]
                
                for i in range(0, inner, inner_batch):
                    dur = sq_dur[i:i+inner_batch, :]
                    pitch = sq_pitch[i:i+inner_batch, :]

                    outputs = model(dur)
                    outputs = outputs[:,7:]

                    output_ids = torch.argmax(outputs, dim=2)
                    pitch_districution = pitch_prob(pitch, vocab_size)

                    loss = criterion(outputs.reshape(-1, vocab_size), pitch.reshape(-1))

                    total, correct, acc = noneAR_token_accuracy(output_ids, pitch)
                    
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

        if val_acc[-1] > best_val_acc or val_loss[-1] < best_val_loss:
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
            }, folder_path + f'/pitch_{epoch+1}_{val_loss[-1]:.4f}_{val_acc[-1]:.4f}.pt')

            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}') 
            log += f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}\n'
            save_log(log)
            
        print(f"SAMPLE PREDICT OUTPUT : ")
        print(output_ids[:5,:20])
        print(pitch[:5,:20])
        log += f"SAMPLE PREDICT OUTPUT : \n"
        log += f'{output_ids[:5,:20]}\n'
        log += f'{pitch[:5,:20]}\n'
        save_log(log)
        
            
def save_log(log):
    with open(f'{folder_path}/log.txt', "w") as file:
        file.write(log)
    return

run()
