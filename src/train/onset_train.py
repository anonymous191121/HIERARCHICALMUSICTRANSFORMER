import os
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from ..utils import token_accuracy
from ..model.onset_model import OnsetTransformer, Onset
from ..loader.onset_loader import create_Obook

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
train_loader, val_loader = create_Obook()
print(f'Split Train : {len(train_loader)}, Val : {len(val_loader)}')

#############################################################
#############################################################
#############################################################
book_size=8
book_vocab_size = []
for i in range(book_size):
    with open(f'data/vocabs/book/onset_B{i}.json', 'r') as file:
        vocab = json.load(file)
    book_vocab_size.append((len(vocab)//2))
print(book_vocab_size)
num_epochs = 500
inner_batch = 36
d_model = 512  # dimension of model
num_layers = 6  # number of decoder layers
num_heads = 8  # number of attention heads
d_ff = d_model*4  # dimension of feed-forward network

# model = OnsetTransformer(d_model=d_model,book_size=book_size, book_vocab_size=book_vocab_size, num_heads=num_heads, d_ff=d_ff)

model = OnsetTransformer(d_model=d_model,book_size=book_size, book_vocab_size=book_vocab_size, num_heads=num_heads, d_ff=d_ff)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

#-----------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------

folder_path = f'src/train/out/onset'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# torch.set_printoptions(precision=None, edgeitems=None, linewidth=None, profile='full', sci_mode=None)

def run(model=model, train_loader=train_loader, val_loader=val_loader, epochs=num_epochs, device=device):
    log = ''
    model.to(device)
    model.train()

    best_val_loss = 1000000
    best_val_acc = 0

    train_loss = []
    val_loss = []
    
    total_train_book_loss = []
    total_train_book_acc = []
    
    total_val_book_loss = []
    total_val_book_acc = []

    for epoch in range(epochs):
        model.train()
        
        book_train_loss = [0 for _ in range(model.book_size)]
        
        book_acc = 0
        
        total_train_loss = 0

        inner_cnt = 0
        
        epoch_loss = []
        
        for batch_idx, (_, book_onset) in enumerate(tqdm(train_loader, ncols=60, desc="Training")):

            book_onset = book_onset.to(device)

            inner = book_onset.shape[0] - 1 # == measure

            encoder_input = book_onset[:-1, :, :].to(device)
            decoder_input = book_onset[1:, :, :].to(device)

            for i in range(0, inner, inner_batch):
                optimizer.zero_grad() 
                inner_en = encoder_input[i:i+inner_batch, :, :].to(device)
                inner_de = decoder_input[i:i+inner_batch, :, :].to(device)

                inner_target = decoder_input[i:i+inner_batch, :, :8].to(device)
                
                output_logit_container = model(encoder_in=inner_en, decoder_in=inner_de)

                loss_container, loss = model.ce_loss(output_logit_container, inner_target)
                
                loss.backward()
                
                epoch_loss.append(loss.item())
                
                pred = model.top_k_sampling(output_logit_container).to(device)
                
                accuracy, train_book_acc = calculate_accuracy(pred, inner_target, pad_token=0)
                
                for i in range(model.book_size):
                    book_train_loss[i] += loss_container[i].item()
                    
                optimizer.step()
                
                total_train_loss += loss.item()
                
                book_acc += accuracy

                inner_cnt += 1
            
            if batch_idx % 1000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch}], Step [{batch_idx}], LR: {current_lr:.6f}, Loss: {loss.item():.4f}, Acc: {accuracy:.6f}")
                log += f"Epoch [{epoch}], Step [{batch_idx}], LR: {current_lr:.6f}, Loss: {loss.item():.4f} Acc: {accuracy:6f}\n"
                log += f"Target : \n {inner_target[0, :4, :]}\n"
                for i in range(model.book_size):
                    log += f'BOOK {i} | {pred[0,:4, i].long()} | {train_book_acc[i]:.6f}\n'
                    
                save_log(log)
        for i in range(model.book_size):
            print(f'BOOK {i} | Train Loss: {(book_train_loss[i]/inner_cnt):.6f}')
            book_train_loss[i] = book_train_loss[i]/inner_cnt
        
        
        train_loss.append(total_train_loss/inner_cnt)
        total_train_book_acc.append(book_acc/inner_cnt)
        
        total_train_book_loss.append(book_train_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.8f}, Acc: {total_train_book_acc[-1]:8f}, LR: {scheduler.get_last_lr()[0]:.12f}')
        log += f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.8f}, Acc: {total_train_book_acc[-1]:8f}, LR: {scheduler.get_last_lr()[0]:.12f}\n'
        save_log(log)
        scheduler.step()
        
        book_val_loss = [0 for _ in range(model.book_size)]
        
        book_acc = 0
        
        total_val_loss = 0
        
        total_cnt = 0
        inner_cnt = 0
        
        epoch_loss = []

        
        with torch.no_grad(): 
            for batch_idx, (_, book_onset) in enumerate(tqdm(val_loader, ncols=60, desc="Training")):

                book_onset = book_onset.to(device)
                
                inner = book_onset.shape[0] - 1 # == measure

                encoder_input = book_onset[:-1, :, :].to(device)
                decoder_input = book_onset[1:, :, :].to(device)
                
                for i in range(0, inner, inner_batch):
                    optimizer.zero_grad()

                    inner_en = encoder_input[i:i+inner_batch, :, :].to(device)
                    inner_de = decoder_input[i:i+inner_batch, :, :].to(device)

                    inner_target = decoder_input[i:i+inner_batch, :, :8].to(device)
                    
                    output_logit_container = model(encoder_in=inner_en, decoder_in=inner_de)
                    
                    loss_container, loss = model.ce_loss(output_logit_container, inner_target)
                    
                    
                    pred = model.top_k_sampling(output_logit_container).to(device)
                    
                    accuracy, train_book_acc = calculate_accuracy(pred, inner_target, pad_token=0)
                    
                    for i in range(model.book_size):
                        book_val_loss[i] += loss_container[i].item()
                    
                    total_val_loss += loss.item()
                    
                    book_acc += accuracy

                    inner_cnt += 1
                
                if batch_idx % 1000 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch [{epoch}], Step [{batch_idx}], Loss: {loss.item():.4f}, Acc: {accuracy:.6f}")
                    log += f"Epoch [{epoch}], Step [{batch_idx}], Loss: {loss.item():.4f} Acc: {accuracy:6f}\n"
                    log += f"Target : \n {inner_target[0, :4, :]}\n"
                    for i in range(model.book_size):
                        log += f'BOOK {i} | {pred[0,:4, i].long()} | {train_book_acc[i]:.6f}\n'
                        
                    save_log(log)
            
            for i in range(model.book_size):
                print(f'BOOK {i} | Val Loss: {(book_val_loss[i]/inner_cnt):.6f}')
                book_val_loss[i] = book_val_loss[i]/inner_cnt
            
            
            val_loss.append(total_val_loss/inner_cnt)
            total_val_book_acc.append(book_acc/inner_cnt)
            
            total_val_book_loss.append(book_train_loss)
            
            print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.8f}, Acc: {total_val_book_acc[-1]/inner_cnt:8f}, LR: {scheduler.get_last_lr()[0]:.12f}')
            log += f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.8f}, Acc: {total_val_book_acc[-1]/inner_cnt:8f}, LR: {scheduler.get_last_lr()[0]:.12f}\n'
            save_log(log)
            scheduler.step()
        
        with open(folder_path + '/train_loss.pkl', 'wb') as file:
            pickle.dump(train_loss, file)
        with open(folder_path + '/val_loss.pkl', 'wb') as file:
            pickle.dump(val_loss, file)

        if val_loss[-1] < best_val_loss or best_val_acc < total_val_book_acc[-1]:
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
            if total_val_book_acc[-1] > best_val_acc:
                best_val_acc = total_val_book_acc[-1]

            # torch.save(model.state_dict(), folder_path + f'/model_{epoch+1}_{val_acc[-1]:.4f}_{test_acc[-1]:.4f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # if applicable
                # add any other states you need
            }, folder_path + f'/onset_{epoch+1}_{val_loss[-1]:.4f}_{total_val_book_acc[-1]:.4f}.pt')
            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}') 
            log += f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}\n'
            save_log(log)

def calculate_accuracy(predictions_, targets_, pad_token=0, ignore_token=4):
    # predictions: shape (batch_size, book_size, seq_len)
    # targets: shape (batch_size, book_size, seq_len)
    book_acc = []
    total_acc = 0
    for i in range(model.book_size):
        # Flatten the predictions and targets for easier comparison
        predictions = predictions_[:, :, i].reshape(-1)
        targets = targets_[:, :, i].reshape(-1)
        
        # Create a mask to exclude both the padding token and the ignore_token (e.g., 4)
        mask = (targets != pad_token) & (targets != ignore_token)
        
        # Count correct predictions for tokens that are not masked out
        correct = (predictions == targets) & mask
        
        # Avoid division by zero if mask.sum() is 0
        accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0
        book_acc.append(accuracy)
        total_acc += accuracy
    
    return total_acc/8, book_acc


def save_log(log):
    with open(f'{folder_path}/log.txt', "w") as file:
        file.write(log)
    return

run()
