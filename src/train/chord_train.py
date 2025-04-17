import os
import sys
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from ..utils import token_accuracy
import torch.distributed as dist
from tqdm import tqdm
from ..model.chord_model import ChordDecoder
from ..loader.chord_loader import create_dataloaders

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

def distance_label_smoothing_loss(outputs, targets, sigma=0.2, pad_idx=0):
    """
    Computes a distance-based label smoothing loss while ignoring padding tokens.
    
    Args:
        outputs: Tensor of shape (batch_size, seq_len, vocab_size) containing raw model outputs.
        targets: Tensor of shape (batch_size, seq_len) with token indices.
        sigma: Standard deviation for the Gaussian smoothing.
        pad_idx: The token index used for padding (e.g., 0).
        
    Returns:
        A scalar loss computed only over non-padding tokens.
    """
    device = outputs.device
    batch_size, seq_len, vocab_size = outputs.size()
    
    # Flatten outputs and targets to shape (batch_size*seq_len, ...)
    outputs = outputs.view(-1, vocab_size)  # (N, vocab_size)
    targets = targets.reshape(-1)              # (N,)
    # outputs.view(-1, vocab_size), target.reshape(-1)
    # Create a mask for non-padding tokens (i.e. targets that are not 0)
    non_pad_mask = (targets != pad_idx)
    if non_pad_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    # Select only the non-padding tokens
    outputs = outputs[non_pad_mask]  # (N_nonpad, vocab_size)
    targets = targets[non_pad_mask]  # (N_nonpad,)
    
    # Exclude the padding token from the smoothing distribution.
    # We remove the logit column corresponding to pad_idx.
    # (Here, pad_idx is assumed to be 0.)
    outputs = outputs[:, 1:]  # new shape: (N_nonpad, vocab_size-1)
    # Adjust targets: since the pad column is removed, subtract 1 from targets.
    targets = targets - 1

    # Now, the number of classes is reduced by one.
    num_classes = outputs.size(1)
    # Create a tensor of class indices for the remaining classes: 0, 1, ..., num_classes-1.
    classes = torch.arange(num_classes, device=device).float()  # (num_classes,)
    
    # Expand targets to (N_nonpad, 1) for broadcasting.
    targets = targets.float().unsqueeze(1)  # (N_nonpad, 1)
    
    # Compute Gaussian weights for each class relative to the target.
    # The weight for class i is given by exp(-0.5*((i - target)/sigma)**2).
    weights = torch.exp(-0.5 * ((classes - targets) / sigma) ** 2)  # (N_nonpad, num_classes)
    
    # Normalize the weights to form valid probability distributions.
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    # Compute log probabilities from the outputs.
    log_probs = F.log_softmax(outputs, dim=1)
    
    # Compute the final loss as the mean negative log likelihood under the smoothed target distribution.
    loss = -(weights * log_probs).sum(dim=1).mean()
    return loss

#############################################################
#############################################################
#############################################################
batch_size = 12
num_epochs = 500

# TODO, 512 VER F Setting
vocab_size = 136  # size of the vocabulary
d_model = 512  # dimension of model
num_layers = 6  # number of decoder layers
num_heads = 8  # number of attention heads
d_ff = d_model*4  # dimension of feed-forward network
max_seq_len = 768
dropout = 0.1  # dropout rate
model = ChordDecoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout)
state_dict = torch.load('src/model/pretrain/chord_pretrain.pt', map_location=device)
if 'fixed_proj' in state_dict:
    del state_dict['fixed_proj']
model.load_state_dict(state_dict, strict=True)
train_loader, val_loader = create_dataloaders(batch_size, '')
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}')


# TODO
# You can modify it accordingly to suit your dataset.
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: max(0.95 ** epoch, 1e-6))
# criterion = nn.CrossEntropyLoss(ignore_index = 0)
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

folder_path = f'src/train/out/chord'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def run(model=model, train_loader=train_loader, val_loader=val_loader, epochs=num_epochs, device=device):
    log = ''
    model.to(device)
    model.train()

    best_val_loss = 10000000
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
        
        epoch_loss = []
        
        for batch_idx, seq in enumerate(tqdm(train_loader, ncols=60, desc="Training")):
            optimizer.zero_grad()

            seq = seq.to(device)
            input = seq[:,:-1] 
            target = seq[:,1:]

            outputs = model(input)
            output_ids = torch.argmax(outputs, dim=2)
            # loss = criterion(outputs.view(-1, vocab_size), target.reshape(-1))
            loss = distance_label_smoothing_loss(outputs, target)
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 1000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch}], Step [{batch_idx}], LR: {current_lr:.6f}, Loss: {loss.item():.4f}")
                log += f"Epoch [{epoch}], Step [{batch_idx}], LR: {current_lr:.6f}, Loss: {loss.item():.4f}\n"
                save_log(log)
                

            total, correct, acc = token_accuracy(output_ids, target, eos_id=1)
            
            total_cnt += total
            total_correct += correct
            total_acc += acc

            total_train_loss += loss.item()
            epoch_loss.append(loss.item())

        # save_epoch_loss(epoch, epoch_loss)
        scheduler.step()
        train_loss.append(batch_size * total_train_loss / len(train_loader.dataset))
        train_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: {scheduler.get_last_lr()[0]:.12f}')
        log += f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: {scheduler.get_last_lr()[0]:.12f}\n'
        log += f'Target : {target[:5,:30]}\n'
        log += f'Output : {output_ids[:5,:30]}\n'
        save_log(log)
        # continue
    
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        with torch.no_grad():
            for batch_idx, seq in enumerate(tqdm(val_loader, ncols=60, desc="Validating")):

                seq = seq.to(device)
                input = seq[:,:-1]
                target = seq[:,1:]

                outputs = model(input)
                output_ids = torch.argmax(outputs, dim=2)
                # loss = criterion(outputs.view(-1, vocab_size), target.reshape(-1))
                loss = distance_label_smoothing_loss(outputs, target)
                total, correct, acc = token_accuracy(output_ids, target, eos_id=1)
                
                total_cnt += total
                total_correct += correct
                total_acc += acc
                
                total_val_loss += loss.item()

        val_loss.append(batch_size * total_val_loss / len(val_loader.dataset))
        val_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}')
        log += f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}\n'
        log += f'Target : {target[:5,:30]}\n'
        log += f'Output : {output_ids[:5,:30]}\n'
        save_log(log)

        with open(folder_path + '/train_loss.pkl', 'wb') as file:
            pickle.dump(train_loss, file)
        with open(folder_path + '/val_loss.pkl', 'wb') as file:
            pickle.dump(val_loss, file)
        with open(folder_path + '/train_acc.pkl', 'wb') as file:
            pickle.dump(train_acc, file)
        with open(folder_path + '/val_acc.pkl', 'wb') as file:
            pickle.dump(val_acc, file)

        if (val_loss[-1] < best_val_loss or val_acc[-1] > best_val_acc):
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
            # if epoch > 100 or epoch % 10 == 0:
            torch.save(model.state_dict(), folder_path + f'/chord_{epoch+1}_{val_loss[-1]:.4f}_{val_acc[-1]:.4f}.pt')
            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}')
            log += f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}\n'
            save_log(log)

def save_log(log):
    with open(f'src/train/out/chord/log.txt', "w") as file:
        file.write(log)
    return

run()
