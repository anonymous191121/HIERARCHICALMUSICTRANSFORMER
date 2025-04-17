import torch
import torch.nn.functional as F

pit2alphabet = ['C', 'd', 'D', 'e', 'E', 'F', 'g', 'G', 'a', 'A', 'b', 'B']
char2pit = {x: id for id, x in enumerate(pit2alphabet)}


def pit2str(x):
    octave = x // 12
    octave = octave - 1 if octave > 0 else 'O'
    rel_pit = x % 12
    return pit2alphabet[rel_pit] + str(octave)


def str2pit(x):
    rel_pit = char2pit[x[0]]
    octave = (int(x[1]) if x[1] != 'O' else -1) + 1
    return octave * 12 + rel_pit


def int2char(x):
    if x <= 9:
        return str(x)
    if x <= 35:
        return chr(ord('a') + (x - 10))
    if x < 62:
        return chr(ord('A') + (x - 36))
    assert False, f'invalid number {x}'


def char2int(c):
    num = ord(c)
    A, a, Z, z = ord('A'), ord('a'), ord('Z'), ord('z')
    if num >= a and num <= z:
        return 10 + num - a
    elif num >= A and num <= Z:
        return 36 + num - A
    elif num >= ord('0') and num <= ord('9'):
        return num - ord('0')
    assert False, f'invalid character {c}'


def pos2str(ons):
    if ons < 62:
        return 'p' + int2char(ons)
    return 'P' + int2char(ons - 62)


def bom2str(ons):
    if ons < 62:
        return 'm' + int2char(ons)
    return 'M' + int2char(ons - 62)


def dur2str(ons):
    if ons < 62:
        return 'r' + int2char(ons)
    return 'R' + int2char(ons - 62)


def trk2str(ons):
    if ons < 62:
        return 't' + int2char(ons)
    return 'T' + int2char(ons - 62)


def ins2str(ons):  # 0 - 128
    if ons < 62:
        return 'x' + int2char(ons)
    ons -= 62
    if ons < 62:
        return 'X' + int2char(ons)
    ons -= 62
    if ons < 62:
        return 'y' + int2char(ons)
    return 'Y' + int2char(ons - 62)


def ispitch(x):  # judge if a event str is a pitch (CO - B9)
    return len(x) == 2 and x[0] in char2pit and (x[1] == 'O' or x[1].isdigit())


def ison(x):  # judge if a event str is a bpe token
    if len(x) % 2 != 0 or len(x) < 2:
        return False
    for i in range(0, len(x), 2):
        if not ispitch(x[i:i + 2]):
            return False

    return True


def bpe_str2int(x):
    if len(x) == 2:
        return (0, str2pit(x))
    res = []
    for i in range(0, len(x), 2):
        res.append(str2pit(x[i:i + 2]))
    return (1,) + tuple(sorted(res))


def sort_tok_str(x):
    c = x[0].lower()
    if c in ('r', 't', 'x', 'y'):
        #         if x in ('RZ', 'TZ', 'YZ'):
        #             return (c if c != 'y' else 'x', False, -1)
        return (c, not x[0].islower(), char2int(x[1]))
    if c in ('m', 'p'):
        return (c, not x[0].islower(), char2int(x[1]))

    if c == 'h':
        return (c, char2pit[x[1]] if x[1] != 'N' else 12, x[2:])
    if c == 'n':
        return ('w', x)
    if ison(x):
        return ('a',) + bpe_str2int(x)

    return ('A', x[1] != 'b', x[1] != 'p', x[1] != 'e')


def token_accuracy(infer, target, eos_id):
    if infer.shape[0] != target.shape[0]:
        raise Exception
    correct = 0
    total = 0

    for i in range(infer.shape[0]):
        zero_target = torch.where(target[i] == eos_id)[0]
        zero_target = zero_target[-1].item() if len(zero_target) > 0 else target[i].shape[0]
        infer_len = infer.shape[1]
        max_zero = zero_target
        
        if max_zero < infer_len:
            infer_slice = infer[i][:max_zero]
            target_slice = target[i][:max_zero]
            
            correct += (infer_slice == target_slice).sum().item()
            total += max_zero
        else:
            infer_slice = infer[i][:infer_len]
            target_slice = target[i][:infer_len]
            
            correct += (infer_slice == target_slice).sum().item()
            total += max_zero

    return total, correct, correct/total

def calculate_accuracy(predictions_, targets_, pad_token=0, book_size=8):
    device = predictions_.device
    predictions_ = predictions_.permute(0,2,1).to(device)
    targets_ = targets_.permute(0,2,1).to(device)
    book_acc = []
    total_acc = 0
    for _ in range(book_size):
        predictions = predictions_[:,_,:].reshape(-1)
        targets = targets_[:,_,:].reshape(-1)

        mask = targets != pad_token
       
        correct = (predictions == targets) & mask
        accuracy = correct.sum().item() / mask.sum().item()
        book_acc.append(accuracy)
        total_acc += accuracy
    
    return total_acc, book_acc

def noneAR_token_accuracy(output, target, pad_token=0):
    """
    output: 모델의 예측 결과, shape [batch_size, seq_len] 또는 [batch_size, seq_len, num_classes]
    target: 정답 텐서, shape [batch_size, seq_len]
    pad_token: 패딩 토큰 값 (여기서는 0)
    
    Returns:
        accuracy: 맞춘 비율 (float)
        total_non_pad: 패딩 토큰을 제외한 전체 토큰 개수 (int)
        correct_count: 맞춘 토큰 개수 (int)
    """
    # 만약 output이 logits 형태라면, argmax를 통해 예측 토큰 인덱스로 변환
    if output.dim() == 3:
        preds = output.argmax(dim=-1)  # shape: [batch_size, seq_len]
    else:
        preds = output

    # 패딩 토큰이 아닌 위치에 대한 마스크 생성 (True: 계산에 포함)
    mask = target != pad_token

    # 예측이 정답과 일치하는지 여부 계산 (패딩이 아닌 위치에 대해서만)
    correct = (preds == target) & mask

    # 총 패딩이 아닌 토큰의 개수와 맞춘 토큰의 개수 계산
    total_non_pad = mask.sum().item()
    correct_count = correct.sum().item()

    # 정확도 계산: 맞춘 토큰 수 / 전체 패딩이 아닌 토큰 수
    accuracy = correct_count / total_non_pad if total_non_pad > 0 else 0.0

    return total_non_pad, correct_count, accuracy

def pitch_prob(x, vocab_size):
    mask = (x != 0)

    # Count non-padding tokens for each row (shape: [bsz, 1])
    non_pad_counts = mask.sum(dim=1, keepdim=True)

    # One-hot encode x (shape: [bsz, seq, vocab_size])
    one_hot = F.one_hot(x, num_classes=vocab_size).float()

    # Zero out positions corresponding to padding tokens
    one_hot = one_hot * mask.unsqueeze(-1)

    # Sum along the sequence dimension to get counts for each vocabulary token
    token_counts = one_hot.sum(dim=1)

    # Divide by the count of non-padding tokens to get uniform probabilities
    probs = token_counts / non_pad_counts
    return probs

def pit_loss(model_output, target_distribution, input_ids, padding_idx=0):
    """
    Compute the KL divergence loss between the model output and a target distribution,
    ignoring contributions from padded positions.
    
    Parameters:
      model_output (Tensor): Raw logits of shape [batch_size, seq_length, vocab_size].
      target_distribution (Tensor): Target probability distribution of shape [batch_size, vocab_size].
      input_ids (Tensor): Input token IDs of shape [batch_size, seq_length] with padding index.
      padding_idx (int): The index used for padding tokens (default: 0).
    
    Returns:
      Tensor: A scalar loss averaged over non-padding tokens.
    """
    # Expand the target distribution along the sequence dimension.
    # It changes from [batch_size, vocab_size] to [batch_size, seq_length, vocab_size].
    target_expanded = target_distribution.unsqueeze(1).expand(-1, model_output.size(1), -1)
    
    # Convert the model logits to log probabilities.
    log_probs = F.log_softmax(model_output, dim=-1)
    
    # Compute KL divergence without reduction, which gives per-element loss.
    loss_tensor = F.kl_div(log_probs, target_expanded, reduction='none')
    
    # Sum the loss over the vocab dimension to obtain a loss per token.
    loss_per_token = loss_tensor.sum(dim=-1)  # shape: [batch_size, seq_length]
    
    # Create a mask that is 1 for non-padding tokens and 0 for padding tokens.
    mask = (input_ids != padding_idx).float()  # shape: [batch_size, seq_length]
    
    # Sum the loss over non-padding tokens and normalize by the number of valid tokens.
    loss = (loss_per_token * mask).sum() / mask.sum()
    
    return loss