import math
import torch
import torch.nn as nn
from .transformer import PositionalEncoding, TransformerEncoder

class PitchEncoder(nn.Module):
    """
    A complete encoder-only Transformer model: embeddings + encoder stack + final projection.
    """
    def __init__(self,
                 vocab_size,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 num_layers=6,
                 max_seq_len=512,
                 dropout=0.1,
                 pad_token_id=0,
                 mode=''):
        super(PitchEncoder, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.mode = mode
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.chord_embedding = nn.Embedding(16,d_model, padding_idx=0)
        
        self.rel_pos_embedding = nn.Embedding(24, d_model)
            
        self.inst_embedding = nn.Embedding(133, d_model, padding_idx=0)
        
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def make_padding_mask(self, x):
        """
        x: (B, S)
        returns mask: (B, 1, 1, S) with True where x == pad_token_id
        """
        return (x == self.pad_token_id).unsqueeze(1).unsqueeze(2)

    def forward(self, x):
        """
        x: (B, S) integer token IDs
        returns: (B, S, vocab_size)
        """
        device = x.device
        B, S = x.shape

        chord_embed = self.chord_embedding(x[:,0:5])
        rel_pos_embed = self.rel_pos_embedding(x[:,5]).unsqueeze(1)
        inst_embed = self.inst_embedding(x[:,6]).unsqueeze(1)
        dur_embed = self.embedding(x[:,7:])

        mask = self.make_padding_mask(x).to(device)  # (B, 1, 1, S)
        x = torch.cat([chord_embed, rel_pos_embed, inst_embed, dur_embed], dim=1)
            
        x = x * math.sqrt(self.d_model)  # (B, S, d_model)
        x = self.pos_encoding(x)

        x = self.encoder(x, mask=mask)  # (B, S, d_model)

        logits = self.fc_out(x)         # (B, S, vocab_size)
        return logits
    
    def sampling(self, logit, k=5, temperature=1.0):
        """
        logits: Tensor of shape [batch, seq, vocab_size]
        k: number of top tokens to consider
        temperature: scaling factor for logits (lower temperature makes distribution sharper)
        """
        batch, seq, vocab_size = logit.shape
        output_pred = torch.zeros(batch, seq)

        logits = logit
        # Apply temperature scaling
        logits = logits / temperature
        batch, seq, vocab_size = logits.shape
        
        # Get top k tokens and their logits along the vocabulary dimension
        topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
        
        # Convert top-k logits to probabilities
        topk_probs = torch.softmax(topk_logits, dim=-1)
        
        # Reshape for sampling: merge batch and sequence dimensions
        topk_probs = topk_probs.view(-1, k)
        topk_indices = topk_indices.view(-1, k)
        
        # Sample one token index from the top-k probabilities
        sampled_topk_idx = torch.multinomial(topk_probs, num_samples=1)
        
        # Map sampled index back to original vocabulary indices
        sampled_tokens = topk_indices.gather(1, sampled_topk_idx)
        
        output_pred[:, :] = sampled_tokens.view(batch, seq)
        
        # Reshape back to [batch, seq, ]
        return output_pred