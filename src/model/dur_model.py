import math
import torch
import torch.nn as nn
from .transformer import PositionalEncoding, TransformerEncoder


class DurEncoder(nn.Module):
    """
    A complete encoder-only Transformer model: embeddings + encoder stack + final projection.
    """
    def __init__(self,
                 dur_size,
                 vocab_size=101,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 num_layers=6,
                 max_seq_len=768,
                 dropout=0.1,
                 pad_token_id=0,
                 mode=''):
        super(DurEncoder, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.mode = mode
        self.dur_size = dur_size

        self.onset_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.rel_embedding = nn.EmbeddingBag(24, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)

        self.dur_fc_out = nn.Linear(d_model,self.dur_size)  # for LM or token-level tasks
        self.mea_fc_out = nn.Linear(d_model,65)  # for LM or token-level tasks

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

        encoder_in = torch.zeros(B, S, self.d_model).to(device)
        
        rel_embed = self.rel_embedding(x[:, 0].unsqueeze(1))
        onset_embed = self.onset_embedding(x[:, 1:])
        
        encoder_in[:, 0, :] = rel_embed
        encoder_in[:, 1:, :] = onset_embed
        
        # 1) Create mask
        mask = self.make_padding_mask(x).to(device)  # (B, 1, 1, S)
            
        # 2) Embed + pos encode
        x = encoder_in * math.sqrt(self.d_model)  # (B, S, d_model)
        x = self.pos_encoding(x)

        # 3) Encoder forward
        x = self.encoder(x, mask=mask)  # (B, S, d_model)

        # 4) Final projection
        m_logits = self.mea_fc_out(x[:, 0, :]).unsqueeze(1)
        d_logits = self.dur_fc_out(x[:, 1:, :])         # (B, S, vocab_size)
        return m_logits, d_logits