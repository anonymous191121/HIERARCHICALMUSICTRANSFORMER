import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import PositionalEncoding, TransformerDecoder

class RotateEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Create a fixed rotate embedding.
        Each token's embedding is built from d_model/2 2D blocks:
            block_k(i) = [cos(i*theta + phi_k), sin(i*theta + phi_k)]
        where theta = 2π/vocab_size and phi_k are distinct phase offsets.
        The resulting vector is normalized to have norm 1.
        """
        super(RotateEmbedding, self).__init__()
        assert d_model % 2 == 0, "d_model must be even"
        num_blocks = d_model // 2
        theta = 2 * math.pi / vocab_size
        
        # Instead of using endpoint=False, we manually adjust the end value:
        phi = torch.linspace(0, 2 * math.pi - (2 * math.pi / num_blocks), steps=num_blocks)
        
        # Compute angles for each token (shape: [vocab_size, num_blocks])
        indices = torch.arange(vocab_size, dtype=torch.float).unsqueeze(1)
        angles = indices * theta + phi  # Each row corresponds to token i
        
        # Compute cosine and sine components for each block
        embeds_cos = torch.cos(angles)
        embeds_sin = torch.sin(angles)
        
        # Concatenate to form the full embedding vector (vocab_size x d_model)
        embeddings = torch.cat([embeds_cos, embeds_sin], dim=1)
        
        # Each block is unit norm, so the full vector has norm sqrt(num_blocks); normalize to unit norm.
        embeddings = embeddings / math.sqrt(num_blocks)
        
        # Register the embeddings as a buffer so they remain fixed (non-trainable)
        self.register_buffer("embeddings", embeddings)

    def forward(self, x):
        # x is a tensor of token indices; simply index into the fixed embedding matrix.
        return self.embeddings[x]

class ChordDecoder(nn.Module):
    """
    Decoder-only Transformer: embeddings + decoder stack + final projection.
    No cross-attention, just causal self-attention.
    """
    def __init__(self,
                 vocab_size,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 num_layers=6,
                 max_seq_len=768,
                 dropout=0.1,
                 pad_token_id=0):
        super(ChordDecoder, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        self.embedding = RotateEmbedding(vocab_size, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers, dropout, cross_attn=False)

        self.fc_out = nn.Linear(d_model, vocab_size)  # for LM
        self.vocab_size = vocab_size

    def make_causal_mask(self, seq_len, device='cpu'):
        """
        Causal mask: shape (1, seq_len, seq_len),
        with True in upper triangle => blocked.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        return mask.unsqueeze(0)  # (1, seq_len, seq_len)

    def forward(self, x):
        """
        x: (B, S)
        returns: (B, S, vocab_size)
        """
        device = x.device
        B, S = x.shape

        # 1) Causal mask for self-attn
        causal_mask = self.make_causal_mask(S, device=device)  # (1, S, S)
        causal_mask = causal_mask.unsqueeze(1).expand(B, -1, -1, -1)  # (B, 1, S, S)

        # 2) Embed + pos encode
        x = self.embedding(x) * math.sqrt(self.d_model)  # (B, S, d_model)
        
        # 3) Pass through decoder
        x = self.decoder(x, enc_out=None, self_mask=causal_mask, cross_mask=None)  # (B, S, d_model)

        # 4) Final projection
        logits = self.fc_out(x)  # (B, S, vocab_size)
        return logits
    
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        eos_token_id: int = None
    ):
        """
        Greedy generation for a decoder-only Transformer.
        
        Args:
            input_ids (Tensor): shape (B, S) of token indices to start from.
            max_length (int): maximum number of new tokens to generate.
            eos_token_id (int): optional, stop generation if this token is produced.
        
        Returns:
            Tensor of shape (B, S + new_tokens) with generated token IDs.
        """
        device = input_ids.device

        for step in range(max_length):
            # 1) Forward pass: (B, seq_len, vocab_size)
            logits = self.forward(input_ids)
            
            # 2) Get the predicted distribution at the last time step
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)
            
            # 3) Greedy pick of the next token
            next_tokens = torch.argmax(next_token_logits, dim=-1)  # (B,)
            
            # 4) Append next token to the sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)  # (B, seq_len+1)
            
            # 5) [Optional] Stop if every sequence just generated <EOS>
            if eos_token_id is not None:
                # Check whether all new tokens == eos_token_id
                if torch.all(next_tokens.eq(eos_token_id)):
                    break

        return input_ids
    
    def sampling(
        self,
        input_ids: torch.Tensor,
        max_length: int = 766,
        eos_token_id: int = 1,
        top_k: int = 3,
        temperature: float = 1.0
    ):
        """
        Top-k sampling generation for a decoder-only Transformer with temperature scaling.
        
        Args:
            input_ids (Tensor): shape (B, S) of token indices to start from.
            max_length (int): maximum number of new tokens to generate.
            eos_token_id (int): optional, stop generation if this token is produced.
            top_k (int): number of top tokens to consider for sampling.
            temperature (float): scaling factor for logits (1.0 means no scaling).
        
        Returns:
            Tensor of shape (B, S + new_tokens) with generated token IDs.
        """
        device = input_ids.device

        for step in range(max_length):
            # 1) Forward pass: (B, seq_len, vocab_size)
            logits = self.forward(input_ids)
            
            # 2) Get the predicted distribution at the last time step and apply temperature scaling
            next_token_logits = logits[:, -1, :] / temperature
            
            # 3) Top-k filtering: Get top-k logits and indices
            topk_logits, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
            # Convert top-k logits to probabilities
            topk_probs = torch.softmax(topk_logits, dim=-1)
            # Sample one token from the top-k distribution
            sampled_indices = torch.multinomial(topk_probs, num_samples=1).squeeze(1)
            # Retrieve the actual token IDs from the top-k indices
            next_tokens = topk_indices.gather(1, sampled_indices.unsqueeze(1)).squeeze(1)
            
            # 4) Append next token to the sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)  # (B, seq_len+1)
            
            # 5) [Optional] Stop if every sequence just generated <EOS>
            if eos_token_id is not None:
                if torch.all(next_tokens.eq(eos_token_id)):
                    break

        return input_ids