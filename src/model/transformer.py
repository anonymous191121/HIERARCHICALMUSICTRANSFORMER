import math
import torch
import torch.nn as nn
import torch.nn.functional as F
###############################################################################
# Common Building Blocks
###############################################################################

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings via sine/cosine waves.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a (max_len, d_model) positional encoding matrix
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # Register as a buffer so it's not a parameter
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (B, S, d_model)
        returns: (B, S, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding: broadcast over batch dimension
        x = x + self.pe[:, :seq_len, :]
        return x


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    query: (B, num_heads, S_q, d_k)
    key:   (B, num_heads, S_k, d_k)
    value: (B, num_heads, S_k, d_v)
    mask:  (B, 1, S_q, S_k) where True indicates positions to mask.

    returns:
      out: (B, num_heads, S_q, d_v)
      attn_weights: (B, num_heads, S_q, S_k)
    """
    d_k = query.size(-1)
    # (B, num_heads, S_q, d_k) x (B, num_heads, d_k, S_k) -> (B, num_heads, S_q, S_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # True in mask => set scores to -inf
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, S_q, S_k)
    # Weighted sum: (B, num_heads, S_q, S_k) x (B, num_heads, S_k, d_v) -> (B, num_heads, S_q, d_v)
    out = torch.matmul(attn_weights, value)
    return out, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module that can be used for self-attn or cross-attn.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv, mask=None):
        """
        x_q: (B, S_q, d_model)
        x_kv: (B, S_k, d_model)
        mask: (B, 1, S_q, S_k) or broadcastable shape. True => masked

        returns:
          out: (B, S_q, d_model)
          attn_weights: (B, num_heads, S_q, S_k)
        """
        B, S_q, _ = x_q.shape
        B, S_k, _ = x_kv.shape

        # 1) Project to Q, K, V
        Q = self.w_q(x_q)   # (B, S_q, d_model)
        K = self.w_k(x_kv)  # (B, S_k, d_model)
        V = self.w_v(x_kv)  # (B, S_k, d_model)

        # 2) Reshape for multi-head
        # (B, S_q, num_heads, d_k) -> (B, num_heads, S_q, d_k)
        Q = Q.view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)

        # 3) Scaled dot-product attention
        out, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        # out: (B, num_heads, S_q, d_k)

        # 4) Reshape heads back
        out = out.transpose(1, 2).contiguous()  # (B, S_q, num_heads, d_k)
        out = out.view(B, S_q, self.d_model)    # (B, S_q, d_model)

        # 5) Final linear
        out = self.w_o(out)                     # (B, S_q, d_model)

        return out, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    FFN with two linear layers and a ReLU in between.
    """
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.GELU()

    def forward(self, x):
        """
        x: (B, S, d_model)
        returns: (B, S, d_model)
        """
        return self.linear2(self.relu(self.linear1(x)))


###############################################################################
# (1) Encoder (stack)
###############################################################################

class EncoderBlock(nn.Module):
    """
    Single Encoder block with self-attention + feed-forward (+ residuals, layernorm).
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (B, S, d_model)
        mask: (B, 1, S, S) or broadcastable; True => masked
        returns: (B, S, d_model)
        """
        # 1) Self-attention
        attn_out, _ = self.self_attn(x, x, mask=mask)  # Q=K=V=x
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # 2) Feed-forward
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """
    N-layer encoder stack.
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x: (B, S, d_model)
        mask: (B, 1, S, S)
        returns: (B, S, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        return x


###############################################################################
# (2) Encoder-Only Model
###############################################################################

class EncoderOnlyModel(nn.Module):
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
                 pad_token_id=0):
        super(EncoderOnlyModel, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)

        self.fc_out = nn.Linear(d_model, vocab_size)  # for LM or token-level tasks

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

        # 1) Create mask
        mask = self.make_padding_mask(x).to(device)  # (B, 1, 1, S)

        # 2) Embed + pos encode
        x = self.embedding(x) * math.sqrt(self.d_model)  # (B, S, d_model)
        x = self.pos_encoding(x)

        # 3) Encoder forward
        x = self.encoder(x, mask=mask)  # (B, S, d_model)

        # 4) Final projection
        logits = self.fc_out(x)         # (B, S, vocab_size)
        return logits


###############################################################################
# (3) Decoder (stack)
###############################################################################

class DecoderBlock(nn.Module):
    """
    Single Decoder block:
      - Masked self-attn
      - Cross-attn
      - Feed-forward (+ residual + layernorm)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out=None, self_mask=None, cross_mask=None):
        """
        x: (B, tgt_len, d_model) - decoder input embeddings
        enc_out: (B, src_len, d_model) - from encoder (if cross-attn needed)
        self_mask: (B, 1, tgt_len, tgt_len) - for causal or padding
        cross_mask: (B, 1, tgt_len, src_len) - for cross-attn
        returns: (B, tgt_len, d_model)
        """
        # 1) Masked self-attn
        self_attn_out, _ = self.self_attn(x, x, mask=self_mask)
        x = x + self.dropout1(self_attn_out)
        x = self.norm1(x)

        # 2) Cross-attn (only if enc_out is provided, otherwise skip)
        if enc_out is not None:
            cross_attn_out, _ = self.cross_attn(x, enc_out, mask=cross_mask)
            x = x + self.dropout2(cross_attn_out)
            x = self.norm2(x)

        # 3) Feed-forward
        ffn_out = self.ffn(x)
        x = x + self.dropout3(ffn_out)
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    """
    N-layer decoder stack.
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1, cross_attn=True):
        super(TransformerDecoder, self).__init__()
        # If cross_attn=False, we skip cross-attention usage
        # (this can be used for "decoder-only" setups).
        self.cross_attn = cross_attn
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out=None, self_mask=None, cross_mask=None):
        """
        x: (B, tgt_len, d_model)
        enc_out: (B, src_len, d_model) (optional for cross-attn)
        self_mask: (B, 1, tgt_len, tgt_len)
        cross_mask: (B, 1, tgt_len, src_len)

        returns: (B, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x,
                      enc_out=enc_out if self.cross_attn else None,
                      self_mask=self_mask,
                      cross_mask=cross_mask if self.cross_attn else None)
        x = self.norm(x)
        return x


###############################################################################
# (4) Decoder-Only Model (GPT-like)
###############################################################################

  
class DecoderOnlyModel(nn.Module):
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
                 max_seq_len=512,
                 dropout=0.1,
                 pad_token_id=0):
        super(DecoderOnlyModel, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Initialize embedding weights with a rotated pattern
        # initial_embedding = create_rotated_embedding_matrix(vocab_size, d_model, 0.1)
        # with torch.no_grad():
        #     self.embedding.weight.copy_(initial_embedding)
        
        # self.embedding = RotateEmbedding(vocab_size, d_model)
        
        
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        # Use cross_attn=False so that the DecoderBlock skip cross-attn
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers, dropout, cross_attn=False)

        self.fc_out = nn.Linear(d_model, vocab_size)  # for LM
        self.vocab_size = vocab_size
        
        if vocab_size != d_model:
            fixed_proj = torch.randn(vocab_size, d_model)
            self.register_buffer('fixed_proj', fixed_proj)

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
        
        # x = F.one_hot(x, num_classes=self.vocab_size).float()
        # if self.vocab_size != self.d_model:
        #     x = x @ self.fixed_proj  # (B, S, d_model)

        x = self.pos_encoding(x)

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
        max_length: int = 50,
        eos_token_id: int = None,
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


###############################################################################
# (5) Full Transformer (Encoder–Decoder)
###############################################################################

class FullTransformer(nn.Module):
    """
    Standard Encoder–Decoder Transformer.
    """
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 max_seq_len=512,
                 dropout=0.1,
                 pad_token_id=0):
        super(FullTransformer, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Embeddings + positional encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)
        self.tgt_pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)

        # Encoder
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)

        # Decoder (cross_attn=True)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_decoder_layers, dropout, cross_attn=True)

        # Final projection to target vocab
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_src_padding_mask(self, src):
        """
        src: (B, src_len)
        returns: (B, 1, 1, src_len) True where pad => masked
        """
        return (src == self.pad_token_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_padding_mask(self, tgt):
        """
        tgt: (B, tgt_len)
        returns: (B, 1, 1, tgt_len) True where pad => masked
        """
        return (tgt == self.pad_token_id).unsqueeze(1).unsqueeze(2)

    def make_causal_mask(self, size, device='cpu'):
        """
        shape (1, size, size) => True in upper triangle => masked
        """
        mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
        return mask.unsqueeze(0)  # (1, size, size)

    def forward(self, src, tgt):
        """
        src: (B, src_len)
        tgt: (B, tgt_len)
        returns: (B, tgt_len, tgt_vocab_size)
        """
        device = src.device
        B, src_len = src.shape
        B, tgt_len = tgt.shape

        # 1) Create source mask (padding)
        src_mask = self.make_src_padding_mask(src).to(device)   # (B, 1, 1, src_len)

        # 2) Create target padding + causal masks
        tgt_pad_mask = self.make_tgt_padding_mask(tgt).to(device) # (B, 1, 1, tgt_len)
        tgt_causal_mask = self.make_causal_mask(tgt_len, device=device)  # (1, tgt_len, tgt_len)

        # Combine them: broadcast causal mask => (B, 1, tgt_len, tgt_len),
        # then logical OR with the padding mask.
        tgt_mask = tgt_pad_mask | tgt_causal_mask.unsqueeze(1)   # (B, 1, tgt_len, tgt_len)

        # 3) Embed + pos encode
        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)  # (B, src_len, d_model)
        src_embed = self.src_pos_enc(src_embed)

        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  # (B, tgt_len, d_model)
        tgt_embed = self.tgt_pos_enc(tgt_embed)

        # 4) Encoder
        enc_out = self.encoder(src_embed, mask=src_mask)  # (B, src_len, d_model)

        # 5) Decoder cross-attn mask: we want to ignore source padding => broadcast
        cross_mask = src_mask.expand(-1, 1, tgt_len, -1)  # (B, 1, tgt_len, src_len)

        dec_out = self.decoder(tgt_embed, enc_out, self_mask=tgt_mask, cross_mask=cross_mask)
        # dec_out: (B, tgt_len, d_model)

        # 6) Final projection
        logits = self.fc_out(dec_out)  # (B, tgt_len, tgt_vocab_size)
        return logits
