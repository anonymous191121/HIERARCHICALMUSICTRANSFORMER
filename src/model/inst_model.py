import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerEncoder, TransformerDecoder

class InstTransformer(nn.Module):
    def __init__(self, book_size=8, book_vocab_size=[], d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=5000, dropout=0.1):
        super(InstTransformer, self).__init__()
        self.book_size = book_size
        self.book_vocab_size = book_vocab_size

        self.pad_token_id = 0
        self.inst_book_embed = nn.ModuleList([nn.Embedding(self.book_vocab_size[i], 64, padding_idx=0) for i in range(8)])
        
        self.chord_size = 140
        self.chord_embed = nn.Embedding(self.chord_size, 512, padding_idx=0)
        self.chord_encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.inst_decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers, dropout, cross_attn=True)
        
        self.d_model = d_model
        self.out = nn.ModuleList([nn.Linear(d_model, self.book_vocab_size[i]) for i in range(self.book_size)])
        
        self.criterion = nn.CrossEntropyLoss(ignore_index = 0)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, infer=False):
        device = src.device
        src_mask = self.make_src_padding_mask(src)  # Generate source mask

        src = self.chord_embed(src)
        encoder_embed = src * math.sqrt(self.d_model)
        # 4) Encoder
        enc_out = self.chord_encoder(encoder_embed, mask=src_mask)

        
        
        bsz, d_l, _ = tgt.shape
        decoder_embed = torch.zeros(bsz, d_l, 512)
        
        for i in range(self.book_size):
            tgt_book = self.inst_book_embed[i](tgt[:, :, i])
            decoder_embed[:, :, i*64:(i+1)*64] = tgt_book
        
        decoder_embed = decoder_embed * math.sqrt(self.d_model)
        
        tgt_pad_mask = self.make_tgt_padding_mask(tgt[:, :, 0]).to(device)
        tgt_causal_mask = self.make_causal_mask(d_l, device=device)
        tgt_mask = tgt_pad_mask | tgt_causal_mask.unsqueeze(1)
        
        cross_mask = src_mask.expand(-1, 1, d_l, -1).to(device)
        dec_out = self.inst_decoder(decoder_embed.to(device), enc_out, self_mask=tgt_mask.to(device), cross_mask=cross_mask)
        
        output_logit_container = []

        for i in range(self.book_size):
            output = self.out[i](dec_out)
            output_logit_container.append(output)
            
        return output_logit_container
    
    def ce_loss(self, logit_container, target):
        loss_container = []
        total_loss = 0
        
        for i in range(self.book_size):
            loss = self.criterion(logit_container[i].view(-1, self.book_vocab_size[i]), target[:,:,i].reshape(-1))
            loss_container.append(loss)
            total_loss += loss
        return loss_container, total_loss/self.book_size
    
    # def inference_sampling(self, src, tgt, k=3, max_len=768, temperature=1.0):
    #     logit_container = self.forward(src, tgt)
    #     batch_size = src.size(0)
    #     max_len = tgt.size(1)
        
    #     inference = torch.zeros(batch_size, max_len, self.book_size)
        
    #     for i in range(self.book_size):
    #         book_logit = logit_container[i]
    #         print(book_logit.shape)
    #         scaled_logit = book_logit / temperature
    #         next_token = self.top_k_sampling(scaled_logit, k=k)
    #         inference[:,:,i] = next_token
    #     inference = inference.to(src.device)
       
    #     return inference

    def inference_sampling(self, src, tgt, k=3, temperature=1.0):
        logit_container = self.forward(src, tgt)
        inference = self.top_k_sampling(logit_container, k=k, temperature=temperature)
       
        return inference
    
    
    def top_k_sampling(self, logit_container, k=3, temperature=1.0):
        """
        logits: Tensor of shape [batch, seq, vocab_size]
        k: number of top tokens to consider
        temperature: scaling factor for logits (lower temperature makes distribution sharper)
        """
        batch, seq, vocab_size = logit_container[0].shape
        output_pred = torch.zeros(batch, seq, 8)
        
        for i in range(8):
            logits = logit_container[i]
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
            
            output_pred[:, :, i] = sampled_tokens.view(batch, seq)
        
        # Reshape back to [batch, seq, ]
        return output_pred
    
    def generate(self, src, tgt, k=5, temperature=1.0, prompt_len=0):
        device = tgt.device
        bsz, seq = src.shape

        for _ in range(seq-prompt_len+1):
            output_container = self.forward(src, tgt)
            pred = self.top_k_sampling(output_container, k=k, temperature=temperature)

            tgt = torch.cat([tgt, pred[:, -1, :].unsqueeze(1).long().to(device)], dim=1)
        return tgt

    
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