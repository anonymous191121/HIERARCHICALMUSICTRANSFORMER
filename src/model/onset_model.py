import math
import torch
import torch.nn as nn
from .transformer import PositionalEncoding, TransformerEncoder, TransformerDecoder

class OnsetTransformer(nn.Module):
    """
    Standard Encoder - Decoder Transformer.
    """
    def __init__(self,
                 d_model=512,
                 book_size=8,
                 book_vocab_size=[],
                 num_heads=8,
                 d_ff=2048,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 tgt_max_len=768,
                 dropout=0.1,
                 pad_token_id=0,
                 mode = ''):
        super(OnsetTransformer, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.mode = mode
        self.book_size = book_size
        self.book_vocab_size = book_vocab_size
        self.criterion = nn.CrossEntropyLoss(ignore_index = 0)
        # Embeddings + positional encoding
        self.en_onset_embedding = nn.ModuleList([nn.Embedding(self.book_vocab_size[i], 56, padding_idx=0) for i in range(8)])
        self.en_inst_embedding = nn.Embedding(133, 16, padding_idx=0)
        self.en_rel_embedding = nn.Embedding(24, 32)
        self.en_chord_embedding = nn.Embedding(136, 16, padding_idx=0)
        
        self.de_inst_embedding = nn.Embedding(133, 128)
        self.de_rel_embedding = nn.Embedding(24, 256)
        self.de_chord_embedding = nn.Embedding(136, 128)

        self.tgt_pos_enc = PositionalEncoding(d_model, max_len=tgt_max_len)

        # Encoder
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)
        # Decoder (cross_attn=True)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_decoder_layers, dropout, cross_attn=True)

        # Final projection to target vocab
        # Include MASK and EOS index
        self.fc_out = nn.ModuleList([nn.Linear(d_model, self.book_vocab_size[i]) for i in range(8)])

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

    def forward(self, encoder_in, decoder_in):

        device = encoder_in.device
        _, e_l, _ = encoder_in.shape
        bsz, d_l, _ = decoder_in.shape
        l = d_l
        # B, src_len = src.shape
        # B, tgt_len = tgt.shape

        encoder_embed = torch.zeros(bsz, e_l, 512).to(device)
        decoder_embed = torch.zeros(bsz, d_l, 512).to(device)

        onset_seq = encoder_in[:, :, :8].to(device)
        inst_seq = encoder_in[:, :, 8].to(device)
        rel_seq = encoder_in[:, :, 9].to(device)
        c_seq = encoder_in[:, :, 11:].to(device)
        
        for i in range(8):
            book = self.en_onset_embedding[i](onset_seq[:, :, i])
            encoder_embed[:, :, 56*i:56*(i+1)] = book
        
        inst_embed = self.en_inst_embedding(inst_seq)
        encoder_embed[:, :, 56*8:56*8+16] = inst_embed
        
        rel_embed = self.en_rel_embedding(rel_seq)
        encoder_embed[:, :, 56*8+16:56*8+48] = rel_embed
        
        c_embed = self.en_chord_embedding(c_seq)
        c_embed = c_embed.mean(dim=2)
        encoder_embed[:, :, 56*8+48:] = c_embed
        
        src_mask = self.make_src_padding_mask(inst_seq).to(device)

        de_inst_seq = decoder_in[:, :, 8].to(device)
        de_rel_seq = decoder_in[:, :, 9].to(device)
        de_ref_chord = decoder_in[:, :, 11:].to(device)
        
        de_inst_embed = self.de_inst_embedding(de_inst_seq)
        de_rel_embed = self.de_rel_embedding(de_rel_seq)
        de_ref_embed = self.de_chord_embedding(de_ref_chord)
        de_ref_embed = de_ref_embed.mean(dim=2)
        
        decoder_embed[:, :, :128] = de_inst_embed
        decoder_embed[:, :, 128:128+256] = de_rel_embed
        decoder_embed[:, :, 128+256:] = de_ref_embed
        
        tgt_pad_mask = self.make_tgt_padding_mask(de_inst_seq).to(device) # (B, 1, 1, tgt_len)
        tgt_causal_mask = self.make_causal_mask(d_l, device=device)  # (1, tgt_len, tgt_len)

        # Combine them: broadcast causal mask => (B, 1, tgt_len, tgt_len),
        # then logical OR with the padding mask.
        tgt_mask = tgt_pad_mask | tgt_causal_mask.unsqueeze(1)   # (B, 1, tgt_len, tgt_len)
        encoder_embed = encoder_embed * math.sqrt(self.d_model)
        decoder_embed = decoder_embed * math.sqrt(self.d_model)
        
        # 4) Encoder
        enc_out = self.encoder(encoder_embed, mask=src_mask)  # (B, src_len, d_model)
        
        cross_mask = src_mask.expand(-1, 1, d_l, -1)  # (B, 1, tgt_len, src_len)
        dec_out = self.decoder(decoder_embed, enc_out, self_mask=tgt_mask, cross_mask=cross_mask)
        # dec_out: (B, tgt_len, d_model)

        # 6) Final projection
        output_logit_container = []
        
        for i in range(8):
            # tgt_mask = self.generate_tgt_mask(tgt[:, i, :])  # Generate target mask
            # output = self.inst_decoder[i](tgt[:, i, :], memory, src_mask, tgt_mask)
            output = self.fc_out[i](dec_out)
            output_logit_container.append(output)
            
        return output_logit_container

    def ce_loss(self, logit_container, target):
        loss_container = []
        total_loss = 0

        # for i in range(8):
        #     tmin = target[:,:,i].min().item()
        #     tmax = target[:,:,i].max().item()
        #     print(f"Book {i}: target min = {tmin}, max = {tmax}, expected range = [0, {self.book_vocab_size[i]-1}]")
        
        for i in range(8):
            # print(f'{i}, {logit_container[i].shape}, {self.book_vocab_size[i]}')
            loss = self.criterion(logit_container[i].view(-1, self.book_vocab_size[i]), target[:,:,i].reshape(-1))
            # print(loss)
            loss_container.append(loss)
            total_loss += loss
        return loss_container, total_loss/self.book_size
    
    def top_k_sampling(self, logit_container, k=5, temperature=1.0):
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
    
    
    
class Onset(nn.Module):
    """
    Standard Encoder - Decoder Transformer.
    """
    def __init__(self,
                 src_vocab_size=None,
                 tgt_vocab_size=None,
                 inst_vocab_size=None,
                 measure_size=None,
                 d_model=512,
                 book_size=8,
                 book_vocab_size=[],
                 num_heads=8,
                 d_ff=2048,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 src_max_len=512,
                 tgt_max_len=512,
                 dropout=0.1,
                 pad_token_id=0,
                 mode = ''):
        super(Onset, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.mode = mode
        self.book_size = book_size
        self.book_vocab_size = book_vocab_size
        self.criterion = nn.CrossEntropyLoss()
        # Embeddings + positional encoding
        if mode == '':
            self.en_onset_embedding = nn.ModuleList([nn.Embedding(self.book_vocab_size[i], 56) for i in range(8)])
            self.en_inst_embedding = nn.Embedding(133, 16)
            self.en_rel_embedding = nn.Embedding(24, 32)
            self.en_chord_embedding = nn.Embedding(136, 16)
            
            self.de_inst_embedding = nn.Embedding(133, 128)
            self.de_rel_embedding = nn.Embedding(24, 256)
            self.de_chord_embedding = nn.Embedding(136, 128)
        elif mode == 'POS':
            self.en_onset_embedding = nn.ModuleList([nn.Embedding(self.book_vocab_size[i], 56, padding_idx=0) for i in range(8)])
            self.en_inst_embedding = nn.Embedding(133, 32, padding_idx=0)
            self.en_chord_embedding = nn.Embedding(136, 32, padding_idx=0)
            
            self.de_inst_embedding = nn.Embedding(133, 256, padding_idx=0)
            self.de_chord_embedding = nn.Embedding(136, 256, padding_idx=0)
        elif mode == 'CHD':
            self.en_onset_embedding = nn.ModuleList([nn.Embedding(self.book_vocab_size[i], 56, padding_idx=0) for i in range(8)])
            self.en_inst_embedding = nn.Embedding(133, 32, padding_idx=0)
            self.en_rel_embedding = nn.Embedding(24, 32)
            
            self.de_inst_embedding = nn.Embedding(133, 256, padding_idx=0)
            self.de_rel_embedding = nn.Embedding(24, 256)

        elif mode == 'INS':
            self.en_onset_embedding = nn.ModuleList([nn.Embedding(self.book_vocab_size[i], 56, padding_idx=0) for i in range(8)])
            self.en_rel_embedding = nn.Embedding(24, 32)
            self.en_chord_embedding = nn.Embedding(136, 32, padding_idx=0)
            
            self.de_rel_embedding = nn.Embedding(24, 256)
            self.de_chord_embedding = nn.Embedding(136, 256, padding_idx=0)
        elif mode == 'MIN':
            self.en_onset_embedding = nn.ModuleList([nn.Embedding(self.book_vocab_size[i], 56, padding_idx=0) for i in range(8)])
            self.en_inst_embedding = nn.Embedding(133, 64, padding_idx=0)
            # self.en_rel_embedding = nn.Embedding(24, 32)
            # self.en_chord_embedding = nn.Embedding(136, 16, padding_idx=0)
            
            self.de_inst_embedding = nn.Embedding(133, 512, padding_idx=0)
            # self.de_rel_embedding = nn.Embedding(24, 256)
            # self.de_chord_embedding = nn.Embedding(136, 128, padding_idx=0)
        
        
        

        # idx 101 is mask to predict onset like [101, 101, 101, 101, ...]
        # idx 102 is EOS
        # self.de_onset_embedding = nn.Embedding(103, 256, padding_idx=0)

        self.tgt_pos_enc = PositionalEncoding(d_model, max_len=tgt_max_len)

        # Encoder
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)
        # Decoder (cross_attn=True)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_decoder_layers, dropout, cross_attn=True)

        # Final projection to target vocab
        # Include MASK and EOS index
        self.fc_out = nn.ModuleList([nn.Linear(d_model, self.book_vocab_size[i]) for i in range(8)])

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

    def forward(self, encoder_in, decoder_in):
        """
        src: (B, src_len)
        tgt: (B, tgt_len)
        returns: (B, tgt_len, tgt_vocab_size)
        """
        device = encoder_in.device
        _, e_l, _ = encoder_in.shape
        bsz, d_l, _ = decoder_in.shape
        l = d_l
        # B, src_len = src.shape
        # B, tgt_len = tgt.shape

        encoder_embed = torch.zeros(bsz, e_l, 512).to(device)
        decoder_embed = torch.zeros(bsz, d_l, 512).to(device)

        if self.mode == '':
            onset_seq = encoder_in[:, :, :8].to(device)
            inst_seq = encoder_in[:, :, 8].to(device)
            rel_seq = encoder_in[:, :, 9].to(device)
            c_seq = encoder_in[:, :, 11:].to(device)
            
            for i in range(8):
                book = self.en_onset_embedding[i](onset_seq[:, :, i])
                encoder_embed[:, :, 56*i:56*(i+1)] = book
            
            inst_embed = self.en_inst_embedding(inst_seq)
            encoder_embed[:, :, 56*8:56*8+16] = inst_embed
            
            rel_embed = self.en_rel_embedding(rel_seq)
            encoder_embed[:, :, 56*8+16:56*8+48] = rel_embed
            
            c_embed = self.en_chord_embedding(c_seq)
            c_embed = c_embed.mean(dim=2)
            encoder_embed[:, :, 56*8+48:] = c_embed
            
            src_mask = self.make_src_padding_mask(inst_seq).to(device)


            de_inst_seq = decoder_in[:, :, 8].to(device)
            de_rel_seq = decoder_in[:, :, 9].to(device)
            de_ref_chord = decoder_in[:, :, 11:].to(device)
            
            de_inst_embed = self.de_inst_embedding(de_inst_seq)
            de_rel_embed = self.de_rel_embedding(de_rel_seq)
            de_ref_embed = self.de_chord_embedding(de_ref_chord)
            de_ref_embed = de_ref_embed.mean(dim=2)
            
            decoder_embed[:, :, :128] = de_inst_embed
            decoder_embed[:, :, 128:128+256] = de_rel_embed
            decoder_embed[:, :, 128+256:] = de_ref_embed
            
            tgt_pad_mask = self.make_tgt_padding_mask(de_inst_seq).to(device) # (B, 1, 1, tgt_len)
            tgt_causal_mask = self.make_causal_mask(d_l, device=device)  # (1, tgt_len, tgt_len)

            # Combine them: broadcast causal mask => (B, 1, tgt_len, tgt_len),
            # then logical OR with the padding mask.
            tgt_mask = tgt_pad_mask | tgt_causal_mask.unsqueeze(1)   # (B, 1, tgt_len, tgt_len)
            
        elif self.mode == 'POS':
            onset_seq = encoder_in[:, :, :8].to(device)
            inst_seq = encoder_in[:, :, 8].to(device)
            c_seq = encoder_in[:, :, 11:].to(device)

            for i in range(8):
                book = self.en_onset_embedding[i](onset_seq[:, :, i])
                encoder_embed[:, :, 56*i:56*(i+1)] = book
            
            inst_embed = self.en_inst_embedding(inst_seq)
            encoder_embed[:, :, 56*8:56*8+32] = inst_embed
            
            c_embed = self.en_chord_embedding(c_seq)
            c_embed = c_embed.mean(dim=2)
            encoder_embed[:, :, 56*8+32:] = c_embed
            
            src_mask = self.make_src_padding_mask(inst_seq).to(device)


            de_inst_seq = decoder_in[:, :, 8].to(device)
            de_ref_chord = decoder_in[:, :, 11:].to(device)
            
            de_inst_embed = self.de_inst_embedding(de_inst_seq)
            de_ref_embed = self.de_chord_embedding(de_ref_chord)
            de_ref_embed = de_ref_embed.mean(dim=2)
            
            decoder_embed[:, :, :256] = de_inst_embed
            decoder_embed[:, :, 256:] = de_ref_embed
            
            tgt_pad_mask = self.make_tgt_padding_mask(de_inst_seq).to(device) # (B, 1, 1, tgt_len)
            tgt_causal_mask = self.make_causal_mask(l, device=device)  # (1, tgt_len, tgt_len)

            # Combine them: broadcast causal mask => (B, 1, tgt_len, tgt_len),
            # then logical OR with the padding mask.
            tgt_mask = tgt_pad_mask | tgt_causal_mask.unsqueeze(1)   # (B, 1, tgt_len, tgt_len)
        elif self.mode == 'CHD':
            onset_seq = encoder_in[:, :, :8].to(device)
            inst_seq = encoder_in[:, :, 8].to(device)
            rel_seq = encoder_in[:, :, 9].to(device)

            for i in range(8):
                book = self.en_onset_embedding[i](onset_seq[:, :, i])
                encoder_embed[:, :, 56*i:56*(i+1)] = book
            
            inst_embed = self.en_inst_embedding(inst_seq)
            encoder_embed[:, :, 56*8:56*8+32] = inst_embed
            
            rel_embed = self.en_rel_embedding(rel_seq)
            encoder_embed[:, :, 56*8+32:] = rel_embed
            
            src_mask = self.make_src_padding_mask(inst_seq).to(device)


            de_inst_seq = decoder_in[:, :, 8].to(device)
            de_rel_seq = decoder_in[:, :, 9].to(device)
            
            de_inst_embed = self.de_inst_embedding(de_inst_seq)
            de_rel_embed = self.de_rel_embedding(de_rel_seq)
            
            decoder_embed[:, :, :256] = de_inst_embed
            decoder_embed[:, :, 256:] = de_rel_embed
            
            tgt_pad_mask = self.make_tgt_padding_mask(de_inst_seq).to(device) # (B, 1, 1, tgt_len)
            tgt_causal_mask = self.make_causal_mask(l, device=device)  # (1, tgt_len, tgt_len)

            # Combine them: broadcast causal mask => (B, 1, tgt_len, tgt_len),
            # then logical OR with the padding mask.
            tgt_mask = tgt_pad_mask | tgt_causal_mask.unsqueeze(1)   # (B, 1, tgt_len, tgt_len)
        elif self.mode == 'INS':
            onset_seq = encoder_in[:, :, :8].to(device)
            rel_seq = encoder_in[:, :, 9].to(device)
            c_seq = encoder_in[:, :, 11:].to(device)
            inst_seq = encoder_in[:, :, 8].to(device)

            for i in range(8):
                book = self.en_onset_embedding[i](onset_seq[:, :, i])
                encoder_embed[:, :, 56*i:56*(i+1)] = book
            
            rel_embed = self.en_rel_embedding(rel_seq)
            encoder_embed[:, :, 56*8:56*8+32] = rel_embed
            
            c_embed = self.en_chord_embedding(c_seq)
            c_embed = c_embed.mean(dim=2)
            encoder_embed[:, :, 56*8+32:] = c_embed
            
            src_mask = self.make_src_padding_mask(inst_seq).to(device)

            de_inst_seq = decoder_in[:, :, 8].to(device)
            de_rel_seq = decoder_in[:, :, 9].to(device)
            de_ref_chord = decoder_in[:, :, 11:].to(device)

            de_rel_embed = self.de_rel_embedding(de_rel_seq)
            de_ref_embed = self.de_chord_embedding(de_ref_chord)
            de_ref_embed = de_ref_embed.mean(dim=2)

            decoder_embed[:, :, :256] = de_rel_embed
            decoder_embed[:, :, 256:] = de_ref_embed
            
            tgt_pad_mask = self.make_tgt_padding_mask(de_inst_seq).to(device) # (B, 1, 1, tgt_len)
            tgt_causal_mask = self.make_causal_mask(l, device=device)  # (1, tgt_len, tgt_len)

            # Combine them: broadcast causal mask => (B, 1, tgt_len, tgt_len),
            # then logical OR with the padding mask.
            tgt_mask = tgt_pad_mask | tgt_causal_mask.unsqueeze(1)   # (B, 1, tgt_len, tgt_len)
        elif self.mode == 'MIN':
            onset_seq = encoder_in[:, :, :8].to(device)
            inst_seq = encoder_in[:, :, 8].to(device)

            for i in range(8):
                book = self.en_onset_embedding[i](onset_seq[:, :, i])
                encoder_embed[:, :, 56*i:56*(i+1)] = book

            inst_embed = self.en_inst_embedding(inst_seq)
            encoder_embed[:, :, 56*8:] = inst_embed
            
            src_mask = self.make_src_padding_mask(inst_seq).to(device)


            de_inst_seq = decoder_in[:, :, 8].to(device)
            de_inst_embed = self.de_inst_embedding(de_inst_seq)
            
            decoder_embed[:, :, :] = de_inst_embed
            
            tgt_pad_mask = self.make_tgt_padding_mask(de_inst_seq).to(device) # (B, 1, 1, tgt_len)
            tgt_causal_mask = self.make_causal_mask(l, device=device)  # (1, tgt_len, tgt_len)

            # Combine them: broadcast causal mask => (B, 1, tgt_len, tgt_len),
            # then logical OR with the padding mask.
            tgt_mask = tgt_pad_mask | tgt_causal_mask.unsqueeze(1)   # (B, 1, tgt_len, tgt_len)

        encoder_embed = encoder_embed * math.sqrt(self.d_model)
        decoder_embed = decoder_embed * math.sqrt(self.d_model)
        
        # 4) Encoder
        enc_out = self.encoder(encoder_embed, mask=src_mask)  # (B, src_len, d_model)
        
        cross_mask = src_mask.expand(-1, 1, d_l, -1)  # (B, 1, tgt_len, src_len)
        dec_out = self.decoder(decoder_embed, enc_out, self_mask=tgt_mask, cross_mask=cross_mask)
        # dec_out: (B, tgt_len, d_model)

        # 6) Final projection
        output_logit_container = []
        
        for i in range(8):
            # tgt_mask = self.generate_tgt_mask(tgt[:, i, :])  # Generate target mask
            # output = self.inst_decoder[i](tgt[:, i, :], memory, src_mask, tgt_mask)
            output = self.fc_out[i](dec_out)
            output_logit_container.append(output)
            
        return output_logit_container
    
    def ce_loss(self, logit_container, target):
        loss_container = []
        total_loss = 0
        
        for i in range(8):
            loss = self.criterion(logit_container[i].view(-1, self.book_vocab_size[i]), target[:,:,i].reshape(-1))
            loss_container.append(loss)
            total_loss +=  loss
        return loss_container, total_loss/8
    
    # def inference_sampling(self, output_logit_container, k=5, max_len=768):
    #     # Start with a beginning-of-sequence token for each book in the batch
    #     logit_container = self.forward(src, src)
    #     batch_size = src.size(0)
    #     max_len = src.size(1)
        
    #     inference = torch.zeros(batch_size, self.book_size, max_len)
        
    #     for i in range(self.book_size):
    #         book_logit = logit_container[i]
    #         next_token = self.top_k_sampling(book_logit, k=k)  # next_token: shape (batch_size)
    #         # print(next_token)
    #         # print(next_token.shape)
    #         # print(generated)
    #         # print(generated.shape)
    #         inference[:,i,:] = next_token
    #     inference = inference.to(src.device)
       
    #     return inference
    
    
    def top_k_sampling(self, logit_container, k=5, temperature=1.0):
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