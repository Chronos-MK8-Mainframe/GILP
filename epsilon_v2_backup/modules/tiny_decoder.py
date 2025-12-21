
import torch
import torch.nn as nn
import math
from epsilon.config import TinyDecoderConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class TinyDecoder(nn.Module):
    """
    A lightweight, deterministic translator from Geometric Paths to Text.
    It uses a small Transformer Decoder to attend to the geometric path (Logic+Psych+Expr)
    and generate the corresponding natural language.
    """
    def __init__(self, config: TinyDecoderConfig):
        super().__init__()
        self.config = config
        
        # Text Embedding
        self.embedding = nn.Embedding(config.output_dim, config.hidden_dim)
        self.pos_encoder = PositionalEncoding(config.hidden_dim)
        
        # Projection for geometric path (Memory)
        # We project the concatenated [Logic, Psych, Expr] vector to hidden_dim
        self.path_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Transformer Decoder
        # Defaults: nhead=4, dim_feedforward=4*d_model
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim, 
            nhead=4,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        
        # Output Head
        self.head = nn.Linear(config.hidden_dim, config.output_dim)
        
    def forward(self, geometric_path: torch.Tensor, target_tokens: torch.Tensor, 
                tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            geometric_path: [Batch, Path_Len, Input_Dim] - The thought trajectory
            target_tokens: [Batch, Seq_Len] - The target text tokens (for training)
            tgt_mask: Optional mask for autoregressive generation
            
        Returns:
            logits: [Batch, Seq_Len, Vocab_Size]
        """
        # 1. Prepare Memory (Geometric Path)
        # [Batch, Path_Len, Input_Dim] -> [Batch, Path_Len, Hidden_Dim]
        memory = self.path_projection(geometric_path)
        # Transpose for Transformer [Path_Len, Batch, Hidden_Dim]
        memory = memory.transpose(0, 1)
        
        # 2. Prepare Target (Text)
        # [Batch, Seq_Len] -> [Batch, Seq_Len, Hidden_Dim]
        tgt = self.embedding(target_tokens)
        tgt = tgt.transpose(0, 1) # [Seq_Len, Batch, Hidden_Dim]
        tgt = self.pos_encoder(tgt)
        
        # 3. Decode
        # output: [Seq_Len, Batch, Hidden_Dim]
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 4. Project to Vocab
        logits = self.head(output.transpose(0, 1)) # [Batch, Seq_Len, Vocab_Size]
        
        return logits
        
    def generate(self, geometric_path: torch.Tensor, start_token: int, 
                 max_len: int = 50, end_token: int = None) -> torch.Tensor:
        """
        Greedy generation for inference.
        """
        batch_size = geometric_path.size(0)
        device = geometric_path.device
        
        # Start with just start_token
        tgt_tokens = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        # Pre-compute memory
        memory = self.path_projection(geometric_path)
        memory = memory.transpose(0, 1)
        
        for _ in range(max_len):
            # Embed current sequence
            tgt = self.embedding(tgt_tokens)
            tgt = tgt.transpose(0, 1)
            tgt = self.pos_encoder(tgt)
            
            # Create causal mask
            seq_len = tgt.size(0)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
            
            # Forward pass
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.head(output[-1:]) # Only take last token output [1, Batch, Hidden]
            
            # Greedy decode
            next_token = logits.argmax(dim=-1).transpose(0, 1) # [Batch, 1]
            
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
            
            # Check for end token (simplified, assumes all finished)
            if end_token is not None and (next_token == end_token).all():
                break
                
        return tgt_tokens
