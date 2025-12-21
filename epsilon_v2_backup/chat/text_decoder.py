"""
Text Decoder for Epsilon Chat

Converts GILP manifold trajectories back to natural language.
Uses a small transformer decoder trained to output fluent text.

Key insight: The decoder doesn't need to understand reasoning —
just translate the manifold's structure into readable language.

TINY VERSION: ~500K params for laptop (CPU-only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class TextDecoder(nn.Module):
    """
    Decodes GILP trajectory to text.
    
    Architecture:
    - GILP vectors → d_model projection
    - Transformer decoder (cross-attention to trajectory)
    - Output projection → vocab
    
    TINY VERSION: ~500K params
    """
    
    def __init__(self,
                 vocab_size: int = 8192,
                 d_model: int = 64,
                 nhead: int = 2,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 gilp_dim: int = 32,
                 max_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.gilp_dim = gilp_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Project GILP vectors to d_model
        self.gilp_projection = nn.Linear(gilp_dim, d_model)
        
        # Token embedding for autoregressive generation
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Special tokens
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        
    def forward(self, gilp_trajectory: torch.Tensor,
                target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass (for training).
        
        Args:
            gilp_trajectory: GILP vectors [batch, seq, gilp_dim]
            target_tokens: Target token IDs [batch, tgt_len]
            
        Returns:
            Logits [batch, tgt_len, vocab_size]
        """
        batch_size = gilp_trajectory.size(0)
        
        # Project GILP trajectory
        memory = self.gilp_projection(gilp_trajectory)
        
        # Embed target tokens
        if target_tokens is None:
            # Start with BOS token
            target_tokens = torch.full((batch_size, 1), self.bos_token_id, 
                                       dtype=torch.long, device=memory.device)
        
        tgt_len = target_tokens.size(1)
        positions = torch.arange(tgt_len, device=target_tokens.device)
        
        tgt_embed = self.token_embedding(target_tokens) + self.pos_embedding(positions)
        
        # Causal mask for autoregressive generation
        tgt_mask = self._generate_square_mask(tgt_len, target_tokens.device)
        
        # Decode
        output = self.transformer(tgt_embed, memory, tgt_mask=tgt_mask)
        
        # Project to vocab
        logits = self.output_proj(output)
        
        return logits
    
    def _generate_square_mask(self, sz: int, device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    @torch.no_grad()
    def generate(self, gilp_trajectory: torch.Tensor,
                max_length: int = 50,
                temperature: float = 1.0,
                top_k: Optional[int] = 50) -> torch.Tensor:
        """
        Generate text from GILP trajectory.
        
        Args:
            gilp_trajectory: GILP vectors [batch, seq, gilp_dim]
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = greedy)
            
        Returns:
            Generated token IDs [batch, length]
        """
        batch_size = gilp_trajectory.size(0)
        device = gilp_trajectory.device
        
        # Project trajectory
        memory = self.gilp_projection(gilp_trajectory)
        
        # Start with BOS
        generated = torch.full((batch_size, 1), self.bos_token_id, 
                               dtype=torch.long, device=device)
        
        for _ in range(max_length):
            tgt_len = generated.size(1)
            positions = torch.arange(tgt_len, device=device)
            
            tgt_embed = self.token_embedding(generated) + self.pos_embedding(positions)
            tgt_mask = self._generate_square_mask(tgt_len, device)
            
            output = self.transformer(tgt_embed, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(output[:, -1, :])  # Last position
            
            # Sample
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None:
                # Top-k sampling
                top_values, top_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                probs = F.softmax(top_values, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_indices.gather(-1, idx)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS
            if (next_token == self.eos_token_id).all():
                break
        
        return generated


class TrajectoryToTextTrainer:
    """
    Train decoder to convert GILP trajectories to fluent text.
    """
    
    def __init__(self, decoder: TextDecoder, learning_rate: float = 1e-4):
        self.decoder = decoder
        self.optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
    def train_step(self, trajectories: torch.Tensor, 
                   target_tokens: torch.Tensor) -> dict:
        """
        Single training step.
        
        Args:
            trajectories: GILP trajectories [batch, seq, gilp_dim]
            target_tokens: Target text tokens [batch, tgt_len]
        """
        self.decoder.train()
        self.optimizer.zero_grad()
        
        # Teacher forcing: input = tokens[:-1], target = tokens[1:]
        input_tokens = target_tokens[:, :-1]
        target = target_tokens[:, 1:]
        
        logits = self.decoder(trajectories, input_tokens)
        
        # Flatten for cross-entropy
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1)
        )
        
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}


def test_decoder():
    """Test the text decoder."""
    print("=== Testing TextDecoder (TINY version) ===\n")
    
    # Create decoder
    decoder = TextDecoder(
        vocab_size=8192,
        d_model=64,
        nhead=2,
        num_layers=2,
        gilp_dim=32
    )
    
    # Count params
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Test forward
    batch_size = 2
    traj_len = 5
    tgt_len = 10
    
    trajectories = torch.randn(batch_size, traj_len, 32)  # GILP trajectories
    target = torch.randint(3, 100, (batch_size, tgt_len))  # Random tokens
    
    logits = decoder(trajectories, target)
    print(f"Input trajectory: {trajectories.shape}")
    print(f"Output logits: {logits.shape}")
    
    # Test generation
    generated = decoder.generate(trajectories, max_length=20)
    print(f"Generated tokens: {generated.shape}")
    print(f"Sample: {generated[0].tolist()[:15]}...")
    
    print("\n=== Decoder Test Complete ===")


if __name__ == "__main__":
    test_decoder()
