"""
Text Encoder for Epsilon Chat

Maps natural language text to GILP manifold vectors.
Uses a small transformer encoder (~50M params) trained with contrastive loss.

Key insight: We don't need the encoder to store knowledge — 
just to translate language into the manifold's coordinate system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """
    Encodes text to GILP manifold vectors.
    
    Architecture:
    - Token embedding (vocab_size → d_model)
    - Positional encoding
    - Transformer encoder (2 layers)
    - Mean pooling
    - Projection to GILP dimension
    - Hyperbolic projection (into Poincaré ball)
    
    The output is a point in the GILP hyperbolic manifold.
    
    TINY VERSION: ~500K params, runs on CPU with 4GB RAM
    """
    
    def __init__(self, 
                 vocab_size: int = 8192,    # Smaller vocab
                 d_model: int = 64,         # Tiny: was 256
                 nhead: int = 2,            # Tiny: was 4
                 num_layers: int = 2,       # Tiny: was 4
                 dim_feedforward: int = 256, # Tiny: was 1024
                 gilp_dim: int = 32,        # Tiny: was 64
                 max_len: int = 128,        # Shorter: was 512
                 dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of token vocabulary
            d_model: Internal transformer dimension
            nhead: Number of attention heads  
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            gilp_dim: Output GILP manifold dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.gilp_dim = gilp_dim
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection to GILP space
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, gilp_dim)
        )
        
        # For hyperbolic projection
        self.eps = 1e-5
        
    def forward(self, tokens: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode token sequence to GILP vector.
        
        Args:
            tokens: Token IDs [batch, seq_len]
            attention_mask: Padding mask [batch, seq_len], 1=valid, 0=pad
            
        Returns:
            GILP vectors in Poincaré ball [batch, gilp_dim]
        """
        # Embed tokens
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # TransformerEncoder expects True=ignore, so invert
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Mean pooling (mask-aware)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Project to GILP dimension
        x = self.projection(x)
        
        # Project to Poincaré ball (norm < 1)
        x = self._project_to_ball(x)
        
        return x
    
    def _project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball with norm < 1."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # Use tanh to smoothly map to (-1, 1) range
        scale = torch.tanh(norm) / (norm + self.eps)
        return x * scale * 0.9  # Leave margin inside ball
    
    def encode_text(self, text: str, tokenizer) -> torch.Tensor:
        """Convenience method to encode a single text string."""
        tokens = tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            return self.forward(tokens)


class SimpleTokenizer:
    """
    Simple character-level tokenizer for initial testing.
    Replace with SentencePiece/BPE for production.
    """
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
    def encode(self, text: str, max_length: int = 128, 
               return_tensors: Optional[str] = None) -> torch.Tensor:
        """Encode text to token IDs."""
        # Simple character-level encoding
        tokens = [self.bos_token_id]
        for char in text[:max_length - 2]:
            token_id = ord(char) % (self.vocab_size - 4) + 4
            tokens.append(token_id)
        tokens.append(self.eos_token_id)
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
        
        tokens = tokens[:max_length]
        
        if return_tensors == 'pt':
            return torch.tensor([tokens], dtype=torch.long)
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for token_id in tokens:
            if token_id <= 3:  # Special tokens
                continue
            char = chr((token_id - 4) % 128 + 32)
            chars.append(char)
        return ''.join(chars)
    
    def batch_encode(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Encode batch of texts."""
        all_tokens = []
        all_masks = []
        
        for text in texts:
            tokens = self.encode(text, max_length=max_length)
            mask = [1 if t != self.pad_token_id else 0 for t in tokens]
            all_tokens.append(tokens)
            all_masks.append(mask)
        
        return {
            'input_ids': torch.tensor(all_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(all_masks, dtype=torch.long)
        }


class ContrastiveTrainer:
    """
    Train text encoder with contrastive loss.
    
    Similar sentences → similar GILP vectors
    Different sentences → different GILP vectors
    """
    
    def __init__(self, encoder: TextEncoder, learning_rate: float = 1e-4,
                 temperature: float = 0.07):
        self.encoder = encoder
        self.optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
        self.temperature = temperature
        
    def contrastive_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss.
        
        Assumes embeddings are pairs: [pos1, pos2, pos1, pos2, ...]
        where consecutive pairs are positive examples.
        """
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create labels: pairs are positives
        labels = torch.arange(batch_size, device=embeddings.device)
        labels = labels ^ 1  # XOR with 1: 0↔1, 2↔3, etc.
        
        # Mask out self-similarities
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.encoder.train()
        self.optimizer.zero_grad()
        
        # Encode batch
        embeddings = self.encoder(
            batch['input_ids'],
            attention_mask=batch.get('attention_mask')
        )
        
        # Compute loss
        loss = self.contrastive_loss(embeddings)
        
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def train_epoch(self, dataloader, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            num_batches += 1
        
        return {'loss': total_loss / num_batches, 'epoch': epoch}


def test_encoder():
    """Quick test of the text encoder."""
    print("=== Testing TextEncoder (TINY version) ===\n")
    
    # Create TINY encoder (laptop-friendly)
    encoder = TextEncoder(
        vocab_size=8192,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=256,
        gilp_dim=32
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Test encoding
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps across the sleepy hound.",  # Similar
        "Python is a programming language.",
        "Python is used for machine learning.",  # Similar
    ]
    
    batch = tokenizer.batch_encode(texts)
    
    with torch.no_grad():
        embeddings = encoder(batch['input_ids'], batch['attention_mask'])
    
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Output norm: {embeddings.norm(dim=-1).tolist()}")
    
    # Check that outputs are in Poincaré ball
    assert (embeddings.norm(dim=-1) < 1).all(), "Outputs should be in unit ball"
    print("\n✓ All outputs in Poincaré ball")
    
    # Compute similarities
    embeddings_norm = F.normalize(embeddings, dim=-1)
    sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
    
    print("\nSimilarity matrix:")
    for i, row in enumerate(sim_matrix):
        print(f"  {i}: {[f'{v:.2f}' for v in row.tolist()]}")
    
    print("\n=== Encoder Test Complete ===")


if __name__ == "__main__":
    test_encoder()
