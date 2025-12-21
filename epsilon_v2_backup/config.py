"""
Epsilon Configuration

Central configuration schema for all Epsilon hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EpsilonConfig:
    """
    Configuration for the Epsilon reasoning substrate.
    
    Attributes:
        shell_radius: Target distance per proof step (default: 0.5)
        rigidity_strength: Weight for anchor enforcement loss
        max_local_hops: k-hop radius for incremental updates
        embedding_dim: Dimension of the hyperbolic embedding space
        vocab_size: Size of token vocabulary
        learning_rate: Base learning rate for optimizer
        
        Failure Detection Thresholds:
        - oscillation_window: Steps to check for cycle detection
        - flat_basin_threshold: Min distance variance to detect missing rules
        - contradiction_margin: Separation margin for contradictions
    """
    
    # Geometry
    shell_radius: float = 0.5
    embedding_dim: int = 64
    manifold_eps: float = 1e-5
    
    # Incremental Fossilization
    rigidity_strength: float = 1.0
    max_local_hops: int = 3
    anchor_decay: float = 0.99  # Anchor strength decay per epoch (1.0 = no decay)
    
    # Model
    vocab_size: int = 1000
    hidden_dim: int = 64
    num_gnn_layers: int = 4
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    
    # Loss weights
    structure_weight: float = 1.0
    shell_ordering_weight: float = 0.5
    contrastive_weight: float = 0.1
    fossilization_weight: float = 1.0
    
    # Failure Detection
    oscillation_window: int = 5
    flat_basin_threshold: float = 1e-4
    contradiction_margin: float = 2.0
    max_navigation_steps: int = 50
    neighbor_radius: float = 0.6
    
    # Cost Tracking
    enable_cost_tracking: bool = True
    
    # Heuristic scalars by edge type
    heuristic_scalars: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,  # prerequisite
        1: 0.5,  # contradiction
        2: 0.1   # composition
    })
    
    def validate(self):
        """Validate configuration constraints."""
        assert 0 < self.shell_radius < 1.0, "shell_radius must be in (0, 1)"
        assert self.rigidity_strength >= 0, "rigidity_strength must be non-negative"
        assert self.max_local_hops >= 1, "max_local_hops must be >= 1"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        return True


@dataclass
class PsychologyConfig(EpsilonConfig):
    """Configuration for Psychology Manifold."""
    # Psychology might need more fluid transitions
    shell_radius: float = 0.3  # Finer granularity for emotions
    embedding_dim: int = 64


@dataclass
class ExpressionConfig(EpsilonConfig):
    """Configuration for Expression Manifold."""
    # Expression needs to capture style nuances
    embedding_dim: int = 128  # Higher dim for style vectors
    shell_radius: float = 0.4


@dataclass
class TinyDecoderConfig:
    """Configuration for the lightweight deterministic decoder."""
    input_dim: int = 256  # Sum of Logic(64) + Psych(64) + Expr(128)
    hidden_dim: int = 128
    output_dim: int = 1000  # Must match vocab_size
    num_layers: int = 2
    dropout: float = 0.1
