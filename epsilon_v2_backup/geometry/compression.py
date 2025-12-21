
"""
Geometric Compression Module

Implements "Vector Translation Rules" to compress Graph Logic into Geometry.
Idea: Instead of storing A->B, B->C, X->Y, we learn a rule vector V such that:
    Start + V ≈ End
    
This allows infinite scaling of "trivial facts" without graph bloat.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

class VectorRule:
    """A compressed logical rule represented as a translation vector."""
    def __init__(self, name: str, vector: torch.Tensor, tolerance: float = 0.1):
        self.name = name
        self.vector = vector # The "Rule" itself
        self.tolerance = tolerance
        
    def apply(self, start_pos: torch.Tensor) -> torch.Tensor:
        """Apply the rule to a concept to hypothesize a result."""
        # Poincare translation (Mobius addition is better, but Euclidean add is fast approximation for local)
        # For v2 demo, we use simple addition + re-projection
        return start_pos + self.vector
        
    def check(self, start_pos: torch.Tensor, end_pos: torch.Tensor) -> bool:
        """Does this rule explain the relationship between Start and End?"""
        prediction = self.apply(start_pos)
        dist = torch.norm(prediction - end_pos, p=2)
        return dist < self.tolerance

class CompressionEngine:
    """
    Manages the compression of explicit graph edges into Vector Rules.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.rules: Dict[str, VectorRule] = {}
        
    def learn_rule(self, name: str, pairs: list[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Learn a universal vector that transforms inputs to outputs in the list.
        Minimizes MSE: ||(Start + V) - End||^2
        """
        if not pairs:
            return
            
        # Stack
        starts = torch.stack([p[0] for p in pairs])
        ends = torch.stack([p[1] for p in pairs])
        
        # Optimal V = mean(End - Start)
        diffs = ends - starts
        v_optimal = torch.mean(diffs, dim=0)
        
        # Calculate consistency (variance)
        variance = torch.var(diffs, dim=0).sum()
        
        # Only keep if rule is consistent enough
        if variance < 0.5:
            self.rules[name] = VectorRule(name, v_optimal)
            print(f"✓ Learned Compressed Rule '{name}' (Variance: {variance:.4f})")
        else:
            print(f"✗ Failed to compress '{name}': Relationship is not geometric (Var: {variance:.4f})")
            
    def get_rule(self, name: str) -> VectorRule:
        return self.rules.get(name)
