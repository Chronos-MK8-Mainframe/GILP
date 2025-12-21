
"""
Parabolic Manifold (Unified Dimensional Space)

Implements the "Parabolic Architecture" where dimension index determines semantic stability.
- Dims [0..CORE_DIM]: "Logic Core" (Stable, Axiomatic).
- Dims [CORE_DIM..]: "Atmosphere" (Context, Personality, Fluid).

Trajectory Rule: High-Dim -> Low-Dim (Grounding) -> High-Dim (Expression).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParabolicManifold(nn.Module):
    def __init__(self, total_dim=64, core_dim=4, core_weight=10.0):
        super().__init__()
        self.total_dim = total_dim
        self.core_dim = core_dim
        self.core_weight = core_weight
        
    def split(self, z: torch.Tensor):
        """Split embeddings into Core and Atmosphere."""
        core = z[..., :self.core_dim]
        atmosphere = z[..., self.core_dim:]
        return core, atmosphere
        
    def parabolic_dist(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Weighted distance metric.
        Changes in Core dimensions allow significantly more "Distance" (Penalty)
        than changes in Atmosphere.
        
        This forces the system to align Logic first, then Context.
        """
        c1, a1 = self.split(z1)
        c2, a2 = self.split(z2)
        
        # Euclidean for now (Poincare stability in mixed curvature is complex)
        d_core = torch.norm(c1 - c2, p=2, dim=-1)
        d_atm = torch.norm(a1 - a2, p=2, dim=-1)
        
        # Weighted sum
        return self.core_weight * d_core + d_atm
        
    def ground(self, z: torch.Tensor) -> torch.Tensor:
        """
        "The Descent": Project high-dim vector to Low-Dim Logic Core.
        Effectively strips context/emotion to find raw truth.
        """
        c, _ = self.split(z)
        # Pad with zeros to keep shape
        padding = torch.zeros_like(z)
        padding[..., :self.core_dim] = c
        return padding
        
    def express(self, core_z: torch.Tensor, personality_z: torch.Tensor) -> torch.Tensor:
        """
        "The Ascent": Attach personality context to raw logic.
        Result is a point on the "Surface" (Atmosphere).
        """
        c, _ = self.split(core_z)
        _, a = self.split(personality_z)
        return torch.cat([c, a], dim=-1)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Keep embeddings bounded."""
        return F.normalize(z, p=2, dim=-1)
