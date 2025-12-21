"""
Quantized Poincaré Manifold

Improvement #2 - Metric Quantization

Extends the Poincaré Ball model with discrete radial shells.
Each shell represents a proof depth level, enforcing that:
- A → C cannot collapse onto A → B → C
- Multi-step proofs are geometrically unavoidable
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

import sys
sys.path.insert(0, 'c:/Users/rupa9/Videos/GILP/GILP-main/GILP-main')
from gilp_core.geometry.hyperbolic import PoincareManifold


class QuantizedPoincareManifold(PoincareManifold):
    """
    Poincaré Ball with discrete radial shells.
    
    The key innovation: positions are constrained to radial shells,
    where each shell represents a proof depth level. This structurally
    prevents "teleportation" across proof depths.
    
    Shell 0: Origin (goals/conjectures)
    Shell 1: One step from goal
    Shell 2: Two steps from goal
    ...and so on.
    
    Distance between adjacent shells is exactly shell_width,
    enforcing unit-step geometry.
    """
    
    def __init__(self, shell_width: float = 0.5, eps: float = 1e-5):
        """
        Args:
            shell_width: Target hyperbolic distance between adjacent shells.
                         This enforces unit-step geometry.
            eps: Numerical stability constant.
        """
        super().__init__(eps)
        self.shell_width = shell_width
        
    def get_shell(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the shell index (proof depth) for each point.
        
        Shell index = floor(||x|| / shell_width)
        
        Points closer to origin are in lower shells (closer to goal).
        
        Args:
            x: Points in Poincaré ball [N, D]
            
        Returns:
            Shell indices [N]
        """
        norm = torch.norm(x, p=2, dim=-1)
        return (norm / self.shell_width).floor().long()
    
    def get_shell_float(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return continuous shell value (for gradient flow).
        
        Args:
            x: Points in Poincaré ball [N, D]
            
        Returns:
            Continuous shell values [N]
        """
        norm = torch.norm(x, p=2, dim=-1)
        return norm / self.shell_width
    
    def project_to_shell(self, x: torch.Tensor, target_shell: int) -> torch.Tensor:
        """
        Project points to specified shell boundary.
        
        Maintains angular direction, adjusts radial distance.
        
        Args:
            x: Points in Poincaré ball [N, D]
            target_shell: Target shell index
            
        Returns:
            Projected points [N, D] at shell boundary
        """
        # Normalize to get direction
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        direction = x / (norm + self.eps)
        
        # Target radius for shell
        target_radius = target_shell * self.shell_width
        
        # Clamp to stay inside Poincaré ball
        target_radius = min(target_radius, 1.0 - self.eps)
        
        return direction * target_radius
    
    def shell_ordering_loss(self, embeddings: torch.Tensor, 
                            edge_index: torch.Tensor) -> torch.Tensor:
        """
        Penalize edges where source is not one shell closer to origin than dest.
        
        For an edge A → B (A implies B), we want:
            shell(A) = shell(B) + 1
            
        This means A is one step further from the goal than B.
        The proof path goes FROM higher shells TO lower shells.
        
        Loss = mean(max(0, shell(dst) - shell(src) + 1))
        
        Args:
            embeddings: Node embeddings [N, D]
            edge_index: Edge indices [2, E]
            
        Returns:
            Shell ordering violation loss (scalar)
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        src, dst = edge_index
        
        # Use continuous shell values for gradient flow
        src_shells = self.get_shell_float(embeddings[src])
        dst_shells = self.get_shell_float(embeddings[dst])
        
        # We want src_shell = dst_shell + 1 (src is one step further)
        # Violation: src_shell < dst_shell + 1
        # Loss = max(0, dst_shell - src_shell + 1)
        violations = torch.clamp(dst_shells - src_shells + 1, min=0)
        
        return violations.mean()
    
    def target_distance_loss(self, embeddings: torch.Tensor,
                             edge_index: torch.Tensor,
                             target: Optional[float] = None) -> torch.Tensor:
        """
        Enforce specific distance for connected edges (unit step).
        
        This prevents collapse (d→0) and enables stepwise navigation.
        Combined with shell ordering, this creates a structured manifold
        where proof depth is geometrically encoded.
        
        Args:
            embeddings: Node embeddings [N, D]
            edge_index: Edge indices [2, E]
            target: Target distance (defaults to shell_width)
            
        Returns:
            MSE loss against target distance (scalar)
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        target = target if target is not None else self.shell_width
        
        src, dst = edge_index
        dists = self.dist(embeddings[src], embeddings[dst])
        
        return ((dists - target) ** 2).mean()
    
    def lateral_motion_loss(self, embeddings: torch.Tensor,
                            edge_index: torch.Tensor) -> torch.Tensor:
        """
        Penalize edges that don't change shell (lateral motion only).
        
        For valid inference edges, we expect shell transition.
        Edges within the same shell suggest missing structure.
        
        Args:
            embeddings: Node embeddings [N, D]
            edge_index: Edge indices [2, E]
            
        Returns:
            Lateral motion penalty (scalar)
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        src, dst = edge_index
        
        src_shells = self.get_shell(embeddings[src])
        dst_shells = self.get_shell(embeddings[dst])
        
        # Penalize same-shell edges
        same_shell = (src_shells == dst_shells).float()
        
        return same_shell.mean()
    
    def quantize_positions(self, embeddings: torch.Tensor,
                           soft: bool = True,
                           temperature: float = 0.1) -> torch.Tensor:
        """
        Quantize positions to shell boundaries.
        
        In soft mode, uses soft assignment for differentiability.
        In hard mode, snaps to nearest shell boundary.
        
        Args:
            embeddings: Node embeddings [N, D]
            soft: Use soft quantization (default True)
            temperature: Softmax temperature for soft mode
            
        Returns:
            Quantized embeddings [N, D]
        """
        norm = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        direction = embeddings / (norm + self.eps)
        
        if soft:
            # Soft assignment to shells
            shell_float = norm / self.shell_width
            shell_lower = shell_float.floor()
            shell_upper = shell_lower + 1
            
            # Soft weights based on distance to each shell
            dist_to_lower = shell_float - shell_lower
            dist_to_upper = shell_upper - shell_float
            
            # Softmax over distances
            weights = F.softmax(
                torch.cat([-dist_to_lower, -dist_to_upper], dim=-1) / temperature,
                dim=-1
            )
            
            # Weighted shell
            target_radius = (weights[:, 0:1] * shell_lower + 
                           weights[:, 1:2] * shell_upper) * self.shell_width
        else:
            # Hard snap to nearest shell
            shell_idx = (norm / self.shell_width).round()
            target_radius = shell_idx * self.shell_width
        
        # Clamp to stay inside ball
        target_radius = torch.clamp(target_radius, max=1.0 - self.eps)
        
        return direction * target_radius
