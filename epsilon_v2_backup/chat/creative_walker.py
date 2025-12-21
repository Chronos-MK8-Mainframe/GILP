"""
Creative Walker for Epsilon Chat

Generates novel content by walking through the GILP manifold.
Different walk styles produce different types of creativity:
- Divergent: Seek novelty, move away from common concepts
- Associative: Random walk with momentum, stream of consciousness
- Structured: Follow shell boundaries, coherent narrative

Key insight: Creativity = traversing unexplored geometry.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import numpy as np


class WalkStyle(Enum):
    """Types of creative walks through the manifold."""
    DIVERGENT = "divergent"         # Seek novelty
    ASSOCIATIVE = "associative"     # Stream of consciousness
    STRUCTURED = "structured"       # Coherent narrative
    EXPLORATORY = "exploratory"     # Random exploration
    GOAL_DIRECTED = "goal_directed" # Move toward target


@dataclass
class WalkStep:
    """Single step in a creative walk."""
    position: torch.Tensor
    direction: torch.Tensor
    concept_idx: Optional[int] = None
    concept_name: Optional[str] = None
    novelty_score: float = 0.0


@dataclass
class CreativeWalk:
    """Complete creative walk through the manifold."""
    start: torch.Tensor
    steps: List[WalkStep] = field(default_factory=list)
    style: WalkStyle = WalkStyle.DIVERGENT
    
    @property
    def trajectory(self) -> torch.Tensor:
        """Get the full trajectory as tensor."""
        positions = [self.start] + [s.position for s in self.steps]
        return torch.stack(positions)
    
    @property
    def concepts(self) -> List[str]:
        """Get list of concepts encountered."""
        return [s.concept_name for s in self.steps if s.concept_name]


class CreativeWalker:
    """
    Generates creative content by walking through GILP manifold.
    
    The manifold encodes concept relationships geometrically.
    Walking through it = generating sequences of related concepts.
    Different walk styles produce different creative outputs.
    """
    
    def __init__(self, embeddings: torch.Tensor,
                 concept_names: Optional[List[str]] = None,
                 manifold=None):
        """
        Args:
            embeddings: All concept embeddings [N, D]
            concept_names: Optional names for each concept
            manifold: QuantizedPoincareManifold for geometry
        """
        self.embeddings = embeddings
        self.concept_names = concept_names or [f"concept_{i}" for i in range(len(embeddings))]
        self.manifold = manifold
        
        self.num_concepts = embeddings.size(0)
        self.dim = embeddings.size(1)
        
        # Precompute concept density (for novelty scoring)
        self._compute_density()
    
    def _compute_density(self, k: int = 10):
        """Compute local density around each concept."""
        # Higher density = more common
        from sklearn.neighbors import NearestNeighbors
        emb_np = self.embeddings.detach().cpu().numpy()
        nn = NearestNeighbors(n_neighbors=min(k, self.num_concepts))
        nn.fit(emb_np)
        distances, _ = nn.kneighbors(emb_np)
        # Density = 1 / average distance to neighbors
        self.density = 1.0 / (distances.mean(axis=1) + 1e-6)
        self.density = torch.tensor(self.density, device=self.embeddings.device)
    
    def walk(self, start: torch.Tensor, num_steps: int = 10,
             style: WalkStyle = WalkStyle.DIVERGENT,
             goal: Optional[torch.Tensor] = None,
             step_size: float = 0.1,
             temperature: float = 1.0) -> CreativeWalk:
        """
        Perform a creative walk through the manifold.
        
        Args:
            start: Starting position in manifold
            num_steps: Number of steps to take
            style: Walk style (divergent, associative, etc.)
            goal: Target position for goal-directed walk
            step_size: Size of each step
            temperature: Randomness (higher = more random)
            
        Returns:
            CreativeWalk with full trajectory
        """
        walk = CreativeWalk(start=start, style=style)
        current = start.clone()
        momentum = torch.zeros_like(current)
        
        for step_idx in range(num_steps):
            # Compute direction based on style
            if style == WalkStyle.DIVERGENT:
                direction = self._divergent_direction(current)
            elif style == WalkStyle.ASSOCIATIVE:
                direction, momentum = self._associative_direction(current, momentum)
            elif style == WalkStyle.STRUCTURED:
                direction = self._structured_direction(current, step_idx)
            elif style == WalkStyle.EXPLORATORY:
                direction = self._exploratory_direction(current, temperature)
            elif style == WalkStyle.GOAL_DIRECTED and goal is not None:
                direction = self._goal_direction(current, goal)
            else:
                direction = self._exploratory_direction(current, temperature)
            
            # Normalize and scale
            direction = F.normalize(direction, dim=-1) * step_size
            
            # Take step
            next_pos = current + direction
            next_pos = self._project_to_ball(next_pos)
            
            # Find nearest concept
            concept_idx, concept_name = self._nearest_concept(next_pos)
            novelty = self._novelty_score(next_pos)
            
            # Record step
            walk.steps.append(WalkStep(
                position=next_pos.clone(),
                direction=direction.clone(),
                concept_idx=concept_idx,
                concept_name=concept_name,
                novelty_score=novelty
            ))
            
            current = next_pos
        
        return walk
    
    def _divergent_direction(self, current: torch.Tensor) -> torch.Tensor:
        """
        Move AWAY from high-density regions.
        Novelty seeking: go where few concepts are.
        """
        # Compute gradient of density field (approximate)
        # Move away from nearest high-density concepts
        dists = torch.norm(self.embeddings - current.unsqueeze(0), dim=-1)
        weights = self.density / (dists + 1e-6)  # High density, close = high weight
        
        # Direction = away from weighted centroid
        weighted_center = (self.embeddings * weights.unsqueeze(-1)).sum(0) / weights.sum()
        direction = current - weighted_center  # Away from dense regions
        
        return direction
    
    def _associative_direction(self, current: torch.Tensor,
                               momentum: torch.Tensor,
                               momentum_decay: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random walk with momentum.
        Stream of consciousness: loosely connected associations.
        """
        # Random perturbation
        noise = torch.randn_like(current) * 0.3
        
        # Update momentum
        new_momentum = momentum_decay * momentum + noise
        
        # Direction = momentum
        return new_momentum, new_momentum
    
    def _structured_direction(self, current: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        Follow shell boundaries (constant radius).
        Coherent narrative: stay at same abstraction level.
        """
        # Move tangent to the sphere (orthogonal to radial direction)
        radial = F.normalize(current, dim=-1)
        
        # Random tangent direction
        random_dir = torch.randn_like(current)
        # Project out radial component
        tangent = random_dir - (random_dir * radial).sum() * radial
        
        return tangent
    
    def _exploratory_direction(self, current: torch.Tensor,
                               temperature: float) -> torch.Tensor:
        """
        Pure random exploration with temperature control.
        """
        noise = torch.randn_like(current) * temperature
        return noise
    
    def _goal_direction(self, current: torch.Tensor,
                       goal: torch.Tensor) -> torch.Tensor:
        """
        Move toward a specific goal.
        """
        return goal - current
    
    def _project_to_ball(self, x: torch.Tensor, max_norm: float = 0.95) -> torch.Tensor:
        """Keep point inside Poincaré ball."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        if norm > max_norm:
            x = x / norm * max_norm
        return x
    
    def _nearest_concept(self, position: torch.Tensor) -> Tuple[int, str]:
        """Find the nearest concept to a position."""
        dists = torch.norm(self.embeddings - position.unsqueeze(0), dim=-1)
        idx = dists.argmin().item()
        return idx, self.concept_names[idx]
    
    def _novelty_score(self, position: torch.Tensor) -> float:
        """
        Score how novel/unusual a position is.
        Higher = less common region.
        """
        # Distance to nearest concept
        dists = torch.norm(self.embeddings - position.unsqueeze(0), dim=-1)
        min_dist = dists.min().item()
        
        # Inverse of local density
        nearest_idx = dists.argmin().item()
        inv_density = 1.0 / (self.density[nearest_idx].item() + 1e-6)
        
        # Combine
        return min_dist * 0.5 + inv_density * 0.5
    
    def generate_concept_sequence(self, start_concept: str,
                                  num_concepts: int = 10,
                                  style: WalkStyle = WalkStyle.ASSOCIATIVE) -> List[str]:
        """
        Generate a sequence of related concepts starting from a word.
        
        Args:
            start_concept: Starting concept name
            num_concepts: Number of concepts to generate
            style: Walk style
            
        Returns:
            List of concept names
        """
        # Find starting position
        if start_concept in self.concept_names:
            start_idx = self.concept_names.index(start_concept)
            start_pos = self.embeddings[start_idx]
        else:
            # Random start
            start_pos = torch.randn(self.dim) * 0.5
            start_pos = self._project_to_ball(start_pos)
        
        # Walk
        walk = self.walk(start_pos, num_steps=num_concepts, style=style)
        
        return walk.concepts
    
    def interpolate_concepts(self, start: str, end: str,
                            num_steps: int = 5) -> List[str]:
        """
        Find concepts along a path between two concepts.
        Useful for generating transitions or explanations.
        """
        # Get positions
        start_idx = self.concept_names.index(start) if start in self.concept_names else 0
        end_idx = self.concept_names.index(end) if end in self.concept_names else 0
        
        start_pos = self.embeddings[start_idx]
        end_pos = self.embeddings[end_idx]
        
        # Interpolate
        concepts = [start]
        for i in range(1, num_steps + 1):
            t = i / (num_steps + 1)
            pos = start_pos * (1 - t) + end_pos * t
            pos = self._project_to_ball(pos)
            _, concept = self._nearest_concept(pos)
            if concept != concepts[-1]:  # Avoid repeats
                concepts.append(concept)
        concepts.append(end)
        
        return concepts


def demo_creative_walker():
    """Demo the creative walker with synthetic data."""
    print("=== Creative Walker Demo ===\n")
    
    # Create fake concept embeddings (clustered)
    num_concepts = 50
    dim = 32
    
    # Create clusters
    clusters = [
        ("animals", torch.randn(10, dim) * 0.1 + torch.tensor([0.5, 0.3] + [0] * (dim-2))),
        ("food", torch.randn(10, dim) * 0.1 + torch.tensor([-0.4, 0.3] + [0] * (dim-2))),
        ("science", torch.randn(10, dim) * 0.1 + torch.tensor([0.0, -0.5] + [0] * (dim-2))),
        ("art", torch.randn(15, dim) * 0.1 + torch.tensor([0.3, -0.3] + [0] * (dim-2))),
        ("misc", torch.randn(5, dim) * 0.1),
    ]
    
    embeddings = torch.cat([c[1] for c in clusters])
    names = []
    for cluster_name, embeds in clusters:
        for i in range(len(embeds)):
            names.append(f"{cluster_name}_{i}")
    
    # Create walker
    walker = CreativeWalker(embeddings, names)
    
    # Test different walk styles
    start = embeddings[0]  # Start from first animal
    
    print("Starting from: animals_0\n")
    
    for style in [WalkStyle.DIVERGENT, WalkStyle.ASSOCIATIVE, WalkStyle.STRUCTURED]:
        walk = walker.walk(start, num_steps=8, style=style, step_size=0.15)
        print(f"{style.value:12s}: {' → '.join(walk.concepts[:6])}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_creative_walker()
