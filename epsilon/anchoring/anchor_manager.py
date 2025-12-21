"""
Anchor Manager

Improvement #1 - Incremental Fossilization

Manages verified logic anchors that represent trusted inference edges.
New learning must "bend around" these anchors, preserving existing proofs.

Core concept: Geometry becomes plastic locally, rigid globally.
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set
from collections import defaultdict


@dataclass
class Anchor:
    """
    Represents a verified inference edge with frozen distance.
    
    An anchor locks the geometric relationship between two nodes,
    preventing existing proofs from collapsing under new learning.
    """
    src: int              # Source node index
    dst: int              # Destination node index
    delta: float          # Frozen hyperbolic distance d(src, dst)
    strength: float       # Confidence in [0, 1] - higher = more rigid
    timestamp: float      # When anchor was created (for ordering)
    rule_name: str = ""   # Human-readable rule name for debugging


class AnchorManager:
    """
    Manages verified logic anchors for incremental fossilization.
    
    The key insight: instead of retraining the entire geometry when
    adding new rules, we:
    1. Lock existing verified inferences as anchors
    2. Only allow gradient updates in local neighborhoods
    3. Penalize any drift from anchored distances
    
    This enables continuous learning without catastrophic forgetting
    of previously verified proofs.
    """
    
    def __init__(self, decay_rate: float = 1.0):
        """
        Args:
            decay_rate: Per-epoch decay for anchor strength (1.0 = no decay)
        """
        self.anchors: Dict[Tuple[int, int], Anchor] = {}
        self.decay_rate = decay_rate
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)  # For k-hop queries
        
    def register_anchor(self, src: int, dst: int, delta: float, 
                       strength: float = 1.0, rule_name: str = "") -> Anchor:
        """
        Register a verified inference as permanent geometry.
        
        Args:
            src: Source node index
            dst: Destination node index
            delta: Current distance to freeze
            strength: Confidence in [0, 1]
            rule_name: Human-readable name
            
        Returns:
            Created Anchor object
        """
        anchor = Anchor(
            src=src,
            dst=dst,
            delta=delta,
            strength=strength,
            timestamp=time.time(),
            rule_name=rule_name
        )
        self.anchors[(src, dst)] = anchor
        
        # Update adjacency for k-hop queries
        self.adjacency[src].add(dst)
        self.adjacency[dst].add(src)
        
        return anchor
    
    def register_from_proof_trace(self, proof_trace, embeddings, manifold):
        """
        Register all steps in a proof trace as anchors.
        
        Args:
            proof_trace: ProofTrace object with verified steps
            embeddings: Current node embeddings
            manifold: Geometry manifold for distance computation
        """
        for step in proof_trace.steps:
            delta = manifold.dist(
                embeddings[step.from_node].unsqueeze(0),
                embeddings[step.to_node].unsqueeze(0)
            ).item()
            
            self.register_anchor(
                src=step.from_node,
                dst=step.to_node,
                delta=delta,
                strength=1.0,
                rule_name=step.rule_applied
            )
    
    def compute_rigidity_loss(self, embeddings: torch.Tensor, 
                              manifold) -> torch.Tensor:
        """
        Compute rigidity loss to prevent anchor drift.
        
        L_rigid = Σ s_uv · |d(u,v) - δ|
        
        This ensures existing proofs cannot collapse under new learning.
        New learning must "bend around" old logic.
        
        Args:
            embeddings: Current node embeddings [N, D]
            manifold: Geometry manifold with dist() method
            
        Returns:
            Rigidity loss (scalar)
        """
        if not self.anchors:
            return torch.tensor(0.0, device=embeddings.device)
        
        total_loss = torch.tensor(0.0, device=embeddings.device)
        
        for (src, dst), anchor in self.anchors.items():
            # Skip if indices out of range (node was removed)
            if src >= embeddings.size(0) or dst >= embeddings.size(0):
                continue
                
            current_dist = manifold.dist(
                embeddings[src].unsqueeze(0),
                embeddings[dst].unsqueeze(0)
            )
            
            # Weighted absolute deviation from frozen distance
            total_loss = total_loss + anchor.strength * torch.abs(current_dist - anchor.delta)
        
        return total_loss / max(1, len(self.anchors))
    
    def get_k_hop_neighborhood(self, center_nodes: List[int], k: int) -> Set[int]:
        """
        Get all nodes within k hops of the center nodes.
        
        Used to determine which nodes are "plastic" (can be updated)
        when adding new rules incrementally.
        
        Args:
            center_nodes: Starting nodes for BFS
            k: Maximum hop distance
            
        Returns:
            Set of node indices within k hops
        """
        visited = set(center_nodes)
        frontier = set(center_nodes)
        
        for _ in range(k):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.adjacency.get(node, []):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
                
        return visited
    
    def get_frozen_mask(self, total_nodes: int, plastic_nodes: Set[int], 
                       device: torch.device) -> torch.Tensor:
        """
        Get a boolean mask of which nodes should NOT be updated.
        
        Args:
            total_nodes: Total number of nodes
            plastic_nodes: Set of nodes that CAN be updated
            device: Torch device
            
        Returns:
            Boolean tensor [N] where True = frozen
        """
        mask = torch.ones(total_nodes, dtype=torch.bool, device=device)
        for node in plastic_nodes:
            if node < total_nodes:
                mask[node] = False
        return mask
    
    def decay_strengths(self):
        """Apply decay to all anchor strengths (called per epoch)."""
        for anchor in self.anchors.values():
            anchor.strength *= self.decay_rate
    
    def prune_weak_anchors(self, threshold: float = 0.1):
        """Remove anchors with strength below threshold."""
        to_remove = [
            key for key, anchor in self.anchors.items()
            if anchor.strength < threshold
        ]
        for key in to_remove:
            del self.anchors[key]
    
    def get_anchor_stats(self) -> Dict:
        """Get statistics about current anchors."""
        if not self.anchors:
            return {"count": 0}
            
        strengths = [a.strength for a in self.anchors.values()]
        deltas = [a.delta for a in self.anchors.values()]
        
        return {
            "count": len(self.anchors),
            "avg_strength": sum(strengths) / len(strengths),
            "min_strength": min(strengths),
            "max_strength": max(strengths),
            "avg_delta": sum(deltas) / len(deltas),
        }
    
    def save(self, path: str):
        """Serialize anchors to file."""
        import json
        data = {
            "anchors": [
                {
                    "src": a.src, "dst": a.dst, "delta": a.delta,
                    "strength": a.strength, "timestamp": a.timestamp,
                    "rule_name": a.rule_name
                }
                for a in self.anchors.values()
            ],
            "decay_rate": self.decay_rate
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load anchors from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.decay_rate = data.get("decay_rate", 1.0)
        self.anchors.clear()
        self.adjacency.clear()
        
        for a in data["anchors"]:
            self.register_anchor(
                src=a["src"], dst=a["dst"], delta=a["delta"],
                strength=a["strength"], rule_name=a.get("rule_name", "")
            )
