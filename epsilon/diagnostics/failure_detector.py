"""
Failure Detector

Improvement #4 - Formal Failure Semantics

"Failure is meaningful" — but this was underspecified in GILP.
This module defines failure modes as geometric states with clear meaning.

Failure Type       | Geometric Signal           | Meaning
-------------------|----------------------------|----------------------------
NO_DESCENT         | All neighbors farther      | Proof impossible
OSCILLATION        | Equal-distance cycle       | Ambiguous logic
REPULSION_LOCK     | Contradiction barrier      | Inconsistent axioms
FLAT_BASIN         | Zero gradient              | Missing rule

Each failure is:
- Detectable
- Logged
- Interpretable
"""

import torch
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set


class FailureType(Enum):
    """
    Geometric failure states with logical meaning.
    
    These aren't just "errors" — they're diagnostic signals that tell us
    what's wrong with the knowledge base or inference attempt.
    """
    NO_DESCENT = "no_descent"           # All neighbors farther → proof impossible
    OSCILLATION = "oscillation"         # Equal-distance cycle → ambiguous logic
    REPULSION_LOCK = "repulsion_lock"   # Contradiction barrier → inconsistent axioms
    FLAT_BASIN = "flat_basin"           # Zero gradient → missing rule
    

@dataclass
class FailureReport:
    """Detailed report of a detected failure."""
    failure_type: FailureType
    message: str
    node: int                           # Where failure occurred
    goal: int                           # What we were trying to reach
    neighbors_checked: int              # Number of neighbors examined
    min_distance_found: float           # Best distance to goal found
    current_distance: float             # Distance from failure node to goal
    history_length: int                 # How many steps before failure
    
    def to_dict(self):
        return {
            "type": self.failure_type.value,
            "message": self.message,
            "node": self.node,
            "goal": self.goal,
            "neighbors_checked": self.neighbors_checked,
            "min_distance_found": round(self.min_distance_found, 4),
            "current_distance": round(self.current_distance, 4),
            "history_length": self.history_length
        }


class FailureDetector:
    """
    Detects and interprets geometric failure states.
    
    This is the key to making GILP's failures "meaningful" rather than
    mysterious. Each failure type has a clear geometric signature and
    a corresponding logical interpretation.
    """
    
    def __init__(self, manifold, config=None):
        """
        Args:
            manifold: Geometry manifold for distance computation
            config: EpsilonConfig with detection thresholds
        """
        self.manifold = manifold
        
        # Detection parameters
        self.oscillation_window = config.oscillation_window if config else 5
        self.flat_basin_threshold = config.flat_basin_threshold if config else 1e-4
        self.contradiction_margin = config.contradiction_margin if config else 2.0
        
        # Statistics
        self.failure_counts = {ft: 0 for ft in FailureType}
        self.failure_history: List[FailureReport] = []
        
    def detect(self, current: int, neighbors: List[int], goal: int,
              history: List[int],
              embeddings: Optional[torch.Tensor] = None) -> Tuple[Optional[FailureType], str]:
        """
        Analyze current state and return failure type if detected.
        
        Args:
            current: Current node index
            neighbors: List of neighbor node indices
            goal: Goal node index
            history: List of previously visited nodes
            embeddings: Optional embeddings for distance computation
            
        Returns:
            (failure_type, message) or (None, None) if no failure
        """
        if embeddings is not None:
            self._embeddings = embeddings
        
        # No neighbors → can't proceed
        if not neighbors:
            failure = FailureType.NO_DESCENT
            msg = "No neighbors within step radius"
            self._record_failure(failure, msg, current, goal, 0, float('inf'), float('inf'), len(history))
            return failure, msg
        
        # Compute distances
        current_dist = self._dist_to_goal(current, goal)
        neighbor_dists = [(n, self._dist_to_goal(n, goal)) for n in neighbors]
        min_neighbor_dist = min(d for _, d in neighbor_dists)
        
        # Check NO_DESCENT: All neighbors farther
        if all(d >= current_dist for _, d in neighbor_dists):
            failure = FailureType.NO_DESCENT
            msg = f"Local minimum - all {len(neighbors)} neighbors farther (best: {min_neighbor_dist:.4f} vs current: {current_dist:.4f})"
            self._record_failure(failure, msg, current, goal, len(neighbors), min_neighbor_dist, current_dist, len(history))
            return failure, msg
        
        # Check OSCILLATION: Cycle detection
        if len(history) >= self.oscillation_window:
            recent = history[-self.oscillation_window:]
            unique_recent = set(recent)
            if len(unique_recent) < self.oscillation_window // 2:
                failure = FailureType.OSCILLATION
                msg = f"Cycle detected in recent {self.oscillation_window} steps: {unique_recent}"
                self._record_failure(failure, msg, current, goal, len(neighbors), min_neighbor_dist, current_dist, len(history))
                return failure, msg
        
        # Check FLAT_BASIN: Zero gradient (all neighbors at same distance)
        if len(neighbor_dists) > 1:
            distances = [d for _, d in neighbor_dists]
            dist_variance = max(distances) - min(distances)
            if dist_variance < self.flat_basin_threshold:
                failure = FailureType.FLAT_BASIN
                msg = f"Flat basin - distance variance {dist_variance:.6f} below threshold"
                self._record_failure(failure, msg, current, goal, len(neighbors), min_neighbor_dist, current_dist, len(history))
                return failure, msg
        
        # Check REPULSION_LOCK: This requires knowing about contradictions
        # We detect this when the best neighbor is significantly worse than expected
        # (indicating we're being "pushed away" by a contradiction)
        best_improvement = current_dist - min_neighbor_dist
        if best_improvement < 0 and abs(best_improvement) > self.contradiction_margin:
            failure = FailureType.REPULSION_LOCK
            msg = f"Repulsion lock - best neighbor {self.contradiction_margin:.2f}+ farther, possible contradiction barrier"
            self._record_failure(failure, msg, current, goal, len(neighbors), min_neighbor_dist, current_dist, len(history))
            return failure, msg
        
        # No failure detected
        return None, ""
    
    def detect_with_report(self, current: int, neighbors: List[int], goal: int,
                          history: List[int],
                          embeddings: Optional[torch.Tensor] = None) -> Optional[FailureReport]:
        """
        Detect failure and return detailed report if found.
        
        Returns:
            FailureReport or None if no failure
        """
        failure_type, msg = self.detect(current, neighbors, goal, history, embeddings)
        
        if failure_type:
            # Get the last recorded failure (which we just created)
            return self.failure_history[-1] if self.failure_history else None
        
        return None
    
    def _dist_to_goal(self, node: int, goal: int) -> float:
        """Compute hyperbolic distance from node to goal."""
        if not hasattr(self, '_embeddings') or self._embeddings is None:
            return float('inf')
        
        node_emb = self._embeddings[node].unsqueeze(0)
        goal_emb = self._embeddings[goal].unsqueeze(0)
        return self.manifold.dist(node_emb, goal_emb).item()
    
    def _record_failure(self, failure_type: FailureType, message: str,
                       node: int, goal: int, neighbors_checked: int,
                       min_distance: float, current_distance: float,
                       history_length: int):
        """Record a failure for statistics and history."""
        self.failure_counts[failure_type] += 1
        
        report = FailureReport(
            failure_type=failure_type,
            message=message,
            node=node,
            goal=goal,
            neighbors_checked=neighbors_checked,
            min_distance_found=min_distance,
            current_distance=current_distance,
            history_length=history_length
        )
        self.failure_history.append(report)
        
    def get_failure_stats(self) -> dict:
        """Get statistics about detected failures."""
        total = sum(self.failure_counts.values())
        return {
            "total_failures": total,
            "by_type": {ft.value: count for ft, count in self.failure_counts.items()},
            "distribution": {
                ft.value: round(count / total, 4) if total > 0 else 0
                for ft, count in self.failure_counts.items()
            }
        }
    
    def get_failure_interpretation(self, failure_type: FailureType) -> str:
        """
        Get human-readable interpretation of what a failure means.
        
        This is the key insight: geometric failures have logical meanings.
        """
        interpretations = {
            FailureType.NO_DESCENT: 
                "PROOF IMPOSSIBLE: The current node is a local minimum in the geometry. "
                "There is no valid inference path from here to the goal. "
                "This likely means the goal is logically unreachable from the current premises.",
            
            FailureType.OSCILLATION:
                "AMBIGUOUS LOGIC: The navigation is cycling between nodes without progress. "
                "This suggests the knowledge base has ambiguous or conflicting inference paths. "
                "The geometry cannot resolve which direction leads to the goal.",
            
            FailureType.REPULSION_LOCK:
                "INCONSISTENT AXIOMS: A contradiction barrier is blocking progress. "
                "The geometry has learned to repel paths that would lead through inconsistent logic. "
                "Check for contradictory rules in the knowledge base.",
            
            FailureType.FLAT_BASIN:
                "MISSING RULE: The geometry shows uniform distance in all directions. "
                "This suggests a missing inference rule — the knowledge base doesn't know "
                "how to proceed from this point. Adding relevant rules may resolve this."
        }
        return interpretations.get(failure_type, "Unknown failure type")
    
    def clear_history(self):
        """Clear failure history and reset counts."""
        self.failure_counts = {ft: 0 for ft in FailureType}
        self.failure_history.clear()
