"""
Proof Navigator

Improvement #3 - Explicit Proof Objects (Part 2)

Enhanced navigator that builds proof traces during search.
Geometry executes, logic validates.

Key features:
- Records each hop with justification
- Integrates with failure detection
- Produces verifiable ProofTrace objects
"""

import torch
from typing import List, Optional, Set, Tuple
from sklearn.neighbors import KDTree
import numpy as np

from epsilon.proofs.proof_trace import ProofStep, ProofTrace, ProofStatus


class ProofNavigator:
    """
    Navigation that produces verifiable proof objects.
    
    Unlike simple pathfinding, this navigator:
    1. Records every step with its justification
    2. Detects and categorizes failures
    3. Produces complete ProofTrace objects
    4. Enables post-hoc verification
    
    Geometry executes, logic validates.
    """
    
    def __init__(self, embeddings: torch.Tensor, 
                 manifold,
                 knowledge_base=None,
                 failure_detector=None,
                 config=None):
        """
        Args:
            embeddings: Node embeddings in Poincaré ball [N, D]
            manifold: QuantizedPoincareManifold for distance/shell computation
            knowledge_base: Optional KB for rule lookups
            failure_detector: Optional FailureDetector for failure categorization
            config: Optional EpsilonConfig for parameters
        """
        self.manifold = manifold
        self.kb = knowledge_base
        self.failure_detector = failure_detector
        self.config = config
        
        # Convert embeddings for computation
        if embeddings is not None:
            self.embeddings = embeddings.detach()
            self.embeddings_np = embeddings.detach().cpu().numpy()
            self.tree = KDTree(self.embeddings_np)
        else:
            self.embeddings = None
            self.embeddings_np = None
            self.tree = None
    
    def update_embeddings(self, embeddings: torch.Tensor):
        """Update embeddings after training."""
        self.embeddings = embeddings.detach()
        self.embeddings_np = embeddings.detach().cpu().numpy()
        self.tree = KDTree(self.embeddings_np)
    
    def navigate(self, start_idx: int, goal_idx: int,
                max_steps: Optional[int] = None,
                step_radius: Optional[float] = None) -> ProofTrace:
        """
        Navigate from start to goal, producing a verifiable proof trace.
        
        Uses hyperbolic greedy best-first search with explicit step recording.
        
        Args:
            start_idx: Starting node index
            goal_idx: Goal node index
            max_steps: Maximum navigation steps (default from config)
            step_radius: Radius for neighbor search (default from config)
            
        Returns:
            ProofTrace with complete step history
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not set. Call update_embeddings first.")
        
        # Get parameters from config or defaults
        max_steps = max_steps or (self.config.max_navigation_steps if self.config else 50)
        step_radius = step_radius or (self.config.neighbor_radius if self.config else 0.6)
        
        # Initialize proof trace
        trace = ProofTrace(start=start_idx, goal=goal_idx)
        current = start_idx
        history: List[int] = [start_idx]
        
        # Get goal embedding for distance computation
        goal_emb = self.embeddings[goal_idx].unsqueeze(0)
        current_dist_to_goal = self._dist_to_goal(current, goal_idx)
        
        for step_num in range(max_steps):
            trace.nodes_explored += 1
            
            # Check if we've reached the goal
            if current == goal_idx:
                trace.status = ProofStatus.SUCCESS
                return trace
            
            # Get neighbors within step radius
            neighbors = self._get_neighbors(current, step_radius)
            
            # Check for failure using failure detector
            if self.failure_detector:
                failure_type, failure_msg = self.failure_detector.detect(
                    current, neighbors, goal_idx, history
                )
                if failure_type:
                    trace.status = self._failure_type_to_status(failure_type)
                    trace.failure_reason = failure_msg
                    return trace
            
            # No neighbors available
            if not neighbors:
                trace.status = ProofStatus.FAIL_NO_NEIGHBORS
                trace.failure_reason = f"No neighbors within radius {step_radius}"
                return trace
            
            # Find best neighbor (minimum distance to goal)
            best_neighbor = None
            best_dist = float('inf')
            
            for n in neighbors:
                if n == current:
                    continue
                dist_to_goal = self._dist_to_goal(n, goal_idx)
                if dist_to_goal < best_dist:
                    best_dist = dist_to_goal
                    best_neighbor = n
            
            # Check for local minimum (no progress possible)
            if best_neighbor is None or best_dist >= current_dist_to_goal:
                trace.status = ProofStatus.FAIL_LOCAL_MINIMUM
                trace.failure_reason = f"Local minimum at node {current}"
                return trace
            
            # Record the step
            step = self._create_step(current, best_neighbor)
            trace.add_step(step)
            
            # Move to best neighbor
            history.append(best_neighbor)
            current = best_neighbor
            current_dist_to_goal = best_dist
        
        # Exceeded max steps
        trace.status = ProofStatus.FAIL_MAX_STEPS
        trace.failure_reason = f"Exceeded {max_steps} steps"
        return trace
    
    def navigate_with_beam(self, start_idx: int, goal_idx: int,
                          beam_width: int = 3,
                          max_steps: Optional[int] = None,
                          step_radius: Optional[float] = None) -> ProofTrace:
        """
        Beam search variant for more robust navigation.
        
        Maintains multiple candidate paths and returns the best one.
        
        Args:
            start_idx: Starting node index
            goal_idx: Goal node index
            beam_width: Number of candidates to maintain
            max_steps: Maximum navigation steps
            step_radius: Radius for neighbor search
            
        Returns:
            ProofTrace for the best path found
        """
        max_steps = max_steps or (self.config.max_navigation_steps if self.config else 50)
        step_radius = step_radius or (self.config.neighbor_radius if self.config else 0.6)
        
        # Each beam entry: (distance_to_goal, trace, current_node, history)
        beams = [(self._dist_to_goal(start_idx, goal_idx), 
                  ProofTrace(start=start_idx, goal=goal_idx),
                  start_idx,
                  [start_idx])]
        
        for step_num in range(max_steps):
            new_beams = []
            
            for dist, trace, current, history in beams:
                # Check if this beam reached goal
                if current == goal_idx:
                    trace.status = ProofStatus.SUCCESS
                    return trace
                
                # Expand this beam
                neighbors = self._get_neighbors(current, step_radius)
                
                for n in neighbors:
                    if n == current or n in history[-5:]:  # Avoid recent cycles
                        continue
                    
                    new_dist = self._dist_to_goal(n, goal_idx)
                    
                    # Create new trace with this step
                    new_trace = ProofTrace(
                        start=start_idx, goal=goal_idx,
                        steps=list(trace.steps),
                        total_distance=trace.total_distance,
                        nodes_explored=trace.nodes_explored + 1
                    )
                    new_trace.add_step(self._create_step(current, n))
                    
                    new_beams.append((new_dist, new_trace, n, history + [n]))
            
            if not new_beams:
                # All beams stuck
                best = min(beams, key=lambda b: b[0])
                best[1].status = ProofStatus.FAIL_LOCAL_MINIMUM
                best[1].failure_reason = "All beams stuck"
                return best[1]
            
            # Keep top beam_width beams
            new_beams.sort(key=lambda b: b[0])
            beams = new_beams[:beam_width]
        
        # Return best beam
        best = min(beams, key=lambda b: b[0])
        best[1].status = ProofStatus.FAIL_MAX_STEPS
        best[1].failure_reason = f"Exceeded {max_steps} steps"
        return best[1]
    
    def _get_neighbors(self, node_idx: int, radius: float) -> List[int]:
        """Get neighbors within Euclidean radius (proxy for locality)."""
        query = self.embeddings_np[node_idx].reshape(1, -1)
        indices = self.tree.query_radius(query, r=radius)[0]
        return [int(i) for i in indices if i != node_idx]
    
    def _dist_to_goal(self, node_idx: int, goal_idx: int) -> float:
        """Compute hyperbolic distance from node to goal."""
        node_emb = self.embeddings[node_idx].unsqueeze(0)
        goal_emb = self.embeddings[goal_idx].unsqueeze(0)
        return self.manifold.dist(node_emb, goal_emb).item()
    
    def _get_shell(self, node_idx: int) -> int:
        """Get the shell (proof depth) of a node."""
        return self.manifold.get_shell(self.embeddings[node_idx].unsqueeze(0)).item()
    
    def _create_step(self, from_node: int, to_node: int) -> ProofStep:
        """Create a ProofStep with all metadata."""
        distance = self._dist(from_node, to_node)
        from_shell = self._get_shell(from_node)
        to_shell = self._get_shell(to_node)
        
        rule_name = ""
        if self.kb is not None:
            rule = self.kb.get_rule_between(from_node, to_node)
            rule_name = rule.name if rule else ""
        
        return ProofStep(
            from_node=from_node,
            to_node=to_node,
            rule_applied=rule_name,
            distance_traveled=distance,
            shell_transition=(from_shell, to_shell)
        )
    
    def _dist(self, node_a: int, node_b: int) -> float:
        """Compute hyperbolic distance between two nodes."""
        emb_a = self.embeddings[node_a].unsqueeze(0)
        emb_b = self.embeddings[node_b].unsqueeze(0)
        return self.manifold.dist(emb_a, emb_b).item()
    
    def _failure_type_to_status(self, failure_type) -> ProofStatus:
        """Convert FailureType enum to ProofStatus."""
        from epsilon.diagnostics.failure_detector import FailureType
        
        mapping = {
            FailureType.NO_DESCENT: ProofStatus.FAIL_LOCAL_MINIMUM,
            FailureType.OSCILLATION: ProofStatus.FAIL_OSCILLATION,
            FailureType.REPULSION_LOCK: ProofStatus.FAIL_REPULSION_LOCK,
            FailureType.FLAT_BASIN: ProofStatus.FAIL_FLAT_BASIN,
        }
        return mapping.get(failure_type, ProofStatus.FAIL_LOCAL_MINIMUM)
