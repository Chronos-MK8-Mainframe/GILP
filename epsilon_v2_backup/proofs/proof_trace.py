"""
Proof Trace

Improvement #3 - Explicit Proof Objects

Make proof paths first-class objects with:
- Recorded hops
- Stored justifications (edge, rule, anchor)
- Structured traces
- Symbolic verification

A reached node ≠ a proof. This module ensures we have REAL proofs.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class ProofStatus(Enum):
    """Status of a proof attempt."""
    SUCCESS = "SUCCESS"
    FAIL_NO_NEIGHBORS = "FAIL_NO_NEIGHBORS"
    FAIL_LOCAL_MINIMUM = "FAIL_LOCAL_MINIMUM"
    FAIL_OSCILLATION = "FAIL_OSCILLATION"
    FAIL_REPULSION_LOCK = "FAIL_REPULSION_LOCK"
    FAIL_FLAT_BASIN = "FAIL_FLAT_BASIN"
    FAIL_MAX_STEPS = "FAIL_MAX_STEPS"
    INCOMPLETE = "INCOMPLETE"


@dataclass
class ProofStep:
    """
    Single step in a proof trace.
    
    Each step represents a logical inference from one node to another,
    justified by a rule in the knowledge base.
    """
    from_node: int                      # Source node index
    to_node: int                        # Destination node index
    rule_applied: str                   # Name of the rule justifying this step
    distance_traveled: float            # Hyperbolic distance for this step
    shell_transition: Tuple[int, int]   # (from_shell, to_shell)
    step_index: int = 0                 # Position in the proof sequence
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize step for logging/output."""
        return {
            "step": self.step_index,
            "from": self.from_node,
            "to": self.to_node,
            "rule": self.rule_applied,
            "distance": round(self.distance_traveled, 4),
            "shells": list(self.shell_transition)
        }


@dataclass 
class ProofTrace:
    """
    Complete proof with verifiable steps.
    
    This is a first-class proof object that can be:
    - Verified against the knowledge base
    - Serialized for logging
    - Used to fossilize verified inferences
    - Inspected for debugging
    
    The key innovation: we don't just say "we found a path",
    we provide a complete, auditable trace of the reasoning.
    """
    start: int                           # Starting node index
    goal: int                            # Goal node index
    steps: List[ProofStep] = field(default_factory=list)
    total_distance: float = 0.0          # Sum of all step distances
    status: ProofStatus = ProofStatus.INCOMPLETE
    failure_reason: str = ""             # Human-readable failure explanation
    nodes_explored: int = 0              # For cost tracking
    
    @property
    def path(self) -> List[int]:
        """Get the node path as a list of indices."""
        if not self.steps:
            return [self.start]
        return [self.start] + [step.to_node for step in self.steps]
    
    @property
    def length(self) -> int:
        """Number of steps in the proof."""
        return len(self.steps)
    
    @property
    def is_success(self) -> bool:
        """Whether the proof reached the goal."""
        return self.status == ProofStatus.SUCCESS
    
    def add_step(self, step: ProofStep):
        """Add a step to the proof trace."""
        step.step_index = len(self.steps)
        self.steps.append(step)
        self.total_distance += step.distance_traveled
    
    def verify(self, knowledge_base) -> Tuple[bool, str]:
        """
        Symbolically check each step against KB rules.
        
        This is the critical function that ensures our geometric
        navigation corresponds to valid logical inference.
        
        Args:
            knowledge_base: KB with has_edge(src, dst) method
            
        Returns:
            (valid: bool, message: str)
        """
        if not self.is_success:
            return False, f"Proof not successful: {self.status.value}"
            
        for i, step in enumerate(self.steps):
            # Check if edge exists in KB
            if not knowledge_base.has_edge(step.from_node, step.to_node):
                return False, f"Invalid step {i}: {step.from_node} → {step.to_node} (no KB edge)"
            
            # Check shell monotonicity (should decrease or stay same)
            from_shell, to_shell = step.shell_transition
            if to_shell > from_shell:
                return False, f"Invalid shell transition at step {i}: {from_shell} → {to_shell}"
        
        # Verify path continuity
        for i in range(1, len(self.steps)):
            if self.steps[i].from_node != self.steps[i-1].to_node:
                return False, f"Discontinuity at step {i}"
        
        # Verify endpoints
        if self.steps and self.steps[-1].to_node != self.goal:
            return False, f"Path does not reach goal: ends at {self.steps[-1].to_node}"
            
        return True, "Proof valid"
    
    def get_rules_used(self) -> List[str]:
        """Get list of all rules used in the proof."""
        return [step.rule_applied for step in self.steps]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/output."""
        valid, message = (True, "Not verified") if not hasattr(self, '_verified') else self._verified
        
        return {
            "start": self.start,
            "goal": self.goal,
            "path": self.path,
            "steps": [step.to_dict() for step in self.steps],
            "total_distance": round(self.total_distance, 4),
            "status": self.status.value,
            "failure_reason": self.failure_reason,
            "nodes_explored": self.nodes_explored,
            "valid": valid,
            "validation_message": message
        }
    
    def to_string(self, knowledge_base=None) -> str:
        """Human-readable proof representation."""
        lines = [
            f"ProofTrace: {self.start} → {self.goal}",
            f"Status: {self.status.value}",
            f"Steps: {self.length}, Distance: {self.total_distance:.4f}",
            "---"
        ]
        
        for step in self.steps:
            rule_str = f" [{step.rule_applied}]" if step.rule_applied else ""
            lines.append(
                f"  {step.step_index}: {step.from_node} → {step.to_node}{rule_str} "
                f"(d={step.distance_traveled:.3f}, shell {step.shell_transition[0]}→{step.shell_transition[1]})"
            )
        
        if self.failure_reason:
            lines.append(f"Failure: {self.failure_reason}")
            
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ProofTrace({self.start}→{self.goal}, status={self.status.value}, steps={self.length})"


@dataclass
class ProofAttemptLog:
    """
    Log of all proof attempts for analysis.
    
    Useful for:
    - Cost tracking
    - Debugging navigation issues
    - Understanding failure patterns
    """
    attempts: List[ProofTrace] = field(default_factory=list)
    
    def add(self, trace: ProofTrace):
        self.attempts.append(trace)
    
    @property
    def success_rate(self) -> float:
        if not self.attempts:
            return 0.0
        successes = sum(1 for t in self.attempts if t.is_success)
        return successes / len(self.attempts)
    
    @property
    def avg_steps(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(t.length for t in self.attempts) / len(self.attempts)
    
    @property
    def failure_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self.attempts:
            key = t.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    def summary(self) -> Dict[str, Any]:
        return {
            "total_attempts": len(self.attempts),
            "success_rate": round(self.success_rate, 4),
            "avg_steps": round(self.avg_steps, 2),
            "failures": self.failure_distribution
        }
