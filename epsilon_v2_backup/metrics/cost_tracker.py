"""
Cost Tracker

Improvement #5 - Training-time vs Inference-time Honesty

The GILP repo blurs the cost tradeoff. This module makes it explicit:
- Measure total search eliminated
- Measure training compute spent
- Compare amortized cost

This reframes the contribution honestly.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class InferenceRecord:
    """Record of a single inference attempt."""
    steps_taken: int           # Actual steps in geometric navigation
    baseline_steps: int        # Estimated brute-force search steps
    success: bool              # Whether proof was found
    time_ms: float             # Inference time in milliseconds


@dataclass
class TrainingRecord:
    """Record of training statistics."""
    epoch: int
    loss: float
    time_ms: float
    num_samples: int


class CostTracker:
    """
    Honest accounting of training vs inference costs.
    
    This is crucial for credibility: we need to show that the
    training investment pays off in reduced inference cost.
    
    Key metrics:
    - Amortized cost: training_time_ms / search_steps_avoided
    - Search reduction ratio: steps_avoided / steps_taken
    - ROI: steps_avoided_per_training_hour
    """
    
    def __init__(self):
        # Training statistics
        self.training_steps = 0
        self.training_time_ms = 0.0
        self.training_records: List[TrainingRecord] = []
        
        # Inference statistics
        self.inference_steps_taken = 0
        self.inference_steps_avoided = 0
        self.inference_count = 0
        self.inference_successes = 0
        self.inference_time_ms = 0.0
        self.inference_records: List[InferenceRecord] = []
        
        # For live tracking
        self._current_start_time: Optional[float] = None
        
    def start_timer(self):
        """Start a timer for the next operation."""
        self._current_start_time = time.time()
        
    def _elapsed_ms(self) -> float:
        """Get elapsed time since start_timer() in ms."""
        if self._current_start_time is None:
            return 0.0
        return (time.time() - self._current_start_time) * 1000
    
    # --- Training Logging ---
    
    def log_training_epoch(self, epoch: int, loss: float, 
                          time_ms: Optional[float] = None,
                          num_samples: int = 1):
        """
        Log a training epoch.
        
        Args:
            epoch: Epoch number
            loss: Training loss
            time_ms: Time taken (or auto from timer)
            num_samples: Number of samples trained
        """
        elapsed = time_ms if time_ms is not None else self._elapsed_ms()
        
        self.training_steps += num_samples
        self.training_time_ms += elapsed
        
        record = TrainingRecord(
            epoch=epoch,
            loss=loss,
            time_ms=elapsed,
            num_samples=num_samples
        )
        self.training_records.append(record)
        
    def log_training_step(self, time_ms: Optional[float] = None):
        """Log a single training step (simpler than epoch)."""
        elapsed = time_ms if time_ms is not None else self._elapsed_ms()
        self.training_steps += 1
        self.training_time_ms += elapsed
    
    # --- Inference Logging ---
    
    def log_inference(self, steps_taken: int, baseline_steps: int,
                     success: bool = True,
                     time_ms: Optional[float] = None):
        """
        Log an inference attempt.
        
        Args:
            steps_taken: Actual steps used by geometric navigation
            baseline_steps: Estimated steps for brute-force search
            success: Whether the proof was found
            time_ms: Inference time (or auto from timer)
        """
        elapsed = time_ms if time_ms is not None else self._elapsed_ms()
        
        self.inference_count += 1
        self.inference_steps_taken += steps_taken
        self.inference_steps_avoided += max(0, baseline_steps - steps_taken)
        self.inference_time_ms += elapsed
        
        if success:
            self.inference_successes += 1
        
        record = InferenceRecord(
            steps_taken=steps_taken,
            baseline_steps=baseline_steps,
            success=success,
            time_ms=elapsed
        )
        self.inference_records.append(record)
    
    def estimate_baseline_steps(self, num_nodes: int, path_length: int) -> int:
        """
        Estimate brute-force search steps for a problem.
        
        Conservative estimate: BFS would explore O(branching_factor^depth) nodes.
        We assume average branching factor of 5 for logical KB.
        """
        avg_branching_factor = 5
        # Geometric series: 1 + b + b^2 + ... + b^d = (b^(d+1) - 1) / (b - 1)
        estimated = min(
            num_nodes,  # Can't explore more than all nodes
            (avg_branching_factor ** (path_length + 1) - 1) // (avg_branching_factor - 1)
        )
        return max(estimated, path_length)  # At minimum, path length
    
    # --- Metrics ---
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get all cost metrics.
        
        Returns dict with:
        - amortized_cost: ms spent training per search step saved
        - search_reduction_ratio: steps avoided / steps taken
        - steps_avoided_per_training_hour: ROI metric
        - success_rate: inference success rate
        - avg_inference_time_ms: average inference time
        """
        return {
            "amortized_cost_ms_per_step": self._safe_div(
                self.training_time_ms, 
                self.inference_steps_avoided
            ),
            "search_reduction_ratio": self._safe_div(
                self.inference_steps_avoided,
                self.inference_steps_taken
            ),
            "steps_avoided_per_training_hour": self._safe_div(
                self.inference_steps_avoided,
                self.training_time_ms / 3600000  # Convert ms to hours
            ),
            "total_training_time_s": self.training_time_ms / 1000,
            "total_inference_time_ms": self.inference_time_ms,
            "total_training_steps": self.training_steps,
            "total_inference_count": self.inference_count,
            "total_steps_taken": self.inference_steps_taken,
            "total_steps_avoided": self.inference_steps_avoided,
            "success_rate": self._safe_div(
                self.inference_successes,
                self.inference_count
            ),
            "avg_inference_time_ms": self._safe_div(
                self.inference_time_ms,
                self.inference_count
            ),
            "avg_steps_per_inference": self._safe_div(
                self.inference_steps_taken,
                self.inference_count
            )
        }
    
    def _safe_div(self, numerator: float, denominator: float) -> float:
        """Safe division handling zero denominator."""
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    def get_summary(self) -> str:
        """Get human-readable cost summary."""
        m = self.get_metrics()
        
        lines = [
            "=== Epsilon Cost Analysis ===",
            f"Training: {m['total_training_steps']} steps in {m['total_training_time_s']:.2f}s",
            f"Inference: {m['total_inference_count']} proofs, {m['success_rate']*100:.1f}% success",
            "",
            "Search Efficiency:",
            f"  Steps taken: {m['total_steps_taken']}",
            f"  Steps avoided: {m['total_steps_avoided']}",
            f"  Reduction ratio: {m['search_reduction_ratio']:.2f}x",
            "",
            "Cost Analysis:",
            f"  Amortized cost: {m['amortized_cost_ms_per_step']:.4f} ms/step saved",
            f"  ROI: {m['steps_avoided_per_training_hour']:.0f} steps saved per training hour",
        ]
        return "\n".join(lines)
    
    def reset(self):
        """Reset all statistics."""
        self.training_steps = 0
        self.training_time_ms = 0.0
        self.training_records.clear()
        
        self.inference_steps_taken = 0
        self.inference_steps_avoided = 0
        self.inference_count = 0
        self.inference_successes = 0
        self.inference_time_ms = 0.0
        self.inference_records.clear()
    
    def save(self, path: str):
        """Save cost data to file."""
        import json
        data = {
            "metrics": self.get_metrics(),
            "training_records": [
                {"epoch": r.epoch, "loss": r.loss, "time_ms": r.time_ms, "samples": r.num_samples}
                for r in self.training_records
            ],
            "inference_records": [
                {"steps": r.steps_taken, "baseline": r.baseline_steps, 
                 "success": r.success, "time_ms": r.time_ms}
                for r in self.inference_records
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
