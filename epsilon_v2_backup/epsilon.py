"""
Epsilon: A Reliable Reasoning Substrate

Not "geometry is logic" — but:
"Logic can be compiled into a stable execution manifold"

Build a map where:
- Reasoning is walking
- Updates don't break roads  
- Dead ends are real

This is how you win:
- A reliable reasoning substrate
- That other systems can call
- That fails honestly
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

import sys
sys.path.insert(0, 'c:/Users/rupa9/Videos/GILP/GILP-main/GILP-main')

from epsilon.config import EpsilonConfig
from epsilon.geometry.quantized_poincare import QuantizedPoincareManifold
from epsilon.anchoring.anchor_manager import AnchorManager
from epsilon.proofs.proof_trace import ProofTrace, ProofStatus
from epsilon.proofs.proof_navigator import ProofNavigator
from epsilon.diagnostics.failure_detector import FailureDetector
from epsilon.metrics.cost_tracker import CostTracker
from epsilon.training.trainer import EpsilonTrainer


class GeometricReasoningEngine:
    """
    The Epsilon reasoning substrate.
    
    Designed to be:
    - Called by other systems
    - Fail honestly
    - Provide verifiable proofs
    - Update incrementally without breaking existing proofs
    
    Core capabilities:
    - learn(): Train or update geometric embedding from knowledge base
    - prove(): Find verifiable proof paths between logical concepts
    - fossilize(): Lock verified proofs as permanent geometry
    - get_costs(): Honest cost/benefit metrics
    """
    
    def __init__(self, model: Optional[nn.Module] = None,
                 knowledge_base=None,
                 config: Optional[EpsilonConfig] = None):
        """
        Initialize Epsilon.
        
        Args:
            model: Neural network for embedding (if None, must be set before learn())
            knowledge_base: Knowledge base with rules and edges
            config: Configuration parameters
        """
        self.config = config or EpsilonConfig()
        self.config.validate()
        
        # Core components
        self.model = model
        self.kb = knowledge_base
        
        # Geometry with metric quantization
        self.manifold = QuantizedPoincareManifold(
            shell_width=self.config.shell_radius,
            eps=self.config.manifold_eps
        )
        
        # Incremental fossilization
        self.anchors = AnchorManager(decay_rate=self.config.anchor_decay)
        
        # Failure detection
        self.failure_detector = FailureDetector(self.manifold, self.config)
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        
        # Trainer and navigator (initialized lazily)
        self._trainer: Optional[EpsilonTrainer] = None
        self._navigator: Optional[ProofNavigator] = None
        self._embeddings: Optional[torch.Tensor] = None
        
    def set_model(self, model: nn.Module):
        """Set or update the neural network model."""
        self.model = model
        self._trainer = None  # Reset trainer
        
    def set_knowledge_base(self, kb):
        """Set or update the knowledge base."""
        self.kb = kb
        
    @property
    def trainer(self) -> EpsilonTrainer:
        """Lazily create trainer."""
        if self._trainer is None:
            if self.model is None:
                raise ValueError("Model must be set before training")
            self._trainer = EpsilonTrainer(self.model, self.anchors, self.config)
        return self._trainer
    
    @property
    def navigator(self) -> ProofNavigator:
        """Lazily create navigator."""
        if self._navigator is None:
            self._navigator = ProofNavigator(
                embeddings=self._embeddings,
                manifold=self.manifold,
                knowledge_base=self.kb,
                failure_detector=self.failure_detector,
                config=self.config
            )
        return self._navigator
    
    def learn(self, rule_tokens: torch.Tensor,
              rule_types: torch.Tensor,
              edge_index: torch.Tensor,
              edge_type: torch.Tensor,
              epochs: int = 100,
              incremental: bool = False,
              new_nodes: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Train or update the geometric embedding.
        
        Args:
            rule_tokens: Token sequences [N, seq_len]
            rule_types: Rule type labels [N]
            edge_index: Edge indices [2, E]
            edge_type: Edge type labels [E]
            epochs: Number of training epochs
            incremental: If True, only update near new_nodes
            new_nodes: Indices of newly added nodes (required if incremental=True)
            
        Returns:
            Final training metrics
        """
        if incremental and new_nodes is None:
            raise ValueError("new_nodes required for incremental learning")
        
        metrics = {}
        for epoch in range(epochs):
            if incremental:
                metrics = self.trainer.train_step_incremental(
                    rule_tokens, rule_types, edge_index, edge_type, new_nodes
                )
            else:
                metrics = self.trainer.train_step(
                    rule_tokens, rule_types, edge_index, edge_type
                )
            
            if epoch % 10 == 0:
                self.cost_tracker.log_training_epoch(epoch, metrics["loss"])
        
        # Update embeddings
        self._embeddings = self.trainer.get_embeddings(
            rule_tokens, rule_types, edge_index, edge_type, mode="text"
        )
        
        # Update navigator
        if self._navigator is not None:
            self._navigator.update_embeddings(self._embeddings)
        
        return metrics
    
    def prove(self, start: int, goal: int,
             use_beam_search: bool = False) -> ProofTrace:
        """
        Attempt to find a proof from start to goal.
        
        Returns a verifiable ProofTrace (success or failure).
        The trace includes:
        - Complete path with step-by-step justifications
        - Shell transitions showing proof depth
        - Failure type and interpretation if unsuccessful
        
        Args:
            start: Starting node index
            goal: Goal node index
            use_beam_search: Use beam search for more robust navigation
            
        Returns:
            ProofTrace with complete proof or diagnostic failure info
        """
        if self._embeddings is None:
            raise ValueError("Must call learn() before prove()")
        
        self.cost_tracker.start_timer()
        
        # Navigate
        if use_beam_search:
            trace = self.navigator.navigate_with_beam(start, goal)
        else:
            trace = self.navigator.navigate(start, goal)
        
        # Log inference cost
        baseline = self.cost_tracker.estimate_baseline_steps(
            self._embeddings.size(0), trace.length
        )
        self.cost_tracker.log_inference(
            steps_taken=trace.nodes_explored,
            baseline_steps=baseline,
            success=trace.is_success
        )
        
        return trace
    
    def verify(self, trace: ProofTrace) -> tuple:
        """
        Verify a proof trace against the knowledge base.
        
        Args:
            trace: ProofTrace to verify
            
        Returns:
            (valid: bool, message: str)
        """
        if self.kb is None:
            return False, "No knowledge base set"
        return trace.verify(self.kb)
    
    def fossilize(self, proven_trace: ProofTrace):
        """
        Lock a verified proof into permanent geometry.
        
        After fossilization:
        - The proof distances become anchored
        - Future learning must bend around this proof
        - The proof cannot collapse under new training
        
        Args:
            proven_trace: Successfully verified ProofTrace
        """
        if not proven_trace.is_success:
            raise ValueError("Can only fossilize successful proofs")
        
        if self._embeddings is None:
            raise ValueError("No embeddings available")
        
        self.anchors.register_from_proof_trace(
            proven_trace, self._embeddings, self.manifold
        )
    
    def get_costs(self) -> Dict[str, float]:
        """Get honest cost/benefit metrics."""
        return self.cost_tracker.get_metrics()
    
    def get_cost_summary(self) -> str:
        """Get human-readable cost summary."""
        return self.cost_tracker.get_summary()
    
    def get_failure_stats(self) -> Dict:
        """Get failure detection statistics."""
        return self.failure_detector.get_failure_stats()
    
    def get_anchor_stats(self) -> Dict:
        """Get anchor/rigidity statistics."""
        return self.anchors.get_anchor_stats()
    
    def explain_failure(self, trace: ProofTrace) -> str:
        """
        Get human-readable explanation of why a proof failed.
        
        Args:
            trace: Failed ProofTrace
            
        Returns:
            Explanation string
        """
        if trace.is_success:
            return "Proof succeeded, no failure to explain"
        
        # Map proof status to failure type
        from epsilon.diagnostics.failure_detector import FailureType
        
        status_to_type = {
            ProofStatus.FAIL_LOCAL_MINIMUM: FailureType.NO_DESCENT,
            ProofStatus.FAIL_OSCILLATION: FailureType.OSCILLATION,
            ProofStatus.FAIL_REPULSION_LOCK: FailureType.REPULSION_LOCK,
            ProofStatus.FAIL_FLAT_BASIN: FailureType.FLAT_BASIN,
        }
        
        failure_type = status_to_type.get(trace.status)
        if failure_type:
            return self.failure_detector.get_failure_interpretation(failure_type)
        
        return f"Proof failed: {trace.status.value} - {trace.failure_reason}"
    
    def save(self, path: str):
        """Save Epsilon state to directory."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save model
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        
        # Save embeddings
        if self._embeddings is not None:
            torch.save(self._embeddings, os.path.join(path, "embeddings.pt"))
        
        # Save anchors
        self.anchors.save(os.path.join(path, "anchors.json"))
        
        # Save costs
        self.cost_tracker.save(os.path.join(path, "costs.json"))
    
    def load(self, path: str):
        """Load Epsilon state from directory."""
        import os
        
        # Load model
        model_path = os.path.join(path, "model.pt")
        if os.path.exists(model_path) and self.model is not None:
            self.model.load_state_dict(torch.load(model_path))
        
        # Load embeddings
        emb_path = os.path.join(path, "embeddings.pt")
        if os.path.exists(emb_path):
            self._embeddings = torch.load(emb_path)
            if self._navigator is not None:
                self._navigator.update_embeddings(self._embeddings)
        
        # Load anchors
        anchor_path = os.path.join(path, "anchors.json")
        if os.path.exists(anchor_path):
            self.anchors.load(anchor_path)
