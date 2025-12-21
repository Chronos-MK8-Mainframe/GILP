
"""
Hypothesis Space (Working Memory Manifold)

A volatile geometric buffer for storing new, unverified information.
Concepts here are "Provisional". They must be verified or reinforced 
before being fossilized into the main Logic Manifold.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from epsilon.geometry.quantized_poincare import QuantizedPoincareManifold

class HypothesisSpace:
    """
    A temporary manifold for short-term memory and new ideas.
    """
    def __init__(self, embedding_dim: int = 64, capacity: int = 1000):
        self.dim = embedding_dim
        self.capacity = capacity
        
        # Volatile Manifold
        self.manifold = QuantizedPoincareManifold()
        
        # Storage
        # We use a simple dictionary for the buffer, mapping concepts to volatile embeddings
        self.concepts: Dict[str, torch.Tensor] = {}
        self.confidence: Dict[str, float] = {} # 0.0 to 1.0
        
    def add_hypothesis(self, concept: str, vector: torch.Tensor, confidence: float = 0.5):
        """
        Add a new concept to the working memory.
        If it exists, update it (reinforcement).
        """
        if len(self.concepts) >= self.capacity:
            self._forget_weakest()
            
        if concept in self.concepts:
            # Reinforcement Learning: Average positions, boost confidence
            old_vec = self.concepts[concept]
            new_vec = (old_vec + vector) / 2.0 
            # Project back to manifold
            new_vec = self.manifold.project_to_shell(new_vec, self.manifold.get_shell(new_vec))
            
            self.concepts[concept] = new_vec
            self.confidence[concept] = min(1.0, self.confidence[concept] + 0.1)
        else:
            # New Idea
            self.concepts[concept] = vector
            self.confidence[concept] = confidence
            
    def query(self, concept: str) -> Optional[torch.Tensor]:
        """Retrieve a vector if exists in hypothesis space."""
        return self.concepts.get(concept)
        
    def promote_strong_hypotheses(self, threshold: float = 0.9) -> List[Tuple[str, torch.Tensor]]:
        """
        Identify hypotheses that are strong enough to become permanent Laws.
        Returns list of (Concept, Vector) to be moved to Logic Manifold.
        """
        promotions = []
        for concept, conf in self.confidence.items():
            if conf >= threshold:
                promotions.append((concept, self.concepts[concept]))
        
        # Remove promoted items from working memory (they are now Facts)
        for concept, _ in promotions:
            del self.concepts[concept]
            del self.confidence[concept]
            
        return promotions
        
    def _forget_weakest(self):
        """Garbage collection: Remove lowest confidence item."""
        if not self.concepts:
            return
            
        weakest = min(self.confidence, key=self.confidence.get)
        del self.concepts[weakest]
        del self.confidence[weakest]

    def clear(self):
        """Wipe working memory (e.g. end of session)."""
        self.concepts.clear()
        self.confidence.clear()
