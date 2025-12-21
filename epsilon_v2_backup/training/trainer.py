"""
Epsilon Trainer

Unified training combining all improvements:
- Incremental fossilization (rigidity loss)
- Metric quantization (shell ordering)
- Local-only updates (masked gradients)
- Cost tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Optional, Dict, List, Set

import sys
sys.path.insert(0, 'c:/Users/rupa9/Videos/GILP/GILP-main/GILP-main')

from epsilon.config import EpsilonConfig
from epsilon.geometry.quantized_poincare import QuantizedPoincareManifold
from epsilon.anchoring.anchor_manager import AnchorManager
from epsilon.anchoring.local_optimizer import LocalOptimizer, IncrementalUpdateContext
from epsilon.metrics.cost_tracker import CostTracker


class EpsilonTrainer:
    """
    Epsilon training with:
    - Incremental fossilization (rigidity loss)
    - Metric quantization (shell ordering)  
    - Local-only updates (masked gradients)
    - Cost transparency
    
    The key innovation: geometry becomes plastic locally, rigid globally.
    New learning must bend around existing verified proofs.
    """
    
    def __init__(self, model: nn.Module, 
                 anchor_manager: AnchorManager,
                 config: Optional[EpsilonConfig] = None):
        """
        Args:
            model: The neural network model (e.g., StructureAwareGraphEmbedding)
            anchor_manager: AnchorManager for incremental fossilization
            config: EpsilonConfig with hyperparameters
        """
        self.model = model
        self.anchors = anchor_manager
        self.config = config or EpsilonConfig()
        
        # Geometry with metric quantization
        self.manifold = QuantizedPoincareManifold(
            shell_width=self.config.shell_radius,
            eps=self.config.manifold_eps
        )
        
        # Local optimizer for masked gradient updates
        self.local_optimizer = LocalOptimizer(model, base_lr=self.config.learning_rate)
        
        # Standard optimizer for full training
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        
    def train_step(self, rule_tokens: torch.Tensor,
                   rule_types: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_type: torch.Tensor,
                   edge_weight: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Full training step (non-incremental).
        
        Used for initial training of the geometry.
        
        Args:
            rule_tokens: Token sequences [N, seq_len]
            rule_types: Rule type labels [N]
            edge_index: Edge indices [2, E]
            edge_type: Edge type labels [E]
            edge_weight: Optional edge weights [E]
            
        Returns:
            Dict of loss metrics
        """
        self.cost_tracker.start_timer()
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get graph-aware embeddings
        z_graph = self.model(rule_tokens, rule_types, edge_index, edge_type, edge_weight)
        
        # 1. Structure Loss: Unit Steps (target distance)
        mask_dep = (edge_type == 0)  # Prerequisite edges
        if mask_dep.sum() > 0:
            dep_edge_index = edge_index[:, mask_dep]
            l_struct = self.manifold.target_distance_loss(
                z_graph, dep_edge_index, self.config.shell_radius
            )
        else:
            l_struct = torch.tensor(0.0, device=z_graph.device)
        
        # 2. Shell Ordering Loss
        l_shell = self.manifold.shell_ordering_loss(z_graph, edge_index)
        
        # 3. Rigidity Loss (anchor preservation)
        l_rigid = self.anchors.compute_rigidity_loss(z_graph, self.manifold)
        
        # 4. Contrastive Loss (push negatives)
        l_contrastive = self._contrastive_loss(z_graph, edge_index)
        
        # 5. Fossilization Loss (text-only → graph-aware)
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        empty_edge_type = torch.empty((0,), dtype=torch.long, device=edge_type.device)
        z_text = self.model(rule_tokens, rule_types, empty_edge_index, empty_edge_type, None)
        l_fossil = self._fossilization_loss(z_text, z_graph.detach())
        
        # Combined loss
        loss = (
            self.config.structure_weight * l_struct +
            self.config.shell_ordering_weight * l_shell +
            self.config.rigidity_strength * l_rigid +
            self.config.contrastive_weight * l_contrastive +
            self.config.fossilization_weight * l_fossil
        )
        
        loss.backward()
        self.optimizer.step()
        
        # Log training cost
        self.cost_tracker.log_training_step()
        
        return {
            "loss": loss.item(),
            "l_struct": l_struct.item(),
            "l_shell": l_shell.item(),
            "l_rigid": l_rigid.item(),
            "l_contrastive": l_contrastive.item(),
            "l_fossil": l_fossil.item()
        }
    
    def train_step_incremental(self, rule_tokens: torch.Tensor,
                               rule_types: torch.Tensor,
                               edge_index: torch.Tensor,
                               edge_type: torch.Tensor,
                               new_nodes: List[int],
                               edge_weight: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Incremental training step: only update geometry near new nodes.
        
        This is the key to incremental fossilization:
        - Existing geometry is frozen (via rigidity loss + masked gradients)
        - Only k-hop neighborhood of new nodes is updated
        - Verified proofs are preserved
        
        Args:
            rule_tokens: Token sequences [N, seq_len]
            rule_types: Rule type labels [N]
            edge_index: Edge indices [2, E]
            edge_type: Edge type labels [E]
            new_nodes: Indices of newly added nodes
            edge_weight: Optional edge weights [E]
            
        Returns:
            Dict of loss metrics
        """
        self.cost_tracker.start_timer()
        self.model.train()
        
        num_nodes = rule_tokens.size(0)
        device = rule_tokens.device
        
        # Use incremental update context
        with IncrementalUpdateContext(
            optimizer=self.local_optimizer,
            anchor_manager=self.anchors,
            new_nodes=new_nodes,
            k_hop=self.config.max_local_hops,
            total_nodes=num_nodes,
            device=device
        ):
            # Get embeddings
            z_graph = self.model(rule_tokens, rule_types, edge_index, edge_type, edge_weight)
            
            # Compute losses (same as full training)
            mask_dep = (edge_type == 0)
            if mask_dep.sum() > 0:
                dep_edge_index = edge_index[:, mask_dep]
                l_struct = self.manifold.target_distance_loss(
                    z_graph, dep_edge_index, self.config.shell_radius
                )
            else:
                l_struct = torch.tensor(0.0, device=z_graph.device)
            
            l_shell = self.manifold.shell_ordering_loss(z_graph, edge_index)
            l_rigid = self.anchors.compute_rigidity_loss(z_graph, self.manifold)
            l_contrastive = self._contrastive_loss(z_graph, edge_index)
            
            # Fossilization
            empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_edge_type = torch.empty((0,), dtype=torch.long, device=device)
            z_text = self.model(rule_tokens, rule_types, empty_edge_index, empty_edge_type, None)
            l_fossil = self._fossilization_loss(z_text, z_graph.detach())
            
            loss = (
                self.config.structure_weight * l_struct +
                self.config.shell_ordering_weight * l_shell +
                self.config.rigidity_strength * l_rigid +
                self.config.contrastive_weight * l_contrastive +
                self.config.fossilization_weight * l_fossil
            )
            
            # Masked gradient update
            self.local_optimizer.step(loss)
        
        self.cost_tracker.log_training_step()
        
        return {
            "loss": loss.item(),
            "l_struct": l_struct.item(),
            "l_shell": l_shell.item(),
            "l_rigid": l_rigid.item(),
            "l_contrastive": l_contrastive.item(),
            "l_fossil": l_fossil.item(),
            "incremental": True,
            "plastic_region_size": len(new_nodes)
        }
    
    def _contrastive_loss(self, embeddings: torch.Tensor,
                          edge_index: torch.Tensor,
                          margin: float = 1.0,
                          num_negatives: int = 20) -> torch.Tensor:
        """
        Hyperbolic contrastive loss: push negatives away.
        
        d_H(u, v) < d_H(u, negative) - margin
        """
        src, dst = edge_index
        if src.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Positive distances
        pos_dist = self.manifold.dist(embeddings[src], embeddings[dst])
        
        # Negative sampling
        batch_size = src.size(0)
        num_nodes = embeddings.size(0)
        device = embeddings.device
        
        loss = torch.tensor(0.0, device=device)
        for _ in range(num_negatives):
            neg_idx = torch.randint(0, num_nodes, (batch_size,), device=device)
            neg_dist = self.manifold.dist(embeddings[src], embeddings[neg_idx])
            
            # Max(0, pos - neg + margin)
            l = torch.clamp(pos_dist - neg_dist + margin, min=0)
            loss = loss + l.mean()
        
        return loss / num_negatives
    
    def _fossilization_loss(self, z_pred: torch.Tensor,
                            z_target: torch.Tensor) -> torch.Tensor:
        """
        Pull text-only embeddings towards graph-aware embeddings.
        """
        dist = self.manifold.dist(z_pred, z_target)
        return dist.mean()
    
    def train_epoch(self, dataloader, epoch: int = 0) -> Dict[str, float]:
        """
        Train for one epoch over all batches.
        
        Args:
            dataloader: DataLoader yielding (tokens, types, edge_index, edge_type, weight)
            epoch: Current epoch number
            
        Returns:
            Aggregated loss metrics
        """
        epoch_losses = {}
        num_batches = 0
        
        for batch in dataloader:
            rule_tokens, rule_types, edge_index, edge_type = batch[:4]
            edge_weight = batch[4] if len(batch) > 4 else None
            
            metrics = self.train_step(rule_tokens, rule_types, edge_index, edge_type, edge_weight)
            
            for key, value in metrics.items():
                epoch_losses[key] = epoch_losses.get(key, 0) + value
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        # Log epoch
        self.cost_tracker.log_training_epoch(
            epoch=epoch,
            loss=epoch_losses.get("loss", 0),
            num_samples=num_batches
        )
        
        return epoch_losses
    
    def get_embeddings(self, rule_tokens: torch.Tensor,
                       rule_types: torch.Tensor,
                       edge_index: torch.Tensor,
                       edge_type: torch.Tensor,
                       mode: str = "graph") -> torch.Tensor:
        """
        Get embeddings in specified mode.
        
        Args:
            mode: "graph" for graph-aware, "text" for text-only (fossilized)
        """
        self.model.eval()
        with torch.no_grad():
            if mode == "text":
                # OFF-mode: No graph edges
                empty_edge = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
                empty_type = torch.empty((0,), dtype=torch.long, device=edge_type.device)
                return self.model(rule_tokens, rule_types, empty_edge, empty_type, None)
            else:
                return self.model(rule_tokens, rule_types, edge_index, edge_type, None)
