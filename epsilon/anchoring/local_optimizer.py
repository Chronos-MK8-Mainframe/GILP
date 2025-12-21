"""
Local Optimizer

Improvement #1 - Incremental Fossilization (Part 2)

Per-node learning rate control for masked gradient updates.
Only updates embeddings in local neighborhood, freezing distant regions.

This is simple to implement with:
- Masked gradients
- Per-node learning rates
"""

import torch
import torch.nn as nn
from typing import Optional, Set, List


class LocalOptimizer:
    """
    Optimizer that only updates embeddings in local neighborhood.
    
    When adding a new rule incrementally:
    - Update embeddings only within k-hop neighborhood of new nodes
    - Freeze distant regions completely
    - This preserves existing proof geometry
    """
    
    def __init__(self, model: nn.Module, base_lr: float = 0.001):
        """
        Args:
            model: The neural network model to optimize
            base_lr: Base learning rate
        """
        self.model = model
        self.base_lr = base_lr
        self.frozen_mask: Optional[torch.Tensor] = None
        self.per_node_lr: Optional[torch.Tensor] = None
        
        # Use Adam for underlying optimization
        self.optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        
    def set_plastic_region(self, plastic_indices: Set[int], 
                          total_nodes: int,
                          device: torch.device):
        """
        Mark which node embeddings are allowed to update.
        
        Nodes NOT in plastic_indices will have their gradients zeroed.
        
        Args:
            plastic_indices: Set of node indices that CAN be updated
            total_nodes: Total number of nodes in the graph
            device: Torch device
        """
        # Create frozen mask (True = frozen, don't update)
        self.frozen_mask = torch.ones(total_nodes, dtype=torch.bool, device=device)
        for idx in plastic_indices:
            if idx < total_nodes:
                self.frozen_mask[idx] = False
                
    def set_per_node_learning_rates(self, lr_tensor: torch.Tensor):
        """
        Set per-node learning rates for fine-grained control.
        
        Args:
            lr_tensor: Learning rate multiplier per node [N]
        """
        self.per_node_lr = lr_tensor
        
    def clear_masks(self):
        """Clear all masks and return to full updates."""
        self.frozen_mask = None
        self.per_node_lr = None
        
    def step(self, loss: torch.Tensor, embedding_param_name: str = ""):
        """
        Perform gradient step with masked updates.
        
        Args:
            loss: Loss tensor to backpropagate
            embedding_param_name: Name pattern for embedding parameters
                                 (to identify which params to mask)
        """
        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient masking if frozen mask is set
        if self.frozen_mask is not None:
            self._apply_gradient_mask(embedding_param_name)
        
        # Apply per-node learning rates if set
        if self.per_node_lr is not None:
            self._apply_per_node_lr(embedding_param_name)
        
        # Standard optimizer step
        self.optimizer.step()
        
    def _apply_gradient_mask(self, param_name_pattern: str):
        """
        Zero out gradients for frozen nodes.
        
        This is applied to parameters matching the pattern.
        For embedding layers, the gradient shape is typically [N, D].
        """
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            # If no pattern specified, try to apply to all params
            # that have the right shape
            if param_name_pattern and param_name_pattern not in name:
                continue
                
            # Check if gradient has compatible shape with mask
            if param.grad.dim() >= 1 and param.grad.size(0) == self.frozen_mask.size(0):
                # Expand mask to match gradient shape
                mask_shape = [self.frozen_mask.size(0)] + [1] * (param.grad.dim() - 1)
                expanded_mask = self.frozen_mask.view(*mask_shape)
                
                # Zero out frozen gradients
                param.grad[expanded_mask.expand_as(param.grad)] = 0.0
                
    def _apply_per_node_lr(self, param_name_pattern: str):
        """
        Scale gradients by per-node learning rates.
        """
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            if param_name_pattern and param_name_pattern not in name:
                continue
                
            if param.grad.dim() >= 1 and param.grad.size(0) == self.per_node_lr.size(0):
                # Scale gradients by per-node LR
                lr_shape = [self.per_node_lr.size(0)] + [1] * (param.grad.dim() - 1)
                param.grad *= self.per_node_lr.view(*lr_shape)


class IncrementalUpdateContext:
    """
    Context manager for incremental learning updates.
    
    Usage:
        with IncrementalUpdateContext(optimizer, anchor_manager, new_nodes, k_hop):
            loss = compute_loss(...)
            optimizer.step(loss)
    """
    
    def __init__(self, optimizer: LocalOptimizer, 
                 anchor_manager, 
                 new_nodes: List[int],
                 k_hop: int,
                 total_nodes: int,
                 device: torch.device):
        """
        Args:
            optimizer: LocalOptimizer instance
            anchor_manager: AnchorManager for k-hop queries
            new_nodes: Indices of newly added nodes
            k_hop: Hop distance for plastic region
            total_nodes: Total number of nodes
            device: Torch device
        """
        self.optimizer = optimizer
        self.anchor_manager = anchor_manager
        self.new_nodes = new_nodes
        self.k_hop = k_hop
        self.total_nodes = total_nodes
        self.device = device
        
    def __enter__(self):
        # Compute plastic region (k-hop neighborhood of new nodes)
        plastic_nodes = self.anchor_manager.get_k_hop_neighborhood(
            self.new_nodes, self.k_hop
        )
        
        # Always include new nodes themselves
        plastic_nodes.update(self.new_nodes)
        
        # Set the plastic region in optimizer
        self.optimizer.set_plastic_region(
            plastic_nodes, self.total_nodes, self.device
        )
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear masks after incremental update
        self.optimizer.clear_masks()
        return False
