
import torch
import torch.nn as nn
import torch.optim as optim


import torch
import torch.nn as nn
import torch.optim as optim


from gilp_core.geometry.hyperbolic import PoincareManifold
from gilp_core.geometry.lorentz import LorentzManifold

class GILPTrainer:
    def __init__(self, model, learning_rate=0.001, heuristic_scalars=None):
        """
        heuristic_scalars: Dict[int, float] mapping edge_type to loss scale.
        e.g., {0: 1.0, 1: 0.5, 2: 0.1}
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.heuristic_scalars = heuristic_scalars if heuristic_scalars else {}
        
        # Use the manifold provided by the model (Lorentz or Poincare)
        self.manifold = getattr(model, 'manifold', None)
        if self.manifold is None:
             # Fallback just in case
             self.manifold = PoincareManifold()
    def geometric_loss(self, embeddings, edge_index, edge_type, edge_weight=None):
        """
        L_geo = sum(scalar_t * w_ij * (||ei - ej|| - 1)^2)
        """
        loss = torch.tensor(0.0, device=embeddings.device)
        per_type_loss = {}
        
        # Unique types in this batch
        unique_types = torch.unique(edge_type)
        
        for t in unique_types:
            t_val = t.item()
            mask = (edge_type == t)
            
            src = edge_index[0, mask]
            dst = edge_index[1, mask]
            
            dist = torch.norm(embeddings[src] - embeddings[dst], dim=1)
            
            # Target distance is 1.0 for all connected edges in this metric space
            # (Relaxed for Contradiction which is handled by separation_loss)
            if t_val == 1: # Contradiction - handled by separation, skip here or keep?
                # Usually we want contradictions FAR, so geometric loss (pulling close) is WRONG for type 1.
                # standard KB: 0=Prereq, 1=Contradiction, 2=Composition
                continue 
            
            squared_errors = (dist - 1.0) ** 2
            
            # Apply edge weights (instance specific)
            if edge_weight is not None:
                weights = edge_weight[mask]
                squared_errors = squared_errors * weights
                
            # Apply Heuristic Scalar (Type specific)
            scalar = self.heuristic_scalars.get(t_val, 1.0)
            
            type_loss = (squared_errors.mean()) * scalar
            loss += type_loss
            per_type_loss[f"geo_type_{t_val}"] = type_loss.item()
            
        return loss, per_type_loss

    def separation_loss(self, embeddings, edge_index, edge_type, margin=2.0):
        """
        L_sep for Contradictions (Type 1)
        """
        loss = torch.tensor(0.0, device=embeddings.device)
        mask = (edge_type == 1) 
        if mask.sum() > 0:
            src = edge_index[0, mask]
            dst = edge_index[1, mask]
            
            dist = torch.norm(embeddings[src] - embeddings[dst], dim=1)
            loss += torch.clamp(margin - dist, min=0).mean()
        return loss

    def repulsion_loss(self, embeddings, rule_tokens, margin=3.0, num_negatives=200):
        """
        L_repulsion with Hard/Easy negative filtering.
        Hard Negative: High token overlap but not connected (assumed not connected for random sample).
        """
        num_nodes = embeddings.size(0)
        
        # Sample random pairs
        idx1 = torch.randint(0, num_nodes, (num_negatives,), device=embeddings.device)
        idx2 = torch.randint(0, num_nodes, (num_negatives,), device=embeddings.device)
        
        # Avoid self-loops
        mask = (idx1 != idx2)
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        
        # Calculate distances
        dist = torch.norm(embeddings[idx1] - embeddings[idx2], dim=1)
        
        # Basic Repulsion (Easy Negatives)
        raw_loss = torch.clamp(margin - dist, min=0)
        
        # Hard Negative Weighting
        # We estimate "Hardness" by token overlap (Jaccard)
        # rule_tokens: [N, SeqLen]. We need a fast way to check overlap.
        # Approximation: Check intersection of token sets.
        # Since tensors are padded, we can't just set(row). 
        # For efficiency in prototype, we'll do a simplified check:
        # If they share at least one non-padding token.
        
        weights = torch.ones_like(raw_loss)
        
        # Retrieve tokens for sampled pairs
        tok1 = rule_tokens[idx1] # [K, L]
        tok2 = rule_tokens[idx2] # [K, L]
        
        # This loop is slow in Python, but for K=200 it's fine.
        # Vectorized overlap is better but complex with padding 0.
        # Let's try to trust the 'Easy vs Hard' random split logic:
        # Just weight ALL repulsions. 
        # But User request: "Weight hard negatives higher".
        # Let's do a fast distinct check.
        
        for k in range(len(idx1)):
            # Convert to sets ignoring 0 (padding)
            s1 = set(tok1[k].tolist()) - {0}
            s2 = set(tok2[k].tolist()) - {0}
            if not s1 or not s2: continue
            
            intersection = len(s1.intersection(s2))
            if intersection > 0:
                # Hard Negative: Shares Vocabulary!
                weights[k] = 5.0 # Weight hard negatives 5x
            else:
                # Easy Negative
                weights[k] = 1.0
                
        weighted_loss = (raw_loss * weights).mean()
        return weighted_loss

    def sample_2hop_paths(self, edge_index, num_paths=100):
        """
        Find ALL 2-hop paths u -> v -> w for small graphs.
        Returns: tensor of shape [num_found, 3]
        """
        device = edge_index.device
        src, dst = edge_index
        
        # Build adjacency
        from collections import defaultdict
        adj = defaultdict(list)
        for s, d in zip(src.tolist(), dst.tolist()):
            adj[s].append(d)
        
        paths = []
        for u, neighbors_v in adj.items():
            for v in neighbors_v:
                if v in adj:
                    for w in adj[v]:
                        if u != w:
                            paths.append([u, v, w])
        
        if not paths:
             # print("No 2-hop paths found!") 
             return torch.empty((0, 3), dtype=torch.long, device=device)
        
        # If too many, sample? For MED dataset, probably small enough.
        # If very large, sample.
        if len(paths) > num_paths:
            import random
            paths = random.sample(paths, num_paths)
            
        return torch.tensor(paths, dtype=torch.long, device=device)

    def path_consistency_loss(self, embeddings, edge_index, num_paths=100):
        """
        L_path = (d(u,w) - (d(u,v) + d(v,w)))^2
        Enforces that v lies on the geodesic between u and w.
        """
        triplets = self.sample_2hop_paths(edge_index, num_paths=num_paths)
        if triplets.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        u, v, w = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        
        emb_u = embeddings[u]
        emb_v = embeddings[v]
        emb_w = embeddings[w]
        
        d_uv = torch.norm(emb_u - emb_v, dim=1)
        d_vw = torch.norm(emb_v - emb_w, dim=1)
        d_uw = torch.norm(emb_u - emb_w, dim=1)
        
        # We want d_uw = d_uv + d_vw
        # Minimize difference
        diff = d_uw - (d_uv + d_vw)
        loss = (diff ** 2).mean()
        
        return loss

    def bellman_loss(self, potentials, edge_index):
        """
        Train phi(u) approx min(phi(v)) + 1 for edges u->v.
        This learns 'Steps-to-Goal'.
        """
        src, dst = edge_index
        if src.numel() == 0:
            return torch.tensor(0.0, device=potentials.device)
            
        # Group by source to find min neighbor
        # Naive loop for prototype (Graph is small)
        # Optimization: use scatter_reduce if scaling needed.
        
        unique_src = torch.unique(src)
        loss = 0.0
        
        for u in unique_src:
            # Find neighbors
            mask = (src == u)
            v_indices = dst[mask]
            
            # Value Iteration Logic
            # Target = min(phi(neighbors)) + step_cost(1.0)
            # Detach target to stabilize training (like DQN)
            min_phi_v = potentials[v_indices].min().detach()
            
            target = min_phi_v + 1.0
            
            # MSE Loss
            loss += (potentials[u] - target) ** 2
            
        return loss / unique_src.size(0)

    def anchor_loss(self, potentials, rule_types):
        """
        Anchor conjectures (Type 1) to 0.
        """
        mask = (rule_types == 1)
        if mask.sum() == 0:
             return torch.tensor(0.0, device=potentials.device)
        return (potentials[mask] ** 2).mean()

    def alignment_loss(self, embeddings, potentials, edge_index):
        """
        Couples geometry and potential: ||z_u - z_v|| ~ phi(u) - phi(v)
        """
        src, dst = edge_index
        
        dist = torch.norm(embeddings[src] - embeddings[dst], dim=1, keepdim=True)
        pot_diff = potentials[src] - potentials[dst]
        
        # Enforce dist ~ pot_diff
        return torch.abs(dist - pot_diff).mean()

    def variance_loss(self, potentials, rule_types, epsilon=1.0):
        """
        Ensure variance of non-conjecture potentials is at least epsilon.
        """
        mask = (rule_types != 1)
        if mask.sum() < 2:
            return torch.tensor(0.0, device=potentials.device)
        
        var = torch.var(potentials[mask])
        return torch.clamp(epsilon - var, min=0)

    def hyperbolic_contrastive_loss(self, embeddings, edge_index, margin=1.0, num_negatives=20):
        """
        Proto-3: Enforce Hyperbolic Locality.
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
        
        loss = 0.0
        for _ in range(num_negatives):
            neg_idx = torch.randint(0, num_nodes, (batch_size,), device=embeddings.device)
            neg_dist = self.manifold.dist(embeddings[src], embeddings[neg_idx])
            
            # Max(0, pos - neg + margin)
            l = torch.clamp(pos_dist - neg_dist + margin, min=0)
            loss += l
            
        return loss.mean() / num_negatives

    def fossilization_loss(self, z_pred, z_target):
        """
        L_fossil: Pull z_pred (Text-Only) towards z_target (Graph-Aware/True)
        Minimizes Hyperbolic Distance between them.
        """
        dist = self.manifold.dist(z_pred, z_target)
        return dist.mean()

    def root_anchoring_loss(self, embeddings, edge_index):
        """
        v6: Anchor nodes with ZERO in-degree (root concepts) to the origin.
        This fixes the "top" of the logical hierarchy.
        """
        num_nodes = embeddings.size(0)
        # Find nodes that ARE NOT targets of any edge
        has_parent = torch.zeros(num_nodes, dtype=torch.bool, device=embeddings.device)
        has_parent[edge_index[1]] = True
        
        roots = torch.where(~has_parent)[0]
        if roots.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        root_embs = embeddings[roots]
        
        origin = torch.zeros_like(root_embs)
        if isinstance(self.manifold, LorentzManifold):
             origin[..., 0] = 1.0 # Lorentz origin [1, 0, ...]
        
        dist_to_origin = self.manifold.dist(root_embs, origin)
        return dist_to_origin.mean()

    def target_distance_loss(self, embeddings, edge_index, target=0.5):
        """
        Enforces a specific distance 'target' for connected edges.
        Prevents collapse (d->0) and enables stepwise navigation.
        """
        src, dst = edge_index
        if src.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        dists = self.manifold.dist(embeddings[src], embeddings[dst])
        # MSE against target
        return ((dists - target) ** 2).mean()

    def neural_consistency_loss(self, embeddings, kb, num_samples=2):
        """
        v9: Neural Consistency.
        Randomly sample node pairs that are GEOMETRICALLY CLOSE but NOT CONNECTED.
        Ask LLM (via explainer) if they make sense together.
        If NO, increase repulsion.
        """
        if not hasattr(self, 'explainer') or self.explainer is None:
             return torch.tensor(0.0, device=embeddings.device)
             
        num_nodes = embeddings.size(0)
        if num_nodes < 2: return torch.tensor(0.0, device=embeddings.device)
        
        loss = torch.tensor(0.0, device=embeddings.device)
        
        # Sample random pairs
        indices = torch.randperm(num_nodes)[:num_samples]
        for i in indices:
            # Find nearest neighbor in embedding space (approx)
            # For simplicity in trainer, we just pick another random node
            # and check if the distance is too small for unconnected nodes.
            j = torch.randint(0, num_nodes, (1,)).item()
            if i == j: continue
            
            u_name = kb.get_rule(i.item()).name
            v_name = kb.get_rule(j).name
            
            # If not connected in graph
            # This check is simplified for the demo
            
            d = self.manifold.dist(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            
            if d.item() < 0.5: # Hardcoded "too close" threshold
                 # Ask LLM if they are related
                 is_related = self.explainer.verify_consistency(u_name, v_name)
                 if not is_related:
                      # Penalize small distance
                      penalty = torch.clamp(1.0 - d, min=0.0).reshape([]) # Ensure scalar
                      loss = loss + penalty
                      
        return loss / (num_samples + 1e-6)
    def train_step(self, rule_tokens, rule_types, edge_index, edge_type, edge_weight=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. True Fossilization (Graph-Aware)
        z_graph = self.model(rule_tokens, rule_types, edge_index, edge_type, edge_weight)
        
        # A. Structure Loss: Unit Steps (Springs)
        # For Type 0 (Prereq) edges, we want d(u,v) approx 0.5 (or user defined step)
        # This prevents "Shortcuts" in geometry, enforcing A->B->C structure.
        # A. Structure Loss: Unit Steps (Springs)
        mask_dep = (edge_type == 0)
        dep_edge_index = edge_index[:, mask_dep] if mask_dep.sum() > 0 else torch.empty((2, 0), dtype=torch.long, device=z_graph.device)
        
        if mask_dep.sum() > 0:
            # Target distance 0.5 ensures that A->C (dist ~1.0) is not a neighbor in 0.5-radius search
            l_struct = self.target_distance_loss(z_graph, dep_edge_index, target=0.5) 
        else:
            l_struct = torch.tensor(0.0, device=z_graph.device)
            
        # B. Repulsion Loss (Contrastive)
        l_repul = self.hyperbolic_contrastive_loss(z_graph, dep_edge_index, margin=1.0, num_negatives=20)
        # Note: The above calculates max(0, d_pos - d_neg + margin). 
        # Since we punish d_pos!=0.5 in l_struct, this might conflict if d_pos tries to go to 0.
        # But if we treat this as just "Separation", it's fine.
        
        # 2. Fossilization (Text-Only -> Graph)
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        empty_edge_type = torch.empty((0,), dtype=torch.long, device=edge_type.device)
        
        z_text = self.model(rule_tokens, rule_types, empty_edge_index, empty_edge_type, None)
        l_fossil = self.fossilization_loss(z_text, z_graph.detach())
        
        # 3. Root Anchoring (v6)
        l_root = self.root_anchoring_loss(z_graph, edge_index)
        
        # Total
        loss = l_struct + l_fossil + 0.1 * l_repul + 0.5 * l_root 
        
        # 4. Neural Consistency (v9) - Every 10 steps to save cost?
        l_neural = torch.tensor(0.0, device=z_graph.device)
        if hasattr(self, 'explainer') and self.explainer:
             l_neural = self.neural_consistency_loss(z_graph, self.model.kb) # Assuming kb is attached to model
             loss += 1.0 * l_neural
             
        loss.backward()
        self.optimizer.step()
        
        metrics = {
            "loss": loss.item(),
            "l_struct": l_struct.item(),
            "l_fossil": l_fossil.item(),
            "l_repul": l_repul.item(),
            "l_root": l_root.item(),
            "l_neural": l_neural.item() if isinstance(l_neural, torch.Tensor) else l_neural,
            "c": self.model.get_curvature().item() if hasattr(self.model, 'get_curvature') else 1.0
        }
        
        return metrics
