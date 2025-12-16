
import numpy as np
from sklearn.neighbors import KDTree
import torch



from gilp_core.geometry.hyperbolic import PoincareManifold

class AHSPSearch:
    def __init__(self, embeddings: torch.Tensor): 
        """
        Adaptive Hieararchical Space Partitioning (Proto-3 Hyperbolic)
        embeddings: Tensor [N, D] in Poincaré Ball.
        """
        self.embeddings = embeddings.detach().cpu().numpy()
        # KDTree uses Euclidean distance.
        # This is a valid proxy for locality in Poincaré ball 
        # (neighbors in ball are neighbors in distance).
        self.tree = KDTree(self.embeddings)
        self.manifold = PoincareManifold()
        
        # Convert embeddings back to torch for manifold ops
        self.emb_torch = getattr(embeddings, 'cpu', lambda: embeddings)().detach()
        
    def find_nearest(self, query_point, k=1):
        """Find k nearest logical rules to a query point (vector)"""
        if isinstance(query_point, torch.Tensor):
            query_point = query_point.detach().cpu().numpy()
            
        if query_point.ndim == 1:
            query_point = query_point.reshape(1, -1)
            
        dist, ind = self.tree.query(query_point, k=k)
        return ind, dist
        
    def find_path_budgeted(self, start_idx, goal_idx, initial_budget=0.1, max_budget=2.0, step_size=0.1):
        """
        Attempts to find a path where every step has Euclidean distance < current_budget.
        If potentials are present, enforces Strict Descent (phi(v) < phi(u)).
        Returns: (path_list, budget_used, fail_reason)
        """
        import heapq
        
        current_budget = initial_budget
        while current_budget <= max_budget + 1e-6:
            # A* Search
            start_emb = self.embeddings[start_idx]
            goal_emb = self.embeddings[goal_idx]
            
            def heuristic(idx):
                return np.linalg.norm(self.embeddings[idx] - goal_emb)

            pq = [(heuristic(start_idx), 0, start_idx, [start_idx])]
            visited = {start_idx: 0} # map node -> g_score
            
            path_found = None
            nodes_expanded = 0
            stuck_count = 0
            
            while pq:
                f, g, u, path = heapq.heappop(pq)
                nodes_expanded += 1
                
                if u == goal_idx:
                    path_found = path
    def find_path_hyperbolic_greedy(self, start_idx, goal_idx, step_radius=0.5, max_steps=50):
        """
        Hyperbolic Greedy Best-First Search.
        Minimize d_H(current, goal).
        """
        path = [start_idx]
        current = start_idx
        steps_taken = 0
        
        goal_emb = self.emb_torch[goal_idx].reshape(1, -1)
        
        # Initial distance
        cur_emb_torch = self.emb_torch[current].reshape(1, -1)
        current_dist_to_goal = self.manifold.dist(cur_emb_torch, goal_emb).item()
        
        total_dist_traveled = 0.0
        
        for _ in range(max_steps):
            if current == goal_idx:
                return path, total_dist_traveled, "SUCCESS"
            
            # Find neighbors (Euclidean proxy)
            # Use larger radius because hyperbolic space expands faster? 
            # Or trust proxy.
            cur_emb = self.embeddings[current].reshape(1, -1)
            indices = self.tree.query_radius(cur_emb, r=step_radius)[0]
            
            best_v = None
            best_dist = float('inf')
            
            for v in indices:
                v = v.item()
                if v == current: continue
                
                # Check Hyperbolic Distance to Goal
                v_emb = self.emb_torch[v].reshape(1, -1)
                d_H_to_goal = self.manifold.dist(v_emb, goal_emb).item()
                
                if d_H_to_goal < best_dist:
                    best_dist = d_H_to_goal
                    best_v = v
            
            # Strict Descent on Distance to Goal
            if best_v is not None and best_dist < current_dist_to_goal:
                 # Move
                 d_step = self.manifold.dist(cur_emb_torch, self.emb_torch[best_v].reshape(1, -1)).item()
                 total_dist_traveled += d_step
                 
                 current = best_v
                 path.append(current)
                 current_dist_to_goal = best_dist
                 
                 # Prepare next iter
                 cur_emb_torch = self.emb_torch[current].reshape(1, -1)
            else:
                return None, total_dist_traveled, "FAIL_LOCAL_MINIMA"
                
        return None, total_dist_traveled, "FAIL_MAX_STEPS"

    def find_path(self, start_idx, goal_idx):
        """
        Find a path from start rule to goal rule in the embedding space.
        This is a heuristic pathfinding. In GILP, we follow the gradient of valid inference?
        Or we stick to the graph? 
        The PDF says "pathfinding in high-dimensional spaces".
        A simple greedy approach: 
        1. Look at neighbors in the embedding space that reduce distance to goal.
        2. Check if there exists a valid edge in the Knowledge Base (or predict one).
        
        For this simplified version, we'll return the Euclidean path in embedding space
        identifying the sequence of rules that are "geometrically intermediate".
        """
        start_emb = self.embeddings[start_idx]
        goal_emb = self.embeddings[goal_idx]
        
        # Linearly interpolate between start and goal
        # Then find nearest rules to these interpolation points
        steps = 10
        path_indices = []
        
        for i in range(steps + 1):
            alpha = i / steps
            point = (1 - alpha) * start_emb + alpha * goal_emb
            
            # Find nearest rule
            ind, _ = self.find_nearest(point, k=1)
            nearest_idx = ind[0][0]
            
            if not path_indices or path_indices[-1] != nearest_idx:
                path_indices.append(nearest_idx)
                
        return path_indices
