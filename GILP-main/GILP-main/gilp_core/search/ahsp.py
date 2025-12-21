
import numpy as np
from sklearn.neighbors import KDTree
import torch



from gilp_core.geometry.hyperbolic import PoincareManifold
from gilp_core.geometry.lorentz import LorentzManifold

class AHSPSearch:
    def __init__(self, embeddings: torch.Tensor, manifold_type='lorentz', explainer=None, kb=None): 
        """
        Adaptive Hierarchical Space Partitioning (Proto-7 Generative)
        embeddings: Tensor [N, D] in the chosen manifold.
        explainer: ReasoningExplainer instance (v7 bridging).
        kb: KnowledgeBase instance (v7 bridging).
        """
        self.embeddings = embeddings.detach().cpu().numpy()
        # KDTree uses Euclidean distance as a proxy for locality.
        self.tree = KDTree(self.embeddings)
        self.manifold_type = manifold_type
        if manifold_type == 'poincare':
            self.manifold = PoincareManifold()
        else:
            self.manifold = LorentzManifold()
        
        # Convert embeddings back to torch for manifold ops
        self.emb_torch = getattr(embeddings, 'cpu', lambda: embeddings)().detach()
        self.explainer = explainer
        self.kb = kb
        
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
    def find_path_hyperbolic_astar(self, start_idx: int, goal_idx: int, step_radius: float = 1.0, max_expanded: int = 500, max_ts: float = float('inf')):
        """
        Hyperbolic A* Search.
        g(n): Accumulated hyperbolic distance from start.
        h(n): Hyperbolic distance to goal (admissible heuristic).
        """
        import heapq
        
        goal_emb = self.emb_torch[goal_idx].reshape(1, -1)
        start_emb = self.emb_torch[start_idx].reshape(1, -1)
        
        # h_start = d_H(start, goal)
        h_start = self.manifold.dist(start_emb, goal_emb).item()
        
        # (f_score, g_score, current_idx, path)
        pq = [(h_start, 0.0, start_idx, [start_idx])]
        visited = {start_idx: 0.0} # node -> g_score
        
        expanded_count = 0
        
        while pq and expanded_count < max_expanded:
            f, g, u, path = heapq.heappop(pq)
            expanded_count += 1
            
            if u == goal_idx:
                return path, g, "SUCCESS"
            
            # Find neighbors via KDTree Euclidean proxy
            u_emb_np = self.emb_torch[u].detach().cpu().numpy().reshape(1, -1)
            # Use query_radius to find neighbors within Euclidean step_radius
            indices = self.tree.query_radius(u_emb_np, r=step_radius)[0]
            
            u_emb_torch = self.emb_torch[u].reshape(1, -1)
            
            for v in indices:
                v = v.item()
                if v == u: continue
                
                # v9: Check timestamp
                if self.kb:
                     rule_v = self.kb.get_rule(v)
                     if rule_v and rule_v.timestamp > max_ts:
                          continue

                v_emb_torch = self.emb_torch[v].reshape(1, -1)
                
                # Edge weight in hyperbolic space
                d_uv = self.manifold.dist(u_emb_torch, v_emb_torch).item()
                new_g = g + d_uv
                
                if v not in visited or new_g < visited[v]:
                    visited[v] = new_g
                    h_v = self.manifold.dist(v_emb_torch, goal_emb).item()
                    f_v = new_g + h_v
                    heapq.heappush(pq, (f_v, new_g, v, path + [v]))
                    
        if expanded_count >= max_expanded:
            return None, 0.0, "FAIL_MAX_EXPANDED"
            
        return None, 0.0, "FAIL_NO_PATH"

    def find_path_hyperbolic_greedy(self, start_idx, goal_idx, step_radius=0.5, max_steps=50):
        # Fallback to A* for robustness
        return self.find_path_hyperbolic_astar(start_idx, goal_idx, step_radius, max_steps * 4)

    def find_path_with_generative_bridge(self, start_idx, goal_idx, bridge_threshold=1.5):
        """
        v7: Find path using A*, but if the distance between two hops is too large,
        query the LLM to 'discover' an intermediate lemma.
        """
        path, g, status = self.find_path_hyperbolic_astar(start_idx, goal_idx)
        if status != "SUCCESS" or self.explainer is None:
            return path, g, status
            
        final_path = []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            u_emb = self.emb_torch[u].reshape(1, -1)
            v_emb = self.emb_torch[v].reshape(1, -1)
            
            d = self.manifold.dist(u_emb, v_emb).item()
            final_path.append(u)
            
            if d > bridge_threshold:
                u_name = self.kb.get_rule(u).name
                v_name = self.kb.get_rule(v).name
                print(f"  [Bridge] Gap detected between {u_name} and {v_name} (d={d:.2f}). Consulting LLM...")
                
                # Query LLM for bridge
                bridge_lemma = self.explainer.propose_bridge_lemma(u_name, v_name)
                print(f"  [Bridge] LLM proposed: {bridge_lemma}")
                
                # In a real system, we'd Re-Embed here. For prototype, we just add to path.
                # create a 'Virtual Node'
                from collections import namedtuple
                VirtualNode = namedtuple('VirtualNode', ['name', 'is_virtual'])
                final_path.append(VirtualNode(bridge_lemma, True))
                
        final_path.append(path[-1])
        return final_path, g, "SUCCESS_WITH_BRIDGES"

class TemporalManifold:
    def __init__(self, base_searcher, current_time):
        """
        v9: Temporal Support.
        Filters nodes based on their 'ts' (timestamp) attribute.
        """
        self.base = base_searcher
        self.current_time = current_time
        
    def find_path(self, start_idx, goal_idx):
        return self.base.find_path_hyperbolic_astar(start_idx, goal_idx, max_ts=self.current_time)

class ManifoldEnsemble:
    def __init__(self, searcher_poincare, searcher_lorentz):
        """
        v9: Ensemble reasoning.
        Runs search in both manifolds and returns the most confident or consensus path.
        """
        self.poincare = searcher_poincare
        self.lorentz = searcher_lorentz
        
    def find_consensus_path(self, start_idx, goal_idx):
        path_p, dist_p, status_p = self.poincare.find_path_hyperbolic_astar(start_idx, goal_idx)
        path_l, dist_l, status_l = self.lorentz.find_path_hyperbolic_astar(start_idx, goal_idx)
        
        # Return Lorentz if both succeed (more stable), otherwise whatever works
        if "SUCCESS" in status_l: return path_l, dist_l, status_l
        return path_p, dist_p, status_p

class DensityAnalyzer:
    def __init__(self, embeddings, manifold):
        """
        v9: Detects logical bottlenecks by identifying regions with high embedding density.
        """
        self.z = embeddings
        self.manifold = manifold
        
    def analyze(self):
        num_nodes = self.z.size(0)
        bottlenecks = []
        for i in range(num_nodes):
            # Count neighbors within 0.2 radius
            dists = self.manifold.dist(self.z[i].unsqueeze(0), self.z)
            count = (dists < 0.2).sum().item()
            if count > num_nodes * 0.5: # 50% nodes in one tiny spot is a bottleneck
                bottlenecks.append(i)
        return bottlenecks

class HypothesisTestingLoop:
    def __init__(self, searcher, kb, explainer):
        """
        v9: Automated hypothesis testing.
        Generates 'What if' scenarios and tests them against the geometric manifold.
        """
        self.searcher = searcher
        self.kb = kb
        self.explainer = explainer
        
    def run_tests(self, num_tests=3):
        print(f"\n--- Hypothesis Testing Loop (v9) ---")
        num_nodes = len(self.kb.rules)
        for _ in range(num_tests):
            i, j = torch.randint(0, num_nodes, (2,))
            u_name = self.kb.get_rule(i.item()).name
            v_name = self.kb.get_rule(j.item()).name
            
            print(f"Testing Hypothesis: {u_name} -> {v_name}")
            path, dist, status = self.searcher.find_path_hyperbolic_astar(i.item(), j.item())
            
            if "SUCCESS" in status:
                print(f"  [RESULT] GEOMETRICALLY POSSIBLE (dist={dist:.2f})")
                print(f"  [PATH] {' -> '.join([self.kb.get_rule(node).name if not hasattr(node, 'is_virtual') else node.name for node in path])}")
            else:
                print(f"  [RESULT] GEOMETRICALLY IMPOSSIBLE. Asking LLM for reason...")
                reason = self.explainer.llm(f"Why might {u_name} not lead to {v_name}?", max_tokens=64)
                print(f"  [LLM REASON] {reason['choices'][0]['text'].strip()}")

class RecursiveSubProofSearch:
    def __init__(self, searcher, kb):
        """
        v10: Hierarchical A*.
        If a direct jump is too large, it triggers a sub-search to find intermediate steps.
        """
        self.searcher = searcher
        self.kb = kb
        
    def find_deep_path(self, start_idx, target_idx, depth=0, max_depth=2):
        if depth > max_depth: return None
        
        path, dist, status = self.searcher.find_path_hyperbolic_astar(start_idx, target_idx)
        if "SUCCESS" in status:
            return path
            
        # If failed, try to find a 'Generative Bridge' AND THEN sub-search for it
        # This combines v7 and v10 behavior.
        return path # Placeholder for complex recursive logic

class FormalCodeExport:
    def __init__(self, kb):
        self.kb = kb
        
    def to_lean(self, path):
        """
        v10: Translates a geometric path to a Lean 4 proof skeleton.
        """
        code = ["-- GILP v10 Generated Proof Skeleton"]
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            u_name = self.kb.get_rule(u).name if not hasattr(u, 'is_virtual') else u.name
            v_name = self.kb.get_rule(v).name if not hasattr(v, 'is_virtual') else v.name
            code.append(f"theorem {u_name}_to_{v_name} : {u_name} → {v_name} := by")
            code.append(f"  sorry")
        return "\n".join(code)

class VolumeAnalyzer:
    def __init__(self, manifold):
        self.manifold = manifold
        
    def get_uncertainty(self, cluster_embeddings):
        """
        v10: Maps hyperbolic volume to reasoning uncertainty.
        Larger spread in Lorentz space = Higher uncertainty.
        """
        if len(cluster_embeddings) < 2: return 0.0
        # Simple proxy: variance of embeddings
        var = torch.var(cluster_embeddings, dim=0).sum().item()
        return var
