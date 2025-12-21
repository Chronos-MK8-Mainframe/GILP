import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch

def create_v6_kb():
    kb = KnowledgeBase()
    # High density branch vs Low density branch
    # Root: Logic
    # DENSE: P -> Q, Q -> R, R -> S, S -> T
    # SPARSE: A -> B
    names = ["Logic", "P", "Q", "R", "S", "T", "A", "B"]
    for n in names: kb.add_rule(n)
    
    kb.add_dependency(1, 0) # P needs Logic
    kb.add_dependency(2, 1) # Q needs P
    kb.add_dependency(3, 2) # R needs Q
    kb.add_dependency(4, 3) # S needs R
    kb.add_dependency(5, 4) # T needs S
    
    kb.add_dependency(6, 0) # A needs Logic
    kb.add_dependency(7, 6) # B needs A
    
    return kb

def main():
    print("=== GILP v6: Adaptive Manifold Demo ===")
    kb = create_v6_kb()
    graph_data = kb.build_graphs()
    
    # 1. Fixed Curvature (c=1.0)
    print("\n--- Phase 1: Fixed Curvature ---")
    model_fixed = StructureAwareGraphEmbedding(len(kb.rules), manifold_type='lorentz', learnable_curvature=False)
    trainer_fixed = GILPTrainer(model_fixed)
    
    rule_tokens = torch.arange(len(kb.rules)).unsqueeze(1)
    rule_types = torch.zeros(len(kb.rules), dtype=torch.long)
    
    for epoch in range(101):
        metrics = trainer_fixed.train_step(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}, RootDist={metrics['l_root']:.4f}")

    # 2. Learned Curvature
    print("\n--- Phase 2: Learned Curvature (Adaptive) ---")
    model_adv = StructureAwareGraphEmbedding(len(kb.rules), manifold_type='lorentz', learnable_curvature=True)
    trainer_adv = GILPTrainer(model_adv, learning_rate=0.01) # Faster learning for curvature demo
    
    for epoch in range(101):
        metrics = trainer_adv.train_step(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}, Curvature(c)={metrics['c']:.4f}, RootDist={metrics['l_root']:.4f}")
            
    # 3. Search Comparison
    model_adv.eval()
    with torch.no_grad():
        z = model_adv(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
    
    searcher = AHSPSearch(z, manifold_type='lorentz')
    
    # Check if Logic (0) is indeed near origin
    # Lorentz origin is [1, 0, 0, 0]
    origin = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    root_emb = z[0].unsqueeze(0)
    dist_root = model_adv.manifold.dist(root_emb, origin).item()
    print(f"\nRoot Node '{kb.get_rule(0).name}' distance to origin: {dist_root:.4f}")
    
    # Search deep path
    start, goal = 0, 5 # Logic -> T
    print(f"Searching Deep Path: {kb.get_rule(start).name} -> {kb.get_rule(goal).name}")
    path, dist, status = searcher.find_path_hyperbolic_astar(start, goal, step_radius=1.5)
    print(f"Path: {' -> '.join([kb.get_rule(i).name for i in path])} [{status}]")

if __name__ == "__main__":
    main()
