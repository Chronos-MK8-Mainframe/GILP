import torch
import sys
import os

# Ensure we can import gilp_core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch
from gilp_core.llm.reasoning_explainer import get_explainer

def create_demo_kb():
    kb = KnowledgeBase()
    # Path: Set Theory -> Group Theory -> Ring Theory -> Field Theory -> Galois Theory
    names = [
        "SetTheory", "GroupTheory", "RingTheory", "FieldTheory", "GaloisTheory",
        "LinearAlgebra", "VectorSpaces", "InnerProductSpaces",
        "Contradiction_A", "Contradiction_B"
    ]
    for name in names: kb.add_rule(name)
    
    # Algebra Branch
    kb.add_dependency(1, 0) # Group needs Set
    kb.add_dependency(2, 1) # Ring needs Group
    kb.add_dependency(3, 2) # Field needs Ring
    kb.add_dependency(4, 3) # Galois needs Field
    
    # Linear Branch
    kb.add_dependency(6, 0) # VectorSpaces needs Set
    kb.add_dependency(7, 6) # InnerProduct needs VectorSpaces
    
    kb.add_contradiction(8, 9)
    
    return kb

def main():
    print("=== GILP Proto-5: Lorentz Reasoning Demo ===")
    
    # 1. Setup
    kb = create_demo_kb()
    graph_data = kb.build_graphs()
    
    # Using Lorentz Manifold by default in Proto-5
    model = StructureAwareGraphEmbedding(vocab_size=len(kb.rules), hidden_dim=64, manifold_type='lorentz')
    trainer = GILPTrainer(model)
    
    rule_tokens = torch.arange(len(kb.rules)).unsqueeze(1)
    rule_types = torch.zeros(len(kb.rules), dtype=torch.long)
    
    # 2. Train
    print("\n--- Training on Manifold (Lorentz Geometry) ---")
    for epoch in range(151):
        metrics = trainer.train_step(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}")
            
    # 3. Search
    model.eval()
    with torch.no_grad():
        z = model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        
    searcher = AHSPSearch(z, manifold_type='lorentz')
    
    # Path: SetTheory (0) -> GaloisTheory (4)
    start, goal = 0, 4
    print(f"\nSearching Path: {kb.get_rule(start).name} -> {kb.get_rule(goal).name}")
    
    path, dist, status = searcher.find_path_hyperbolic_astar(start, goal, step_radius=1.0, max_expanded=500)
    
    if status == "SUCCESS":
        rule_names = [kb.get_rule(i).name for i in path]
        print(f"Path Found: {' -> '.join(rule_names)}")
        print(f"Total Hyperbolic Distance: {dist:.4f}")
        
        # 4. LLM Explanation
        try:
            print("\nGenerating LLM Explanation...")
            explainer = get_explainer()
            explanation = explainer.explain_proof_path(rule_names)
            print("-" * 40)
            print(explanation)
            print("-" * 40)
        except Exception as e:
            print(f"LLM Explanation failed: {e}")
    else:
        print(f"Search Failed: {status}")

if __name__ == "__main__":
    main()
