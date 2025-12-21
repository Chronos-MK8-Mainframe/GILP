import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch
from gilp_core.llm.reasoning_explainer import get_explainer

def create_v7_kb():
    kb = KnowledgeBase()
    # Create a intentional gap: Calculus -> ??? -> DifferentialEquations
    names = ["Calculus", "Integration", "Differentiation", "DifferentialEquations"]
    for n in names: kb.add_rule(n)
    
    kb.add_dependency(1, 0) # Integration needs Calculus
    kb.add_dependency(2, 0) # Differentiation needs Calculus
    
    # Gap: No link between Integration/Differentiation and DifferentialEquations
    # But in the embedding space they might be close enough to trigger a bridge.
    # To force a gap in A*, we need them to be reachable but with high edge cost?
    # Actually, if there is NO PATH, A* fails. 
    # v7 refinement: If A* fails, we try a latent search or bridge between nearest known neighbors.
    # Simplified Demo: We have a path but it's "stretched".
    kb.add_dependency(3, 1) # Force a link but we'll show it as a high-dist link
    
    return kb

def main():
    print("=== GILP v7: Generative Bridging Demo ===")
    kb = create_v7_kb()
    graph_data = kb.build_graphs()
    
    model = StructureAwareGraphEmbedding(len(kb.rules), manifold_type='lorentz')
    trainer = GILPTrainer(model)
    
    rule_tokens = torch.arange(len(kb.rules)).unsqueeze(1)
    rule_types = torch.zeros(len(kb.rules), dtype=torch.long)
    
    print("\n--- Training Model ---")
    for epoch in range(101):
        metrics = trainer.train_step(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        
    model.eval()
    with torch.no_grad():
        z = model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        
    # Initialize Searcher with LLM Explainer (v7)
    explainer = get_explainer()
    searcher = AHSPSearch(z, manifold_type='lorentz', explainer=explainer, kb=kb)
    
    start, goal = 0, 3 # Calculus -> DiffEq
    print(f"\nSearching Path with Bridging: {kb.get_rule(start).name} -> {kb.get_rule(goal).name}")
    
    # Low threshold to trigger bridging on any reasonable distance
    path, dist, status = searcher.find_path_with_generative_bridge(start, goal, bridge_threshold=0.3)
    
    print("\nFinal Resulting Path:")
    for node in path:
        if hasattr(node, 'is_virtual'):
            print(f"  [VIRTUAL] {node.name}")
        else:
            print(f"  [KNOWN] {kb.get_rule(node).name}")
            
    # Explain the path
    print("\nGenerating Logical Explanation...")
    names_for_explainer = []
    for node in path:
        if hasattr(node, 'is_virtual'): names_for_explainer.append(node.name)
        else: names_for_explainer.append(kb.get_rule(node).name)
        
    explanation = explainer.explain_proof_path(names_for_explainer)
    print("-" * 40)
    print(explanation)
    print("-" * 40)

if __name__ == "__main__":
    main()
