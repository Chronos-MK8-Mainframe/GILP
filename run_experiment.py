
import torch
import numpy as np
import networkx as nx

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch

def create_arithmetic_dataset():
    """
    Creates a small synthetic dataset of arithmetic concepts.
    Rules:
    0: Number
    1: Zero
    2: One (Successor of Zero)
    3: Two (Successor of One)
    4: Addition
    5: 0 + x = x
    6: x + 0 = x
    7: S(x) + y = S(x + y)
    8: Multiplication
    9: 0 * x = 0
    10: S(x) * y = y + (x * y)
    """
    kb = KnowledgeBase()
    
    # Define concepts/rules
    names = [
        "Number", "Zero", "One", "Two", 
        "Addition", "IdentityAddLeft", "IdentityAddRight", "RecursiveAdd",
        "Multiplication", "ZeroMult", "RecursiveMult",
        "ContradictionTest_A", "ContradictionTest_B"
    ]
    
    for name in names:
        kb.add_rule(name)
        
    # Relationships
    
    # Composition / Hierarchy (Type 2) (using add_composition: "Zero is composed of Number"?? 
    # Maybe "Number is composed of Zero..." usually it's "Zero IS A Number")
    # Let's say Number is higer level. 
    # Creating some hierarchy for structure
    kb.add_dependency(1, 0) # Zero -> Number (Prereq logic: defined in terms of)
    kb.add_dependency(2, 1) # One -> Zero (Successor)
    
    # Prerequisites (Type 0)
    kb.add_dependency(4, 0) # Addition requires Number
    kb.add_dependency(5, 4) # Rule 5 requires Addition and Zero
    kb.add_dependency(5, 1)
    
    kb.add_dependency(8, 4) # Mult requires Add
    
    # Contradictions (Type 1)
    kb.add_contradiction(11, 12) # Test logic
    
    return kb

def run_experiment():
    print("--- 1. Initializing Knowledge Base ---")
    kb = create_arithmetic_dataset()
    graph_data = kb.build_graphs()
    print(f"Nodes: {graph_data.num_nodes}, Edges: {graph_data.edge_index.size(1)}")
    
    print("\n--- 2. Initializing Model ---")
    # Vocab size = number of rules (simplification, using rule ID as token)
    model = StructureAwareGraphEmbedding(vocab_size=len(kb.rules), hidden_dim=64)
    trainer = GILPTrainer(model)
    
    # Inputs
    rule_tokens = torch.arange(len(kb.rules)).unsqueeze(1) # [N, 1]
    rule_types = torch.zeros(len(kb.rules), dtype=torch.long) # All type 0 for now
    
    print("\n--- 3. Training Loop ---")
    for epoch in range(51):
        loss, l_geo, l_sep = trainer.train_step(
            rule_tokens, 
            rule_types, 
            graph_data.edge_index, 
            graph_data.edge_type
        )
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss={loss:.4f} (Geo={l_geo:.4f}, Sep={l_sep:.4f})")
            
    print("\n--- 4. Search / Inference ---")
    model.eval()
    with torch.no_grad():
        embeddings = model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        
    searcher = AHSPSearch(embeddings)
    
    # Try to find path from "Zero" (1) to "RecursiveMult" (10)
    start_id = 1
    goal_id = 10
    
    path = searcher.find_path(start_id, goal_id)
    print(f"Finding path from {kb.get_rule(start_id).name} to {kb.get_rule(goal_id).name}")
    print("Path found (IDs):", path)
    print("Path names:", [kb.get_rule(idx).name for idx in path])
    
    # Verify contradiction separation
    c1 = embeddings[11]
    c2 = embeddings[12]
    dist = torch.norm(c1 - c2).item()
    print(f"\nDist between contradictions: {dist:.4f} (Should be > margin)")

if __name__ == "__main__":
    run_experiment()
