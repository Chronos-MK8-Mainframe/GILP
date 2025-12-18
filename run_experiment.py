
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
    for epoch in range(201):
        metrics = trainer.train_step(
            rule_tokens, 
            rule_types, 
            graph_data.edge_index, 
            graph_data.edge_type
        )
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss={metrics['loss']:.4f} (Geo={metrics['l_hyp']:.4f}, Fossil={metrics['l_fossil']:.4f})")
            
    print("\n--- 4. Search / Inference ---")
    model.eval()
    
    # Test 1: ON Mode (Full Graph - Baseline)
    print("\n[Mode: ON] Using Full Graph for Inference (Heuristics ON)")
    with torch.no_grad():
        z_on = model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        
    searcher_on = AHSPSearch(z_on)
    start_id = 1 # Zero
    goal_id = 10 # RecursiveMult
    
    path_on, d_on, status_on = searcher_on.find_path_hyperbolic_greedy(start_id, goal_id)
    print(f"Path (ON): {path_on} [{status_on}]")
    if path_on:
        print("Path names:", [kb.get_rule(idx).name for idx in path_on])
    
    # Test 2: OFF Mode (Empty Graph - Fossilization Test)
    print("\n[Mode: OFF] Using Empty Graph for Inference (Heuristics OFF)")
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=graph_data.edge_index.device)
    empty_edge_type = torch.empty((0,), dtype=torch.long, device=graph_data.edge_type.device)
    
    with torch.no_grad():
        z_off = model(rule_tokens, rule_types, empty_edge_index, empty_edge_type)
        
    searcher_off = AHSPSearch(z_off)
    path_off, d_off, status_off = searcher_off.find_path_hyperbolic_greedy(start_id, goal_id)
    print(f"Path (OFF): {path_off} [{status_off}]")
    if path_off:
        print("Path names:", [kb.get_rule(idx).name for idx in path_off])
    
    
    if status_off == "SUCCESS":
        print("\n>>> SUCCESS (Claim 2): Fossilization worked! Model navigated correctly without edges.")
    else:
        print(f"\n>>> FAILURE: Model failed to navigate without edges. Reason: {status_off}")
    
    # Test 3: Deep Composition (Claim 5)
    # Trying path from Rule 0 ("Number") to Rule 10 ("RecursiveMult").
    # Path: 0 -> 4 -> 8 -> 10 (or similar)
    print("\n[Test 3: Composition] Navigation Depth Test (Claim 5)")
    start_deep = 0
    goal_deep = 10
    path_deep, d_deep, status_deep = searcher_off.find_path_hyperbolic_greedy(start_deep, goal_deep)
    print(f"Path (0->10): {path_deep} [{status_deep}]")
    if path_deep:
        print("Path names:", [kb.get_rule(idx).name for idx in path_deep])
        
    if status_deep == "SUCCESS" and len(path_deep) > 2:
         print(">>> SUCCESS (Claim 5): Long-range composition preserved.")
    
    # Test 4: Meaningful Failure (Claim 4)
    # Try to navigate from a Contradiction node (11) to a Logic node (1).
    # They should be disconnected or remarkably far.
    print("\n[Test 4: Meaningful Failure] Disconnected Graph Test (Claim 4)")
    start_fail = 11
    goal_fail = 1
    
    # Check initial distance
    d_initial = trainer.manifold.dist(z_off[11:12], z_off[1:2]).item()
    print(f"Distance 11->1: {d_initial:.4f} (Expect Large)")
    
    path_fail, d_fail_total, status_fail = searcher_off.find_path_hyperbolic_greedy(start_fail, goal_fail, max_steps=5)
    print(f"Path (11->1): {path_fail} [{status_fail}]")
    
    if status_fail == "FAIL_LOCAL_MINIMA": 
        print(">>> SUCCESS (Claim 4): Agent rightfully refused to move (No descent possible).")
    elif status_fail == "FAIL_MAX_STEPS":
         print(">>> WARNING: Agent wandered but didn't reach goal (Ambiguous result).")
    elif status_fail == "SUCCESS":
         print(f">>> WARNING: Agent found a path? (Maybe graph is connected via similarity?). Path: {path_fail}")

    # Verify contradiction separation (Claim 3)
    c1 = z_on[11]
    c2 = z_on[12]
    dist = torch.norm(c1 - c2).item()
    print(f"\nDist between contradictions (Claim 3): {dist:.4f} (Should be > margin)")

if __name__ == "__main__":
    run_experiment()
