import torch
from common import get_trained_model
from gilp_core.search.ahsp import AHSPSearch

def test_claim_1():
    print("\n=== Testing Claim 1: Directional Progress (Monotonic Descent) ===")
    print("Claim: There exists a learned metric where each step monotonically reduces task-relevant distance.")
    
    model, kb, graph_data, trainer, tokens, types = get_trained_model()
    
    # ON Mode
    with torch.no_grad():
        z_on = model(tokens, types, graph_data.edge_index, graph_data.edge_type)
    
    searcher = AHSPSearch(z_on)
    
    # Path: Zero (1) -> RecursiveMult (10)
    start, goal = 1, 10
    print(f"Seeking path: {kb.get_rule(start).name} -> {kb.get_rule(goal).name}")
    
    path, dists, status = searcher.find_path_hyperbolic_greedy(start, goal)
    
    if status != "SUCCESS":
        print(f"FAILED: Search did not succeed. Status: {status}")
        return False
        
    print(f"Path Found: {path}")
    
    # Verify Monotonicity
    # We need to check if d(step_i, goal) < d(step_{i-1}, goal)
    goal_emb = z_on[goal].unsqueeze(0)
    
    prev_dist = float('inf')
    monotonic = True
    
    print("\nStep-by-Step Distance to Goal:")
    for step_idx in path:
        step_emb = z_on[step_idx].unsqueeze(0)
        d_to_goal = trainer.manifold.dist(step_emb, goal_emb).item()
        
        diff = prev_dist - d_to_goal
        print(f"  Node {step_idx} ({kb.get_rule(step_idx).name}): d={d_to_goal:.4f} (Reduction: {diff:.4f})")
        
        if d_to_goal >= prev_dist and step_idx != path[0]: # Allow start node to set initial
             print(f"  [!] VIOLATION: Metric increased or stayed same!")
             monotonic = False
        
        prev_dist = d_to_goal
        
    if monotonic:
        print("\n>>> CLAIM 1 VERIFIED: Descent is monotonic.")
        return True
    else:
        print("\n>>> CLAIM 1 FAILED: Distance metric is not strictly monotonic.")
        return False

if __name__ == "__main__":
    test_claim_1()
