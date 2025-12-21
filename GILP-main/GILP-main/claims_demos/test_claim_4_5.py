import torch
from common import get_trained_model
from gilp_core.search.ahsp import AHSPSearch

def test_claims_4_and_5():
    print("\n=== Testing Claim 4 (Meaningful Failure) & Claim 5 (Composition) ===")
    
    model, kb, graph_data, trainer, tokens, types = get_trained_model()
    
    # Use OFF Mode for stricter test of Fossilization reliability
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=graph_data.edge_index.device)
    empty_edge_type = torch.empty((0,), dtype=torch.long, device=graph_data.edge_type.device)
    with torch.no_grad():
        z_off = model(tokens, types, empty_edge_index, empty_edge_type)
    searcher = AHSPSearch(z_off)
    
    # --- Claim 5: Deep Composition ---
    print("\n[Claim 5] Composing many steps preserves global direction.")
    start_c, goal_c = 0, 10 # Number -> RecursiveMult (Deepest path)
    
    print(f"Seeking Deep Path: {kb.get_rule(start_c).name} -> {kb.get_rule(goal_c).name}")
    path_c, _, status_c = searcher.find_path_hyperbolic_greedy(start_c, goal_c)
    print(f"Path: {path_c} ({[kb.get_rule(i).name for i in path_c]})")
    
    claim_5_passed = False
    if status_c == "SUCCESS" and len(path_c) > 2:
        print(">>> CLAIM 5 VERIFIED: Long range composition successful.")
        claim_5_passed = True
    else:
        print(">>> CLAIM 5 FAILED: Path too short or failed.")
        
    # --- Claim 4: Meaningful Failure ---
    print("\n[Claim 4] No-descent states correspond to real logical dead-ends.")
    # Attempting jump from Contradiction -> Logic
    start_f, goal_f = 11, 1 
    print(f"Seeking Impossible Path: {kb.get_rule(start_f).name} -> {kb.get_rule(goal_f).name}")
    
    path_f, _, status_f = searcher.find_path_hyperbolic_greedy(start_f, goal_f)
    print(f"Result: Status={status_f}, Path={path_f}")
    
    claim_4_passed = False
    if status_f == "FAIL_LOCAL_MINIMA":
        print(">>> CLAIM 4 VERIFIED: Agent stopped at dead-end (Local Minima).")
        claim_4_passed = True
    elif status_f == "SUCCESS":
        print(">>> CLAIM 4 FAILED: Agent hallucinated a path through disjoint concepts!")
    else:
        print(">>> CLAIM 4 AMBIGUOUS: Agent wandered.")
        
    return claim_5_passed and claim_4_passed

if __name__ == "__main__":
    test_claims_4_and_5()
