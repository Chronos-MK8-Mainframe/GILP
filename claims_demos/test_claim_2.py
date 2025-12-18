import torch
from common import get_trained_model
from gilp_core.search.ahsp import AHSPSearch

def test_claim_2():
    print("\n=== Testing Claim 2: Fossilization (OFF Mode Navigation) ===")
    print("Claim: Learned geometry can execute reasoning without reconstructing logic.")
    
    model, kb, graph_data, trainer, tokens, types = get_trained_model()
    
    # OFF Mode (Empty Graph)
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=graph_data.edge_index.device)
    empty_edge_type = torch.empty((0,), dtype=torch.long, device=graph_data.edge_type.device)
    
    with torch.no_grad():
        z_off = model(tokens, types, empty_edge_index, empty_edge_type)
        
    searcher = AHSPSearch(z_off)
    
    start, goal = 1, 10
    print(f"Seeking path (Heuristics OFF): {kb.get_rule(start).name} -> {kb.get_rule(goal).name}")
    
    path, dists, status = searcher.find_path_hyperbolic_greedy(start, goal)
    
    print(f"Path Found: {path}")
    print(f"Status: {status}")
    
    if status == "SUCCESS" and path[-1] == goal:
        print("\n>>> CLAIM 2 VERIFIED: Navigated successfully without graph edges.")
        return True
    else:
        print("\n>>> CLAIM 2 FAILED: Could not navigate without graph structure.")
        return False

if __name__ == "__main__":
    test_claim_2()
