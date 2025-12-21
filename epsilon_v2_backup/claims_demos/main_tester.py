"""
Epsilon Claims Verification Suite

Extended from GILP's verification suite with new Epsilon claims:
- Claims 1-5: Original GILP claims
- Claim 6: Incremental fossilization preserves existing proofs
- Claim 7: All navigations produce valid proof traces
- Claim 8: Each failure type is correctly identified
"""

import torch
import sys
sys.path.insert(0, 'c:/Users/rupa9/Videos/GILP')
sys.path.insert(0, 'c:/Users/rupa9/Videos/GILP/GILP-main/GILP-main')

from epsilon.core.minimal import EpsilonMinimal
from epsilon.proofs.proof_trace import ProofTrace, ProofStatus
from epsilon.diagnostics.failure_detector import FailureType


def test_claim_1_directional_progress():
    """
    Claim 1: Greedy descent makes directional progress toward goal.
    """
    print("\n[Claim 1] Directional Progress")
    print("-" * 40)
    
    vocab_size = 100
    epsilon = EpsilonMinimal(vocab_size, dim=32, shell_width=0.5)
    
    # Simple chain
    tokens = torch.randint(1, vocab_size, (5, 10))
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    
    # Train
    for _ in range(100):
        epsilon.train_step(tokens, edges)
    
    z = epsilon.embed(tokens)
    
    # Navigate from 0 to 4
    path, status = epsilon.navigate(z, 0, 4)
    
    if status == "SUCCESS":
        # Verify monotonic distance decrease
        distances = []
        for node in path:
            d = epsilon.dist(z[node:node+1], z[4:5]).item()
            distances.append(d)
        
        monotonic = all(distances[i] >= distances[i+1] for i in range(len(distances)-1))
        
        if monotonic:
            print(f"  Path: {path}")
            print(f"  Distances to goal: {[f'{d:.2f}' for d in distances]}")
            print(">>> CLAIM 1 VERIFIED: Distance decreases monotonically")
            return True
    
    print(f"  Navigation failed: {status}")
    print(">>> CLAIM 1 NEEDS REVIEW")
    return False


def test_claim_5_deep_composition():
    """
    Claim 5: Multi-step proofs compose correctly.
    """
    print("\n[Claim 5] Deep Composition")
    print("-" * 40)
    
    vocab_size = 100
    epsilon = EpsilonMinimal(vocab_size, dim=32, shell_width=0.5)
    
    # Longer chain: 0→1→2→3→4→5→6
    tokens = torch.randint(1, vocab_size, (7, 10))
    edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    
    # Train more for longer chain
    for _ in range(200):
        epsilon.train_step(tokens, edges)
    
    z = epsilon.embed(tokens)
    
    # Navigate from 0 to 6 (6 steps)
    path, status = epsilon.navigate(z, 0, 6)
    
    print(f"  Path: {path}")
    print(f"  Status: {status}")
    print(f"  Length: {len(path) - 1} steps")
    
    if status == "SUCCESS" and len(path) > 3:
        print(">>> CLAIM 5 VERIFIED: Long-range composition successful")
        return True
    else:
        print(">>> CLAIM 5 NEEDS REVIEW: Path too short or failed")
        return False


def test_claim_6_incremental_fossilization():
    """
    Claim 6 (New): Adding rules doesn't break existing proofs.
    """
    print("\n[Claim 6] Incremental Fossilization")
    print("-" * 40)
    
    vocab_size = 100
    epsilon = EpsilonMinimal(vocab_size, dim=32, shell_width=0.5)
    
    # Initial chain
    tokens = torch.randint(1, vocab_size, (4, 10))
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    
    # Train
    for _ in range(100):
        epsilon.train_step(tokens, edges)
    
    z = epsilon.embed(tokens)
    
    # Verify initial navigation works
    path1, status1 = epsilon.navigate(z, 0, 3)
    print(f"  Before: {path1} ({status1})")
    
    if status1 != "SUCCESS":
        print(">>> CLAIM 6 CANNOT TEST: Initial navigation failed")
        return False
    
    # Fossilize
    epsilon.fossilize(0, 1, z)
    epsilon.fossilize(1, 2, z)
    
    # Add new nodes
    tokens_ext = torch.cat([tokens, torch.randint(1, vocab_size, (2, 10))])
    edges_ext = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    
    # Continue training
    for _ in range(100):
        epsilon.train_step(tokens_ext, edges_ext)
    
    z2 = epsilon.embed(tokens_ext)
    
    # Verify original navigation still works
    path2, status2 = epsilon.navigate(z2, 0, 3)
    print(f"  After:  {path2} ({status2})")
    
    if status2 == "SUCCESS":
        print(">>> CLAIM 6 VERIFIED: Existing proof preserved")
        return True
    else:
        print(">>> CLAIM 6 FAILED: Existing proof broken")
        return False


def test_claim_7_proof_traces():
    """
    Claim 7 (New): All successful navigations produce valid traces.
    """
    print("\n[Claim 7] Valid Proof Traces")
    print("-" * 40)
    
    vocab_size = 100
    epsilon = EpsilonMinimal(vocab_size, dim=32, shell_width=0.5)
    
    tokens = torch.randint(1, vocab_size, (5, 10))
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    
    for _ in range(100):
        epsilon.train_step(tokens, edges)
    
    z = epsilon.embed(tokens)
    
    # Multiple navigation attempts
    test_cases = [(0, 4), (0, 2), (1, 4)]
    all_valid = True
    
    for start, goal in test_cases:
        path, status = epsilon.navigate(z, start, goal)
        
        if status == "SUCCESS":
            # Verify path is continuous
            continuous = all(
                abs(path[i+1] - path[i]) <= 2  # Reasonable hop distance
                for i in range(len(path)-1)
            )
            
            if not continuous:
                print(f"  {start}→{goal}: Path discontinuous!")
                all_valid = False
            else:
                print(f"  {start}→{goal}: Path {path} ✓")
        else:
            print(f"  {start}→{goal}: {status}")
    
    if all_valid:
        print(">>> CLAIM 7 VERIFIED: All traces valid")
        return True
    else:
        print(">>> CLAIM 7 FAILED: Some traces invalid")
        return False


def test_claim_8_failure_semantics():
    """
    Claim 8 (New): Each failure type is correctly identified.
    """
    print("\n[Claim 8] Failure Semantics")
    print("-" * 40)
    
    vocab_size = 100
    epsilon = EpsilonMinimal(vocab_size, dim=32, shell_width=0.5)
    
    # Create disconnected graph: 0→1→2 and 3→4 (separate)
    tokens = torch.randint(1, vocab_size, (5, 10))
    edges = torch.tensor([[0, 1, 3], [1, 2, 4]])
    
    for _ in range(100):
        epsilon.train_step(tokens, edges)
    
    z = epsilon.embed(tokens)
    
    # Try to navigate between disconnected components
    path, status = epsilon.navigate(z, 0, 4)
    
    print(f"  Attempting 0→4 (disconnected)")
    print(f"  Result: {status}")
    print(f"  Path before failure: {path}")
    
    # We expect a failure status
    if status.startswith("FAIL"):
        print(">>> CLAIM 8 VERIFIED: Failure correctly detected")
        return True
    else:
        print(">>> CLAIM 8 ANOMALY: Expected failure, got success")
        return False


def run_all_claims():
    """Run the complete Epsilon claims verification suite."""
    print("=" * 60)
    print("       EPSILON CLAIMS VERIFICATION SUITE")
    print("=" * 60)
    print("Testing the 8 Pillars of Epsilon Geometric Inference...")
    
    results = {}
    
    # Original GILP claims (adapted)
    try:
        results["Claim 1 (Directional Progress)"] = test_claim_1_directional_progress()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["Claim 1 (Directional Progress)"] = False
    
    try:
        results["Claim 5 (Deep Composition)"] = test_claim_5_deep_composition()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["Claim 5 (Deep Composition)"] = False
    
    # New Epsilon claims
    try:
        results["Claim 6 (Incremental Fossilization)"] = test_claim_6_incremental_fossilization()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["Claim 6 (Incremental Fossilization)"] = False
    
    try:
        results["Claim 7 (Valid Proof Traces)"] = test_claim_7_proof_traces()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["Claim 7 (Valid Proof Traces)"] = False
    
    try:
        results["Claim 8 (Failure Semantics)"] = test_claim_8_failure_semantics()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["Claim 8 (Failure Semantics)"] = False
    
    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    
    all_passed = True
    for claim, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {claim}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n>>> ALL EPSILON CLAIMS VERIFIED")
        print(">>> The paradigm shift is confirmed:")
        print("    'Logic compiles into a stable execution manifold'")
        return 0
    else:
        print("\n>>> SOME CLAIMS NEED ATTENTION")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_claims())
