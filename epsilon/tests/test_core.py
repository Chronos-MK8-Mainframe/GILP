"""
Epsilon Core Unit Tests

Tests for core Epsilon components:
- QuantizedPoincareManifold
- AnchorManager
- FailureDetector
- ProofTrace
"""

import torch
import sys
sys.path.insert(0, 'c:/Users/rupa9/Videos/GILP')

from epsilon.geometry.quantized_poincare import QuantizedPoincareManifold
from epsilon.anchoring.anchor_manager import AnchorManager
from epsilon.proofs.proof_trace import ProofTrace, ProofStep, ProofStatus
from epsilon.diagnostics.failure_detector import FailureDetector, FailureType
from epsilon.config import EpsilonConfig


def test_quantized_poincare_shells():
    """Test radial shell computation."""
    print("\n=== Testing QuantizedPoincareManifold ===")
    
    manifold = QuantizedPoincareManifold(shell_width=0.5)
    
    # Create points at different radii
    points = torch.tensor([
        [0.0, 0.0, 0.0],   # Origin - shell 0
        [0.3, 0.0, 0.0],   # shell 0 (0.3 < 0.5)
        [0.5, 0.0, 0.0],   # shell 1 (0.5 / 0.5 = 1)
        [0.7, 0.0, 0.0],   # shell 1 (0.7 / 0.5 = 1.4 -> floor = 1)
        [0.8, 0.0, 0.0],   # shell 1 (0.8 / 0.5 = 1.6 -> floor = 1)
    ])
    
    shells = manifold.get_shell(points)
    print(f"  Points: {points[:, 0].tolist()}")
    print(f"  Shells: {shells.tolist()}")
    
    # Expected: [0, 0, 1, 1, 1]
    assert shells[0] == 0, "Origin should be shell 0"
    assert shells[1] == 0, "0.3 should be shell 0"
    assert shells[2] == 1, "0.5 should be shell 1"
    
    print("  [PASS] Shell computation correct")
    
    # Test shell ordering loss
    edges = torch.tensor([[0, 1], [2, 3]])  # 0->2, 1->3
    loss = manifold.shell_ordering_loss(points, edges)
    print(f"  Shell ordering loss: {loss.item():.4f}")
    
    print("  [PASS] Shell ordering loss computes")
    return True


def test_anchor_manager():
    """Test anchor management and rigidity loss."""
    print("\n=== Testing AnchorManager ===")
    
    manager = AnchorManager()
    
    # Register some anchors
    manager.register_anchor(0, 1, delta=0.5, strength=1.0, rule_name="A->B")
    manager.register_anchor(1, 2, delta=0.5, strength=0.8, rule_name="B->C")
    
    print(f"  Registered {len(manager.anchors)} anchors")
    
    # Check k-hop neighborhood
    neighborhood = manager.get_k_hop_neighborhood([0], k=2)
    print(f"  2-hop from node 0: {neighborhood}")
    assert 0 in neighborhood
    assert 1 in neighborhood
    assert 2 in neighborhood
    print("  [PASS] K-hop neighborhood correct")
    
    # Test rigidity loss
    manifold = QuantizedPoincareManifold(shell_width=0.5)
    embeddings = torch.tensor([
        [0.1, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.8, 0.0, 0.0],
    ])
    
    loss = manager.compute_rigidity_loss(embeddings, manifold)
    print(f"  Rigidity loss: {loss.item():.4f}")
    assert loss.item() >= 0, "Rigidity loss should be non-negative"
    print("  [PASS] Rigidity loss computes")
    
    # Test stats
    stats = manager.get_anchor_stats()
    print(f"  Stats: {stats}")
    assert stats["count"] == 2
    print("  [PASS] Anchor stats correct")
    
    return True


def test_proof_trace():
    """Test ProofTrace creation and verification."""
    print("\n=== Testing ProofTrace ===")
    
    # Create a proof trace
    trace = ProofTrace(start=0, goal=3)
    
    trace.add_step(ProofStep(
        from_node=0, to_node=1, rule_applied="A->B",
        distance_traveled=0.5, shell_transition=(2, 1)
    ))
    trace.add_step(ProofStep(
        from_node=1, to_node=2, rule_applied="B->C",
        distance_traveled=0.5, shell_transition=(1, 0)
    ))
    trace.add_step(ProofStep(
        from_node=2, to_node=3, rule_applied="C->D",
        distance_traveled=0.5, shell_transition=(0, 0)
    ))
    trace.status = ProofStatus.SUCCESS
    
    print(f"  Path: {trace.path}")
    print(f"  Length: {trace.length}")
    print(f"  Total distance: {trace.total_distance:.2f}")
    
    assert trace.path == [0, 1, 2, 3], "Path should be [0, 1, 2, 3]"
    assert trace.length == 3, "Length should be 3"
    assert abs(trace.total_distance - 1.5) < 0.01, "Total distance should be 1.5"
    print("  [PASS] ProofTrace construction correct")
    
    # Test serialization
    data = trace.to_dict()
    print(f"  Serialized keys: {list(data.keys())}")
    assert "path" in data
    assert "steps" in data
    print("  [PASS] ProofTrace serialization works")
    
    # Test string output
    string = trace.to_string()
    print("  String representation:")
    for line in string.split('\n')[:4]:
        print(f"    {line}")
    print("  [PASS] ProofTrace string output works")
    
    return True


def test_failure_detector():
    """Test failure detection and interpretation."""
    print("\n=== Testing FailureDetector ===")
    
    config = EpsilonConfig()
    manifold = QuantizedPoincareManifold(shell_width=0.5)
    detector = FailureDetector(manifold, config)
    
    # Create test embeddings
    embeddings = torch.tensor([
        [0.1, 0.0, 0.0],  # 0: close to origin
        [0.5, 0.0, 0.0],  # 1
        [0.6, 0.1, 0.0],  # 2
        [0.7, 0.0, 0.0],  # 3
        [0.8, 0.0, 0.0],  # 4: far
    ])
    detector._embeddings = embeddings
    
    # Test NO_DESCENT detection
    # Current=4 (far), neighbors=[3], goal=0 (close)
    # Neighbor 3 is still farther from goal than current
    failure, msg = detector.detect(
        current=4,
        neighbors=[],  # No neighbors
        goal=0,
        history=[4]
    )
    print(f"  No neighbors: {failure} - {msg[:50]}...")
    assert failure == FailureType.NO_DESCENT
    print("  [PASS] NO_DESCENT detected correctly")
    
    # Test OSCILLATION detection
    failure, msg = detector.detect(
        current=2,
        neighbors=[1, 3],
        goal=0,
        history=[1, 2, 1, 2, 1, 2]  # Oscillating between 1 and 2
    )
    print(f"  Oscillation: {failure} - {msg[:50]}..." if failure else "  Oscillation: None")
    # May or may not trigger depending on distances
    
    # Test interpretation
    interp = detector.get_failure_interpretation(FailureType.NO_DESCENT)
    print(f"  Interpretation: {interp[:60]}...")
    assert "PROOF IMPOSSIBLE" in interp
    print("  [PASS] Failure interpretation works")
    
    # Test stats
    stats = detector.get_failure_stats()
    print(f"  Stats: {stats}")
    print("  [PASS] Failure stats computed")
    
    return True


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("       EPSILON CORE UNIT TESTS")
    print("=" * 60)
    
    results = {}
    
    try:
        results["QuantizedPoincareManifold"] = test_quantized_poincare_shells()
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["QuantizedPoincareManifold"] = False
    
    try:
        results["AnchorManager"] = test_anchor_manager()
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["AnchorManager"] = False
    
    try:
        results["ProofTrace"] = test_proof_trace()
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["ProofTrace"] = False
    
    try:
        results["FailureDetector"] = test_failure_detector()
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["FailureDetector"] = False
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n>>> ALL CORE TESTS PASSED")
    else:
        print("\n>>> SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
