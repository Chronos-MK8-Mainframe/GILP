
"""
Test Advanced Features (Phase 4)

Verifies:
1. RAG (Search Integration)
2. Hypothesis Space (Working Memory)
3. Geometric Compression (Vector Rules)
"""

import sys
import os
import torch

# Add root to path
sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon

def main():
    print("="*60)
    print("     EPSILON v2 ADVANCED FEATURES TEST")
    print("="*60)
    
    # 1. Initialize
    epsilon = Epsilon()
    
    # 2. Test RAG & Hypothesis Space
    print("\n--- Test 1: RAG & Working Memory ---")
    question = "What is the Capital of France?"
    print(f"User: {question}")
    
    # Run think() which triggers RAG
    epsilon.think(question)
    
    # Verify it hit memory
    mem_item = epsilon.memory.query("Capital of France:Paris")
    if mem_item is not None:
        print("✓ Success: Fact 'Capital of France:Paris' found in Hypothesis Space.")
    else:
        print("✗ Failure: Fact not found in memory.")
        
    # 3. Test Geometric Compression
    print("\n--- Test 2: Geometric Compression ---")
    # Simulate some vector pairs for "Country -> Capital"
    # A + v = B
    rule_vec = torch.tensor([0.5] * 64)
    
    paris = torch.randn(64)
    france = paris - rule_vec # So France + Rule = Paris
    
    tokyo = torch.randn(64)
    japan = tokyo - rule_vec
    
    pairs = [(france, paris), (japan, tokyo)]
    
    print("Learning Rule: 'is_capital_of' from 2 pairs...")
    epsilon.compression.learn_rule("is_capital_of", pairs)
    
    rule = epsilon.compression.get_rule("is_capital_of")
    if rule:
        print(f"✓ Success: Rule Learned. Vector Mean: {rule.vector.mean():.4f}")
    else:
        print("✗ Failure: Rule not learned.")
        
    # 4. Verify Compression Logic
    # Predict new capital
    berlin_gt = torch.randn(64)
    germany = berlin_gt - rule_vec
    
    prediction = rule.apply(germany)
    error = torch.norm(prediction - berlin_gt)
    print(f"Prediction Error on new data: {error:.6f}")
    
    if error < 1e-5:
        print("✓ Success: Compression Logic works (Vector Arithmetic holds).")
    else:
        print("✗ Failure: Prediction too far off.")

if __name__ == "__main__":
    main()
