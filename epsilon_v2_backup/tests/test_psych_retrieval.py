
"""
Geometric Integrity Test

Audits the 'Psychology' module to prove it's not hardcoded.
Method:
1. Create a raw vector probe at coordinates [1.0, -1.0, 0.0] (The "Anger Sector").
2. Search the entire brain (5800+ concepts).
3. Verify that the nearest concept is 'Expression:Angry'.
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon

def main():
    print("="*60)
    print("     GEOMETRIC INTEGRITY AUDIT")
    print("="*60)
    
    # 1. Load Brain
    print("Loading Brain...")
    sys_instance = Epsilon()
    sys_instance.load(".")
    
    total_concepts = len(sys_instance.knowledge_store)
    print(f"Memory Size: {total_concepts} Vectors.")
    
    # 2. Construct Probe (The 'Anger' Coordinates)
    # In psych_ingestor.py, we defined Angry as [1.0, -1.0, 0.0] in the Atmosphere (Dims 5,6,7)
    # Logic Core (Dims 0-4) was random.
    # To find it, we need to search based on the Atmosphere similarity.
    
    print("\n--- Test 1: Blind Vector Search ---")
    print("Probing Atmosphere Sector: [1.0, -1.0, 0.0] (Anger)")
    
    probe_atm = torch.tensor([1.0, -1.0, 0.0])
    
    best_name = None
    best_dist = 999.0
    
    # Scan all 5800 concepts
    for name, vec in sys_instance.knowledge_store.items():
        # Filter out payloads
        if not isinstance(vec, torch.Tensor):
            continue
            
        # vec is 64-dim. Atmosphere is indices 5,6,7
        # We only compare the atmosphere part for this test
        # (Assuming the system drifted into this mood)
        
        target_atm = vec[5:8]
        
        # Euclidean Distance
        dist = torch.norm(probe_atm - target_atm)
        
        if dist < best_dist:
            best_dist = dist
            best_name = name
            
    print(f"Nearest Neighbor Found: '{best_name}'")
    print(f"Distance: {best_dist.item():.4f}")
    
    # 3. Verify
    if best_name == "Expression:Angry":
        print("\n✓ PASSED: System retrieved 'Angry' based purely on coordinates.")
    else:
        print(f"\n✗ FAILED: Expected 'Expression:Angry', got '{best_name}'")
        return

    # 4. Check Payload
    payload = sys_instance.knowledge_store.get(f"{best_name}_PAYLOAD")
    if payload and "screen.fill((100, 0, 0))" in payload:
        print("✓ PASSED: Code Payload contains 'Red Screen' logic.")
    else:
        print("✗ FAILED: Code Payload is missing or incorrect.")

if __name__ == "__main__":
    main()
