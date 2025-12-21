
"""
Epsilon v3 Full System Regression
"The Health Check"

Verifies:
1. Memory Integrity (No NaNs)
2. Semantic Logic (Embeddings make sense)
3. Cognitive Stability (Thinking works)
4. Physical Reaction (Action works)
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.env.interface import ActionInterface
from epsilon.env.world import avatar_instance

def main():
    print("="*60)
    print("     EPSILON FULL SYSTEM REGRESSION")
    print("="*60)
    
    # 1. Load System
    print("[1] Loading System...")
    sys_instance = Epsilon()
    # Assume Memory is already loaded or we ingest mini-net
    from epsilon.knowledge.conceptnet_ingestor import ConceptNetIngestor
    ingestor = ConceptNetIngestor(sys_instance)
    ingestor.ingest_mininet()
    
    # 2. Memory Audit (NaN Check)
    print("\n[2] Memory Integrity Audit")
    nan_count = 0
    total = len(sys_instance.knowledge_store)
    
    for k, v in sys_instance.knowledge_store.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"  ! CORRUPTION DETECTED in '{k}'")
                nan_count += 1
                
    if nan_count == 0:
        print(f"✓ Memory Healthy ({total} nodes). No NaNs.")
    else:
        print(f"✗ CRITICAL FAILURE: {nan_count} corrupted vectors.")
        return

    # 3. Geometric Logic Check
    print("\n[3] Geometric Logic Verification")
    # Rule: Dist(Computer, Coding) < Dist(Computer, Wet)
    
    def get_vec(name):
        return sys_instance.knowledge_store[name]
        
    v_comp = get_vec("Computer")
    v_code = get_vec("Coding")
    v_wet = get_vec("Wet")
    
    d1 = torch.norm(v_comp - v_code).item()
    d2 = torch.norm(v_comp - v_wet).item()
    
    print(f"  Dist(Computer, Coding): {d1:.4f}")
    print(f"  Dist(Computer, Wet):    {d2:.4f}")
    
    if d1 < d2:
        print("✓ SEMANTICS PASS: Computer is closer to Coding.")
    else:
        print("✗ SEMANTICS FAIL: Bindings are broken.")
        
    # 4. Cognitive Loop (Think)
    print("\n[4] Cognitive Loop Test")
    try:
        # Force a conflict to test the complex path
        with torch.no_grad():
            sys_instance.personality_vector.fill_(0)
            sys_instance.personality_vector[5:8] = torch.tensor([1.0, 1.0, 0.0]) # Happy
            
        # Provoke with "Chaos" (Antonym of Logic)
        print("  Thinking about 'Chaos'...")
        response, trace = sys_instance.think("Chaos")
        print(f"  Response: {response}")
        print("✓ Think Cycle Completed.")
        
    except Exception as e:
        print(f"✗ CRASH IN THINK LOOP: {e}")
        import traceback
        traceback.print_exc()

    # 5. Action Loop (Body)
    print("\n[5] Action Loop Test")
    try:
        interface = ActionInterface()
        # "Jump" Vector
        vec = torch.zeros(64)
        vec[11] = 1.0 
        
        # Reset avatar
        avatar_instance.y = 460 # Ground
        avatar_instance.vy = 0
        
        interface.execute_vector(vec)
        
        if avatar_instance.vy != 0:
            print("✓ Action System Healthy (Physics Triggered).")
        else:
            print("✗ ACTION FAIL: Avatar did not respond.")
            
    except Exception as e:
        print(f"✗ CRASH IN ACTION LOOP: {e}")

if __name__ == "__main__":
    main()
