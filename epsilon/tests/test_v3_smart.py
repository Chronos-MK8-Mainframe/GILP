
"""
Epsilon v3 "Smart Bot" Verification

Tests:
1. Vector Vision (Can she see the code?)
2. Critical Thinking (Does she hesitate?)
"""

import sys
import os
import torch
import time

sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.env.visual_cortex import VisualCortex

def main():
    print("="*60)
    print("     EPSILON V3 SMART DIAGNOSTIC")
    print("="*60)
    
    sys_instance = Epsilon()
    
    # --- TEST 1: VECTOR VISION ---
    print("\n[TEST 1] Vector Vision (The Eye)")
    cortex = VisualCortex(sys_instance.manifold)
    
    # 1. Write Angry Code (Synthetic)
    angry_code = """
import pygame
import random
def render(screen, ticks):
    screen.fill((255, 0, 0)) # RED
    x = random.randint(0, 100) # CHAOS
"""
    with open("epsilon/env/dynamic_render.py", "w") as f:
        f.write(angry_code)
        
    print(" > Wrote 'Angry/Red' code to sandbox.")
    
    # 2. Analyze
    vec, stats = cortex.analyze_scene()
    print(f" > Visual Stats: {stats}")
    print(f" > Visual Vector (Atmosphere): {vec[5:8].tolist()}")
    
    # We expect Anger Vector [1.0, -1.0, 0.0] roughly
    if stats.get("red", 0) > 0 and vec[6] < 0:
        print("✓ VISION CONFIRMED: Detected Anger (Red).")
    else:
        print("✗ VISION FAILED.")
        
    # --- TEST 2: CRITICAL THINKING ---
    print("\n[TEST 2] Critical Thinking (The Conscience)")
    
    # 1. Setup Conflict
    # Personality = Calm [ -1.0, 1.0, 0.0 ]
    with torch.no_grad():
        sys_instance.personality_vector.fill_(0)
        sys_instance.personality_vector[5:8] = torch.tensor([-1.0, 1.0, 0.0])
    
    # Logic = Anger [ 1.0, -1.0, 0.0 ] (We cheat and inject a concept)
    # We need a concept that maps to Anger logic
    sys_instance.knowledge_store["Provocation"] = torch.zeros(64)
    sys_instance.knowledge_store["Provocation"][5:8] = torch.tensor([1.0, -1.0, 0.0])
    
    print(" > Personality set to CALM.")
    print(" > Input triggers 'Provocation' (ANGRY logic).")
    
    # 2. Think (High Tension expected)
    response, trace = sys_instance.think("Provocation")
    
    conflict_found = False
    for label, vec in trace:
        if "Conflict" in label:
            conflict_found = True
            
    if conflict_found:
        print("✓ CONSCIOUSNESS CONFIRMED: System entered Debate Loop due to conflict.")
    else:
        print("✗ FAILED: No conflict detected.")

if __name__ == "__main__":
    main()
