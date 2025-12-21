
"""
Fill Brain Script (Knowledge Ingestion)

Uses the Ingestor to scan code and populate the Parabolic Manifold.
Target: Epsilon's own source code (Self-Knowledge).
"""

import sys
import os
import torch
import random

sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.knowledge.ingestor import PythonIngestor

def main():
    print("="*60)
    print("     EPSILON KNOWLEDGE INGESTION")
    print("="*60)
    
    # 1. Load System
    sys_instance = Epsilon()
    # sys_instance.load(".") # Start fresh or load? Start fresh for clean demo.
    
    # 2. Scan Knowledge
    ingestor = PythonIngestor()
    target_dir = os.path.join(os.getcwd(), "epsilon")
    concepts = ingestor.scan_directory(target_dir)
    
    # 3. Embed into Geometry
    print(f"\nEmbedding {len(concepts)} concepts into Parabolic Space...")
    
    for name, desc in concepts:
        # Create a Logic Core vector for this concept
        # In a real system, we'd use a semantic encoder (e.g. BERT) to initialize position
        # For this Ground Up demo, we hash the name to get a consistent random position
        
        torch.manual_seed(sum(ord(c) for c in name))
        
        # Logic Core: Dimensions 0-4 are active
        z = torch.zeros(64)
        z[:4] = torch.randn(4) 
        
        # Normalize in manifold
        z = sys_instance.manifold.normalize(z)
        
        sys_instance.knowledge_store[name] = z
        # print(f"  + Learned: {name}")
        
    print(f"\n✓ Ingestion Complete. Total Knowledge: {len(sys_instance.knowledge_store)} Concepts.")
    
    # 4. Save
    sys_instance.save(".")
    print("✓ System Saved to ./system_v2.pt")
    
    # 5. Verify
    print("\n--- Verification Query ---")
    response, trace = sys_instance.think("tell me about class Epsilon")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
