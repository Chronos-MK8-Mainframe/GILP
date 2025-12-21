
"""
Test Parabolic Visualization

Runs a thought cycle and generates the brain scan.
"""

import sys
import os
import torch

sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.visualization.visualizer import visualize_trajectory

def main():
    print("Initializing Parabolic Mind...")
    sys = Epsilon()
    
    # 1. Embed Rules (Simulated Training)
    # We create some "Logic Cores" (Zeros in atmosphere)
    def create_concept(core_vals):
        z = torch.zeros(64)
        z[:4] = torch.tensor(core_vals) # Hardcode core logic
        return sys.manifold.normalize(z)
        
    sys.knowledge_store["Failure"] = create_concept([1.0, 0.0, 0.0, 0.0])
    sys.knowledge_store["Success"] = create_concept([-1.0, 1.0, 0.0, 0.0])
    sys.knowledge_store["Math"] = create_concept([0.0, 0.0, 1.0, 1.0])
    
    print(f"Embedded {len(sys.knowledge_store)} Logic Concepts.")
    
    # 2. Think
    prompt = "I feel like a failure"
    response, trace = sys.think(prompt)
    
    print("\nTrace Steps:")
    for step, vec in trace:
        print(f"  - {step} [Norm: {vec.norm():.2f}]")
        
    # 3. Visualize
    print("\nGenerating Brain Scan...")
    visualize_trajectory(trace, sys.knowledge_store, "thought_trajectory.png")

if __name__ == "__main__":
    main()
