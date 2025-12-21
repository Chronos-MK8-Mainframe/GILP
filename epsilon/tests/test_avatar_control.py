
"""
Test Avatar Control

Verifies Epsilon can drive the Pygame Avatar.
"""

import sys
import os
import torch

sys.path.insert(0, os.getcwd())

import epsilon.env.world as world
from epsilon.system import Epsilon
from epsilon.knowledge.conceptnet_ingestor import ConceptNetIngestor
from epsilon.env.interface import ActionInterface

def main():
    print("="*60)
    print("     EPSILON AVATAR DIAGNOSTIC")
    print("="*60)
    
    # 1. Init System & Knowledge
    sys_instance = Epsilon()
    ingestor = ConceptNetIngestor(sys_instance)
    ingestor.ingest_mininet()
    
    # 2. Init Action Interface
    actor = ActionInterface()
    avatar = world.avatar_instance
    
    print(f"Initial Avatar State: Y={avatar.y}, Color={avatar.color}")
    
    # 3. Construct "Joyful Jump" Thought
    # Logic: "Happy" (Atmosphere [1,1,0]) + "Action:Jump" (Dim 11 = 1.0)
    
    # Place avatar on ground so it can jump
    avatar.y = 460
    
    thought = torch.zeros(64)
    # Emotion
    thought[5:8] = torch.tensor([1.0, 1.0, 0.0]) 
    # Action
    thought[11] = 1.0 
    
    print("Executing 'Joyful Jump' Vector...")
    actor.execute_vector(thought)
    
    # 4. Verify Physics
    # Jump sets vy to -12
    if avatar.vy == -12:
        print("✓ PHYSICS CONFIRMED: Avatar is jumping (VY = -12).")
    else:
        print(f"✗ MOVE FAILED: VY = {avatar.vy}")
        
    # 5. Verify Emotion
    # Happy sets color to Yellow (255, 255, 0)
    if avatar.color == (255, 255, 0):
        print("✓ EMOTION CONFIRMED: Avatar turned Yellow (Happy).")
    else:
        print(f"✗ COLOR FAILED: {avatar.color}")

if __name__ == "__main__":
    main()
