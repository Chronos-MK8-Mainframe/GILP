
"""
Test Sandbox Interaction

Simulates Epsilon modifying its own environment code.
"""

import sys
import os
import time

sys.path.insert(0, os.getcwd())

# Mocking Epsilon modifying the file directly for this test
def epsilon_modify_env(new_code):
    path = "epsilon/env/dynamic_render.py"
    print(f"Epsilon is rewriting {path}...")
    with open(path, "w") as f:
        f.write(new_code)
    print("✓ Rewrite Complete.")

def main():
    print("="*60)
    print("     EPSILON SELF-MODIFICATION TEST")
    print("="*60)
    
    # 1. Define new behavior (Blue Mood)
    new_code = """
\"\"\"
Dynamic Render Script (Modified by Epsilon)
\"\"\"
import pygame
import math

def render(screen, ticks):
    width, height = screen.get_size()
    
    # Blue Background (Calm)
    b = int(127 + 127 * math.sin(ticks * 0.001))
    screen.fill((0, 0, b)) 
    
    # Draw Square instead of Circle
    rect = pygame.Rect(width//2 - 50, height//2 - 50, 100, 100)
    pygame.draw.rect(screen, (255, 255, 0), rect)
    
    font = pygame.font.SysFont("Arial", 32)
    text = font.render("Epsilon Modified This!", True, (255, 255, 255))
    screen.blit(text, (50, 50))
"""

    print("Simulating cognitive decision to change environment...")
    time.sleep(1)
    
    # 2. Apply Change
    epsilon_modify_env(new_code)
    
    print("\nSUCCESS: Code updated.")
    print("Run 'python epsilon/env/sandbox.py' to see the Blue Screen!")

if __name__ == "__main__":
    main()
