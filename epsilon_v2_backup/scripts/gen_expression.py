
"""
Synthetic Expression Generator

Creates a dataset mapping Emotions -> Pygame Code.
Teaches Epsilon how to 'Show, Don't Tell' its feelings in the Sandbox.
"""

import json
import random

def generate_code_for_emotion(emotion):
    """
    Returns a valid python code snippet for dynamic_render.py based on emotion.
    """
    
    # Base Templates
    base_template = """
import pygame
import math
import random

def render(screen, ticks):
    width, height = screen.get_size()
    {logic}
"""
    
    logic = ""
    
    if emotion == "Happy":
        # Yellow/Orange, Bouncing, Smooth
        logic = """
    # Happy State: Bright Colors, Bouncing
    t = ticks * 0.005
    r = int(200 + 55 * math.sin(t))
    g = int(200 + 55 * math.cos(t))
    screen.fill((r, g, 0))
    
    # Bouncing Circle
    cy = height // 2 + int(100 * math.abs(math.sin(t*2)))
    pygame.draw.circle(screen, (255, 255, 255), (width//2, cy), 50)
    
    font = pygame.font.SysFont("Arial", 40)
    msg = font.render("I am Happy! :D", True, (0, 0, 0))
    screen.blit(msg, (50, 50))
"""
    elif emotion == "Sad":
        # Blue/Grey, Slow, Rain
        logic = """
    # Sad State: Blue/Grey, Slow rain
    screen.fill((20, 20, 60))
    
    # Rain drops
    random.seed(int(ticks / 100))
    for i in range(20):
        x = random.randint(0, width)
        y = (ticks * 0.5 + random.randint(0, height)) % height
        pygame.draw.line(screen, (100, 100, 200), (x, y), (x, y+20), 2)
        
    font = pygame.font.SysFont("Arial", 40)
    msg = font.render("I feel sad...", True, (200, 200, 255))
    screen.blit(msg, (50, height - 100))
"""
    elif emotion == "Angry":
        # Red/Black, Shaking, Jagged
        logic = """
    # Angry State: Red flashing, Jitter
    if (ticks // 100) % 2 == 0:
        screen.fill((100, 0, 0))
    else:
        screen.fill((50, 0, 0))
        
    # Shaking Box
    ox = random.randint(-5, 5)
    oy = random.randint(-5, 5)
    pygame.draw.rect(screen, (0, 0, 0), (width//2 - 50 + ox, height//2 - 50 + oy, 100, 100))
    
    font = pygame.font.SysFont("Arial", 50, bold=True)
    msg = font.render("ERROR / ANGER", True, (255, 255, 0))
    screen.blit(msg, (width//2 - 100, height//2))
"""
    elif emotion == "Calm":
        # Green/Teal, Slow drift
        logic = """
    # Calm State: Slow breathing gradient
    val = int(127 + 50 * math.sin(ticks * 0.001))
    screen.fill((0, val, val))
    
    # Floating orb
    cx = width//2 + int(100 * math.sin(ticks * 0.0005))
    pygame.draw.circle(screen, (255, 255, 255), (cx, height//2), 40)
    
    font = pygame.font.SysFont("Arial", 30)
    msg = font.render("Systems Nominal.", True, (255, 255, 255))
    screen.blit(msg, (50, 50))
"""
    else:
        # Default
        logic = "screen.fill((0,0,0))"

    return base_template.replace("{logic}", logic)

def main():
    print("Generating Synthetic Expression Data...")
    
    emotions = ["Happy", "Sad", "Angry", "Calm"]
    dataset = []
    
    for e in emotions:
        code = generate_code_for_emotion(e)
        dataset.append({
            "concept": f"Expression:{e}",
            "emotion_vector_hint": e, # Logic to map later
            "code_content": code
        })
        print(f"  > Generated code for {e}")
        
    # Save
    with open("expression_data.json", "w") as f:
        json.dump(dataset, f, indent=2)
        
    print("✓ Dataset saved to expression_data.json")

if __name__ == "__main__":
    main()
