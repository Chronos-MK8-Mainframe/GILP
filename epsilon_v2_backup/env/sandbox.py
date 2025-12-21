
"""
Epsilon Pygame Sandbox (Self-Modifying Environment)

A container that runs a dynamic script.
Auto-reloads when the script is modified by Epsilon.
"""

import pygame
import sys
import importlib
import os
import time
import traceback

# The dynamic module we will reload
import epsilon.env.dynamic_render as dynamic_render

WIDTH, HEIGHT = 800, 600
FPS = 60

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Epsilon Sandbox (Hot Reload)")
    clock = pygame.time.Clock()
    
    # Track file modification
    script_path = os.path.abspath(dynamic_render.__file__)
    last_mtime = os.path.getmtime(script_path)
    
    font = pygame.font.SysFont("Arial", 18)
    status_msg = "Running"
    status_color = (0, 255, 0)
    
    running = True
    while running:
        # 1. Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # 2. Hot Reload Check
        try:
            current_mtime = os.path.getmtime(script_path)
            if current_mtime > last_mtime:
                print(">>> Reloading Dynamic Script...")
                importlib.reload(dynamic_render)
                last_mtime = current_mtime
                status_msg = "Reloaded!"
                status_color = (0, 255, 255)
        except Exception as e:
            print(f"Reload Error: {e}")
            
        # 3. Render
        screen.fill((0, 0, 0)) # Clean slate
        
        try:
            # Call the dynamic render function
            if hasattr(dynamic_render, 'render'):
                dynamic_render.render(screen, pygame.time.get_ticks())
            else:
                s = font.render("No 'render(screen, ticks)' found in script.", True, (255, 0, 0))
                screen.blit(s, (10, 50))
                
        except Exception as e:
            # Catch runtime errors in the dynamic script so sandbox stays alive
            status_msg = f"Runtime Error: {str(e)}"
            status_color = (255, 50, 50)
            # print(traceback.format_exc()) # Optional: Print to console
            
        # UI Overlay
        overlay = font.render(f"Status: {status_msg}", True, status_color)
        screen.blit(overlay, (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()

if __name__ == "__main__":
    main()
