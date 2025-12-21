
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

# Ensure path for imports
sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.knowledge.conceptnet_ingestor import ConceptNetIngestor
from epsilon.env.interface import ActionInterface
import epsilon.env.world as world

# The dynamic module we will reload
import epsilon.env.dynamic_render as dynamic_render

WIDTH, HEIGHT = 800, 600
FPS = 60

def main():
    print("Initializing Pygame Sandbox + Brain...")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Epsilon Sandbox [Chat Enabled]")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)
    
    # --- BRAIN BOOT ---
    brain = Epsilon()
    ingestor = ConceptNetIngestor(brain)
    ingestor.ingest_mininet()
    ingestor.ingest_chat_pack()
    actor = ActionInterface()
    print("Brain Online.")
    
    # --- CHAT STATE ---
    user_text = ''
    chat_history = ["System: Welcome. Chat with Epsilon below."]
    is_thinking = False
    
    running = True
    last_reload = time.time()
    
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # Chat Input
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                elif event.key == pygame.K_RETURN:
                    if user_text.strip():
                        # Submit to Brain
                        chat_history.append(f"You: {user_text}")
                        
                        # Process (Simple blocking for now, could be threaded)
                        try:
                            response, trace = brain.think(user_text)
                            chat_history.append(f"Epsilon: {response}")
                            
                            # Execute Actions
                            final_vec = None
                            for label, vec in trace:
                                if "Expression" in label or "Reflex" in label:
                                    final_vec = vec
                            if final_vec is not None:
                                actor.execute_vector(final_vec)
                                
                        except Exception as e:
                            print(f"Brain Error: {e}")
                            chat_history.append("System: Brain Error.")
                            
                        user_text = '' # Clear input
                        
                        # Keep history short
                        if len(chat_history) > 10:
                            chat_history.pop(0)
                            
                else:
                    # Append char
                    user_text += event.unicode

        # Auto-Reload Logic (Every 2 seconds check)
        if time.time() - last_reload > 2.0:
            try:
                importlib.reload(dynamic_render)
                # print("Reloaded render logic.")
            except Exception as e:
                print(f"Reload Error: {e}")
            last_reload = time.time()
            
        # 3. Physics
        world.update_world()
        
        # 4. Render World (Dynamic)
        screen.fill((0, 0, 0)) # Clean slate
        try:
            if hasattr(dynamic_render, 'render'):
                dynamic_render.render(screen, pygame.time.get_ticks())
            else:
                s = font.render("No 'render' found.", True, (255, 0, 0))
                screen.blit(s, (10, 10))
        except Exception as e:
            s = font.render(f"Render Error: {e}", True, (255, 0, 0))
            screen.blit(s, (10, 10))

        # 5. Render Chat Overlay
        # Transparent background for chat
        chat_bg = pygame.Surface((WIDTH, 250))
        chat_bg.set_alpha(150)
        chat_bg.fill((0, 0, 0))
        screen.blit(chat_bg, (0, HEIGHT - 250))
        
        # Draw History
        y_offset = HEIGHT - 240
        for line in chat_history:
            col = (100, 255, 100) if "Epsilon" in line else (200, 200, 200)
            txt_surf = font.render(line, True, col)
            screen.blit(txt_surf, (20, y_offset))
            y_offset += 20
            
        # Draw Input Line
        pygame.draw.line(screen, (255, 255, 255), (0, HEIGHT - 40), (WIDTH, HEIGHT - 40))
        input_surf = font.render(f"> {user_text}", True, (255, 255, 255))
        screen.blit(input_surf, (20, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()

if __name__ == "__main__":
    main()
