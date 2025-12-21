
"""
Dynamic Render Script (Modified by Epsilon)
"""
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
