
"""
Epsilon Avatar

The 'Body' that Epsilon controls in the Sandbox.
"""

import pygame
import random

class Avatar:
    def __init__(self, x=400, y=300):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.color = (255, 255, 255) # Default White
        self.size = 40
        self.ground_y = 500
        
    def update(self):
        """Physics Update"""
        # Gravity
        self.vy += 0.5
        
        # Movement
        self.x += self.vx
        self.y += self.vy
        
        # Ground Collision
        if self.y > self.ground_y - self.size:
            self.y = self.ground_y - self.size
            self.vy = 0
            
        # Friction
        self.vx *= 0.9
        
    def move(self, dx):
        self.vx += dx
        
    def jump(self):
        if self.y >= self.ground_y - self.size - 5:
            self.vy = -12
            
    def set_emotion(self, emotion):
        """Changes color based on emotion"""
        if emotion == "Happy":
            self.color = (255, 255, 0) # Yellow
        elif emotion == "Sad":
            self.color = (0, 0, 255) # Blue
        elif emotion == "Angry":
            self.color = (255, 0, 0) # Red
        elif emotion == "Calm":
            self.color = (0, 255, 255) # Cyan
        else:
            self.color = (255, 255, 255)
            
    def draw(self, screen):
        # Draw Body
        rect = pygame.Rect(int(self.x), int(self.y), self.size, self.size)
        pygame.draw.rect(screen, self.color, rect)
        
        # Draw Eyes (Direction)
        eye_color = (0,0,0)
        look_dir = 10 if self.vx >= 0 else -10
        pygame.draw.circle(screen, eye_color, (int(self.x + self.size/2 + look_dir), int(self.y + 10)), 5)
