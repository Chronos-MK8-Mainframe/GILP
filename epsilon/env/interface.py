
"""
Action Interface (Brain -> Body)

Maps Parabolic Vectors to Avatar Commands.
"""

import torch
import epsilon.env.world as world

class ActionInterface:
    def __init__(self):
        pass
        
    def execute_vector(self, vec: torch.Tensor):
        """
        Translates a 64-dim thought vector into action.
        Mapping:
        - Dims 5,6,7 (Atmosphere) -> Emotion (Color)
        - Dim 10 -> X Velocity (>0.5 Right, <-0.5 Left)
        - Dim 11 -> Jump (>0.5 Jump)
        """
        avatar = world.avatar_instance
        
        # 1. Motion
        val_x = vec[10].item()
        val_jump = vec[11].item()
        
        if val_x > 0.5:
            avatar.move(1)
        elif val_x < -0.5:
            avatar.move(-1)
            
        if val_jump > 0.5:
            avatar.jump()
            
        # 2. Emotion (Using our standard Atmosphere mapping)
        # 1.0, -1.0, 0.0 = Angry
        # 1.0, 1.0, 0.0 = Happy
        atm = vec[5:8]
        
        # Simple heuristic matching
        if atm[0] > 0.5 and atm[1] < -0.5:
            avatar.set_emotion("Angry")
        elif atm[0] > 0.5 and atm[1] > 0.5:
            avatar.set_emotion("Happy")
        elif atm[0] < -0.5 and atm[1] < -0.5:
            avatar.set_emotion("Sad")
        elif atm[0] < -0.5 and atm[1] > 0.5:
            avatar.set_emotion("Calm")
            
        # print(f"Action Executed: X={val_x:.2f} J={val_jump:.2f}")

