
"""
Visual Cortex (Vector Vision)

Parses 'dynamic_render.py' to extract geometric intent.
Maps Code -> Visual Semantics -> Manifold Vector.
"""

import ast
import torch

class VisualCortex:
    def __init__(self, system_manifold):
        self.manifold = system_manifold
        
    def analyze_scene(self, code_path="epsilon/env/dynamic_render.py"):
        """
        Reads the code and returns a 'Visual Awareness' vector.
        """
        try:
            with open(code_path, 'r') as f:
                code = f.read()
                
            tree = ast.parse(code)
            
            # Heuristics for visual sentiment
            red_score = 0
            blue_score = 0
            green_score = 0
            chaos_score = 0 # Jitter/Randomness
            
            for node in ast.walk(tree):
                # Check for color tuples
                if isinstance(node, ast.Tuple):
                    # Simple heuristic: look for (R, G, B) tuples
                    if len(node.elts) == 3:
                        vals = []
                        valid = True
                        for elt in node.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                                vals.append(elt.value)
                            else:
                                valid = False
                        
                        if valid:
                            r, g, b = vals
                            # Basic Color Logic
                            if r > 200 and g < 100: red_score += 1
                            if b > 200 and r < 100: blue_score += 1
                            if g > 200: green_score += 1
                            
                # Check for Chaos (Randomness)
                if isinstance(node, ast.Attribute):
                    if node.attr == "randint":
                        chaos_score += 1
                        
            # Construct Visual Vector based on scores
            # Mapping to Atmosphere (Dims 5,6,7)
            # Happy (Green/Bright) = [1, 1, 0]
            # Angry (Red/Chaos) = [1, -1, 0]
            # Sad (Blue) = [-1, -1, 0]
            
            z_visual = torch.zeros(64)
            
            if red_score > blue_score and red_score > green_score:
                z_visual[5:8] = torch.tensor([1.0, -1.0, 0.0]) # Angry
            elif blue_score > red_score:
                z_visual[5:8] = torch.tensor([-1.0, -1.0, 0.0]) # Sad
            elif green_score > red_score:
                z_visual[5:8] = torch.tensor([1.0, 1.0, 0.0]) # Happy
            elif chaos_score > 5:
                # Confused/Anxious
                z_visual[5:8] = torch.tensor([0.0, -1.0, 1.0])
                
            return self.manifold.normalize(z_visual), {"red": red_score, "blue": blue_score, "chaos": chaos_score}
            
        except Exception as e:
            print(f"Visual Cortex Blind: {e}")
            return torch.zeros(64), {}
