
"""
Psychology Ingestor

Ingests Emotional Mappings and Expression Code.
Maps 'Atmosphere' vectors (Feelings) to 'Logic Core' vectors (Code Actions).
"""

import json
import torch
import os
from epsilon.system import Epsilon

class PsychIngestor:
    def __init__(self, system: Epsilon):
        self.system = system
        
        # Define Emotion Anchors in Atmosphere (Dims 5,6,7 for simplicity in demo)
        # 64 dim total. Dims 0-4 are Logic. Dims 5+ are Atmosphere.
        self.emotion_anchors = {
            "Happy": [1.0, 1.0, 0.0],
            "Sad": [-1.0, -1.0, 0.0],
            "Angry": [1.0, -1.0, 0.0],
            "Calm": [-1.0, 1.0, 0.0]
        }
        
    def ingest_expressions(self, json_path: str):
        print(f"Ingesting Expression Data from {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            concept_name = item["concept"] # e.g. "Expression:Happy"
            emotion = item["emotion_vector_hint"]
            code = item["code_content"]
            
            # 1. Create the Logic Node (The Code)
            # This lives in the Core (0-4) because it's a hard fact/action
            torch.manual_seed(sum(ord(c) for c in concept_name))
            z_code = torch.zeros(64)
            z_code[:4] = torch.randn(4) # Random logic position
            
            # 2. Encode the Emotion (The Atmosphere)
            # We want this node to be "accessible" from the Emotion Region
            if emotion in self.emotion_anchors:
                anchor = torch.tensor(self.emotion_anchors[emotion])
                # Set dimensions 5,6,7 to this anchor
                z_code[5:8] = anchor
                
            # Normalize
            z_code = self.system.manifold.normalize(z_code)
            
            # Store
            self.system.knowledge_store[concept_name] = z_code
            
            # Store the actual code payload (Mocking a 'Value' store)
            # In a real system, nodes point to data blocks.
            # Here we just attach it to the key string for retrieval in the demo
            self.system.knowledge_store[f"{concept_name}_PAYLOAD"] = code
            
            print(f"  > Learned Expression: {concept_name} linked to {emotion}")
            
        print("✓ Expression Ingestion Complete.")

