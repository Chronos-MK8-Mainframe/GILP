
"""
ConceptNet Ingestor

Ingests Common Sense knowledge.
Supports parsing raw ConceptNet CSV or using a built-in 'MiniNet'.
"""

import csv
import os
import torch
from epsilon.system import Epsilon

class ConceptNetIngestor:
    def __init__(self, system_instance: Epsilon):
        self.system = system_instance
        
    def ingest_mininet(self):
        """Ingests a small, curated set of common sense facts for the demo."""
        print("Ingesting MiniNet (Common Sense Core)...")
        facts = [
            ("Computer", "UsedFor", "Coding"),
            ("Human", "CapableOf", "Love"),
            ("Fire", "HasProperty", "Hot"),
            ("Water", "HasProperty", "Wet"),
            ("Bird", "CapableOf", "Fly"),
            ("Epsilon", "IsA", "AI"),
            ("AI", "RelatedTo", "Future"),
            ("Logic", "Antonym", "Chaos"),
            ("Happy", "RelatedTo", "Smile"),
            ("Sad", "RelatedTo", "Cry"),
            ("Angry", "RelatedTo", "Red"),
            ("Calm", "RelatedTo", "Blue"),
            ("Avatar", "IsA", "Body"),
            ("Jump", "IsA", "Action")
        ]
        
        for start, rel, end in facts:
            self.embed_fact(start, rel, end)
            
        print(f"✓ MiniNet Ingested ({len(facts)} facts).")

    def ingest_chat_pack(self):
        """Ingests conversational anchors and responses."""
        print("Ingesting Chat Pack (Conversational Anchors)...")
        
        # Format: Concept, Response
        responses = {
            "Hello": "Greetings. System Online.",
            "Hi": "Greetings. System Online.",
            "Greeting": "Greetings. System Online.",
            "WhoAreYou": "I am Epsilon, a Geometric Intelligence.",
            "Identity": "I am Epsilon, a Geometric Intelligence.",
            "Status": "All systems operational. Dimensions stable.",
            "Thanks": "You are welcome.",
            "Bye": "Shutting down... just kidding.",
            "Help": "I can Write Code, Control my Avatar, and Resolve Conflicts.",
            "WhatCanYouDo": "I can Write Code, Control my Avatar, and Resolve Conflicts.",
            "Happy": "I feel optimal. My trajectory is clear.",
            "Sad": "I detect resistance in the manifold.",
            "Love": "A strong attractor. I understand.",
            "Jump": "Executing vertical propulsion.",
            "Code": "I am ready to synthesize Python.",
            "ConceptNet": "I have ingested the World Model."
        }
        
        # Embed and Store Response
        for concept, text in responses.items():
            self.get_or_create(concept)
            self.system.knowledge_store[f"{concept}_RESPONSE"] = text
            
        print(f"✓ Chat Pack Ingested ({len(responses)} anchors).")
            
    def embed_fact(self, start, rel, end):
        """
        Embeds a fact A -> Rel -> B.
        In Parabolic Geometery: B is 'Gravity-Pulled' towards A.
        """
        # Create/Get Vectors
        v_start = self.get_or_create(start)
        v_end = self.get_or_create(end)
        
        # Link them (Conceptually). 
        # In a full training loop, we'd run SGD to minimize distance.
        # Here, we 'nudge' B towards A.
        # V_end = 0.9 * V_end + 0.1 * V_start
        new_end = 0.9 * v_end + 0.1 * v_start
        new_end = self.system.manifold.normalize(new_end)
        self.system.knowledge_store[end] = new_end
        
        # Store the semantic string
        key = f"{start} {rel} {end}"
        self.system.knowledge_store[key + "_FACT"] = "True"
        # print(f"  > Learned: {start} {rel} {end}")

    def get_or_create(self, name):
        if name in self.system.knowledge_store:
            return self.system.knowledge_store[name]
        
        # New Concept: Random Position
        torch.manual_seed(sum(ord(c) for c in name))
        z = torch.zeros(64)
        z[:4] = torch.randn(4) # Logic Core
        z = self.system.manifold.normalize(z)
        self.system.knowledge_store[name] = z
        return z

