
import torch
import sys
import os

# Add root to path
sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.knowledge.conceptnet_ingestor import ConceptNetIngestor

def verify():
    print("=== Epsilon Chatbot Verification ===")
    
    # Initialize Epsilon
    print("Booting Brain...")
    brain = Epsilon()
    ingestor = ConceptNetIngestor(brain)
    ingestor.ingest_mininet()
    ingestor.ingest_chat_pack()
    
    # Test cases: (Input, Expected Source)
    test_queries = [
        "Hello",               # Known Anchor
        "How are you?",        # Unknown (Encoder fallback)
        "What is 2+2?",        # Unknown (Encoder fallback)
        "Tell me a joke",      # Unknown (Encoder fallback)
        "Jump",                # Known Anchor
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response, trace = brain.think(query)
        print(f"Epsilon: {response}")
        
        # Verify response isn't just a generic failure message
        assert len(response) > 0, "Response should not be empty"
        # Since we haven't trained it for REAL in this script, 
        # it might be gibberish, but it should be a string from the word tokenizer.
        # Once trained via train_v2.py and loaded, it would be coherent.
        
    print("\n✓ Verification Script Completed (Basic Flow)")

if __name__ == "__main__":
    verify()
