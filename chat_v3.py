
"""
Epsilon v3 Chat Interface
"Talk to the Machine"

Run this to chat with Epsilon.
"""

import sys
import os
import torch
import time

# Ensure we can find the modules
sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.knowledge.conceptnet_ingestor import ConceptNetIngestor
from epsilon.env.interface import ActionInterface

def main():
    print("="*60)
    print("     EPSILON V3 [Smart Bot Online]")
    print("="*60)
    print("Initializing Brain...")
    
    # 1. Boot System
    epsilon = Epsilon()
    
    # 2. Ingest Knowledge (The "Education")
    print("Ingesting World Model...")
    ingestor = ConceptNetIngestor(epsilon)
    ingestor.ingest_mininet()   # Common Sense
    ingestor.ingest_chat_pack() # Language
    
    # 3. Init Action Interface (Body)
    actor = ActionInterface()
    
    print("\n[SYSTEM READY]")
    print("She is listening. (Type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Epsilon: Goodbye.")
                break
            
            if not user_input.strip():
                continue
                
            # THINK
            start_t = time.time()
            response_text, trace = epsilon.think(user_input)
            end_t = time.time()
            
            # ACT (Check if thoughts triggered action)
            # We look at the FINAL resolved vector in the trace
            final_vec = None
            for label, vec in trace:
                if "Expression" in label or "Reflex" in label:
                    final_vec = vec
            
            if final_vec is not None:
                # Execute Action
                # (This modifies the internal Avatar state)
                actor.execute_vector(final_vec)
                
                # Check for physical reaction in vector (Dims 5-8, 10-12)
                # Just for CLI feedback
                if final_vec[11] > 0.5:
                    print("  *Epsilon Jumps*")
                elif final_vec[10] > 0.5:
                    print("  *Epsilon moves Right*")
            
            # SPEAK
            # Formatting for "Hesitation"
            if "think about that" in response_text:
                print(f"Epsilon: ... {response_text}")
            else:
                print(f"Epsilon: {response_text}")
                
            # DEBUG INFO (Optional)
            # print(f"  [Thought Time: {end_t-start_t:.2f}s]")
            
        except KeyboardInterrupt:
            print("\nEpsilon: Session Terminated.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
