
"""
Test Chat Capability

Verifies Epsilon can retrieve conversational responses.
"""

import sys
import os
import torch

sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.knowledge.conceptnet_ingestor import ConceptNetIngestor

def main():
    print("="*60)
    print("     EPSILON CHAT DIAGNOSTIC")
    print("="*60)
    
    sys_instance = Epsilon()
    ingestor = ConceptNetIngestor(sys_instance)
    
    # 1. Ingest Knowledge
    ingestor.ingest_mininet()
    ingestor.ingest_chat_pack()
    
    # 2. Test Hello
    print("\nThinking about 'Hello'...")
    # Note: think() usually takes a 'concept_name' or raw input.
    # We pass "Hello" which matches a known anchor.
    response, trace = sys_instance.think("Hello")
    
    print(f"Response: {response}")
    
    if "Greetings" in response:
        print("✓ CHAT PASSED: Retrieved greeting successfully.")
    else:
        print("✗ CHAT FAILED: Default response only.")
        
    # 3. Test Unknown (Status)
    print("\nThinking about 'Status'...")
    response, trace = sys_instance.think("Status")
    print(f"Response: {response}")
    
    if "operational" in response:
        print("✓ STATUS PASSED.")
    else:
        print("✗ STATUS FAILED.")

if __name__ == "__main__":
    main()
