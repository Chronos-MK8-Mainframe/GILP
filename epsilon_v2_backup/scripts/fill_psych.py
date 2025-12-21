
"""
Fill Psychology Script

Runs the Expression Generator and Ingests the data into Epsilon.
"""

import sys
import os
import torch
import json

sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.scripts import gen_expression
from epsilon.knowledge.psych_ingestor import PsychIngestor

def main():
    print("="*60)
    print("     EPSILON PSYCHOLOGY INGESTION")
    print("="*60)
    
    # 1. Generate Data
    gen_expression.main()
    
    # 2. Load System
    sys_instance = Epsilon()
    sys_instance.load(".")
    print(f"Loaded Brain with {len(sys_instance.knowledge_store)} concepts.")
    
    # 3. Ingest
    ingestor = PsychIngestor(sys_instance)
    ingestor.ingest_expressions("expression_data.json")
    
    # 4. Save
    sys_instance.save(".")
    print("✓ Brain Updated & Saved.")
    
    # 5. Verification
    print("\n--- Verification: Retrieving Angry Code ---")
    payload = sys_instance.knowledge_store.get("Expression:Angry_PAYLOAD")
    if payload:
        print("✓ Success: Found Code Payload for 'Expression:Angry'")
        print(payload[:100] + "...")
    else:
        print("✗ Failed to retrieve payload.")

if __name__ == "__main__":
    main()
