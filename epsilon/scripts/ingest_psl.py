
"""
Ingest Python Standard Library (PSL)

Locates the Python installation and ingests the Standard Library into Epsilon.
Focuses on core libs: os, sys, math, json, collections, etc.
"""

import sys
import os
import torch
sys.path.insert(0, os.getcwd())
from epsilon.system import Epsilon
from epsilon.knowledge.ingestor import PythonIngestor

def main():
    print("="*60)
    print("     EPSILON PSL INGESTION")
    print("="*60)
    
    # 1. Locate Libs
    # Trick: os.__file__ usually points to Lib/os.py
    lib_path = os.path.dirname(os.__file__)
    print(f"Target Library Path: {lib_path}")
    
    # 2. Init System
    epsilon = Epsilon()
    # Try loading existing brain to add to it, else fresh
    epsilon.load(".")
    print(f"Current Knowledge: {len(epsilon.knowledge_store)} concepts.")
    
    # 3. Scan
    ingestor = PythonIngestor()
    
    # Filter for core libraries (avoid huge test suites or site-packages)
    # We'll just scan the root of Lib and a few subdirs
    target_files = []
    
    # Core .py files in Lib root
    for f in os.listdir(lib_path):
        if f.endswith(".py"):
            target_files.append(os.path.join(lib_path, f))
            
    # Important subdirectories
    subdirs = ["json", "collections", "asyncio", "threading", "http", "urllib"]
    for d in subdirs:
        dpath = os.path.join(lib_path, d)
        if os.path.exists(dpath):
            for root, _, files in os.walk(dpath):
                for f in files:
                    if f.endswith(".py"):
                        target_files.append(os.path.join(root, f))
                        
    print(f"Found {len(target_files)} target Python files.")
    
    # 4. Ingest
    print("Ingesting...")
    count = 0
    for path in target_files:
        concepts = ingestor.ingest_file(path)
        for name, desc in concepts:
            if name not in epsilon.knowledge_store:
                # Embed
                torch.manual_seed(sum(ord(c) for c in name))
                z = torch.zeros(64)
                z[:4] = torch.randn(4)
                z = epsilon.manifold.normalize(z)
                epsilon.knowledge_store[name] = z
                count += 1
                
        if count % 100 == 0:
            print(f"  > Learned {count} concepts...")
            
    print(f"\n✓ Ingestion Complete. Added {count} new concepts.")
    print(f"Total Knowledge: {len(epsilon.knowledge_store)}")
    
    # 5. Save
    epsilon.save(".")
    print("✓ Brain Saved.")

if __name__ == "__main__":
    main()
