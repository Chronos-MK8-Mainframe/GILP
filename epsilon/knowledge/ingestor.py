
"""
Epsilon Knowledge Ingestor

Parses structured data (Code, Logic) into Proto-Concepts.
These concepts are then embedded into the Parabolic Manifold.
"""

import ast
import os
import glob
from typing import List, Dict, Tuple

class PythonIngestor:
    """
    Parses Python source code into Logical Concept Nodes.
    """
    def __init__(self):
        pass
        
    def ingest_file(self, filepath: str) -> List[Tuple[str, str]]:
        """
        Parse a .py file and return (Name, Docstring/Type) pairs.
        """
        concepts = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    concepts.append((f"class {node.name}", ast.get_docstring(node) or "Class Definition"))
                    # Recursively get methods? For now, flat structure.
                elif isinstance(node, ast.FunctionDef):
                    concepts.append((f"def {node.name}", ast.get_docstring(node) or "Function Definition"))
                    
        except Exception as e:
            # print(f"Skipping {filepath}: {e}")
            pass
            
        return concepts

    def scan_directory(self, root_dir: str) -> List[Tuple[str, str]]:
        """Recursively ingest all .py files."""
        all_concepts = []
        files = glob.glob(os.path.join(root_dir, "**", "*.py"), recursive=True)
        print(f"Scanning {len(files)} Python files in {root_dir}...")
        
        for f in files:
            all_concepts.extend(self.ingest_file(f))
            
        print(f"  > Extracted {len(all_concepts)} Python concepts.")
        return all_concepts

class TPTPIngestor:
    """
    Parses TPTP (Thousands of Problems for Theorem Provers) Axioms.
    Placeholder for Phase 2 of Ingestion.
    """
    def ingest_file(self, filepath: str):
        # TODO: Implement TPTP regex parser
        pass
