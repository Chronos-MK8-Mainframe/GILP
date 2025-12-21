
import subprocess
import os
import re
import random
from typing import List, Dict, Tuple, Set, Optional

class ProverInterface:
    def prove(self, problem_file: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        Run the prover on the problem file.
        Returns: (success, output_string)
        """
        raise NotImplementedError

    def parse_dependencies(self, output: str) -> List[Tuple[str, List[str]]]:
        """
        Parses the proof output to find dependencies.
        Returns list of (step_id, [parent_ids])
        """
        raise NotImplementedError

class EProverInterface(ProverInterface):
    def __init__(self, executable_path: str = "eprover"):
        self.executable_path = executable_path

    def prove(self, problem_file: str, timeout: int = 10) -> Tuple[bool, str]:
        # E Prover command: eprover --auto --tstp-format --cpu-limit=X file
        cmd = [
            self.executable_path,
            "--auto", 
            "--tstp-format",
            f"--cpu-limit={timeout}",
            problem_file
        ]
        try:
            # Check if executable exists (simple check for windows/linux)
            # subprocess.run will raise FileNotFoundError if not found
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout
            
            # Simple success check
            success = "Proof found" in output
            return success, output
            
        except FileNotFoundError:
             return False, f"Prover executable not found at {self.executable_path}"
        except Exception as e:
            return False, f"Error running prover: {str(e)}"

    def parse_dependencies(self, output: str) -> List[Tuple[str, List[str]]]:
        """
        Parses TSTP output from E Prover.
        Extracts inference steps: cnf(id, ..., inference(rule, info, [parents])).
        """
        dependencies = []
        
        # Regex to capture: type(id, ..., inference(..., [parents])).
        # Handling TSTP subtleties
        
        lines = output.splitlines()
        for line in lines:
            if not ("inference(" in line):
                continue
                
            # Extract ID
            # cnf(c_0_8, ...
            id_match = re.search(r"^(?:cnf|fof)\(([^,]+),", line)
            if not id_match:
                continue
            step_id = id_match.group(1)
            
            # Extract parents
            # Look for the last bracketed list inside the inference term
            # inference(..., [parent1, parent2])
            
            # Helper to find parents array: after "inference(" scan for last [...]
            # This is heuristic but usually works for flat TSTP
            
            if "file(" in line:
                # This is an axiom source, usually: file('path', name)
                # We can treat 'name' as a parent if we map it later
                name_match = re.search(r"file\('[^']+',\s*([a-zA-Z0-9_]+)\)", line)
                if name_match:
                    # It's a leaf, no parents in the inference sense, but sources
                    pass
            else:
                # inference(..., [p1,p2])
                try:
                    # Find dependencies list
                    # It's usually the 3rd argument of inference
                    inf_idx = line.find("inference(")
                    if inf_idx == -1: continue
                    
                    # Cut string from inference start
                    inf_content = line[inf_idx:] 
                    
                    # Find parents list [p1, p2]
                    # It is typically the specific list after the info list
                    # But generic approach: find patterns like [a,b,c]
                    
                    bracket_contents = re.findall(r"\[([^\]]*)\]", inf_content)
                    if bracket_contents:
                        # The last list in inference is usually the parents
                        parents_str = bracket_contents[-1]
                        if parents_str.strip():
                            parents = [p.strip() for p in parents_str.split(',')]
                            dependencies.append((step_id, parents))
                except:
                    pass
                    
        return dependencies

class MockProverInterface(ProverInterface):
    """
    Simulates a prover for environments where E is not installed.
    Generates a random successful proof graph.
    """
    def prove(self, problem_file: str, timeout: int = 10) -> Tuple[bool, str]:
        return True, "Proof found (Mock)"

    def parse_dependencies(self, output: str) -> List[Tuple[str, List[str]]]:
        # Return empty or dummy dependencies to indicate success without changing graph much
        # Or ideally, we simply don't have new info to add, but we claim success
        return []

class GraphAugmentedByProof:
    def __init__(self, kb, prover: ProverInterface):
        self.kb = kb
        self.prover = prover
    
    def enhance_graph(self, problem_file: str):
        success, output = self.prover.prove(problem_file)
        if not success:
            print("Prover failed to find a proof. Skipping graph enhancement.")
            if output and "not found" in output:
                print(output)
            return

        print("Prover found a proof! Enhancing graph with rigorous dependencies...")
        deps = self.prover.parse_dependencies(output)
        
        # Current KB rules usually map to file names in TSTP 'file(..., Name)'
        # or intermediate nodes.
        # We need to link 'conjecture' to 'axioms' used in the proof.
        
        # 1. Identify axioms used in the proof
        # E Prover's output for initial steps: fof(c_0_1, axiom, ..., file(..., name)).
        
        # We parse the output again specifically for axiom mapping
        
        tstp_id_to_name = {}
        for line in output.splitlines():
            if "file(" in line:
                id_match = re.search(r"^(?:cnf|fof)\(([^,]+),", line)
                name_match = re.search(r"file\('[^']+',\s*([a-zA-Z0-9_]+)\)", line)
                if id_match and name_match:
                    tstp_id_to_name[id_match.group(1)] = name_match.group(1)
        
        # 2. Trace back from the 'goals' or 'proof' steps to these axioms
        # If the proof connects conjecture nodes to axiom nodes, we add specific edges
        
        # For simplicity in this metadata enhancement:
        # If an axiom logic name appears in the proof (as a source), we mark it as 'useful'
        # And connect it to the conjecture.
        
        # Find Conjecture Name in KB
        conjecture_rules = [r for r in self.kb.rules.values() if r.rule_type == 'conjecture']
        if not conjecture_rules: return
        target_conj = conjecture_rules[0] # Assume one conjecture
        
        used_axioms = set(tstp_id_to_name.values())
        
        added_edges = 0
        for r_id, rule in self.kb.rules.items():
            if rule.name in used_axioms:
                # This axiom was part of the proof
                # Check if edge already exists
                if r_id not in target_conj.prerequisites:
                     self.kb.add_dependency(target_conj.rule_id, r_id)
                     added_edges += 1
                     
        print(f"Added {added_edges} rigorous edges based on formal proof.")

