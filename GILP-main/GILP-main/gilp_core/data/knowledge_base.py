
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import List, Tuple, Dict, Optional

class Rule:
    def __init__(self, rule_id: int, name: str, complexity: float = 1.0):
        self.rule_id = rule_id
        self.name = name
        self.complexity = complexity
        self.weight = 1.0
        self.timestamp = 0.0 # v9: Temporal support
        self.prerequisites: List[Dict] = [] # List of dicts: {'id': int, 'type': str, 'weight': float, 'ts': float}
        self.contradictions: List[int] = [] # List of rule_ids that contradict this rule
        self.compositions: List[int] = [] # List of rule_ids that this rule is composed of
        
        # GILP / TPTP extensions
        self.content: str = ""     # Raw formula string
        self.rule_type: str = ""   # e.g., 'axiom', 'conjecture'

    def add_prerequisite(self, rule_id: int, edge_type: str = "dependency", weight: float = 1.0):
        self.prerequisites.append({'id': rule_id, 'type': edge_type, 'weight': weight})

    def add_contradiction(self, rule_id: int):
        self.contradictions.append(rule_id)
    
    def add_composition(self, rule_id: int):
        self.compositions.append(rule_id)

    def __repr__(self):
        return f"Rule(id={self.rule_id}, name={self.name})"

class KnowledgeBase:
    def __init__(self):
        self.rules: Dict[int, Rule] = {}
        self.next_id = 0
        self.graph_builder = GraphBuilder(self)

    def add_rule(self, name: str, complexity: float = 1.0, timestamp: float = 0.0) -> Rule:
        rule = Rule(self.next_id, name, complexity)
        rule.timestamp = timestamp
        self.rules[self.next_id] = rule
        self.next_id += 1
        return rule
    
    def get_rule(self, rule_id: int) -> Optional[Rule]:
        return self.rules.get(rule_id)

    def add_dependency(self, target_id: int, source_id: int, edge_type: str = "dependency", weight: float = 1.0):
        """Source is a prerequisite for Target"""
        if target_id in self.rules and source_id in self.rules:
            self.rules[target_id].add_prerequisite(source_id, edge_type, weight)

    def add_contradiction(self, id1: int, id2: int):
        """id1 and id2 differ"""
        if id1 in self.rules and id2 in self.rules:
            self.rules[id1].add_contradiction(id2)
            self.rules[id2].add_contradiction(id1)

    def get_or_create_rule(self, name: str) -> Rule:
        """Finds rule with name or creates it."""
        for r in self.rules.values():
            if r.name.lower() == name.lower():
                return r
        return self.add_rule(name)

    def ingest_extracted_rules(self, rules: List[Dict]):
        """Ingests rules from LogicExtractor format."""
        for r_data in rules:
            if not isinstance(r_data, dict):
                continue
            src_name = r_data.get("source")
            dst_name = r_data.get("target")
            r_type = r_data.get("type", "dependency")
            weight = r_data.get("confidence", 1.0)
            ts = r_data.get("timestamp", 0.0)
            
            if src_name and dst_name:
                src_rule = self.get_or_create_rule(src_name)
                dst_rule = self.get_or_create_rule(dst_name)
                
                if r_type == "dependency":
                    self.add_dependency(dst_rule.rule_id, src_rule.rule_id, weight=weight)
                elif r_type == "contradiction":
                    self.add_contradiction(dst_rule.rule_id, src_rule.rule_id)

    def prune_redundant_rules(self, searcher, threshold: float = 0.1):
        """
        v10: Geometric Pruning.
        Removes rules that are geometrically too close to others (logical redundancy).
        """
        embeddings = searcher.emb_torch
        num_rules = len(self.rules)
        to_remove = set()
        
        for i in range(num_rules):
            if i in to_remove: continue
            for j in range(i+1, num_rules):
                if j in to_remove: continue
                
                dist = searcher.manifold.dist(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
                if dist < threshold:
                     # Keep the one with higher ID (newer) or lower ID (legacy)? 
                     # Let's keep legacy (lower ID).
                     to_remove.add(j)
        
        for r_id in to_remove:
            print(f"[v10 Pruner] Pruning redundant rule: {self.rules[r_id].name}")
            del self.rules[r_id]
            
        print(f"[v10 Pruner] Pruned {len(to_remove)} redundant rules.")

    def merge_with_other_kb(self, other_kb):
        """
        v10: Collaborative Knowledge Merging.
        Merges rules from another KB into this one, maintaining IDs.
        """
        print(f"[v10 Collaborator] Merging with external KB...")
        for other_rule in other_kb.rules:
            # Simple merge: if name exists, add dependencies
            existing = self.get_or_create_rule(other_rule.name)
            for dep_id in other_rule.dependencies:
                dep_name = other_kb.get_rule(dep_id).name
                self.add_dependency(existing.name, dep_name)
        print(f"[v10 Collaborator] Merged {len(other_kb.rules)} rules.")

    def build_graphs(self):
        return self.graph_builder.build()

class GraphBuilder:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def build(self):
        """
        Constructs the PyG Data object containing edge_index and edge_type.
        Edge types: 0: Prerequisite, 1: Contradiction, 2: Composition
        """
        edge_indices = [[], []]
        edge_types = []
        edge_weights = []
        node_timestamps = []

        for r_id, rule in self.kb.rules.items():
            node_timestamps.append(rule.timestamp)
            # Prerequisites: source -> target
            for pre in rule.prerequisites:
                edge_indices[0].append(pre['id'])
                edge_indices[1].append(r_id)
                edge_types.append(0) # Prerequisite
                edge_weights.append(pre['weight'])

            # Contradictions: bidirectional
            for contra_id in rule.contradictions:
                # Add one direction, loop will catch the other
                edge_indices[0].append(r_id)
                edge_indices[1].append(contra_id)
                edge_types.append(1) # Contradiction
                edge_weights.append(1.0) # Standard weight

            # Composition: sub -> super
            for comp_id in rule.compositions:
                edge_indices[0].append(comp_id)
                edge_indices[1].append(r_id)
                edge_types.append(2) # Composition
                edge_weights.append(1.0) # Standard weight

        if not edge_types:
             return Data(x=torch.zeros((len(self.kb.rules), 1)), 
                         edge_index=torch.empty((2, 0), dtype=torch.long), 
                         edge_type=torch.empty(0, dtype=torch.long),
                         edge_weight=torch.empty(0, dtype=torch.float),
                         ts=torch.tensor(node_timestamps, dtype=torch.float))

        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        ts = torch.tensor(node_timestamps, dtype=torch.float)
        
        return Data(edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight, ts=ts, num_nodes=len(self.kb.rules))

