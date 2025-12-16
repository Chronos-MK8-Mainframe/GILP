
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import List, Tuple, Dict, Optional

class Rule:
    def __init__(self, rule_id: int, name: str, complexity: float = 1.0):
        self.rule_id = rule_id
        self.name = name
        self.complexity = complexity
        self.prerequisites: List[Dict] = [] # List of dicts: {'id': int, 'type': str, 'weight': float}
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

    def add_rule(self, name: str, complexity: float = 1.0) -> Rule:
        rule = Rule(self.next_id, name, complexity)
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
        
        # Mapping for easy lookup, though ids are ints already
        # We assume ids are contiguous 0..N-1 for PyG
        
        edge_weights = []

        for r_id, rule in self.kb.rules.items():
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
                         edge_weight=torch.empty(0, dtype=torch.float))

        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # Node features? For now, just identity or random. 
        # In LSA-GNN, they use embeddings of text + type. 
        # We will let the dataset/model handle feature initialization if passed, 
        # but here we can return the graph structure.
        
        return Data(edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight, num_nodes=len(self.kb.rules))

