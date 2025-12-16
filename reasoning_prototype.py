
import torch
import os
import re
from collections import defaultdict

# Force unbuffered for debugging
import sys
sys.stdout.reconfigure(encoding='utf-8')

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.data.tptp_parser import TPTPParser
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch
from gilp_core.visualization import EmbeddingVisualizer
from gilp_core.data.prover_interface import EProverInterface, MockProverInterface, GraphAugmentedByProof

def extract_predicates(formula):
    matches = re.findall(r'([a-z][a-zA-Z0-9_]*)\s*\(', formula)
    exclude = {'fof', 'cnf', 'include', 'axiom', 'conjecture', 'lemma'}
    return set([m for m in matches if m not in exclude])

def build_symbol_graph(kb):
    print("Building dependency graph based on symbol overlap...")
    symbol_to_rules = defaultdict(list)
    
    for r_id, rule in kb.rules.items():
        preds = extract_predicates(rule.content)
        for p in preds:
            symbol_to_rules[p].append(r_id)
            
    edge_count = 0
    for p, ids in symbol_to_rules.items():
        axioms = [i for i in ids if kb.rules[i].rule_type == 'axiom']
        conjectures = [i for i in ids if kb.rules[i].rule_type == 'conjecture']
        others = [i for i in ids if i not in axioms and i not in conjectures]
        
        for a in axioms:
            for c in conjectures:
                kb.add_dependency(c, a)
                edge_count += 1
            for o in others:
                kb.add_dependency(o, a)
                edge_count += 1
                
        for o in others:
            for c in conjectures:
                kb.add_dependency(c, o)
                edge_count += 1

    print(f"Added {edge_count} heuristic edges.")

def main():
    tptp_root = r"C:\Users\rupa9\Videos\GILP\TPTP-v9.2.1"
    problem_file = "Problems/MED/MED001+1.p" # Example problem
    
    kb = KnowledgeBase()
    parser = TPTPParser(tptp_root)
    
    # 2. Parse
    print(f"Parsing {problem_file}...")
    parser.parse_file(problem_file, kb)
    print(f"Parsed {len(kb.rules)} rules.")
    
    # 3. Build Graph
    build_symbol_graph(kb)

    # --- ADVANCED GRAPH CONSTRUCTION ---
    # Try to use formal prover to enhance graph
    # We use MockProver by default as E Prover is likely not installed on Windows
    # To use E Prover, install it and change to: prover = EProverInterface("path/to/eprover")
    prover_path = "eprover" 
    # Check if we should use Mock or Real
    use_mock = True # Default for safety in this environment
    
    if use_mock:
        prover = MockProverInterface()
    else:
        prover = EProverInterface(prover_path)

    graph_enhancer = GraphAugmentedByProof(kb, prover)
    # Full path to problem file for the prover
    full_problem_path = os.path.join(tptp_root, problem_file)
    graph_enhancer.enhance_graph(full_problem_path)
    # -----------------------------------

    graph_data = kb.build_graphs()
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.size(1)} edges.")
    
    # NEW: Build Vocabulary and Tokenize for Transformer
    vocab = parser.build_vocab(kb, min_freq=1)
    print(f"Vocab size: {len(vocab)}")
    rule_tokens = parser.encode_rules(kb, vocab) # [N, MaxLen]
    
    # 4. Train
    print("\n--- Training LSA-GNN (Transformer) on TPTP Data ---")
    model = StructureAwareGraphEmbedding(vocab_size=len(vocab), hidden_dim=64)
    trainer = GILPTrainer(model, learning_rate=0.005)
    
    type_map = {'axiom': 0, 'conjecture': 1, 'lemma': 2, 'definition': 3}
    rule_types = torch.tensor([type_map.get(r.rule_type, 0) for r in kb.rules.values()], dtype=torch.long)
    
    for epoch in range(61):
        loss, l_geo, l_sep = trainer.train_step(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f} (Geo={l_geo:.4f})")
            
    # 5. Reasoning / Query
    model.eval()
    with torch.no_grad():
        embeddings = model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        
    searcher = AHSPSearch(embeddings)
    
    conjecture_ids = [rid for rid, r in kb.rules.items() if r.rule_type == 'conjecture']
    if not conjecture_ids:
        print("No conjectures found.")
        return

    target_id = conjecture_ids[0]
    target_rule = kb.rules[target_id]
    print(f"Target Conjecture: {target_rule.name}")
    
    indices, dists = searcher.find_nearest(embeddings[target_id], k=6)
    
    print("\nNearest Logical Premises:")
    for i, idx in enumerate(indices[0]):
        if idx == target_id: continue
        r = kb.rules[idx.item()]
        d = dists[0][i]
        print(f"[{i}] {r.name} ({r.rule_type}): Dist={d:.4f}")
        
    axiom_ids = [rid for rid, r in kb.rules.items() if r.rule_type == 'axiom']
    if axiom_ids:
        start_id = axiom_ids[0]
        path = searcher.find_path(start_id, target_id)
        print(f"\nInference Path from {kb.rules[start_id].name} -> {target_rule.name}:")
        print(" -> ".join([kb.rules[i].name for i in path]))

    # --- VISUALIZATION ---
    print("\nLaunching 3D Visualizer...")
    # Get labels for visualization
    labels = [r.name for r in kb.rules.values()]
    # Re-use rule_types from training block
    
    # We need to map types back to ints for the visualizer if we want colors
    # The 'rule_types' tensor is already a tensor of ints.
    # We can pass that directly as a list.
    vis_types = rule_types.cpu().tolist()
    
    vis = EmbeddingVisualizer(embeddings, labels=labels, types=vis_types)
    # Save to file to ensure it works in non-interactive environments too
    vis.plot_3d(save_path="embedding_space.png", block=False)
    print("Visualization saved to embedding_space.png")
    # ---------------------

if __name__ == "__main__":
    main()
