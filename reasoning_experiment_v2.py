
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.data.tptp_parser import TPTPParser
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch
from gilp_core.data.prover_interface import MockProverInterface, GraphAugmentedByProof

# Enable accurate logging
sys.stdout.reconfigure(encoding='utf-8')


def extract_predicates(formula):
    import re
    matches = re.findall(r'([a-z][a-zA-Z0-9_]*)\s*\(', formula)
    exclude = {'fof', 'cnf', 'include', 'axiom', 'conjecture', 'lemma'}
    return set([m for m in matches if m not in exclude])

def build_robust_graph(kb):
    print("Building robust dependency graph (Proto-2.5)...")
    symbol_to_rules = defaultdict(list)
    
    for r_id, rule in kb.rules.items():
        preds = extract_predicates(rule.content)
        for p in preds:
            symbol_to_rules[p].append(r_id)
            
    edge_count = 0
    
    # Track added edges to avoid duplicates
    added_edges = set()
    
    for p, ids in symbol_to_rules.items():
        axioms = [i for i in ids if kb.rules[i].rule_type == 'axiom']
        conjectures = [i for i in ids if kb.rules[i].rule_type == 'conjecture']
        others = [i for i in ids if i not in axioms and i not in conjectures]
        
        # 1. Axiom <-> Axiom (Related Premises) - NEW for Proto-2.5
        for i in range(len(axioms)):
            for j in range(i + 1, len(axioms)):
                a1, a2 = axioms[i], axioms[j]
                if (a1, a2) not in added_edges:
                    kb.add_dependency(a1, a2, weight=0.5) # Weak link between axioms
                    kb.add_dependency(a2, a1, weight=0.5)
                    added_edges.add((a1, a2))
                    added_edges.add((a2, a1))
                    edge_count += 2

        # 2. Axiom -> Conjecture (Direct)
        for a in axioms:
            for c in conjectures:
                if (c, a) not in added_edges:
                    kb.add_dependency(c, a, weight=1.0)
                    added_edges.add((c, a))
                    edge_count += 1
            
            # 3. Axiom -> Other -> Conjecture (Chain)
            for o in others:
                if (o, a) not in added_edges:
                    kb.add_dependency(o, a, weight=1.0) # o <- a
                    added_edges.add((o, a))
                    edge_count += 1

        for o in others:
            for c in conjectures:
                if (c, o) not in added_edges:
                    kb.add_dependency(c, o, weight=1.0) # c <- o
                    added_edges.add((c, o))
                    edge_count += 1

    print(f"Added {edge_count} heuristic edges (including A<->A).")

def run_experiment():
    print("=== GILP Proto-2.5 Experiment: Path Consistency ===")
    
    # 1. Setup
    tptp_root = r"C:\Users\rupa9\Videos\GILP\TPTP-v9.2.1"
    problem_file = "Problems/MED/MED001+1.p" 
    
    kb = KnowledgeBase()
    parser = TPTPParser(tptp_root)
    print(f"Parsing {problem_file}...")
    parser.parse_file(problem_file, kb)
    
    # Use Robust Graph Builder
    build_robust_graph(kb) 
    
    # Build initial graph
    graph_data = kb.build_graphs()
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.size(1)} edges.")
    
    # Vocab
    vocab = parser.build_vocab(kb, min_freq=1)
    rule_tokens = parser.encode_rules(kb, vocab)
    
    # Types
    type_map = {'axiom': 0, 'conjecture': 1, 'lemma': 2, 'definition': 3}
    rule_types = torch.tensor([type_map.get(r.rule_type, 0) for r in kb.rules.values()], dtype=torch.long)
    
    # 2. Training with Proto-3 (Hyperbolic Contrastive)
    print("\n--- Training (Hyperbolic Logic) ---")
    
    model = StructureAwareGraphEmbedding(vocab_size=len(vocab), hidden_dim=64)
    trainer = GILPTrainer(model, learning_rate=0.005)
    
    losses = defaultdict(list)
    
    for epoch in range(101):
        metrics = trainer.train_step(rule_tokens, rule_types, 
                                   graph_data.edge_index, 
                                   graph_data.edge_type, 
                                   graph_data.edge_weight)
        
        for k, v in metrics.items():
            losses[k].append(v)
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={metrics['loss']:.4f} (Hyp={metrics['l_hyp']:.4f})")
            
    # 3. Evaluation Setup
    model.eval()
    
    # Identify Task
    conjecture_ids = [rid for rid, r in kb.rules.items() if r.rule_type == 'conjecture']
    if not conjecture_ids:
        print("No conjectures found.")
        return
    target_id = conjecture_ids[0]
    
    axiom_ids = [rid for rid, r in kb.rules.items() if r.rule_type == 'axiom']
    start_id = axiom_ids[0] # Pick first axiom
    
    print(f"\nTask: Path from {kb.rules[start_id].name} -> {kb.rules[target_id].name}")
    
    # 4. Critical Experiment: Fossilization Test
    results = {}
    
    # Enable Potentials for both modes to guide search
    # But in OFF mode it relies on potential learned from Tokens (no graph input)
    
    for mode in ["Heuristics ON", "Heuristics OFF"]:
        print(f"\nrunning Inference Mode: {mode}")
        
        with torch.no_grad():
            if mode == "Heuristics ON":
                # Use full graph
                emb = model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type, graph_data.edge_weight)
            else:
                # Use empty graph
                empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=rule_tokens.device)
                empty_edge_type = torch.empty(0, dtype=torch.long, device=rule_tokens.device)
                emb = model(rule_tokens, rule_types, empty_edge_index, empty_edge_type)
        
        searcher = AHSPSearch(emb)
        
        # Run Hyperbolic Greedy Search (Proto-3)
        path, dist_traveled, reason = searcher.find_path_hyperbolic_greedy(start_id, target_id, 
                                                                           step_radius=0.5, 
                                                                           max_steps=50)
        
        if path:
            print(f"SUCCESS. Path len: {len(path)}. Hyp Dist: {dist_traveled:.2f}")
            path_names = [kb.rules[i].name for i in path]
            print(" -> ".join(path_names))
            results[mode] = {"success": True, "dist": dist_traveled, "len": len(path), "reason": reason}
        else:
            print(f"FAILURE. Dist: {dist_traveled:.2f}. Reason: {reason}")
            results[mode] = {"success": False, "dist": dist_traveled, "len": 0, "reason": reason}
            
    # 5. Metrics & Histograms
    print("\n--- Metrics Summary ---")
    
    # Compare modes
    print("\nFOSSILIZATION RESULT:")
    on_res = results["Heuristics ON"]
    off_res = results["Heuristics OFF"]
    
    print(f"ON:  Success={on_res['success']}, Dist={on_res['dist']}, Reason={on_res['reason']}")
    print(f"OFF: Success={off_res['success']}, Dist={off_res['dist']}, Reason={off_res['reason']}")
    
    if off_res['success']:
        print(">> VERIFIED: Geometry/Potential guided inference without graph edges!")
    else:
        print(">> FAILED: Model relies on graph edges.")


    # Distance Histogram (Hyperbolic)
    from gilp_core.geometry.hyperbolic import PoincareManifold
    manifold = PoincareManifold()
    
    with torch.no_grad():
        final_emb = model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type, graph_data.edge_weight)
        final_emb = final_emb.cpu()
    
    # Calculate pairwise distances of a subset
    n = final_emb.size(0)
    dists = []
    import random
    indices = list(range(n))
    # Sample 1000 pairs
    for _ in range(1000):
        i, j = random.sample(indices, 2)
        d = manifold.dist(final_emb[i], final_emb[j]).item()
        dists.append(d)
        
    plt.figure()
    plt.hist(dists, bins=30, alpha=0.7)
    plt.title("Pairwise Hyperbolic Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig("distance_histogram.png")
    print("Saved distance_histogram.png")

if __name__ == "__main__":
    run_experiment()
