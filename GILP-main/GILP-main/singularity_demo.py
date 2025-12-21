import torch
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.data.logic_extractor import LogicExtractor
from gilp_core.data.autonomous_crawler import AutonomousCrawler
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch, FormalCodeExport, VolumeAnalyzer
from gilp_core.llm.reasoning_explainer import get_explainer

def main():
    print("=== GILP v10: The Autonomous Singularity (Singularity Prototype) ===")
    
    # 1. Setup KB and Extractor
    kb = KnowledgeBase()
    explainer = get_explainer()
    extractor = LogicExtractor(explainer)
    
    # 2. Start Autonomous Crawler (Background Knowledge Expansion)
    crawler = AutonomousCrawler(kb, extractor)
    crawler.start(topics=["QuantumComputing", "ArtificialIntelligence"], interval=2)
    
    print("\nWaiting for crawler to expand knowledge base...")
    time.sleep(8) # Let it ingest a few rules
    crawler.stop()
    
    # v10 Fallback if LLM is slow/offline
    if len(kb.rules) == 0:
        print("[v10 System] LLM Ingestion delayed. Injecting synthetic Singularity clusters...")
        kb.ingest_extracted_rules([
            {"source": "Singularity", "target": "SuperIntelligence", "type": "dependency", "confidence": 0.99},
            {"source": "SuperIntelligence", "target": "InfiniteReasoning", "type": "dependency", "confidence": 0.95},
            {"source": "Entropy", "target": "Singularity", "type": "contradiction", "confidence": 0.8}
        ])
    
    graph_data = kb.build_graphs()
    print(f"\nKnowledge Base Size: {len(kb.rules)} rules.")
    
    # 3. 256-Dimension Ultra-Manifold Training
    print("\n--- Training 256-Dimension Lorentz Manifold ---")
    model = StructureAwareGraphEmbedding(len(kb.rules), hidden_dim=256, manifold_type='lorentz', kb=kb)
    trainer = GILPTrainer(model, learning_rate=0.0001)
    
    rule_tokens = torch.arange(len(kb.rules)).unsqueeze(1)
    rule_types = torch.zeros(len(kb.rules), dtype=torch.long)
    
    for epoch in range(11):
        metrics = trainer.train_step(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}")

    # 4. Search & Pruning
    searcher = AHSPSearch(model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type), 
                         manifold_type='lorentz', explainer=explainer, kb=kb)
    
    print("\n--- Geometric Redundancy Pruning ---")
    kb.prune_redundant_rules(searcher, threshold=0.01)

    # 5. Volume/Uncertainty Analysis
    print("\n--- Volume-Based Uncertainty Analysis ---")
    analyzer = VolumeAnalyzer(searcher.manifold)
    uncertainty = analyzer.get_uncertainty(searcher.emb_torch)
    print(f"Global Manifold Spread (Uncertainty Proxy): {uncertainty:.4f}")

    # 6. Formal Code Export (Lean 4)
    print("\n--- Formal Code Export (Lean 4 Skeleton) ---")
    exporter = FormalCodeExport(kb)
    # Pick a random path if possible, or just mock
    if len(kb.rules) > 2:
        path = [0, 1, 2] # Mock path indices
        lean_code = exporter.to_lean(path)
        print(lean_code)

    print("\n=== Singularity Demo Complete ===")

if __name__ == "__main__":
    main()
