import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.data.logic_extractor import ImageLogicExtractor
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch, TemporalManifold, ManifoldEnsemble, DensityAnalyzer, HypothesisTestingLoop
from gilp_core.llm.reasoning_explainer import get_explainer

def main():
    try:
        _main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()

def _main():
    print("=== GILP v9: Multimodal & Neural Verification Demo ===")
    
    # 1. Image Logic Extraction (Multimodal)
    explainer = get_explainer()
    img_extractor = ImageLogicExtractor(explainer)
    
    # "Image": A flowchart showing that "Investment" leads to "Productivity" but "High Interest Rates" contradict "Investment".
    img_desc = "A flowchart showing that Investment increases Productivity, and High Interest Rates counteract Investment."
    print(f"\nProcessing Image: {img_desc}")
    extracted = img_extractor.extract_from_image(img_desc)
    
    # 2. Knowledge Base with Temporal & Confidence
    kb = KnowledgeBase()
    # Add some initial "Legacy" knowledge (t=0)
    kb.add_rule("BasicArithmetic", timestamp=0.0)
    kb.add_rule("EconomicTheory", timestamp=0.0)
    
    # Ingest image logic with higher timestamp (t=1.0) and confidence (0.9)
    # We'll mock the extracted rules if it fails to ensure demo continues
    if not extracted:
        extracted = [
            {"source": "Investment", "target": "Productivity", "type": "dependency", "confidence": 0.9, "timestamp": 1.0},
            {"source": "HighInterestRates", "target": "Investment", "type": "contradiction", "confidence": 0.8, "timestamp": 1.0}
        ]
    kb.ingest_extracted_rules(extracted)
    graph_data = kb.build_graphs()
    
    # 3. Training with Neural Consistency Loss
    # 3. Training with Neural Consistency Loss
    model = StructureAwareGraphEmbedding(len(kb.rules), hidden_dim=256, manifold_type='lorentz', kb=kb)
    trainer = GILPTrainer(model)
    trainer.explainer = explainer # Attach explainer for NeuralConsistencyLoss
    
    rule_tokens = torch.arange(len(kb.rules)).unsqueeze(1)
    rule_types = torch.zeros(len(kb.rules), dtype=torch.long)
    
    print("\n--- Training with Neural Consistency Loss ---")
    for epoch in range(21): # Short for demo
        try:
            metrics = trainer.train_step(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}, L_Neural={metrics.get('l_neural', 0):.4f}")
        except Exception as e:
            print(f"  [ERROR] Training failed at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            return

    # 4. Temporal Search
    print("\n--- Temporal Search Verification ---")
    searcher = AHSPSearch(model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type), 
                         manifold_type='lorentz', explainer=explainer, kb=kb)
    
    # Search at t=0 (should fail to find Investment)
    t0_searcher = TemporalManifold(searcher, current_time=0.0)
    # Search at t=1.0 (should succeed)
    t1_searcher = TemporalManifold(searcher, current_time=1.0)
    
    inv_rule = kb.get_or_create_rule("Investment")
    prod_rule = kb.get_or_create_rule("Productivity")
    
    print(f"Searching {inv_rule.name} -> {prod_rule.name} at T=0.0...")
    p0, d0, s0 = t0_searcher.find_path(inv_rule.rule_id, prod_rule.rule_id)
    print(f"  Result: {s0}")
    
    print(f"Searching {inv_rule.name} -> {prod_rule.name} at T=1.0...")
    p1, d1, s1 = t1_searcher.find_path(inv_rule.rule_id, prod_rule.rule_id)
    print(f"  Result: {s1} (Path Length: {len(p1) if p1 else 0})")

    # 5. Density Analysis
    print("\n--- Manifold Density Analysis ---")
    analyzer = DensityAnalyzer(searcher.emb_torch, searcher.manifold)
    bottlenecks = analyzer.analyze()
    print(f"Detected {len(bottlenecks)} logical bottlenecks.")

    # 6. Hypothesis Testing Loop
    tester = HypothesisTestingLoop(searcher, kb, explainer)
    tester.run_tests(num_tests=2)

if __name__ == "__main__":
    main()
