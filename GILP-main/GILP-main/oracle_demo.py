import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.data.logic_extractor import LogicExtractor
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch
from gilp_core.llm.reasoning_explainer import get_explainer

def main():
    print("=== GILP v8: Universal Oracle Demo ===")
    
    # 1. Topic Ingestion
    topic = "The intersection of Renewable Energy, Government Subsidies, and Economic Growth."
    print(f"\nTarget Topic: {topic}")
    
    # Mock 'Research' text that would normally come from a web search or RAG pipeline
    research_text = """
    Renewable Energy production is influenced by Government Subsidies.
    Government Subsidies often lead to a Budget Deficit if not managed.
    Renewable Energy adoption stimulates Economic Growth through green jobs.
    Budget Deficit can sometimes slow down Economic Growth in the long term.
    Fossil Fuel consumption contradicts Renewable Energy goals.
    """
    
    # 2. Logic Extraction
    explainer = get_explainer()
    extractor = LogicExtractor(explainer)
    
    print("\n--- Extracting Logic from Research Data ---")
    extracted_rules = extractor.extract_rules(research_text)
    print(f"Extracted {len(extracted_rules)} logical relationships.")
    
    # 3. Knowledge Base Construction
    kb = KnowledgeBase()
    kb.ingest_extracted_rules(extracted_rules)
    graph_data = kb.build_graphs()
    
    if graph_data.num_nodes == 0:
        print("Error: No nodes generated. Logic extraction might have failed.")
        return

    print(f"Knowledge Graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.size(1)} edges.")
    
    # 4. Training
    model = StructureAwareGraphEmbedding(len(kb.rules), manifold_type='lorentz')
    trainer = GILPTrainer(model)
    
    rule_tokens = torch.arange(len(kb.rules)).unsqueeze(1)
    rule_types = torch.zeros(len(kb.rules), dtype=torch.long)
    
    print("\n--- Universal Fossilization (Training Manifold) ---")
    for epoch in range(101):
        metrics = trainer.train_step(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}")

    # 5. Reasoning
    model.eval()
    with torch.no_grad():
        z = model(rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type)
        
    searcher = AHSPSearch(z, manifold_type='lorentz', explainer=explainer, kb=kb)
    
    # Reasoning Question: How do Subsidies affect Growth?
    # Find matching nodes using fuzzy/normalize search
    def find_rule_by_fuzzy_name(kb, name):
        name_norm = name.replace(" ", "").lower()
        for r in kb.rules.values():
            if r.name.replace(" ", "").lower() in name_norm or name_norm in r.name.replace(" ", "").lower():
                return r
        return None

    subs_rule = find_rule_by_fuzzy_name(kb, "Subsidies")
    growth_rule = find_rule_by_fuzzy_name(kb, "Growth")
    
    if not subs_rule or not growth_rule:
        print(f"Error: Could not find relevant nodes for Subsidies/Growth in the graph.")
        # Fallback to first and last node if possible
        if len(kb.rules) >= 2:
            subs_rule = kb.rules[0]
            growth_rule = kb.rules[len(kb.rules)-1]
        else:
            return

    print(f"\nReasoning Query: How does {subs_rule.name} impact {growth_rule.name}?")
    
    path, dist, status = searcher.find_path_with_generative_bridge(subs_rule.rule_id, growth_rule.rule_id, bridge_threshold=1.5)
    
    print("\nLogical Chain of Evidence:")
    rule_names = []
    for node in path:
        if hasattr(node, 'is_virtual'):
            print(f"  [BRIDGE] {node.name}")
            rule_names.append(node.name)
        else:
            name = kb.get_rule(node).name
            print(f"  [KNOWN]  {name}")
            rule_names.append(name)
            
    # Final Narrative
    print("\nSynthesizing Universal Conclusion...")
    explanation = explainer.explain_proof_path(rule_names)
    print("-" * 40)
    print(explanation)
    print("-" * 40)

if __name__ == "__main__":
    main()
