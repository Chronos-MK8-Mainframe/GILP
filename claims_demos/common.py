import torch
import sys
import os

# Add parent directory to path to import gilp_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
from gilp_core.trainer import GILPTrainer
from gilp_core.search.ahsp import AHSPSearch

def get_trained_model(epochs=200):
    """
    Trains the GILP model on the Arithmetic Logic dataset and returns the artifacts.
    """
    print("--- [Setup] Initializing Knowledge Base ---")
    kb = KnowledgeBase()
    # Define concepts/rules (Arithmetic)
    names = [
        "Number", "Zero", "One", "Two", 
        "Addition", "IdentityAddLeft", "IdentityAddRight", "RecursiveAdd",
        "Multiplication", "ZeroMult", "RecursiveMult",
        "ContradictionTest_A", "ContradictionTest_B"
    ]
    for name in names: kb.add_rule(name)
        
    # Topology
    kb.add_dependency(1, 0) # Zero -> Number
    kb.add_dependency(2, 1) # One -> Zero
    kb.add_dependency(4, 0) # Addition requires Number
    kb.add_dependency(5, 4); kb.add_dependency(5, 1) # Identity needs Add + Zero
    kb.add_dependency(8, 4) # Mult requires Add
    kb.add_dependency(10, 8); kb.add_dependency(10, 2) # RecursiveMult needs Mult + One (Successor)
    
    # Disconnected Contradictions
    kb.add_contradiction(11, 12) 

    graph_data = kb.build_graphs()
    
    print("--- [Setup] Training Model (Fossilization) ---")
    model = StructureAwareGraphEmbedding(vocab_size=len(kb.rules), hidden_dim=64)
    trainer = GILPTrainer(model)
    
    rule_tokens = torch.arange(len(kb.rules)).unsqueeze(1)
    rule_types = torch.zeros(len(kb.rules), dtype=torch.long)
    
    for epoch in range(epochs + 1):
        metrics = trainer.train_step(
            rule_tokens, rule_types, graph_data.edge_index, graph_data.edge_type
        )
        if epoch % 50 == 0:
            print(f"    Epoch {epoch}: Loss={metrics['loss']:.4f} (Struct={metrics.get('l_struct', 0):.4f}, Repul={metrics.get('l_repul', 0):.4f}, Fossil={metrics['l_fossil']:.4f})")
            
    model.eval()
    return model, kb, graph_data, trainer, rule_tokens, rule_types
