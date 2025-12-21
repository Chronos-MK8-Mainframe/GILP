
print("Start test...", flush=True)
import torch
print("Torch imported", flush=True)
from gilp_core.models.lsa_gnn import StructureAwareGraphEmbedding
print("Model class imported", flush=True)
model = StructureAwareGraphEmbedding(vocab_size=100, hidden_dim=64)
print("Model initialized", flush=True)
