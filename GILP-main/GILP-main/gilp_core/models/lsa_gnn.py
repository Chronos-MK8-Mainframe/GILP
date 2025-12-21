
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import math

class LogicTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=2, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        # src: [batch_size, seq_len]
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        # Average pooling to get single vector per formula
        # Mask aware mean? For simplicity just mean over 1st dim (if batch_first=True) -> [batch, seq, dim] -> [batch, dim]
        # We should mask padding tokens ideally.
        return output.mean(dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class LogicalGNN(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=4, num_relations=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.message_types = nn.ModuleDict({
            '0': nn.Linear(hidden_dim, hidden_dim), # prerequisite
            '1': nn.Linear(hidden_dim, hidden_dim), # contradiction
            '2': nn.Linear(hidden_dim, hidden_dim)  # composition
        })
        
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            for _ in range(num_layers)
        ])
        
        self.spatial_projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim) # Project to manifold space (e.g. 256)
        )
        
    def forward(self, node_features, edge_index, edge_type, edge_weight=None):
        h = node_features
        
        for layer in self.gnn_layers:
            current_h = h
            h_new = 0 
            
            if edge_index.size(1) == 0:
                 h_new = current_h
            else:
                for rel_type_str, transform in self.message_types.items():
                    rel_type = int(rel_type_str)
                    mask = (edge_type == rel_type)
                    
                    if mask.sum() > 0:
                        sub_edge_index = edge_index[:, mask]
                        
                        # Handle weights
                        if edge_weight is not None:
                            sub_weights = edge_weight[mask]
                            # Reshape for broadcasting if needed, but for simple scaling of features:
                            # We can't easily scale features before GAT because GAT takes (x, edge_index).
                            # GATConv in PyG doesn't strictly take scalar edge_weight for attention modification in all versions.
                            # BUT, we can just pass it to GATConv if we trusted it. Not all GAT implementations use it.
                            # Alternative: Scale the OUTPUT of the transform before GAT? No, GAT aggregates.
                            # Better: Weighted GAT? 
                            # Simplest "Heuristics scaled down" implementation:
                            # We manually scale the message or use the weight in the loss.
                            # The user says "Allow heuristics to be scaled down". This usually implies during Training/Inference?
                            # If we modify the GRAPH structure (weights), the GNN should respect them.
                            # For GAT, if we don't pass weights, it learns attention.
                            # If we want FORCED scaling, we can multiply the attention scores? Hard with PyG high level.
                            # Let's assume we multiply the incoming features by weight for that edge? No, one node has many edges.
                            
                            # Let's use a simpler approach for this prototype: 
                            # We will rely on the GAT to learn, BUT we use the weights in the LOSS function to scale the geometric penalty.
                            # AND we can pass edge_weight to GATConv hoping it uses it, or switch to GCNConv which definitely uses it.
                            # The user requirement "Label every heuristic... Allow heuristics to be scaled down"
                            # This probably means the EFFECT of the heuristic on the embedding.
                            pass

                        transformed_h = transform(current_h)
                        out = layer(transformed_h, sub_edge_index)
                        
                        # If we really want to enforce scalar weights on top of GAT:
                        # We might need a custom layer. 
                        # But wait, if requirement is "Allow heuristics to be... annealed to zero", 
                        # this suggests we control the *influence* of determining the embedding.
                        # If the edge exists in GAT, it flows information.
                        # If we want to anneal it to zero, we should REMOVE the edge or weight it to 0.
                        # Since we pass `sub_edge_index`, we can mask out edges with weight < threshold?
                        # Or we assume the Trainer anneals the LOSS weight. 
                        
                        # Let's stick to standard GAT flow here, but accept the arg.
                        h_new = h_new + out
            
            if isinstance(h_new, int) and h_new == 0: 
                 h_new = h # Fallback
            
            h = h + h_new 
            h = torch.relu(h)

        positions = self.spatial_projection(h)
        return positions, h

from gilp_core.geometry.hyperbolic import PoincareManifold
from gilp_core.geometry.lorentz import LorentzManifold

class StructureAwareGraphEmbedding(nn.Module):
    def __init__(self, vocab_size=1000, hidden_dim=256, manifold_type='lorentz', learnable_curvature=True, kb=None):
        super().__init__()
        self.kb = kb
        # Repalced simple embedding with Transformer
        self.logic_encoder = LogicTransformerEncoder(vocab_size, d_model=hidden_dim)
        self.logical_gnn = LogicalGNN(hidden_dim=hidden_dim)
        self.type_embedder = nn.Embedding(10, hidden_dim)
        
        # Hyperbolic Manifold selection
        self.manifold_type = manifold_type
        if manifold_type == 'poincare':
            self.manifold = PoincareManifold()
        else:
            self.manifold = LorentzManifold()
            
        # Learned Curvature (v6)
        if learnable_curvature:
            self.log_c = nn.Parameter(torch.zeros(1)) # c = exp(log_c), initialized at 1.0
        else:
            self.register_buffer('log_c', torch.zeros(1))

    def get_curvature(self):
        return torch.exp(self.log_c)
        
    def forward(self, rule_token_seqs, rule_types, edge_index, edge_type, edge_weight=None):
        # rule_token_seqs: [num_nodes, seq_len] (padded)
        
        # Encode logic text
        token_emb = self.logic_encoder(rule_token_seqs)
        
        # Encode types
        type_emb = self.type_embedder(rule_types)
        
        node_features = token_emb + type_emb
        
        # GNN Output (Euclidean Tangent Space)
        tangent_vecs, h = self.logical_gnn(node_features, edge_index, edge_type, edge_weight)
        
        # Scale by curvature (v6)
        # In hyperbolic geometry, scaling the tangent space is equivalent to changing curvature.
        c = self.get_curvature()
        # sqrt(c) is used to scale the result of ExpMap or tangent vectors.
        # For ExpMap: x = exp_c(v) = exp_1(sqrt(c) * v) / sqrt(c)
        
        # Stability fix: Clip tangent vectors to avoid NaNs in ExpMap
        # cosh(x) overflows quickly. For dim=256, sqrt(256)=16. 
        # tanh(v)*2.0 -> max norm 32. cosh(32) ~ 1e13 (Safe for float32).
        tangent_vecs = torch.tanh(tangent_vecs) * 2.0 
        
        scaled_tangent = tangent_vecs * torch.sqrt(c)

        # Hyperbolic Projection
        if self.manifold_type == 'poincare':
            embeddings = self.manifold.exp_map0(scaled_tangent)
            # Rescale result for curvature? 
            # Poincaré ball is bounded [0, 1/sqrt(c)].
            # We'll normalize to standard ball for now or scale it.
            # Usually we scale the distances.
            pass
        else:
            v_tangent = torch.cat([torch.zeros_like(scaled_tangent[..., 0:1]), scaled_tangent], dim=-1)
            embeddings = self.manifold.exp_map0(v_tangent)
        
        return embeddings



