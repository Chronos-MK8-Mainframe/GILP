
"""
Epsilon v2 Training Script

Trains the Multi-Manifold System:
1. Logic Layer (Reasoning)
2. Psychology Layer (Emotion)
3. Expression Layer (Style)
4. Tiny Decoder (Text Generation)

Hardware: Optimized for CPU (i5, 10GB RAM)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add root to path
sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.system import Epsilon
from epsilon.epsilon import GeometricReasoningEngine
from epsilon.chat.tokenizer import SimpleWordTokenizer

# Initialize Global Tokenizer
TOKENIZER = SimpleWordTokenizer()

# ============================================================================
# 1. Dataset Wrappers (Simplified for Demo)
# ============================================================================

class LogicDataset(Dataset):
    """Simple dataset for Logic: A implies B pairs."""
    def __init__(self, size=1000):
        self.size = size
        # Demo data: Simple syllogisms and commonsense
        self.data = [
            ("User is sad", "Comfort user"),
            ("User asked question", "Provide answer"),
            ("Code has error", "Debug code"),
            ("System constraint", "Optimize usage")
        ]
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Cycle through demo data
        src, dst = self.data[idx % len(self.data)]
        return src, dst

class CodingDataset(Dataset):
    """Dataset for Coding Logic: Problem -> Solution Concept."""
    def __init__(self, size=1000):
        self.size = size
        # Demo data: Coding logic patterns
        self.data = [
            ("Write Fibonacci", "Recursive Implementation"),
            ("Sort Array", "Quicksort/Mergesort"),
            ("Handle HTTP Request", "Requests Library"),
            ("Parse JSON", "json.load"),
            ("Define Class", "__init__ method"),
            ("Train Model", "Forward Pass & Loss")
        ]
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        src, dst = self.data[idx % len(self.data)]
        return src, dst

class PsychologyDataset(Dataset):
    """Dataset for Psychology: Event -> Emotion transitions."""
    def __init__(self, size=1000):
        self.size = size
        # Demo data: Event -> Big Sister Reaction
        self.data = [
            ("User failed", "Encouraging"),
            ("User succeeded", "Proud"),
            ("User confused", "Patient"),
            ("User rude", "Stern but caring")
        ]
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        src, dst = self.data[idx % len(self.data)]
        return src, dst

class ExpressionDataset(Dataset):
    """Dataset for Expression: Tone -> Tone transitions."""
    def __init__(self, size=1000):
        self.size = size
        # Demo data: Flow of conversation style
        self.data = [
            ("Formal start", "Casual middle"),
            ("Casual middle", "Warm end"),
            ("Questioning", "Explaining"),
            ("Listening", "Advising")
        ]
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        src, dst = self.data[idx % len(self.data)]
        return src, dst

# ============================================================================
# 2. Training Loop
# ============================================================================

def train_manifold(engine: GeometricReasoningEngine, dataset: Dataset, 
                  name: str, epochs: int = 100):
    """Generic training loop for a geometric engine."""
    print(f"\nTraining {name} Manifold...")
    
    # In a real run, we'd tokenize text -> embeddings.
    # For this demo, we'll simulate embeddings updates.
    
    # 1. Initialize embeddings if needed
    # (In real code, this would come from a tokenizer + encoder)
    # We'll just optimize random embeddings for demonstration of the geometric constraints.
    
    vocab_size = 100 # Demo vocab
    dim = engine.config.embedding_dim
    
    # Mock embeddings parameter
    embeddings = nn.Parameter(torch.randn(vocab_size, dim) * 0.1)
    optimizer = optim.Adam([embeddings], lr=0.01)
    
    engine.metric_loss = engine.manifold.target_distance_loss # Bind loss function
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Simulate batch processing
        # We map string -> random index for demo
        import random
        batch_src = torch.randint(0, vocab_size, (32,))
        batch_dst = torch.randint(0, vocab_size, (32,))
        
        optimizer.zero_grad()
        
        # 1. Project to Manifold (Poincare Ball)
        # Ensure normalization < 1
        z = embeddings 
        norm = z.norm(dim=-1, keepdim=True)
        z = z / (norm + 1e-5) * 0.9 # Project inside ball
        
        # 2. Compute Geometric Loss (Structure)
        # We want dist(src, dst) -> shell_radius
        # Create edge_index for batch [2, 32]
        edge_index = torch.stack([batch_src, batch_dst], dim=0)
        
        loss = engine.manifold.target_distance_loss(
            z, edge_index, target=engine.config.shell_radius
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if epoch % 1 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
            
    # Save simulated result
    engine._embeddings = z.detach()
    engine.navigator.update_embeddings(z.detach())
    print(f"✓ {name} Trained")

def train_decoder(system: Epsilon, epochs: int = 300):
    """
    Train the Tiny Decoder to be a Translator (Geometry -> Text).
    Real End-to-End Training.
    """
    print(f"\nTraining Tiny Decoder (Geometric Translation)...")
    
    # Training Data: (LogicConcept, PsychState, TargetText)
    training_pairs = [
        # (LogicNodeName, PsychState, ActualResponse)
        ("Failure", "Sympathy", "Oh no! Failures are just learning steps. I believe in you!"),
        ("Recursion", "Helpful", "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"),
        ("Math", "Helpful", "The answer is 4. Basic arithmetic is fundamental!"),
        ("Unknown", "Confused", "I am not sure I understand."),
        ("Hello", "Warm", "Hello there! It is wonderful to see you."),
        ("Hi", "Friendly", "Hi! I am ready to help with whatever you need."),
        ("Status", "Clarity", "All systems are operational and the manifold is stable."),
        ("Thanks", "Appreciative", "You are very welcome! I am happy to be of service."),
        ("Jump", "Energetic", "Initiating vertical propulsion! Look at me go!"),
        ("Code", "Analytical", "I have initialized the Python synthesis engine. What shall we build?"),
        ("Love", "Warm", "A beautiful attractor in the human manifold. I understand its importance."),
        ("Sad", "Comforting", "I detect resistance in your manifold. I am here for you.")
    ]
    
    # 1. Train Tokenizer Vocab first
    all_texts = [pair[2] for pair in training_pairs]
    TOKENIZER.train(all_texts)
    
    # 2. Resize Decoder to match new Vocab
    # In V2, we dynamically update the config
    new_vocab_size = TOKENIZER.vocab_size
    print(f"  > Resizing Decoder Head to Vocab Size: {new_vocab_size}")
    
    # Create new head with correct size
    system.decoder.embedding = nn.Embedding(new_vocab_size, system.decoder_config.hidden_dim)
    system.decoder.head = nn.Linear(system.decoder_config.hidden_dim, new_vocab_size)
    
    decoder = system.decoder
    optimizer = optim.Adam(decoder.parameters(), lr=0.005) # Higher LR for fast convergence
    criterion = nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_token_id)
    
    def get_concept_vec(text, dim, seed):
        torch.manual_seed(sum(ord(c) for c in text) + seed)
        return torch.randn(dim)

    # Aggressive Training Loop
    progress_bar = tqdm(range(epochs), desc="Decoder Training")
    
    for epoch in progress_bar:
        optimizer.zero_grad()
        total_loss = 0
        
        for logic_concept, psych_state, target_text in training_pairs:
            # 1. Synthesize Geometric State
            v_logic = get_concept_vec(logic_concept, 64, seed=1)
            v_psych = get_concept_vec(psych_state, 64, seed=2)
            v_expr = get_concept_vec(psych_state, 128, seed=3) 
            
            geo_state = torch.cat([v_logic, v_psych, v_expr]).view(1, 1, -1)
            
            # 2. Prepare Target Text
            ids = [TOKENIZER.bos_token_id] + TOKENIZER.encode(target_text) + [TOKENIZER.eos_token_id]
            tgt_tensor = torch.tensor(ids).unsqueeze(0)
            
            # Input/Target
            dec_input = tgt_tensor[:, :-1]
            loss_target = tgt_tensor[:, 1:]
            
            # 3. Forward
            logits = decoder(geo_state, dec_input)
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), loss_target.reshape(-1))
            loss.backward()
            total_loss += loss.item()
            
        optimizer.step()
        progress_bar.set_postfix({'loss': f"{total_loss:.4f}"})
            
    print("✓ Tiny Decoder Trained (Word Level)")

# ============================================================================
# 3. Main
# ============================================================================

def main():
    print("="*60)
    print("     EPSILON v2 TRAINING (CPU Optimized)")
    print("="*60)
    
    # 1. Instantiate System
    print("Initializing System...")
    epsilon = Epsilon()
    
    # 2. Train Manifolds
    # In real pipeline, these use specific datasets
    train_manifold(epsilon.logic, LogicDataset(), "Logic (General)")
    train_manifold(epsilon.logic, CodingDataset(), "Logic (Coding)") # Train logic on code tasks too
    train_manifold(epsilon.psychology, PsychologyDataset(), "Psychology (Emotion)")
    train_manifold(epsilon.expression, ExpressionDataset(), "Expression (Style)")
    
    # 3. Train Decoder
    train_decoder(epsilon)
    
    # 4. Save
    print("\nSaving Checkpoints...")
    epsilon.save("epsilon_v2_checkpoints")
    print("✓ Saved to ./epsilon_v2_checkpoints")
    
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
