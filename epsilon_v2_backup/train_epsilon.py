"""
Epsilon Chat Training - Phase-by-Phase

Trains encoder, GILP geometry, and decoder separately.
Designed for CPU-only hardware (10GB RAM, no dGPU).

Usage:
  python train_epsilon.py --phase encoder
  python train_epsilon.py --phase gilp
  python train_epsilon.py --phase decoder
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import json
import os
import sys
sys.path.insert(0, '.')

from epsilon.chat.text_encoder import TextEncoder, SimpleTokenizer, ContrastiveTrainer
from epsilon.chat.text_decoder import TextDecoder, TrajectoryToTextTrainer


# ============================================================
# PHASE 1: Encoder Training (Contrastive)
# ============================================================

class TextPairDataset(Dataset):
    """Simple dataset of similar text pairs."""
    
    def __init__(self, pairs, tokenizer, max_length=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        text1, text2 = self.pairs[idx]
        
        tokens1 = self.tokenizer.encode(text1, max_length=self.max_length)
        tokens2 = self.tokenizer.encode(text2, max_length=self.max_length)
        
        return {
            'text1': torch.tensor(tokens1, dtype=torch.long),
            'text2': torch.tensor(tokens2, dtype=torch.long)
        }


def create_sample_pairs():
    """Create some sample training pairs (similar sentences)."""
    pairs = [
        # Paraphrases
        ("The cat sat on the mat.", "A cat was sitting on the mat."),
        ("The dog barked loudly.", "A loud bark came from the dog."),
        ("It is raining outside.", "Rain is falling outside."),
        ("The sun is shining brightly.", "Bright sunshine today."),
        ("I love to eat pizza.", "Pizza is my favorite food."),
        ("She reads many books.", "She enjoys reading books."),
        ("The car is very fast.", "That's a fast car."),
        ("He plays the guitar well.", "He's a good guitar player."),
        ("The movie was exciting.", "What an exciting movie!"),
        ("Coffee keeps me awake.", "I stay awake with coffee."),
        
        # Related concepts
        ("Fire is hot.", "Flames produce heat."),
        ("Water is wet.", "Wet things contain water."),
        ("The sky is blue.", "Blue sky above us."),
        ("Trees grow tall.", "Tall trees in the forest."),
        ("Birds can fly.", "Flying birds in the sky."),
        
        # Simple facts
        ("The earth is round.", "Our planet is spherical."),
        ("Humans need oxygen.", "Oxygen is vital for humans."),
        ("Snow is cold.", "Cold snow on the ground."),
        ("Music sounds pleasant.", "Pleasant music playing."),
        ("Books contain knowledge.", "Knowledge comes from books."),
    ]
    
    # Duplicate to make more training data
    pairs = pairs * 10  # 200 pairs
    return pairs


def train_encoder(epochs=50, batch_size=8, lr=1e-3, save_path="encoder.pt"):
    """Train the text encoder with contrastive learning."""
    print("\n" + "="*50)
    print("PHASE 1: Training Text Encoder")
    print("="*50)
    
    # Create encoder
    encoder = TextEncoder()
    tokenizer = SimpleTokenizer()
    
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {num_params:,}")
    
    # Create dataset
    pairs = create_sample_pairs()
    dataset = TextPairDataset(pairs, tokenizer)
    
    print(f"Training pairs: {len(dataset)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Training loop
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    
    encoder.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Simple batching
        indices = torch.randperm(len(dataset))
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            
            # Get batch
            batch_tokens = []
            for idx in batch_idx:
                item = dataset[idx]
                batch_tokens.append(item['text1'])
                batch_tokens.append(item['text2'])
            
            tokens = torch.stack(batch_tokens)
            
            # Forward
            embeddings = encoder(tokens)
            
            # Contrastive loss (InfoNCE-style)
            embeddings = F.normalize(embeddings, dim=-1)
            sim = torch.matmul(embeddings, embeddings.T) / 0.07
            
            # Pairs are positive (0-1, 2-3, etc.)
            labels = torch.arange(len(embeddings))
            labels = labels ^ 1  # XOR with 1
            
            loss = F.cross_entropy(sim, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}: loss = {avg_loss:.4f} ({elapsed:.1f}s)")
    
    # Save
    torch.save(encoder.state_dict(), save_path)
    print(f"\nSaved encoder to {save_path}")
    
    return encoder


# ============================================================
# PHASE 2: GILP Geometry (Knowledge Embeddings)
# ============================================================

def train_gilp_geometry(encoder, save_path="gilp_knowledge.pt"):
    """Build knowledge embeddings from concepts."""
    print("\n" + "="*50)
    print("PHASE 2: Building GILP Knowledge Base")
    print("="*50)
    
    tokenizer = SimpleTokenizer()
    
    # Define concepts (would come from Wikipedia/KB in production)
    concepts = [
        # Elements
        "fire", "water", "earth", "air",
        # Properties
        "hot", "cold", "wet", "dry", "bright", "dark",
        # Nature
        "sun", "moon", "star", "sky", "cloud", "rain", "wind", "storm",
        "tree", "flower", "grass", "forest", "mountain", "river", "ocean",
        # Animals
        "cat", "dog", "bird", "fish", "horse", "cow", "sheep",
        # People
        "human", "person", "man", "woman", "child", "family",
        # Objects
        "house", "car", "book", "food", "water", "fire",
        # Abstract
        "love", "hate", "happy", "sad", "good", "bad", "time", "space",
    ]
    
    print(f"Concepts: {len(concepts)}")
    
    # Encode all concepts
    encoder.eval()
    embeddings = []
    
    with torch.no_grad():
        for concept in concepts:
            batch = tokenizer.batch_encode([concept])
            emb = encoder(batch['input_ids'])
            embeddings.append(emb[0])
    
    embeddings = torch.stack(embeddings)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save
    torch.save({
        'embeddings': embeddings,
        'concepts': concepts
    }, save_path)
    
    print(f"Saved knowledge to {save_path}")
    
    # Show some similarities
    print("\nSample similarities:")
    embeddings_norm = F.normalize(embeddings, dim=-1)
    
    test_pairs = [("fire", "hot"), ("water", "wet"), ("sun", "bright")]
    for c1, c2 in test_pairs:
        if c1 in concepts and c2 in concepts:
            i1, i2 = concepts.index(c1), concepts.index(c2)
            sim = (embeddings_norm[i1] @ embeddings_norm[i2]).item()
            print(f"  {c1} <-> {c2}: {sim:.3f}")
    
    return embeddings, concepts


# ============================================================
# PHASE 3: Decoder Training
# ============================================================

def train_decoder(encoder, knowledge, epochs=50, batch_size=4, lr=1e-3, 
                 save_path="decoder.pt"):
    """Train decoder to convert trajectories to text."""
    print("\n" + "="*50)
    print("PHASE 3: Training Text Decoder")
    print("="*50)
    
    embeddings, concepts = knowledge
    tokenizer = SimpleTokenizer()
    
    # Create decoder
    decoder = TextDecoder()
    
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder params: {num_params:,}")
    
    # Create simple training data: concept -> "This is about [concept]"
    # (In production, use real text passages)
    training_data = []
    for i, concept in enumerate(concepts):
        response = f"This is about {concept}."
        traj = embeddings[i:i+1].unsqueeze(0)  # [1, 1, dim]
        tokens = tokenizer.encode(response, max_length=32)
        training_data.append((traj, torch.tensor([tokens])))
    
    print(f"Training examples: {len(training_data)}")
    
    # Train
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    decoder.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for traj, tokens in training_data:
            # Teacher forcing
            input_tokens = tokens[:, :-1]
            target = tokens[:, 1:]
            
            logits = decoder(traj, input_tokens)
            
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(training_data)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}: loss = {avg_loss:.4f} ({elapsed:.1f}s)")
    
    # Save
    torch.save(decoder.state_dict(), save_path)
    print(f"\nSaved decoder to {save_path}")
    
    # Test generation
    print("\nTest generation:")
    decoder.eval()
    with torch.no_grad():
        for i in range(3):
            traj = embeddings[i:i+1].unsqueeze(0)
            tokens = decoder.generate(traj, max_length=20, temperature=0.7)
            text = tokenizer.decode(tokens[0].tolist())
            print(f"  {concepts[i]}: {text}")
    
    return decoder


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Epsilon Chat")
    parser.add_argument("--phase", choices=["encoder", "gilp", "decoder", "all"],
                       default="all", help="Training phase")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs("checkpoints", exist_ok=True)
    
    if args.phase in ["encoder", "all"]:
        encoder = train_encoder(
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path="checkpoints/encoder.pt"
        )
    else:
        # Load existing encoder
        encoder = TextEncoder()
        if os.path.exists("checkpoints/encoder.pt"):
            encoder.load_state_dict(torch.load("checkpoints/encoder.pt"))
            print("Loaded encoder from checkpoints/encoder.pt")
    
    if args.phase in ["gilp", "all"]:
        knowledge = train_gilp_geometry(
            encoder,
            save_path="checkpoints/gilp_knowledge.pt"
        )
    else:
        if os.path.exists("checkpoints/gilp_knowledge.pt"):
            data = torch.load("checkpoints/gilp_knowledge.pt")
            knowledge = (data['embeddings'], data['concepts'])
            print(f"Loaded knowledge: {len(knowledge[1])} concepts")
        else:
            knowledge = None
    
    if args.phase in ["decoder", "all"] and knowledge:
        decoder = train_decoder(
            encoder, knowledge,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path="checkpoints/decoder.pt"
        )
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == "__main__":
    main()
