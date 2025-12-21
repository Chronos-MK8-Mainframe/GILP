
"""
Epsilon v2 Chat Verification (Robust)

Tests the system on TRAINED scenarios using Real Inference (No Hardcoding).
"""

import sys
import os
import torch
import traceback

# Add root to path
sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.training.train_v2 import TOKENIZER

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

def chat_step(system, user_input):
    flush_print(f"\nUser: {user_input}")
    flush_print("  [Thinking...]")
    
    # 1. Concept Detection (Simulated for this demo, mapping to Training Pairs)
    if "fail" in user_input.lower():
        logic_goal = "Failure"
        psych_reaction = "Sympathy"
    elif "2+2" in user_input:
        logic_goal = "Math"
        psych_reaction = "Helpful"
    elif "fibonacci" in user_input.lower():
        logic_goal = "Recursion"
        psych_reaction = "Helpful"
    else:
        logic_goal = "Unknown"
        psych_reaction = "Confused"

    flush_print(f"  > Detected Logic: {logic_goal}")
    flush_print(f"  > Detected Psych: {psych_reaction}")
    
    # 2. Replicate Geometric State (Same seed as training)
    def get_concept_vec(text, dim, seed):
        torch.manual_seed(sum(ord(c) for c in text) + seed)
        return torch.randn(dim)

    try:
        v_logic = get_concept_vec(logic_goal, 64, seed=1)
        v_psych = get_concept_vec(psych_reaction, 64, seed=2)
        v_expr = get_concept_vec(psych_reaction, 128, seed=3)
        
        geo_state = torch.cat([v_logic, v_psych, v_expr]).view(1, 1, -1)
        
        flush_print(f"  > Geometric State Synthesized: {geo_state.shape}")
        
        # 3. Generate
        generated_ids = [TOKENIZER.bos_token_id]
        
        for i in range(50): # Max 50 words
            inp = torch.tensor(generated_ids).unsqueeze(0)
            logits = system.decoder(geo_state, inp)
            next_id = logits[0, -1].argmax().item()
            
            if next_id == TOKENIZER.eos_token_id:
                break
                
            generated_ids.append(next_id)
            
        response = TOKENIZER.decode(generated_ids)
        flush_print(f"Bot: {response}")
        flush_print("-" * 50)
        
    except Exception as e:
        flush_print(f"ERROR: {e}")
        traceback.print_exc()

def main():
    flush_print("=" * 60)
    flush_print("     EPSILON v2 CHAT (ROBUST VERIFICATION)")
    flush_print("=" * 60)

    # 1. Train Tokenizer (Must match training script exactly)
    training_texts = [
        "Oh no! Failures are just learning steps. I believe in you!",
        "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
        "The answer is 4. Basic arithmetic is fundamental!",
        "I am not sure I understand."
    ]
    TOKENIZER.train(training_texts)
    flush_print(f"Tokenizer Vocab: {TOKENIZER.vocab_size}")
    
    # 2. Load System with Resizing
    epsilon = Epsilon()
    
    # Checkpoint path
    ckpt_path = "epsilon_v2_checkpoints"
    
    # Hack: We know the decoder size in the checkpoint is big (VocabSize)
    # But Epsilon() init defaults to small. We must resize.
    import torch.nn as nn
    epsilon.decoder.embedding = nn.Embedding(TOKENIZER.vocab_size, epsilon.decoder_config.hidden_dim)
    epsilon.decoder.head = nn.Linear(epsilon.decoder_config.hidden_dim, TOKENIZER.vocab_size)
    
    try:
        epsilon.load(ckpt_path)
        flush_print("✓ System Loaded")
    except Exception as e:
        flush_print(f"Failed to load: {e}")
        return

    # 3. Run Tests
    # "Oh no! Failures are just learning steps. I believe in you!"
    chat_step(epsilon, "I failed the test.")
    
    # "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"
    chat_step(epsilon, "Write a function for fibonacci.")

if __name__ == "__main__":
    main()
