
"""
Epsilon v2 System (The Coordinator)

This class orchestrates the Multi-Manifold Architecture:
Layer 0: Logic (Reasoning)
Layer 1: Psychology (Emotion/State)
Layer 2: Expression (Tone/Style)
Layer 3: Tiny Decoder (Text Rendering)
"""

import torch
import torch.nn as nn
import difflib # For Fuzzy Matching
import re
from typing import Optional, List, Dict, Tuple

from epsilon.config import EpsilonConfig, PsychologyConfig, ExpressionConfig, TinyDecoderConfig
from epsilon.modules.tiny_decoder import TinyDecoder
from epsilon.geometry.parabolic import ParabolicManifold
from epsilon.knowledge.search import KnowledgeSearch
from epsilon.cognition.conflict import TensionEngine
from epsilon.memory.hypothesis import HypothesisSpace
from epsilon.geometry.compression import CompressionEngine
from epsilon.chat.text_encoder import TextEncoder, SimpleTokenizer
from epsilon.chat.tokenizer import SimpleWordTokenizer

class Epsilon:
    """
    The Epsilon v2 System.
    
    It is not a single model, but a stack of geometric engines.
    """
    
    def __init__(self, 
                 logic_config: Optional[EpsilonConfig] = None,
                 psych_config: Optional[PsychologyConfig] = None,
                 expr_config: Optional[ExpressionConfig] = None,
                 decoder_config: Optional[TinyDecoderConfig] = None):
         
        # 2. Initialize Parabolic Engine (The Unified Brain)
        self.manifold = ParabolicManifold()
        
        # Knowledge Graph (Embeddings)
        # We store known concepts here. Key -> Vector
        self.knowledge_store: Dict[str, torch.Tensor] = {}
        
        # Personality Anchor (The "Sister" vector in the Atmosphere)
        # Initialized randomly in the "Warm" region of the shell
        self.personality_vector = torch.nn.Parameter(torch.randn(64))
        
        # Conflict Engine (The Conscience)
        self.tension_engine = TensionEngine(threshold=1.0) # High enough to ignore noise, low enough to catch conflict
        
        # 3. Helpers
        self.search = KnowledgeSearch()
        
        # 4. Tiny Decoder (The Mouth)
        self.decoder_config = TinyDecoderConfig()
        self.decoder = TinyDecoder(self.decoder_config)
        self.word_tokenizer = SimpleWordTokenizer()
        
        # 5. Text Encoder (The Ears)
        self.text_encoder = TextEncoder(
            vocab_size=8192,
            d_model=64,
            gilp_dim=64 # To match system dim
        )
        self.char_tokenizer = SimpleTokenizer(vocab_size=8192)
        
    def embed_text(self, text):
        """
        Maps text to a geometric vector.
        Strategy: Fuzzy Snapping (Force input to nearest known Anchor).
        """
        text = text.strip()
        text_lower = text.lower()
        
        # 1. Exact Match (Case-Insensitive)
        for key in self.knowledge_store:
            # print(f"Checking key: {key}") 
            if key.lower() == text_lower:
                if isinstance(self.knowledge_store[key], torch.Tensor):
                    # print(f"  > Exact Match: '{text}' -> '{key}'") 
                    return self.knowledge_store[key]
                
        # 2. Fuzzy Match (Snap to Grid)
        keys = [k for k in self.knowledge_store.keys() if isinstance(self.knowledge_store[k], torch.Tensor)]
        
        best_key = None
        best_score = 0.0
        
        # Use whole-word regex check
        for key in keys:
            pattern = rf"\b{re.escape(key.lower())}\b"
            if re.search(pattern, text_lower):
                # Length bias: prefer longer matches (e.g. "ConceptNet" over "Net")
                score = len(key) / len(text) + 0.5 
                if score > best_score:
                    best_score = score
                    best_key = key
                    
        if best_key:
            # print(f"  > Snapped '{text}' to '{best_key}' (Keyword)")
            return self.knowledge_store[best_key]
                    
        # 3. Fallback: Use TextEncoder (The ears)
        print(f"  > Unknown Input '{text}'. Encoding via TextEncoder.")
        tokens = self.char_tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            v = self.text_encoder(tokens)
        
        # Ensure it's 64-dim (TextEncoder might be 32, we project or pad if needed)
        # For now, we assume gilp_dim was set to 64 in __init__
        return self.manifold.normalize(v[0])
        
    def think(self, context: str):
        """
        Parabolic Thinking Loop: High -> Low -> High.
        Returns response and trace data for visualization.
        """
        # 1. Input Projection (Observation)
        # Uses embed_text (Fuzzy Snapping) to find initial coordinate
        observation = self.embed_text(context)
        
        print(f"Thinking about: {context}")
        trace = []
        trace.append(("Observation", observation))
        # print(f"  > Input 'Observation' (High-Dim)")
        
        # 2. The Descent (Grounding)
        # Strip context to find Logic Core
        core_thought = self.manifold.ground(observation)
        trace.append(("Grounding (Descent)", core_thought))
        print(f"  > Descent to Logic Core (Dims 0-4)")
        
        # 3. Descent (Logic Search)
        # Find nearest concept in Core Logic dimensions (0-4)
        print("  > Descent to Logic Core (Dims 0-4)")
        
        best_match = None
        min_dist = float('inf')
        
        for name, vec in self.knowledge_store.items():
            if not isinstance(vec, torch.Tensor):
                continue
                
            # Compare only Logic Core (Dims 0-4) to simulate gravity
            d = torch.norm(core_thought[:4] - vec[:4])
            
            if d < min_dist:
                min_dist = d
                best_match = (name, vec)
                
        if best_match:
            concept_name, concept_vec = best_match
            print(f"  > Logic Lock: Identified '{concept_name}'")
            trace.append((f"Logic Node: {concept_name}", self.manifold.ground(concept_vec)))
            
            # 1. Reflex Check (Do we know this exact script?)
            # If we have a prepared response, bypass conflict engine.
            payload_key = f"{concept_name}_RESPONSE"
            if payload_key in self.knowledge_store:
                final_vec = concept_vec
                response_text = self.knowledge_store[payload_key]
                print(f"  > Reflex Triggered: {response_text}")
                trace.append(("Reflex/RAG", final_vec))
                return response_text, trace
            
            # 2. Critical Thinking (Conflict Check)
            # Logic says: concept_vec (Ground)
            # Emotion says: personality_vector (Atmosphere)
            # Do they align?
            
            # We construct the "Impulse" (what she WANTS to say) vs "Logic" (what is TRUE)
            # For this demo, let's treat the 'personality_vector' as the 'Desire'
            # and 'concept_vec' as 'Duty'.
            
            # Project logic to atmosphere to compare apples to apples
            logic_in_atm = concept_vec.clone()
            logic_in_atm[5:] = concept_vec[5:] # Use concept's native atmosphere? 
            # Actually, standard parabolic flow: Logic IS the core.
            
            # Let's check tension between Current Mood (Personality) and the Concept
            # e.g. If Mood is Happy but Concept is "Failure", Tension is High.
            
            # Tension calculation (comparing Atmosphere parts)
            mood_vec = self.personality_vector
            fact_vec = concept_vec
            
            # We resolve the vectors to get the final "Conscious Decision"
            final_vec, tension, steps = self.tension_engine.resolve(fact_vec, mood_vec)
            
            if tension > self.tension_engine.threshold:
                print(f"  ! CONFLICT DETECTED (Tension: {tension:.2f})")
                print(f"  ! Entering Debate Loop ({steps} steps)...")
                trace.append(("Conflict/Hesitation", final_vec))
                response_text = f"I... I had to think about that. ({concept_name})"
            else:
                # Flow: Logic -> Add Personality -> Output
                final_vec = self.manifold.express(fact_vec, mood_vec)

            # 5. Descent (Expression)
            trace.append(("Expression (Resolved)", final_vec))
            print(f"  > Ascent to Atmosphere (Resolved)")
            
            # --- NEW: GENERATE NATURAL RESPONSE ---
            # Construct geometric trajectory for decoder: [Logic, Psych, Expr]
            # Match TinyDecoderConfig: input_dim=256
            # Logic (64) + Psych (64) + Expr (128)
            # Match TinyDecoderConfig: input_dim=256
            # Logic (64) + Psych (64) + Expr (128)
            v_logic = torch.zeros(64)
            v_logic[:64] = fact_vec[:64] # fact_vec is already 64
            
            v_psych = torch.zeros(64)
            v_psych[:64] = mood_vec[:64] # mood_vec is already 64
            
            v_expr = final_vec # final_vec IS 128-dim
            
            geo_state = torch.cat([v_logic, v_psych, v_expr]).view(1, 1, -1)
            
            print(f"  > Decoder: Translating trajectory to text... (Input Dim: {geo_state.shape[-1]})")
            generated_ids = self.decoder.generate(
                geo_state, 
                start_token=self.word_tokenizer.bos_token_id,
                end_token=self.word_tokenizer.eos_token_id,
                max_len=self.decoder_config.max_response_length if hasattr(self.decoder_config, 'max_response_length') else 50
            )
            
            response_text = self.word_tokenizer.decode(generated_ids[0].tolist())
            
        else:
            print("  > Unknown Concept. Staying in High Orbit.")
            response_vector = observation
            # Even for unknown concepts, try to decode the observation
            # observation is 128-dim (Expression level), we need 256
            # observation is 128-dim (Expression level), we need 256
            v_logic = torch.zeros(64)
            v_logic[:64] = observation[:64]
            
            v_psych = torch.zeros(64)
            v_psych[:64] = self.personality_vector[:64]
            
            v_expr = observation 
            
            geo_state = torch.cat([v_logic, v_psych, v_expr]).view(1, 1, -1)
            generated_ids = self.decoder.generate(
                geo_state, 
                start_token=self.word_tokenizer.bos_token_id,
                end_token=self.word_tokenizer.eos_token_id
            )
            response_text = self.word_tokenizer.decode(generated_ids[0].tolist())
            
        return response_text, trace

    def train_step(self, data_batch):
        """
        Unified training step.
        """
        # 1. Train Logic Engine (if logic data)
        # 2. Train Psych Engine (if psych data)
        # 3. Train Expr Engine (if expr data)
        # 4. Train Decoder (end-to-end or teacher forced)
        pass
    
    def save(self, path: str):
        """Save System."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save({
            "knowledge": self.knowledge_store,
            "personality": self.personality_vector,
            "decoder": self.decoder.state_dict()
        }, os.path.join(path, "system_v2.pt"))
        
    def load(self, path: str):
        """Load System."""
        import os
        pt_path = os.path.join(path, "system_v2.pt")
        try:
            data = torch.load(pt_path)
            self.knowledge_store = data["knowledge"]
            self.personality_vector = data["personality"]
            self.decoder.load_state_dict(data["decoder"])
        except FileNotFoundError:
            print("No checkpoint found.")
