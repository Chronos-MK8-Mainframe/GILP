"""
Epsilon Chat - Main Chat Interface

LLM-free conversational AI using geometric reasoning.

Architecture:
  Text → Encoder → GILP Manifold → Walker/Reasoner → Trajectory → Decoder → Response

Total: ~1.2M params (runs on CPU with 4GB RAM)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from epsilon.chat.text_encoder import TextEncoder, SimpleTokenizer
from epsilon.chat.text_decoder import TextDecoder
from epsilon.chat.creative_walker import CreativeWalker, WalkStyle


@dataclass
class ChatConfig:
    """Configuration for Epsilon Chat."""
    # Model dimensions (TINY for laptop)
    vocab_size: int = 8192
    d_model: int = 64
    gilp_dim: int = 32
    num_layers: int = 2
    
    # Generation
    max_response_length: int = 50
    temperature: float = 0.8
    top_k: int = 50
    
    # Walking
    walk_style: str = "associative"
    walk_steps: int = 5
    step_size: float = 0.1


@dataclass
class ConversationState:
    """Tracks conversation as a trajectory through manifold."""
    history: List[torch.Tensor] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    def add_turn(self, query_vec: torch.Tensor, response_vec: torch.Tensor,
                 user_text: str, bot_text: str):
        """Add a conversation turn."""
        self.history.append(query_vec)
        self.history.append(response_vec)
        self.messages.append({"role": "user", "text": user_text})
        self.messages.append({"role": "bot", "text": bot_text})
    
    @property
    def trajectory(self) -> Optional[torch.Tensor]:
        """Get full conversation as trajectory."""
        if not self.history:
            return None
        return torch.stack(self.history)


class EpsilonChat:
    """
    LLM-free conversational AI.
    
    Uses:
    - Small encoder to map text → GILP manifold
    - Creative walker for generation/exploration
    - Small decoder to map trajectories → text
    
    The manifold IS the intelligence — small transformers just translate.
    """
    
    def __init__(self, config: Optional[ChatConfig] = None,
                 knowledge_embeddings: Optional[torch.Tensor] = None,
                 concept_names: Optional[List[str]] = None):
        """
        Args:
            config: Chat configuration
            knowledge_embeddings: Pre-computed concept embeddings [N, gilp_dim]
            concept_names: Names for each concept
        """
        self.config = config or ChatConfig()
        
        # Initialize models
        self.encoder = TextEncoder(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            gilp_dim=self.config.gilp_dim,
            num_layers=self.config.num_layers
        )
        
        self.decoder = TextDecoder(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            gilp_dim=self.config.gilp_dim,
            num_layers=self.config.num_layers
        )
        
        self.tokenizer = SimpleTokenizer(vocab_size=self.config.vocab_size)
        
        # Knowledge base
        if knowledge_embeddings is not None:
            self.walker = CreativeWalker(knowledge_embeddings, concept_names)
        else:
            self.walker = None
        
        # Conversation state
        self.conversation = ConversationState()
        
        # Track total params
        self._count_params()
    
    def _count_params(self):
        """Count and print total parameters."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total = enc_params + dec_params
        print(f"Epsilon Chat initialized:")
        print(f"  Encoder: {enc_params:,} params")
        print(f"  Decoder: {dec_params:,} params")
        print(f"  Total:   {total:,} params ({total/1e6:.1f}M)")
    
    def encode_query(self, text: str) -> torch.Tensor:
        """Encode user query to GILP vector."""
        batch = self.tokenizer.batch_encode([text])
        with torch.no_grad():
            vec = self.encoder(batch['input_ids'], batch['attention_mask'])
        return vec[0]
    
    def generate_response_trajectory(self, query_vec: torch.Tensor) -> torch.Tensor:
        """
        Generate response trajectory by walking through manifold.
        
        This is where the "reasoning" happens — we navigate the
        geometric structure to find relevant concepts.
        """
        if self.walker is None:
            # No knowledge base — just use query as context
            trajectory = query_vec.unsqueeze(0).unsqueeze(0)
        else:
            # Walk through manifold from query position
            style = WalkStyle(self.config.walk_style)
            walk = self.walker.walk(
                start=query_vec,
                num_steps=self.config.walk_steps,
                style=style,
                step_size=self.config.step_size
            )
            trajectory = walk.trajectory.unsqueeze(0)  # [1, steps, dim]
        
        # Include conversation context
        if self.conversation.trajectory is not None:
            context = self.conversation.trajectory.unsqueeze(0)
            trajectory = torch.cat([context, trajectory], dim=1)
        
        return trajectory
    
    def decode_trajectory(self, trajectory: torch.Tensor) -> str:
        """Decode trajectory to text response."""
        with torch.no_grad():
            tokens = self.decoder.generate(
                trajectory,
                max_length=self.config.max_response_length,
                temperature=self.config.temperature,
                top_k=self.config.top_k
            )
        
        # Decode tokens to text
        text = self.tokenizer.decode(tokens[0].tolist())
        return text
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and generate response.
        
        Full pipeline:
        1. Encode user text → GILP vector
        2. Walk through manifold (reasoning)
        3. Decode trajectory → response text
        """
        # Encode query
        query_vec = self.encode_query(user_input)
        
        # Generate response trajectory
        trajectory = self.generate_response_trajectory(query_vec)
        
        # Decode to text
        response = self.decode_trajectory(trajectory)
        
        # Update conversation state
        response_vec = trajectory[0, -1] if trajectory.dim() == 3 else query_vec
        self.conversation.add_turn(query_vec, response_vec, user_input, response)
        
        return response
    
    def chat_with_trace(self, user_input: str) -> Tuple[str, Dict]:
        """
        Chat with debugging trace.
        
        Returns response and trace info (embeddings, walk, etc.)
        """
        query_vec = self.encode_query(user_input)
        trajectory = self.generate_response_trajectory(query_vec)
        response = self.decode_trajectory(trajectory)
        
        trace = {
            "query_vector": query_vec.tolist(),
            "trajectory_shape": list(trajectory.shape),
            "walk_concepts": self.walker.walk(query_vec, 5).concepts if self.walker else [],
            "response": response
        }
        
        return response, trace
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation = ConversationState()
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location='cpu')
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])


def demo_chat():
    """Demo Epsilon Chat (untrained — outputs will be random)."""
    print("=" * 50)
    print("     EPSILON CHAT DEMO (Untrained)")
    print("=" * 50)
    print()
    
    # Create chat with some fake knowledge
    num_concepts = 100
    gilp_dim = 32
    fake_embeddings = torch.randn(num_concepts, gilp_dim) * 0.5
    fake_names = [f"concept_{i}" for i in range(num_concepts)]
    
    chat = EpsilonChat(
        knowledge_embeddings=fake_embeddings,
        concept_names=fake_names
    )
    
    print("\n[Note: Responses are random — model is untrained]\n")
    
    # Demo conversation
    queries = [
        "What is the meaning of life?",
        "Tell me about Python programming.",
        "How does the weather work?"
    ]
    
    for query in queries:
        print(f"User: {query}")
        response, trace = chat.chat_with_trace(query)
        print(f"Bot:  {response[:60]}...")
        print(f"      (walked through: {trace['walk_concepts'][:3]})")
        print()
    
    print("=" * 50)
    print("Demo complete. Train the model for real responses!")
    print("=" * 50)


if __name__ == "__main__":
    demo_chat()
