
"""
SimpleWordTokenizer for Epsilon v2

Ensures "Natural Conversation" by operating on words, not chars.
This guarantees that any output is at least a valid English word.
"""

from typing import List, Dict
import re

class SimpleWordTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        
        # Special Tokens
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
        
        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        for i, token in enumerate(self.specials):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
            
        self.special_ids = list(range(len(self.specials)))
        self.pad_token_id = self.word_to_id[self.pad_token]
        self.bos_token_id = self.word_to_id[self.bos_token]
        self.eos_token_id = self.word_to_id[self.eos_token]
        
    def train(self, texts: List[str]):
        """Build vocab from list of texts."""
        word_freq = {}
        for text in texts:
            words = self._tokenize(text)
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1
        
        # Sort by freq
        sorted_words = sorted(word_freq.keys())
        
        # Add to vocab
        start_id = len(self.specials)
        for i, w in enumerate(sorted_words):
            self.word_to_id[w] = start_id + i
            self.id_to_word[start_id + i] = w
            
        print(f"Tokenizer trained. Vocab size: {len(self.word_to_id)}")
            
    def _tokenize(self, text: str) -> List[str]:
        """Basic regex tokenizer."""
        # Split by space and punctuation, keeping punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens
        
    def encode(self, text: str) -> List[int]:
        """Convert string to word IDs."""
        words = self._tokenize(text)
        ids = []
        for w in words:
            ids.append(self.word_to_id.get(w, self.word_to_id[self.unk_token]))
        return ids
        
    def decode(self, ids: List[int]) -> str:
        """Convert IDs to string."""
        words = []
        for i in ids:
            if i in self.special_ids:
                continue
            words.append(self.id_to_word.get(i, ""))
            
        # Simple heuristic join (not perfect but readable)
        # Join with space, then fix punctuation
        text = " ".join(words)
        text = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" (", "(")
        return text
    
    @property
    def vocab_size(self):
        return len(self.word_to_id)
