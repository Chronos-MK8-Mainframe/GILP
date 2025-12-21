
import os
import re
import torch

class TPTPParser:
    def __init__(self, tptp_root):
        self.tptp_root = tptp_root
        self.parsed_files = set()

    def parse_file(self, file_path, kb):
        """
        Parses a TPTP file and populates the KnowledgeBase.
        Handles: individual files, recursion via 'include'.
        """
        # Resolve path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.tptp_root, file_path)
        
        if file_path in self.parsed_files:
            return
        self.parsed_files.add(file_path)

        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove comments (lines starting with %)
        # content = re.sub(r'(^|\n)%[^\n]*', '', content) 
        # Better: remove comments but keep newlines to avoid merging lines incorrectly, 
        # though TPTP is largely whitespace insensitive.
        
        # Tokenizer-like approach for robust parsing of balanced parentheses
        tokens = self._tokenize(content)
        self._parse_statements(tokens, kb)

    def _tokenize(self, content):
        """
        Splits content into tokens, removing comments.
        """
        tokens = []
        i = 0
        n = len(content)
        while i < n:
            char = content[i]
            if char == '%': # Comment start
                while i < n and content[i] != '\n':
                    i += 1
            elif char.isspace():
                i += 1
            elif char in "(),.":
                tokens.append(char)
                i += 1
            elif char == "'": # Quoted string
                start = i
                i += 1
                while i < n and content[i] != "'":
                     if content[i] == '\\': i += 1 # escape
                     i += 1
                i += 1 # skip closing quote
                tokens.append(content[start:i])
            else: # Alphanumeric / Symbol
                start = i
                while i < n and (content[i].isalnum() or content[i] in "_-"): # Simple ident
                     i += 1
                # If we hit a non-ident char that is not space/special, consume it? 
                # TPTP allows ~ ! ? etc.
                if start == i:
                    # Capture symbols
                     tokens.append(content[i])
                     i += 1
                else:
                    tokens.append(content[start:i])
        return tokens

    def _parse_statements(self, tokens, kb):
        """
        Recursive descent parser for TPTP statements.
        Statement: name(arg1, arg2, ...).
        """
        i = 0
        n = len(tokens)
        
        while i < n:
            token = tokens[i]
            
            if token in ['fof', 'cnf']:
                i = self._parse_formula(tokens, i, kb, token)
            elif token == 'include':
                i = self._parse_include(tokens, i, kb)
            else:
                # Unknown or syntax error, skip to next '.'
                # print(f"Skipping unknown token: {token}")
                while i < n and tokens[i] != '.':
                    i += 1
                i += 1
                
    def _parse_include(self, tokens, start_idx, kb):
        # include('path').
        i = start_idx + 1 # skip 'include'
        if tokens[i] != '(': return i
        i += 1
        path = tokens[i].strip("'")
        i += 1
        if tokens[i] == ',': # Optional selection args? include('A', [a,b]).
             # Skip until ')'
             while i < len(tokens) and tokens[i] != ')':
                 i += 1
        
        if tokens[i] == ')':
            i += 1
        if tokens[i] == '.':
            i += 1
            
        # Recursive call
        # print(f"Including: {path}")
        self.parse_file(path, kb)
        return i

    def _parse_formula(self, tokens, start_idx, kb, form_type):
        # fof(name, role, formula).
        i = start_idx + 1
        if tokens[i] != '(': return i
        i += 1
        
        name = tokens[i]
        i += 1
        if tokens[i] == ',': i += 1
        
        role = tokens[i]
        i += 1
        if tokens[i] == ',': i += 1
        
        # Capture formula until matching paren
        formula_tokens = []
        current_paren_balance = 0
        
        while i < len(tokens):
            t = tokens[i]
            if t == '(':
                current_paren_balance += 1
            elif t == ')':
                current_paren_balance -= 1
                
            if current_paren_balance < 0:
                break
            
            formula_tokens.append(t)
            i += 1
            
        formula_str = " ".join(formula_tokens)
        
        # Register in KB
        rule = kb.add_rule(name=name, complexity=1.0)
        rule.content = formula_str 
        rule.rule_type = role
        rule.tokens = formula_tokens # NEW: Store tokens for transformer
        
        if tokens[i] == ')': i += 1
        if tokens[i] == '.': i += 1
        return i

    def build_vocab(self, kb, min_freq=1):
        """
        Builds vocabulary from all rules in KB.
        Returns: word2idx dictionary
        """
        freqs = {}
        for rule in kb.rules.values():
            if hasattr(rule, 'tokens'):
                for t in rule.tokens:
                    freqs[t] = freqs.get(t, 0) + 1
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for t, f in freqs.items():
            if f >= min_freq:
                vocab[t] = len(vocab)
        return vocab

    def encode_rules(self, kb, vocab, max_len=128):
        """
        Encodes rule tokens into integer sequences.
        Returns: Tensor [Num_Rules, Max_Len]
        """
        sequences = []
        for i in range(len(kb.rules)):
            rule = kb.get_rule(i)
            seq = [vocab.get(t, 1) for t in getattr(rule, 'tokens', [])] # 1 is UNK
            
            # Pad or Trim
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [0] * (max_len - len(seq)) # 0 is PAD
            sequences.append(seq)
            
        return torch.tensor(sequences, dtype=torch.long)
