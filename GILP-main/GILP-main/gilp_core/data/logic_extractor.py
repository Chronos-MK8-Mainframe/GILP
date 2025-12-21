import re
import json
from typing import List, Dict

class LogicExtractor:
    def __init__(self, explainer):
        """
        explainer: ReasoningExplainer instance (provides LLM access).
        """
        self.explainer = explainer

    def extract_rules(self, text: str) -> List[Dict]:
        """
        Transforms raw text into a list of logical rules (triples).
        Format: {"premise": "A", "conclusion": "B", "type": "dependency/contradiction"}
        """
        prompt = f"""<|im_start|>system
You are a logical structure extractor. Read the provided text and extract the core logical relationships.
Focus on causal links, prerequisites, and contradictions.
Return ONLY a JSON list of objects. No intro, no outro, no markdown formatting.
Example output: [{{ "source": "Sunlight", "target": "Photosynthesis", "type": "dependency" }}]
<|im_end|>
<|im_start|>user
Extract logic from: {text}<|im_end|>
<|im_start|>assistant
"""
        response = self.explainer.llm(
            prompt,
            max_tokens=512,
            stop=["<|im_end|>"],
            echo=False
        )
        
        raw_output = response["choices"][0]["text"].strip()
        print(f"\n[DEBUG] Raw LLM Output: {raw_output}\n")
        
        # Robust JSON cleaning
        if isinstance(raw_output, str) and "[" in raw_output:
            try:
                start = raw_output.find("[")
                end = raw_output.rfind("]") + 1
                json_data = raw_output[start:end]
                rules = json.loads(json_data)
                
                # Further validation: ensure it's a list of dicts
                if isinstance(rules, dict):
                    # Sometimes LLM wraps in an object e.g. {"objects": [...]}
                    for key in ["objects", "rules", "relationships", "triples"]:
                        if key in rules and isinstance(rules[key], list):
                            rules = rules[key]
                            break
                    if isinstance(rules, dict): # Still a dict? 
                        rules = [rules] # Wrap single object in list
                
                if isinstance(rules, list):
                    return [r for r in rules if isinstance(r, dict)]
            except Exception as e:
                print(f"  [LogicExtractor] Error parsing JSON: {e}")
        
        return self._fallback_regex(raw_output)

    def _fallback_regex(self, text: str) -> List[Dict]:
        rules = []
        # Look for simpler patterns if JSON fails
        # 1. Matches "A leads to B" or "A causes B" or "A -> B"
        logic_patterns = [
            (r'(\w+)\s+(?:leads to|causes|requires|affects|influences)\s+(\w+)', 'dependency'),
            (r'(\w+)\s*->\s*(\w+)', 'dependency'),
            (r'(\w+)\s+(?:contradicts|annihilates|opposes)\s+(\w+)', 'contradiction'),
            (r'(\w+)\s*!=\s*(\w+)', 'contradiction')
        ]
        
        for pattern, rel_type in logic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for src, dst in matches:
                rules.append({"source": src, "target": dst, "type": rel_type})
        return rules

class ImageLogicExtractor(LogicExtractor):
    def extract_from_image(self, image_description: str) -> List[Dict]:
        """
        v9: Multimodal extraction. 
        Given a natural language description of an image (e.g. from a vision model),
        extract logical relationships.
        """
        prompt = f"An image shows: {image_description}. What are the logical dependencies or contradictions? Answer in JSON list format."
        return self.extract_rules(prompt)
