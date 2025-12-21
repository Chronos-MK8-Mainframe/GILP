import threading
import time
from typing import List, Dict
from gilp_core.data.knowledge_base import KnowledgeBase
from gilp_core.data.logic_extractor import LogicExtractor

class AutonomousCrawler:
    def __init__(self, kb: KnowledgeBase, extractor: LogicExtractor):
        self.kb = kb
        self.extractor = extractor
        self.running = False
        self.thread = None
        
    def start(self, topics: List[str], interval: int = 10):
        """
        v10: Starts a background thread that 'researches' topics.
        """
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(topics, interval), daemon=True)
        self.thread.start()
        print(f"[v10 Crawler] Background research started for topics: {topics}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print("[v10 Crawler] Background research stopped.")
        
    def _run(self, topics, interval):
        while self.running:
            for topic in topics:
                if not self.running: break
                print(f"[v10 Crawler] Synthesizing knowledge for: {topic}...")
                
                # In a real v10, this would use a web-search tool. 
                # For this prototype, we simulate research text.
                mock_text = f"New discovery in {topic}: It has been found that {topic} is directly dependent on AdvancedLogic and it contradicts Stagnation."
                
                rules = self.extractor.extract_rules(mock_text)
                if rules:
                    self.kb.ingest_extracted_rules(rules)
                    print(f"[v10 Crawler] Ingested {len(rules)} new rules for {topic}.")
                
                time.sleep(interval)
