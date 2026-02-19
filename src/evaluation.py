import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

EVAL_LOG_FILE = "eval_logs.json"

class EvaluationLogger:
    """
    Simples JSON logger for retrieval performance and evaluation metrics.
    """
    def __init__(self, log_file=EVAL_LOG_FILE):
        self.log_file = log_file

    def log_interaction(self, 
                        query: str, 
                        answer: str, 
                        grounding_score: float, 
                        retrieval_sim: float,
                        citation_cov: float,
                        source_overlap: float,
                        risk: float,
                        timestamp: Optional[float] = None):
        
        entry = {
            "timestamp": timestamp or time.time(),
            "query": query,
            "answer_length": len(answer),
            "grounding_score": grounding_score,
            "metrics": {
                "retrieval_similarity": retrieval_sim,
                "citation_coverage": citation_cov,
                "source_overlap": source_overlap,
                "hallucination_risk": risk
            }
        }
        
        self._append_to_log(entry)

    def _append_to_log(self, entry: Dict[str, Any]):
        data = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
            except Exception:
                pass
        
        data.append(entry)
        
        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_logs(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

# Evaluation Metrics for Benchmark

def compute_recall_at_k(retrieved_docs: List[Any], relevant_docs: List[Any], k: int) -> float:
    """
    Compute Recall@K.
    Requires ground truth relevant_docs (e.g., specific paper titles or section types).
    Here implemented as: Ratio of relevant documents found in top K.
    """
    if not relevant_docs:
        return 0.0
        
    retrieved_set = {d.metadata.get("source") for d in retrieved_docs[:k]}
    relevant_set = {d.metadata.get("source") for d in relevant_docs}
    
    intersection = retrieved_set.intersection(relevant_set)
    return len(intersection) / len(relevant_set)

def compute_faithfulness(answer: str, context: str) -> float:
    """
    Compute Faithfulness Score (simplified).
    A proxy metric: checks if key nouns/entities in answer appear in context.
    Real implementation would use LLM ("Is statement supported by context?").
    """
    # Simple overlap check for demonstration
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    
    if not answer_words:
        return 0.0
        
    overlap = len(answer_words.intersection(context_words))
    return overlap / len(answer_words)

