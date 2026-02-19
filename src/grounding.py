import numpy as np
import re
from dataclasses import dataclass, asdict
from typing import List, Optional
from langchain_core.documents import Document

@dataclass
class GroundingResult:
    overall_score: float  # 0-100
    retrieval_similarity: float  # 0-100
    citation_coverage: float  # 0-100
    source_overlap: float  # 0-100
    hallucination_risk: float  # 0-100 (inverse of risk, so higher = safer)
    explanation: str

def compute_retrieval_similarity(distances: List[float]) -> float:
    """
    Convert FAISS L2 distances to 0-100 similarity score.
    Lower distance = higher similarity.
    """
    if not distances:
        return 0.0
    
    # Simple conversion: 1 / (1 + distance)
    # Weighted average of top K
    sims = [1 / (1 + (d * 0.7)) for d in distances]
    
    if len(sims) == 1:
        weights = [1.0]
    elif len(sims) == 2:
        weights = [0.7, 0.3]
    else:
        weights = [0.6, 0.3, 0.1] + [0.0] * (len(sims) - 3)
        weights = weights[:len(sims)]
        # Normalize weights
        total = sum(weights)
        weights = [w/total for w in weights]
        
    score = np.average(sims, weights=weights)
    return float(score * 100.0)

def compute_citation_coverage(answer: str, docs: List[Document]) -> float:
    """
    Check if the answer cites the retrieved sources.
    Looks for [Source: filename] patterns.
    """
    if not docs:
        return 0.0
        
    cited_count = 0
    # Extract unique source names from docs
    available_sources = set()
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src:
            available_sources.add(src)
            
    if not available_sources:
        return 0.0

    # Check how many form available_sources are mentioned in answer
    # Simple check for filename substring
    lower_answer = answer.lower()
    for src in available_sources:
        if src.lower() in lower_answer:
            cited_count += 1
            
    # If partial match or "Source:" tag present without specific filename match (hallucinated filename?)
    # Actually, let's just count ratio of cited vs available top-3
    # If answer cites at least 1 valid source, give points.
    
    # strict citation check
    coverage = cited_count / len(available_sources)
    
    # Bonus if ALL are cited? No, rarely happens.
    # Cap at 1.0
    return min(100.0, coverage * 100.0)

def compute_source_overlap(answer: str, docs: List[Document], query: str) -> float:
    """
    Check lexical overlap (unigrams/bigrams) between answer and source content.
    Excludes query terms to avoid cheating (answering with query words).
    """
    if not docs:
        return 0.0
        
    combined_source_text = " ".join([d.page_content.lower() for d in docs[:3]])
    answer_tokens = set(re.findall(r'\w+', answer.lower()))
    source_tokens = set(re.findall(r'\w+', combined_source_text))
    query_tokens = set(re.findall(r'\w+', query.lower()))
    
    # Filter out stopwords and query words from answer tokens
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
    relevant_answer_tokens = answer_tokens - stopwords - query_tokens
    
    if not relevant_answer_tokens:
        return 0.0
        
    overlap_count = len(relevant_answer_tokens.intersection(source_tokens))
    overlap_ratio = overlap_count / len(relevant_answer_tokens)
    
    return min(100.0, overlap_ratio * 100.0)

def compute_grounding_score(query: str, answer: str, docs: List[Document], distances: List[float]) -> GroundingResult:
    """
    Calculate composite grounding score (Confidence 2.0).
    Weighted sum of signals.
    """
    
    # 1. Retrieval Similarity (30%)
    retrieval_sim = compute_retrieval_similarity(distances)
    
    # 2. Citation Coverage (20%)
    citation_cov = compute_citation_coverage(answer, docs)
    
    # 3. Source Overlap (30%)
    source_ov = compute_source_overlap(answer, docs, query)
    
    # 4. Hallucination Risk (20%) - Simplified heuristic for now (check for warning)
    # If answer starts with "Warning:", automatic penalty
    hallucination_safe = 100.0
    if "warning:" in answer.lower()[:50]:
        hallucination_safe = 20.0
    
    # Weighted Sum
    # Weights: Sim=0.4, Cov=0.2, Overlap=0.2, Safety=0.2
    overall = (retrieval_sim * 0.4) + (citation_cov * 0.2) + (source_ov * 0.2) + (hallucination_safe * 0.2)
    
    # Explanation
    explanation = "High confidence"
    if overall < 50:
        explanation = "Low confidence: Weak retrieval match or missing citations."
    elif overall < 75:
        explanation = "Medium confidence: Good retrieval but partial overlap."
        
    return GroundingResult(
        overall_score=overall,
        retrieval_similarity=retrieval_sim,
        citation_coverage=citation_cov,
        source_overlap=source_ov,
        hallucination_risk=hallucination_safe,
        explanation=explanation
    )
