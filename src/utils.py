import numpy as np

def calculate_confidence_score(distances, query=None, docs=None):
    """
    Convert FAISS L2 distances to a percentage confidence score.
    Logic:
    1. Convert L2 distance to similarity: similarity = 1 / (1 + distance)
    2. Normalize to 0-100% scale.
    3. Take weighted average of top results.
    4. Apply Boost if exact query (or huge part) is found in text.
    """
    if not distances or len(distances) == 0:
        return 0.0

    similarities = []
    for d in distances:
        # Avoid division by zero, though L2 is >= 0
        # Tuning: Scale distance by 0.7 to be slightly stricter than 0.5
        # but still lenient enough for L2 (unnormalized)
        similarity = 1 / (1 + (d * 0.7))
        similarities.append(similarity)

    # Weighted average: give more weight to the top result
    count = len(similarities)
    if count == 1:
        weights = [1.0]
    elif count == 2:
        weights = [0.7, 0.3]
    else:
        similarities = similarities[:3]
        weights = [0.6, 0.3, 0.1]
        
    score = np.average(similarities, weights=weights)
    score_percent = score * 100.0
    
    # Exact Match Boost
    if query and docs:
        query_norm = query.lower().strip().replace("?", "").replace(".", "")
        # Check top doc
        top_doc_content = docs[0].page_content.lower()
        
        # 1. Direct substring match
        if query_norm in top_doc_content:
            score_percent += 20.0
        
        # 2. Keyword overlap (simple heuristic)
        # If > 80% of unique query words exist in doc
        query_words = set(query_norm.split())
        doc_words = set(top_doc_content.split())
        if query_words:
            overlap = len(query_words.intersection(doc_words)) / len(query_words)
            if overlap > 0.8:
                 score_percent += 10.0
    
    # Cap at 100
    return min(100.0, score_percent)
