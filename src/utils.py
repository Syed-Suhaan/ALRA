import numpy as np

def calculate_confidence_score(distances, query=None, docs=None):
   
    if not distances or len(distances) == 0:
        return 0.0

    similarities = []
    for d in distances:
        similarity = 1 / (1 + (d * 0.7))
        similarities.append(similarity)

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
    
    if query and docs:
        query_norm = query.lower().strip().replace("?", "").replace(".", "")
        top_doc_content = docs[0].page_content.lower()
        
        if query_norm in top_doc_content:
            score_percent += 20.0
        
        query_words = set(query_norm.split())
        doc_words = set(top_doc_content.split())
        if query_words:
            overlap = len(query_words.intersection(doc_words)) / len(query_words)
            if overlap > 0.8:
                 score_percent += 10.0
    
    return min(100.0, score_percent)
