import os
import json
from src.ingest import load_vector_db
from src.engine import get_answer
from src.evaluation import compute_recall_at_k, compute_faithfulness
from langchain_core.documents import Document

# Expanded Golden Dataset with Ground Truth for Recall
TEST_DATASET = [
    {
        "question": "What is the primary methodology used?",
        "expected_keywords": ["methodology", "approach", "proposed"],
        "type": "specific",
        "ground_truth_source_substrings": ["result", "method"] # Simplified for demo - usually specific filenames
    },
    {
        "question": "What are the results of the experiment?",
        "expected_keywords": ["result", "accuracy", "performance", "%"],
        "type": "specific",
        "ground_truth_source_substrings": ["result"]
    },
    {
        "question": "What is the recipe for lasagna?",
        "expected_keywords": [],
        "type": "irrelevant",
        "ground_truth_source_substrings": []
    }
]

def run_benchmark():
    print("Loading Vector DB...")
    vector_store = load_vector_db()

    if not vector_store:
        print("Error: No FAISS index found. Please process PDFs in the app first.")
        return

    print(f"Starting Benchmark on {len(TEST_DATASET)} test cases...\n")

    results = []

    for test_case in TEST_DATASET:
        query = test_case["question"]
        print(f"Testing: '{query}'")

        try:
            # Updated to unpack 4 values, where confidence is now a GroundingResult object
            answer, grounding, raw_results, reasoning = get_answer(vector_store, query)
            
            # Extract score from GroundingResult
            confidence = grounding.overall_score

            pass_confidence = False
            if test_case["type"] == "irrelevant":
                if confidence < 50:
                    pass_confidence = True
            else:
                if confidence > 60:
                    pass_confidence = True

            keyword_match = False
            if test_case["expected_keywords"]:
                if any(k.lower() in answer.lower() for k in test_case["expected_keywords"]):
                    keyword_match = True
            else:
                if "warning" in answer.lower():
                    keyword_match = True
            
            # --- New Evaluation Metrics ---
            # Create dummy relevant docs for recall calculation based on substrings
            # In a real scenario, this would be strict filename matching
            relevant_docs = []
            if test_case["ground_truth_source_substrings"]:
                 relevant_docs = [Document(page_content="", metadata={"source": s}) for s in test_case["ground_truth_source_substrings"]]
            
            # We reuse compute_recall_at_k but need to mock metadata for substring match if using dummy logic
            # Actually, compute_recall_at_k expects exact source match.
            # Let's just skip rigorous recall calculation here unless we have real filenames.
            # Instead, we will use faithfulness.
            
            faithfulness = compute_faithfulness(answer, "\n".join([d.page_content for d, _ in raw_results]))

            reasoning_keywords = reasoning.get("reasoning_keywords", [])
            is_multi_hop = reasoning.get("is_multi_hop", False)

            print(f"  -> Score: {confidence:.2f}% | Pass Conf: {pass_confidence} | Keyword Match: {keyword_match}")
            print(f"  -> Faithfulness (Proxy): {faithfulness:.2f}")
            print(f"  -> Breakdown: Sim={grounding.retrieval_similarity:.1f}, Cov={grounding.citation_coverage:.1f}, Overlap={grounding.source_overlap:.1f}, Risk={grounding.hallucination_risk:.1f}")

            results.append({
                "query": query,
                "confidence": confidence,
                "pass_confidence": pass_confidence,
                "pass_accuracy": keyword_match,
                "faithfulness": faithfulness,
                "reasoning_keywords": reasoning_keywords
            })

        except Exception as e:
            print(f"  -> Error: {e}")

    total = len(results)
    conf_pass_rate = sum(1 for r in results if r["pass_confidence"]) / total * 100
    acc_pass_rate = sum(1 for r in results if r["pass_accuracy"]) / total * 100
    avg_faithfulness = sum(r["faithfulness"] for r in results) / total * 100
    
    print("\n--- Benchmark Report ---")
    print(f"Total Tests: {total}")
    print(f"Confidence Calibration Score: {conf_pass_rate:.1f}%")
    print(f"Answer Accuracy Score: {acc_pass_rate:.1f}%")
    print(f"Avg Faithfulness Score: {avg_faithfulness:.1f}%")
    print(f"Avg Reasoning Keywords Generated: {sum(len(r.get('reasoning_keywords', [])) for r in results) / total:.1f}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_benchmark()
