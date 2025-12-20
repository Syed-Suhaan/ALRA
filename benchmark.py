import os
import json
from src.ingest import load_vector_db
from src.engine import get_answer

# Define "Golden Dataset" (Line 6) - Keeping simplified
TEST_DATASET = [
    {
        "question": "What is the primary methodology used?", 
        "expected_keywords": ["methodology", "approach", "proposed"],
        "type": "specific"
    },
    {
        "question": "What are the results of the experiment?",
        "expected_keywords": ["result", "accuracy", "performance", "%"],
        "type": "specific"
    },
    {
        "question": "What is the recipe for lasagna?",
        "expected_keywords": [],
        "type": "irrelevant"
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
            answer, confidence, _ = get_answer(vector_store, query)
            
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

            print(f"  -> Score: {confidence:.2f}% | Pass Conf: {pass_confidence} | Keyword Match: {keyword_match}")
            
            results.append({
                "query": query,
                "confidence": confidence,
                "pass_confidence": pass_confidence,
                "pass_accuracy": keyword_match
            })
            
        except Exception as e:
            print(f"  -> Error: {e}")

    # Aggregation
    total = len(results)
    conf_pass_rate = sum(1 for r in results if r["pass_confidence"]) / total * 100
    acc_pass_rate = sum(1 for r in results if r["pass_accuracy"]) / total * 100
    
    print("\n--- Benchmark Report ---")
    print(f"Total Tests: {total}")
    print(f"Confidence Calibration Score: {conf_pass_rate:.1f}%")
    print(f"Answer Accuracy Score: {acc_pass_rate:.1f}%")

if __name__ == "__main__":
    # Ensure env is loaded
    from dotenv import load_dotenv
    load_dotenv()
    run_benchmark()
