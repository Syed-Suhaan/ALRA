import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

@dataclass
class SynthesisResult:
    claims_table: Dict[str, str]  # Paper -> Claim
    method_comparison: Dict[str, str] # Paper -> Method
    results_summary: Dict[str, str] # Paper -> Results
    synthesis_summary: str
    contradictions: List[str]

SYNTHESIS_PROMPT = """You are a research synthesis engine.
Given a collection of excerpts from multiple papers about a topic, your task is to synthesize them into a structured comparison.

Topic: {topic}

Documents:
{context}

Instructions:
1. Identify the distinct papers mentioned (group by [Source: Title] or [Source: Filename]).
2. For EACH paper, extract:
   - Key Claim related to the topic
   - Methodology used
   - Main Result/Finding
3. Identify any contradictions or disagreements between the papers.
4. Write a brief synthesis summary weaving the findings together.

Respond ONLY with valid JSON in this structure:
{{
    "comparison": [
        {{
            "paper": "Paper Title 1 (or Filename)",
            "claim": "Extracted claim...",
            "method": "Method used...",
            "result": "Key result..."
        }},
        {{
            "paper": "Paper Title 2...",
            ...
        }}
    ],
    "contradictions": ["Contradiction 1...", "Contradiction 2..."],
    "summary": "A cohesive paragraph summarizing the landscape..."
}}
"""

def get_synthesis_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found.")
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile", # Using larger model for complex synthesis
        temperature=0.3
    )

def synthesize_papers(vector_store, topic: str, k=15) -> SynthesisResult:
    """
    Retrieve broad context and synthesize a structured comparison across papers.
    """
    llm = get_synthesis_llm()
    
    # search with expanded query? Yes, let's use broad search.
    # We can reuse expand_query logic or just plain search.
    # For synthesis, we want diversity.
    
    results = vector_store.similarity_search(topic, k=k)
    
    # Build context grouped by paper if possible
    context_parts = []
    for doc in results:
        meta = doc.metadata
        title = meta.get("paper_title") or meta.get("source", "Unknown")
        section = meta.get("section_type", "general")
        context_parts.append(f"--- Paper: {title} (Section: {section}) ---\n{doc.page_content}\n")
        
    context_text = "\n".join(context_parts)
    
    prompt = PromptTemplate(
        input_variables=["topic", "context"],
        template=SYNTHESIS_PROMPT
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        raw_response = chain.invoke({"topic": topic, "context": context_text})
        
        # Clean JSON
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        
        data = json.loads(cleaned)
        
        # Transform to SynthesisResult format
        claims = {}
        methods = {}
        results_map = {}
        
        for item in data.get("comparison", []):
            paper = item.get("paper", "Unknown")
            claims[paper] = item.get("claim", "")
            methods[paper] = item.get("method", "")
            results_map[paper] = item.get("result", "")
            
        return SynthesisResult(
            claims_table=claims,
            method_comparison=methods,
            results_summary=results_map,
            synthesis_summary=data.get("summary", ""),
            contradictions=data.get("contradictions", [])
        )
        
    except Exception as e:
        print(f"Synthesis error: {e}")
        return SynthesisResult({}, {}, {}, f"Error generating synthesis: {str(e)}", [])
