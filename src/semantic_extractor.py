import os
import re
import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

@dataclass
class SemanticChunk:
    content: str
    section_type: str  # objective, methodology, results, claims, limitations, other
    paper_title: Optional[str] = "Unknown"
    source: Optional[str] = "Unknown"
    page: Optional[int] = 0

SEMANTIC_PROMPT = """You are a research paper parser.
Analyze the following text chunk and classify it into ONE of these categories:
- objective (goals, problem statement, hypothesis)
- methodology (methods, data, setup, algorithms)
- results (findings, tables, metrics, performance)
- claims (arguments, main contributions, discussion)
- limitations (weaknesses, future work)
- other (references, headers, boilerplate)

Then, if possible, identify the probable paper title if it appears in the text (otherwise null).

Text Chunk:
{text_chunk}

Respond ONLY with valid JSON:
{{
    "section_type": "category_name",
    "paper_title": "extracted title or null"
}}
"""

def get_tagging_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )

def extract_semantic_sections(chunks: List[Document]) -> List[Document]:
    """
    Process a list of raw chunks and enrich them with semantic metadata.
    Uses LLM for classification. fallbacks to regex/heuristic if LLM fails or for speed.
    """
    llm = get_tagging_llm()
    enriched_docs = []
    
    # Simple regex fallback map
    fallback_map = {
        r"(?i)abstract|introduction|goal|objective": "objective",
        r"(?i)method|algorithm|setup|data": "methodology",
        r"(?i)result|performance|accuracy|table": "results",
        r"(?i)discussion|conclusion|claim": "claims",
        r"(?i)limitation|future work": "limitations"
    }

    print(f"Enriching {len(chunks)} chunks with semantic metadata...")

    if llm:
        try:
            prompt = PromptTemplate(
                input_variables=["text_chunk"],
                template=SEMANTIC_PROMPT
            )
            chain = prompt | llm | StrOutputParser()
        except Exception:
            chain = None
    else:
        chain = None

    for i, doc in enumerate(chunks):
        content = doc.page_content
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        
        section_type = "other"
        paper_title = None
        
        # Heuristic check first to save tokens - if very short, skip LLM
        if len(content) < 200:
            section_type = "other"
        elif chain:
            try:
                # Rate limit protection: simple sleep every 10 calls if needed
                if i > 0 and i % 10 == 0:
                    time.sleep(1)

                response = chain.invoke({"text_chunk": content[:1500]}) # Limit context window usage
                
                cleaned_response = response.strip()
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response.split("\n", 1)[1]
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3]
                
                data = json.loads(cleaned_response)
                section_type = data.get("section_type", "other").lower()
                paper_title = data.get("paper_title")

            except Exception as e:
                # Fallback on error
                # print(f"Error classifying chunk {i}: {e}")
                pass
        
        # Fallback if section_type is still default or empty
        if section_type == "other" or not section_type:
            for pattern, stype in fallback_map.items():
                if re.search(pattern, content):
                    section_type = stype
                    break

        if paper_title == "null":
            paper_title = None

        doc.metadata["section_type"] = section_type
        if paper_title:
             doc.metadata["paper_title"] = paper_title
        
        enriched_docs.append(doc)
        
    return enriched_docs
