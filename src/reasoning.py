import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

REASONING_PROMPT = """You are a research query analysis engine. Given a user's question about research papers, your job is to:

1. Identify the core intent of the question
2. Generate reasoning keywords — related concepts, methods, or entities that are implied but not explicitly stated
3. Decompose the query into sub-questions if it requires multi-hop reasoning
4. Produce an expanded search query that combines the original question with inferred terms

User Query: {query}

Respond ONLY with valid JSON in this exact format:
{{
    "core_intent": "one sentence describing what the user really wants to know",
    "reasoning_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "sub_queries": ["sub-question 1", "sub-question 2"],
    "expanded_query": "the original query enriched with inferred terms for better semantic search",
    "is_multi_hop": true or false
}}

Rules:
- Generate 3-7 reasoning keywords that are semantically related but not in the original query
- Only create sub_queries if the question genuinely requires multiple retrieval steps
- The expanded_query should be a natural sentence, not just keywords concatenated
- Think about what a researcher would need to find to answer this question completely
"""


def get_reasoning_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )


def expand_query(query):
    """
    Expand a user query using LLM-based reasoning to improve retrieval.
    Returns a dict with reasoning_keywords, sub_queries, expanded_query, etc.
    Falls back to returning the original query if expansion fails.
    """
    try:
        llm = get_reasoning_llm()

        prompt = PromptTemplate(
            input_variables=["query"],
            template=REASONING_PROMPT
        )

        chain = prompt | llm | StrOutputParser()
        raw_response = chain.invoke({"query": query})

        raw_response = raw_response.strip()
        if raw_response.startswith("```"):
            raw_response = raw_response.split("\n", 1)[1]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            raw_response = raw_response.strip()

        result = json.loads(raw_response)

        required_keys = ["core_intent", "reasoning_keywords", "sub_queries", "expanded_query"]
        for key in required_keys:
            if key not in result:
                result[key] = [] if key in ("reasoning_keywords", "sub_queries") else query

        if "is_multi_hop" not in result:
            result["is_multi_hop"] = len(result.get("sub_queries", [])) > 1

        result["original_query"] = query

        return result

    except (json.JSONDecodeError, Exception) as e:
        return {
            "original_query": query,
            "core_intent": query,
            "reasoning_keywords": [],
            "sub_queries": [],
            "expanded_query": query,
            "is_multi_hop": False,
            "error": str(e)
        }


def get_search_query(reasoning_result):
    """
    Build the final search query string from reasoning results.
    Uses the expanded query if available, otherwise falls back to original.
    """
    expanded = reasoning_result.get("expanded_query", "")
    original = reasoning_result.get("original_query", "")

    if expanded and expanded != original:
        return expanded
    return original
