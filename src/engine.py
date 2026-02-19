import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.grounding import compute_grounding_score, compute_retrieval_similarity
from src.reasoning import expand_query, get_search_query
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

def search_with_context(vector_store, query, k=5):
    """
    Search and prepare context string using reasoning and semantic metadata.
    Returns: context_text, pre_gen_confidence, results_with_score, reasoning_result
    """
    reasoning_result = expand_query(query)
    search_query = get_search_query(reasoning_result)

    results_with_score = vector_store.similarity_search_with_score(search_query, k=k)

    distances = []
    docs = []
    processed_context = []
    
    for doc, score in results_with_score:
        distances.append(score)
        docs.append(doc)
        
        # Build context string with semantic tags
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        section = doc.metadata.get('section_type', 'other').upper()
        title = doc.metadata.get('paper_title', '')
        
        header = f"[Source: {source} | Section: {section} | Page: {page}]"
        if title and title != "Unknown":
            header = f"[Source: {source} ('{title}') | Section: {section} | Page: {page}]"
            
        processed_context.append(f"{header}\n{doc.page_content}")

    # Use retrieval similarity for prompt guidance (pre-generation confidence)
    pre_gen_confidence = compute_retrieval_similarity(distances)
    
    context_text = "\n\n".join(processed_context)

    return context_text, pre_gen_confidence, results_with_score, reasoning_result

def get_answer(vector_store, query):
    llm = get_llm()

    context, pre_gen_confidence, raw_results, reasoning = search_with_context(vector_store, query)

    reasoning_info = ""
    if reasoning.get("reasoning_keywords"):
        reasoning_info = f"\nReasoning Keywords: {', '.join(reasoning['reasoning_keywords'])}"
    if reasoning.get("sub_queries"):
        reasoning_info += f"\nSub-questions identified: {'; '.join(reasoning['sub_queries'])}"

    system_prompt = """You are a Research Analysis Agent.
context_match_score: {pre_gen_confidence:.2f}%
{reasoning_info}

Instructions:
1. Answer the user's question using ONLY the provided context.
2. Pay attention to the [Section: TYPE] tags in the context to understand if the information comes from Results, Methodology, etc.
3. If 'context_match_score' is below 50%, start your answer with: 'Warning: The available documents do not contain a strong match for this query, but based on partial information...'
4. Reference your sources using the [Source: filename] format provided in the context.
5. If reasoning keywords were provided, ensure your answer addresses the broader context they suggest.

Context:
{context}

Question:
{question}
"""

    prompt = PromptTemplate(
        input_variables=["pre_gen_confidence", "reasoning_info", "context", "question"],
        template=system_prompt
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "pre_gen_confidence": pre_gen_confidence,
        "reasoning_info": reasoning_info,
        "context": context,
        "question": query
    })

    # Compute full grounding score post-generation
    distances = [doc_score[1] for doc_score in raw_results]
    docs = [doc_score[0] for doc_score in raw_results]
    grounding_result = compute_grounding_score(query, response, docs, distances)

    return response, grounding_result, raw_results, reasoning
