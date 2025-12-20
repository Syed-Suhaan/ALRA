import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils import calculate_confidence_score
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    # Using Llama 3.3 70b as the new default
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

def search_with_confidence(vector_store, query, k=5):
    """
    Search vector db and calculate confidence score.
    """
    results_with_score = vector_store.similarity_search_with_score(query, k=k)
    
    distances = []
    docs = []
    for doc, score in results_with_score:
        distances.append(score)
        docs.append(doc)
        
    confidence_score = calculate_confidence_score(distances, query=query, docs=docs)
    
    context_text = "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')} page {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content}" for doc, score in results_with_score])
    
    return context_text, confidence_score, results_with_score

def get_answer(vector_store, query):
    llm = get_llm()
    
    context, confidence, raw_results = search_with_confidence(vector_store, query)
    
    system_prompt = """You are a Research Analysis Agent.
context_confidence_score: {confidence_score:.2f}%

Instructions:
1. Answer the user's question using ONLY the provided context.
2. If 'context_confidence_score' is below 50%, start your answer with: 'Warning: The available documents do not contain a strong match for this query, but based on partial information...'
3. Reference your sources using the [Source: filename page X] format provided in the context.

Context:
{context}

Question:
{question}
"""
    
    prompt = PromptTemplate(
        input_variables=["confidence_score", "context", "question"],
        template=system_prompt
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "confidence_score": confidence, 
        "context": context, 
        "question": query
    })
    
    return response, confidence, raw_results
