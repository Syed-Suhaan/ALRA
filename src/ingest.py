import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from src.semantic_extractor import extract_semantic_sections

def load_and_process_pdfs(pdf_files):
    """
    Load PDFs using PyMuPDF (faster) and return chunks.
    Now includes Semantic Extraction step.
    """
    documents = []
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    
    for pdf_file in pdf_files:
        temp_path = os.path.join(temp_dir, pdf_file.name)
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())
            
        try:
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()
            for doc in docs:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = pdf_file.name
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
        finally:
            pass
            
    try:
        if os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
    except Exception:
        pass
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # --- Semantic Extraction Step ---
    print(f"Applying semantic extraction to {len(chunks)} chunks...")
    chunks = extract_semantic_sections(chunks)
    
    return chunks

def create_vector_db(chunks):
    """
    Create specific FAISS index from chunks and save it.
    Chunks now carry semantic metadata (section_type, paper_title).
    """
    if not chunks:
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_db():
    """
    Load existing FAISS index.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None
