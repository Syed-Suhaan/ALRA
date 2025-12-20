import streamlit as st
import os
import shutil
import streamlit_shadcn_ui as ui
from src.ingest import load_and_process_pdfs, create_vector_db, load_vector_db
from src.engine import get_answer
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Auto-LitReview Agent", layout="wide")

st.title("Auto-LitReview Agent (ALRA)")
st.markdown("### AI-Powered Research Assistant with Confidence Scoring")

# Sidebar for Setup
with st.sidebar:
    st.header("Document Setup")
    uploaded_files = st.file_uploader("Upload Research Papers (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if ui.button("Process PDFs", key="process_btn"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    if os.path.exists("faiss_index"):
                        shutil.rmtree("faiss_index")
                        
                    chunks = load_and_process_pdfs(uploaded_files)
                    create_vector_db(chunks)
                    st.success(f"Processed {len(chunks)} chunks from {len(uploaded_files)} files.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
        else:
            st.warning("Please upload files first.")

    st.markdown("---")
    st.markdown("**Core Features:**")
    st.markdown("- Semantic Search (FAISS)")
    st.markdown("- Confidence Scoring")
    st.markdown("- Groq Llama3 Integration")
    
    if ui.button("Clear Conversation", variant="destructive", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    
    with st.expander("Benchmark Dashboard"):
        st.info("Run a quick evaluation on your current knowledge base.")
    
        with st.form("benchmark_form"):
            st.subheader("1. Positive Test Case")
            pos_q = st.text_input("Question (expecting answer)", value="What is the main subject?")
            pos_kw = st.text_input("Expected Keyword", value="subject")
            
            st.subheader("2. Negative Test Case")
            neg_q = st.text_input("Irrelevant Question", value="What is the recipe for lasagna?")
            
            run_bench = st.form_submit_button("Run Evaluation")
        
        if run_bench:
            if not os.path.exists("faiss_index"):
                st.error("Please process PDFs first!")
            else:
                with st.spinner("Running Benchmark..."):
                    vector_store = load_vector_db()
                    
                    ans_pos, conf_pos, _ = get_answer(vector_store, pos_q)
                    pass_pos_conf = conf_pos > 60
                    
                    keywords = [k.strip().lower() for k in pos_kw.split(",")]
                    pass_pos_kw = any(k in ans_pos.lower() for k in keywords)
                    
                    ans_neg, conf_neg, _ = get_answer(vector_store, neg_q)
                    pass_neg_conf = conf_neg < 50
                    pass_neg_warn = "warning" in ans_neg.lower().split(":")[0]
                    
                    st.write("---")
                    st.markdown("### Evaluation Results")
                    
                    c1 = "green" if pass_pos_conf else "red"
                    c2 = "green" if pass_neg_conf else "red"
                    st.markdown(f"**Confidence Calibration**")
                    st.markdown(f"- Positive Query: :{c1}[{conf_pos:.1f}%] (Goal: >60%)")
                    st.markdown(f"- Negative Query: :{c2}[{conf_neg:.1f}%] (Goal: <50%)")
                    
                    k1 = "green" if pass_pos_kw else "red"
                    k2 = "green" if pass_neg_warn else "red"
                    st.markdown(f"**Answer Quality**")
                    st.markdown(f"- Keyword Match: :{k1}[{'Yes' if pass_pos_kw else 'No'}]")
                    st.markdown(f"- Hallucination Check: :{k2}[{'Passed' if pass_neg_warn else 'Failed'}] ({'Warning found' if pass_neg_warn else 'No warning'})")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        vector_store = load_vector_db()
        if not vector_store:
            st.error("No knowledge base found. Please upload and process PDFs first.")
        else:
            with st.spinner("Analyzing papers..."):
                try:
                    response, confidence, raw_results = get_answer(vector_store, prompt)
                    
                    if confidence >= 75:
                        label = "High Confidence"
                        desc = "Strong Evidence Found"
                    elif confidence >= 50:
                        label = "Medium Confidence"
                        desc = "Partial Match"
                    else:
                        label = "Low Confidence"
                        desc = "Speculative Answer"
                        
                    ui.metric_card(
                        title=f"{confidence:.1f}%",
                        content=label,
                        description=desc,
                        key=f"metric_{len(st.session_state.messages)}"
                    )
                    
                    st.markdown(response)
                    
                    with st.expander("View Source Evidence"):
                        for i, (doc, score) in enumerate(raw_results):
                            st.markdown(f"**Chunk {i+1} (Score: {1/(1+score):.2f})**")
                            st.caption(f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
                            st.text(doc.page_content[:300] + "...")
                            st.write("---")
                            
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error during retrieval: {str(e)}")
