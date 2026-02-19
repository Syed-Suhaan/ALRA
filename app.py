import streamlit as st
import os
import shutil
import pandas as pd
import streamlit_shadcn_ui as ui
from dataclasses import asdict
from src.ingest import load_and_process_pdfs, create_vector_db, load_vector_db
from src.engine import get_answer
from src.synthesis import synthesize_papers
from src.evaluation import EvaluationLogger, compute_recall_at_k
from dotenv import load_dotenv

load_dotenv()
logger = EvaluationLogger()

st.set_page_config(page_title="ALRA 2.0 — Auto-LitReview Agent", layout="wide")

st.title("Auto-LitReview Agent (ALRA) 2.0")

# Mode Toggle
mode = st.radio("Mode", ["Q&A Chat", "Multi-Paper Synthesis"], horizontal=True, label_visibility="collapsed")

if mode == "Q&A Chat":
    st.markdown("### AI-Powered Research Assistant with Query Reasoning & Grounding")
else:
    st.markdown("### Multi-Paper Synthesis & Comparison Engine")

with st.sidebar:
    st.header("Document Setup")
    uploaded_files = st.file_uploader("Upload Research Papers (PDF)", type=["pdf"], accept_multiple_files=True)

    if ui.button("Process PDFs", key="process_btn"):
        if uploaded_files:
            with st.spinner("Processing documents (Parsing semantic sections... this may take a moment)..."):
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
    st.markdown("**ALRA 2.0 Features:**")
    st.markdown("- 🧠 Query Reasoning Expansion")
    st.markdown("- 🔍 Semantic Search (FAISS)")
    st.markdown("- 📊 Grounding Score CoE")
    st.markdown("- 📑 Multi-Paper Synthesis")
    st.markdown("- 📈 Performance Logging")

    if ui.button("Clear Conversation", variant="destructive", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    
    # --- Performance History Tab ---
    with st.expander("📊 Performance History"):
        logs = logger.get_logs()
        if logs:
            df = pd.DataFrame(logs)
            st.markdown(f"**Total Queries:** {len(df)}")
            st.markdown(f"**Avg Grounding:** {df['grounding_score'].mean():.1f}%")
            
            # Extract metrics from dict
            metrics_df = pd.json_normalize(df['metrics'])
            if not metrics_df.empty:
                 st.markdown("**Avg Metrics:**")
                 st.caption(f"Sim: {metrics_df['retrieval_similarity'].mean():.1f} | Cov: {metrics_df['citation_coverage'].mean():.1f}")
            
            st.line_chart(df['grounding_score'], use_container_width=True)
        else:
            st.info("No interaction logs yet.")

    # --- Benchmark Dashboard ---
    with st.expander("⚡ Rapid Benchmark"):
        st.info("Run a quick evaluation on your knowledge base.")

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

                    ans_pos, conf_pos, _, reasoning_pos = get_answer(vector_store, pos_q)
                    # conf_pos is now GroundingResult object
                    pass_pos_conf = conf_pos.overall_score > 60

                    keywords = [k.strip().lower() for k in pos_kw.split(",")]
                    pass_pos_kw = any(k in ans_pos.lower() for k in keywords)

                    ans_neg, conf_neg, _, reasoning_neg = get_answer(vector_store, neg_q)
                    pass_neg_conf = conf_neg.overall_score < 50
                    pass_neg_warn = "warning" in ans_neg.lower().split(":")[0]

                    st.write("---")
                    st.markdown("### Evaluation Results")

                    c1 = "green" if pass_pos_conf else "red"
                    c2 = "green" if pass_neg_conf else "red"
                    st.markdown(f"**Confidence Calibration**")
                    st.markdown(f"- Positive Query: :{c1}[{conf_pos.overall_score:.1f}%] (Goal: >60%)")
                    st.markdown(f"- Negative Query: :{c2}[{conf_neg.overall_score:.1f}%] (Goal: <50%)")

                    k1 = "green" if pass_pos_kw else "red"
                    k2 = "green" if pass_neg_warn else "red"
                    st.markdown(f"**Answer Quality**")
                    st.markdown(f"- Keyword Match: :{k1}[{'Yes' if pass_pos_kw else 'No'}]")
                    st.markdown(f"- Hallucination Check: :{k2}[{'Passed' if pass_neg_warn else 'Failed'}] ({'Warning found' if pass_neg_warn else 'No warning'})")

                    st.markdown("**Reasoning Expansion (Positive Query)**")
                    if reasoning_pos.get("reasoning_keywords"):
                        st.markdown(f"- Keywords: {', '.join(reasoning_pos['reasoning_keywords'])}")

# --- QA Mode ---
if mode == "Q&A Chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        # Filter out synthesis messages if any (though keeping them separate might be better, simple for now)
        if message.get("role") in ["user", "assistant"]:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                
                if message.get("reasoning"):
                    with st.expander("🧠 Query Reasoning"):
                        r = message["reasoning"]
                        st.markdown(f"**Core Intent:** {r.get('core_intent', 'N/A')}")
                        if r.get("reasoning_keywords"):
                            st.markdown(f"**Reasoning Keywords:** {', '.join(r['reasoning_keywords'])}")
                        if r.get("sub_queries"):
                            st.markdown("**Sub-questions:**")
                            for sq in r["sub_queries"]:
                                st.markdown(f"- {sq}")
                        if r.get("is_multi_hop"):
                            st.caption("🔗 Multi-hop query detected")
                
                if message.get("grounding"):
                    g = message["grounding"]
                    with st.expander("📊 Confidence Score Details"):
                        cols = st.columns(4)
                        cols[0].metric("Overall", f"{g['overall_score']:.1f}%")
                        cols[1].metric("Retrieval Sim", f"{g['retrieval_similarity']:.1f}%")
                        cols[2].metric("Citation Cov", f"{g['citation_coverage']:.1f}%")
                        cols[3].metric("Source Overlap", f"{g['source_overlap']:.1f}%")
                        st.caption(f"Explanation: {g['explanation']}")

    if prompt := st.chat_input("Ask a question about the papers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            vector_store = load_vector_db()
            if not vector_store:
                st.error("No knowledge base found. Please upload and process PDFs first.")
            else:
                with st.spinner("Reasoning & analyzing papers..."):
                    try:
                        response, grounding, raw_results, reasoning = get_answer(vector_store, prompt)
                        
                        # Log interaction
                        logger.log_interaction(
                            query=prompt,
                            answer=response,
                            grounding_score=grounding.overall_score,
                            retrieval_sim=grounding.retrieval_similarity,
                            citation_cov=grounding.citation_coverage,
                            source_overlap=grounding.source_overlap,
                            risk=grounding.hallucination_risk
                        )
                        
                        confidence = grounding.overall_score

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
                        
                        with st.expander("📊 Confidence Score Details", expanded=True):
                            cols = st.columns(4)
                            cols[0].metric("Retrieval Sim", f"{grounding.retrieval_similarity:.1f}%", help="Semantic similarity of retrieved chunks")
                            cols[1].metric("Citation Coverage", f"{grounding.citation_coverage:.1f}%", help="Are retrieved sources cited?")
                            cols[2].metric("Source Overlap", f"{grounding.source_overlap:.1f}%", help="Keyword overlap between answer and source")
                            cols[3].metric("Hallucination Risk", f"{100 - grounding.hallucination_risk:.1f}%", help="Risk of fabricated content (lower is better)", delta_color="inverse")
                            st.info(f"**Analysis:** {grounding.explanation}")

                        with st.expander("🧠 Query Reasoning", expanded=False):
                            st.markdown(f"**Original Query:** {reasoning.get('original_query', prompt)}")
                            st.markdown(f"**Core Intent:** {reasoning.get('core_intent', 'N/A')}")
                            if reasoning.get("reasoning_keywords"):
                                st.markdown(f"**Reasoning Keywords:** {', '.join(reasoning['reasoning_keywords'])}")
                            
                            if reasoning.get("sub_queries"):
                                st.markdown("**Sub-questions:**")
                                for sq in reasoning["sub_queries"]:
                                    st.markdown(f"- {sq}")
                                    
                            if reasoning.get("is_multi_hop"):
                                st.info("🔗 Multi-hop query detected — reasoning expanded across sub-questions")

                        st.markdown(response)

                        with st.expander("View Source Evidence"):
                            for i, (doc, score) in enumerate(raw_results):
                                st.markdown(f"**Chunk {i+1} (Score: {1/(1+score):.2f})**")
                                section = doc.metadata.get('section_type', 'other').upper()
                                title = doc.metadata.get('paper_title', 'Unknown')
                                source = doc.metadata.get('source', 'Unknown')
                                page = doc.metadata.get('page', '?')
                                st.caption(f"Source: {source} | Page: {page}")
                                st.caption(f"Paper: {title} | Type: {section}")
                                st.text(doc.page_content[:300] + "...")
                                st.write("---")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "reasoning": reasoning,
                            "grounding": asdict(grounding)
                        })

                    except Exception as e:
                        st.error(f"Error during retrieval: {str(e)}")

# --- Synthesis Mode ---
else:
    st.info("Enter a topic to generate a structured comparison across all uploaded papers.")
    
    topic = st.text_input("Research Topic", placeholder="e.g., Transformer architecture variants")
    
    if ui.button("Synthesize Literature", key="synth_btn"):
        vector_store = load_vector_db()
        if not vector_store:
            st.error("Please upload PDFs first.")
        elif not topic:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Retrieving and synthesizing across papers..."):
                try:
                    result = synthesize_papers(vector_store, topic)
                    
                    st.success("Synthesis Complete!")
                    
                    st.markdown("### 📝 Synthesis Summary")
                    st.markdown(result.synthesis_summary)
                    
                    if result.contradictions:
                         st.warning(f"⚠️ **Contradictions / Disagreements:**\n" + "\n".join([f"- {c}" for c in result.contradictions]))

                    st.markdown("### 📊 Structured Comparison")
                    
                    # Claims Table
                    st.subheader("Key Claims")
                    if result.claims_table:
                        df_claims = pd.DataFrame(list(result.claims_table.items()), columns=["Paper", "Claim"])
                        st.table(df_claims)
                    
                    # Methods
                    st.subheader("Methodologies")
                    if result.method_comparison:
                        df_methods = pd.DataFrame(list(result.method_comparison.items()), columns=["Paper", "Methodology"])
                        st.table(df_methods)
                        
                    # Results
                    st.subheader("Results & Findings")
                    if result.results_summary:
                        df_results = pd.DataFrame(list(result.results_summary.items()), columns=["Paper", "Main Result"])
                        st.table(df_results)

                except Exception as e:
                    st.error(f"Synthesis failed: {str(e)}")
