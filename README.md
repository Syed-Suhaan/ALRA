# Auto-LitReview Agent (ALRA)

ALRA is an AI-powered research assistant designed to automate literature reviews. It enables users to upload research papers (PDFs), perform semantic searches, and receive answers with confidence scores using RAG (Retrieval-Augmented Generation).

## Key Features

-   **PDF Ingestion**: Fast processing of multiple research papers using PyMuPDF.
-   **Semantic Search**: Utilizes FAISS and HuggingFace embeddings for accurate retrieval.
-   **AI Analysis**: Powered by Groq (Llama 3) for high-speed, quality answers.
-   **Confidence Scoring**: Custom metric to gauge the reliability of generated answers.
-   **Benchmark Dashboard**: Built-in tool to evaluate retrieval performance.

## ALRA 2.0 Upgrades (Research-Grade Agentic RAG)

ALRA 2.0 transforms the system from a basic RAG pipeline to an advanced research agent based on the **CLARA** (Composite Latent-Augmented Retrieval & Analysis) framework.

### New Features:
- **🧠 Query Reasoning**: LLM-based query expansion and decomposition to handle complex research questions.
- **🔍 Semantic Tagging**: Intelligent classification of PDF chunks into sections (Methodology, Results, etc.).
- **📊 Composite Grounding (Confidence 2.0)**: Multi-signal evaluation of answer accuracy.
- **📑 Multi-Paper Synthesis**: Generate structured comparison tables across multiple papers.
- **📈 Performance Dashboard**: Visual analytics and interaction logging.

### Upgrades Gallery
![ALRA 2.0 Dashboard](screenshots/Screenshot%202026-02-19%20191218.png)
![Synthesis Mode](screenshots/Screenshot%202026-02-19%20191238.png)

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Syed-Suhaan/ALRA.git
    cd ALRA/auto-lit-review
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables**:
    Create a `.env` file and add your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```
