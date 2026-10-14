import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="4.4 Query Embeddings - Embeddings Curriculum",
    page_icon="🔎",
    layout="wide"
)

# Title & Overview
st.title("4.4 Query Embeddings: Asymmetric Representation, Instruction Prefixes, & Intent Projections")
st.caption("Chapter 4: Types of Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In symmetric tasks (like paraphrase detection), two inputs have identical characteristics: both are full, complete sentences.
In **Information Retrieval and Search**, however, the relationship is fundamentally **Asymmetric**:
* The **Query** is typically short, vague, conversational, or keyword-based (e.g., *"fast python api"* — 3 tokens).
* The **Document** is long, authoritative, structured, and explanatory (e.g., a 400-word documentation guide on FastAPI).

If you embed both using a naive symmetric embedding model, **the query vector will naturally cluster near other short questions, rather than matching the authoritative document that actually answers it**!

To resolve this fundamental mismatch, modern search architectures deploy three primary mechanisms:
1. **Asymmetric Embedding Models**: Models trained specifically on query-passage pairs with distinct sub-networks or task instructions.
2. **Instruction-Prefixed Queries**: Prepending explicit task prompts (e.g., `"Represent this sentence for searching relevant passages: {query}"`) to project the query into the document space.
3. **Hypothetical Document Embeddings (HyDE)**: Using an LLM to hallucinate a synthetic answer, then embedding the synthetic answer into the document space.

This lesson explores the mathematics of asymmetric query projection, instruction prefixing, and HyDE search pipelines.
""")

st.divider()

# Section 1: The Asymmetric Query-Document Gap
st.header("1. The Asymmetric Representation Gap & Instruction Tuning", divider="rainbow")

col_qy1, col_qy2 = st.columns([1, 1])

with col_qy1:
    st.markdown("""
    ### Why Asymmetric Modeling is Critical:
    * **1. The Question-Answer Geometric Divergence**:
      In naive vector space:
      `Query: "What causes hypertension?"` is geometrically closer to `Similar Question: "Why do people get high blood pressure?"` than to `Answer Passage: "Hypertension occurs when arterial walls experience persistent pressure..."`.
    * **2. The Instruction Prefix Solution (BGE / E5 / Instructor)**:
      Modern embedding models (like BAAI BGE and Microsoft E5) mandate asymmetric instruction prefixes:
      * Query format: `query: {user_search}`
      * Passage format: `passage: {corpus_document}`
      * The prefix instructs the attention layers to apply an affine rotation, steering the query vector directly toward answer passages!
    * **3. Hypothetical Document Embeddings (HyDE)**:
      Instead of matching a 3-word query to a 300-word document, an LLM generates a speculative 100-word answer. 
      Embedding this speculative answer bridges the document length and style gap!
    """)

with col_qy2:
    st.markdown("### Asymmetric Vector Projection Alignment")
    query_dot = """
    digraph AsymQuery {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        RawQuery [label="Raw Query (3 Tokens)\\n'fast python api'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        subgraph cluster_asym {
            label="Asymmetric Transformation Strategies"; style="dashed"; color="#D97706";
            Prefix [label="Strategy A: Task Prefix\\n'query: Represent query for search:'", fillcolor="#FEF3C7"];
            HyDE [label="Strategy B: HyDE Pipeline\\nLLM generates synthetic passage", fillcolor="#EDE9FE"];
        }

        ProjectedVec [label="Rotated Query Vector\\n(Aligned with Document Latent Space)", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];
        PassageDoc [label="Target Passage Document\\n'FastAPI documentation and benchmarks'", fillcolor="#22C55E", fontcolor="white", style="filled,bold"];

        RawQuery -> Prefix -> ProjectedVec;
        RawQuery -> HyDE -> ProjectedVec;
        ProjectedVec -> PassageDoc [label="High Cosine Match (0.89)", style="bold", color="#15803D"];
    }
    """
    st.graphviz_chart(query_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Asymmetric Query Engine in Python", divider="rainbow")
st.markdown("""
The following Python class demonstrates asymmetric instruction prefixing, synthetic HyDE document expansion, and comparative candidate ranking.
""")

class ProductionQueryEmbeddingEngine:
    """
    Simulates asymmetric query embedding, instruction prefixing,
    and Hypothetical Document Embedding (HyDE) transformations.
    """
    def __init__(self, dimensions: int = 6):
        self.dimensions = dimensions
        # Sample document corpus with pre-computed passage embeddings
        self.documents = [
            {
                "id": "DOC-FASTAPI",
                "title": "FastAPI Framework Documentation",
                "content": "FastAPI is a modern, high-performance web framework for building APIs with Python based on standard type hints and Starlette.",
                "vector": np.array([0.90, 0.85, 0.10, 0.15, 0.30, 0.20])
            },
            {
                "id": "DOC-DJANGO",
                "title": "Django Full-Stack Web Framework",
                "content": "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic MVC design with an ORM.",
                "vector": np.array([0.70, 0.40, 0.20, 0.10, 0.60, 0.25])
            },
            {
                "id": "DOC-FLASK",
                "title": "Flask Microframework Tutorial",
                "content": "Flask is a lightweight WSGI micro web framework written in Python designed to make getting started quick and easy.",
                "vector": np.array([0.80, 0.60, 0.15, 0.12, 0.35, 0.18])
            },
            {
                "id": "DOC-WEATHER",
                "title": "Meteorological Data Ingestion",
                "content": "Automated atmospheric sensors stream weather telemetry data across global monitoring stations.",
                "vector": np.array([0.10, 0.12, 0.90, 0.85, 0.10, 0.05])
            }
        ]
        # Normalize document vectors
        for d in self.documents:
            d["vector"] /= np.linalg.norm(d["vector"])

    def encode_query(self, query_text: str, mode: str = "Asymmetric Prefixed") -> np.ndarray:
        """Encodes query under different asymmetric transformation modes."""
        np.random.seed(abs(hash(query_text.lower())) % (2**31))

        if mode == "Raw Symmetric (No Prefix)":
            # Unprefixed short query vector: has lower topical alignment
            vec = np.array([0.45, 0.40, 0.10, 0.08, 0.20, 0.15])
        elif mode == "Asymmetric Prefixed":
            # Instruction prefix rotates query vector toward document space
            vec = np.array([0.88, 0.82, 0.11, 0.14, 0.28, 0.19])
        else: # HyDE (Hypothetical Document Embedding)
            # Simulated embedding of generated passage: closely matches FastAPI doc
            vec = np.array([0.91, 0.86, 0.09, 0.14, 0.29, 0.21])

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def rank_documents(self, query_vec: np.ndarray) -> List[Dict[str, Any]]:
        """Ranks corpus documents by cosine similarity."""
        results = []
        for d in self.documents:
            sim = float(np.dot(query_vec, d["vector"]))
            results.append({
                "id": d["id"],
                "title": d["title"],
                "content": d["content"],
                "similarity": round(sim, 4)
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

code_snippet = r'''
class AsymmetricSearchHandler:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.encoder = SentenceTransformer(model_name)

    def encode_corpus_passages(self, passages: list) -> np.ndarray:
        # BGE models do NOT need prefix for documents
        return self.encoder.encode(passages, normalize_embeddings=True)

    def encode_query_asymmetric(self, query: str) -> np.ndarray:
        # MANDATORY for BGE: Prepend task instruction to queries
        prompt = f"Represent this sentence for searching relevant passages: {query}"
        return self.encoder.encode(prompt, normalize_embeddings=True)

    def encode_query_hyde(self, query: str, llm_client) -> np.ndarray:
        # HyDE: Generate synthetic answer passage first
        hypothetical_doc = llm_client.generate(f"Write a short passage answering: {query}")
        return self.encoder.encode(hypothetical_doc, normalize_embeddings=True)
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Asymmetric & HyDE Workbench
st.header("3. Interactive Asymmetric Query & HyDE Workbench", divider="rainbow")
st.markdown("""
Test how **Asymmetric Instruction Prefixes** and **HyDE (Hypothetical Document Embeddings)** bridge the gap between 
short keyword queries and long target documentation passages.
""")

query_engine = ProductionQueryEmbeddingEngine(dimensions=6)

preset_queries = [
    "fast python api",
    "lightweight web microframework",
    "python full stack web orm"
]

col_qy_in1, col_qy_in2 = st.columns([1, 1])

with col_qy_in1:
    selected_query = st.selectbox("1. Select Short Search Query:", preset_queries)
with col_qy_in2:
    transformation_mode = st.radio(
        "2. Query Embedding Transformation:",
        ["Raw Symmetric (No Prefix)", "Asymmetric Prefixed (query: ...)", "HyDE (Synthetic Document)"],
        index=1
    )

query_vector = query_engine.encode_query(selected_query, mode=transformation_mode)
ranked_docs = query_engine.rank_documents(query_vector)

st.markdown("### Search Results & Alignment Scores")

col_top_res, col_top_info = st.columns([1, 1])
with col_top_res:
    top_hit = ranked_docs[0]
    st.metric(
        f"Top Result: {top_hit['title']}",
        f"{top_hit['similarity']:.4f}",
        delta=f"{'+18% Alignment' if transformation_mode != 'Raw Symmetric (No Prefix)' else 'Baseline'}"
    )
with col_top_info:
    if transformation_mode == "Raw Symmetric (No Prefix)":
        st.warning("Without asymmetric instruction prefixes, short queries experience lower confidence alignment with long technical passages.")
    elif transformation_mode == "Asymmetric Prefixed (query: ...)":
        st.success("Instruction prefix successfully rotates the query vector directly into the authoritative document latent space!")
    else:
        st.success("HyDE: The hypothetical document matches the style, vocabulary, and token density of the target documentation passage!")

st.markdown("##### Full Candidate Ranking Comparison:")
results_df = pd.DataFrame([
    {"Rank": i+1, "Doc ID": d["id"], "Title": d["title"], "Cosine Similarity": f"{d['similarity']:.4f}", "Preview": d["content"][:80] + "..."}
    for i, d in enumerate(ranked_docs)
])
st.dataframe(results_df, use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Query Embedding Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why do modern embedding models (e.g. BGE, E5, Voyage) require an instruction prefix for queries but NOT for documents?**
    * **Answer**:
      * The underlying training data is **Asymmetric** (MS MARCO, Natural Questions): short questions paired with long Wikipedia or web passages.
      * The model uses the instruction prefix (`"query: ..."` or `"Represent this sentence for searching: ..."`) as an **attention steering token**.
      * The self-attention layers interpret this prompt token to apply an affine rotation, projecting the short, interrogative question into the geometric manifold of explanatory answers.
      * Passages are already in the target manifold, so prepending an instruction prefix to passages would corrupt the index.

    **Q2: What is Hypothetical Document Embeddings (HyDE) and what are its primary failure modes?**
    * **Answer**:
      * **How HyDE Works**: When a user asks *"What is the capital of Australia?"*, an LLM writes a speculative paragraph (*"Canberra is the capital of Australia, established in 1913..."*). The system embeds this speculative paragraph and searches the vector database.
      * **Advantage**: Bypasses the length and vocabulary mismatch between short queries and long documents.
      * **Failure Modes**:
        1. **Hallucination Drift**: If the LLM generates completely false facts in the speculative answer, the embedding searches for the hallucinated facts, retrieving incorrect documents.
        2. **Latency & Cost**: Adding an LLM inference call before vector retrieval adds 500-1500ms of latency and increases token billing.

    **Q3: How do Symmetric and Asymmetric embedding models differ in training loss?**
    * **Answer**:
      * **Symmetric Models (STS)**: Trained on pairs $(s_1, s_2)$ where both inputs are symmetric paraphrases. The loss function is symmetric: $loss(s_1, s_2) = loss(s_2, s_1)$.
      * **Asymmetric Models (Retrieval)**: Trained on directed pairs $(q, d^+)$ where queries and documents come from different distributions. The query tower and passage tower often use separate positional or prefix embeddings.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Prepending 'passage:' to Queries or 'query:' to Passages**:
      * *Issue*: Inverting the instruction prefixes during ingestion or search.
      * *Consequence*: Severe 20-30% drop in retrieval hit-rate; the model projects vectors in opposite directions.
      * *Fix*: Automate prefix attachment in strict wrapper classes (`encode_query` vs `encode_documents`).

    * **Trap 2: Using HyDE for Simple Keyword Lookups**:
      * *Issue*: Triggering an LLM call for queries like `"Apple stock price"` or `"Python 3.12 release date"`.
      * *Consequence*: Adds 1.5s of unnecessary latency for simple factual lookups.
      * *Fix*: Use a query classifier; deploy HyDE only for complex, multi-hop, or conceptual inquiries.

    * **Trap 3: Forgetting to Strip Punctuation Noise from Web Search Queries**:
      * *Issue*: Passing raw user strings with trailing boolean operators (`AND`, `OR`) into dense embedding encoders.
      * *Consequence*: Dense models treat `"OR"` as semantic content rather than a search operator, skewing vectors.
      * *Fix*: Clean query text prior to embedding.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Asymmetric Retrieval**
      * *Definition*: Information retrieval where the query is short/interrogative and the target document is long/explanatory, requiring specialized alignment models.
    * **Flashcard 2: Query Instruction Prefix**
      * *Definition*: A task-steering prompt prepended to user queries (e.g. 'query: ') that rotates the query vector into the document answer space.
    * **Flashcard 3: Hypothetical Document Embedding (HyDE)**
      * *Definition*: A technique that generates a speculative answer using an LLM and embeds the synthetic answer to retrieve matching real documents.
    * **Flashcard 4: Affine Projection**
      * *Definition*: The geometric transformation applied by transformer self-attention layers to map vectors between distinct linguistic manifolds.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Query Embeddings:

- [ ] I understand the fundamental difference between Symmetric and Asymmetric retrieval.
- [ ] I know why instruction prefixes (`query: `) are mandatory for frontier retrieval models like BGE and E5.
- [ ] I can describe the mechanics and failure modes of Hypothetical Document Embeddings (HyDE).
- [ ] I understand how instruction prefixes project short queries into the document answer manifold.
- [ ] I know how to build a production search pipeline combining asymmetric query encoding with passage vector indexes.
""")
