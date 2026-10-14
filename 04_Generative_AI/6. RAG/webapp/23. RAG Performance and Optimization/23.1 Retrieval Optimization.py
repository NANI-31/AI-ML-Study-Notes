import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="23.1 Retrieval Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class RetrievalOptimizerEngine:
    """Enterprise Retrieval Tuning & Multi-Objective Hyperparameter Optimization Engine."""

    def __init__(self):
        # Simulated benchmark evaluation matrix
        self.benchmark_database = {
            (128, 3, False, False): {"recall_at_10": 0.62, "precision": 0.78, "latency_ms": 15.0, "prompt_tokens": 384},
            (128, 10, True, True): {"recall_at_10": 0.88, "precision": 0.91, "latency_ms": 85.0, "prompt_tokens": 640},
            (512, 3, False, False): {"recall_at_10": 0.74, "precision": 0.81, "latency_ms": 22.0, "prompt_tokens": 1536},
            (512, 5, True, True): {"recall_at_10": 0.93, "precision": 0.94, "latency_ms": 110.0, "prompt_tokens": 1536},
            (1024, 3, False, False): {"recall_at_10": 0.69, "precision": 0.65, "latency_ms": 32.0, "prompt_tokens": 3072},
            (1024, 5, True, True): {"recall_at_10": 0.82, "precision": 0.79, "latency_ms": 140.0, "prompt_tokens": 2560}
        }

    def evaluate_configuration(
        self,
        chunk_size: int,
        top_k: int,
        use_reranker: bool,
        use_hybrid: bool
    ) -> Dict[str, Any]:
        """Calculates multi-objective performance trade-offs for a given retrieval configuration."""
        # Find nearest key or calculate interpolated performance
        base_recall = 0.70 if chunk_size == 512 else (0.64 if chunk_size == 128 else 0.66)
        if use_hybrid:
            base_recall += 0.12
        if top_k >= 5:
            base_recall += 0.08
            
        base_precision = 0.75
        if use_reranker:
            base_precision += 0.18
        if chunk_size == 1024:
            base_precision -= 0.10  # Context dilution penalty
            
        latency = 18.0 + (top_k * 2.5)
        if use_hybrid:
            latency += 12.0
        if use_reranker:
            latency += 75.0
            
        tokens = chunk_size * (3 if use_reranker else top_k)
        
        recall = round(min(0.98, base_recall), 3)
        precision = round(min(0.96, base_precision), 3)

        return {
            "chunk_size": chunk_size,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "use_hybrid": use_hybrid,
            "recall_at_10": recall,
            "precision_at_k": precision,
            "total_latency_ms": round(latency, 1),
            "prompt_tokens": tokens,
            "pareto_grade": "Optimal Enterprise Tier" if recall >= 0.90 and precision >= 0.90 else "Sub-Optimal Tradeoff"
        }


# --- Streamlit Layout ---

st.title("23.1 Retrieval Optimization in RAG")
st.caption("Multi-Objective Tuning: Chunk Granularity, Candidate Sizing, Domain Embeddings, Cross-Encoder Reranking & Hybrid Fusion")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Two-Stage Multi-Objective Retrieval Optimization Pipeline", divider="rainbow")

st.markdown(
    """
    Retrieval quality in production is governed by a **fundamental engineering trade-off**:
    maximizing factual recall (ensuring the answer chunk is captured) while maximizing precision (preventing irrelevant noise from diluting LLM attention),
    all while adhering to strict **latency SLOs** and **token cost envelopes**.
    
    The diagram below models the complete **Two-Stage Multi-Objective Optimization Architecture**, demonstrating how
    hybrid retrieval, adaptive candidate pools, and cross-encoder rerankers work in synergy to achieve optimal Pareto efficiency.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Retrieval_Optimization {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_stage1_recall {
            label = "STAGE 1: HIGH-RECALL HYBRID CANDIDATE RETRIEVAL (TOP-K = 20)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_query [label="Incoming Query (Dense & Lexical Channels)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            dense_search [label="Dense Bi-Encoder Search\\n(512-Token Semantic Chunks, HNSW Index)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            sparse_search [label="Sparse Lexical Search\\n(BM25 Exact Keyword Token Index)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            rrf_fusion [label="Reciprocal Rank Fusion (RRF)\\nCombines Top-20 Dense & Sparse Candidates", fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            user_query -> dense_search;
            user_query -> sparse_search;
            dense_search -> rrf_fusion [label="Rank Vector"];
            sparse_search -> rrf_fusion [label="Rank Lexical"];
        }

        subgraph cluster_stage2_precision {
            label = "STAGE 2: HIGH-PRECISION RERANKING & CONTEXT PRUNING (TOP-K = 3)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            cross_reranker [label="Cross-Encoder Reranker\\n(BGE-Reranker-Large / Cohere Rerank v3)\\nFull Token-to-Token Cross-Attention", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            relevance_gate [label="Relevance Threshold Gate\\n(Prunes candidates with score < 0.40)", fillcolor="#78350F", fontcolor="#FDE68A"];
            optimal_context [label="Optimized Lean Context Window\\nTop-3 Chunks (Highest Precision, Zero Noise)", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            rrf_fusion -> cross_reranker [label="Top-20 Pool"];
            cross_reranker -> relevance_gate -> optimal_context;
        }

        subgraph cluster_pareto_output {
            label = "STAGE 3: GROUNDED LLM SYNTHESIS";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            llm_synthesizer [label="Frontier LLM Generator\\nHigh Attention Focus on Pure Signal Chunks", fillcolor="#1E3A8A", fontcolor="#10B981"];
            verified_output [label="High-Accuracy Verified Response\\n(Zero Lost-in-the-Middle Decay)", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            optimal_context -> llm_synthesizer -> verified_output;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: CHOOSING CORRECT CHUNK SIZE ---
st.header("2. Choosing Correct Chunk Size (Token Granularity, Semantic Boundaries & Information Density)", divider="rainbow")

st.markdown(
    """
    Chunk size is the single most impactful architectural decision in RAG:
    
    - **Small Chunks (128 Tokens)**: Excellent semantic precision; matches specific sentences with minimal noise. *Limitation*: Lacks global context; often detaches pronouns from antecedents.
    - **Medium Chunks (512 Tokens)**: The enterprise sweet spot; captures complete paragraphs, table summaries, and coherent technical explanations without attention dilution.
    - **Large Chunks (1024+ Tokens)**: Preserves complete multi-paragraph narratives. *Limitation*: Semantic vector averaging dilutes specific details; induces heavy **lost-in-the-middle** degradation.
    """
)

# --- SECTION 3: OPTIMIZING TOP-K ---
st.header("3. Optimizing Top-K (Recall vs Precision Curves, Candidate Pool Sizing & Attention Budgeting)", divider="rainbow")

st.markdown(
    """
    Tuning the number of retrieved candidates (Top-K) involves balancing two opposing curves:
    
    - **Recall Curve**: As K increases from 3 to 20, the probability of capturing the answer chunk rises from 70% to 95%.
    - **Precision Curve**: As K increases, the ratio of irrelevant distracting tokens rises exponentially, degrading LLM reasoning accuracy.
    - **The Two-Stage Solution**: Set stage-1 retrieval to high recall (K = 20), followed by cross-encoder reranking that prunes down to high precision (K = 3).
    """
)

# --- SECTION 4: IMPROVING EMBEDDINGS ---
st.header("4. Improving Embeddings (Fine-Tuning Domain Bi-Encoders, Matryoshka Embeddings & Instruction Tuning)", divider="rainbow")

st.markdown(
    """
    Off-the-shelf embedding models struggle with specialized proprietary vocabulary:
    
    - **Domain Fine-Tuning**: Training contrastive encoders on synthetic (query, positive_chunk, hard_negative) triplets generated from company documents yields 15% - 25% recall improvements.
    - **Matryoshka Representation Learning (MRL)**: Embeddings where the first 128 or 256 dimensions contain the majority of information, allowing 4x vector compression with under 1% loss in recall.
    - **Instruction-Tuned Models (BGE-M3, Voyage-Large)**: Prepending task instructions (*'Represent this query for retrieving regulatory compliance manuals'*) to adapt vector representations dynamically.
    """
)

# --- SECTION 5: ADDING RERANKING ---
st.header("5. Adding Reranking (Cross-Encoder Verification, Listwise Attention & Cutoff Gating)", divider="rainbow")

st.markdown(
    """
    Bi-encoders embed queries and documents independently. They cannot perform token-to-token cross-attention, missing subtle negation and numeric constraints.
    
    **Adding a Cross-Encoder Reranker** (Cohere Rerank v3, BGE-Reranker-Large):
    - Evaluates joint query-document cross-attention across all transformer layers.
    - Re-orders the top-20 candidates based on true factual answering utility.
    - Filters out candidate chunks whose cross-attention score collapses below safety floors (e.g., `score < 0.40`).
    """
)

# --- SECTION 6: HYBRID SEARCH ---
st.header("6. Hybrid Search (Dense Vector + Sparse BM25 Reciprocal Rank Fusion & Parameter Tuning)", divider="rainbow")

st.markdown(
    """
    Combining dense vector search with sparse BM25 keyword matching eliminates blind spots:
    
    - Dense vectors capture conceptual intent and synonyms.
    - Sparse BM25 captures exact serial numbers, error codes, and unique acronyms.
    - **Reciprocal Rank Fusion (RRF)**: Merges rankings mathematically:  
      `RRF_Score(doc) = Sum( 1.0 / (60 + rank_i(doc)) )`  
      completely bypassing the need for brittle score normalization across different algorithms.
    """
)

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Enterprise Multi-Stage Retrieval Optimizer", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `RetrievalOptimizerEngine`.
    It models multi-objective retrieval hyperparameter evaluation, simulating trade-offs across chunk sizes,
    top-K candidate pools, cross-encoder rerankers, and hybrid search fusion.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class RetrievalOptimizerEngine:
    """Enterprise Retrieval Tuning & Multi-Objective Hyperparameter Optimization Engine."""

    def __init__(self):
        pass

    def evaluate_configuration(
        self,
        chunk_size: int,
        top_k: int,
        use_reranker: bool,
        use_hybrid: bool
    ) -> Dict[str, Any]:
        """Calculates multi-objective performance trade-offs for a given retrieval configuration."""
        base_recall = 0.70 if chunk_size == 512 else (0.64 if chunk_size == 128 else 0.66)
        if use_hybrid:
            base_recall += 0.12
        if top_k >= 5:
            base_recall += 0.08
            
        base_precision = 0.75
        if use_reranker:
            base_precision += 0.18
        if chunk_size == 1024:
            base_precision -= 0.10
            
        latency = 18.0 + (top_k * 2.5)
        if use_hybrid:
            latency += 12.0
        if use_reranker:
            latency += 75.0
            
        tokens = chunk_size * (3 if use_reranker else top_k)
        recall = round(min(0.98, base_recall), 3)
        precision = round(min(0.96, base_precision), 3)

        return {
            "chunk_size": chunk_size,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "use_hybrid": use_hybrid,
            "recall_at_10": recall,
            "precision_at_k": precision,
            "total_latency_ms": round(latency, 1),
            "prompt_tokens": tokens,
            "pareto_grade": "Optimal Enterprise Tier" if recall >= 0.90 and precision >= 0.90 else "Sub-Optimal Tradeoff"
        }
''',
    language="python"
)

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Chunk Size, Top-K, and Reranking Parameter Grid Explorer", divider="rainbow")

st.markdown(
    """
    Explore the multi-objective retrieval parameter grid.
    Adjust chunk granularity, candidate top-K, and toggles for cross-encoder reranking and hybrid search
    to inspect simulated Recall@10, Precision@K, pipeline latency, and prompt token consumption.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Retrieval Parameter Configuration")
    
    chunk_val = st.select_slider(
        "1. Chunk Size Granularity (Tokens):",
        options=[128, 512, 1024],
        value=512
    )
    
    top_k_val = st.slider("2. Retrieval Top-K Sizing:", min_value=1, max_value=15, value=5, step=1)
    
    use_hybrid = st.toggle("Enable Hybrid Search (Dense Vector + BM25 RRF)", value=True)
    use_rerank = st.toggle("Enable Cross-Encoder Neural Reranking", value=True)

with wb_col2:
    st.subheader("📊 Multi-Objective Performance Telemetry")
    
    engine = RetrievalOptimizerEngine()
    metrics = engine.evaluate_configuration(chunk_val, top_k_val, use_rerank, use_hybrid)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall@10", f"{metrics['recall_at_10'] * 100:.1f}%")
    m2.metric("Precision@K", f"{metrics['precision_at_k'] * 100:.1f}%")
    m3.metric("Latency", f"{metrics['total_latency_ms']} ms")
    m4.metric("Prompt Tokens", metrics["prompt_tokens"])
    
    st.markdown("#### Pareto Frontier Assessment")
    if "Optimal" in metrics["pareto_grade"]:
        st.success(f"🏆 **{metrics['pareto_grade']}**: Configuration achieves > 90% Recall and > 90% Precision.")
    else:
        st.warning(f"⚠️ **{metrics['pareto_grade']}**: Low recall or high context noise detected. Enable Hybrid Search and Reranking.")

st.markdown("### Architectural Hyperparameter Recommendations")
rec_df = pd.DataFrame([
    {"Strategy": "Naive Vector Only", "Chunk Size": 1024, "Top-K": 3, "Reranker": "No", "Recall@10": "69.0%", "Precision": "65.0%", "Latency": "32 ms"},
    {"Strategy": "Fast Low-Latency Tier", "Chunk Size": 128, "Top-K": 5, "Reranker": "No", "Recall@10": "76.0%", "Precision": "75.0%", "Latency": "30 ms"},
    {"Strategy": "Enterprise Production Gold", "Chunk Size": 512, "Top-K": 5, "Reranker": "Yes", "Recall@10": "93.0%", "Precision": "94.0%", "Latency": "110 ms"}
])
st.dataframe(rec_df, use_container_width=True)

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: Why is a two-stage retrieval architecture (High Recall -> High Precision) optimal for RAG?**"):
    st.markdown(
        """
        - **Stage 1 (Bi-Encoder + BM25)**: Fast (O(1) vector search and inverted index lookup) capable of scanning millions of chunks in milliseconds. It casts a wide net (K = 20) to guarantee high recall, but precision is moderate.
        - **Stage 2 (Cross-Encoder Reranker)**: Heavy transformer model that computes joint cross-attention across all tokens. Evaluating it over millions of documents is computationally impossible.
        Applying the cross-encoder strictly to the top 20 candidates achieves 95%+ precision without sacrificing throughput or blowing up latency budgets.
        """
    )

with st.expander("**Q2: How does chunk size directly influence embedding vector representation quality?**"):
    st.markdown(
        """
        Embedding models compress text into a single fixed-dimensional vector (e.g., 768 dimensions):
        - When chunk size is too large (1,024+ tokens), the vector averages diverse topics together (**semantic washing**), reducing cosine similarity for specific fine-grained facts.
        - When chunk size is too small (under 100 tokens), the vector captures isolated phrases but lacks semantic context, causing false positive matches.
        A 512-token chunk represents the empirical optimal balance for technical and regulatory corpora.
        """
    )

with st.expander("**Q3: What are Matryoshka Embeddings (MRL) and how do they reduce vector storage costs?**"):
    st.markdown(
        """
        **Matryoshka Representation Learning (MRL)** trains embedding models (such as OpenAI `text-embedding-3-large` or `nomic-embed-text`) so that the front d dimensions are explicitly trained to represent the full semantic meaning.
        - A 1,536-dimensional vector can be truncated to 256 or 512 dimensions by simple slicing.
        - Reduces vector RAM storage by 3x - 6x and speeds up HNSW distance computations proportionally with under 1.5% drop in retrieval accuracy.
        """
    )

with st.expander("**Q4: How do you tune the smoothing constant k in Reciprocal Rank Fusion (RRF)?**"):
    st.markdown(
        """
        In RRF: `Score(d) = Sum( 1.0 / (k + rank_i(d)) )`.
        - The parameter k controls the penalty slope for lower-ranked documents.
        - A small k (k = 10) places massive weight on the top 1-2 ranks, heavily penalizing candidates ranked 5th or lower.
        - A large k (k = 60, the standard default) smooths the ranking curve, ensuring documents that rank moderately well across both dense and sparse retrievers (e.g., ranked 3rd in both) outscore documents that ranked 1st in one but were absent in the other.
        """
    )

with st.expander("**Q5: Why does increasing Top-K past 10 often decrease downstream LLM answer quality?**"):
    st.markdown(
        """
        Increasing Top-K increases recall, but it simultaneously introduces **context dilution** and triggers the **lost-in-the-middle** effect.
        Transformer attention heads struggle to isolate signal when 80% of the prompt context consists of tangential noise.
        Passing 15 un-reranked chunks increases prompt token costs, inflates Time To First Token (TTFT), and frequently causes the LLM to hallucinate or miss the primary evidence chunk.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Arbitrary Character Slicing Without Semantic Boundary Respect**  
        **Antipattern**: Slicing documents into fixed 500-character blocks using string index slicing.  
        **Consequence**: Slices numbers, tables, and sentences in half, producing corrupted embeddings.  
        **Production Fix**: Employ recursive character text splitters or AST-based semantic chunkers that respect sentence and markdown boundaries.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Context Stuffing to Maximize Recall**  
        **Antipattern**: Increasing Top-K from 3 to 15 chunks under the naive assumption that more context always equals better answers.  
        **Consequence**: Quadruples prompt token consumption, inflates TTFT latency, and triggers lost-in-the-middle degradation.  
        **Production Fix**: Retrieve 20 candidates, but use a cross-encoder reranker to prune down to the top 3-4 highly relevant chunks.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Deploying Without a Cross-Encoder Score Safety Floor**  
        **Antipattern**: Statically accepting the Top-K chunks regardless of how low their reranker score is.  
        **Consequence**: Out-of-domain queries retrieve irrelevant documents, forcing the LLM to invent hallucinations.  
        **Production Fix**: Enforce a strict minimum reranker cutoff threshold (e.g., `score >= 0.40`); if not met, gracefully abstain.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Discarding BM25 in Favor of Pure Vector Search**  
        **Antipattern**: Relying exclusively on dense vector search for technical software documentation.  
        **Consequence**: Fails on exact error codes, function signatures, and configuration keys.  
        **Production Fix**: Implement hybrid search with Reciprocal Rank Fusion (RRF) unconditionally.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is the Two-Stage Retrieval Architecture?")
        st.markdown("**Back**: A pipeline that first retrieves a broad candidate pool (Top-20) using fast bi-encoders and BM25 for high recall, then reranks with a cross-encoder to select the Top-3 for high precision.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is Semantic Washing in large chunks?")
        st.markdown("**Back**: The loss of fine-grained detail that occurs when a 1,024+ token text chunk is compressed into a single vector, causing specific facts to be diluted by surrounding prose.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What are Matryoshka Embeddings?")
        st.markdown("**Back**: Embeddings engineered so that initial dimensions contain core semantic representations, allowing truncation to 256 or 512 dimensions for 4x memory savings.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Why is RRF smoothing constant k=60 used?")
        st.markdown("**Back**: A smoothing constant that balances rank decay, ensuring documents supported across both dense and sparse retrievers outscore single-source outliers.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of retrieval optimization, hyperparameter tuning, and hybrid architectures:")

chk1 = st.checkbox("I understand the multi-objective trade-offs between chunk size, recall, precision, and latency.", value=True)
chk2 = st.checkbox("I can implement a two-stage retrieval pipeline combining broad hybrid recall with cross-encoder precision.", value=True)
chk3 = st.checkbox("I understand how Matryoshka embeddings optimize vector database storage footprints.", value=True)
chk4 = st.checkbox("I know how to tune the RRF smoothing constant k to balance dense and sparse rank signals.", value=True)
chk5 = st.checkbox("I understand why increasing Top-K past 10 triggers context dilution and lost-in-the-middle degradation.", value=True)
chk6 = st.checkbox("I know how to establish cross-encoder score cutoff floors to prevent out-of-domain hallucinations.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Retrieval Optimization and Hyperparameter Tuning.")
