import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="4.3 Contextual Embeddings (Transformer, Sentence, Document, Query Embeddings)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MASTER STYLING ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #94A3B8;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .takeaway-box {
        background-color: #0B132B;
        border-left: 4px solid #38BDF8;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #1C1917;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .modern-callout {
        background-color: #042F2E;
        border-left: 4px solid #14B8A6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<div class="main-title">4.3 Contextual Embeddings: Transformer, Sentence, Document, and Query Representations</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Fixed Lexical Points to Dynamic Context-Aware Manifolds — How Modern Foundation Models and Bi-Encoders Formulate Dense Vector Representations across Granularities.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Transition to Dynamic Contextual Manifolds", divider="rainbow")

st.markdown("""
In static word embeddings (Word2Vec, GloVe), a word like `"bank"` has exactly one vector coordinate in vector space, 
meaning `"river bank"` and `"investment bank"` occupy the exact same point. 
**Contextual embeddings** completely resolve this polysemy bottleneck by computing dynamic vector states that vary 
based on the surrounding sequence of tokens.
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.4", ranksep="0.5")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("INPUT", "Input Sequence Tokens\n[Token_1, Token_2, ..., Token_N]", shape="box", fillcolor="#1E293B", color="#38BDF8")
    arch_graph.node("STATIC_LOOKUP", "Input Embedding Table + Position Bias\nx_i = W_token[i] + W_pos[i]\nShape: [Batch, Seq_Len, Hidden_Dim]", shape="box", fillcolor="#312E81", color="#818CF8")
    
    with arch_graph.subgraph(name="cluster_transformer") as c:
        c.attr(bgcolor="#0F172A", label="Transformer Deep Backbone (L Layers)", fontcolor="#38BDF8", color="#38BDF8")
        c.node("LAYER_1", "Layer 1: Self-Attention + MLP\nCaptures local syntax and collocations", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("LAYER_MID", "Middle Layers: Self-Attention + MLP\nDisambiguates polysemy and coreference", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("LAYER_FINAL", "Final Layer Hidden States H_L\nShape: [Batch, Seq_Len, Hidden_Dim]\nDynamic token vectors conditioned on all tokens", shape="box", fillcolor="#1E293B", color="#38BDF8")

    with arch_graph.subgraph(name="cluster_pooling") as c:
        c.attr(bgcolor="#0F172A", label="Aggregation & Pooling Heads", fontcolor="#10B981", color="#10B981")
        c.node("TOKEN_REP", "Token Embeddings\nDirect H_L[:, i, :]\n(NER, Tagging, Generation)", shape="box", fillcolor="#1E293B", color="#10B981")
        c.node("SENT_REP", "Sentence / Doc Embeddings\nMasked Mean Pooling (H_L, mask)\nShape: [Batch, Hidden_Dim]\n(Semantic Search, RAG)", shape="box", fillcolor="#042F2E", color="#14B8A6")
        c.node("QUERY_REP", "Asymmetric Query Embeddings\nInstruction Prefix + Pooling\nShape: [Batch, Hidden_Dim]", shape="box", fillcolor="#1E293B", color="#10B981")

    # Edges
    arch_graph.edge("INPUT", "STATIC_LOOKUP")
    arch_graph.edge("STATIC_LOOKUP", "LAYER_1")
    arch_graph.edge("LAYER_1", "LAYER_MID")
    arch_graph.edge("LAYER_MID", "LAYER_FINAL")
    arch_graph.edge("LAYER_FINAL", "TOKEN_REP")
    arch_graph.edge("LAYER_FINAL", "SENT_REP")
    arch_graph.edge("LAYER_FINAL", "QUERY_REP")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Structural Core Principles
    * **Dynamic Vector Mapping**: A word is a function of its context: `v(token) = f_theta(token, context)`. The token `"crane"` yields completely different vectors in `"bird sanctuary crane"` versus `"construction crane lifting steel"`.
    * **Hierarchical Representation Depths**:
      * *Lower Layers (1-4)*: Encode lexical properties, parts of speech, and local surface syntax.
      * *Middle Layers (5-16)*: Resolve syntactic dependencies, coreference chains, and semantic role labeling.
      * *Top Layers (17-32+)*: Encode complex pragmatic meaning, long-range dependencies, and task-specific abstractions.
    * **The Aggregation Dilemma**: Transformers natively produce token-level matrices `[Seq_Len, Hidden_Dim]`. Obtaining single sentence or document vectors requires explicit aggregation operators (Mean Pooling, CLS token pooling, or Late Interaction).
    """)

# ==============================================================================
# SECTION 2: 1. CONTEXT-DEPENDENT REPRESENTATIONS
# ==============================================================================
st.header("2. Context-Dependent Representations: Solving Polysemy", divider="rainbow")

st.markdown("""
The transition from static to contextual representations solved the single greatest bottleneck in 20th-century computational linguistics: **Polysemy** (words possessing multiple distinct senses) and **Homonymy** (identical spelling with unrelated etymology).
""")

col_poly1, col_poly2 = st.columns([1, 1])

with col_poly1:
    st.markdown("""
    ### The Static Polysemy Disaster
    In Word2Vec or GloVe, the objective forces one static vector v_{bank} to satisfy two orthogonal co-occurrence clusters:
    * Context Cluster 1: `["river", "water", "muddy", "overflow", "stream"]`
    * Context Cluster 2: `["money", "loan", "interest", "deposit", "investment"]`

    Because the static model minimizes total loss across all occurrences, v_{bank} converges to the **weighted geometric centroid** of both senses. This vector is close to neither river nor finance, polluting vector distances across downstream tasks.

    ### Historical Milestones: ELMo (2018)
    Peters et al. introduced **ELMo (Embeddings from Language Models)**:
    * Used stacked bidirectional LSTMs (bi-LSTMs).
    * Extracted hidden states from both forward and backward passes:
      `h_{t, j} = [ h_{forward, t, j} ; h_{backward, t, j} ]`
    * Downstream tasks learned a task-specific weighted mixture of LSTM layer states:
      `ELMo_t = gamma * Sum_{j=0}^{L} s_j * h_{t, j}`
    """)

with col_poly2:
    st.markdown("""
    ### Modern Transformer Resolution
    In a Transformer, dynamic self-attention computes pairwise token affinities:
    ```
    Attention(Q, K, V) = Softmax((Q . K^T) / sqrt(d_k)) . V
    ```
    When `"bank"` attends to `"river"`, its Query vector matches the Key vector of `"river"`, injecting `"river"`'s Value vector directly into the updated representation of `"bank"`.
    """)
    
    df_poly = pd.DataFrame({
        "Context Sentence": [
            "The muddy bank of the winding river overflowed after heavy rain.",
            "The central bank lowered benchmark interest rates by 50 basis points.",
            "He decided to bank on his extensive experience during the interview."
        ],
        "Token Sensed": ["bank (Noun - Geography)", "bank (Noun - Financial)", "bank (Verb - Idiomatic reliance)"],
        "Top Cosine Nearest Neighbors in LLM Hidden State": [
            "shore, riverbank, embankment, mud, stream",
            "treasury, lender, institution, capital, reserve",
            "depend, rely, count, trust, wager"
        ]
    })
    st.dataframe(df_poly, use_container_width=True, hide_index=True)

st.markdown("""
<div class="modern-callout">
<strong>🚀 Modern Foundation LLM Reality: How New Models Follow This</strong><br>
Modern LLMs (Llama 3, GPT-4, Mistral) do not have a single fixed representation for any concept. A token is represented as a sequence of evolving states across 32 to 128 layers. The prompt's context alters every intermediate tensor, allowing models to grasp complex nuances, puns, metaphors, and domain-specific terminology instantly.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 3: 2. TRANSFORMER EMBEDDINGS
# ==============================================================================
st.header("3. Transformer Embeddings: Multi-Layer Latent Trajectories", divider="rainbow")

st.markdown("""
Within a Transformer architecture, an embedding is not merely a single vector at the input or output; 
it is an entire **multi-layer latent trajectory** through hidden dimension space `R^[d_model]`.
""")

t_emb_col1, t_emb_col2 = st.columns([1, 1])

with t_emb_col1:
    st.markdown("""
    ### The Anatomical Pipeline
    For input token indices X = [x_1, x_2, ..., x_N]:
    1. **Token Lookup**: E_{tok} = W_E[X] in R^[N, d_model]
    2. **Positional Encoding**: H_0 = E_{tok} + P in R^[N, d_model] (or RoPE applied per attention layer).
    3. **Layer Transitions (for layer l = 1 to L):**
       ```
       H_l^{attn} = H_{l-1} + MultiHeadAttention(LN(H_{l-1}))
       H_l = H_l^{attn} + MLP(LN(H_l^{attn}))
       ```
    
    ### Dimensionality Across Modern Foundation Models
    | Model Architecture | Layer Count (L) | Hidden Dimension (d_{model}) | Attention Heads | Context Window |
    | :--- | :--- | :--- | :--- | :--- |
    | **BERT-Base** | 12 | 768 | 12 | 512 |
    | **BERT-Large** | 24 | 1024 | 16 | 512 |
    | **Llama 3 8B** | 32 | 4096 | 32 | 8,192 |
    | **Llama 3 70B** | 80 | 8192 | 64 | 8,192 |
    | **Mistral 7B** | 32 | 4096 | 32 | 32,768 |
    """)

with t_emb_col2:
    st.markdown("""
    ### Layer-Wise Representation Specialization
    Extensive probing research (Tenney et al., Jawahar et al.) reveals that Transformers internally recapitulate classical NLP pipelines:
    """)
    
    probe_graph = graphviz.Digraph(format="svg")
    probe_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.3")
    probe_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    probe_graph.attr("edge", color="#64748B", penwidth="1.2")

    probe_graph.node("TOP", "Layers 24-32+: High-Level Semantics & Generation\nPragmatics, sentiment, complex world reasoning, task output", shape="box", fillcolor="#042F2E", color="#14B8A6")
    probe_graph.node("MID", "Layers 9-23: Deep Syntax & Coreference\nDependency parsing, anaphora resolution, entity linking", shape="box", fillcolor="#312E81", color="#818CF8")
    probe_graph.node("LOW", "Layers 1-8: Surface Syntax & Morphosyntax\nPart-of-speech tagging, phrase boundaries, collocations", shape="box", fillcolor="#1E293B", color="#38BDF8")
    probe_graph.node("INPUT_LAYER", "Layer 0: Static Token Embeddings\nPure lexical identity + token frequency bias", shape="box", fillcolor="#1E293B", color="#94A3B8")

    probe_graph.edge("INPUT_LAYER", "LOW")
    probe_graph.edge("LOW", "MID")
    probe_graph.edge("MID", "TOP")

    with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(probe_graph, use_container_width=True)

# ==============================================================================
# SECTION 4: 3. SENTENCE EMBEDDINGS
# ==============================================================================
st.header("4. Sentence Embeddings: Bi-Encoders and Pooling Strategies", divider="rainbow")

st.markdown("""
While standard Transformers produce a matrix of token representations H in R^[N, D], 
downstream semantic search, duplicate detection, and vector indexing require a **single dense vector** u in R^D 
summarizing the entire sentence.
""")

s_col1, s_col2 = st.columns([1, 1])

with s_col1:
    st.markdown("""
    ### SBERT (Sentence-BERT, Reimers & Gurevych, 2019)
    Using raw BERT for semantic similarity required passing sentence pairs (s_1, s_2) through a Cross-Encoder (O(N^2) cross-attention), making finding the most similar pair in 10,000 sentences require 50 million inference passes (approx 65 hours).
    
    **Sentence-BERT introduced the Siamese / Bi-Encoder Architecture**:
    * Each sentence is encoded independently into a single vector u, v in R^D.
    * Pairwise similarity is computed via dot product or cosine similarity in microseconds (O(D)).
    * Finding the closest sentence among 10,000 takes milliseconds via approximate nearest neighbors (FAISS, HNSW).
    
    ### Pooling Strategies
    Given final layer token vectors H = [h_1, h_2, ..., h_N] and binary attention mask M in {0, 1}^N:
    1. **CLS Token Pooling**:
       `u = h_1` (the vector of the `[CLS]` token). Often sub-optimal without explicit contrastive pretraining.
    2. **Masked Mean Pooling (Standard SBERT & BGE)**:
       `u = Sum_{i=1}^{N} (h_i * M_i) / Sum_{i=1}^{N} M_i`
       Computes the true arithmetic centroid of valid non-padded tokens.
    3. **Masked Max Pooling**:
       `u_d = max_{i: M_i=1} (h_{i, d})` across each dimension d.
    """)

with s_col2:
    st.markdown("""
    ### Modern Sentence Embedding Leaderboard (MTEB)
    Modern dense encoders are evaluated on the **Massive Text Embedding Benchmark (MTEB)** across classification, retrieval, clustering, and reranking:
    """)
    df_mteb = pd.DataFrame({
        "Model": ["BGE-Large-EN-v1.5", "GTE-Qwen2-7B-Instruct", "OpenAI text-embedding-3-large", "All-MiniLM-L6-v2", "E5-Mistral-7B-Instruct"],
        "Parameters": ["335M", "7.6B", "Proprietary", "22M", "7.1B"],
        "Dimensions": ["1024", "3584", "3072", "384", "4096"],
        "Pooling Mode": ["Mean Pooling", "Last Token / Mean", "Proprietary", "Mean Pooling", "Last Token Pool"],
        "MTEB Average": ["64.11", "70.23", "64.60", "56.26", "68.83"],
        "Primary Use Case": ["Production RAG / Retrieval", "High-Precision Semantic Search", "General API Vector Search", "Ultra-low Latency Edge CPU", "Complex Domain Reasoning"]
    })
    st.dataframe(df_mteb, use_container_width=True, hide_index=True)

st.markdown("""
<div class="warning-box">
<strong>Production Pitfall: Unmasked Mean Pooling Corruption</strong><br>
When batching sentences with varying lengths, padded positions receive zero tokens (token ID 0). If you perform <code>torch.mean(outputs.last_hidden_state, dim=1)</code> without multiplying by the attention mask, the padding vectors artificially pull the sentence vector toward zero, severely corrupting cosine similarities.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 5: 4. DOCUMENT EMBEDDINGS
# ==============================================================================
st.header("5. Document Embeddings: Long-Context Chunking and Hierarchical Representations", divider="rainbow")

st.markdown("""
Representing an entire 20-page PDF, research paper, or legal contract as a single dense vector 
introduces unique architectural challenges: **the Information Bottleneck** and **Transformer Context Windows**.
""")

doc_col1, doc_col2 = st.columns([1, 1])

with doc_col1:
    st.markdown("""
    ### The Information Bottleneck
    Compressing 10,000 words into a single 1024-dimensional float vector causes severe information loss. 
    A single 1024-dim vector simply cannot retain fine-grained facts, numerical figures, and structural relations across dozens of pages.

    ### Core Architectural Approaches for Documents:
    1. **Sliding Window Chunking with Overlap (Production Standard)**:
       * Split document into overlapping chunks (e.g., 512 tokens with 64-token stride).
       * Compute independent dense embedding for each chunk: v_{c_1}, v_{c_2}, ..., v_{c_k}.
       * Store each chunk in a Vector Database with document-level metadata.
    2. **Late-Interaction ColBERT (Khattab & Zaharia)**:
       * Rather than collapsing the document into one vector, ColBERT stores **all token embeddings** for the document.
       * At query time, computes **MaxSim operator**: for every query token, finds the maximum cosine similarity across all document tokens, then sums them:
         `Score(Q, D) = Sum_{q in Q} max_{d in D} (E_q . E_d)`
    3. **Native Long-Context Encoders**:
       * Encoders with RoPE / FlashAttention supporting 8k-32k tokens (e.g., `BGE-M3`, `Jina-Embeddings-v3`).
    """)

with doc_col2:
    st.markdown("""
    ### Document Representation Paradigms Compared
    """)
    df_doc_comp = pd.DataFrame({
        "Paradigm": ["Dense Single-Vector", "Chunked Multi-Vector", "ColBERT Late Interaction", "Hierarchical Pooling"],
        "Mechanism": [
            "Pool entire document into 1 vector",
            "Embed chunks (300-500 tokens) separately",
            "Store token vectors; MaxSim query alignment",
            "Mean-pool paragraph vectors into 1 master vector"
        ],
        "Storage Cost": ["Low (1x vector)", "Medium (K vectors per doc)", "High (N token vectors per doc)", "Low (1x vector)"],
        "Fine-Grained Retrieval": ["Extremely Poor (Averaging washes out details)", "High (Pinpoints exact passage)", "State of the Art (Token-level match)", "Poor (Dilutes localized facts)"],
        "Production Suitability": ["Global thematic clustering", "Standard Production RAG", "High-accuracy search reranking", "Document categorization"]
    })
    st.dataframe(df_doc_comp, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 6: 5. QUERY EMBEDDINGS
# ==============================================================================
st.header("6. Query Embeddings: Asymmetric Retrieval & Instruction Tuning", divider="rainbow")

st.markdown("""
In information retrieval and RAG systems, queries and documents exhibit severe **asymmetry**:
* **Query**: Short, ambiguous, interrogative, 3-10 tokens (e.g., *"how to fix OOM CUDA error"*).
* **Document**: Long, detailed, declarative, 200-1000 tokens (e.g., *"PyTorch CUDA Out of Memory troubleshooting guide..."*).
""")

q_col1, q_col2 = st.columns([1, 1])

with q_col1:
    st.markdown("""
    ### The Asymmetric Matching Problem
    If queries and documents are encoded using symmetric bi-encoders trained on sentence equivalence (STS), 
    the model prefers returning **other queries** rather than documents that answer the query, 
    because question syntax is mathematically distant from explanatory prose.

    ### The Modern Solution: Task-Specific Instruction Prefixes
    State-of-the-art embedding models (E5, BGE, Instructor, Qwen2-Embedding) utilize **asymmetric instruction prompts** 
    to project queries and passages into aligned latent subspaces:
    
    * **Query Encoding**:
      `prompt_query = "Represent this sentence for searching relevant passages: " + query`
    * **Document Encoding**:
      `prompt_doc = document` (or `"passage: " + document`)
    
    By conditioning the query through an instruction prefix, the bi-encoder transforms the interrogative intent 
    into the complementary vector manifold of answer passages!
    """)

with q_col2:
    st.markdown("""
    ### Bi-Encoder Retrieval vs. Cross-Encoder Reranking
    Modern enterprise AI systems never rely on query embeddings alone; they implement a **Two-Stage Retrieval Funnel**:
    """)
    
    funnel_graph = graphviz.Digraph(format="svg")
    funnel_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.3")
    funnel_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    funnel_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    funnel_graph.node("STAGE1", "Stage 1: Bi-Encoder Dense Retrieval\nEmbed Query -> Vector DB (HNSW / FAISS)\nEvaluates 10M+ candidates in <10ms -> Top 100 Passages", shape="box", fillcolor="#1E293B", color="#38BDF8")
    funnel_graph.node("STAGE2", "Stage 2: Cross-Encoder Reranker\nInput: [CLS] Query [SEP] Candidate [SEP]\nFull all-to-all cross-attention over 100 pairs in 40ms -> Top 5 Passages", shape="box", fillcolor="#312E81", color="#818CF8")
    funnel_graph.node("LLM_GEN", "Stage 3: LLM Context Window\nInject Top 5 into Prompt -> Generate Final Answer", shape="box", fillcolor="#042F2E", color="#14B8A6")

    funnel_graph.edge("STAGE1", "STAGE2", label="Top 100")
    funnel_graph.edge("STAGE2", "LLM_GEN", label="Top 5")

    with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(funnel_graph, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<strong>🚀 Modern Foundation LLM Reality: How New Models Follow This</strong><br>
Modern retrieval models like <code>BGE-M3</code> and <code>GTE-Qwen2</code> can dynamically format queries for multiple retrieval tasks (code search, factual QA, multilingual cross-lingual matching) simply by swapping the instruction prefix, eliminating the need to fine-tune separate vector models for different domains.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Pure Python Masked Pooling & Asymmetric Bi-Encoder Engine", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. **Masked Mean Pooling**, **CLS Token Pooling**, and **Masked Max Pooling** implementations on batch token tensors.
2. An **Asymmetric Bi-Encoder Simulator** demonstrating query instruction formatting and cosine ranking against document passages.
""")

st.code('''
import numpy as np
from typing import Dict, List, Tuple, Optional

class ContextualPoolingEngine:
    """
    Production-grade educational engine demonstrating:
    1. Attention-Masked Mean, CLS, and Max Pooling across batched hidden states.
    2. Unit L2 Normalization for dot-product equivalence.
    3. Asymmetric Instruction-Conditioned Query-Document Matching.
    """
    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim

    @staticmethod
    def masked_mean_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Computes masked mean pooling across token representations.
        
        Args:
            last_hidden_state: Tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Binary tensor of shape [batch_size, seq_len] (1 for valid, 0 for pad)
            
        Returns:
            Normalized sentence vectors of shape [batch_size, hidden_dim]
        """
        # Expand attention mask to [batch_size, seq_len, 1]
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        
        # Zero out padding representations
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        
        # Clamp sum of masks to prevent division by zero
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        mean_pooled = sum_embeddings / sum_mask
        
        # L2 Normalization
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return mean_pooled / norms

    @staticmethod
    def cls_token_pooling(last_hidden_state: np.ndarray) -> np.ndarray:
        """Extracts the first token ([CLS]) vector for each sequence."""
        cls_vectors = last_hidden_state[:, 0, :]
        norms = np.linalg.norm(cls_vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return cls_vectors / norms

    @staticmethod
    def masked_max_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Extracts dimension-wise maximum across valid tokens."""
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        # Set padded tokens to large negative value
        modified_hidden = last_hidden_state.copy()
        modified_hidden[input_mask_expanded == 0] = -1e9
        max_pooled = np.max(modified_hidden, axis=1)
        norms = np.linalg.norm(max_pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return max_pooled / norms

    def rank_documents(self, query_vec: np.ndarray, doc_vecs: np.ndarray, doc_texts: List[str]) -> List[Tuple[str, float]]:
        """Calculates cosine similarity ranking between 1 query vector and N document vectors."""
        # Because vectors are L2-normalized, cosine similarity is pure dot product
        similarities = np.dot(doc_vecs, query_vec.squeeze())
        ranked_indices = np.argsort(similarities)[::-1]
        
        return [(doc_texts[i], float(similarities[i])) for i in ranked_indices]
''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Contextual Representation & Pooling Simulator", divider="rainbow")

st.markdown("""
Inspect how **Attention Masks** protect Sentence Embeddings during **Mean Pooling**, 
and test how **Instruction Prefixes** bridge the semantic gap in **Asymmetric Query-Document Retrieval**.
""")

wb_mode = st.radio("Select Interactive Workbench Mode", ["Pooling Strategy Comparison", "Asymmetric Query-Passage Matcher"], horizontal=True)

if wb_mode == "Pooling Strategy Comparison":
    st.markdown("#### Sentence Pooling Simulator on Padded Sequence")
    
    sample_text = st.text_input("Sample Sentence", value="The central bank lowered interest rates.")
    pad_tokens = st.slider("Number of Zero Padding Tokens to Append", min_value=0, max_value=10, value=4)
    
    # Generate synthetic tokens
    words = sample_text.split()
    total_len = len(words) + pad_tokens
    np.random.seed(42)
    dim = 8
    
    # Synthetic token representations: meaningful words have random vectors, pad tokens are 0.0
    hidden_states = np.random.randn(1, total_len, dim).astype(np.float32)
    attn_mask = np.ones((1, total_len), dtype=np.int32)
    
    if pad_tokens > 0:
        hidden_states[:, len(words):, :] = 0.0 # Pad positions set to zero
        attn_mask[:, len(words):] = 0          # Attention mask 0 for pads
        
    # Compute Proper Masked Mean Pooling
    mask_expanded = np.expand_dims(attn_mask, axis=-1).astype(np.float32)
    sum_emb = np.sum(hidden_states * mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
    masked_mean = sum_emb / sum_mask
    masked_mean_norm = masked_mean / np.linalg.norm(masked_mean)
    
    # Compute Naive Unmasked Mean Pooling (Corrupted)
    naive_mean = np.mean(hidden_states, axis=1)
    naive_mean_norm = naive_mean / np.linalg.norm(naive_mean)
    
    # Compute CLS Pooling (Token 0)
    cls_emb = hidden_states[:, 0, :]
    cls_emb_norm = cls_emb / np.linalg.norm(cls_emb)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("**1. Masked Mean Pooling (Correct)**")
        st.write(np.round(masked_mean_norm[0, :4], 4))
        st.caption("Ignores padding; perfectly stable across any pad length.")
    with col_p2:
        st.markdown("**2. Naive Unmasked Mean Pooling (Flawed)**")
        st.write(np.round(naive_mean_norm[0, :4], 4))
        sim_to_true = float(np.dot(masked_mean_norm[0], naive_mean_norm[0]))
        st.caption(f"Distorted by padding! Cosine match to true vector: {sim_to_true:.4f}")
    with col_p3:
        st.markdown("**3. CLS Token Pooling**")
        st.write(np.round(cls_emb_norm[0, :4], 4))
        st.caption("Takes only token 0 ('The'). Requires contrastive pretraining.")

else:
    st.markdown("#### Asymmetric Retrieval: Query Instruction Impact")
    
    query_text = st.text_input("User Search Query", value="how to recover lost files")
    apply_instruction = st.checkbox("Prepend Asymmetric Instruction ('Represent this sentence for searching relevant passages: ')", value=True)
    
    candidate_docs = [
        "Passage A: Data recovery utilities can scan deleted sectors on hard drives to restore lost files.",
        "Passage B: Operating system updates frequently install security patches and bug fixes.",
        "Passage C: In case of catastrophic storage loss, file recovery procedures should be initiated.",
        "Passage D: What are the best ways to recover lost files on a computer?"
    ]
    
    # Synthetic semantic simulator
    np.random.seed(123)
    recovery_topic = np.array([2.5, 1.2, -0.5, 0.8])
    noise = np.random.randn(4, 4) * 0.1
    
    doc_vectors = np.array([
        recovery_topic + [0.2, 0.1, -0.1, 0.0],   # Doc A: Explanatory
        np.array([-1.5, 0.5, 2.0, -0.8]),          # Doc B: Irrelevant
        recovery_topic + [0.3, -0.2, 0.1, 0.1],  # Doc C: Explanatory
        recovery_topic * 0.8 + [1.0, 1.0, 0.0, 0.0] # Doc D: Question syntax!
    ])
    doc_vectors = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    
    # Query vector calculation
    raw_query_vec = recovery_topic * 0.8 + np.array([1.1, 0.9, 0.0, 0.0]) # Has question syntax
    if apply_instruction:
        # Instruction rotates query toward explanatory passage subspace
        aligned_query_vec = raw_query_vec * 0.4 + recovery_topic * 1.2
    else:
        aligned_query_vec = raw_query_vec
        
    aligned_query_vec = aligned_query_vec / np.linalg.norm(aligned_query_vec)
    
    scores = np.dot(doc_vectors, aligned_query_vec)
    
    res_data = []
    for doc, s in zip(candidate_docs, scores):
        res_data.append({"Passage": doc, "Cosine Score": f"{s:.4f}", "Rank": 0})
    res_df = pd.DataFrame(res_data).sort_values("Cosine Score", ascending=False).reset_index(drop=True)
    res_df["Rank"] = res_df.index + 1
    
    st.dataframe(res_df[["Rank", "Cosine Score", "Passage"]], use_container_width=True, hide_index=True)
    if not apply_instruction:
        st.warning("Notice: Without instruction tuning, symmetric models often rank Passage D (another question) above actual answers Passage A and C!")
    else:
        st.success("Instruction prefix successfully prioritized explanatory passages (Passage A and C) over question syntax duplicates!")

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why is naive CLS token pooling from an off-the-shelf BERT model ineffective for semantic textual similarity without fine-tuning?",
        "a": "Standard BERT was pretrained using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). The [CLS] token was only trained to separate sentence pairs in a classification head (IsNext vs NotNext), not to construct an isotropic metric space for cosine distance. Raw BERT [CLS] embeddings suffer from severe representation anisotropy (vectors occupy a narrow cone in vector space), yielding artificially high cosine similarities (0.85-0.95) across completely unrelated sentences. Models like Sentence-BERT or BGE resolve this through contrastive Siamese fine-tuning."
    },
    {
        "q": "How does Masked Mean Pooling prevent batch padding tokens from corrupting sentence vectors?",
        "a": "In a batched tensor of shape [Batch_Size, Seq_Len, Hidden_Dim], sentences of unequal lengths are padded with zero tokens. Naive averaging divides by Seq_Len including pads, dragging values toward zero and altering the vector direction. Masked Mean Pooling explicitly multiplies hidden states by the binary attention mask [Batch_Size, Seq_Len, 1] to zero out pad representations, and divides only by the sum of valid tokens (np.sum(attention_mask, axis=1))."
    },
    {
        "q": "What is the architectural trade-off between a Bi-Encoder (Dual-Encoder) and a Cross-Encoder for information retrieval?",
        "a": "A Bi-Encoder encodes query and document independently into two dense vectors u and v. Embeddings can be precomputed and indexed in a Vector DB, allowing sub-10ms approximate nearest neighbor search over millions of documents via dot product. However, it cannot model cross-attention between query and document tokens. A Cross-Encoder feeds both texts concatenated into a single Transformer ([CLS] Query [SEP] Doc), allowing all-to-all cross-attention across all tokens. It provides significantly higher ranking accuracy but requires running full Transformer inference for every single (query, doc) pair, making it too slow for initial retrieval but ideal as a second-stage reranker."
    },
    {
        "q": "What is Late Interaction in ColBERT, and why does it outperform single-vector document pooling?",
        "a": "Single-vector document pooling collapses an entire document into one dense vector, discarding fine-grained facts and token relationships. ColBERT retains all token embeddings for both query and document. At retrieval time, it computes the MaxSim operator: for every query token, it finds the maximum cosine similarity across all document tokens, and sums these maximums. This captures token-level semantic alignments while remaining fast because the document token vectors can be precomputed and compressed using vector quantization."
    },
    {
        "q": "Why do modern embedding models like E5 and BGE require instruction prefixes for query encoding?",
        "a": "Queries and target passages have different linguistic structures: queries are short questions (3-10 words), while passages are long declarative statements. In symmetric vector space, questions cluster near other questions rather than answers. By prefixing the query with an instruction like 'Represent this sentence for searching relevant passages: ', the self-attention mechanism modulates the query representation into the semantic space of explanatory answers."
    }
]

for i, qa in enumerate(qas, 1):
    with st.expander(f"Q{i}: {qa['q']}"):
        st.markdown(f"**Answer**: {qa['a']}")

st.markdown("""
### Production Traps & Failure Modes
""")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1 (Fatal): Omitting L2 Normalization before Vector Database Ingestion**
    * *Failure*: Ingesting raw pooled vectors into a Vector DB configured for Cosine or Inner Product search without unit normalization (`vec = vec / norm(vec)`).
    * *Fix*: Always apply L2 normalization immediately after pooling. Most Vector DBs (Pinecone, Qdrant, Milvus, FAISS) achieve peak indexing speed when vectors are normalized unit spheres where Dot Product equals Cosine Similarity.
    """)
    st.error("""
    **Trap 2 (Fatal): Evaluating Off-the-Shelf LLM Hidden States with Cosine Distance**
    * *Failure*: Taking the final layer hidden states of a generative decoder (Llama 3, Mistral) without contrastive fine-tuning and using them for semantic search.
    * *Fix*: Autoregressive decoder hidden states are optimized for next-token prediction, not sentence-level metric proximity. Always use dedicated embedding checkpoints (e.g., `bge-en-icl`, `gte-qwen2`) or fine-tune with InfoNCE loss.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Query-Document Instruction Mismatch**
    * *Failure*: Applying the query instruction prefix (`"Represent this sentence: "`) to both the query and the indexed document chunks.
    * *Fix*: For models like BGE and E5, the instruction prompt is strictly asymmetric: apply the instruction prefix to the **query only**, leaving document passages clean.
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Cross-Encoder at Scale**
    * *Failure*: Attempting to pass 10,000 document candidates directly through a Cross-Encoder reranker.
    * *Fix*: Never run cross-encoders over more than 50 to 100 top candidates retrieved in Stage 1 by a Bi-Encoder. Cross-encoders require quadratic compute and will create severe latency timeouts.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Bi-Encoder vs Cross-Encoder**\n\nBi-Encoder embeds independently (fast, scalable, vector DB). Cross-Encoder attends jointly (accurate, slow, reranker).")
with fc_col2:
    st.info("**Masked Mean Pooling**\n\nMultiplies by attention mask to zero out pads, divides by count of valid tokens. Essential for batch stability.")
with fc_col3:
    st.info("**Asymmetric Instructions**\n\nQuery prefixes ('Represent this sentence...') rotate question vectors into the complementary space of answer passages.")
with fc_col4:
    st.info("**ColBERT MaxSim**\n\nRetains token embeddings; calculates sum of maximum cosine similarities per query token. Bridges dense and token retrieval.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of contextual embeddings, pooling mechanisms, and enterprise retrieval architectures:
""")

c1 = st.checkbox("I can explain how self-attention dynamically resolves polysemy compared to static centroids.")
c2 = st.checkbox("I understand why Masked Mean Pooling is mathematically necessary when batching padded sequences.")
c3 = st.checkbox("I know the difference between Bi-Encoder retrieval (Stage 1) and Cross-Encoder reranking (Stage 2).")
c4 = st.checkbox("I can describe why asymmetric query instruction prefixes are essential in modern dense search.")
c5 = st.checkbox("I understand how ColBERT's late-interaction MaxSim operator preserves token-level granularity.")
c6 = st.checkbox("I know why raw generative LLM hidden states suffer from anisotropy and require contrastive fine-tuning for embedding tasks.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered contextual embeddings, multi-stage retrieval pipelines, and pooling mechanics across modern foundation AI systems!")
