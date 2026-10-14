import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="18.3 Metadata + Semantic Search - Embeddings Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("18.3 Metadata + Semantic Search: Unified Filtering & Continuous Vector Proximity")
st.caption("Chapter 18: Metadata and Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In real-world applications, search problems rarely conform to either pure relational SQL or pure vector similarity alone:
* **Relational SQL Filters** provide **deterministic hard truth**: binary yes/no decisions on security clearance, tenant ID, or publication date. However, SQL fails at understanding natural language meaning.
* **Vector Embeddings** provide **continuous probabilistic relevance**: ranking documents by semantic nuance and conceptual proximity. However, vectors cannot enforce hard security boundaries or date ranges.

**Unified Metadata + Semantic Search** combines both paradigms into a cohesive two-tier engine:
1. **Hard Filtering Tier**: Deterministically enforces organizational boundaries, security permissions, and temporal freshness.
2. **Soft Semantic Tier**: Ranks the pre-filtered candidate pool by dense cosine similarity.

This lesson explores how production systems synthesize metadata constraints with vector similarity to power enterprise RAG.
""")

st.divider()

# Section 1: The Unified Dual-Tier Query Architecture
st.header("1. Dual-Tier Unified Query Architecture", divider="rainbow")

col_uni1, col_uni2 = st.columns([1, 1])

with col_uni1:
    st.markdown("""
    ### Hard Constraints vs. Soft Semantic Proximity
    * **Tier 1: Hard Constraints (Deterministic Filtering)**:
      * Enforces Boolean truth values: `tenant == 'acme' AND clearance >= 3 AND year >= 2023`.
      * Evaluated via inverted bitmasks or SQLite predicates.
      * Non-negotiable: a document that fails Tier 1 is **strictly forbidden** from appearing in search results, regardless of how high its vector similarity is.
    * **Tier 2: Soft Semantic Proximity (Continuous Vector Ranking)**:
      * Evaluates high-dimensional geometric distance: `CosineSimilarity(q, d)`.
      * Produces a continuous score between -1.0 and 1.0.
      * Ranks candidate documents that successfully passed Tier 1 from most relevant to least relevant.
    """)

with col_uni2:
    st.markdown("### Unified Query Processing Pipeline")
    uni_dot = """
    digraph UnifiedSearch {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserPrompt [label="User Input:\\n'HIPAA audit compliance in S3'\\n+ Context: {dept='healthcare', year>=2023}", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_tier1 {
            label="Tier 1: Hard Filter Gate"; style="dashed"; color="#DC2626";
            GateFilter [label="Metadata Bitmask Check\\ndept == 'healthcare' AND year >= 2023", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];
            SubsetEligible [label="Eligible Candidates\\n(Strictly Authorized Subset)", fillcolor="#FED7AA"];
        }

        subgraph cluster_tier2 {
            label="Tier 2: Soft Semantic Ranking"; style="dashed"; color="#10B981";
            DenseScore [label="Cosine Vector Traversal\\nEvaluates similarity strictly on eligible subset", fillcolor="#D1FAE5"];
            RankedTopK [label="Final Grounded Top-K\\nHigh relevance + 100% compliant", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
        }

        UserPrompt -> GateFilter [label="Apply hard constraints"];
        GateFilter -> SubsetEligible [label="Prune unauthorized rows"];
        SubsetEligible -> DenseScore [label="Score eligible nodes"];
        UserPrompt -> DenseScore [label="Query embedding vector"];
        DenseScore -> RankedTopK [label="Top-K nearest"];
    }
    """
    st.graphviz_chart(uni_dot, use_container_width=False)

st.markdown("### The Three Architectural Configurations")
st.markdown("""
| Search Mode | Hard Filter Support | Semantic Ranking | Speed (1M items) | Risk Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Semantic Search** | None | Yes (Top-K) | ~3 - 5 ms | High security risk (leaks cross-tenant data, outdated docs). |
| **Pure Metadata Filter** | Yes (Exact match) | None (Arbitrary order) | ~1 - 2 ms | Poor relevance (returns unordered matching rows). |
| **Unified Search** | **Yes (Guaranteed)** | **Yes (Ranked Top-K)** | **~3 - 6 ms** | **Zero cross-tenant leaks, optimal factual precision.** |
""")

st.divider()

# Section 2: Production Code: Unified Hybrid Engine
st.header("2. Production Code: Unified Hybrid Search Engine", divider="rainbow")

st.markdown("""
The production class below models a Unified Hybrid Search engine: evaluating hard boolean predicates before ranking eligible candidates by cosine similarity.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class UnifiedHybridSearchEngine:
    """
    Combines deterministic metadata predicate filtering with
    continuous cosine vector similarity ranking.
    """
    def __init__(self, embedding_dim: int = 64):
        self.dim = embedding_dim
        self.corpus: List[Dict[str, Any]] = []

    def mock_embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.lower().strip())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        vec = self.mock_embed(text)
        self.corpus.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata,
            "vector": vec
        })

    def search(
        self,
        query: str,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        q_vec = self.mock_embed(query)

        # Step 1: Evaluate Hard Metadata Constraints (Pruning)
        eligible_docs = []
        for doc in self.corpus:
            if filter_fn is None or filter_fn(doc["metadata"]):
                eligible_docs.append(doc)

        if not eligible_docs:
            return []

        # Step 2: Evaluate Soft Semantic Similarity on Eligible Subset
        scored = []
        for doc in eligible_docs:
            sim = float(np.dot(doc["vector"], q_vec))
            scored.append((doc, sim))

        # Step 3: Rank and return top K
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
''', language="python")

st.divider()

# Section 3: Interactive Hybrid Search Arena
st.header("3. Interactive Workbench: Pure Semantic vs. Filtered vs. Unified", divider="rainbow")

st.markdown("""
Experience why Unified Metadata + Semantic Search is mandatory for enterprise applications.
Test all 3 search modes across a curated corpus of 8 documents spanning Healthcare, Cloud, and Finance.
""")

class InteractiveHybridSimulator:
    def __init__(self, dim: int = 48):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        t_low = text.lower()
        np.random.seed(abs(hash(t_low)) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        
        # Add semantic topic bias
        if any(w in t_low for w in ["hipaa", "patient", "medical", "health", "ehr"]):
            v[:16] += 2.8
        elif any(w in t_low for w in ["s3", "storage", "cloud", "aws", "bucket"]):
            v[16:32] += 2.8
        elif any(w in t_low for w in ["tax", "revenue", "audit", "finance"]):
            v[32:] += 2.8
            
        return v / np.maximum(np.linalg.norm(v), 1e-12)

hybrid_sim = InteractiveHybridSimulator()

hybrid_corpus = [
    {"id": "doc_med_01", "dept": "healthcare", "year": 2024, "clearance": "confidential", "text": "HIPAA compliance rules mandate encryption of electronic protected health information stored in cloud buckets."},
    {"id": "doc_med_02", "dept": "healthcare", "year": 2021, "clearance": "public", "text": "Outdated 2021 medical privacy advisory notice regarding local server storage guidelines."},
    {"id": "doc_cld_03", "dept": "cloud_eng", "year": 2024, "clearance": "public", "text": "Amazon S3 server-side encryption with KMS provides scalable data protection for cloud applications."},
    {"id": "doc_cld_04", "dept": "cloud_eng", "year": 2023, "clearance": "internal", "text": "Cloud object lifecycle policies transition inactive data partitions to cold archive tiers."},
    {"id": "doc_fin_05", "dept": "finance", "year": 2024, "clearance": "confidential", "text": "Sarbanes-Oxley audit trails require immutable log retention for all financial transactions."},
    {"id": "doc_fin_06", "dept": "finance", "year": 2022, "clearance": "internal", "text": "Archived 2022 quarterly revenue reconciliation spreadsheets for domestic business units."},
    {"id": "doc_med_07", "dept": "healthcare", "year": 2024, "clearance": "restricted", "text": "Clinical trial patient data access protocols require multi-factor biometrics and hardware tokens."},
    {"id": "doc_cld_08", "dept": "cloud_eng", "year": 2024, "clearance": "confidential", "text": "Healthcare cloud workload isolation using dedicated single-tenant VPC peering connections."}
]

col_u1, col_u2 = st.columns([1, 1])

with col_u1:
    st.subheader("1. Query & Metadata Criteria")
    user_q = st.text_input("Semantic Search Query:", value="HIPAA encryption for cloud storage")

    st.subheader("2. Hard Metadata Governance Filter")
    f_dept = st.selectbox("Department Restriction (`dept`):", ["all", "healthcare", "cloud_eng", "finance"], index=1)
    f_min_year = st.slider("Minimum Publication Year (`year` >=):", min_value=2021, max_value=2024, value=2023)
    f_clearance = st.multiselect("Allowed Clearance Levels:", ["public", "internal", "confidential", "restricted"], default=["public", "internal", "confidential"])

    st.subheader("3. Select Execution Mode")
    exec_mode = st.radio(
        "Search Execution Paradigm:",
        ["1. Pure Semantic Vector Search (No Filters)", "2. Pure Metadata Filter (No Vector Ranking)", "3. Unified Metadata + Semantic Search"],
        index=2
    )
    k_res = st.slider("Max Results ($K$):", min_value=1, max_value=5, value=3)

with col_u2:
    st.subheader(f"4. Execution Results: {exec_mode.split('(')[0]}")

    q_vec = hybrid_sim.embed(user_q)

    # Mode 1: Pure Semantic Search
    if "1. Pure Semantic" in exec_mode:
        scored = []
        for d in hybrid_corpus:
            sim = float(np.dot(hybrid_sim.embed(d["text"]), q_vec))
            scored.append((d, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:k_res]

        st.warning("⚠️ **Pure Semantic Mode**: Notice outdated 2021 documents or unauthorized departments can leak into top results!")
        for rank, (doc, sim) in enumerate(results, 1):
            with st.expander(f"Rank {rank} | {doc['id']} | Cosine: {sim:.4f}", expanded=True):
                st.markdown(f"> {doc['text']}")
                st.caption(f"Dept: `{doc['dept']}` | Year: `{doc['year']}` | Clearance: `{doc['clearance']}`")

    # Mode 2: Pure Metadata Filter
    elif "2. Pure Metadata" in exec_mode:
        filtered = []
        for d in hybrid_corpus:
            if (f_dept == "all" or d["dept"] == f_dept) and d["year"] >= f_min_year and d["clearance"] in f_clearance:
                filtered.append(d)
        
        st.info("ℹ️ **Pure Metadata Mode**: Documents pass hard security checks, but results have ZERO semantic relevance ranking.")
        if not filtered:
            st.error("No documents matched the hard filter criteria.")
        else:
            for idx, doc in enumerate(filtered[:k_res], 1):
                with st.expander(f"Match {idx} | {doc['id']} (Arbitrary Order)", expanded=True):
                    st.markdown(f"> {doc['text']}")
                    st.caption(f"Dept: `{doc['dept']}` | Year: `{doc['year']}` | Clearance: `{doc['clearance']}`")

    # Mode 3: Unified Metadata + Semantic Search
    else:
        unified_candidates = []
        for d in hybrid_corpus:
            # Check hard constraints
            if (f_dept == "all" or d["dept"] == f_dept) and d["year"] >= f_min_year and d["clearance"] in f_clearance:
                sim = float(np.dot(hybrid_sim.embed(d["text"]), q_vec))
                unified_candidates.append((d, sim))
        
        unified_candidates.sort(key=lambda x: x[1], reverse=True)
        results = unified_candidates[:k_res]

        st.success("✅ **Unified Mode**: 100% compliant with security/date filters AND sorted by semantic relevance!")
        if not results:
            st.warning("No documents matched both the hard filter criteria and semantic search.")
        else:
            for rank, (doc, sim) in enumerate(results, 1):
                with st.expander(f"Rank {rank} | {doc['id']} | Cosine: {sim:.4f}", expanded=True):
                    st.markdown(f"> {doc['text']}")
                    st.caption(f"Dept: `{doc['dept']}` | Year: `{doc['year']}` | Clearance: `{doc['clearance']}`")
                    st.progress(float(min(1.0, max(0.0, sim))))

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Hybrid Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is a 'Self-Querying Retriever' and how does it automate Metadata + Semantic Search?**
    * **Answer**: In standard RAG, the user prompt is a raw natural language string (e.g. *"Show me 2024 healthcare security manuals"*). A **Self-Querying Retriever** uses an LLM to decompose the user prompt into two distinct components:
      1. A **Semantic Search Query**: `"security manuals"`
      2. A **Structured Metadata Filter**: `{"year": 2024, "department": "healthcare"}`
      It executes the combined query against the vector database automatically without requiring the user to fill out form fields or dropdown menus.

    **Q2: What is the primary operational tradeoff between Pre-Filtering and Single-Stage Filtered HNSW search?**
    * **Answer**:
      * **Pre-Filtering (Subset Brute Force)**: Evaluates metadata in relational storage first. If the resulting subset is small (e.g. 50 items), a quick brute-force dot product is blazing fast (< 1ms). However, if the subset contains 500,000 items, brute-forcing is slow.
      * **Single-Stage Filtered HNSW**: Navigates the vector graph directly while filtering node bitmasks. Outstanding when the subset is large, but can suffer graph routing slowdowns if the filter is hyper-restrictive (< 0.1% matching nodes).

    **Q3: How do you handle queries where the hard metadata filter returns zero results?**
    * **Answer**: In production, implement **Graceful Fallback Logic**:
      1. Check if the candidate count after hard filtering is 0.
      2. Rather than returning an empty prompt to the LLM (which induces hallucinations), fall back to relaxing non-critical filters (e.g. expand the date window from 2024 to 2022).
      3. Return the relaxed results with an explicit warning banner: *"No exact matches found for 2024; displaying relevant documents from 2022-2023."*
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Hardcoding Security Clearance Filters in the Client Application**:
      * *Issue*: Allowing the frontend client to specify which `tenant_id` to filter by.
      * *Consequence*: A malicious user modifies the HTTP payload to request another tenant's ID, bypassing multi-tenant isolation.
      * *Fix*: Always inject tenant IDs and user security clearance levels server-side from verified JWT session claims.

    * **Trap 2: Ignoring Cold-Start Cache Misses on Filtered Indices**:
      * *Issue*: Users submit unpredictable random combinations of dynamic metadata filters.
      * *Consequence*: The vector database cannot keep all secondary payload indexes cached in RAM, causing heavy disk IO latency spikes.
      * *Fix*: Restrict filtering to curated categorical and temporal fields, and warm secondary indexes during startup.

    * **Trap 3: Over-Constraining Filters Resulting in Empty Prompts**:
      * *Issue*: Combining 5 strict filters that leave 0 matching documents.
      * *Consequence*: The LLM generates excuses or hallucinated answers.
      * *Fix*: Implement fallback thresholding or alert the user to broaden their search criteria.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Unified Dual-Tier Search**
      * *Definition*: The architectural fusion of deterministic metadata filtering (hard gate) with continuous vector similarity ranking (soft sorting).
    * **Flashcard 2: Self-Querying Retriever**
      * *Definition*: An LLM-powered retriever that automatically parses natural language questions into an embedding query and a structured metadata filter.
    * **Flashcard 3: Server-Side Tenant Injection**
      * *Definition*: Enforcing tenant isolation by injecting validated user identity into vector query filters at the backend layer.
    * **Flashcard 4: Graceful Filter Relaxation**
      * *Definition*: The fallback mechanism of widening restrictive date or category filters when an initial query yields zero results.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Metadata + Semantic Search:

- [ ] I can articulate the structural difference between hard metadata constraints and soft semantic ranking.
- [ ] I understand why Pure Semantic Search introduces security and obsolescence risks.
- [ ] I know how a Self-Querying Retriever decomposes a prompt into query and filter components.
- [ ] I can implement Unified Hybrid search that guarantees both compliance and relevance.
- [ ] I understand how to handle empty-result edge cases with graceful filter relaxation.
""")
