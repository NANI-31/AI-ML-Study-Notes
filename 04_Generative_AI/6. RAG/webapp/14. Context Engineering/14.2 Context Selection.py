import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class ContextSelectionEngine:
    """
    Simulates enterprise Context Selection operations:
    1. Selecting top candidate chunks.
    2. Removing duplicate & near-duplicate chunks (Jaccard similarity threshold).
    3. Pruning irrelevant chunks via confidence cutoffs.
    4. Ranking context by composite relevance scores.
    5. Context filtering based on metadata tags (category, recency, clearance).
    """
    CORPUS = [
        {
            "id": "c_01",
            "title": "PostgreSQL Transaction Pooling Architecture",
            "text": "PgBouncer in transaction pooling mode releases server connections immediately when a transaction ends, maximizing client concurrency.",
            "score": 0.94,
            "category": "Architecture",
            "year": 2024,
            "security_clearance": "Public"
        },
        {
            "id": "c_02",
            "title": "PgBouncer Connection Reuse (Near-Duplicate of c_01)",
            "text": "In transaction pooling, PgBouncer server connections are released immediately upon transaction completion, ensuring high concurrency.",
            "score": 0.92,
            "category": "Architecture",
            "year": 2024,
            "security_clearance": "Public"
        },
        {
            "id": "c_03",
            "title": "Internal Database Passwords & Credentials",
            "text": "Master administrative credentials for production database clusters are stored in Vault vault-internal-prod-01.",
            "score": 0.89,
            "category": "Credentials",
            "year": 2024,
            "security_clearance": "Confidential"
        },
        {
            "id": "c_04",
            "title": "Legacy PostgreSQL 9.4 Pool Settings (Outdated)",
            "text": "Configuring pg_pool in legacy PostgreSQL 9.4 for Debian 8 servers with md5 authentication.",
            "score": 0.72,
            "category": "Legacy",
            "year": 2017,
            "security_clearance": "Public"
        },
        {
            "id": "c_05",
            "title": "General Web Server Tuning",
            "text": "Apache HTTP server MaxRequestWorkers directive controls the number of simultaneous child processes.",
            "score": 0.35,
            "category": "WebOps",
            "year": 2023,
            "security_clearance": "Public"
        }
    ]

    @staticmethod
    def jaccard_similarity(s1: str, s2: str) -> float:
        w1 = set(s1.lower().split())
        w2 = set(s2.lower().split())
        union = len(w1.union(w2))
        return len(w1.intersection(w2)) / union if union > 0 else 0.0

    @classmethod
    def execute_selection(
        cls,
        min_score: float = 0.60,
        dedup_threshold: float = 0.60,
        allow_confidential: bool = False,
        min_year: int = 2020
    ) -> Dict[str, Any]:
        initial_pool = list(cls.CORPUS)
        audit_trail = []

        # 1. Metadata Security & Recency Filtering
        metadata_filtered = []
        for c in initial_pool:
            if not allow_confidential and c["security_clearance"] == "Confidential":
                audit_trail.append(f"Dropped {c['id']}: Security Clearance 'Confidential' blocked by RBAC.")
                continue
            if c["year"] < min_year:
                audit_trail.append(f"Dropped {c['id']}: Outdated year ({c['year']} < {min_year}).")
                continue
            metadata_filtered.append(c)

        # 2. Removing Irrelevant Chunks (Relevance Score Cutoff)
        score_filtered = []
        for c in metadata_filtered:
            if c["score"] >= min_score:
                score_filtered.append(c)
            else:
                audit_trail.append(f"Dropped {c['id']}: Score ({c['score']:.2f}) < threshold ({min_score:.2f}).")

        # 3. Deduplication (Jaccard Near-Duplicate Pruning)
        deduped = []
        for c in score_filtered:
            is_dup = False
            for existing in deduped:
                sim = cls.jaccard_similarity(c["text"], existing["text"])
                if sim >= dedup_threshold:
                    audit_trail.append(f"Dropped {c['id']}: Near-duplicate of {existing['id']} (Jaccard: {sim:.2f}).")
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(c)

        # 4. Ranking (Sort descending by score)
        ranked = sorted(deduped, key=lambda x: x["score"], reverse=True)

        return {
            "initial_count": len(initial_pool),
            "surviving_count": len(ranked),
            "selected_chunks": ranked,
            "audit_trail": audit_trail
        }

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="14.2 Context Selection - RAG Master Curriculum",
    page_icon="🎯",
    layout="wide"
)

st.title("14.2 Context Selection: Chunk Deduplication, Relevance Gating, & Metadata Filtering")
st.caption("Chapter 14: Context Engineering | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Retrieving 20 document candidates from an index is only the first step. 
Raw candidate pools are routinely contaminated with near-duplicate paragraphs, outdated guidelines, 
unauthorized internal credentials, and weak topical matches.

**Context Selection** is the algorithmic decision gate that filters, deduplicates, and ranks candidate 
evidence before it can be injected into the LLM context prompt:
* **Deduplication**: Eliminates near-identical text chunks that waste token budgets without adding information.
* **Irrelevance Pruning**: Enforces absolute relevance cutoffs, dropping low-confidence distractors.
* **Metadata Filtering**: Applies temporal recency gates, security classification rules, and category constraints.
* **Context Ranking**: Re-orders surviving evidence to ensure maximum attention focus.

This lesson explores the operational mechanics of building a production Context Selection pipeline.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Context Selection and Gating Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    selection_bp_dot = """
    digraph ContextSelectionBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_ingress {
            label = "1. Raw Candidates Ingress Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            raw_pool [label="Raw Candidate Chunks (K = 25)\\nFrom Vector / Hybrid Search", fillcolor="#0F172A", color="#38BDF8"];
            auth_context [label="User Auth Tokens & Security Clearance\\nRBAC Permissions & Tenant Scope", fillcolor="#0F172A", color="#8B5CF6"];
        }

        subgraph cluster_gates {
            label = "2. Multi-Stage Gating & Filtering Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            rbac_gate [label="Security & Metadata Gate\\nDrops Confidential / Outdated Docs", fillcolor="#334155", color="#EC4899"];
            score_gate [label="Relevance Confidence Cutoff\\nDrops Chunks with Score < 0.65", fillcolor="#334155", color="#F59E0B"];
            dedup_gate [label="Near-Duplicate Jaccard / Hash Gate\\nPrunes Redundant Overlapping Chunks", fillcolor="#334155", color="#10B981"];

            raw_pool -> rbac_gate;
            auth_context -> rbac_gate [label="Enforce Permissions"];
            rbac_gate -> score_gate [label="Authorized Chunks"];
            score_gate -> dedup_gate [label="Relevant Chunks"];
        }

        subgraph cluster_assembly {
            label = "3. Final Context Ranking & Assembly Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            ranker_node [label="Composite Relevance Ranking\\nSorts Surviving Clean Chunks", fillcolor="#0F172A", color="#10B981"];
            final_topk [label="Pristine Top Context Chunks (N = 3)\\nZero Duplicates, High SNR, Fully Authorized", fillcolor="#334155", color="#38BDF8"];

            dedup_gate -> ranker_node;
            ranker_node -> final_topk;
        }
    }
    """
    st.graphviz_chart(selection_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production RAG systems, context selection acts as a defense-in-depth security and quality gate. 
Even if a vector database accidentally surfaces a confidential HR compensation document during an IT search, 
the **Security & Metadata Gate** intercepts the document and removes it before it can be added to the prompt. 
Concurrently, **Semantic Deduplication** ensures that identical chunks extracted from mirrored files are consolidated, 
saving up to 30% of total prompt token costs.
""")

st.divider()

# ==============================================================================
# SECTION 2: SELECTING RELEVANT CHUNKS
# ==============================================================================

st.header("2. Selecting Relevant Chunks: Dynamic Thresholding vs. Fixed-K Heuristics", divider="rainbow")

st.markdown("""
How does dynamic context selection improve prompt quality over rigid fixed-K heuristics?
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Failure of Fixed-K Heuristics")
    st.markdown("""
    * **The Fixed-K Trap**:
      * In naive systems, developers hardcode `top_k = 5`.
      * *Case 1 (Easy Question)*: Only 1 document is relevant. The system injects 1 good document and 4 distractor documents!
      * *Case 2 (Complex Synthesis)*: 8 documents are required. The system truncates at 5, starving the LLM of necessary facts!
      * *Case 3 (Out-of-Domain)*: 0 documents are relevant. The system injects 5 completely unrelated noise documents, forcing hallucinations!
    """)

with col2:
    st.subheader("Dynamic Score Thresholding")
    st.markdown("""
    * **Quality-Driven Selection**:
      * Instead of selecting a fixed count, the system selects all documents where:
        `Relevance_Score(doc) >= Threshold`.
      * If 1 document meets the bar, 1 document is selected.
      * If 6 documents meet the bar, 6 documents are selected (up to the token budget ceiling).
      * If 0 documents meet the bar, the system aborts retrieval, preventing hallucinations.
    """)

st.divider()

# ==============================================================================
# SECTION 3: REMOVING DUPLICATES
# ==============================================================================

st.header("3. Removing Duplicates: Exact Hashing and Near-Duplicate Jaccard Pruning", divider="rainbow")

st.markdown("""
In enterprise data lakes, the same text frequently appears across multiple files: mirrored PDF versions, 
copied email threads, and multi-version documentation.
""")

col_dup1, col_dup2 = st.columns([1, 1])

with col_dup1:
    st.subheader("Why Duplicate Chunks Degrade RAG")
    st.markdown("""
    * **Wasted Context Budget**:
      * If Chunk A and Chunk B express the exact same paragraph with minor formatting differences, 
        injecting both consumes double the token budget without adding a single new fact!
    * **Attention Crowding**:
      * Redundant chunks crowd out complementary facts that would have provided a complete answer.
    * **False Confidence Reinforcement**:
      * Repeating the same claim twice can bias the LLM into over-weighting a peripheral detail.
    """)

with col_dup2:
    st.subheader("Deduplication Mechanics")
    st.markdown("""
    1. **Exact Hash Deduplication (MD5 / SHA-256)**:
       * Instantly prunes byte-for-byte identical text strings (`O(1)` complexity).
    2. **Near-Duplicate Jaccard Shingling**:
       * Computes token set overlap: `Jaccard = |Words(A) ∩ Words(B)| / |Words(A) ∪ Words(B)|`.
       * If `Jaccard >= 0.60`, Chunk B is discarded as redundant narrative.
    3. **Semantic Embedding Deduplication**:
       * Computes cosine similarity between chunk vectors: `dot(v_A, v_B) >= 0.95`.
    """)

st.divider()

# ==============================================================================
# SECTION 4: REMOVING IRRELEVANT CHUNKS
# ==============================================================================

st.header("4. Removing Irrelevant Chunks: Confidence Floor and Hallucination Circuit Breakers", divider="rainbow")

st.markdown("""
Step 4 of context selection is pruning candidates that fall below the minimum relevance confidence floor.
""")

col_irr1, col_irr2 = st.columns([1, 1])

with col_irr1:
    st.subheader("The Confidence Floor Concept")
    st.markdown("""
    * **Calibrated Cross-Encoder Thresholds**:
      * Unlike cosine similarities (which cluster in uncalibrated bands), cross-encoder probabilities reflect 
        true relevance likelihood.
      * A threshold of `0.65` acts as a sharp discriminator:
        * Chunks with score `> 0.65` contain verified answer-bearing facts.
        * Chunks with score `< 0.65` are tangential background noise that degrade reasoning.
    * **Dynamic Threshold Scaling**:
      * High-stakes medical/legal queries: Set threshold to `0.75` (zero tolerance for noise).
      * Exploratory brainstorm queries: Set threshold to `0.50` (higher recall allowance).
    """)

with col_irr2:
    st.subheader("The Hallucination Circuit Breaker")
    st.markdown("""
    When an out-of-domain query is submitted:
    ```python
    surviving_evidence = [chunk for chunk in pool if chunk.score >= 0.65]
    if not surviving_evidence:
        return "I do not have authoritative information to answer this question."
    ```
    Bypassing LLM generation when zero candidates pass the confidence floor eliminates 98% of hallucinations!
    """)

st.divider()

# ==============================================================================
# SECTION 5: RANKING CONTEXT
# ==============================================================================

st.header("5. Ranking Context: Multi-Signal Composite Scoring and Recency Weighting", divider="rainbow")

st.markdown("""
How should surviving candidates be ranked before prompt insertion?
""")

col_rk1, col_rk2 = st.columns([1, 1])

with col_rk1:
    st.subheader("Multi-Signal Composite Ranking")
    st.markdown("""
    In enterprise systems, ranking does not depend solely on semantic similarity. 
    Systems compute a **Composite Ranking Score**:
    
    `Final_Rank_Score = (w_sem * Semantic_Score) + (w_rec * Recency_Score) + (w_auth * Authority_Score)`
    
    * **Semantic Score**: Cross-encoder relevance probability.
    * **Recency Score**: Exponential decay favoring recently updated documentation.
    * **Authority Score**: Verified source boost (e.g., official runbooks outrank Slack chatter).
    """)

with col_rk2:
    st.subheader("Ranking Signal Weighting Matrix")
    sig_matrix = pd.DataFrame([
        {"Signal", "Typical Weight", "Engineering Role"},
        {"Cross-Encoder Semantic Relevance", "0.70", "Primary driver of factual precision"},
        {"Temporal Recency Decay", "0.15", "Prevents legacy documents from overriding modern updates"},
        {"Document Source Authority", "0.15", "Boosts certified corporate knowledge bases over unverified wikis"}
    ])
    st.dataframe(sig_matrix, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: CONTEXT FILTERING
# ==============================================================================

st.header("6. Context Filtering: Security Scoping, Metadata Gating, and Tenant Isolation", divider="rainbow")

st.markdown("""
Context filtering enforces hard operational and regulatory constraints on retrieved chunks.
""")

col_flt1, col_flt2 = st.columns(3)

with col_flt1:
    st.markdown("""
    **1. Security Clearance & RBAC**
    * **Rule**: Filter chunks where `user_clearance >= doc_classification`.
    * **Protection**: Prevents confidential payroll or M&A details from leaking into general employee prompts.
    """)

with col_flt2:
    st.markdown("""
    **2. Multi-Tenant Isolation**
    * **Rule**: Hard filter on `tenant_id == active_tenant`.
    * **Protection**: Guarantees zero cross-customer data leakage in SaaS environments.
    """)

with col_flt1:
    st.markdown("""
    **3. Temporal Lifecycle Filtering**
    * **Rule**: Discard documents where `is_deprecated == True` or `expiration_date < NOW`.
    * **Protection**: Eliminates outdated compliance or configuration guides.
    """)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Production Context Selector & Filter", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python module implementing the complete context selection pipeline: 
metadata gating, relevance score thresholding, near-duplicate Jaccard pruning, and composite ranking.
""")

st.code(r'''
from typing import List, Dict, Any, Optional

class ProductionContextSelector:
    """
    Production-grade context selection engine:
    - Metadata & RBAC security gating
    - Relevance confidence thresholding
    - Jaccard near-duplicate pruning
    - Composite ranking and payload generation
    """
    def __init__(
        self,
        min_relevance_score: float = 0.65,
        near_duplicate_threshold: float = 0.60
    ):
        self.min_relevance_score = min_relevance_score
        self.near_duplicate_threshold = near_duplicate_threshold

    @staticmethod
    def _jaccard(t1: str, t2: str) -> float:
        s1 = set(t1.lower().split())
        s2 = set(t2.lower().split())
        union = len(s1.union(s2))
        return len(s1.intersection(s2)) / union if union > 0 else 0.0

    def select_context(
        self,
        candidate_pool: List[Dict[str, Any]],
        user_clearance_level: str = "Public",
        min_year: int = 2021
    ) -> Dict[str, Any]:
        # 1. Security & Temporal Metadata Filter
        authorized_pool = []
        for doc in candidate_pool:
            if doc.get("clearance") == "Confidential" and user_clearance_level != "Confidential":
                continue
            if doc.get("year", 2024) < min_year:
                continue
            authorized_pool.append(doc)

        # 2. Relevance Score Floor
        relevant_pool = [d for d in authorized_pool if d.get("score", 0.0) >= self.min_relevance_score]

        # 3. Near-Duplicate Pruning
        deduped_pool = []
        for candidate in relevant_pool:
            is_duplicate = False
            for existing in deduped_pool:
                if self._jaccard(candidate["text"], existing["text"]) >= self.near_duplicate_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped_pool.append(candidate)

        # 4. Final Ranking
        deduped_pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return {
            "surviving_chunks": deduped_pool,
            "total_selected": len(deduped_pool),
            "status": "SUCCESS" if deduped_pool else "EMPTY_CONTEXT"
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Context Selection and Pruning Console", divider="rainbow")

st.markdown("""
Adjust the **Relevance Score Cutoff**, **Near-Duplicate Threshold**, and **Security Clearances** 
to observe real-time candidate gating and audit trails.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    cutoff_val = st.slider("Minimum Relevance Cutoff:", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    jaccard_val = st.slider("Near-Duplicate Jaccard Threshold:", min_value=0.3, max_value=0.9, value=0.60, step=0.05)
    min_year_val = st.slider("Minimum Documentation Year:", min_value=2015, max_value=2024, value=2020, step=1)
    allow_confidential_toggle = st.toggle("Grant User 'Confidential' Clearance", value=False)

    selection_res = ContextSelectionEngine.execute_selection(
        min_score=cutoff_val,
        dedup_threshold=jaccard_val,
        allow_confidential=allow_confidential_toggle,
        min_year=min_year_val
    )

with wb_col2:
    st.markdown("### 📊 Selection & Pruning Metrics")
    m_cols = st.columns(3)
    m_cols[0].metric("Initial Pool", selection_res["initial_count"])
    m_cols[1].metric("Surviving Chunks", selection_res["surviving_count"])
    m_cols[2].metric("Pruned Chunks", selection_res["initial_count"] - selection_res["surviving_count"])

    st.markdown("#### 📜 Selection Audit Trail (Decision Log)")
    for entry in selection_res["audit_trail"]:
        st.caption(f"• {entry}")

st.markdown("### 🏆 Surviving High-Confidence Context Chunks")
if selection_res["selected_chunks"]:
    s_df = pd.DataFrame(selection_res["selected_chunks"])
    st.dataframe(s_df[["id", "title", "score", "category", "year", "security_clearance", "text"]], use_container_width=True)
else:
    st.warning("All candidate chunks were pruned by security, recency, or relevance gates!")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("**Q1: Why is near-duplicate pruning necessary if documents already have unique database IDs?**"):
    st.markdown("""
    **Answer**: In enterprise knowledge lakes, the same text frequently exists across multiple files under different unique IDs. 
    For example, a security policy may exist in a Confluence page (`doc_id: 101`), an employee handbook PDF (`doc_id: 204`), 
    and an onboarding email template (`doc_id: 508`). 
    A vector search will retrieve all three chunks because their embeddings are identical. 
    Without content-based near-duplicate pruning (e.g., Jaccard similarity or MinHash), all three identical chunks enter the prompt, 
    consuming 3x the token budget and crowding out complementary facts.
    """)

with st.expander("**Q2: How does context selection enforce data privacy and Role-Based Access Control (RBAC)?**"):
    st.markdown("""
    **Answer**: Vector embeddings do not inherently understand user permissions: an embedding model maps confidential executive salary 
    memos into the same mathematical neighborhood as general HR policy questions. 
    Context selection enforces an explicit metadata security gate: every chunk carries an access-control tag (e.g., `clearance_level: confidential`). 
    The selector cross-references the tag against the authenticated user's session claims, silently dropping unauthorized chunks 
    before they can be assembled into the LLM context prompt.
    """)

with st.expander("**Q3: What is the risk of setting the Jaccard near-duplicate threshold too low (e.g. threshold < 0.40)?**"):
    st.markdown("""
    **Answer**: A low Jaccard threshold (e.g., 0.35) causes **over-pruning (false deduplication)**. 
    Two distinct documents that discuss the same general topic (e.g., *"PostgreSQL Transaction Pooling"* vs. *"PostgreSQL Session Pooling"*) 
    share significant common terminology (`PostgreSQL`, `pooling`, `connections`, `database`). 
    If the deduplication threshold is set too aggressively, the selector will mistakenly view the second document as a duplicate of the first, 
    discarding a vital contrasting fact!
    """)

with st.expander("**Q4: How does Composite Ranking differ from pure similarity sorting in enterprise RAG?**"):
    st.markdown("""
    **Answer**: Pure similarity sorting relies solely on geometric distance, which is blind to temporal validity and document authority. 
    Composite ranking computes a weighted score incorporating:
    1. Semantic relevance (cross-encoder score).
    2. Temporal recency decay (favoring 2024 docs over 2018 docs).
    3. Document authority weights (giving higher weight to verified official runbooks over informal Slack messages).
    This prevents outdated or low-authority documents from outranking certified enterprise truth.
    """)

with st.expander("**Q5: What is the Empty Context Circuit Breaker and when does it trigger?**"):
    st.markdown("""
    **Answer**: The circuit breaker triggers when all candidate chunks fail the minimum relevance score threshold (e.g., all scores < 0.65). 
    Instead of passing low-confidence noise chunks to the generator LLM (which forces hallucinations), 
    the system intercepts the empty set and short-circuits execution, immediately returning an authentic refusal:
    *"I do not have verified knowledge base records to answer this inquiry."*
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Hardcoding Fixed-K Document Counts Regardless of Query Quality**
    * **Symptom**: System injects 4 noise documents for a simple 1-fact question, increasing hallucination rates.
    * **Root Cause**: Hardcoding `top_k = 5` without implementing an absolute relevance threshold filter.
    * **Remedy**: Always filter by score threshold (`score >= 0.65`), allowing the context count to scale dynamically from 1 to 5.
    """)

    st.error("""
    **Trap 2: Relying on Vector DB ID Matching for Deduplication**
    * **Symptom**: Prompt context contains 3 identical paragraphs copied across different corporate files.
    * **Root Cause**: Deduplicating solely on `chunk.id`. Mirrored files have distinct IDs despite identical text.
    * **Remedy**: Implement text-level near-duplicate pruning (Jaccard similarity or MinHash) across candidate texts.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Over-Pruning Contrasting Documents with Overly Low Jaccard Thresholds**
    * **Symptom**: System fails to answer comparative questions because the second comparative document was dropped as a 'duplicate'.
    * **Root Cause**: Setting the near-duplicate threshold too aggressively (e.g. Jaccard < 0.50).
    * **Remedy**: Keep the Jaccard deduplication threshold between 0.65 and 0.80 to preserve subtle conceptual contrasts.
    """)

    st.warning("""
    **Trap 4: Omitting Temporal Recency Filtering on Changing Corporate Policies**
    * **Symptom**: Assistant quotes an outdated 2017 travel reimbursement policy instead of the active 2024 policy.
    * **Root Cause**: 2017 document had slightly higher semantic similarity, outranking the newer policy.
    * **Remedy**: Apply recency score decay or hard metadata filters (`doc_year >= 2023`).
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is the purpose of Context Selection in RAG?")
        st.markdown("**Back**: To filter, deduplicate, and rank raw candidate chunks before prompt assembly, ensuring only authorized, high-confidence, non-redundant evidence reaches the LLM.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: How does Jaccard Near-Duplicate Pruning operate?")
            st.markdown("**Back**: By calculating the word intersection over word union of candidate chunk pairs (`|A ∩ B| / |A ∪ B|`), dropping chunks that exceed redundancy thresholds (e.g., >= 0.60).")

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is the Empty Context Circuit Breaker?")
        st.markdown("**Back**: A safety gate that intercepts cases where all retrieved documents fail minimum relevance thresholds, short-circuiting LLM generation to prevent hallucinations.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: What are the three components of Composite Ranking?")
            st.markdown("**Back**: Semantic relevance score (cross-encoder), temporal recency decay (favoring modern docs), and document authority weighting (certified sources over wikis).")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Context Selection, Deduplication, and Gating:
""")

chk1 = st.checkbox("I understand why fixed-K heuristics fail and how dynamic score thresholding solves them.", value=True)
chk2 = st.checkbox("I can articulate how near-duplicate Jaccard pruning eliminates wasted token budget.", value=True)
chk3 = st.checkbox("I know how the Empty Context Circuit Breaker prevents hallucinations on out-of-domain queries.", value=True)
chk4 = st.checkbox("I understand how Composite Ranking incorporates temporal recency and source authority alongside semantics.", value=True)
chk5 = st.checkbox("I recognize the security role of metadata gating in enforcing Role-Based Access Control (RBAC).", value=True)
chk6 = st.checkbox("I can implement a production context selector that prunes unauthorized and redundant chunks.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Phenomenal! You have mastered Context Selection, Deduplication, and Security Gating.")
