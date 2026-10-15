import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="20.1 Multi-Document RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class MultiDocumentReconcilerEngine:
    """Enterprise Multi-Document Ingestion, Sharded Retrieval, Conflict Resolution & Deduplication Engine."""

    def __init__(self):
        # Multi-document corpus with intentional temporal & authoritative conflicts
        self.documents = [
            {
                "doc_id": "sec_filing_2024",
                "source": "SEC 10-K Annual Report (FY2024)",
                "authority_score": 0.95,
                "timestamp": "2025-01-15",
                "content": "Enterprise Cloud revenue grew 24% year-over-year reaching $48.2 billion. Operating margin expanded to 31.5% with total enterprise headcount at 124,000."
            },
            {
                "doc_id": "pr_draft_q4",
                "source": "Internal PR Draft Q4 (Preliminary)",
                "authority_score": 0.60,
                "timestamp": "2024-12-28",
                "content": "Preliminary estimates indicate Cloud revenue growth of approximately 25% with operating margin approaching 32%. Total headcount reported as 126,000."
            },
            {
                "doc_id": "analyst_briefing_2025",
                "source": "Goldman Sachs Tech Briefing",
                "authority_score": 0.75,
                "timestamp": "2025-02-10",
                "content": "Confirmed Enterprise Cloud revenue achieved $48.2 billion in FY24. We model 2025 operating margins at 33.0% assuming headcount remains disciplined at 124,000."
            },
            {
                "doc_id": "sec_filing_duplicate",
                "source": "SEC 10-K Syndicate Mirror (Mirror Site)",
                "authority_score": 0.90,
                "timestamp": "2025-01-15",
                "content": "Enterprise Cloud revenue grew 24% year-over-year reaching $48.2 billion. Operating margin expanded to 31.5% with total enterprise headcount at 124,000."
            }
        ]

    def compute_jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """Lexical Jaccard similarity for near-duplicate identification."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return float(intersection / union) if union > 0 else 0.0

    def deduplicate_documents(self, docs: List[Dict[str, Any]], sim_threshold: float = 0.85) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prunes exact and near-duplicate documents across disparate sources."""
        unique_docs = []
        duplicate_docs = []
        
        for doc in docs:
            is_duplicate = False
            for u_doc in unique_docs:
                sim = self.compute_jaccard_similarity(doc["content"], u_doc["content"])
                if sim >= sim_threshold:
                    is_duplicate = True
                    duplicate_docs.append({
                        "duplicate_id": doc["doc_id"],
                        "matched_with": u_doc["doc_id"],
                        "similarity": round(sim, 3)
                    })
                    break
            if not is_duplicate:
                unique_docs.append(doc)
                
        return unique_docs, duplicate_docs

    def retrieve_and_prioritize(
        self,
        query: str,
        recency_weight: float = 0.3,
        authority_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Multi-document retrieval with joint relevance, authoritative hierarchy, and recency scoring."""
        scored = []
        q_tokens = set(query.lower().split())
        
        for doc in self.documents:
            d_tokens = set((doc["source"] + " " + doc["content"]).lower().split())
            rel_score = len(q_tokens.intersection(d_tokens)) / (len(q_tokens) + 1e-5)
            
            # Recency factor
            recency_factor = 1.0 if "2025" in doc["timestamp"] else 0.75
            
            # Composite priority score
            composite_score = (rel_score * 0.4) + (doc["authority_score"] * authority_weight * 0.4) + (recency_factor * recency_weight * 0.2)
            
            scored.append({
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "authority_score": doc["authority_score"],
                "timestamp": doc["timestamp"],
                "content": doc["content"],
                "composite_score": round(composite_score, 4),
                "relevance_raw": round(rel_score, 3)
            })
            
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored

    def reconcile_conflicts(self, prioritized_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detects contradictions across multiple documents and executes authoritative reconciliation."""
        facts = {}
        conflicts = []
        
        for doc in prioritized_docs:
            if "124,000" in doc["content"]:
                facts.setdefault("headcount", []).append(("124,000", doc["source"], doc["authority_score"]))
            if "126,000" in doc["content"]:
                facts.setdefault("headcount", []).append(("126,000", doc["source"], doc["authority_score"]))
                
        if len(facts.get("headcount", [])) > 1:
            candidates = facts["headcount"]
            candidates.sort(key=lambda x: x[2], reverse=True)
            winner = candidates[0]
            conflicts.append({
                "metric": "Total Headcount",
                "claimed_values": [f"{val} ({src})" for val, src, _ in candidates],
                "resolved_value": winner[0],
                "justification": f"Prioritized {winner[1]} due to higher regulatory authority score ({winner[2]})."
            })
            
        synthesis = (
            f"Consensus Financial Synthesis:\n"
            f"- Cloud Revenue: $48.2 Billion (24% YoY growth, verified by SEC 10-K)\n"
            f"- Operating Margin: 31.5% (SEC verified FY24)\n"
            f"- Enterprise Headcount: 124,000 employees [Resolved conflict: Preliminary PR draft claimed 126k, rejected in favor of SEC filing]."
        )
        
        return {
            "conflicts_detected": conflicts,
            "synthesized_response": synthesis
        }


# --- Streamlit Layout ---

st.title("20.1 Multi-Document RAG")
st.caption("Cross-Document Sharding, Multi-Source Synthesis, Authoritative Prioritization & Deduplication Pipelines")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Multi-Document Cross-Referencing & Synthesis Pipeline", divider="rainbow")

st.markdown(
    """
    In enterprise environments, answers rarely exist within a single isolated document.
    A complete answer to a strategic question (*"How did cloud operating margins evolve between the Q3 forecast and the official 10-K filing?"*)
    requires gathering, cross-referencing, prioritizing, and reconciling information distributed across multiple disparate corporate documents.
    
    The diagram below models the complete **Multi-Document RAG Architecture**, detailing multi-corpus ingestion, parallel candidate retrieval,
    content deduplication, authoritative conflict resolution, and cross-document synthesis.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Multi_Document_Architecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_ingress {
            label = "1. MULTI-DOCUMENT INGESTION & SHARDING";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            doc_10k [label="Doc A: Regulatory 10-K Filing\\n(Authority: 0.95, Timestamp: Jan 2025)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            doc_pr [label="Doc B: Internal PR / Drafts\\n(Authority: 0.60, Timestamp: Dec 2024)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            doc_analyst [label="Doc C: Wall Street Equity Report\\n(Authority: 0.75, Timestamp: Feb 2025)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            doc_mirror [label="Doc D: Syndicated Web Mirror\\n(Authority: 0.90, Timestamp: Jan 2025)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        subgraph cluster_retrieval_dedup {
            label = "2. PARALLEL RETRIEVAL & NEAR-DUPLICATE PRUNING";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            federated_retriever [label="Federated Vector Search\\nParallel Query Execution Across Shards", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            dedup_filter [label="Semantic Deduplication Engine\\n- MinHash LSH / Jaccard Overlap (> 0.85)\\n- Exact Document SHA-256 Hashing", fillcolor="#312E81", fontcolor="#A5B4FC"];
            pruned_pool [label="Unique Filtered Candidate Pool\\n(Eliminated Duplicate Mirror Chunks)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            doc_10k -> federated_retriever;
            doc_pr -> federated_retriever;
            doc_analyst -> federated_retriever;
            doc_mirror -> federated_retriever;
            federated_retriever -> dedup_filter -> pruned_pool;
        }

        subgraph cluster_prioritization_reconcile {
            label = "3. HIERARCHICAL PRIORITIZATION & CONFLICT RECONCILER";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            prioritization_tier [label="Authoritative & Temporal Scoring\\nScore = w_rel*Sim + w_auth*Auth + w_rec*Time", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            conflict_engine [label="Cross-Document Conflict Detector\\n- Detects Numerical Discrepancies (124k vs 126k)\\n- Identifies Regulatory Overrides", fillcolor="#78350F", fontcolor="#FDE68A"];
            reconciled_manifest [label="Resolved Evidence Manifest\\n(Winner: Official SEC 10-K Authority)", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            pruned_pool -> prioritization_tier -> conflict_engine -> reconciled_manifest;
        }

        subgraph cluster_synthesis {
            label = "4. CROSS-DOCUMENT GENERATIVE SYNTHESIS";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            llm_synthesizer [label="Frontier LLM Synthesizer\\n(Joint Multi-Document Reasoning & Citations)", fillcolor="#0F172A", fontcolor="#10B981"];
            verified_response [label="Multi-Document Grounded Output\\nExplicitly Discloses Resolved Conflicts", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            reconciled_manifest -> llm_synthesizer -> verified_response;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: SEARCHING MULTIPLE DOCUMENTS ---
st.header("2. Searching Multiple Documents (Multi-Corpus Sharding, Federated Queries & Disparate Formats)", divider="rainbow")

st.markdown(
    """
    In real-world RAG architectures, documents are rarely homogeneous. A query might require correlating:
    - An audited financial 10-K report (structured tables and legal disclaimers).
    - An executive earnings call transcript (conversational dialogue and Q&A).
    - An internal Notion engineering design document (markdown and bulleted specifications).
    
    ### Challenges of Multi-Document Search:
    1. **Format Asymmetry**: Different document structures generate disparate embedding distributions; raw conversational transcripts embed differently from tabular SEC filings.
    2. **Federated Query Scattering**: A single user query must be dispatched concurrently across multiple index shards, followed by centralized candidate gathering.
    3. **Context Fragmentation**: The critical answer requires piecing together Fragment A from Document 1 with Fragment B from Document 3.
    """
)

# --- SECTION 3: COMBINING DOCUMENT SOURCES ---
st.header("3. Combining Document Sources (Cross-Document Coreference, Contradiction Detection & Semantic Merging)", divider="rainbow")

st.markdown(
    """
    When evidence from multiple documents is concatenated together, language models face severe **cross-document reasoning pitfalls**:
    
    - **Cross-Document Coreference Ambiguity**: Document A says *"Project Titan achieved $10M ARR"*, while Document B says *"The automotive initiative was canceled"*. The LLM must determine whether *"Project Titan"* and *"the automotive initiative"* refer to the exact same entity.
    - **Contradiction Detection**: Different documents frequently state conflicting facts due to differing publication dates, revisions, or author bias (e.g., preliminary draft claiming 126k headcount vs final filing confirming 124k).
    - **Semantic Merging**: The synthesizer must weave multiple complementary perspectives into a coherent narrative without repeating boilerplate preamble.
    """
)

# --- SECTION 4: SOURCE PRIORITIZATION ---
st.header("4. Source Prioritization (Authority Weighting, Recency Biasing & Cryptographic Provenance)", divider="rainbow")

st.markdown(
    """
    When documents contradict one another, naive RAG has no principled method to decide which document represents the truth.
    Enterprise multi-document architectures enforce a **Hierarchical Priority Model**:
    """
)

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.subheader("Source Authority Hierarchy")
    auth_df = pd.DataFrame({
        "Tier": ["Tier 1: Regulatory & Legal", "Tier 2: Audited Financials", "Tier 3: Executive Communications", "Tier 4: Internal Drafts / Wikis", "Tier 5: External Web Scrapes"],
        "Authority Score": ["0.95 - 1.00", "0.90 - 0.94", "0.75 - 0.85", "0.50 - 0.70", "0.30 - 0.50"],
        "Resolution Policy": ["Overrides all other sources unconditionally", "Overrides internal drafts & press releases", "Provides context for historical shifts", "Subordinate to official filings", "Used only for missing external facts"]
    })
    st.dataframe(auth_df, use_container_width=True)

with col_p2:
    st.subheader("Composite Priority Formulation")
    st.markdown(
        """
        The overall candidate ranking is governed by a multi-objective scoring formula:
        
        `Composite_Score = (w_rel * Similarity) + (w_auth * Authority) + (w_rec * Recency)`
        
        - `w_rel` (Relevance Weight): Ensures query semantic alignment.
        - `w_auth` (Authority Weight): Elevates audited, cryptographically verified documents.
        - `w_rec` (Recency Weight): Penalizes stale, outdated revisions.
        """
    )

# --- SECTION 5: DUPLICATE REMOVAL ---
st.header("5. Duplicate Removal (MinHash LSH, Exact Content Hashing & Semantic Deduplication)", divider="rainbow")

st.markdown(
    """
    In enterprise corpora, 20% to 40% of all documents are redundant mirrors, email forwards, identical slide decks, or re-syndicated web articles.
    If multiple near-identical chunks are retrieved, they consume precious context window tokens and push complementary documents out of the prompt.
    
    ### Deduplication Tiers in Production:
    1. **Exact Deduplication (Cryptographic Hashing)**: SHA-256 hashes computed over normalized text strips duplicate documents instantly at zero GPU cost.
    2. **Near-Duplicate Detection (MinHash & Locality Sensitive Hashing - LSH)**: Detects documents with identical paragraphs but differing disclaimers or headers.
    3. **Semantic Vector Deduplication**: If cosine similarity between two retrieved candidate chunks exceeds `0.92`, the lower-authority chunk is pruned from the candidate pool.
    """
)

# --- SECTION 6: PRODUCTION IMPLEMENTATION ---
st.header("6. Production Implementation: Enterprise Multi-Document Reconciler Engine", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `MultiDocumentReconcilerEngine`.
    It handles multi-document candidate ingestion, Jaccard near-duplicate filtering, authority-weighted prioritization,
    and automatic conflict resolution between contradictory records.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class MultiDocumentReconcilerEngine:
    """Enterprise Multi-Document Ingestion, Sharded Retrieval, Conflict Resolution & Deduplication Engine."""

    def __init__(self):
        self.documents = [
            {
                "doc_id": "sec_filing_2024",
                "source": "SEC 10-K Annual Report (FY2024)",
                "authority_score": 0.95,
                "timestamp": "2025-01-15",
                "content": "Enterprise Cloud revenue grew 24% year-over-year reaching $48.2 billion. Operating margin expanded to 31.5% with total enterprise headcount at 124,000."
            },
            {
                "doc_id": "pr_draft_q4",
                "source": "Internal PR Draft Q4 (Preliminary)",
                "authority_score": 0.60,
                "timestamp": "2024-12-28",
                "content": "Preliminary estimates indicate Cloud revenue growth of approximately 25% with operating margin approaching 32%. Total headcount reported as 126,000."
            },
            {
                "doc_id": "analyst_briefing_2025",
                "source": "Goldman Sachs Tech Briefing",
                "authority_score": 0.75,
                "timestamp": "2025-02-10",
                "content": "Confirmed Enterprise Cloud revenue achieved $48.2 billion in FY24. We model 2025 operating margins at 33.0% assuming headcount remains disciplined at 124,000."
            },
            {
                "doc_id": "sec_filing_duplicate",
                "source": "SEC 10-K Syndicate Mirror (Mirror Site)",
                "authority_score": 0.90,
                "timestamp": "2025-01-15",
                "content": "Enterprise Cloud revenue grew 24% year-over-year reaching $48.2 billion. Operating margin expanded to 31.5% with total enterprise headcount at 124,000."
            }
        ]

    def compute_jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """Lexical Jaccard similarity for near-duplicate identification."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return float(intersection / union) if union > 0 else 0.0

    def deduplicate_documents(self, docs: List[Dict[str, Any]], sim_threshold: float = 0.85) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prunes exact and near-duplicate documents across disparate sources."""
        unique_docs = []
        duplicate_docs = []
        
        for doc in docs:
            is_duplicate = False
            for u_doc in unique_docs:
                sim = self.compute_jaccard_similarity(doc["content"], u_doc["content"])
                if sim >= sim_threshold:
                    is_duplicate = True
                    duplicate_docs.append({
                        "duplicate_id": doc["doc_id"],
                        "matched_with": u_doc["doc_id"],
                        "similarity": round(sim, 3)
                    })
                    break
            if not is_duplicate:
                unique_docs.append(doc)
                
        return unique_docs, duplicate_docs

    def retrieve_and_prioritize(
        self,
        query: str,
        recency_weight: float = 0.3,
        authority_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Multi-document retrieval with joint relevance, authoritative hierarchy, and recency scoring."""
        scored = []
        q_tokens = set(query.lower().split())
        
        for doc in self.documents:
            d_tokens = set((doc["source"] + " " + doc["content"]).lower().split())
            rel_score = len(q_tokens.intersection(d_tokens)) / (len(q_tokens) + 1e-5)
            recency_factor = 1.0 if "2025" in doc["timestamp"] else 0.75
            composite_score = (rel_score * 0.4) + (doc["authority_score"] * authority_weight * 0.4) + (recency_factor * recency_weight * 0.2)
            
            scored.append({
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "authority_score": doc["authority_score"],
                "timestamp": doc["timestamp"],
                "content": doc["content"],
                "composite_score": round(composite_score, 4),
                "relevance_raw": round(rel_score, 3)
            })
            
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored

    def reconcile_conflicts(self, prioritized_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detects contradictions across multiple documents and executes authoritative reconciliation."""
        facts = {}
        conflicts = []
        
        for doc in prioritized_docs:
            if "124,000" in doc["content"]:
                facts.setdefault("headcount", []).append(("124,000", doc["source"], doc["authority_score"]))
            if "126,000" in doc["content"]:
                facts.setdefault("headcount", []).append(("126,000", doc["source"], doc["authority_score"]))
                
        if len(facts.get("headcount", [])) > 1:
            candidates = facts["headcount"]
            candidates.sort(key=lambda x: x[2], reverse=True)
            winner = candidates[0]
            conflicts.append({
                "metric": "Total Headcount",
                "claimed_values": [f"{val} ({src})" for val, src, _ in candidates],
                "resolved_value": winner[0],
                "justification": f"Prioritized {winner[1]} due to higher regulatory authority score ({winner[2]})."
            })
            
        synthesis = (
            f"Consensus Financial Synthesis:\\n"
            f"- Cloud Revenue: $48.2 Billion (24% YoY growth, verified by SEC 10-K)\\n"
            f"- Operating Margin: 31.5% (SEC verified FY24)\\n"
            f"- Enterprise Headcount: 124,000 employees [Resolved conflict: Preliminary PR draft claimed 126k, rejected in favor of SEC filing]."
        )
        
        return {
            "conflicts_detected": conflicts,
            "synthesized_response": synthesis
        }
''',
    language="python"
)

# --- SECTION 7: INTERACTIVE WORKBENCH ---
st.header("7. Interactive Workbench: Cross-Document Conflict Resolution & Deduplication Simulator", divider="rainbow")

st.markdown(
    """
    Test how the multi-document reconciler identifies duplicate mirror articles, balances authoritative weighting vs recency,
    and programmatically resolves numerical contradictions across conflicting sources.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Ingestion & Prioritization Controls")
    
    user_query = st.text_input(
        "User Query:",
        "What was the total enterprise cloud revenue, operating margin, and headcount?"
    )
    
    dedup_threshold = st.slider(
        "Near-Duplicate Jaccard Threshold:",
        min_value=0.50,
        max_value=0.98,
        value=0.85,
        step=0.05
    )
    
    auth_weight = st.slider("Authority Weight (w_auth):", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    recency_weight = st.slider("Recency Weight (w_rec):", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

with wb_col2:
    st.subheader("🔍 Prioritization & Reconciliation Telemetry")
    
    engine = MultiDocumentReconcilerEngine()
    
    # 1. Deduplicate
    unique_docs, duplicates = engine.deduplicate_documents(engine.documents, sim_threshold=dedup_threshold)
    
    # 2. Prioritize
    prioritized_docs = engine.retrieve_and_prioritize(user_query, recency_weight=recency_weight, authority_weight=auth_weight)
    
    # 3. Reconcile
    reconciliation = engine.reconcile_conflicts(prioritized_docs)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Ingested Documents", len(engine.documents))
    m2.metric("Duplicates Pruned", len(duplicates))
    m3.metric("Conflicts Resolved", len(reconciliation["conflicts_detected"]))
    
    st.markdown("#### Synthesized Consensus Answer")
    st.success(reconciliation["synthesized_response"])
    
    if reconciliation["conflicts_detected"]:
        st.markdown("#### Detected Cross-Document Conflicts")
        for c in reconciliation["conflicts_detected"]:
            st.warning(
                f"**Contradiction in {c['metric']}**:  \n"
                f"- Claims: {', '.join(c['claimed_values'])}  \n"
                f"- **Resolved Winner**: `{c['resolved_value']}`  \n"
                f"- *Rationale*: {c['justification']}"
            )

st.markdown("### Ranked Document Manifest")
manifest_df = pd.DataFrame([
    {
        "Rank": i + 1,
        "Source Name": d["source"],
        "Authority": d["authority_score"],
        "Timestamp": d["timestamp"],
        "Composite Priority Score": d["composite_score"],
        "Content Excerpt": d["content"][:90] + "..."
    }
    for i, d in enumerate(prioritized_docs)
])
st.dataframe(manifest_df, use_container_width=True)

# --- SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: How do you resolve factual contradictions when retrieving context across multiple documents?**"):
    st.markdown(
        """
        Contradictions across multi-document corpora cannot be resolved through vector similarity alone. Production systems employ:
        1. **Authoritative Hierarchy Weighting**: Hard-coding verified domain trust levels (e.g., Regulatory 10-K > Internal Press Release > Blog).
        2. **Temporal Decay Biasing**: For dynamic facts (such as product pricing or policies), timestamp decay formulas penalize older document chunks.
        3. **Explicit Conflict Extraction**: Prompting the generator to detect discrepancies and explicitly state the discrepancy rather than silently picking one.
        """
    )

with st.expander("**Q2: Why is MinHash LSH preferred over pairwise embedding cosine similarity for corpus deduplication?**"):
    st.markdown(
        """
        Pairwise vector comparison over `N` chunks requires `O(N^2)` distance calculations, which becomes computationally impossible for millions of documents.
        
        **MinHash Locality Sensitive Hashing (LSH)** converts documents into sets of character n-gram shingles, hashes them with `K` independent hash functions,
        and groups them into hash buckets. Candidate duplicates are discovered in `O(N)` expected time without calculating pairwise distances across the entire corpus.
        """
    )

with st.expander("**Q3: What is the 'Syndication Problem' in enterprise web RAG?**"):
    st.markdown(
        """
        When external press releases, technical blogs, or news stories are syndicated across 50 different news aggregators,
        a standard vector retriever retrieves 10 copies of the exact same article under slightly different URLs.
        This completely fills the LLM's context window with redundant tokens, pushing complementary documents out of the prompt.
        
        Deduplication at the ingestion and pre-ranking stages is required to collapse syndicated articles into a single canonical source.
        """
    )

with st.expander("**Q4: How do you handle cross-document coreference resolution in multi-hop RAG?**"):
    st.markdown(
        """
        Cross-document coreference resolution links mentions of entities across different texts (e.g., *"Titan"* in Document 1 and *"the autonomous vehicle unit"* in Document 2).
        
        Production pipelines use:
        1. **Named Entity Resolution (NER) & Knowledge Graphs**: Linking textual mentions to a canonical Knowledge Graph Entity ID before chunking.
        2. **Document Context Prepending**: Prepending metadata headers (`[Entity: Project Titan | Subsidiary: AutoCorp]`) to every chunk so dense vectors capture the entity relationship.
        """
    )

with st.expander("**Q5: How does multi-document synthesis affect LLM context window attention distributions?**"):
    st.markdown(
        """
        Concatenating 5 to 10 distinct documents into a single prompt exacerbates **context dilution** and the **lost-in-the-middle** effect.
        The LLM's self-attention mechanism must attend across document boundaries.
        If supporting evidence is split across Document 2 and Document 7, attention heads frequently fail to correlate the facts.
        
        Mitigation requires **Extractive Cross-Document Compression** to prune non-essential paragraphs before context assembly.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Concatenating Redundant Mirror Chunks**  
        **Antipattern**: Retrieving 5 duplicate chunks from mirror sites because they all share high cosine similarity.  
        **Consequence**: Wastes 80% of prompt token budget on duplicate text and prevents alternative evidence from reaching the LLM.  
        **Production Fix**: Implement near-duplicate clustering (Jaccard / MinHash > 0.85) to retain only the single highest-authority chunk.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Ignoring Document Publication Timestamps**  
        **Antipattern**: Treating a 2021 policy document as having equal authority to a revised 2025 policy document.  
        **Consequence**: The LLM synthesizes outdated regulations, generating factually superseded guidance.  
        **Production Fix**: Incorporate temporal decay functions into the composite document ranking score.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Silent Conflict Resolution**  
        **Antipattern**: Allowing the LLM to pick one conflicting number at random without alerting the user that contradictory documents exist.  
        **Consequence**: Users receive plausible answers with zero visibility into enterprise data inconsistencies.  
        **Production Fix**: Prompt the LLM to explicitly highlight discovered conflicts and cite both sources.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Missing Source Provenance Headers**  
        **Antipattern**: Stuffing raw document chunks into prompts without source IDs, timestamps, or authority metadata.  
        **Consequence**: The LLM cannot perform authoritative reasoning or provide granular citations.  
        **Production Fix**: Prepend standardized structured headers to every chunk: `[Source: Doc_A | Authority: 0.95 | Date: 2025-01-15]`.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Cross-Document Coreference?")
        st.markdown("**Back**: The NLP challenge of determining when different phrases across separate documents (e.g., 'Project Titan' and 'the car initiative') refer to the exact same real-world entity.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Why is Authority Weighting critical?")
        st.markdown("**Back**: When documents state contradictory facts, authority weighting programmatically prioritizes verified sources (e.g., SEC 10-K filings) over internal drafts or web scrapes.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is MinHash LSH in RAG?")
        st.markdown("**Back**: An efficient probabilistic hashing algorithm that identifies near-duplicate document chunks in linear time without expensive pairwise vector comparisons.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is the Syndication Problem?")
        st.markdown("**Back**: The phenomenon where identical articles syndicated across multiple web sources flood the vector retriever with duplicate chunks, crowding out unique evidence.")

# --- SECTION 9: SELF-ASSESSMENT CHECKLIST ---
st.header("9. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of multi-document RAG architectures, conflict resolution, and deduplication:")

chk1 = st.checkbox("I understand the end-to-end pipeline for federated multi-document search and cross-document reasoning.", value=True)
chk2 = st.checkbox("I can implement Jaccard and MinHash near-duplicate filtering to prune redundant mirror chunks.", value=True)
chk3 = st.checkbox("I know how to design composite priority scoring balancing semantic relevance, authority tiers, and recency.", value=True)
chk4 = st.checkbox("I understand how to detect and reconcile factual contradictions across disparate document sources.", value=True)
chk5 = st.checkbox("I can structure prompt manifests with explicit document provenance and authority metadata.", value=True)
chk6 = st.checkbox("I know how to configure LLM prompts to explicitly disclose resolved data conflicts to the end user.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Multi-Document RAG Architecture and Reconciliation.")
