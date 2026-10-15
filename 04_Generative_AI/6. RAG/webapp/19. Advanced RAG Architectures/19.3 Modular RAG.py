import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="19.3 Modular RAG Architecture",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class ModularRAGEngine:
    """Enterprise Modular RAG Engine featuring pluggable micro-components and dynamic routing."""

    def __init__(self):
        # Sharded Domain Corpora
        self.corpora = {
            "legal_compliance": [
                {"id": "leg_1", "title": "GDPR Article 17 (Right to Erasure)", "text": "Data subjects have the right to request erasure of personal data without undue delay where data is no longer necessary for original processing purposes."},
                {"id": "leg_2", "title": "SOC 2 Type II Audit Standards", "text": "Trust Services Criteria require strict change management logging, annual penetration testing, and role-based access control with least privilege."}
            ],
            "engineering_specs": [
                {"id": "eng_1", "title": "Kafka Partition Rebalancing Protocol", "text": "Kafka consumer groups trigger rebalances upon consumer failure. Eager rebalancing revokes all partitions, while Cooperative Sticky rebalancing enables zero-downtime progressive assignment."},
                {"id": "eng_2", "title": "PostgreSQL Connection Pooling Architecture", "text": "PgBouncer operating in transaction pooling mode reduces server process fork overhead, capping active backend connections at 100 while supporting 10,000 clients."}
            ],
            "hr_policies": [
                {"id": "hr_1", "title": "Parental Leave & Wellness Policy", "text": "Eligible employees receive 16 weeks of fully paid parental leave, accessible continuously or in two separate blocks within 12 months of birth or adoption."},
                {"id": "hr_2", "title": "Remote Work Equipment Stipend", "text": "New hires are allocated a one-time $1,000 home office ergonomics stipend and a recurring $75 monthly internet reimbursement."}
            ]
        }

    def route_query_intent(self, query: str) -> Dict[str, Any]:
        """Query Router: Classifies query intent, complexity, and optimal generation tier."""
        q_lower = query.lower()
        
        # Complexity classification
        if any(w in q_lower for w in ["compare", "synthesize", "tradeoff", "architecture", "difference"]):
            complexity = "HIGH_REASONING"
            recommended_model = "Frontier Reasoning LLM (70B+ / Claude 3.5 Sonnet)"
        else:
            complexity = "LOW_LATENCY"
            recommended_model = "Distilled SLM (8B / GPT-4o-mini)"
            
        return {
            "complexity": complexity,
            "recommended_model": recommended_model,
            "estimated_reasoning_steps": 3 if complexity == "HIGH_REASONING" else 1
        }

    def route_retrieval_corpus(self, query: str) -> Dict[str, Any]:
        """Retrieval Router: Shards query to appropriate domain corpus and retriever type."""
        q_lower = query.lower()
        
        if any(w in q_lower for w in ["gdpr", "soc", "erasure", "compliance", "legal", "audit"]):
            selected_domain = "legal_compliance"
            retriever_type = "Hybrid (Dense Vector + Exact Token BM25)"
        elif any(w in q_lower for w in ["kafka", "postgres", "partition", "pgbouncer", "latency", "connection"]):
            selected_domain = "engineering_specs"
            retriever_type = "Dense Vector Search (HNSW Code/Tech Corpus)"
        else:
            selected_domain = "hr_policies"
            retriever_type = "Sparse Lexical BM25 (Policy FAQs)"
            
        return {
            "selected_domain": selected_domain,
            "retriever_type": retriever_type,
            "corpus_doc_count": len(self.corpora[selected_domain])
        }

    def retrieve_documents(self, domain: str, query: str, retriever_type: str) -> List[Dict[str, Any]]:
        """Modular Retriever execution based on router assignment."""
        docs = self.corpora.get(domain, [])
        scored = []
        q_words = set(query.lower().split())
        
        for doc in docs:
            d_words = set((doc["title"] + " " + doc["text"]).lower().split())
            overlap = len(q_words.intersection(d_words))
            score = float(overlap / (len(q_words) + 1e-5)) + 0.50
            scored.append({
                "id": doc["id"],
                "domain": domain,
                "title": doc["title"],
                "text": doc["text"],
                "raw_score": round(score, 3)
            })
            
        scored.sort(key=lambda x: x["raw_score"], reverse=True)
        return scored

    def rerank_documents(self, docs: List[Dict[str, Any]], reranker_choice: str) -> List[Dict[str, Any]]:
        """Modular Reranker execution: Cross-Encoder, Reciprocal Rank Fusion, or Bypass."""
        if reranker_choice == "None (Bypass)":
            for d in docs:
                d["rerank_score"] = d["raw_score"]
            return docs
            
        boost = 0.25 if reranker_choice == "Cross-Encoder (High Precision)" else 0.10
        for d in docs:
            d["rerank_score"] = round(d["raw_score"] + boost, 3)
            
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return docs

    def generate_response(self, query: str, docs: List[Dict[str, Any]], model_choice: str) -> Dict[str, Any]:
        """Modular Generator execution: Fast SLM vs Frontier LLM."""
        if not docs:
            return {"answer": "No relevant documents found across routed domains.", "latency_ms": 45.0, "tokens": 20}
            
        top_doc = docs[0]
        if "Frontier" in model_choice:
            answer = (
                f"[Frontier Synthesis]: Analyzing {top_doc['domain']} governance for query '{query}':\n"
                f"Under {top_doc['title']}, {top_doc['text']}\n"
                f"Engineering Assessment: Architecture ensures regulatory and operational compliance with zero identified regressions."
            )
            latency_ms = 480.0
            tokens = 115
        else:
            answer = (
                f"[SLM Rapid Response]: According to {top_doc['title']}, {top_doc['text'][:120]}..."
            )
            latency_ms = 95.0
            tokens = 45

        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "top_doc": top_doc["title"]
        }


# --- Streamlit Layout ---

st.title("19.3 Modular RAG Architecture")
st.caption("Pluggable Micro-Components: Modular Retrievers, Rerankers, Generators & Dynamic Routing Infrastructure")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise Modular RAG & Pluggable Micro-Component Architecture", divider="rainbow")

st.markdown(
    """
    As generative AI systems scale to enterprise operations, a one-size-fits-all pipeline becomes unmanageable.
    **Modular RAG** departs from monolithic pipelines by decoupling retrieval, reranking, generation, and routing into
    **independent, swappable micro-services**.
    
    Rather than hard-coding a static vector database or a single frontier LLM, a centralized **Routing Orchestration Engine**
    evaluates incoming queries in real time. It triages user requests based on query intent, sharded corpus affinity, latency budgets,
    and computational cost constraints.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Modular_RAG_Architecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_ingress_routing {
            label = "1. DYNAMIC ROUTING & TRIAGE LAYER";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_input [label="User Ingress Query", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            query_router [label="Query Router (Intent & Complexity)\\n- Factoid vs Multi-Hop Synthesis\\n- Cost & Latency Budget Triage", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            retrieval_router [label="Retrieval Router (Corpus Sharding)\\n- Domain Dispatch (Legal / Eng / HR)\\n- Tenant Isolation Gating", fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            user_input -> query_router -> retrieval_router;
        }

        subgraph cluster_modular_retrievers {
            label = "2. MODULAR RETRIEVER POOL (PLUGGABLE CONNECTORS)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            dense_retriever [label="Retriever A: Dense Vector Engine\\n(Milvus / Qdrant HNSW)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            sparse_retriever [label="Retriever B: Sparse BM25 / SPLADE\\n(Elasticsearch / OpenSearch)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            graph_retriever [label="Retriever C: Knowledge Graph Traversal\\n(Neo4j / Cypher Multi-Hop)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            web_retriever [label="Retriever D: Web / External API Tool\\n(Tavily / Perplexity Live Search)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            retrieval_router -> dense_retriever [label="Semantic Match"];
            retrieval_router -> sparse_retriever [label="Keyword Match"];
            retrieval_router -> graph_retriever [label="Relational Match"];
            retrieval_router -> web_retriever [label="Live Data Match"];
        }

        subgraph cluster_modular_rerankers {
            label = "3. MODULAR RERANKER POOL (DYNAMIC PRECISION)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            cross_reranker [label="Reranker 1: Heavy Cross-Encoder\\n(Cohere Rerank v3 / BGE-Large)\\nHigh Precision (180ms)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            fast_rrf [label="Reranker 2: Reciprocal Rank Fusion\\nPure Mathematical Rank Merge\\nZero GPU Overhead (< 2ms)", fillcolor="#0F172A", fontcolor="#10B981"];
            bypass_rerank [label="Reranker 3: Pass-Through Bypass\\nDirect Top-K Slicing\\nUltra Low Latency (0ms)", fillcolor="#78350F", fontcolor="#FDE68A"];
            
            dense_retriever -> cross_reranker;
            sparse_retriever -> fast_rrf;
            graph_retriever -> cross_reranker;
            web_retriever -> bypass_rerank;
        }

        subgraph cluster_modular_generators {
            label = "4. MODULAR GENERATOR POOL (DYNAMIC COMPUTE TIERS)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            slm_model [label="Generator A: Low-Latency SLM\\n(Llama-3.2-3B / Mistral-7B / GPT-4o-mini)\\nCost: $0.0002 | Latency: 80ms", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            frontier_model [label="Generator B: Frontier Reasoning LLM\\n(GPT-4o / Claude 3.5 Sonnet / o1-preview)\\nCost: $0.0150 | Latency: 950ms", fillcolor="#4C1D95", fontcolor="#DDD6FE"];
            
            cross_reranker -> frontier_model [label="Complex Multi-Hop"];
            fast_rrf -> slm_model [label="Standard Factoid"];
            bypass_rerank -> slm_model [label="Quick Lookup"];
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: MODULAR RETRIEVERS ---
st.header("2. Modular Retrievers (Pluggable Dense Vector, BM25 Sparse, Knowledge Graph & Web Search Modules)", divider="rainbow")

st.markdown(
    """
    In a modular architecture, the retrieval step is defined as an abstract interface (`BaseRetriever`).
    Different implementations can be instantiated, swapped, or combined dynamically depending on runtime conditions.
    
    ### Specialized Modular Retriever Profiles
    """
)

col_mr1, col_mr2 = st.columns(2)
with col_mr1:
    with st.container(border=True):
        st.markdown("#### 1. Dense Vector Modules (Semantic)")
        st.markdown(
            """
            - **Engine**: Qdrant, Milvus, Weaviate, Pinecone.
            - **Strength**: High conceptual recall, synonym handling, multilingual queries.
            - **Weakness**: Blind to specific serial numbers, error codes, and strict Boolean constraints.
            """
        )
    with st.container(border=True):
        st.markdown("#### 2. Sparse Lexical Modules (Keyword)")
        st.markdown(
            """
            - **Engine**: BM25, Elasticsearch, SPLADE.
            - **Strength**: Exact keyword matching, compliance article numbers, code identifiers.
            - **Weakness**: Zero understanding of synonyms or cross-lingual semantics.
            """
        )
with col_mr2:
    with st.container(border=True):
        st.markdown("#### 3. Knowledge Graph Modules (Relational)")
        st.markdown(
            """
            - **Engine**: Neo4j, Amazon Neptune, GraphRAG.
            - **Strength**: Explicit entity relationships, multi-hop transitive paths (*'Company A owns Subsidiary B which holds Patent C'*).
            - **Weakness**: High schema construction and graph extraction overhead.
            """
        )
    with st.container(border=True):
        st.markdown("#### 4. Web & External API Modules (Live)")
        st.markdown(
            """
            - **Engine**: Tavily Search, Serper, Enterprise REST APIs.
            - **Strength**: Real-time stock prices, live weather, external vendor documentation.
            - **Weakness**: Unbounded third-party latency and inconsistent response formatting.
            """
        )

# --- SECTION 3: MODULAR RERANKERS ---
st.header("3. Modular Rerankers (Configurable Cross-Encoders, Cohere Rerank API & Local Monolingual Engines)", divider="rainbow")

st.markdown(
    """
    Reranking represents the single highest compute bottleneck in the retrieval tier.
    A modular RAG system allows administrators to dynamically toggle between reranking strategies based on SLA budgets:
    
    - **Tier 1: Heavy Cross-Encoder Transformer** (`Cohere Rerank v3`, `BGE-Reranker-Large`). Injected for mission-critical legal or clinical queries where ranking precision is non-negotiable.
    - **Tier 2: Reciprocal Rank Fusion (RRF)**. A purely mathematical rank-merging algorithm running in $< 2\text{ ms}$ on CPU. Injected for standard business queries.
    - **Tier 3: Pass-Through (Bypass)**. Bypasses the reranking step entirely for ultra-low latency interactive chatbots ($< 50\text{ ms}$ SLA).
    """
)

# --- SECTION 4: MODULAR GENERATORS ---
st.header("4. Modular Generators (Dynamic LLM Switching: Low-Latency SLMs vs Frontier Reasoning Models)", divider="rainbow")

st.markdown(
    """
    Using a 70B+ parameter frontier model for every query is commercially unviable.
    Modular generation dynamically routes prompts across a spectrum of model tiers:
    """
)

gen_table = pd.DataFrame({
    "Generator Tier": ["Small Language Model (SLM)", "Mid-Tier Workhorse Model", "Frontier Reasoning LLM"],
    "Example Models": ["Llama-3.2-3B, Mistral-7B, Phi-3.5", "GPT-4o-mini, Claude 3.5 Haiku", "GPT-4o, Claude 3.5 Sonnet, o1-preview"],
    "Inference Latency": ["40ms - 120ms", "150ms - 400ms", "800ms - 3,500ms"],
    "Cost per 1M Tokens": ["$0.05 - $0.20", "$0.15 - $0.60", "$2.50 - $15.00"],
    "Optimal Query Category": ["Factoid Q&A, Keyword Lookups", "Standard Document Summarization", "Cross-Document Synthesis, Complex Logic"]
})
st.dataframe(gen_table, use_container_width=True)

# --- SECTION 5: QUERY ROUTERS ---
st.header("5. Query Routers (Intent Classification, Query Complexity Scoring & Dynamic Branching Logic)", divider="rainbow")

st.markdown(
    """
    The **Query Router** acts as the front-line dispatcher of the modular architecture.
    Before any downstream retriever or LLM is touched, the router performs semantic intent classification and complexity triage.
    
    ### Query Routing Mechanisms:
    1. **Embedding Classifier / Zero-Shot Router**: Computes cosine similarity between the user query and pre-defined intent centroid embeddings.
    2. **Lightweight LLM Router**: A distilled 3B parameter model evaluates the query against JSON schema routing rules.
    3. **Regex & Rule-Based Fast Path**: Deterministic keyword filters that route queries with zero compute overhead ($< 1\text{ ms}$).
    """
)

# --- SECTION 6: RETRIEVAL ROUTERS ---
st.header("6. Retrieval Routers (Multi-Corpus Sharding, Tenant Isolation & Selective Database Dispatch)", divider="rainbow")

st.markdown(
    """
    In enterprise systems, data is segmented into distinct regulatory, legal, and operational boundaries.
    The **Retrieval Router** prevents cross-corpus contamination by directing the search exclusively to verified knowledge shards:
    
    - **Domain-Specific Sharding**: Routes HR questions strictly to the HR policies index, preventing irrelevant engineering documents from diluting context.
    - **Tenant Isolation & Security Boundaries**: Enforces cryptographic tenant segregation in multi-tenant SaaS environments, ensuring Tenant A never queries Tenant B's vector index.
    - **Temporal Sharding**: Routes questions regarding historical records to cold archive indices while routing current operations to warm, active indices.
    """
)

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Enterprise Modular RAG Routing & Execution Engine (`ModularRAGEngine`)", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `ModularRAGEngine`.
    It implements intent and complexity routing, multi-corpus sharded retrieval, modular reranking toggles,
    and adaptive generator dispatching.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class ModularRAGEngine:
    """Enterprise Modular RAG Engine featuring pluggable micro-components and dynamic routing."""

    def __init__(self):
        self.corpora = {
            "legal_compliance": [
                {"id": "leg_1", "title": "GDPR Article 17 (Right to Erasure)", "text": "Data subjects have the right to request erasure of personal data without undue delay where data is no longer necessary for original processing purposes."},
                {"id": "leg_2", "title": "SOC 2 Type II Audit Standards", "text": "Trust Services Criteria require strict change management logging, annual penetration testing, and role-based access control with least privilege."}
            ],
            "engineering_specs": [
                {"id": "eng_1", "title": "Kafka Partition Rebalancing Protocol", "text": "Kafka consumer groups trigger rebalances upon consumer failure. Eager rebalancing revokes all partitions, while Cooperative Sticky rebalancing enables zero-downtime progressive assignment."},
                {"id": "eng_2", "title": "PostgreSQL Connection Pooling Architecture", "text": "PgBouncer operating in transaction pooling mode reduces server process fork overhead, capping active backend connections at 100 while supporting 10,000 clients."}
            ],
            "hr_policies": [
                {"id": "hr_1", "title": "Parental Leave & Wellness Policy", "text": "Eligible employees receive 16 weeks of fully paid parental leave, accessible continuously or in two separate blocks within 12 months of birth or adoption."},
                {"id": "hr_2", "title": "Remote Work Equipment Stipend", "text": "New hires are allocated a one-time $1,000 home office ergonomics stipend and a recurring $75 monthly internet reimbursement."}
            ]
        }

    def route_query_intent(self, query: str) -> Dict[str, Any]:
        """Query Router: Classifies query intent, complexity, and optimal generation tier."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["compare", "synthesize", "tradeoff", "architecture", "difference"]):
            complexity = "HIGH_REASONING"
            recommended_model = "Frontier Reasoning LLM (70B+ / Claude 3.5 Sonnet)"
        else:
            complexity = "LOW_LATENCY"
            recommended_model = "Distilled SLM (8B / GPT-4o-mini)"
            
        return {
            "complexity": complexity,
            "recommended_model": recommended_model,
            "estimated_reasoning_steps": 3 if complexity == "HIGH_REASONING" else 1
        }

    def route_retrieval_corpus(self, query: str) -> Dict[str, Any]:
        """Retrieval Router: Shards query to appropriate domain corpus and retriever type."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["gdpr", "soc", "erasure", "compliance", "legal", "audit"]):
            selected_domain = "legal_compliance"
            retriever_type = "Hybrid (Dense Vector + Exact Token BM25)"
        elif any(w in q_lower for w in ["kafka", "postgres", "partition", "pgbouncer", "latency", "connection"]):
            selected_domain = "engineering_specs"
            retriever_type = "Dense Vector Search (HNSW Code/Tech Corpus)"
        else:
            selected_domain = "hr_policies"
            retriever_type = "Sparse Lexical BM25 (Policy FAQs)"
            
        return {
            "selected_domain": selected_domain,
            "retriever_type": retriever_type,
            "corpus_doc_count": len(self.corpora[selected_domain])
        }

    def retrieve_documents(self, domain: str, query: str, retriever_type: str) -> List[Dict[str, Any]]:
        """Modular Retriever execution based on router assignment."""
        docs = self.corpora.get(domain, [])
        scored = []
        q_words = set(query.lower().split())
        
        for doc in docs:
            d_words = set((doc["title"] + " " + doc["text"]).lower().split())
            overlap = len(q_words.intersection(d_words))
            score = float(overlap / (len(q_words) + 1e-5)) + 0.50
            scored.append({
                "id": doc["id"],
                "domain": domain,
                "title": doc["title"],
                "text": doc["text"],
                "raw_score": round(score, 3)
            })
            
        scored.sort(key=lambda x: x["raw_score"], reverse=True)
        return scored

    def rerank_documents(self, docs: List[Dict[str, Any]], reranker_choice: str) -> List[Dict[str, Any]]:
        """Modular Reranker execution: Cross-Encoder, Reciprocal Rank Fusion, or Bypass."""
        if reranker_choice == "None (Bypass)":
            for d in docs:
                d["rerank_score"] = d["raw_score"]
            return docs
            
        boost = 0.25 if reranker_choice == "Cross-Encoder (High Precision)" else 0.10
        for d in docs:
            d["rerank_score"] = round(d["raw_score"] + boost, 3)
            
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return docs

    def generate_response(self, query: str, docs: List[Dict[str, Any]], model_choice: str) -> Dict[str, Any]:
        """Modular Generator execution: Fast SLM vs Frontier LLM."""
        if not docs:
            return {"answer": "No relevant documents found across routed domains.", "latency_ms": 45.0, "tokens": 20}
            
        top_doc = docs[0]
        if "Frontier" in model_choice:
            answer = (
                f"[Frontier Synthesis]: Analyzing {top_doc['domain']} governance for query '{query}':\\n"
                f"Under {top_doc['title']}, {top_doc['text']}\\n"
                f"Engineering Assessment: Architecture ensures regulatory and operational compliance with zero identified regressions."
            )
            latency_ms = 480.0
            tokens = 115
        else:
            answer = (
                f"[SLM Rapid Response]: According to {top_doc['title']}, {top_doc['text'][:120]}..."
            )
            latency_ms = 95.0
            tokens = 45

        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "top_doc": top_doc["title"]
        }
''',
    language="python"
)

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Modular Pipeline Switchboard & Dynamic Routing Simulator", divider="rainbow")

st.markdown(
    """
    Test dynamic routing across sharded corpora and configure modular pipeline components.
    Select query presets to observe automated intent triage, or manually override retrievers, rerankers, and generator models.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🎛️ Router Input & Component Overrides")
    
    preset_choice = st.selectbox(
        "Select Query Preset:",
        [
            "Preset 1 (Engineering): Compare Kafka Cooperative Sticky rebalancing with eager rebalancing architecture",
            "Preset 2 (Legal Compliance): What are the GDPR Article 17 requirements for personal data erasure?",
            "Preset 3 (HR Policies): How many weeks of paid parental leave do employees receive?",
            "Preset 4 (Custom Query): Enter your own query below"
        ],
        index=0
    )
    
    if "Custom" in preset_choice:
        active_query = st.text_input("Active Query:", "How does PgBouncer connection pooling improve database performance?")
    else:
        active_query = preset_choice.split(": ", 1)[1]
        st.text_input("Active Query:", active_query, disabled=True)

    engine = ModularRAGEngine()
    auto_intent = engine.route_query_intent(active_query)
    auto_corpus = engine.route_retrieval_corpus(active_query)
    
    st.markdown("#### Dynamic Router Decision (Automated)")
    st.info(
        f"**Target Corpus**: `{auto_corpus['selected_domain']}`  \n"
        f"**Retriever Type**: `{auto_corpus['retriever_type']}`  \n"
        f"**Recommended Model**: `{auto_intent['recommended_model']}`"
    )
    
    st.markdown("#### Manual Component Overrides")
    reranker_choice = st.radio(
        "Modular Reranker Module:",
        ["Cross-Encoder (High Precision)", "Reciprocal Rank Fusion (Fast CPU)", "None (Bypass)"],
        index=0,
        horizontal=True
    )
    
    generator_choice = st.radio(
        "Modular Generator Module:",
        ["Frontier Reasoning LLM (70B+)", "Distilled SLM (8B)"],
        index=0 if auto_intent["complexity"] == "HIGH_REASONING" else 1,
        horizontal=True
    )

with wb_col2:
    st.subheader("⚡ Execution Telemetry & Output")
    
    # Execute modular pipeline
    routed_docs = engine.retrieve_documents(auto_corpus["selected_domain"], active_query, auto_corpus["retriever_type"])
    reranked_docs = engine.rerank_documents(routed_docs, reranker_choice)
    gen_result = engine.generate_response(active_query, reranked_docs, generator_choice)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Routed Corpus Chunks", len(routed_docs))
    m2.metric("Pipeline Latency", f"{gen_result['latency_ms']} ms")
    m3.metric("Output Tokens", gen_result["tokens"])
    
    st.markdown("#### Generated Synthesized Answer")
    st.success(gen_result["answer"])
    
    st.markdown("#### Routed Documents & Rerank Scores")
    doc_display = pd.DataFrame([
        {
            "Doc ID": d["id"],
            "Domain Shard": d["domain"],
            "Title": d["title"],
            "Raw Score": d["raw_score"],
            "Rerank Score": d["rerank_score"]
        }
        for d in reranked_docs
    ])
    st.dataframe(doc_display, use_container_width=True)

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: What are the primary differences between Modular RAG and Advanced RAG?**"):
    st.markdown(
        """
        - **Advanced RAG** introduces a standardized, fixed sequence of enhanced stages (Rewriting -> Hybrid Search -> Reranking -> Compression -> Grounded Generation). Every query traverses the same multi-stage DAG.
        - **Modular RAG** introduces architectural flexibility, component independence, and **dynamic runtime routing**. Queries do not follow a static DAG; rather, an intelligent router determines which specific retrievers (Vector vs Graph vs Web), rerankers, and LLM tiers are activated based on the query's complexity, cost budget, and domain affinity.
        """
    )

with st.expander("**Q2: How do you design an ultra-low-latency Query Router in production?**"):
    st.markdown(
        """
        Invoking a heavy LLM to classify query intent adds 300ms - 800ms to the critical path.
        Production architectures implement a **multi-tiered router hierarchy**:
        1. **Tier 1 (Deterministic Rules & Regex)**: Checks for specific prefixes or keyword patterns (`< 1ms`).
        2. **Tier 2 (Embedding Centroid Classifier)**: Encodes the query using a fast embedding model and computes cosine similarity against pre-clustered domain centroids (`< 15ms`).
        3. **Tier 3 (Distilled SLM Classifier)**: For ambiguous queries, a 1B - 3B quantized parameter model evaluates routing constraints (`< 60ms`).
        """
    )

with st.expander("**Q3: What is Corpus Sharding and how does it prevent cross-domain hallucination?**"):
    st.markdown(
        """
        In monolithic RAG, indexing millions of disparate documents (legal, IT support, code repos, HR policies) into a single vector index creates severe vector crowding:
        an HR query about *"vacation policy"* might pull an engineering doc discussing *"thread idle sleep times"*.
        
        **Corpus Sharding** partitions data into isolated physical or logical indices.
        The Retrieval Router directs the search query strictly to the relevant domain shard, completely eliminating cross-domain vector collisions and drastically improving precision.
        """
    )

with st.expander("**Q4: How does dynamic generator switching optimize the RAG cost-latency Pareto frontier?**"):
    st.markdown(
        """
        80% of enterprise queries are simple factoid lookups (*"What is the deductible?"*, *"Where is the remote policy?"*) that require basic extractive synthesis easily handled by an 8B SLM ($0.15 / 1M tokens, 80ms latency).
        Only 20% of queries require multi-step comparative analysis across multiple documents, requiring a frontier model ($5.00 / 1M tokens, 1,200ms latency).
        
        Dynamic generator routing ensures that expensive frontier models are reserved exclusively for complex queries, reducing aggregate inference costs by over 70% while improving P50 latency.
        """
    )

with st.expander("**Q5: What are the operational challenges of maintaining a modular micro-component architecture?**"):
    st.markdown(
        """
        1. **Schema & Contract Drift**: Independent micro-services evolving different JSON payloads for document chunks and metadata.
        2. **Cascading Failure Modes**: If the Query Router misclassifies an intent, downstream retrievers receive malformed inputs.
        3. **Distributed Observability**: Tracing requests across separate retriever pods, reranker endpoints, and multiple LLM providers requires unified OpenTelemetry span propagation.
        4. **Cold Start Latencies**: Managing auto-scaling across diverse specialized GPU containers (e.g., cross-encoder pods vs embedding pods).
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Using Frontier LLMs as Synchronous Query Routers**  
        **Antipattern**: Prompting GPT-4o to route every incoming user question before initiating retrieval.  
        **Consequence**: Adds 800ms of latency and $0.02 of cost before retrieval even starts, completely destroying the latency budget.  
        **Production Fix**: Employ lightweight embedding classifiers or local SLMs (1B-3B) with latency $< 20\text{ ms}$.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Monolithic Vector Index Collisions**  
        **Antipattern**: Storing engineering source code, HR documents, and customer tickets in a single flat vector index.  
        **Consequence**: Semantic vector crowding causes technical queries to return HR policies, inducing severe context pollution.  
        **Production Fix**: Shard indices by business domain and enforce retrieval routing based on user permissions and query intent.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Unmonitored Fallback Routing Loops**  
        **Antipattern**: Setting a router to continuously retry alternative retrievers in a loop if the first retriever returns zero hits.  
        **Consequence**: An unanswerable query triggers cascading API calls across dense, sparse, graph, and web retrievers, taking 15+ seconds.  
        **Production Fix**: Cap maximum fallback hops to 1 and enforce a strict global request timeout (e.g., 2,500ms).
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Inconsistent Metadata Contracts Across Modular Retrievers**  
        **Antipattern**: Dense retriever returns `doc_text` while the sparse retriever returns `content` and graph returns `node_body`.  
        **Consequence**: Downstream rerankers and compressors fail with key errors or parse empty strings silently.  
        **Production Fix**: Enforce strict Pydantic data schemas (`RetrievedChunk`) across all modular components.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Modular RAG?")
        st.markdown("**Back**: An architecture where retrieval, reranking, generation, and routing exist as decoupled, interchangeable micro-services coordinated by an intelligent routing engine.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is the role of a Query Router?")
        st.markdown("**Back**: Classifies user query intent and complexity to dynamically select the optimal retriever, reranker, and generator tier without human intervention.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is Corpus Sharding?")
        st.markdown("**Back**: Partitioning documents into distinct domain-specific vector indices (e.g., Legal, HR, Engineering) to prevent semantic crowding and cross-domain pollution.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Why use Dynamic Generator Switching?")
        st.markdown("**Back**: Routes simple factoids to fast, cheap SLMs and reserves expensive frontier reasoning models for complex multi-hop queries, optimizing cost and latency.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of the enterprise Modular RAG architecture:")

chk1 = st.checkbox("I understand the architectural differences between linear Basic RAG, multi-stage Advanced RAG, and pluggable Modular RAG.", value=True)
chk2 = st.checkbox("I can design a multi-tier Query Router that classifies intent and complexity without adding multi-second LLM latency.", value=True)
chk3 = st.checkbox("I understand how Corpus Sharding and Retrieval Routers prevent vector collisions and cross-domain context dilution.", value=True)
chk4 = st.checkbox("I know how to dynamically toggle between heavy cross-encoders, fast RRF, and bypass modes based on SLA requirements.", value=True)
chk5 = st.checkbox("I can implement dynamic generator routing to optimize the cost-latency Pareto frontier across SLMs and frontier models.", value=True)
chk6 = st.checkbox("I understand the operational challenges of schema standardization, distributed tracing, and fallback cascading in modular systems.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Excellent! You have mastered the Enterprise Modular RAG Architecture.")
