import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class ProductionIndexRecord:
    """Represents a document tracked by an incremental RecordManager index."""
    def __init__(self, key: str, doc_hash: str, text: str, tenant_id: str, acl_roles: List[str]):
        self.key = key
        self.doc_hash = doc_hash
        self.text = text
        self.tenant_id = tenant_id
        self.acl_roles = acl_roles

class MockProductionRAGStudio:
    """
    Simulates production RAG operations:
    Incremental Indexing (RecordManager), Semantic Cache (Redis),
    RBAC Access Control Filtering, and Observability Telemetry.
    """
    def __init__(self):
        self.semantic_cache = {
            "what is our uptime sla?": {
                "response": "Enterprise cloud infrastructure guarantees 99.995% uptime with 12s automated failover.",
                "cached": True,
                "latency_ms": 4.2
            }
        }
        self.indexed_records: List[ProductionIndexRecord] = [
            ProductionIndexRecord(
                "rec-001", "hash_a1b2",
                "Executive Compensation: C-suite annual retention bonuses are governed by Board approval.",
                "tenant-alpha", ["executive", "hr_admin"]
            ),
            ProductionIndexRecord(
                "rec-002", "hash_c3d4",
                "Standard Leave Policy: Full-time employees receive 25 days paid time off annually.",
                "tenant-alpha", ["employee", "manager", "executive"]
            ),
            ProductionIndexRecord(
                "rec-003", "hash_e5f6",
                "Production Kubernetes Ingress SLA guarantees 99.99% availability with automated node recovery.",
                "tenant-beta", ["devops", "cloud_admin"]
            )
        ]

    def query_pipeline(
        self,
        query: str,
        user_tenant: str,
        user_roles: List[str],
        enable_cache: bool
    ) -> Dict[str, Any]:
        q_norm = query.strip().lower()

        # 1. Semantic Cache Check
        if enable_cache and q_norm in self.semantic_cache:
            cache_hit = self.semantic_cache[q_norm]
            return {
                "source": "Semantic Cache (Redis Hit)",
                "response": cache_hit["response"],
                "latency_ms": cache_hit["latency_ms"],
                "tokens_saved": 450,
                "cost_usd": 0.000,
                "authorized_docs": []
            }

        # 2. RBAC & Tenant Access Control Gating
        authorized_docs = []
        for rec in self.indexed_records:
            if rec.tenant_id != user_tenant:
                continue
            if any(role in rec.acl_roles for role in user_roles):
                authorized_docs.append(rec)

        if not authorized_docs:
            return {
                "source": "LLM Synthesis (Vector DB)",
                "response": "Access Denied / No Documents Found: User credentials do not satisfy the required security ACL roles or tenant boundary constraints.",
                "latency_ms": 32.5,
                "tokens_saved": 0,
                "cost_usd": 0.001,
                "authorized_docs": []
            }

        # 3. Simulated Generation from authorized context
        top_rec = authorized_docs[0]
        sim_response = f"Authorized Context ({top_rec.key}): {top_rec.text}"
        return {
            "source": "LLM Synthesis (Vector DB)",
            "response": sim_response,
            "latency_ms": 485.2,
            "tokens_saved": 0,
            "cost_usd": 0.0042,
            "authorized_docs": [r.key for r in authorized_docs]
        }

    def simulate_incremental_indexing(self, new_docs: List[Dict[str, str]]) -> Dict[str, int]:
        num_added = 0
        num_skipped = 0
        existing_hashes = {r.doc_hash for r in self.indexed_records}

        for d in new_docs:
            if d["hash"] in existing_hashes:
                num_skipped += 1
            else:
                self.indexed_records.append(
                    ProductionIndexRecord(d["key"], d["hash"], d["text"], d["tenant"], d["roles"])
                )
                existing_hashes.add(d["hash"])
                num_added += 1

        return {"added": num_added, "skipped": num_skipped}


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="16.4 Production RAG — LangChain Course",
    page_icon="🛡️",
    layout="wide"
)

st.title("16.4 Production RAG")
st.caption("Enterprise Infrastructure: Incremental Indexing, Semantic Caching, RBAC Access Control, Multi-Tenancy, and Observability")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: The Enterprise Production RAG Infrastructure", divider="rainbow")
st.markdown("""
Moving a RAG application from a developer laptop to production requires hardened infrastructure. 
A production RAG architecture must manage **dynamic document updates without downtime**, enforce **strict Role-Based 
Access Control (RBAC)** across multi-tenant boundaries, slash latency using **semantic caching**, and emit **real-time observability metrics**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ProductionRAGArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_ingress {
            label = "Step 1: Ingress Gateway & Security Authentication";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";
            user_req [label="Client HTTP Request\\n(JWT Token: tenant_id, user_id, roles)", fillcolor="#1E293B", color="#0284C7"];
            sem_cache [label="Semantic Cache (Redis / GPTCache)\\nChecks query similarity threshold (sim > 0.96)", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_indexing {
            label = "Continuous Ingestion: LangChain Indexing API";
            style = "rounded,dashed";
            color = "#475569";
            fontcolor = "#94A3B8";
            doc_update [label="Document Updates / CDC\\n(Google Drive, Confluence, S3)", fillcolor="#1E293B", color="#64748B"];
            rec_mgr [label="SQL RecordManager\\nTracks content hashes: adds new, updates changed, deletes orphaned", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_retrieval {
            label = "Step 2: RBAC Gated Vector Retrieval";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";
            rbac_filter [label="Enforce Hard Security Metadata Filter\\nWHERE tenant_id = current_tenant AND role IN user_roles", fillcolor="#1E293B", color="#EF4444"];
            v_index [label="Vector Store (Pinecone / Qdrant / pgvector)\\nMulti-tenant namespace isolation", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_llm_ops {
            label = "Step 3: Monitored Generation & Telemetry";
            style = "rounded";
            color = "#10B981";
            fontcolor = "#34D399";
            llm_call [label="LLM Generation (Streaming)\\nCost-controlled temperature=0.0", fillcolor="#064E3B", color="#10B981"];
            langsmith_obs [label="Observability & Tracing (LangSmith / OpenTelemetry)\\nLatency, token burn, evaluation feedback", fillcolor="#022C22", color="#059669"];
        }

        user_req -> sem_cache [label="Incoming Query"];
        sem_cache -> user_req [label="Cache Hit (< 10ms)", color="#10B981"];
        sem_cache -> rbac_filter [label="Cache Miss"];
        
        doc_update -> rec_mgr;
        rec_mgr -> v_index [label="Incremental Index Sync"];

        rbac_filter -> v_index [label="Restricted HNSW Search"];
        v_index -> llm_call [label="Authorized Chunks"];
        llm_call -> langsmith_obs [label="Trace Spans"];
        llm_call -> user_req [label="Streamed Answer"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: DOCUMENT UPDATES ---
st.header("2. Document Updates & Lifecycle Synchronization", divider="rainbow")
st.markdown("""
In enterprise settings, source documents are continually mutated, amended, and deleted. 
Naively re-indexing the entire corpus every night is financially prohibitive and causes downtime.

### Production Update Mechanisms:
1. **Change Data Capture (CDC)**: Listening to webhooks from Google Drive, Notion, or SharePoint to trigger targeted chunk ingestion.
2. **Document Tombstoning**: Marking deprecated chunks with `{"is_deleted": True}` in vector metadata so they are instantly excluded 
   from active retrieval before physical garbage collection occurs.
3. **Version Stamping**: Appending sequential version IDs (`version=3`) to document metadata to enforce temporal validity.
""")

st.divider()

# --- SECTION 3: INCREMENTAL INDEXING ---
st.header("3. Incremental Indexing with RecordManager", divider="rainbow")
st.markdown("""
The **LangChain Indexing API** (`index()`) synchronizes documents from any loader into a vector store while avoiding 
redundant embedding computation and automatically purging deleted documents.

It utilizes a centralized **`RecordManager`** (backed by SQLite or PostgreSQL) to track document hashes, write times, 
and vector store source IDs.
""")

st.code("""
from langchain.indexes import index, SQLRecordManager
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Initialize RecordManager
record_manager = SQLRecordManager(
    namespace="chroma/enterprise_docs",
    db_url="sqlite:///record_manager.sql"
)
record_manager.create_schema()

# 2. Run Incremental Sync (cleanup="incremental")
# - Existing unchanged docs: Skipped (0 embedding API cost)
# - Mutated docs: Re-embedded and updated
# - Deleted docs: Purged from vector store
index_result = index(
    docs_source=latest_documents,
    record_manager=record_manager,
    vector_store=vectorstore,
    cleanup="incremental",
    source_id_key="source"
)

print(index_result)
# Output: {'num_added': 12, 'num_updated': 3, 'num_skipped': 450, 'num_deleted': 2}
""", language="python")

st.divider()

# --- SECTION 4: CACHING ---
st.header("4. Exact & Semantic Caching", divider="rainbow")
st.markdown("""
Calling LLM APIs for repeated queries is slow and expensive. Caching slashes latency from 2,000ms down to sub-10ms.
""")

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.subheader("Exact Prompt Cache")
    st.markdown("""
    Hashes the normalized prompt string (SHA-256). If an identical query was submitted previously, the cached answer 
    is returned instantly with zero LLM API costs:
    ```python
    from langchain_community.cache import RedisCache
    from langchain.globals import set_llm_cache
    import redis

    redis_client = redis.Redis.from_url("redis://localhost:6379")
    set_llm_cache(RedisCache(redis_client))
    ```
    """)
with col_c2:
    st.subheader("Semantic Vector Cache")
    st.markdown("""
    Embeds the user's query and searches a cache vector index. If a previous query shares cosine similarity > 0.96 
    (e.g. *"What is our cloud SLA?"* vs *"Tell me our cloud uptime SLA"*), the stored response is returned:
    ```python
    from langchain_community.cache import RedisSemanticCache
    
    set_llm_cache(RedisSemanticCache(
        redis_url="redis://localhost:6379",
        embedding=embeddings,
        score_threshold=0.96
    ))
    ```
    """)

st.divider()

# --- SECTION 5: ACCESS CONTROL ---
st.header("5. Access Control (RBAC & ABAC)", divider="rainbow")
st.markdown("""
Enterprise knowledge bases contain sensitive data (HR policies, executive compensation, confidential roadmaps) 
that must never be exposed to unauthorized personnel.

**Hard Pre-Filtering** injects the authenticated user's Role-Based Access Control (RBAC) permissions directly 
into the vector search query:
""")

st.code("""
# Extracted securely from verified JWT claims
current_user_tenant = request.auth.tenant_id
current_user_roles = request.auth.roles  # e.g. ["engineering", "team_lead"]

# Vector store query with mandatory security pre-filter
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {
            "tenant_id": current_user_tenant,
            "acl_roles": {"in": current_user_roles}
        }
    }
)
""", language="python")

st.divider()

# --- SECTION 6: MULTI-USER RAG ---
st.header("6. Multi-User RAG & Tenant Governance", divider="rainbow")
st.markdown("""
Operating a multi-tenant RAG service requires architectural isolation across three tiers:
""")

ten_df = pd.DataFrame({
    "Isolation Tier": ["Collection / Namespace Isolation", "Metadata Tenant Gating", "Physical Dedicated Index"],
    "Implementation Strategy": ["Separate Pinecone namespace or Chroma collection per organization", "Shared collection with mandatory WHERE tenant_id = 'org_99'", "Separate dedicated vector DB cluster per enterprise customer"],
    "Security Level": ["High: Logical collection separation", "Moderate: Reliant on strict query filter enforcement", "Maximum: Complete hardware and network isolation"],
    "Infrastructure Cost": ["Low to Moderate", "Lowest: Maximum resource sharing", "Highest: Dedicated cloud infrastructure"]
})
st.dataframe(ten_df, use_container_width=True)

st.divider()

# --- SECTION 7: MONITORING ---
st.header("7. Monitoring, Tracing & Cost Telemetry", divider="rainbow")
st.markdown("""
Production RAG systems require continuous telemetry across four operational dimensions:
1. **Latency Breakdown**: Time spent in Query Embedding vs ANN Vector Search vs Reranking vs LLM First-Token (TTFT).
2. **Token Burn Rate**: Tracking prompt context tokens vs generation tokens to control monthly OpenAI/Anthropic bills.
3. **Retrieval Degradation**: Alerting when the mean top-1 similarity score drops below 0.60 across a 1-hour window.
4. **LangSmith Integration**: Full distributed trace visualization capturing every input, intermediate prompt, and tool call.
""")

st.code("""
# Configure LangSmith telemetry via environment configuration (.env)
# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_API_KEY="lsv2_pt_..."
# LANGCHAIN_PROJECT="production-enterprise-rag"
""", language="python")

st.divider()

# --- SECTION 8: EVALUATION ---
st.header("8. Continuous Online Production Evaluation", divider="rainbow")
st.markdown("""
Offline golden test sets are necessary but insufficient; production systems must capture **online user feedback** 
and run automated shadow evaluation:
- **Explicit Feedback**: Capturing user thumbs-up / thumbs-down ratings linked to specific LangSmith trace IDs.
- **Implicit Feedback**: Tracking copy-to-clipboard actions, follow-up clarification queries, and session abandonment rates.
- **Shadow Testing**: Routing 5% of production queries to a candidate retriever pipeline to compare Context Relevance in parallel.
""")

st.divider()

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Enterprise RAG Engine with Caching & RBAC", divider="rainbow")
st.markdown("""
The production utility below demonstrates how to construct an enterprise RAG service that integrates 
JWT-based RBAC filtering and semantic caching.
""")

st.code("""
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class EnterpriseProductionRAGService:
    \"\"\"
    Production-grade RAG service enforcing multi-tenant isolation, RBAC gating,
    and structured citation generation.
    \"\"\"
    def __init__(self, vectorstore: Chroma, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.llm = llm

    def query(
        self,
        question: str,
        tenant_id: str,
        user_roles: List[str],
        top_k: int = 4
    ) -> str:
        # Enforce security pre-filter
        security_filter = {
            "and": [
                {"tenant_id": {"eq": tenant_id}},
                {"acl_role": {"in": user_roles}}
            ]
        }

        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": security_filter
            }
        )

        def format_docs(docs):
            if not docs:
                return "NO_AUTHORIZED_CONTEXT_AVAILABLE"
            return "\\n\\n".join([f"[{d.metadata.get('source')}]: {d.page_content}" for d in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based strictly on context. If context says NO_AUTHORIZED_CONTEXT_AVAILABLE, refuse to answer:\\n\\n{context}"),
            ("human", "{question}")
        ])

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)
""", language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Production Operations Simulator", divider="rainbow")
st.markdown("""
Simulate enterprise production RAG operations: test **Semantic Caching** to observe sub-10ms responses, 
and toggle **RBAC Roles** to observe how access control boundaries dynamically permit or deny context retrieval.
""")

# Instantiate module-scope simulator
prod_studio = MockProductionRAGStudio()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    test_q_prod = st.selectbox(
        "Select User Query:",
        [
            "What is our uptime SLA?",
            "What is the executive bonus and compensation policy?",
            "What is our standard employee paid time off leave policy?",
            "What is our production Kubernetes ingress SLA?"
        ]
    )

    tenant_select = st.selectbox("Current Request Tenant ID:", ["tenant-alpha", "tenant-beta"])
    
    role_options = ["employee", "manager", "executive", "hr_admin", "devops", "cloud_admin"]
    user_roles_selected = st.multiselect(
        "Current User RBAC Roles (JWT Claims):",
        options=role_options,
        default=["employee"]
    )

    enable_cache_toggle = st.toggle("Enable Redis Semantic Cache", value=True)
    run_prod_btn = st.button("🛡️ Execute Production Query", type="primary", use_container_width=True)

with wb_col2:
    st.subheader("Operational Execution Telemetry")

    if run_prod_btn or test_q_prod:
        telemetry = prod_studio.query_pipeline(
            query=test_q_prod,
            user_tenant=tenant_select,
            user_roles=user_roles_selected,
            enable_cache=enable_cache_toggle
        )

        t_c1, t_c2, t_c3 = st.columns(3)
        t_c1.metric("Execution Latency", f"{telemetry['latency_ms']} ms")
        t_c2.metric("Tokens Saved", telemetry["tokens_saved"])
        t_c3.metric("Estimated Cost", f"${telemetry['cost_usd']:.4f}")

        st.info(f"**Execution Route**: `{telemetry['source']}`")
        st.markdown("### 🤖 Response Payload")
        if "Access Denied" in telemetry["response"]:
            st.error(telemetry["response"])
        else:
            st.success(telemetry["response"])

        if telemetry["authorized_docs"]:
            st.caption(f"**Authorized Records Accessed**: `{telemetry['authorized_docs']}`")

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does LangChain's RecordManager prevent unnecessary embedding API expenditures during re-indexing?",
     "The RecordManager maintains a persistent key-value store (SQLite/PostgreSQL) tracking content hashes of ingested documents. When the indexing API runs with cleanup='incremental', it hashes incoming documents. If a document's hash matches an existing record in the database, it is skipped entirely, incurring 0 embedding API calls. Only new or mutated documents are embedded, and deleted source files are automatically purged from the vector index."),
    ("What is the difference between Exact Caching and Semantic Caching in production RAG?",
     "Exact caching uses a cryptographic hash (e.g. SHA-256) of the raw query string; it only hits if the query is a 100% character-for-character match. Semantic caching embeds the query and searches a vector cache; if the cosine similarity between the new query and a cached query exceeds a threshold (e.g. 0.96), the cached answer is returned, handling paraphrased queries with sub-10ms response times."),
    ("Why is pre-filtering strictly mandatory for Role-Based Access Control (RBAC) in vector databases?",
     "If post-filtering is used, the vector database executes an approximate nearest neighbor search over the entire global corpus first, returning the top-k vectors. If the top-k vectors belong to a different tenant or higher classification tier, post-filtering will discard them, leaving the user with an empty result or exposing sensitive metadata. Pre-filtering applies boolean access constraints during graph traversal, guaranteeing that non-authorized documents are never evaluated."),
    ("How do you handle real-time document deletions in an HNSW-indexed vector database?",
     "HNSW graphs cannot easily delete nodes without risking graph fragmentation. In production, deletions are handled via a two-stage strategy: (1) Instant soft deletion / tombstoning: setting is_deleted=True in metadata so the pre-filter excludes the vector instantly, and (2) Periodic compaction / rebuild: batch rebuilding the HNSW index during off-peak hours to reclaim memory."),
    ("What operational telemetry metrics are most critical to monitor in production RAG?",
     "The most critical metrics are: (1) Time-to-First-Token (TTFT) and End-to-End Latency broken down by retrieval vs generation, (2) Token Burn Rate (prompt context vs completion tokens) to forecast API costs, (3) Mean Top-1 Retrieval Score to identify knowledge base coverage drift, and (4) User Feedback Ratios (thumbs-up/thumbs-down) tagged with trace IDs for debugging.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using In-Memory RecordManagers in Distributed Cloud Deployments. Storing the RecordManager in local SQLite within an auto-scaling Kubernetes pod means pods do not share state. Pod restarts will trigger full, duplicate re-indexing runs. Always back the RecordManager with persistent PostgreSQL.")
    st.error("**Trap 2 (Breaking Error)**: Semantic Cache Pollution from Hallucinated Responses. If a bad or hallucinated response is written into the semantic cache, every subsequent user who asks a similar question will receive the hallucinated response. Only cache responses that pass automated faithfulness evaluation.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Running Unbounded Semantic Cache Without TTL. Without a Time-To-Live (TTL) expiration policy, the semantic cache will serve stale answers long after authoritative underlying policy documents have been updated.")
    st.warning("**Trap 4 (Quality Warning)**: Setting Semantic Cache Threshold Too Low (e.g. sim < 0.90). A low similarity cutoff causes subtly different questions (e.g. 'How to create user' vs 'How to delete user') to hit the same cached response, causing incorrect answers.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: RecordManager"):
        st.markdown("**Tracks document hashes to enable zero-duplicate incremental indexing and automated garbage collection.**")
    with st.expander("🎴 Flashcard 2: Semantic Cache"):
        st.markdown("**Returns cached answers for semantically equivalent queries (similarity > 0.96) in sub-10ms.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: RBAC Pre-Filtering"):
        st.markdown("**Restricts vector search traversal to chunks matching current user tenant and authorization roles.**")
    with st.expander("🎴 Flashcard 4: Document Tombstoning"):
        st.markdown("**Soft-deleting documents in vector metadata for immediate exclusion before batch graph compaction.**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand how the LangChain Indexing API and RecordManager prevent duplicate embeddings.", value=True)
chk2 = st.checkbox("I know the architectural difference between Exact Prompt Caching and Semantic Vector Caching.", value=True)
chk3 = st.checkbox("I can implement hard metadata pre-filtering for multi-tenant RBAC security.", value=True)
chk4 = st.checkbox("I understand the mechanics of document tombstoning and HNSW index compaction.", value=True)
chk5 = st.checkbox("I know which operational telemetry metrics (TTFT, token burn, top-1 score) to monitor in production.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 production RAG technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 16.4 Production RAG!")
