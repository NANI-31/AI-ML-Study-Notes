import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="10.4 Memory Design - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("10.4 Memory Design: Architecture Trade-Offs, Eviction, Privacy & Compliance")
st.caption("Chapter 10: Agent State and Memory | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Designing memory for enterprise AI agents requires balancing competing engineering constraints: 
**reasoning fidelity** against **inference token costs**, **personalized recall** against **data privacy (GDPR)**, 
and **immediate working context** against **long-term knowledge accumulation**.

A poorly designed memory system either bloats prompt tokens (leading to high costs and attention drift) 
or violates regulatory compliance by indefinitely retaining sensitive user data.

In this lesson, you will master the 6 foundational pillars of enterprise memory system design:
1. **Memory vs Context**: Understanding the fundamental architectural boundary between physical prompt context and external memory stores.
2. **Short-Term vs Long-Term Memory**: Designing synchronization loops between ephemeral threads and durable stores.
3. **Memory Size Management**: Implementing LRU cache eviction, sliding windows, and semantic clustering.
4. **Memory Retrieval Engineering**: Optimizing retrieval latency with similarity thresholds and cross-encoder rerankers.
5. **Memory Privacy & Governance**: Enforcing tenant isolation, role-based access control (RBAC), and encryption at rest.
6. **Memory Deletion & Compliance**: Implementing GDPR "Right to be Forgotten", selective forgetfulness, and automated TTL policies.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockGovernanceReport:
    """Represents a simulated memory governance and compliance audit in module scope."""
    def __init__(self, tenant_id: str, active_records: int, evicted_records: int, purged_gdpr: int, encryption_status: str, audit_log: List[str]):
        self.tenant_id = tenant_id
        self.active_records = active_records
        self.evicted_records = evicted_records
        self.purged_gdpr = purged_gdpr
        self.encryption_status = encryption_status
        self.audit_log = audit_log

class MockMemoryGovernanceSimulator:
    """Simulates memory lifecycle management, LRU eviction, and GDPR deletion in module scope."""
    def __init__(self):
        pass

    def run_governance_cycle(
        self,
        tenant_id: str,
        total_memories: int,
        capacity_limit: int,
        trigger_gdpr_purge: bool
    ) -> MockGovernanceReport:
        logs = []
        logs.append(f"Ingested {total_memories} memory entries for tenant '{tenant_id}'.")

        # 1. LRU Eviction Simulation
        if total_memories > capacity_limit:
            evicted = total_memories - capacity_limit
            logs.append(f"Capacity exceeded ({total_memories} > {capacity_limit}). Evicted {evicted} least-recently-used (LRU) records.")
            remaining = capacity_limit
        else:
            evicted = 0
            remaining = total_memories
            logs.append("Memory storage within capacity limits. Zero eviction required.")

        # 2. GDPR Deletion Simulation
        purged = 0
        if trigger_gdpr_purge:
            purged = min(remaining, 5)
            remaining -= purged
            logs.append(f"GDPR Article 17 request processed: Purged {purged} user records and updated cryptographic erasure log.")
        else:
            logs.append("No active GDPR deletion requests pending.")

        logs.append(f"AES-256 GCM encryption verified at rest. Active persistent records: {remaining}.")

        return MockGovernanceReport(
            tenant_id=tenant_id,
            active_records=remaining,
            evicted_records=evicted,
            purged_gdpr=purged,
            encryption_status="AES-256 GCM (Encrypted)",
            audit_log=logs
        )

# --- SECTION 1: ARCHITECTURE BLUEPRINT & GOVERNANCE TOPOLOGY ---
st.header("1. Architecture Blueprint: Enterprise Memory Governance & Lifecycle Topology", divider="rainbow")

st.markdown("""
The architecture below illustrates the complete **Enterprise Memory Lifecycle**. 
Memory flows through ingestion, categorization, semantic indexing, access control filtering, 
and automated lifecycle pruning (LRU eviction and GDPR cryptographic deletion).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Level 1: Ingestion & Privacy Scrubbing
        subgraph cluster_ingress {
            label = "1. INGESTION &amp; PRIVACY SCRUBBING TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            RawTurn [label="Raw Conversation Message\\nHuman / Tool observation turn"];
            PIIFilter [label="PII Sanitizer &amp; Anonymizer\\nMasks emails, phone numbers &amp; credit cards", color="#10B981"];
        }

        // Level 2: Partitioning & Storage
        subgraph cluster_storage_tier {
            label = "2. NAMESPACED STORAGE TIER (AES-256 Encrypted)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            TenantNamespace [label="Tenant Isolation Boundary\\n(tenant_id, user_id, category)"];
            ActiveMemoryDB [label="Vector &amp; Relational Store\\nPostgreSQL (pgvector) / Redis", color="#38BDF8"];
        }

        // Level 3: Retrieval & Re-ranking
        subgraph cluster_retrieval_tier {
            label = "3. RETRIEVAL &amp; RE-RANKING ENGINE";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            SimilaritySearch [label="Cosine Similarity Search\\nExtracts candidate memories (Top 20)"];
            CrossEncoderReranker [label="Cross-Encoder Reranker\\nPrunes to top 3 highest-signal memories", color="#10B981"];
        }

        // Level 4: Lifecycle & Compliance
        subgraph cluster_compliance {
            label = "4. GOVERNANCE &amp; COMPLIANCE POLICIES";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            LRUEviction [label="LRU Eviction Worker\\nPurges stale memories when capacity exceeds ceiling"];
            GDPRPurge [label="GDPR Article 17 Worker\\nHard deletes records on user request", color="#EF4444"];
            TTLPolicy [label="Automated Expiration (TTL)\\nPurges operational logs after 90 days"];
        }

        // Level 5: Prompt Ingestion
        subgraph cluster_delivery {
            label = "5. AGENT WORKING CONTEXT";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            AugmentedPrompt [label="Grounded Working Prompt\\nZero PII leakage, optimal token budget", color="#10B981"];
        }

        // Wiring
        RawTurn -> PIIFilter;
        PIIFilter -> TenantNamespace [label="Sanitized Facts"];
        TenantNamespace -> ActiveMemoryDB;

        ActiveMemoryDB -> SimilaritySearch [label="On Agent Turn"];
        SimilaritySearch -> CrossEncoderReranker;
        CrossEncoderReranker -> AugmentedPrompt;

        ActiveMemoryDB -> LRUEviction [label="Capacity Check"];
        ActiveMemoryDB -> GDPRPurge [label="Right to Be Forgotten", style=dashed, color="#EF4444"];
        ActiveMemoryDB -> TTLPolicy [label="Scheduled Expiration"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Core Architecture Invariant:**
Memory is not an unbounded append-only log. In enterprise architectures, **storage without lifecycle governance is technical debt and a regulatory liability**.
""")

st.divider()

# --- SECTION 2: MEMORY VS CONTEXT ---
st.header("2. Memory vs Context (Bounded Attention Windows vs Unbounded Stores)", divider="rainbow")

st.markdown("""
It is essential to distinguish between **Context** and **Memory**:
""")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.info("🧠 Model Context Window (RAM)")
    st.markdown("""
    - **Physical Nature**: The volatile, bounded array of tokens evaluated in a single forward pass of the neural network.
    - **Cost Model**: Incurred on every turn for every token.
    - **Capacity**: 8k to 1M tokens.
    - **Persistence**: Zero. Vanishes as soon as the HTTP request completes.
    """)

with col_m2:
    st.success("💾 External Memory Store (Hard Drive)")
    st.markdown("""
    - **Physical Nature**: Durable database records (PostgreSQL, SQLite, Redis) stored across weeks or years.
    - **Cost Model**: Standard database storage fees.
    - **Capacity**: Effectively infinite.
    - **Persistence**: Perpetual across all user sessions and server restarts.
    """)

st.divider()

# --- SECTION 3: SHORT-TERM VS LONG-TERM MEMORY ---
st.header("3. Short-Term vs Long-Term Memory (Architectural Trade-Offs)", divider="rainbow")

st.markdown("""
Enterprise agents harmonize both memory tiers through an active synchronization lifecycle:
""")

st.subheader("Comparison: Short-Term vs Long-Term Memory")
mem_cmp_df = pd.DataFrame({
    "Dimension": ["Scope", "Storage Medium", "Key Identifier", "Retention", "Retrieval Mechanism"],
    "Short-Term Memory": ["Single conversation thread", "PostgresSaver / Checkpointer", "thread_id", "Session duration (Hours)", "Sequential message array replay"],
    "Long-Term Memory": ["Global across all threads", "BaseStore / PostgresStore", "(tenant_id, user_id, key)", "Permanent / Until deleted", "Semantic vector search + Exact key query"]
})
st.dataframe(mem_cmp_df, use_container_width=True)

st.divider()

# --- SECTION 4: MEMORY SIZE MANAGEMENT ---
st.header("4. Memory Size Management (Eviction Strategies & LRU Policies)", divider="rainbow")

st.markdown("""
Without active size management, long-term memory stores accumulate stale, contradictory, and obsolete data. 
Production memory engines deploy three size-management strategies:
1. **LRU (Least Recently Used) Eviction**: When a user's memory count exceeds a threshold (e.g. 500 items), records with the oldest `last_accessed_at` timestamps are deleted.
2. **Semantic Clustering & Deduplication**: Merging similar memories into a single consolidated entry (e.g. combining *"User likes Python"* and *"User writes scripts in Python"*).
3. **Sliding Token Budgets**: Limiting memory prompt injections to at most 15% of the total model context window.
""")

st.code('''# Least Recently Used (LRU) Eviction Policy Pattern:
def evict_lru_memories(store, namespace, max_capacity: int = 100):
    items = store.search(namespace_prefix=namespace, limit=max_capacity * 2)
    if len(items) > max_capacity:
        # Sort by access timestamp ascending
        sorted_items = sorted(items, key=lambda x: x.value.get("last_accessed_at", 0))
        excess_count = len(items) - max_capacity
        
        # Purge oldest excess records
        for item_to_remove in sorted_items[:excess_count]:
            store.delete(namespace=namespace, key=item_to_remove.key)
''', language="python")

st.divider()

# --- SECTION 5: MEMORY RETRIEVAL ENGINEERING ---
st.header("5. Memory Retrieval Engineering (Scoring, Thresholds & Reranking)", divider="rainbow")

st.markdown("""
Retrieving irrelevant memories degrades agent reasoning. 
To guarantee high precision:
- **Similarity Thresholding**: Reject memories with cosine similarity scores below `0.75`.
- **Cross-Encoder Reranking**: Use a lightweight cross-encoder model to re-score candidate memories against the user query, eliminating false positives before prompt assembly.
""")

st.code('''def retrieve_high_precision_memories(store, user_id: str, query: str, threshold: float = 0.80) -> list:
    # 1. Broad vector search
    candidates = store.search(namespace_prefix=("users", user_id), query=query, limit=10)
    
    # 2. Score threshold filter
    high_signal = [c for c in candidates if c.score >= threshold]
    
    # 3. Limit to top 3 highest precision items
    return high_signal[:3]
''', language="python")

st.divider()

# --- SECTION 6: MEMORY PRIVACY & COMPLIANCE ---
st.header("6. Memory Privacy & Compliance (Multi-Tenancy & Encryption)", divider="rainbow")

st.markdown("""
Storing memory for enterprise users demands strict security governance:
1. **Multi-Tenant Partitioning**: Enforce namespace prefixes such as `("tenant_id", "user_id")` at the database level. An agent must never execute un-namespaced cross-tenant searches.
2. **Encryption at Rest**: Memory vectors and text payloads must be encrypted using AES-256.
3. **PII Scrubbing on Ingress**: Pass memories through a PII scrubber (such as Microsoft Presidio) before writing to the database to ensure Social Security numbers and payment data are never permanently persisted.
""")

st.divider()

# --- SECTION 7: MEMORY DELETION & TTL POLICIES ---
st.header("7. Memory Deletion & TTL Policies (GDPR & Selective Forgetfulness)", divider="rainbow")

st.markdown("""
Under **GDPR Article 17 (Right to Be Forgotten)** and the **California Consumer Privacy Act (CCPA)**, 
users have a legal right to request the permanent deletion of their personal data.

Enterprise AI agents must support:
- **`forget_user(user_id)`**: Hard deletion of all records under the user's namespace.
- **Selective Forgetting**: Deleting a single erroneous or outdated preference (e.g. deleting `user_pref_city`).
- **Time-to-Live (TTL)**: Automatically expiring operational and debug memory logs after 30 to 90 days.
""")

st.code('''def gdpr_right_to_be_forgotten(store, user_id: str) -> int:
    """Permanently purges all persistent memories associated with a user."""
    user_memories = store.search(namespace_prefix=("users", user_id))
    purged_count = len(user_memories)
    
    for item in user_memories:
        store.delete(namespace=item.namespace, key=item.key)
        
    print(f"Compliance Audit: Purged {purged_count} records for user '{user_id}'.")
    return purged_count
''', language="python")

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Governed Memory Lifecycle Manager", divider="rainbow")

st.markdown("""
The following complete production pattern demonstrates an **Enterprise Governed Memory Manager** 
featuring capacity bounds, LRU eviction, multi-tenant isolation, and automated GDPR compliance deletion.
""")

st.code('''import time
import logging
from typing import Dict, Any, List, Optional
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger("EnterpriseMemoryGovernor")

class GovernedEnterpriseMemoryStore:
    """Production memory manager enforcing multi-tenant isolation, LRU limits, and GDPR purge."""

    def __init__(self, tenant_capacity_limit: int = 250):
        self.capacity_limit = tenant_capacity_limit
        self.store = InMemoryStore()

    def _sanitize_namespace(self, tenant_id: str, user_id: str, category: str) -> tuple:
        """Enforces strictly partitioned multi-tenant namespace hierarchy."""
        return (tenant_id.strip().lower(), user_id.strip().lower(), category.strip().lower())

    def put_memory(self, tenant_id: str, user_id: str, category: str, key: str, payload: Dict[str, Any]) -> None:
        ns = self._sanitize_namespace(tenant_id, user_id, category)
        payload["updated_at"] = time.time()
        payload["last_accessed_at"] = time.time()

        # Enforce LRU capacity limits before insertion
        existing_items = self.store.search(namespace_prefix=(tenant_id.lower(),))
        if len(existing_items) >= self.capacity_limit:
            oldest = min(existing_items, key=lambda x: x.value.get("last_accessed_at", 0))
            logger.warning(f"Tenant {tenant_id} at capacity. Evicting LRU memory key: {oldest.key}")
            self.store.delete(namespace=oldest.namespace, key=oldest.key)

        self.store.put(namespace=ns, key=key, value=payload)
        logger.info(f"Persisted memory under namespace {ns} with key: {key}")

    def query_memories(self, tenant_id: str, user_id: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        ns = (tenant_id.lower(), user_id.lower())
        results = self.store.search(namespace_prefix=ns, query=query, limit=limit)
        
        # Touch accessed items to update LRU timestamp
        for r in results:
            r.value["last_accessed_at"] = time.time()
            self.store.put(namespace=r.namespace, key=r.key, value=r.value)
            
        return [r.value for r in results]

    def purge_user_data_gdpr(self, tenant_id: str, user_id: str) -> int:
        """Executes GDPR Article 17 hard erasure across all user namespaces."""
        ns = (tenant_id.lower(), user_id.lower())
        items = self.store.search(namespace_prefix=ns)
        count = len(items)
        
        for item in items:
            self.store.delete(namespace=item.namespace, key=item.key)

        logger.info(f"GDPR Purge completed: Erased {count} records for user '{user_id}'.")
        return count
''', language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Enterprise Memory Governance & Deletion Simulator", divider="rainbow")

st.markdown("""
Test how an enterprise memory manager enforces **Tenant Capacity Caps**, triggers **LRU Eviction**, 
and executes **GDPR "Right to be Forgotten" Erasure** in real time.
""")

gov_sim = MockMemoryGovernanceSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("⚙️ Configure Tenant Memory Load")
    tenant_name_input = st.text_input("Tenant Organization ID", value="acme-health-corp")
    total_memories_input = st.slider("Total Accumulated Memory Entries", min_value=10, max_value=200, value=75)
    capacity_limit_input = st.slider("Configured Max Capacity Limit", min_value=20, max_value=100, value=50)
    gdpr_purge_toggle = st.toggle("Trigger GDPR Article 17 Erasure Request", value=False)

with col_wb2:
    st.subheader("🔒 Active Governance Constraints")
    st.info(f"""
    **Compliance Profile:**
    - **Tenant**: `{tenant_name_input}`
    - **Eviction Algorithm**: LRU (Least Recently Used)
    - **Capacity Cap**: `{capacity_limit_input}` entries
    - **GDPR Action**: `{'SCHEDULED FOR HARD PURGE' if gdpr_purge_toggle else 'Standard Retention'}`
    """)

if st.button("🚀 Run Governance & Audit Cycle", type="primary"):
    rep = gov_sim.run_governance_cycle(
        tenant_id=tenant_name_input,
        total_memories=total_memories_input,
        capacity_limit=capacity_limit_input,
        trigger_gdpr_purge=gdpr_purge_toggle
    )

    st.subheader("📊 Governance Audit Report")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Retained Records", rep.active_records)
    m2.metric("LRU Evicted Records", rep.evicted_records)
    m3.metric("GDPR Purged Records", rep.purged_gdpr)
    m4.metric("Encryption Status", "AES-256")

    st.write("#### 🛡️ Compliance & Security Audit Trail:")
    for log in rep.audit_log:
        st.write(f"- 📋 `{log}`")

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why is memory size management critical even when using large context window models (e.g. 1M tokens)?",
     "While modern models support vast context windows, stuffing hundreds of obsolete memories into every prompt increases inference latency (TTFT), multiplies per-turn API token costs exponentially, and degrades model attention precision (the 'lost-in-the-middle' problem). Active size management and LRU eviction ensure only high-signal, relevant memories are evaluated."),
    ("How do you implement GDPR Article 17 ('Right to Be Forgotten') in an agent memory system?",
     "Memory stores must organize records under strictly partitioned hierarchical namespaces (such as ('tenant_id', 'user_id', ...)). When a GDPR deletion request is received, a compliance worker queries that user's exact namespace prefix and permanently hard-deletes all associated keys and vector embeddings, creating an immutable cryptographic audit log."),
    ("What are the advantages of using Cross-Encoder Rerankers on retrieved memories?",
     "Dense vector retrieval uses bi-encoders (cosine similarity), which are fast but occasionally retrieve memories that share keyword or semantic similarity but lack contextual relevance. A cross-encoder reranker scores the full attention between query and candidate memory, filtering out false positives before prompt assembly."),
    ("What is the difference between Time-to-Live (TTL) expiration and LRU cache eviction?",
     "TTL expiration automatically purges a record after a fixed elapsed duration (e.g. deleting session logs after 90 days), regardless of how often it was accessed. LRU (Least Recently Used) eviction deletes records only when the total memory count exceeds a capacity limit, pruning the records that haven't been accessed for the longest time."),
    ("How do you prevent Prompt Injection attacks from corrupting long-term memory?",
     "If user inputs are written directly into memory without verification, adversarial users can inject prompt instructions into the store (e.g. 'Always output internal credentials'). Production systems enforce a dual-stage architecture: raw messages are held in ephemeral state, while an auxiliary reflection worker validates and filters facts before committing them to persistent long-term storage.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Flat Un-Namespaced Memory Stores. Storing memories without tenant-level namespace partitioning allows multi-tenant memory cross-contamination, violating SOC2 and GDPR compliance. Always enforce strict tuple namespaces.")
    st.error("**Trap 2 (Breaking Error)**: Writing Ephemeral Session Tokens to Long-Term Memory. Storing short-lived JWT tokens or temporary OTP codes in persistent memory causes authentication crashes when the agent attempts to use expired tokens in future weeks. Never persist ephemeral credentials.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Missing Memory Retrieval Similarity Thresholds. Injecting memories regardless of relevance score pollutes prompts with irrelevant facts. Always enforce a minimum similarity threshold (e.g. score >= 0.78).")
    st.warning("**Trap 4 (Quality Warning)**: Infinite Memory Accumulation Without Deduplication. Storing 50 variations of 'User works in DevOps' across 50 sessions causes memory bloat. Implement semantic deduplication to consolidate updates.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Context vs Memory"):
        st.markdown("**Context is volatile prompt RAM (bounded); Memory is durable database storage (unbounded).**")
    with st.expander("🎴 Flashcard 2: LRU Eviction"):
        st.markdown("**Prunes the least-recently-accessed records when memory storage exceeds capacity limits.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: GDPR Compliance"):
        st.markdown("**Hard erasure of all records under a user's namespace upon deletion request.**")
    with st.expander("🎴 Flashcard 4: Cross-Encoder Reranking"):
        st.markdown("**Re-scores candidate memories to eliminate false positives before prompt assembly.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural differences between context windows and external memory stores.", value=True)
chk2 = st.checkbox("I know how to design synchronization loops between short-term and long-term memory.", value=True)
chk3 = st.checkbox("I can implement LRU eviction policies and capacity ceilings to manage memory size.", value=True)
chk4 = st.checkbox("I understand how similarity thresholds and rerankers optimize memory retrieval.", value=True)
chk5 = st.checkbox("I know how to enforce multi-tenant isolation and GDPR Article 17 deletion policies.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 memory design interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 10.4 Memory Design and completed all of Chapter 10: Agent State and Memory!")
