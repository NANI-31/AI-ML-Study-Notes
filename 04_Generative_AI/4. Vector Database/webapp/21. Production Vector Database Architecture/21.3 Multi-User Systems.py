import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="21.3 Multi-User Systems - Vector DB Curriculum",
    page_icon="👥",
    layout="wide"
)

# Title & Overview
st.title("21.3 Multi-User Systems: User & Tenant Isolation, Metadata Permissions & Collection Routing")
st.caption("Chapter 21: Production Vector Database Architecture | Domain: 4. Vector Database")
st.markdown("""
In a multi-user SaaS platform, thousands of individual users and hundreds of enterprise organizations 
share the same underlying AI infrastructure. 

Architecting a multi-user vector database requires balancing **two conflicting engineering forces**:
1. **Security & Regulatory Isolation**: Guaranteeing that User A can never retrieve documents belonging to User B (preventing cross-tenant data leaks).
2. **Resource & Memory Scalability**: Ensuring that the database does not exhaust server RAM, file descriptors, and CPU cores by creating an unmaintainable number of separate indexes.

To navigate this trade-off, systems apply the **Multi-Tenant Isolation Spectrum**:
* **User-Specific Collections**: A dedicated collection for each individual user (maximum security, high overhead).
* **Tenant-Specific Collections**: One collection per enterprise company, shared among internal employees.
* **Shared Multi-Tenant Collection with Enforced Metadata Pre-Filtering**: Single global collection with compound security predicates.

In this lesson, we explore multi-user isolation patterns, implement an adaptive collection router in Python, 
tackle the **Noisy Neighbor Problem**, and test an interactive Multi-User Architecture Sizing Simulator.
""")

st.divider()

# Section 1: The Multi-User Isolation Spectrum
st.header("1. The Multi-User Isolation Spectrum", divider="rainbow")

col_spec1, col_spec2 = col_spec1, col_spec2 = st.columns([1.1, 0.9])

with col_spec1:
    st.markdown("""
    ### Three Architectural Patterns Compared
    
    * **Pattern 1: User-Specific Collections (`user_{user_id}_kb`)**:
      * *Mechanism*: Every single user gets a dedicated ChromaDB collection.
      * *Pros*: Physical separation; zero data bleed; deleting a user account is an instantaneous collection drop.
      * *Cons*: Severe scalability ceiling. Having 50,000 users creates 50,000 independent HNSW graphs, crashing server RAM and exhausting file descriptors.
      * *Best For*: Premium enterprise VIP tiers or desktop AI apps with single-user local stores.
    
    * **Pattern 2: Tenant-Specific Collections (`org_{tenant_id}_kb`)**:
      * *Mechanism*: One collection per enterprise client organization.
      * *Pros*: Isolates corporate boundaries completely; balances resource consumption (hundreds of collections rather than tens of thousands).
      * *Best For*: B2B enterprise SaaS (e.g., Salesforce, Notion workspaces).
    
    * **Pattern 3: Shared Partitioned Collection (Metadata-Gated)**:
      * *Mechanism*: All users share a single, massive collection. Security is enforced via mandatory filters:  
        `where={"$and": [{"tenant_id": "org_42"}, {"user_id": "user_99"}]}`.
      * *Pros*: Maximum hardware efficiency; scales seamlessly to millions of consumer users.
      * *Best For*: High-volume B2C applications.
    """)

with col_spec2:
    st.markdown("### Isolation Spectrum Topology")
    spec_dot = """
    digraph IsolationSpectrum {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_p1 {
            label="Pattern 1: User-Specific Collections"; style="dashed"; color="#DC2626";
            U1 [label="Collection: user_alice\\n(Isolated HNSW)", fillcolor="#FEE2E2"];
            U2 [label="Collection: user_bob\\n(Isolated HNSW)", fillcolor="#FEE2E2"];
        }

        subgraph cluster_p2 {
            label="Pattern 2: Tenant-Specific Collections"; style="dashed"; color="#D97706";
            T1 [label="Collection: org_acme_corp\\n(Shared Acme Employees)", fillcolor="#FEF3C7"];
            T2 [label="Collection: org_globex_corp\\n(Shared Globex Employees)", fillcolor="#FEF3C7"];
        }

        subgraph cluster_p3 {
            label="Pattern 3: Shared Global Collection"; style="dashed"; color="#059669";
            GlobalColl [label="Single Shared Enterprise Collection\\nEnforces where={'tenant_id': ...}", fillcolor="#D1FAE5", style="filled,bold"];
        }
    }
    """
    st.graphviz_chart(spec_dot, use_container_width=False)

st.divider()

# Section 2: Implementing Dynamic Multi-User Routing in Python
st.header("2. Implementing Dynamic Multi-User Routing in Python", divider="rainbow")

st.markdown("""
Modern enterprise platforms use a **Tiered Hybrid Router**: VIP enterprise customers receive dedicated collections, 
while standard users share a partitioned multi-tenant index:
""")

st.code("""import chromadb
from typing import Dict, Any, Optional

class AdaptiveMultiUserRouter:
    \"\"\"Dynamically routes vector queries based on customer enterprise tier.\"\"\"
    
    def __init__(self, chroma_client: chromadb.HttpClient):
        self.client = chroma_client
        # Pre-initialize shared consumer collection
        self.shared_collection = self.client.get_or_create_collection(
            name="shared_b2c_knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

    def route_query(
        self,
        query_text: str,
        user_jwt: Dict[str, Any],
        k: int = 5
    ) -> Dict[str, Any]:
        tenant_id = user_jwt["tenant_id"]
        user_id = user_jwt["user_id"]
        tier = user_jwt.get("tier", "standard")  # "enterprise" or "standard"

        if tier == "enterprise":
            # VIP Route: Dedicated Isolated Collection per Enterprise Tenant
            collection_name = f"tenant_{tenant_id}_vault"
            tenant_collection = self.client.get_collection(collection_name)
            
            # Filter internally by document access level
            results = tenant_collection.query(
                query_texts=[query_text],
                n_results=k,
                where={"allowed_users": {"$in": [user_id, "all_org"]}}
            )
        else:
            # Standard Route: Shared Partitioned Collection with Strict Predicates
            results = self.shared_collection.query(
                query_texts=[query_text],
                n_results=k,
                where={
                    "$and": [
                        {"tenant_id": tenant_id},
                        {"user_id": user_id}
                    ]
                }
            )

        return results
""", language="python")

st.divider()

# Section 3: Mitigating the Noisy Neighbor Problem & Resource Quotas
st.header("3. Mitigating the Noisy Neighbor Problem & Resource Quotas", divider="rainbow")

col_nn1, col_nn2 = col_nn1, col_nn2 = st.columns([1, 1])

with col_nn1:
    st.markdown("""
    ### The Noisy Neighbor Pathology
    In a shared cluster, one aggressive user or misconfigured bot script can:
    * Ingest 500,000 vectors in rapid succession, acquiring SQLite write locks and starving other users.
    * Spam heavy queries with `efSearch=256` across large collections, pinning CPU cores at 100%.
    * Starve innocent users sharing the same node, driving up search p99 latencies from 2ms to 1,500ms!
    """)

with col_nn2:
    st.markdown("""
    ### Production Quota & Throttling Safeguards
    1. **Rate Limiting (QPM / TPM)**:
       * Restrict users to 60 Queries Per Minute (QPM) at the API gateway layer using Redis Token Bucket.
    2. **Ingestion Quota Limits**:
       * Enforce hard document limits per account: Standard users capped at 50,000 vectors; Enterprise capped at 5,000,000 vectors.
    3. **Query Timeout Guardrails**:
       * Hard abort any vector query exceeding **500 milliseconds** to prevent rogue queries from locking CPU threads.
    4. **Separate Read/Write Pools**:
       * Ingestion ETL workers write to a private writer node; read traffic hits autoscaling follower replicas.
    """)

st.divider()

# Section 4: Interactive Workbench: Multi-User Architecture Sizing Simulator
st.header("4. Interactive Workbench: Multi-User Architecture Sizing Simulator", divider="rainbow")

st.markdown("""
Configure your SaaS user base and observe how different isolation strategies impact 
total collection count, RAM overhead, file descriptor consumption, and security guarantees.
""")

col_sim_u, col_sim_m = col_sim_u, col_sim_m = st.columns(2)

with col_sim_u:
    st.subheader("1. SaaS Customer Demographics")
    num_organizations = st.slider("Total Enterprise Organizations (Tenants):", min_value=10, max_value=2000, value=250, step=10)
    users_per_org = st.slider("Average Users per Organization:", min_value=5, max_value=500, value=50, step=5)
    total_users = num_organizations * users_per_org
    st.caption(f"Total Platform Users: **{total_users:,} Registered Employees**")

with col_sim_m:
    st.subheader("2. Isolation Strategy")
    active_strategy = st.radio(
        "Choose Isolation Strategy:",
        [
            "Strategy 1: User-Specific Collections (1 collection per user)",
            "Strategy 2: Tenant-Specific Collections (1 collection per company)",
            "Strategy 3: Shared Partitioned Collection (Single shared index)"
        ]
    )

# Sizing modeling
if "Strategy 1" in active_strategy:
    coll_count = total_users
    ram_overhead_gb = coll_count * 0.045  # ~45MB base RAM per HNSW graph
    fd_handles = coll_count * 6
    sec_risk = "🟢 Zero Risk (Physical separation per user)"
    scalability_verdict = "❌ Unmaintainable (Exhausts system file descriptors and crashes RAM)" if coll_count > 5000 else "🟡 Heavy RAM overhead"
elif "Strategy 2" in active_strategy:
    coll_count = num_organizations
    ram_overhead_gb = coll_count * 0.085
    fd_handles = coll_count * 6
    sec_risk = "🟢 Zero Cross-Tenant Risk (Physical boundary per company)"
    scalability_verdict = "🏆 Production Gold Standard for B2B SaaS"
else:  # Strategy 3
    coll_count = 1
    ram_overhead_gb = 4.5
    fd_handles = 12
    sec_risk = "⚠️ Software Logic Dependent (Relies on strict where={'tenant_id': ...} query filters)"
    scalability_verdict = "🏆 Maximum Hardware Efficiency for B2C SaaS"

st.divider()

col_res_box1, col_res_box2 = col_res_box1, col_res_box2 = st.columns([1, 1.2])

with col_res_box1:
    st.subheader("Hardware & Resource Footprint")
    st.metric(label="Active HNSW Collections", value=f"{coll_count:,} Collections")
    st.metric(label="HNSW Graph RAM Overhead", value=f"{ram_overhead_gb:.1f} GB RAM")
    st.metric(label="Open File Descriptors", value=f"{fd_handles:,} File Handles")

with col_res_box2:
    st.subheader("Architecture Feasibility & Security Assessment")
    st.info(f"**Security Isolation Guarantee**: {sec_risk}")
    
    if "Strategy 1" in active_strategy and total_users > 3000:
        st.error(f"🚨 **CRITICAL OVERLOAD**: Creating {total_users:,} collections requires {ram_overhead_gb:.1f} GB of RAM just for empty graph metadata pointers! Move to Tenant Collections.")
    elif "Strategy 2" in active_strategy:
        st.success(f"✅ **BALANCED EXCELLENCE**: Managing {num_organizations} collections is well within Kubernetes memory limits while guaranteeing zero cross-company data bleed.")
    else:
        st.success("✅ **HIGH EFFICIENCY**: Minimal resource footprint. Ensure non-bypassable query wrapper proxies are deployed to prevent cross-tenant queries.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What are the primary trade-offs between Tenant Collections vs. Metadata-Filtered Collections?"):
    st.markdown("""
    **Answer**:
    * **Tenant Collections (Collection-per-Tenant)**:
      * *Pros*: Physical isolation eliminates any possibility of cross-tenant data bleed. Deleting a tenant upon churn is an instant `drop collection`.
      * *Cons*: Memory overhead scales with tenant count. 10,000 tenants create 10,000 HNSW graphs, exhausting file descriptors and OS page cache.
    * **Metadata-Filtered Collections (Shared Collection)**:
      * *Pros*: Scales to millions of tenants on minimal RAM; single HNSW graph index.
      * *Cons*: Relies entirely on application software discipline to include `where={"tenant_id": ...}`. A single bug can expose customer data across tenants.
    """)

with st.expander("Q2: How do you solve the 'Noisy Neighbor Problem' in a multi-tenant vector database?"):
    st.markdown("""
    **Answer**:
    1. **Rate Limiting at API Gateway**: Enforce per-tenant token-bucket rate limits (e.g., 100 QPM for Tenant A, 500 QPM for VIP Tenant B).
    2. **Dedicated Read Replicas for VIPs**: Route high-paying enterprise tenants to dedicated read replicas, while free-tier tenants share a pooled cluster.
    3. **Ingestion Queue Throttling**: Run background ingestion jobs through Celery/Temporal queues with concurrency limits so batch indexing does not monopolize CPU cores needed for search.
    4. **Query Execution Deadlines**: Enforce hard client timeouts (`timeout=350ms`) to abort runaway queries.
    """)

with st.expander("Q3: How does GDPR Right-to-be-Forgotten differ between Tenant Collections and Shared Collections?"):
    st.markdown("""
    **Answer**:
    * **In Tenant Collections**: Dropping the entire tenant collection (`client.delete_collection("tenant_acme")`) physically deletes the SQLite database and index files from disk, providing immediate cryptographic data erasure.
    * **In Shared Collections**: Calling `collection.delete(where={"tenant_id": "tenant_acme"})` only creates soft-delete tombstones in HNSW memory. The physical disk segments continue to store the vector coordinates until the database runs an offline index vacuum and compaction cycle.
    """)

with st.expander("Q4: Why does creating 10,000 ChromaDB collections cause Linux server crashes?"):
    st.markdown("""
    **Answer**:
    1. **File Descriptor Exhaustion (`EMFILE` error)**: Every collection maintains open file handles for SQLite WAL logs, index binary files, and lockfiles. Linux defaults to 1,024 open files per process (`ulimit -n`).
    2. **OS Page Cache Thrashing**: The Linux kernel must maintain file metadata and directory inode caches for 10,000 separate folder trees, evicting vector search pages from RAM.
    3. **Memory Fragmentation**: In-process memory allocators (malloc/jemalloc) suffer high fragmentation maintaining tens of thousands of tiny sparse graph heaps.
    """)

with st.expander("Q5: What is a Tiered Hybrid Multi-Tenant Router?"):
    st.markdown("""
    **Answer**:
    A production architecture that stratifies customers based on contractual tier:
    * **Tier 1 (Enterprise / VIP)**: Hosted on dedicated, single-tenant collections or dedicated private nodes with custom encryption keys.
    * **Tier 2 (Pro / Teams)**: Hosted on shared tenant collections (one collection per company).
    * **Tier 3 (Free / Individual B2C)**: Pooled into a massive shared collection partitioned via `tenant_id` and `user_id` metadata filters.
    * This optimizes infrastructure costs while providing strict compliance isolation for paying enterprise accounts.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Creating a Collection per Consumer User**  
    Spinning up separate collections for 50,000 consumer mobile users crashes server memory and exhausts file descriptors within days. Use metadata partitioning.
    """)
    st.warning("""
    **Trap 2: Optional Tenant Filtering in Query Wrappers**  
    Allowing developers to write ad-hoc queries where `where={"tenant_id": ...}` is optional invites accidental cross-tenant data leaks. Enforce mandatory injection at gateway.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Unbounded Search Beam Widths for Free Users**  
    Allowing free-tier users to run searches with `efSearch=512` locks server CPU cores and degrades performance for paying customers. Cap parameters by user tier.
    """)
    st.warning("""
    **Trap 4: Missing Inode and File Descriptor Alerts**  
    Monitoring only CPU and RAM while ignoring open file descriptor counts (`cat /proc/sys/fs/file-nr`) allows collection growth to trigger sudden OS crash failures.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the Noisy Neighbor Problem?"):
        st.markdown("""
        **Resource Starvation**:  
        One heavy user spams queries or batch writes, exhausting CPU cores and degrading latency for all other users.
        """)
    with st.expander("🗂️ Flashcard: Why choose Tenant Collections?"):
        st.markdown("""
        **B2B Gold Standard**:  
        Physical isolation per company eliminates cross-customer data leaks with manageable collection counts.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is Tiered Hybrid Routing?"):
        st.markdown("""
        **Adaptive Strategy**:  
        VIP enterprises receive dedicated collections; standard B2C users share a partitioned multi-tenant index.
        """)
    with st.expander("🗂️ Flashcard: How to mitigate Noisy Neighbors?"):
        st.markdown("""
        **Safeguards**:  
        Rate limiting (QPM), query timeout deadlines (350ms), and separate read/write node pools.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the Multi-Tenant Isolation Spectrum: User Collections vs Tenant Collections vs Metadata Partitioning.", value=True)
chk2 = st.checkbox("I know why creating a collection per individual consumer user causes file descriptor and memory exhaustion.", value=True)
chk3 = st.checkbox("I can implement a Tiered Hybrid Multi-User Router in Python using ChromaDB.", value=True)
chk4 = st.checkbox("I know how to prevent the Noisy Neighbor problem using rate limiting, timeouts, and separate node pools.", value=True)
chk5 = st.checkbox("I understand the compliance and GDPR Right-to-be-Forgotten implications of shared vector storage.", value=True)
chk6 = st.checkbox("I can architect a resilient, highly-scalable multi-user enterprise vector search platform.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Multi-User Systems and are ready for Chapter 21.4: Deployment, Backups & Production Operations!")
