import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="31.2 Duplicate Detection Pipeline - Embeddings Curriculum",
    page_icon="🕸️",
    layout="wide"
)

# Title & Overview
st.title("31.2 Duplicate Detection Pipeline: Similarity Comparison, Threshold Tuning & Graph Grouping")
st.caption("Chapter 31: Embeddings for Duplicate Detection | Domain: 3. Vector Embeddings")
st.markdown("""
Identifying duplicates in a small list of 10 sentences is straightforward. 
However, deploying deduplication in production across **millions of documents, customer questions, or catalog products** 
requires a disciplined, scalable algorithmic pipeline.

A naive approach that compares every document against every other document requires $O(N^2)$ pairwise comparisons. 
For 1,000,000 documents, that represents **500 billion dot products**—which would take days of compute time!

The production **Duplicate Detection Pipeline** solves this through five disciplined stages:
1. **Batch Embedding Generation**: Encoding documents into dense L2-normalized vectors.
2. **Index-Based ANN Self-Querying**: Using an HNSW vector database to find nearest neighbors in $O(N \\log N)$ time.
3. **Calibrated Threshold Filtering**: Retaining only candidate pairs exceeding a rigorously validated Cosine Similarity threshold (e.g., $\\ge 0.88$).
4. **Graph Edge Construction**: Treating duplicate pairs as undirected edges in a graph.
5. **Connected Component Grouping (Union-Find)**: Merging connected documents into canonical duplicate clusters and designating a single **Master Document**.
""")

st.divider()

# Section 1: The 5-Stage Production Deduplication Pipeline
st.header("1. Architecture of the 5-Stage Deduplication Pipeline", divider="rainbow")

col_pipe1, col_pipe2 = col_pipe1, col_pipe2 = st.columns([1, 1])

with col_pipe1:
    st.markdown("""
    ### The 5 Core Pipeline Stages
    * **Stage 1: Vector Extraction**:
      * Ingest raw documents: pass through encoder model (e.g., `text-embedding-3-small`).
      * Normalize all vectors: `v = v / ||v||_2`.
    * **Stage 2: Approximate Nearest Neighbor Search ($O(N \\log N)$)**:
      * Instead of an $N \\times N$ all-pairs matrix, insert vectors into an HNSW index.
      * For each document, execute an **ANN radius search** (or top-$k$ search) to retrieve candidates with cosine similarity above threshold $\\tau$.
    * **Stage 3: Threshold Filtering & Precision Verification**:
      * Filter out pairs below the calibrated similarity boundary $\\tau$.
      * Optionally run top pairs through a Cross-Encoder to eliminate false-positive negations.
    * **Stage 4: Duplicate Graph Construction**:
      * Treat documents as vertices $V$.
      * Add an undirected edge $(u, v)$ for every confirmed duplicate pair.
    * **Stage 5: Connected Component Clustering (Disjoint Set Union)**:
      * Execute the **Union-Find algorithm** to find all connected components.
      * Each component forms a **Duplicate Group**.
      * Elect the longest, oldest, or highest-quality document as the **Canonical Master**.
    """)

with col_pipe2:
    st.markdown("### Production Deduplication Architecture")
    pipe_dot = """
    digraph DedupPipeline {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Corpus [label="Enterprise Corpus\\n1,000,000 Inbound Documents", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Encoder [label="Stage 1: Batch Embedder\\nGenerates 1536-dim Normalized Vectors", fillcolor="#DBEAFE"];

        HNSWIndex [label="Stage 2: HNSW Vector Index\\nExecutes Radius ANN Search\\n(O(N log N) scaling)", fillcolor="#FEF3C7", color="#F59E0B"];

        ThresholdFilter [label="Stage 3: Calibrated Threshold\\nRetains pairs with Cosine Sim >= 0.88", fillcolor="#FDE68A"];

        GraphUnion [label="Stage 4 & 5: Union-Find (DSU)\\nGroups Connected Duplicate Components", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];

        CleanCorpus [label="Deduplicated Clean Corpus\\nCanonical Masters Retained (65% size)\\nRedundant Clones Archived", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        Corpus -> Encoder;
        Encoder -> HNSWIndex;
        HNSWIndex -> ThresholdFilter;
        ThresholdFilter -> GraphUnion;
        GraphUnion -> CleanCorpus;
    }
    """
    st.graphviz_chart(pipe_dot, use_container_width=False)

st.divider()

# Section 2: Connected Components & The Union-Find Algorithm
st.header("2. Grouping Duplicate Clusters with Union-Find (Disjoint-Set Union)", divider="rainbow")

col_ds1, col_ds2 = col_ds1, col_ds2 = st.columns([1, 1])

with col_ds1:
    st.markdown("""
    ### Why Union-Find (DSU) Is Essential
    When pairwise duplicate checks run, duplicates naturally form **transitive chains**:
    * Document 1 is a duplicate of Document 2.
    * Document 2 is a duplicate of Document 3.
    * Therefore, Documents 1, 2, and 3 belong to the **exact same duplicate group**, even if Document 1 and Document 3 were never directly compared!

    **The Disjoint-Set Union (DSU) Data Structure**:
    * Maintains a collection of disjoint sets.
    * Supports two near-constant time operations:
      1. `find(x)`: Finds the canonical representative (root) of the set containing element $x$ with **path compression**.
      2. `union(x, y)`: Merges the set containing $x$ with the set containing $y$ using **union by rank**.
    * **Complexity**: $O(N \\cdot \\alpha(N))$, where $\\alpha$ is the Inverse Ackermann function (effectively $O(1)$ per operation).
    * Groups millions of duplicate pairs into clean clusters in milliseconds!
    """)

with col_ds2:
    st.markdown("### Production Code: Union-Find Duplicate Clusterer")
    st.code("""
class DisjointSetUnion:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, x):
        if self.parent[x] != x:
            # Path compression
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

def group_duplicate_documents(duplicate_pairs: list[tuple[str, str]], all_doc_ids: list[str]) -> dict:
    dsu = DisjointSetUnion(all_doc_ids)
    
    # Merge all connected duplicate pairs
    for doc_a, doc_b in duplicate_pairs:
        dsu.union(doc_a, doc_b)
        
    # Group by canonical root
    clusters = {}
    for doc_id in all_doc_ids:
        root = dsu.find(doc_id)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(doc_id)
        
    return clusters
""", language="python")

st.divider()

# Section 3: Interactive Workbench — Duplicate Pipeline & Graph Cluster Simulator
st.header("3. Interactive Workbench: Live Graph Deduplication & Union-Find Studio", divider="rainbow")

st.markdown("""
Watch the duplicate detection pipeline run live on a simulated enterprise ticket corpus. 
Adjust the **Cosine Similarity Threshold slider** to see how graph edges form between similar documents, 
how the Union-Find algorithm clusters connected components, and how each group selects its canonical master!
""")

# Sample corpus with multiple duplicate clusters and distinct topics
corpus_items = [
    # Cluster A: Password recovery (3 duplicates)
    {"id": "DOC_01", "text": "How do I reset my account password?", "cat": "Auth"},
    {"id": "DOC_02", "text": "What are the instructions to recover forgotten account credentials?", "cat": "Auth"},
    {"id": "DOC_03", "text": "I forgot my password, how can I regain access to my profile?", "cat": "Auth"},
    
    # Cluster B: Kubernetes 137 OOM (2 duplicates)
    {"id": "DOC_04", "text": "Kubernetes pods are crashing with exit code 137 OOMKilled.", "cat": "DevOps"},
    {"id": "DOC_05", "text": "Container terminated by Linux kernel OOM killer error 137.", "cat": "DevOps"},
    
    # Cluster C: Duplicate Invoice requests (2 duplicates)
    {"id": "DOC_06", "text": "Please provide an updated VAT invoice for our accounting team.", "cat": "Billing"},
    {"id": "DOC_07", "text": "Where can our finance department download past VAT invoice receipts?", "cat": "Billing"},
    
    # Singleton unique documents
    {"id": "DOC_08", "text": "How do I configure Kafka consumer partitions in Go?", "cat": "DevOps"},
    {"id": "DOC_09", "text": "What is the recommended recipe for baking sourdough bread?", "cat": "Other"}
]

# Simulated 16-d embeddings
np.random.seed(42)
category_anchors = {
    "Auth": np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]),
    "DevOps": np.array([0.1, 0.1, 0.9, 0.8, 0.7, 0.9, 0.1, 0.1, 0.1, 0.9, 0.8, 0.9, 0.1, 0.9, 0.8, 0.1]),
    "Billing": np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1]),
    "Other": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
}

doc_vectors = {}
for d in corpus_items:
    base = category_anchors[d["cat"]]
    noise = np.random.normal(0, 0.08, 16)
    vec = base + noise
    doc_vectors[d["id"]] = vec / np.linalg.norm(vec)

# Pre-calculate pairwise similarity matrix
n_docs = len(corpus_items)
pairwise_sims = {}
for i in range(n_docs):
    id_i = corpus_items[i]["id"]
    for j in range(i + 1, n_docs):
        id_j = corpus_items[j]["id"]
        # Explicit high calibration for known duplicate pairs
        if id_i in ["DOC_01", "DOC_02", "DOC_03"] and id_j in ["DOC_01", "DOC_02", "DOC_03"]:
            sim = 0.92 if (id_i == "DOC_01" and id_j == "DOC_02") else (0.89 if (id_i == "DOC_02" and id_j == "DOC_03") else 0.87)
        elif id_i in ["DOC_04", "DOC_05"] and id_j in ["DOC_04", "DOC_05"]:
            sim = 0.93
        elif id_i in ["DOC_06", "DOC_07"] and id_j in ["DOC_06", "DOC_07"]:
            sim = 0.90
        elif corpus_items[i]["cat"] == corpus_items[j]["cat"]:
            sim = 0.65
        else:
            sim = round(float(np.dot(doc_vectors[id_i], doc_vectors[id_j])), 3)
            sim = max(0.10, sim)
            
        pairwise_sims[(id_i, id_j)] = sim

col_th_ctrl, col_th_view = col_th_ctrl, col_th_view = st.columns([1, 1])

with col_th_ctrl:
    st.subheader("Step 1: Threshold Tuning")
    sim_threshold = st.slider("Cosine Similarity Threshold (tau):", min_value=0.75, max_value=0.98, value=0.88, step=0.01)
    
    st.markdown(f"**Current Threshold**: `CosineSim >= {sim_threshold:.2f}`")
    
    # Identify duplicate pairs exceeding threshold
    detected_pairs = []
    for (id_a, id_b), s in pairwise_sims.items():
        if s >= sim_threshold:
            detected_pairs.append((id_a, id_b, s))
            
    st.metric("Confirmed Duplicate Edges", len(detected_pairs))
    
    st.markdown("### Detected Duplicate Edges:")
    if detected_pairs:
        df_edges = pd.DataFrame([
            {"Doc A": p[0], "Doc B": p[1], "Cosine Similarity": f"{p[2]:.3f}"}
            for p in detected_pairs
        ])
        st.dataframe(df_edges, use_container_width=True, hide_index=True)
    else:
        st.info("No pairs exceed threshold! Increase sensitivity by lowering threshold.")

# Execute Union-Find to group components
class LocalDSU:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry

dsu_instance = LocalDSU([d["id"] for d in corpus_items])
for id_a, id_b, _ in detected_pairs:
    dsu_instance.union(id_a, id_b)

clusters_map = {}
for d in corpus_items:
    root = dsu_instance.find(d["id"])
    if root not in clusters_map:
        clusters_map[root] = []
    clusters_map[root].append(d)

with col_th_view:
    st.subheader("Step 2: Discovered Duplicate Clusters (Union-Find)")
    
    n_unique_clusters = len(clusters_map)
    st.metric("Total Deduplicated Master Items", n_unique_clusters, delta=f"{len(corpus_items) - n_unique_clusters} Duplicates Eliminated")
    
    for c_idx, (root_id, members) in enumerate(clusters_map.items(), 1):
        is_dup_group = len(members) > 1
        
        # Canonical master election: longest text
        canonical = max(members, key=lambda m: len(m["text"]))
        
        with st.container():
            if is_dup_group:
                st.markdown(f"### Duplicate Cluster #{c_idx}: `{len(members)} Documents Merged` :link:")
                st.success(f"**Canonical Master**: `[{canonical['id']}]` {canonical['text']}")
                st.markdown("**Duplicate Clones Merged Into Master:**")
                for m in members:
                    if m["id"] != canonical["id"]:
                        st.caption(f"• `[{m['id']}]` {m['text']}")
            else:
                st.markdown(f"### Unique Document #{c_idx}: `[{members[0]['id']}]`")
                st.caption(f"{members[0]['text']}")
            st.divider()

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: Why is an all-pairs O(N^2) similarity matrix unacceptable for web-scale deduplication, and how does ANN solve it?", expanded=True):
    st.markdown("""
    **Answer:**
    An all-pairs similarity matrix requires computing $\\frac{N(N - 1)}{2}$ dot products:
    * For $N = 1,000$ documents: $\\sim 500,000$ operations (takes a fraction of a second).
    * For $N = 1,000,000$ documents: $\\sim 500,000,000,000$ operations (500 billion dot products). At 100M dot products/second on GPU, this requires hours of sustained compute and hundreds of gigabytes of RAM.
    * For $N = 50,000,000$ documents: Mathematically impossible with brute-force.
    
    **The Vector Database / ANN Solution**:
    * Instead of all-pairs comparison, documents are inserted into an **HNSW graph index**.
    * Each document vector is queried against the index using a **Radius Nearest Neighbor Search** with radius threshold $\\tau$:
      `results = index.search_range(v_query, radius=tau)`
    * HNSW graph traversal identifies candidate neighbors in sub-linear $O(\\log N)$ time per document.
    * Total pipeline complexity drops from $O(N^2)$ down to $O(N \\log N)$, completing 1,000,000 document deduplication in under 15 minutes!
    """)

with st.expander("Q2: How does the Disjoint-Set Union (Union-Find) algorithm prevent quadratic clustering overhead in deduplication?"):
    st.markdown("""
    **Answer:**
    Once duplicate pairs are identified as edges $(u, v)$, grouping them into connected components can naively require expensive graph traversal (BFS/DFS).
    
    **Union-Find (DSU)** solves this with near-optimal efficiency:
    * It maintains parent pointers with **Path Compression** (`parent[x] = find(parent[x])`) and **Union by Rank**.
    * Whenever an edge $(u, v)$ with similarity $\\ge \\tau$ is discovered, `dsu.union(u, v)` merges their sets in amortized $O(\\alpha(N))$ time (where $\\alpha$ is the Inverse Ackermann function, effectively constant time $\\le 4$).
    * At the end of the stream, a single $O(N)$ pass groups all documents by their root parent, cleanly outputting all duplicate clusters in linear time and minimal memory!
    """)

with st.expander("Q3: How do you choose the Canonical Master Document from a cluster of duplicates?"):
    st.markdown("""
    **Answer:**
    When merging a cluster of 5 duplicate documents, the pipeline must designate one **Canonical Master** to retain while archiving the clones. 
    Production master election policies use deterministic multi-criteria scoring:
    1. **Textual Completeness & Length**: Favor the document with the highest word count or most detailed specifications (e.g., product listing with full specs over a listing with 5 words).
    2. **Temporal Precedence (First-In Priority)**: Retain the oldest document (e.g., original Quora/StackOverflow question with historical comments and upvotes).
    3. **Authority / Source Quality**: In customer support, prioritize verified enterprise tickets over anonymous free-tier tickets.
    4. **Image & Metadata Richness**: In e-commerce, choose the listing with high-resolution image assets and structured category tags.
    """)

with st.expander("Q4: What is the 'Transitive Drift' problem in connected component deduplication, and how is it mitigated?"):
    st.markdown("""
    **Answer:**
    **Transitive Drift** occurs when pairwise similarity creates an elongated semantic bridge across unrelated documents:
    * Document A is similar to Document B (sim: 0.88).
    * Document B is similar to Document C (sim: 0.88).
    * Document C is similar to Document D (sim: 0.88).
    * However, Document A and Document D have a similarity of only 0.62 and discuss completely different topics!
    * Pure connected component grouping merges A, B, C, and D into a single giant cluster.
    
    **Mitigations**:
    * **Cliques / Minimum Degree Constraints**: Require that documents have edges to *multiple* members of the cluster (not just a single bridge) before merging.
    * **Centroid Radius Check**: After merging, verify that all members in the cluster remain within cosine distance $\\tau$ from the cluster centroid; if an item drifts beyond the radius, prune it into a separate cluster.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Brute-Force All-Pairs Dot Product on Production Databases**: Writing an `O(N^2)` Python double loop over 100,000 database rows. The process exhausts RAM and times out workers. Always use HNSW indexing.
* **Trap 2: Ignoring Transitive Drift**: Using pure Union-Find on noisy data with low threshold (0.80). Half your database will collapse into a single monolithic cluster.
* **Trap 3: Deleting Duplicate Clones Irreversibly**: Hard-deleting duplicate documents from the database. If a user queries with an exact URL or ID of the duplicate, they get a 404. Always soft-delete or redirect: `redirect_to = canonical_master_id`.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: O(N log N) Scaling"):
        st.markdown("**Front**: How does the deduplication pipeline avoid O(N^2) complexity?")
        st.divider()
        st.markdown("**Back**: By indexing vectors in an HNSW graph index and executing radius search, reducing search complexity to O(N log N).")

with card_cols[1]:
    with st.popover("Card 2: Union-Find (DSU)"):
        st.markdown("**Front**: What does Union-Find do in deduplication?")
        st.divider()
        st.markdown("**Back**: It groups pairwise duplicate connections into transitively merged connected clusters in amortized O(1) time.")

with card_cols[2]:
    with st.popover("Card 3: Canonical Master"):
        st.markdown("**Front**: How is the Canonical Master elected in a duplicate group?")
        st.divider()
        st.markdown("**Back**: Selected via deterministic business rules: oldest timestamp, longest text completeness, or highest authority rating.")

with card_cols[3]:
    with st.popover("Card 4: Transitive Drift"):
        st.markdown("**Front**: What is Transitive Drift?")
        st.divider()
        st.markdown("**Back**: When a chain of pairwise similarities accidentally connects unrelated documents into one giant mega-cluster.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Duplicate Detection Pipelines:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 5-stage production deduplication pipeline.", key="chk31_2_1")
    st.checkbox("I know how HNSW radius search scales deduplication to O(N log N).", key="chk31_2_2")
    st.checkbox("I can implement the Disjoint-Set Union (Union-Find) algorithm with path compression.", key="chk31_2_3")
with c2:
    st.checkbox("I understand the Canonical Master Document election criteria.", key="chk31_2_4")
    st.checkbox("I know how Transitive Drift occurs and how centroid checks prevent it.", key="chk31_2_5")
    st.checkbox("I understand soft-deletion and redirect pointer best practices.", key="chk31_2_6")
