import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="16.1 Similarity Retrieval - Vector DB Curriculum",
    page_icon="📐",
    layout="wide"
)

# Title & Overview
st.title("16.1 Similarity Retrieval: Geometric Metrics, Top-K Heaps & Confidence Thresholds")
st.caption("Chapter 16: Advanced Retrieval Techniques | Domain: 4. Vector Database")
st.markdown("""
At the heart of every vector database lies **Similarity Retrieval**—the mathematical process of identifying 
which indexed vectors reside closest to a query vector in high-dimensional space.

While conceptually straightforward, executing similarity retrieval in production requires mastering 
three foundational engineering concepts:
1. **Geometric Distance Space Calibration**: Understanding when to use Cosine Distance, Euclidean (L2) Distance, or Inner Product (Dot Product).
2. **Top-K Min-Heap Mechanics**: How search algorithms track the K best candidates during HNSW graph traversal without sorting millions of vectors.
3. **Similarity Score Thresholding**: Enforcing hard mathematical cutoffs to reject out-of-domain queries and prevent hallucinations before context reaches the LLM.

In this lesson, we explore the geometry of high-dimensional vector spaces, analyze priority queue mechanics, 
learn how to calibrate confidence thresholds, and experiment with an interactive Similarity Retrieval Simulator.
""")

st.divider()

# Section 1: The Geometry of Similarity Retrieval
st.header("1. The Geometry of Similarity Retrieval & Distance Spaces", divider="rainbow")

col_geom1, col_geom2 = col_geom1, col_geom2 = st.columns([1.1, 0.9])

with col_geom1:
    st.markdown("""
    ### Three Primary Distance Formulations
    
    1. **Cosine Similarity / Cosine Distance (`hnsw:space: "cosine"`)**:
       * Measures the **cosine of the angle** between two vectors, completely ignoring their magnitudes:
         ```text
         Cosine_Similarity = Dot_Product(A, B) / (Norm(A) * Norm(B))
         Cosine_Distance = 1.0 - Cosine_Similarity
         ```
       * **Range**: Cosine Distance is strictly bounded between `0.0` (identical direction) and `2.0` (diametrically opposite).
       * **Best For**: Text embeddings where document length variations should not distort semantic meaning.
    
    2. **Euclidean / L2 Distance (`hnsw:space: "l2"`)**:
       * Measures the straight-line physical geometric distance between two coordinate points:
         ```text
         L2_Distance = sqrt( sum( (A_i - B_i)^2 ) )
         ```
       * **Best For**: Image embeddings, spatial coordinates, and audio feature vectors where magnitude carries physical meaning.
    
    3. **Inner Product / Dot Product (`hnsw:space: "ip"`)**:
       * Multiplies components directly: `sum(A_i * B_i)`.
       * When vectors are **L2-normalized to unit length** (`||v|| = 1.0`), Inner Product is mathematically identical to Cosine Similarity, but executes **3x faster** on modern CPU SIMD registers.
    """)

with col_geom2:
    st.markdown("### Geometric Space Comparison")
    geom_dot = """
    digraph MetricGeometry {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Origin [label="Origin (0, 0)", shape=circle, fillcolor="#E5E7EB", width=0.6];

        VecA [label="Vector A (Query)\\nNormalized Unit Hypersphere", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        VecB [label="Vector B (Doc 1)\\nSmall Angle -> High Cosine Sim", fillcolor="#D1FAE5", color="#059669"];
        VecC [label="Vector C (Doc 2)\\nLarge Angle -> High Cosine Dist", fillcolor="#FEE2E2", color="#DC2626"];

        Origin -> VecA [label="Length = 1.0"];
        Origin -> VecB [label="Length = 1.0"];
        Origin -> VecC [label="Length = 1.0"];

        VecA -> VecB [label="Cosine Dist: 0.08\\n(Near Match)", color="#059669", style="dashed"];
        VecA -> VecC [label="Cosine Dist: 0.72\\n(Dissimilar)", color="#DC2626", style="dashed"];
    }
    """
    st.graphviz_chart(geom_dot, use_container_width=False)

st.divider()

# Section 2: Priority Queues & Top-K Min-Heap Mechanics
st.header("2. Priority Queues & Top-K Min-Heap Mechanics", divider="rainbow")

col_heap1, col_heap2 = col_heap1, col_heap2 = st.columns([1, 1])

with col_heap1:
    st.markdown("""
    ### Why Full Sorting is Forbidden
    In a database of 10,000,000 vectors, sorting all distance values using quicksort requires `O(N * log N)` operations—taking several seconds and gigabytes of memory.
    
    **The Bounded Min-Heap / Max-Heap Pattern**:
    * Vector search algorithms maintain a **Bounded Priority Queue (Heap)** of size `K`:
      1. Traverse the HNSW graph or inverted list.
      2. For each candidate node, compute its distance `d`.
      3. If the heap contains fewer than `K` items, push the candidate.
      4. If the heap is full, compare `d` against the **worst element currently in the heap**:
         * If `d` is better than the worst element, pop the worst element and insert the new candidate.
         * Otherwise, discard the candidate immediately!
    * **Complexity**: `O(N * log K)`. When `K = 5`, `log(5) ≈ 2.3`, keeping search overhead strictly under 2 milliseconds!
    """)

with col_heap2:
    st.markdown("### Python Priority Queue Traversal Simulation")
    st.code("""import heapq
from typing import List, Tuple

def top_k_search(query_vec, candidate_vectors, k: int = 3) -> List[Tuple[float, str]]:
    \"\"\"Maintains a max-heap of size K using negative distances.\"\"\"
    heap = []  # Stores (-distance, doc_id)
    
    for doc_id, vec in candidate_vectors.items():
        # Cosine distance computation
        dist = 1.0 - float(np.dot(query_vec, vec))
        
        if len(heap) < k:
            heapq.heappush(heap, (-dist, doc_id))
        else:
            # -dist > heap[0][0] means dist < current worst distance
            if -dist > heap[0][0]:
                heapq.heappushpop(heap, (-dist, doc_id))
                
    # Return sorted from best (smallest distance) to worst
    results = [(-neg_d, doc_id) for neg_d, doc_id in sorted(heap, reverse=True)]
    return results
""", language="python")

st.divider()

# Section 3: Calibrating Confidence Thresholds for Production
st.header("3. Calibrating Confidence Thresholds for Production", divider="rainbow")

st.markdown("""
In naive vector search, querying for a topic that does not exist in your database still returns `K` results. 
To build reliable AI agents, you must enforce a **Mathematical Confidence Gate**:
""")

col_thresh1, col_thresh2 = col_thresh1, col_thresh2 = st.columns([1, 1])

with col_thresh1:
    st.markdown("""
    ### Calibrating the Cutoff Boundary
    In **Cosine Distance space** (`distance = 1.0 - cosine_sim`):
    
    * **`Distance < 0.15` (Similarity > 0.85)**:
      * **Exact Semantic Match**: Direct factual grounding. High confidence.
    * **`0.15 <= Distance <= 0.35` (Similarity 0.65 - 0.85)**:
      * **Broad / Contextual Match**: Useful related context, but may not contain exact specifics.
    * **`Distance > 0.35` (Similarity < 0.65)**:
      * **Out-of-Domain Noise**: The document merely shares generic English vocabulary with the query. 
      * **Action**: **DROP IT!** Never pass chunks with distance > 0.35 into an LLM context prompt.
    """)

with col_thresh2:
    st.markdown("""
    ### Production Threshold Implementation in Chroma
    ```python
    # 1. Query ChromaDB with n_results=5
    results = collection.query(
        query_texts=["What is our quantum encryption key?"],
        n_results=5
    )

    qualified_docs = []
    MAX_ALLOWABLE_DISTANCE = 0.35

    # 2. Filter by calibrated threshold
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        if dist <= MAX_ALLOWABLE_DISTANCE:
            qualified_docs.append((doc, dist, meta))

    # 3. Handle zero-result fallback
    if not qualified_docs:
        print("Query is OUT-OF-DOMAIN. Returning verified fallback response.")
    ```
    """)

st.divider()

# Section 4: Interactive Workbench: Distance Metric & Confidence Threshold Simulator
st.header("4. Interactive Workbench: Distance Metric & Confidence Threshold Simulator", divider="rainbow")

st.markdown("""
Test how varying distance thresholds and top-K configurations affect retrieval quality, 
filtering out irrelevant noise while retaining grounded facts.
""")

col_sim_in, col_sim_params = st.columns([1.1, 0.9])

with col_sim_in:
    st.subheader("1. Test Search Scenario")
    inquiry_scenario = st.selectbox(
        "Select User Inquiry:",
        [
            "In-Domain Query: 'How do I configure mTLS in Envoy proxy?'",
            "Related Query: 'What is our certificate rotation frequency?'",
            "Out-of-Domain Query: 'What are the rules of professional tennis?'"
        ]
    )

with col_sim_params:
    st.subheader("2. Retrieval Gatekeeper Knobs")
    k_slider = st.slider("Requested Top-K Results:", min_value=1, max_value=4, value=3)
    dist_threshold = st.slider(
        "Max Allowable Cosine Distance (Cutoff):",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="Chunks with distance greater than this value will be pruned."
    )

# Simulated Database Records
MOCK_CORPUS = [
    {
        "id": "DOC-ENV-01",
        "title": "Envoy Ingress mTLS Configuration",
        "text": "Mount client certificates in /etc/envoy/certs and set require_client_certificate: true to enforce mTLS.",
        "in_domain_dist": 0.06,
        "related_dist": 0.22,
        "out_domain_dist": 0.74
    },
    {
        "id": "DOC-SPIRE-02",
        "title": "SPIRE Automated Certificate Rotation",
        "text": "SPIRE agents issue X.509 SVID certificates with an automated 1-hour renewal cycle over local domain sockets.",
        "in_domain_dist": 0.18,
        "related_dist": 0.08,
        "out_domain_dist": 0.78
    },
    {
        "id": "DOC-K8S-03",
        "title": "Kubernetes Ingress Gateway Architecture",
        "text": "Envoy runs as an ingress gateway pod managing ingress routes, rate limiting, and mTLS termination.",
        "in_domain_dist": 0.24,
        "related_dist": 0.29,
        "out_domain_dist": 0.81
    },
    {
        "id": "DOC-HR-04",
        "title": "Corporate Code of Conduct",
        "text": "Employees must maintain professional conduct and protect company intellectual property at all times.",
        "in_domain_dist": 0.76,
        "related_dist": 0.72,
        "out_domain_dist": 0.85
    }
]

# Pick distances based on scenario
for d in MOCK_CORPUS:
    if "In-Domain" in inquiry_scenario:
        d["active_dist"] = d["in_domain_dist"]
    elif "Related" in inquiry_scenario:
        d["active_dist"] = d["related_dist"]
    else:
        d["active_dist"] = d["out_domain_dist"]

sorted_candidates = sorted(MOCK_CORPUS, key=lambda x: x["active_dist"])[:k_slider]
passed_candidates = [d for d in sorted_candidates if d["active_dist"] <= dist_threshold]

st.divider()

col_raw_hits, col_passed_hits = st.columns([1, 1])

with col_raw_hits:
    st.subheader(f"1. Raw Unfiltered Top-{k_slider} Candidates")
    st.caption("What standard vector search returns blindly without thresholding.")
    
    for idx, item in enumerate(sorted_candidates):
        is_passed = item["active_dist"] <= dist_threshold
        box_type = st.success if is_passed else st.error
        with st.expander(f"{'✅' if is_passed else '❌'} Rank {idx+1}: {item['title']} (Dist: {item['active_dist']:.2f})", expanded=True):
            st.markdown(f"*{item['text']}*")
            st.caption(f"Status: **{'PASSED THRESHOLD' if is_passed else 'PRUNED (Distance too high)'}**")

with col_passed_hits:
    st.subheader(f"2. Qualified Results Sent to LLM ({len(passed_candidates)} Chunks)")
    st.caption(f"Only candidates with Cosine Distance <= {dist_threshold:.2f} are admitted.")
    
    if passed_candidates:
        st.success(f"Admitted **{len(passed_candidates)} grounded chunks** into context prompt.")
        for item in passed_candidates:
            st.markdown(f"• **{item['title']}** (Confidence: `{(1.0 - item['active_dist']) * 100:.1f}%`)")
    else:
        st.error("🚨 Zero chunks passed the confidence threshold!")
        st.info("System Guardrail Triggered: The RAG assistant will safely report: *'No verified documentation found to answer this inquiry.'* (Hallucination completely prevented!)")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the relationship between Cosine Distance, Inner Product, and Euclidean (L2) Distance when vectors are normalized?"):
    st.markdown("""
    **Answer**:
    When all vectors are **L2-normalized to unit length** (`||A|| = 1.0` and `||B|| = 1.0`):
    1. `Cosine_Similarity(A, B) = <A, B> = Inner_Product(A, B)`.
    2. `Cosine_Distance = 1.0 - <A, B>`.
    3. `L2_Distance^2 = ||A - B||^2 = ||A||^2 + ||B||^2 - 2 * <A, B> = 2.0 - 2 * <A, B> = 2 * Cosine_Distance`.
    
    * **Conclusion**: On unit-normalized vectors, all three metrics produce **100% mathematically identical relative ranking orders**! 
    Databases use Inner Product because computing a dot product requires zero square roots or divisions, maximizing CPU SIMD throughput.
    """)

with st.expander("Q2: Why does standard Top-K search without a distance threshold cause hallucinations in RAG?"):
    st.markdown("""
    **Answer**:
    Top-K search is an **unconstrained optimization problem**: it returns the K nearest vectors regardless of absolute geometric distance.
    * If a user asks a question about an alien topic (e.g. asking about gardening in a medical database), the database still returns the 3 "closest" medical vectors, even if their distance is 0.85 (effectively random noise).
    * When these irrelevant vectors are injected into the LLM context prompt, the LLM attempts to synthesize an answer from irrelevant facts, generating catastrophic hallucinations.
    * Enforcing a strict **distance cutoff threshold** stops out-of-domain chunks before they reach the prompt.
    """)

with st.expander("Q3: How does a Bounded Min-Heap achieve O(N log K) complexity during vector retrieval?"):
    st.markdown("""
    **Answer**:
    Instead of sorting all `N` candidate distances (`O(N log N)`):
    * The search maintains a heap of maximum size `K`.
    * The root of the heap always stores the **current worst distance** among the top-K candidates seen so far.
    * For each of the `N` vectors evaluated:
      * If distance `d` is worse than the root, it is rejected in `O(1)` time.
      * If `d` is better, popping the root and inserting `d` takes `O(log K)` time.
    * Because `K` is small (typically 3 to 10), `log K` is negligible, allowing the search to evaluate tens of thousands of vectors in sub-millisecond time.
    """)

with st.expander("Q4: How should confidence thresholds be calibrated when migrating between different embedding models?"):
    st.markdown("""
    **Answer**:
    **Never copy distance thresholds across different embedding models!**
    * Different models (e.g. `all-MiniLM-L6-v2` vs `text-embedding-3-small` vs `bge-large`) possess different vector manifold dimensionalities, anisotropies, and geometric score distributions.
    * A distance of `0.20` might represent a strong 90% confidence match in OpenAI `text-embedding-3`, but represent an average match in BGE.
    * **Calibration Protocol**:
      1. Run a validation benchmark of 500 labeled `(query, ground_truth_doc)` pairs.
      2. Plot the precision-recall curve across varying distance cutoffs.
      3. Select the cutoff threshold that maximizes F1 score or satisfies your target precision SLA (e.g. 95% precision).
    """)

with st.expander("Q5: What is the 'Curse of Dimensionality' and how does it impact vector similarity search?"):
    st.markdown("""
    **Answer**:
    As dimensionality `D` increases (e.g., from 3 dims to 1,536 dims):
    1. **Distance Concentration**: The ratio of the distance to the nearest neighbor versus the distance to the farthest neighbor approaches 1.0 (`(d_max - d_min) / d_min -> 0`). Points appear almost equidistant from one another.
    2. **Hypersphere Volume**: The volume of high-dimensional space grows exponentially (`2^D`), making exhaustive grid search impossible.
    3. **Solution**: Modern embedding models combat this by training on contrastive loss functions that force semantic points onto dense, non-uniform conceptual sub-manifolds, while graph indices like HNSW exploit small-world cluster structures.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Hardcoding Distance Cutoffs Across Different Models**  
    Copying `threshold = 0.35` when switching from OpenAI to BGE causes retrieval recall to collapse. Calibrate thresholds empirically for every model version.
    """)
    st.warning("""
    **Trap 2: Using Euclidean Distance on Unnormalized Embeddings**  
    If vectors are not normalized to unit length, Euclidean distance penalizes long documents simply because they have larger coordinate norms. Always normalize vectors for text search.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Omitting Zero-Result Fallbacks in RAG**  
    When a distance threshold prunes all chunks, failing to provide an explicit fallback message causes downstream code to crash with `IndexError` on empty lists. Always handle empty results gracefully.
    """)
    st.warning("""
    **Trap 4: Requesting Unnecessarily Large K**  
    Requesting `k = 50` when the LLM only reads 3 chunks increases priority queue overhead, network serialization time, and memory consumption. Set K to what you need.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Cosine Distance?"):
        st.markdown("""
        **Cosine Distance**:  
        `1.0 - Cosine_Similarity`. Measures the angular separation between vectors regardless of magnitude (bounded in 0.0 to 2.0).
        """)
    with st.expander("🗂️ Flashcard: How does a Bounded Min-Heap work?"):
        st.markdown("""
        **Heap Complexity**:  
        Tracks the top-K items in `O(N log K)` time by comparing new candidates against the current worst item at the heap root.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why enforce a distance threshold?"):
        st.markdown("""
        **Confidence Gate**:  
        Prunes out-of-domain noise (e.g. distance > 0.35) before it reaches the prompt, completely preventing hallucinations.
        """)
    with st.expander("🗂️ Flashcard: When are Cosine and Dot Product identical?"):
        st.markdown("""
        **Unit Normalization**:  
        When `||v|| = 1.0`, Cosine Similarity is mathematically identical to Inner Product, enabling fast hardware SIMD dot products.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the geometric differences between Cosine, L2, and Inner Product distance spaces.", value=True)
chk2 = st.checkbox("I know why unit normalization allows replacing Cosine Distance with fast Inner Products.", value=True)
chk3 = st.checkbox("I understand how Bounded Priority Queues achieve O(N log K) retrieval complexity.", value=True)
chk4 = st.checkbox("I can calibrate mathematical confidence thresholds to filter out-of-domain queries.", value=True)
chk5 = st.checkbox("I know how to implement zero-result fallback guardrails in production RAG systems.", value=True)
chk6 = st.checkbox("I understand the Curse of Dimensionality and how contrastive training mitigates distance concentration.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Similarity Retrieval and are ready for Chapter 16.2: Maximal Marginal Relevance (MMR)!")
