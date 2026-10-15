import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="16.2 MMR - Vector DB Curriculum",
    page_icon="🔀",
    layout="wide"
)

# Title & Overview
st.title("16.2 Maximal Marginal Relevance (MMR): Diversity-Aware Retrieval & Redundancy Reduction")
st.caption("Chapter 16: Advanced Retrieval Techniques | Domain: 4. Vector Database")
st.markdown("""
In enterprise retrieval, high cosine similarity does not guarantee high information utility. 
When search algorithms optimize exclusively for similarity, they fall victim to the **Redundancy Pathology**: 
returning multiple passages that repeat the exact same underlying facts.

First introduced by Carbonell & Goldstein (1998) in information retrieval research, **Maximal Marginal Relevance (MMR)** 
transforms search from a pure distance ranking into a **multi-objective optimization problem**:
* **Objective 1 (Relevance)**: Maximize alignment between candidate documents and the user query.
* **Objective 2 (Diversity)**: Minimize redundancy between candidate documents and previously selected results.

By penalizing information overlap, MMR ensures that every retrieved passage introduces **novel, complementary context**, 
maximizing the knowledge density supplied to the LLM's finite context window.

In this lesson, we break down the algorithmic mechanics of MMR, analyze the trade-off spectrum governed by `lambda_mult`, 
implement MMR in clean NumPy, and experiment with an interactive Diversity Optimizer Workbench.
""")

st.divider()

# Section 1: The Redundancy Pathology & Algorithmic Foundation
st.header("1. The Redundancy Pathology & Algorithmic Foundation", divider="rainbow")

col_path1, col_path2 = col_path1, col_path2 = st.columns([1.1, 0.9])

with col_path1:
    st.markdown("""
    ### Why Pure Similarity Fails in Production
    Consider an engineer querying: *"How do we handle Kubernetes ingress failover?"*
    * In a naive vector database, the top 3 nearest vectors might be:
      1. *"Section 1.1: When ingress nodes fail, DNS failover reroutes traffic to Region B."* (Similarity: 0.94)
      2. *"Section 1.2: Ingress node failures automatically trigger DNS rerouting to Region B."* (Similarity: 0.93)
      3. *"Executive Summary: If ingress fails, DNS routes to Region B."* (Similarity: 0.91)
    * **Result**: The LLM learns about DNS failover 3 times, but receives **zero information** about health check intervals, BGP routing, or database connection draining!
    * **The MMR Solution**: MMR forces the retriever to reject Chunk 2 (because it is 95% redundant with Chunk 1) and instead surface Chunk 4 (*"Health checks ping endpoints every 5 seconds"*), providing a comprehensive, multi-faceted answer.
    """)

with col_path2:
    st.markdown("### MMR Selection Step-by-Step Flow")
    mmr_step_dot = """
    digraph MMRAlgorithm {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Init [label="1. Fetch Candidate Pool R\\n(size = fetch_k, e.g. 20)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        PickFirst [label="2. Pick First Item S_0\\narg max Sim(D, Q)", fillcolor="#D1FAE5", color="#059669"];

        Loop [label="3. For each remaining unpicked D_i in R:\\nCompute MMR Score", fillcolor="#FEF3C7", color="#D97706"];

        Formula [label="MMR Score =\\nlambda * Sim(D_i, Q) - (1-lambda) * max Sim(D_i, D_j in S)", fillcolor="#EDE9FE", style="filled,bold"];

        SelectBest [label="4. Add highest scoring D_i to S", fillcolor="#D1FAE5", color="#059669"];

        CheckDone [label="5. Is |S| == k?", shape=diamond, fillcolor="#FEE2E2"];

        Done [label="Return Diverse Set S", fillcolor="#A7F3D0", style="filled,bold"];

        Init -> PickFirst;
        PickFirst -> Loop;
        Loop -> Formula;
        Formula -> SelectBest;
        SelectBest -> CheckDone;
        CheckDone -> Loop [label="No"];
        CheckDone -> Done [label="Yes"];
    }
    """
    st.graphviz_chart(mmr_step_dot, use_container_width=False)

st.divider()

# Section 2: Mathematical Derivation & Vector Implementation
st.header("2. Mathematical Derivation & Vector Implementation", divider="rainbow")

col_code_m1, col_code_m2 = col_code_m1, col_code_m2 = st.columns([1, 1])

with col_code_m1:
    st.markdown("""
    ### The MMR Mathematical Objective
    At each step `t`, the next document is selected according to:
    
    ```text
    D_{next} = argmax_{D_i in R \\ S} [ lambda * Sim(D_i, Q) - (1 - lambda) * max_{D_j in S} Sim(D_i, D_j) ]
    ```
    
    * **`R`**: Unranked candidate set of size `fetch_k`.
    * **`S`**: Set of already selected documents (initially empty).
    * **`Sim(D_i, Q)`**: Marginal relevance to the user question.
    * **`max_{D_j in S} Sim(D_i, D_j)`**: Maximum redundancy with any document already chosen.
    * **`lambda` (`0.0 <= lambda <= 1.0`)**: The balance knob between relevance and diversity.
    """)

with col_code_m2:
    st.markdown("### Pure NumPy Implementation")
    st.code("""import numpy as np
from typing import List

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    lambda_mult: float = 0.7,
    k: int = 3
) -> List[int]:
    \"\"\"Returns indices of the top-k diverse documents.\"\"\"
    # 1. Cosine similarity of all candidates to query
    sim_to_query = np.dot(candidate_embeddings, query_embedding)
    
    # 2. Select initial best document
    selected = [int(np.argmax(sim_to_query))]
    unselected = list(set(range(len(candidate_embeddings))) - set(selected))
    
    # 3. Iteratively select remaining k-1 documents
    while len(selected) < k and unselected:
        best_score = -float("inf")
        best_idx = None
        
        for idx in unselected:
            relevance = sim_to_query[idx]
            # Max similarity to any already selected document
            redundancy = max(np.dot(candidate_embeddings[idx], candidate_embeddings[s]) for s in selected)
            
            score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * redundancy)
            if score > best_score:
                best_score = score
                best_idx = idx
                
        selected.append(best_idx)
        unselected.remove(best_idx)
        
    return selected
""", language="python")

st.divider()

# Section 3: The `lambda_mult` Spectrum Across Enterprise Domains
st.header("3. The `lambda_mult` Spectrum Across Enterprise Domains", divider="rainbow")

st.markdown("""
How should you set `lambda_mult` in your production RAG pipeline? The optimal value depends on the underlying domain:
""")

LAMBDA_DOMAIN_GUIDE = [
    {
        "Domain / Use Case": "Legal Contract Discovery & M&A Due Diligence",
        "Recommended Lambda": "0.4 - 0.5 (High Diversity)",
        "Rationale": "Contracts repeat clauses across amendments. High diversity surfaces indemnity, termination, and governance clauses rather than 5 copies of the indemnity paragraph."
    },
    {
        "Domain / Use Case": "Enterprise IT Helpdesk & Customer Support",
        "Recommended Lambda": "0.65 - 0.75 (Balanced)",
        "Rationale": "Ensures the primary answer is directly relevant while pulling in troubleshooting steps or prerequisite runbooks."
    },
    {
        "Domain / Use Case": "Medical Diagnosis & Clinical Guidelines",
        "Recommended Lambda": "0.80 - 0.90 (High Relevance)",
        "Rationale": "Clinical answers require precision. Penalizing similarity too heavily risks replacing critical dosage instructions with unrelated contraindications."
    },
    {
        "Domain / Use Case": "Code Repository Search & Semantic Grep",
        "Recommended Lambda": "0.55 - 0.65 (Medium Diversity)",
        "Rationale": "Retrieves the function definition, unit test example, and import configuration rather than 3 identical docstring references."
    }
]

st.dataframe(pd.DataFrame(LAMBDA_DOMAIN_GUIDE), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Interactive MMR Diversity Optimizer
st.header("4. Interactive Workbench: Interactive MMR Diversity Optimizer", divider="rainbow")

st.markdown("""
Experiment with MMR in a simulated multi-cluster embedding space. Adjust **`lambda_mult`** in real time 
to observe how the algorithm penalizes intra-cluster redundancy and recruits diverse knowledge chunks.
""")

# Simulated Multi-Topic Candidate Corpus
SIM_CANDIDATES = [
    {"id": "DOC-1A", "topic": "Cluster 1: Ingress Gateway mTLS", "text": "Envoy proxy enforces mTLS across all external edge ingress controllers.", "q_sim": 0.95, "c_sim": 0.92},
    {"id": "DOC-1B", "topic": "Cluster 1: Ingress Gateway mTLS", "text": "All edge ingress connections negotiate TLS 1.3 with client cert validation.", "q_sim": 0.93, "c_sim": 0.92},
    {"id": "DOC-1C", "topic": "Cluster 1: Ingress Gateway mTLS", "text": "Mutual TLS is mandatory for edge Envoy proxy ingress traffic.", "q_sim": 0.90, "c_sim": 0.90},
    {"id": "DOC-2A", "topic": "Cluster 2: SPIFFE Cert Rotation", "text": "SPIRE agents rotate pod X.509 certificates automatically every 60 minutes.", "q_sim": 0.84, "c_sim": 0.25},
    {"id": "DOC-2B", "topic": "Cluster 2: SPIFFE Cert Rotation", "text": "Automated certificate issuance uses local SPIFFE Workload APIs.", "q_sim": 0.80, "c_sim": 0.25},
    {"id": "DOC-3A", "topic": "Cluster 3: TLS Handshake Debugging", "text": "Use envoy admin endpoint /stats and grep ssl.handshake_error to debug drops.", "q_sim": 0.76, "c_sim": 0.20},
    {"id": "DOC-3B", "topic": "Cluster 3: TLS Handshake Debugging", "text": "Common TLS drops stem from expired CA root trust bundles.", "q_sim": 0.72, "c_sim": 0.20}
]

col_opt_ctl, col_opt_res = st.columns([1, 1.2])

with col_opt_ctl:
    st.subheader("1. MMR Tuning Knobs")
    k_target = st.slider("Target Number of Retrieved Chunks (k):", min_value=2, max_value=4, value=3)
    user_lambda = st.slider(
        "Diversity Weight (`lambda_mult`):",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="1.0 = Pure Query Similarity; 0.0 = Maximal Diversity."
    )
    
    st.info(f"""
    **Current Balance Profile**:
    * **Relevance Weight**: `{user_lambda * 100:.0f}%`
    * **Diversity Penalty**: `{(1.0 - user_lambda) * 100:.0f}%`
    """)

with col_opt_res:
    st.subheader("2. Real-Time MMR Execution Trace")
    
    # Execute MMR simulation
    # Step 1: pick DOC-1A
    chosen = [SIM_CANDIDATES[0]]
    pool = SIM_CANDIDATES[1:]
    
    trace_log = [f"Step 1: Picked DOC-1A (Initial Highest Query Sim: {SIM_CANDIDATES[0]['q_sim']:.2f})"]
    
    while len(chosen) < k_target and pool:
        step_num = len(chosen) + 1
        best_cand = None
        best_mmr_score = -999.0
        
        for cand in pool:
            rel = cand["q_sim"]
            # Redundancy: high if same cluster as already picked
            max_redundancy = 0.0
            for picked in chosen:
                if cand["topic"] == picked["topic"]:
                    max_redundancy = max(max_redundancy, cand["c_sim"])
                else:
                    max_redundancy = max(max_redundancy, 0.22)
            
            score = (user_lambda * rel) - ((1.0 - user_lambda) * max_redundancy)
            if score > best_mmr_score:
                best_mmr_score = score
                best_cand = cand
                
        chosen.append(best_cand)
        pool.remove(best_cand)
        trace_log.append(f"Step {step_num}: Picked {best_cand['id']} ({best_cand['topic'].split(':')[1].strip()}) — MMR Score: {best_mmr_score:.3f}")

    for t in trace_log:
        st.write(f"• **{t}**")

st.divider()

col_comp_std, col_comp_mmr = st.columns([1, 1])

with col_comp_std:
    st.subheader(f"Standard Similarity Top-{k_target}")
    std_picks = SIM_CANDIDATES[:k_target]
    for idx, p in enumerate(std_picks):
        with st.expander(f"Chunk {idx+1}: {p['id']} ({p['topic']})", expanded=True):
            st.markdown(f"*{p['text']}*")
            st.caption(f"Query Similarity: `{p['q_sim']}`")
    
    topics_in_std = len(set(p['topic'] for p in std_picks))
    st.metric("Unique Topics Covered", f"{topics_in_std} of 3 Clusters", delta=f"{topics_in_std - 3} Deficit" if topics_in_std < 3 else "Complete")

with col_comp_mmr:
    st.subheader(f"MMR Diverse Top-{k_target}")
    for idx, p in enumerate(chosen):
        with st.expander(f"Chunk {idx+1}: {p['id']} ({p['topic']})", expanded=True):
            st.markdown(f"*{p['text']}*")
            st.caption(f"Query Similarity: `{p['q_sim']}`")
            
    topics_in_mmr = len(set(p['topic'] for p in chosen))
    st.metric("Unique Topics Covered", f"{topics_in_mmr} of 3 Clusters", delta="Diverse Coverage" if topics_in_mmr > topics_in_std else "Identical")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the computational time complexity of the MMR re-ranking algorithm?"):
    st.markdown("""
    **Answer**:
    Let `fetch_k` be the size of the initial candidate pool (typically 20 to 50), `k` be the final number of retrieved documents (typically 3 to 5), and `D` be the vector dimension (e.g. 1536).
    * Initial candidate retrieval from HNSW takes `O(log N)` or ~2ms.
    * In step 1, finding the top candidate takes `O(fetch_k)`.
    * For each of the remaining `k - 1` iterations, MMR evaluates each unselected document against all currently selected documents.
    * This requires computing at most `O(k * fetch_k * D)` floating-point operations.
    * Because `fetch_k = 20` and `k = 4`, total re-ranking takes **under 0.3 milliseconds** in pure NumPy, adding virtually zero latency to the pipeline.
    """)

with st.expander("Q2: Why does MMR select the first document purely based on query similarity?"):
    st.markdown("""
    **Answer**:
    At step `t = 1`, the set of already-selected documents `S` is empty (`S = []`).
    * In the objective equation `lambda * Sim(D, Q) - (1 - lambda) * max Sim(D, D_j in S)`, the second redundancy term is zero because there are no previously selected documents to be redundant with.
    * Therefore, the score simplifies to `lambda * Sim(D, Q)`.
    * Maximizing this term guarantees that the **single most relevant, authoritative document** always anchors the retrieved context. Diversity penalties only begin on step 2.
    """)

with st.expander("Q3: How does MMR compare to Cross-Encoder Reranking?"):
    st.markdown("""
    **Answer**:
    * **MMR**: Evaluates already-computed bi-encoder embeddings to **maximize diversity and penalize redundancy**. It cannot re-evaluate subtle query-document token interactions; it only looks at geometric distances. It runs in <0.3ms.
    * **Cross-Encoder Reranker**: Passes the raw query and candidate text through full transformer cross-attention layers to **maximize precision**. It does NOT penalize redundancy (two duplicate high-relevance chunks will both receive high scores). It takes 15-30ms.
    * **Production Best Practice**: Combine both! Use Cross-Encoder reranking first to pick the top 20 most accurate candidates, then apply MMR to select the final 4 most diverse chunks.
    """)

with st.expander("Q4: Can MMR hurt RAG generation quality if tuned incorrectly?"):
    st.markdown("""
    **Answer**:
    **Yes**. If `lambda_mult` is set too low (e.g. `lambda < 0.3`):
    * The algorithm penalizes similarity so aggressively that it actively searches for documents that have nothing in common with Chunk 1.
    * The retriever will pick Chunk 1 (which answers the question), and then pick Chunk 2 and Chunk 3 from completely unrelated topics simply because their vectors are far away.
    * The LLM context prompt becomes contaminated with irrelevant noise, degrading generation fidelity. Always keep `lambda >= 0.5`.
    """)

with st.expander("Q5: What is the difference between MMR in ChromaDB versus MMR in LangChain?"):
    st.markdown("""
    **Answer**:
    * **ChromaDB Native**: ChromaDB does not have a built-in C++ MMR HNSW traversal routine in its core C++ engine; it returns standard top-K nearest neighbors.
    * **LangChain MMR**: LangChain executes MMR **in Python memory** on top of ChromaDB: it calls `collection.query(n_results=fetch_k)`, pulls the embeddings into NumPy, and runs the MMR loop in the Python application layer.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Setting `lambda_mult = 0.0`**  
    Setting lambda to 0 causes the algorithm to completely ignore the user's question after Step 1, selecting the most divergent noise in the database. Keep lambda between 0.5 and 0.8.
    """)
    st.warning("""
    **Trap 2: Setting `fetch_k` Too Small (e.g. `fetch_k = k`)**  
    If `fetch_k = 3` and `k = 3`, MMR cannot introduce diversity because there are zero unpicked candidate alternatives to choose from. Ensure `fetch_k >= 4 * k`.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Forgetting to Request Embeddings for MMR**  
    Running MMR requires comparing candidate embeddings against one another. If querying ChromaDB directly, you must include `include=['documents', 'embeddings', 'metadatas']` in the query call.
    """)
    st.warning("""
    **Trap 4: Applying MMR on Small Collections (< 20 chunks)**  
    If a collection only contains 15 total chunks, running MMR with `fetch_k = 20` fetches the entire database, which is unnecessary overhead. Standard Top-K is sufficient for tiny collections.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the MMR objective?"):
        st.markdown("""
        **Dual Objective**:  
        `lambda * Sim(Doc, Query) - (1 - lambda) * max Sim(Doc, Selected)`.
        """)
    with st.expander("🗂️ Flashcard: What happens at step 1 of MMR?"):
        st.markdown("""
        **Anchor Selection**:  
        Picks the candidate with highest query similarity because redundancy penalty is zero.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is the recommended lambda_mult?"):
        st.markdown("""
        **0.6 to 0.7**:  
        Industry sweet spot: 60-70% query relevance, 30-40% diversity penalty.
        """)
    with st.expander("🗂️ Flashcard: How does MMR differ from Cross-Encoders?"):
        st.markdown("""
        **Diversity vs Precision**:  
        MMR penalizes redundancy; Cross-Encoders maximize token-level semantic precision.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the Redundancy Pathology and why pure similarity search wastes prompt tokens.", value=True)
chk2 = st.checkbox("I know how the MMR mathematical formula balances query relevance against candidate redundancy.", value=True)
chk3 = st.checkbox("I can implement the MMR selection loop in pure NumPy and vectorize candidate scoring.", value=True)
chk4 = st.checkbox("I know how to tune `lambda_mult` across legal, medical, and technical support domains.", value=True)
chk5 = st.checkbox("I understand the computational complexity of MMR and why it adds <0.5ms of latency.", value=True)
chk6 = st.checkbox("I can combine Cross-Encoder reranking and MMR into a two-stage high-precision retrieval pipeline.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered MMR and are ready for Chapter 16.3: Hybrid Retrieval (Dense + Sparse)!")
