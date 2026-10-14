import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="19.1 What Makes a Good Embedding - Embeddings Curriculum",
    page_icon="⭐",
    layout="wide"
)

# Title & Overview
st.title("19.1 What Makes a Good Embedding: Semantic Fidelity, Separability, & Isotropy")
st.caption("Chapter 19: Embedding Quality | Domain: 3. Vector Embeddings")
st.markdown("""
In vector search and RAG architectures, your system's intelligence is fundamentally bounded by the **mathematical quality of its embeddings**.
If an embedding model produces degraded vector coordinates, no amount of prompt engineering or vector database tuning can salvage the application.

What distinguishes a world-class embedding model from a mediocre one?
* **Semantic Fidelity**: True conceptual meaning is encoded rather than superficial word overlap.
* **Cluster Compactness**: Paraphrases and synonyms are tightly clustered together.
* **Inter-Cluster Separability**: Unrelated topics are mapped far apart, eliminating false positives.
* **Domain Relevance**: The model understands industry-specific acronyms, jargon, and syntax.
* **Isotropy & Uniformity**: Vectors occupy the high-dimensional hypersphere uniformly, avoiding the **Dimensional Collapse** and **Hubness** traps.

This lesson explores the mathematical and empirical standards that define high-quality embeddings.
""")

st.divider()

# Section 1: The 5 Hallmarks of High-Quality Embeddings
st.header("1. The 5 Hallmarks of Embedding Quality & The Isotropy Problem", divider="rainbow")

col_hal1, col_hal2 = st.columns([1, 1])

with col_hal1:
    st.markdown("""
    ### The 5 Core Quality Pillars
    * **1. Semantic Representation Fidelity**:
      * The model captures underlying intent, causality, and sentiment rather than literal token matches.
      * Example: *"The engine stalled"* and *"The motor stopped working"* map to virtually identical vectors despite zero shared words.
    * **2. High Intra-Cluster Compactness**:
      * Paraphrases of the same concept achieve high cosine similarity ($> 0.85$).
    * **3. High Inter-Cluster Separability**:
      * Unrelated concepts (e.g. quantum physics vs. corporate tax law) exhibit low baseline similarity ($< 0.20$), maximizing contrast.
    * **4. Domain Idiom Alignment**:
      * Understands technical codes (`CVE-2024-3094`), legal clauses, and medical abbreviations without hallucinating semantic connections.
    * **5. Latent Space Isotropy (Uniformity)**:
      * Vectors are distributed evenly in all directions across the hypersphere.
      * Degraded models suffer from **Anisotropy / Dimensional Collapse**: vectors crowd into a narrow cone where *all* pairs share an artificially high baseline similarity of 0.80+, destroying search contrast!
    """)

with col_hal2:
    st.markdown("### Isotropic Hypersphere vs. Anisotropic Cone Collapse")
    iso_dot = """
    digraph SpaceIsotropy {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_bad {
            label="Anisotropic Collapsed Space (Poor Quality)"; style="dashed"; color="#EF4444";
            Cone [label="Narrow Cone Distribution\\nAll vectors clustered together\\nBaseline cosine = 0.85+\\nSevere 'Hubness' problem", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];
        }

        subgraph cluster_good {
            label="Isotropic Uniform Space (High Quality)"; style="dashed"; color="#10B981";
            Uniform [label="Uniform Hypersphere Coverage\\nDistinct concepts separated\\nBaseline cosine ~ 0.0\\nMaximum discriminative contrast", fillcolor="#ECFDF5", color="#059669", style="filled,bold"];
        }

        EvalQ [label="Query Evaluation: Quality Diagnostic", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        EvalQ -> Cone [label="Low discriminative power\\nHigh false positives"];
        EvalQ -> Uniform [label="Sharp ranking recall\\nRobust contrast"];
    }
    """
    st.graphviz_chart(iso_dot, use_container_width=False)

st.markdown("### Embedding Space Failure Modes")
st.markdown("""
| Failure Mode | Mathematical Symptom | Practical RAG Consequence | How to Diagnose |
| :--- | :--- | :--- | :--- |
| **Dimensional Collapse** | Singular values decay rapidly; few active dimensions. | All documents look similar; cosine scores cluster between 0.85 and 0.95. | Compute SVD eigenvalues of embedding matrix. |
| **Hubness Problem** | A small subset of 'hub' vectors are nearest neighbors to almost everything. | Irrelevant hub documents appear in every single user search result. | Calculate in-degree distribution of nearest neighbor graphs. |
| **Lexical Overfitting** | High similarity strictly when exact keywords match. | Fails on synonyms, paraphrases, and cross-lingual queries. | Benchmark on semantic textual similarity (STS) datasets. |
| **Domain Blindness** | Treats rare medical/code tokens as unknown `[UNK]` or noise. | Complete failure to retrieve specialized technical knowledge. | Test on domain-specific MTEB benchmarks. |
""")

st.divider()

# Section 2: Production Code: Embedding Quality Diagnostic Suite
st.header("2. Production Code: Quality & Isotropy Diagnostic Suite", divider="rainbow")

st.markdown("""
The production utility below evaluates embedding quality mathematically: computing intra-cluster compactness, inter-cluster separation, and singular value spectrum to quantify space isotropy.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple

class EmbeddingQualityDiagnostics:
    """
    Analyzes mathematical properties of an embedding space:
    - Intra-cluster compactness vs. Inter-cluster separation
    - Isotropy score via Singular Value Decomposition (SVD)
    """
    @staticmethod
    def compute_compactness_ratio(
        cluster_a: np.ndarray,
        cluster_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates average intra-cluster similarity vs inter-cluster similarity.
        Higher ratio indicates superior discriminative clustering.
        """
        # Normalize vectors to unit sphere
        norm_a = cluster_a / np.linalg.norm(cluster_a, axis=1, keepdims=True)
        norm_b = cluster_b / np.linalg.norm(cluster_b, axis=1, keepdims=True)

        # 1. Intra-cluster similarity (Within Cluster A)
        sim_within_a = np.mean(np.dot(norm_a, norm_a.T))

        # 2. Intra-cluster similarity (Within Cluster B)
        sim_within_b = np.mean(np.dot(norm_b, norm_b.T))

        # 3. Inter-cluster similarity (Between Cluster A and Cluster B)
        sim_between = np.mean(np.dot(norm_a, norm_b.T))

        avg_intra = (sim_within_a + sim_within_b) / 2.0
        contrast_ratio = avg_intra / max(1e-6, abs(sim_between))

        return {
            "avg_intra_cluster_sim": float(avg_intra),
            "inter_cluster_sim": float(sim_between),
            "contrast_ratio": float(contrast_ratio)
        }

    @staticmethod
    def compute_isotropy_score(embeddings: np.ndarray) -> float:
        """
        Computes isotropy score using SVD singular value variance.
        A score near 1.0 indicates perfect uniform spherical distribution.
        A score near 0.0 indicates severe dimensional collapse.
        """
        norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Center embeddings
        centered = norm_emb - np.mean(norm_emb, axis=0)
        
        # Singular Value Decomposition
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        
        # Energy ratio of top singular value vs mean singular value
        isotropy = float(np.min(s) / max(1e-12, np.max(s)))
        return min(1.0, max(0.0, isotropy))
''', language="python")

st.divider()

# Section 3: Interactive Quality & Isotropy Analyzer
st.header("3. Interactive Workbench: Embedding Quality & Isotropy Studio", divider="rainbow")

st.markdown("""
Compare three simulated embedding models representing different mathematical quality profiles:
1. **Model A (High-Quality Isotropic)**: Balanced spherical coverage with sharp semantic cluster boundaries.
2. **Model B (Anisotropic Collapsed Cone)**: Squeezed into a narrow cone with high baseline similarity across all topics.
3. **Model C (Lexical / High-Noise)**: Disorganized representations failing to cluster related concepts.
""")

class QualityModelSimulator:
    def __init__(self, mode: str, dim: int = 32):
        self.mode = mode
        self.dim = dim

    def embed_texts(self, texts: List[str], topic_ids: List[int]) -> np.ndarray:
        vectors = []
        for text, tid in zip(texts, topic_ids):
            np.random.seed(abs(hash(text)) % (2**31 - 1))
            base = np.random.randn(self.dim).astype(np.float32)

            if self.mode == "High-Quality Isotropic":
                # Distinct orthogonal centers with compact noise
                center = np.zeros(self.dim, dtype=np.float32)
                center[tid * 8 : (tid + 1) * 8] = 3.5
                v = center + base * 0.4

            elif self.mode == "Anisotropic Collapsed Cone":
                # Large common bias vector compressing all points into narrow cone
                cone_bias = np.ones(self.dim, dtype=np.float32) * 5.0
                topic_shift = np.zeros(self.dim, dtype=np.float32)
                topic_shift[tid] = 0.5
                v = cone_bias + topic_shift + base * 0.2

            else: # High Noise / Low Quality
                # Weak topic signal dominated by random noise
                topic_shift = np.zeros(self.dim, dtype=np.float32)
                topic_shift[tid] = 0.3
                v = topic_shift + base * 2.0

            # L2 normalize
            v = v / np.maximum(np.linalg.norm(v), 1e-12)
            vectors.append(v)

        return np.array(vectors)

benchmark_corpus = [
    # Topic 0: Healthcare / Medicine
    ("Patients diagnosed with acute hypertension require cardiovascular monitoring.", 0, "Healthcare"),
    ("Clinical dosages of beta-blockers reduce arterial blood pressure in trials.", 0, "Healthcare"),
    ("Oncologists evaluate malignant biopsy tissue using histological staining.", 0, "Healthcare"),
    # Topic 1: Cloud & DevOps
    ("Kubernetes clusters schedule pods across worker nodes with horizontal autoscaling.", 1, "DevOps"),
    ("Docker container images package application binaries with runtime dependencies.", 1, "DevOps"),
    ("Terraform provisions infrastructure as code across multiple cloud providers.", 1, "DevOps"),
    # Topic 2: Finance & Banking
    ("Federal interest rate hikes influence mortgage debt yields and corporate bonds.", 2, "Finance"),
    ("Commercial banks calculate capital adequacy ratios under Basel III regulations.", 2, "Finance"),
    ("Equity index funds offer diversified market exposure with minimal expense ratios.", 2, "Finance")
]

col_m_ctrl, col_m_view = st.columns([1, 1])

with col_m_ctrl:
    st.subheader("1. Model Quality Profile")
    chosen_profile = st.selectbox(
        "Select Simulated Model Architecture:",
        ["High-Quality Isotropic", "Anisotropic Collapsed Cone", "Lexical / High-Noise"],
        index=0
    )

    st.write(f"Evaluating corpus of **{len(benchmark_corpus)}** documents across 3 distinct semantic domains: **Healthcare**, **DevOps**, and **Finance**.")
    df_corp = pd.DataFrame([{"Domain": b[2], "Text": b[0]} for b in benchmark_corpus])
    st.dataframe(df_corp, use_container_width=True)

with col_m_view:
    st.subheader(f"2. Mathematical Quality Evaluation: {chosen_profile}")

    sim_engine = QualityModelSimulator(chosen_profile)
    b_texts = [b[0] for b in benchmark_corpus]
    b_topics = [b[1] for b in benchmark_corpus]
    
    vecs = sim_engine.embed_texts(b_texts, b_topics)

    # Compute pairwise cosine similarity matrix
    sim_matrix = np.dot(vecs, vecs.T)

    # Calculate Intra vs Inter cluster metrics
    intra_sims = []
    inter_sims = []
    for i in range(len(b_topics)):
        for j in range(i + 1, len(b_topics)):
            s = float(sim_matrix[i, j])
            if b_topics[i] == b_topics[j]:
                intra_sims.append(s)
            else:
                inter_sims.append(s)

    avg_intra = float(np.mean(intra_sims))
    avg_inter = float(np.mean(inter_sims))
    contrast_spread = avg_intra - avg_inter

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Intra-Cluster Sim", f"{avg_intra:.3f}", help="Average similarity within the same topic (higher is better)")
    m_col2.metric("Inter-Cluster Sim", f"{avg_inter:.3f}", help="Average similarity between unrelated topics (lower is better)")
    m_col3.metric("Contrast Spread", f"{contrast_spread:.3f}", help="Spread = Intra - Inter (Higher spread indicates cleaner search)")

    if chosen_profile == "High-Quality Isotropic":
        st.success("✅ **Ideal Distribution**: High intra-topic similarity with near-zero cross-topic noise. Maximum search contrast!")
    elif chosen_profile == "Anisotropic Collapsed Cone":
        st.error("❌ **Severe Dimensional Collapse**: Notice Inter-Cluster similarity is above 0.85! Unrelated topics look nearly identical to the vector index.")
    else:
        st.warning("⚠️ **High-Noise Model**: Weak intra-topic clustering. Semantically related documents fail to align closely.")

    # Show 2D Projection (First 2 PCA components simulation)
    st.markdown("#### 2D Latent Space Projection (PCA)")
    centered_vecs = vecs - np.mean(vecs, axis=0)
    _, _, vt = np.linalg.svd(centered_vecs, full_matrices=False)
    proj_2d = np.dot(centered_vecs, vt[:2].T)

    df_proj = pd.DataFrame({
        "PCA_1": proj_2d[:, 0],
        "PCA_2": proj_2d[:, 1],
        "Domain": [b[2] for b in benchmark_corpus]
    })
    
    st.scatter_chart(df_proj, x="PCA_1", y="PCA_2", color="Domain")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Quality Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is 'Dimensional Collapse' (Anisotropy) in embedding models and how does it harm retrieval?**
    * **Answer**: Anisotropy occurs when an embedding model maps all vectors into a narrow, low-dimensional cone rather than utilizing the full hypersphere uniformly.
    * **Consequence**: The baseline cosine similarity between *any two random sentences* is artificially inflated (e.g. 0.85+). Because all documents score within a narrow band [0.85, 0.95], the signal-to-noise ratio collapses, causing frequent false positives and ranking instability.

    **Q2: What is the 'Hubness Problem' in high-dimensional vector spaces?**
    * **Answer**: In high-dimensional spaces ($d > 500$), mathematical artifacts known as 'hubs' frequently emerge: a small subset of vectors that appear as the nearest neighbors to a disproportionately large number of query vectors, regardless of topic.
    * **Impact**: In RAG, a 'hub' document (often a generic introductory paragraph or boilerplate text) is returned on almost every search, polluting prompts with irrelevant context.

    **Q3: How do modern contrastive loss functions (e.g. InfoNCE) ensure embedding space uniformity?**
    * **Answer**: Modern training objectives combine two forces:
      1. **Alignment (Pull Force)**: Pulls positive query-document pairs close together in latent space.
      2. **Uniformity (Push Force)**: Pushes negative documents away across the entire sphere, encouraging isotropic distribution and preventing vectors from collapsing into a single cluster.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Evaluating Embeddings Purely on Training Loss**:
      * *Issue*: Measuring embedding model quality strictly by contrastive loss on in-domain training pairs.
      * *Consequence*: The model overfits to specific syntactic patterns while suffering latent collapse on out-of-distribution text.
      * *Fix*: Benchmark on diverse standardized benchmarks like MTEB (Massive Text Embedding Benchmark).

    * **Trap 2: Comparing Raw Cosine Scores Across Different Embedding Models**:
      * *Issue*: Assuming a 0.80 cosine similarity in OpenAI `text-embedding-3` means the same thing as 0.80 in `all-MiniLM-L6-v2`.
      * *Consequence*: Ruined threshold filters and unpredictable retrieval behavior.
      * *Fix*: Calibrate similarity thresholds independently for each specific embedding model checkpoint.

    * **Trap 3: Ignoring Non-Normalized Vectors in Inner Product Space**:
      * *Issue*: Using dot product search on embeddings that have not been L2-normalized.
      * *Consequence*: Longer documents with larger vector magnitudes dominate search results regardless of semantic relevance.
      * *Fix*: Always unit-normalize embeddings ($L2\\;\\text{Norm} = 1.0$) before indexing.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Isotropy**
      * *Definition*: The mathematical property of an embedding space where vectors are uniformly distributed in all directions across the hypersphere.
    * **Flashcard 2: Dimensional Collapse**
      * *Definition*: A failure mode where embeddings occupy a low-dimensional subspace or narrow cone, causing artificially inflated baseline similarities.
    * **Flashcard 3: Hubness Problem**
      * *Definition*: The emergence of rogue vectors in high-dimensional spaces that act as false nearest neighbors to a vast number of unrelated queries.
    * **Flashcard 4: Intra- vs Inter-Cluster Contrast**
      * *Definition*: The metric comparing similarity within identical topics against similarity between disparate topics to measure discriminative power.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of What Makes a Good Embedding:

- [ ] I can explain the 5 core hallmarks of high-quality embeddings.
- [ ] I understand the causes and symptoms of Dimensional Collapse and Anisotropy.
- [ ] I can describe the Hubness Problem and its impact on RAG retrieval.
- [ ] I know why Intra-to-Inter cluster contrast spread dictates retrieval precision.
- [ ] I understand how contrastive loss (InfoNCE) enforces latent space uniformity.
""")
