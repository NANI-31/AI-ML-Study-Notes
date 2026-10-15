import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="4.2 Sentence Embeddings - Embeddings Curriculum",
    page_icon="💬",
    layout="wide"
)

# Title & Overview
st.title("4.2 Sentence Embeddings: Sentence-BERT, Contrastive Triplet Objectives, & Dense Semantic Comparison")
st.caption("Chapter 4: Types of Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Averaging static word vectors fails fundamentally when dealing with human language:
`"The dog bit the man"` and `"The man bit the dog"` contain the exact same bag of words, but express completely opposite real-world events.

**Sentence Embeddings** capture the holistic semantic meaning of entire sentences, phrases, and short paragraphs:
* Pioneered by **Sentence-BERT (SBERT)** (Reimers & Gurevych, 2019), sentence encoders use **Siamese and Triplet transformer networks** to map complete sentences to dense vector spaces.
* Rather than predicting masked words (like standard BERT), sentence encoders are explicitly trained using **Contrastive Loss (Multiple Negatives Ranking Loss)** to pull semantically equivalent sentences close together while repelling unrelated ones.
* This allows computing pairwise similarity across millions of sentences in **under 10 milliseconds**, unlocking modern semantic search, clustering, and duplicate detection.

This lesson explores Siamese neural architectures, contrastive triplet training, and production sentence comparison.
""")

st.divider()

# Section 1: Siamese & Triplet Network Architecture
st.header("1. SBERT Architecture: Siamese Networks & Contrastive Objectives", divider="rainbow")

col_sb1, col_sb2 = st.columns([1, 1])

with col_sb1:
    st.markdown("""
    ### Why Standard BERT Fails at Sentence Comparison:
    * **The 65-Hour Cross-Encoder Bottleneck**:
      Finding the most similar pair among 10,000 sentences using standard BERT requires evaluating 50 million pairs through a 12-layer transformer, requiring roughly **65 hours of GPU compute**.
    * **The SBERT Bi-Encoder Breakthrough**:
      SBERT uses a Siamese transformer architecture to encode each sentence **independently into a fixed vector**.
      Comparing 10,000 sentences reduces to 10,000 forward passes plus high-speed dot products, taking **under 5 seconds** (a 50,000x speedup).
    * **Contrastive Triplet Training**:
      Trained on triplets: **Anchor (A), Positive (P), and Negative (N)**.
      Loss penalizes the model unless:
      `distance(A, P) + margin < distance(A, N)`.
    """)

with col_sb2:
    st.markdown("### SBERT Siamese Architecture")
    sbert_dot = """
    digraph SBERT {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        SentA [label="Sentence A\\n'A boy is playing soccer'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        SentB [label="Sentence B\\n'A kid kicks a football'", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        BERT_A [label="BERT Encoder (Tied Weights)", fillcolor="#EDE9FE", color="#7C3AED"];
        BERT_B [label="BERT Encoder (Tied Weights)", fillcolor="#EDE9FE", color="#7C3AED"];

        Pool_A [label="Masked Mean Pooling -> u", fillcolor="#DCFCE7", color="#16A34A"];
        Pool_B [label="Masked Mean Pooling -> v", fillcolor="#DCFCE7", color="#16A34A"];

        Sim [label="Cosine Similarity / Contrastive Objective\\ncos(u, v) ≈ 0.94", fillcolor="#22C55E", fontcolor="white", style="filled,bold"];

        SentA -> BERT_A -> Pool_A -> Sim;
        SentB -> BERT_B -> Pool_B -> Sim;
    }
    """
    st.graphviz_chart(sbert_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Sentence Embedding Comparator in Python", divider="rainbow")
st.markdown("""
The following Python class demonstrates SBERT-style sentence comparison, semantic textual similarity (STS) scoring, and contrastive triplet loss evaluation.
""")

class ProductionSentenceEmbeddingComparator:
    """
    Simulates Sentence-BERT Siamese encoding, semantic textual similarity (STS),
    and contrastive triplet loss distance calculations.
    """
    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
        # Seeded concept anchors
        np.random.seed(42)
        self.anchors = {
            "sports": np.array([0.90, 0.82, 0.10, -0.15, 0.20, 0.40, 0.10, 0.15]),
            "finance": np.array([0.15, 0.12, 0.92, 0.80, -0.10, 0.25, 0.40, 0.20]),
            "animals": np.array([0.20, 0.85, 0.12, -0.10, 0.88, 0.30, -0.15, 0.10]),
            "tech": np.array([0.80, 0.10, 0.20, 0.90, 0.15, 0.75, 0.30, 0.25])
        }
        for k in self.anchors:
            self.anchors[k] /= np.linalg.norm(self.anchors[k])

    def encode_sentence(self, text: str) -> np.ndarray:
        """Simulates contextual sentence encoding based on semantic keywords."""
        vec = np.zeros(self.dimensions)
        t_low = text.lower()
        matched = False

        if any(w in t_low for w in ["soccer", "football", "player", "game", "kicks", "ball", "match"]):
            vec += self.anchors["sports"] * 2.0
            matched = True
        if any(w in t_low for w in ["stock", "revenue", "money", "earnings", "quarter", "bank"]):
            vec += self.anchors["finance"] * 2.0
            matched = True
        if any(w in t_low for w in ["dog", "cat", "animal", "pet", "puppy", "chased", "barks"]):
            vec += self.anchors["animals"] * 2.0
            matched = True
        if any(w in t_low for w in ["ai", "python", "model", "algorithm", "software", "code", "neural"]):
            vec += self.anchors["tech"] * 2.0
            matched = True

        if not matched:
            np.random.seed(abs(hash(text)) % (2**31))
            vec = np.random.uniform(-0.4, 0.4, self.dimensions)

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def compute_sts_similarity(self, sent1: str, sent2: str) -> Tuple[float, float]:
        """
        Calculates cosine similarity and converts to 0-5 STS Benchmark scale.
        """
        v1 = self.encode_sentence(sent1)
        v2 = self.encode_sentence(sent2)
        cos_sim = float(np.dot(v1, v2))
        # Map [-1, 1] to [0, 5]
        sts_score = max(0.0, min(5.0, (cos_sim + 1.0) * 2.5))
        return round(cos_sim, 4), round(sts_score, 2)

    def evaluate_triplet(self, anchor: str, positive: str, negative: str, margin: float = 0.3) -> Dict[str, Any]:
        """Evaluates Triplet Loss: max(0, dist(A, P) - dist(A, N) + margin)."""
        v_a = self.encode_sentence(anchor)
        v_p = self.encode_sentence(positive)
        v_n = self.encode_sentence(negative)

        dist_ap = float(np.linalg.norm(v_a - v_p))
        dist_an = float(np.linalg.norm(v_a - v_n))

        raw_loss = dist_ap - dist_an + margin
        triplet_loss = max(0.0, raw_loss)

        return {
            "dist_anchor_positive": round(dist_ap, 4),
            "dist_anchor_negative": round(dist_an, 4),
            "margin": margin,
            "triplet_loss": round(triplet_loss, 4),
            "is_satisfied": triplet_loss == 0.0
        }

code_snippet = r'''
from sentence_transformers import SentenceTransformer, util

# 1. Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Encode sentences into fixed dense vectors
sentences = [
    "A boy is playing soccer in the park.",
    "A kid is kicking a football outdoors.",
    "Stock markets experienced heavy trading volume today."
]
embeddings = model.encode(sentences, convert_to_tensor=True)

# 3. Compute cosine similarity matrix
cosine_scores = util.cos_sim(embeddings, embeddings)
# Score between 0 and 1: 0.85+ (Paraphrase)
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive STS & Triplet Loss Workbench
st.header("3. Interactive STS Comparison & Triplet Loss Workbench", divider="rainbow")
st.markdown("""
Test sentence similarity on the **Semantic Textual Similarity (STS)** scale, or inspect how **Contrastive Triplet Loss** enforces semantic boundaries.
""")

sbert_engine = ProductionSentenceEmbeddingComparator(dimensions=8)

tab_sts1, tab_sts2 = st.tabs(["1. Pairwise STS Similarity Benchmark", "2. Contrastive Triplet Loss Inspector"])

with tab_sts1:
    preset_pairs = {
        "Paraphrase: 'A young boy is playing soccer' vs 'A kid kicks a football'": (
            "A young boy is playing soccer in the park.",
            "A kid kicks a football on the green field."
        ),
        "Topic Divergence: 'A dog is barking at the cat' vs 'Tech stocks surged on earnings'": (
            "A brown dog is barking at the neighbor's cat.",
            "Tech stocks surged today following quarterly earnings reports."
        ),
        "Syntactic Shift: 'The dog chased the cat' vs 'The cat chased the dog'": (
            "The playful dog chased the fast cat.",
            "The fast cat chased the playful dog."
        )
    }

    selected_pair = st.selectbox("Select semantic sentence pair to test:", list(preset_pairs.keys()))
    s_a, s_b = preset_pairs[selected_pair]

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        text_a = st.text_area("Sentence A:", value=s_a, height=80)
    with col_s2:
        text_b = st.text_area("Sentence B:", value=s_b, height=80)

    sim_score, sts_benchmark = sbert_engine.compute_sts_similarity(text_a, text_b)

    col_sts_m1, col_sts_m2 = st.columns([1, 1])
    with col_sts_m1:
        st.metric("Cosine Similarity", f"{sim_score:.4f}", delta="Paraphrase" if sim_score > 0.8 else ("Related" if sim_score > 0.4 else "Unrelated"))
    with col_sts_m2:
        st.metric("STS Benchmark Score (0 to 5)", f"{sts_benchmark:.2f} / 5.0")

with tab_sts2:
    st.markdown("### Contrastive Triplet Loss Simulator")
    st.markdown("""
    During SBERT training, triplets enforce that the **Anchor** is closer to the **Positive** than to the **Negative** by at least a predefined margin:
    `Loss = max(0, dist(Anchor, Positive) - dist(Anchor, Negative) + Margin)`.
    """)

    col_t_a, col_t_p, col_t_n = st.columns(3)
    with col_t_a:
        anchor_text = st.text_input("Anchor Sentence:", value="A software developer writes clean Python code.")
    with col_t_p:
        positive_text = st.text_input("Positive (Similar) Sentence:", value="An engineer authors scalable machine learning scripts.")
    with col_t_n:
        negative_text = st.text_input("Negative (Unrelated) Sentence:", value="A golden retriever dog barks at the mailman.")

    margin_val = st.slider("Contrastive Margin:", min_value=0.1, max_value=1.0, value=0.3, step=0.05)

    triplet_results = sbert_engine.evaluate_triplet(anchor_text, positive_text, negative_text, margin=margin_val)

    st.markdown("##### Triplet Optimization Trace:")
    col_t_res1, col_t_res2, col_t_res3 = st.columns(3)
    col_t_res1.metric("Anchor-Positive Dist", f"{triplet_results['dist_anchor_positive']:.4f}")
    col_t_res2.metric("Anchor-Negative Dist", f"{triplet_results['dist_anchor_negative']:.4f}")
    col_t_res3.metric("Triplet Loss", f"{triplet_results['triplet_loss']:.4f}", delta="Objective Satisfied (Loss = 0)" if triplet_results['is_satisfied'] else "Loss > 0 (Gradient Update Required)", delta_color="normal" if triplet_results['is_satisfied'] else "inverse")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Sentence Embedding Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why does standard pre-trained BERT without fine-tuning produce poor sentence embeddings?**
    * **Answer**:
      * The original BERT model was pre-trained using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). It was **never trained to map whole sentences to meaningful cosine similarity coordinates**.
      * Untrained BERT token representations suffer from extreme **anisotropy** (the 'cone effect'); the average pairwise cosine similarity between random sentences is artificially high (~0.65 to 0.80), yielding poor discrimination on Semantic Textual Similarity (STS) benchmarks.
      * **SBERT's Contribution**: Fine-tunes BERT using Siamese contrastive loss (NLI datasets + STS), pulling the average cosine similarity of unrelated pairs down to ~0.0 and dramatically widening dynamic range.

    **Q2: What is Multiple Negatives Ranking Loss (MNRL) and why is it so effective in SBERT?**
    * **Answer**:
      * In a training batch of B (query, positive) pairs: [(q1, p1), (q2, p2), ..., (qB, pB)].
      * For query q1, p1 is the positive document, while all other B - 1 documents in the batch (p2, p3, ..., pB) serve as **in-batch negative examples**!
      * **Computational Efficiency**: A batch of size 64 automatically yields 1 positive and 63 negative pairs per query without requiring manual negative mining.

    **Q3: When should you use a Bi-Encoder vs. a Cross-Encoder for sentence matching?**
    * **Answer**:
      * **Bi-Encoder (Sentence Transformers)**: Encodes sentences independently into vectors. Ideal for 1-to-many retrieval (searching 10M documents in a vector DB) in sub-10ms.
      * **Cross-Encoder**: Feeds both sentences simultaneously into the transformer: `[CLS] SentA [SEP] SentB [SEP]`. Full cross-attention computes deeper interaction scores, yielding higher accuracy but O(N) inference cost.
      * **Production Pattern**: Use Bi-Encoder for initial Top-100 candidate retrieval, followed by Cross-Encoder for Top-5 reranking.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Using Cross-Encoders as Vector Database Embeddings**:
      * *Issue*: Attempting to store cross-encoder outputs in Pinecone or ChromaDB.
      * *Consequence*: Cross-encoders output a single similarity classification float score, NOT an independent vector. They cannot be indexed.
      * *Fix*: Index bi-encoder sentence embeddings; use cross-encoders strictly as secondary rerankers.

    * **Trap 2: Ignoring In-Domain STS Calibration**:
      * *Issue*: Using a general-domain model (`all-MiniLM-L6-v2`) to compare technical legal contracts.
      * *Consequence*: Sub-optimal ranking because legal terminology ("indemnification") has distinct semantics not captured in web crawl data.
      * *Fix*: Fine-tune on domain-specific pairs or use specialized models (e.g. LegalBERT).

    * **Trap 3: SBERT Token Length Truncation**:
      * *Issue*: Passing a 1,000-word essay into SBERT expecting a full-document representation.
      * *Consequence*: SBERT quietly truncates text at 128 or 256 tokens, discarding the final 75% of the text.
      * *Fix*: Use document chunking or models with extended context windows.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Siamese Network**
      * *Definition*: An artificial neural network architecture using tied weights to process two different input sentences into the same vector space.
    * **Flashcard 2: Multiple Negatives Ranking Loss (MNRL)**
      * *Definition*: A contrastive loss function that treats all other positive documents in the batch as negative examples for the current query.
    * **Flashcard 3: Cross-Encoder vs. Bi-Encoder**
      * *Definition*: Bi-encoders produce separate indexable vectors; cross-encoders process pair tokens together for high-accuracy reranking.
    * **Flashcard 4: Triplet Loss Margin**
      * *Definition*: The minimum required geometric distance buffer enforcing that the Anchor is closer to the Positive than to the Negative.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Sentence Embeddings:

- [ ] I can explain why naive word vector averaging fails to represent sentence meaning.
- [ ] I understand the Siamese transformer architecture and why it enables 10ms retrieval over millions of sentences.
- [ ] I know how Contrastive Triplet Loss and Multiple Negatives Ranking Loss (MNRL) train sentence encoders.
- [ ] I can contrast Bi-Encoders and Cross-Encoders in production retrieval pipelines.
- [ ] I understand why raw pre-trained BERT exhibits severe anisotropy on sentence similarity tasks.
""")
