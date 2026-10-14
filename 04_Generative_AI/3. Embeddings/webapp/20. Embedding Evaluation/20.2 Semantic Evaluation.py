import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from scipy.stats import spearmanr, pearsonr
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="20.2 Semantic Evaluation - Embeddings Curriculum",
    page_icon="🎯",
    layout="wide"
)

# Title & Overview
st.title("20.2 Semantic Evaluation: Ground-Truth Pairs, Hard Negatives, & MTEB Benchmarks")
st.caption("Chapter 20: Embedding Evaluation | Domain: 3. Vector Embeddings")
st.markdown("""
While retrieval metrics evaluate how a vector database ranks documents in a search engine, **Semantic Evaluation** directly measures how well an embedding model represents human language nuances.

A model might score high on trivial search tasks simply by acting as an expensive keyword matcher.
True semantic evaluation tests whether the model understands deep semantic relationships:
* **Ground-Truth Pairs**: Measuring correlation between model cosine similarities and human-annotated similarity scores (0.0 to 5.0).
* **Positive Examples**: Paraphrases and entailments that must cluster together despite completely different vocabularies.
* **Hard Negatives**: Adversarial sentence pairs that share heavy lexical keyword overlap but describe diametrically opposite facts!
* **Global Benchmark Suites**: The Massive Text Embedding Benchmark (MTEB), STS Benchmark, and BEIR.

This lesson explores how embedding models are empirically benchmarked against human linguistic ground truth.
""")

st.divider()

# Section 1: The Semantic Evaluation Paradigm & MTEB Architecture
st.header("1. The Semantic Evaluation Paradigm & The MTEB Suite", divider="rainbow")

col_sem1, col_sem2 = col_sem1, col_sem2 = st.columns([1, 1])

with col_sem1:
    st.markdown("""
    ### The Contrastive Triplet Paradigm
    High-quality embedding evaluation relies on three sentence categories:
    * **1. Anchor ($A$)**: The reference query or concept (e.g. *"The court granted an injunction"*).
    * **2. Positive ($P$)**: A paraphrase or direct conceptual match (e.g. *"A judicial restraining order was issued"*). Zero shared words, but identical meaning.
    * **3. Hard Negative ($N_{hard}$)**: An adversarial distractor sharing almost identical words, but expressing a different fact (e.g. *"The court denied the motion for an injunction"*).
    * **The Semantic Goal**: The embedding distance `Dist(A, P)` must be dramatically smaller than `Dist(A, N_hard)`. Weak models fail because shared words pull the hard negative closer!
    """)

with col_sem2:
    st.markdown("### Contrastive Triplet Latent Alignment")
    triplet_dot = """
    digraph TripletAlignment {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Anchor [label="Anchor (A):\\n'Court granted an injunction'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Positive [label="Positive (P):\\n'Judicial restraining order issued'\\n(0 shared words, same meaning)", fillcolor="#ECFDF5", color="#10B981", style="filled,bold"];
        HardNeg [label="Hard Negative (N):\\n'Court DENIED the injunction'\\n(90% shared words, OPPOSITE fact)", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];

        Anchor -> Positive [label="Pull Force (Minimize Distance)\\nCosSim ~ 0.88", color="#059669"];
        Anchor -> HardNeg [label="Push Force (Maximize Distance)\\nCosSim < 0.35", color="#DC2626"];
    }
    """
    st.graphviz_chart(triplet_dot, use_container_width=False)

st.markdown("### The Massive Text Embedding Benchmark (MTEB)")
st.markdown("""
**MTEB (Massive Text Embedding Benchmark)** is the global industry standard for evaluating text embeddings across 8 distinct task categories and 50+ datasets:
* **1. Retrieval (BEIR / MS MARCO)**: Finding relevant documents for queries.
* **2. Semantic Textual Similarity (STS)**: Predicting human similarity ratings (measured via Spearman correlation).
* **3. Reranking**: Re-ordering top-K candidate passages.
* **4. Clustering**: Grouping unlabeled text into semantic topics (measured via V-measure).
* **5. Classification**: Using linear probes on frozen embeddings to predict categories.
* **6. Pair Classification**: Determining if two sentences are duplicates or entailments.
* **7. Summarization**: Scoring generated summary quality against human reference summaries.
* **8. Bitext Mining**: Matching translated sentence pairs across parallel multilingual corpora.
""")

st.divider()

# Section 2: Production Code: STS & Correlation Evaluation Suite
st.header("2. Production Code: STS Evaluation & Correlation Suite", divider="rainbow")

st.markdown("""
The production utility below evaluates predicted model similarities against human ground-truth ratings, calculating Pearson correlation, Spearman rank correlation, and hard-negative margin separation.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.stats import spearmanr, pearsonr

class SemanticEvaluationSuite:
    """
    Evaluates embedding models on Semantic Textual Similarity (STS) pairs
    and computes Spearman rank correlation against human ground truth.
    """
    @staticmethod
    def evaluate_sts_correlation(
        predicted_similarities: List[float],
        human_ground_truth_scores: List[float]
    ) -> Dict[str, float]:
        """
        Computes Pearson and Spearman correlation.
        Spearman rank correlation is the official metric on STS benchmarks.
        """
        pred = np.array(predicted_similarities, dtype=np.float64)
        human = np.array(human_ground_truth_scores, dtype=np.float64)

        # Pearson correlation (linear alignment)
        p_corr, _ = pearsonr(pred, human)
        
        # Spearman correlation (monotonic rank alignment)
        s_corr, _ = spearmanr(pred, human)

        return {
            "pearson_correlation": float(p_corr),
            "spearman_rank_correlation": float(s_corr),
            "mean_squared_error": float(np.mean((pred - human / 5.0) ** 2))
        }

    @staticmethod
    def evaluate_hard_negative_margin(
        anchor_pos_sim: float,
        anchor_neg_sim: float
    ) -> Dict[str, Any]:
        """
        Calculates contrastive margin separation.
        Positive must score significantly higher than the hard negative.
        """
        margin = anchor_pos_sim - anchor_neg_sim
        is_separated = margin > 0.20
        return {
            "anchor_pos_similarity": anchor_pos_sim,
            "anchor_neg_similarity": anchor_neg_sim,
            "margin": margin,
            "passed_hard_negative_test": is_separated
        }
''', language="python")

st.divider()

# Section 3: Interactive Semantic Evaluation Studio
st.header("3. Interactive Workbench: The STS Benchmark & Hard Negative Arena", divider="rainbow")

st.markdown("""
Compare two models on a 5-pair STS benchmark containing subtle paraphrases and deceptive hard negatives:
* **Model 1 (Modern Contrastive Model)**: Trained with hard negatives; understands intent over words.
* **Model 2 (Lexical Surface Model)**: Naive keyword matcher; fooled by shared surface vocabulary.
""")

sts_dataset = [
    {
        "id": "Pair 1 (Direct Paraphrase)",
        "s1": "A plane is landing on the runway.",
        "s2": "An airplane touches down on the tarmac.",
        "human_score": 4.8, # Max 5.0
        "notes": "Paraphrase with zero shared content words."
    },
    {
        "id": "Pair 2 (Topical Entailment)",
        "s1": "The doctor prescribed antibiotics for the infection.",
        "s2": "The patient took medication for a bacterial disease.",
        "human_score": 3.7,
        "notes": "Conceptually related with mild entailment."
    },
    {
        "id": "Pair 3 (Hard Negative Trap)",
        "s1": "The company reported record quarterly net profits.",
        "s2": "The company reported record quarterly net losses.",
        "human_score": 0.8,
        "notes": "90% shared words, but polar opposite financial meaning!"
    },
    {
        "id": "Pair 4 (Negation Hard Negative)",
        "s1": "This chemical compound is safe for human consumption.",
        "s2": "This chemical compound is NOT safe for human consumption.",
        "human_score": 0.5,
        "notes": "Exact match except for the word NOT."
    },
    {
        "id": "Pair 5 (Orthogonal Unrelated)",
        "s1": "The chef baked a chocolate sourdough loaf.",
        "s2": "Quantum entanglement occurs across particle pairs.",
        "human_score": 0.1,
        "notes": "Completely unrelated topics."
    }
]

col_d1, col_d2 = st.columns([1, 1])

with col_d1:
    st.subheader("1. STS Benchmark Dataset")
    st.caption("Human ground-truth similarity scale: 0.0 (Unrelated) to 5.0 (Identical meaning).")
    
    df_pairs = pd.DataFrame([
        {"ID": p["id"], "Sentence A": p["s1"], "Sentence B": p["s2"], "Human (0-5)": p["human_score"]}
        for p in sts_dataset
    ])
    st.dataframe(df_pairs, use_container_width=True)

    st.subheader("2. Select Model Evaluation Profile")
    model_mode = st.radio(
        "Choose Model to Evaluate on Benchmark:",
        ["Model 1: Modern Contrastive Embedding Model (BGE / E5)", "Model 2: Naive Lexical Overfitting Model (Word Overlap)"],
        index=0
    )

with col_d2:
    st.subheader(f"3. Evaluation Results: {model_mode.split(':')[0]}")

    human_scores = [p["human_score"] for p in sts_dataset]

    if "Model 1" in model_mode:
        # High-quality contrastive predictions
        predicted_sims = [0.92, 0.74, 0.18, 0.12, 0.04]
    else:
        # Lexical naive model fooled by shared words in pairs 3 and 4!
        predicted_sims = [0.25, 0.35, 0.89, 0.94, 0.02]

    # Normalize human scores to 0-1 scale for comparison
    human_norm = [h / 5.0 for h in human_scores]

    s_corr, _ = spearmanr(predicted_sims, human_scores)
    p_corr, _ = pearsonr(predicted_sims, human_scores)

    c1, c2, c3 = st.columns(3)
    c1.metric("Spearman Rank Corr", f"{s_corr:.3f}", help="Official MTEB metric (higher is better)")
    c2.metric("Pearson Correlation", f"{p_corr:.3f}")
    c3.metric("Hard-Neg Accuracy", "100% Passed" if "Model 1" in model_mode else "0% FAILED")

    if "Model 1" in model_mode:
        st.success("✅ **Superb Semantic Alignment**: High correlation with human ratings. Correctly recognized that Pair 3 & Pair 4 are opposites despite shared words!")
    else:
        st.error("❌ **Lexical Trap Failure**: The naive model gave Pair 4 a 0.94 similarity just because 90% of words were identical, missing the negation entirely!")

    # Scatter plot of Predicted vs Human Ground Truth
    df_eval = pd.DataFrame({
        "Pair": [f"P{i+1}" for i in range(5)],
        "Predicted Cosine (0-1)": predicted_sims,
        "Human Score (0-5)": human_scores,
        "Human Normalized (0-1)": human_norm
    })
    st.markdown("#### Predicted Cosine vs. Human Ground Truth Alignment")
    st.line_chart(df_eval.set_index("Pair")[["Predicted Cosine (0-1)", "Human Normalized (0-1)"]])

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Evaluation Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is Spearman Rank Correlation preferred over Pearson Correlation when evaluating Semantic Textual Similarity (STS)?**
    * **Answer**:
      * **Pearson Correlation** measures strict linear alignment between predicted cosine values and human scores.
      * **Spearman Rank Correlation** measures **monotonic rank ordering**: whether Pair A is ranked higher than Pair B, regardless of whether the scale is linear, sigmoidal, or compressed.
      * In real-world retrieval, the exact numerical distance scale does not matter—what matters is the **relative ordering** of candidates. Therefore, MTEB and academic benchmarks universally report Spearman rank correlation.

    **Q2: What is 'Shortcut Learning' in contrastive embedding training and how do Hard Negatives prevent it?**
    * **Answer**: If an embedding model is trained using random in-batch negatives (e.g. comparing a sentence about basketball against a sentence about cooking), the model takes an easy shortcut: it learns to identify broad vocabulary domains rather than deep semantics.
    * When deployed in production, it fails on hard domain queries.
    * **Hard Negative Mining** (e.g. mining passages that share keywords but disagree on facts) forces the model to attend to syntax, negation, verbs, and relational entities rather than relying on shallow vocabulary shortcuts.

    **Q3: What is the BEIR benchmark and how does it evaluate out-of-domain retrieval generalization?**
    * **Answer**: BEIR (Benchmarking Information Retrieval) is a zero-shot retrieval benchmark comprising 17 heterogeneous datasets across diverse domains (BioASQ for biomedical, FiQA for finance, SciFact for scientific fact-checking, Climate-FEVER, CQADupStack). It tests whether an embedding model trained on general data can generalize to specialized domains without fine-tuning.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Evaluating Models Exclusively on In-Distribution Validation Data**:
      * *Issue*: Training on MS MARCO and only evaluating on MS MARCO test splits.
      * *Consequence*: The model scores 90% on common queries but collapses on internal enterprise technical docs.
      * *Fix*: Always benchmark on zero-shot out-of-domain suites like BEIR.

    * **Trap 2: Using Mean Squared Error (MSE) to Compare Different Embedding Models on STS**:
      * *Issue*: Computing MSE between cosine similarity and human labels across models with different dimensional scales.
      * *Consequence*: Model A might have an anisotropic baseline of 0.80 and Model B has a baseline of 0.20, distorting raw MSE.
      * *Fix*: Use rank-based Spearman correlation, which is scale-invariant.

    * **Trap 3: Overlooking Negation Handling in Production Testing**:
      * *Issue*: Assuming high overall MTEB score guarantees the model handles negations.
      * *Consequence*: Queries for *"policies excluding part-time workers"* retrieve documents stating *"policies including part-time workers"*.
      * *Fix*: Curate an adversarial test suite specifically testing syntactic polarity and negation flips.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: MTEB (Massive Text Embedding Benchmark)**
      * *Definition*: The preeminent multi-task benchmark assessing embedding models across 8 tasks: Retrieval, STS, Clustering, Classification, Reranking, etc.
    * **Flashcard 2: Spearman Rank Correlation**
      * *Definition*: The monotonic rank-order correlation metric used as the official evaluation standard on STS benchmarks.
    * **Flashcard 3: Hard Negative Mining**
      * *Definition*: Finding challenging negative training samples that share heavy lexical overlap with the anchor but represent different factual concepts.
    * **Flashcard 4: BEIR Benchmark**
      * *Definition*: A 17-dataset heterogeneous benchmark designed to evaluate zero-shot domain transfer in information retrieval.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Semantic Evaluation:

- [ ] I can articulate the contrastive triplet paradigm (Anchor, Positive, Hard Negative).
- [ ] I understand why Spearman rank correlation is preferred over Pearson for STS evaluation.
- [ ] I know the 8 core task categories evaluated in MTEB.
- [ ] I understand how Hard Negatives prevent shortcut learning in embedding training.
- [ ] I can describe the purpose of zero-shot domain evaluation via BEIR.
""")
