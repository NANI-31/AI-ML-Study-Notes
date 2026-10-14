import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="5.2 Sentence Transformers - Embeddings Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("5.2 Sentence Transformers: SBERT Architecture, Pooling Layers, & High-Throughput Encoding")
st.caption("Chapter 5: Embedding Models | Domain: 3. Vector Embeddings")
st.markdown("""
The `sentence-transformers` library (originally developed by Nils Reimers and UKPLab, now an open-source standard under Hugging Face) revolutionized dense semantic retrieval.
While raw BERT required feeding sentence pairs jointly through cross-attention (taking **65 hours** to find the most similar pair in a 10,000-sentence corpus), **Sentence Transformers (SBERT)** produces independent, dense vector embeddings for individual sentences, reducing the search to a simple vector dot product taking just **5 milliseconds**!

This lesson explores:
* **The SBERT Pipeline**: Tokenizer -> Transformer Backbone -> Pooling Layer -> Normalization.
* **Pooling Strategies**: Mean Pooling vs. CLS Token vs. Max Pooling.
* **Production API**: `model.encode()` parameters, multi-GPU processing, and `util.semantic_search`.
* **Interactive Simulator**: Benchmark pooling strategies, similarity matrices, and batch throughput dynamics.
""")

st.divider()

# Section 1: The SBERT Architecture & Pooling Mechanisms
st.header("1. SBERT Architecture & Token Pooling Mechanisms", divider="rainbow")

col_sb1, col_sb2 = st.columns([1, 1])

with col_sb1:
    st.markdown("""
    ### Why Raw BERT is NOT a Sentence Embedder:
    * Standard BERT outputs a sequence of contextual hidden vectors: one vector per token:
      `[CLS], t_1, t_2, ..., t_n, [SEP]`
    * If you simply take the `[CLS]` token from a standard pretrained BERT model without fine-tuning, the sentence embeddings are notoriously poor—often worse than averaging static GloVe word vectors!
    * BERT's token representations suffer from **anisotropy**: all vectors are tightly squeezed into a narrow geometric cone with high average cosine similarity.

    ### How Sentence Transformers Solves This:
    1. **Siamese Fine-Tuning**: Trains Siamese networks on NLI (Natural Language Inference) and STS (Semantic Textual Similarity) pairs.
    2. **Explicit Pooling Layer**: Aggregates token-level hidden states into a single fixed-size sentence vector:
       * **Mean Pooling (Default)**: Takes the average of all contextual token embeddings, masked by the attention mask.
       * **CLS Pooling**: Uses the final hidden state of the first `[CLS]` token (common in models specifically tuned for CLS).
       * **Max Pooling**: Takes the dimension-wise maximum across all token representations.
    3. **L2 Unit Normalization**: Ensures that cosine similarity simplifies to a blazing-fast inner dot product.
    """)

with col_sb2:
    st.markdown("### Sentence Transformers Forward Pipeline")
    sbert_dot = """
    digraph SBERTPipeline {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        InputText [label="Input Text\\n'Sentence Transformers are fast'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Tokenizer [label="WordPiece / BPE Tokenizer\\nToken IDs + Attention Mask", fillcolor="#FEF3C7"];
        Transformer [label="Transformer Backbone (e.g. MiniLM / MPNet)\\nToken Hidden States: (B, seq_len, 384)", fillcolor="#D1FAE5"];
        
        subgraph cluster_pool {
            label="Pooling Module"; style="dashed"; color="#8B5CF6";
            Pool [label="Mean Pooling\\nsum(h_i * mask) / sum(mask)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];
        }

        Norm [label="L2 Normalization Layer\\nv / norm(v)", fillcolor="#FCE7F3"];
        OutputVec [label="Dense Sentence Embedding\\n(384-d or 768-d Unit Vector)", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        InputText -> Tokenizer -> Transformer -> Pool -> Norm -> OutputVec;
    }
    """
    st.graphviz_chart(sbert_dot, use_container_width=False)

st.markdown("### Top Open-Source Sentence Transformer Models")
st.markdown("""
| Model Identifier | Base Architecture | Output Dim | Context Window | Speed (Sentences/sec) | MTEB Score | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **all-MiniLM-L6-v2** | 6-layer MiniLM | 384 | 256 tokens | **~14,200 (CPU/GPU)** | 56.3 | Ultra-fast local edge inference, real-time search |
| **all-mpnet-base-v2** | MPNet | 768 | 384 tokens | ~2,800 | 57.8 | High-accuracy general purpose text embedding |
| **BAAI/bge-small-en-v1.5** | BERT-small | 384 | 512 tokens | ~9,500 | 62.1 | RAG retrieval with instruction prefixes |
| **BAAI/bge-large-en-v1.5** | BERT-large | 1024 | 512 tokens | ~1,200 | 64.1 | Enterprise search requiring deep semantic discrimination |
""")

st.divider()

# Section 2: Production Code: Sentence Transformers API & High-Throughput Batching
st.header("2. Production Implementation: Sentence Transformers Batch Engine", divider="rainbow")

st.markdown("""
The production class below shows how to configure `sentence-transformers` for real-world deployments: controlling precision (FP16 / FP32), batch sizing, attention mask pooling, and computing cross-sentence similarity matrices.
""")

st.code(r'''
import torch
import numpy as np
from typing import List, Dict, Any, Union

# In production:
# from sentence_transformers import SentenceTransformer, util

class ProductionSentenceEmbedder:
    """
    Production-grade wrapper for SentenceTransformer inference.
    Handles device placement, FP16 mixed precision, normalized encoding, and search.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        # In production:
        # self.model = SentenceTransformer(model_name, device=device)
        # if device == "cuda":
        #     self.model.half()  # Enable FP16 for 2x faster GPU inference

    def encode_corpus(
        self, 
        sentences: List[str], 
        batch_size: int = 64, 
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encodes a list of sentences into a float32/float16 numpy matrix.
        """
        # In production:
        # return self.model.encode(
        #     sentences,
        #     batch_size=batch_size,
        #     show_progress_bar=len(sentences) > 1000,
        #     normalize_embeddings=normalize,
        #     convert_to_numpy=True
        # )
        
        # Synthetic simulation for illustration:
        rng = np.random.default_rng(42)
        dim = 384 if "MiniLM" in self.model_name else 768
        vectors = rng.standard_normal((len(sentences), dim)).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return vectors

    def semantic_search(
        self, 
        query: str, 
        corpus: List[str], 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Computes cosine similarity between a single query and corpus.
        Returns top_k matching documents.
        """
        q_emb = self.encode_corpus([query], normalize=True)
        c_emb = self.encode_corpus(corpus, normalize=True)
        
        # Dot product equals cosine similarity for normalized vectors
        scores = np.dot(c_emb, q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "corpus_id": idx,
                "text": corpus[idx],
                "score": float(scores[idx])
            })
        return results
''', language="python")

# Actual Python implementation for the interactive workbench
class MockSentenceTransformerService:
    def __init__(self):
        # Semantic vectors for demonstrative sentences in 6-dimensional latent space
        # Dimensions: [Tech/AI, Finance/Money, Nature/Environment, Health/Medicine, Food/Cooking, Space/Physics]
        self.concept_labels = ["Tech/AI", "Finance", "Ecology", "Health", "Food", "Astronomy"]
        
        self.sample_sentences = {
            "Transformers use self-attention mechanisms to process tokens": np.array([0.95, 0.05, 0.02, 0.01, 0.01, 0.15]),
            "Large language models excel at natural language processing": np.array([0.92, 0.08, 0.01, 0.02, 0.01, 0.10]),
            "Federal interest rates impact stock market valuations": np.array([0.05, 0.96, 0.02, 0.01, 0.01, 0.01]),
            "Global warming threatens coastal ecosystems and reefs": np.array([0.02, 0.05, 0.95, 0.12, 0.02, 0.01]),
            "Aerobic exercise strengthens cardiovascular wellness": np.array([0.01, 0.02, 0.10, 0.96, 0.05, 0.01]),
            "Fresh ingredients enhance authentic Italian pasta sauces": np.array([0.01, 0.01, 0.04, 0.08, 0.97, 0.01])
        }

    def compute_similarity_matrix(self, sentences: List[str], pooling_noise: float = 0.0):
        raw_vecs = []
        for s in sentences:
            base = self.sample_sentences.get(s, np.random.default_rng(abs(hash(s)) % (2**31)).standard_normal(6))
            if pooling_noise > 0:
                noise = np.random.default_rng(42).standard_normal(6) * pooling_noise
                base = base + noise
            norm = base / np.linalg.norm(base)
            raw_vecs.append(norm)
        
        mat = np.array(raw_vecs)
        # Cosine similarity matrix: mat @ mat.T
        sim_matrix = np.dot(mat, mat.T)
        return sim_matrix

st.divider()

# Section 3: Interactive Workbench: SBERT Similarity Matrix & Benchmark
st.header("3. Interactive Workbench: SBERT Semantic Similarity Matrix Explorer", divider="rainbow")

st.markdown("""
Select sentences to encode and explore how Sentence Transformers measures fine-grained semantic distance. Observe how semantically related sentences exhibit high cosine similarity while distinct domains remain orthogonal.
""")

svc = MockSentenceTransformerService()

col_bench_left, col_bench_right = st.columns([1, 1])

with col_bench_left:
    st.subheader("1. Sentence Selection")
    all_options = list(svc.sample_sentences.keys())
    selected_sentences = st.multiselect(
        "Select Sentences to Encode & Compare:",
        options=all_options,
        default=[
            all_options[0],  # Transformers use self-attention
            all_options[1],  # LLMs excel at NLP
            all_options[2],  # Federal interest rates
            all_options[3]   # Global warming
        ]
    )

    pooling_mode = st.radio(
        "Simulated Pooling Mode:",
        ["Mean Pooling (Standard SBERT)", "CLS Token Pooling (Raw Transformer)", "Max Pooling"]
    )
    
    noise_factor = 0.0 if "Mean" in pooling_mode else (0.15 if "Max" in pooling_mode else 0.35)

    st.markdown("#### Hardware Throughput Calculator")
    hw_choice = st.selectbox("Inference Hardware Target:", ["NVIDIA A100 (GPU FP16)", "NVIDIA T4 (GPU FP16)", "Intel Xeon (CPU 8-core)"])
    batch_size = st.slider("Encoding Batch Size:", min_value=8, max_value=256, value=64, step=8)
    
    speed_map = {
        "NVIDIA A100 (GPU FP16)": 28_000,
        "NVIDIA T4 (GPU FP16)": 8_500,
        "Intel Xeon (CPU 8-core)": 950
    }
    throughput = speed_map[hw_choice] * (batch_size / 64) ** 0.5
    st.metric("Estimated Throughput (Sentences/sec)", f"{int(throughput):,} sent/sec")

with col_bench_right:
    st.subheader("2. Semantic Cosine Similarity Matrix")
    if len(selected_sentences) < 2:
        st.warning("Please select at least 2 sentences to compute the similarity matrix.")
    else:
        sim_mat = svc.compute_similarity_matrix(selected_sentences, pooling_noise=noise_factor)
        
        # Display as clean styled DataFrame
        short_labels = [f"S{i+1}: {s[:30]}..." for i, s in enumerate(selected_sentences)]
        df_sim = pd.DataFrame(sim_mat, index=short_labels, columns=short_labels)
        
        st.dataframe(
            df_sim.style.format("{:.4f}").background_gradient(cmap="Blues", vmin=0.0, vmax=1.0),
            use_container_width=True
        )

        st.markdown("#### Sentence Mapping Key")
        for i, s in enumerate(selected_sentences):
            st.caption(f"**S{i+1}**: {s}")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Sentence Transformers Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why cannot we simply use the [CLS] token embedding from BERT directly as a sentence embedding?**
    * **Answer**:
      * Pretrained BERT was trained on Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
      * The `[CLS]` token's final hidden state is trained only to serve as the input to a binary classification head (is next sentence vs is not). It is never forced to capture the complete semantic summary of the entire sentence.
      * Furthermore, standard BERT representations suffer from **anisotropy**: vectors occupy a narrow, high-density cone in the vector space where almost all pairs have high cosine similarity (e.g. 0.8+).
      * SBERT applies contrastive/triplet fine-tuning with **Mean Pooling**, which aggregates all contextual tokens across the sequence and centers the embedding space, producing isotropic, highly discriminative sentence representations.

    **Q2: How does Mean Pooling handle variable-length sequences with padding tokens?**
    * **Answer**:
      * In a batch, sentences have different lengths and are right-padded with `[PAD]` tokens.
      * If you take a naive average of all token embeddings in the sequence, the `[PAD]` token embeddings (which are non-zero) will drag and distort the sentence embedding!
      * **Attention Masked Mean Pooling**:
        1. Multiply token hidden states by the binary `attention_mask` (1 for real tokens, 0 for PAD tokens).
        2. Sum the masked token hidden states across the sequence dimension.
        3. Sum the attention mask values across the sequence dimension (clamped to a minimum of 1e-9 to prevent division by zero).
        4. Divide the summed vectors by the sum of mask values: `mean_pooled = sum(H * mask) / sum(mask)`.

    **Q3: What is the computational complexity difference between Cross-Encoder and Bi-Encoder (Sentence Transformers)?**
    * **Answer**:
      * **Cross-Encoder**: Feeds `(query, document)` jointly into BERT. Full cross-attention occurs across all tokens of both inputs ($O((L_q + L_d)^2)$). For a corpus of N documents, finding the top match requires N forward passes through BERT. Finding all pairs among N documents takes $O(N^2)$ forward passes—computationally intractable for large corpora.
      * **Bi-Encoder (SBERT)**: Encodes query and documents independently into fixed-size vectors ($O(N)$ forward passes offline). Search is computed using dot product ($O(N)$ inner products), taking microseconds.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Forgetting normalize_embeddings=True during Encoding**:
      * *Issue*: Calling `model.encode(sentences)` without specifying `normalize_embeddings=True`.
      * *Consequence*: The returned vectors have variable norms. If stored in a vector DB configured for dot-product or inner-product metric, longer sentences or higher-norm vectors will dominate retrieval artificially.
      * *Fix*: Always set `normalize_embeddings=True` to enforce unit length.

    * **Trap 2: Ignoring the 256 / 512 Maximum Sequence Length Ceiling**:
      * *Issue*: Passing a 2,000-word essay into `all-MiniLM-L6-v2` (which has `max_seq_length = 256`).
      * *Consequence*: The tokenizer silently truncates the essay at 256 tokens, discarding the remaining 85% of the text without throwing an error!
      * *Fix*: Inspect `model.max_seq_length` and chunk long documents into overlapping passages before encoding.

    * **Trap 3: Running Batch Encoding on CPU without Threading Tuning**:
      * *Issue*: Encoding 100,000 sentences on a multi-core CPU with default PyTorch thread settings.
      * *Consequence*: PyTorch over-subscribes threads, causing CPU core thrashing and 3x slower execution.
      * *Fix*: Set `torch.set_num_threads(num_physical_cores)` or use `model.start_multi_process_pool()`.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Bi-Encoder**
      * *Definition*: An architecture where two texts are encoded separately into dense vectors, enabling fast retrieval via cosine similarity or dot product.
    * **Flashcard 2: Mean Pooling**
      * *Definition*: Aggregating token-level representations into a single sentence vector by averaging token hidden states masked by the attention mask.
    * **Flashcard 3: Anisotropy**
      * *Definition*: The geometric phenomenon in transformer representations where vectors cluster in a narrow cone rather than dispersing uniformly across the vector space.
    * **Flashcard 4: all-MiniLM-L6-v2**
      * *Definition*: A ubiquitous 384-dimensional sentence transformer model optimized for blazing speed and lightweight CPU/edge deployment.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Sentence Transformers:

- [ ] I can articulate why raw BERT [CLS] tokens perform poorly for semantic similarity.
- [ ] I understand the mathematical mechanics of attention-masked Mean Pooling.
- [ ] I know the trade-offs between Bi-Encoders (fast retrieval) and Cross-Encoders (deep interaction).
- [ ] I understand why `normalize_embeddings=True` is critical when storing embeddings in vector databases.
- [ ] I know how to check and handle the maximum sequence length (`max_seq_length`) of an SBERT model.
""")
