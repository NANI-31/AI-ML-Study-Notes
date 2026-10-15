import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="6.3 Batch Embeddings - Embeddings Curriculum",
    page_icon="📦",
    layout="wide"
)

# Title & Overview
st.title("6.3 Batch Embeddings: Vectorized Parallelism, Smart Bucketing, & VRAM Optimization")
st.caption("Chapter 6: Generating Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Embedding a single text at a time in a loop is catastrophically slow:
* On a GPU, single-sample inference utilizes less than **3% of CUDA cores**, leaving memory bandwidth and tensor cores sitting idle.
* On an API (OpenAI / Cohere), making individual HTTP POST requests for 100,000 documents results in **100,000 separate TCP/TLS handshakes**, adding hours of pure network latency!

**Batch Embedding** groups multiple strings together into unified multidimensional tensor payloads.
However, naive batching introduces two major engineering hurdles:
1. **The Padding Tax**: When a 10-token sentence is batched with a 500-token document, the short sentence is padded with 490 useless `[PAD]` tokens, wasting 98% of its compute.
2. **GPU Out-Of-Memory (OOM) Crashes**: Setting an arbitrary batch size (e.g. 256) without considering sequence length can easily trigger CUDA OOM on long texts.

This lesson explores vectorized batching mechanics, smart length-sorted bucketing, dynamic batch sizing, and API batch packing.
""")

st.divider()

# Section 1: Vectorized Batching & The Padding Dilemma
st.header("1. Vectorized Batching & The Padding Dilemma", divider="rainbow")

col_bat1, col_bat2 = st.columns([1, 1])

with col_bat1:
    st.markdown("""
    ### Why Batching Yields 10x-50x Speedups:
    * **GPU Tensor Cores**: Modern GPUs (NVIDIA A100, H100, T4) are massive matrix multiplication engines. They achieve peak FLOPS only when processing large matrix batches (e.g. `[Batch Size, Seq Len, Hidden Dim]`).
    * **Kernel Launch Overhead**: Launching a CUDA kernel carries fixed CPU-to-GPU overhead (~5-10 microseconds). In single-text loops, this launch overhead dominates runtime.
    * **Network Throughput**: Hosted APIs permit batching hundreds or thousands of texts per HTTP call, collapsing network round-trips.

    ### Naive Batching vs. Smart Bucketing:
    * **Naive Fixed-Length Batching**: Every sentence is padded to the model's global maximum (e.g. 512 tokens). If the average sentence is 30 tokens, **94% of tensor computations process empty zeros**!
    * **Batch-Dynamic Padding**: Pad only to the maximum length *within that specific batch*.
    * **Smart Length-Sorted Bucketing**:
      1. Sort all corpus texts by token length.
      2. Group sentences of near-identical lengths into batches.
      3. Encode each homogeneous bucket with minimal padding.
      4. Restore original document order after encoding.
    """)

with col_bat2:
    st.markdown("### Naive vs. Smart Bucketing Comparison")
    bat_dot = """
    digraph BatchingFlow {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        subgraph cluster_naive {
            label="Naive Batch (Unsorted)"; style="dashed"; color="#EF4444";
            N1 [label="Text A: 12 tokens + 488 PAD\\nText B: 500 tokens\\nText C: 25 tokens + 475 PAD", fillcolor="#FEE2E2"];
        }

        subgraph cluster_smart {
            label="Smart Bucketed Batches (Sorted)"; style="dashed"; color="#10B981";
            S1 [label="Batch 1 (Short): 10-25 tokens\\nPad length = 25 tokens\\nWasted compute: < 5%", fillcolor="#ECFDF5"];
            S2 [label="Batch 2 (Long): 480-500 tokens\\nPad length = 500 tokens\\nWasted compute: < 2%", fillcolor="#D1FAE5"];
        }

        Corpus [label="Corpus Texts (Arbitrary Lengths)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Corpus -> N1 [label="Random grouping\\nMassive wasted pad FLOPS"];
        Corpus -> S1 [label="Length sort & bucket\\nMinimal padding"];
        Corpus -> S2;
    }
    """
    st.graphviz_chart(bat_dot, use_container_width=False)

st.markdown("### Batching Strategies Tradeoff Table")
st.markdown("""
| Batching Strategy | Padding Method | Wasted FLOPs | Implementation Complexity | Throughput (Sentences/sec) | Memory Predictability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-Item Loop** | No padding | 0% (pure waste in overhead) | Trivial | ~150 sent/sec | Fixed minimal |
| **Global Fixed Padding** | Pad all to 512 | **70% - 90%** | Easy | ~2,500 sent/sec | Constant |
| **Batch-Dynamic Padding** | Pad to max in batch | 30% - 50% | Moderate | ~6,500 sent/sec | Variable |
| **Smart Bucketing (Sorted)** | Sort by length + bucket | **< 8%** | Moderate | **~18,000 sent/sec** | Highly predictable |
""")

st.divider()

# Section 2: Production Code: Smart Length-Sorted Batch Engine
st.header("2. Production Implementation: Length-Sorted Smart Batch Engine", divider="rainbow")

st.markdown("""
The production utility below demonstrates **Smart Bucketing**: sorting inputs by token length, processing homogeneous batches with dynamic padding, and reassembling the resulting vectors into the exact original corpus order.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple

class SmartBatchEmbedder:
    """
    Production-grade batch embedder featuring length-sorted smart bucketing.
    Minimizes wasted padding computations while preserving original input order.
    """
    def __init__(self, batch_size: int = 64, embedding_dim: int = 384):
        self.batch_size = batch_size
        self.dim = embedding_dim

    def encode_corpus_smart(self, texts: List[str]) -> np.ndarray:
        """
        Sorts texts by character/token length, batches dynamically,
        encodes with minimal padding, and reorders vectors to original indices.
        """
        n = len(texts)
        if n == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        # 1. Store original indices alongside estimated length
        indexed_texts = [(idx, text, len(text.split())) for idx, text in enumerate(texts)]
        
        # 2. Sort by sequence length
        sorted_texts = sorted(indexed_texts, key=lambda x: x[2])
        
        output_matrix = np.empty((n, self.dim), dtype=np.float32)

        # 3. Process batches along the sorted sequence
        for start_idx in range(0, n, self.batch_size):
            batch_items = sorted_texts[start_idx : start_idx + self.batch_size]
            original_positions = [item[0] for item in batch_items]
            batch_strings = [item[1] for item in batch_items]
            
            # In production:
            # batch_vectors = model.encode(batch_strings, normalize_embeddings=True)
            
            # Synthetic simulation for illustration:
            rng = np.random.default_rng(start_idx)
            batch_vectors = rng.standard_normal((len(batch_strings), self.dim)).astype(np.float32)
            norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            batch_vectors = batch_vectors / np.maximum(norms, 1e-12)
            
            # 4. Scatter back into the exact original indices
            output_matrix[original_positions] = batch_vectors

        return output_matrix
''', language="python")

# Actual Python implementation for the interactive workbench
class BatchingSimulator:
    def __init__(self):
        # Sample realistic corpus text lengths (tokens)
        self.sample_lengths = [
            12, 15, 18, 22, 25, 28, 32, 45,
            120, 135, 140, 165, 180,
            380, 410, 450, 490, 510
        ]

    def analyze_padding_waste(self, batch_size: int, strategy: str) -> Dict[str, Any]:
        lengths = self.sample_lengths.copy()
        total_tokens = sum(lengths)
        
        if strategy == "Global Max (512 tokens)":
            total_padded_cells = len(lengths) * 512
        elif strategy == "Batch Dynamic (Unsorted)":
            total_padded_cells = 0
            for i in range(0, len(lengths), batch_size):
                batch = lengths[i : i + batch_size]
                max_len = max(batch)
                total_padded_cells += len(batch) * max_len
        else: # Smart Bucketing (Sorted)
            sorted_lens = sorted(lengths)
            total_padded_cells = 0
            for i in range(0, len(sorted_lens), batch_size):
                batch = sorted_lens[i : i + batch_size]
                max_len = max(batch)
                total_padded_cells += len(batch) * max_len

        wasted_pad_tokens = total_padded_cells - total_tokens
        waste_percentage = (wasted_pad_tokens / total_padded_cells) * 100.0
        useful_percentage = (total_tokens / total_padded_cells) * 100.0

        return {
            "real_tokens": total_tokens,
            "padded_cells": total_padded_cells,
            "wasted_tokens": wasted_pad_tokens,
            "waste_pct": waste_percentage,
            "useful_pct": useful_percentage
        }

st.divider()

# Section 3: Interactive Workbench: Batch Padding & Compute Efficiency Lab
st.header("3. Interactive Workbench: Batch Padding & Compute Efficiency Lab", divider="rainbow")

st.markdown("""
Experiment with batch sizes and grouping strategies on a heterogeneous corpus. Observe how **Smart Length-Sorted Bucketing** collapses wasted padding computations from 80%+ down to less than 15%!
""")

sim = BatchingSimulator()

col_lab_l, col_lab_r = st.columns([1, 1])

with col_lab_l:
    st.subheader("1. Configure Batch Parameters")
    strat_choice = st.radio(
        "Batching Strategy:",
        [
            "Global Max (512 tokens) — Naive Fixed Padding",
            "Batch Dynamic (Unsorted) — Pad to Max in Each Batch",
            "Smart Bucketing (Sorted) — Sort by Length + Dynamic Pad"
        ]
    )

    batch_size_sel = st.slider("Batch Size:", min_value=2, max_value=8, value=4, step=2)

    st.markdown("#### Sample Corpus Sequence Lengths (18 Texts)")
    st.caption("Distribution includes short sentences (12-45 tokens), medium paragraphs (120-180 tokens), and long articles (380-510 tokens).")
    df_corpus = pd.DataFrame({
        "Document Index": [f"Doc_{i+1}" for i in range(len(sim.sample_lengths))],
        "Token Length": sim.sample_lengths
    })
    st.dataframe(df_corpus.T, use_container_width=True)

with col_lab_r:
    st.subheader("2. Efficiency & Compute Waste Analysis")
    clean_strat = strat_choice.split(" — ")[0]
    stats = sim.analyze_padding_waste(batch_size_sel, clean_strat)

    m1, m2 = st.columns(2)
    m1.metric("Total Real Tokens", f"{stats['real_tokens']:,}")
    m2.metric("Total Padded Cell Computations", f"{stats['padded_cells']:,}")

    m3, m4 = st.columns(2)
    m3.metric("Compute Efficiency (Useful Work)", f"{stats['useful_pct']:.1f}%")
    m4.metric("Wasted Padding FLOPs", f"{stats['waste_pct']:.1f}%", delta=f"-{stats['waste_pct']:.1f}%", delta_color="inverse")

    st.markdown("#### Performance Metric Summary")
    df_perf = pd.DataFrame({
        "Metric": [
            "Batching Architecture",
            "Batch Size",
            "Real Informational Tokens",
            "Wasted Pad Tokens Computed",
            "Estimated Relative GPU Speedup"
        ],
        "Value": [
            clean_strat,
            f"{batch_size_sel} texts/batch",
            f"{stats['real_tokens']} tokens",
            f"{stats['wasted_tokens']} pad cells",
            f"{stats['useful_pct'] / (stats['real_tokens']/ (len(sim.sample_lengths)*512) * 100):.2f}x vs Global Max"
        ]
    })
    st.dataframe(df_perf, use_container_width=True)

    if stats['useful_pct'] > 75.0:
        st.success(f"🚀 **Optimal Efficiency**: Over {stats['useful_pct']:.1f}% of GPU matrix multiplications perform useful token encoding!")
    else:
        st.warning(f"⚠️ **High Waste Warning**: {stats['waste_pct']:.1f}% of GPU tensor operations are computing attention over meaningless zero-padding tokens.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Batching Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is 'Smart Bucketing' (or Length-Sorted Batching) and why is it used in Sentence Transformers?**
    * **Answer**:
      * In a diverse corpus, text lengths vary wildly: from 5-word titles to 400-word passages.
      * If a short text and a long text share the same batch, the short text must be padded to the longest text in the batch. Transformer self-attention scales quadratically with sequence length ($O(L^2)$), so calculating attention over hundreds of padding tokens wastes massive GPU FLOPs.
      * **Smart Bucketing**:
        1. Pre-sort all texts in ascending order of length.
        2. Create batches containing texts of almost identical lengths.
        3. Pad each batch only to its local maximum.
        4. Reorder the resulting vectors back to the original index positions.
      * This routinely yields a **3x to 5x throughput increase** on GPUs with zero loss of vector precision!

    **Q2: How do you prevent GPU CUDA Out-Of-Memory (OOM) errors during large batch embedding jobs?**
    * **Answer**:
      * 1. **Token-Budget Batching (Dynamic Batch Sizing)**: Instead of a fixed batch size of N texts, define a maximum token budget per batch (e.g. 4,096 tokens). A batch might contain 100 short 40-token sentences, but only 8 long 500-token passages.
      * 2. **FP16 / BF16 Mixed Precision**: Halves activation memory requirements.
      * 3. **Automatic Fallback with Backoff**: Catch `torch.cuda.OutOfMemoryError`, empty CUDA cache (`torch.cuda.empty_cache()`), cut batch size in half, and retry that chunk.

    **Q3: How does API batching differ from local GPU batching?**
    * **Answer**:
      * Local GPU batching is constrained by **VRAM capacity and tensor dimensions**.
      * API batching (OpenAI, Cohere) is constrained by **HTTP payload limits, TPM (Tokens Per Minute), and RPM (Requests Per Minute)** ceilings.
      * In API batching, packing the maximum allowed number of strings (e.g. 512 or 2,048 strings per HTTP POST) minimizes HTTP connection overhead and TLS handshakes, reducing API latency from hours to minutes.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Forgetting to Restore Input Order after Length Sorting**:
      * *Issue*: Sorting texts by length, encoding them in batches, and saving them directly into a database without restoring original corpus indices.
      * *Consequence*: Total corruption of the vector index! Document A's vector is assigned to Document B's metadata.
      * *Fix*: Always track `(original_index, text)` tuples and scatter vectors back into an array sized to the original corpus length.

    * **Trap 2: Fixed Batch Size on Long Texts Causing Spurious OOMs**:
      * *Issue*: Hardcoding `batch_size = 128` because it worked fine on short sentences, then running the pipeline on 2,000-token PDF chunks.
      * *Consequence*: Instant CUDA OOM on the first long batch because memory scales with `batch_size * max_len^2`.
      * *Fix*: Use token-budget batching: `batch_size = max(1, token_budget // max_seq_len)`.

    * **Trap 3: Exceeding API Payload Size Limits**:
      * *Issue*: Sending 2,048 long paragraphs in a single OpenAI API request.
      * *Consequence*: Request fails with HTTP 413 `Payload Too Large` or exceeds the 8M token per call ceiling.
      * *Fix*: Count tokens client-side with `tiktoken` before packing batches.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Smart Bucketing**
      * *Definition*: Grouping texts by sequence length before batching to minimize wasted padding computations in transformer attention layers.
    * **Flashcard 2: Token-Budget Batching**
      * *Definition*: Sizing batches by total token count rather than text count to guarantee consistent GPU VRAM memory consumption.
    * **Flashcard 3: Padding Tax**
      * *Definition*: The wasteful computation spent calculating self-attention and hidden states over zero-padding tokens.
    * **Flashcard 4: Batch Dynamic Padding**
      * *Definition*: Right-padding texts only up to the maximum sequence length present in the current batch, rather than the model's global limit.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Batch Embeddings:

- [ ] I can articulate why single-sample embedding loops under-utilize GPU tensor cores.
- [ ] I understand how Smart Bucketing (length sorting) eliminates the Padding Tax.
- [ ] I know how to implement an order-preserving bucketing batch encoder in Python.
- [ ] I understand token-budget batching and how it prevents CUDA Out-Of-Memory crashes.
- [ ] I know the differences and constraints between local GPU batching and hosted API batching.
""")
