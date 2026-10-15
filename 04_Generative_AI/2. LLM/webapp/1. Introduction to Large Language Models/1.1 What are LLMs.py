import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="1.1 What are LLMs? - LLM Curriculum",
    page_icon="🧠",
    layout="wide"
)

# ==============================================================================
# Title Block & Overview
# ==============================================================================
st.title("1.1 What are Large Language Models (LLMs)?")
st.caption("Chapter 1: Introduction to Large Language Models | Domain: 2. Large Language Models")
st.markdown("""
A **Large Language Model (LLM)** is a deep neural network—predominantly built on the Transformer decoder architecture—trained 
on massive corpora (hundreds of billions to trillions of tokens) to optimize a single, elegant mathematical objective: 
predicting the probability distribution of the next token in a sequence.

By scaling parameter count, training compute, and dataset volume under empirical scaling laws, LLMs transcend classical 
statistical pattern recognition, displaying emergent capabilities in natural reasoning, code synthesis, cross-lingual translation, 
and autonomous tool execution.

In this lesson, you will master:
1. The mathematical definition of language models and autoregressive next-token prediction.
2. The evolutionary trajectory from statistical N-grams and recurrent networks to decoder-only foundation models.
3. The architectural distinctions separating traditional machine learning, classical NLP, and modern foundation LLMs.
4. Production unit economics, scaling laws, and GPU VRAM calculation across FP16, INT8, and INT4 precisions.
5. Critical engineering antipatterns in context budgeting, KV-cache sizing, and model selection.
""")

st.divider()

# ==============================================================================
# Section 1: End-to-End Autoregressive LLM System Architecture
# ==============================================================================
st.header("1. End-to-End Autoregressive LLM System Architecture", divider="rainbow")

col_arch_a, col_arch_b = col_arch_a, col_arch_b = st.columns([1.2, 0.8])

with col_arch_a:
    dot_heavy_architecture = """
    digraph LLMSystemArchitecture {
        rankdir=TB;
        bgcolor="#0B1120";
        compound=true;
        fontname="Helvetica";
        

        // Global Node Defaults
        node [fontname="Helvetica", fontsize=9, fontcolor="#FFFFFF", penwidth=1.5, margin="0.15,0.08"];

        // Global Edge Defaults
        edge [fontname="Helvetica", fontsize=8, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.2, arrowsize=0.7];

        // 1. Ingress Tier
        subgraph cluster_ingress {
            label="Tier 1: Context Ingress & Tokenization";
            fontname="Helvetica";
            fontsize=10;
            fontcolor="#94A3B8";
            style="dashed";
            color="#334155";
            penwidth=1.2;

            raw_prompt [label="User Prompt Text\\n'The capital of France is'", shape=box, style="filled,rounded", fillcolor="#0891B2", color="#38BDF8"];
            bpe_tokenizer [label="Byte-Level BPE Tokenizer\\n(tiktoken / SentencePiece)", shape=box, style="filled,rounded", fillcolor="#1E40AF", color="#3B82F6"];
            token_ids [label="Token ID Vector\\n[464, 3139, 286, 4881, 318]", shape=box, style="filled,rounded", fillcolor="#0284C7", color="#38BDF8"];
        }

        // 2. Transformer Backbone Tier
        subgraph cluster_backbone {
            label="Tier 2: Transformer Decoder Stack";
            fontname="Helvetica";
            fontsize=10;
            fontcolor="#94A3B8";
            style="rounded";
            color="#334155";
            penwidth=1.2;

            emb_lookup [label="Embedding Table W_emb\\n[Vocab_Size x Hidden_Dim]", shape=cylinder, style="filled", fillcolor="#15803D", color="#22C55E"];
            rope_enc [label="Rotary Position Embedding (RoPE)\\nComplex Coordinate Rotation", shape=box, style="filled,rounded", fillcolor="#5B21B6", color="#8B5CF6"];
            decoder_layers [label="32 to 128 Stacked Decoder Blocks\\n(RMSNorm + GQA Self-Attention + SwiGLU FFN)", shape=box, style="filled,rounded", fillcolor="#6D28D9", color="#A78BFA"];
            final_norm [label="Final RMSNorm Normalization\\nShape: [1, Seq_Len, Hidden_Dim]", shape=box, style="filled,rounded", fillcolor="#5B21B6", color="#8B5CF6"];
        }

        // 3. Logit Projection & Sampling Tier
        subgraph cluster_output {
            label="Tier 3: Unembedding, Sampling & Recurrence";
            fontname="Helvetica";
            fontsize=10;
            fontcolor="#94A3B8";
            style="rounded";
            color="#334155";
            penwidth=1.2;

            lm_head [label="Unembedding LM Head\\nMatrix W_unemb [Hidden_Dim x Vocab_Size]", shape=cylinder, style="filled", fillcolor="#991B1B", color="#EF4444"];
            logits_vec [label="Logits Score Vector\\nShape: [1, Vocab_Size = 128,256]", shape=box, style="filled,rounded", fillcolor="#B45309", color="#F59E0B"];
            softmax_dist [label="Softmax Normalization\\nSum_i P(token_i) = 1.0", shape=box, style="filled,rounded", fillcolor="#B45309", color="#F59E0B"];
            sampler [label="Nucleus Sampler (Temp=0.7, Top-p=0.9)\\nSampled Token: ' Paris' (ID: 6342)", shape=box, style="filled,rounded", fillcolor="#047857", color="#10B981"];
        }

        // Connections
        raw_prompt -> bpe_tokenizer [label="Raw String Stream"];
        bpe_tokenizer -> token_ids [label="Subword Segmentation"];
        token_ids -> emb_lookup [label="Vocab Index Lookup"];
        emb_lookup -> rope_enc [label="Dense Vector [d=4096]"];
        rope_enc -> decoder_layers [label="Positional Vector"];
        decoder_layers -> final_norm [label="Autoregressive Attention"];
        final_norm -> lm_head [label="Last Hidden State h_T"];
        lm_head -> logits_vec [label="Dot Product Scoring"];
        logits_vec -> softmax_dist [label="Exponential Scaling"];
        softmax_dist -> sampler [label="Categorical Distribution"];
        sampler -> raw_prompt [label="Append Token & Repeat Loop", style="dashed", color="#10B981"];
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(dot_heavy_architecture, use_container_width=True)

with col_arch_b:
    st.markdown("""
    ### Tier-by-Tier Architectural Breakdown
    * **Tier 1: Context Ingress & Tokenization**: Ingests UTF-8 text strings and converts them into discrete vocabulary indices using lossless Byte-Pair Encoding.
    * **Tier 2: Transformer Decoder Stack**: Projects integer IDs into continuous dense vectors, applies Rotary Positional Embeddings (RoPE), and evaluates deep self-attention across 32 to 128 stacked decoder blocks.
    * **Tier 3: Unembedding, Sampling & Recurrence**: Projects the final hidden state vector back to vocabulary space via the LM head, calculates Softmax probabilities, samples the next token, and appends it to the prefix context for autoregressive recurrence.
    """)

st.divider()

# ==============================================================================
# Section 2: Definition of Large Language Models (Roadmap Bullet 1)
# ==============================================================================
st.header("2. Definition of Large Language Models", divider="rainbow")

col_b1_a, col_b1_b = col_b1_a, col_b1_b = st.columns([1.2, 0.8])

with col_b1_a:
    st.markdown("""
    ### Foundational Definition
    A **Large Language Model (LLM)** is a statistical foundation model parameterized by billions of trainable weights 
    (typically 7B to 405B+ parameters) optimized on massive multimodal text datasets. Unlike shallow models that predict 
    fixed class labels, an LLM learns the underlying compressed manifold of human language, reasoning, and programming code.

    The operational definition rests on three foundational pillars:
    1. **Parameter Scale**: Billions of interconnected weights capable of storing factual knowledge in feed-forward layers.
    2. **Self-Supervised Pretraining**: Training on unlabeled text by predicting the next token without human annotation.
    3. **General-Purpose Representation**: A single unified model that solves translation, summarization, logic, and coding zero-shot.
    """)

with col_b1_b:
    st.markdown("""
    #### Architectural Dimension Profile (Llama 3 8B)
    * **Parameters**: 8.03 Billion (`fp16` footprint: 16.06 GB)
    * **Layers (Depth)**: 32 stacked Transformer decoder blocks
    * **Hidden Dimension**: 4,096 channels
    * **Attention Heads**: 32 query heads, 8 key-value heads (GQA)
    * **Vocabulary Size**: 128,256 subword tokens
    * **Context Window**: 8,192 to 131,072 tokens
    """)

st.info("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **How New LLMs Handle This**: In modern foundation models (GPT-4, Llama 3, Claude 3.5), models are not treated as specialized text predictors. They serve as **universal cognitive engines** capable of executing multi-turn tool calls, writing executable software, and following structured schema constraints natively.
* **Why Legacy Approach Was Abandoned**: Classical NLP trained isolated models for single tasks (e.g., one model for sentiment, one for grammar). Modern LLMs replaced this fragmentation with a single pre-trained generalist backbone adapted via instruction tuning and RLHF.
* **Production Standard Today**: High-throughput inference engines (vLLM, TensorRT-LLM) serve unified foundation models using PagedAttention and continuous batching.
""")

st.markdown("#### Definition & Scope: Classical NLP vs Modern Foundation LLMs")
comp_b1_df = pd.DataFrame({
    "Dimension": ["Model Scope", "Parameter Horizon", "Supervision Signal", "Inference Delivery"],
    "Classical NLP Approach": ["Narrow task-specific classifier", "Thousands to millions (<100M)", "Manually annotated labels", "Single classification label"],
    "Modern Foundation LLM": ["Universal general-purpose engine", "Billions to hundreds of billions (7B–405B)", "Self-supervised next-token loss", "Open-ended autoregressive token stream"],
    "Engineering Impact": ["Requires managing hundreds of micro-models", "Single backbone handles enterprise workflows", "Eliminates manual dataset annotation bottlenecks", "Enables zero-shot and few-shot reasoning"]
})
st.dataframe(comp_b1_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: What Makes a Model a Language Model? (Roadmap Bullet 2)
# ==============================================================================
st.header("3. What Makes a Model a Language Model?", divider="rainbow")

col_b2_a, col_b2_b = col_b2_a, col_b2_b = st.columns([1.2, 0.8])

with col_b2_a:
    st.markdown("""
    ### The Probabilistic Objective Formulation
    A model qualifies mathematically as a **Language Model (LM)** if and only if it assigns a formal joint probability 
    distribution over sequences of words or tokens:

    `P(W) = P(w_1, w_2, ..., w_T)`

    By applying the probability chain rule, this joint distribution decomposes into the product of conditional probabilities:

    `P(w_1, ..., w_T) = Product_{t=1}^T P(w_t | w_1, ..., w_{t-1})`

    ### Causal Autoregressive Property
    At each step `t`, the language model takes all preceding tokens `w_{<t}` as an unmodifiable prefix context and computes a 
    Softmax probability vector over the entire vocabulary dictionary `V`:

    `P(w_t | w_{<t}) = Softmax(W_unemb * RMSNorm(h_t))`

    Where:
    * `h_t` is the final contextualized hidden vector emitted by the top decoder block for position `t-1`.
    * `W_unemb` is the unembedding projection matrix of shape `[Hidden_Dimension x Vocab_Size]`.
    """)

with col_b2_b:
    st.markdown("""
    #### Causal Attention Mask Property
    To strictly preserve the autoregressive property during parallel GPU training, future tokens are masked out:
    
    ```
    M_{i,j} = 0      if j <= i (Past & Present)
    M_{i,j} = -inf   if j > i  (Future Masked)
    ```
    
    This triangular mask ensures token `i` can never peek at token `i+1`, enabling $O(1)$ parallel training across the entire sequence.
    """)

st.info("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **How New LLMs Handle This**: While bidirectional models like BERT masked 15% of tokens in the middle (`MLM`), modern foundation LLMs (Llama 3, Mistral, GPT-4) are almost exclusively **Causal Decoder-Only Models (`CLM`)**.
* **Why Legacy Approach Was Abandoned**: Bidirectional masked models cannot generate free-form text efficiently; generating 100 tokens requires 100 full re-evaluations without KV-caching. Causal decoder models naturally generate text autoregressively while reusing key-value attention caches.
* **Production Standard Today**: Autoregressive causal decoders paired with static Key-Value (KV) caching in GPU VRAM deliver sub-20ms per-token latency.
""")

st.markdown("#### Language Model Architecture Paradigms")
comp_b2_df = pd.DataFrame({
    "Architecture Type": ["Causal Decoder-Only (GPT, Llama)", "Bidirectional Encoder (BERT)", "Encoder-Decoder (T5, BART)"],
    "Attention Matrix": ["Lower-triangular causal mask", "Full bidirectional visibility", "Bidirectional encoder + Causal decoder"],
    "Primary Objective": ["Autoregressive next-token prediction", "Masked token reconstruction (MLM)", "Seq2Seq conditional reconstruction"],
    "Generation Efficiency": ["High (native KV-cache reuse)", "Incapable of autoregressive generation", "Moderate (cross-attention overhead)"]
})
st.dataframe(comp_b2_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: Evolution of Language Models (Roadmap Bullet 3)
# ==============================================================================
st.header("4. Evolution of Language Models", divider="rainbow")

st.markdown("""
The transition from hand-crafted statistical tables to multi-trillion-token foundation models marks the most rapid acceleration 
in computational history. The timeline below illustrates the four major epochs:
""")

dot_timeline = """
digraph EvolutionTimeline {
    rankdir=TB;
    bgcolor="#0B1120";
    ranksep=0.25;
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded", margin="1.25,0.06"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    subgraph cluster_era1 {
        label="Epoch 1: 1990s - 2012\\nStatistical NLP";
        bgcolor="#1E293B";
        fontcolor="#EF4444";
        color="#334155";
        e1 [label="N-gram Frequency Tables\\n* Markov Assumption (N=3 to 5)\\n* Catastrophic data sparsity\\n* Zero semantic generalization", fillcolor="#450A0A", fontcolor="#FCA5A5", color="#EF4444"];
    }

    subgraph cluster_era2 {
        label="Epoch 2: 2013 - 2017\\nStatic Vectors & Recurrence";
        bgcolor="#1E293B";
        fontcolor="#F59E0B";
        color="#334155";
        e2 [label="Word2Vec, GloVe, LSTMs, GRUs\\n* Dense geometric vector space\\n* Sequential recurrent hidden state\\n* Vanishing gradients on long contexts", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
    }

    subgraph cluster_era3 {
        label="Epoch 3: 2017 - 2021\\nTransformer Revolution";
        bgcolor="#1E293B";
        fontcolor="#3B82F6";
        color="#334155";
        e3 [label="Self-Attention (Vaswani et al.)\\n* BERT (Encoder) & GPT-2/3 (Decoder)\\n* Parallel GPU training\\n* Scaling to 175B parameters", fillcolor="#1E3A8A", fontcolor="#DBEAFE", color="#3B82F6"];
    }

    subgraph cluster_era4 {
        label="Epoch 4: 2022 - Present\\nFrontier Foundation Models";
        bgcolor="#1E293B";
        fontcolor="#10B981";
        color="#334155";
        e4 [label="Llama 3, GPT-4, DeepSeek, Claude\\n* 15+ Trillion Training Tokens\\n* Instruction Tuning & RLHF / DPO\\n* Emergent multi-step reasoning", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
    }

    e1 -> e2 [label="   Distributed Vectors   "];
    e2 -> e3 [label="   Self-Attention   "];
    e3 -> e4 [label="   Scale + Alignment   "];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.graphviz_chart(dot_timeline, use_container_width=True)

st.info("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **How New LLMs Handle This**: Modern LLMs (Llama 3, Claude 3.5, Gemini 1.5) operate in the **Frontier Foundation Epoch**. Models are trained on over 15 Trillion tokens and feature context windows extending from 128k up to 1M+ tokens.
* **Why Legacy Approach Was Abandoned**: N-grams suffered from combinatorial explosion for contexts beyond 5 words. Recurrent models (LSTMs) could not be parallelized across sequence dimensions on modern GPU clusters.
* **Production Standard Today**: Decoder-only Transformers using FlashAttention-2 kernels and Grouped-Query Attention (GQA) for ultra-low latency inference.
""")

st.markdown("#### Evolutionary Comparison Across Epochs")
timeline_df = pd.DataFrame({
    "Epoch": ["1. Statistical N-grams", "2. Recurrent Networks (LSTM)", "3. Early Transformers (BERT/GPT-2)", "4. Modern Foundation Models (Llama 3)"],
    "Context Capacity": ["3 to 5 words", "50 to 100 tokens", "512 to 2,048 tokens", "128,000 to 1,000,000+ tokens"],
    "Training Parallelism": ["None (Counting tables)", "Sequential O(T) recurrent steps", "Parallel O(1) self-attention steps", "Thousands of GPUs via Megatron/FSDP"],
    "Generalization Ability": ["Zero (Exact match only)", "Weak (Fixed word vectors)", "Moderate (Task fine-tuning)", "High (Zero-shot in-context reasoning)"]
})
st.dataframe(timeline_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: NLP vs LLMs (Roadmap Bullet 4)
# ==============================================================================
st.header("5. NLP vs Large Language Models", divider="rainbow")

col_b4_a, col_b4_b = col_b4_a, col_b4_b = st.columns([1.2, 0.8])

with col_b4_a:
    st.markdown("""
    ### The Paradigm Shift: Subfield vs Generative Backbone
    **Natural Language Processing (NLP)** is the overarching computer science discipline concerned with computational 
    interactions between computers and human language. **Large Language Models (LLMs)** represent a specific, modern deep learning 
    paradigm within NLP that has largely subsumed traditional NLP pipelines.

    Historically, NLP was divided into dozens of isolated sub-tasks:
    * Part-of-speech (POS) tagging
    * Named Entity Recognition (NER)
    * Dependency parsing
    * Coreference resolution
    * Text classification and sentiment analysis

    In traditional NLP, an engineering team had to build, train, deploy, and monitor separate machine learning pipelines 
    for each sub-task. If the upstream POS tagger made an error, that error cascaded through the downstream dependency parser, 
    causing the final entity extraction to fail catastrophically.
    """)

with col_b4_b:
    st.markdown("""
    #### The Unified LLM Interface
    Modern foundation LLMs collapse this entire multi-model cascade into a **single prompt interface**:
    
    ```
    Prompt: "Extract entities and sentiment from:
    'Apple announced record Q4 earnings in Cupertino.'"
    
    Response: {
      "entities": [{"name": "Apple", "type": "ORG"}, {"name": "Cupertino", "type": "GPE"}],
      "sentiment": "POSITIVE"
    }
    ```
    One single model forward pass handles extraction, classification, and formatting simultaneously.
    """)

st.info("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **How New LLMs Handle This**: Modern LLMs treat all classical NLP sub-tasks as **conditional text-to-text generation**. By providing natural language instructions or few-shot exemplars, a single foundation model solves tasks that previously required 10 separate specialized models.
* **Why Legacy Approach Was Abandoned**: Cascaded NLP pipelines were brittle, high-maintenance, and suffered from compounding upstream errors. Maintaining 15 distinct microservices for individual NLP tasks is economically unfeasible compared to querying a foundation model API.
* **Production Standard Today**: Structured outputs via constrained JSON decoding (Instructor, Outlines) replacing specialized NER and classification models.
""")

st.markdown("#### Classical Multi-Task NLP vs Modern Unified LLM Architecture")
nlp_vs_llm_df = pd.DataFrame({
    "Criterion": ["System Architecture", "Deployment Footprint", "Handling of Novel Domains", "Error Propagation"],
    "Traditional NLP Pipeline": ["Multi-stage cascade of specialized models", "10+ microservices per task", "Fails without domain retraining", "Upstream errors cascade fatally"],
    "Modern Foundation LLM": ["Unified single foundation model", "Single inference API endpoint", "Strong zero-shot generalization", "End-to-end holistic self-attention"],
    "Business Impact": ["High engineering maintenance cost", "Low operational overhead", "Rapid time-to-market for new features", "Drastically higher accuracy"]
})
st.dataframe(nlp_vs_llm_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: Traditional NLP vs Deep Learning NLP (Roadmap Bullet 5)
# ==============================================================================
st.header("6. Traditional NLP vs Deep Learning NLP", divider="rainbow")

col_b5_a, col_b5_b = col_b5_a, col_b5_b = st.columns([1.1, 0.9])

with col_b5_a:
    st.markdown("""
    ### Hand-Crafted Features vs Representation Learning
    The evolution from Traditional NLP to Deep Learning NLP represents the shift from human-engineered symbolic heuristics 
    to autonomous representation learning in continuous vector space.

    #### Traditional NLP (1990s - 2012):
    * **Representations**: Discrete, sparse high-dimensional matrices (Bag of Words, TF-IDF).
    * **Heuristics**: Rule-based grammars, regex patterns, Gazetteers (name dictionaries), Porter stemming.
    * **Orthogonal Vectors**: The word `"cat"` and `"feline"` were represented as orthogonal basis vectors with `Cosine Similarity = 0.0`. The system had zero understanding that they share semantic meaning.

    #### Deep Learning NLP (2013 - Present):
    * **Representations**: Dense continuous latent embeddings (e.g. 768 to 16,384 dimensions).
    * **Geometric Geometry**: Words with similar meanings cluster together in geometric space (`Cosine Similarity("cat", "kitten") > 0.85`).
    * **Contextual Polysemy**: Self-attention layers compute dynamic representations where `"apple"` (fruit) has a different vector than `"Apple"` (tech corporation) based on surrounding tokens.
    """)

with col_b5_b:
    st.markdown("""
    #### Representation Geometry Contrast
    ```
    Traditional NLP (One-Hot / TF-IDF):
    cat:    [1, 0, 0, 0, 0, ..., 0]
    feline: [0, 0, 0, 1, 0, ..., 0]
    Cosine Similarity = 0.00 (Zero Semantic Link)

    Deep Learning NLP (Dense Embeddings):
    cat:    [+0.21, -0.45, +0.88, ..., +0.12]
    feline: [+0.19, -0.42, +0.85, ..., +0.15]
    Cosine Similarity = 0.94 (Dense Geometric Affinity)
    ```
    """)

st.info("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **How New LLMs Handle This**: Modern LLMs take Deep Learning NLP to its frontier: they completely discard static word embeddings (like Word2Vec) in favor of **multi-layer contextualized latent states**. Each token's representation is transformed across 32 to 128 self-attention layers.
* **Why Legacy Approach Was Abandoned**: Traditional NLP's sparse matrices failed to capture word order, compositionality, or polysemy. Manual feature engineering hit a performance plateau that could not scale with internet-scale data.
* **Production Standard Today**: Pre-trained Transformer backbones with billions of parameters that autonomously learn syntax, morphology, world facts, and code logic directly from raw tokens.
""")

st.markdown("#### Technical Comparison: Traditional NLP vs Deep Learning NLP")
trad_vs_dl_df = pd.DataFrame({
    "Attribute": ["Vector Representation", "Semantic Understanding", "Polysemy Handling", "Word Order Awareness"],
    "Traditional NLP": ["Sparse, orthogonal (TF-IDF, BoW)", "Zero geometric affinity", "Fails (single static bin per word)", "Completely discarded (Bag of Words)"],
    "Deep Learning NLP": ["Dense continuous vectors (d=4096)", "High cosine similarity for synonyms", "Context-dependent dynamic vectors", "Explicitly preserved via Positional Encodings (RoPE)"],
    "Scalability": ["Plateaus on large datasets", "Continuously improves with compute and data", "Learns cross-lingual representations", "Enables programmatic reasoning"]
})
st.dataframe(trad_vs_dl_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: Foundation Models (Roadmap Bullet 6)
# ==============================================================================
st.header("7. Foundation Models: The Generalist Paradigm", divider="rainbow")

col_b6_a, col_b6_b = col_b6_a, col_b6_b = st.columns([1.2, 0.8])

with col_b6_a:
    st.markdown("""
    ### Definition & The Stanford Taxonomy
    Coined by the Stanford Institute for Human-Centered Artificial Intelligence (HAI) in 2021, the term **Foundation Model** 
    describes any massive neural network trained on broad, multimodal data at scale that can be adapted to a wide range of 
    downstream applications.

    Key architectural properties of Foundation Models:
    1. **Scale**: Trained on unprecedented compute budgets (`10^24` to `10^26` FLOPs) across trillions of diverse tokens.
    2. **Self-Supervised Generality**: Pretrained on broad raw data without task-specific labels, capturing generalized world knowledge.
    3. **Adaptability (Transfer Learning)**: Serves as a central bedrock adapted downstream via:
       * **In-Context Prompting**: Zero-shot and few-shot task specification.
       * **Instruction Fine-Tuning**: Supervised alignment on conversational instructions.
       * **Parameter-Efficient Fine-Tuning (PEFT / LoRA)**: Freezing base weights and training low-rank adapter matrices.
    """)

with col_b6_b:
    st.markdown("""
    #### The Foundation Model Tree
    ```
                    [ Foundation Model ]
               (e.g., Llama 3 / GPT-4 / Claude)
                              │
         ┌────────────┬───────┴───────┬────────────┐
         ▼            ▼               ▼            ▼
     [Coding]   [Legal RAG]      [Healthcare]  [Robotics]
    (SQL, Py)   (Doc Search)     (Diagnostics)  (Actions)
    ```
    A single foundation model replaces dozens of specialized domain models.
    """)

st.info("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **How New LLMs Handle This**: In modern enterprise AI, companies rarely train foundation models from scratch due to prohibitive costs ($10M–$100M+ per training run). Instead, organizations deploy open-weight foundation models (Llama 3, Mistral) and adapt them via Retrieval-Augmented Generation (RAG) or LoRA adapters.
* **Why Legacy Approach Was Abandoned**: Training separate models from scratch for every enterprise domain resulted in massive engineering overhead, high infrastructure bills, and brittle models lacking general reasoning.
* **Production Standard Today**: Freezing the foundation model weights and attaching domain-specific LoRA adapters (`rank=16`, `alpha=32`) or grounding via Vector Databases.
""")

st.markdown("#### Foundation Model Adaptation Spectrum")
adapt_df = pd.DataFrame({
    "Adaptation Method": ["Prompt Engineering", "RAG (Retrieval-Augmented)", "PEFT / LoRA Fine-Tuning", "Full Pre-Training"],
    "Compute Cost": ["$0 (API cost only)", "Low (Vector DB indexing)", "Moderate ($50 - $500 GPU compute)", "Astronomical ($1M - $100M+)"],
    "Data Requirements": ["Zero to few examples", "Unstructured enterprise documents", "Hundreds to thousands of prompt/response pairs", "Trillions of high-quality tokens"],
    "Knowledge Freshness": ["Static (Training cutoff)", "Real-time dynamic updates", "Static (Embedded in adapter weights)", "Static at time of training"]
})
st.dataframe(adapt_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: Generative AI vs LLMs (Roadmap Bullet 7)
# ==============================================================================
st.header("8. Generative AI vs Large Language Models", divider="rainbow")

col_b7_a, col_b7_b = col_b7_a, col_b7_b = st.columns([1.1, 0.9])

with col_b7_a:
    st.markdown("""
    ### Taxonomic Relationship: Ecosystem vs Core Engine
    **Generative Artificial Intelligence (GenAI)** is the broad umbrella discipline encompassing all AI systems that generate 
    novel content across any modality. **Large Language Models (LLMs)** represent the text-focused, autoregressive engine that 
    serves as the primary reasoning core of the Generative AI ecosystem.

    #### The Generative AI Modality Spectrum:
    * **Text & Code Generation**: Powered natively by Large Language Models (GPT-4, Llama 3, Claude).
    * **Image Generation**: Powered by Diffusion Models and Flow Matching (Stable Diffusion, Midjourney, FLUX).
    * **Audio & Speech Synthesis**: Powered by Neural Audio Codecs and Diffusion (Whisper, ElevenLabs).
    * **Video Generation**: Powered by Spatio-Temporal Diffusion Transformers (Sora, Runway Gen-3).
    * **3D Asset Generation**: Powered by NeRFs (Neural Radiance Fields) and Gaussian Splatting.
    """)

with col_b7_b:
    st.markdown("""
    #### LLM as the Central Reasoning Core
    ```
    ┌────────────────────────────────────────────────────────┐
    │              Generative AI Ecosystem                   │
    │                                                        │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │       Large Language Models (Reasoning Core)     │  │
    │  │   Text Generation · Code · Logic · Planning      │  │
    │  └────────────────────────┬─────────────────────────┘  │
    │                           │ Orchestration              │
    │         ┌─────────────────┼─────────────────┐          │
    │         ▼                 ▼                 ▼          │
    │    [Diffusion]       [Speech TTS]      [Video DiT]     │
    │     (Images)           (Audio)           (Video)       │
    └────────────────────────────────────────────────────────┘
    ```
    """)

st.info("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **How New LLMs Handle This**: Frontier LLMs are rapidly converging into **Omni-Modal Foundation Models** (e.g. GPT-4o, Gemini 1.5 Pro). Rather than processing text alone, the model natively ingests and emits interleaved text, audio, video frames, and code tokens within a single unified Transformer architecture.
* **Why Legacy Approach Was Abandoned**: Separate modality silos (e.g. chaining Speech-to-Text -> LLM -> Text-to-Speech) introduced 1,000ms+ latency penalties and lost vocal inflections, emotional cadence, and visual nuances.
* **Production Standard Today**: Native multimodal tokenizers that project image patches and audio spectrograms directly into the LLM's continuous hidden dimension space.
""")

st.markdown("#### Generative AI Architecture Modalities")
genai_df = pd.DataFrame({
    "Modality": ["Text & Code", "Image Synthesis", "Video Generation", "Audio & Voice", "Omni-Modal"],
    "Dominant Architecture": ["Autoregressive Transformer Decoder", "Latent Diffusion / Flow Matching", "Diffusion Transformer (DiT)", "Neural Codec / Flow Matching", "Unified Multimodal Transformer"],
    "Input Representation": ["Discrete Subword Tokens", "Latent VAE Feature Patches", "Spatio-Temporal Video Patches", "Mel-Spectrogram Audio Frames", "Interleaved Cross-Modal Tokens"],
    "Flagship Models": ["Llama 3, Claude 3.5 Sonnet", "FLUX.1, Midjourney v6", "Sora, Runway Gen-3", "ElevenLabs, CosyVoice", "GPT-4o, Gemini 1.5 Pro"]
})
st.dataframe(genai_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: LLMs vs Traditional Machine Learning Models (Roadmap Bullet 8)
# ==============================================================================
st.header("9. LLMs vs Traditional Machine Learning Models", divider="rainbow")

col_b8_a, col_b8_b = col_b8_a, col_b8_b = st.columns([1.1, 0.9])

with col_b8_a:
    st.markdown("""
    ### Tabular Narrow Models vs In-Context Reasoners
    Machine learning practitioners must carefully choose between **Traditional Machine Learning** (e.g. Gradient Boosted 
    Trees, Random Forests, Logistic Regression) and **Large Language Models** based on data modalities, latency requirements, 
    and economic cost.

    #### Traditional Machine Learning:
    * **Optimal Modality**: Structured tabular databases (SQL rows, financial transactions, sensor logs).
    * **Latency**: Ultra-low latency inference (0.1ms to 5ms on standard CPU cores).
    * **Cost**: Negligible compute cost; models train in seconds to minutes.
    * **Interpretability**: Feature importance trees (SHAP, Gain, Split counts) provide exact mathematical audits.

    #### Large Language Models:
    * **Optimal Modality**: Unstructured text, conversational dialogues, source code, multimodal media.
    * **Latency**: High sequential latency (20ms to 50ms per generated token; 500ms–2000ms per request).
    * **Cost**: Expensive GPU VRAM footprints (A100/H100 clusters) and high electricity costs.
    * **Interpretability**: Opaque deep neural latent activations; requires mechanistic interpretability probes.
    """)

with col_b8_b:
    st.markdown("""
    #### Enterprise Architecture Decision Guide
    ```
    Is the data structured tabular columns (numbers, categories)?
    ├── YES ──> Use XGBoost / LightGBM / CatBoost
    │           (Faster, 100x cheaper, higher tabular accuracy)
    └── NO (Free-form text, code, complex reasoning)
         └── Is task simple extraction or complex synthesis?
              ├── Simple Extraction ──> Small fine-tuned model / LLM
              └── Complex Synthesis ──> Frontier Foundation LLM
    ```
    Never replace an XGBoost model on fraud detection tables with an LLM!
    """)

st.info("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **How New LLMs Handle This**: In production enterprise architecture, modern systems use **Hybrid Ensembles**: Traditional ML (XGBoost) processes tabular records and triggers business alerts, while an LLM (Llama 3) generates natural language explanations and communicates with customers.
* **Why Legacy Approach Was Abandoned**: Early GenAI hype led companies to mistakenly apply LLMs to numerical tabular prediction, resulting in severe latency regressions, high cloud bills, and lower accuracy than tree-based ensembles.
* **Production Standard Today**: GBDT models for tabular classification and regression; LLMs for unstructured language understanding, summarization, and human-facing dialogue.
""")

st.markdown("#### Comprehensive Comparison: Traditional ML vs Foundation LLMs")
trad_ml_df = pd.DataFrame({
    "Criterion": ["Data Modality", "Hardware Requirement", "Inference Latency", "Training Duration", "Primary Strength"],
    "Traditional ML (XGBoost, SVM)": ["Structured tabular / numerical matrices", "Standard CPU cores", "0.5ms to 5ms", "Minutes to hours", "Unbeatable accuracy on tabular business records"],
    "Foundation LLMs (Llama 3, GPT-4)": ["Unstructured text, code, multimodal media", "Enterprise GPUs (A100 / H100)", "20ms - 50ms per token", "Months on supercomputer clusters", "General reasoning, summarization, and code synthesis"],
    "Enterprise Recommendation": ["Use for fraud, credit scoring, churn", "Avoid for tabular; use for NLP/AI agents", "Direct deployment on edge devices", "Deploy via specialized inference servers", "Combine via hybrid architecture"]
})
st.dataframe(trad_ml_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 10: Production Python Implementation: Text Generation via Hugging Face
# ==============================================================================
st.header("10. Production Python Implementation: Text Generation via Hugging Face", divider="rainbow")

st.markdown("""
The following runnable snippet demonstrates how production systems interact with foundation models using the official 
Hugging Face `transformers` library, illustrating device placement, data types, and decoding configurations:
""")

prod_code = '''from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Specify foundation model repository
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# 2. Inspect device capability (CUDA GPU vs Apple MPS vs CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# 3. Load tokenizer with automatic padding configuration
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Load causal decoder model with optimized memory mapping
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map="auto"
)

# 5. Formulate structured conversational prompt
messages = [
    {"role": "system", "content": "You are a principal AI systems architect."},
    {"role": "user", "content": "Explain why decoder-only models dominate generative AI."}
]

prompt_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

# 6. Execute autoregressive generation with decoding hyperparameters
with torch.no_grad():
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# 7. Decode generated output tokens back to human-readable string
response_ids = generated_ids[0][prompt_ids.shape[1]:]
response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
print("Generated Architect Response:")
print(response_text)'''

st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# Section 11: Interactive Workbench: LLM Parameter Scale & VRAM Simulator
# ==============================================================================
st.header("11. Interactive Workbench: LLM Parameter Scale & VRAM Simulator", divider="rainbow")

st.markdown("""
Calculate the exact GPU VRAM footprint required to load and serve LLMs across common parameter scales, quantization 
formats, and serving context windows.
""")

col_sim_in, col_sim_out = col_sim_in, col_sim_out = st.columns([1, 1.2])

with col_sim_in:
    st.subheader("Model Configuration Panel")
    
    preset_model = st.selectbox(
        "Select Benchmark Model Preset or Custom:",
        [
            "Llama 3 8B (Compact Foundation)",
            "Mistral NeMo 12B",
            "Llama 3 70B (Enterprise Workhorse)",
            "DeepSeek-V2 236B (MoE Architecture)",
            "Llama 3 405B (Frontier Giant)",
            "Custom Parameter Size"
        ]
    )
    
    if "8B" in preset_model:
        default_params = 8.0
    elif "12B" in preset_model:
        default_params = 12.0
    elif "70B" in preset_model:
        default_params = 70.0
    elif "236B" in preset_model:
        default_params = 236.0
    elif "405B" in preset_model:
        default_params = 405.0
    else:
        default_params = 14.0
        
    param_count = st.number_input("Active Parameters (Billions):", min_value=0.5, max_value=1000.0, value=default_params, step=1.0)
    
    precision_format = st.selectbox(
        "Weight Quantization Precision Format:",
        [
            "FP16 / BF16 (16-bit: 2.0 Bytes / Parameter)",
            "INT8 (8-bit: 1.0 Byte / Parameter)",
            "INT4 / AWQ / GPTQ (4-bit: 0.5 Bytes / Parameter)"
        ]
    )
    
    context_window = st.select_slider(
        "Serving Context Window (Tokens):",
        options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
        value=8192
    )

with col_sim_out:
    st.subheader("Hardware & VRAM Telemetry")
    
    bytes_per_param = 2.0 if "FP16" in precision_format else (1.0 if "INT8" in precision_format else 0.5)
    raw_weights_gb = (param_count * 1e9 * bytes_per_param) / (1024 ** 3)
    
    # Estimate KV-Cache and runtime activation overhead based on context window
    kv_cache_gb = (2 * param_count * 0.05 * context_window * 2) / (1024 ** 2 * 10)
    total_serving_vram_gb = raw_weights_gb * 1.20 + kv_cache_gb
    
    met1, met2, met3 = st.columns(3)
    with met1:
        st.metric("Raw Model Weights", f"{raw_weights_gb:.1f} GB")
    with met2:
        st.metric("KV-Cache Overhead", f"{kv_cache_gb:.2f} GB")
    with met3:
        st.metric("Recommended VRAM", f"{total_serving_vram_gb:.1f} GB")
        
    st.markdown("#### Hardware Deployment Viability Tier")
    if total_serving_vram_gb <= 16:
        st.success(f"🟢 **Consumer Single-GPU**: Fits on an RTX 4080 (16 GB) or Apple Mac M-Series (16+ GB unified memory).")
    elif total_serving_vram_gb <= 24:
        st.info(f"🟡 **Enthusiast Workstation**: Fits on a single RTX 3090 / 4090 (24 GB VRAM).")
    elif total_serving_vram_gb <= 80:
        st.warning(f"🟠 **Enterprise Single-Node**: Requires a single NVIDIA A100 / H100 (80 GB VRAM).")
    else:
        gpus_needed = int(np.ceil(total_serving_vram_gb / 80))
        st.error(f"🔴 **Multi-GPU Cluster Required**: Requires at least {gpus_needed}x NVIDIA H100 (80GB) GPUs using Tensor Parallelism (TP={gpus_needed}).")

    st.markdown("#### VRAM Allocation Breakdown")
    vram_breakdown = pd.DataFrame({
        "Component": ["Model Weights (Static)", "KV-Cache (Context Dependent)", "Activation Buffer Overhead", "CUDA Runtime Context"],
        "Memory (GB)": [f"{raw_weights_gb:.2f}", f"{kv_cache_gb:.2f}", f"{(raw_weights_gb * 0.15):.2f}", f"{(raw_weights_gb * 0.05):.2f}"],
        "Description": ["Static neural network layers in VRAM", "Cached key-value attention tensors", "Scratchpad memory for backward/forward math", "GPU kernel runtime and CUDA context"]
    })
    st.dataframe(vram_breakdown, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 12: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("12. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the fundamental difference between Masked Language Modeling (BERT) and Causal Language Modeling (GPT)?"):
    st.markdown("""
    **Answer**:
    * **Masked Language Modeling (MLM - BERT / Encoder)**: Uses bidirectional self-attention where tokens can attend to both left and right context simultaneously. 15% of tokens are corrupted with `[MASK]`. Ideal for extraction, classification, and embeddings, but mathematically incapable of efficient autoregressive text generation.
    * **Causal Language Modeling (CLM - GPT / Decoder)**: Enforces a strictly lower-triangular causal attention mask that prevents any token from attending to future positions. Trains solely on predicting token `t+1` given tokens `1` through `t`. Perfectly aligned with open-ended generation, reasoning, and conversational dialogue.
    """)

with st.expander("Q2: What are the Chinchilla Scaling Laws, and why did they invalidate previous frontier model scaling practices?"):
    st.markdown("""
    **Answer**:
    * **The Finding**: Research by DeepMind (Hoffmann et al., 2022) proved that model parameter size and training token volume should scale in equal proportions (1:1) for compute-optimal training.
    * **The Flaw in Earlier Models**: Models like GPT-3 (175B parameters trained on only 300B tokens) were severely token-starved.
    * **Modern Industry Practice**: Modern foundation models (e.g., Llama 3 8B trained on 15 Trillion tokens) prioritize massive token overtraining. This produces highly compressed, high-performance compact weights that are significantly cheaper to serve at inference time.
    """)

with st.expander("Q3: Why are modern frontier LLMs predominantly Decoder-Only rather than Encoder-Decoder or Encoder-Only?"):
    st.markdown("""
    **Answer**:
    * **Simplicity & Unification**: Decoder-only architectures unify pre-training, fine-tuning, in-context learning, and sampling into a single uninterrupted sequence-to-sequence paradigm.
    * **Serving Throughput & KV-Caching**: Autoregressive decoding benefits heavily from static Key-Value (KV) caching kernels. Having a single causal stack simplifies GPU memory management and multi-query attention kernels compared to cross-attention encoder-decoder structures.
    * **Zero-Shot Task Generalization**: Conditioning solely on causal text prefixes enables generalized in-context few-shot learning across any arbitrary task without architectural task heads.
    """)

with st.expander("Q4: How does weight quantization (e.g., FP16 down to INT4) alter memory requirements and model latency?"):
    st.markdown("""
    **Answer**:
    * **Memory Reduction**: Reduces the bytes per parameter from 2.0 (FP16) to 0.5 (INT4), decreasing static model weight VRAM by 75%.
    * **Memory Bandwidth Bound**: LLM token generation is strictly memory-bandwidth bound (reading weights from GPU VRAM to compute cores). Because INT4 weights require moving only 25% of the data volume across the memory bus, generation throughput often increases substantially.
    * **Quantization Trade-off**: Aggressive quantization can introduce perplexity degradation or outlier weight clipping unless mitigated by modern quantization algorithms (AWQ, GPTQ, SmoothQuant).
    """)

with st.expander("Q5: What constitutes an 'Emergent Ability' in Large Language Models, and what causes it?"):
    st.markdown("""
    **Answer**:
    * **Definition**: An ability that is not present in smaller models (near zero accuracy) but appears abruptly when model scale, dataset size, and compute surpass critical thresholds. Examples include multi-step mathematical reasoning, program synthesis, and theory-of-mind tests.
    * **Mechanistic Cause**: Emergent behavior is often explained by multi-step reasoning: if solving a complex problem requires 5 distinct logical sub-steps and a small model has a 30% chance on each step, its end-to-end success is `0.30^5 = 0.2%` (appearing as zero). Once a scaled model achieves 85% accuracy per step, end-to-end success jumps to `0.85^5 = 44%`, creating the perception of sudden emergent capability.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Conflating Parameter Count with Model Intelligence**  
    Assuming an 80B model is inherently smarter than an 8B model. A modern 8B model trained on 15 Trillion curated tokens routinely outperforms an undertrained 70B model trained on 1 Trillion tokens while consuming 10x less GPU inference RAM.
    """)
    st.warning("""
    **Trap 2: Ignoring KV-Cache Memory Overhead**  
    Budgeting VRAM solely for static model weights without accounting for context length. Serving long contexts (e.g. 64k or 128k tokens) requires tens of gigabytes of dynamic VRAM for KV-cache tensors, leading to unexpected Out-of-Memory (OOM) crashes during peak concurrent user requests.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Using Encoder Models for Autoregressive Text Generation**  
    Attempting to use BERT or RoBERTa for conversational chat or text completion. Encoder models use bidirectional attention and are mathematically structured for classification, NER, and embeddings—not autoregressive generation.
    """)
    st.warning("""
    **Trap 4: Naive Precision Casting Without Outlier Awareness**  
    Applying naive INT8/INT4 rounding without outlier-preserving algorithms. Frontier LLMs feature systematic outlier activations in specific hidden channels; naive truncation collapses perplexity into complete model degradation.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: Autoregressive Model?"):
        st.markdown("""
        **Autoregressive Model**:  
        A model that generates text sequence by sequence, using its own previously predicted tokens as the input context for all subsequent prediction steps.
        """)
    with st.expander("🗂️ Flashcard: Foundation Model?"):
        st.markdown("""
        **Foundation Model**:  
        A massive neural network trained on broad data at scale that serves as a generalized base adapted to hundreds of diverse downstream applications via prompting or PEFT.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Chinchilla Scaling?"):
        st.markdown("""
        **Chinchilla Scaling Laws**:  
        The empirical discovery that compute-optimal LLM training requires scaling model parameter count and token volume in equal 1:1 proportion.
        """)
    with st.expander("🗂️ Flashcard: Emergent Abilities?"):
        st.markdown("""
        **Emergent Abilities**:  
        Capabilities (e.g., complex code generation, multi-step math) that appear abruptly as model scale crosses critical compute and dataset thresholds.
        """)

st.divider()

# ==============================================================================
# Section 13: Self-Assessment Checklist (Mandatory Final Section — No Trailing Divider)
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the mathematical autoregressive next-token objective of causal language models.", value=True)
chk2 = st.checkbox("I can trace the evolutionary timeline from statistical N-grams to modern foundation models.", value=True)
chk3 = st.checkbox("I know the architectural differences between Traditional ML, Classical NLP, and Foundation LLMs.", value=True)
chk4 = st.checkbox("I understand Chinchilla scaling laws and why token volume is as critical as parameter count.", value=True)
chk5 = st.checkbox("I can calculate the static weight and dynamic KV-cache VRAM requirements for serving LLMs at FP16, INT8, and INT4.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 1.2: Why LLMs are Important!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered 1.1 What are Large Language Models. Ready for Chapter 1.2: Why LLMs are Important!")
