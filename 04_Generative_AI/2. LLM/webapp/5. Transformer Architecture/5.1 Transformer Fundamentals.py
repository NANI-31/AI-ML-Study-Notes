import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="5.1 Transformer Fundamentals (Encoder, Decoder, Seq2Seq, Architectures)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MASTER STYLING ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #94A3B8;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .takeaway-box {
        background-color: #0B132B;
        border-left: 4px solid #38BDF8;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #1C1917;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .modern-callout {
        background-color: #042F2E;
        border-left: 4px solid #14B8A6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<div class="main-title">5.1 Transformer Fundamentals: Architecture, Encoders, Decoders, and Seq2Seq</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From the Original 2017 Vaswani Blueprint to Modern Causal Decoder Supremacy — The Macro-Architectural Taxonomy of Self-Attention Systems.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Universal Transformer Family Taxonomy", divider="rainbow")

st.markdown("""
Published in June 2017 by Ashish Vaswani et al. at Google Brain and Google Research, 
*"Attention Is All You Need"* eliminated recurrent loops (RNNs, LSTMs, GRUs) and temporal convolutions (CNNs), 
replacing them entirely with **parallelizable dot-product self-attention mechanisms**.
Over the subsequent decade, this singular foundational breakthrough branched into three major structural families:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.4", ranksep="0.5")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    # Nodes
    arch_graph.node("ROOT", "Attention Is All You Need (2017)\nFull Encoder-Decoder Transformer Stack\n(Machine Translation German -> English)", shape="box", fillcolor="#312E81", color="#818CF8")

    with arch_graph.subgraph(name="cluster_enc") as c:
        c.attr(bgcolor="#0F172A", label="Encoder-Only Family (Bidirectional)", fontcolor="#38BDF8", color="#38BDF8")
        c.node("BERT", "BERT / RoBERTa / DeBERTa\nBidirectional Full Attention Matrix\nTask: Classification, NER, Embedding", shape="box", fillcolor="#1E293B", color="#38BDF8")

    with arch_graph.subgraph(name="cluster_encdec") as c:
        c.attr(bgcolor="#0F172A", label="Encoder-Decoder Family (Seq2Seq)", fontcolor="#A855F7", color="#A855F7")
        c.node("T5", "T5 / BART / Whisper\nBidirectional Encoder + Causal Decoder\nCross-Attention Bridge\nTask: Translation, Audio-to-Text", shape="box", fillcolor="#1E293B", color="#A855F7")

    with arch_graph.subgraph(name="cluster_dec") as c:
        c.attr(bgcolor="#0F172A", label="Decoder-Only Family (Causal Generative)", fontcolor="#10B981", color="#10B981")
        c.node("GPT", "GPT-4 / Llama 3 / Mistral / DeepSeek\nLower-Triangular Causal Mask\nTask: Autoregressive General Intelligence", shape="box", fillcolor="#042F2E", color="#14B8A6")

    # Edges
    arch_graph.edge("ROOT", "BERT", label="Extracts Encoder Stack")
    arch_graph.edge("ROOT", "T5", label="Maintains Dual Stack")
    arch_graph.edge("ROOT", "GPT", label="Extracts Decoder Stack (The Winner)")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Structural Comparison of the Three Paradigms
    * **Encoder-Only (BERT)**: All tokens attend to all other tokens (N * N full visibility). Exceptional for language comprehension, sentence embeddings, and token tagging, but cannot generate open-ended text efficiently.
    * **Encoder-Decoder (T5 / BART)**: Separates input encoding from output token generation. Features an explicit cross-attention layer where the decoder attends to encoder keys and values.
    * **Decoder-Only (GPT / Llama)**: Every token only attends to previous tokens (causal mask). Unified architecture: processes prompts and emits responses inside the same autoregressive framework.
    """)
    st.info("Modern Generative AI has standardized on Decoder-Only architectures (Llama 3, GPT-4, Mistral, Qwen 2.5) due to their unified KV-caching efficiency and superior scaling-law performance.")

# ==============================================================================
# SECTION 2: 1. WHAT IS A TRANSFORMER?
# ==============================================================================
st.header("2. What is a Transformer? From Recurrence to Parallel Self-Attention", divider="rainbow")

st.markdown("""
Before the Transformer, state-of-the-art NLP relied on **Recurrent Neural Networks (LSTMs and GRUs)**. 
LSTMs enforced a strict sequential bottleneck: token h_t could not be computed until token h_{t-1} finished executing:
`h_t = f(h_{t-1}, x_t)`.
""")

col_trans1, col_trans2 = st.columns([1, 1])

with col_trans1:
    st.markdown("""
    ### The Fatal Flaws of Recurrent Architectures
    1. **Zero Training Parallelism**: Recurrence processes sequences sequentially step-by-step (O(N) operations), completely starving modern parallel GPUs and TPUs.
    2. **Vanishing Gradient / Information Bottleneck**: Long-range dependencies (e.g., matching a subject at token 5 with a verb at token 500) degraded because gradients had to backpropagate through 500 consecutive matrix multiplications.
    3. **Path Length**: The maximum signal path between any two tokens in an LSTM was O(N).

    ### The Transformer Solution
    The Transformer replaced sequential recurrence with **Multi-Head Self-Attention**:
    * **Constant Path Length O(1)**: Any token can attend directly to any other token in a single operation.
    * **Full Parallelism**: All N tokens are processed simultaneously across matrix tensor cores during training.
    * **Complexity Trade-off**: Path length drops to O(1), but compute complexity scales quadratically (O(N^2)) with sequence length.
    """)

with col_trans2:
    st.markdown("""
    ### Architectural Evolution: LSTM vs. Transformer
    """)
    df_trans_comp = pd.DataFrame({
        "Attribute": ["Sequential Processing", "Maximum Path Length", "Training Parallelism", "Computational Complexity", "Long-Range Memory", "Hardware Suitability"],
        "RNN / LSTM (Legacy)": ["Strictly sequential (Step-by-step)", "O(N) (Linear in sequence length)", "None (Time-step dependency)", "O(N * d^2) operations", "Degrades due to vanishing gradients", "Poor GPU utilization"],
        "Transformer (Modern)": ["Fully parallel across sequence", "O(1) (Direct pairwise attention)", "Total (Matrix multiplication)", "O(N^2 * d) attention operations", "Flawless direct signal routing", "Near 100% GPU Tensor Core efficiency"]
    })
    st.dataframe(df_trans_comp, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 3: 2. TRANSFORMER ARCHITECTURE
# ==============================================================================
st.header("3. Transformer Architecture: The Macro-Topological Stack", divider="rainbow")

st.markdown("""
The original 2017 Transformer architecture comprises two coupled towers: 
an **Encoder stack of N=6 identical layers** and a **Decoder stack of N=6 identical layers**.
""")

stack_col1, stack_col2 = st.columns([1, 1])

with stack_col1:
    t_stack = graphviz.Digraph(format="svg")
    t_stack.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.3", ranksep="0.5")
    t_stack.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    t_stack.attr("edge", color="#64748B", penwidth="1.2")

    with t_stack.subgraph(name="cluster_enc_block") as c:
        c.attr(bgcolor="#0F172A", label="Encoder Layer (x N)", fontcolor="#38BDF8", color="#38BDF8")
        c.node("ENC_ATTN", "Multi-Head Self-Attention\n(Bidirectional)", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("ENC_NORM1", "Add & LayerNorm", shape="ellipse", fillcolor="#312E81", color="#818CF8")
        c.node("ENC_FFN", "Feed-Forward Network (FFN)\nLinear -> ReLU / GELU -> Linear", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("ENC_NORM2", "Add & LayerNorm", shape="ellipse", fillcolor="#312E81", color="#818CF8")
        c.edge("ENC_ATTN", "ENC_NORM1")
        c.edge("ENC_NORM1", "ENC_FFN")
        c.edge("ENC_FFN", "ENC_NORM2")

    with t_stack.subgraph(name="cluster_dec_block") as c:
        c.attr(bgcolor="#0F172A", label="Decoder Layer (x N)", fontcolor="#10B981", color="#10B981")
        c.node("DEC_MASK", "Masked Multi-Head Attention\n(Causal Lower-Triangular)", shape="box", fillcolor="#1E293B", color="#10B981")
        c.node("DEC_NORM1", "Add & LayerNorm", shape="ellipse", fillcolor="#042F2E", color="#14B8A6")
        c.node("DEC_CROSS", "Multi-Head Cross-Attention\nQ from Dec, K & V from Enc", shape="box", fillcolor="#312E81", color="#A855F7")
        c.node("DEC_NORM2", "Add & LayerNorm", shape="ellipse", fillcolor="#042F2E", color="#14B8A6")
        c.node("DEC_FFN", "Feed-Forward Network (FFN)", shape="box", fillcolor="#1E293B", color="#10B981")
        c.node("DEC_NORM3", "Add & LayerNorm", shape="ellipse", fillcolor="#042F2E", color="#14B8A6")
        c.edge("DEC_MASK", "DEC_NORM1")
        c.edge("DEC_NORM1", "DEC_CROSS")
        c.edge("DEC_CROSS", "DEC_NORM2")
        c.edge("DEC_NORM2", "DEC_FFN")
        c.edge("DEC_FFN", "DEC_NORM3")

    t_stack.edge("ENC_NORM2", "DEC_CROSS", label="Keys & Values\nfrom Final Enc Layer", color="#A855F7", style="dashed")

    with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(t_stack, use_container_width=True)

with stack_col2:
    st.markdown("""
    ### Core Invariants of the Transformer Block
    1. **Residual Connections (`Add`)**:
       Every sublayer output is wrapped in a residual bypass: `x_{out} = LayerNorm(x + Sublayer(x))`. Residual paths preserve clean gradient flow through dozens of layers, preventing gradient dissipation.
    2. **Dimension Uniformity (`d_model`)**:
       All vectors flowing between sublayers share the exact same hidden dimensionality (e.g., d_{model} = 512 in base Transformer, 4096 in Llama 3 8B, 8192 in Llama 3 70B).
    3. **Position-Wise Feed-Forward Networks**:
       Applies two linear projections with a non-linearity to each position independently:
       `FFN(x) = max(0, x W_1 + b_1) W_2 + b_2`.
       The inner dimension d_{ff} is typically expanded to 4 * d_{model} (or 8/3 * d_{model} with SwiGLU).
    """)

# ==============================================================================
# SECTION 4: 3. THE ENCODER
# ==============================================================================
st.header("4. The Encoder: Bidirectional Contextual Representation", divider="rainbow")

st.markdown("""
The **Encoder** consumes source sequences and generates high-dimensional continuous representations 
where each token's embedding is contextualized by all tokens in the sequence simultaneously.
""")

enc_col1, enc_col2 = st.columns([1, 1])

with enc_col1:
    st.markdown("""
    ### Encoder Characteristics
    * **Full Bidirectional Attention**:
      The attention mask is all ones (M_{ij} = 1). A token at position 2 can attend directly to token 1, token 3, and token 500 without restriction.
    * **Context Absorption**:
      At layer 1, representations are largely lexical. As signals propagate through 12-24 encoder layers, representations become deeply abstracted, capturing grammatical agreement, syntactic roles, and long-range semantic coreference.
    * **Deterministic Static Output**:
      Given an input of length N, the encoder runs a **single parallel forward pass** to emit a final representation tensor H_{enc} in R^[N, d_model]. It never runs an autoregressive generation loop.
    """)

with enc_col2:
    st.markdown("""
    ### Post-LN vs. Modern Pre-LN Stability
    * **Original Post-LN (Vaswani 2017)**:
      `x_{l+1} = LayerNorm(x_l + Sublayer(x_l))`
      LayerNorm sits on the residual path. Requires an extensive learning rate warm-up schedule; deep networks (>24 layers) suffer from vanishing gradients at initialization.
    * **Modern Pre-LN (Radford et al., Llama, Mistral)**:
      `x_{l+1} = x_l + Sublayer(LayerNorm(x_l))`
      LayerNorm is applied to the input of each sublayer before attention/MLP. Preserves an unobstructed identity residual highway, enabling stable training of 100+ layer architectures.
    """)

st.markdown("""
<div class="takeaway-box">
<strong>Architectural Rule</strong>: Always implement <strong>Pre-Layer Normalization (Pre-LN)</strong> or <strong>RMSNorm</strong> for stable gradient backpropagation when designing deep Transformer architectures.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 5: 4. THE DECODER
# ==============================================================================
st.header("5. The Decoder: Autoregressive Generation with Causal Masking", divider="rainbow")

st.markdown("""
The **Decoder** generates output sequences token-by-token in an **autoregressive loop**. 
To preserve causality during parallel training, future tokens are strictly masked from view.
""")

dec_col1, dec_col2 = st.columns([1, 1])

with dec_col1:
    st.markdown("""
    ### The Causal Attention Mask
    During training, the entire target sequence is fed to the decoder simultaneously. 
    To prevent token t from looking ahead at token t+1:
    * A **lower-triangular mask** M in R^[N, N] is added to attention logits before softmax:
      ```
      M_{ij} = 0      if j <= i (Past & Current)
      M_{ij} = -inf   if j > i  (Future / Lookahead)
      ```
    * Because exp(-infty) = 0, attention weights to all future positions become strictly zero!
    
    ### The Two Sub-Layers of the Classical Decoder:
    1. **Masked Self-Attention**: Queries, Keys, and Values all derive from the target sequence (Q, K, V <=ftarrow H_{dec}), constrained by the causal mask.
    2. **Cross-Attention**: Queries derive from the decoder (Q <=ftarrow H_{dec}), but Keys and Values derive from the encoder (K, V <=ftarrow H_{enc}).
    """)

with dec_col2:
    st.markdown("""
    ### Autoregressive Inference Loop & KV Caching
    At inference time, the model cannot parallelize token emission:
    1. Pass prompt tokens through model -> emit token 1.
    2. Append token 1 to prompt -> pass through model -> emit token 2.
    3. Repeat until an `<|end_of_text|>` token is sampled.
    
    **The KV Cache Optimization**:
    Because past tokens do not change, their Key and Value tensors are stored in GPU memory (`KV Cache`). 
    On subsequent decoding steps, only the single newly generated token is passed through the model (O(1) forward pass instead of recomputing O(N) historical tokens).
    """)

# ==============================================================================
# SECTION 6: 5. ENCODER-DECODER ARCHITECTURE
# ==============================================================================
st.header("6. Encoder-Decoder Architecture: The Seq2Seq Paradigm", divider="rainbow")

st.markdown("""
The **Encoder-Decoder** architecture represents the original Sequence-to-Sequence (Seq2Seq) model, 
ideal for tasks where an input sequence of length N is transformed into an output sequence of length M 
with distinct syntactic and lexical structures (e.g., German to English, Audio to Text).
""")

encdec_col1, encdec_col2 = st.columns([1, 1])

with encdec_col1:
    st.markdown("""
    ### Iconic Encoder-Decoder Models:
    * **Original Vaswani (2017)**: 6 Encoder layers, 6 Decoder layers, trained on WMT English-German translation.
    * **T5 (Raffel et al., 2020)**: *"Text-to-Text Transfer Transformer"*. Formulated all NLP tasks (classification, question answering, summarization) into a unified text-in, text-out Seq2Seq framework.
    * **BART (Lewis et al., 2019)**: Bidirectional encoder with corrupting denoising autoencoder objectives + autoregressive decoder.
    * **OpenAI Whisper (2022)**: Audio spectrogram encoder + autoregressive text decoder.

    ### The Cross-Attention Bridge Mechanics
    In cross-attention:
    * Query vector Q comes from the target sequence being generated.
    * Key vector K and Value vector V come from the frozen source representations emitted by the encoder.
    ```
    CrossAttention(Q_dec, K_enc, V_enc) = Softmax((Q_dec . K_enc^T) / sqrt(d_k)) . V_enc
    ```
    This allows every newly generated output word to dynamically query the full source document for relevant context!
    """)

with encdec_col2:
    st.markdown("""
    ### Why Encoder-Decoder Fell Behind Pure Decoders
    1. **Hardware Inefficiency**: Encoders and decoders execute separate compute graphs, complicating distributed tensor parallelism across large GPU clusters.
    2. **Inflexible In-Context Learning**: Encoder-decoders require strict source-target separation, making general conversational few-shot prompting awkward.
    3. **Parameter Asymmetry**: If the encoder is 50% of the parameter count, half the model weights remain completely idle during autoregressive token generation!
    """)

# ==============================================================================
# SECTION 7: 6. ENCODER-ONLY MODELS
# ==============================================================================
st.header("7. Encoder-Only Models: Bidirectional Feature Extraction", divider="rainbow")

st.markdown("""
**Encoder-Only Transformers** (BERT, RoBERTa, DeBERTa, ELECTRA) discard the decoder stack entirely, 
specializing in **deep bidirectional language understanding and discriminative tasks**.
""")

bert_col1, bert_col2 = st.columns([1, 1])

with bert_col1:
    st.markdown("""
    ### The Masked Language Modeling (MLM) Paradigm
    Because bidirectional attention allows all tokens to see each other, an encoder-only model cannot be trained with standard next-token prediction (it would trivially look ahead).
    
    Devlin et al. (2018) introduced **Masked Language Modeling (MLM)**:
    * Randomly replace 15% of tokens in the sequence with a special `[MASK]` token.
    * The encoder passes the sequence through bidirectional self-attention.
    * A linear classification head predicts the original identity of the masked tokens:
      `Loss = -log P(token_original | context_{all})`.

    ### Primary Production Workloads:
    * **Dense Embeddings & Bi-Encoders**: Sentence-BERT, BGE, E5 for Vector DBs.
    * **Named Entity Recognition (NER) & Token Classification**: Legal contract parsing, medical entity extraction.
    * **Cross-Encoder Rerankers**: Reranking search candidates with full cross-attention.
    """)

with bert_col2:
    st.markdown("""
    ### Architectural Comparison: Encoder-Only vs. Decoder-Only
    """)
    df_enc_only = pd.DataFrame({
        "Attribute": ["Attention Visibility", "Pre-training Objective", "Generation Capability", "Contextual Richness", "Primary Domain"],
        "Encoder-Only (BERT)": ["Full Bidirectional (No causal mask)", "Masked Language Modeling (MLM)", "Incapable of open-ended generation", "Highest per-token bidirectional context", "Search Reranking, Embeddings, Classification"],
        "Decoder-Only (GPT / Llama)": ["Causal Mask (Past tokens only)", "Next-Token Prediction (Causal LM)", "Native fluent open-ended generation", "Unidirectional per-token context", "General Reasoning, Chat, Code Generation, RAG"]
    })
    st.dataframe(df_enc_only, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 8: 7. DECODER-ONLY MODELS
# ==============================================================================
st.header("8. Decoder-Only Models: The Modern Foundation Model Standard", divider="rainbow")

st.markdown("""
**Decoder-Only Transformers** (GPT-4, Llama 3, Mistral, DeepSeek, Claude) discard the encoder and cross-attention sublayers, 
operating as a **single, unified autoregressive stack** conditioned by causal attention masks.
""")

dec_only_col1, dec_only_col2 = st.columns([1, 1])

with dec_only_col1:
    st.markdown("""
    ### Why Decoder-Only Won the Generative Era
    1. **Unified Prompt & Response Representation**:
       Prompt tokens and generated tokens pass through the exact same weights. There is no artificial partition between "source" and "target".
    2. **Linear KV Cache Scaling**:
       All tokens share a single KV cache. During generation, only 1 new vector pair (K_t, V_t) is appended per layer, enabling high throughput serving engines (vLLM, TensorRT-LLM).
    3. **In-Context Learning (Prompt Engineering)**:
       Brown et al. (2020) demonstrated that autoregressive decoders trained on vast corpora exhibit emergent zero-shot and few-shot reasoning, solving tasks simply by conditioning on instructions without parameter updates.
    4. **Empirical Scaling Law Superiority**:
       Kaplan et al. and Chinchilla showed that decoder-only models scale with strict predictability: loss decreases as a clean power-law function of compute (C), dataset tokens (D), and parameters (N).
    """)

with dec_only_col2:
    st.markdown("""
    ### Modern Decoder-Only Foundation LLM Milestones
    | Foundation Model | Parameters | Context Window | Attention Variant | Normalization | Activation |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **GPT-3** | 175B | 2,048 | Multi-Head Attention (MHA) | Post-LN | GELU |
    | **Llama 2** | 7B / 70B | 4,096 | Grouped-Query Attention (GQA) | RMSNorm (Pre-LN) | SwiGLU |
    | **Llama 3** | 8B / 70B | 8,192 | GQA (8 KV heads) | RMSNorm (Pre-LN) | SwiGLU |
    | **Mistral 7B** | 7.3B | 32,768 | Sliding Window Attention + GQA | RMSNorm | SwiGLU |
    | **DeepSeek-V2/V3** | 236B / 671B | 128,000 | Multi-Head Latent Attention (MLA) | RMSNorm | SwiGLU (MoE) |
    """)

st.markdown("""
<div class="modern-callout">
<strong>🚀 Modern Foundation LLM Reality: How New Models Follow This</strong><br>
Every frontier foundation LLM in 2024–2026 is a Decoder-Only Transformer. The architecture has proven so universally scalable that even multimodal models (GPT-4o, Llama 3.2 Vision) tokenize images, audio, and video into discrete tokens and feed them directly into the same causal decoder backbone!
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 9: 8. SEQUENCE-TO-SEQUENCE MODELS
# ==============================================================================
st.header("9. Sequence-to-Sequence Models: Conditioning on Fixed Sequences", divider="rainbow")

st.markdown("""
While pure decoder-only models dominate general text, **Sequence-to-Sequence (Seq2Seq)** modeling remains 
crucial in specialized modalities where input and output domains are structurally disjoint.
""")

seq_col1, seq_col2 = st.columns([1, 1])

with seq_col1:
    st.markdown("""
    ### The Two Ways to Build Seq2Seq Today
    1. **True Dual-Tower Encoder-Decoder (e.g., Whisper, T5)**:
       * Source input is encoded bidirectionally.
       * Decoder attends to source via cross-attention.
       * Essential when the source domain is not text (e.g., audio mel-spectrogram frames in Whisper or protein amino acid sequences).
    2. **Prefix Language Model (Decoder-Only Seq2Seq)**:
       * Input prompt tokens attend bidirectionally to each other (*prefix attention*).
       * Output response tokens attend causally to past tokens.
       * Implemented inside a single decoder-only backbone by modifying the attention mask matrix!
    """)

with seq_col2:
    st.markdown("""
    ### When to Choose Encoder-Decoder vs. Decoder-Only
    * **Choose Encoder-Decoder when**:
      * Inputs and outputs reside in completely different continuous modalities (Audio-to-Text, Protein-to-Structure).
      * Strict latency constraints require small dedicated models (<1B parameters) deployed on edge devices (e.g., on-device Whisper translation).
    * **Choose Decoder-Only when**:
      * Building general conversational agents, code generation tools, multi-turn reasoning, or text-to-text RAG pipelines.
      * Parameter scale exceeds 3B parameters.
    """)

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Pure Python Minimal Transformer Blocks & Causal Mask Engine", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. A **Causal Attention Mask Generator** supporting pure causal, bidirectional, and prefix-LM attention.
2. A **Pure-Numpy Multi-Head Attention Block** with scaled dot-product attention.
3. A **Minimal Pre-LN Transformer Layer** with residual skip-connections and Feed-Forward expansion.
""")

st.code('''
import numpy as np
from typing import Optional, Tuple

class MinimalTransformerBlock:
    """
    Production-grade educational Transformer implementation demonstrating:
    1. Scaled Dot-Product Multi-Head Attention (MHA).
    2. Causal masking vs Bidirectional masking.
    3. Pre-Layer Normalization and Residual connections.
    4. Two-layer Position-Wise Feed-Forward Network (FFN).
    """
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 256, seed: int = 42):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff
        
        rng = np.random.RandomState(seed)
        # Attention projection weights
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_o = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        
        # FFN weights
        self.W_ff1 = rng.randn(d_model, d_ff).astype(np.float32) * 0.02
        self.b_ff1 = np.zeros(d_ff, dtype=np.float32)
        self.W_ff2 = rng.randn(d_ff, d_model).astype(np.float32) * 0.02
        self.b_ff2 = np.zeros(d_model, dtype=np.float32)

    @staticmethod
    def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Applies Layer Normalization over the last dimension."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    @staticmethod
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """Creates lower-triangular causal attention mask of shape [seq_len, seq_len]."""
        # 0.0 for allowed positions, -inf for masked lookahead positions
        mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        return mask

    def multi_head_attention(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Executes Multi-Head Self-Attention.
        x shape: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Project to Q, K, V
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        
        # 2. Scaled Dot-Product: (Q @ K^T) / sqrt(d_k)
        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        
        # 3. Apply causal mask if provided
        if mask is not None:
            scores = scores + mask # Broadcasts across heads
            
        # 4. Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 5. Multiply by Values
        out = np.matmul(attn_weights, V)
        out = out.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # 6. Final linear projection
        return np.dot(out, self.W_o)

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Two-layer FFN with ReLU activation."""
        h = np.maximum(0, np.dot(x, self.W_ff1) + self.b_ff1)
        return np.dot(h, self.W_ff2) + self.b_ff2

    def forward_pre_ln(self, x: np.ndarray, is_causal: bool = True) -> np.ndarray:
        """Complete Transformer block forward pass with Pre-LayerNorm."""
        seq_len = x.shape[1]
        mask = self.create_causal_mask(seq_len) if is_causal else None
        
        # Sublayer 1: Attention with Pre-LN and Residual Connection
        norm_x1 = self.layer_norm(x)
        attn_out = self.multi_head_attention(norm_x1, mask=mask)
        x = x + attn_out
        
        # Sublayer 2: FFN with Pre-LN and Residual Connection
        norm_x2 = self.layer_norm(x)
        ffn_out = self.feed_forward(norm_x2)
        x = x + ffn_out
        
        return x
''', language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Transformer Family Topologies & Mask Matrix Visualizer", divider="rainbow")

st.markdown("""
Explore how **Attention Mask Matrices** fundamentally define whether a Transformer functions as an 
**Encoder-Only model**, a **Decoder-Only model**, or a **Prefix Language Model**.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.markdown("#### Configure Sequence and Mask Paradigm")
    seq_length = st.slider("Sequence Length (Tokens)", min_value=3, max_value=8, value=5)
    arch_mode = st.radio(
        "Select Architecture Paradigm",
        ["Encoder-Only (Bidirectional - BERT)", "Decoder-Only (Causal - GPT / Llama)", "Prefix LM (Bidirectional Prompt + Causal Generation)"]
    )
    
    # Generate mask matrix
    if arch_mode.startswith("Encoder"):
        mask_mat = np.ones((seq_length, seq_length))
        desc = "Full Bidirectional: Every token attends to all tokens without restriction."
    elif arch_mode.startswith("Decoder"):
        mask_mat = np.tril(np.ones((seq_length, seq_length)))
        desc = "Causal Lower-Triangular: Token t only attends to past tokens (columns <= t). Future is masked."
    else:
        prefix_len = max(2, seq_length // 2)
        mask_mat = np.zeros((seq_length, seq_length))
        mask_mat[:prefix_len, :prefix_len] = 1.0 # Bidirectional prefix
        for i in range(prefix_len, seq_length):
            mask_mat[i, :i+1] = 1.0 # Causal generation
        desc = f"Prefix LM: First {prefix_len} tokens attend bidirectionally; remaining tokens attend causally."

    st.info(desc)

with wb_col2:
    st.markdown("#### Attention Visibility Matrix (N * N)")
    st.markdown("Rows = Query Position (i), Columns = Key Position (j)")
    
    df_mask = pd.DataFrame(
        mask_mat,
        columns=[f"K_{j}" for j in range(seq_length)],
        index=[f"Q_{i}" for i in range(seq_length)]
    )
    st.dataframe(df_mask.style.background_gradient(cmap="Blues", vmin=0, vmax=1), use_container_width=True)

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why does Multi-Head Attention divide the model dimension d_model by the number of heads h (d_k = d_model / h)?",
        "a": "Dividing the hidden dimension ensures that total computational complexity and parameter count remain identical to single-head attention with dimension d_model. Specifically, running h independent attention heads with dimension d_k takes O(h * (N^2 * d_k)) = O(N^2 * d_model) operations. Multi-head projection allows the model to jointly attend to information from different representation subspaces at different positions without increasing compute costs."
    },
    {
        "q": "Why did the AI industry converge on Decoder-Only architectures for LLMs over Encoder-Decoder architectures?",
        "a": "Decoder-only models unify prompt understanding and token generation within a single autoregressive compute graph, enabling efficient KV-caching during decoding. Encoder-decoder architectures suffer from parameter asymmetry during generation (the entire encoder remains idle while the decoder emits tokens), and scaling laws demonstrated that causal language modeling on decoders achieves superior zero-shot and few-shot in-context learning per FLOP."
    },
    {
        "q": "What is the mathematical difference between Pre-LN and Post-LN, and why does Pre-LN dominate modern LLMs?",
        "a": "In Post-LN (Vaswani 2017), LayerNorm is applied after the residual addition: x_{l+1} = LN(x_l + Sublayer(x_l)). In Pre-LN, LayerNorm is applied to the input of each sublayer before processing: x_{l+1} = x_l + Sublayer(LN(x_l)). Pre-LN preserves an unobstructed identity residual path through all layers (x_L = x_0 + sum(Sublayers)), preventing vanishing or exploding gradients and enabling stable training of 100+ layer foundation models without delicate learning rate warm-up."
    },
    {
        "q": "How does Cross-Attention differ from Self-Attention in an Encoder-Decoder Transformer?",
        "a": "In Self-Attention, Queries, Keys, and Values all derive from the exact same input tensor (Q, K, V = X W_q, X W_k, X W_v). In Cross-Attention, Queries derive from the decoder's previous hidden states (Q = H_dec W_q), while Keys and Values derive from the final layer of the encoder (K = H_enc W_k, V = H_enc W_v). This enables the decoder to query the source sequence dynamically at every generated token position."
    },
    {
        "q": "Why cannot an Encoder-Only model like BERT generate fluent open-ended text autoregressively?",
        "a": "BERT was pretrained with bidirectional attention and Masked Language Modeling (MLM). Its representations assume full visibility of both preceding and subsequent tokens simultaneously. If forced to generate autoregressively, BERT lacks a causal mask to enforce left-to-right generation, causing severe out-of-distribution hallucinations and requiring re-evaluating the entire sequence for every emitted token."
    }
]

for i, qa in enumerate(qas, 1):
    with st.expander(f"Q{i}: {qa['q']}"):
        st.markdown(f"**Answer**: {qa['a']}")

st.markdown("""
### Production Traps & Failure Modes
""")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Trap 1 (Fatal): Causal Mask Leakage via Inverted Indices**
    * *Failure*: Setting the causal attention mask to upper-triangular instead of lower-triangular, allowing tokens to attend exclusively to the *future* while hiding the *past*.
    * *Fix*: Always verify the mask: position (i, j) must be 0 for j <= i, and -infty for j > i.
    """)
    st.error("""
    **Trap 2 (Fatal): Training Deep Transformers (>30 layers) with Post-LN**
    * *Failure*: Using original 2017 Post-LayerNorm on large models, resulting in catastrophic gradient divergence and NaN loss within the first 1,000 training steps.
    * *Fix*: Always use Pre-LN or RMSNorm (`x + Sublayer(RMSNorm(x))`) to preserve gradient highways.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Discarding KV Caching in Decoder Inference**
    * *Failure*: Passing the entire prompt plus all generated tokens through the model at every decoding step (O(N^2) inference latency).
    * *Fix*: Cache past Keys and Values (`past_key_values` in HuggingFace or vLLM PagedAttention), reducing per-token step latency to O(1).
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Deploying Encoder-Decoders for General Chatbots**
    * *Failure*: Selecting T5 or BART for general conversational tasks, experiencing slow serving throughput due to dual compute graph overhead.
    * *Fix*: Use decoder-only foundation models (Llama 3, Mistral) for conversational and generative workflows.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**Encoder vs Decoder**\n\nEncoder is bidirectional (BERT, embeddings). Decoder is causally masked (GPT, generative emission).")
with fc_col2:
    st.info("**Pre-LN Highway**\n\nApplies LayerNorm before sublayers, preserving an unobstructed identity residual path for stable deep training.")
with fc_col3:
    st.info("**Cross-Attention**\n\nQueries from Decoder, Keys and Values from Encoder. Enables translation and Seq2Seq grounding.")
with fc_col4:
    st.info("**KV Cache Speedup**\n\nCaches past Key and Value tensors in GPU RAM, avoiding redundant computation during autoregressive token emission.")

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of Transformer architectures, attention topologies, and generative foundation model mechanics:
""")

c1 = st.checkbox("I can explain why the Transformer eliminated recurrent loops in favor of parallel self-attention.")
c2 = st.checkbox("I understand the macro-topological stack of the original 2017 Encoder-Decoder architecture.")
c3 = st.checkbox("I know how the causal lower-triangular attention mask enforces autoregressive generation during training.")
c4 = st.checkbox("I understand the mathematical difference between Pre-LN and Post-LN and why modern LLMs use Pre-LN.")
c5 = st.checkbox("I can explain why the AI industry standardized on Decoder-Only architectures (Llama, GPT) over Encoder-Decoders.")
c6 = st.checkbox("I know how the KV Cache optimizes autoregressive inference from O(N^2) to O(N).")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered Transformer fundamentals, architectural topologies, and the foundations of modern generative LLMs!")
