import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="5.3 Transformer Architectures (BERT, GPT, T5, Encoder, Decoder, Encoder-Decoder)",
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
st.markdown('<div class="main-title">5.3 Transformer Architectures: BERT, GPT, and T5 Paradigm Deep Dive</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">From Masked Encoders and Sequence-to-Sequence Bridges to Autoregressive Foundation Decoders — Structural Analysis of Modern Attention Archetypes.</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Tripartite Transformer Paradigm Matrix", divider="rainbow")

st.markdown("""
The modern natural language processing landscape is divided into three distinct architectural archetypes, 
each distinguished by its **attention visibility mask**, **training objective**, and **downstream operational role**:
""")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    arch_graph = graphviz.Digraph(format="svg")
    arch_graph.attr(bgcolor="#0B1120", rankdir="TB", nodesep="0.4", ranksep="0.5")
    arch_graph.attr("node", style="filled", fontname="sans-serif", fontsize="10", fontcolor="#F8FAFC", penwidth="1.5")
    arch_graph.attr("edge", color="#64748B", penwidth="1.2", fontcolor="#94A3B8", fontsize="9")

    with arch_graph.subgraph(name="cluster_bert") as c:
        c.attr(bgcolor="#0F172A", label="1. BERT Archetype\n(Encoder-Only)", fontcolor="#38BDF8", color="#38BDF8")
        c.node("BERT_MASK", "Bidirectional Mask\nFull all-to-all visibility", shape="box", fillcolor="#1E293B", color="#38BDF8")
        c.node("BERT_OBJ", "MLM Pretraining\nPredict masked tokens [MASK]", shape="box", fillcolor="#312E81", color="#818CF8")

    with arch_graph.subgraph(name="cluster_t5") as c:
        c.attr(bgcolor="#0F172A", label="2. T5 Archetype\n(Encoder-Decoder)", fontcolor="#A855F7", color="#A855F7")
        c.node("T5_STRUCT", "Dual Tower + Cross-Attention\nEncodes input -> Decodes output", shape="box", fillcolor="#1E293B", color="#A855F7")
        c.node("T5_OBJ", "Span Denoising Objective\nReplace text spans with sentinels", shape="box", fillcolor="#312E81", color="#818CF8")

    with arch_graph.subgraph(name="cluster_gpt") as c:
        c.attr(bgcolor="#0F172A", label="3. GPT Archetype\n(Decoder-Only)", fontcolor="#10B981", color="#10B981")
        c.node("GPT_MASK", "Causal Mask\nStrict lower-triangular visibility", shape="box", fillcolor="#042F2E", color="#14B8A6")
        c.node("GPT_OBJ", "Next-Token Prediction\nP(w_t | w_1...w_{t-1})", shape="box", fillcolor="#042F2E", color="#14B8A6")

    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(arch_graph, use_container_width=True)

with col_arch2:
    st.markdown("""
    ### Structural Essence of the Archetypes
    * **BERT-style (Autoencoding)**: Corrupts text with noise (`[MASK]`) and reconstructs it. Produces rich, context-aware embeddings where every token attends to the entire document.
    * **T5-style (Sequence-to-Sequence)**: Transforms input sequences into output sequences. Excellent for translation and structured transformation tasks where input and output domains differ.
    * **GPT-style (Autoregressive Generative)**: Predicts the next token given all preceding tokens. Unified architecture that powers modern frontier AI (Llama 3, GPT-4, Mistral).
    """)

# Master side-by-side comparison matrix across the three archetypes
st.markdown("### Master Paradigm Comparison Matrix")
df_paradigms = pd.DataFrame({
    "Criterion": ["Pioneering Paper", "Attention Mask", "Pre-training Task", "Generation Capability", "Parameter Efficiency during Inference", "Primary Enterprise Workload"],
    "BERT-style (Encoder-Only)": ["Devlin et al., 2018", "Bidirectional (All-to-all)", "Masked LM (MLM) + NSP", "Incapable of open-ended generation", "100% active (One forward pass)", "Vector Embeddings, Search Reranking, Token Classification"],
    "T5-style (Encoder-Decoder)": ["Raffel et al., 2020", "Bidirectional (Enc) + Causal (Dec)", "Span Corruption / Denoising", "Structured text-to-text generation", "50% idle (Encoder idle during generation)", "Translation, Abstractive Summarization, Whisper Audio"],
    "GPT-style (Decoder-Only)": ["Radford et al., 2018", "Causal Lower-Triangular", "Next-Token Causal LM", "Native fluent open-ended generation", "100% active (Shared weights with KV cache)", "General Reasoning, Chatbots, Code Generation, RAG Agents"]
})
st.dataframe(df_paradigms, use_container_width=True, hide_index=True)

# ==============================================================================
# SECTION 2: 1. BERT-STYLE ARCHITECTURE
# ==============================================================================
st.header("2. BERT-Style Architecture: Bidirectional Masked Autoencoding", divider="rainbow")

st.markdown("""
Introduced by Jacob Devlin et al. at Google AI in 2018, **BERT (Bidirectional Encoder Representations from Transformers)** 
demolished previous NLP benchmarks by demonstrating that pretraining deep bidirectional representations 
yields profound linguistic understanding.
""")

bert_col1, bert_col2 = st.columns([1, 1])

with bert_col1:
    st.markdown("""
    ### Pretraining Mechanics
    1. **Masked Language Modeling (MLM - 80/10/10 Rule)**:
       * 15% of all tokens in a text are randomly selected for corruption.
       * Of selected tokens:
         * **80%** replaced with the `[MASK]` token.
         * **10%** replaced with a random vocabulary token (forces model to maintain representation fidelity).
         * **10%** kept unchanged (biases representation toward true token).
    2. **Next Sentence Prediction (NSP)**:
       * Fed pairs of sentences `[CLS] Sentence A [SEP] Sentence B [SEP]`.
       * Binary classification predicting whether Sentence B logically follows Sentence A (later removed by RoBERTa as harmful).
    """)

with bert_col2:
    st.markdown("""
    ### Key Evolutionary Variants of BERT:
    * **RoBERTa (Liu et al., 2019)**: Stripped NSP loss, trained on 10x more data with dynamic masking and larger batch sizes, substantially outperforming original BERT.
    * **DeBERTa (He et al., 2021)**: Introduced **Disentangled Attention** (separates content and relative position into two independent vectors) and enhanced mask decoding.
    * **ELECTRA (Clark et al., 2020)**: Replaced `[MASK]` with **Replaced Token Detection (RTD)** using a small generator to insert plausible fakes, training the discriminator on 100% of tokens rather than just 15%.
    """)

# ==============================================================================
# SECTION 3: 2. GPT-STYLE ARCHITECTURE
# ==============================================================================
st.header("3. GPT-Style Architecture: Causal Autoregressive Language Modeling", divider="rainbow")

st.markdown("""
Originating with Alec Radford et al. at OpenAI in 2018 (GPT-1), the **Generative Pre-trained Transformer (GPT)** 
relies entirely on a single stack of **Causal Transformer Decoders**.
""")

gpt_col1, gpt_col2 = st.columns([1, 1])

with gpt_col1:
    st.markdown("""
    ### The Causal Autoregressive Objective
    Given a token sequence U = (u_1, u_2, ..., u_n), maximize the causal log-likelihood:
    ```
    L_CL = Sum_{i=1}^{n} log P(u_i | u_1, u_2, ..., u_{i-1}; Theta)
    ```
    Where conditional probability is modeled via Softmax over hidden states:
    ```
    P(u_i | u_{<i}) = Softmax(h_i . W_vocab^T)
    ```

    ### Key Generative Capabilities:
    * **Next-Token Prediction at Scale**: By learning to predict the next token across trillions of web tokens, models develop emergent world models, logical deduction, and coding abilities.
    * **KV Caching**: During generation, the Key and Value tensors for historical tokens u_{<i} are cached in GPU memory, avoiding redundant computation and enabling high-throughput serving engines.
    """)

with gpt_col2:
    st.markdown("""
    ### From GPT-1 to Modern Frontier Decoders:
    * **GPT-1 (2018)**: 117M parameters, 12 layers. Demonstrated generative pretraining followed by discriminative fine-tuning.
    * **GPT-2 (2019)**: 1.5B parameters. Moved LayerNorm to Pre-LN; demonstrated zero-shot multi-task potential without fine-tuning.
    * **GPT-3 (2020)**: 175B parameters. Proved **In-Context Learning (Few-shot prompting)** emerges at scale.
    * **Llama 3 / Mistral (2023-2024)**: Open-weights supremacy; introduced RMSNorm, RoPE rotary embeddings, SwiGLU activations, and Grouped-Query Attention (GQA).
    """)

st.markdown("""
<div class="modern-callout">
<strong>🚀 Modern Foundation LLM Reality: How New Models Follow This</strong><br>
Virtually all state-of-the-art open and closed foundation models (GPT-4o, Llama 3.3, Mistral Large, Qwen 2.5, DeepSeek-V3) are direct descendants of the GPT-style causal decoder architecture.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 4: 3. T5-STYLE ARCHITECTURE
# ==============================================================================
st.header("4. T5-Style Architecture: The Text-to-Text Transfer Framework", divider="rainbow")

st.markdown("""
In 2020, Colin Raffel et al. introduced **T5 (Text-to-Text Transfer Transformer)**, 
proposing that **every single NLP problem** can be cleanly formulated as receiving text as input and generating new text as output.
""")

t5_col1, t5_col2 = st.columns([1, 1])

with t5_col1:
    st.markdown("""
    ### Unified "Text-to-Text" Formulation
    Instead of adding custom classification heads for sentiment or regression heads for similarity:
    * **Translation**: Input: `"translate English to German: The house is wonderful."` -> Output: `"Das Haus ist wunderbar."`
    * **Classification**: Input: `"sst2 sentence: The movie was exciting and thrilling."` -> Output: `"positive"`
    * **Summarization**: Input: `"summarize: [Article text]..."` -> Output: `"[Summary text]"`

    ### The Span Corruption Objective
    Unlike BERT which masks individual tokens with `[MASK]`, T5 replaces **contiguous spans of text** with unique sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, ...):
    * **Corrupted Input**: `"Thank you <extra_id_0> me to your party <extra_id_1> week."`
    * **Target Output**: `"<extra_id_0> for inviting <extra_id_1> last <extra_id_2>"`
    This dramatically reduces target sequence length, saving significant pretraining compute!
    """)

with t5_col2:
    st.markdown("""
    ### Enduring Legacy of T5 in Modern AI:
    * **FLAN-T5 (Chung et al., 2022)**: Demonstrated that instruction-tuning an encoder-decoder model across hundreds of diverse tasks creates exceptional zero-shot reasoning.
    * **OpenAI Whisper (2022)**: The world standard for speech-to-text uses a pure T5-style architecture: a Transformer audio encoder coupled to an autoregressive text decoder.
    * **Diffusion Model Text Encoders**: Text-to-image models (Stable Diffusion 3, Imagen, Flux) use **T5-XXL** as their text comprehension backbone due to its rich bidirectional representations!
    """)

# ==============================================================================
# SECTION 5: 4. ENCODER-ONLY TRANSFORMERS
# ==============================================================================
st.header("5. Encoder-Only Transformers: Structural Mechanics & Best Practices", divider="rainbow")

st.markdown("""
An **Encoder-Only Transformer** consists exclusively of bidirectional self-attention blocks. 
It processes the entire input sequence in a **single parallel forward pass**, outputting a contextualized tensor `H in R^[Batch, Seq_Len, d_model]`.
""")

enc_col1, enc_col2 = st.columns([1, 1])

with enc_col1:
    st.markdown("""
    ### Downstream Task Adaptation Patterns:
    1. **Sentence Classification (Sentiment, Spam, Intent)**:
       Feed the representation of the first token `h_[CLS]` into a linear classification head:
       `logits = h_[CLS] . W_class + b_class`
    2. **Token Classification (NER, POS Tagging)**:
       Pass every token's hidden state h_i into a shared linear classification head:
       `logits_i = h_i . W_tag + b_tag`
    3. **Extractive Question Answering (SQuAD)**:
       Compute two dot products per token to predict the start and end position of the answer span in the passage.
    4. **Bi-Encoder Dense Embeddings (SBERT, BGE)**:
       Apply masked mean pooling over H to produce a single normalized vector u in R^[d_model] for vector search indexing.
    """)

with enc_col2:
    st.markdown("""
    ### Why Encoder-Only Models Remain Indispensable:
    * **Ultra-Low Latency**: A single parallel matrix multiplication evaluates 512 tokens in under 5ms on CPU or GPU.
    * **No Autoregressive Bottlenecks**: No KV cache management, no sequential sampling loops.
    * **High-Throughput Ingestion**: Ideal for indexing millions of documents in vector databases or screening millions of transactions for fraud in real time.
    """)

# ==============================================================================
# SECTION 6: 5. DECODER-ONLY TRANSFORMERS
# ==============================================================================
st.header("6. Decoder-Only Transformers: Architectural Scaling & The Generative Standard", divider="rainbow")

st.markdown("""
The **Decoder-Only Transformer** has become the undisputed standard for modern foundation models. 
Its strength lies in its **simplicity, uniform compute graph, and alignment with autoregressive scaling laws**.
""")

dec_col1, dec_col2 = st.columns([1, 1])

with dec_col1:
    st.markdown("""
    ### The Scaling Law Monopoly
    Research by Kaplan et al. (OpenAI, 2020) and Hoffmann et al. (Chinchilla, DeepMind, 2022) revealed that 
    cross-entropy loss on causal language modeling follows clean power-law scaling across compute (C), dataset tokens (D), and parameters (N):
    ```
    Loss(N, D) = (N_c / N)^alpha_N + (D_c / D)^alpha_D + L_0
    ```
    Decoder-only architectures optimize this frontier more efficiently than encoder-decoder models because:
    * **Every FLOP updates universal generative weights**.
    * There is no parameter asymmetry or idle encoder hardware during generation.
    * Autoregressive next-token prediction inherently forces the model to compress world knowledge, syntax, semantics, and logic into its weights.
    """)

with dec_col2:
    st.markdown("""
    ### Modern Architectural Enhancements in Decoder LLMs:
    1. **Mixture of Experts (MoE - Mixtral, DeepSeek)**:
       Replaces the dense FFN with E sparse expert networks, activating only top-K experts per token (e.g., activating 2 of 8 experts). Multiplies parameter capacity by 4x-8x while keeping compute latency constant!
    2. **Grouped-Query Attention (GQA)**:
       Slashes KV cache memory by sharing Key-Value heads across multiple Query heads.
    3. **Rotary Positional Embeddings (RoPE)**:
       Enables context window scaling from 4k tokens to 128k+ tokens via NTK-aware and YaRN interpolation.
    """)

# ==============================================================================
# SECTION 7: 6. ENCODER-DECODER TRANSFORMERS
# ==============================================================================
st.header("7. Encoder-Decoder Transformers: Dual-Tower Cross-Attention Topologies", divider="rainbow")

st.markdown("""
The **Encoder-Decoder Transformer** decouples source understanding from target generation via an explicit **Cross-Attention Bridge**.
""")

ed_col1, ed_col2 = st.columns([1, 1])

with ed_col1:
    st.markdown("""
    ### Cross-Attention Information Flow
    Let source sequence X_{src} have length N, and target sequence X_{tgt} have generated length t:
    1. **Encoder Pass (Executed ONCE)**:
       Computes bidirectional hidden states: H_{enc} in R^[N, d_model].
       Computes Key and Value projections:
       ```
       K_{enc} = H_{enc} W_K,   V_{enc} = H_{enc} W_V
       ```
    2. **Decoder Step t (Executed Autoregressively)**:
       Computes causal masked self-attention over target tokens 1...t.
       Generates Query vector Q_t = h_{dec, t} W_Q.
       Performs Cross-Attention against the frozen encoder states:
       ```
       CrossAttn(Q_t, K_{enc}, V_{enc}) = Softmax((Q_t . K_{enc}^T) / sqrt(d_k)) . V_{enc}
       ```
    """)

with ed_col2:
    st.markdown("""
    ### Niche Strengths Where Encoder-Decoder Still Excels:
    * **Multi-Modal Translation**: When the input is a continuous spectrogram (Whisper) or video frames, an encoder naturally compresses the dense raw signal before the text decoder generates tokens.
    * **Extreme Compression Summarization**: Generating a 20-word summary from a 5,000-word document without filling the generation context window with the source text.
    * **On-Device Edge Translation**: Compact Seq2Seq models (<300M parameters) execute faster translation on mobile neural engines than large general decoders.
    """)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Pure Python Unified Multi-Architecture Forward Pass Engine", divider="rainbow")

st.markdown("""
The following self-contained Python implementation provides:
1. An **Encoder-Only (BERT-style) Forward Pass** with full bidirectional attention and masked pooling.
2. A **Decoder-Only (GPT-style) Forward Pass** with causal attention and KV-cache tracking.
3. An **Encoder-Decoder (T5-style) Forward Pass** with decoupled cross-attention.
""")

st.code('''
import numpy as np
from typing import Dict, List, Tuple, Optional

class UnifiedTransformerTopologyEngine:
    """
    Production-grade educational engine demonstrating the structural forward pass
    differences between BERT (Encoder-only), GPT (Decoder-only), and T5 (Encoder-Decoder).
    """
    def __init__(self, d_model: int = 32, seed: int = 42):
        self.d_model = d_model
        rng = np.random.RandomState(seed)
        
        # Shared projection weights
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * 0.05
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * 0.05
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * 0.05
        self.W_cross_k = rng.randn(d_model, d_model).astype(np.float32) * 0.05
        self.W_cross_v = rng.randn(d_model, d_model).astype(np.float32) * 0.05

    def attention_core(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Scaled dot-product attention kernel."""
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        return np.matmul(weights, V)

    def forward_encoder_only_bert(self, x: np.ndarray) -> np.ndarray:
        """
        BERT-style: Full bidirectional self-attention (no mask).
        x shape: [batch, seq_len, d_model]
        """
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        # No mask: all tokens attend to all tokens
        return self.attention_core(Q, K, V, mask=None)

    def forward_decoder_only_gpt(self, x: np.ndarray, kv_cache: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        GPT-style: Causal lower-triangular masked attention with KV caching.
        """
        seq_len = x.shape[1]
        Q = np.dot(x, self.W_q)
        new_K = np.dot(x, self.W_k)
        new_V = np.dot(x, self.W_v)
        
        if kv_cache is not None:
            K = np.concatenate([kv_cache["K"], new_K], axis=1)
            V = np.concatenate([kv_cache["V"], new_V], axis=1)
        else:
            K = new_K
            V = new_V
            
        total_len = K.shape[1]
        # Causal mask for the current query slice
        mask = np.triu(np.full((seq_len, total_len), -1e9, dtype=np.float32), k=total_len - seq_len + 1)
        out = self.attention_core(Q, K, V, mask=mask)
        
        updated_cache = {"K": K, "V": V}
        return out, updated_cache

    def forward_encoder_decoder_t5(self, x_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
        """
        T5-style: Bidirectional encoder + Causal decoder + Cross-Attention bridge.
        """
        # 1. Bidirectional Encoder
        H_enc = self.forward_encoder_only_bert(x_src)
        K_enc = np.dot(H_enc, self.W_cross_k)
        V_enc = np.dot(H_enc, self.W_cross_v)
        
        # 2. Causal Masked Decoder Self-Attention
        tgt_len = x_tgt.shape[1]
        Q_tgt = np.dot(x_tgt, self.W_q)
        K_tgt = np.dot(x_tgt, self.W_k)
        V_tgt = np.dot(x_tgt, self.W_v)
        causal_mask = np.triu(np.full((tgt_len, tgt_len), -1e9, dtype=np.float32), k=1)
        H_dec_self = self.attention_core(Q_tgt, K_tgt, V_tgt, mask=causal_mask)
        
        # 3. Cross-Attention: Q from Decoder, K & V from Encoder
        Q_cross = np.dot(H_dec_self, self.W_q)
        H_out = self.attention_core(Q_cross, K_enc, V_enc, mask=None)
        return H_out
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Architecture Benchmark & Latency Simulator", divider="rainbow")

st.markdown("""
Compare the computational efficiency, **Time-To-First-Token (TTFT)**, and **Per-Token Generation Latency** 
between Encoder-Only, Decoder-Only, and Encoder-Decoder architectures across varying input and generation lengths.
""")

bench_col1, bench_col2 = st.columns([1, 1])

with bench_col1:
    st.markdown("#### Workload Parameters")
    prompt_len = st.slider("Input Prompt Length (Tokens)", min_value=128, max_value=4096, value=1024, step=128)
    gen_len = st.slider("Target Generation Length (Tokens)", min_value=16, max_value=1024, value=256, step=32)
    model_size = st.selectbox("Model Parameter Class", ["300M Parameters", "7B Parameters", "70B Parameters"], index=1)
    
    # Model config multiplier
    if "300M" in model_size:
        base_ms = 0.5
    elif "7B" in model_size:
        base_ms = 5.0
    else:
        base_ms = 35.0

    # Simulation metrics
    # Encoder-Only (Evaluates prompt only, cannot generate)
    enc_latency = base_ms * (prompt_len / 512.0)
    
    # Decoder-Only (Prompt prefill + autoregressive generation with KV cache)
    dec_prefill_ms = base_ms * (prompt_len / 512.0)
    dec_decode_per_token_ms = base_ms * 0.15 # O(1) step with KV cache
    dec_total_ms = dec_prefill_ms + (gen_len * dec_decode_per_token_ms)
    
    # Encoder-Decoder (Encoder runs prompt + Decoder decodes with Cross-Attention)
    ed_prefill_ms = (base_ms * 0.5) * (prompt_len / 512.0) # Encoder is half model
    ed_decode_per_token_ms = base_ms * 0.22 # Cross-attention overhead
    ed_total_ms = ed_prefill_ms + (gen_len * ed_decode_per_token_ms)

with bench_col2:
    st.markdown("#### Benchmark Latency Comparison")
    
    df_perf = pd.DataFrame([
        {"Architecture": "Encoder-Only (BERT)", "Prefill / TTFT": f"{enc_latency:.1f} ms", "Generation Time": "N/A (Non-generative)", "Total Latency": f"{enc_latency:.1f} ms", "Throughput": "Instant Parallel"},
        {"Architecture": "Decoder-Only (GPT / Llama)", "Prefill / TTFT": f"{dec_prefill_ms:.1f} ms", "Generation Time": f"{gen_len * dec_decode_per_token_ms:.1f} ms", "Total Latency": f"{dec_total_ms:.1f} ms", "Throughput": f"{1000 / dec_decode_per_token_ms:.1f} tok/s"},
        {"Architecture": "Encoder-Decoder (T5)", "Prefill / TTFT": f"{ed_prefill_ms:.1f} ms", "Generation Time": f"{gen_len * ed_decode_per_token_ms:.1f} ms", "Total Latency": f"{ed_total_ms:.1f} ms", "Throughput": f"{1000 / ed_decode_per_token_ms:.1f} tok/s"}
    ])
    st.dataframe(df_perf, use_container_width=True, hide_index=True)
    
    st.info(f"Summary for {model_size}: Decoder-Only achieves a generation throughput of **{1000 / dec_decode_per_token_ms:.1f} tokens/second** using KV caching.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Core Technical Q&A
""")

qas = [
    {
        "q": "Why does RoBERTa substantially outperform BERT on GLUE benchmarks despite sharing the identical architecture?",
        "a": "Liu et al. (2019) discovered that original BERT was severely undertrained. RoBERTa introduced four key optimizations: 1) Removed the Next Sentence Prediction (NSP) task, which proved detrimental to single-sentence and multi-sentence representations. 2) Replaced static masking with dynamic masking (generating new mask patterns at every epoch). 3) Trained on 10x more text (160GB vs 16GB) with larger mini-batch sizes (8,000 vs 256). 4) Utilized a larger byte-level BPE vocabulary (50,000 tokens)."
    },
    {
        "q": "What is the architectural bottleneck of Cross-Attention during long-context generation in Encoder-Decoder models?",
        "a": "In cross-attention, every newly generated decoder token must compute dot-product attention against all N source tokens emitted by the encoder: Attention(Q_dec, K_enc, V_enc). If the source context is long (e.g., N = 32,000 tokens), every single decoding step requires loading the entire 32k encoder Key-Value states from GPU memory, creating severe memory bandwidth bottlenecks and slowing down token emission compared to decoder-only models using sliding window or state-space optimizations."
    },
    {
        "q": "How does Prefix Language Modeling bridge the gap between Decoder-Only and Encoder-Decoder architectures?",
        "a": "A Prefix LM is implemented within a single decoder-only Transformer backbone by modifying the attention mask matrix. For input prompt tokens (the 'prefix'), the mask is bidirectional (all prefix tokens attend to all prefix tokens, exactly like a BERT encoder). For target tokens, the mask is causal lower-triangular (tokens only attend to past tokens). This provides the bidirectional prompt comprehension of an encoder with the zero-overhead unified weights of a decoder."
    },
    {
        "q": "Why is the [CLS] token representation in BERT insufficient for semantic search without contrastive fine-tuning?",
        "a": "BERT's pretraining objectives (Masked LM and Next Sentence Prediction) do not construct an isotropic metric space for cosine distance. The [CLS] token is optimized solely to separate binary pairs in a downstream classification head. Un-finetuned BERT [CLS] embeddings suffer from severe representation collapse (anisotropy), where all sentence vectors crowd into a narrow cone and exhibit artificially high cosine similarities (0.80-0.95) across unrelated sentences."
    },
    {
        "q": "What is the primary operational trade-off of Mixture of Experts (MoE) in modern decoder-only foundation models?",
        "a": "MoE replaces the dense FFN with multiple sparse expert networks (e.g., 8 or 16 experts), routing each token to top-2 experts via a learned gating network. This provides massive parameter capacity (e.g., 47B total parameters) while executing compute equivalent to a much smaller model (e.g., 13B active FLOPs). The trade-off is massive VRAM consumption: the entire 47B parameter model must reside in GPU RAM, and distributed serving requires complex all-to-all communication across GPUs for expert routing."
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
    **Trap 1 (Fatal): Using BERT for Generative Chat Tasks**
    * *Failure*: Attempting to generate multi-turn conversational responses using BERT by masking future tokens and predicting iteratively.
    * *Fix*: Never use encoder-only models for text generation. BERT lacks autoregressive training and will produce catastrophic, incoherent repetitions. Use decoder-only LLMs.
    """)
    st.error("""
    **Trap 2 (Fatal): Neglecting Masked Mean Pooling on BERT Embeddings**
    * *Failure*: Extracting `outputs.last_hidden_state[:, 0, :]` ([CLS] token) directly for vector search indexing without SBERT fine-tuning.
    * *Fix*: Always use dedicated fine-tuned embedding models (e.g., `BGE-Large`, `all-MiniLM-L6-v2`) with attention-masked mean pooling.
    """)

with trap_col2:
    st.warning("""
    **Trap 3 (Degraded Performance): Discarding RoBERTa Dynamic Masking in Custom MLM**
    * *Failure*: Pre-computing static mask patterns on the training dataset once before pretraining BERT, causing the model to see identical masks across all epochs.
    * *Fix*: Generate dynamic mask patterns on the fly in the PyTorch `Dataset` `__getitem__` pipeline.
    """)
    st.warning("""
    **Trap 4 (Latency Trap): Encoder-Decoder Parameter Asymmetry in Production Serving**
    * *Failure*: Hosting a 50/50 encoder-decoder model where the encoder sits completely idle during a 2,000-token generation stream.
    * *Fix*: Standardize production serving on decoder-only architectures where all weights participate in token generation.
    """)

st.markdown("### Rapid-Fire Flashcards")
fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.info("**BERT Masking Rule**\n\n15% corrupted: 80% [MASK], 10% random token, 10% unchanged. Preserves bidirectional linguistic context.")
with fc_col2:
    st.info("**T5 Span Corruption**\n\nReplaces contiguous multi-token spans with sentinel tokens (<extra_id_0>), slashing target sequence length and pretraining FLOPs.")
with fc_col3:
    st.info("**Decoder Supremacy**\n\nUnified autoregressive weights with linear KV-caching eliminate idle parameter overhead and scale predictably.")
with fc_col4:
    st.info("**Prefix Language Model**\n\nCombines bidirectional prompt attention with causal generation inside a single decoder-only backbone.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your comprehension of the 3 Transformer archetypes, pretraining objectives, and architectural trade-offs:
""")

c1 = st.checkbox("I can articulate the attention mask differences between BERT (all-to-all), GPT (causal), and T5 (dual).")
c2 = st.checkbox("I understand BERT's 80/10/10 masking rule and why RoBERTa discarded the Next Sentence Prediction task.")
c3 = st.checkbox("I can explain how T5 unifies all NLP tasks into a 'Text-to-Text' framework using span corruption.")
c4 = st.checkbox("I know why Decoder-Only architectures won the foundation model era over Encoder-Decoders.")
c5 = st.checkbox("I understand the cross-attention bridge mechanics in Encoder-Decoder models like Whisper.")
c6 = st.checkbox("I can explain how Mixture of Experts (MoE) decouples parameter capacity from per-token compute FLOPs.")

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Confirmed: You have thoroughly mastered Transformer architectural archetypes, structural trade-offs, and foundation model design patterns!")
