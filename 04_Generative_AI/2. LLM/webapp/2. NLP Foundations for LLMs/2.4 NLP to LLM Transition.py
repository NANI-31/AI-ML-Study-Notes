import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="2.4 NLP to LLM Transition - LLM Curriculum",
    page_icon="🔄",
    layout="wide"
)

# ==============================================================================
# Title Block and Executive Overview
# ==============================================================================
st.title("2.4 The Architectural Transition: From RNNs and LSTMs to Transformers")
st.caption("Chapter 2: NLP Foundations for LLMs | Domain: 2. Large Language Models")

st.markdown("""
The path from classical natural language processing to modern Foundation Large Language Models was fundamentally 
paved by breaking through the **sequential compute bottleneck**. 

For decades, recurrent neural architectures—**Vanilla RNNs, LSTMs, GRUs, and Seq2Seq networks**—reigned supreme in 
sequence modeling. However, their core mathematical formulation imposed an impassable physical constraint: 
hidden states had to be computed sequentially step-by-step, leaving thousands of GPU Tensor cores idle during training 
while suffering from exponential gradient decay over long sequences.

The definitive breakthrough occurred when **Vaswani et al. (2017) discarded recurrence entirely**, replacing temporal loops 
with parallel Multi-Head Self-Attention. This single architectural shift reduced maximum token communication path lengths 
from O(N) to O(1), unlocking mass parallelization across GPU clusters and launching the modern foundation model era.

In this lesson, you will master:
1. The 8 evolutionary milestones that transitioned natural language processing from recurrent loops to scalable foundation models.
2. The mathematical mechanics of the Sequential Compute Bottleneck and Backpropagation Through Time (BPTT).
3. How LSTMs and GRUs attempted to solve vanishing gradients with additive gating highways, and why they still hit a scaling wall.
4. Why Bahdanau dynamic attention eliminated the fixed-vector Seq2Seq bottleneck, paving the way for "Attention Is All You Need".
5. A runnable pure Python implementation directly comparing serial recurrent unrolling with parallel self-attention.
""")

st.divider()

# ==============================================================================
# Section 1: Architectural Blueprint: The Sequential Compute Bottleneck & Attention
# ==============================================================================
st.header("1. Architectural Blueprint: The Sequential Compute Bottleneck and the Attention Revolution", divider="rainbow")

st.markdown("""
Observe the multi-decade journey of sequence modeling: from serial temporal recurrence (where step t must wait for step t-1) 
to fully parallelized spatial self-attention matrices processed simultaneously across modern GPU Tensor cores:
""")

col_diag1, col_diag2 = st.columns([1.3, 0.7])

with col_diag1:
    evolution_dot = """
    digraph SequenceEvolution {
        bgcolor="#0B1120";
        rankdir=TB;
        compound=true;
        nodesep=0.35;
        ranksep=0.65;

        node [
            fontname="Helvetica",
            fontsize=10,
            fontcolor="#F8FAFC",
            shape=box,
            style="filled,rounded",
            color="#334155",
            penwidth=1.5
        ];

        edge [
            fontname="Helvetica",
            fontsize=8,
            fontcolor="#94A3B8",
            color="#38BDF8",
            penwidth=1.6,
            arrowsize=0.7
        ];

        subgraph cluster_rnn {
            label="1. Vanilla RNN (1980s - 1990s)";
            style=filled;
            color="#1E293B";
            fontcolor="#EF4444";
            R_Cell [label="Vanilla RNN Cell\\nh_t = tanh(W_h h_{t-1} + W_x x_t)\\nSevere Vanishing Gradient\\nContext: ~10 Tokens", fillcolor="#450A0A", color="#EF4444"];
        }

        subgraph cluster_lstm {
            label="2. Gated RNNs (1997 - 2014)";
            style=filled;
            color="#1E293B";
            fontcolor="#F59E0B";
            L_Cell [label="LSTM and GRU\\nForget, Input, Output Gates\\nAdditive cell state highway\\nContext: ~100-200 Tokens\\nSTILL STRICTLY SERIAL", fillcolor="#78350F", color="#F59E0B"];
        }

        subgraph cluster_seq2seq {
            label="3. Seq2Seq and Attention (2014 - 2016)";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            S_Cell [label="Seq2Seq + Bahdanau Attention\\nEncoder compresses to bottleneck\\nDecoder uses soft alignment\\nPath length: O(1) cross-attention\\nStill sequential recurrence", fillcolor="#0C4A6E", color="#38BDF8"];
        }

        subgraph cluster_transformer {
            label="4. The Transformer (2017+)";
            style=filled;
            color="#1E293B";
            fontcolor="#34D399";
            T_Cell [label="Attention Is All You Need\\nDiscard recurrence entirely!\\nQ, K, V Spatial Dot-Product\\nPath length: O(1) anywhere\\n100% Fully Parallel Training", fillcolor="#064E3B", color="#34D399", shape=hexagon, penwidth=2.5];
        }

        R_Cell -> L_Cell -> S_Cell -> T_Cell;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(evolution_dot, use_container_width=True)

with col_diag2:
    st.markdown("#### The Fundamental Mathematical Paradigm Shift")
    st.markdown("""
    * **Recurrence (Temporal Loop)**:
      `h_t = f(h_{t-1}, x_t)`
      * Time complexity for sequence of length N: **O(N) sequential steps**.
      * Training cannot be parallelized along the sequence dimension.
    * **Self-Attention (Spatial Matrix Multiply)**:
      `Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V`
      * Time complexity for sequence of length N: **O(1) sequential steps**.
      * All N tokens interact in a single parallel tensor multiplication across GPU streaming multiprocessors.
    """)

# Data Structure Architecture Diagram
st.markdown("#### Data Structure Architecture: The Sequence Modeling Progression Matrix")

seq_table_dot = """
digraph SequenceArchitectureTable {
    bgcolor="#0B1120";
    rankdir=TB;

    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=plaintext];

    TableNode [label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="8" bgcolor="#1E293B">
            <tr>
                <td bgcolor="#0F172A" colspan="5"><font color="#38BDF8"><b>Architectural Comparison: Sequence Modeling Paradigms Across Eras</b></font></td>
            </tr>
            <tr bgcolor="#334155">
                <td><b>Architecture</b></td>
                <td><b>Sequential Bottleneck</b></td>
                <td><b>Max Path Length</b></td>
                <td><b>Effective Context Horizon</b></td>
                <td><b>Training Parallelizability</b></td>
            </tr>
            <tr>
                <td bgcolor="#450A0A"><b>Vanilla RNN</b></td>
                <td>Strictly O(N) serial steps</td>
                <td>O(N) sequential steps</td>
                <td>~10 - 15 tokens</td>
                <td>Impossible (GPU underutilized)</td>
            </tr>
            <tr>
                <td bgcolor="#78350F"><b>LSTM / GRU</b></td>
                <td>Strictly O(N) serial steps</td>
                <td>O(N) sequential steps</td>
                <td>~100 - 200 tokens</td>
                <td>Impossible (GPU underutilized)</td>
            </tr>
            <tr>
                <td bgcolor="#0C4A6E"><b>Seq2Seq + Attention</b></td>
                <td>Recurrent decoder loop</td>
                <td>O(1) in cross-attention</td>
                <td>~500 - 1,000 tokens</td>
                <td>Encoder parallel; Decoder serial</td>
            </tr>
            <tr>
                <td bgcolor="#064E3B"><b>Transformer Decoder</b></td>
                <td>None during training</td>
                <td>O(1) single matrix step</td>
                <td>32k - 2,000,000 tokens</td>
                <td>100% Fully Parallel Matrix Multiplies</td>
            </tr>
        </table>
    >];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(seq_table_dot, use_container_width=True)

st.divider()

# ==============================================================================
# Section 2: Limitations of Traditional NLP
# ==============================================================================
st.header("2. Limitations of Traditional NLP: The Sequential Wall, Sparsity, and Memory Amnesia", divider="rainbow")

st.markdown("""
Traditional Natural Language Processing methods were crippled by three fundamental mathematical and physical limitations: 
**The Sequential Compute Bottleneck, Exponential Gradient Vanishing, and Fixed-Vector Information Compression**.
""")

col_lim_a, col_lim_b = st.columns(2)

with col_lim_a:
    st.markdown("#### 1. The Sequential Compute Bottleneck on Modern GPUs")
    st.markdown("""
    * **Serial Dependency**: To compute the hidden representation at step t, the recurrent network must wait for step t-1 to complete:
      `h_t = f(h_{t-1}, x_t)`
    * **Severe Hardware Underutilization**: Modern GPU architectures (such as NVIDIA H100 clusters) achieve petaflops of throughput by running millions of floating-point operations simultaneously across thousands of Tensor cores.
    * A recurrent network processing a sequence of 4,096 tokens must perform **4,096 serial lock-step memory accesses**, leaving the vast majority of GPU cores idle waiting for the next temporal state.
    """)

    st.markdown("#### 2. Vanishing and Exploding Gradients (BPTT)")
    st.markdown("""
    * **Backpropagation Through Time (BPTT)**: Computing the gradient of the loss at step N with respect to the initial hidden state h_1 requires chaining Jacobian matrices across all intermediate steps:
      `dh_N / dh_1 = Product_{t=2}^N (dh_t / dh_{t-1})`
    * If the maximum singular value of the recurrent weight matrix is less than 1.0, the gradient vanishes exponentially toward zero as N grows:
      `||dh_N / dh_1|| -> 0`
    * The network becomes mathematically incapable of learning relationships between words separated by more than 10 to 20 steps.
    """)

with col_lim_b:
    st.markdown("#### 3. Context Amnesia and Lexical Rigidity")
    st.markdown("""
    * **Context Amnesia**: Even with clever gating mechanisms, recurrent models continuously overwrite their internal memory vector at every new token. By token 200, information introduced at token 1 is virtually extinguished.
    * **Vocabulary Sparsity and Out-of-Vocabulary (OOV) Collapse**: Early NLP relied on closed word vocabularies. Any unseen morphological variation or typo produced an `<UNK>` token, destroying sentence meaning.
    * **Polysemy Collapse**: Traditional static representations (like Word2Vec) assigned a single static coordinate vector to each word, rendering them incapable of distinguishing between *"Apple the technology company"* and *"apple the fruit"*.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern Foundation LLMs overcome every single one of these limitations:
1. **Parallel Pre-training**: By replacing temporal recurrence with self-attention, training scales across tens of thousands of GPUs simultaneously using 3D parallelism (Data, Tensor, and Pipeline Parallelism).
2. **Context Horizon Expansion**: Through FlashAttention-2, RoPE positional encodings, and RingAttention, context windows have expanded from 20 tokens to **over 1,000,000 tokens** (Llama 3.1, Gemini 1.5 Pro).
3. **Lossless Byte-Fallback Tokenization**: Modern BPE tokenizers eliminate `<UNK>` tokens entirely by falling back to raw UTF-8 bytes.
""")

lim_comp_df = pd.DataFrame({
    "System Property": [
        "Training Parallelism",
        "Gradient Backpropagation Path",
        "Effective Context Window",
        "Hardware Utilization (MFU)",
        "Handling of Polysemy",
        "Out-of-Vocabulary Resilience"
    ],
    "Traditional NLP (RNNs / LSTMs)": [
        "Strictly serial; 1 token at a time",
        "Chained through N serial matrix products",
        "10 to 200 tokens maximum",
        "Very low (< 15% Model FLOPs Utilization)",
        "Collapsed to single static vector",
        "Crashes or emits <UNK> tokens"
    ],
    "Modern Foundation LLMs (Llama 3 / GPT-4)": [
        "100% fully parallel across sequence dimension",
        "Direct O(1) residual skip connections",
        "32k to 2,000,000 tokens",
        "High (45% - 60% Model FLOPs Utilization)",
        "Dynamic high-dimensional contextual manifold",
        "100% coverage via byte-fallback subword BPE"
    ]
})
st.dataframe(lim_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Word Embeddings vs Contextual Embeddings
# ==============================================================================
st.header("3. Word Embeddings vs. Contextual Embeddings: From Static Dictionaries to Deep Residual Streams", divider="rainbow")

st.markdown("""
Natural language cannot be computed as raw characters; it must be mapped into continuous geometric vector spaces. 
The representation mechanism underwent a fundamental paradigm shift from static dictionary lookups to dynamic deep 
contextual tensor representations.
""")

col_emb_a, col_emb_b = st.columns(2)

with col_emb_a:
    st.markdown("#### Static Word Embeddings (Word2Vec, GloVe, FastText)")
    st.markdown("""
    * **Dictionary Mapping**: Maintained a static lookup table `W_vocab` of dimension `[Vocab_Size, Embedding_Dim]`:
      `vector = Lookup(W_vocab, token_id)`
    * **Single Point in Space**: Every unique word in the vocabulary had exactly one fixed vector coordinate, regardless of context:
      * *Sentence 1*: "She sat on the river **bank** to read a book."
      * *Sentence 2*: "He deposited funds in the commercial **bank**."
    * In both sentences, the word **"bank"** received the exact same vector representation. The embedding could not distinguish between a geological riverbank and a financial institution.
    """)

with col_emb_b:
    st.markdown("#### Contextual Dynamic Embeddings (ELMo, BERT, Transformers)")
    st.markdown("""
    * **Dynamic Function of Entire Context**: The numerical representation of token i is not a static lookup; it is a dynamic function of all tokens in the sequence:
      `h_i = Transformer_Layer_L(x_1, x_2, ..., x_N)[i]`
    * **Residual Stream Geometry**: In modern foundation models, the initial token embedding is merely an entry point into the **residual stream**.
    * As the representation passes through 32 to 128 layers of multi-head self-attention, each token's vector dynamically shifts its coordinates based on surrounding syntax, modifiers, negations, and semantic relations.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
In models like Llama 3 and GPT-4, representations are dynamic trajectories across high-dimensional manifolds. 
A token's hidden state vector at layer 1 reflects its surface subword identity; by layer 16 it encodes syntactic dependencies; 
and by layer 32 it encodes high-level factual knowledge and conversational intent.
""")

emb_comp_df = pd.DataFrame({
    "Dimension": [
        "Mathematical Representation",
        "Polysemy Disambiguation",
        "Syntactic Sensitivity",
        "Layer Depth",
        "Storage Mechanism"
    ],
    "Static Embeddings (Word2Vec / GloVe)": [
        "v = W_vocab[token_id] (Constant vector)",
        "Impossible (All senses blended into single vector)",
        "Zero; word order is completely ignored",
        "Single shallow projection matrix",
        "Flat embedding table on disk (~1 GB)"
    ],
    "Contextual Foundation LLM Tensors": [
        "h = Deep_Transformer(x_1, ..., x_N) (Dynamic tensor)",
        "Native; coordinates shift based on context",
        "Deeply aware of syntax, negation, and word order",
        "Deep multi-layer residual stream (32 to 128 layers)",
        "Learned weight tensors across billions of parameters"
    ]
})
st.dataframe(emb_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: Recurrent Neural Networks (RNNs)
# ==============================================================================
st.header("4. Recurrent Neural Networks (RNNs): Mechanics, Backpropagation Through Time, and Failure Modes", divider="rainbow")

st.markdown("""
The Vanilla Recurrent Neural Network (Elman RNN, 1990) introduced internal memory to neural networks by feeding 
the previous hidden state back into the current step computation.
""")

col_rnn_a, col_rnn_b = st.columns(2)

with col_rnn_a:
    st.markdown("#### Mathematical Formulation of the Elman RNN")
    st.markdown("""
    At each discrete time step t, the recurrent cell computes:
    
    `h_t = tanh(W_h * h_{t-1} + W_x * x_t + b_h)`
    `y_t = Softmax(W_y * h_t + b_y)`
    
    * **State Dimension**: `h_t` has dimension `[Batch_Size, Hidden_Dimension]`.
    * **Weight Sharing**: The weight matrices `W_h` (hidden-to-hidden) and `W_x` (input-to-hidden) are identical across all time steps.
    * **Sequential Dependency**: Notice the recurrence: `h_t` cannot be computed until `h_{t-1}` is fully known.
    """)

with col_rnn_b:
    st.markdown("#### The Mechanics of Backpropagation Through Time (BPTT)")
    st.markdown("""
    To train an RNN, errors from the final loss are backpropagated backward through time across all historical steps:
    
    `dL / dh_1 = (dL / dh_N) * (dh_N / dh_{N-1}) * ... * (dh_2 / dh_1)`
    
    * Each Jacobian term `dh_t / dh_{t-1}` involves multiplying by `W_h^T * diag(1 - tanh^2(...))`.
    * Because the derivative of `tanh` is bounded between 0 and 1:
      `|1 - tanh^2(z)| <= 1.0`
    * Chaining N consecutive matrix multiplications causes the gradient magnitude to decay exponentially as (|lambda_max| * 1)^N.
    * For sequences longer than 10-15 tokens, gradients completely vanish before reaching the earliest tokens.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern LLMs completely eliminated temporal recurrent loops. During training, all N tokens of a sequence are passed into 
the Transformer simultaneously as a single 3D tensor `[Batch_Size, Sequence_Length, Hidden_Dimension]`. Gradients propagate 
directly across residual skip connections in O(1) operations, eliminating vanishing gradients along the sequence length.
""")

rnn_comp_df = pd.DataFrame({
    "Property": [
        "Compute Architecture",
        "Sequence Execution",
        "Gradient Propagation",
        "Effective Context Span",
        "Hardware Compatibility"
    ],
    "Vanilla RNN": [
        "Recurrent temporal loop",
        "Strictly sequential O(N) steps",
        "Multiplicative chain through BPTT (Vanishing)",
        "~5 to 15 tokens",
        "Poor; cannot exploit GPU matrix engines"
    ],
    "Modern Foundation LLM": [
        "Spatial self-attention tensor operations",
        "100% fully parallel matrix multiplications",
        "Direct additive residual skip connections",
        "32k to 2,000,000 tokens",
        "Optimal; runs at peak Tensor Core throughput"
    ]
})
st.dataframe(rnn_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: Long Short-Term Memory (LSTM)
# ==============================================================================
st.header("5. Long Short-Term Memory (LSTM): The Constant Error Carousel and Gated Information Flow", divider="rainbow")

st.markdown("""
Invented by Hochreiter and Schmidhuber (1997), the Long Short-Term Memory network was engineered specifically 
to overcome the vanishing gradient problem in Vanilla RNNs.
""")

col_lstm_a, col_lstm_b = st.columns(2)

with col_lstm_a:
    st.markdown("#### The 3 Gates and the Constant Error Carousel")
    st.markdown("""
    LSTMs split the recurrent state into two vectors: the **Hidden State** `h_t` and the **Cell State** `c_t`. 
    Information flow is regulated by three learned non-linear gates:
    
    1. **Forget Gate (f_t)**: Decides what fraction of prior memory to discard:
       `f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)`
    2. **Input Gate (i_t)**: Decides which new candidate values to write to memory:
       `i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)`
       `c_tilde_t = tanh(W_c * [h_{t-1}, x_t] + b_c)`
    3. **Cell State Update (Additive Highway)**:
       `c_t = f_t * c_{t-1} + i_t * c_tilde_t`
    4. **Output Gate (o_t)**: Decides what parts of the cell state to emit:
       `o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)`
       `h_t = o_t * tanh(c_t)`
    """)

with col_lstm_b:
    st.markdown("#### Why LSTMs Still Hit a Hard Scaling Wall")
    st.markdown("""
    * **Additive Gradient Highway**: Notice the cell state update: `c_t = f_t * c_{t-1} + ...`. 
      Because the update is **additive** rather than purely multiplicative, errors can propagate backward without decaying exponentially when f_t ~ 1.0.
    * **The Unbreakable Sequential Wall**: Despite solving vanishing gradients for moderate sequences (~100 to 200 tokens), LSTMs **remained strictly sequential**.
    * Step t still requires the output of step t-1. When trying to train models on billions of tokens across GPU clusters, LSTMs could not be parallelized across the sequence dimension.
    * **Information Bottleneck**: Compressing an entire paragraph into a single fixed hidden vector h_t inevitably caused memory saturation and forgotten context.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Transformers borrowed the most profound intuition of the LSTM—**additive linear highways that protect gradient flow**—and 
generalized it into the **Residual Stream** (`x_{l+1} = x_l + Attention(x_l)`). By replacing gated recurrence with 
pure additive skip connections, Transformers achieved perfect gradient propagation without any sequential compute bottleneck.
""")

lstm_comp_df = pd.DataFrame({
    "Feature": [
        "Memory State",
        "Gradient Highway",
        "Gating Mechanism",
        "Max Path Length",
        "Parallel Training"
    ],
    "LSTM (Hochreiter & Schmidhuber 1997)": [
        "Dual vectors: Hidden State h_t and Cell State c_t",
        "Additive cell state carousel (f_t * c_{t-1})",
        "3 separate gates (Forget, Input, Output)",
        "O(N) sequential recurrent steps",
        "Impossible (Serial temporal loop)"
    ],
    "Modern Foundation LLM": [
        "Residual stream tensor [Batch, Seq_Len, Hidden_Dim]",
        "Residual skip connections: x + SubLayer(x)",
        "Gated activations in MLP (SwiGLU) instead of recurrence",
        "O(1) single matrix dot-product step",
        "100% Fully Parallel across GPUs"
    ]
})
st.dataframe(lstm_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: Gated Recurrent Units (GRUs)
# ==============================================================================
st.header("6. Gated Recurrent Units (GRUs): Parameter Pruning and Coupled State Dynamics", divider="rainbow")

st.markdown("""
Introduced by Cho et al. (2014), the Gated Recurrent Unit (GRU) simplified the complex multi-gate architecture 
of the LSTM to reduce computational overhead and parameter counts.
""")

col_gru_a, col_gru_b = st.columns(2)

with col_gru_a:
    st.markdown("#### The Coupled Architecture of the GRU")
    st.markdown("""
    The GRU collapses the separate cell state and hidden state into a single hidden vector `h_t`, regulated by only **two gates**:
    
    1. **Update Gate (z_t)**: Controls how much of the past state is kept versus updated (combining LSTM forget and input gates):
       `z_t = sigmoid(W_z * [h_{t-1}, x_t] + b_z)`
    2. **Reset Gate (r_t)**: Controls how much of the previous hidden state is remembered when computing candidate state:
       `r_t = sigmoid(W_r * [h_{t-1}, x_t] + b_r)`
    3. **Candidate Hidden State**:
       `h_tilde_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)`
    4. **Final Hidden State (Linear Interpolation)**:
       `h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t`
    """)

with col_gru_b:
    st.markdown("#### Performance vs. Fundamental Limits")
    st.markdown("""
    * **Efficiency Gain**: By eliminating one gate and merging states, GRUs reduced parameter count by ~25% compared to LSTMs and trained slightly faster on small datasets.
    * **Coupled Trade-Off**: The linear interpolation `(1 - z_t) * h_{t-1} + z_t * h_tilde_t` strictly couples forgetting and remembering: forgetting old information is mandatory to write new information.
    * **The Persistent Recurrent Barrier**: Like LSTMs, GRUs remained trapped in the sequential temporal loop. They could not solve the fundamental challenge of scaling to massive datasets on GPU clusters.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
In modern architectures, computational efficiency is achieved not by pruning recurrent gates, but by **sparse parameter activation** 
(Mixture of Experts - MoE, like Mixtral and DeepSeek) and **Grouped-Query Attention (GQA)**. These innovations reduce inference FLOPs 
and memory bandwidth while retaining 100% parallel GPU pretraining.
""")

gru_comp_df = pd.DataFrame({
    "Criterion": [
        "Number of Gates",
        "Internal Memory States",
        "Parameter Efficiency",
        "Temporal Coupling",
        "Modern LLM Role"
    ],
    "GRU (Cho et al. 2014)": [
        "2 gates (Reset and Update)",
        "Single hidden state vector h_t",
        "~25% fewer parameters than LSTM",
        "Forgetting and writing are strictly coupled",
        "Largely obsolete; found only in legacy edge microcontrollers"
    ],
    "Modern Foundation LLM": [
        "No recurrent gates; Self-Attention + MLP",
        "High-dimensional continuous residual stream",
        "Massive parameter scaling with MoE sparsity",
        "Decoupled attention across all sequence positions",
        "Dominates all frontier natural language systems"
    ]
})
st.dataframe(gru_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: Seq2Seq Models
# ==============================================================================
st.header("7. Seq2Seq Models: Encoder-Decoder Architecture and the Fixed-Vector Bottleneck", divider="rainbow")

st.markdown("""
Sequence-to-Sequence (Seq2Seq) architectures (Sutskever et al., 2014) extended deep learning from fixed-length inputs 
to arbitrary variable-length mapping tasks, such as machine translation, document summarization, and dialogue.
""")

col_seq_a, col_seq_b = st.columns(2)

with col_seq_a:
    st.markdown("#### The Classical Encoder-Decoder Formulation")
    st.markdown("""
    Seq2Seq connects two distinct recurrent networks:
    
    1. **Encoder RNN**: Consumes the input sequence `(x_1, x_2, ..., x_T)` token-by-token, updating its hidden state until the final token:
       `v = h_T = Encoder(x_1, ..., x_T)`
       Where `v` is the **Fixed-Length Context Vector** (typically 512 or 1024 floats).
    2. **Decoder RNN**: Initialized with vector `v`, the decoder autoregressively emits target tokens one by one:
       `s_t = Decoder(s_{t-1}, y_{t-1}, v)`
       `P(y_t) = Softmax(W_out * s_t)`
    """)

with col_seq_b:
    st.markdown("#### The Fixed-Vector Information Bottleneck")
    st.markdown("""
    * **The Fundamental Bottleneck**: Regardless of whether the source sentence contained 5 words or a complex 80-word legal clause, the encoder was forced to compress **all information into a single fixed vector `v`**.
    * **Performance Degradation Curve**: In empirical studies (Bahdanau et al., 2014), Seq2Seq translation performance plummeted precipitously for sentences longer than 20-30 words.
    * The vector `v` simply did not have the information capacity to preserve names, numbers, negation, and complex grammatical relations simultaneously.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern Foundation LLMs (such as Llama 3 and GPT-4) are **Decoder-Only Causal Transformers**. They completely eliminate 
the separate encoder-decoder split and the fixed context vector bottleneck. The prompt and generation share a single 
unbroken causal context window where every generated token has direct self-attention access to every single prompt token.
""")

seq_comp_df = pd.DataFrame({
    "System Property": [
        "Architecture",
        "Context Passing Mechanism",
        "Information Bottleneck",
        "Long-Sentence Degradation",
        "Multi-Task Flexibility"
    ],
    "Classical Seq2Seq (2014)": [
        "Separate Encoder RNN + Decoder RNN",
        "Single fixed-size context vector v = h_T",
        "Severe; 512 numbers must store entire sentence",
        "Catastrophic drop in BLEU score after 25 words",
        "Bespoke model trained per language pair"
    ],
    "Modern Foundation LLM (Decoder-Only)": [
        "Unified Autoregressive Causal Transformer",
        "Full Key-Value (KV) cache of all prior tokens",
        "None; direct token-to-token attention access",
        "Retains coherence across multi-million token contexts",
        "Unified multi-task polyglot zero-shot generalist"
    ]
})
st.dataframe(seq_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: Attention-Based Models
# ==============================================================================
st.header("8. Attention-Based Models: Bahdanau Additive Attention and Dynamic Soft Alignment", divider="rainbow")

st.markdown("""
In 2014, Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio introduced **Neural Machine Translation by Jointly Learning to Align and Translate**. 
This paper introduced the attention mechanism that fundamentally altered the history of artificial intelligence.
""")

col_att_a, col_att_b = st.columns(2)

with col_att_a:
    st.markdown("#### The Bahdanau Additive Attention Formulation")
    st.markdown("""
    Instead of forcing the decoder to rely on a single static context vector `v`, the Bahdanau architecture computes a 
    **dynamic context vector c_t at every decoding step**:
    
    1. **Alignment Score (Additive MLP)**: Compares the decoder's prior state s_{t-1} with each encoder hidden state h_i:
       `e_{t, i} = v_a^T * tanh(W_a * s_{t-1} + U_a * h_i)`
    2. **Softmax Attention Weights**: Converts scores into a normalized probability distribution across input tokens:
       `alpha_{t, i} = exp(e_{t, i}) / Sum_j exp(e_{t, j})`
    3. **Dynamic Context Vector**: Computes the weighted sum over all encoder hidden states:
       `c_t = Sum_i alpha_{t, i} * h_i`
    * The decoder uses c_t to emit the next target token.
    """)

with col_att_b:
    st.markdown("#### Why Attention Was Revolutionary")
    st.markdown("""
    * **Eliminated the Bottleneck**: The decoder can dynamically look back at any part of the input sentence, focusing on the specific words relevant to the current translation step.
    * **Path Length Dropped to O(1)**: The maximum path length between any input token h_i and output token dropped from O(N) sequential steps to **a single O(1) soft-alignment lookup**.
    * **Interpretability**: For the first time, researchers could visualize attention weight heatmaps, revealing soft word-alignment dictionaries between languages (e.g., aligning French nouns with English adjectives).
    * **The Lingering Limitation**: While attention connected encoder and decoder states in O(1), the underlying backbone **was still a recurrent network** requiring step-by-step temporal unrolling.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Vaswani et al. (2017) observed Bahdanau's breakthrough and realized: **If attention provides direct O(1) communication 
between tokens, why do we need recurrent loops at all?** They replaced Bahdanau's additive attention (`tanh(W s + U h)`) 
with **Scaled Dot-Product Attention** (`Softmax(Q K^T / sqrt(d_k)) V`), replacing slow sequential loops with blazing-fast 
hardware-optimized matrix multiplications.
""")

att_comp_df = pd.DataFrame({
    "Attention Formulation": [
        "Alignment Function",
        "Mathematical Form",
        "Hardware Optimization",
        "Underlying Model",
        "Sequential Bottleneck"
    ],
    "Bahdanau Attention (2014)": [
        "Additive (Single-layer MLP)",
        "v^T * tanh(W s + U h)",
        "Poor; requires explicit tanh and vector loops",
        "Recurrent RNN / BiLSTM",
        "Still present in recurrent state propagation"
    ],
    "Transformer Attention (Vaswani 2017+)": [
        "Multiplicative (Scaled Dot-Product)",
        "Softmax(Q K^T / sqrt(d_k)) * V",
        "Optimal; dense BLAS General Matrix Multiplies (GEMM)",
        "Pure Feed-Forward & Self-Attention",
        "Zero during training (100% fully parallel)"
    ]
})
st.dataframe(att_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: Transformer-Based NLP
# ==============================================================================
st.header("9. Transformer-Based NLP: Discarding Recurrence and Unleashing GPU Scalability", divider="rainbow")

st.markdown("""
In June 2017, the Google Brain team published the seminal paper: **"Attention Is All You Need"**. 
By completely stripping away recurrence and convolution, the Transformer established the universal computational 
substrate powering all modern foundation artificial intelligence.
""")

col_tf_a, col_tf_b = st.columns(2)

with col_tf_a:
    st.markdown("#### The Core Pillars of the Transformer Architecture")
    st.markdown("""
    1. **Scaled Dot-Product Self-Attention**:
       Every token generates three vector projections: **Query (Q)**, **Key (K)**, and **Value (V)**:
       `Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V`
       * Computed simultaneously for all N tokens via parallel matrix multiplications:
         `Q, K, V in R^{[Batch, Seq_Len, Hidden_Dim]}`
    2. **Multi-Head Attention (MHA)**: Projects inputs into H distinct representation subspaces simultaneously, allowing the model to attend to syntax, facts, and coreference in parallel.
    3. **Positional Encoding (RoPE / ALiBi)**: Since Transformers discard temporal loops, spatial word order is injected via sinusoidal or Rotary Position Embeddings.
    4. **Residual Streams & LayerNorm**: Additive skip connections maintain uninhibited gradient highways throughout 100+ transformer layers.
    """)

with col_tf_b:
    st.markdown("#### Why Transformers Scaled to Frontier AI")
    st.markdown("""
    * **Compute Alignment with GPUs**: Every core operation in a Transformer is a **Dense Matrix Multiplication (GEMM)**. GPU hardware accelerators are specifically engineered to execute GEMMs at maximum theoretical FLOPS.
    * **The Scaling Laws (Kaplan et al., Chinchilla)**: As model parameter count (N) and pretraining token dataset size (D) scale, test cross-entropy loss drops predictably as a clean power-law:
      `Loss approx (N_c / N)^{alpha_N} + (D_c / D)^{alpha_D}`
    * This empirical scaling reliability unlocked tens of billions of dollars in AI supercomputing investments, scaling models from 110M parameters (BERT) to 405B+ parameters (Llama 3.1).
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
State-of-the-art foundation models (Llama 3, Claude 3.5, Mistral Large, GPT-4) employ refined Transformer decoders:
* **FlashAttention-2 / 3**: Kernel fusion techniques that eliminate slow GPU High Bandwidth Memory (HBM) read/writes, computing attention directly in fast SRAM.
* **Grouped-Query Attention (GQA)**: Sharing key and value heads across multiple query heads to slash KV cache memory consumption during autoregressive inference.
* **SwiGLU Activation Functions**: Replacing ReLU with gated non-linearities for faster parameter convergence.
""")

tf_comp_df = pd.DataFrame({
    "Feature": [
        "Sequence Modeling Mechanism",
        "Path Length Across Sequence",
        "Training Throughput",
        "Scaling Law Predictability",
        "State of the Art Dominance"
    ],
    "Classical Recurrent NLP": [
        "Serial hidden state loops h_t = f(h_{t-1}, x_t)",
        "O(N) sequential steps",
        "Extremely slow; bounded by serial dependencies",
        "Hits severe performance saturation early",
        "Obsolete in modern foundation AI"
    ],
    "Modern Transformer Foundation LLMs": [
        "Parallel Multi-Head Scaled Dot-Product Self-Attention",
        "O(1) single matrix dot-product step",
        "Ultra-high; scales across 100,000+ GPUs simultaneously",
        "Predictable power-law scaling across compute, data, and parameters",
        "Universal standard for all frontier AI models"
    ]
})
st.dataframe(tf_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 10: Production Implementation: Pure Python Recurrent Hidden State vs. Self-Attention Step
# ==============================================================================
st.header("10. Production Implementation: Recurrence vs. Attention in Pure Python", divider="rainbow")

st.markdown("""
The self-contained Python script below directly contrasts the computational mechanics of **Serial Recurrent State Propagation** 
with **Parallel Scaled Dot-Product Self-Attention**. It measures execution cycles, simulates gradient path attenuation, 
and validates tensor outputs using pure NumPy:
""")

st.code('''import numpy as np
import time
from typing import Dict, Any, Tuple

class SequenceModelingBenchmark:
    """
    Directly contrasts the computational mechanics of Recurrent Unrolling (O(N) steps)
    with Scaled Dot-Product Self-Attention (O(1) parallel steps).
    """

    def __init__(self, hidden_dim: int = 64):
        self.d = hidden_dim
        np.random.seed(42)
        
        # Recurrent weights: W_h (hidden-to-hidden) and W_x (input-to-hidden)
        self.W_h = np.random.randn(self.d, self.d) * 0.1
        self.W_x = np.random.randn(self.d, self.d) * 0.1
        
        # Self-Attention projection weights: W_q, W_k, W_v
        self.W_q = np.random.randn(self.d, self.d) * 0.1
        self.W_k = np.random.randn(self.d, self.d) * 0.1
        self.W_v = np.random.randn(self.d, self.d) * 0.1

    def run_recurrent_forward(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Executes Vanilla RNN unrolling step-by-step.
        X shape: [Sequence_Length, Hidden_Dim]
        Enforces O(N) sequential dependency: h_t requires h_{t-1}.
        """
        seq_len, _ = X.shape
        h = np.zeros(self.d)
        h_states = []
        
        start_time = time.perf_counter()
        for t in range(seq_len):
            # Strict serial dependency: must wait for previous h
            h = np.tanh(np.dot(self.W_h, h) + np.dot(self.W_x, X[t]))
            h_states.append(h)
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return np.array(h_states), elapsed_ms

    def run_self_attention_forward(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Executes Scaled Dot-Product Self-Attention.
        X shape: [Sequence_Length, Hidden_Dim]
        100% Fully Parallel Matrix Multiplications: zero temporal loops!
        """
        seq_len, _ = X.shape
        start_time = time.perf_counter()
        
        # 1. Parallel linear projections across all N tokens simultaneously
        Q = np.dot(X, self.W_q)  # [N, d]
        K = np.dot(X, self.W_k)  # [N, d]
        V = np.dot(X, self.W_v)  # [N, d]
        
        # 2. Scaled dot-product scores: [N, N]
        scores = np.dot(Q, K.T) / np.sqrt(self.d)
        
        # 3. Softmax across sequence dimension
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 4. Context aggregation: [N, d]
        output = np.dot(weights, V)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return output, elapsed_ms

    def compute_gradient_path_lengths(self, seq_len: int) -> Dict[str, Any]:
        """
        Quantifies the theoretical distance that gradients must travel 
        from token N back to token 1.
        """
        return {
            "sequence_length": seq_len,
            "recurrent_path_length": seq_len,  # Must traverse all intermediate states
            "transformer_path_length": 1,      # Direct O(1) attention connection
            "recurrent_sequential_steps": seq_len,
            "transformer_sequential_steps": 1
        }

# Execution demonstration
if __name__ == "__main__":
    benchmark = SequenceModelingBenchmark(hidden_dim=128)
    sequence_length = 500
    dummy_input = np.random.randn(sequence_length, 128)
    
    rnn_out, rnn_time = benchmark.run_recurrent_forward(dummy_input)
    attn_out, attn_time = benchmark.run_self_attention_forward(dummy_input)
    metrics = benchmark.compute_gradient_path_lengths(sequence_length)
    
    print(f"RNN Forward ({sequence_length} steps): {rnn_time:.3f} ms")
    print(f"Self-Attention ({sequence_length} tokens): {attn_time:.3f} ms")
    print(f"Max Gradient Path Length: RNN={metrics['recurrent_path_length']} vs Transformer={metrics['transformer_path_length']}")
''', language="python")

st.divider()

# ==============================================================================
# Section 11: Interactive Workbench: Context Horizon & Vanishing Gradient Simulator
# ==============================================================================
st.header("11. Interactive Workbench: Context Horizon & Vanishing Gradient Simulator", divider="rainbow")

st.markdown("""
Simulate how gradient signals and semantic information degrade over sequence length across different 
sequence modeling architectures. Observe the mathematical contrast between **exponential recurrent decay** 
and **constant attention connectivity**:
""")

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    seq_length_slider = st.slider("Sequence Length (Tokens)", min_value=10, max_value=200, value=60, step=10)
    
    decay_rate = st.slider(
        "Recurrent Weight Matrix Eigenvalue / Attenuation Factor (lambda)",
        min_value=0.70,
        max_value=0.99,
        value=0.88,
        step=0.01,
        help="Simulates the largest eigenvalue of the transition matrix W_h. Values < 1.0 cause exponential vanishing."
    )

with col_wb2:
    selected_arch = st.selectbox(
        "Select Sequence Modeling Architecture to Inspect:",
        [
            "Vanilla RNN (Elman)",
            "LSTM (Gated Highway)",
            "Bahdanau Seq2Seq Attention",
            "Transformer Self-Attention"
        ]
    )

# Simulation computations
steps = np.arange(1, seq_length_slider + 1)

# Theoretical gradient retention curve from token N backward to token 1
if selected_arch == "Vanilla RNN (Elman)":
    # Exponential decay: lambda^(N - t)
    retention = np.power(decay_rate, seq_length_slider - steps)
    path_len = seq_length_slider
    parallel_train = "0% (Strictly Serial)"
    gpu_mfu = "8% - 14%"
elif selected_arch == "LSTM (Gated Highway)":
    # Slower decay due to constant error carousel: (decay + 0.08)^(N - t)
    lstm_rate = min(0.97, decay_rate + 0.07)
    retention = np.power(lstm_rate, seq_length_slider - steps)
    path_len = seq_length_slider
    parallel_train = "0% (Strictly Serial)"
    gpu_mfu = "12% - 18%"
elif selected_arch == "Bahdanau Seq2Seq Attention":
    # Cross-attention provides high signal retention, but recurrent decoder attenuates
    retention = 0.5 + 0.45 * np.power(min(0.98, decay_rate + 0.05), (seq_length_slider - steps) / 4)
    path_len = 1
    parallel_train = "Partial (Encoder only)"
    gpu_mfu = "25% - 35%"
else:
    # Transformer: Direct O(1) attention connection + residual highway keeps retention ~1.0
    retention = np.full(seq_length_slider, 0.96)
    path_len = 1
    parallel_train = "100% (Fully Parallel)"
    gpu_mfu = "48% - 60%"

chart_df = pd.DataFrame({
    "Token Position (1 = Start, N = Current)": steps,
    "Gradient Signal Retained": np.clip(retention, 0.0, 1.0)
})

st.markdown("#### Gradient Backpropagation Signal Retention from Token N to Token 1")
st.line_chart(chart_df.set_index("Token Position (1 = Start, N = Current)"), height=260)

col_wb_res1, col_wb_res2, col_wb_res3, col_wb_res4 = st.columns(4)

with col_wb_res1:
    st.metric("Max Token-to-Token Path Length", f"O({path_len})")

with col_wb_res2:
    st.metric("Signal at Token 1", f"{retention[0] * 100:.2f}%")

with col_wb_res3:
    st.metric("Training Parallelizability", parallel_train)

with col_wb_res4:
    st.metric("Expected GPU MFU", gpu_mfu)

st.divider()

# ==============================================================================
# Section 12: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Mastering the transition from recurrent architectures to modern Transformers requires answering 
in-depth questions on GPU systems mechanics, memory bounds, and scaling dynamics.
""")

st.subheader("Principal / Staff-Level Interview Questions")

with st.expander("Q1: Why is Transformer pretraining 100% parallel, but Transformer inference is still inherently sequential?"):
    st.markdown("""
    **Direct Answer:**
    The difference stems from **training with Teacher Forcing** versus **inference with Autoregressive Generation**:
    
    * **During Training (Teacher Forcing)**: The ground-truth target sequence is completely known in advance. In causal language modeling, causal masking (a lower-triangular mask) ensures that token t cannot attend to tokens t+1. Because all inputs are available simultaneously, all N tokens are multiplied in parallel via large General Matrix Multiplies (GEMMs) in O(1) operations across thousands of GPU cores.
    * **During Inference (Autoregressive Decoding)**: Token t cannot be computed until token t-1 has been sampled. Therefore, generation is inherently sequential: token-by-token generation requires N sequential memory-bound forward passes, storing historical keys and values in the **KV Cache**.
    """)

with st.expander("Q2: Mathematically, why does Scaled Dot-Product Attention scale as O(N^2) in compute and memory, while RNN inference is O(1)?"):
    st.markdown("""
    **Direct Answer:**
    In Self-Attention, computing attention scores requires multiplying the Query matrix Q of shape `[N, d]` with the Key matrix K of shape `[N, d]`:
    
    `Scores = Q * K^T   # Shape: [N, N]`
    
    * This matrix multiply requires 2 * N^2 * d floating point operations.
    * Storing the intermediate attention score matrix requires O(N^2) memory bandwidth. If N = 100,000, N^2 = 10,000,000,000 values per attention head per layer, causing immediate GPU Out-Of-Memory (OOM) errors unless FlashAttention is used.
    * Conversely, an RNN updates a single constant-sized hidden vector `h_t = f(h_{t-1}, x_t)` regardless of sequence length, consuming O(1) compute and memory per step at the cost of catastrophic memory amnesia.
    """)

with st.expander("Q3: How did LSTMs address vanishing gradients, and why did Transformers abandon recurrence for residual skip connections?"):
    st.markdown("""
    **Direct Answer:**
    LSTMs introduced the **Constant Error Carousel (Additive Cell State Highway)**:
    `c_t = f_t * c_{t-1} + i_t * c_tilde_t`
    Because the cell state update is additive rather than purely multiplicative, errors can propagate backward without decaying exponentially when the forget gate f_t ~ 1.0.
    
    However, LSTMs still required O(N) sequential operations. Transformers eliminated recurrence entirely by adopting **Residual Skip Connections**:
    `x_{l+1} = x_l + SubLayer(x_l)`
    In backpropagation, the gradient of the loss with respect to x_l is:
    `dL / dx_l = dL / dx_{l+1} * (I + dSubLayer / dx_l)`
    The identity matrix I provides an unobstructed linear highway that routes gradient signals directly across 100+ transformer layers without attenuation, while completely decoupling operations across the sequence length.
    """)

with st.expander("Q4: What are State Space Models (Mamba, RWKV) and why are researchers exploring them as alternatives to Transformers?"):
    st.markdown("""
    **Direct Answer:**
    While Transformers achieve unmatched pretraining scaling, their O(N^2) attention compute and linear KV cache memory during inference create severe latency and VRAM bottlenecks for long documents.
    
    **State Space Models (like Mamba)** discretize continuous differential equations into selective linear recurrent state spaces:
    1. **Parallel Training**: They reformulate linear recurrence into a parallel associative scan, training in O(N) time on GPUs.
    2. **Constant-Memory Inference**: During generation, Mamba maintains a constant O(1) hidden state vector, generating tokens without a growing KV cache.
    * Frontier models currently explore **Hybrid Architectures** (e.g. Jamba), combining Mamba layers for long-range state compression with Transformer attention layers for precise associative fact retrieval.
    """)

with st.expander("Q5: What is the KV Cache in Transformer inference, and why does its memory footprint grow linearly with sequence length?"):
    st.markdown("""
    **Direct Answer:**
    During autoregressive inference, generating token t requires computing attention against all prior tokens 1, ..., t-1. 
    Without caching, the model would redundantly recompute Key (K) and Value (V) projections for all past tokens at every step, causing O(N^2) redundant compute.
    
    The **KV Cache** stores the precomputed Key and Value tensors in GPU memory for all historical tokens:
    `KV_Memory = 2 * Batch_Size * Sequence_Length * Num_Layers * Num_KV_Heads * Head_Dim * Bytes_Per_Param`
    * For a 70B model with a 128k context window in FP16, the KV Cache for a single user query consumes **over 40 GB of VRAM**, exceeding the memory required for the model weights themselves!
    * This is why techniques like Grouped-Query Attention (GQA), FP8/FP4 KV caching, and vLLM PagedAttention are essential in production.
    """)

st.subheader("Production Engineering Traps")

st.error("""
🚨 **Trap 1 (Error): Confusing Parallel Training with Autoregressive Inference Decoding**
Engineers often assume that because Transformers train in parallel across millions of tokens, generation must also be fast and parallel. 
In reality, generation is strictly sequential and memory-bandwidth bound. Every single token requires streaming all model weights and KV cache tensors from GPU HBM into Tensor Cores. 
*Remediation*: Deploy continuous batching runtimes (vLLM, TensorRT-LLM) using PagedAttention and speculative decoding to maximize memory bandwidth utilization.
""")

st.error("""
🚨 **Trap 2 (Error): Quadratic Memory OOM on Long Sequences without FlashAttention**
Naively executing standard PyTorch eager attention (`torch.matmul(Q, K.T)`) on sequences longer than 8,192 tokens creates a massive `[Batch, Heads, N, N]` intermediate matrix in high-bandwidth memory (HBM), instantly causing GPU CUDA Out-Of-Memory errors.
*Remediation*: Always enable FlashAttention-2 or FlashAttention-3 (`torch.nn.functional.scaled_dot_product_attention`), which fuses the Softmax calculation into GPU SRAM blocks using online softmax normalization, reducing intermediate memory from O(N^2) to O(N).
""")

st.warning("""
⚠️ **Trap 3 (Warning): Recurrent Gradient Shattering in High-Depth BPTT**
When maintaining legacy RNN or LSTM pipelines for edge microcontrollers, unrolling Backpropagation Through Time over more than 100 steps without gradient clipping (`torch.nn.utils.clip_grad_norm_`) causes catastrophic numerical NaN explosions or vanished gradients.
*Remediation*: Enforce strict gradient norm clipping (threshold 1.0) and use Truncated BPTT (TBPTT) with window sizes under 35 steps.
""")

st.warning("""
⚠️ **Trap 4 (Warning): The Fallacy of Unlimited Context Windows (The Lost-in-the-Middle Phenomenon)**
Although modern Transformers support 128k to 2M token context windows, empirical studies show that attention retrieval accuracy degrades severely when critical evidence is located in the middle of a massive prompt ("Lost-in-the-Middle"). 
*Remediation*: Do not dump hundreds of raw unranked documents into the context window. Use hybrid search and cross-encoder re-ranking to place the highest-signal context immediately adjacent to the query at the very beginning or end of the prompt.
""")

st.subheader("Concept Flashcards")

col_fc1, col_fc2 = st.columns(2)

with col_fc1:
    with st.container():
        st.info("**Sequential Compute Bottleneck**")
        st.caption("The fundamental hardware limitation where recurrent states h_t must be computed one step at a time, leaving parallel GPU Tensor cores idle during training.")

    with col_fc2:
        with st.container():
            st.info("**Path Length: O(N) vs. O(1)**")
            st.caption("Recurrent networks require N sequential steps for signals to travel between tokens; Self-Attention connects any two tokens in a single O(1) dot-product matrix multiply.")

with col_fc1:
    with st.container():
        st.info("**Scaled Dot-Product Attention**")
        st.caption("The core matrix operation of the Transformer: Softmax(Q K^T / sqrt(d_k)) * V, running as dense GEMM operations at peak GPU floating point utilization.")

with col_fc2:
    with st.container():
        st.info("**Residual Gradient Highway**")
        st.caption("Additive skip connections x + SubLayer(x) that route backpropagation gradients directly across 100+ layers without exponential decay or recurrence.")

st.divider()

# ==============================================================================
# Section 13: Self-Assessment Checklist
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of the architectural transition from classical recurrent NLP to modern Foundation Transformers. 
Complete all 6 checklist items:
""")

cb1 = st.checkbox("1. I understand why the sequential compute bottleneck in RNNs prevents parallel training on modern GPU clusters.", key="chk_24_1")
cb2 = st.checkbox("2. I can explain the mechanics of Backpropagation Through Time (BPTT) and why Vanilla RNNs suffer from vanishing gradients.", key="chk_24_2")
cb3 = st.checkbox("3. I know how LSTMs and GRUs use additive gating highways to preserve memory, and why they still hit a scaling wall.", key="chk_24_3")
cb4 = st.checkbox("4. I understand how Bahdanau attention eliminated the fixed-vector Seq2Seq bottleneck by computing dynamic context vectors.", key="chk_24_4")
cb5 = st.checkbox("5. I can formulate Scaled Dot-Product Self-Attention and explain how it reduced token communication path length to O(1).", key="chk_24_5")
cb6 = st.checkbox("6. I understand why Transformer pretraining is 100% parallel while autoregressive inference is sequential and memory-bound.", key="chk_24_6")

if cb1 and cb2 and cb3 and cb4 and cb5 and cb6:
    st.success("🎉 Outstanding! You have mastered the complete architectural evolution from classical NLP to modern Foundation Large Language Models!")
