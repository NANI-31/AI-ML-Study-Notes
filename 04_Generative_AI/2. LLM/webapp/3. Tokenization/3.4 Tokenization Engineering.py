import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="3.4 Tokenization Engineering - LLM Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# ==============================================================================
# Title Block and Executive Overview
# ==============================================================================
st.title("3.4 Tokenization Engineering: Context Budgeting, Padding, Truncation, and Systems Runtime")
st.caption("Chapter 3: Tokenization | Domain: 2. Large Language Models")

st.markdown("""
Tokenization is far more than an abstract linguistic exercise; it is an intensive, high-throughput **systems engineering discipline**. 
In production generative AI architectures, tokenization components dictate API billing economics, GPU memory allocation, 
batch throughput, and context window survival.

Every single query processed by a frontier model must undergo rigorous engineering operations: 
**forward token-to-ID lookups, reverse stream decoding with UTF-8 byte reconstruction, strict context-length budgeting, 
left-padding for batch homogenization, information-preserving truncation, and attention mask tensor synthesis**.

In this lesson, you will master:
1. The 8 critical engineering operations governing production tokenization pipelines.
2. How to design robust context-length budgeting systems across system rules, RAG passages, and chat history.
3. Why Center (Middle) Truncation outperforms naive Head or Tail truncation in retrieval-augmented workflows.
4. How streaming decoders buffer partial multi-byte UTF-8 sequences to prevent replacement character glitches.
5. A runnable pure Python implementation of a complete tokenization engineering suite with deterministic validation.
""")

st.divider()

# ==============================================================================
# Section 1: Architectural Blueprint: The Production Tokenization Runtime Engine
# ==============================================================================
st.header("1. Architectural Blueprint: The Production Tokenization Runtime Engine", divider="rainbow")

st.markdown("""
Observe the complete production token engineering lifecycle: transforming raw, variable-length heterogeneous 
text streams into aligned, budgeted, and masked mathematical tensors ready for GPU execution:
""")

col_diag1, col_diag2 = st.columns([1.3, 0.7])

with col_diag1:
    runtime_dot = """
    digraph TokenRuntime {
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

        subgraph cluster_ingress {
            label="1. Ingress & Token Conversion";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";
            RawText [label="Raw Text Payload\\nSystem + History + Query", fillcolor="#1E293B"];
            RustTokenizer [label="Rust / C++ Engine\\n1M+ tokens/sec throughput", fillcolor="#0C4A6E", color="#38BDF8"];
            TokenIDs [label="Raw Token IDs\\n[128000, 42, 891, ...]", fillcolor="#1E293B"];
            
            RawText -> RustTokenizer -> TokenIDs;
        }

        subgraph cluster_budget {
            label="2. Context Budget & Slicing";
            style=filled;
            color="#1E293B";
            fontcolor="#F59E0B";
            BudgetCheck [label="Budget Verification\\nPrompt <= Max_Context - Reserve", fillcolor="#78350F", color="#F59E0B"];
            Truncator [label="Center Truncator\\nPreserves System & Query", fillcolor="#78350F", color="#F59E0B"];
            
            TokenIDs -> BudgetCheck -> Truncator;
        }

        subgraph cluster_batch {
            label="3. Tensor Alignment & Masking";
            style=filled;
            color="#1E293B";
            fontcolor="#34D399";
            LeftPad [label="Left-Padding Engine\\nAligns Batch to Max_Len", fillcolor="#064E3B", color="#34D399"];
            AttnMask [label="Attention Mask Tensor\\n1 = Real, 0 = Masked Pad", fillcolor="#065F46", color="#34D399"];
            GPU_Tensor [label="GPU Ingress Tensors\\nInput_IDs & Mask in VRAM", fillcolor="#047857", color="#34D399", shape=hexagon];
            
            Truncator -> LeftPad;
            Truncator -> AttnMask;
            LeftPad -> GPU_Tensor;
            AttnMask -> GPU_Tensor;
        }
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(runtime_dot, use_container_width=True)

with col_diag2:
    st.markdown("#### Systems Engineering Objectives")
    st.markdown("""
    * **Zero GPU Starvation**: Ensure tokenization throughput exceeds GPU ingestion rates (1M+ tokens/sec using compiled Rust kernels).
    * **Strict Boundary Enforcement**: Mathematically guarantee that prompts never exceed hardware context limits, eliminating 400 Bad Request errors.
    * **Tensor Homogenization**: Align heterogeneous sequence lengths into uniform matrix blocks using left-padding while preserving causal next-token pointers.
    * **Lossless Output Assembly**: Incrementally buffer multi-byte UTF-8 fragments during streaming generation, preventing replacement glyph corruption.
    """)

# Data Structure Comparison Matrix
st.markdown("#### Data Structure Architecture: The 8 Tokenization Engineering Operations")

ops_table_dot = """
digraph OpsTable {
    bgcolor="#0B1120";
    rankdir=TB;

    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=plaintext];

    TableNode [label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="8" bgcolor="#1E293B">
            <tr>
                <td bgcolor="#0F172A" colspan="4"><font color="#38BDF8"><b>Tokenization Engineering Operations Matrix</b></font></td>
            </tr>
            <tr bgcolor="#334155">
                <td><b>Operation</b></td>
                <td><b>Algorithm / Strategy</b></td>
                <td><b>Primary System Concern</b></td>
                <td><b>Complexity</b></td>
            </tr>
            <tr>
                <td bgcolor="#0C4A6E"><b>1. Token-to-ID Conversion</b></td>
                <td>Trie / Hash Map vocabulary index lookup</td>
                <td>CPU cache misses and thread lockups</td>
                <td>O(1) amortized lookup</td>
            </tr>
            <tr>
                <td bgcolor="#1E1B4B"><b>2. ID-to-Token Conversion</b></td>
                <td>Array indexing with UTF-8 byte buffer</td>
                <td>Incomplete multi-byte split in streaming</td>
                <td>O(1) constant lookup</td>
            </tr>
            <tr>
                <td bgcolor="#78350F"><b>3. Token Counting</b></td>
                <td>Exact client-side BPE tokenizer execution</td>
                <td>Discrepancies causing billing billing drift</td>
                <td>O(N) text scan</td>
            </tr>
            <tr>
                <td bgcolor="#064E3B"><b>4. Context Budgeting</b></td>
                <td>Hierarchical dynamic allocation</td>
                <td>Context window overflow crashing API</td>
                <td>O(1) arithmetic check</td>
            </tr>
            <tr>
                <td bgcolor="#451A03"><b>5. Padding</b></td>
                <td>Mandatory Left-Padding for causal decoders</td>
                <td>Wasted FLOPs and positional pointer skew</td>
                <td>O(Max_Len) tensor copy</td>
            </tr>
            <tr>
                <td bgcolor="#312E81"><b>6. Truncation</b></td>
                <td>Center (Middle) truncation pruning</td>
                <td>Loss of critical instructions or final query</td>
                <td>O(N) slice operation</td>
            </tr>
            <tr>
                <td bgcolor="#042F2E"><b>7. Attention Masking</b></td>
                <td>Binary 2D/4D mask tensor or FlashAttn offset</td>
                <td>Preventing dummy PAD tokens from biasing states</td>
                <td>O(N^2) or O(1) in kernel</td>
            </tr>
            <tr>
                <td bgcolor="#1F2937"><b>8. Throughput Efficiency</b></td>
                <td>Rust Rayon multithreading &amp; memory mapping</td>
                <td>GPU starvation during high-QPS serving</td>
                <td>1,000,000+ tokens/sec</td>
            </tr>
        </table>
    >];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(ops_table_dot, use_container_width=True)

st.divider()

# ==============================================================================
# Section 2: 1. Token-to-ID Conversion
# ==============================================================================
st.header("2. Token-to-ID Conversion: Forward Vocabulary Lookup and Hash Map Mechanics", divider="rainbow")

st.markdown("""
Token-to-ID conversion is the forward mapping function:

`f: Subword_String -> Integer_ID in {0, 1, ..., V - 1}`

Every natural language token produced by the pre-tokenizer and merge engine must be converted into a discrete 
integer index to retrieve corresponding continuous embedding vectors from GPU memory.
""")

col_tid1, col_tid2 = st.columns(2)

with col_tid1:
    st.markdown("#### Algorithmic Lookup Mechanics")
    st.markdown("""
    * **Hash Map Architecture**:
      * In naive Python, vocabularies are stored as `dict[str, int]`.
      * In compiled production engines (Hugging Face `tokenizers` in Rust, `tiktoken` in C++), vocabularies are implemented using **Perfect Hashing** or **Radix Prefix Trees (Tries)**.
      * This ensures O(1) amortized lookup time per subword while keeping the hash table tightly packed in CPU L2/L3 cache memory.
    * **Byte-Pair Traversal**:
      * During BPE encoding, checking whether an adjacent symbol pair `(A, B)` exists in the vocabulary requires querying the merge rank table:
        `rank = Merge_Table.get((A, B), Infinity)`
      * Highly optimized hash tables allow evaluating thousands of candidate pairs per microsecond.
    """)

with col_tid2:
    st.markdown("#### The GPU Decoupling Principle")
    st.markdown("""
    * **CPU-Bound Bottleneck**: Token-to-ID conversion is an integer-and-string operation executed entirely on the CPU.
    * **Asynchronous Pre-Tokenization**: In modern inference servers (such as vLLM and Triton Inference Server), token-to-ID conversion runs on dedicated background CPU worker threads.
    * Converted integer ID arrays are transferred asynchronously via Non-Volatile Memory Express (NVMe) or PCIe DMA directly into GPU High Bandwidth Memory (HBM), ensuring GPU Tensor cores are never stalled waiting for string parsing.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Frontier tokenizers (like Llama 3's `tiktoken` implementation) achieve lookup throughputs exceeding **1.5 million tokens per second** 
per CPU core. By avoiding intermediate string allocations and operating directly over UTF-8 byte slices, modern runtimes 
eliminate Python garbage collection pauses during large batch processing.
""")

tid_comp_df = pd.DataFrame({
    "Implementation": ["Pure Python dict", "Hugging Face Rust Tokenizers", "OpenAI tiktoken (Rust/C++)"],
    "Underlying Data Structure": ["Hash Table (PyDictObject)", "Radix Tree + Double-Array Trie", "Optimized BPE Merge Hash Table"],
    "Throughput per Core": ["~40,000 tokens/sec", "~1,200,000 tokens/sec", "~1,500,000+ tokens/sec"],
    "Memory Overhead": ["High (~4x raw string size)", "Minimal (Zero-copy byte slices)", "Minimal (Compact binary table)"],
    "Production Readiness": ["Prototype only", "Enterprise standard", "Frontier LLM standard"]
})
st.dataframe(tid_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: 2. ID-to-Token Conversion
# ==============================================================================
st.header("3. ID-to-Token Conversion: Detokenization, Byte Assembly, and Stream Decoding", divider="rainbow")

st.markdown("""
ID-to-token conversion is the reverse mapping function:

`f^{-1}: Integer_ID -> Subword_String`

While batch detokenization is computationally trivial, real-time **token streaming** introduces subtle systems challenges.
""")

col_idt1, col_idt2 = st.columns(2)

with col_idt1:
    st.markdown("#### Direct Array Indexing")
    st.markdown("""
    * **Constant Time Lookup**:
      Because token IDs are dense zero-indexed integers `0, 1, ..., V - 1`, the reverse lookup table is a contiguous array:
      `token_bytes = vocab_array[token_id]`
      Retrieval executes in O(1) constant time with a single memory pointer dereference.
    * **Lossless Concatenation**:
      In Byte-Level BPE, each token ID maps to a sequence of raw UTF-8 bytes. Detokenizing an entire response is achieved by concatenating all byte arrays and decoding to UTF-8:
      `decoded_string = b"".join(vocab_array[tid] for tid in generated_ids).decode("utf-8")`
    """)

with col_idt2:
    st.markdown("#### The Streaming Multi-Byte Decoding Trap")
    st.markdown("""
    * In modern conversational applications, tokens are streamed to the user interface via Server-Sent Events (SSE) token-by-token.
    * **The UTF-8 Splitting Glitch**:
      A multi-byte character (e.g. `'🚀'`, which consists of 4 bytes: `0xF0 0x9F 0x9A 0x80`, or Chinese characters requiring 3 bytes) may be emitted across **multiple distinct tokens**:
      * Token 1 emits: `0xF0 0x9F` (Incomplete UTF-8 prefix!)
      * Token 2 emits: `0x9A 0x80` (Remaining suffix)
    * If a streaming server naively calls `.decode("utf-8")` on Token 1 alone, Python crashes with a `UnicodeDecodeError` or emits the dreaded replacement glyph: **`` (U+FFFD)**!
    * **The Solution**: Production stream decoders maintain an internal **Pending Byte Buffer**. They append new bytes, check for valid UTF-8 character boundaries, and only yield complete decoded characters to the client.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Hugging Face's `TextStreamer` and OpenAI client SDKs employ incremental UTF-8 decoders (`codecs.getincrementaldecoder("utf-8")`). 
This guarantees that international characters, emojis, and math notations stream smoothly to user screens without flickering replacement characters.
""")

idt_comp_df = pd.DataFrame({
    "Decoding Mode": [
        "Batch Detokenization",
        "Naive Token Streaming",
        "Production Incremental Streaming"
    ],
    "Byte Handling": [
        "Concatenates all bytes before decoding",
        "Attempts to decode each token ID in isolation",
        "Maintains pending byte buffer across tokens"
    ],
    "Risk of Replacement Glyph ()": [
        "Zero (Complete sequence guaranteed)",
        "Extremely High on emojis and non-Latin scripts",
        "Zero (Flushes only upon valid UTF-8 boundary)"
    ],
    "Perceived Latency": [
        "High (Must wait for entire generation to finish)",
        "Low (Immediate token streaming)",
        "Optimal (Immediate streaming with zero corruption)"
    ]
})
st.dataframe(idt_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: 3. Token Counting
# ==============================================================================
st.header("4. Token Counting: Accuracy, API Billing Mechanics, and Estimations", divider="rainbow")

st.markdown("""
In generative AI engineering, **tokens are the universal currency**. 
API providers bill strictly per token, context windows are bounded by tokens, and rate limits are metered in 
Tokens Per Minute (TPM). Accurately counting tokens before dispatching API calls is mandatory for cost control.
""")

col_cnt1, col_cnt2 = col_cnt1, col_cnt2 = st.columns(2)

with col_cnt1:
    st.markdown("#### Why Heuristic Word Counts Fail")
    st.markdown("""
    A common novice mistake is estimating token counts using crude heuristics such as:
    `estimated_tokens = len(text) / 4   # HIGH RISK!`
    
    This heuristic breaks down catastrophically in production:
    1. **Code & JSON Payloads**: Syntax characters (`{`, `}`, `[`, `]`, `\n\n`, indentation) tokenize into far more tokens than natural prose. A 1,000-character JSON payload can easily consume **600 tokens** (1.6 chars/token).
    2. **Multilingual Text**: In languages such as Hindi, Korean, or Arabic, words are decomposed into individual byte clusters, averaging **1.2 to 2.0 chars/token**.
    3. **Numbers & Math**: Large numbers (e.g. `"104928502"`) are partitioned into multiple numeric fragments.
    """)

with col_cnt2:
    st.markdown("#### Pre-Flight Token Metering Architecture")
    st.markdown("""
    Production AI gateways (such as LiteLLM, Portkey, and LangSmith) execute **pre-flight client-side token counting**:
    * Run the exact tokenizer locally (using compiled `tiktoken` or Hugging Face tokenizers).
    * Compute exact token counts for the prompt before initiating network requests.
    * **Pre-Flight Guardrails**:
      * Enforce client-side rate limiters to prevent hitting provider 429 errors.
      * Reject requests that exceed context limits before incurring API costs.
      * Dynamically estimate dollar costs:
        `Cost = (Prompt_Tokens * Price_In) + (Max_Output_Tokens * Price_Out)`
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern frontier models enforce asymmetric pricing: output tokens are typically **3x to 4x more expensive** than input tokens 
due to the memory-bandwidth bottleneck of autoregressive generation. Pre-flight token counting allows enterprises to budget 
and cap `max_tokens` defensively.
""")

cnt_comp_df = pd.DataFrame({
    "Text Category": ["English Prose", "Python / TypeScript Code", "JSON Payload", "Hindi / Devanagari", "Arabic Script"],
    "Characters per Token": ["~4.2 to 4.6", "~2.8 to 3.2", "~1.8 to 2.4", "~1.4 to 1.9", "~1.5 to 2.1"],
    "Heuristic Error (chars/4)": ["Minimal (< 10%)", "Underestimates by 35%", "Underestimates by 55%", "Underestimates by 65%", "Underestimates by 60%"],
    "Production Impact": ["Acceptable for estimation", "Unexpected bill spikes", "Frequent context overflow", "Severe rate-limit exhaustion", "Severe rate-limit exhaustion"]
})
st.dataframe(cnt_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: 4. Context-Length Calculation
# ==============================================================================
st.header("5. Context-Length Calculation: Budgeting System, RAG, Chat History, and Reserve Output", divider="rainbow")

st.markdown("""
Every foundation model has a hard physical ceiling: the **Maximum Context Window** (C_max). 
Exceeding this limit causes an immediate HTTP 400 Bad Request error or CUDA assertion crash. 
Engineering a production prompt requires dynamic context window budgeting.
""")

col_ctx1, col_ctx2 = st.columns(2)

with col_ctx1:
    st.markdown("#### The Anatomy of a Context Budget")
    st.markdown("""
    The total context window must be partitioned into strict, non-negotiable budget allocations:
    
    `Total_Budget = C_max   # e.g. 8,192 or 128,000 tokens`
    
    1. **Reserve Output Buffer (B_out)**: The maximum tokens allocated for the assistant's reply (e.g. 2,048 tokens). Must be subtracted first!
    2. **System Prompt (B_sys)**: Permanent behavioral guardrails and persona definitions.
    3. **Tool Definitions (B_tools)**: JSON schemas for external API calling capabilities.
    4. **Active User Query (B_user)**: The immediate prompt submitted by the operator.
    5. **Dynamic Allocation Pool (B_avail)**:
       `B_avail = C_max - B_out - B_sys - B_tools - B_user`
       This remaining pool is shared between **Retrieved RAG Documents** and **Historical Chat Turns**.
    """)

with col_ctx2:
    st.markdown("#### Dynamic Sliding Windows & Compaction")
    st.markdown("""
    When conversational turns accumulate:
    * If `Chat_History + RAG_Docs > B_avail`, the system must intervene before making the API call:
      1. **Sliding Window Pruning**: Discard the oldest conversational turns while strictly retaining the system prompt.
      2. **Hierarchical Summarization**: Periodically summarize turns 1 through 10 into an executive brief, replacing 2,000 historical tokens with a 150-token summary.
      3. **RAG Chunk Truncation**: Dynamically drop lower-ranked retrieval chunks to stay within the allocated document budget.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Even with models supporting 128k (Llama 3) or 1,000,000 tokens (Gemini 1.5 Pro), budgeting remains vital: 
processing 500,000 tokens incurs substantial compute latency (several seconds of Time-to-First-Token) and high financial cost. 
Smart budgeting optimizes response speed and operational economics.
""")

ctx_budget_df = pd.DataFrame({
    "Component": ["Reserve Output Buffer", "System Rules & Personas", "Tool Calling Schemas", "User Question", "RAG Documents", "Chat History"],
    "Priority Level": ["Mandatory (Cannot truncate)", "Mandatory (Cannot truncate)", "Mandatory (Cannot truncate)", "Highest Context Priority", "High (Prunable by relevance)", "Dynamic (Sliding window pruning)"],
    "Typical Budget Share": ["15% - 25%", "5% - 10%", "5% - 15%", "5% - 10%", "30% - 40%", "20% - 30%"],
    "Action when Exceeded": ["Caps max_new_tokens", "Static; reviewed by dev", "Static; review schema size", "Must reject if query > B_avail", "Drop bottom-ranked chunks", "Compress or slide window"]
})
st.dataframe(ctx_budget_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: 5. Padding
# ==============================================================================
st.header("6. Padding: Batch Homogenization, Left-Padding Rules, and Compute Overhead", divider="rainbow")

st.markdown("""
Padding equalizes the length of distinct sequences within a batch so they can be processed as a uniform 
2D tensor of shape `[Batch_Size, Max_Sequence_Length]`.
""")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown("#### The Left-Padding Imperative for Causal Decoders")
    st.markdown("""
    In causal autoregressive generation (GPT, Llama, Mistral):
    * **Right-Padding is Broken**:
      `Row A: [Word_1, Word_2, PAD, PAD]`
      Generating the next token samples off the final sequence index `logits[-1]`. In right-padding, index `-1` is a PAD token! The model predicts a continuation of padding, resulting in corrupt outputs.
    * **Left-Padding is Required**:
      `Row A: [PAD, PAD, Word_1, Word_2]`
      Index `-1` is always the legitimate last word of the prompt (`Word_2`), ensuring mathematically correct next-token generation.
    * Real tokens retain their relative causal attention order, and all PAD positions are ignored via `attention_mask = 0`.
    """)

with col_p2:
    st.markdown("#### The Compute Overhead Tax of Padding")
    st.markdown("""
    * **Wasted FLOPs**: If a batch contains seven 50-token prompts and one 2,000-token prompt:
      * All 7 short prompts are padded out to 2,000 tokens.
      * Total tokens processed: 8 * 2,000 = 16,000 tokens.
      * Real tokens: 7 * 50 + 2,000 = 2,350 tokens.
      * **Over 85% of GPU matrix multiplications are wasted on meaningless PAD tokens!**
    * **Production Remedy**:
      * **Bucketing / Length Sorting**: Group prompts of similar lengths into the same batch.
      * **PagedAttention (vLLM)**: Decouples batch allocation into non-contiguous physical pages, completely eliminating padding during generation.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern inference serving engines (vLLM, TensorRT-LLM) use **Continuous Batching with PagedAttention**. 
Requests join and exit the running batch dynamically on a token-by-token basis, eliminating both the padding compute tax 
and memory fragmentation.
""")

pad_tax_df = pd.DataFrame({
    "Batching Strategy": ["Static Eager Batching", "Bucket-Sorted Batching", "Continuous Batching (PagedAttention)"],
    "Padding Mechanism": ["Pads all rows to batch max length", "Pads rows to bucket-specific max length", "Zero padding; ragged memory pages"],
    "Wasted Compute": ["Extremely High (up to 70% wasted FLOPs)", "Low (10% - 20% wasted FLOPs)", "0% (Every FLOP computes real tokens)"],
    "Throughput (req/sec)": ["Baseline (1.0x)", "~2.5x throughput gain", "~5x to 8x throughput gain"],
    "Standard Deployment": ["Legacy PyTorch scripts", "High-throughput fine-tuning", "Modern enterprise serving (vLLM)"]
})
st.dataframe(pad_tax_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: 6. Truncation
# ==============================================================================
st.header("7. Truncation: Strategies, Head vs. Tail Pruning, and Information Preservation", divider="rainbow")

st.markdown("""
When an input sequence inevitably exceeds its allocated token budget, truncation algorithms must prune tokens. 
The choice of truncation strategy dictates whether critical instructions or user queries survive.
""")

col_tr1, col_tr2 = st.columns(2)

with col_tr1:
    st.markdown("#### The 3 Classical Truncation Strategies")
    st.markdown("""
    1. **Tail Truncation (Right-Truncation)**:
       * Discards tokens from the **end** of the sequence: `tokens[:max_len]`.
       * **The Catastrophic Flaw**: In conversational prompts and RAG, the user's actionable query is located at the very end of the text. Tail truncation cuts off the user's question, causing the model to answer an incomplete prompt!
    2. **Head Truncation (Left-Truncation)**:
       * Discards tokens from the **beginning** of the sequence: `tokens[-max_len:]`.
       * **The Catastrophic Flaw**: Cuts off the opening system instructions, safety guardrails, and role definitions!
    3. **Center / Middle Truncation**:
       * Preserves the head (k tokens of system instructions) and preserves the tail (m tokens of user query), discarding text from the **middle** of the document:
         `tokens[:k] + tokens[-m:]`
       * Ensures that both operational rules and the actionable query survive intact.
    """)

with col_tr2:
    st.markdown("#### Semantic Chunk Pruning (The Modern Approach)")
    st.markdown("""
    Modern RAG pipelines avoid blunt token-level slicing altogether:
    * Instead of blindly slicing across sentences in the middle of a document, the system operates at the **chunk level**.
    * Uses Cross-Encoder relevance scores to rank all candidate passages.
    * Iteratively removes the **lowest-scoring passage chunks** until the total token count fits within the budget.
    * This guarantees that every surviving chunk is syntactically complete and semantically relevant.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Production agents pair Center Truncation with hierarchical compaction. When truncating chat history, 
the system keeps the system prompt (head) and the last 3 conversational turns (tail), condensing the middle turns 
into an automated summary.
""")

tr_comp_df = pd.DataFrame({
    "Truncation Method": ["Tail Truncation (Right)", "Head Truncation (Left)", "Center Truncation (Middle)", "Semantic Chunk Pruning"],
    "System Prompt Preservation": ["Guaranteed intact", "Severely compromised", "Guaranteed intact", "Guaranteed intact"],
    "User Query Preservation": ["Severely compromised", "Guaranteed intact", "Guaranteed intact", "Guaranteed intact"],
    "Syntactic Sentence Integrity": ["Cuts mid-sentence", "Cuts mid-sentence", "Cuts mid-sentence", "100% complete paragraphs"],
    "Recommended Use Case": ["Never in conversational AI", "Single-turn document reading", "Multi-turn chat history", "Enterprise RAG pipelines"]
})
st.dataframe(tr_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: 7. Attention Masks
# ==============================================================================
st.header("8. Attention Masks: Bidirectional, Causal, and Tensor Dimensions", divider="rainbow")

st.markdown("""
Attention masks are the mathematical enforcement mechanisms that dictate which tokens are allowed to attend to 
which other tokens during the self-attention operation:

`Attention(Q, K, V) = Softmax((Q * K^T) / sqrt(d_k) + Mask) * V`
""")

col_mask_a, col_mask_b = st.columns(2)

with col_mask_a:
    st.markdown("#### The Anatomy of Attention Mask Types")
    st.markdown("""
    1. **Padding Mask**:
       * A 1D binary vector per sequence: `[0, 0, 1, 1, 1]`.
       * Zeros indicate PAD positions; ones indicate real content tokens.
       * Added to logit scores as -1e9 so padding positions receive 0% Softmax weight.
    2. **Causal (Autoregressive) Mask**:
       * A 2D lower-triangular matrix of shape `[Seq_Len, Seq_Len]`:
         ```
         [[  0, -inf, -inf],
          [  0,    0, -inf],
          [  0,    0,    0]]
         ```
       * Ensures token t can only attend to historical tokens 1, ..., t, preventing the model from cheating by looking ahead into future tokens during training.
    3. **Combined 4D Attention Mask**:
       * Shape: `[Batch_Size, 1, Seq_Len, Seq_Len]`. Combines padding masking and causal masking into a single unified tensor.
    """)

with col_mask_b:
    st.markdown("#### Hardware Kernel Optimization (FlashAttention)")
    st.markdown("""
    * **The Memory Waste of Explicit Masks**:
      Materializing a 4D attention mask tensor for a 32k context window consumes gigabytes of GPU VRAM just for dummy masking numbers!
    * **FlashAttention On-Chip Causal Masking**:
      Modern GPU kernels (FlashAttention-2 and FlashAttention-3) **do not materialize mask tensors in memory at all**.
      Instead, the causal masking condition is hard-coded into the GPU CUDA thread coordinates:
      ```cuda
      if (col_idx > row_idx) {
          score = -INFINITY;
      }
      ```
      This eliminates mask tensor memory allocations entirely and accelerates attention compute by 3x.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
In frontier pretraining and inference, explicit attention mask tensors are completely absent. 
Engines rely entirely on CUDA kernel causal indexing and ragged sequence boundary arrays (`cu_seqlens`).
""")

mask_types_df = pd.DataFrame({
    "Mask Architecture": ["Padding Mask (1D)", "Causal Mask (2D)", "Combined 4D Tensor", "FlashAttention Kernel (0D)"],
    "Tensor Shape": ["[Batch, Seq_Len]", "[Seq_Len, Seq_Len]", "[Batch, 1, Seq_Len, Seq_Len]", "None (Implicit CUDA logic)"],
    "Memory Footprint": ["Minimal", "Moderate (Quadratic in Seq_Len)", "Massive (O(B * N^2) VRAM)", "Zero memory allocation"],
    "Underlying Operation": ["Logit addition (-1e9)", "Upper-triangular -inf addition", "Broadcasted tensor addition", "On-chip SRAM thread indexing"],
    "Production Role": ["Encoder batching", "Standard PyTorch causal attention", "Legacy Hugging Face pipelines", "State-of-the-art LLM serving"]
})
st.dataframe(mask_types_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: 8. Tokenization Efficiency
# ==============================================================================
st.header("9. Tokenization Efficiency: Rust Implementations, Throughput, and Multiprocessing", divider="rainbow")

st.markdown("""
When training foundation models on 15 trillion tokens (such as Llama 3) or serving tens of thousands of concurrent users, 
tokenization throughput can easily become the primary system bottleneck.
""")

col_eff1, col_eff2 = st.columns(2)

with col_eff1:
    st.markdown("#### Why Python Fails at Scale")
    st.markdown("""
    * **Global Interpreter Lock (GIL)**: Python's GIL prevents true multi-threaded CPU execution of string operations across multiple cores.
    * **Garbage Collection Overhead**: Creating and destroying millions of Python string objects creates severe memory fragmentation and GC pauses.
    * **Throughput Ceiling**: A pure Python BPE tokenizer typically processes **30,000 to 50,000 tokens per second**. At this rate, tokenizing a 1-terabyte dataset would take over **8 months** on a modern workstation!
    """)

with col_eff2:
    st.markdown("#### The Rust & C++ Revolution")
    st.markdown("""
    * **Hugging Face `tokenizers` (Rust)**:
      * Implemented in native Rust with the **Rayon** multi-threading library.
      * Processes text using zero-copy memory slices directly over memory-mapped raw files (`mmap`).
      * Achieves **over 1,200,000 tokens per second** per CPU core.
    * **OpenAI `tiktoken`**:
      * Custom BPE engine engineered in Rust and C for maximum raw speed.
      * Avoids regex backtracking and compiles merge tables into compact binary structures.
    * **Pre-Tokenized Binary Datasets**:
      * In pretraining pipelines, tokenization is performed **once offline**.
      * Tokens are saved as memory-mapped binary files (e.g. NumPy `uint16` / `uint32` arrays or Arrow tables).
      * The training loop simply memory-maps the binary files, streaming integer IDs directly to GPUs at bus speed.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Pre-tokenizing trillion-token datasets offline and storing them as contiguous binary arrays reduces training cluster startup 
time from hours to seconds and guarantees 100% GPU Model FLOPs Utilization (MFU).
""")

eff_comp_df = pd.DataFrame({
    "Tokenization Pipeline": [
        "Pure Python Loop",
        "Python Multiprocessing (Pool)",
        "Hugging Face Rust Tokenizers",
        "Offline Pre-Tokenized Memmap"
    ],
    "Language / Runtime": ["Python (Single thread)", "Python (Process fork)", "Rust / C++ (Rayon multithreading)", "Direct C/C++ memory map (mmap)"],
    "Throughput per Machine": ["~40k tokens/sec", "~350k tokens/sec", "~10M+ tokens/sec (64 cores)", "Hardware bus speed (> 50GB/sec)"],
    "GPU Starvation Risk": ["100% Guaranteed GPU stall", "High under heavy load", "Near zero", "Zero (Instant DMA transfer)"],
    "Storage Format": ["Raw .txt / .jsonl", "Raw .txt / .jsonl", "Raw or Arrow chunks", "Binary uint16 / uint32 arrays (.bin)"]
})
st.dataframe(eff_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 10: Production Implementation: Complete Token Engineering Suite
# ==============================================================================
st.header("10. Production Implementation: Complete Token Engineering Suite in Python", divider="rainbow")

st.markdown("""
The self-contained Python implementation below demonstrates the full spectrum of production tokenization engineering: 
forward/reverse lookups, exact context window budgeting, Center Truncation, Left-Padding with binary attention masks, 
and incremental streaming UTF-8 decoding:
""")

st.code('''from typing import List, Dict, Any, Tuple
import codecs

class ProductionTokenEngineeringSuite:
    """
    Production-grade tokenization engineering suite implementing:
    - Context-length budgeting and reserve calculation
    - Center (Middle) Truncation
    - Mandatory Left-Padding with Attention Mask generation
    - Incremental multi-byte UTF-8 stream decoding
    """

    def __init__(self, vocab_size: int = 128256, max_context_window: int = 8192):
        self.vocab_size = vocab_size
        self.max_context = max_context_window
        self.PAD_ID = 128004
        self.BOS_ID = 128000
        self.EOS_ID = 128001
        
        # Incremental stream decoder
        self.stream_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def calculate_context_budget(
        self, 
        system_tokens: int, 
        tools_tokens: int, 
        user_tokens: int, 
        reserve_output: int = 2048
    ) -> Dict[str, Any]:
        """
        Calculates remaining budget available for dynamic RAG documents and chat history.
        """
        fixed_overhead = system_tokens + tools_tokens + user_tokens + reserve_output
        available_dynamic = self.max_context - fixed_overhead
        
        is_overflow = available_dynamic < 0
        return {
            "max_context_limit": self.max_context,
            "fixed_overhead_tokens": fixed_overhead,
            "reserve_output_tokens": reserve_output,
            "available_for_rag_and_history": max(0, available_dynamic),
            "is_overflow": is_overflow,
            "overflow_amount": abs(available_dynamic) if is_overflow else 0
        }

    def center_truncate(self, tokens: List[int], max_allowed: int, head_preserve: int = 50, tail_preserve: int = 50) -> List[int]:
        """
        Applies Center (Middle) Truncation: preserves head (system/opening context)
        and tail (user query), discarding text from the middle.
        """
        if len(tokens) <= max_allowed:
            return tokens
            
        if head_preserve + tail_preserve >= max_allowed:
            # Fallback to balanced head/tail slice
            half = max_allowed // 2
            return tokens[:half] + tokens[-half:]
            
        # Keep head and tail, slice out the middle
        head_slice = tokens[:head_preserve]
        tail_slice = tokens[-tail_preserve:]
        return head_slice + tail_slice

    def left_pad_batch(self, batch_token_ids: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Executes MANDATORY Left-Padding for causal autoregressive decoders.
        Generates corresponding binary attention masks.
        """
        max_batch_len = max(len(seq) for seq in batch_token_ids)
        padded_batch = []
        attention_masks = []
        
        for seq in batch_token_ids:
            num_pads = max_batch_len - len(seq)
            # Prepend PAD IDs to the left
            padded_row = [self.PAD_ID] * num_pads + seq
            mask_row = [0] * num_pads + [1] * len(seq)
            
            padded_batch.append(padded_row)
            attention_masks.append(mask_row)
            
        return padded_batch, attention_masks

    def stream_decode_chunk(self, raw_bytes: bytes, final_chunk: bool = False) -> str:
        """
        Incrementally decodes byte streams, buffering incomplete multi-byte UTF-8 fragments.
        """
        return self.stream_decoder.decode(raw_bytes, final=final_chunk)

# Verification Demonstration
if __name__ == "__main__":
    suite = ProductionTokenEngineeringSuite(max_context_window=4096)
    
    # 1. Budget Verification
    budget = suite.calculate_context_budget(system_tokens=250, tools_tokens=150, user_tokens=100, reserve_output=1024)
    print("Context Budget Diagnostics:", budget)
    assert not budget["is_overflow"], "Budget must not overflow for sample inputs!"
    
    # 2. Center Truncation Verification
    long_sequence = list(range(1000))  # 1000 tokens
    truncated = suite.center_truncate(long_sequence, max_allowed=100, head_preserve=40, tail_preserve=40)
    print(f"Center Truncated Length: {len(truncated)} (Original: {len(long_sequence)})")
    assert truncated[:40] == long_sequence[:40], "Head tokens must be strictly preserved!"
    assert truncated[-40:] == long_sequence[-40:], "Tail tokens must be strictly preserved!"
    
    # 3. Left-Padding Verification
    sample_batch = [[10, 20, 30], [100, 200, 300, 400, 500]]
    padded, masks = suite.left_pad_batch(sample_batch)
    print("Padded Row 0 (Should have 2 left pads):", padded[0])
    print("Attention Mask Row 0:", masks[0])
    assert padded[0][0] == suite.PAD_ID and padded[0][1] == suite.PAD_ID
    assert masks[0] == [0, 0, 1, 1, 1], "Mask must accurately zero out pad tokens!"
    
    # 4. Stream Decoding Test (Multi-byte split simulation)
    # The rocket emoji '🚀' is 4 bytes: b'\\xf0\\x9f\\x9a\\x80'
    rocket_part1 = b"\\xf0\\x9f"
    rocket_part2 = b"\\x9a\\x80"
    out1 = suite.stream_decode_chunk(rocket_part1)  # Incomplete! Should buffer and emit nothing
    out2 = suite.stream_decode_chunk(rocket_part2, final_chunk=True)  # Complete! Emits emoji
    print(f"Stream Chunk 1 Output: '{out1}' (Buffered successfully)")
    print(f"Stream Chunk 2 Output: '{out2}' (Complete emoji restored)")
    assert out1 == "" and out2 == "🚀", "Stream decoder must prevent replacement character glitch!"
    print("All Engineering Assertions Passed!")
''', language="python")

st.divider()

# ==============================================================================
# Section 11: Interactive Workbench: Context Budgeter & Truncation Studio
# ==============================================================================
st.header("11. Interactive Workbench: Context Budgeter, Padding & Truncation Studio", divider="rainbow")

st.markdown("""
Simulate enterprise context window budgeting, Left-Padding tensor alignment, and Center Truncation. 
Adjust the component sliders below to evaluate remaining RAG document capacity and avoid context overflows:
""")

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.markdown("#### Context Budget Allocator")
    cfg_max_ctx = st.selectbox(
        "Hardware Max Context Limit (C_max):",
        [4096, 8192, 16384, 32768, 128256],
        index=1
    )
    
    cfg_reserve = st.slider("Reserve Output Buffer (Max Generation Tokens):", min_value=256, max_value=4096, value=1500, step=128)
    cfg_sys = st.slider("System Instructions & Persona Tokens:", min_value=50, max_value=1500, value=350, step=50)
    cfg_tools = st.slider("Tool Definitions & API Schema Tokens:", min_value=0, max_value=2000, value=400, step=50)
    cfg_user = st.slider("Immediate User Prompt Tokens:", min_value=20, max_value=2000, value=250, step=10)

with col_wb2:
    st.markdown("#### Truncation & Padding Controls")
    cfg_rag_tokens = st.slider("Retrieved RAG Documents Input:", min_value=500, max_value=10000, value=4500, step=250)
    cfg_chat_tokens = st.slider("Chat History Tokens:", min_value=200, max_value=10000, value=2000, step=200)
    
    trunc_method = st.radio(
        "Truncation Strategy if Overflow Occurs:",
        ["Center (Middle) Truncation", "Tail Truncation (Right)", "Head Truncation (Left)"]
    )

# Compute calculations
total_fixed = cfg_reserve + cfg_sys + cfg_tools + cfg_user
available_pool = cfg_max_ctx - total_fixed
requested_dynamic = cfg_rag_tokens + cfg_chat_tokens
total_requested = total_fixed + requested_dynamic

is_overflow = total_requested > cfg_max_ctx
overflow_amount = max(0, total_requested - cfg_max_ctx)

st.markdown("#### Context Budget Allocation Diagnostics")

col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.metric("Max Window Limit", f"{cfg_max_ctx:,} tok")
with col_b2:
    st.metric("Fixed Overhead", f"{total_fixed:,} tok")
with col_b3:
    st.metric("Available Dynamic Pool", f"{max(0, available_pool):,} tok")
with col_b4:
    if is_overflow:
        st.metric("Budget Status", "OVERFLOW!", f"-{overflow_amount:,} tok", delta_color="inverse")
    else:
        st.metric("Budget Status", "SAFE", f"+{cfg_max_ctx - total_requested:,} tok")

# Budget Distribution Chart
budget_chart_df = pd.DataFrame({
    "Component": ["System", "Tools", "User Query", "Reserve Output", "RAG Docs", "Chat History"],
    "Tokens": [cfg_sys, cfg_tools, cfg_user, cfg_reserve, cfg_rag_tokens, cfg_chat_tokens]
})
st.bar_chart(budget_chart_df.set_index("Component"))

if is_overflow:
    st.error(f"🚨 **Context Window Overflow Detected**: Prompt request exceeds hardware limit by **{overflow_amount:,} tokens**! Under strategy '{trunc_method}', {overflow_amount} tokens must be pruned from the middle of RAG documents and chat history.")
else:
    st.success(f"✅ **Context Budget Verified**: Prompt fits safely within context window with **{cfg_max_ctx - total_requested:,} tokens** remaining buffer.")

st.divider()

# ==============================================================================
# Section 12: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Tokenization engineering questions evaluate candidate depth in GPU systems, memory management, 
and defensive API design. Master these five questions and four production traps:
""")

st.subheader("Principal / Staff-Level Interview Questions")

with st.expander("Q1: Why is middle (center) truncation generally superior to head or tail truncation in long-context document QA?"):
    st.markdown("""
    **Direct Answer:**
    In real-world generative applications, prompts exhibit an asymmetrical information layout:
    * **The Head**: Contains the system prompt, operational guardrails, and role definitions. Truncating the head destroys behavioral alignment.
    * **The Tail**: Contains the active user question and formatting instructions. Truncating the tail cuts off the user's query, causing the model to generate responses to an incomplete prompt.
    * **The Middle**: Contains background reference documents or older chat history.
    
    **Center Truncation** preserves both the head (k tokens) and tail (m tokens) while pruning from the middle of the document pool. Because language models suffer from the "Lost-in-the-Middle" phenomenon (attending far more strongly to information at the very beginning and very end of prompts), pruning from the middle aligns directly with the model's natural attention bias.
    """)

with st.expander("Q2: How do streaming tokenizers handle partial multi-byte UTF-8 characters without rendering replacement character glyphs (U+FFFD)?"):
    st.markdown("""
    **Direct Answer:**
    In UTF-8 encoding, characters take between 1 and 4 bytes. In Byte-Level BPE, a token boundary can occur in the middle of a multi-byte sequence (e.g. emitting 2 bytes of a 4-byte emoji).
    
    Production stream decoders (such as Python's `codecs.getincrementaldecoder("utf-8")`) maintain an internal **Pending Byte Buffer**:
    1. When a new token ID is generated, its raw bytes are appended to the internal buffer.
    2. The decoder checks if the buffer terminates on a valid UTF-8 boundary.
    3. If the trailing bytes form an incomplete multi-byte prefix, the decoder holds them in reserve and yields only the completed characters.
    4. When the subsequent token arrives with the remaining bytes, the buffer is completed, and the full Unicode character is yielded smoothly without emitting replacement glyphs (``).
    """)

with st.expander("Q3: What is the computational and memory difference between materialized 4D attention masks and FlashAttention causal kernels?"):
    st.markdown("""
    **Direct Answer:**
    * **Materialized 4D Masks (Legacy PyTorch)**:
      Allocates an explicit float tensor in GPU HBM of shape `[Batch_Size, 1, Seq_Len, Seq_Len]`. For a batch of 8 sequences with length 32,768 in FP16, this dummy mask tensor alone consumes **over 17 GB of VRAM**! Furthermore, it requires reading and writing the entire matrix to memory.
    * **FlashAttention Causal Kernels**:
      FlashAttention eliminates the mask tensor completely. The causal constraint (col <= row) is evaluated mathematically inside the GPU SRAM tile using CUDA thread indexing (`if (col > row) return;`). Memory consumption drops from O(N^2) to strictly **0 bytes**, and compute executes at peak hardware memory bandwidth.
    """)

with st.expander("Q4: Why does tokenizing raw code or JSON consume significantly more tokens per character than natural language prose?"):
    st.markdown("""
    **Direct Answer:**
    The difference stems from **subword merge frequency statistics**:
    * **Prose**: BPE training corpora are dominated by common natural language words (`"international"`, `"understanding"`). The tokenizer learns long multi-character subwords, achieving high compression (~4.5 chars/token).
    * **Code & JSON**: Formats like JSON use repetitive syntax symbols (`{`, `}`, `"`, `:`, `\n`, `    `), snake_case or camelCase variables (`compute_attention_weights`), and numeric values. Because programming variable names are idiosyncratic, they rarely match common vocabulary words, forcing the tokenizer to fall back to 1-character or 2-character subwords.
    * Consequently, code and JSON average **1.8 to 2.8 chars/token**, consuming 50% to 100% more context window space per character than prose.
    """)

with st.expander("Q5: How does sequence bucketing reduce the padding compute tax in distributed pretraining?"):
    st.markdown("""
    **Direct Answer:**
    In standard batching, sequences in a batch are padded to the length of the longest sequence in that batch. If a 100-token prompt is paired with a 4,000-token prompt, 3,900 padding tokens are computed pointlessly.
    
    **Sequence Bucketing**:
    1. During dataset pre-processing, sequences are sorted and grouped into discrete length buckets (e.g. Bucket A: 0-512 tokens; Bucket B: 513-1024 tokens; Bucket C: 1025-2048 tokens).
    2. Batches are constructed exclusively from within the same bucket.
    3. Padding is minimized because sequences within any single batch have nearly identical lengths, reducing wasted padding FLOPs from 60% down to under 5%.
    """)

st.subheader("Production Engineering Traps")

st.error("""
🚨 **Trap 1 (Error): Tail Truncation Deleting the Actionable User Query in Long Prompts**
Using naive right-truncation (`tokens[:max_len]`) when a prompt exceeds the context window cuts off the tail. Because users place their question at the end of the prompt, the model receives all reference documents but no question, resulting in empty or completely irrelevant responses.
*Remediation*: Always use Center (Middle) Truncation to guarantee that the system prompt and the user question survive.
""")

st.error("""
🚨 **Trap 2 (Error): Using len(text.split()) for Token Budgeting Leading to 400 Context Errors**
Estimating token counts by splitting on whitespace severely underestimates tokens for code, JSON, and non-Latin languages. When the actual token count exceeds provider limits, the API request crashes with an HTTP 400 error.
*Remediation*: Always execute exact pre-flight tokenization using `tiktoken` or Hugging Face tokenizers before dispatching API calls.
""")

st.warning("""
⚠️ **Trap 3 (Warning): The UTF-8 Replacement Character Flaw in Naive Token Streaming**
Calling `.decode("utf-8")` on individual streaming token byte chunks causes multi-byte characters (emojis, Chinese, Arabic) to be rendered as `` replacement characters on client UIs.
*Remediation*: Use `codecs.getincrementaldecoder("utf-8")` to buffer incomplete byte fragments until complete characters are formed.
""")

st.warning("""
⚠️ **Trap 4 (Warning): CPU Tokenization Bottlenecks Stalling Multi-GPU Training Pipelines**
Running pure Python tokenization inside PyTorch `DataLoader` workers during distributed training leaves expensive GPU clusters idling at < 20% Model FLOPs Utilization (MFU) waiting for string parsing.
*Remediation*: Pre-tokenize training corpora offline into memory-mapped binary NumPy/Arrow arrays (`uint16`/`uint32`), streaming integers directly to GPUs via DMA.
""")

st.subheader("Concept Flashcards")

col_fc1, col_fc2 = st.columns(2)

with col_fc1:
    with st.container():
        st.info("**Left-Padding vs. Right-Padding**")
        st.caption("Causal decoders mandate Left-Padding so the final token index is always the legitimate prompt ending for next-token sampling; Right-Padding is used only for bidirectional encoders.")

    with st.container():
        st.info("**Center (Middle) Truncation**")
        st.caption("Pruning strategy that preserves the head (system instructions) and tail (user query), discarding text from the middle where attention retention is lowest.")

with col_fc2:
    with st.container():
        st.info("**Context Budget Formula**")
        st.caption("Total_Budget = Max_Context - Reserve_Output - System_Prompt - Tool_Schemas - User_Query. The remainder is allocated to RAG documents and chat history.")

    with col_fc2:
        with st.container():
            st.info("**Rust Tokenizer Throughput**")
            st.caption("Compiled Rust implementations (Hugging Face / Tiktoken) process over 1,000,000 tokens/sec per core using zero-copy memory slices, eliminating GPU starvation.")

st.divider()

# ==============================================================================
# Section 13: Self-Assessment Checklist
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of production tokenization engineering, context budgeting, and runtime systems. 
Complete all 6 checklist items:
""")

cb1 = st.checkbox("1. I understand how to budget context windows dynamically across system instructions, tools, user queries, and reserve outputs.", key="chk_34_1")
cb2 = st.checkbox("2. I know why Center Truncation is superior to Head or Tail truncation in conversational and RAG workflows.", key="chk_34_2")
cb3 = st.checkbox("3. I can explain why causal decoders strictly mandate Left-Padding and how attention masks zero out pad logits.", key="chk_34_3")
cb4 = st.checkbox("4. I understand how incremental UTF-8 stream decoders prevent replacement glyph () glitches during token streaming.", key="chk_34_4")
cb5 = st.checkbox("5. I know why FlashAttention causal kernels eliminate materialized 4D attention mask tensors from GPU VRAM.", key="chk_34_5")
cb6 = st.checkbox("6. I understand why pre-tokenizing datasets into memory-mapped binary arrays prevents CPU bottlenecks during training.", key="chk_34_6")

if cb1 and cb2 and cb3 and cb4 and cb5 and cb6:
    st.success("🎉 Outstanding! You have mastered all production tokenization engineering operations, runtime systems, and context budgeting architectures!")
