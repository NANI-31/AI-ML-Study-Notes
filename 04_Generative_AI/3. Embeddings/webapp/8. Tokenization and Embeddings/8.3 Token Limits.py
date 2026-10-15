import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="8.3 Token Limits - Embeddings Curriculum",
    page_icon="📏",
    layout="wide"
)

# Title & Overview
st.title("8.3 Token Limits: Context Windows, Silent Truncation Hazards, & Long-Doc Chunking")
st.caption("Chapter 8: Tokenization and Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Every neural embedding model operates within a rigid mathematical boundary: its **Maximum Input Context Window** ($L_{\\text{max}}$).
Whether bounded by fixed sinusoidal positional embeddings ($L_{\\text{max}} = 512$) or rotary position embeddings (RoPE up to $8,192$ or $32,768$), passing text that exceeds this threshold triggers one of two production disasters:
1. **Silent Truncation**: The tokenizer quietly discards all tokens beyond the limit, discarding crucial conclusions, legal waivers, or financial figures at the end of the document without throwing any error!
2. **Crash / Out-Of-Bounds Exception**: If truncation is disabled, PyTorch crashes with `IndexError: index out of range` or causes catastrophic CUDA Out-Of-Memory errors.

This lesson explores context limits across model architectures, accurate client-side token counting, and production strategies for embedding oversized documents.
""")

st.divider()

# Section 1: The Spectrum of Context Limits & Truncation Mechanics
st.header("1. Context Limits Across Architectures & The Truncation Hazard", divider="rainbow")

col_lim1, col_lim2 = st.columns([1, 1])

with col_lim1:
    st.markdown("""
    ### Context Windows Across Embedding Models:
    * **Classical Short-Context (256 - 512 tokens)**:
      * `all-MiniLM-L6-v2` (256 tokens), `all-mpnet-base-v2` (384 tokens), `bge-large-en-v1.5` (512 tokens).
      * Bound by fixed learned absolute position embedding tables. Cannot extrapolate to 513 tokens.
    * **Modern Extended-Context (8,192 tokens)**:
      * OpenAI `text-embedding-3-small` / `3-large`, `nomic-embed-text-v1.5`, `ModernBERT-base`.
      * Uses Rotary Position Embedding (RoPE) or ALiBi attention, processing multi-page documents in a single forward pass.
    * **Frontier Long-Context (32,768 tokens)**:
      * `gte-Qwen2-1.5B-instruct`, `bge-m3`. Can ingest complete corporate whitepapers or books.

    ### The Silent Truncation Hazard:
    When you call `model.encode(long_text)` with default settings:
    * The tokenizer encodes the first 512 tokens and **discards the remaining 90% of the document**.
    * If a user asks a question answered on page 3 of a 10-page PDF, the retriever will **never find it** because those tokens were never converted into vectors!
    """)

with col_lim2:
    st.markdown("### The Silent Truncation Cliff")
    cliff_dot = """
    digraph TruncationCliff {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        InputDoc [label="Ingested Contract (1,500 Tokens)\\nIntro + Terms + Liability + Signatures", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        TokenizerCut [label="Embedding Tokenizer\\nHard Limit: 512 Tokens", fillcolor="#FEF3C7"];
        
        Preserved [label="Tokens [0 to 512]\\nIntro & Preamble\\nEncoded into Vector", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];
        
        Discarded [label="Tokens [513 to 1500]\\nLiability & Signatures\\nSILENTLY DISCARDED!", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];

        InputDoc -> TokenizerCut;
        TokenizerCut -> Preserved [label="Retained in index"];
        TokenizerCut -> Discarded [label="Silent deletion\\nZero error thrown!"];
    }
    """
    st.graphviz_chart(cliff_dot, use_container_width=False)

st.markdown("### Model Context Window & Positional Encoding Matrix")
st.markdown("""
| Model Identifier | Max Context Window | Positional Encoding Type | Truncation Risk on 2-Page Document | Recommended Chunk Size |
| :--- | :--- | :--- | :--- | :--- |
| **all-MiniLM-L6-v2** | **256 tokens** | Absolute Positional Embeddings | **Extreme (Cuts off ~75%)** | 150 - 200 tokens |
| **bge-base-en-v1.5** | **512 tokens** | Absolute Positional Embeddings | **High (Cuts off ~50%)** | 250 - 350 tokens |
| **text-embedding-3-small** | **8,192 tokens** | Modern Rotary / Extended | Low | 500 - 1,000 tokens (for RAG precision) |
| **nomic-embed-text-v1.5** | **8,192 tokens** | Rotary Position Embedding (RoPE) | Low | 500 - 1,000 tokens |
| **gte-Qwen2-1.5B** | **32,768 tokens** | RoPE with YaRN extrapolation | Zero | Multi-page section chunks |
""")

st.divider()

# Section 2: Production Code: Token-Aware Long-Document Chunking Engine
st.header("2. Production Implementation: Token-Aware Document Chunking Engine", divider="rainbow")

st.markdown("""
To eliminate silent truncation, production pipelines enforce **token-aware chunking**: measuring exact token counts before embedding and partitioning oversized texts into overlapping passages.
""")

st.code(r'''
from typing import List, Dict, Any, Tuple

class TokenAwareDocumentChunker:
    """
    Splits long documents based on strict token ceilings rather than arbitrary word counts.
    Eliminates silent truncation while maintaining contextual overlap.
    """
    def __init__(self, max_tokens_per_chunk: int = 400, overlap_tokens: int = 50):
        self.max_tokens = max_tokens_per_chunk
        self.overlap = overlap_tokens

    def chunk_document(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Token-aware splitting with sliding window overlap.
        In production, use actual model tokenizer (e.g. tiktoken or HF AutoTokenizer).
        """
        words = text.split()
        # Word-to-token heuristic: 1 word ~ 1.3 tokens
        chunks = []
        words_per_chunk = int(self.max_tokens / 1.3)
        overlap_words = int(self.overlap / 1.3)
        step = max(1, words_per_chunk - overlap_words)

        for i in range(0, len(words), step):
            slice_words = words[i : i + words_per_chunk]
            chunk_text = " ".join(slice_words)
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "doc_id": doc_id,
                "text": chunk_text,
                "estimated_tokens": int(len(slice_words) * 1.3),
                "is_safe": int(len(slice_words) * 1.3) <= self.max_tokens
            })
            if i + words_per_chunk >= len(words):
                break

        return chunks
''', language="python")

# Actual Python implementation for the interactive workbench
class LongDocSimulator:
    def __init__(self):
        self.models = {
            "all-MiniLM-L6-v2 (256 tokens)": 256,
            "bge-base-en-v1.5 (512 tokens)": 512,
            "text-embedding-3-small (8,192 tokens)": 8192
        }

    def analyze_truncation(self, text: str, model_limit: int) -> Dict[str, Any]:
        words = text.strip().split()
        # Simulated token count: words * 1.3 + special tokens
        tokens_est = int(len(words) * 1.3) + 2
        
        is_truncated = tokens_est > model_limit
        retained_tokens = min(tokens_est, model_limit)
        discarded_tokens = max(0, tokens_est - model_limit)
        
        retained_pct = (retained_tokens / max(tokens_est, 1)) * 100.0
        discarded_pct = (discarded_tokens / max(tokens_est, 1)) * 100.0
        
        # Word-level split
        retained_word_count = int(model_limit / 1.3)
        retained_text = " ".join(words[:retained_word_count])
        discarded_text = " ".join(words[retained_word_count:])

        return {
            "total_tokens": tokens_est,
            "model_limit": model_limit,
            "is_truncated": is_truncated,
            "retained_tokens": retained_tokens,
            "discarded_tokens": discarded_tokens,
            "retained_pct": retained_pct,
            "discarded_pct": discarded_pct,
            "retained_text": retained_text,
            "discarded_text": discarded_text
        }

st.divider()

# Section 3: Interactive Workbench: Truncation Danger & Chunking Visualizer
st.header("3. Interactive Workbench: Truncation Danger & Chunking Visualizer", divider="rainbow")

st.markdown("""
Paste or generate a long multi-paragraph document. Select an embedding model and observe whether the document exceeds the context window.
If truncation occurs, the system highlights the **silently discarded tokens** in red, demonstrating what information would be lost in an unchunked pipeline.
""")

sample_long_doc = """Section 1: Executive Overview. The enterprise cloud infrastructure delivers fault-tolerant distributed databases across multi-region clusters. Automated replication safeguards operational continuity during localized data center outages.

Section 2: Architecture Specifications. Primary nodes handle transactional read-write operations with ACID guarantees, while asynchronous read replicas scale analytics throughput. Connection pooling throttles incoming client concurrency to prevent socket exhaustion.

Section 3: Security & Compliance. Mutual TLS encrypts data in transit, while envelope encryption with customer-managed keys protects data at rest. Role-based access control enforces least-privilege policies across all administrative interfaces.

Section 4: Critical Disaster Recovery & SLA Terms. In the event of catastrophic data center destruction, failover initiates within 180 seconds. The service level agreement guarantees 99.99% uptime with financial credits issued for unscheduled outages exceeding four consecutive minutes."""

sim_long = LongDocSimulator()

col_trunc_l, col_trunc_r = st.columns([1, 1])

with col_trunc_l:
    st.subheader("1. Ingested Document & Model Selection")
    user_doc = st.text_area("Input Document Text:", value=sample_long_doc, height=180)
    
    model_choice = st.selectbox(
        "Target Embedding Model Context Limit:",
        list(sim_long.models.keys())
    )
    
    limit_val = sim_long.models[model_choice]
    res_trunc = sim_long.analyze_truncation(user_doc, limit_val)

with col_trunc_r:
    st.subheader("2. Context Limit & Truncation Diagnostics")
    m1, m2 = st.columns(2)
    m1.metric("Estimated Tokens", f"{res_trunc['total_tokens']} tokens")
    m2.metric("Model Context Limit", f"{res_trunc['model_limit']} tokens")

    m3, m4 = st.columns(2)
    m3.metric("Retained Tokens", f"{res_trunc['retained_tokens']} ({res_trunc['retained_pct']:.1f}%)")
    m4.metric("Discarded Tokens", f"{res_trunc['discarded_tokens']} ({res_trunc['discarded_pct']:.1f}%)", delta=f"-{res_trunc['discarded_tokens']}" if res_trunc['is_truncated'] else "Zero", delta_color="inverse")

    if res_trunc['is_truncated']:
        st.error(f"🚨 **Silent Truncation Active**: {res_trunc['discarded_pct']:.1f}% of the document exceeds the context limit and will be discarded!")
        with st.expander("Inspect Retained vs Discarded Text"):
            st.markdown("#### ✅ Retained Text (Embedded):")
            st.write(res_trunc['retained_text'])
            st.markdown("#### ❌ Discarded Text (Silently Lost):")
            st.error(res_trunc['discarded_text'])
    else:
        st.success(f"✅ **Safe Ingestion**: Full text fits comfortably within the {res_trunc['model_limit']}-token context limit with 100% information preservation.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Context Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is 'Silent Truncation' and why is it considered a critical production failure mode in RAG?**
    * **Answer**:
      * In PyTorch and Hugging Face `transformers`, the default tokenizer configuration often specifies `truncation=True` or clips inputs to `max_position_embeddings`.
      * When a user passes an 800-token document into a 512-token model, the tokenizer truncates tokens 513 through 800 **without raising an exception, warning, or log message**.
      * The downstream vector index silently indexes only the first 512 tokens.
      * If a user subsequently queries for a fact present in the second half of the document, the retrieval engine will never match it, causing hallucinations in the LLM generator.

    **Q2: If modern models like OpenAI text-embedding-3 support 8,192 tokens, why should we still chunk text into 200-500 token passages for RAG?**
    * **Answer**:
      * 1. **Needle-in-a-Haystack Semantic Dilution**: Squeezing an 8,000-token document into a single 1536-dimensional vector smears dozens of different subtopics into a general centroid. A specific question about a single paragraph on page 7 will have lower cosine similarity with the full-document vector than with a targeted 300-token chunk vector.
      * 2. **LLM Context Window Economy**: When retrieving top-k chunks for generation, injecting three 300-token chunks consumes 900 prompt tokens. Injecting three 8,000-token documents consumes 24,000 tokens, increasing latency and API costs by 25x!

    **Q3: How do you handle token counting reliably when using closed-source APIs like OpenAI?**
    * **Answer**:
      * Never use word count approximations (`len(text.split())`) in production.
      * Use **`tiktoken`**, OpenAI's open-source fast BPE tokenizer.
      * For `text-embedding-3-small` and `text-embedding-3-large`, instantiate the `cl100k_base` encoding:
        `encoding = tiktoken.get_encoding("cl100k_base")`
        `num_tokens = len(encoding.encode(text))`
      * This provides bit-exact token counting to verify chunk bounds and track API token billing.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Forgetting Special Tokens in Context Budget Calculations**:
      * *Issue*: Sizing chunks to exactly 512 tokens.
      * *Consequence*: The tokenizer adds `[CLS]` and `[SEP]`, expanding the sequence to 514 tokens. The model clips the last two tokens of the text!
      * *Fix*: Always leave a safety margin of at least 10-20 tokens (e.g. max chunk size = 490 tokens).

    * **Trap 2: Using tiktoken to Measure Hugging Face Model Tokens**:
      * *Issue*: Counting tokens with `tiktoken` for a model running `bge-small-en-v1.5` (WordPiece).
      * *Consequence*: BPE and WordPiece have completely different vocabularies. A code snippet might be 300 tokens in `cl100k_base` but 650 tokens in BERT WordPiece, triggering unexpected truncation.
      * *Fix*: Always use the exact tokenizer paired with the target embedding model.

    * **Trap 3: Disabling Truncation Without Exception Handling**:
      * *Issue*: Setting `truncation=False` to prevent silent data loss.
      * *Consequence*: When an oversized document arrives, PyTorch crashes with `IndexError: index out of range`, halting the entire ingestion pipeline.
      * *Fix*: Set `truncation=False` inside a `try / except` block that routes oversized documents to an automated chunker.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Maximum Context Window**
      * *Definition*: The hard upper bound on input token sequence length that an embedding model's positional embeddings can accept.
    * **Flashcard 2: Silent Truncation**
      * *Definition*: When an embedding tokenizer discards tokens beyond its maximum sequence length without throwing an exception.
    * **Flashcard 3: tiktoken**
      * *Definition*: OpenAI's high-speed BPE tokenizer library used to count exact token consumption for OpenAI embedding and LLM APIs.
    * **Flashcard 4: Safety Margin**
      * *Definition*: Sizing text chunks slightly below the context ceiling (e.g. 480 vs 512) to accommodate mandatory special tokens like [CLS] and [SEP].
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Token Limits:

- [ ] I can state the maximum context windows of common embedding models (MiniLM, BGE, OpenAI, GTE).
- [ ] I understand the danger of Silent Truncation and how it corrupts RAG recall.
- [ ] I know why we still chunk text even when using 8k-token embedding models.
- [ ] I know how to use `tiktoken` and Hugging Face tokenizers for accurate client-side token budgeting.
- [ ] I understand why a safety margin for special tokens ([CLS], [SEP]) is mandatory during chunking.
""")
