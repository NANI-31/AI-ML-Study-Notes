import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="6.1 Chunking Basics - RAG Curriculum",
    page_icon="✂️",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulations
class SlidingWindowChunker:
    """Demonstrates sliding-window chunking with configurable size and overlap."""
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        tokens = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_str = " ".join(chunk_tokens)
            is_overlap = i > 0
            chunks.append({
                "chunk_id": f"chk_{len(chunks) + 1}",
                "start_token": i,
                "end_token": i + len(chunk_tokens),
                "token_count": len(chunk_tokens),
                "text": chunk_str,
                "has_overlap": is_overlap
            })
            if i + self.chunk_size >= len(tokens):
                break
        return chunks

# Title & Overview
st.title("6.1 Chunking Basics: Foundations, Windowing & Granularity")
st.caption("Chapter 6: Text Chunking | Domain: 6. RAG")
st.markdown("""
Text chunking is the **foundational transformation stage** in any Retrieval-Augmented Generation pipeline. 
Foundation language models and embedding models do not ingest arbitrary multi-megabyte document streams 
as single monolithic blocks—they operate over finite token context windows with distinct attention budgets.

Chunking partitions continuous enterprise documents into discrete, semantically coherent passages. 
The choice of **chunk size**, **overlap boundary**, and **retrieval granularity** directly determines whether 
downstream semantic retrieval succeeds in isolating exact factual answers or drowns the LLM in irrelevant noise.

In this lesson, you will master:
1. **The Core Definition & Purpose**: Why raw documents must be segmented before embedding and vector storage.
2. **Chunk Size Dynamics**: Navigating the trade-offs between dense semantic focus and broad contextual coverage.
3. **Chunk Overlap Mechanics**: Preventing conceptual amnesia and reference severing across partition boundaries.
4. **Retrieval Granularity Hierarchy**: Comparing sentence, paragraph, and hierarchical parent-document chunk structures.
5. **Production Failure Traps**: Avoiding anaphora severance, boundary drift, and vector index duplication.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: Chunking & Ingestion Processing Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph ChunkingPipelineBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_raw {
            label="Tier 1: Document Ingress & Normalization";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            raw_doc [label="Raw Ingestion Document\\nMulti-page PDF, Confluence page, API report", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            cleaner [label="Text Sanitizer & Normalizer\\nUnicode cleanup, whitespace collapse, markup removal", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            raw_doc -> cleaner;
        }

        subgraph cluster_chunking {
            label="Tier 2: Boundary Segmentation & Sliding Window Engine";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            window_engine [label="Sliding Window Chunker\\nToken / character budget partitioner", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            overlap_buffer [label="Overlap Buffer (10-20%)\\nInjects trailing tokens of Chunk N into Chunk N+1", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            metadata_injector [label="Metadata Breadcrumb Injector\\nAttaches document_id, section_header, chunk_index", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            cleaner -> window_engine;
            window_engine -> overlap_buffer -> metadata_injector;
        }

        subgraph cluster_vector {
            label="Tier 3: Embedding Projection & Index Persistence";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            dense_embedder [label="Dense Embedding Model\\nOpenAI text-embedding-3 / BGE-Large", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            vector_store [label="Vector Database Storage\\nHNSW Index + Document Payload Store", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            metadata_injector -> dense_embedder -> vector_store;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: What is Text Chunking?
st.header("2. What is Text Chunking?", divider="rainbow")
st.markdown("""
**Text chunking** is the algorithmic process of dividing a large continuous piece of text into smaller, 
bounded segments (termed *chunks*) prior to embedding generation and vector database insertion.

* **The Unit of Retrieval**: In a RAG pipeline, the chunk is the fundamental unit of retrieval. When a user asks a question, the vector database retrieves *chunks*, not entire books or entire file systems.
* **Semantic Independence**: An ideal chunk represents a self-contained conceptual unit—it contains sufficient surrounding facts, nouns, and context to be fully understood by an LLM without requiring the rest of the original document.
* **Token Constraints**: Embedding models enforce hard token limits (e.g., 512 tokens for `all-MiniLM-L6-v2`, 8192 tokens for `text-embedding-3`). A document exceeding these limits cannot be processed without chunking.
""")
st.divider()

# Section 3: Why Chunking is Important
st.header("3. Why Chunking is Important: The Retrieval Trade-Off", divider="rainbow")
st.markdown("""
Why not simply embed entire documents at once or, conversely, embed each word individually?

Chunking directly resolves the tension between **Retrieval Precision** and **Context Completeness**:
""")

st.markdown("""
| Dimension | Too Large (e.g., 4000+ tokens) | Balanced (e.g., 300–600 tokens) | Too Small (e.g., 50 tokens) |
| :--- | :--- | :--- | :--- |
| **Semantic Specificity** | Diluted; embedding vector averages out many disparate topics. | High; vector represents a coherent, focused subject. | Extreme; single sentences lack broader topical grounding. |
| **Noise in LLM Context** | Massive; LLM prompt is flooded with irrelevant filler text. | Minimal; retrieved passages directly answer the query. | Very low, but context is insufficient to answer complex questions. |
| **Context Sufficiency** | Excellent; full surrounding explanations are preserved. | Optimal; contains facts plus necessary qualifying conditions. | Poor; misses antecedent nouns and prerequisite rules. |
| **Embedding Cost & Speed**| Slower inference, higher memory footprint per vector. | Fast batch embedding, balanced memory footprint. | Excessive vector count; index bloat in vector database. |
""")
st.divider()

# Section 4: Chunk Size
st.header("4. Chunk Size: Balancing Resolution and Breadth", divider="rainbow")
st.markdown("""
**Chunk Size** denotes the maximum number of characters, words, or tokens allocated to each individual text partition:

* **Small Chunks (100–250 tokens)**:
  * Best for fine-grained factual lookup (e.g., customer support FAQs, glossaries, telephone numbers).
  * High embedding specificity: the vector accurately reflects a single atomic statement.
  * Disadvantage: May sever causal links (e.g., separating a rule from its statutory exceptions).
* **Medium Chunks (300–600 tokens)**:
  * The industry standard for enterprise documentation, technical manuals, and corporate wikis.
  * Strikes the optimal balance between topical specificity and narrative context.
* **Large Chunks (800–1500 tokens)**:
  * Best for long-form narrative synthesis, legal contracts, and historical analysis.
  * Disadvantage: Can dilute specific keyword relevance during cosine similarity matching.
""")
st.divider()

# Section 5: Chunk Overlap
st.header("5. Chunk Overlap: Preserving Boundary Continuity", divider="rainbow")
st.markdown("""
**Chunk Overlap** specifies the number of trailing tokens from Chunk `N` that are duplicated at the start of Chunk `N+1`.

Why is overlap essential?
Consider this sentence split exactly at a boundary:
> *Chunk 1*: "The system administrator must disable remote root login because..."
> *Chunk 2*: "...it exposes SSH port 22 to automated dictionary brute-force attacks."

Without overlap, a query searching for *"Why disable remote root login?"* might retrieve Chunk 1, which lacks the causal explanation, 
or Chunk 2, which lacks the subject noun. Overlapping by 15–20% guarantees that boundary transitions maintain complete semantic continuity.
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
State-of-the-art production RAG architectures decouple the **Unit of Retrieval** from the **Unit of Generation** using 
**Parent-Document / Small-to-Big Retrieval**:
1. Documents are split into tiny child chunks (100 tokens) for high-precision semantic vector search.
2. Each child chunk stores a reference to a larger parent chunk (1000 tokens) or parent document.
3. When the child chunk matches the query, the retrieval engine fetches and passes the *parent chunk* to the LLM. 
This delivers maximum retrieval accuracy without sacrificing context completeness.
""")
st.divider()

# Section 6: Context Preservation
st.header("6. Context Preservation: Mitigating Anaphora and Reference Severing", divider="rainbow")
st.markdown("""
A primary failure mode in naive chunking is **anaphoric reference severance**:

* **What is Anaphora?**: When a sentence uses a pronoun (*"it"*, *"he"*, *"she"*, *"these parameters"*) whose noun antecedent was defined earlier in the document.
* **The Severance Problem**: If Chunk 1 defines *"PostgreSQL connection pooler PgBouncer"*, and Chunk 2 begins with *"It maintains active socket connections..."*, the vector embedding for Chunk 2 has zero semantic affinity for the word *"PgBouncer"*. A query for *"PgBouncer sockets"* will completely miss Chunk 2!
* **Context Preservation Tactics**:
  * **Document Title & Breadcrumb Prepending**: Prepending the document title and section header to every single chunk (e.g., `Document: PgBouncer Guide | Section: Connection Pooling | Text: ...`).
  * **Semantic Boundary Snapping**: Forcing chunks to snap to paragraph or section header boundaries rather than rigid character counts.
""")
st.divider()

# Section 7: Retrieval Granularity
st.header("7. Retrieval Granularity: The Structural Spectrum", divider="rainbow")
st.markdown("""
Choosing the appropriate **Retrieval Granularity** depends heavily on corpus structure and query modality:
""")

st.markdown("""
| Granularity Level | Typical Chunk Unit | Optimal Search Task | Trade-offs |
| :--- | :--- | :--- | :--- |
| **Sentence-Level** | 1–2 sentences (~40–80 tokens) | Precise fact-checking, entity extraction | Low context; LLM cannot synthesize broader rationale |
| **Paragraph-Level** | 1 cohesive paragraph (~150–350 tokens) | Standard customer support, API documentation | High coherence; natural boundary preserves author intent |
| **Section-Level** | Full Markdown header section (~500–1200 tokens) | Legal contract clauses, technical runbooks | High context; risk of multi-topic semantic dilution |
| **Hierarchical (Parent-Child)** | Child (100 tokens) -> Parent (800 tokens) | Enterprise knowledge bases, policy manuals | Optimal retrieval + generation; requires two-tier storage |
""")
st.divider()

# Section 8: Production Implementation
st.header("8. Production Implementation: Sliding Window Chunker with Overlap Tracking", divider="rainbow")
st.markdown("""
Below is a clean, production-grade Python implementation of a token-aware sliding window chunker 
that tracks boundary offsets, word counts, and generates traceable chunk identifiers.
""")

st.code('''
from typing import List, Dict, Any

class ProductionSlidingWindowChunker:
    """Production-grade sliding window chunker with boundary tracking and metadata injection."""
    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 30):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document_id: str, text: str, doc_title: str = "") -> List[Dict[str, Any]]:
        words = text.split()
        total_words = len(words)
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        
        chunk_idx = 0
        for start_idx in range(0, total_words, step):
            end_idx = min(start_idx + self.chunk_size, total_words)
            chunk_words = words[start_idx:end_idx]
            
            # Construct chunk with breadcrumb header
            chunk_body = " ".join(chunk_words)
            formatted_text = f"Title: {doc_title} | Chunk: {chunk_idx + 1}\\n{chunk_body}" if doc_title else chunk_body
            
            chunks.append({
                "chunk_id": f"{document_id}_chk_{chunk_idx + 1:04d}",
                "document_id": document_id,
                "chunk_index": chunk_idx,
                "start_word": start_idx,
                "end_word": end_idx,
                "word_count": len(chunk_words),
                "is_truncated": end_idx < total_words,
                "content": formatted_text
            })
            chunk_idx += 1
            if end_idx >= total_words:
                break

        return chunks
''', language="python")
st.divider()

# Section 9: Interactive Workbench
st.header("9. Interactive Workbench: Live Chunk Size & Overlap Boundary Visualizer", divider="rainbow")
st.markdown("""
Manipulate chunk size and overlap sliders to observe in real time how a sample enterprise policy 
document is segmented into discrete retrieval units.
""")

sample_policy_text = """
The Enterprise Data Retention Policy dictates that all customer access logs must be retained for a minimum of 365 calendar days.
Logs older than 365 days must be archived to immutable cold storage (Amazon S3 Glacier Vault) with compliance object locking enabled.
Under no circumstances may an engineer delete production audit logs without written authorization from the Chief Information Security Officer.
Failure to comply with these retention mandates constitutes a critical compliance violation under SOC-2 Type II standards.
All cryptographic key rotation events must be logged to a separate, tamper-evident append-only ledger monitored by automated security alerting pipelines.
"""

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    user_chunk_size = st.slider("Chunk Size (Words):", min_value=15, max_value=80, value=30, step=5)
with col_ctrl2:
    user_overlap = st.slider("Chunk Overlap (Words):", min_value=0, max_value=25, value=8, step=1)

if user_overlap >= user_chunk_size:
    st.error("Overlap must be strictly smaller than Chunk Size!")
else:
    chunker = SlidingWindowChunker(chunk_size=user_chunk_size, chunk_overlap=user_overlap)
    generated_chunks = chunker.chunk_text(sample_policy_text)
    
    st.subheader(f"Generated Chunks ({len(generated_chunks)} Segments)")
    df_chunks = pd.DataFrame(generated_chunks)
    st.dataframe(df_chunks[["chunk_id", "token_count", "start_token", "end_token", "text"]], use_container_width=True)
    
    for c in generated_chunks:
        with st.expander(f"Inspect {c['chunk_id']} ({c['token_count']} words)", expanded=False):
            st.markdown(f"**Content**:\n\n`{c['text']}`")
st.divider()

# Section 10: Technical Interview Preparation & Production Traps
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: What is the primary trade-off between small chunk sizes and large chunk sizes in RAG?", expanded=True):
    st.markdown("""
    **Answer:**
    * **Small Chunks (e.g., 100–200 tokens)**: High semantic resolution. Embedding vectors closely represent individual atomic facts, minimizing noise during vector retrieval. However, small chunks lack surrounding context, causing the LLM to miss qualifying conditions or fail on multi-sentence reasoning.
    * **Large Chunks (e.g., 800–1200 tokens)**: High contextual completeness, preserving complex explanations and surrounding qualifications. However, embedding vectors represent the average of multiple topics, diluting search precision and consuming substantial LLM prompt token bandwidth with irrelevant filler.
    """)

with st.expander("Q2: Why is chunk overlap necessary in sliding window chunking pipelines?"):
    st.markdown("""
    **Answer:**
    Without overlap, critical information that spans a rigid chunk boundary (such as a condition split from its conclusion, or a noun separated from its pronoun) becomes disjointed. Queries seeking the relationship between those concepts will fail to retrieve either chunk with high similarity. Overlapping by 10–20% guarantees that every sentence transition is represented continuously in at least one chunk.
    """)

with st.expander("Q3: How does Parent-Document (Small-to-Big) retrieval solve the chunk size dilemma?"):
    st.markdown("""
    **Answer:**
    Parent-Document retrieval decouples the **retrieval index** from the **generation context**:
    1. Documents are partitioned into small child chunks (e.g., 100 tokens) that are embedded into the vector database for fine-grained retrieval.
    2. Each child chunk stores a reference metadata ID pointing to a larger parent chunk (e.g., 800 tokens) or parent document.
    3. When a child chunk achieves high cosine similarity during search, the retrieval engine swaps it for the larger parent chunk before feeding the context into the LLM prompt.
    """)

with st.expander("Q4: What is anaphoric reference severance, and how do you prevent it during document chunking?"):
    st.markdown("""
    **Answer:**
    Anaphoric reference severance occurs when a pronoun (e.g., *"it"*, *"they"*, *"this policy"*) is isolated in a chunk while its antecedent entity is located in the preceding chunk. Because the isolated chunk lacks the entity keyword, its embedding has zero proximity to queries referencing that entity. 
    Prevention techniques include:
    * Prepending document titles and section breadcrumbs to every chunk.
    * Employing coreference resolution models (e.g., FastCoref) during pre-processing.
    * Snapping boundaries to natural paragraph or Markdown header transitions.
    """)

with st.expander("Q5: How do you measure the quality of a chunking strategy empirically?"):
    st.markdown("""
    **Answer:**
    Quality is evaluated across retrieval and generation dimensions:
    * **Retrieval Metrics**: Hit Rate@K and Mean Reciprocal Rank (MRR) on a golden question-answer test dataset across different chunk configurations.
    * **Generation Metrics**: Faithfulness (verifying the LLM answer is derived strictly from the chunk) and Answer Relevance (using RAGAS or TruLens evaluators).
    * **Chunk Coherence**: Measuring semantic similarity between adjacent sentences within chunks vs across chunk boundaries.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Arbitrary Character Splitting Severing Token Encodings (Garbage Tokens)**
    Splitting raw text purely by string character index (`text[:500]`) rather than token boundaries. Characters inside multi-byte UTF-8 sequences (such as emojis or non-English characters) or multi-token words get bisected, producing invalid replacement tokens and corrupted embeddings.
    """)
    st.error("""
    **Trap 2: Excessive Overlap Driving Vector Index Bloat & Duplication (Wasted Cost)**
    Setting chunk overlap above 50% (e.g., 500-token chunk with 350-token overlap). The number of chunks increases by 300%, blowing out vector database memory storage costs and flooding the LLM context with redundant, near-identical passages.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Stripping Document Hierarchy and Header Breadcrumbs (Context Amnesia)**
    Discarding Markdown headers (`# Header`) and table captions during text extraction. Chunks containing bulleted instructions lose their governing context, causing the LLM to generate answers that confuse policy guidelines across different departments.
    """)
    st.warning("""
    **Trap 4: Inconsistent Tokenizer Alignment with Embedding Models (Silent Truncation)**
    Chunking text using word counts or GPT-4 tokenizer counts while embedding with a model (like BERT or BGE) that uses a WordPiece tokenizer with a smaller max sequence length (512 tokens). Chunks exceed the model's sequence length and are silently truncated at ingestion time.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary purpose of text chunking in RAG?")
        st.markdown("**Back**: To divide documents into cohesive semantic units that fit within embedding and LLM token budgets while optimizing retrieval precision.")
    with st.container(border=True):
        st.markdown("**Front**: What is the standard recommended chunk overlap ratio?")
        st.markdown("**Back**: 10% to 20% of the chunk size, providing continuity without bloating the vector database.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: How does Small-to-Big (Parent-Document) chunking work?")
        st.markdown("**Back**: Embeds small child chunks for precise vector lookup, but returns the larger parent chunk to the LLM for rich context during generation.")
    with st.container(border=True):
        st.markdown("**Front**: Why prepend document breadcrumbs to individual chunks?")
        st.markdown("**Back**: To prevent context amnesia, ensuring chunks retain vital document-level and section-level metadata.")

st.divider()

# Section 11: Self-Assessment Checklist
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Text Chunking Foundations before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand what text chunking is and why it represents the primary unit of retrieval.", value=True, key="chk_chk_1")
    chk2 = st.checkbox("I can explain the trade-offs between small chunks (high precision) and large chunks (high context).", value=True, key="chk_chk_2")
    chk3 = st.checkbox("I know how chunk overlap prevents reference severing and conceptual boundary loss.", value=True, key="chk_chk_3")
with c2:
    chk4 = st.checkbox("I understand the mechanics of Small-to-Big and Parent-Document retrieval.", value=True, key="chk_chk_4")
    chk5 = st.checkbox("I know how to inject breadcrumb headers to mitigate anaphoric pronoun severance.", value=True, key="chk_chk_5")
    chk6 = st.checkbox("I can identify production traps such as UTF-8 character splitting and tokenizer mismatches.", value=True, key="chk_chk_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Chunking Basics Mastery Demonstrated!** You are ready to advance to Lesson 6.2 in the curriculum.")
