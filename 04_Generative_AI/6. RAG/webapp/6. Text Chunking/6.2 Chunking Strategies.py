import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="6.2 Chunking Strategies - RAG Curriculum",
    page_icon="✂️",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulations
class MultiStrategyChunker:
    """Demonstrates and compares multiple chunking strategies across the identical text passage."""
    def __init__(self):
        pass

    def fixed_chunk(self, text: str, size: int = 120) -> List[str]:
        return [text[i:i + size] for i in range(0, len(text), size)]

    def sentence_chunk(self, text: str) -> List[str]:
        raw_sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return [f"{s}." for s in raw_sentences]

    def paragraph_chunk(self, text: str) -> List[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def recursive_split(self, text: str, max_chars: int = 150) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        for p in paragraphs:
            if len(p) <= max_chars:
                chunks.append(p.strip())
            else:
                sentences = p.split(". ")
                current = ""
                for s in sentences:
                    if len(current) + len(s) + 2 <= max_chars:
                        current = f"{current} {s}.".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = f"{s}."
                if current:
                    chunks.append(current)
        return chunks

# Title & Overview
st.title("6.2 Chunking Strategies: Algorithms, Boundaries & Semantic Splitting")
st.caption("Chapter 6: Text Chunking | Domain: 6. RAG")
st.markdown("""
No single chunking strategy works universally across all document types. Code files demand abstract syntax tree 
awareness, legal contracts require section hierarchy preservation, conversational chats need turn-based grouping, 
and dense technical manuals benefit from semantic distance thresholds.

Choosing the appropriate **chunking strategy** is an engineering decision balancing computational efficiency, 
grammatical integrity, and downstream vector clustering precision.

In this lesson, you will master:
1. **The Core Taxonomy**: From rigid fixed-character splitting to advanced recursive and semantic boundary detection.
2. **Recursive Character Splitting**: The LangChain/LlamaIndex gold standard that respects natural text hierarchies.
3. **Token-Based Alignment**: Aligning partitions directly with Byte-Pair Encoding (BPE) subword tokenizers.
4. **Semantic Chunking**: Detecting embedding divergence points to split text where conceptual topics actually shift.
5. **Production Failure Traps**: Avoiding mid-word bisection, sentence fragmentation, and semantic threshold instability.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: Chunking Strategy Taxonomy & Execution Flow", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph ChunkingTaxonomyBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_input {
            label="Tier 1: Document Corpus Ingestion";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            doc_stream [label="Continuous Text Document Stream\\nMarkdown, Articles, Transcripts, Documentation", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_strategies {
            label="Tier 2: Algorithmic Chunking Strategy Routing";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            fixed_strat [label="1. Fixed-Size & Character Chunker\\nUniform char/token windows; naive slicing", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            recursive_strat [label="2. Recursive Character Splitter\\nHierarchical separator fallback: [\\n\\n, \\n, ' ', '']", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            sentence_strat [label="3. Linguistic Chunker (Sentence/Paragraph)\\nspaCy / NLTK sentence boundary detection", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            semantic_strat [label="4. Semantic Distance Chunker\\nCosine embedding divergence thresholding", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            doc_stream -> fixed_strat [style=dashed];
            doc_stream -> recursive_strat [label="Standard RAG"];
            doc_stream -> sentence_strat [style=dashed];
            doc_stream -> semantic_strat [label="Advanced RAG"];
        }

        subgraph cluster_output {
            label="Tier 3: Structured Chunks & Downstream Vector Indexing";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            chunk_batches [label="Cohesive Passage Chunks\\nPreserved grammatical boundaries + metadata tags", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            embedding_engine [label="Dense Vector Embedding Engine\\nGenerates 1536-d semantic vectors per passage", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];

            recursive_strat -> chunk_batches;
            semantic_strat -> chunk_batches;
            chunk_batches -> embedding_engine;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Fixed-Size Chunking
st.header("2. Fixed-Size Chunking: Naive Constant-Window Slicing", divider="rainbow")
st.markdown("""
**Fixed-Size Chunking** partitions a text string at rigid, constant intervals (e.g., every 500 characters or every 100 words) 
regardless of linguistic grammar or document structure.

* **Implementation**: Uses simple string slicing (`text[i:i + size]`) with an optional step offset.
* **Advantages**: Minimal CPU overhead; instantaneous processing speeds across gigabytes of text.
* **Severe Drawbacks**: Completely agnostic to words and sentences. It will cut words in half (*"super|conductor"*), sever sentence punctuation, and produce disjointed phrases that damage downstream embedding representations.
""")
st.divider()

# Section 3: Character-Based Chunking
st.header("3. Character-Based Chunking", divider="rainbow")
st.markdown("""
**Character-Based Chunking** measures boundaries in character length (`len(text)`). 
While commonly confused with token-based chunking, characters and tokens diverge significantly:

* **Character-to-Token Ratio**: In English, 1 token roughly corresponds to 4 characters (0.75 words).
* **Code & Non-English Divergence**: In source code (indents, brackets) or non-Latin scripts (Chinese, Arabic, Hindi), 
a single character may consume 2 to 4 tokens.
* **Risk**: Setting a character chunk size of 1,000 characters could result in 250 tokens in English, but 800+ tokens in Python code—potentially overflowing an embedding model with a 512-token limit!
""")
st.divider()

# Section 4: Recursive Character Splitting
st.header("4. Recursive Character Splitting: The Industry Standard", divider="rainbow")
st.markdown("""
**Recursive Character Splitting** (popularized by LangChain's `RecursiveCharacterTextSplitter`) is the most widely adopted 
general-purpose chunking algorithm in modern RAG.

Instead of slicing blindly, it attempts to split text using a **hierarchical list of separators**:
1. First, attempt to split along **Paragraphs** (`\n\n`). If each paragraph fits within the target chunk size, keep them as chunks.
2. If a paragraph exceeds the target size, recursively split that paragraph along **Line Breaks** (`\n`).
3. If a line is still too large, split along **Sentence Endings** (`. `).
4. If a sentence is still too large, split along **Word Spaces** (` `).
5. Only as a last resort does it split between individual characters (`""`).

By cascading through these boundaries, it guarantees that sentences and paragraphs remain intact whenever possible, 
preserving maximum grammatical coherence.
""")
st.divider()

# Section 5: Sentence-Based Chunking
st.header("5. Sentence-Based Chunking", divider="rainbow")
st.markdown("""
**Sentence-Based Chunking** leverages natural language processing libraries (such as NLTK, spaCy, or regular expression rules) 
to detect true sentence termination boundaries:

* **Distinguishing Abbreviations from Endings**: Naive regex splitting on periods breaks on abbreviations (*"Dr. Smith"*, *"e.g."*, *"Inc."*). NLP tokenizers evaluate grammatical syntax to identify true sentence breaks.
* **Multi-Sentence Windows**: Chunks are assembled by grouping `N` consecutive complete sentences (e.g., 3–5 sentences per chunk) with a 1-sentence sliding overlap.
* **Advantage**: Zero mid-sentence severance; every chunk consists of complete, syntactically valid thoughts.
""")
st.divider()

# Section 6: Paragraph-Based Chunking
st.header("6. Paragraph-Based Chunking", divider="rainbow")
st.markdown("""
In human writing, authors naturally organize distinct ideas into **paragraphs**:
* **Natural Topical Boundaries**: A paragraph transition almost always signals a shift in focus, argument, or topic.
* **Splitting Mechanism**: Segments documents along double newline characters (`\n\n`).
* **Short Paragraph Merging**: When paragraphs are very short (e.g., single-sentence dialogue or list items), the algorithm accumulates consecutive paragraphs until the target token budget is reached.
""")
st.divider()

# Section 7: Token-Based Chunking
st.header("7. Token-Based Chunking: Exact Tokenizer Alignment", divider="rainbow")
st.markdown("""
Language models and embedding models do not see characters—they process **tokens** generated via 
Byte-Pair Encoding (BPE) or WordPiece tokenizers.

**Token-Based Chunking** executes windowing directly in tokenizer space using libraries like OpenAI's `tiktoken`:
1. Encode raw text into an array of integer token IDs: `tokens = tokenizer.encode(text)`.
2. Slice the integer array into fixed windows: `chunk_tokens = tokens[i:i + max_tokens]`.
3. Decode each slice back into text: `chunk_text = tokenizer.decode(chunk_tokens)`.

* **Advantage**: Guarantees that chunks will never exceed the embedding model's maximum sequence length (e.g., exactly 512 tokens), preventing silent truncation.
""")
st.divider()

# Section 8: Semantic Chunking
st.header("8. Semantic Chunking: Detecting Concept Divergence Points", divider="rainbow")
st.markdown("""
Rather than relying on arbitrary word counts or punctuation marks, **Semantic Chunking** uses 
neural embeddings to detect where the conceptual topic actually changes:

1. **Sentence Splitting**: The document is split into individual sentences.
2. **Embedding Generation**: Every sentence is encoded into a dense vector embedding.
3. **Distance Calculation**: The algorithm computes the Cosine Distance between consecutive sentences:
   `Distance(S_i, S_{i+1}) = 1.0 - Cosine_Similarity(v_i, v_{i+1})`.
4. **Breakpoint Identification**: When the distance between sentence `i` and sentence `i+1` spikes above a designated statistical threshold (e.g., the 90th percentile of distance shifts), a chunk boundary is placed!
""")

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production enterprise search, **Semantic Chunking** is paired with **Agentic Chunking**: 
an LLM inspects a document outline and dynamically groups passages based on conceptual intent, 
producing variable-length chunks that map 1:1 with standalone business topics.
""")
st.divider()

# Section 9: Production Implementation
st.header("9. Production Implementation: Adaptive Recursive & Sentence Chunker", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation of an adaptive chunking engine supporting recursive splitting 
with configurable separators and minimum passage lengths.
""")

st.code('''
from typing import List

class ProductionAdaptiveChunker:
    """Adaptive recursive chunker prioritizing natural structural boundaries."""
    def __init__(self, max_chunk_size: int = 400, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.separators = ["\\n\\n", "\\n", ". ", " "]

    def split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators or len(text) <= self.max_chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0]
        splits = text.split(sep)
        chunks = []
        current = ""

        for s in splits:
            part = s + (sep if sep != " " else "")
            if len(current) + len(part) <= self.max_chunk_size:
                current += part
            else:
                if current:
                    chunks.append(current.strip())
                # If a single piece exceeds max_chunk_size, recursively subdivide with next separator
                if len(part) > self.max_chunk_size:
                    chunks.extend(self._recursive_split(part, separators[1:]))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks
''', language="python")
st.divider()

# Section 10: Interactive Workbench
st.header("10. Interactive Workbench: Multi-Strategy Chunking Comparison Laboratory", divider="rainbow")
st.markdown("""
Compare how four distinct chunking strategies process the identical text document. 
Observe the differences in chunk count, boundary placement, and grammatical integrity.
""")

multi_chunker = MultiStrategyChunker()

benchmark_text = """
The Kubernetes control plane manages worker nodes and pods across the cluster. The API server acts as the primary gateway for all administrative interactions.
When a deployment is created, etcd records the desired state in an append-only distributed key-value store.

The kube-scheduler evaluates resource requests and node affinities to assign pods to suitable host machines. If a node suffers a kernel panic or network partition, the controller manager detects node unresponsiveness within forty seconds and initiates pod re-scheduling on surviving nodes.

Security policies mandate that all inter-pod traffic must be encrypted via mutual TLS using an automated service mesh sidecar proxy.
"""

col_w1, col_w2 = st.columns(2)

with col_w1:
    st.subheader("1. Fixed-Size Slicing (120 chars)")
    fixed_res = multi_chunker.fixed_chunk(benchmark_text, size=120)
    st.caption(f"Produced {len(fixed_res)} chunks (Naive mid-word splits possible)")
    for i, c in enumerate(fixed_res[:3]):
        st.text(f"Chunk {i+1}: {c[:60]}...")

    st.subheader("2. Paragraph Splitting")
    para_res = multi_chunker.paragraph_chunk(benchmark_text)
    st.caption(f"Produced {len(para_res)} chunks (Cohesive macro-sections)")
    for i, c in enumerate(para_res):
        st.text(f"Chunk {i+1}: {c[:60]}...")

with col_w2:
    st.subheader("3. Sentence Boundary Splitting")
    sent_res = multi_chunker.sentence_chunk(benchmark_text)
    st.caption(f"Produced {len(sent_res)} chunks (Atomic factual statements)")
    for i, c in enumerate(sent_res[:3]):
        st.text(f"Chunk {i+1}: {c}")

    st.subheader("4. Recursive Splitting (150 chars)")
    rec_res = multi_chunker.recursive_split(benchmark_text, max_chars=150)
    st.caption(f"Produced {len(rec_res)} chunks (Hierarchical fallback)")
    for i, c in enumerate(rec_res[:3]):
        st.text(f"Chunk {i+1}: {c[:60]}...")
st.divider()

# Section 11: Technical Interview Preparation & Production Traps
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: How does Recursive Character Splitting work, and why is it preferred over character splitting?", expanded=True):
    st.markdown("""
    **Answer:**
    Recursive Character Splitting evaluates a prioritized hierarchy of natural boundary separators (typically `["\\n\\n", "\\n", ". ", " "]`). 
    It attempts to segment text along the largest semantic boundary first (paragraphs). If any paragraph exceeds the target chunk size, 
    it recursively descends to the next separator (line breaks, then sentences, then words). This ensures chunks align with natural grammatical 
    structures whenever possible, completely eliminating mid-word cuts and preserved syntactic coherence.
    """)

with st.expander("Q2: What is Semantic Chunking, and what are its performance trade-offs?"):
    st.markdown("""
    **Answer:**
    Semantic Chunking splits documents based on conceptual divergence:
    1. Text is partitioned into sentences, and each sentence is passed through an embedding model.
    2. Cosine distance between adjacent sentence vectors is computed.
    3. Breakpoints are injected wherever the distance exceeds a statistical percentile threshold.
    * **Advantage**: Chunks map cleanly to topical transitions rather than arbitrary length budgets.
    * **Trade-off**: Requires generating vector embeddings for every sentence during ingestion—increasing pre-processing time and API costs by 10x compared to rule-based recursive chunking.
    """)

with st.expander("Q3: Why should token-based chunking use the exact tokenizer of the downstream embedding model?"):
    st.markdown("""
    **Answer:**
    Different foundation models employ distinct tokenizers with different vocabularies (e.g., tiktoken `cl100k_base` for OpenAI vs WordPiece for BERT/BGE vs SentencePiece for LLaMA). 
    A text passage measuring 450 tokens in tiktoken might translate to 580 tokens in WordPiece. If you chunk using tiktoken but embed with BGE (max limit 512 tokens), the model will silently truncate the trailing 68 tokens, resulting in permanent context loss.
    """)

with st.expander("Q4: When should you choose Sentence-based chunking over Paragraph-based chunking?"):
    st.markdown("""
    **Answer:**
    * **Sentence-based chunking** is optimal for dense fact-verification, entity question answering, and customer support FAQs where queries seek specific atomic data points.
    * **Paragraph-based chunking** is optimal for conceptual synthesis, legal documents, and narrative prose where individual sentences rely heavily on surrounding qualifying clauses and antecedent explanations.
    """)

with st.expander("Q5: What is Agentic Chunking, and when is it justified?"):
    st.markdown("""
    **Answer:**
    Agentic Chunking uses an LLM to read the entire document, identify its structural themes, and programmatically decide where chunks should begin and end, frequently writing a short summary header for each chunk. It is justified for high-value enterprise repositories (such as executive strategy whitepapers or regulatory compliance filings) where standard rule-based splitters fail on intricate formatting.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Naive Fixed-Character Splitting Bisecting Markdown Tables (Syntax Destruction)**
    Using character splitting on Markdown or HTML documents. Table row pipes (`|`) and tags (`<table>`) are sliced in half across chunk boundaries, producing malformed syntax that confuses LLM generation and breaks parser scrapers.
    """)
    st.error("""
    **Trap 2: Threshold Drift in Semantic Chunking Across Heterogeneous Corpora (Erratic Sizes)**
    Using a static Cosine Distance threshold (e.g., `threshold = 0.4`) across diverse documents. Highly dense technical text may never exceed 0.3, generating massive 20-page chunks, while conversational dialog fluctuates wildly, creating single-sentence fragments. Always use dynamic percentile-based thresholds.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Tokenizer Mismatch Silently Truncating Document Chunks (Hidden Context Loss)**
    Chunking with an LLM tokenizer budget (e.g., 2000 tokens for GPT-4o) when the embedding model's maximum context length is only 512 tokens. The embedding model silently truncates the passage at token 512, rendering the last 75% of the chunk unsearchable.
    """)
    st.warning("""
    **Trap 4: Missing Fallback in Sentence Splitters on Unpunctuated Text (Single Huge Chunk)**
    Using a sentence splitter that only splits on periods (`. `) when ingesting bulleted checklists, logs, or transcribed speech lacking standard punctuation. The entire document is treated as a single mammoth sentence, overflowing buffers. Always configure recursive character fallbacks.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the separator order in Recursive Character Splitting?")
        st.markdown("**Back**: Double newline (`\\n\\n`), single newline (`\\n`), sentence space (`. `), word space (` `), and character (`""`).")
    with st.container(border=True):
        st.markdown("**Front**: How does Semantic Chunking determine boundary breaks?")
        st.markdown("**Back**: By measuring cosine distance shifts between adjacent sentence embeddings and breaking where distance exceeds a threshold.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: Why is Token-based chunking safer than Character-based chunking?")
        st.markdown("**Back**: It directly mirrors the embedding model's internal sequence memory, preventing silent context truncation.")
    with st.container(border=True):
        st.markdown("**Front**: What is the danger of fixed-size character chunking?")
        st.markdown("**Back**: It cuts words, token encodings, and sentences in half, producing malformed embeddings and corrupted context.")

st.divider()

# Section 12: Self-Assessment Checklist
st.header("12. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Chunking Strategies before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand the hierarchical separator fallback of Recursive Character Splitting.", value=True, key="chk_strat_1")
    chk2 = st.checkbox("I can explain how Semantic Chunking uses embedding distance to find topical shifts.", value=True, key="chk_strat_2")
    chk3 = st.checkbox("I know why Token-based chunking must align with the embedding model's tokenizer.", value=True, key="chk_strat_3")
with c2:
    chk4 = st.checkbox("I know when to choose Sentence-based vs Paragraph-based chunking strategies.", value=True, key="chk_strat_4")
    chk5 = st.checkbox("I understand the risks of static distance thresholds in semantic splitting.", value=True, key="chk_strat_5")
    chk6 = st.checkbox("I can identify production traps such as Markdown table corruption and tokenizer mismatches.", value=True, key="chk_strat_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Chunking Strategies Mastery Demonstrated!** You are ready to advance to Lesson 6.3 in the curriculum.")
