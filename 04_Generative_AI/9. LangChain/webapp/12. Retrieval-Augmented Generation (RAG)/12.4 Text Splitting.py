import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class ChunkResult:
    """Represents a generated text chunk with positional metadata."""
    def __init__(self, index: int, text: str, char_len: int, token_est: int, overlap_prev: int, metadata: Dict[str, Any]):
        self.index = index
        self.text = text
        self.char_len = char_len
        self.token_est = token_est
        self.overlap_prev = overlap_prev
        self.metadata = metadata

class MockChunkingPlaygroundSimulator:
    """Simulates multi-strategy document splitting with token estimates and overlap tracking."""
    def __init__(self):
        pass

    def split_recursive(self, text: str, chunk_size: int, chunk_overlap: int) -> List[ChunkResult]:
        chunks: List[ChunkResult] = []
        step = max(1, chunk_size - chunk_overlap)
        paragraphs = text.split("\n\n")

        # Simplified recursive fallback simulation
        accumulated = ""
        chunk_idx = 1
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(accumulated) + len(p) + 2 <= chunk_size:
                accumulated = f"{accumulated}\n\n{p}".strip()
            else:
                if accumulated:
                    ov = min(chunk_overlap, len(accumulated))
                    chunks.append(ChunkResult(
                        chunk_idx, accumulated, len(accumulated),
                        len(accumulated.split()) * 4, ov if chunk_idx > 1 else 0,
                        {"strategy": "RecursiveCharacterTextSplitter", "separator": "\\n\\n"}
                    ))
                    chunk_idx += 1
                    # Retain overlap tail
                    overlap_tail = accumulated[-ov:] if ov > 0 else ""
                    accumulated = f"{overlap_tail}\n\n{p}".strip() if overlap_tail else p
                else:
                    # Single paragraph exceeds chunk_size, split by sentences
                    sentences = p.split(". ")
                    for s in sentences:
                        chunks.append(ChunkResult(
                            chunk_idx, s.strip(), len(s), len(s.split()) * 4, 0,
                            {"strategy": "RecursiveCharacterTextSplitter", "separator": ". "}
                        ))
                        chunk_idx += 1

        if accumulated:
            ov = min(chunk_overlap, len(accumulated))
            chunks.append(ChunkResult(
                chunk_idx, accumulated, len(accumulated),
                len(accumulated.split()) * 4, ov if chunk_idx > 1 else 0,
                {"strategy": "RecursiveCharacterTextSplitter", "separator": "EOF"}
            ))

        return chunks

    def split_markdown_headers(self, text: str) -> List[ChunkResult]:
        lines = text.split("\n")
        chunks: List[ChunkResult] = []
        current_header = "Preamble"
        current_sub = "General"
        accumulated = []
        chunk_idx = 1

        for line in lines:
            if line.startswith("# "):
                if accumulated:
                    body = "\n".join(accumulated).strip()
                    if body:
                        chunks.append(ChunkResult(
                            chunk_idx, body, len(body), len(body.split()) * 4, 0,
                            {"Header_1": current_header, "Header_2": current_sub}
                        ))
                        chunk_idx += 1
                    accumulated = []
                current_header = line.replace("# ", "").strip()
                current_sub = "General"
            elif line.startswith("## "):
                if accumulated:
                    body = "\n".join(accumulated).strip()
                    if body:
                        chunks.append(ChunkResult(
                            chunk_idx, body, len(body), len(body.split()) * 4, 0,
                            {"Header_1": current_header, "Header_2": current_sub}
                        ))
                        chunk_idx += 1
                    accumulated = []
                current_sub = line.replace("## ", "").strip()
            else:
                accumulated.append(line)

        if accumulated:
            body = "\n".join(accumulated).strip()
            if body:
                chunks.append(ChunkResult(
                    chunk_idx, body, len(body), len(body.split()) * 4, 0,
                    {"Header_1": current_header, "Header_2": current_sub}
                ))

        return chunks

    def split_semantic(self, text: str) -> List[ChunkResult]:
        # Simulates breakpoint detection when topic shifts significantly
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[ChunkResult] = []
        for idx, p in enumerate(paragraphs, 1):
            chunks.append(ChunkResult(
                idx, p, len(p), len(p.split()) * 4, 0,
                {"strategy": "SemanticChunker", "topic_shift_score": f"{0.72 + (idx * 0.05):.2f}"}
            ))
        return chunks


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="12.4 Text Splitting",
    page_icon="✂️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">12.4 Text Splitting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Chunking strategies, chunk size, overlap dynamics, recursive splitting, markdown headers, and semantic boundaries.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Document Chunking & Text Splitting Hierarchy", divider="rainbow")

st.markdown("""
Document chunking is the single most critical engineering determinant of RAG retrieval precision.
If chunks are too large, the embedding vector averages out distinct facts, burying the specific answer in semantic noise. 
If chunks are too small, sentences are severed from their necessary supporting context.
Text Splitters apply structured grammatical hierarchies to decompose long prose into high-density vector chunks.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ChunkingArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_long_doc {
            label="1. Ingested Full-Length Document";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            FullDoc [label="Full Document Stream (15,000 Tokens)\\n'Enterprise Policy: Section 1... Section 42...'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_splitting_engine {
            label="2. Text Splitting Hierarchy (Recursive Fallback)";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            SepPara [label="Separator 1: '\\n\\n' (Paragraph Boundary)\\nKeep paragraphs together if <= chunk_size", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            SepSentence [label="Separator 2: '\\n' or '. ' (Sentence Boundary)\\nFall back to sentence breaks if paragraph too long", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            SepWord [label="Separator 3: ' ' (Word Boundary)\\nFall back to word spacing if sentence too long", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            SepChar [label="Separator 4: '' (Character Boundary)\\nAbsolute emergency fallback; never breaks words", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
        }

        subgraph cluster_chunk_sliding_window {
            label="3. Bounded Chunks with Overlap Windows";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            fontsize=12;

            Chunk1 [label="Chunk 1: [Tokens 0 to 500]\\n'Policy Section 1: Travel per-diem...'", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            OverlapZone [label="Sliding Overlap Window (50 Tokens)\\nShared context to prevent severed sentence loss", shape=parallelogram, fillcolor="#B45309", fontcolor="#FFFFFF", color="#F59E0B"];
            Chunk2 [label="Chunk 2: [Tokens 450 to 950]\\n'...Travel per-diem rules and expense SLAs...'", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        }

        subgraph cluster_vectorization {
            label="4. Downstream Vector Indexing";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            DenseEmbed [label="Individual Vector Embeddings\\nEach chunk mapped to distinct 1536D vector", shape=cylinder, fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399", penwidth=2];
        }

        FullDoc -> SepPara [label="1. Ingest text"];
        SepPara -> SepSentence [label="If > chunk_size"];
        SepSentence -> SepWord [label="If > chunk_size"];
        SepWord -> SepChar [label="If > chunk_size"];

        SepPara -> Chunk1 [label="Fits chunk_size"];
        SepSentence -> Chunk1 [label="Fits chunk_size"];

        Chunk1 -> OverlapZone [label="Trailing 50 tokens"];
        OverlapZone -> Chunk2 [label="Leading 50 tokens"];

        Chunk1 -> DenseEmbed [label="Vector 1"];
        Chunk2 -> DenseEmbed [label="Vector 2"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: WHY CHUNKING IS NEEDED ---
st.header("2. Why Chunking is Needed", divider="rainbow")

st.markdown("""
Why not simply embed entire documents as single vectors?
Embedding an entire 100-page manual into one embedding vector produces an **average semantic representation** of everything in the book. 
When a user asks: *"What is the torque specification for bolt 4B?"*, the general document vector will fail to match the query 
because the specific fact was diluted across 50,000 irrelevant words.

### Core Objectives of Chunking:
1. **High Semantic Density**: Each chunk represents a focused, coherent topic or discrete fact.
2. **Embedding Model Limits**: Most embedding models enforce hard input limits (e.g. 512, 2048, or 8192 tokens).
3. **Prompt Budget Optimization**: Minimizes token consumption by injecting only the precise 3 paragraphs needed to answer the question.
""")

st.divider()

# --- SECTION 3: CHUNK SIZE ---
st.header("3. Chunk Size", divider="rainbow")

st.markdown("""
`chunk_size` defines the maximum length of an individual chunk. 
Choosing the optimal chunk size depends on your document genre and retrieval queries:
""")

size_df = pd.DataFrame([
    {"Chunk Size": "Small (100 - 300 tokens)", "Target Domain": "Customer Support FAQs, Glossary, Factoid Q&A", "Retrieval Pros": "Extremely high precision; matches exact keywords and concise answers.", "Tradeoffs": "Lacks surrounding context; may miss conditional clauses."},
    {"Chunk Size": "Medium (400 - 800 tokens)", "Target Domain": "Standard Enterprise Docs, HR Policies, Wikis", "Retrieval Pros": "The industry standard sweet spot. Captures complete paragraphs and reasoning.", "Tradeoffs": "Balanced; requires moderate context window budgeting."},
    {"Chunk Size": "Large (1000 - 2000 tokens)", "Target Domain": "Legal Contracts, Financial Analysis, Narrative Summaries", "Retrieval Pros": "Preserves deep cross-clause context and complex contractual logic.", "Tradeoffs": "Higher token cost; risks noise dilution in cosine similarity search."}
])
st.dataframe(size_df, use_container_width=True)

st.divider()

# --- SECTION 4: CHUNK OVERLAP ---
st.header("4. Chunk Overlap", divider="rainbow")

st.markdown("""
`chunk_overlap` creates a sliding window between sequential chunks, repeating the trailing tokens of Chunk 1 at the beginning of Chunk 2.

Without overlap, critical information that spans a chunk boundary is severed:
- *Chunk 1 End*: "The executive bonus pool is authorized for all VP-level staff..."
- *Chunk 2 Start*: "...provided that division revenue exceeds 100M USD, otherwise bonus is zero."

If a user asks *"What are the bonus conditions for VPs?"*, Chunk 1 matches without the condition, and Chunk 2 matches without the subject.
A **10% - 20% overlap** ensures the semantic connection is preserved in both vectors.
""")

st.divider()

# --- SECTION 5: RECURSIVE CHARACTER SPLITTING ---
st.header("5. Recursive Character Splitting", divider="rainbow")

st.markdown("""
The `RecursiveCharacterTextSplitter` is LangChain's recommended default splitter. 
It attempts to keep text semantically intact by splitting on a prioritized list of separators:
1. `\n\n` (Paragraphs)
2. `\n` (Lines)
3. ` ` (Spaces / Words)
4. `""` (Characters)
""")

st.code("""from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\\n\\n", "\\n", " ", ""]
)

chunks = text_splitter.split_text(raw_document_text)""", language="python")

st.divider()

# --- SECTION 6: MARKDOWN SPLITTING ---
st.header("6. Markdown Splitting", divider="rainbow")

st.markdown("""
When chunking technical manuals or documentation, standard character splitters blindly cut across section headers.
`MarkdownHeaderTextSplitter` splits along document structure, capturing the header hierarchy directly into chunk metadata:
""")

st.code("""from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)
header_chunks = markdown_splitter.split_text(markdown_text)

# Every chunk's metadata tracks its exact position:
# metadata: {'Header 1': 'System Architecture', 'Header 2': 'Ingress Gateways'}""", language="python")

st.divider()

# --- SECTION 7: TOKEN-BASED SPLITTING ---
st.header("7. Token-Based Splitting", divider="rainbow")

st.markdown("""
Character counts vary dramatically from token counts: 100 characters of English prose might be 22 tokens, 
while 100 characters of JSON, code, or non-Latin scripts can be 60+ tokens.
`TokenTextSplitter` (backed by `tiktoken`) guarantees chunks strictly adhere to LLM token boundaries:
""")

st.code("""from langchain_text_splitters import TokenTextSplitter

token_splitter = TokenTextSplitter(
    chunk_size=256,
    chunk_overlap=24,
    encoding_name="cl100k_base"  # OpenAI GPT-4 tokenizer
)
token_chunks = token_splitter.split_text(code_or_json_payload)""", language="python")

st.divider()

# --- SECTION 8: SEMANTIC CHUNKING ---
st.header("8. Semantic Chunking", divider="rainbow")

st.markdown("""
Instead of splitting at arbitrary character or token limits, **Semantic Chunking** uses an embedding model 
to compute the semantic distance between consecutive sentences. 
When the cosine distance between Sentence N and Sentence N+1 exceeds a statistical breakpoint threshold 
(e.g. 95th percentile shift), the splitter inserts a chunk boundary.
""")

st.divider()

# --- SECTION 9: CHUNKING STRATEGIES ---
st.header("9. Chunking Strategies", divider="rainbow")

st.markdown("""
### Advanced Pattern: The Parent-Document Retriever
A common dilemma in RAG is:
- **Small chunks** are required for precise vector matching.
- **Large chunks** are required for the LLM to understand context.

The **Parent-Document Retriever** solves this by creating two tiers of chunks:
1. Splits raw documents into large **Parent Chunks** (e.g. 1,500 tokens).
2. Sub-divides each parent into small **Child Chunks** (e.g. 200 tokens).
3. Indexes only the small child chunks in the vector database.
4. When a child chunk matches a user query, the retriever fetches its **parent chunk** from docstore and injects the parent into the LLM prompt.
""")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Multi-Strategy Document Splitting Engine", divider="rainbow")

st.markdown("""
The following complete Python program implements multiple chunking strategies with token estimation, 
metadata enrichment, and sliding overlap window verification.
""")

st.code("""from typing import List, Dict, Any

class ProductionTextSplitter:
    \"\"\"Production chunking harness supporting recursive, markdown, and token splitting.\"\"\"
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_by_paragraphs(self, text: str, source_id: str) -> List[Dict[str, Any]]:
        paragraphs = [p.strip() for p in text.split("\\n\\n") if p.strip()]
        chunks = []
        current_chunk = ""
        chunk_index = 0

        for p in paragraphs:
            if len(current_chunk) + len(p) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\\n\\n{p}".strip()
            else:
                if current_chunk:
                    chunks.append({
                        "id": f"{source_id}-chunk-{chunk_index}",
                        "content": current_chunk,
                        "char_count": len(current_chunk),
                        "token_est": len(current_chunk.split()) * 4,
                        "metadata": {"source": source_id, "index": chunk_index}
                    })
                    chunk_index += 1
                    # Apply overlap window
                    tail = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_chunk = f"{tail}\\n\\n{p}".strip()
                else:
                    current_chunk = p

        if current_chunk:
            chunks.append({
                "id": f"{source_id}-chunk-{chunk_index}",
                "content": current_chunk,
                "char_count": len(current_chunk),
                "token_est": len(current_chunk.split()) * 4,
                "metadata": {"source": source_id, "index": chunk_index}
            })

        return chunks""", language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Real-Time Chunking Playground & Overlap Visualizer", divider="rainbow")

st.markdown("""
Experiment with different chunk sizes, overlap windows, and splitting algorithms in real time. 
Inspect the resulting chunks and verify how overlapping windows preserve semantic boundaries.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Splitting Configuration")
    sample_text_input = st.text_area(
        "Document Text to Split",
        value="""# Enterprise Cloud Infrastructure Policy

## Section 1: Ingress Gateways
All public HTTP traffic must traverse the AWS CloudFront CDN with AWS WAF enabled. Direct IP exposure on application load balancers is strictly prohibited. Security groups must restrict port 443 access to verified CloudFront IP ranges.

## Section 2: Database Snapshot Retention
Production Aurora PostgreSQL clusters must take automated continuous backups with a 35-day point-in-time recovery window. Weekly manual snapshots must be encrypted with KMS and replicated to eu-west-1 for disaster recovery.

## Section 3: Identity & Access Management
IAM users are deprecated. All human access must use Okta SSO via AWS IAM Identity Center with mandatory WebAuthn FIDO2 hardware keys. Session durations must not exceed 8 hours.""",
        height=220
    )

    strategy_choice = st.selectbox(
        "Splitting Strategy",
        options=["Recursive Character Splitting", "Markdown Header Splitting", "Semantic Chunking"]
    )

    wb_size = st.slider("Target Chunk Size (Characters)", min_value=150, max_value=800, value=300, step=50)
    wb_overlap = st.slider("Chunk Overlap (Characters)", min_value=0, max_value=150, value=40, step=10)

    btn_split = st.button("🚀 Execute Text Splitter", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Emitted Chunks & Boundary Breakdown")

    if btn_split:
        sim = MockChunkingPlaygroundSimulator()
        if strategy_choice == "Recursive Character Splitting":
            results = sim.split_recursive(sample_text_input, wb_size, wb_overlap)
        elif strategy_choice == "Markdown Header Splitting":
            results = sim.split_markdown_headers(sample_text_input)
        else:
            results = sim.split_semantic(sample_text_input)

        st.metric("Total Chunks Created", len(results))

        for c in results:
            with st.expander(f"📦 Chunk #{c.index} ({c.char_len} chars | ~{c.token_est} tokens)", expanded=True):
                st.write(f"`{c.text}`")
                if c.overlap_prev > 0:
                    st.caption(f"🔁 Overlap with previous chunk: ~{c.overlap_prev} characters")
                st.json(c.metadata)
    else:
        st.info("Adjust chunk settings and click **Execute Text Splitter** to visualize chunks.")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why is RecursiveCharacterTextSplitter preferred over naive CharacterTextSplitter?",
     "CharacterTextSplitter splits rigidly on a single separator (e.g. character count or space), frequently cutting mid-paragraph or mid-sentence. RecursiveCharacterTextSplitter uses a hierarchical list of separators (paragraph '\\n\\n', then newline '\\n', then space ' '). It tries to keep paragraphs together first, only falling back to finer separators when a chunk exceeds chunk_size, preserving grammatical structure."),
    ("How do you determine the optimal chunk size and overlap for an enterprise RAG application?",
     "Optimal chunk size depends on query complexity and document density. Highly specific factoid lookups benefit from smaller chunks (200-400 tokens) to maximize vector similarity. Multi-hop reasoning and policy interpretation require medium-to-large chunks (600-1000 tokens) with 10-20% overlap. Teams empirically benchmark using RAG evaluation frameworks (such as Ragas or TruLens) across answer relevancy and context precision metrics."),
    ("Explain the Parent-Document Retriever pattern and what problem it solves.",
     "The Parent-Document Retriever solves the tension between search precision and synthesis context. It indexes small child chunks (e.g. 200 tokens) in the vector store for high-accuracy embedding search. When a child chunk is retrieved, the system fetches the larger parent document (e.g. 1500 tokens) from docstore and passes the parent to the LLM, giving the model full surrounding context without search dilution."),
    ("What are the advantages of MarkdownHeaderTextSplitter over character-based splitters for technical documentation?",
     "Markdown documents contain intrinsic semantic structure in their headers (#, ##, ###). MarkdownHeaderTextSplitter splits along header boundaries and embeds the full breadcrumb path into chunk metadata (e.g. {'Header 1': 'Auth', 'Header 2': 'OAuth2'}). This preserves contextual hierarchy even if a section body is only 3 sentences long."),
    ("How does Semantic Chunking differ from rule-based text splitting?",
     "Rule-based splitters split at predetermined character or token counts regardless of content. Semantic Chunking generates embeddings for individual sentences and calculates the cosine distance between adjacent sentences. When the distance crosses a statistical threshold (such as a 95th percentile shift), it detects a change in subject matter and places a breakpoint, keeping coherent topics together.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Setting Chunk Overlap Greater Than Chunk Size. Setting chunk_overlap >= chunk_size causes an infinite loop or duplicate redundant chunks. Overlap must always be strictly smaller than chunk_size (typically 10-20%).")
    st.error("**Trap 2 (Breaking Error)**: Blind Splitting of Markdown Tables. Character splitters often cut directly across table rows, splitting Markdown tables into unparseable text fragments. Always use Markdown or specialized table splitters to keep rows intact.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Measuring Chunk Size in Characters While Vector Limits Are in Tokens. Assuming 1 character equals 1 token causes unexpected token truncation when embedding code or dense JSON. Always use TokenTextSplitter for strict token quotas.")
    st.warning("**Trap 4 (Quality Warning)**: Zero Chunk Overlap. Splitting text with 0 overlap risks breaking sentences exactly at the answer boundary, causing vector search to miss the query entirely.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Recursive Splitting"):
        st.markdown("**Hierarchical fallback: splits on paragraphs, then newlines, then spaces, preserving structure.**")
    with st.expander("🎴 Flashcard 2: Chunk Overlap Ratio"):
        st.markdown("**Typically 10-20% of chunk size; prevents semantic loss across boundary cuts.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Parent-Document Pattern"):
        st.markdown("**Index small child chunks for vector search; retrieve large parent chunk for LLM generation.**")
    with st.expander("🎴 Flashcard 4: Markdown Header Metadata"):
        st.markdown("**Splits by headers and injects section hierarchy into chunk metadata breadcrumbs.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand why document chunking is essential for high-precision semantic retrieval.", value=True)
chk2 = st.checkbox("I know how to balance chunk size and overlap ratios based on domain requirements.", value=True)
chk3 = st.checkbox("I can implement RecursiveCharacterTextSplitter with hierarchical separator fallbacks.", value=True)
chk4 = st.checkbox("I understand how MarkdownHeaderTextSplitter captures breadcrumbs in metadata.", value=True)
chk5 = st.checkbox("I can articulate the architectural mechanics of the Parent-Document Retriever pattern.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 text splitting interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 12.4 Text Splitting and completed all of Chapter 12: Retrieval-Augmented Generation (RAG)!")
