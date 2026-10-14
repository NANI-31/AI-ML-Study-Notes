import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="15.3 Text Splitters - Vector DB Curriculum",
    page_icon="✂️",
    layout="wide"
)

# Title & Overview
st.title("15.3 LangChain Text Splitters: Character, Recursive, Token-Based & Overlap Tuning")
st.caption("Chapter 15: LangChain + Vector Databases | Domain: 4. Vector Database")
st.markdown("""
Once documents are loaded into memory, they must be partitioned into semantically cohesive, bite-sized segments 
before embedding. In LangChain, this task is managed by the **`TextSplitter`** hierarchy in the dedicated 
`langchain-text-splitters` package.

Text splitters solve the delicate tension between **model context limits** and **semantic completeness**:
* If chunks are too small, they lose surrounding context and coreferences (*"it"*, *"they"*).
* If chunks are too large, specific facts get diluted, and retrieval similarity drops.
* If splitters cut text naively, they sever words in half or tear sentences apart across boundaries.

In this lesson, we explore the core LangChain splitters: **`CharacterTextSplitter`**, **`RecursiveCharacterTextSplitter`**, 
and **`TokenTextSplitter`**, examine the critical difference between **character counts and tiktoken tokens**, 
and test an interactive side-by-side Splitter Comparison Lab.
""")

st.divider()

# Section 1: The LangChain TextSplitter Hierarchy
st.header("1. The LangChain TextSplitter Hierarchy", divider="rainbow")

col_hier1, col_hier2 = col_hier1, col_hier2 = st.columns([1.1, 0.9])

with col_hier1:
    st.markdown("""
    ### The Core TextSplitter Contract
    Every splitter inherits from `TextSplitter` and implements two foundational methods:
    
    1. **`.split_text(text: str) -> List[str]`**:
       * Takes a raw string and returns a list of chunk strings.
    2. **`.split_documents(documents: List[Document]) -> List[Document]`**:
       * Takes a list of `Document` objects and breaks them into chunk `Document` objects.
       * **Automatic Metadata Inheritance**: Every generated chunk inherits the parent document's complete `metadata` dictionary (source, page, author, etc.), ensuring uninterrupted provenance.
       * **Optional Header Extraction**: Splitters like `MarkdownHeaderTextSplitter` add new metadata keys (e.g. `{"Header 1": "Intro", "Header 2": "Architecture"}`) directly to each chunk!
    """)

with col_hier2:
    st.markdown("### TextSplitter Class Hierarchy")
    split_dot = """
    digraph TextSplitterHierarchy {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Base [label="langchain_text_splitters.TextSplitter\\n(Abstract Base Class)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        CharSplit [label="CharacterTextSplitter\\n(Single Separator)", fillcolor="#FEE2E2", color="#DC2626"];
        RecursiveSplit [label="RecursiveCharacterTextSplitter\\n(Hierarchical Fallback)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
        TokenSplit [label="TokenTextSplitter\\n(BPE Tiktoken Encoder)", fillcolor="#FEF3C7", color="#D97706"];
        MarkdownSplit [label="MarkdownHeaderTextSplitter\\n(Structural Outline)", fillcolor="#E0E7FF", color="#4338CA"];

        Base -> CharSplit;
        Base -> RecursiveSplit;
        Base -> TokenSplit;
        Base -> MarkdownSplit;
    }
    """
    st.graphviz_chart(split_dot, use_container_width=False)

st.divider()

# Section 2: Detailed Breakdown of the Three Core Splitters
st.header("2. Detailed Breakdown of the Three Core Splitters", divider="rainbow")

tab_sp1, tab_sp2, tab_sp3, tab_sp4 = st.tabs([
    "1. RecursiveCharacterTextSplitter",
    "2. CharacterTextSplitter",
    "3. Token-Based Splitters",
    "4. Markdown & Code Splitters"
])

with tab_sp1:
    st.markdown("### 1. RecursiveCharacterTextSplitter (The Gold Standard)")
    st.markdown("""
    * **Algorithm**: Attempts to split along a list of separators in descending order of semantic size:
      `["\\n\\n", "\\n", " ", ""]`.
    * **Behavior**: Keeps entire paragraphs together whenever possible. If a paragraph exceeds `chunk_size`, it splits that paragraph along lines. If a line is still too long, it splits along words.
    * **Rule**: This should be your **default choice for 95% of text RAG applications**.
    """)
    st.code("""from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

chunks = splitter.split_text(long_text)
print(f"Produced {len(chunks)} chunks.")
""", language="python")

with tab_sp2:
    st.markdown("### 2. CharacterTextSplitter (Single Separator)")
    st.markdown("""
    * **Algorithm**: Splits strictly on a single configured separator (default `separator="\\n\\n"`).
    * **The Trap**: If a single paragraph is 4,000 characters long and does not contain `\\n\\n`, `CharacterTextSplitter` **refuses to split it**, outputting a chunk that grossly exceeds `chunk_size` and printing a warning!
    * **Verdict**: Avoid unless your documents have guaranteed strict paragraph limits.
    """)
    st.code("""from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\\n\\n",
    chunk_size=1000,
    chunk_overlap=100
)
# Caution: Can produce oversized chunks if \\n\\n is missing!
""", language="python")

with tab_sp3:
    st.markdown("### 3. TokenTextSplitter (Tiktoken BPE Encoding)")
    st.markdown("""
    * **Algorithm**: Uses OpenAI's `tiktoken` byte-pair encoder (BPE) to count exact tokens (`cl100k_base` or `o200k_base`) rather than raw characters.
    * **Why It Matters**: Embedding models have strict **token limits**, not character limits. An 800-character string containing code or mathematical symbols could be 400 tokens, while plain English might only be 200 tokens.
    * **Alternative Pattern**: Use `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)` to combine recursive separator preservation with exact token budgeting!
    """)
    st.code("""from langchain_text_splitters import RecursiveCharacterTextSplitter

# Best of both worlds: Recursive boundaries + Exact token budget!
token_aware_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o",
    chunk_size=500,     # Exactly 500 TOKENS
    chunk_overlap=50    # Exactly 50 TOKENS
)
""", language="python")

with tab_sp4:
    st.markdown("### 4. Markdown & Language Splitters")
    st.markdown("""
    * **MarkdownHeaderTextSplitter**: Slices documents based on header levels (`#`, `##`, `###`) and injects the header hierarchy into the metadata of every child chunk:
    """)
    st.code("""from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = md_splitter.split_text(markdown_text)

# Every chunk contains metadata: {'Header 1': 'Intro', 'Header 2': 'Installation'}
""", language="python")

st.divider()

# Section 3: Characters vs Tokens: The Sizing Trap
st.header("3. Characters vs Tokens: The Sizing Trap", divider="rainbow")

col_trap_c1, col_trap_c2 = col_trap_c1, col_trap_c2 = st.columns([1, 1])

with col_trap_c1:
    st.markdown("""
    ### The 4:1 Character-to-Token Rule of Thumb
    In standard English text, 1 token is approximately **4 characters** (or ~0.75 words):
    * `chunk_size = 1000` (characters) ≈ **~250 tokens**.
    * `chunk_size = 2000` (characters) ≈ **~500 tokens**.
    
    ### Where the Heuristic Breaks Down
    * **Code & JSON**: Semicolons, brackets, and indentations inflate token count (1 token ≈ 1.5 - 2 characters).
    * **Non-English Languages**: Asian languages (Chinese, Japanese) or accented languages use multiple byte tokens per character (1 character ≈ 1 - 3 tokens!).
    * **Solution**: Always use `.from_tiktoken_encoder()` or pass `length_function=tiktoken_len` when working with hard embedding model token limits.
    """)

with col_trap_c2:
    st.markdown("""
    ### Chunk Overlap: Preserving Context Across Boundaries
    ```text
    Source Text: "The database password is SecretPass. It rotates every 30 days."
    
    Split with Overlap = 0:
    [Chunk 0]: "The database password is SecretPass."
    [Chunk 1]: "It rotates every 30 days."  <-- What rotates? Entity lost!
    
    Split with Overlap = 20%:
    [Chunk 0]: "The database password is SecretPass."
    [Chunk 1]: "password is SecretPass. It rotates every 30 days."  <-- Entity preserved!
    ```
    * **Sweet Spot**: Maintain **10% to 20% overlap** (e.g. 100 tokens overlap for a 500-token chunk).
    """)

st.divider()

# Section 4: Interactive Workbench: Side-by-Side Splitter Comparison Lab
st.header("4. Interactive Workbench: Side-by-Side Splitter Comparison Lab", divider="rainbow")

st.markdown("""
Test how different LangChain text splitters segment a live technical document. 
Observe chunk count, boundary decisions, and token distributions.
""")

SAMPLE_TEXT = """1.0 Authentication and Security Gateway Architecture

The enterprise ingress tier utilizes Envoy proxies deployed as edge gateway sidecars. All incoming client connections must negotiate TLS 1.3 with forward secrecy.

Mutual TLS (mTLS) is enforced for internal microservice communication. Each Kubernetes pod mounts an ephemeral SPIFFE X.509 certificate managed by the cluster SPIRE agent. Certificates have a strict 1-hour time-to-live (TTL) and rotate automatically.

2.0 Database Persistence and Disaster Recovery

Primary transactional storage is hosted on multi-region PostgreSQL clusters. Database write-ahead logs (WAL) are streamed continuously to Amazon S3 with server-side encryption enabled.

Point-In-Time Recovery (PITR) is maintained for a rolling window of 35 days. The Recovery Point Objective (RPO) is strictly under 5 minutes, while the Recovery Time Objective (RTO) for full region failover is 4 hours."""

col_lab_cfg1, col_lab_cfg2 = col_lab_cfg1, col_lab_cfg2 = st.columns([1, 1])

with col_lab_cfg1:
    st.subheader("1. Splitter Configuration")
    
    active_splitter_name = st.selectbox(
        "Choose LangChain Splitter:",
        [
            "RecursiveCharacterTextSplitter (Recommended)",
            "CharacterTextSplitter (Single Separator)",
            "TokenTextSplitter (Simulated Tiktoken BPE)"
        ]
    )
    
    ui_chunk_chars = st.slider("Target Chunk Size (Chars):", min_value=150, max_value=600, value=300, step=25)
    ui_overlap_chars = st.slider("Chunk Overlap (Chars):", min_value=0, max_value=100, value=40, step=10)

with col_lab_cfg2:
    st.subheader("2. Source Document Preview")
    st.info(f"Source Document Length: **{len(SAMPLE_TEXT)} characters** (~{len(SAMPLE_TEXT)//4} tokens)")
    st.caption("Contains two major numbered sections (1.0 Security, 2.0 Database Persistence).")

# Splitter Execution Logic
def simulate_recursive_splitter(text: str, chunk_size: int, overlap: int) -> List[str]:
    paras = text.split("\n\n")
    chunks = []
    curr = ""
    for p in paras:
        if len(curr) + len(p) + 2 <= chunk_size:
            curr = (curr + "\n\n" + p).strip()
        else:
            if curr:
                chunks.append(curr)
                tail = curr[-overlap:] if overlap > 0 and len(curr) > overlap else ""
                curr = (tail + "\n\n" + p).strip() if tail else p
            else:
                chunks.append(p[:chunk_size])
                curr = p[chunk_size - overlap:]
    if curr:
        chunks.append(curr)
    return chunks

def simulate_character_splitter(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Splits only on \n\n
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras

def simulate_token_splitter(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Fixed word slice
    words = text.split()
    target_words = chunk_size // 5
    overlap_words = overlap // 5
    step = max(1, target_words - overlap_words)
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + target_words]))
    return chunks

if "Recursive" in active_splitter_name:
    sim_chunks = simulate_recursive_splitter(SAMPLE_TEXT, ui_chunk_chars, ui_overlap_chars)
elif "CharacterTextSplitter" in active_splitter_name:
    sim_chunks = simulate_character_splitter(SAMPLE_TEXT, ui_chunk_chars, ui_overlap_chars)
else:
    sim_chunks = simulate_token_splitter(SAMPLE_TEXT, ui_chunk_chars, ui_overlap_chars)

st.divider()

col_lab_chunks, col_lab_metrics = col_lab_chunks, col_lab_metrics = st.columns([1.2, 0.8])

with col_lab_chunks:
    st.subheader(f"Generated Chunks ({len(sim_chunks)} Total)")
    for idx, c in enumerate(sim_chunks):
        with st.expander(f"✂️ Chunk {idx+1} ({len(c)} chars / ~{len(c)//4} tokens)", expanded=True):
            st.markdown(f"**Content**:\n\n{c}")

with col_lab_metrics:
    st.subheader("Chunking Metrics & Health Check")
    lengths = [len(c) for c in sim_chunks]
    st.metric(label="Total Chunks Produced", value=len(sim_chunks))
    st.metric(label="Average Chunk Size", value=f"{int(np.mean(lengths))} chars")
    st.metric(label="Max Chunk Size", value=f"{max(lengths)} chars")
    
    if "CharacterTextSplitter" in active_splitter_name and max(lengths) > ui_chunk_chars:
        st.warning("⚠️ Warning: CharacterTextSplitter produced chunks larger than target chunk_size because paragraphs lacked `\\n\\n` breaks!")
    else:
        st.success("✅ Clean boundaries: All chunks strictly comply with target size limits.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: How does LangChain's RecursiveCharacterTextSplitter avoid cutting words in half?"):
    st.markdown("""
    **Answer**:
    The recursive splitter checks separators in order: `["\n\n", "\n", " ", ""]`.
    * When a paragraph is too long, it steps down to newlines (`\n`).
    * When a line is too long, it steps down to spaces (`" "`).
    * Because space (`" "`) is checked before raw characters (`""`), the algorithm splits strictly along word boundaries.
    * Only if a single isolated word is longer than `chunk_size` (e.g. an unbroken 1,000-character base64 string or URL) does it fall back to character slicing.
    """)

with st.expander("Q2: Why should you use `RecursiveCharacterTextSplitter.from_tiktoken_encoder` instead of standard `RecursiveCharacterTextSplitter`?"):
    st.markdown("""
    **Answer**:
    By default, `RecursiveCharacterTextSplitter` measures length using Python's `len()` function, which counts **characters**.
    * Embedding models (like OpenAI `text-embedding-3` or Cohere) have limits defined strictly in **BPE tokens** (e.g., 512 or 8,192 tokens).
    * Setting `chunk_size=1000` assuming tokens will actually create tiny ~250-token chunks. Conversely, in languages with non-Latin scripts, 1,000 characters could translate to 1,500 tokens, silently exceeding model limits.
    * `.from_tiktoken_encoder` uses `tiktoken` as the length counter while retaining recursive paragraph/sentence boundary preservation.
    """)

with st.expander("Q3: What is the purpose of `MarkdownHeaderTextSplitter` and how is it used in two-stage chunking?"):
    st.markdown("""
    **Answer**:
    `MarkdownHeaderTextSplitter` splits structured Markdown documents along header outlines (`#`, `##`, `###`) and injects the header path into the chunk's `metadata` dictionary.
    
    **The Two-Stage Chunking Pattern**:
    1. **Stage 1 (Structural Split)**: Use `MarkdownHeaderTextSplitter` to divide the document into major thematic sections, attaching metadata like `{"section": "Installation", "topic": "Docker"}`.
    2. **Stage 2 (Sub-Chunking)**: Pass the resulting sections through `RecursiveCharacterTextSplitter(chunk_size=500)` to ensure individual passages fit within embedding limits.
    Every resulting chunk retains the structural context of the section it came from!
    """)

with st.expander("Q4: What happens if `chunk_overlap` is set larger than `chunk_size`?"):
    st.markdown("""
    **Answer**:
    Setting `chunk_overlap >= chunk_size` raises a `ValueError` in LangChain. 
    Mathematically, overlap dictates how many characters are re-used from the previous chunk. If overlap equals or exceeds chunk size, the sliding window step (`step = chunk_size - overlap`) becomes zero or negative, causing an infinite loop.
    """)

with st.expander("Q5: How do code splitters like `PythonCodeTextSplitter` differ from general prose splitters?"):
    st.markdown("""
    **Answer**:
    Prose splitters prioritize double newlines and punctuation. 
    `PythonCodeTextSplitter` uses Python-specific syntactic separators:
    `["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]`.
    This ensures that classes and function signatures remain bundled with their docstrings and bodies, preventing code chunks from being syntactically broken.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Confusing Character Counts with Token Counts**  
    Setting `chunk_size = 512` in a character splitter assuming tokens creates tiny 120-token fragments that lack context. Use `.from_tiktoken_encoder()`.
    """)
    st.warning("""
    **Trap 2: Using CharacterTextSplitter on Markdown Without Blank Lines**  
    If documentation uses single newlines instead of double newlines, `CharacterTextSplitter` fails to split, outputting massive chunks that cause model truncation.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Zero Overlap on Legal or Regulatory Text**  
    Setting overlap to 0 severs cross-sentence references (*"Such indemnity shall not apply..."*), causing retrieval to miss critical clause conditions. Maintain 15-20% overlap.
    """)
    st.warning("""
    **Trap 4: Excessive Overlap (> 40%)**  
    Setting 50% overlap fills the vector store with redundant duplicate sentences, wasting vector storage and polluting the LLM context prompt.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the fallback hierarchy of Recursive splitter?"):
        st.markdown("""
        **Fallback Hierarchy**:  
        `["\n\n", "\n", " ", ""]` — Paragraphs -> Lines -> Words -> Characters.
        """)
    with st.expander("🗂️ Flashcard: What does split_documents do?"):
        st.markdown("""
        **Metadata Inheritance**:  
        Splits `Document` objects while copying all parent metadata attributes to every child chunk.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How does MarkdownHeaderTextSplitter work?"):
        st.markdown("""
        **Header Outlines**:  
        Splits on `#` and `##`, attaching the section names as metadata tags to child chunks.
        """)
    with st.expander("🗂️ Flashcard: What is the recommended overlap ratio?"):
        st.markdown("""
        **10% to 20%**:  
        Preserves cross-boundary entities without inflating token redundancy.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the TextSplitter hierarchy and how `.split_documents()` preserves metadata.", value=True)
chk2 = st.checkbox("I know how RecursiveCharacterTextSplitter navigates its hierarchical fallback separators.", value=True)
chk3 = st.checkbox("I understand the difference between character length functions and tiktoken BPE token counts.", value=True)
chk4 = st.checkbox("I can implement two-stage structural chunking using MarkdownHeaderTextSplitter.", value=True)
chk5 = st.checkbox("I know how to tune chunk size and overlap (10-20%) for enterprise RAG accuracy.", value=True)
chk6 = st.checkbox("I can implement domain-specific code and document splitters in LangChain.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Text Splitters and are ready for Chapter 15.4: LangChain Retrieval Strategies!")
