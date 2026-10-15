import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="12.3 Document Chunking - Vector DB Curriculum",
    page_icon="✂️",
    layout="wide"
)

# Title & Overview
st.title("12.3 Document Chunking: Strategies, Overlap, Boundaries & Semantic Splitting")
st.caption("Chapter 12: Document Processing for Vector Databases | Domain: 4. Vector Database")
st.markdown("""
In vector database engineering, **Document Chunking**—the strategy used to break long documents into smaller segments—is 
the **single most influential factor governing RAG retrieval accuracy**.

Chunking solves two foundational engineering constraints:
1. **Hard Token Limits**: Embedding models cannot accept infinite text (e.g., `bge-base` has a 512-token limit; `text-embedding-3` caps at 8,192).
2. **Semantic Dilution (Information Loss)**: Compressing an entire 40-page technical manual into a single 1536-dimensional vector 
   averages out all distinct topics into generic noise. Specific facts become impossible to retrieve.

However, poorly chunked documents introduce disastrous failure modes: cutting words in half, severing subject-verb propositions, 
or separating questions from their answers across boundaries.

In this lesson, we break down the five major chunking paradigms: **Fixed-Size, Sentence-Based, Paragraph-Based, Recursive Character, and Semantic Chunking**, 
analyze the mechanics of **Chunk Overlap**, and experiment with an interactive multi-strategy chunking visualizer.
""")

st.divider()

# Section 1: The Core Chunking Dilemma & Architectural Rationale
st.header("1. The Core Chunking Dilemma & Architectural Rationale", divider="rainbow")

col_dil1, col_dil2 = col_dil1, col_dil2 = st.columns([1.1, 0.9])

with col_dil1:
    st.markdown("""
    ### Why Chunking is Mandatory
    * **The Semantic Dilution Law**: A vector embedding represents a single point in concept space. As document length grows, the vector regresses toward the geometric center of all topics, losing specificity.
    * **Context Window Precision**: If an LLM needs to answer *"What is the timeout for Redis connection pooling?"*, injecting a precise 200-word chunk containing the exact config flag allows the LLM to generate an immediate, hallucination-free answer. Injecting a 20-page document drowns the signal in irrelevant noise.
    
    ### The Chunk Size Trade-Off Spectrum
    * **Small Chunks (100 - 250 tokens)**:
      * *Strength*: Hyper-focused semantic specificity. High cosine similarity matches.
      * *Weakness*: Lacks surrounding context (e.g., pronoun references like *"it"* or *"they"* lose their antecedent).
    * **Large Chunks (800 - 1,500 tokens)**:
      * *Strength*: Retains full narrative context, multi-paragraph reasoning, and table structures.
      * *Weakness*: Lower cosine similarity against brief search queries; fewer unique chunks fit into prompt context windows.
    * **Industry Sweet Spot**: **400 to 600 tokens (~1,600 - 2,400 characters) with 10% to 20% overlap**.
    """)

with col_dil2:
    st.markdown("### Chunking Architecture & Hierarchy")
    chunk_dot = """
    digraph ChunkHierarchy {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        FullDoc [label="Source Document\\n(10,000 words / 40 pages)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        Splitter [label="Chunking Strategy\\nRecursive / Semantic / Sentence", fillcolor="#FEF3C7", color="#D97706"];

        subgraph cluster_chunks {
            label="Chunk Sequence (with 15% sliding overlap)"; style="dashed"; color="#059669";
            C1 [label="Chunk 0 (500 tokens)\\nTokens 0..500", fillcolor="#D1FAE5"];
            C2 [label="Chunk 1 (500 tokens)\\nTokens 425..925 (Overlap: 75)", fillcolor="#A7F3D0"];
            C3 [label="Chunk 2 (500 tokens)\\nTokens 850..1350 (Overlap: 75)", fillcolor="#6EE7B7"];
        }

        VectorDB [label="Vector Store (ChromaDB)\\nIndexed Independent Chunks", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        FullDoc -> Splitter;
        Splitter -> C1;
        Splitter -> C2;
        Splitter -> C3;
        C1 -> VectorDB;
        C2 -> VectorDB;
        C3 -> VectorDB;
    }
    """
    st.graphviz_chart(chunk_dot, use_container_width=False)

st.divider()

# Section 2: Deep Dive into the Five Major Chunking Strategies
st.header("2. Deep Dive into the Five Major Chunking Strategies", divider="rainbow")

tab_strat1, tab_strat2, tab_strat3, tab_strat4, tab_strat5 = st.tabs([
    "1. Fixed-Size Chunking",
    "2. Sentence-Based",
    "3. Paragraph-Based",
    "4. Recursive Character",
    "5. Semantic Chunking"
])

with tab_strat1:
    st.markdown("""
    ### 1. Fixed-Size Chunking (Naive Windowing)
    * **Mechanism**: Slices text strictly every `N` characters or tokens (e.g., exactly 1,000 characters).
    * **Pros**: Trivially fast (`O(1)` slice computation); uniform vector sizes.
    * **Cons**: Completely ignores syntax. Frequently cuts words in half (`"datab"` in chunk 1, `"ase"` in chunk 2) and severs mid-sentence clauses.
    * **Verdict**: Strongly discouraged in modern production pipelines.
    """)

with tab_strat2:
    st.markdown(r"""
    ### 2. Sentence-Based Chunking
    * **Mechanism**: Uses sentence boundary detection (regex `[.!?]\s+`, NLTK `sent_tokenize`, or spaCy) to ensure every chunk contains a whole number of complete grammatical sentences.
    * **Pros**: Never breaks a propositional statement; preserves grammatical integrity.
    * **Cons**: Sentences vary wildly in length (from 3 words to 80 words); requires an accumulator to bundle sentences up to a target token limit.
    * **Code Pattern**:
    ```python
    import re

    def sentence_chunker(text: str, max_chars: int = 1000) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) > max_chars and current:
                chunks.append(current.strip())
                current = s + " "
            else:
                current += s + " "
        if current.strip():
            chunks.append(current.strip())
        return chunks
    ```
    """)

with tab_strat3:
    st.markdown("""
    ### 3. Paragraph-Based Chunking
    * **Mechanism**: Splits on double newlines (`\\n\\n`).
    * **Pros**: Respects the original author's conceptual structure. Each paragraph represents a single coherent thought or topic.
    * **Cons**: Single paragraphs in academic papers or legal filings can be 2,000 tokens long (exceeding model limits), while in dialogues they may be 5 tokens long.
    """)

with tab_strat4:
    st.markdown("""
    ### 4. Recursive Character Text Splitting (The Industry Standard)
    * **Mechanism**: Popularized by LangChain (`RecursiveCharacterTextSplitter`). 
    * Evaluates a **hierarchical list of separators** in descending order of semantic granularity:
      1. `\\n\\n` (Paragraph break)
      2. `\\n` (Line break)
      3. `" "` (Word boundary)
      4. `""` (Individual character fallback)
    * **Algorithm**: Tries to split on paragraphs first. If a paragraph exceeds `chunk_size`, it recursively splits that paragraph on line breaks. If a line is still too long, it splits on spaces.
    * **Why It Dominates**: Guarantees that chunks never exceed `chunk_size` while keeping the largest possible semantic blocks intact without cutting words.
    """)

with tab_strat5:
    st.markdown("""
    ### 5. Semantic Chunking (Embedding-Based Segmentation)
    * **Mechanism**: Rather than relying on arbitrary character counts, semantic chunking identifies **topic transitions**:
      1. Splits text into individual sentences.
      2. Generates embeddings for each sentence (or sliding windows of 3 sentences).
      3. Computes cosine distance between consecutive sentence embeddings.
      4. Identifies **distance spikes** (drops in similarity greater than a threshold, e.g. 95th percentile).
      5. Splits text at the semantic breakpoint where the topic changes.
    * **Pros**: Creates variable-length chunks grouped strictly by conceptual coherence.
    * **Cons**: Computationally expensive during ingestion (requires embedding every sentence individually).
    """)

st.divider()

# Section 3: The Mechanics of Chunk Overlap
st.header("3. The Mechanics of Chunk Overlap", divider="rainbow")

col_ov1, col_ov2 = col_ov1, col_ov2 = st.columns([1, 1])

with col_ov1:
    st.markdown("""
    ### Why Chunk Overlap is Critical
    Consider a document containing:
    > *"To reset the master password, an admin must generate a one-time token. This token remains valid for exactly 15 minutes before expiring."*

    If a text splitter without overlap cuts the document right after *"one-time token"*:
    * **Chunk 1**: *"To reset the master password, an admin must generate a one-time token."*
    * **Chunk 2**: *"This token remains valid for exactly 15 minutes before expiring."*

    Now, a user asks: *"How long is a password reset token valid?"*
    * Chunk 2 contains "15 minutes", but the embedding model does not know what "This token" refers to (the entity definition is in Chunk 1).
    * Search recall fails or the LLM cannot confidently attribute the answer!

    ### The Overlap Solution
    By introducing **Chunk Overlap** (e.g., 20% overlap), the tail of Chunk 1 is duplicated as the head of Chunk 2:
    * **Chunk 2 with Overlap**: *"generate a one-time token. This token remains valid for exactly 15 minutes before expiring."*
    * Context and coreferences are perfectly preserved.
    """)

with col_ov2:
    st.markdown("""
    ### Overlap Tuning Rules of Thumb
    
    | Chunk Size (Tokens) | Recommended Overlap (Tokens) | Overlap % | Typical Use Case |
    | :--- | :--- | :--- | :--- |
    | **200 - 300** | 30 - 50 | ~15% | FAQ items, API endpoints |
    | **500 - 600** | 50 - 100 | ~15% - 18% | Standard RAG, Knowledge bases |
    | **1000 - 1500** | 150 - 250 | ~15% - 20% | Legal contracts, Research papers |
    
    * **Caution**: Overlap higher than 30% creates excessive redundancy, inflating vector storage costs and filling prompt windows with duplicate sentences.
    """)

st.divider()

# Section 4: Interactive Workbench: Multi-Strategy Chunking Simulator
st.header("4. Interactive Workbench: Multi-Strategy Chunking Simulator", divider="rainbow")

st.markdown("""
Compare how different chunking algorithms segment a live technical document. Adjust chunk size and overlap 
to visualize chunk boundaries, token distributions, and overlap regions.
""")

SAMPLE_CORPUS = """Kubernetes Pod Autoscaling Architecture

The Horizontal Pod Autoscaler (HPA) automatically adjusts the number of replica pods based on observed CPU utilization, memory consumption, or custom application metrics.

The HPA controller queries the resource metrics API every 15 seconds by default. When the observed metric exceeds the target threshold by more than the configured tolerance (typically 10%), the controller calculates the desired replica count using the scaling equation.

During a scale-down event, the controller enforces a stabilization window of 300 seconds to prevent thrashing or rapid flapping. This ensures temporary dips in load do not prematurely terminate pods.

For persistent workloads requiring stateful storage, the Vertical Pod Autoscaler (VPA) is preferred over HPA. VPA modifies the resource requests and limits of existing pods rather than spinning up additional replicas."""

col_sim_cfg1, col_sim_cfg2 = col_sim_cfg1, col_sim_cfg2 = st.columns([1, 1])

with col_sim_cfg1:
    st.subheader("1. Chunking Strategy & Knobs")
    chosen_strat = st.selectbox(
        "Select Chunking Strategy:",
        [
            "Recursive Character Splitting (Recommended)",
            "Sentence-Based Splitting",
            "Fixed-Size Windowing (Character Slice)",
            "Paragraph Splitting (\\n\\n)"
        ]
    )
    
    ui_chunk_size = st.slider("Target Chunk Size (Characters):", min_value=150, max_value=800, value=350, step=25)
    ui_overlap = st.slider("Chunk Overlap (Characters):", min_value=0, max_value=150, value=50, step=10)

with col_sim_cfg2:
    st.subheader("2. Sample Input Preview")
    st.info(f"Source Text Length: **{len(SAMPLE_CORPUS)} characters** (~{len(SAMPLE_CORPUS)//4} tokens)")
    st.caption("Document contains 4 distinct paragraphs covering HPA, metrics API, scale-down stabilization, and VPA.")

# Python Implementation of Chunkers
def execute_recursive_splitter(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Simplified recursive splitter simulation
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + p).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Apply overlap from the tail of current_chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    # snap to word boundary
                    words = overlap_text.split(" ", 1)
                    if len(words) > 1:
                        overlap_text = words[1]
                    current_chunk = (overlap_text + "\n\n" + p).strip()
                else:
                    current_chunk = p
            else:
                # p itself is larger than chunk_size, split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', p)
                for s in sentences:
                    if len(current_chunk) + len(s) + 1 <= chunk_size:
                        current_chunk = (current_chunk + " " + s).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = s
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def execute_sentence_splitter(text: str, chunk_size: int, overlap: int) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, curr = [], ""
    for s in sentences:
        if len(curr) + len(s) + 1 > chunk_size and curr:
            chunks.append(curr.strip())
            curr = s + " "
        else:
            curr += s + " "
    if curr.strip():
        chunks.append(curr.strip())
    return chunks

def execute_fixed_slice(text: str, chunk_size: int, overlap: int) -> List[str]:
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Run Selected Strategy
if "Recursive" in chosen_strat:
    generated_chunks = execute_recursive_splitter(SAMPLE_CORPUS, ui_chunk_size, ui_overlap)
elif "Sentence" in chosen_strat:
    generated_chunks = execute_sentence_splitter(SAMPLE_CORPUS, ui_chunk_size, ui_overlap)
elif "Fixed" in chosen_strat:
    generated_chunks = execute_fixed_slice(SAMPLE_CORPUS, ui_chunk_size, ui_overlap)
else:  # Paragraph
    generated_chunks = [p.strip() for p in SAMPLE_CORPUS.split("\n\n") if p.strip()]

st.divider()

col_chunks_view, col_chunks_meta = st.columns([1.3, 0.7])

with col_chunks_view:
    st.subheader(f"Generated Chunks ({len(generated_chunks)} Total)")
    
    for idx, c_text in enumerate(generated_chunks):
        with st.expander(f"📦 Chunk {idx} ({len(c_text)} chars / ~{len(c_text)//4} tokens)", expanded=True):
            st.markdown(f"**Content**:\n\n{c_text}")
            st.caption(f"Starts with: `'{c_text[:30]}...'` | Ends with: `'...{c_text[-30:]}'`")

with col_chunks_meta:
    st.subheader("Chunking Metrics & Quality")
    
    avg_len = int(np.mean([len(c) for c in generated_chunks]))
    max_len = max([len(c) for c in generated_chunks])
    min_len = min([len(c) for c in generated_chunks])
    
    st.metric(label="Total Chunks Produced", value=len(generated_chunks))
    st.metric(label="Average Chunk Size", value=f"{avg_len} chars")
    st.metric(label="Size Range", value=f"{min_len} - {max_len} chars")
    
    if "Fixed" in chosen_strat:
        st.error("⚠️ Warning: Fixed slicing breaks words mid-string. Note how words are cut at boundaries.")
    else:
        st.success("✅ Clean boundaries: Preserved natural sentences or paragraph breaks.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: How does LangChain's RecursiveCharacterTextSplitter prioritize separators?"):
    st.markdown("""
    **Answer**:
    The recursive splitter operates via a hierarchical fallback list: `["\n\n", "\n", " ", ""]`.
    1. It first attempts to split text on **paragraph boundaries (`\n\n`)**, keeping complete paragraphs intact.
    2. If a paragraph is larger than `chunk_size`, it steps down to the next separator (`\n`) to split along lines.
    3. If a single line exceeds `chunk_size`, it steps down to word spaces (`" "`) to avoid breaking words.
    4. If a single word exceeds `chunk_size` (e.g. a long base64 string or URL), it falls back to raw character splitting (`""`).
    This ensures that semantic structure is preserved as high in the hierarchy as possible.
    """)

with st.expander("Q2: What is the 'Lost in the Middle' problem and how does chunk size affect it?"):
    st.markdown("""
    **Answer**:
    Research by Liu et al. (Stanford, 2023) demonstrated that LLMs are most effective at retrieving information placed at the **very beginning** or **very end** of the context prompt; information placed in the middle of a large context prompt is frequently overlooked.
    * If you use **large chunks (1,500+ tokens)**, the critical fact might sit in the middle of a retrieved chunk, increasing the likelihood that the LLM ignores it.
    * Using **smaller, focused chunks (300-500 tokens)** allows retrieving multiple highly specific passages that are directly relevant, mitigating the "Lost in the Middle" phenomenon.
    """)

with st.expander("Q3: How does Semantic Chunking differ from character-based splitting, and what is its primary drawback?"):
    st.markdown("""
    **Answer**:
    * **Semantic Chunking**: Breaks text into sentences, computes embeddings for each sentence, and measures the cosine distance between consecutive sentences. A split is placed wherever the distance exceeds a statistical threshold (e.g., 95th percentile jump), indicating that the author transitioned to a new topic.
    * **Primary Drawback**: **Ingestion Latency & Cost**. A 100-page document containing 3,000 sentences requires making 3,000 embedding API calls or running 3,000 local inferences just to decide where to place chunk cuts, slowing down document processing by 10x-50x.
    """)

with st.expander("Q4: Why should code files (Python, JavaScript) NOT be chunked using standard paragraph splitters?"):
    st.markdown("""
    **Answer**:
    In programming languages, indentation and scope blocks define syntax. 
    * A naive double-newline split can separate a function signature (`def compute_tax(user):`) from its docstring, body, or return statement, producing non-executable, syntactically broken code snippets.
    * **Code Chunking Best Practice**: Use Abstract Syntax Tree (AST) splitters (e.g., Tree-sitter or LangChain `PythonCodeTextSplitter`) that split cleanly at **class and function definition boundaries**, keeping imports attached to scope.
    """)

with st.expander("Q5: What is the Parent Document Retriever pattern and how does it reconcile small-chunk search with large-chunk context?"):
    st.markdown("""
    **Answer**:
    The Parent Document Retriever resolves the fundamental chunk size dilemma:
    * **Small chunks (100 tokens)** are ideal for **vector search** because their embeddings are hyper-focused and match user queries tightly.
    * **Large chunks (800 tokens)** are ideal for **LLM prompt context** because they provide rich surrounding narrative.
    
    **The Pattern**:
    1. During ingestion, store large parent documents in a key-value store (e.g. SQLite / Redis).
    2. Split parents into small child chunks (100 tokens) and store their embeddings in the vector database with metadata: `{"parent_id": doc_123}`.
    3. At query time, vector search retrieves the top child chunks, but instead of feeding child chunks to the LLM, the system looks up and injects the **full parent document**!
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Zero Overlap on Technical Text**  
    Setting `chunk_overlap = 0` guarantees that sentences crossing chunk boundaries lose their antecedents (*"This variable..."*), causing frequent retrieval failures on technical specifications. Maintain 10-20% overlap.
    """)
    st.warning("""
    **Trap 2: Character vs Token Confusion**  
    Setting `chunk_size = 512` assuming tokens when your splitter counts characters produces tiny 120-token chunks. 1 token ≈ 4 characters in English. Always verify your splitter's length function (`len` vs `tiktoken`).
    """)

with col_trap2:
    st.error("""
    **Trap 3: Chunking Markdown Tables Line-by-Line**  
    Splitting a Markdown table across chunks detaches data rows from their column header row, rendering the numbers meaningless. Wrap tables in atomic, non-split blocks.
    """)
    st.warning("""
    **Trap 4: Massive Overlap (> 40%)**  
    Setting overlap to 50% causes adjacent chunks to be nearly identical. Both Chunk A and Chunk B will match the query, filling the LLM context prompt with redundant duplicates.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is the industry standard chunker?"):
        st.markdown("""
        **Recursive Character Splitter**:  
        Splits hierarchically along `\n\n`, `\n`, `" "`, and `""` to preserve semantic boundaries without cutting words.
        """)
    with st.expander("🗂️ Flashcard: What is the purpose of Chunk Overlap?"):
        st.markdown("""
        **Context Preservation**:  
        Carries trailing context from chunk N into chunk N+1, preventing entity definitions and pronouns from being severed.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is the Parent Document Retriever?"):
        st.markdown("""
        **Parent-Child Pattern**:  
        Searches small child vector chunks for precision, but passes the larger parent document to the LLM for context.
        """)
    with st.expander("🗂️ Flashcard: What is Semantic Chunking?"):
        st.markdown("""
        **Embedding-Based Cuts**:  
        Splits text wherever cosine distance between consecutive sentences spikes, indicating a topic shift.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand why chunking is required to prevent semantic dilution and respect model token limits.", value=True)
chk2 = st.checkbox("I know how LangChain's RecursiveCharacterTextSplitter hierarchy functions.", value=True)
chk3 = st.checkbox("I can calculate optimal chunk size and overlap ratios (10-20%) for enterprise RAG.", value=True)
chk4 = st.checkbox("I understand the mechanics of Semantic Chunking and its ingestion latency trade-off.", value=True)
chk5 = st.checkbox("I know how the Parent Document Retriever reconciles small-chunk search with large-chunk context.", value=True)
chk6 = st.checkbox("I can implement robust code and table chunkers that protect syntactic scope.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have completed Chapter 12: Document Processing for Vector Databases and are ready for Chapter 13: Embedding Pipelines!")
