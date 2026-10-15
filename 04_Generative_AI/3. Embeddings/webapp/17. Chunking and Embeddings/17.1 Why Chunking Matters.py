import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="17.1 Why Chunking Matters - Embeddings Curriculum",
    page_icon="✂️",
    layout="wide"
)

# Title & Overview
st.title("17.1 Why Chunking Matters: The Long Document Problem, Vector Dilution, & Boundary Integrity")
st.caption("Chapter 17: Chunking and Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
A common beginner mistake in AI engineering is attempting to feed an entire 50-page PDF or 10,000-word report directly into an embedding model.
Even if an embedding model supports an 8,192-token context window, doing so is catastrophic for retrieval quality.

This occurs because of **Vector Averaging Dilution**:
* An embedding vector is a fixed-dimensional coordinate (e.g. 1536 floats).
* Compressing 10,000 words into a single vector averages out all specific facts, dates, numbers, and nuances into a generic semantic blur.
* Searching for a specific clause (e.g. *"What is the warranty period for replacement batteries?"*) will fail because that single sentence represents only 0.1% of the whole document's average vector!

**Chunking** is the disciplined engineering process of partitioning documents into semantically coherent passages.
This lesson explores the physics of chunking, context window budgets, the Goldilocks chunk size tradeoff, and boundary preservation.
""")

st.divider()

# Section 1: The Physics of Chunking & Vector Dilution
st.header("1. The Long Document Problem & Vector Averaging Dilution", divider="rainbow")

col_phys1, col_phys2 = st.columns([1, 1])

with col_phys1:
    st.markdown("""
    ### Why Whole-Document Embeddings Fail
    * **1. Vector Averaging Dilution**:
      * Most transformer embedding models perform **mean pooling** over output token vectors.
      * In a 5,000-word document discussing finance, legal liabilities, product roadmaps, and employee benefits, the resulting average vector lands in a generic middle ground that matches none of those topics with high specificity.
    * **2. Hard Model Token Limits**:
      * Open-source models like `all-MiniLM-L6-v2` or `bge-base-en` have a strict hard limit of **512 tokens**. Any text beyond token 512 is silently truncated and discarded!
    * **3. LLM Prompt Budget & Needle Precision**:
      * Retrieving an entire 20-page document into an LLM prompt forces the model to search through irrelevant distractors, triggering the **Lost-in-the-Middle** hallucination effect.
    """)

with col_phys2:
    st.markdown("### Whole-Document vs. Chunked Embeddings")
    dilute_dot = """
    digraph VectorDilution {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_bad {
            label="Whole Document (Diluted)"; style="dashed"; color="#EF4444";
            DocAll [label="Whole 50-Page PDF\\n(25,000 words across 10 topics)", fillcolor="#FEE2E2"];
            VecDilute [label="Single Average Vector (d=1536)\\nGeneric centroid coordinate\\nLost specific facts", fillcolor="#FCA5A5", color="#DC2626", style="filled,bold"];
        }

        subgraph cluster_good {
            label="Chunked Partitioning (High Fidelity)"; style="dashed"; color="#10B981";
            C1 [label="Chunk 1: Pricing Tier\\n(300 tokens)", fillcolor="#D1FAE5"];
            C2 [label="Chunk 2: Warranty SLA\\n(300 tokens)", fillcolor="#D1FAE5"];
            C3 [label="Chunk 3: Security Certs\\n(300 tokens)", fillcolor="#D1FAE5"];
            V1 [label="Vector 1 (Pricing)", fillcolor="#A7F3D0"];
            V2 [label="Vector 2 (Warranty)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
            V3 [label="Vector 3 (Security)", fillcolor="#A7F3D0"];
        }

        DocAll -> VecDilute [label="Mean pooling over 25k words"];
        C1 -> V1; C2 -> V2; C3 -> V3;
        UserQ [label="Query: 'Battery warranty duration?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        UserQ -> VecDilute [label="Low match (0.41)\\nDiluted", color="#DC2626"];
        UserQ -> V2 [label="High match (0.92)\\nPinpoint precision!", color="#059669"];
    }
    """
    st.graphviz_chart(dilute_dot, use_container_width=False)

st.markdown("### The Goldilocks Chunk Size Tradeoff")
st.markdown("""
| Dimension | Small Chunks (50 - 150 Tokens) | Medium Chunks (250 - 500 Tokens) | Large Chunks (800 - 1500 Tokens) |
| :--- | :--- | :--- | :--- |
| **Semantic Specificity** | **Extremely High** (Sharp vector focus) | **Balanced** (Optimal for most RAG) | Low (Diluted across multiple concepts) |
| **Context Completeness** | Low (Missing background & pronouns) | **High** (Self-contained thought) | Very High (Full document section) |
| **LLM Generation Risk** | Fragmented facts, hallucinated context | **Minimal** (Industry standard) | Slower generation, high prompt cost |
| **Best Used For** | FAQ matching, pinpoint entity lookup | Enterprise manuals, policies, blogs | Long-form legal contracts, book chapters |
""")

st.divider()

# Section 2: Production Code: Sliding Window Chunking with Overlap
st.header("2. Production Code: Boundary-Aware Sliding Window Chunker", divider="rainbow")

st.markdown("""
The production utility below demonstrates how sliding-window chunking preserves semantic boundaries: chunking text into controlled token budgets while maintaining overlap to prevent fact severance.
""")

st.code(r'''
from typing import List, Dict, Any

class BoundaryAwareChunker:
    """
    Splits text along sentence boundaries while enforcing
    strict maximum token size and sliding overlap.
    """
    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_into_sentences(self, text: str) -> List[str]:
        # Basic sentence tokenizer splitting on sentence terminators
        import re
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw_sentences if s.strip()]

    def chunk(self, text: str, doc_id: str = "doc_01") -> List[Dict[str, Any]]:
        sentences = self.split_into_sentences(text)
        chunks: List[Dict[str, Any]] = []
        
        current_words: List[str] = []
        current_sentences: List[str] = []
        
        for sent in sentences:
            sent_words = sent.split()
            
            # If adding this sentence exceeds chunk_size, finalize current chunk
            if len(current_words) + len(sent_words) > self.chunk_size and current_words:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "chunk_id": f"{doc_id}_c{len(chunks)+1:02d}",
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "sentences_count": len(current_sentences)
                })
                
                # Sliding window overlap: keep last N words for overlap
                overlap_words: List[str] = []
                overlap_sentences: List[str] = []
                for s in reversed(current_sentences):
                    s_w = s.split()
                    if len(overlap_words) + len(s_w) <= self.overlap:
                        overlap_words = s_w + overlap_words
                        overlap_sentences.insert(0, s)
                    else:
                        break
                
                current_words = overlap_words + sent_words
                current_sentences = overlap_sentences + [sent]
            else:
                current_words.extend(sent_words)
                current_sentences.append(sent)

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "chunk_id": f"{doc_id}_c{len(chunks)+1:02d}",
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "sentences_count": len(current_sentences)
            })

        return chunks
''', language="python")

st.divider()

# Section 3: Interactive Chunking Dilemma & Boundary Workbench
st.header("3. Interactive Workbench: The Boundary Severance & Overlap Studio", divider="rainbow")

st.markdown("""
Test how chunk size and overlap affect fact retention. If a critical causal statement is split directly across a boundary without overlap, neither chunk will contain the full factual relationship.
""")

col_ch1, col_ch2 = st.columns([1, 1])

with col_ch1:
    st.subheader("1. Document Input")
    default_policy = (
        "Article 4: Employee Severance and Termination Benefits.\n"
        "Employees involuntarily terminated without cause are eligible for standard severance pay. "
        "Severance is calculated at the rate of two weeks base salary per completed year of continuous service. "
        "However, employees terminated for gross misconduct, breach of confidentiality, or insubordination "
        "are strictly disqualified from all severance packages and accrued bonus distributions. "
        "All severance payments require the signed execution of a comprehensive release of claims agreement "
        "within 45 calendar days of official separation. "
        "Health insurance coverage under COBRA will be subsidized by the company for a period of up to 90 days "
        "following the formal separation date for all qualifying eligible employees."
    )

    doc_text = st.text_area("Enterprise Policy Text:", value=default_policy, height=180)

    st.subheader("2. Chunking Parameters")
    target_size = st.slider("Max Chunk Size (Words):", min_value=20, max_value=80, value=35, step=5)
    overlap_val = st.slider("Chunk Overlap (Words):", min_value=0, max_value=20, value=10, step=2)

with col_ch2:
    st.subheader("3. Partitioned Chunks & Overlap Inspection")

    # Simple word-level sliding window for demonstration
    words = doc_text.split()
    step_size = max(1, target_size - overlap_val)
    
    generated_chunks = []
    for i in range(0, len(words), step_size):
        chunk_w = words[i : i + target_size]
        generated_chunks.append({
            "ID": f"Chunk {len(generated_chunks)+1:02d}",
            "Words": len(chunk_w),
            "Start": i,
            "End": i + len(chunk_w),
            "Text": " ".join(chunk_w)
        })

    st.success(f"Generated **{len(generated_chunks)}** chunks (Step: {step_size} words, Overlap: {overlap_val} words)")

    for c in generated_chunks:
        with st.expander(f"{c['ID']} ({c['Words']} words, Indices [{c['Start']} : {c['End']}])", expanded=True):
            st.markdown(f"> {c['Text']}")

    st.subheader("4. Boundary Severance Vulnerability Test")
    needle_query = st.selectbox(
        "Test Retrieval of Specific Fact:",
        [
            "Are employees terminated for gross misconduct eligible for severance?",
            "How many days to sign the release of claims agreement?",
            "What is the COBRA health insurance subsidy duration?"
        ]
    )

    # Check which chunks contain the keywords
    st.markdown(f"**Query:** *{needle_query}*")
    matches = []
    for c in generated_chunks:
        # Simple keyword presence check
        c_text_lower = c["Text"].lower()
        if "misconduct" in needle_query and ("misconduct" in c_text_lower and "severance" in c_text_lower):
            matches.append(c["ID"])
        elif "release of claims" in needle_query and ("release" in c_text_lower and "45" in c_text_lower):
            matches.append(c["ID"])
        elif "cobra" in needle_query and ("cobra" in c_text_lower and "90" in c_text_lower):
            matches.append(c["ID"])

    if matches:
        st.success(f"Fact successfully captured in: **{', '.join(matches)}**! (Overlap preserved the fact complete).")
    else:
        st.error("Boundary Severance! The fact was cut in half across adjacent chunks. Increase chunk size or overlap.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Chunking Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is 'Vector Averaging Dilution' and why does it prevent whole-document embedding?**
    * **Answer**: In transformer models, sequence embeddings are generated via mean-pooling across all token hidden states. When a document spans thousands of words covering multiple distinct topics, the mathematical mean of these thousands of vectors converges toward a generic centroid in high-dimensional space. Specific facts, numbers, and clauses become washed out, causing cosine similarity against targeted queries to plummet.

    **Q2: What is the primary purpose of Chunk Overlap in sliding-window text splitters?**
    * **Answer**: Without overlap, text is arbitrarily severed at a fixed token or character threshold. If a critical causal statement or relationship spans across that exact boundary (e.g. *'The application crashed [Boundary] because the database connection timed out'*), neither chunk contains the complete fact. Introducing 10-20% overlap guarantees that boundary sentences are fully represented in at least one contiguous chunk.

    **Q3: How do you choose between small (100 token) vs large (1000 token) chunks for an enterprise RAG application?**
    * **Answer**:
      * Choose **Small Chunks (100-200 tokens)** when: queries are specific factual questions (e.g., error codes, dates, numbers, definitions), vector search requires high precision, or prompt context budget is constrained.
      * Choose **Large Chunks (500-1000 tokens)** when: queries require understanding broad narratives, complex reasoning, legal clauses with extensive cross-references, or when downstream LLMs need rich surrounding context to synthesize an answer.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Zero Overlap Splitting**:
      * *Issue*: Splitting documents with `chunk_overlap=0` to save vector database storage.
      * *Consequence*: Up to 15% of critical factual claims are sliced directly at clause boundaries, causing retrieval failures on subtle questions.
      * *Fix*: Always maintain an overlap between 10% and 20% of `chunk_size`.

    * **Trap 2: Ignoring Semantic Formatting Boundaries (Tables & Code Blocks)**:
      * *Issue*: Using a character splitter that slices markdown tables or python functions right through the middle.
      * *Consequence*: The embedding model receives corrupted table syntax or syntax-error code snippets that fail semantic search.
      * *Fix*: Use markdown-aware or code-aware splitters that treat code blocks and tables as atomic, unsplittable units.

    * **Trap 3: Oversized Chunks Exceeding Embedding Context Windows**:
      * *Issue*: Setting `chunk_size=1000` tokens on a model with a 512-token context window (`all-MiniLM-L6-v2`).
      * *Consequence*: The model silently truncates the last 488 tokens of every chunk, losing half the knowledge base.
      * *Fix*: Always verify the `max_seq_length` of your embedding model and set `chunk_size <= max_seq_length`.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Vector Averaging Dilution**
      * *Definition*: The loss of specific granular factual signal that occurs when long multi-topic documents are mean-pooled into a single fixed-dimension vector.
    * **Flashcard 2: Chunk Overlap**
      * *Definition*: Duplicating a sliding window of trailing tokens from chunk $N$ into the beginning of chunk $N+1$ to ensure facts crossing boundaries are not lost.
    * **Flashcard 3: The Goldilocks Chunk Size**
      * *Definition*: Balancing chunk size (typically 256-512 tokens) so it is small enough for vector specificity yet large enough for context completeness.
    * **Flashcard 4: Hard Token Truncation**
      * *Definition*: The silent dropping of all tokens beyond an embedding model's maximum sequence length (e.g. 512 tokens in standard BERT architectures).
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Why Chunking Matters:

- [ ] I can explain the mechanics of Vector Averaging Dilution and why whole-document embedding fails.
- [ ] I know why embedding models have hard token truncation limits.
- [ ] I can articulate the tradeoffs between small, medium, and large chunks.
- [ ] I understand how chunk overlap prevents boundary fact severance.
- [ ] I know how to select the optimal chunk size based on document structure and LLM requirements.
""")
