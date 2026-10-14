import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="18.3 Chunking Strategies - LLM Engineering",
    page_icon="✂️",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class ProductionChunker:
    """
    Self-contained implementation of enterprise chunking algorithms:
    - Fixed-Size Chunking with Overlap
    - Recursive Character Chunking
    - Simulated Semantic Distance Breakpoint Chunking
    """
    
    @staticmethod
    def fixed_size_chunk(text: str, chunk_size: int = 200, overlap: int = 40) -> List[Dict[str, Any]]:
        """Splits strictly by character length with sliding overlap."""
        chunks = []
        start = 0
        text_len = len(text)
        chunk_id = 1
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_content = text[start:end]
            chunks.append({
                "id": f"CHUNK-{chunk_id:03d}",
                "start": start,
                "end": end,
                "length": len(chunk_content),
                "text": chunk_content,
                "overlap_prev": text[start:start+overlap] if start > 0 else ""
            })
            if end == text_len:
                break
            start += (chunk_size - overlap)
            chunk_id += 1
            
        return chunks

    @staticmethod
    def recursive_character_chunk(
        text: str,
        chunk_size: int = 300,
        overlap: int = 50,
        separators: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively splits text by prioritizing natural semantic boundaries:
        1. Paragraphs ("\\n\\n")
        2. Lines ("\\n")
        3. Sentences (". ")
        4. Words (" ")
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " "]
            
        def _split_text(txt: str, seps: List[str]) -> List[str]:
            final_chunks = []
            separator = seps[-1]
            new_seps = []
            
            for i, s in enumerate(seps):
                if s == "":
                    separator = ""
                    break
                if s in txt:
                    separator = s
                    new_seps = seps[i+1:]
                    break
                    
            splits = txt.split(separator) if separator else list(txt)
            
            good_splits = []
            for s in splits:
                if len(s) < chunk_size:
                    good_splits.append(s)
                else:
                    if new_seps:
                        good_splits.extend(_split_text(s, new_seps))
                    else:
                        good_splits.append(s)
                        
            # Merge small adjacent splits with overlap
            merged = []
            current = ""
            for s in good_splits:
                candidate = (current + separator + s) if current else s
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        merged.append(current)
                    current = s
            if current:
                merged.append(current)
                
            return merged

        raw_splits = _split_text(text, separators)
        result = []
        for i, chunk_text in enumerate(raw_splits, 1):
            result.append({
                "id": f"RECURSIVE-{i:03d}",
                "length": len(chunk_text),
                "text": chunk_text
            })
        return result

    @staticmethod
    def semantic_distance_chunk(text: str, threshold_percentile: float = 80.0) -> List[Dict[str, Any]]:
        """
        Simulates Semantic Chunking (Kamradt algorithm):
        Splits text by sentences, calculates simulated cosine distance spikes between adjacent sentences,
        and cuts chunk boundaries at distance peaks.
        """
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) <= 1:
            return [{"id": "SEMANTIC-001", "length": len(text), "text": text, "sentence_count": len(sentences)}]
            
        # Simulate semantic distance between adjacent sentences
        np.random.seed(42)
        distances = []
        for i in range(len(sentences) - 1):
            # Check for keyword discontinuity to create deterministic semantic spikes
            s1, s2 = sentences[i], sentences[i+1]
            if ("however" in s2.lower() or "furthermore" in s2.lower() or "in contrast" in s2.lower() or "\n\n" in s1):
                dist = 0.85 # Strong semantic shift
            else:
                dist = float(np.random.uniform(0.15, 0.45))
            distances.append(dist)
            
        # Determine distance breakpoint threshold
        cutoff = np.percentile(distances, threshold_percentile)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        chunk_id = 1
        
        for i, dist in enumerate(distances):
            if dist >= cutoff:
                # Breakpoint triggered!
                chunk_str = " ".join(current_chunk_sentences)
                chunks.append({
                    "id": f"SEMANTIC-{chunk_id:03d}",
                    "length": len(chunk_str),
                    "text": chunk_str,
                    "sentence_count": len(current_chunk_sentences),
                    "split_distance": dist
                })
                chunk_id += 1
                current_chunk_sentences = [sentences[i+1]]
            else:
                current_chunk_sentences.append(sentences[i+1])
                
        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            chunks.append({
                "id": f"SEMANTIC-{chunk_id:03d}",
                "length": len(chunk_str),
                "text": chunk_str,
                "sentence_count": len(current_chunk_sentences),
                "split_distance": 0.0
            })
            
        return chunks


# --- HEADER & INTRODUCTION ---
st.title("18.3 Chunking Strategies: Recursive, Semantic, Sliding Window, & Size Optimization")
st.caption("Chapter 18: Retrieval-Augmented Generation — RAG | Domain: 2. Large Language Models")

st.markdown("""
Why can't we simply embed an entire 100-page enterprise manual into a single vector?

Because embedding models compress text into a fixed-dimensional latent coordinate (e.g. 1536 floats). 
Compressing 50,000 words into a single vector averages out fine-grained details into an indistinct semantic blur: 
specific numbers, error codes, and clauses are completely diluted. 
Conversely, splitting text into individual 5-word sentences creates fragmented vectors stripped of surrounding context and pronouns.

**Chunking** is the foundational segmentation step in RAG that negotiates the physical tension between 
**retrieval precision** (finding the needle) and **contextual completeness** (giving the LLM enough evidence to reason). 
This lesson covers fixed-size, sentence, paragraph, recursive character, and semantic distance chunking strategies.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: The Chunking Geometry & Semantic Granularity Spectrum", divider="rainbow")

st.markdown("""
How document segmentation balances vector precision, contextual completeness, and boundary preservation:
""")

chunk_blueprint_dot = """
digraph ChunkingSpectrum {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    subgraph cluster_input {
        label="Raw Document";
        color="#1E293B";
        fontcolor="#38BDF8";
        RawDoc [label="Extracted Clean Text\\n(e.g., 20,000 tokens)", fillcolor="#1E293B"];
    }

    subgraph cluster_strategies {
        label="Segmentation Strategies";
        color="#1E293B";
        fontcolor="#38BDF8";
        Fixed [label="Fixed-Size\\n(Naive Character Slices)\\nSplits words mid-thought", fillcolor="#7F1D1D"];
        SentencePara [label="Sentence / Paragraph\\n(Punctuation & \\n\\n)\\nVariable ragged chunk sizes", fillcolor="#0F766E"];
        Recursive [label="Recursive Character\\n(['\\n\\n', '\\n', '. ', ' '])\\nIndustry Standard Baseline", fillcolor="#0369A1"];
        Semantic [label="Semantic Distance\\n(Cosine Spikes Between Sentences)\\nDynamic concept groupings", fillcolor="#6D28D9"];
    }

    subgraph cluster_overlap {
        label="Boundary Protection";
        color="#1E293B";
        fontcolor="#38BDF8";
        SlidingWindow [label="Sliding Window Overlap\\n(10% - 20% token buffer)\\nPrevents boundary severed facts", fillcolor="#4338CA"];
    }

    subgraph cluster_dest {
        label="Vector Ingestion";
        color="#1E293B";
        fontcolor="#38BDF8";
        EmbeddingReady [label="Embedding Engine\\nUniform, context-rich chunks\\nready for Vector DB", fillcolor="#059669"];
    }

    RawDoc -> Fixed;
    RawDoc -> SentencePara;
    RawDoc -> Recursive;
    RawDoc -> Semantic;

    Fixed -> SlidingWindow;
    SentencePara -> SlidingWindow;
    Recursive -> SlidingWindow;
    Semantic -> SlidingWindow;

    SlidingWindow -> EmbeddingReady;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(chunk_blueprint_dot, use_container_width=True)

st.markdown("""
### Strategic Comparison of Modern Chunking Methodologies
""")

chunk_matrix = [
    {
        "Methodology": "Fixed-Size Chunking",
        "Segmentation Mechanism": "Slices text strictly every N characters or tokens",
        "Boundary Fidelity": "Extremely Low (breaks mid-word, splits sentences in half)",
        "Computational Cost": "Near Zero (string slicing)",
        "Best Use Cases": "Structured fixed-width log files, time series, DNA sequences"
    },
    {
        "Methodology": "Sentence Chunking",
        "Segmentation Mechanism": "Splits on punctuation regex or sentence tokenizers",
        "Boundary Fidelity": "High (clean grammatical units)",
        "Computational Cost": "Very Low (regex / NLTK)",
        "Best Use Cases": "Short FAQ pairs, translation units"
    },
    {
        "Methodology": "Paragraph Chunking",
        "Segmentation Mechanism": "Splits on double newlines ('\\n\\n')",
        "Boundary Fidelity": "Very High (natural human thought boundaries)",
        "Computational Cost": "Very Low",
        "Best Use Cases": "Well-formatted articles, blog posts, books"
    },
    {
        "Methodology": "Recursive Character Chunking",
        "Segmentation Mechanism": "Recursively falls back through ['\\n\\n', '\\n', '. ', ' '] until chunk < target",
        "Boundary Fidelity": "High (preserves paragraphs where possible, sentences otherwise)",
        "Computational Cost": "Low (fast CPU recursion)",
        "Best Use Cases": "General enterprise documentation, PDFs, manuals (Industry Standard)"
    },
    {
        "Methodology": "Semantic Chunking (Kamradt)",
        "Segmentation Mechanism": "Embeds every sentence; splits when cosine distance between neighbors spikes",
        "Boundary Fidelity": "Highest (groups conceptually aligned ideas dynamically)",
        "Computational Cost": "High (requires N embedding calls during chunking)",
        "Best Use Cases": "Complex academic research papers, dense legal contracts"
    }
]
st.dataframe(pd.DataFrame(chunk_matrix), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - FIXED-SIZE CHUNKING ---
st.header("2. Fixed-Size Chunking: Naive Slicing & Boundary Ruptures", divider="rainbow")

st.markdown("""
In **Fixed-Size Chunking**, text is segmented strictly every `N` characters or tokens (e.g., every 500 characters), 
regardless of where words, sentences, or paragraphs begin and end.

### The Semantic Rupture Danger
Because fixed-size chunking ignores linguistic structure:
- It frequently severs critical negations:
  ```text
  Chunk 1: "Under standard enterprise licensing, Acme Corp will not"
  Chunk 2: "be liable for any incidental, indirect, or consequential damages..."
  ```
  If a user asks *"Is Acme liable for damages?"*, Chunk 2 alone will be retrieved, **completely reversing the legal meaning of the contract!**
- It slices numbers, code variables, and names in half (e.g. `100,000` becomes `100,` in Chunk 1 and `000` in Chunk 2).
""")

st.divider()

# --- SECTION 3: TOPIC 2 - SENTENCE CHUNKING ---
st.header("3. Sentence Chunking: Grammatical Integrity & Pronoun Resolution", divider="rainbow")

st.markdown("""
**Sentence Chunking** segments text at grammatical punctuation boundaries using sentence tokenizers 
(regular expressions, NLTK `sent_tokenize`, or spaCy).

### The Pronoun Disconnection Challenge
While every individual sentence is grammatically complete, sentences rarely make sense in isolation:
- Consider this sequence:
  ```text
  Sentence 1: "The Apollo 11 Lunar Module touched down on the lunar surface on July 20, 1969."
  Sentence 2: "It was piloted by Neil Armstrong and Buzz Aldrin."
  ```
- If Sentence 2 is embedded as an independent chunk, its semantic vector is dominated by the generic pronoun `"It"`.
- When a user asks *"Who piloted the Apollo 11 Lunar Module?"*, Sentence 2 will score low semantic similarity 
  because the noun *"Apollo 11"* only exists in Sentence 1!
- **Solution**: Modern sentence chunking bundles sentences into rolling windows of 3 to 5 sentences to preserve referential context.
""")

st.divider()

# --- SECTION 4: TOPIC 3 - PARAGRAPH CHUNKING ---
st.header("4. Paragraph Chunking: Topic Boundaries & Natural Semantic Units", divider="rainbow")

st.markdown("""
**Paragraph Chunking** treats double newlines (`\n\n`) as natural thematic delimiters created by human authors.

### The Ragged Chunk Size Dilemma
While paragraphs represent complete thoughts, real-world human writing varies wildly in density:
- One paragraph might be a 4-word sub-header: `### Next Steps`.
- The following paragraph might be a 1,200-word uninterrupted explanation of a pharmaceutical drug mechanism.
- The 4-word chunk produces a tiny, noisy vector that matches random search queries.
- The 1,200-word chunk exceeds typical embedding model context limits (512 or 8192 tokens) and dilutes specific factual data.
- Paragraph chunking works best as the **primary tier of a recursive chunking strategy**, rather than an isolated rule.
""")

st.divider()

# --- SECTION 5: TOPIC 4 - RECURSIVE CHUNKING ---
st.header("5. Recursive Character Chunking: The Industry Standard Hierarchy", divider="rainbow")

st.markdown("""
**Recursive Character Chunking** is the gold standard for enterprise RAG (popularized by LangChain's 
`RecursiveCharacterTextSplitter` and LlamaIndex's `SentenceSplitter`).

It attempts to split text using a hierarchical list of separators, falling back to smaller granularities 
only when a chunk exceeds `chunk_size`:

```python
separators = ["\n\n", "\n", ". ", " ", ""]
```
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### The Recursive Execution Order
    1. **Tier 1: Paragraphs (`\n\n`)**: Tries to keep complete paragraphs together. If a paragraph is smaller than `chunk_size`, it remains unbroken.
    2. **Tier 2: Lines (`\n`)**: If a single paragraph is too large, it splits the paragraph by line breaks (preserving list items).
    3. **Tier 3: Sentences (`. `)**: If a single line is still too large, it splits on sentence boundaries.
    4. **Tier 4: Words (` `)**: If an individual sentence exceeds `chunk_size`, it splits on spaces between words.
    5. **Tier 5: Characters (`""`)**: Emergency fallback if a single unbroken word exceeds `chunk_size`.
    """)
with col2:
    st.info("""
    **Why Recursive Chunking Dominates Production**:
    - Keeps paragraphs intact whenever possible.
    - Guarantees that no resulting chunk ever exceeds the hard `chunk_size` token ceiling.
    - Preserves semantic flow without requiring expensive runtime embedding calls.
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - SEMANTIC CHUNKING ---
st.header("6. Semantic Chunking: Embedding Distance Spikes & Concept Drift Breakpoints", divider="rainbow")

st.markdown("""
First introduced by Greg Kamradt, **Semantic Chunking** does not split text based on character counts or punctuation rules; 
instead, it uses an embedding model to measure **semantic concept drift** between consecutive sentences.
""")

semantic_flow = """
digraph SemanticChunkingFlow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    Sentences [label="1. Split into Sentences\\n[S1, S2, S3, S4, S5]", fillcolor="#1E293B"];
    Embeddings [label="2. Embed Each Sentence\\n[e1, e2, e3, e4, e5]", fillcolor="#0369A1"];
    Distances [label="3. Compute Cosine Distances\\nd_i = 1 - CosSim(e_i, e_i+1)", fillcolor="#0F766E"];
    PeakDetect [label="4. Distance Spike Detection\\nFlag d_i > 90th Percentile", fillcolor="#7F1D1D"];
    DynamicChunks [label="5. Create Semantic Chunks\\nSplit strictly at distance peaks", fillcolor="#059669"];

    Sentences -> Embeddings -> Distances -> PeakDetect -> DynamicChunks;
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(semantic_flow, use_container_width=True)

st.markdown("""
### How Semantic Chunking Works Mathematically
1. Split the document into an array of individual sentences `[s_1, s_2, ..., s_m]`.
2. Generate a dense embedding vector `e_i` for each sentence.
3. Compute the cosine distance between each adjacent pair:
   ```text
   Distance(i) = 1.0 - Cosine_Similarity(e_i, e_{i+1})
   ```
4. Plot the distance curve across the document:
   - When consecutive sentences discuss the same topic, distance remains low (e.g. `0.1` to `0.3`).
   - When the author abruptly transitions to a new topic (e.g. moving from financial results to security policy), the distance **spikes** to `0.8+`.
5. Apply a statistical threshold (e.g. 90th or 95th percentile of distances) to cut chunk boundaries at distance peaks!
""")

st.divider()

# --- SECTION 7: TOPIC 6 - CHUNK OVERLAP ---
st.header("7. Chunk Overlap: The Sliding Window & Boundary Information Bleed", divider="rainbow")

st.markdown("""
Even with intelligent separators, critical facts will inevitably lie across the boundary between two adjacent chunks. 
**Chunk Overlap** applies a sliding window buffer, retaining the trailing `O` tokens of Chunk K as the leading prefix of Chunk K+1.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Sliding Window Mechanics")
    st.markdown("""
    Suppose `chunk_size = 500` characters and `overlap = 100` characters:
    - **Chunk 1**: Characters `0` to `500`
    - **Chunk 2**: Characters `400` to `900` *(Characters 400-500 are shared!)*
    - **Chunk 3**: Characters `800` to `1300` *(Characters 800-900 are shared!)*

    **Benefits**:
    - Ensures that facts near the end of Chunk 1 are repeated in Chunk 2.
    - Prevents sudden conceptual truncation.
    """)

with col2:
    st.subheader("The Overlap Goldilocks Rule")
    st.markdown("""
    - **Standard Rule of Thumb**: Set overlap to **10% to 20% of the chunk size** (e.g., 50-token overlap for a 500-token chunk).
    - **Too Little Overlap (< 5%)**: Fails to capture multi-sentence relationships spanning the boundary.
    - **Too Much Overlap (> 35%)**: Inflates total vector index size by 40%+, wasting VRAM and causing near-duplicate search results that crowd out distinct evidence.
    """)

st.divider()

# --- SECTION 8: TOPIC 7 - CHUNK-SIZE OPTIMIZATION ---
st.header("8. Chunk-Size Optimization: Balancing Retrieval Precision vs Context Sufficiency", divider="rainbow")

st.markdown("""
Selecting chunk size is an empirical optimization problem that balances two competing forces:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Small Chunks (128 - 256 Tokens)
    - **High Retrieval Precision**: The embedding vector is razor-sharp; ideal for exact factual lookups (dates, product names, phone numbers).
    - **Low Context Sufficiency**: The LLM may receive fragmented sentences without enough surrounding rationale to answer complex questions.
    """)
with col2:
    st.markdown("""
    ### Large Chunks (1,024 - 2,048 Tokens)
    - **High Context Sufficiency**: Contains full paragraphs, detailed arguments, and entire code functions.
    - **Low Retrieval Precision**: Blends multiple sub-topics into a single average vector; risks 'Lost in the Middle' attention degradation.
    """)

st.markdown("""
### Advanced Architecture: Parent-Document Retrieval (Hierarchical Chunking)
To achieve both razor-sharp retrieval AND complete context, modern enterprise systems decouple retrieval chunks from generation chunks:
1. Split documents into **large parent chunks (1,000 tokens)**.
2. Sub-divide each parent chunk into **small child chunks (128 tokens)**.
3. Embed and search across the child chunks for maximum vector precision.
4. When a child chunk matches the query, **pass its entire 1,000-token parent chunk to the LLM**!
""")

st.divider()

# --- SECTION 9: PRODUCTION CODE ---
st.header("9. Production Implementation: Recursive Character & Semantic Distance Chunkers in Python", divider="rainbow")

st.markdown("""
The following complete, zero-dependency Python script implements the mathematical core of **Fixed-Size Chunking with Overlap**, 
**Recursive Character Text Splitting**, and **Semantic Distance Breakpoint Chunking**:
""")

code_block = r'''import re
import numpy as np
from typing import List, Dict, Any

class ProductionTextSplitter:
    """
    Self-contained chunking engine implementing:
    - Fixed-size sliding window chunking
    - Recursive character text splitting
    - Semantic distance breakpoint chunking
    """
    
    @staticmethod
    def recursive_split(
        text: str,
        chunk_size: int = 400,
        overlap: int = 50,
        separators: List[str] = None
    ) -> List[str]:
        """Splits recursively through paragraphs, lines, sentences, and words."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " "]
            
        def _split(txt: str, seps: List[str]) -> List[str]:
            sep = seps[-1]
            remaining = []
            for i, s in enumerate(seps):
                if s == "":
                    sep = ""
                    break
                if s in txt:
                    sep = s
                    remaining = seps[i+1:]
                    break
                    
            raw_parts = txt.split(sep) if sep else list(txt)
            splits = []
            for p in raw_parts:
                if len(p) <= chunk_size:
                    splits.append(p)
                elif remaining:
                    splits.extend(_split(p, remaining))
                else:
                    splits.append(p)
                    
            # Merge with overlap
            merged = []
            curr = ""
            for p in splits:
                cand = (curr + sep + p) if curr else p
                if len(cand) <= chunk_size:
                    curr = cand
                else:
                    if curr:
                        merged.append(curr)
                    curr = p
            if curr:
                merged.append(curr)
            return merged

        return _split(text, separators)

    @staticmethod
    def fixed_sliding_window(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        """Slices text strictly by character length with sliding overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += (chunk_size - overlap)
        return chunks
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 10: WORKBENCH ---
st.header("10. Interactive Workbench: Real-Time Chunk Boundary Visualizer & Overlap Inspector", divider="rainbow")

st.markdown("""
Test how different chunking strategies segment real multi-paragraph text in real time:
""")

default_sample = (
    "Retrieval-Augmented Generation (RAG) is an architectural framework that enhances Large Language Models by retrieving external knowledge.\n\n"
    "Traditional models suffer from knowledge cutoffs and hallucinations. When queried on private proprietary data, standard foundation models fail.\n\n"
    "Chunking is the critical first step in document preparation. Selecting an appropriate chunk size determines whether the vector search engine retrieves exact factual details or broad thematic noise.\n\n"
    "Recursive character chunking preserves paragraph and sentence boundaries. It falls back through double newlines, single newlines, and sentence punctuation to keep ideas intact."
)

wb_col1, wb_col2 = st.columns(2)
with wb_col1:
    input_doc = st.text_area("Input Document Text", default_sample, height=200)
    strategy_choice = st.selectbox("Chunking Strategy", ["Recursive Character Splitter", "Fixed-Size with Overlap", "Semantic Distance Breakpoints"])

with wb_col2:
    target_size = st.slider("Target Chunk Size (Characters)", min_value=100, max_value=600, value=250, step=25)
    target_overlap = st.slider("Sliding Window Overlap (Characters)", min_value=0, max_value=100, value=40, step=10)

# Run selected chunker
if strategy_choice == "Recursive Character Splitter":
    generated_chunks = ProductionChunker.recursive_character_chunk(input_doc, chunk_size=target_size, overlap=target_overlap)
elif strategy_choice == "Fixed-Size with Overlap":
    generated_chunks = ProductionChunker.fixed_size_chunk(input_doc, chunk_size=target_size, overlap=target_overlap)
else:
    generated_chunks = ProductionChunker.semantic_distance_chunk(input_doc)

st.subheader(f"Generated Chunks ({len(generated_chunks)} total)")
stat_col1, stat_col2 = st.columns(2)
stat_col1.metric("Total Document Length", f"{len(input_doc)} chars")
stat_col2.metric("Average Chunk Length", f"{int(np.mean([c['length'] for c in generated_chunks]))} chars")

for c in generated_chunks:
    with st.container(border=True):
        st.markdown(f"**Chunk ID: {c['id']}** | Length: `{c['length']} chars`")
        st.info(c["text"])
        if "overlap_prev" in c and c["overlap_prev"]:
            st.caption(f"Shared Overlap with Previous Chunk: `{c['overlap_prev']}`")

st.divider()

# --- SECTION 11: INTERVIEW PREP ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "Why does embedding an entire 50-page document as a single vector fail in RAG?",
        "a": "Dense embedding models compress input text into a single fixed-size mathematical coordinate vector (e.g. 1536 dimensions). In transformer embedding models (like text-embedding-3 or BGE), pooling operations average token representations across the sequence. When 50 pages of disparate topics, numbers, and dates are compressed into one vector, individual facts are diluted into a generic semantic mean. High-precision vector similarity search requires narrow, high-density chunks that clearly represent specific ideas."
    },
    {
        "q": "How does Recursive Character Chunking balance paragraph cohesion with token length constraints?",
        "a": "Recursive character chunking operates through an ordered hierarchy of separators: ['\\n\\n', '\\n', '. ', ' ']. It attempts to segment text using the highest-level structural separator first (double newlines for paragraphs). If a paragraph fits within the target chunk size, it is left completely intact. Only if a single paragraph exceeds the chunk limit does the algorithm recursively split that paragraph using the next separator (single newlines for lines, then periods for sentences). This guarantees that paragraphs and sentences are preserved wherever possible while enforcing a strict maximum token ceiling."
    },
    {
        "q": "Explain how Semantic Chunking uses cosine distance spikes between adjacent sentences to find breakpoints.",
        "a": "Semantic chunking (Kamradt algorithm) splits text into sentences, computes an embedding for each sentence, and evaluates the cosine distance between consecutive pairs: d_i = 1 - CosSim(e_i, e_{i+1}). Sentences discussing the same topic exhibit low distance. When the author shifts topics, the semantic distance spikes. By computing a statistical threshold (such as the 90th percentile of adjacent distances), the algorithm identifies transition points and creates chunks of variable size that each represent a single coherent concept."
    },
    {
        "q": "What is the operational purpose of chunk overlap and what is the optimal overlap ratio?",
        "a": "Chunk overlap ensures that semantic relationships and facts spanning the boundary between two adjacent chunks are not lost. Without overlap, a critical sentence split in half across a boundary loses its meaning in both chunks. The recommended rule of thumb is 10% to 20% of the chunk size (e.g. 50 tokens for a 500-token chunk). Overlap below 5% fails to bridge multi-sentence relationships, while overlap above 35% inflates vector storage and causes duplicate chunks to crowd out diverse evidence in top-k retrieval."
    },
    {
        "q": "How does Parent-Document Retrieval resolve the trade-off between small and large chunks?",
        "a": "Parent-document retrieval decouples the retrieval unit from the generation unit. The ingestion pipeline segments text into large parent chunks (e.g. 1,000 tokens) and then sub-divides each parent into small child chunks (e.g. 128 tokens). The system embeds and searches across the small child chunks (achieving razor-sharp vector similarity precision). However, once a child chunk is retrieved, the engine looks up its parent ID and passes the full 1,000-token parent chunk to the LLM's context window, providing full context without vector dilution."
    }
]

for idx, qa in enumerate(qas, 1):
    with st.expander(f"Question {idx}: {qa['q']}"):
        st.write(qa["a"])

st.markdown("""
### Production Traps & Architecture Failure Modes
""")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **Trap 1: Character-Count vs Token-Count Mismatch (Embedding Truncation)**
    Setting `chunk_size = 500` assuming characters equal tokens. In technical text, code, or non-English languages, 
    500 characters can easily expand to 600+ tokens. If the embedding model context limit is 512 tokens, 
    the trailing tokens are silently clipped and never embedded! Always use a real tokenizer (e.g. `tiktoken`) to measure chunk limits.
    """)
    st.warning("""
    **Trap 3: Severing Markdown Tables and Code Blocks Mid-Grid**
    Applying character-based text splitters to Markdown files without table awareness. Splitting a Markdown table across two chunks 
    creates two broken fragments lacking header columns, rendering both chunks useless to the LLM. 
    Always use specialized Markdown or code splitters that keep code blocks and tables atomic.
    """)

with col_trap2:
    st.error("""
    **Trap 2: Excessive Overlap Creating Redundant Top-K Clones**
    Configuring overlap at 50% or higher. When a user executes a search, the top-4 retrieved chunks are often 
    the same overlapping paragraph with minor offsets. The context window is completely filled with redundant duplicate text, 
    blocking other relevant source documents from being seen.
    """)
    st.warning("""
    **Trap 4: Missing Parent Breadcrumb Metadata in Deep Hierarchies**
    Chunking sub-sections without prepending the parent document title and section header. A chunk stating 
    *"The fee is $50"* without the heading *"Section 3.2: Wire Transfers"* cannot be interpreted by the LLM. 
    Always prepend header breadcrumbs to chunk text before embedding.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: Recursive Order"):
        st.markdown("**Hierarchy of separators?**\\n\\n`['\\n\\n', '\\n', '. ', ' ', '']`\\n\\nFalls back through paragraphs, lines, sentences, words.")
with fc2:
    with st.popover("Flashcard 2: Overlap Ratio"):
        st.markdown("**Standard overlap rule?**\\n\\n10% to 20% of chunk size. Prevents boundary severed facts without duplicate bloat.")
with fc3:
    with st.popover("Flashcard 3: Semantic Spikes"):
        st.markdown("**How does Semantic Chunking split?**\\n\\nEmbeds sentences; splits at cosine distance spikes (>90th percentile) between neighbors.")
with fc4:
    with st.popover("Flashcard 4: Parent-Child"):
        st.markdown("**What is Parent-Document Retrieval?**\\n\\nEmbeds small child chunks (128t) for search; passes full parent chunk (1000t) to LLM for context.")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT ---
st.header("12. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why embedding whole documents causes vector dilution and why chunking is required.",
    "I can explain why fixed-size chunking severs sentences and inverts legal or technical meaning.",
    "I understand the hierarchical fallback mechanism of Recursive Character Chunking (['\\n\\n', '\\n', '. ', ' ']).",
    "I can derive the cosine distance spike formula used in Semantic Chunking and explain its trade-offs.",
    "I know how to select an optimal sliding window overlap (10% - 20%) to protect boundary facts.",
    "I understand the Parent-Document Retrieval pattern and how it decouples search precision from context generation."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of document chunking strategies, boundary preservation, and chunk-size optimization.")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
