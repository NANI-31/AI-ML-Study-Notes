import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="17.2 Chunking Strategies - Embeddings Curriculum",
    page_icon="🧩",
    layout="wide"
)

# Title & Overview
st.title("17.2 Chunking Strategies: Fixed, Token, Sentence, Recursive, & Semantic Chunking")
st.caption("Chapter 17: Chunking and Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
How you divide text into chunks is one of the highest-leverage engineering decisions in any RAG or semantic search system.
A crude chunking algorithm splits coherent thoughts in half, producing gibberish vectors. An intelligent chunking algorithm aligns chunk boundaries with natural topical transitions.

In this lesson, you will master the 7 fundamental chunking strategies:
* **Fixed-Size & Character-Based**: Simple slice-by-length splitting.
* **Token-Based**: Tokenizer-aware splitting guaranteeing compliance with model attention budgets.
* **Sentence-Based & Paragraph-Based**: Linguistic boundary splitting preserving complete human ideas.
* **Recursive Chunking**: The hierarchical standard combining paragraph, sentence, and word fallbacks.
* **Semantic Chunking**: Using embedding distance drops between adjacent sentences to detect true topical shifts.
""")

st.divider()

# Section 1: Taxonomy of the 7 Chunking Strategies
st.header("1. Taxonomy & Mechanics of Chunking Strategies", divider="rainbow")

col_tax1, col_tax2 = st.columns([1, 1])

with col_tax1:
    st.markdown("""
    ### The 7 Core Chunking Strategies
    * **1. Fixed-Size / Character-Based**: Slices text strictly every $N$ characters. Fastest, but frequently slices words in half (e.g., `"Transf"` and `"ormer"`).
    * **2. Token-Based**: Slices text every $N$ tokens using the model's exact BPE tokenizer (e.g., `tiktoken` for OpenAI). Prevents silent model truncation.
    * **3. Sentence-Based**: Groups text strictly by complete grammatical sentences. Ensures syntactic integrity.
    * **4. Paragraph-Based**: Splits on double newlines (`\n\n`). Respects the author's intentional conceptual units.
    * **5. Recursive Chunking**: Progressively attempts paragraphs (`\n\n`), then sentences (`\n`), then spaces (` `), and characters (`""`). The industry standard in LangChain and LlamaIndex.
    * **6. Semantic Chunking**: Evaluates the embedding distance between consecutive sentences. When cosine similarity drops below a threshold (a semantic "valley"), a chunk boundary is placed!
    """)

with col_tax2:
    st.markdown("### Semantic Chunking Boundary Detection")
    sem_dot = """
    digraph SemanticChunking {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        S1 [label="Sentence 1: 'S3 provides scalable object storage.'", fillcolor="#EFF6FF"];
        S2 [label="Sentence 2: 'Buckets store unstructured files.'", fillcolor="#EFF6FF"];
        S3 [label="Sentence 3: 'PyTorch trains deep neural networks.'", fillcolor="#FEF3C7"];
        S4 [label="Sentence 4: 'Backpropagation updates model weights.'", fillcolor="#FEF3C7"];

        DistHigh [label="High Similarity (0.88)\\nKeep in Chunk 1", fillcolor="#D1FAE5", color="#10B981"];
        DistDrop [label="Similarity Valley Drop (0.19)\\nTOPICAL BREAKPOINT DETECTED!", fillcolor="#FEE2E2", color="#DC2626", style="filled,bold"];
        DistHigh2 [label="High Similarity (0.91)\\nKeep in Chunk 2", fillcolor="#D1FAE5", color="#10B981"];

        S1 -> S2 [label="Pairwise sim"];
        S2 -> S3 [label="Pairwise sim"];
        S3 -> S4 [label="Pairwise sim"];
        S1 -> DistHigh [style="dotted"];
        S2 -> DistDrop [style="dotted"];
        S3 -> DistHigh2 [style="dotted"];
    }
    """
    st.graphviz_chart(sem_dot, use_container_width=False)

st.markdown("### Comparative Evaluation of Chunking Strategies")
st.markdown("""
| Strategy | Compute Overhead | Respects Grammar | Respects Token Limits | Detects Topic Shifts | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fixed / Character** | Almost zero | Terrible (Cuts words) | Rough estimate | No | Quick prototyping only. |
| **Token-Based** | Very low (Tokenizer) | Moderate | **Exact (100%)** | No | Strict model context compliance. |
| **Sentence-Based** | Low (Regex / NLTK) | **Outstanding** | Variable chunk lengths | No | Linguistic analysis, transcripts. |
| **Paragraph-Based** | Very low | High | Variable (Oversized chunks) | Moderate | Well-structured articles, essays. |
| **Recursive Chunking** | Low | **Outstanding** | **High** | Moderate | **General RAG Production Default**. |
| **Semantic Chunking** | High ($N$ embedding calls) | **Outstanding** | High | **Outstanding** | Unstructured mixed-topic enterprise docs. |
""")

st.divider()

# Section 2: Production Code: Multi-Strategy Chunker & Semantic Splitter
st.header("2. Production Code: Multi-Strategy Engine & Semantic Splitter", divider="rainbow")

st.markdown("""
The production utility below implements both a Recursive Boundary Chunker and a pure-Python Semantic Chunker that detects topical drops using sentence embeddings.
""")

st.code(r'''
import re
import numpy as np
from typing import List, Dict, Any, Tuple

class ProductionChunkingEngine:
    """
    Demonstrates Token-Aware, Recursive, and Semantic Chunking implementations.
    """
    def __init__(self, embedding_dim: int = 64):
        self.dim = embedding_dim

    def mock_embed_sentence(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.lower().strip())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

    def split_sentences(self, text: str) -> List[str]:
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if s.strip()]

    def recursive_chunk(self, text: str, max_chars: int = 200, overlap: int = 30) -> List[str]:
        """Hierarchical recursive splitting simulating LangChain."""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) <= max_chars:
                current = f"{current}\n\n{p}".strip()
            else:
                if current:
                    chunks.append(current)
                current = p
        if current:
            chunks.append(current)
        return chunks

    def semantic_chunk(self, text: str, similarity_threshold: float = 0.50) -> List[Dict[str, Any]]:
        """
        Embeds each sentence, calculates adjacent cosine similarity,
        and splits text at semantic valley breakpoints.
        """
        sentences = self.split_sentences(text)
        if len(sentences) <= 1:
            return [{"chunk_id": "c01", "text": text, "sentences": sentences}]

        # Compute sentence embeddings
        vectors = [self.mock_embed_sentence(s) for s in sentences]

        # Calculate adjacent pairwise similarities
        similarities = []
        for i in range(len(vectors) - 1):
            sim = float(np.dot(vectors[i], vectors[i + 1]))
            similarities.append(sim)

        # Identify breakpoints where similarity < threshold
        chunks = []
        current_chunk_sentences = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < similarity_threshold:
                # Semantic boundary breakpoint!
                chunks.append({
                    "chunk_id": f"chunk_{len(chunks)+1:02d}",
                    "text": " ".join(current_chunk_sentences),
                    "sentence_count": len(current_chunk_sentences),
                    "split_reason": f"Similarity dropped to {sim:.3f} (< {similarity_threshold:.2f})"
                })
                current_chunk_sentences = [sentences[i + 1]]
            else:
                current_chunk_sentences.append(sentences[i + 1])

        if current_chunk_sentences:
            chunks.append({
                "chunk_id": f"chunk_{len(chunks)+1:02d}",
                "text": " ".join(current_chunk_sentences),
                "sentence_count": len(current_chunk_sentences),
                "split_reason": "End of document"
            })

        return chunks
''', language="python")

st.divider()

# Section 3: Interactive Multi-Strategy Chunking Arena
st.header("3. Interactive Workbench: Multi-Strategy Chunking Arena", divider="rainbow")

st.markdown("""
Compare how different chunking strategies partition a mixed-topic enterprise document transitioning between **Cloud Infrastructure**, **Machine Learning Training**, and **Corporate Compliance**.
""")

class InteractiveStrategyArena:
    def __init__(self, dim: int = 48):
        self.dim = dim

    def embed_sentence(self, text: str) -> np.ndarray:
        # Give distinct topics distinct embedding clusters
        t_low = text.lower()
        np.random.seed(abs(hash(t_low)) % (2**31 - 1))
        base = np.random.randn(self.dim).astype(np.float32)
        
        # Add topic bias vector to simulate real transformer clustering
        bias = np.zeros(self.dim, dtype=np.float32)
        if any(w in t_low for w in ["s3", "storage", "cloud", "aws", "bucket"]):
            bias[:12] += 2.5
        elif any(w in t_low for w in ["pytorch", "model", "training", "weights", "neural"]):
            bias[12:24] += 2.5
        elif any(w in t_low for w in ["compliance", "audit", "gdpr", "legal", "security"]):
            bias[24:36] += 2.5
            
        v = base + bias
        return v / np.maximum(np.linalg.norm(v), 1e-12)

arena_engine = InteractiveStrategyArena()

default_multi_topic = (
    "Amazon S3 provides durable object storage designed for eleven nines of data availability. "
    "Cloud storage buckets can be encrypted at rest utilizing AWS Key Management Service keys. "
    "Object lifecycle policies automatically transition cold files into Glacier Flexible Archive.\n\n"
    "PyTorch accelerates deep learning model training using distributed data parallelism across GPUs. "
    "Gradient clipping prevents exploding gradients during backpropagation in deep neural networks. "
    "Learning rate schedulers decay the optimizer step size as validation loss plateaus.\n\n"
    "Corporate compliance policies mandate that all employee access must be logged for annual audits. "
    "Under GDPR regulations, European customer data must be erasable upon formal right-to-be-forgotten requests. "
    "Security incident disclosures must be delivered to regulators within seventy-two hours of discovery."
)

col_strat1, col_strat2 = st.columns([1, 1])

with col_strat1:
    st.subheader("1. Ingestion Document")
    sample_text = st.text_area("Multi-Topic Knowledge Document:", value=default_multi_topic, height=180)

    st.subheader("2. Select Strategy")
    chosen_strategy = st.selectbox(
        "Choose Chunking Algorithm:",
        [
            "1. Fixed Character Splitting (Naive)",
            "2. Sentence-Based Splitting",
            "3. Paragraph-Based Splitting",
            "4. Recursive Character Splitting (LangChain Standard)",
            "5. Semantic Chunking (Embedding Distance Valleys)"
        ],
        index=4
    )

    if "Fixed Character" in chosen_strategy:
        fixed_chars = st.slider("Chunk Character Length:", min_value=100, max_value=350, value=200, step=25)
    elif "Recursive" in chosen_strategy:
        rec_size = st.slider("Max Recursive Chunk Size:", min_value=150, max_value=400, value=250, step=25)
        rec_ovlp = st.slider("Recursive Overlap:", min_value=0, max_value=60, value=30, step=5)
    elif "Semantic" in chosen_strategy:
        sem_threshold = st.slider(
            "Semantic Breakpoint Threshold:",
            min_value=0.20,
            max_value=0.85,
            value=0.55,
            step=0.05,
            help="Pairs with cosine similarity below this threshold trigger a chunk boundary."
        )

with col_strat2:
    st.subheader(f"3. Partitioning Results: {chosen_strategy.split('(')[0]}")

    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', sample_text.strip()) if s.strip()]

    if "Fixed Character" in chosen_strategy:
        chunks = []
        for i in range(0, len(sample_text), fixed_chars):
            chunks.append(sample_text[i : i + fixed_chars])
        
        st.warning("⚠️ Notice words chopped mid-string at arbitrary character offsets!")
        for idx, c in enumerate(chunks, 1):
            with st.expander(f"Chunk {idx:02d} ({len(c)} chars)", expanded=True):
                st.code(c)

    elif "Sentence-Based" in chosen_strategy:
        st.info("Each sentence forms its own discrete chunk.")
        for idx, s in enumerate(sentences, 1):
            with st.expander(f"Chunk {idx:02d} ({len(s.split())} words)", expanded=False):
                st.markdown(f"> {s}")

    elif "Paragraph-Based" in chosen_strategy:
        paragraphs = [p.strip() for p in sample_text.split("\n\n") if p.strip()]
        st.success(f"Split cleanly into **{len(paragraphs)}** topical paragraphs based on double newlines.")
        for idx, p in enumerate(paragraphs, 1):
            with st.expander(f"Chunk {idx:02d} ({len(p.split())} words)", expanded=True):
                st.markdown(f"> {p}")

    elif "Recursive" in chosen_strategy:
        # Simulate recursive splitter
        paras = sample_text.split("\n\n")
        rec_chunks = []
        cur = ""
        for p in paras:
            if len(cur) + len(p) <= rec_size:
                cur = f"{cur}\n\n{p}".strip()
            else:
                if cur:
                    rec_chunks.append(cur)
                cur = p
        if cur:
            rec_chunks.append(cur)
            
        st.success(f"Generated **{len(rec_chunks)}** recursive chunks respecting paragraph hierarchies.")
        for idx, rc in enumerate(rec_chunks, 1):
            with st.expander(f"Chunk {idx:02d} ({len(rc)} chars)", expanded=True):
                st.markdown(f"> {rc}")

    else: # Semantic Chunking
        # Compute adjacent cosine similarities
        vecs = [arena_engine.embed_sentence(s) for s in sentences]
        sim_scores = []
        for i in range(len(vecs) - 1):
            sim = float(np.dot(vecs[i], vecs[i + 1]))
            sim_scores.append(sim)

        # Plot similarity curve
        df_sim = pd.DataFrame({
            "Sentence Transition": [f"S{i+1} -> S{i+2}" for i in range(len(sim_scores))],
            "Cosine Similarity": sim_scores,
            "Threshold": [sem_threshold] * len(sim_scores)
        })
        st.markdown("#### Pairwise Sentence Cosine Similarity Curve")
        st.line_chart(df_sim.set_index("Sentence Transition"))

        # Build semantic chunks
        sem_chunks = []
        cur_sents = [sentences[0]]
        for i, sim in enumerate(sim_scores):
            if sim < sem_threshold:
                sem_chunks.append((" ".join(cur_sents), sim))
                cur_sents = [sentences[i + 1]]
            else:
                cur_sents.append(sentences[i + 1])
        if cur_sents:
            sem_chunks.append((" ".join(cur_sents), 1.0))

        st.success(f"Semantic Chunking detected **{len(sem_chunks)}** coherent topical chunks!")
        for idx, (c_text, split_val) in enumerate(sem_chunks, 1):
            with st.expander(f"Topical Chunk {idx:02d} ({len(c_text.split())} words)", expanded=True):
                st.markdown(f"> {c_text}")
                if split_val < sem_threshold:
                    st.caption(f"Breakpoint triggered: Sentence transition similarity dropped to {split_val:.3f}")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Chunking Strategy Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How does Semantic Chunking work algorithmically and what is its primary advantage?**
    * **Answer**:
      * 1. The document is first split into individual sentences using a grammatical tokenizer.
      * 2. An embedding model generates a vector for every sentence.
      * 3. The algorithm calculates the pairwise cosine distance between adjacent sentences: `Distance(S_i, S_{i+1})`.
      * 4. Breakpoints are placed wherever distance exceeds a statistical threshold (or similarity drops into a local valley).
      * **Advantage**: It does not rely on arbitrary character counts or whitespace formatting; chunks represent true cohesive semantic thoughts, isolating topic changes regardless of paragraph formatting.

    **Q2: What is the computational downside of Semantic Chunking in high-volume production pipelines?**
    * **Answer**: Semantic Chunking requires calling the embedding model for **every single sentence** in the document during the ingestion phase. For a 100,000-page corpus with 2 million sentences, this requires 2,000,000 embedding inference calls just to determine where to cut chunks, vastly increasing ingestion latency and API costs compared to instant regex-based Recursive Character Chunking.

    **Q3: Why is token-based chunking mandatory when interfacing with models like OpenAI `text-embedding-3`?**
    * **Answer**: Characters do not map 1:1 to tokens. In English prose, 1 token is roughly 4 characters; however, in code, JSON payloads, or non-Latin languages, 1 character can consume 1 to 3 tokens. A character splitter configured for 1,000 characters could easily generate a 1,500-token chunk that exceeds an embedding model's context window, causing silent truncation or API rejections.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Naive Fixed-Character Chunking in Production**:
      * *Issue*: Slicing text strictly every 500 characters.
      * *Consequence*: Slices words and numbers in half (e.g. `$100,000` becomes `$10` in chunk 1 and `0,000` in chunk 2), destroying searchable financial entities.
      * *Fix*: Always use token-aware or sentence-aware boundary splitters.

    * **Trap 2: Over-Splitting with Pure Sentence Chunking**:
      * *Issue*: Treating every single sentence as an individual chunk.
      * *Consequence*: A vector index filled with fragments like `"However, this was not the case."` or `"She agreed."` that contain zero self-contained context without their antecedents.
      * *Fix*: Group sentences into semantic paragraphs (200-400 tokens).

    * **Trap 3: Semantic Chunking Threshold Sensitivity**:
      * *Issue*: Hardcoding a single threshold (e.g. 0.70) across different document genres.
      * *Consequence*: In highly homogeneous legal texts, zero splits occur (one giant chunk); in fast-paced dialogue, every sentence becomes a separate chunk.
      * *Fix*: Use dynamic percentiles (e.g. split on the 90th percentile of distance drops within the document) rather than a rigid static float.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Semantic Chunking**
      * *Definition*: An embedding-driven splitting technique that detects topical transition valleys by monitoring cosine distance drops between consecutive sentences.
    * **Flashcard 2: Recursive Character Splitter**
      * *Definition*: The default production splitter that iteratively checks separators in order: `["\n\n", "\n", " ", ""]` to keep related text together.
    * **Flashcard 3: Token-Aware Splitting**
      * *Definition*: Measuring text length using the target model's Byte Pair Encoding (BPE) tokenizer to ensure exact compliance with token budgets.
    * **Flashcard 4: Semantic Valley**
      * *Definition*: A local minimum in adjacent sentence similarity indicating a topical shift where a chunk boundary should be inserted.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Chunking Strategies:

- [ ] I can describe the mechanics and tradeoffs of all 7 chunking strategies.
- [ ] I understand how Semantic Chunking uses adjacent sentence distance curves.
- [ ] I know why token-aware chunking is required for strict model context window compliance.
- [ ] I understand why Recursive Chunking is the general production industry standard.
- [ ] I can articulate the latency and cost tradeoff of Semantic Chunking vs Regex splitters.
""")
