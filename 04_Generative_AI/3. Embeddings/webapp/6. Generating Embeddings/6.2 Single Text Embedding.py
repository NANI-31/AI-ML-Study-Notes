import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="6.2 Single Text Embedding - Embeddings Curriculum",
    page_icon="📄",
    layout="wide"
)

# Title & Overview
st.title("6.2 Single Text Embedding: Granularity Hierarchy, Sequence Saturation, & Context Limits")
st.caption("Chapter 6: Generating Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
When generating embeddings for single text units, one fundamental question determines retrieval success: **At what granularity should we embed?**
A single text unit can range from a short query (3 words), to an atomic sentence (15 words), to a cohesive paragraph (150 words), to a comprehensive multi-page document (3,000 words).

As text length scales from a single sentence to a full document, three critical engineering phenomena emerge:
1. **Semantic Dilution (Topic Smearing)**: A single sentence holds sharp, specific intent. A 10-page document discusses 20 topics; compressing it into a single 768-d vector smears all topics into an ambiguous average.
2. **Context Window Clipping**: Models with fixed context windows (e.g. 512 tokens for MiniLM/BGE) silently truncate excess tokens, completely ignoring the end of the text!
3. **Quadratic Attention & Information Saturation**: Transformer self-attention scales quadratically with sequence length, and standard pooling layers struggle to represent multi-clause dependencies equally.

This lesson analyzes single text embedding across the granularity hierarchy: sentence vs. paragraph vs. document.
""")

st.divider()

# Section 1: The Text Granularity Hierarchy
st.header("1. The Text Granularity Hierarchy: Sentence vs. Paragraph vs. Document", divider="rainbow")

col_gran1, col_gran2 = st.columns([1, 1])

with col_gran1:
    st.markdown("""
    ### Comparing Semantic Granularities:
    * **1. Sentence-Level Embedding (10 - 30 tokens)**:
      * **Characteristics**: Atomic semantic proposition. High precision, sharp directional vector.
      * **Best For**: Paraphrase identification, intent classification, fine-grained fact lookup.
      * **Limitation**: Lacks surrounding context; pronouns (*"It was discovered in 1928"*) lack resolution.
    * **2. Paragraph-Level Embedding (100 - 300 tokens)**:
      * **Characteristics**: Cohesive topical block containing context, explanation, and entity resolution.
      * **Best For**: Standard RAG passage retrieval, knowledge base FAQs, documentation search.
      * **Sweet Spot**: Fits cleanly within standard 512-token context windows without truncation.
    * **3. Document-Level Embedding (1,000 - 10,000+ tokens)**:
      * **Characteristics**: Macro-level overview or theme.
      * **Best For**: Document clustering, high-level genre classification, library categorization.
      * **Hazard**: Fails to retrieve specific needles in a haystack. A user asking *"What is the timeout parameter on page 42?"* will get near-zero cosine similarity with a whole-document vector.
    """)

with col_gran2:
    st.markdown("### Granularity Spectrum vs. Information Sharpness")
    gran_dot = """
    digraph Granularity {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        Sentence [label="Sentence (15 Tokens)\\n'Penicillin was discovered by Alexander Fleming.'\\nSharp Precision | High Vector Specificity", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Paragraph [label="Paragraph (150 Tokens)\\nHistorical narrative of discovery + laboratory notes.\\nBalanced Context & Specificity (RAG Sweet Spot)", fillcolor="#ECFDF5", color="#10B981", style="filled,bold"];
        
        Document [label="Full Textbook / Paper (4,000 Tokens)\\nHistory of 20th Century Antibiotics\\nHigh Topic Dilution | Low Fine-Grained Needle Match", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

        Sentence -> Paragraph [label="Broadens context"];
        Paragraph -> Document [label="Dilutes specific needles"];
    }
    """
    st.graphviz_chart(gran_dot, use_container_width=False)

st.markdown("### Tradeoff Matrix: Granularity Comparison")
st.markdown("""
| Granularity | Typical Length | Token Count | Information Sharpness | Context Completeness | Truncation Risk | Primary Application |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentence** | 1 - 2 lines | 10 - 40 | **Ultra-High** | Low (Missing referents) | Zero | Semantic similarity, STS |
| **Paragraph** | 4 - 8 lines | 80 - 250 | **High (Optimal)** | **High (Self-contained)** | Low | **Production RAG Chunks** |
| **Section** | 1 - 2 pages | 400 - 1,000 | Moderate | Very High | High (>512 cutoff) | Long-context RAG (ModernBERT) |
| **Full Document** | 5 - 50 pages | 2,000 - 30,000 | **Low (Diluted)** | Complete | Extreme | Topic clustering, genre tagging |
""")

st.divider()

# Section 2: Production Code: Hierarchical Text Embedder
st.header("2. Production Implementation: Multi-Granularity Text Encoder", divider="rainbow")

st.markdown("""
When processing a single document, a production system can either embed it directly (if short), truncate it, or construct a **hierarchical representation** by chunking into paragraphs and pooling the paragraph vectors.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Optional

class MultiGranularityEmbedder:
    """
    Demonstrates embedding generation across single sentences, paragraphs,
    and whole documents with context-length checking and hierarchical pooling.
    """
    def __init__(self, max_seq_length: int = 512, embedding_dim: int = 384):
        self.max_seq_length = max_seq_length
        self.dim = embedding_dim

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norm, 1e-12)

    def embed_sentence(self, sentence: str) -> np.ndarray:
        """Embeds a single atomic sentence."""
        # In production: return model.encode(sentence, normalize_embeddings=True)
        rng = np.random.default_rng(abs(hash(sentence)) % (2**31))
        vec = rng.standard_normal(self.dim)
        return self._normalize(vec)

    def embed_document_hierarchical(self, paragraphs: List[str]) -> Dict[str, Any]:
        """
        Embeds each paragraph individually, then computes both:
        1. Paragraph-level vectors (for granular passage search)
        2. Mean-pooled document vector (for macro topic matching)
        """
        para_vectors = [self.embed_sentence(p) for p in paragraphs]
        stacked = np.array(para_vectors)
        
        # Macro document vector: mean across all paragraph vectors + re-normalize
        doc_macro_vector = self._normalize(np.mean(stacked, axis=0))
        
        return {
            "paragraph_count": len(paragraphs),
            "paragraph_vectors": stacked,
            "document_macro_vector": doc_macro_vector
        }
''', language="python")

# Actual Python implementation for the interactive workbench
class GranularitySimulator:
    def __init__(self):
        # 6 semantic coordinates: [Biomedical, Computing/AI, History, Law, Astronomy, Culinary]
        self.categories = ["Biomedical", "AI/Computing", "History", "Legal", "Astronomy", "Culinary"]
        
        # Benchmark texts across 3 granularities:
        self.sample_data = {
            "Sentence": {
                "text": "CRISPR-Cas9 enables targeted genome editing in living organisms with precise enzymatic cuts.",
                "vector": np.array([0.96, 0.08, 0.02, 0.01, 0.01, 0.01]),
                "token_count": 18
            },
            "Paragraph": {
                "text": "CRISPR-Cas9 revolutionized molecular biology by providing a modular RNA-guided endonuclease system. Discovered as a bacterial adaptive immune mechanism against bacteriophages, it was adapted for eukaryotic genome engineering in 2012 by Doudna and Charpentier, sparking extensive clinical trials for monogenic hereditary disorders.",
                "vector": np.array([0.88, 0.15, 0.35, 0.05, 0.01, 0.01]),
                "token_count": 52
            },
            "Full Multi-Topic Document": {
                "text": "The 21st Century Scientific Revolution spans biomedical genetics, computational intelligence, and space exploration. While CRISPR tools allow clinical gene therapeutics, neural network models automate legal contract analysis and autonomous vehicles. Concurrently, space agencies utilize robotic rovers to analyze Martian mineralogy, while agricultural gene editing enhances crop yields for culinary security.",
                "vector": np.array([0.48, 0.45, 0.30, 0.35, 0.42, 0.38]),
                "token_count": 145
            }
        }

    def evaluate_query_match(self, specific_query_vector: np.ndarray):
        results = []
        norm_q = specific_query_vector / np.linalg.norm(specific_query_vector)
        for level, item in self.sample_data.items():
            norm_v = item["vector"] / np.linalg.norm(item["vector"])
            score = float(np.dot(norm_q, norm_v))
            results.append({
                "Granularity Level": level,
                "Token Length": item["token_count"],
                "Cosine Similarity with Query": score,
                "Preview": item["text"][:80] + "..."
            })
        return results

st.divider()

# Section 3: Interactive Workbench: Granularity & Semantic Dilution Simulator
st.header("3. Interactive Workbench: Granularity & Semantic Dilution Simulator", divider="rainbow")

st.markdown("""
Observe how embedding text at different granularities alters cosine similarity with a specific user query.
Notice how a **targeted factual query** achieves maximum similarity with the single sentence, while the full document suffers from **semantic dilution**!
""")

gran_sim = GranularitySimulator()

col_bench_l, col_bench_r = st.columns([1, 1])

with col_bench_l:
    st.subheader("1. Configure User Search Query")
    query_choice = st.selectbox(
        "Select Specific Needle Query:",
        [
            "How does CRISPR-Cas9 perform enzymatic genome cuts?",
            "What clinical trials use RNA-guided endonuclease therapeutics?",
            "Autonomous neural network vehicles and Mars rovers"
        ]
    )

    query_map = {
        "How does CRISPR-Cas9 perform enzymatic genome cuts?": np.array([0.98, 0.05, 0.01, 0.01, 0.01, 0.01]),
        "What clinical trials use RNA-guided endonuclease therapeutics?": np.array([0.92, 0.10, 0.20, 0.02, 0.01, 0.01]),
        "Autonomous neural network vehicles and Mars rovers": np.array([0.02, 0.70, 0.05, 0.02, 0.65, 0.01])
    }
    
    active_q_vec = query_map[query_choice]

    st.markdown("#### Text Granularity Previews")
    for lvl, data in gran_sim.sample_data.items():
        with st.expander(f"{lvl} ({data['token_count']} tokens)"):
            st.write(data["text"])

with col_bench_r:
    st.subheader("2. Retrieval Match Across Granularities")
    res_list = gran_sim.evaluate_query_match(active_q_vec)
    df_res = pd.DataFrame(res_list)
    
    st.dataframe(
        df_res.style.format({"Cosine Similarity with Query": "{:.4f}"}).background_gradient(subset=["Cosine Similarity with Query"], cmap="Blues"),
        use_container_width=True
    )

    top_item = max(res_list, key=lambda x: x["Cosine Similarity with Query"])
    bottom_item = min(res_list, key=lambda x: x["Cosine Similarity with Query"])
    
    st.info(f"🎯 **Best Match**: **{top_item['Granularity Level']}** (Score: {top_item['Cosine Similarity with Query']:.4f})")
    
    dilution_delta = ((top_item['Cosine Similarity with Query'] - bottom_item['Cosine Similarity with Query']) / top_item['Cosine Similarity with Query']) * 100
    st.warning(f"📉 **Topic Dilution Penalty**: The lowest match experienced a **{dilution_delta:.1f}% drop** in cosine similarity because its vector coordinates were diluted across multiple competing topics!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Granularity Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is 'Topic Dilution' (or semantic smearing) in long-document embeddings?**
    * **Answer**:
      * A dense vector has a fixed capacity (e.g. 768 floating-point coordinates).
      * When a single sentence is embedded, the vector dedicates its geometric capacity to that single concept.
      * When a 5,000-word document covering finance, technology, legal disclaimers, and human resources is averaged into the same 768 coordinates, the vector represents a muddy centroid of all topics.
      * Consequently, a specific query asking for a factual detail will have low cosine similarity with the document vector because the document vector is pulled toward the centroid of unrelated sections.

    **Q2: What is the difference between Sentence-Level, Paragraph-Level, and Document-Level embedding in production RAG?**
    * **Answer**:
      * **Sentence-Level**: Best for high-precision exact matching, but chunks lack context (anaphora, missing subject pronouns).
      * **Paragraph-Level (150-300 tokens)**: The gold standard for RAG. Self-contained, retains contextual referents, fits inside standard embedding windows, and preserves high semantic sharpness.
      * **Document-Level**: Useful only for hierarchical routing (deciding which domain index to query) or coarse document clustering, but inadequate for direct LLM context injection.

    **Q3: How does Late Chunking resolve the trade-off between sentence precision and document context?**
    * **Answer**:
      * In traditional chunking, you split text first, then embed each chunk in isolation (losing global context).
      * In **Late Chunking**, you pass the entire document through a long-context transformer backbone (e.g. 8k tokens) so every token attends to the full document.
      * Then, you pool token vectors corresponding only to each specific sentence/paragraph span.
      * Each chunk embedding retains sharp chunk boundaries while carrying contextual knowledge from the entire document!
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Silent End-of-Document Truncation**:
      * *Issue*: Passing a 1,200-word document into a model with `max_seq_length = 512`.
      * *Consequence*: The tokenizer silently truncates at token 512; everything in the second half of the document is completely discarded from the index with zero warning!
      * *Fix*: Implement token count checks and text chunking before passing strings to the model.

    * **Trap 2: Embedding Tiny 3-Word Sentence Fragments**:
      * *Issue*: Splitting text aggressively on every period, producing fragments like `"However."` or `"See Figure 1."`.
      * *Consequence*: Floods the vector database with useless vectors that match generic transition queries.
      * *Fix*: Enforce a minimum chunk length threshold (e.g. `min_tokens = 30`).

    * **Trap 3: Using Naive Document Centroids as Passage Proxies**:
      * *Issue*: Averaging all chunk vectors to represent an entire chapter in a vector index.
      * *Consequence*: Destroys fine-grained search capability for sub-sections.
      * *Fix*: Store chunk vectors individually and link them to parent document IDs.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Semantic Dilution**
      * *Definition*: The loss of fine-grained conceptual specificity that occurs when diverse multi-topic paragraphs are compressed into a single fixed-dimension vector.
    * **Flashcard 2: Paragraph Chunking Sweet Spot**
      * *Definition*: Text units of 100-300 tokens that provide optimal balance between self-contained contextual coherence and high vector specificity.
    * **Flashcard 3: Silent Truncation**
      * *Definition*: When an embedding tokenizer discards tokens exceeding the model's maximum sequence length without raising an exception.
    * **Flashcard 4: Hierarchical Embedding**
      * *Definition*: Embedding granular chunks for retrieval while maintaining a macro document centroid for high-level clustering or index routing.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Single Text Embedding:

- [ ] I understand the tradeoffs between sentence, paragraph, and whole-document embeddings.
- [ ] I can explain why paragraph-level chunks (100-300 tokens) represent the sweet spot for production RAG.
- [ ] I know the cause and consequence of Topic Dilution in long texts.
- [ ] I can identify the hazard of silent sequence length truncation in transformer tokenizers.
- [ ] I understand how hierarchical and Late Chunking approaches bridge the gap between global context and granular precision.
""")
