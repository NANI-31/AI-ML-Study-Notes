import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="4.3 Document Embeddings - Embeddings Curriculum",
    page_icon="📄",
    layout="wide"
)

# Title & Overview
st.title("4.3 Document Embeddings: Paragraph Vectors, Doc2Vec, Hierarchical Pooling, & Long-Context Encoders")
st.caption("Chapter 4: Types of Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
Representing a single sentence with a 768-dimensional vector is straightforward; **representing a 20-page legal contract, medical record, or financial report with a single vector is a major engineering challenge**.

Long documents contain multiple disparate subtopics, section transitions, tables, and fine-grained facts. 
If an entire document is compressed into a single static point, **specific details are diluted and lost ("Topic Dilution")**.

To address long-document representation, production systems employ four primary paradigms:
1. **Doc2Vec (Distributed Memory & DBOW)**: Historical unsupervised paragraph vector models learning document ID embeddings.
2. **Hierarchical Chunk-and-Pool**: Slicing documents into semantically coherent passages and pooling their vectors.
3. **Parent-Document Indexing**: Indexing granular 256-token passage vectors for high-precision retrieval, but returning the full parent document to the LLM.
4. **Native Long-Context Encoders (8k+ Tokens)**: Modern models (e.g. Jina-embeddings-v2, BGE-M3, Voyage-large-2) utilizing RoPE and linear attention to ingest full multi-page documents in a single forward pass.

This lesson explores the evolution from Doc2Vec to modern multi-vector document architectures.
""")

st.divider()

# Section 1: The Long-Document Architecture Spectrum
st.header("1. Architectural Spectrum: Single-Vector vs. Multi-Vector Representation", divider="rainbow")

col_doc1, col_doc2 = st.columns([1, 1])

with col_doc1:
    st.markdown("""
    ### Engineering Approaches to Long Text:
    * **1. The Naive Truncation Failure**:
      Standard BERT/MiniLM models accept a maximum of 512 tokens. Passing a 3,000-word document silently truncates 80% of the content.
    * **2. Doc2Vec (Le & Mikolov, 2014)**:
      * *PV-DM (Distributed Memory)*: Treats the document ID as an extra context word, predicting the next word.
      * *PV-DBOW (Distributed Bag-of-Words)*: Trains the document vector to predict words randomly sampled from the document.
    * **3. Hierarchical Chunk-and-Pool**:
      Chunks C1, C2, ..., CK are embedded independently, then merged via weighted mean or max pooling:
      `v_doc = weighted_mean(v_C1, v_C2, ..., v_CK)`.
    * **4. Late Chunking (Jina AI, 2024)**:
      Feeds the entire 8k-token document through the transformer so token representations attend to the full document context *before* pooling into passage vectors!
    """)

with col_doc2:
    st.markdown("### Document Representation Strategies")
    doc_dot = """
    digraph DocStrategies {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        DocIn [label="Long Document (3,000 Words)\\nIntro | Architecture | Benchmarks | Security", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        subgraph cluster_pool {
            label="Strategy A: Chunk-and-Pool";
            style="dashed"; color="#D97706";
            Chunk1 [label="Embed Chunk 1 (Intro)", fillcolor="#FEF3C7"];
            Chunk2 [label="Embed Chunk 2 (Arch)", fillcolor="#FEF3C7"];
            Chunk3 [label="Embed Chunk 3 (Bench)", fillcolor="#FEF3C7"];
            Chunk4 [label="Embed Chunk 4 (Sec)", fillcolor="#FEF3C7"];
            PoolVec [label="Pooled Document Vector\\n(Averaged Representation)", fillcolor="#F59E0B", fontcolor="white", style="filled,bold"];
            Chunk1 -> PoolVec; Chunk2 -> PoolVec; Chunk3 -> PoolVec; Chunk4 -> PoolVec;
        }

        subgraph cluster_parent {
            label="Strategy B: Parent-Doc Indexing (Recommended)";
            style="dashed"; color="#10B981";
            GranularIdx [label="Index Granular Passages in VDB\\n(Search matches specific facts)", fillcolor="#DCFCE7", style="filled,bold"];
            ReturnParent [label="Return Complete Parent Document\\nto LLM Generator Context", fillcolor="#22C55E", fontcolor="white", style="filled,bold"];
            GranularIdx -> ReturnParent;
        }

        DocIn -> Chunk1;
        DocIn -> GranularIdx;
    }
    """
    st.graphviz_chart(doc_dot, use_container_width=False)

st.divider()

# Section 2: Production Code Implementation
st.header("2. Complete Production Document Embedding Pipeline in Python", divider="rainbow")
st.markdown("""
The following Python class demonstrates document chunking, hierarchical pooling, and passage-level semantic search.
""")

class ProductionDocumentEmbeddingPipeline:
    """
    Simulates long-document representation:
    Chunking, hierarchical pooling, and granular passage-level retrieval.
    """
    def __init__(self, dimensions: int = 6):
        self.dimensions = dimensions
        # Pre-set topic anchor vectors
        np.random.seed(42)
        self.topic_vectors = {
            "executive_summary": np.array([0.90, 0.40, 0.10, 0.15, 0.30, 0.20]),
            "technical_architecture": np.array([0.15, 0.92, 0.85, 0.10, 0.20, 0.35]),
            "financial_cost": np.array([0.10, 0.20, 0.95, 0.88, 0.15, 0.12]),
            "security_compliance": np.array([0.25, 0.15, 0.10, 0.95, 0.80, 0.10])
        }
        for k in self.topic_vectors:
            self.topic_vectors[k] /= np.linalg.norm(self.topic_vectors[k])

    def chunk_document(self, document_sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Splits document into section chunks and generates passage embeddings."""
        chunks = []
        for sec_name, content in document_sections.items():
            base_vec = self.topic_vectors.get(sec_name, np.random.uniform(-0.3, 0.3, self.dimensions))
            norm_vec = base_vec / np.linalg.norm(base_vec)
            chunks.append({
                "section": sec_name,
                "content": content,
                "vector": norm_vec
            })
        return chunks

    def hierarchical_pool(self, chunks: List[Dict[str, Any]], method: str = "Mean") -> np.ndarray:
        """Combines passage vectors into a single document-level vector."""
        vectors = np.array([c["vector"] for c in chunks])
        if method == "Mean":
            pooled = np.mean(vectors, axis=0)
        else: # Max Pooling
            pooled = np.max(vectors, axis=0)
        norm = np.linalg.norm(pooled)
        return pooled / norm if norm > 0 else pooled

    def search_document(self, query_vec: np.ndarray, chunks: List[Dict[str, Any]], pooled_doc_vec: np.ndarray) -> Dict[str, Any]:
        """
        Compares query similarity against the single pooled document vector
        versus granular passage retrieval.
        """
        q_norm = query_vec / np.linalg.norm(query_vec)
        
        # 1. Similarity to pooled document vector
        doc_level_sim = float(np.dot(q_norm, pooled_doc_vec))

        # 2. Granular passage similarities
        passage_scores = []
        for c in chunks:
            sim = float(np.dot(q_norm, c["vector"]))
            passage_scores.append({"section": c["section"], "similarity": round(sim, 4)})

        passage_scores.sort(key=lambda x: x["similarity"], reverse=True)
        return {
            "pooled_doc_similarity": round(doc_level_sim, 4),
            "top_passage_section": passage_scores[0]["section"],
            "top_passage_similarity": passage_scores[0]["similarity"],
            "all_passages": passage_scores
        }

code_snippet = r'''
import numpy as np

class DocumentEmbeddingHandler:
    def chunk_and_pool(self, document_chunks: list, encoder) -> np.ndarray:
        # 1. Embed each chunk independently
        chunk_embeddings = encoder.encode(document_chunks) # (Num_Chunks, Dim)
        
        # 2. Hierarchical mean pooling across chunks
        doc_vector = np.mean(chunk_embeddings, axis=0)
        
        # 3. L2 re-normalization
        norm = np.linalg.norm(doc_vector)
        return doc_vector / norm if norm > 0 else doc_vector

    def parent_document_retrieval(self, query: str, vector_db):
        # Search against granular chunk index
        matching_chunks = vector_db.search(query, top_k=3)
        
        # Return unique parent document IDs
        parent_doc_ids = list(set(chunk.metadata["parent_id"] for chunk in matching_chunks))
        return parent_doc_ids
'''
st.code(code_snippet, language="python")

st.divider()

# Section 3: Interactive Document Representation Workbench
st.header("3. Interactive Document Pooling vs. Granular Retrieval Workbench", divider="rainbow")
st.markdown("""
Explore what happens when a **4-section enterprise document** is compressed into a single pooled vector versus indexed as granular passages.
Notice how searching for specific technical or security questions yields **high similarity on individual passages, but weak similarity against the diluted pooled vector**!
""")

doc_engine = ProductionDocumentEmbeddingPipeline(dimensions=6)

sample_doc_sections = {
    "executive_summary": "Q3 Strategic Overview: Accelerated cloud adoption and enterprise expansion across global territories.",
    "technical_architecture": "Infrastructure Specification: Microservices cluster running on Kubernetes with distributed Redis caching and Kafka ingestion.",
    "financial_cost": "Budget Analysis: Cloud infrastructure expenditure totaled $420,000 for the quarter, reflecting a 12% reduction in compute overhead.",
    "security_compliance": "Compliance Audit: Successfully completed SOC2 Type II certification and achieved HIPAA data governance alignment."
}

doc_chunks = doc_engine.chunk_document(sample_doc_sections)

col_p_meth, col_p_stat = st.columns([1, 1])
with col_p_meth:
    pooling_choice = st.radio("Document Pooling Method:", ["Mean Pooling", "Max Pooling"])
with col_p_stat:
    pooled_vector = doc_engine.hierarchical_pool(doc_chunks, method="Mean" if "Mean" in pooling_choice else "Max")
    st.info(f"Generated single {len(pooled_vector)}-dimensional document vector representing all 4 sections.")

st.markdown("### Test Targeted Retrieval Queries")

queries = {
    "Specific Security Inquiry: 'What certifications and compliance audits were achieved?'": doc_engine.topic_vectors["security_compliance"],
    "Specific Financial Inquiry: 'What was our total cloud infrastructure spend?'": doc_engine.topic_vectors["financial_cost"],
    "Specific Technical Inquiry: 'Which distributed messaging queues and caches are deployed?'": doc_engine.topic_vectors["technical_architecture"],
    "General High-Level Inquiry: 'Give me a broad overview of corporate quarterly performance'": doc_engine.topic_vectors["executive_summary"]
}

selected_query_label = st.selectbox("Select user inquiry targeting a specific section:", list(queries.keys()))
active_query_vec = queries[selected_query_label]

search_results = doc_engine.search_document(active_query_vec, doc_chunks, pooled_vector)

st.markdown("### Comparison: Single Pooled Vector vs. Granular Passage Retrieval")

col_cmp1, col_cmp2 = st.columns([1, 1])

with col_cmp1:
    st.markdown("#### 1. Single Pooled Document Vector")
    st.metric(
        "Document Cosine Similarity",
        f"{search_results['pooled_doc_similarity']:.4f}",
        delta="Topic Dilution Effect" if search_results['pooled_doc_similarity'] < 0.70 else "Balanced Topic Match",
        delta_color="inverse" if search_results['pooled_doc_similarity'] < 0.70 else "normal"
    )
    st.caption("Because the document vector averages across all 4 sections, specific security or financial details are diluted by unrelated sections.")

with col_cmp2:
    st.markdown("#### 2. Granular Passage-Level Indexing")
    st.metric(
        f"Top Passage Match ('{search_results['top_passage_section']}')",
        f"{search_results['top_passage_similarity']:.4f}",
        delta="Exact Needle Match"
    )
    st.caption("Granular passage indexing matches the exact relevant section with extreme precision!")

st.markdown("##### Detailed Breakdown Across All Document Sections:")
breakdown_df = pd.DataFrame([
    {"Section": p["section"].replace("_", " ").title(), "Cosine Similarity": f"{p['similarity']:.4f}", "Match Quality": "Strong Target Match" if p["similarity"] > 0.8 else "Low (Different Section)"}
    for p in search_results["all_passages"]
])
st.dataframe(breakdown_df, use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Document Embedding Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is 'Topic Dilution' and why does pooling long documents into a single vector degrade retrieval?**
    * **Answer**:
      * A 20-page document rarely discusses one single concept; it contains distinct discussions on security, pricing, engineering, and legal terms.
      * When chunk vectors are averaged into a single document vector, the disparate topics cancel each other out, shifting the vector toward a generic, non-specific centroid.
      * When a user searches for a specific technical needle (*"What is the Kafka replication factor?"*), the query vector is far from the diluted document vector, causing retrieval to fail.

    **Q2: What is Parent-Document Retrieval and why is it superior to both single-vector pooling and small-chunk retrieval?**
    * **Answer**:
      * **The Dilemma**: Small chunks (128 tokens) are ideal for vector search precision, but lack sufficient context for the LLM to generate comprehensive answers. Large chunks (1024 tokens) have high context, but poor retrieval precision.
      * **The Parent-Document Solution**:
        1. Split the document into small child chunks (e.g. 200 tokens) for embedding and vector indexing.
        2. In chunk metadata, attach the `parent_id` linking to the full document or larger 1000-token section.
        3. Search matches the small child chunk, but the system **returns the full parent section to the LLM prompt**.

    **Q3: How does ColBERT (Contextualized Late Interaction) represent documents differently from single-vector models?**
    * **Answer**:
      * In single-vector models, an entire document is reduced to one 768D vector.
      * In **ColBERT**, every single token in the document retains its own 128D embedding (a **Multi-Vector representation**).
      * At query time, ColBERT computes the **MaxSim** operator: for each query token, find the maximum cosine similarity among all document tokens, and sum them. This preserves granular token-level interactions without the computational cost of a cross-encoder.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Compressing an Entire 50-Page Book into a Single 768-Dim Vector**:
      * *Issue*: Attempting to average or pool hundreds of paragraphs into one vector.
      * *Consequence*: Total semantic blur; the vector matches everything weakly and nothing strongly.
      * *Fix*: Always use passage-level chunking and parent-document retrieval.

    * **Trap 2: Ignoring Section Headers in Passage Embeddings**:
      * *Issue*: Splitting raw paragraphs without prepending the document title or section heading.
      * *Consequence*: A chunk saying *"It costs $50/month"* loses the context of which product is being priced.
      * *Fix*: Prepend document and section metadata: `[Doc: Cloud Pricing | Section: Starter Plan] It costs $50/month`.

    * **Trap 3: Slicing Text Mid-Sentence**:
      * *Issue*: Using fixed character splitting (`text[:1000]`) that cuts words in half.
      * *Consequence*: Produces broken subword tokens and degrades embedding quality.
      * *Fix*: Use recursive or sentence-aware text splitters.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Topic Dilution**
      * *Definition*: The degradation of retrieval accuracy when a long, multi-topic document is averaged into a single vector, washing out specific factual details.
    * **Flashcard 2: Parent-Document Retrieval**
      * *Definition*: Indexing small child passages for high-precision vector search while retrieving the full parent document for LLM context generation.
    * **Flashcard 3: Doc2Vec (PV-DBOW)**
      * *Definition*: An early unsupervised algorithm that learns a dedicated document vector by predicting words sampled randomly from the document.
    * **Flashcard 4: Late Chunking**
      * *Definition*: Feeding full documents through a long-context transformer before pooling chunk token spans to retain global cross-chunk attention.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Document Embeddings:

- [ ] I understand why compressing long documents into a single vector causes Topic Dilution.
- [ ] I can describe the mechanics of historical Doc2Vec (PV-DM vs PV-DBOW).
- [ ] I understand how Parent-Document Retrieval resolves the trade-off between search precision and LLM context.
- [ ] I know how Late Chunking leverages long-context transformer attention prior to passage pooling.
- [ ] I understand ColBERT's multi-vector token representation and MaxSim operator.
""")
