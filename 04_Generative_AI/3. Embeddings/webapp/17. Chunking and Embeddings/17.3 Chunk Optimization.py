import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="17.3 Chunk Optimization - Embeddings Curriculum",
    page_icon="⚡",
    layout="wide"
)

# Title & Overview
st.title("17.3 Chunk Optimization: Small-to-Big Retrieval, Metadata Enrichment, & Parent-Child Hierarchies")
st.caption("Chapter 17: Chunking and Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In naive RAG architectures, the **unit of search** is identical to the **unit of generation**: you search for a 300-token chunk, and you pass that exact 300-token chunk to the LLM.
This creates an inescapable engineering conflict:
* **Small chunks** produce sharp, high-precision vector matches, but lack the surrounding background context the LLM needs to synthesize a comprehensive answer.
* **Large chunks** provide rich surrounding context to the LLM, but suffer from vector dilution and low retrieval precision.

**Advanced Chunk Optimization** resolves this conflict by decoupling search from generation:
* **Parent-Child Chunks (Hierarchical / Small-to-Big Retrieval)**: Index small child chunks for pinpoint vector search, but retrieve the larger parent chunk for LLM prompt context!
* **Metadata Enrichment & Section Breadcrumbs**: Prepending document titles and section hierarchies to child chunks to eliminate pronoun ambiguity.
* **Sentence Window Retrieval**: Embedding individual sentences, but retrieving the surrounding $N$ neighbor sentences at query time.

This lesson explores how to optimize chunk architectures for enterprise-grade retrieval fidelity.
""")

st.divider()

# Section 1: Optimization Architectures: Parent-Child & Metadata Enrichment
st.header("1. Optimization Architectures: Parent-Child Hierarchies & Metadata Enrichment", divider="rainbow")

col_opt1, col_opt2 = st.columns([1, 1])

with col_opt1:
    st.markdown("""
    ### The 3 Master Chunk Optimizations
    * **1. Parent-Child (Small-to-Big) Architecture**:
      * **Ingestion**: Split document into large **Parent Chunks** (e.g. 800 tokens). Then, split each parent into small **Child Chunks** (e.g. 150 tokens).
      * **Indexing**: Only child chunks are vectorized and indexed in the vector database. Each child carries a `parent_id` in its metadata.
      * **Retrieval**: The query finds the best-matching child chunk via cosine similarity, but the retriever looks up and returns the **entire Parent Chunk** to the LLM!
    * **2. Metadata Enrichment & Heading Prepending**:
      * Raw chunk: *"It was increased by 15% in Q3."* (Ambiguous: what is 'it'?).
      * Enriched chunk: *"Document: Financial Report 2024 > Section: Cloud Division > Revenue: It was increased by 15% in Q3."*
      * Drastically enhances semantic embedding clarity and BM25 match rates.
    * **3. Sentence Window Retrieval**:
      * Embed single sentences (high precision). When retrieved, return the target sentence along with the preceding 2 sentences and following 2 sentences.
    """)

with col_opt2:
    st.markdown("### Parent-Child (Small-to-Big) Workflow")
    pc_dot = """
    digraph ParentChildRAG {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_parent {
            label="Parent Storage (Key-Value Store / DocStore)"; style="dashed"; color="#3B82F6";
            P1 [label="Parent Chunk 1 (800 tokens)\\nComprehensive Section Context", fillcolor="#EFF6FF", color="#1D4ED8", style="filled,bold"];
        }

        subgraph cluster_child {
            label="Child Vector Store (HNSW Index)"; style="dashed"; color="#10B981";
            C1 [label="Child 1 (150 tokens)\\n(parent_id='P1')", fillcolor="#D1FAE5"];
            C2 [label="Child 2 (150 tokens)\\n(parent_id='P1')", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
            C3 [label="Child 3 (150 tokens)\\n(parent_id='P1')", fillcolor="#D1FAE5"];
        }

        UserQ [label="Query: 'Specific SLA Penalty Clause'", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        UserQ -> C2 [label="Pinpoint vector match (0.94)"];
        C2 -> P1 [label="parent_id lookup in DocStore"];
        P1 -> LLMContext [label="Inject full 800 tokens to prompt"];
        LLMContext [label="LLM Context Prompt\\nComplete context, zero fragmentation!", fillcolor="#34D399", color="#047857", style="filled,bold"];
    }
    """
    st.graphviz_chart(pc_dot, use_container_width=False)

st.markdown("### Chunk Architecture Strategy Comparison")
st.markdown("""
| Architecture | Storage Overhead | Vector Search Precision | Context Delivered to LLM | Pronoun / Context Loss |
| :--- | :--- | :--- | :--- | :--- |
| **Flat Small Chunks (150 tokens)** | Lowest | **High** | Fragmented (Few sentences) | Severe (Antecedents lost) |
| **Flat Large Chunks (800 tokens)** | Moderate | Low (Diluted vectors) | **High** | Low |
| **Enriched Flat Chunks** | Low (+10% tokens) | **Very High** | Moderate | Minimal (Breadcrumbs explain context) |
| **Parent-Child (Small-to-Big)** | Moderate (+DocStore) | **Outstanding** | **Comprehensive (Full Parent)** | **Zero (Best of both worlds)** |
""")

st.divider()

# Section 2: Production Code: Parent-Child Hierarchical Store
st.header("2. Production Code: Hierarchical Parent-Child Retriever", divider="rainbow")

st.markdown("""
The production utility below demonstrates how to implement a Parent-Child (Small-to-Big) retriever: maintaining an in-memory Document Store for large parents and a Vector Store for small child embeddings.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple

class ParentChildRetrieverEngine:
    """
    Implements Hierarchical Small-to-Big Retrieval:
    - Child chunks (100 tokens) indexed in vector space for precision.
    - Parent chunks (500 tokens) stored in document store for generation.
    """
    def __init__(self, embedding_dim: int = 64):
        self.dim = embedding_dim
        self.parent_docstore: Dict[str, Dict[str, Any]] = {}
        self.child_vectorstore: List[Dict[str, Any]] = []

    def mock_embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.lower().strip())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

    def ingest_hierarchical_document(
        self,
        doc_id: str,
        section_title: str,
        full_text: str,
        parent_size: int = 60,
        child_size: int = 20
    ):
        words = full_text.split()
        
        # 1. Create Parent Chunks
        for p_idx in range(0, len(words), parent_size):
            p_words = words[p_idx : p_idx + parent_size]
            parent_id = f"{doc_id}_parent_{p_idx // parent_size + 1}"
            parent_text = " ".join(p_words)
            
            self.parent_docstore[parent_id] = {
                "parent_id": parent_id,
                "title": section_title,
                "full_text": parent_text,
                "word_count": len(p_words)
            }
            
            # 2. Create Child Chunks from this Parent
            for c_idx in range(0, len(p_words), child_size):
                c_words = p_words[c_idx : c_idx + child_size]
                child_id = f"{parent_id}_child_{c_idx // child_size + 1}"
                
                # Metadata enrichment: prepend breadcrumb hierarchy
                enriched_text = f"[{section_title}] {' '.join(c_words)}"
                c_vec = self.mock_embed(enriched_text)
                
                self.child_vectorstore.append({
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "text": " ".join(c_words),
                    "enriched_text": enriched_text,
                    "vector": c_vec
                })

    def retrieve(self, query: str, top_k_parents: int = 2) -> List[Dict[str, Any]]:
        q_vec = self.mock_embed(query)
        
        # 1. Search child vectors for nearest matches
        scored_children = []
        for child in self.child_vectorstore:
            sim = float(np.dot(child["vector"], q_vec))
            scored_children.append((child, sim))
        scored_children.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Deduplicate and resolve to Parent Chunks
        retrieved_parents = []
        seen_parent_ids = set()
        
        for child, sim in scored_children:
            p_id = child["parent_id"]
            if p_id not in seen_parent_ids:
                seen_parent_ids.add(p_id)
                parent_data = self.parent_docstore[p_id].copy()
                parent_data["matched_child_id"] = child["child_id"]
                parent_data["matched_score"] = sim
                parent_data["matched_snippet"] = child["text"]
                retrieved_parents.append(parent_data)
                
            if len(retrieved_parents) >= top_k_parents:
                break
                
        return retrieved_parents
''', language="python")

st.divider()

# Section 3: Interactive Parent-Child vs. Flat Chunking Arena
st.header("3. Interactive Workbench: Small-to-Big vs. Naive Flat Retrieval", divider="rainbow")

st.markdown("""
Compare how **Naive Flat Chunking** vs. **Parent-Child Retrieval** handles an enterprise legal contract containing specific clauses.
Notice how Naive Flat retrieval leaves the LLM with an isolated fragment, whereas Parent-Child delivers the complete context!
""")

class DemoHierarchyEngine:
    def __init__(self, dim: int = 48):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        t_low = text.lower()
        np.random.seed(abs(hash(t_low)) % (2**31 - 1))
        base = np.random.randn(self.dim).astype(np.float32)
        
        # Topic bias
        if "liability" in t_low or "breach" in t_low or "cap" in t_low:
            base[:16] += 2.8
        elif "warranty" in t_low or "remedy" in t_low:
            base[16:32] += 2.8
        elif "governing" in t_low or "jurisdiction" in t_low or "law" in t_low:
            base[32:] += 2.8
            
        return base / np.maximum(np.linalg.norm(base), 1e-12)

demo_engine = DemoHierarchyEngine()

# Sample Legal Contract Parent Text
contract_parent_1 = (
    "Section 8: Limitation of Liability and Aggregate Damages Cap.\n"
    "8.1. In no event shall either party be liable to the other for indirect, special, incidental, "
    "or consequential damages, including loss of profits or loss of data, arising out of this agreement.\n"
    "8.2. Each party's aggregate cumulative liability for direct damages shall be strictly capped "
    "at the total fees actually paid by customer in the twelve months preceding the incident.\n"
    "8.3. The liability cap in Section 8.2 shall not apply to damages resulting from gross negligence, "
    "willful misconduct, or infringement of intellectual property rights."
)

contract_parent_2 = (
    "Section 9: Limited Warranty and Exclusive Remedies.\n"
    "9.1. Licensor warrants that the software will conform materially to published specifications for 90 days.\n"
    "9.2. Customer's sole and exclusive remedy for breach of warranty shall be software repair or replacement.\n"
    "9.3. Except as stated herein, the software is provided as-is without warranties of merchantability."
)

col_ctrl1, col_ctrl2 = st.columns([1, 1])

with col_ctrl1:
    st.subheader("1. Legal Knowledge Ingestion")
    st.markdown("**Parent Section 8 (Limitation of Liability):**")
    st.caption("Parent Size: ~120 words. Broken into 3 Child Chunks (~40 words each).")
    
    st.markdown("**Parent Section 9 (Limited Warranty):**")
    st.caption("Parent Size: ~80 words. Broken into 2 Child Chunks (~40 words each).")

    st.subheader("2. Query Execution")
    test_queries = [
        "What are the exceptions to the liability cap?",
        "What is the exclusive remedy for breach of warranty?",
        "What is the monetary cap on direct damages?"
    ]
    selected_q = st.selectbox("Select User Legal Query:", test_queries)

with col_ctrl2:
    st.subheader("3. Retrieval Comparison")

    q_vec = demo_engine.embed(selected_q)

    # 1. Flat Naive Retrieval Simulation (Child only)
    child_snippets = [
        ("Child 8.1", "Section 8: Limitation of Liability... In no event shall either party be liable for indirect, special, or consequential damages.", contract_parent_1),
        ("Child 8.2", "8.2. Each party's aggregate cumulative liability for direct damages shall be strictly capped at the total fees actually paid in 12 months.", contract_parent_1),
        ("Child 8.3", "8.3. The liability cap in Section 8.2 shall not apply to damages resulting from gross negligence, willful misconduct, or IP infringement.", contract_parent_1),
        ("Child 9.1", "Section 9: Limited Warranty... Licensor warrants software conforms to specifications for 90 days.", contract_parent_2),
        ("Child 9.2", "9.2. Customer's sole remedy for breach of warranty shall be software repair or replacement.", contract_parent_2)
    ]

    scored = []
    for cid, ctext, parent_full in child_snippets:
        sim = float(np.dot(demo_engine.embed(ctext), q_vec))
        scored.append((cid, ctext, parent_full, sim))
    scored.sort(key=lambda x: x[3], reverse=True)
    top_match = scored[0]

    st.markdown("#### Approach A: Naive Flat Chunk Retrieval (Child Only)")
    with st.expander(f"Naive Output: {top_match[0]} (Similarity: {top_match[3]:.4f})", expanded=True):
        st.error(f"What the LLM receives:\n\n> {top_match[1]}")
        st.caption("⚠️ Notice: The LLM gets only clause 8.3. It does not know what Section 8.2 contained, creating context deficit!")

    st.markdown("#### Approach B: Parent-Child Retrieval (Small-to-Big)")
    with st.expander(f"Parent-Child Output: {top_match[2].splitlines()[0]} (Triggered by {top_match[0]})", expanded=True):
        st.success(f"What the LLM receives (Full Parent Section):\n\n{top_match[2]}")
        st.caption("✅ Perfection! Child vector gave 0.95+ match precision; Parent DocStore gave the LLM the complete section context.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Optimization Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is 'Small-to-Big' (Parent-Child) retrieval and why does it outperform standard RAG?**
    * **Answer**: In standard RAG, the same text chunk is used for both vector searching and LLM prompt generation. This forces an impossible compromise: small chunks produce sharp embeddings but lack context; large chunks have context but produce diluted embeddings.
    **Small-to-Big retrieval** decouples search from generation:
    * It splits documents into small child chunks (e.g. 100 tokens) that are embedded into vector space for sharp, pinpoint similarity search.
    * Each child references a larger parent chunk (e.g. 500-1000 tokens) stored in a key-value DocStore.
    * At query time, the system matches on the child vector, but resolves and feeds the entire parent chunk to the LLM.

    **Q2: How does Metadata Enrichment solve the 'pronoun ambiguity' problem in chunking?**
    * **Answer**: If a document section contains *"It increased by 25% year-over-year,"* the standalone embedding is almost meaningless because the vector model does not know what *"It"* refers to.
    Metadata enrichment prepends hierarchical document metadata directly into the chunk text:
    `[Document: Q3 Financials | Section: AWS Cloud Revenue] It increased by 25% year-over-year.`
    This grounds ambiguous pronouns in explicit entity terms, drastically boosting both dense cosine recall and BM25 keyword matching.

    **Q3: What is Sentence Window Retrieval?**
    * **Answer**: A variant of small-to-big retrieval where the unit of search is a single sentence. When that sentence is retrieved by vector search, the retriever automatically expands the context window by fetching the $K$ sentences before it and $K$ sentences after it (e.g. window size = 3), returning a cohesive 7-sentence paragraph to the LLM.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Redundant Parent Injection when Multiple Sibling Children Match**:
      * *Issue*: If Child 1, Child 2, and Child 3 all belong to Parent A, and all three appear in Top-K results, naively injecting Parent A three times.
      * *Consequence*: The LLM prompt context is flooded with 3 identical copies of an 800-token parent document, exhausting token limits.
      * *Fix*: Always deduplicate parent IDs before assembling the augmented prompt context.

    * **Trap 2: Forgetting to Strip Enriched Metadata Before LLM Generation**:
      * *Issue*: Prepending internal database IDs and raw JSON strings into child chunks and passing them verbatim to the user-facing LLM output.
      * *Consequence*: The LLM generates ugly responses repeating database internal keys to customers.
      * *Fix*: Clean formatting breadcrumbs into natural readable citations before prompt synthesis.

    * **Trap 3: Oversized Parent Chunks Diluting LLM Attention**:
      * *Issue*: Defining Parent chunks as entire 50-page documents.
      * *Consequence*: Reintroduces the Lost-in-the-Middle problem and high inference costs.
      * *Fix*: Cap Parent chunks at cohesive sections (500-1000 tokens).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Small-to-Big Retrieval**
      * *Definition*: The architectural pattern of decoupling search units (small child chunks) from generation units (large parent chunks).
    * **Flashcard 2: Sentence Window Retrieval**
      * *Definition*: A retrieval technique where single sentences are embedded, but surrounding adjacent sentences are retrieved to restore context.
    * **Flashcard 3: Metadata Enrichment**
      * *Definition*: Prepending document titles, section headers, and breadcrumbs to chunks to eliminate entity and pronoun ambiguity.
    * **Flashcard 4: Parent Deduplication**
      * *Definition*: The essential step in hierarchical retrieval of filtering duplicate parent IDs when multiple child chunks from the same parent match.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Chunk Optimization:

- [ ] I can articulate the fundamental conflict between unit of search and unit of generation.
- [ ] I understand how Parent-Child (Small-to-Big) architecture resolves this conflict.
- [ ] I know how to implement Parent Deduplication when multiple sibling children match.
- [ ] I understand how Metadata Enrichment and breadcrumbs resolve pronoun ambiguity.
- [ ] I can explain the mechanics of Sentence Window Retrieval.
""")
