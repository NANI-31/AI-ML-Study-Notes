import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="1.2 RAG Architecture - RAG Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title Block & 5-Pillar Overview
st.title("1.2 RAG Architecture: Dual-Stage Pipelines, Chunking, Embeddings, & Synthesis")
st.caption("Chapter 1: Introduction to RAG | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
A production-grade Retrieval-Augmented Generation (RAG) system is not a single script calling an API—it is an 
enterprise distributed data system composed of two asynchronous, highly coordinated lifecycles:
1. **The Ingestion Pipeline (Offline/Asynchronous)**: Raw document harvesting, multimodal parsing, semantic chunking, vector embedding generation, and persistent index storage.
2. **The Query Serving Loop (Online/Sub-second)**: Query analysis, hybrid vector/lexical retrieval, cross-encoder precision reranking, context assembly, and conditioned LLM synthesis with auditable provenance citations.

In this lesson, you will master:
1. The structural anatomy connecting offline ingestion and real-time query serving.
2. The operational trade-offs between chunking strategies (fixed-window, recursive, semantic boundary).
3. Embedding generation, geometric similarity spaces, and high-dimensional ANN index structures.
4. Hybrid retrieval fusion (BM25 + Dense Vectors) and precision cross-attention reranking.
5. End-to-end Python pipeline construction and mission-critical deployment traps.
""")

st.divider()

# Module-Scope Helper Classes for Interactive Workbench & Production Pipeline
class SimulatedRAGArchitectureEngine:
    def __init__(self):
        self.raw_documents = {
            "Policy_Cloud_Infra.pdf": (
                "Kubernetes clusters in production must enforce egress network policies. "
                "All microservices communicating across namespaces must authenticate via mutual TLS (mTLS). "
                "Secrets must never be stored in plain text environment variables."
            ),
            "HR_Global_Leave_2026.docx": (
                "Full-time employees receive 25 days of paid annual leave. "
                "Medical leave requests extending beyond 3 consecutive business days require a licensed physician note. "
                "Parental leave offers 16 weeks of 100% compensated leave."
            ),
            "Quarterly_Security_Patching.md": (
                "Critical CVE vulnerabilities must be remediated and deployed within 48 hours of public advisory. "
                "Zero-day exploits require an immediate severity-1 war room assembly. "
                "Automated container vulnerability scans run nightly at 02:00 UTC."
            )
        }

    def chunk_document(self, text: str, chunk_size: int = 80, overlap: int = 15) -> List[str]:
        words = text.split()
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk_slice = " ".join(words[i:i + chunk_size])
            if chunk_slice:
                chunks.append(chunk_slice)
        return chunks

    def simulate_retrieval(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        query_terms = set(query.lower().split())
        scored = []
        for doc_name, text in self.raw_documents.items():
            chunks = self.chunk_document(text, chunk_size=20, overlap=5)
            for idx, ch in enumerate(chunks):
                ch_terms = set(ch.lower().split())
                overlap_count = len(query_terms.intersection(ch_terms))
                sim_score = (overlap_count * 0.35) + (0.55 if any(t in ch.lower() for t in query_terms) else 0.1)
                scored.append({
                    "chunk_id": f"{doc_name[:6]}_C{idx+1}",
                    "document": doc_name,
                    "text": ch,
                    "score": round(min(0.99, sim_score), 3)
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

# ==============================================================================
# 1. Architecture Blueprint
# ==============================================================================
st.header("1. Architecture Blueprint: The End-to-End Enterprise RAG Mesh Pipeline", divider="rainbow")
st.markdown("""
A production RAG deployment consists of an **Ingestion Pipeline** (data prep) and a **Query Serving Loop** (runtime execution). 
The diagram below maps every component across the two operational stages in a readable top-to-bottom orientation:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_rag_architecture = """
    digraph RAGArchitecture {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingestion {
            label="Phase I: Offline Ingestion Pipeline Tier";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            docs [label="1. Raw Enterprise Documents\\n(PDF, Word, Markdown, Confluence)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            parser [label="2. Document Processing\\nText Extraction, Cleaning, & Layout Analysis", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            splitter [label="3. Semantic & Recursive Chunking\\nChunks text into coherent passages with overlap", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            embedder [label="4. Embedding Transformation\\nDense Vector Projection (e.g. 1024 dimensions)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            vstore [label="5. Enterprise Vector DB & Lexical Index\\n(HNSW ANN Index + BM25 Inverted Index)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            docs -> parser -> splitter -> embedder -> vstore;
        }

        subgraph cluster_query_loop {
            label="Phase II: Online Real-Time Query Serving Tier";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            fontsize=12;
            q [label="6. User Query Ingress\\n'What is our CVE remediation deadline?'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            qproc [label="7. Query Transformation\\nAcronym Expansion & Sub-Query Decomposition", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            retriever [label="8. Parallel Hybrid Retrieval\\nSparse BM25 Keyword + Dense Vector ANN", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            reranker [label="9. Precision Cross-Encoder Reranker\\nJoint All-to-All Cross-Attention Top-5 Scoring", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            augment [label="10. Context Augmentation Layer\\nPrompt Assembly with Citations & Abstention Rules", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            llm [label="11. Grounded LLM Generation\\nSynthesizes Response Strictly Conditioned on Context", fillcolor="#1E1B4B", fontcolor="#E0E7FF", color="#818CF8"];
            resp [label="12. Audited Output with Verifiable Citations\\n'48 hours [Citation: Security_Patching.md]'", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];

            q -> qproc -> retriever;
            retriever -> reranker [label="Top-50 Candidates"];
            reranker -> augment [label="Top-5 Chunks"];
            augment -> llm -> resp;
        }

        vstore -> retriever [label="Index Search", color="#10B981"];
    }
    """
    st.graphviz_chart(dot_rag_architecture, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Early naive RAG implementations piped single-stage vector lookups directly into prompts. 
State-of-the-art enterprise stacks (Anthropic Contextual Retrieval, Cohere Rerank 3, LlamaIndex Advanced RAG) 
implement **two-stage hybrid retrieval**: a high-recall first stage (BM25 sparse search + Dense vector ANN) 
followed by a precision **cross-encoder reranker** that evaluates deep token-level interactions before presenting 
the top 3-5 chunks to the LLM context window.
""")

st.divider()

# ==============================================================================
# 2. Basic RAG Pipeline
# ==============================================================================
st.header("2. Basic RAG Pipeline", divider="rainbow")
st.markdown("""
At its core, a RAG pipeline executes three linear phases:
* **Index**: Process documents into indexable chunks and numerical representations.
* **Retrieve**: For a given user prompt, search the index and fetch the highest-scoring candidate chunks.
* **Generate**: Feed the user prompt and the retrieved chunks into the LLM, prompting it to synthesize an answer.
""")

df_basic_pipeline = pd.DataFrame([
    {"Phase": "1. Ingestion Phase", "Execution Mode": "Asynchronous / Offline", "Primary Bottleneck": "PDF parsing throughput & embedding API rate limits."},
    {"Phase": "2. Retrieval Phase", "Execution Mode": "Real-time Online (10ms - 80ms)", "Primary Bottleneck": "Vector index graph search (HNSW) and network I/O."},
    {"Phase": "3. Synthesis Phase", "Execution Mode": "Real-time Online (500ms - 3000ms)", "Primary Bottleneck": "LLM time-to-first-token (TTFT) and token generation velocity."}
])
st.dataframe(df_basic_pipeline, use_container_width=True)

st.divider()

# ==============================================================================
# 3. Document Ingestion
# ==============================================================================
st.header("3. Document Ingestion", divider="rainbow")
st.markdown("""
**Document Ingestion** is the extract-transform-load (ETL) stage that connects enterprise data sources 
(SharePoint, Google Drive, Jira, S3, SQL databases) to the AI pipeline. 

Key architectural requirements for production ingestion:
1. **Change Data Capture (CDC)**: Ingest only modified, created, or deleted documents to keep the index synced without full corpus re-indexing.
2. **Access Control List (ACL) Inheritance**: Ingest document metadata including user permissions, tenant IDs, and confidentiality tags so downstream retrieval can enforce access rules.
3. **Multimodal Ingestion**: Extract tables, flowcharts, and embedded images alongside plain text.
""")

st.divider()

# ==============================================================================
# 4. Document Processing
# ==============================================================================
st.header("4. Document Processing", divider="rainbow")
st.markdown("""
Raw files cannot be fed directly into splitters without cleaning. **Document Processing** transforms messy, 
unstructured formats into clean, structured Markdown or JSON text:
* **Header & Footer Stripping**: Eliminates page numbers, headers, and copyright disclaimers that pollute embeddings.
* **Table Extraction**: Serializes tabular financial or technical data into structured Markdown or HTML tables to preserve row-column semantics.
* **OCR & Layout Analysis**: Employs vision models to identify reading order in multi-column documents (academic papers, legal briefs).
""")

st.divider()

# ==============================================================================
# 5. Chunking
# ==============================================================================
st.header("5. Chunking Strategies & Boundary Optimization", divider="rainbow")
st.markdown("""
LLMs and embedding models have finite input token windows. **Chunking** divides long documents into coherent, 
semantically self-contained segments. The chunk size governs the trade-off between **precision** and **context**:
""")

col_ch1, col_ch2, col_ch3 = st.columns(3)
with col_ch1:
    st.info("""
    **Small Chunks (100–250 tokens)**
    * **Pros**: High semantic specificity; vector matches pinpoint exact sentences.
    * **Cons**: Fragmented context; misses overarching narrative and global relationships.
    """)
with col_ch2:
    st.info("""
    **Medium Chunks (400–800 tokens)**
    * **Pros**: The enterprise standard sweet spot; preserves complete paragraphs and logical flow.
    * **Cons**: Slight noise introduction if multiple topics are merged into one passage.
    """)
with col_ch3:
    st.info("""
    **Large Chunks (1000–2000 tokens)**
    * **Pros**: Captures entire subsections and multi-part procedural guidelines.
    * **Cons**: Vector dilution; embedding averages out distinct facts, lowering precision.
    """)

with st.expander("🔍 View Chunking Boundary Strategy Diagram", expanded=True):
    dot_chunking = """
    digraph ChunkingComparison {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        doc [label="Full Enterprise Document (10,000 Words)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        
        c_fixed [label="Fixed-Character Chunking\\nBlind split every 1000 chars (Risk of severed words)", fillcolor="#450A0A", fontcolor="#FCA5A5", color="#EF4444"];
        c_rec [label="Recursive Character Chunking\\nSplits on Paragraphs -> Sentences -> Words", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        c_hier [label="Parent-Child Hierarchical Chunking\\nSmall 150t Child for Search -> 800t Parent to Prompt", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];

        doc -> c_fixed;
        doc -> c_rec;
        doc -> c_hier;
    }
    """
    st.graphviz_chart(dot_chunking, use_container_width=True)

st.markdown("""
### Chunking Algorithms Comparative Breakdown
1. **Fixed-Size Chunking with Overlap**: Divides text by a fixed token count with sliding overlap to avoid splitting keywords across boundaries.
2. **Recursive Character Chunking**: Recursively splits on paragraphs, newlines, and sentence ends to preserve logical units naturally.
3. **Semantic Boundary Chunking**: Analyzes consecutive sentence transitions and splits when topical distance shifts significantly.
4. **Parent-Child (Hierarchical) Chunking**: Embeds small 150-token child chunks for precision retrieval, but returns the larger 800-token parent chunk to the LLM for synthesis.
""")

st.divider()

# ==============================================================================
# 6. Embedding Generation
# ==============================================================================
st.header("6. Embedding Generation & Vector Space Projection", divider="rainbow")
st.markdown("""
An **embedding model** (Bi-Encoder) maps textual chunks into continuous, dense vector spaces where semantically 
similar concepts reside in geometric proximity:

* **Semantic Proximity**: Related concepts (e.g. `"mTLS configuration"` and `"mutual TLS in K8s"`) map to closely aligned vectors, identified when cosine similarity exceeds a threshold (e.g. `similarity >= 0.80`).
* **Dense Vectors**: Every text passage is represented as a dense numerical vector across standard dimensions (e.g., 768, 1024, or 1536).
* **Modern Trend**: Matryoshka Representation Learning (MRL) allows vectors to be truncated (e.g. from 1536 down to 256 dimensions) with minimal loss of retrieval accuracy.
""")

st.divider()

# ==============================================================================
# 7. Vector Storage
# ==============================================================================
st.header("7. Vector Storage & Index Structures", divider="rainbow")
st.markdown("""
Standard relational databases cannot efficiently perform similarity search across millions of high-dimensional vectors. 
**Vector Databases** use **Approximate Nearest Neighbor (ANN)** index algorithms:
""")

df_ann = pd.DataFrame([
    {
        "Index Algorithm": "HNSW (Hierarchical Navigable Small World)",
        "Search Latency": "Sub-10ms (Ultra-Fast)",
        "Memory Footprint": "High (Maintains multi-layer graph in RAM)",
        "Recall Precision": "95% - 99%",
        "Ideal Production Use Case": "Real-time interactive user-facing query serving."
    },
    {
        "Index Algorithm": "IVF-PQ (Inverted File with Product Quantization)",
        "Search Latency": "Moderate",
        "Memory Footprint": "Low (Compresses vector representations)",
        "Recall Precision": "85% - 92%",
        "Ideal Production Use Case": "Billion-scale datasets with constrained RAM budgets."
    },
    {
        "Index Algorithm": "Flat (Exhaustive Search)",
        "Search Latency": "High (Scans every vector linearly)",
        "Memory Footprint": "Minimal overhead",
        "Recall Precision": "100% (Exact match)",
        "Ideal Production Use Case": "Offline evaluation datasets and benchmarks (< 10,000 vectors)."
    }
])
st.dataframe(df_ann, use_container_width=True)

st.divider()

# ==============================================================================
# 8. Query Processing
# ==============================================================================
st.header("8. Query Processing & Intent Extraction", divider="rainbow")
st.markdown("""
Raw user queries are often ambiguous, typo-ridden, or missing crucial domain context. **Query Processing** 
transforms raw input into optimized retrieval prompts:
* **Query Rewriting**: Expanding abbreviations and resolving conversational pronouns (e.g. converting `"What about its pricing?"` to `"What is the enterprise pricing for Snowflake in 2026?"`).
* **HyDE (Hypothetical Document Embeddings)**: Instructs an LLM to generate a hypothetical answer passage, then embeds the *hypothetical passage* to match real documents in vector space.
* **Query Decomposition**: Breaks complex multi-hop questions into multiple atomic sub-queries.
""")

st.divider()

# ==============================================================================
# 9. Retrieval
# ==============================================================================
st.header("9. Retrieval: Dense, Sparse, & Hybrid Fusion", divider="rainbow")
st.markdown("""
A production retrieval engine must avoid relying on a single search modality. Modern architectures use **Hybrid Search**:
""")

col_ret1, col_ret2 = st.columns(2)
with col_ret1:
    st.subheader("Sparse Keyword Search (BM25)")
    st.markdown("""
    * **Strength**: Exact keyword matching, product serial numbers, alphanumeric error codes (`CVE-2026-1194`, `404 Not Found`).
    * **Weakness**: Zero understanding of semantic synonyms or conceptual paraphrasing.
    """)
with col_ret2:
    st.subheader("Dense Vector Search (Semantic ANN)")
    st.markdown("""
    * **Strength**: High conceptual abstraction, multilingual alignment, semantic generalization.
    * **Weakness**: Fails on exact serial codes, rare proper nouns, and short acronyms.
    """)

with st.expander("🔍 View Hybrid Fusion & RRF Diagram", expanded=True):
    dot_hybrid = """
    digraph HybridRetrieval {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        q [label="Transformed Query: 'CVE-2026 remediation deadline'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        bm25 [label="Branch 1: Sparse BM25 Search\\n(Matches exact 'CVE-2026' string)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        ann [label="Branch 2: Dense Vector ANN\\n(Matches 'security vulnerability fix SLA')", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        rrf [label="Reciprocal Rank Fusion (RRF)\\nAggregates rank positions into unified Top-50 candidates", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        rerank [label="Cross-Encoder Reranker\\nJoint attention selects Top-5 precision chunks", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];

        q -> bm25;
        q -> ann;
        bm25 -> rrf;
        ann -> rrf;
        rrf -> rerank;
    }
    """
    st.graphviz_chart(dot_hybrid, use_container_width=True)

st.markdown("""
### Reciprocal Rank Fusion (RRF) Principle
Reciprocal Rank Fusion merges candidate lists by evaluating the relative position of items across disparate search mechanisms. 
Documents that rank near the top in both keyword and semantic searches receive highest candidate priority, without 
requiring fragile score normalization.
""")

st.divider()

# ==============================================================================
# 10. Context Augmentation
# ==============================================================================
st.header("10. Context Augmentation & Prompt Framing", divider="rainbow")
st.markdown("""
Once top-K chunks are retrieved and reranked, **Context Augmentation** packages them into a strict prompt contract:
* **Delimited Context Enclosure**: Placing retrieved passages inside clear XML or Markdown delimiters (`<context>...</context>`).
* **Citation Grounding Directives**: Instructing the model to include `[Source: DocID]` brackets after every factual assertion.
* **Negative Abstention Rule**: Explicit instruction to reply `"Information not found in context"` rather than guessing when evidence is missing.
""")

st.divider()

# ==============================================================================
# 11. LLM Generation
# ==============================================================================
st.header("11. LLM Generation: Constrained Context Synthesis", divider="rainbow")
st.markdown("""
The generative LLM receives the augmented prompt and synthesizes a fluent, contextual answer:
* **Decoding Temperature**: Set low (`temperature = 0.0` to `0.2`) for factual RAG tasks to minimize stochastic hallucinations.
* **Constrained Decoding**: Output structured schemas (e.g. JSON or Pydantic models) when feeding downstream enterprise microservices.
""")

st.divider()

# ==============================================================================
# 12. Final Answer with Citations
# ==============================================================================
st.header("12. Final Answer with Citations & Provenance Auditing", divider="rainbow")
st.markdown("""
The final deliverable of a production RAG system is not just text, but an **auditable response object** 
containing the synthesized answer alongside verified provenance citations:
""")

st.json({
    "answer": "Kubernetes clusters in production must enforce egress network policies and require mutual TLS (mTLS) for all cross-namespace communication.",
    "citations": [
        {
            "citation_id": "CITE-1",
            "source_document": "Policy_Cloud_Infra.pdf",
            "section": "Section 3.2 Network Security",
            "verbatim_quote": "Kubernetes clusters in production must enforce egress network policies... require mutual TLS (mTLS)."
        }
    ],
    "groundedness_verified": True,
    "latency_telemetry": {
        "retrieval_ms": 34,
        "rerank_ms": 52,
        "llm_ttft_ms": 310,
        "total_ms": 680
    }
})

st.divider()

# ==============================================================================
# 13. Production Implementation
# ==============================================================================
st.header("13. Production Implementation: Multi-Stage Ingestion & Query Serving Pipeline", divider="rainbow")
st.markdown("""
Below is a runnable, end-to-end Python script implementing multi-stage document ingestion, chunking, 
and query retrieval with citation assembly:
""")

prod_arch_code = """
from typing import List, Dict, Any

class DocumentIngestionPipeline:
    def __init__(self, chunk_size: int = 50, overlap: int = 10):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vector_store: List[Dict[str, Any]] = []

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            slice_text = " ".join(words[i:i + self.chunk_size])
            if slice_text:
                chunks.append(slice_text)
        return chunks

    def ingest(self, doc_id: str, title: str, text: str):
        chunks = self.chunk_text(text)
        for idx, chunk in enumerate(chunks):
            # In production: embedding = embedding_model.encode(chunk)
            self.vector_store.append({
                "chunk_id": f"{doc_id}_C{idx+1}",
                "doc_title": title,
                "text": chunk,
                "token_count": len(chunk.split())
            })

    def query(self, user_query: str, top_k: int = 2) -> Dict[str, Any]:
        q_tokens = set(user_query.lower().split())
        scored_chunks = []
        for item in self.vector_store:
            item_tokens = set(item["text"].lower().split())
            overlap = len(q_tokens.intersection(item_tokens))
            score = overlap / (len(q_tokens) + 1e-6)
            scored_chunks.append((score, item))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [item for score, item in scored_chunks[:top_k] if score > 0]
        
        # Build prompt contract
        context = "\\n\\n".join([f"[{c['chunk_id']}] {c['text']}" for c in top_chunks])
        prompt = f"CONTEXT:\\n{context}\\n\\nQUERY: {user_query}\\nANSWER WITH CITATIONS:"
        
        return {
            "query": user_query,
            "top_chunks": top_chunks,
            "assembled_prompt": prompt
        }
"""
st.code(prod_arch_code, language="python")

st.divider()

# ==============================================================================
# 14. Interactive Workbench
# ==============================================================================
st.header("14. Interactive Workbench: Ingestion Tuning & Retrieval Studio", divider="rainbow")
st.markdown("""
Experiment with document chunking parameters and observe how the retrieval engine extracts 
grounded evidence passages in real time:
""")

engine = SimulatedRAGArchitectureEngine()

col_w1, col_w2 = st.columns(2)
with col_w1:
    selected_doc = st.selectbox("Select Raw Ingestion Document:", list(engine.raw_documents.keys()))
    chunk_sz = st.slider("Chunk Size (Words):", min_value=10, max_value=50, value=25)
    overlap_sz = st.slider("Chunk Overlap (Words):", min_value=2, max_value=15, value=5)

with col_w2:
    st.subheader("Raw Document Content")
    st.text_area("Original Text:", engine.raw_documents[selected_doc], height=130, disabled=True)

raw_text = engine.raw_documents[selected_doc]
generated_chunks = engine.chunk_document(raw_text, chunk_sz, overlap_sz)

st.subheader(f"Generated Document Chunks ({len(generated_chunks)} Total)")
df_chunks = pd.DataFrame([
    {"Chunk Index": f"Chunk #{i+1}", "Word Count": len(c.split()), "Preview": c}
    for i, c in enumerate(generated_chunks)
])
st.dataframe(df_chunks, use_container_width=True)

st.divider()

st.subheader("Live Query & Retrieval Simulation")
user_test_q = st.text_input("Enter Query to Test Retrieval Across Ingested Docs:", value="What is the CVE vulnerability patch requirement?")
if st.button("Simulate Retrieval & Rerank", type="primary"):
    results = engine.simulate_retrieval(user_test_q, top_k=2)
    st.markdown("**Top Retrieved & Ranked Chunks:**")
    for r in results:
        st.success(f"**[{r['chunk_id']}]** ({r['document']}) — Relevance Score: `{r['score']}`\n\n\"{r['text']}\"")

st.divider()

# ==============================================================================
# 15. Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("15. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why does fixed-token chunking often fail in production RAG?"):
    st.markdown("""
    **Answer**: Fixed-token chunking ignores natural document syntax, frequently splitting a single semantic statement 
    or tabular row across two chunks (e.g. splitting a question from its answer, or a clause from its negation). 
    Production systems use **Recursive Character Chunking** or **Semantic Chunking** with sentence-aware boundaries.
    """)

with st.expander("Q2: Explain the difference between HNSW and IVF-PQ vector indexing."):
    st.markdown("""
    **Answer**: **HNSW (Hierarchical Navigable Small World)** is a graph-based index providing near-perfect recall (98%+) 
    and sub-10ms latency by routing queries through a multi-layer skip-graph, but requires storing the entire graph in RAM. 
    **IVF-PQ (Inverted File with Product Quantization)** clusters vectors and quantizes them into compressed byte codes, 
    reducing memory usage by 80-90% at the cost of lower recall (85-92%).
    """)

with st.expander("Q3: What is HyDE (Hypothetical Document Embeddings) and when should it be used?"):
    st.markdown("""
    **Answer**: HyDE generates a hypothetical answer passage using a generative LLM before searching the vector database. 
    It is useful when user queries are very short or formulated as questions, bridging the semantic gap between 
    a *question embedding* and a *declarative document embedding*.
    """)

with st.expander("Q4: What is the Parent-Child (Hierarchical) Chunking architecture?"):
    st.markdown("""
    **Answer**: In parent-child chunking, documents are split into small 100-200 token child chunks for high-granularity 
    vector embedding, but each child references a 1000-token parent chunk. When a child chunk matches during retrieval, 
    the system injects the full *parent chunk* into the prompt, giving the LLM rich surrounding context.
    """)

with st.expander("Q5: How do you mitigate the 'Lost-in-the-Middle' phenomenon during prompt assembly?"):
    st.markdown("""
    **Answer**: Sort retrieved chunks so that the highest-scoring evidence chunks are placed at the very beginning 
    or the very end of the injected context window, where Transformer self-attention is empirically strongest.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The Table Shredding Trap)**: Running standard naive character splitters over PDF tables shreds row and column alignments, turning structured financial statements into meaningless unlinked number fragments.")
    st.warning("**Trap 2 (Neglecting Chunk Overlap)**: Setting chunk overlap to zero causes sentences split across chunk boundaries to lose critical qualifying adjectives or negation clauses.")
with col_trap2:
    st.error("**Trap 3 (Embedding Model Incompatibility Mismatch)**: Indexing document chunks with one embedding model and later querying the index with a different model. The vector spaces are geometrically orthogonal and yield zero valid matches.")
    st.warning("**Trap 4 (Blind First-Stage Top-K Over-Retrieval)**: Injecting 30 raw vector search chunks directly into the prompt without a cross-encoder reranker introduces noise, dilutes attention, and inflates API costs.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the architectural difference between Ingestion and the Query Loop?")
        st.caption("Flip for answer:")
        st.info("**Back**: Ingestion is an offline/async ETL pipeline (parse, chunk, embed, index). The Query Loop is an online real-time serving path (query, retrieve, rerank, synthesize).")
    with st.container(border=True):
        st.markdown("**Front**: What is the primary benefit of HNSW indexing?")
        st.caption("Flip for answer:")
        st.info("**Back**: Provides sub-10ms similarity search latency with 95%+ recall accuracy by traversing a multi-layer graph structure in memory.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How does Parent-Child Chunking operate?")
        st.caption("Flip for answer:")
        st.info("**Back**: Embeds small 150-token child chunks for high-precision search, but injects the larger 800-token parent chunk into the LLM context for complete narrative coherence.")
    with st.container(border=True):
        st.markdown("**Front**: Why is Cross-Encoder Reranking applied as Stage 2?")
        st.caption("Flip for answer:")
        st.info("**Back**: Stage 1 bi-encoders provide fast candidate recall. Stage 2 cross-encoders apply all-to-all attention over top candidates to eliminate false-positive semantic matches.")

st.divider()

# ==============================================================================
# 16. Self-Assessment Checklist
# ==============================================================================
st.header("16. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I can differentiate between offline document ingestion and the online query serving loop.", value=True)
c2 = st.checkbox("I understand the trade-offs of fixed-size, recursive, and semantic boundary chunking.", value=True)
c3 = st.checkbox("I know how high-dimensional embedding vectors are indexed and queried via ANN algorithms (HNSW).", value=True)
c4 = st.checkbox("I understand why hybrid search combines sparse BM25 and dense cosine vectors via RRF.", value=True)
c5 = st.checkbox("I can articulate how cross-encoders eliminate false-positive semantic matches before prompt injection.", value=True)
c6 = st.checkbox("I know how to construct auditable citation objects linking generated claims to chunk IDs.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Module 1.2 Complete! You are ready to advance to Lesson 1.3: RAG Applications.")
