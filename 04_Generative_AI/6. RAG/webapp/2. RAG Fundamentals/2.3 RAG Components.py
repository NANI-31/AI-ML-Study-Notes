import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="2.3 RAG Components - RAG Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title Block & 5-Pillar Overview
st.title("2.3 RAG Components: The Ten Modular Pillars of Production Retrieval Systems")
st.caption("Chapter 2: RAG Fundamentals | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
A production-grade Retrieval-Augmented Generation system is not an indivisible monolith—it is a composable, 
modular microservices architecture comprised of ten interconnected subsystems: **Documents**, **Document Loaders**, 
**Text Splitters**, **Embedding Models**, **Vector Databases**, **Retrievers**, **Rerankers**, **LLMs**, 
**Prompt Templates**, and **Evaluation Systems**.

Architecting an enterprise RAG stack requires understanding the failure boundaries, latency trade-offs, and 
scaling economics of each component. Swapping an embedding model impacts vector database dimensionality; tuning a text 
splitter alters cross-encoder reranking efficacy; upgrading prompt templates alters hallucination rates.

In this lesson, you will master:
1. The architectural contracts governing the ten primary RAG components.
2. Ingestion components: Loaders, multimodal extraction, and recursive text splitters.
3. Indexing & Retrieval components: High-dimensional embeddings, vector databases, and hybrid retrievers.
4. Precision components: Cross-encoder rerankers, prompt contracts, and grounded LLM generators.
5. Automated evaluation frameworks (Ragas, TruLens) for continuous quality regression testing.
""")

st.divider()

# Module-Scope Helper Classes for Interactive Workbench & Production Pipeline
class PluggableComponentPipeline:
    def __init__(self, chunk_strategy: str, embedding_dim: int, reranker_enabled: bool):
        self.chunk_strategy = chunk_strategy
        self.embedding_dim = embedding_dim
        self.reranker_enabled = reranker_enabled
        self.raw_document = (
            "Section 4.1 Enterprise Security Policy: Production Kubernetes clusters must enforce egress network policies. "
            "All inter-service traffic must be encrypted with mutual TLS. Tokens must be rotated every 14 days. "
            "Violation of these rules will result in immediate automated cluster quarantine."
        )

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        # Chunking
        if self.chunk_strategy == "Small (10 words)":
            chunks = [" ".join(self.raw_document.split()[i:i+10]) for i in range(0, len(self.raw_document.split()), 8)]
        elif self.chunk_strategy == "Medium (25 words)":
            chunks = [" ".join(self.raw_document.split()[i:i+25]) for i in range(0, len(self.raw_document.split()), 20)]
        else:
            chunks = [self.raw_document]

        # Simulated First-Stage Retrieval (Top 3)
        scored = []
        q_words = set(query.lower().split())
        for idx, ch in enumerate(chunks):
            ch_words = set(ch.lower().split())
            score = len(q_words.intersection(ch_words)) * 0.3 + 0.2
            scored.append({"chunk_id": f"CHUNK-{idx+1}", "text": ch, "raw_score": round(score, 3)})
        scored.sort(key=lambda x: x["raw_score"], reverse=True)
        top_candidates = scored[:3]

        # Cross-Encoder Reranker
        if self.reranker_enabled:
            for item in top_candidates:
                item["final_score"] = round(min(0.98, item["raw_score"] * 1.4), 3)
                item["reranker_status"] = "RERANKED_CROSS_ATTENTION"
        else:
            for item in top_candidates:
                item["final_score"] = item["raw_score"]
                item["reranker_status"] = "RAW_VECTOR_SCORE"

        top_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return {
            "total_chunks_produced": len(chunks),
            "top_selected": top_candidates,
            "estimated_latency_ms": 45 + (35 if self.reranker_enabled else 0)
        }

# ==============================================================================
# 1. Architecture Blueprint
# ==============================================================================
st.header("1. Architecture Blueprint: The 10-Component Enterprise RAG Stack Topology", divider="rainbow")
st.markdown("""
The diagram below maps the complete ten-component topology across the offline data preparation tier, 
the online storage tier, the serving tier, and the continuous evaluation telemetry loop in a top-to-bottom layout:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_10_components = """
    digraph TenComponents {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingestion {
            label="Tier 1: Ingestion & Document Preparation";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            c1 [label="1. Documents\\nRaw Unstructured Enterprise Files", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            c2 [label="2. Document Loaders\\nPDF, OCR, Audio, HTML Extractors", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            c3 [label="3. Text Splitters\\nRecursive & Semantic Boundary Chunkers", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            c1 -> c2 -> c3;
        }

        subgraph cluster_storage {
            label="Tier 2: Vector Representation & Index Storage";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            fontsize=12;
            c4 [label="4. Embedding Models\\nDense Vector Transform (Bi-Encoders)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            c5 [label="5. Vector Database\\nHNSW Graph & IVF-PQ Clustered Index", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            c3 -> c4 -> c5;
        }

        subgraph cluster_serving {
            label="Tier 3: Query Serving, Reranking, & Synthesis";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            fontsize=12;
            c6 [label="6. Retrievers\\nHybrid Sparse BM25 + Dense ANN Search", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            c7 [label="7. Rerankers\\nCross-Encoder Joint All-to-All Attention", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            c8 [label="8. Prompt Templates\\nStrict Evidence Bounding Contracts", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            c9 [label="9. Large Language Models\\nGrounded Context Reasoning & Synthesis", fillcolor="#1E1B4B", fontcolor="#E0E7FF", color="#818CF8"];
            c5 -> c6 -> c7 -> c8 -> c9;
        }

        subgraph cluster_eval {
            label="Tier 4: Observability & Continuous Evaluation Telemetry";
            bgcolor="#1E293B";
            fontcolor="#EC4899";
            color="#334155";
            fontsize=12;
            c10 [label="10. Evaluation Systems\\nRagas / TruLens / LLM-as-a-Judge Audit", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
        }

        c9 -> c10 [label="Audit Groundedness"];
        c6 -> c10 [label="Audit Context Relevance"];
    }
    """
    st.graphviz_chart(dot_10_components, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
While early hobbyist tutorials glued these components into fragile 5-line scripts, modern enterprise systems 
treat each of these ten components as an independent, observable microservice. Organizations can swap their 
vector database (e.g. from ChromaDB to Qdrant or Milvus) or upgrade their embedding model (from text-embedding-ada-002 
to text-embedding-3 or BAAI/bge-m3) with zero downtime and automated regression evaluation.
""")

st.divider()

# ==============================================================================
# 2. Documents
# ==============================================================================
st.header("2. Documents: Unstructured Data Formats & Metadata Contracts", divider="rainbow")
st.markdown("""
The **Document** is the foundational unit of data ingested into a RAG pipeline. A production document consists 
not just of raw text, but of a rich **Metadata Contract**:
* **Payload Text**: The verbatim human-readable content.
* **Structural Metadata**: Source URI, file name, document title, page numbers, author, creation timestamp, and tenant ID.
* **Security ACL Metadata**: Security clearance level, allowed department user groups, and compliance categorization (PII, HIPAA).
""")

st.divider()

# ==============================================================================
# 3. Document Loaders
# ==============================================================================
st.header("3. Document Loaders: Extractors, Parsers, & Multimodal Ingestion", divider="rainbow")
st.markdown("""
**Document Loaders** connect directly to enterprise data repositories and parse proprietary file formats:
* **Unstructured Text Loaders**: PyPDF, PDFPlumber, PyMuPDF, python-docx, Unstructured.io.
* **Cloud Connectors**: SharePoint connectors, Google Drive API, AWS S3 listeners, Confluence integrations.
* **Multimodal Parsers**: Vision LLMs (e.g., ColPali, GPT-4o-mini) that convert complex scanned PDFs with diagrams into structured markdown representations.
""")

st.divider()

# ==============================================================================
# 4. Text Splitters
# ==============================================================================
st.header("4. Text Splitters: Recursive, Semantic, & Boundary-Aware Chunkers", divider="rainbow")
st.markdown("""
**Text Splitters** partition long document streams into discrete, semantically intact chunks:
""")

df_splitters = pd.DataFrame([
    {"Splitter Type": "Character Text Splitter", "Mechanism": "Splits on a fixed character count (e.g. every 1,000 chars).", "Trade-Off": "Fastest; frequently shreds sentences in half."},
    {"Splitter Type": "Recursive Character Splitter", "Mechanism": "Splits iteratively on paragraphs, newlines, and spaces to preserve boundaries.", "Trade-Off": "The enterprise standard default; balances speed and cohesion."},
    {"Splitter Type": "Semantic Splitter", "Mechanism": "Measures semantic distance between adjacent sentences and cuts when distance spikes.", "Trade-Off": "High semantic purity; requires running embedding models during ingestion."},
    {"Splitter Type": "Markdown / HTML Header Splitter", "Mechanism": "Splits along document header tags (#, ##, <h3>).", "Trade-Off": "Preserves hierarchy and attaches parent titles to chunks."}
])
st.dataframe(df_splitters, use_container_width=True)

st.divider()

# ==============================================================================
# 5. Embedding Models
# ==============================================================================
st.header("5. Embedding Models: Dense Vector Representation & Geometric Spaces", divider="rainbow")
st.markdown("""
**Embedding Models** project textual passages into dense continuous vector spaces (e.g. 768 or 1536 dimensions):
* **Bi-Encoder Architecture**: Converts queries and document chunks independently into static vector representations.
* **Leaderboards (MTEB)**: Evaluated on Massive Text Embedding Benchmark across retrieval, classification, and clustering tasks.
* **Multilingual Embeddings**: High-performance models (e.g., Cohere embed-v3, BAAI bge-m3) align concepts across dozens of languages in the same geometric space.
""")

st.divider()

# ==============================================================================
# 6. Vector Databases
# ==============================================================================
st.header("6. Vector Databases: Distributed Storage & High-Dimensional ANN Indexes", divider="rainbow")
st.markdown("""
**Vector Databases** provide high-throughput persistence, metadata filtering, and sub-second similarity search:
* **Dedicated Engines**: Milvus, Qdrant, Pinecone, ChromaDB, Weaviate.
* **Relational / Hybrid Extensions**: pgvector (PostgreSQL), Elasticsearch Dense Vector, OpenSearch.
* **Core Capabilities**: Real-time CRUD operations, multi-tenant RBAC filtering, HNSW graph indexing, and zero-downtime replication.
""")

st.divider()

# ==============================================================================
# 7. Retrievers
# ==============================================================================
st.header("7. Retrievers: Sparse BM25, Dense Vector, & Hybrid Fusion Fabrics", divider="rainbow")
st.markdown("""
The **Retriever** encapsulates the search logic that accepts a user query and returns candidate chunks:
* **Vector Retriever**: Computes semantic similarity over ANN graphs.
* **Sparse Lexical Retriever**: Computes BM25 keyword matching over inverted word indexes.
* **Hybrid Fusion Retriever**: Queries both vector and BM25 systems concurrently, merging rankings via Reciprocal Rank Fusion (RRF).
""")

st.divider()

# ==============================================================================
# 8. Rerankers
# ==============================================================================
st.header("8. Rerankers: Cross-Encoder Precision Filtering", divider="rainbow")
st.markdown("""
While vector search is fast (bi-encoder), it compresses passages into fixed vectors, losing inter-token interactions. 
A **Cross-Encoder Reranker** (Cohere Rerank, BGE-Reranker-Large) evaluates the query and candidate chunk jointly:
""")

with st.expander("🔍 View Bi-Encoder vs Cross-Encoder Architecture Diagram", expanded=True):
    dot_bi_cross = """
    digraph BiVsCross {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_bi {
            label="Bi-Encoder (First-Stage Retriever)";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            q_bi [label="Query Vector", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            doc_bi [label="Document Vector (Precomputed)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            dot_comp [label="Fast Vector Dot Product Comparison (sub-10ms)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            q_bi -> dot_comp;
            doc_bi -> dot_comp;
        }

        subgraph cluster_cross {
            label="Cross-Encoder (Second-Stage Reranker)";
            bgcolor="#1E293B";
            fontcolor="#EC4899";
            color="#334155";
            fontsize=12;
            joint_in [label="Concatenated Input: [CLS] Query [SEP] Document [SEP]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#EC4899"];
            deep_attn [label="All-to-All Self-Attention Layers across Query and Passage", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            score_out [label="High-Precision Relevance Logit (Filters False Positives)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            joint_in -> deep_attn -> score_out;
        }
    }
    """
    st.graphviz_chart(dot_bi_cross, use_container_width=True)

st.markdown("""
Rerankers filter 50 first-stage candidates down to the top 3-5 high-signal chunks, eliminating false positives 
before prompt injection.
""")

st.divider()

# ==============================================================================
# 9. Large Language Models
# ==============================================================================
st.header("9. Large Language Models: Grounded Reasoning & Context Synthesis Engines", divider="rainbow")
st.markdown("""
The **LLM** acts as the synthesis engine. In a RAG pipeline, the model's job is not to provide facts, but to:
1. Synthesize coherent answers strictly bounded by the supplied context chunks.
2. Resolve conflicting evidence across multiple retrieved sources.
3. Formulate structured outputs (JSON, citations, tables) requested by the user.
""")

st.divider()

# ==============================================================================
# 10. Prompt Templates
# ==============================================================================
st.header("10. Prompt Templates: Strict Evidence Bounding & Citation Frameworks", divider="rainbow")
st.markdown("""
**Prompt Templates** enforce the operational contract between the retriever and the generator. 
A production prompt template must include:
* **System Persona**: Clarifying domain boundaries.
* **Delimited Context Block**: Enclosing retrieved chunks in unambiguous XML/Markdown tags.
* **Negative Constraint Directive**: *"If the answer cannot be established from the context, state 'Information not found.'"*
* **Citation Directive**: *"Attach [DocID] citations to every factual claim."*
""")

st.divider()

# ==============================================================================
# 11. Evaluation Systems
# ==============================================================================
st.header("11. Evaluation Systems: The RAG Triad, Ragas, & Automated Faithfulness Audits", divider="rainbow")
st.markdown("""
You cannot improve what you do not measure. Production RAG stacks integrate automated evaluation telemetry:
""")

df_rag_triad = pd.DataFrame([
    {"Evaluation Metric": "Context Relevance", "What it Measures": "Does the retriever fetch chunks relevant to the query?", "Failure Indicator": "Low score means poor embedding or noisy chunking."},
    {"Evaluation Metric": "Groundedness / Faithfulness", "What it Measures": "Is every statement in the answer supported by retrieved chunks?", "Failure Indicator": "Low score indicates model hallucination."},
    {"Evaluation Metric": "Answer Relevance", "What it Measures": "Does the answer directly answer the user's question?", "Failure Indicator": "Low score means the model drifted or gave evasive answers."}
])
st.dataframe(df_rag_triad, use_container_width=True)

st.divider()

# ==============================================================================
# 12. Production Implementation
# ==============================================================================
st.header("12. Production Implementation: Pluggable Modular Component Orchestrator", divider="rainbow")
st.markdown("""
Below is a runnable Python production implementation of a **Modular RAG Stack Orchestrator** demonstrating 
clean separation of concerns across the components:
""")

prod_components_code = """
from typing import List, Dict, Any

class MockDocument:
    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata

class ModularRAGStack:
    def __init__(self):
        self.documents: List[MockDocument] = []
        self.vector_index: List[Dict[str, Any]] = []

    def loader(self, file_path: str, raw_text: str, tenant_id: str):
        \"\"\"Component 2: Document Loader with Metadata\"\"\"
        doc = MockDocument(
            doc_id=file_path,
            content=raw_text,
            metadata={"tenant_id": tenant_id, "file_path": file_path}
        )
        self.documents.append(doc)

    def text_splitter(self, doc: MockDocument, chunk_size: int = 20) -> List[Dict[str, Any]]:
        \"\"\"Component 3: Text Splitter\"\"\"
        words = doc.content.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            slice_str = " ".join(words[i:i+chunk_size])
            chunks.append({
                "chunk_id": f"{doc.doc_id}_c{i//chunk_size}",
                "text": slice_str,
                "metadata": doc.metadata
            })
        return chunks

    def index(self, chunk_size: int = 20):
        \"\"\"Components 4 & 5: Embedding & Vector Storage\"\"\"
        self.vector_index = []
        for doc in self.documents:
            chunks = self.text_splitter(doc, chunk_size)
            self.vector_index.extend(chunks)

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        \"\"\"Component 6: Retriever\"\"\"
        q_tokens = set(query.lower().split())
        scored = []
        for item in self.vector_index:
            tokens = set(item["text"].lower().split())
            score = len(q_tokens.intersection(tokens))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored[:top_k] if score > 0]

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Component 7: Cross-Encoder Reranker\"\"\"
        for c in candidates:
            c["rerank_score"] = round(0.85 + (0.1 if query.split()[0].lower() in c["text"].lower() else 0.0), 3)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates
"""
st.code(prod_components_code, language="python")

st.divider()

# ==============================================================================
# 13. Interactive Workbench
# ==============================================================================
st.header("13. Interactive Workbench: Component Swap & Pipeline Simulator", divider="rainbow")
st.markdown("""
Configure individual components of the RAG stack and observe how chunk size, embedding dimensionality, 
and reranking activation impact retrieval output and latency:
""")

col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    chunk_opt = st.selectbox("Text Splitter Granularity:", ["Small (10 words)", "Medium (25 words)", "Large (Full Text)"])
with col_cfg2:
    embed_dim = st.selectbox("Embedding Model Dimension:", [384, 768, 1536, 3072])
with col_cfg3:
    use_reranker = st.toggle("Enable Cross-Encoder Reranker (Stage 7)", value=True)

test_comp_query = st.text_input("Enter Query to Test Component Interaction:", value="What are production Kubernetes egress policy requirements?")

if st.button("Run Pluggable Pipeline", type="primary"):
    pipe = PluggableComponentPipeline(chunk_opt, embed_dim, use_reranker)
    out = pipe.run_pipeline(test_comp_query)
    
    st.markdown(f"**Total Document Chunks Produced**: `{out['total_chunks_produced']}`")
    st.markdown(f"**Estimated End-to-End Latency**: `{out['estimated_latency_ms']} ms`")
    
    st.subheader("Selected & Ranked Chunks:")
    for item in out["top_selected"]:
        st.success(f"**[{item['chunk_id']}]** — Score: `{item['final_score']}` ({item['reranker_status']})\n\n\"{item['text']}\"")

st.divider()

# ==============================================================================
# 14. Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("14. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why are bi-encoders fast for retrieval while cross-encoders are slow?"):
    st.markdown("""
    **Answer**: **Bi-encoders** embed queries and documents independently into static vectors offline. At query time, similarity 
    is computed via a fast vector comparison over an indexed graph (sub-10ms). **Cross-encoders** concatenate the query and candidate 
    passage, passing them jointly through full all-to-all transformer self-attention layers. This produces superior relevance scores 
    but costs 50-100x more compute per pair, restricting rerankers to evaluating only the top 50 candidates.
    """)

with st.expander("Q2: Explain the trade-offs of using a relational vector extension (pgvector) versus a dedicated vector database (Milvus/Qdrant)."):
    st.markdown("""
    **Answer**: **pgvector** is ideal for small-to-medium enterprise applications (< 5M vectors) because it reuses existing PostgreSQL ACID transactions, 
    relational schemas, and operational backups with zero new infrastructure. **Dedicated vector databases** (Milvus, Qdrant) excel at billion-scale 
    vector workloads, offering dedicated memory quantization, distributed sharding, and GPU acceleration.
    """)

with st.expander("Q3: How does Ragas evaluate RAG pipeline quality without ground-truth labels?"):
    st.markdown("""
    **Answer**: Ragas uses an **LLM-as-a-Judge** framework with specialized prompt contracts to compute heuristics:
    * **Faithfulness**: Breaks generated answers into atomic claims and verifies the percentage of claims directly entailed by retrieved context.
    * **Answer Relevance**: Generates synthetic questions from the output answer and calculates semantic similarity with the original query.
    * **Context Precision**: Evaluates whether ground-truth evidence ranks near the top of retrieved candidates.
    """)

with st.expander("Q4: What is the difference between semantic chunking and recursive character chunking?"):
    st.markdown("""
    **Answer**: Recursive character chunking splits on structural delimiters (paragraphs, newlines, spaces) to stay within a target character budget. 
    Semantic chunking embeds consecutive sentences and calculates distance between sentence transitions, creating a split boundary 
    only when semantic topicality shifts significantly.
    """)

with st.expander("Q5: Why are prompt templates considered a critical security component in RAG?"):
    st.markdown("""
    **Answer**: Prompt templates are the primary defense against **Indirect Prompt Injection**. If an attacker poisons a public document 
    with hidden malicious instructions, a well-designed prompt template places retrieved text inside isolated XML delimiters 
    (`<context>...</context>`) and instructs the model to treat context strictly as passive data rather than executable instructions.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The Unsynchronized Embedding Dimension Mismatch)**: Changing your embedding model from 768-dim to 1536-dim without rebuilding your vector database collection, causing runtime dimension mismatch exceptions.")
    st.warning("**Trap 2 (Neglecting Metadata Payload Indexing)**: Storing tenant IDs, dates, and category tags as unstructured JSON without enabling database payload indexing, forcing the vector database to perform slow brute-force scans after vector search.")
with col_trap2:
    st.error("**Trap 3 (Indirect Prompt Injection via Poisoned Retrieval)**: Ingesting uncurated external web scrapes containing adversarial prompt override instructions directly into prompt templates without sanitization.")
    st.warning("**Trap 4 (The Single Reranker Latency Bottleneck)**: Sending 200 candidate chunks to a heavy cross-encoder reranker on every query, adding 1,500ms of latency and destroying user experience.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the difference between a Bi-Encoder and a Cross-Encoder?")
        st.caption("Flip for answer:")
        st.info("**Back**: Bi-Encoders embed query and document separately for fast vector search. Cross-Encoders pass query and document jointly through full attention layers for high-precision reranking.")
    with st.container(border=True):
        st.markdown("**Front**: What are the 10 components of an enterprise RAG stack?")
        st.caption("Flip for answer:")
        st.info("**Back**: 1. Docs · 2. Loaders · 3. Splitters · 4. Embeddings · 5. VectorDB · 6. Retrievers · 7. Rerankers · 8. Prompts · 9. LLMs · 10. Eval.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What does the Ragas Framework measure?")
        st.caption("Flip for answer:")
        st.info("**Back**: Automated evaluation of Faithfulness, Context Relevance, and Answer Relevance using an LLM-as-a-judge framework.")
    with st.container(border=True):
        st.markdown("**Front**: What is Indirect Prompt Injection in RAG?")
        st.caption("Flip for answer:")
        st.info("**Back**: Malicious adversarial instructions hidden in ingested documents designed to hijack the LLM generation prompt.")

st.divider()

# ==============================================================================
# 15. Self-Assessment Checklist
# ==============================================================================
st.header("15. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I can articulate all ten primary components of a production RAG system architecture.", value=True)
c2 = st.checkbox("I understand the architectural distinction between bi-encoder retrievers and cross-encoder rerankers.", value=True)
c3 = st.checkbox("I know how to structure document metadata payloads to enable tenant isolation and ACL filtering.", value=True)
c4 = st.checkbox("I understand the trade-offs of recursive character chunking vs semantic boundary chunking.", value=True)
c5 = st.checkbox("I can define the three metrics of the RAG Triad (Context Relevance, Faithfulness, Answer Relevance).", value=True)
c6 = st.checkbox("I know how to design secure prompt templates that isolate retrieved context against indirect prompt injection.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Chapter 2 Complete! You have mastered RAG Fundamentals and are ready for Chapter 3: Setting Up a RAG Environment.")
