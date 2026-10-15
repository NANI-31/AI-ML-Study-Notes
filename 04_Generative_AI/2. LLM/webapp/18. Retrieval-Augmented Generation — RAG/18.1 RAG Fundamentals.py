import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="18.1 RAG Fundamentals - Retrieval-Augmented Generation",
    page_icon="🔍",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class MinimalRAGEngine:
    """
    Self-contained in-memory demonstration engine for RAG Fundamentals:
    - Document corpus indexing with simulated dense embedding vectors
    - Cosine similarity calculation
    - Top-k retrieval with metadata filtering
    - Context-augmented prompt construction with source attribution
    """
    
    CORPUS = [
        {
            "id": "DOC-001",
            "title": "Enterprise Cloud Migration SLA",
            "text": "Enterprise cloud tier guarantees 99.99% uptime with 15-minute response time for Priority 1 outages. Backups run hourly across three availability zones.",
            "category": "Infrastructure",
            "embedding": np.array([0.85, 0.12, 0.45, 0.08, 0.22], dtype=np.float32)
        },
        {
            "id": "DOC-002",
            "title": "Corporate Hardware Return Policy",
            "text": "Employees departing the company must return laptops and access badges within 7 business days via prepaid courier. Retaining devices beyond 14 days incurs payroll deduction.",
            "category": "HR_Policy",
            "embedding": np.array([0.10, 0.88, 0.15, 0.40, 0.18], dtype=np.float32)
        },
        {
            "id": "DOC-003",
            "title": "Quantum Encryption Deployment Roadmap",
            "text": "Post-quantum cryptography (Kyber-768 and Dilithium) will replace legacy RSA-2048 keys across all production API gateways starting Q3 2026.",
            "category": "Security",
            "embedding": np.array([0.25, 0.30, 0.92, 0.10, 0.05], dtype=np.float32)
        },
        {
            "id": "DOC-004",
            "title": "Internal Travel Reimbursement Limits",
            "text": "Daily meal allowance is capped at $75 per diem for domestic travel and $120 for international travel. Receipts are mandatory for expenses exceeding $25.",
            "category": "Finance",
            "embedding": np.array([0.05, 0.75, 0.10, 0.65, 0.12], dtype=np.float32)
        },
        {
            "id": "DOC-005",
            "title": "Priority 1 Incident Escalation Matrix",
            "text": "For P1 outages, the incident commander must page the on-call Site Reliability Engineer within 5 minutes and post a customer-facing status bulletin within 20 minutes.",
            "category": "Infrastructure",
            "embedding": np.array([0.80, 0.18, 0.50, 0.12, 0.30], dtype=np.float32)
        }
    ]
    
    # Query embedding lookup table for realistic simulation
    QUERY_EMBEDDINGS = {
        "What is the SLA uptime guarantee and P1 response time?": np.array([0.88, 0.10, 0.42, 0.05, 0.20], dtype=np.float32),
        "What are the rules for returning corporate laptops?": np.array([0.12, 0.85, 0.12, 0.42, 0.15], dtype=np.float32),
        "When will post-quantum cryptography replace RSA keys?": np.array([0.22, 0.28, 0.95, 0.08, 0.04], dtype=np.float32),
        "What is the daily meal budget for domestic business travel?": np.array([0.08, 0.72, 0.08, 0.68, 0.10], dtype=np.float32)
    }

    @classmethod
    def cosine_similarity(cls, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @classmethod
    def retrieve(
        cls,
        query: str,
        top_k: int = 2,
        category_filter: str = "All"
    ) -> List[Dict[str, Any]]:
        query_vec = cls.QUERY_EMBEDDINGS.get(query, np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        
        scored_docs = []
        for doc in cls.CORPUS:
            if category_filter != "All" and doc["category"] != category_filter:
                continue
            sim = cls.cosine_similarity(query_vec, doc["embedding"])
            doc_entry = dict(doc)
            doc_entry["score"] = sim
            scored_docs.append(doc_entry)
            
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    @classmethod
    def construct_augmented_prompt(cls, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_blocks.append(f"[{doc['id']}] (Source: {doc['title']})\n{doc['text']}")
        
        joined_context = "\n\n".join(context_blocks)
        prompt = (
            "You are an enterprise AI assistant. Answer the user query using ONLY the verified context below.\n"
            "If the context does not contain the answer, state 'Insufficient verified context available.'\n"
            "Always cite the source document ID [DOC-XXX] for every claim made.\n\n"
            f"--- BEGIN CONTEXT ---\n{joined_context}\n--- END CONTEXT ---\n\n"
            f"User Question: {query}\n\nAnswer:"
        )
        return prompt

    @classmethod
    def generate_simulated_answer(cls, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "Insufficient verified context available to answer this question accurately."
        
        top_doc = retrieved_docs[0]
        if "SLA uptime" in query and "DOC-001" in [d["id"] for d in retrieved_docs]:
            return "According to enterprise policy [DOC-001], the cloud tier guarantees a 99.99% uptime SLA with a 15-minute response time for Priority 1 outages. Backups are executed hourly across 3 availability zones."
        elif "laptop" in query and "DOC-002" in [d["id"] for d in retrieved_docs]:
            return "Departing employees must return laptops and badges within 7 business days via prepaid courier [DOC-002]. Holding hardware past 14 days results in payroll deductions."
        elif "post-quantum" in query and "DOC-003" in [d["id"] for d in retrieved_docs]:
            return "Per the security roadmap [DOC-003], post-quantum algorithms (Kyber-768 and Dilithium) will officially replace legacy RSA-2048 keys across production API gateways starting in Q3 2026."
        elif "meal budget" in query and "DOC-004" in [d["id"] for d in retrieved_docs]:
            return "The corporate per diem meal allowance is capped at $75 per day for domestic travel [DOC-004]. Receipts are strictly required for any individual expense over $25."
        else:
            return f"Based on retrieved documentation [{top_doc['id']}]: {top_doc['text']}"


# --- HEADER & INTRODUCTION ---
st.title("18.1 RAG Fundamentals: Parametric vs Non-Parametric Memory & Core Pipelines")
st.caption("Chapter 18: Retrieval-Augmented Generation — RAG | Domain: 2. Large Language Models")

st.markdown("""
A foundation language model's weights act as a **static parametric memory**: they encode linguistic structure, 
reasoning frameworks, and broad world knowledge frozen at the exact second pretraining finished.

However, enterprise generative AI faces three fatal barriers:
1. **Knowledge Cutoffs**: The model is completely blind to events, policies, and code written after its pretraining date.
2. **Hallucination Risk**: When queried on unlearned facts, autoregressive decoding prioritizes plausible phrasing over truth, inventing fictitious citations.
3. **Proprietary & Regulated Data**: Corporate intranets, private customer records, and confidential PDFs cannot be baked into public model weights.

**Retrieval-Augmented Generation (RAG)** resolves this paradox by combining the generative reasoning power of an LLM with 
an external, dynamic **non-parametric memory** (vector search databases, keyword inverted indexes, and document stores). 
This lesson establishes the architectural foundation of RAG, the dual-pipeline lifecycle, and the strategic calculus of RAG versus fine-tuning.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: The Dual-Pipeline RAG Lifecycle & Parametric/Non-Parametric Continuum", divider="rainbow")

st.markdown("""
A production RAG system is not a single script; it is two asynchronous, decoupled distributed pipelines: 
the **Offline Ingestion Pipeline** and the **Online Inference Pipeline**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    rag_blueprint_dot = """
    digraph RAGBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155", fontsize=11, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10, penwidth=1.5];

        subgraph cluster_ingestion {
            label="Pipeline 1: Asynchronous Document Ingestion (Offline / Batch)";
            style="rounded";
            bgcolor="#151E32";
            color="#38BDF8";
            fontcolor="#38BDF8";
            fontsize=12;

            Docs [label="Enterprise Document Store\\n(PDFs, Markdown, Web Crawls, SQL Records)", fillcolor="#1E293B", color="#38BDF8"];
            Parser [label="Document Parsing & Normalization\\n(Extracts Clean Text, Tables, Document Metadata)", fillcolor="#334155", color="#64748B"];
            Chunker [label="Semantic Chunking Engine\\n(512-Token Units with 10% Overlap)", fillcolor="#0F766E", color="#14B8A6"];
            Embedder [label="Dense Embedding Model\\n(Generates 1536-Dimensional Vectors)", fillcolor="#0369A1", color="#38BDF8"];
            VectorDB [label="Enterprise Hybrid Vector Index\\n(HNSW Vector Index + Inverted BM25 Index)", fillcolor="#4338CA", color="#818CF8", shape=cylinder];

            Docs -> Parser -> Chunker -> Embedder -> VectorDB;
        }

        subgraph cluster_inference {
            label="Pipeline 2: Synchronous Retrieval & Generation (Online / User Facing)";
            style="rounded";
            bgcolor="#151E32";
            color="#10B981";
            fontcolor="#10B981";
            fontsize=12;

            UserQuery [label="User Ingress Query\\n'What is our enterprise cloud SLA?'", fillcolor="#1E293B", color="#38BDF8"];
            QueryEncoder [label="Query Dense Encoder\\n(Creates Query Representation Vector)", fillcolor="#0369A1", color="#38BDF8"];
            Retriever [label="Hybrid Retrieval Engine\\n(Parallel ANN Dense + BM25 Sparse -> Top-20 Candidates)", fillcolor="#0F766E", color="#14B8A6"];
            Reranker [label="Cross-Encoder Precision Reranker\\n(Joint Attention Scoring -> Top-3 Grounded Chunks)", fillcolor="#6D28D9", color="#A78BFA"];
            PromptBuilder [label="Context Augmentation & Prompt Assembler\\n(Injects Grounded System Persona + Verified Excerpts)", fillcolor="#334155", color="#64748B"];
            LLM [label="Foundation LLM Reasoner\\n(Grounded Generation with Strict Citation Attribution)", fillcolor="#059669", color="#34D399"];
            Response [label="Verified Output with Exact Citations\\n[Source: DOC-001, Page 4, Section 3.2]", fillcolor="#047857", color="#10B981"];

            UserQuery -> QueryEncoder -> Retriever -> Reranker -> PromptBuilder -> LLM -> Response;
        }

        VectorDB -> Retriever [label="External Vector Match (Cosine >= 0.75)", style="dashed", color="#F59E0B", fontcolor="#FCD34D"];
        UserQuery -> PromptBuilder [label="Original Question", style="dotted", color="#94A3B8"];
    }
    """
    st.graphviz_chart(rag_blueprint_dot, use_container_width=True)


st.markdown("""
### The Knowledge Divide: Parametric vs Non-Parametric Memory
""")

knowledge_matrix = [
    {
        "Dimension": "Storage Mechanism",
        "Parametric Memory (Model Weights)": "Static floating-point matrices (billions of parameters)",
        "Non-Parametric Memory (Vector Store)": "External indices, vector databases, document stores"
    },
    {
        "Dimension": "Update Frequency & Latency",
        "Parametric Memory (Model Weights)": "Weeks to months (requires retraining / fine-tuning)",
        "Non-Parametric Memory (Vector Store)": "Milliseconds (instant write/delete operations via API)"
    },
    {
        "Dimension": "Fact Verification & Auditability",
        "Parametric Memory (Model Weights)": "Black box; cannot reliably trace facts back to specific source training tokens",
        "Non-Parametric Memory (Vector Store)": "Exact deterministic source attribution with chunk IDs and page numbers"
    },
    {
        "Dimension": "Security & Permissions (RBAC)",
        "Parametric Memory (Model Weights)": "Impossible to restrict; all users share the same weight matrices",
        "Non-Parametric Memory (Vector Store)": "Native document metadata filtering enforcing user role access control"
    },
    {
        "Dimension": "Primary Strength",
        "Parametric Memory (Model Weights)": "Syntactic fluency, multi-step logic, code generation, reasoning",
        "Non-Parametric Memory (Vector Store)": "Factual accuracy, proprietary knowledge, zero hallucination"
    }
]
st.dataframe(pd.DataFrame(knowledge_matrix), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - WHAT IS RAG? ---
st.header("2. What is RAG? (Retrieval-Augmented Generation Origins & Mechanics)", divider="rainbow")

st.markdown("""
Introduced by Patrick Lewis et al. (Facebook AI Research / UCL / NYU, 2020), **Retrieval-Augmented Generation (RAG)** 
fundamentally changed how machine learning engineers build question-answering and knowledge-intensive systems.

Prior to RAG, NLP engineers either:
1. Relied purely on extractive search engines (returning a raw paragraph without synthesizing an answer), or
2. Relied purely on generative language models (generating fluent answers riddled with hallucinated facts).

### The Mathematical Formulation (Lewis et al., 2020)
Lewis et al. unified retrieval and generation into a single probabilistic framework. 
Given an input prompt `x`, the model accesses an external index of document chunks `z in Z`:

```text
P(y | x) = Sum_{z in Top-K} P(z | x) * P(y | x, z)
```

Where:
- `P(z | x)` is the **Retrieval Probability**: computed using dense vector similarity (Dense Passage Retrieval / DPR) between the query `x` and document chunk `z`.
- `P(y | x, z)` is the **Generation Probability**: the autoregressive probability of token `y` produced by the sequence-to-sequence model conditioned on both the prompt `x` and retrieved context `z`.

### RAG-Sequence vs RAG-Token
The authors distinguished two primary formulation modes:
- **RAG-Sequence**: The model retrieves top-k documents once for the entire answer, generating the full sequence using the same retrieved documents.
- **RAG-Token**: The model can theoretically retrieve different documents at every individual token generation step (computationally expensive, rarely used in production today).

Today, modern production RAG follows the **RAG-Sequence paradigm**: retrieve once (or multi-hop iteratively), assemble the context window, and generate the final completion.
""")

st.divider()

# --- SECTION 3: TOPIC 2 - WHY RAG IS NEEDED ---
st.header("3. Why RAG is Needed: Enterprise Accuracy, Privacy, & Real-Time Relevance", divider="rainbow")

st.markdown("""
Why does every Fortune 500 company adopt RAG as their primary LLM deployment architecture?
Because deploying pure foundation models in enterprise settings immediately encounters four existential risks:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. The Knowledge Refresh Dilemma")
    st.markdown("""
    - Fine-tuning a 70B parameter model on company updates costs thousands of dollars per run, takes days of compute, and introduces gradient instabilities.
    - Enterprise knowledge changes every second: bug tickets are closed, inventory prices fluctuate, legal contracts are signed.
    - **RAG decouples knowledge updates from model training**: indexing 10,000 new documents into a vector database takes seconds and costs less than $0.10.
    """)
    st.subheader("2. Compliance & Legal Attribution")
    st.markdown("""
    - In healthcare, finance, and law, an un-cited assertion is legally inadmissible and dangerous.
    - If an LLM recommends a medical dosage without citing the exact medical guideline document, the company faces severe regulatory penalties.
    - RAG guarantees that every sentence can be accompanied by an interactive citation linking directly to the source paragraph and page number.
    """)

with col2:
    st.subheader("3. Granular Access Control (RBAC)")
    st.markdown("""
    - An employee asking *"What is the severance compensation policy?"* should only see documents authorized for their specific clearance level.
    - Model weights cannot enforce access control: once a fact is fine-tuned into a model's weights, any user can extract it through clever prompting.
    - RAG enforces access control at query time: the vector search engine filters chunks by `user_role` before the LLM ever sees the text!
    """)
    st.subheader("4. Hallucination Suppression")
    st.markdown("""
    - Providing the LLM with the exact reference text in its context window drops hallucination rates by over 90%.
    - With strict system prompting (*"Only answer using the provided facts; say 'I don't know' otherwise"*), the LLM functions as an analytical summarizer rather than a creative writer.
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - LLM KNOWLEDGE LIMITATIONS ---
st.header("4. LLM Knowledge Limitations: Knowledge Cutoffs, Hallucinations, & Memory Fading", divider="rainbow")

st.markdown("""
To design resilient RAG architectures, an engineer must understand the intrinsic physical limitations of transformer memory:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### 1. The Pretraining Knowledge Cutoff
    A model pretrained with a cutoff of December 2024 has zero knowledge of any event, election, API update, 
    or vulnerability disclosed in 2025 or 2026. Without RAG, it will confidently answer queries about modern events 
    using obsolete historical analogies.

    ### 2. Catastrophic Forgetting in Continual Training
    If you attempt to keep an LLM updated by continuously fine-tuning it on new daily news articles:
    - The new gradient updates alter early attention weights.
    - The model experiences **Catastrophic Forgetting**: it learns the new news articles, but suddenly forgets how to write Python code or solves basic math problems!
    """)
with col2:
    st.markdown("""
    ### 3. The 'Lost in the Middle' Phenomenon
    Modern frontier LLMs support context windows of 128,000 to 2,000,000 tokens (e.g. Gemini 1.5 Pro). 
    Why not simply dump an entire company's documentation into the prompt?
    - **Liu et al. (Stanford, 2023)** proved that transformer attention exhibits strong U-shaped bias: 
      models retrieve facts located at the very beginning and very end of the prompt with high accuracy, 
      but performance drops by over 30% when the critical fact is buried in the middle of a massive context!
    - **Economic & Latency Costs**: Processing 1,000,000 tokens per query costs over $1.00 and takes 10+ seconds to prefill. RAG retrieves only the 3 critical chunks (1,500 tokens), executing in milliseconds for fractions of a cent.
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - EXTERNAL KNOWLEDGE ---
st.header("5. External Knowledge: Structured, Unstructured, & Semi-Structured Modalities", divider="rainbow")

st.markdown("""
External knowledge ingested by RAG systems spans three disparate data modalities, each requiring specialized parsing pipelines:
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    ### Unstructured Text
    - **Formats**: PDFs, Word documents (.docx), text files, scanned paperwork, customer emails.
    - **Characteristics**: Free-flowing natural language, complex multi-column layouts, embedded diagrams, variable formatting.
    - **Target Index**: Dense vector embeddings (Cosine similarity) paired with keyword BM25 indexes.
    """)

with col2:
    st.markdown("""
    ### Semi-Structured Data
    - **Formats**: Markdown files, HTML web pages, JSON API logs, Jira tickets, Slack threads.
    - **Characteristics**: Clear structural markers (headers, bullet points, author tags, timestamps).
    - **Target Index**: Hierarchical chunking preserving header breadcrumbs (`# Chapter > ## Section`) for rich retrieval context.
    """)

with col3:
    st.markdown("""
    ### Structured Tabular Data
    - **Formats**: Relational SQL databases, CSV spreadsheets, ERP tables, Knowledge Graphs.
    - **Characteristics**: Strict schemas, foreign keys, quantitative numbers.
    - **Target Index**: Text-to-SQL generation or GraphRAG (knowledge graph entity-relationship traversal).
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - THE RETRIEVAL PIPELINE ---
st.header("6. The Retrieval Pipeline: Ingestion, Chunking, Embedding, & Indexing", divider="rainbow")

st.markdown("""
The **Offline Ingestion Pipeline** processes raw multi-format files into searchable mathematical vectors stored inside an index.
Every stage in this pipeline introduces trade-offs that determine retrieval precision:
""")

ingestion_stages = """
digraph IngestionDetail {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    RawDoc [label="1. Raw Documents\\n(PDFs, HTML, Markdown)", fillcolor="#1E293B"];
    Extract [label="2. Extraction & Cleaning\\nStrip HTML tags, OCR images,\\nclean control characters", fillcolor="#334155"];
    Chunk [label="3. Chunking Strategy\\nSplit into 512 tokens\\n10% overlapping boundary", fillcolor="#0F766E"];
    Embed [label="4. Vector Embedding\\nDense representation model\\n(e.g., text-embedding-3)", fillcolor="#0369A1"];
    Index [label="5. Approximate Nearest Neighbor Index\\nHNSW (Hierarchical Navigable Small World)\\n+ Metadata Store", fillcolor="#4338CA"];

    RawDoc -> Extract -> Chunk -> Embed -> Index;
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(ingestion_stages, use_container_width=True)

st.markdown("""
### Critical Ingestion Engineering Considerations
1. **Garbage-In, Garbage-Out**: If a PDF parser scrambles a table or misinterprets a two-column layout as a single horizontal line, the embedded chunk will contain gibberish. Clean text extraction accounts for 60% of real-world RAG success.
2. **Chunk Boundary Dilution**: Chunks that are too small (50 tokens) lose surrounding context; chunks that are too large (2,000 tokens) dilute specific facts inside a sea of irrelevant text.
3. **Index Refresh Strategies**: Production systems implement **Change Data Capture (CDC)**: webhooks detect document edits in SharePoint, Notion, or GitHub, immediately re-embedding only the modified sections.
""")

st.divider()

# --- SECTION 7: TOPIC 6 - THE GENERATION PIPELINE ---
st.header("7. The Generation Pipeline: Query Encoding, Context Assembly, & In-Context Synthesis", divider="rainbow")

st.markdown("""
The **Online Generation Pipeline** runs synchronously when a user submits a query. It executes in five precise micro-steps:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### The 5 Online Generation Steps
    1. **Query Encoding**: Convert the user's natural language question into a dense query vector `q_vec in R^D` using the same embedding model used during ingestion.
    2. **Candidate Search**: Execute fast Approximate Nearest Neighbor (ANN) search across the vector database (fetching top-20 chunks).
    3. **Reranking**: Pass the query and 20 candidates through a high-precision Cross-Encoder model to select the top-3 to top-5 most semantically relevant chunks.
    4. **Context Injection**: Format retrieved chunks into a structured system prompt template with explicit ground-truth instructions.
    5. **Synthesis with Citations**: The LLM streams the completion, appending verifiable brackets `[DOC-XXX]` corresponding to the specific evidence used.
    """)

with col2:
    st.markdown("""
    ### Strategic Decision Matrix: RAG vs Fine-Tuning
    """)
    rag_vs_ft = [
        {"Criterion": "Goal", "RAG": "Inject dynamic external knowledge", "Fine-Tuning": "Adapt style, tone, syntax, or format"},
        {"Criterion": "Data Freshness", "RAG": "Real-time (instant updates)", "Fine-Tuning": "Static snapshot at time of training"},
        {"Criterion": "Hallucination Risk", "RAG": "Low (verifiable in prompt)", "Fine-Tuning": "High (unverifiable memory)"},
        {"Criterion": "Cost per Update", "RAG": "Pennies (vector insertion)", "Fine-Tuning": "Hundreds to thousands in GPU hours"},
        {"Criterion": "Setup Complexity", "RAG": "Medium (Vector DB, chunking)", "Fine-Tuning": "High (Dataset curation, eval)"}
    ]
    st.dataframe(pd.DataFrame(rag_vs_ft), use_container_width=True)

st.divider()

# --- SECTION 8: PRODUCTION CODE ---
st.header("8. Production Implementation: End-to-End Minimal RAG Pipeline in Pure Python", divider="rainbow")

st.markdown("""
The following complete, dependency-free Python script implements a production-grade minimal RAG engine. 
It defines in-memory document indexing, dense vector similarity retrieval, metadata filtering, prompt formatting, 
and grounded answer synthesis:
""")

code_block = '''import numpy as np
from typing import Dict, List, Any

class MinimalRAGPipeline:
    """
    Self-contained in-memory RAG pipeline implementing cosine similarity retrieval,
    metadata filtering, and context-injected prompt synthesis.
    """
    
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        
    def add_document(self, doc_id: str, title: str, text: str, category: str, embedding: np.ndarray):
        """Indexes a document chunk with its metadata and dense vector."""
        self.documents.append({
            "id": doc_id,
            "title": title,
            "text": text,
            "category": category,
            "embedding": embedding.astype(np.float32)
        })
        
    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two dense vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        
    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        category_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Retrieves top-k documents matching query vector, respecting metadata filters."""
        candidates = []
        for doc in self.documents:
            if category_filter and doc["category"] != category_filter:
                continue
            sim = self.cosine_similarity(query_vector, doc["embedding"])
            entry = dict(doc)
            entry["similarity_score"] = sim
            candidates.append(entry)
            
        # Rank by similarity descending
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]
        
    def build_prompt(self, user_query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Constructs an audited in-context prompt template."""
        context_blocks = []
        for chunk in retrieved_chunks:
            context_blocks.append(f"[{chunk['id']}] (Source: {chunk['title']})\n{chunk['text']}")
            
        context_str = "\n\n".join(context_blocks)
        
        prompt = (
            "You are a helpful, factual enterprise assistant. Use ONLY the verified context below.\n"
            "If the context does not provide the answer, say 'Insufficient verified context available.'\n"
            "Every fact must cite its source bracket [DOC-XXX].\n\n"
            f"--- BEGIN VERIFIED CONTEXT ---\n{context_str}\n--- END VERIFIED CONTEXT ---\n\n"
            f"Question: {user_query}\n\nAnswer:"
        )
        return prompt
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 9: WORKBENCH ---
st.header("9. Interactive Workbench: Retrieval Augmented Simulator & Parametric vs Non-Parametric Sandbox", divider="rainbow")

st.markdown("""
Experience the direct contrast between **Pure Parametric Generation** (without context, high hallucination risk) 
and **Retrieval-Augmented Generation** (grounded context with source citations) in real time:
""")

wb_col1, wb_col2, wb_col3 = st.columns(3)
with wb_col1:
    sample_queries = list(MinimalRAGEngine.QUERY_EMBEDDINGS.keys())
    selected_query = st.selectbox("Select Enterprise Query", sample_queries)
with wb_col2:
    selected_top_k = st.slider("Retrieval Top-K Chunks", min_value=1, max_value=5, value=2, step=1)
with wb_col3:
    cat_filter = st.selectbox("Metadata Role / Category Filter", ["All", "Infrastructure", "HR_Policy", "Security", "Finance"])

# Execute retrieval
retrieved_docs = MinimalRAGEngine.retrieve(selected_query, top_k=selected_top_k, category_filter=cat_filter)
augmented_prompt = MinimalRAGEngine.construct_augmented_prompt(selected_query, retrieved_docs)
simulated_answer = MinimalRAGEngine.generate_simulated_answer(selected_query, retrieved_docs)

# Display side-by-side comparison
res_col1, res_col2 = st.columns(2)
with res_col1:
    st.subheader("1. Without RAG: Pure Parametric Memory")
    st.error("""
    **Simulated Parametric Output (No Context)**:
    "Our enterprise cloud tier typically provides 99.9% uptime, and P1 response times are usually handled within 1 to 2 hours by support staff."
    """)
    st.caption("Notice the hallucination: the model used generic training knowledge and invented incorrect SLA numbers because it cannot see your private company documentation.")

with res_col2:
    st.subheader("2. With RAG: Grounded Non-Parametric Memory")
    st.success(f"""
    **Grounded RAG Output**:
    {simulated_answer}
    """)
    st.caption("Notice exact factual accuracy: the response quotes exact numbers and embeds verifiable source citations.")

st.markdown("### Retrieved Context Chunks (Vector Search Results)")
if retrieved_docs:
    table_data = []
    for d in retrieved_docs:
        table_data.append({
            "Doc ID": d["id"],
            "Title": d["title"],
            "Category": d["category"],
            "Cosine Similarity": f"{d['score']:.4f}",
            "Excerpt": d["text"][:80] + "..."
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
else:
    st.warning("No documents matched the query and category filter criteria.")

with st.expander("Inspect Raw Injected Prompt Template"):
    st.code(augmented_prompt, language="text")

st.divider()

# --- SECTION 10: INTERVIEW PREP ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "What is the fundamental difference between parametric and non-parametric memory in LLM systems?",
        "a": "Parametric memory is encoded inside the neural network's learned weights during pretraining. It is static, expensive to update (requiring retraining or fine-tuning), and acts as a black box where facts cannot be cleanly audited or restricted by access control. Non-parametric memory resides outside the neural network in external search indices, vector databases, or relational stores. It is dynamic, updated in milliseconds via standard CRUD operations, enforces granular Role-Based Access Control (RBAC), and allows deterministic verification of sources via chunk IDs and citations."
    },
    {
        "q": "Why does a 1-million-token context window not eliminate the need for RAG?",
        "a": "While models like Gemini 1.5 Pro can ingest 1M+ tokens, relying purely on massive context windows in production fails for three reasons: 1) Economic Cost: processing a million tokens per request costs dollars instead of fractions of a cent; 2) Latency: prompt prefill over 1M tokens takes several seconds to minutes; 3) The 'Lost in the Middle' phenomenon: transformers have an inherent attention bias toward tokens at the beginning and end of long contexts, suffering notable degradation when critical facts are buried in the middle. RAG acts as an intelligent filter, reducing 1M tokens of data down to the top-3 relevant chunks (1,500 tokens), yielding sub-second latency and pennies in cost."
    },
    {
        "q": "How did Lewis et al. (2020) formulate RAG mathematically?",
        "a": "Lewis et al. formulated RAG as a marginalization over top-K retrieved document chunks: P(y|x) = sum_{z in Top-K} P(z|x) * P(y|x, z). Here, P(z|x) is the retrieval probability computed by dense vector inner product between query x and document z (using Dense Passage Retrieval). P(y|x, z) is the autoregressive sequence generation probability conditioned on both query x and retrieved chunk z. They introduced RAG-Sequence (retrieves top-k documents once per answer) and RAG-Token (retrieves different documents at each generated token step)."
    },
    {
        "q": "When should an engineering team choose Fine-Tuning instead of RAG?",
        "a": "Choose Fine-Tuning when the primary objective is adapting style, tone, specialized syntax, vocabulary, or output formatting (e.g., teaching an LLM to generate valid SQL in an internal company dialect, or teaching a medical model clinical summary formatting). Choose RAG when the objective is providing new, dynamic, proprietary, or frequently updating facts. In production, the optimal pattern is often hybrid: fine-tune the model to understand domain jargon and output structure, then use RAG to inject factual source ground truth."
    },
    {
        "q": "What causes the 'lost-in-the-middle' phenomenon in transformer attention during RAG?",
        "a": "The lost-in-the-middle phenomenon (Liu et al., 2023) arises because standard self-attention mechanisms and positional encodings (such as RoPE or learned embeddings) exhibit an intrinsic bias toward the start (primacy bias) and end (recency bias) of the input sequence. Intermediate tokens receive lower attention weights across multi-layer aggregation. In production RAG, this is mitigated by ranking retrieved chunks and placing the most critical evidence at the very top or very bottom of the context block."
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
    **Trap 1: Confidential Data Leakage via Unfiltered Retrieval (Broken RBAC)**
    Ingesting sensitive documents (e.g. executive compensation, HR disciplinary records) into a global vector database without metadata access control lists (ACLs). 
    A non-privileged user asking *"What did the CEO say about upcoming layoffs?"* will retrieve confidential chunks into their context window. 
    Always apply pre-retrieval metadata filtering using user identity tokens.
    """)
    st.warning("""
    **Trap 3: Chunk Deduplication Neglect (Context Flooding)**
    Ingesting identical or near-duplicate documents across multiple directories or email chains without hashing or semantic deduplication. 
    At query time, the top-5 retrieved chunks may all be identical copies of the same boilerplate disclaimer, pushing the actual answer 
    out of the retrieved window. Always deduplicate at the chunking stage.
    """)

with col_trap2:
    st.error("""
    **Trap 2: Mismatched Query and Document Embedding Models**
    Using one embedding model (e.g. `text-embedding-3-small`) to index documents during offline ingestion, but using a different model 
    (e.g. `bge-large-en`) to embed the user query at runtime. The resulting vector coordinates inhabit completely different geometric spaces, 
    causing vector similarity scores to become pure random noise.
    """)
    st.warning("""
    **Trap 4: Hallucinated Citations in Generation**
    Asking the LLM to provide citations without enforcing strict bracket formats or validating that cited document IDs actually exist 
    in the injected context. LLMs will invent plausible fake citations like `[DOC-999]` if prompted loosely. Always validate citations programmatically post-generation.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: Lewis Formulation"):
        st.markdown("**What is the RAG marginalization formula?**\\n\\n`P(y|x) = sum P(z|x) * P(y|x, z)`\\n\\nMarginalizes over retrieved chunks z.")
with fc2:
    with st.popover("Flashcard 2: Memory Types"):
        st.markdown("**Parametric vs Non-Parametric?**\\n\\nParametric = Weights (static reasoning).\\nNon-Parametric = Vector DB (dynamic verified facts).")
with fc3:
    with st.popover("Flashcard 3: Lost in Middle"):
        st.markdown("**What is Lost in the Middle?**\\n\\nAttention bias prioritizing beginning and end of prompts over middle content.")
with fc4:
    with st.popover("Flashcard 4: RAG vs Fine-Tune"):
        st.markdown("**When to use which?**\\n\\nRAG for facts and real-time updates.\\nFine-tuning for style, tone, and grammar.")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT ---
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I can explain the difference between parametric (model weights) and non-parametric (external index) memory.",
    "I understand the mathematical formulation of RAG introduced by Lewis et al. (2020) and the difference between RAG-Sequence and RAG-Token.",
    "I know why RAG is essential for enterprise compliance, verifiable citations, and Role-Based Access Control (RBAC).",
    "I understand the physical causes of knowledge cutoffs, catastrophic forgetting, and the 'lost-in-the-middle' phenomenon.",
    "I can articulate the architectural trade-offs between RAG and Supervised Fine-Tuning across cost, latency, and data freshness.",
    "I can trace the 5 stages of offline document ingestion and the 5 stages of online query generation."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of RAG fundamentals, memory modalities, and core pipeline architectures.")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
