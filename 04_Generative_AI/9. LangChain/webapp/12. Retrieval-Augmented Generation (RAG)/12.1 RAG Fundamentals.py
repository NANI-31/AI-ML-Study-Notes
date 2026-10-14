import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class KnowledgeDocument:
    """Represents a chunked document item stored in the non-parametric knowledge base."""
    def __init__(self, doc_id: str, title: str, content: str, embedding_vector: np.ndarray, updated_date: str):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.embedding_vector = embedding_vector
        self.updated_date = updated_date

class MockRAGKnowledgeSimulator:
    """Simulates Parametric (Frozen LLM) vs Non-Parametric (RAG) knowledge retrieval."""
    def __init__(self):
        # Sample internal enterprise knowledge base with synthetic 4D embeddings
        self.knowledge_base = [
            KnowledgeDocument(
                "DOC-001", "Enterprise Travel & Expense Policy 2026",
                "Daily domestic per-diem meal allowance is capped at 85 USD. International hotel cap is 240 USD/night. Receipts mandatory above 25 USD.",
                np.array([0.85, 0.20, 0.10, 0.45]),
                "2026-01-15"
            ),
            KnowledgeDocument(
                "DOC-002", "Remote Work & Hardware Refresh SLA",
                "Employees are eligible for a laptop refresh every 24 months. Standard issue hardware is Apple M3 Pro or ThinkPad P1 Gen 6 with 32GB RAM.",
                np.array([0.15, 0.90, 0.25, 0.10]),
                "2026-03-01"
            ),
            KnowledgeDocument(
                "DOC-003", "Kubernetes Multi-Region Failover Architecture",
                "Production clusters use Cilium CNI with BGP mesh. Regional failover latency target is under 450ms across us-east-1 and eu-central-1.",
                np.array([0.10, 0.25, 0.95, 0.30]),
                "2025-11-20"
            )
        ]

    def _mock_query_embed(self, query: str) -> np.ndarray:
        q_lower = query.lower()
        if "travel" in q_lower or "expense" in q_lower or "meal" in q_lower:
            return np.array([0.82, 0.18, 0.12, 0.48])
        elif "laptop" in q_lower or "hardware" in q_lower or "remote" in q_lower:
            return np.array([0.18, 0.88, 0.22, 0.15])
        elif "kubernetes" in q_lower or "cluster" in q_lower or "failover" in q_lower:
            return np.array([0.12, 0.22, 0.92, 0.32])
        return np.array([0.30, 0.30, 0.30, 0.30])

    def retrieve(self, query: str, top_k: int = 1, min_similarity: float = 0.65) -> List[Tuple[KnowledgeDocument, float]]:
        q_vec = self._mock_query_embed(query)
        results = []
        for doc in self.knowledge_base:
            # Cosine similarity in pure numpy
            norm_q = np.linalg.norm(q_vec)
            norm_doc = np.linalg.norm(doc.embedding_vector)
            sim = float(np.dot(q_vec, doc.embedding_vector) / (norm_q * norm_doc))
            if sim >= min_similarity:
                results.append((doc, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def synthesize(self, query: str, use_rag: bool, top_k: int = 1, min_similarity: float = 0.65) -> Dict[str, Any]:
        if not use_rag:
            # Parametric-only: hallucination or outdated knowledge
            return {
                "mode": "Parametric Only (Frozen Weights, Training Cutoff 2023)",
                "answer": "Based on general information up to my training cutoff, corporate meal limits are usually around 50 USD, and hardware refreshes happen roughly every 3-4 years. (Note: These numbers are estimates and may not reflect your organization's real 2026 policy).",
                "citations": [],
                "retrieved_count": 0,
                "hallucination_risk": "HIGH",
                "grounded": False
            }

        retrieved = self.retrieve(query, top_k, min_similarity)
        if not retrieved:
            return {
                "mode": "RAG Augmented (Knowledge Base Lookup)",
                "answer": "I searched the internal enterprise knowledge base, but no verified documents exceeded the relevance threshold for this query. Refusing to speculate to prevent hallucinations.",
                "citations": [],
                "retrieved_count": 0,
                "hallucination_risk": "ZERO (Refusal on low confidence)",
                "grounded": True
            }

        doc, sim = retrieved[0]
        context_str = f"[{doc.doc_id}: {doc.title} (Updated {doc.updated_date})]: {doc.content}"
        augmented_answer = f"According to verified enterprise policy [{doc.doc_id}], {doc.content}"

        return {
            "mode": "RAG Augmented (Knowledge Base Lookup)",
            "answer": augmented_answer,
            "citations": [f"{doc.doc_id} — {doc.title} (Similarity: {sim:.2f})"],
            "retrieved_count": len(retrieved),
            "hallucination_risk": "VERY LOW (Factually Grounded)",
            "grounded": True,
            "context_excerpt": context_str
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="12.1 RAG Fundamentals",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">12.1 RAG Fundamentals</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Parametric vs non-parametric memory, LLM knowledge limitations, RAG vs fine-tuning, and end-to-end retrieval pipelines.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Modern RAG Paradigm", divider="rainbow")

st.markdown("""
Large Language Models exhibit extraordinary general reasoning capabilities, but their internal **parametric memory** 
is static, costly to update, prone to hallucinations, and blind to private organizational data.
**Retrieval-Augmented Generation (RAG)** bridges this gap by marrying parametric reasoning with dynamic, non-parametric 
external knowledge retrieval.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RAGArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_indexing {
            label="1. Offline Knowledge Ingestion Pipeline";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            RawDocs [label="Raw Documents\\n(PDFs, Markdown, Confluence, SQL)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            Chunker [label="Document Chunker & Cleaner\\n(Recursive Token/Character Splitting)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            EmbeddingModel [label="Dense Embedding Model\\n(text-embedding-3-large / BGE)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            VectorStore [label="Vector Store & IndexDB\\n(Chroma, Pinecone, pgvector)", shape=cylinder, fillcolor="#0F172A", fontcolor="#38BDF8", color="#38BDF8"];
        }

        subgraph cluster_query_retrieval {
            label="2. Online Query & Retrieval Pipeline";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            fontsize=12;

            UserQuery [label="User Query Inbound\\n('What is our 2026 meal allowance?')", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            QueryEmbed [label="Query Vectorization\\n(Generate query embedding vector)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            SimilaritySearch [label="Vector Similarity Search\\n(Cosine / HNSW / MMR / BM25 Hybrid)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            Reranker [label="Cross-Encoder Reranker\\n(Context Filtering & Relevance Scoring)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        }

        subgraph cluster_generation {
            label="3. Augmented Synthesis & Delivery";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            PromptAssembler [label="Augmented Prompt Assembly\\n(System Directive + Retrieved Chunks + Question)", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399"];
            GeneratorLLM [label="Generative LLM (Parametric Reasoner)\\n(GPT-4o / Claude 3.5 Sonnet / Llama 3)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC", penwidth=2];
            ClientAnswer [label="Grounded Output with Citations\\n('Per DOC-001, daily meal limit is 85 USD')", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#34D399"];
        }

        RawDocs -> Chunker [label="1. Ingest & Parse"];
        Chunker -> EmbeddingModel [label="2. Semantic Chunks"];
        EmbeddingModel -> VectorStore [label="3. Dense Vectors (d=1536)"];

        UserQuery -> QueryEmbed [label="4. Ingress"];
        QueryEmbed -> SimilaritySearch [label="5. Vector Query"];
        VectorStore -> SimilaritySearch [label="6. Top-K Candidates", style="dashed", color="#38BDF8"];
        SimilaritySearch -> Reranker [label="7. Filtered Matches"];
        Reranker -> PromptAssembler [label="8. Grounding Context"];
        UserQuery -> PromptAssembler [label="9. Original Intent"];
        PromptAssembler -> GeneratorLLM [label="10. Augmented Prompt"];
        GeneratorLLM -> ClientAnswer [label="11. Verifiable Answer"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: WHAT IS RAG? ---
st.header("2. What is RAG?", divider="rainbow")

st.markdown("""
**Retrieval-Augmented Generation (RAG)** is an AI framework that retrieves relevant facts from an external 
knowledge repository to ground an LLM before it generates a response.

Originally introduced by Lewis et al. (Facebook AI Research / Meta, 2020), RAG decomposes language processing into two complementary pillars:
1. **Parametric Memory**: The neural weights inside the transformer model. This provides general language syntax, reasoning, common sense, and conversational fluency.
2. **Non-Parametric Memory**: An external, indexable store of text documents, databases, or vector indices. This provides factual accuracy, current records, domain specifics, and auditability.
""")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown("#### 🧠 Parametric Memory Alone")
    st.markdown("""
    - Knowledge is frozen in neural weights at training cutoff.
    - Cannot cite sources or verify facts.
    - Prone to plausible hallucinations when uncertain.
    - Retraining/fine-tuning costs thousands of dollars and weeks of GPU compute.
    """)
with col_r2:
    st.markdown("#### ⚡ RAG (Parametric + Non-Parametric)")
    st.markdown("""
    - Knowledge is live and updated instantly by updating documents in the index.
    - Emits verifiable source citations and page numbers.
    - Drastically reduces hallucinations by instructing model: *"Answer strictly using the provided context."*
    - Document updates take milliseconds with zero GPU retraining.
    """)

st.divider()

# --- SECTION 3: WHY RAG IS REQUIRED ---
st.header("3. Why RAG is Required", divider="rainbow")

st.markdown("""
In enterprise software engineering, raw LLMs fail in four major operational dimensions:
""")

why_df = pd.DataFrame([
    {
        "Operational Requirement": "Private & Proprietary Knowledge",
        "Raw LLM Failure": "Base models have zero awareness of your internal git repos, customer tickets, or proprietary medical records.",
        "RAG Solution": "Indexes private VPC documents locally; documents are retrieved on demand without training third-party models."
    },
    {
        "Operational Requirement": "Eliminating Hallucinations",
        "Raw LLM Failure": "LLMs predict the most likely token sequence, not truth. They author confident, fabricated facts.",
        "RAG Solution": "Constrains LLM generation to provided source documents; instruct model to state 'I do not know' if context is missing."
    },
    {
        "Operational Requirement": "Auditability & Regulatory Compliance",
        "Raw LLM Failure": "Neural weights are a black box; impossible to prove which neuron generated a specific claim.",
        "RAG Solution": "Every sentence can be mapped directly to document chunk IDs, filenames, and timestamps for legal compliance."
    },
    {
        "Operational Requirement": "Real-Time Freshness & Dynamic Updates",
        "Raw LLM Failure": "Requires periodic re-training or continual fine-tuning, which causes catastrophic forgetting.",
        "RAG Solution": "Updating an internal policy takes 5 milliseconds: simply delete the old chunk and insert the new chunk into the vector database."
    }
])
st.dataframe(why_df, use_container_width=True)

st.divider()

# --- SECTION 4: LLM KNOWLEDGE LIMITATIONS ---
st.header("4. LLM Knowledge Limitations", divider="rainbow")

st.markdown("""
Understanding why RAG is necessary requires understanding the inherent computational limitations of foundation models:
""")

lim_c1, lim_c2, lim_c3 = st.columns(3)
with lim_c1:
    st.subheader("⏳ Temporal Knowledge Cutoff")
    st.markdown("""
    A model trained with data up to October 2023 has zero awareness of legal rulings, API updates, or product releases in 2024–2026. 
    Attempting to query current events forces the model to either guess or decline.
    """)

with lim_c2:
    st.subheader("🎯 Context Window Economics")
    st.markdown("""
    While modern models support 1M+ token context windows, stuffing an entire corporate wiki (e.g. 500M tokens) into every prompt 
    is economically impossible (costing hundreds of dollars per turn) and introduces severe latency (15+ seconds TTFT).
    """)

with lim_c3:
    st.subheader("🌫️ Lost-in-the-Middle Phenomenon")
    st.markdown("""
    Research demonstrates that LLMs attend accurately to the beginning and end of long prompts, but their attention degrades 
    dramatically across information buried in the middle 80% of an oversized prompt. RAG retrieves only the highest-signal top-K chunks.
    """)

st.divider()

# --- SECTION 5: RAG ARCHITECTURE ---
st.header("5. RAG Architecture", divider="rainbow")

st.markdown("""
A production RAG architecture is split into two fundamentally decoupled processing lifecycles:
1. **The Ingestion (Indexing) Pipeline (Asynchronous / Batch)**
2. **The Retrieval & Generation Pipeline (Synchronous / Online)**
""")

arch_dot = """
digraph DualPipeline {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.3, ranksep=0.4, fontname="Segoe UI"];
    node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
    edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

    subgraph cluster_offline {
        label="Offline Ingestion Lifecycle";
        style="dashed,rounded";
        bgcolor="#1E293B";
        fontcolor="#38BDF8";
        
        DocSource [label="Document Sources\\n(SharePoint, S3, PostgreSQL)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        Loader [label="Document Loaders\\n(Extract raw text + metadata)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
        Splitter [label="Text Chunking\\n(Recursive character chunking)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
        Embedder [label="Embedding Engine\\n(Batch vector computation)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
        VDB [label="Vector Database Storage\\n(Index creation & metadata filtering)", shape=cylinder, fillcolor="#0F172A", fontcolor="#38BDF8", color="#38BDF8"];

        DocSource -> Loader -> Splitter -> Embedder -> VDB;
    }

    subgraph cluster_online {
        label="Online Inference Lifecycle";
        style="dashed,rounded";
        bgcolor="#1E293B";
        fontcolor="#34D399";

        QueryIn [label="User Question", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#34D399"];
        QueryVec [label="Query Vectorization", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399"];
        KSearch [label="Top-K Similarity Retrieval", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399"];
        Augment [label="Prompt Context Injection", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399"];
        LLM [label="LLM Answer Synthesis", fillcolor="#581C87", fontcolor="#FFFFFF", color="#C084FC"];

        QueryIn -> QueryVec -> KSearch -> Augment -> LLM;
    }

    VDB -> KSearch [label="Vector Nearest Neighbor", style="dotted", color="#F59E0B"];
}
"""
st.graphviz_chart(arch_dot, use_container_width=True)

st.divider()

# --- SECTION 6: RAG VS FINE-TUNING ---
st.header("6. RAG vs Fine-Tuning", divider="rainbow")

st.markdown("""
A foundational architectural question in enterprise AI strategy is: **Should we use RAG, or should we Fine-Tune a model?**

A helpful industry mental model:
- **Fine-Tuning is for teaching form, style, vocabulary, tone, and formatting** (e.g. teaching a model to output medical SOAP notes or speak like a tax lawyer).
- **RAG is for providing dynamic facts, documents, customer records, and truth** (e.g. looking up a specific patient's latest lab results).
""")

comp_df = pd.DataFrame([
    {
        "Dimension": "Primary Use Case",
        "RAG (Retrieval-Augmentation)": "Information retrieval, fact lookup, dynamic corporate knowledge",
        "Fine-Tuning (Weights Adjustment)": "Style adaptation, tone, domain terminology, specialized syntax"
    },
    {
        "Dimension": "Knowledge Freshness",
        "RAG (Retrieval-Augmentation)": "Instantaneous (millisecond vector store updates)",
        "Fine-Tuning (Weights Adjustment)": "Static (snapshot of training dataset; stale over time)"
    },
    {
        "Dimension": "Hallucination Control",
        "RAG (Retrieval-Augmentation)": "High (model grounded in cited excerpts; verifiable truth)",
        "Fine-Tuning (Weights Adjustment)": "Low (model can still fabricate details with high confidence)"
    },
    {
        "Dimension": "Auditability & Citations",
        "RAG (Retrieval-Augmentation)": "Exact (provides document ID, page number, and source URL)",
        "Fine-Tuning (Weights Adjustment)": "None (weights are opaque; cannot trace answer to a source)"
    },
    {
        "Dimension": "Engineering Cost & Complexity",
        "RAG (Retrieval-Augmentation)": "Low to Moderate (standard database and API engineering)",
        "Fine-Tuning (Weights Adjustment)": "High (GPU clusters, dataset curation, loss curve tuning)"
    },
    {
        "Dimension": "Access Control (ACLs)",
        "RAG (Retrieval-Augmentation)": "Easy (filter document chunks by user permissions during query)",
        "Fine-Tuning (Weights Adjustment)": "Impossible (all trained weights are visible to any user)"
    }
])
st.dataframe(comp_df, use_container_width=True)

st.divider()

# --- SECTION 7: RAG PIPELINE ---
st.header("7. RAG Pipeline", divider="rainbow")

st.markdown("""
The evolution of RAG systems spans three architectural generations:
""")

gen_c1, gen_c2, gen_c3 = st.columns(3)
with gen_c1:
    st.subheader("1. Naive RAG")
    st.markdown("""
    - Simple character chunking
    - Flat dense embedding search
    - Direct top-K stuffing into prompt
    - **Weakness**: Suffers from semantic drift, missing context, and noisy chunk retrieval.
    """)

with gen_c2:
    st.subheader("2. Advanced RAG")
    st.markdown("""
    - Semantic / recursive chunking
    - Hybrid Search (BM25 keyword + dense vector with Reciprocal Rank Fusion)
    - Cross-encoder reranking
    - Query rewriting (HyDE - Hypothetical Document Embeddings)
    """)

with gen_c3:
    st.subheader("3. Agentic / Modular RAG")
    st.markdown("""
    - Autonomous routing (determines whether retrieval is even needed)
    - Self-RAG reflection loops (evaluates if retrieved chunks actually answer the question)
    - Iterative multi-hop search across diverse vector and SQL sources
    """)

st.divider()

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: End-to-End Modular RAG Engine", divider="rainbow")

st.markdown("""
Below is a clean, self-contained Python implementation of an enterprise-grade modular RAG pipeline. 
It defines `Document` objects, performs in-memory vector cosine similarity ranking, injects retrieved facts into an 
augmented prompt template, and calculates hallucination risk confidence scores.
""")

st.code("""import numpy as np
from typing import List, Dict, Any, Tuple

class Document:
    \"\"\"Core document entity matching LangChain's schema.\"\"\"
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

class SimpleRAGRetriever:
    \"\"\"In-memory dense retriever demonstrating vector search mechanics.\"\"\"
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []

    def add_document(self, doc: Document, embedding: np.ndarray):
        self.documents.append(doc)
        self.embeddings.append(embedding / np.linalg.norm(embedding))

    def retrieve(self, query_vector: np.ndarray, top_k: int = 2) -> List[Tuple[Document, float]]:
        q_norm = query_vector / np.linalg.norm(query_vector)
        scored_docs = []
        for doc, emb in zip(self.documents, self.embeddings):
            sim = float(np.dot(q_norm, emb))
            scored_docs.append((doc, sim))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

class ProductionRAGPipeline:
    \"\"\"Orchestrates retrieval, prompt augmentation, and grounded synthesis.\"\"\"
    def __init__(self, retriever: SimpleRAGRetriever, model_client):
        self.retriever = retriever
        self.model = model_client

    def answer_query(self, user_question: str, query_vector: np.ndarray) -> Dict[str, Any]:
        # 1. Retrieve top relevant chunks
        matches = self.retriever.retrieve(query_vector, top_k=2)
        
        # 2. Format context block
        context_blocks = []
        citations = []
        for doc, score in matches:
            source = doc.metadata.get("source", "Unknown")
            context_blocks.append(f"[Source: {source}]: {doc.page_content}")
            citations.append({"source": source, "relevance_score": round(score, 3)})

        formatted_context = "\\n\\n".join(context_blocks)

        # 3. Assemble Augmented Prompt with Grounding Guardrails
        augmented_prompt = f\"\"\"You are an enterprise AI assistant.
Answer the user question strictly using the provided context below.
If the context does not contain the answer, say 'I cannot find that in verified documents.'

### Context:
{formatted_context}

### Question:
{user_question}

### Answer:\"\"\"

        # 4. Invoke LLM
        response = self.model.generate(augmented_prompt)
        return {
            "answer": response,
            "citations": citations,
            "prompt_length_tokens": len(augmented_prompt.split()) * 4
        }""", language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Parametric vs Non-Parametric Knowledge Simulator", divider="rainbow")

st.markdown("""
Experience the practical difference between querying an LLM relying solely on static weights (Parametric Memory) 
versus an LLM grounded in real-time document retrieval (RAG).
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Query & Retrieval Configuration")
    user_query_choice = st.selectbox(
        "Select Enterprise Query Scenario",
        options=[
            "What is our daily domestic meal allowance for 2026 travel?",
            "What is the company policy on laptop refresh cycles?",
            "What is the failover latency target for Kubernetes clusters?",
            "What is the CEO's favorite vacation destination?"
        ]
    )

    toggle_rag = st.toggle("Enable RAG Grounding (Non-Parametric Lookup)", value=True)
    top_k_slider = st.slider("Max Chunks Retrieved (Top-K)", min_value=1, max_value=3, value=1)
    min_sim_slider = st.slider("Minimum Cosine Similarity Threshold", min_value=0.50, max_value=0.95, value=0.65, step=0.05)

    btn_exec = st.button("🚀 Execute Knowledge Query", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Synthesis & Verification Dashboard")

    if btn_exec:
        sim = MockRAGKnowledgeSimulator()
        res = sim.synthesize(user_query_choice, use_rag=toggle_rag, top_k=top_k_slider, min_similarity=min_sim_slider)

        m1, m2 = st.columns(2)
        m1.metric("Execution Mode", "⚡ Grounded RAG" if toggle_rag else "🧠 Parametric Only")
        m2.metric("Hallucination Risk", res["hallucination_risk"])

        st.markdown("#### 💬 Synthesized Response:")
        if res["grounded"]:
            st.success(res["answer"])
        else:
            st.warning(res["answer"])

        if res["citations"]:
            st.markdown("#### 📚 Verified Source Citations:")
            for cit in res["citations"]:
                st.write(f"- 📄 `{cit}`")

        if "context_excerpt" in res:
            with st.expander("🔍 View Injected Context Chunk"):
                st.code(res["context_excerpt"], language="markdown")
    else:
        st.info("Select a query scenario and click **Execute Knowledge Query** to compare hallucination risks and source grounding.")

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the fundamental architectural difference between Parametric and Non-Parametric memory?",
     "Parametric memory consists of the static weights learned by the neural network during pre-training and fine-tuning. It is fixed at the training cutoff, non-verifiable, and expensive to update. Non-parametric memory consists of an external, indexable document database or vector index. It is dynamically updated in real time, inspectable, and supports verifiable source citations."),
    ("When should an engineering team choose RAG over Fine-Tuning?",
     "Choose RAG when the primary objective is factual accuracy, knowledge freshness, accessing private or frequently changing documents, or providing source citations with access-control filtering. Choose Fine-Tuning when the primary objective is teaching the model a specific style, tone, domain syntax (e.g. specialized medical shorthand), or reducing prompt size for fixed reasoning tasks."),
    ("What causes the 'Lost-in-the-Middle' problem in long-context models, and how does RAG resolve it?",
     "Empirical research demonstrates that transformer attention mechanisms heavily prioritize tokens at the extreme beginning and end of long prompts. Information located in the middle 80% of an extensive context window suffers from significant retrieval degradation. RAG solves this by retrieving only the top-K most relevant chunks and placing them directly adjacent to the user query."),
    ("How does Hybrid Search (BM25 + Dense Vectors) improve retrieval quality over pure vector search?",
     "Dense vector embeddings capture abstract semantic similarity (meaning and concepts) but often fail on exact keyword lookups, such as part numbers ('ABC-9842'), acronyms, or specific customer names. BM25 excels at exact keyword matching. Hybrid search combines both scores using Reciprocal Rank Fusion (RRF), delivering superior recall across both conceptual and literal queries."),
    ("How do you implement document-level Access Control (ACLs) in enterprise RAG systems?",
     "During document ingestion, metadata tags indicating user IDs, group memberships, or tenant classifications are attached to each chunk in the vector store. At query time, the user's authenticated permission scopes are passed as a metadata pre-filter to the vector database query, guaranteeing the vector index only returns documents the user is legally permitted to see.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Ingestion Chunk Size Disconnected from Retrieval Granularity. Splitting documents into 5,000-token chunks means each retrieved chunk dilutes the prompt with irrelevant noise. Conversely, 20-token chunks destroy semantic context. Balance chunk size (e.g. 300-800 tokens) with appropriate overlap.")
    st.error("**Trap 2 (Breaking Error)**: Missing Cosine Similarity Threshold Ceilings. Always injecting top-K documents even when similarity is low (e.g. 0.35) forces irrelevant text into the prompt, confusing the LLM into generating false answers. Always enforce a minimum similarity threshold.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Redundant Vector Database Queries on Duplicate Prompts. Running vector queries on recurring identical questions incurs database latency. Implement a semantic cache ahead of the vector retriever.")
    st.warning("**Trap 4 (Quality Warning)**: Storing Uncleaned Documents. Ingesting raw HTML tags, page headers, footers, and copyright notices into vector stores pollutes embedding representations. Always clean document text prior to chunking.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: RAG Definition"):
        st.markdown("**Grounding generative LLMs in external, non-parametric knowledge retrieval.**")
    with st.expander("🎴 Flashcard 2: RAG vs Fine-Tuning"):
        st.markdown("**RAG provides dynamic facts and citations; Fine-Tuning provides style and domain form.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Lost-in-the-Middle"):
        st.markdown("**Degradation of LLM attention on tokens situated in the middle of oversized prompts.**")
    with st.expander("🎴 Flashcard 4: Hybrid Search"):
        st.markdown("**Combines BM25 exact keyword matching with dense vector semantic search.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural distinction between parametric and non-parametric memory.", value=True)
chk2 = st.checkbox("I can articulate the four primary enterprise drivers for deploying RAG.", value=True)
chk3 = st.checkbox("I understand the trade-offs between RAG and model Fine-Tuning across cost and auditability.", value=True)
chk4 = st.checkbox("I know how the offline ingestion pipeline differs from the online retrieval pipeline.", value=True)
chk5 = st.checkbox("I can implement similarity score thresholding to prevent prompt pollution and hallucinations.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 RAG fundamentals interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 12.1 RAG Fundamentals!")
