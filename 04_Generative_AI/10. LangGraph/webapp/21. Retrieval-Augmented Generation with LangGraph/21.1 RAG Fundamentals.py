import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="21.1 RAG Fundamentals - LangGraph Curriculum",
    page_icon="📚",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .badge-rag {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class RagFundamentalsSimulator:
    """
    Simulates core RAG mechanics:
    - Text chunking (Fixed vs Recursive vs Sentence-based)
    - Simulated vector embeddings
    - Cosine similarity ranking against query
    """
    def __init__(self):
        self.sample_corpus = (
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
            "It extends LangChain with cyclic computational graphs. "
            "Unlike traditional DAG engines, LangGraph allows loops and state updates. "
            "Retrieval-Augmented Generation grounds LLM answers on external enterprise documents. "
            "Vector databases index dense embeddings to enable sub-second nearest-neighbor search."
        )

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        words = text.split(" ")
        chunks = []
        i = 0
        step = max(chunk_size - chunk_overlap, 1)
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += step
        return chunks

    def compute_simulated_similarity(self, query: str, chunks: List[str]) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        results = []
        for idx, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            intersection = query_words.intersection(chunk_words)
            # Simulated Jaccard/overlap score as proxy for semantic cosine similarity
            score = len(intersection) / max(len(query_words), 1)
            score = round(min(0.5 + score * 0.45 + np.random.uniform(0.01, 0.04), 0.99), 3)
            results.append({
                "chunk_id": f"chunk_{idx + 1}",
                "text": chunk,
                "similarity_score": score
            })
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Core RAG Lifecycle and Information Retrieval Pipeline", divider="rainbow")

st.markdown("""
Large Language Models have static knowledge boundaries frozen at their pre-training cutoff date.
Furthermore, proprietary enterprise data (internal documentation, customer records, codebase repositories) is inaccessible to generic foundation models.

**Retrieval-Augmented Generation (RAG)** bridges this gap by decoupling the LLM's reasoning engine from its knowledge store:
1. **Ingestion Tier**: Raw enterprise documents are partitioned into granular chunks and converted into dense vector embeddings.
2. **Indexing Tier**: Vectors are stored in a specialized Vector Database equipped with Approximate Nearest Neighbor (ANN) indexes.
3. **Retrieval Tier**: Incoming user queries are embedded into the same vector space, and the most relevant document chunks are retrieved.
4. **Synthesis Tier**: Retrieved chunks are injected into the prompt as factual context, grounding the LLM's response.

The blueprint below illustrates the complete RAG lifecycle from offline document ingestion to live online query generation.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RagFundamentalsArchitecture {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_ingestion {
            label="1. Document Ingestion & Chunking Pipeline (Offline)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            raw_docs [label="Raw Documents\\n(PDF, Markdown, HTML, SQL)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            chunker [label="Text Splitter (Chunking)\\nChunk Size: 500 | Overlap: 50", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            embedder [label="Embedding Model\\n(text-embedding-3-small -> 1536d)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            vector_db [label="Vector Database (Chroma / Pinecone)\\nHNSW Index / Cosine Metric", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            raw_docs -> chunker -> embedder -> vector_db;
        }

        subgraph cluster_online_query {
            label="2. Online Query & Context Synthesis (Runtime)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            user_query [label="User Query\\n'How do subgraphs work in LangGraph?'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            query_embed [label="Embed Query\\nDense Vector Conversion", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            similarity_search [label="Vector Similarity Search\\nk-NN Top 3 Nearest Chunks", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            prompt_synth [label="Context Construction\\nSystem Prompt + Retrieved Chunks", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            llm_gen [label="LLM Synthesis\\nGrounded Factual Response", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];

            user_query -> query_embed -> similarity_search;
            vector_db -> similarity_search [label="Index Lookup", style=dashed, color="#F97316"];
            similarity_search -> prompt_synth [label="Relevant Passages"];
            prompt_synth -> llm_gen;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: WHAT IS RAG?
# ==============================================================================
st.header("2. What is RAG?: Knowledge Grounding, Eliminating Hallucinations, and Domain Adaptation", divider="rainbow")

st.markdown("""
**Retrieval-Augmented Generation (RAG)** is an architectural pattern that enhances language model outputs by retrieving authoritative facts from external knowledge bases before generating a response.
""")

col_rg1, col_rg2 = st.columns(2)
with col_rg1:
    st.markdown("#### Primary Benefits of RAG")
    st.markdown("""
    1. **Elimination of Hallucinations**: Constrains the model's generation to cited factual context rather than unverified internal memory.
    2. **Real-Time Data Access**: Enables agents to answer questions on documents uploaded seconds ago without fine-tuning.
    3. **Source Attribution & Citations**: Enables footnoted links back to original source paragraphs for human auditability.
    4. **Cost Efficiency**: Exponentially cheaper and faster to maintain than continuously fine-tuning foundation models.
    """)

with col_rg2:
    st.markdown("#### Fine-Tuning vs RAG")
    df_ft = pd.DataFrame({
        "Attribute": ["Primary Purpose", "Knowledge Freshness", "Hallucination Risk", "Setup Cost", "Auditability"],
        "Fine-Tuning": ["Teaching style, tone, domain syntax", "Static at training time", "Moderate to High", "High (GPU training runs)", "Opaque / Black-box weights"],
        "RAG": ["Providing factual knowledge & context", "Real-time updates", "Low (Grounded in passages)", "Low (Embeddings + VectorDB)", "Direct document citations"]
    })
    st.dataframe(df_ft, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 3: DOCUMENTS
# ==============================================================================
st.header("3. Documents: Document Abstractions, Metadata Schemas, and Corpus Ingestion", divider="rainbow")

st.markdown("""
In LangChain and LangGraph, the atomic unit of knowledge is the **`Document`** object.
A Document consists of:
- `page_content`: The raw text payload of the passage.
- `metadata`: A dictionary storing provenance attributes (e.g. `source_url`, `page_number`, `created_at`, `author`, `department`).
""")

st.code('''from langchain_core.documents import Document

doc = Document(
    page_content="LangGraph supports cyclic multi-agent graphs with persistent state checkpointers.",
    metadata={
        "source": "https://langchain.ai/docs/langgraph",
        "author": "LangChain Core Team",
        "section": "Architecture Overview",
        "doc_id": "doc_9918"
    }
)''', language="python")

st.info("💡 **Metadata Filtering Principle**: Rich metadata allows retrieval queries to filter by business scope (e.g. `department == 'Legal' and year == 2024`) before computing vector similarities.")

# ==============================================================================
# SECTION 4: CHUNKING
# ==============================================================================
st.header("4. Chunking: Recursive Character, Semantic Boundary, and Token Window Strategies", divider="rainbow")

st.markdown("""
Feeding an entire 50-page PDF into an embedding model produces a single blurry vector that loses specific details.
**Chunking** divides text into concise, coherent segments.
""")

st.markdown("""
- **Fixed-Size Chunking**: Slices text strictly every $N$ characters. Simple but frequently cuts sentences in half.
- **Recursive Character Chunking (`RecursiveCharacterTextSplitter`)**: Attempts to split along natural boundaries in descending priority: `["\\n\\n", "\\n", " ", ""]`. Preserves paragraph and sentence integrity.
- **Semantic Chunking**: Computes embedding distances between consecutive sentences and places split boundaries wherever the semantic topic shifts significantly.
- **Chunk Overlap**: Retains 10-20% of preceding text in the next chunk to preserve context across boundaries.
""")

st.code('''from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Target characters per chunk
    chunk_overlap=50,       # Retain 50 characters overlap to avoid cutoff context
    separators=["\\n\\n", "\\n", ". ", " ", ""]
)
chunks = splitter.split_documents([doc])''', language="python")

# ==============================================================================
# SECTION 5: EMBEDDINGS
# ==============================================================================
st.header("5. Embeddings: Dense Vector Representations, Cosine Geometry, and High-Dimensional Space", divider="rainbow")

st.markdown("""
An **Embedding Model** transforms textual passages into high-dimensional numerical vectors (typically 1536 or 3072 floating-point numbers).
- **Semantic Proximity**: Sentences with similar meanings produce vectors that point in nearly identical geometric directions.
- **Vector Space Example**: The vector for `"How do I configure checkpointers?"` will have a high cosine similarity to `"Persistence setup guide"`, even if they share zero identical words!
""")

# ==============================================================================
# SECTION 6: VECTOR DATABASES
# ==============================================================================
st.header("6. Vector Databases: Vector Indexing Topologies, ANN Algorithms, and Storage Engines", divider="rainbow")

st.markdown("""
Calculating exact cosine distance between a query and 10 million document vectors using a brute-force sweep ($O(N)$) takes seconds.
**Vector Databases** use **Approximate Nearest Neighbor (ANN)** indexing algorithms to perform sub-10ms retrieval:
- **HNSW (Hierarchical Navigable Small World)**: Builds a multi-layer graph skip-list for super-fast logarithmic navigation.
- **IVF (Inverted File Index)**: Partitions the vector space into Voronoi clusters to search only nearby partitions.
- **Popular Engines**: Chroma (embedded/local), Pinecone (managed cloud), Qdrant (Rust-based high throughput), pgvector (PostgreSQL native extension).
""")

# ==============================================================================
# SECTION 7: SIMILARITY SEARCH
# ==============================================================================
st.header("7. Similarity Search: Distance Metrics (Cosine, Dot Product, Euclidean) and k-NN Queries", divider="rainbow")

st.markdown("""
During retrieval, the vector database ranks candidate chunks using mathematical distance metrics:
""")

col_sim_m1, col_sim_m2, col_sim_m3 = st.columns(3)
with col_sim_m1:
    st.markdown("#### Cosine Similarity")
    st.markdown("""
    Measures the cosine of the angle between two vectors:
    `similarity = dot(A, B) / (norm(A) * norm(B))`
    Normalized between -1 and 1. Standard for text embeddings.
    """)

with col_sim_m2:
    st.markdown("#### Dot Product")
    st.markdown("""
    `dot = sum(A_i * B_i)`
    Equivalent to cosine similarity when vectors are pre-normalized to unit length (`norm = 1.0`).
    Computationally very fast on modern CPUs/GPUs.
    """)

with col_sim_m3:
    st.markdown("#### Euclidean Distance (L2)")
    st.markdown("""
    `distance = sqrt(sum((A_i - B_i)^2))`
    Measures straight-line geometric distance. Smaller distance indicates higher similarity.
    """)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Document Ingestion and Vector Search Pipeline", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing document loading, recursive chunking, Chroma vector indexing, and similarity retrieval.
""")

st.code('''from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Ingest Documents
documents = [
    Document(
        page_content="LangGraph Checkpointers persist graph state across multi-turn sessions.",
        metadata={"source": "docs/persistence.md", "topic": "checkpoints"}
    ),
    Document(
        page_content="Streaming with version='v2' emits standardized StreamPart dictionaries.",
        metadata={"source": "docs/streaming.md", "topic": "streaming"}
    ),
    Document(
        page_content="Subgraphs allow isolated child state schemas with input/output adapters.",
        metadata={"source": "docs/subgraphs.md", "topic": "subgraphs"}
    )
]

# 2. Split into Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# 3. Create Vector Store with In-Memory Chroma
# In production with OpenAI: embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vector_store = Chroma.from_documents(chunks, embeddings)

# 4. Perform Similarity Search with Metadata Filter
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 2, "filter": {"topic": "persistence"}}
# )
# retrieved_docs = retriever.invoke("How do I store state across sessions?")
print(f"Successfully processed {len(documents)} docs into {len(chunks)} chunks.")''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Chunking Strategy & Vector Similarity Playground", divider="rainbow")

st.markdown("""
Experiment with chunk size parameters and test simulated semantic similarity ranking across document passages.
""")

sim = RagFundamentalsSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("⚙️ Ingestion & Chunking Parameters")
    c_size = st.slider("Chunk Size (Words)", min_value=5, max_value=25, value=12)
    c_overlap = st.slider("Chunk Overlap (Words)", min_value=0, max_value=5, value=2)
    
    generated_chunks = sim.chunk_text(sim.sample_corpus, c_size, c_overlap)
    st.info(f"Split corpus into **{len(generated_chunks)}** chunks.")

    with st.expander("🔍 View Generated Text Chunks"):
        for i, ch in enumerate(generated_chunks):
            st.markdown(f"**Chunk {i+1}**: `{ch}`")

with col_wb2:
    st.subheader("🔎 Query Retrieval Simulation")
    test_query = st.text_input("Search Query", value="cyclic computational graphs with loops")
    
    if st.button("🚀 Execute Similarity Search"):
        ranked_res = sim.compute_simulated_similarity(test_query, generated_chunks)
        st.success(f"Top {min(3, len(ranked_res))} Ranked Passages:")
        
        df_ranked = pd.DataFrame(ranked_res[:3])
        st.dataframe(df_ranked, use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is the primary difference between RAG and Fine-Tuning?", expanded=False):
    st.markdown("""
    **Answer**:
    Fine-tuning updates model weights to adapt tone, format, or specialized task syntax, but is static and prone to hallucinations.
    RAG retrieves external, dynamic documents at query time to ground the LLM in up-to-date facts with verifiable source citations.
    """)

with st.expander("Q2: Why is RecursiveCharacterTextSplitter preferred over simple character splitting?", expanded=False):
    st.markdown("""
    **Answer**:
    Recursive character splitting attempts to split along semantic paragraph and sentence delimiters (`\\n\\n`, `\\n`, `. `) before falling back to individual words or characters.
    This preserves natural linguistic units and prevents cutting sentences across chunk boundaries.
    """)

with st.expander("Q3: What role does chunk overlap play in document ingestion?", expanded=False):
    st.markdown("""
    **Answer**:
    Overlap retains trailing context from the previous chunk at the start of the next chunk.
    This prevents contextual rupture if a vital phrase or relationship spans across an arbitrary character split boundary.
    """)

with st.expander("Q4: How does HNSW indexing accelerate vector search compared to brute-force k-NN?", expanded=False):
    st.markdown("""
    **Answer**:
    Brute-force k-NN requires computing distance against every vector in the dataset ($O(N)$).
    HNSW organizes vectors into a multi-layered hierarchical graph where upper layers have sparse, long-range edges and lower layers have dense local edges, enabling approximate nearest neighbor search in logarithmic time ($O(\\log N)$).
    """)

with st.expander("Q5: When should Dot Product be chosen over Cosine Similarity in production?", expanded=False):
    st.markdown("""
    **Answer**:
    When embedding vectors are pre-normalized to unit length (magnitude = 1.0).
    Under unit normalization, the dot product is mathematically equivalent to cosine similarity but avoids costly square-root normalization calculations, speeding up vector searches.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Incompatible Embedding Dimensions Between Ingestion and Query")
    st.markdown("""
    *Mechanism*: Ingesting documents with a 1536-dimensional model and querying with a 3072-dimensional model.  
    *Consequence*: Vector database throws dimension mismatch error immediately, breaking retrieval.  
    *Fix*: Always use the exact same embedding model version for indexing and querying.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Oversized Chunks Exceeding Retriever Context")
    st.markdown("""
    *Mechanism*: Setting `chunk_size=4000` tokens and retrieving top 10 chunks.  
    *Consequence*: Injected context consumes 40,000 tokens, blowing up costs and degrading LLM focus.  
    *Fix*: Keep chunk sizes concise (300-600 tokens) and retrieve fewer, highly relevant passages.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Stripping Metadata During PDF Parsing")
    st.markdown("""
    *Mechanism*: Extracting raw text from documents without preserving document IDs, page numbers, or dates.  
    *Consequence*: Impossible to filter by date or provide verifiable footnotes in generated answers.  
    *Fix*: Store rich metadata dictionaries with every Document chunk.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Recall Warning): Zero Chunk Overlap Causing Context Clipping")
    st.markdown("""
    *Mechanism*: Setting `chunk_overlap=0` in text splitters.  
    *Consequence*: Key entities (e.g. subject in sentence 1, predicate in sentence 2) get severed across chunk boundaries.  
    *Fix*: Always set chunk overlap to 10-20% of chunk size.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**RAG**\\nRetrieval-Augmented Generation: grounding LLMs with real-time external document retrieval.")
with col_fc2:
    st.info("**Document Object**\\nThe core LangChain abstraction containing page_content and metadata.")
with col_fc3:
    st.info("**Chunk Overlap**\\nTrailing context retained between sequential chunks to preserve boundary coherence.")
with col_fc4:
    st.info("**HNSW**\\nHierarchical Navigable Small World: fast logarithmic graph-based vector index.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know the architectural differences between RAG and fine-tuning.", value=True)
c2 = st.checkbox("I understand the components of a LangChain Document (page_content and metadata).", value=True)
c3 = st.checkbox("I know how RecursiveCharacterTextSplitter partitions text along natural boundaries.", value=True)
c4 = st.checkbox("I understand the geometric concepts of dense vector embeddings and cosine similarity.", value=True)
c5 = st.checkbox("I know how HNSW graph indexes achieve logarithmic ANN vector retrieval.", value=True)
c6 = st.checkbox("I understand why consistent embedding models must be used for ingestion and queries.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 21.1 'RAG Fundamentals' with full architectural and practical comprehension!")
