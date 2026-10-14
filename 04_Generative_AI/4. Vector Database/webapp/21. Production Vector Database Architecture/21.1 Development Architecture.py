import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="21.1 Development Architecture - Vector DB Curriculum",
    page_icon="💻",
    layout="wide"
)

# Title & Overview
st.title("21.1 Development Architecture: Local Vector Stores, Embedded Models & Offline RAG Stacks")
st.caption("Chapter 21: Production Vector Database Architecture | Domain: 4. Vector Database")
st.markdown("""
Before building a distributed, multi-region production cluster, every enterprise AI engineer must master the 
**Local Development Architecture**. 

Developing with cloud APIs (OpenAI, Pinecone, Cohere) during rapid prototyping incurs unnecessary API billing, 
exposes proprietary unreleased code to third-party endpoints, and creates network latency bottlenecks that slow down 
the edit-compile-test developer loop.

The modern local development stack enables **100% offline, air-gapped, zero-cost development**:
1. **Local Embedded Vector Database**: Running `chromadb.PersistentClient` directly inside the Python process with zero external daemon dependencies.
2. **Local Embedding Engine**: Utilizing quantized CPU runtimes like `fastembed` or `sentence-transformers` (`all-MiniLM-L6-v2`).
3. **Local Generative LLM**: Running quantized open models (Llama 3, Mistral 7B) on local hardware via **Ollama** or **llama.cpp**.
4. **Self-Contained Retrieval**: Executing the entire ingestion, search, and generation pipeline on a developer laptop with zero Internet connectivity.

In this lesson, we explore the local developer stack, assemble an end-to-end offline RAG pipeline in Python, 
compare local development capabilities against cloud limits, and test an interactive Local Stack Simulator.
""")

st.divider()

# Section 1: The 100% Offline Air-Gapped Development Stack
st.header("1. The 100% Offline Air-Gapped Development Stack", divider="rainbow")

col_dev1, col_dev2 = col_dev1, col_dev2 = st.columns([1.1, 0.9])

with col_dev1:
    st.markdown("""
    ### Four Core Components of the Offline Stack
    
    * **1. Embedded ChromaDB (`PersistentClient`)**:
      * Runs in the same process memory as your Python script.
      * Persists vectors and metadata directly to a local directory (`./local_chroma_db`).
      * Requires zero Docker daemons, zero external ports, and zero cloud accounts.
    
    * **2. Local Embedding Runtime (`fastembed` / ONNX)**:
      * Uses ONNX Runtime with quantized weights (`BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`).
      * Embeds 500 words in **3–6 milliseconds on standard CPU** without a GPU.
    
    * **3. Local Quantized LLM (Ollama / `llama.cpp`)**:
      * Serves 4-bit quantized models (`llama3.2:3b` or `mistral:7b-instruct-q4_K_M`) via a local HTTP endpoint (`http://localhost:11434`).
      * Consumes 4–6 GB of system RAM with zero token costs.
    
    * **4. Instant Developer Feedback Loop**:
      * Unit tests execute in milliseconds without external HTTP timeouts or rate limit (HTTP 429) errors.
    """)

with col_dev2:
    st.markdown("### The Offline Developer Loop Topology")
    dev_dot = """
    digraph LocalDevStack {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_laptop {
            label="Developer Laptop (100% Offline / No Internet)"; style="dashed"; color="#059669";
            DevCode [label="Local Python Script / Jupyter Notebook", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

            LocalChroma [label="Embedded ChromaDB\\n(PersistentClient on NVMe)\\n./local_chroma", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

            LocalEmbed [label="Local FastEmbed / ONNX\\n(MiniLM-L6-v2 on CPU)\\n384-dim | ~4ms", fillcolor="#FEF3C7", color="#D97706"];

            LocalLLM [label="Local Ollama Server\\n(Llama 3.2 3B Q4 on RAM)\\nhttp://localhost:11434", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

            DevCode -> LocalEmbed [label="Compute Query Vector"];
            LocalEmbed -> LocalChroma [label="In-Memory Search"];
            LocalChroma -> DevCode [label="Retrieved Chunks"];
            DevCode -> LocalLLM [label="Synthesize Answer"];
        }
    }
    """
    st.graphviz_chart(dev_dot, use_container_width=False)

st.divider()

# Section 2: Complete Offline Python RAG Stack Implementation
st.header("2. Complete Offline Python RAG Stack Implementation", divider="rainbow")

st.markdown("""
Below is a self-contained, air-gapped RAG pipeline utilizing **Embedded ChromaDB** and **Ollama** 
with zero cloud dependencies:
""")

st.code("""import chromadb
import requests
import json
from typing import List, Dict

class OfflineLocalRAG:
    \"\"\"100% local, zero-cloud RAG pipeline using ChromaDB and Ollama.\"\"\"
    
    def __init__(self, storage_path: str = "./local_chroma_store", ollama_model: str = "llama3.2:3b"):
        # 1. Initialize embedded persistent ChromaDB (no server process needed)
        self.chroma_client = chromadb.PersistentClient(path=storage_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="local_dev_docs",
            metadata={"hnsw:space": "cosine"}
        )
        self.model_name = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"

    def ingest_documents(self, docs: List[str], ids: List[str]):
        \"\"\"Chroma's default ONNX all-MiniLM-L6-v2 embeds text locally.\"\"\"
        self.collection.add(
            documents=docs,
            ids=ids
        )
        print(f"Ingested {len(docs)} documents locally.")

    def query_and_generate(self, user_question: str, k: int = 2) -> str:
        # 2. Local vector similarity retrieval
        results = self.collection.query(
            query_texts=[user_question],
            n_results=k
        )
        retrieved_passages = results["documents"][0]
        context_str = "\\n\\n".join(retrieved_passages)

        # 3. Construct local prompt
        prompt = f\"\"\"Answer the question using only the context below:
<context>
{context_str}
</context>

Question: {user_question}\"\"\"

        # 4. Stream response from local Ollama instance
        response = requests.post(
            self.ollama_url,
            json={"model": self.model_name, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]
""", language="python")

st.divider()

# Section 3: Development vs Production Capability Gap
st.header("3. Development vs Production Capability Gap", divider="rainbow")

st.markdown("""
Understanding what the local development stack can and cannot replicate prevents architectural surprises 
when migrating to production:
""")

DEV_VS_PROD = [
    {
        "Dimension": "Runtime Topology",
        "Local Development Stack": "In-process embedded (`PersistentClient`); single SQLite process.",
        "Production Enterprise Stack": "Distributed Client-Server cluster; decoupled query, indexing, and storage nodes.",
        "Engineering Consideration": "Local SQLite locks under multi-process writes. Production requires distributed Raft consensus."
    },
    {
        "Dimension": "Embedding Latency & Cost",
        "Local Development Stack": "CPU ONNX Runtime (~4ms / chunk); $0.00 cost; 100% private.",
        "Production Enterprise Stack": "Batched GPU Triton server or OpenAI API; $0.02 / 1M tokens.",
        "Engineering Consideration": "Local embeddings use 384 dimensions; production often uses 1,536 dimensions."
    },
    {
        "Dimension": "Concurrent Search Throughput",
        "Local Development Stack": "Single-user interactive testing (~10 to 50 QPS).",
        "Production Enterprise Stack": "Horizontally autoscaling read replicas (1,000 to 5,000+ QPS).",
        "Engineering Consideration": "Benchmarking throughput locally produces invalid capacity projections."
    },
    {
        "Dimension": "Access Control & Security",
        "Local Development Stack": "Zero authentication; filesystem permissions only.",
        "Production Enterprise Stack": "mTLS, Token Auth, Tenant-level ABAC, VPC security groups.",
        "Engineering Consideration": "Must test security wrappers and tenant filters before shipping to cloud."
    }
]

st.dataframe(pd.DataFrame(DEV_VS_PROD), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Local vs Cloud Stack Simulator
st.header("4. Interactive Workbench: Local vs Cloud Stack Simulator", divider="rainbow")

st.markdown("""
Compare the execution profile, latency breakdown, cost economics, and privacy boundaries 
of the Local Development Stack against a Cloud Enterprise Stack for a 1,000-query test run.
""")

col_sim_stk, col_sim_view = col_sim_stk, col_sim_view = st.columns([1, 1.2])

with col_sim_stk:
    st.subheader("1. Active Architectural Profile")
    active_stack = st.radio(
        "Choose Stack Configuration:",
        [
            "100% Offline Local Stack (Embedded Chroma + ONNX MiniLM + Ollama Llama-3.2)",
            "Hybrid Stack (Local Chroma + OpenAI API Embeddings + Claude 3.5 Haiku)",
            "Cloud Enterprise Stack (Chroma Cloud / Qdrant + OpenAI Large + GPT-4o)"
        ]
    )

with col_sim_view:
    st.subheader("2. 1,000-Query Benchmark Projections")
    
    if "Offline Local" in active_stack:
        lat_embed = "4.2 ms (CPU)"
        lat_search = "1.8 ms (Local NVMe)"
        lat_llm = "320 ms (Ollama Q4)"
        total_lat = "326 ms"
        cost_total = "$0.00 (Zero API Bills)"
        privacy_status = "🔒 100% Air-Gapped (Zero packets leave machine)"
    elif "Hybrid" in active_stack:
        lat_embed = "22.5 ms (OpenAI HTTP)"
        lat_search = "1.8 ms (Local NVMe)"
        lat_llm = "450 ms (Claude Haiku)"
        total_lat = "474 ms"
        cost_total = "$1.85 / 1,000 queries"
        privacy_status = "⚠️ External API Transfer (Payload leaves VPC)"
    else:  # Cloud Enterprise
        lat_embed = "28.0 ms (OpenAI HTTP)"
        lat_search = "14.5 ms (Cloud gRPC)"
        lat_llm = "950 ms (GPT-4o)"
        total_lat = "992 ms"
        cost_total = "$18.50 / 1,000 queries"
        privacy_status = "☁️ Multi-Cloud SaaS Distributed"
        
    m1, m2 = st.columns(2)
    m1.metric(label="Total Query Latency", value=total_lat)
    m2.metric(label="Cost for 1,000 Queries", value=cost_total)
    
    st.info(f"**Data Boundary**: {privacy_status}")
    st.markdown(f"""
    * **Embedding Latency**: `{lat_embed}`
    * **Vector Search Latency**: `{lat_search}`
    * **LLM Synthesis Latency**: `{lat_llm}`
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is an embedded vector database like ChromaDB PersistentClient, and how does it work?"):
    st.markdown("""
    **Answer**:
    An **embedded database** runs inside the exact same operating system process and memory space as the host application:
    * It does not run as a standalone background daemon or Docker container.
    * ChromaDB's `PersistentClient` initializes the C++ HNSW index library (via `hnswlib` or `chroma-core`) directly in Python memory.
    * It manages metadata and file pointers using an embedded SQLite database persisted to a local disk folder.
    * **Benefits**: Zero network serialization overhead, zero port conflicts, instantaneous startup, and seamless integration in local unit tests.
    """)

with st.expander("Q2: Why should engineers avoid developing with cloud API embedding models during daily feature work?"):
    st.markdown("""
    **Answer**:
    1. **Network Latency**: Generating an embedding via an external HTTP API takes 15ms–40ms per call. Local ONNX models take 2ms–5ms on CPU.
    2. **Rate Limits & Flakiness**: Running test suites with 500 unit tests easily triggers HTTP 429 Too Many Requests errors or network timeouts.
    3. **Financial Waste**: Re-indexing a 10,000-chunk development dataset 5 times a day incurs recurring API billing.
    4. **Air-Gapped Compliance**: Enables working on airplanes, trains, or in high-security air-gapped corporate environments without Internet access.
    """)

with st.expander("Q3: What happens if multiple Python processes attempt to write to the same ChromaDB PersistentClient folder?"):
    st.markdown("""
    **Answer**:
    **Concurrency Lock Conflict (Database Locked Error)**:
    * ChromaDB `PersistentClient` relies on SQLite for metadata state.
    * When Process A begins a write transaction, SQLite acquires an exclusive file lock.
    * If Process B attempts to write simultaneously, it throws `sqlite3.OperationalError: database is locked`.
    * **Production Rule**: `PersistentClient` is strictly designed for **single-process development**. Production multi-process services must use `chromadb.HttpClient` connecting to a standalone ChromaDB server container.
    """)

with st.expander("Q4: How does Ollama enable local LLM inference without requiring an enterprise GPU?"):
    st.markdown("""
    **Answer**:
    Ollama is built on `llama.cpp`:
    * It uses **GGUF 4-bit and 5-bit quantization** (e.g. `Q4_K_M`), which compresses 16-bit floating point model weights down to 4 bits per parameter with minimal perplexity loss.
    * An 8-billion parameter model that normally requires a 24 GB GPU fits comfortably into **5.5 GB of standard system RAM**!
    * It compiles optimized CPU vector instructions (AVX2, AVX-512, and Apple Silicon Metal) to achieve 15–30 tokens/second on standard laptops.
    """)

with st.expander("Q5: How do you mock vector search in CI/CD pipeline automated unit tests?"):
    st.markdown("""
    **Answer**:
    Use ChromaDB's in-memory ephemeral client:
    ```python
    import chromadb
    import pytest

    @pytest.fixture
    def test_vector_store():
        # Creates an ephemeral in-memory database that wipes automatically upon test completion
        client = chromadb.EphemeralClient()
        collection = client.create_collection("test_col")
        return collection
    ```
    This eliminates filesystem disk cleanup, ensures hermetic test isolation, and executes unit tests in sub-second times.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Deploying `PersistentClient` in Multi-Worker Web Servers (Gunicorn/Uvicorn)**  
    Using `PersistentClient` across 4 Uvicorn worker processes triggers SQLite database locking errors. Run Chroma as a standalone Docker server.
    """)
    st.warning("""
    **Trap 2: Changing Embedding Models Between Dev and Production**  
    Testing with 384-dim MiniLM locally and deploying 1536-dim OpenAI in production hides chunk boundary and retrieval quality bugs. Mirror model dimensions.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Checking Local Vector Index Folders into Git**  
    Committing `./local_chroma_store` binary files bloats Git repositories with megabytes of binary noise. Always add vector storage directories to `.gitignore`.
    """)
    st.warning("""
    **Trap 4: Expecting Production QPS on Local Laptops**  
    Measuring search latency on a single laptop with 500 documents does not reflect production performance on 10M vectors. Benchmark on cloud staging environments.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is an Embedded Vector DB?"):
        st.markdown("""
        **In-Process Database**:  
        Runs inside the Python process using local files/RAM without an external daemon (e.g. `chromadb.PersistentClient`).
        """)
    with st.expander("🗂️ Flashcard: What is Ollama?"):
        st.markdown("""
        **Local LLM Server**:  
        Runs quantized GGUF models (Llama 3, Mistral) on local CPU/Metal with zero cloud dependencies.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is EphemeralClient?"):
        st.markdown("""
        **In-Memory Testing**:  
        ChromaDB in-memory client that creates instant, temporary test collections that vanish upon process exit.
        """)
    with st.expander("🗂️ Flashcard: Why avoid cloud APIs in dev?"):
        st.markdown("""
        **Speed & Cost**:  
        Eliminates network latency, prevents rate limit errors (HTTP 429), and avoids unnecessary API bills.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the components of a 100% offline, air-gapped development stack.", value=True)
chk2 = st.checkbox("I know how to instantiate and query ChromaDB's embedded `PersistentClient` and `EphemeralClient`.", value=True)
chk3 = st.checkbox("I understand how local quantized runtimes (FastEmbed and Ollama) eliminate cloud API costs.", value=True)
chk4 = st.checkbox("I know why `PersistentClient` must never be used in multi-worker production web servers.", value=True)
chk5 = st.checkbox("I can implement hermetic, fast CI/CD unit tests for vector retrieval pipelines.", value=True)
chk6 = st.checkbox("I can architect a seamless bridge between local development prototypes and production clusters.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Development Architecture and are ready for Chapter 21.2: Production Architecture!")
