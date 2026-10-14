import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="3.3 Basic RAG Project Structure - RAG Curriculum",
    page_icon="📁",
    layout="wide"
)

# Title Block & 5-Pillar Overview
st.title("3.3 Basic RAG Project Structure: Directory Topologies, ETL Decoupling, & Configuration")
st.caption("Chapter 3: Setting Up a RAG Environment | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
A production RAG application is not a single sprawling Jupyter Notebook or flat directory of disconnected scripts. 
It requires a clean, maintainable software engineering layout that separates **offline data ingestion (ETL)** 
from **online real-time query serving**.

Failing to decouple ingestion from serving leads to severe operational bottlenecks: index building scripts block 
interactive API queries, configuration hyperparameters get hardcoded across files, and debugging retrieval quality 
regressions becomes nearly impossible.

In this lesson, you will master:
1. Standard production directory topologies for enterprise generative AI microservices.
2. Partitioning document storage: Raw, Processed, and Staging tiers.
3. Clean architectural separation between batch ingestion pipelines and low-latency query handlers.
4. Type-safe configuration management using Pydantic Settings and structured YAML files.
5. Structured JSON logging, telemetry observability, and hermetic dependency management.
""")

st.divider()

# Module-Scope Helper Classes for Interactive Workbench & Production Pipeline
class ProjectLayoutGenerator:
    def __init__(self):
        self.directory_structures = {
            "Enterprise Decoupled (Recommended)": {
                "tree": [
                    "rag_project/",
                    "├── data/",
                    "│   ├── raw/             # Original PDFs, Markdown, Word files",
                    "│   ├── processed/       # Sanitized JSON chunks with metadata",
                    "│   └── vector_store/    # Local vector index / Chroma persistent DB",
                    "├── src/",
                    "│   ├── ingestion/       # Offline ETL: Loaders, Splitters, Embedders",
                    "│   ├── retrieval/       # Online Serving: Hybrid Search, Rerankers",
                    "│   ├── generation/      # Prompt Templates, LLM Clients",
                    "│   └── config/          # Pydantic Settings & YAML parsers",
                    "├── tests/               # Unit, integration, and RAG evaluation tests",
                    "├── configs/             # settings.yaml, logging.yaml",
                    "├── .env.example         # Template for required environment variables",
                    "├── pyproject.toml       # Hermetic project dependencies",
                    "└── Dockerfile           # Production container build manifest"
                ],
                "description": "Decouples offline document processing from online low-latency query APIs."
            },
            "Lightweight Prototype": {
                "tree": [
                    "rag_app/",
                    "├── documents/          # Input enterprise files",
                    "├── index_data/         # Local vector store storage",
                    "├── app.py              # Single Streamlit / FastAPI entrypoint",
                    "├── pipeline.py         # Ingest + retrieve + generate functions",
                    "├── requirements.txt    # Pinned dependency requirements",
                    "└── .env                # Local secrets configuration"
                ],
                "description": "Compact structure suitable for local experiments and proof-of-concept demos."
            }
        }

    def generate_config_yaml(self, chunk_size: int, chunk_overlap: int, model_name: str, top_k: int) -> str:
        return f"""# Enterprise RAG Configuration Manifest
ingestion:
  chunk_size: {chunk_size}
  chunk_overlap: {chunk_overlap}
  supported_extensions:
    - .pdf
    - .docx
    - .md

embedding:
  model_name: "{model_name}"
  batch_size: 64
  normalize_embeddings: true

retrieval:
  top_k_candidates: {top_k}
  use_hybrid_search: true
  similarity_threshold: 0.75

generation:
  model_name: "gpt-4o"
  temperature: 0.1
  max_tokens: 1024
"""

# ==============================================================================
# 1. Architecture Blueprint
# ==============================================================================
st.header("1. Architecture Blueprint: Enterprise RAG Repository Topology", divider="rainbow")
st.markdown("""
A production-grade RAG repository strictly separates batch ingestion from online serving in a readable top-to-bottom layout:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_repo = """
    digraph RepoArchitecture {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_data {
            label="Tier 1: Document & Storage Partitioning Tier (data/)";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            raw [label="data/raw/\\nOriginal Immutable Enterprise Files (PDF, Word)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            proc [label="data/processed/\\nCleaned JSON Chunks with Associated Metadata", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            vstore [label="data/vector_store/\\nPersistent Vector Index Files & BM25 Inverted Store", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            raw -> proc -> vstore;
        }

        subgraph cluster_etl {
            label="Tier 2: Offline Ingestion Pipeline Tier (src/ingestion/)";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            fontsize=12;
            loaders [label="src/ingestion/loaders.py\\nDocument Parsing & Metadata Extraction", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            splitters [label="src/ingestion/chunker.py\\nRecursive & Semantic Boundary Text Splitters", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            embed_job [label="src/ingestion/indexer.py\\nBatch Vector Embedding & Index Population", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            loaders -> splitters -> embed_job;
        }

        subgraph cluster_serving {
            label="Tier 3: Online Real-Time Serving Tier (src/retrieval/ & src/generation/)";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            fontsize=12;
            ret_engine [label="src/retrieval/hybrid_search.py\\nSub-second BM25 + Vector ANN Lookup", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            reranker [label="src/retrieval/reranker.py\\nCross-Encoder Precision Candidate Reranking", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            gen_engine [label="src/generation/generator.py\\nPrompt Assembly & LLM Synthesis with Citations", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            ret_engine -> reranker -> gen_engine;
        }

        embed_job -> vstore [label="Populates Index", color="#10B981"];
        vstore -> ret_engine [label="Sub-second Search Queries", color="#38BDF8"];
    }
    """
    st.graphviz_chart(dot_repo, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In early projects, developers wrote monolithic scripts where every query re-parsed PDF documents on the fly. 
Production systems decouple the **Ingestion Pipeline** (executed via batch worker queues like Celery, Argo, or AWS Lambda 
upon document change) from the **Query Serving Gateway** (high-speed FastAPI or gRPC microservices serving user prompts 
in under 800ms).
""")

st.divider()

# ==============================================================================
# 2. Creating Project Directories
# ==============================================================================
st.header("2. Creating Project Directories & Package Layout", divider="rainbow")
st.markdown("""
A well-structured RAG codebase uses standard Python packaging rules, treating source code as an importable module:
""")

st.code("""
# Terminal: Bootstrap a production RAG layout
mkdir -p rag_project/{data/{raw,processed,vector_store},src/{ingestion,retrieval,generation,config},configs,tests}
touch rag_project/{pyproject.toml,README.md,.env.example,.gitignore,Dockerfile}
touch rag_project/src/__init__.py
touch rag_project/src/ingestion/{__init__.py,loaders.py,chunker.py,indexer.py}
touch rag_project/src/retrieval/{__init__.py,hybrid_search.py,reranker.py}
touch rag_project/src/generation/{__init__.py,prompts.py,generator.py}
touch rag_project/src/config/{__init__.py,settings.py}
""", language="bash")

st.divider()

# ==============================================================================
# 3. Organizing Documents
# ==============================================================================
st.header("3. Organizing Enterprise Documents: Raw, Processed, & Staging Stores", divider="rainbow")
st.markdown("""
Never store ingested documents in an unstructured single directory. Separate document assets into three tiers:
* **`data/raw/`**: The immutable archive of original enterprise files (PDFs, Word docs, Markdown files). Never modify files in this directory.
* **`data/processed/`**: Normalized JSON or Parquet files containing clean extracted text, chunk boundaries, token counts, and parent document metadata.
* **`data/vector_store/`**: The persistent local database files (e.g. ChromaDB SQLite files or Qdrant storage snapshots).
""")

st.divider()

# ==============================================================================
# 4. Separating Ingestion and Retrieval
# ==============================================================================
st.header("4. Separating Ingestion and Retrieval: Offline ETL vs. Online Serving", divider="rainbow")
st.markdown("""
Decoupling offline ingestion from online query serving provides clear operational benefits:
""")

df_separation = pd.DataFrame([
    {"Operational Metric": "Execution Schedule", "Offline Ingestion (src/ingestion/)": "Batch jobs triggered on document upload or nightly cron.", "Online Serving (src/retrieval/)": "Sub-second event loop triggered by user queries."},
    {"Operational Metric": "Hardware Profile", "Offline Ingestion (src/ingestion/)": "High-memory CPU or GPU clusters for batch embedding.", "Online Serving (src/retrieval/)": "Low-latency network I/O with fast RAM access for HNSW graphs."},
    {"Operational Metric": "Failure Handling", "Offline Ingestion (src/ingestion/)": "Retry queues (Celery/RabbitMQ) with dead-letter storage.", "Online Serving (src/retrieval/)": "Immediate graceful fallback (abstention message or keyword search)."}
])
st.dataframe(df_separation, use_container_width=True)

st.divider()

# ==============================================================================
# 5. Configuration Files
# ==============================================================================
st.header("5. Configuration Management: Type-Safe Settings & YAML Hyperparameters", divider="rainbow")
st.markdown("""
Avoid hardcoding chunk sizes, overlap percentages, and model names in Python logic. Use **Pydantic Settings** 
paired with structured **YAML configuration files**:
""")

col_cfg_code1, col_cfg_code2 = st.columns(2)
with col_cfg_code1:
    st.subheader("`configs/settings.yaml`")
    st.code("""
    ingestion:
      chunk_size: 500
      chunk_overlap: 50
    retrieval:
      top_k: 5
      use_hybrid: true
    models:
      embedding: "text-embedding-3-small"
      generation: "gpt-4o"
    """, language="yaml")

with col_cfg_code2:
    st.subheader("`src/config/settings.py`")
    st.code("""
    from pydantic import BaseModel
    from pydantic_settings import BaseSettings

    class IngestionConfig(BaseModel):
        chunk_size: int = 500
        chunk_overlap: int = 50

    class RAGSettings(BaseSettings):
        app_name: str = "Enterprise-RAG"
        ingestion: IngestionConfig = IngestionConfig()
    """, language="python")

st.divider()

# ==============================================================================
# 6. Logging
# ==============================================================================
st.header("6. Production Logging & Telemetry: Structured JSON & Latency Monitoring", divider="rainbow")
st.markdown("""
Standard unstructured `print()` statements are unusable in production Kubernetes clusters. Use **Structured JSON Logging** 
to track latency breakdowns and retrieval provenance:
""")

st.code("""
# Example Structured JSON Log Line
{
  "timestamp": "2026-10-13T15:30:00Z",
  "level": "INFO",
  "event": "query_served",
  "query_id": "QRY-9812",
  "user_id": "usr_442",
  "metrics": {
    "retrieval_latency_ms": 42.1,
    "rerank_latency_ms": 38.5,
    "total_latency_ms": 612.0
  },
  "provenance": {
    "retrieved_chunk_ids": ["DOC-SEC-01", "DOC-SEC-02"],
    "faithfulness_score": 0.98
  }
}
""", language="json")

st.divider()

# ==============================================================================
# 7. Dependency Management
# ==============================================================================
st.header("7. Dependency Management: `pyproject.toml`, Lockfiles, & Docker Containerization", divider="rainbow")
st.markdown("""
Modern Python standards mandate `pyproject.toml` as the single source of truth for build tools, dependencies, and packaging:
""")

col_dep1, col_dep2 = st.columns(2)
with col_dep1:
    st.subheader("`pyproject.toml` Specification")
    st.code("""
    [project]
    name = "enterprise-rag"
    version = "1.0.0"
    requires-python = ">=3.11,<3.13"
    dependencies = [
        "langchain-core>=0.3.0",
        "chromadb>=0.5.0",
        "pydantic-settings>=2.0.0",
        "pypdf>=4.0.0"
    ]
    """, language="toml")

with col_dep2:
    st.subheader("Production Multi-Stage Dockerfile")
    st.code("""
    FROM python:3.11-slim as runtime
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY src/ ./src/
    COPY configs/ ./configs/
    CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0"]
    """, language="dockerfile")

st.divider()

# ==============================================================================
# 8. Production Implementation
# ==============================================================================
st.header("8. Production Implementation: Full Enterprise RAG Project Skeleton Generator", divider="rainbow")
st.markdown("""
Below is a runnable Python production script that programmatically validates a RAG directory structure 
and generates default configuration manifests:
""")

prod_scaffold_code = """
from typing import Dict, List, Any

class RAGProjectScaffolder:
    def __init__(self, project_name: str = "enterprise_rag"):
        self.project_name = project_name
        self.expected_directories = [
            "data/raw",
            "data/processed",
            "data/vector_store",
            "src/ingestion",
            "src/retrieval",
            "src/generation",
            "src/config",
            "configs",
            "tests"
        ]

    def generate_manifest(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "architecture_standard": "Enterprise Decoupled RAG",
            "directory_layout": self.expected_directories,
            "default_config": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "embedding_model": "text-embedding-3-small",
                "retrieval_top_k": 5
            }
        }

if __name__ == "__main__":
    scaffolder = RAGProjectScaffolder()
    manifest = scaffolder.generate_manifest()
    print("Generated Enterprise Project Manifest:", manifest)
"""
st.code(prod_scaffold_code, language="python")

st.divider()

# ==============================================================================
# 9. Interactive Workbench
# ==============================================================================
st.header("9. Interactive Workbench: RAG Project Layout Explorer & Config Generator", divider="rainbow")
st.markdown("""
Explore standard enterprise RAG repository directory trees and generate customized YAML configuration manifests:
""")

layout_engine = ProjectLayoutGenerator()

col_lay1, col_lay2 = st.columns(2)
with col_lay1:
    selected_layout = st.selectbox(
        "Select Architectural Layout Strategy:",
        list(layout_engine.directory_structures.keys())
    )
    st.caption(layout_engine.directory_structures[selected_layout]["description"])
    st.code("\n".join(layout_engine.directory_structures[selected_layout]["tree"]), language="text")

with col_lay2:
    st.subheader("Config Hyperparameter Generator")
    c_size = st.slider("Target Chunk Size (Tokens):", min_value=100, max_value=1000, value=500, step=50)
    c_overlap = st.slider("Chunk Overlap (Tokens):", min_value=10, max_value=200, value=50, step=10)
    emb_model = st.selectbox("Embedding Model:", ["text-embedding-3-small", "BAAI/bge-large-en-v1.5", "cohere-embed-english-v3.0"])
    top_k_select = st.slider("Retrieval Top-K Chunks:", min_value=1, max_value=10, value=5)

    generated_yaml = layout_engine.generate_config_yaml(c_size, c_overlap, emb_model, top_k_select)
    st.code(generated_yaml, language="yaml")

st.divider()

# ==============================================================================
# 10. Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why is coupling ingestion and retrieval code in a single script a major antipattern?"):
    st.markdown("""
    **Answer**: Ingestion is a computationally intensive batch ETL job (parsing PDFs, computing embeddings over millions of tokens) 
    that runs infrequently. Retrieval is a low-latency, real-time query service requiring sub-second response times. 
    Coupling them prevents independent horizontal scaling, couples memory profiles, and causes deployment fragility.
    """)

with st.expander("Q2: How do you partition enterprise document storage across a RAG lifecycle?"):
    st.markdown("""
    **Answer**: Divide storage into three distinct tiers:
    1. **Raw Tier**: Immutable archive of original uploaded files (PDFs, Word documents).
    2. **Processed Tier**: Sanitized and structured JSON chunks with extracted metadata.
    3. **Vector Store Tier**: Indexed embedding vectors and inverted lexical files.
    """)

with st.expander("Q3: What role does Pydantic Settings play in enterprise RAG configuration management?"):
    st.markdown("""
    **Answer**: Pydantic Settings provides type-safe validation for application configuration at startup. It parses and validates 
    nested values from YAML files, environment variables, and `.env` files, failing fast with clear error messages if critical 
    parameters (such as embedding dimensions or API endpoints) are misconfigured.
    """)

with st.expander("Q4: Why is structured JSON logging preferred over standard string formatting in RAG?"):
    st.markdown("""
    **Answer**: Structured JSON logs allow centralized log aggregation tools (Datadog, Elastic, CloudWatch) to index fields 
    such as `retrieval_latency_ms`, `user_id`, `chunk_ids`, and `faithfulness_score`. This enables real-time metric dashboards, 
    latency anomaly alerts, and automated audit trails.
    """)

with st.expander("Q5: How does a multi-stage Docker build optimize production RAG container deployments?"):
    st.markdown("""
    **Answer**: Multi-stage builds use a temporary builder image with C++ compilers and build tools to compile wheels, 
    then copy only the compiled binaries and application code into a lightweight runtime image. This eliminates build tools 
    from the final container, reducing image size by 70%+ and minimizing security vulnerability exposure.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The Flat Directory Antipattern)**: Placing all scripts, documents, and output files in a single flat directory, resulting in untracked data file commits and unmaintainable monolithic codebases.")
    st.warning("**Trap 2 (Hardcoded Ingestion Hyperparameters)**: Hardcoding chunk size and overlap values inside ingestion functions instead of reading from centralized YAML configuration manifests.")
with col_trap2:
    st.error("**Trap 3 (Committing Local Vector Store Snapshots to Git)**: Committing local ChromaDB or FAISS index binary files to version control repositories, bloating git history and corrupting indices across platforms.")
    st.warning("**Trap 4 (Unstructured Print Debugging in Production)**: Relying on `print()` statements for diagnostic output, preventing cloud log aggregators from parsing latency metrics and query provenance.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: Why decouple Ingestion from Retrieval?")
        st.caption("Flip for answer:")
        st.info("**Back**: Ingestion is heavy batch ETL (hours/minutes); Retrieval is low-latency serving (sub-second). Decoupling allows independent scaling and error boundaries.")
    with st.container(border=True):
        st.markdown("**Front**: What are the 3 document storage partitions?")
        st.caption("Flip for answer:")
        st.info("**Back**: 1. `data/raw/` (immutable originals), 2. `data/processed/` (cleaned JSON chunks), 3. `data/vector_store/` (persistent index binaries).")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is Pydantic Settings used for in RAG?")
        st.caption("Flip for answer:")
        st.info("**Back**: Validates application configuration at startup with strict type checking, loading from YAML, environment variables, and `.env` files.")
    with st.container(border=True):
        st.markdown("**Front**: Why use Structured JSON Logging?")
        st.caption("Flip for answer:")
        st.info("**Back**: Emits machine-readable logs containing query IDs, latency breakdown metrics, and chunk citations for centralized cloud observability.")

st.divider()

# ==============================================================================
# 11. Self-Assessment Checklist
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I can structure a production RAG repository using decoupled ingestion and retrieval modules.", value=True)
c2 = st.checkbox("I know how to partition enterprise document assets across raw, processed, and vector tiers.", value=True)
c3 = st.checkbox("I understand why local vector index files must be excluded from git version control.", value=True)
c4 = st.checkbox("I can manage application configuration using type-safe Pydantic Settings and YAML manifests.", value=True)
c5 = st.checkbox("I know how to implement structured JSON logging to monitor retrieval and rerank latencies.", value=True)
c6 = st.checkbox("I understand how multi-stage Docker builds reduce container image attack surfaces and file sizes.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Chapter 3 Complete! You have mastered RAG Environment Setup and are ready for Chapter 4: Documents and Data Sources.")
