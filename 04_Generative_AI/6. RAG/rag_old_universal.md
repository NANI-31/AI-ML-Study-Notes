# Universal Streamlit Lesson File Standard & Architecture Specification

### Single Source of Truth & Master Format for Domain: `6. RAG` (Retrieval-Augmented Generation)

### Target Curriculum: `04_Generative_AI/6. RAG`

### Reference Design Standard: Continuous Vertical Scroll · Dark Graphviz · Zero LaTeX Math

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `6. RAG` represents an interactive, production-grade learning module covering document ingestion, semantic chunking, hybrid retrieval, cross-encoder reranking, and grounded synthesis.

### 1.1 Core Principles

1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden**.
2. **Mandatory 1:1 Section Density**: Every numbered bullet point under a lesson in the syllabus roadmap MUST receive its own dedicated `st.header("X. <Topic Name>", divider="rainbow")` section. Section count is locked by formula: `Total Sections = 1 (Architecture) + K (Roadmap Bullets) + 1 (Production Code) + 1 (Interactive Workbench) + 1 (Technical Interview Prep) + 1 (Self-Assessment Checklist) = K + 5`.
3. **Modern Foundation LLM & RAG Contrast**: Contrast legacy naive RAG patterns (simple top-k vector cosine match, naive fixed-character chunking) with modern production RAG (Agentic RAG, GraphRAG, HyDE, Corrective RAG, Contextual Retrieval), featuring a dedicated **`### 🚀 Modern Foundation LLM Reality: How New Models Follow This`** contrast block.
4. **Zero LaTeX Math Notation**: Never use `$$...$$` or `$x$`. Express all retrieval fusion and reranking formulas in plain English and code formulas (e.g., `Reciprocal Rank Fusion RRF Score = 1 / (60 + Rank)`, `BM25 Score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (|D| / avgdl)))`, `Context Ratio: Chunks / Prompt Window`).
5. **Heavy Architectural Diagrams (Mandatory)**: Every lesson must feature Graphviz diagrams with dark theme styling (`bgcolor="#0B1120"`):
   - **End-to-End RAG Mesh**: Ingestion Pipeline (Splitter -> Embedder -> Vector DB) and Query Retrieval Loop (Query -> Hybrid Search -> Reranker -> LLM Synthesis).
   - **Hybrid Retrieval Fusion**: Sparse BM25 + Dense ANN Parallel Branches -> Reciprocal Rank Fusion (RRF) -> Top-K Chunks.
   - **Cross-Encoder Reranker**: Query + Chunk Pairs -> Full Self-Attention Cross-Encoder -> Logit Relevance Score.
   - **Chunking Boundary Visualizers**: Sliding Window, Recursive Character, Semantic Boundary, Parent-Child Chunking.
6. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
7. **Dual Theory & Practical Delivery**: Combine RAG system diagrams, chunking trade-off matrices, runnable production retrieval pipelines, and an interactive RAG query/rerank studio.
8. **Uniform Pedagogical Endings**:
   - **Section K+4: Technical Interview Preparation & Production Traps** (Exactly 5 Q&As + 4 Traps [2 error, 2 warning] + 4 Flashcards).
   - **Section K+5: Self-Assessment Checklist** (Exactly 6 Checkboxes + clean `st.success(...)` completion banner. Strictly prohibit `st.balloons()`).

---

## 2. Complete File Anatomy (Top-to-Bottom Structure)

```
┌────────────────────────────────────────────────────────────────────────┐
│ [A] Standard Imports Block (Exactly 5 Standard Imports)                │
├────────────────────────────────────────────────────────────────────────┤
│ [B] st.set_page_config(page_title=..., page_icon=..., layout="wide")   │
├────────────────────────────────────────────────────────────────────────┤
│ [C] Title Block: st.title + st.caption + Narrative with 5-Point List   │
├────────────────────────────────────────────────────────────────────────┤
│ [D] First st.divider()                                                 │
├────────────────────────────────────────────────────────────────────────┤
│ [E] Content Sections 1 through N (Separated by st.divider())           │
│     ├── Type 1: Dual-Stage Retrieval / RAG Mesh Diagram (Graphviz Dark)│
│     ├── Type 2: Chunking & Retrieval Strategy Comparative Matrix       │
│     ├── Type 3: Production Runnable RAG Code Pipeline                  │
│     ├── Type 4: Interactive RAG Query, Rerank & Citation Simulator     │
│     └── Type 6: Retrieved Chunks & Similarity Score DataFrames         │
├────────────────────────────────────────────────────────────────────────┤
│ [F] Section N+1: Technical Interview Preparation (Mandatory 2nd-Last)  │
│     ├── Exactly 5 High-Frequency Q&A Expanders                         │
│     ├── Exactly 4 Production Traps (2x st.error + 2x st.warning)       │
│     └── Exactly 4 Interactive Flashcards in 2 Columns                  │
├────────────────────────────────────────────────────────────────────────┤
│ [G] st.divider()                                                       │
├────────────────────────────────────────────────────────────────────────┤
│ [H] Section N+2: Self-Assessment Checklist (Mandatory Last Section)    │
│     ├── Exactly 6 Checkboxes (All Initialized with value=True)         │
│     └── if chk1..chk6: -> st.success(Next Lesson) [Strictly NO balloons]│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Structural Component Specifications

### 3.1 Standard Imports Block

```python
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional
```

Strictly prohibit `import os` and `from pathlib import Path`.

### 3.2 Page Configuration (`st.set_page_config`)

```python
st.set_page_config(
    page_title="<X.Y> <Lesson Title> - RAG Curriculum",
    page_icon="🔍",
    layout="wide"
)
```

### 3.3 Title Block & Opening Narrative

```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
<2–4 sentence high-level narrative detailing grounded contextual retrieval, knowledge injection,
and how RAG eliminates hallucinations while providing verifiable citations in enterprise LLMs.>

In this lesson, you will master:
1. <Retrieval pipeline architecture and chunking strategy 1>
2. <Hybrid search and Reciprocal Rank Fusion (RRF) 2>
3. <Cross-encoder reranking and relevance scoring 3>
4. <Production retrieval pipeline implementation 4>
5. <Context stuffing, lost-in-the-middle, and hallucination failure modes 5>
""")
st.divider()
```

---

## 4. Heavy Multi-Tier & Dual-Stage RAG Diagrams Standard

### 4.1 Production Dual-Stage RAG Pipeline Template

```python
dot_rag_mesh = """
digraph RAGPipeline {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    subgraph cluster_query {
        label="Query Ingress";
        bgcolor="#1E293B";
        fontcolor="#38BDF8";
        color="#334155";
        q [label="User Question\\n'How does FlashAttention work?'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_stage1 {
        label="Stage 1: Hybrid Retrieval";
        bgcolor="#1E293B";
        fontcolor="#F59E0B";
        color="#334155";
        sparse [label="Sparse BM25 Index\\n(Exact Keyword Match)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        dense [label="Dense Vector Store\\n(Semantic Vector Search)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        rrf [label="Reciprocal Rank Fusion\\nMerges Top-50 Chunks", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
    }

    subgraph cluster_stage2 {
        label="Stage 2: Precision Reranking";
        bgcolor="#1E293B";
        fontcolor="#10B981";
        color="#334155";
        rerank [label="Cross-Encoder Reranker\\nFull Cross-Attention Top-5 Scoring", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
    }

    subgraph cluster_synthesis {
        label="Stage 3: Grounded Synthesis";
        bgcolor="#1E293B";
        fontcolor="#EC4899";
        color="#334155";
        llm [label="LLM Generator\\nProduces Answer with Exact Citations", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
    }

    q -> sparse [label="Keywords"];
    q -> dense [label="Query Vector"];
    sparse -> rrf [label="Ranked BM25"];
    dense -> rrf [label="Ranked Dense"];
    rrf -> rerank [label="Top-50 Candidates"];
    rerank -> llm [label="Top-5 Grounded Chunks"];
}
"""
st.graphviz_chart(dot_rag_mesh, use_container_width=True)
```

---

## 5. Mandatory Pedagogical Endings

### 5.1 Technical Interview Preparation (Section N+1)

Must contain:

1. **Exactly 5 Expanders**: High-frequency RAG architecture, chunking, reranking, and evaluation interview questions.
2. **Exactly 4 Traps**:
   - `Trap 1` (`st.error`): Lost-in-the-middle trap (placing critical retrieval facts in the middle of a massive context window).
   - `Trap 2` (`st.warning`): Fixed character chunking splitting crucial sentences across two fragmented chunks.
   - `Trap 3` (`st.error`): Bi-encoder single-stage retrieval failure on domain acronyms and IDs (requiring hybrid BM25).
   - `Trap 4` (`st.warning`): Outdated index sync drift (failing to invalidate deleted knowledge base documents).
3. **Exactly 4 Flashcards**: Rendered in a 2-column layout (`col_fc1`, `col_fc2`).

### 5.2 Self-Assessment Checklist (Section N+2)

Must contain:

- Exactly 6 checkboxes initialized with `value=True`.
- `if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:` triggering `st.success(...)` (strictly prohibit `st.balloons()`).
