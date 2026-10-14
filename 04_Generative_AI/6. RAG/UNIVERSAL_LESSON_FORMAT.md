# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for Domain: `6. RAG` (Retrieval-Augmented Generation)
### Target Curriculum: `04_Generative_AI/6. RAG`
### Core Reference Standard: Continuous Vertical Scroll · Dark Graphviz · Zero LaTeX Math · Advanced & Agentic RAG

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `6. RAG` represents an interactive, production-grade learning module covering enterprise document ingestion, semantic chunking strategies, hybrid dense and sparse search, cross-encoder precision reranking, context window compression, and agentic self-correcting retrieval loops.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden** because they hide educational content behind clicks.
2. **Mandatory 1:1 Section Density (Every Roadmap Bullet Gets a Dedicated Section)**: Every single numbered bullet point under a lesson in `rag.txt` **MUST receive its own dedicated `st.header("X. <Topic Name>", divider="rainbow")` section**. Never compress or merge topics.
   - Total Sections per lesson file is locked by formula:
     `Total Sections = 1 (Architecture Blueprint) + K (Roadmap Syllabus Topics) + 1 (Production Implementation) + 1 (Interactive Workbench) + 1 (Technical Interview Prep) + 1 (Self-Assessment Checklist) = K + 5`.
3. **Clean Single Sequential Section Numbering (Strictly Zero Compound Numbering)**: Section headers MUST ALWAYS use clean, single sequential integer numbering from 1 to N (`st.header("1. <Blueprint>", divider="rainbow")`, `st.header("2. <Topic 1 Title>", divider="rainbow")`, ..., `st.header(f"{K+2}. Production Implementation: ...")`). Strictly prohibit compound, dotted, or nested section numbering (NEVER use `st.header("2. 1. ...")`, `st.header("2.1 ...")`, `st.header("3. 2. ...")`).
4. **ABSOLUTELY ZERO MATHEMATICAL EQUATIONS**:
   - This curriculum is strictly focused on Retrieval Systems Engineering, Context Management, and Search Pipelines.
   - Strictly prohibit ALL LaTeX notation (`$$...$$`, `$x$`), calculus, linear algebra matrix operations, and vector dot product formulas.
   - Express all retrieval fusion and scoring concepts in pure Systems Engineering:
     - Instead of cosine similarity equations: Explain similarity as a normalized threshold filter (e.g. `similarity_score >= 0.75`).
     - Instead of BM25 formulas: Explain BM25 as term-frequency keyword matching with document length normalization.
     - Instead of algebraic rank equations: Explain Reciprocal Rank Fusion (RRF) as rank-order list aggregation where higher positions in sparse and dense lists yield priority candidate status.
5. **Modern Advanced & Agentic RAG Production Reality Contrast**:
   - Every section covering legacy retrieval patterns must feature a prominent callout block:
     `### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This`
   - Explicitly contrast Naive RAG vs Modern Advanced / Agentic RAG:
     - Naive single-stage vector top-K lookup vs Dual-Stage Hybrid Search (BM25 sparse + Dense vector) merged via RRF.
     - Blind fixed-character chunking vs Semantic, recursive, and hierarchical Parent-Document / Multi-Vector chunking.
     - Static unverified context stuffing vs Precision Cross-Encoder Reranking and Contextual Compression.
     - Linear ungrounded generation vs Agentic Corrective RAG (CRAG) with web search fallbacks and Self-RAG reflection loops.
6. **Heavy Multi-Tier Architectural & Topic Diagrams (Mandatory Whenever Required)**: Every lesson must feature Graphviz diagrams with dark theme styling (`bgcolor="#0B1120"`, cluster containers `#1E293B`, edges `#38BDF8`).
   - **Mandatory Top-to-Bottom (`rankdir=TB;`) Orientation for Long Pipelines**: Strictly prohibit long horizontal `rankdir=LR;` layouts (more than 3–4 sequential nodes). Long horizontal graphs get squeezed into flat, unreadable strips in Streamlit. Always use **`rankdir=TB;` (Top to Bottom)** so nodes expand downwards with large, readable fonts and generous padding.
   - **Mandatory `st.expander` Container**: Always wrap long, multi-stage, or blueprint architectural diagrams inside an `st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):` calling `st.graphviz_chart(dot_code, use_container_width=True)`. This ensures diagrams render at full readable scale and do not get vertically or horizontally crushed.
   - **Section 1 Blueprint**: The macro end-to-end pipeline (Ingestion -> Retrieval -> Rerank -> Generation).
   - **Contextual In-Section Diagrams (Mandatory Whenever Required)**: Diagrams must NOT be restricted to Section 1 alone. Whenever any roadmap topic section covers a multi-step workflow, data structure, or algorithmic mechanism (e.g. Chunking boundaries, Sparse/Dense Hybrid Fusion, Cross-Encoder pairwise attention, Context Compression, CRAG routing), it **MUST include its own dedicated contextual Graphviz diagram or visual data layout directly inside that section**.
7. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
8. **Module Scope for Interactive Workbench Classes**: Any classes, retrievers, or mock pipelines needed by the Interactive Workbench must be defined in real Python module scope within the script. Never define executable classes solely inside string literals passed to `st.code(...)`.
9. **Standard Imports**: Exactly 5 standard imports at the top of the file:
   ```python
   import streamlit as st
   import pandas as pd
   import numpy as np
   import graphviz
   from typing import List, Dict, Any, Tuple, Optional, Union
   ```
   Strictly prohibit `import os` and `from pathlib import Path`.
10. **Uniform Pedagogical Endings**: Every single lesson concludes with:
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
│ [E] Section 1: End-to-End Architectural Diagram (Graphviz Dark)        │
├────────────────────────────────────────────────────────────────────────┤
│ [F] Sections 2 through K+1: Mandatory 1:1 Syllabus Roadmap Topics      │
│     ├── Deep conceptual explanation (Pure Systems Engineering)        │
│     ├── 📊 Visual Diagram / Flowchart (Mandatory whenever required:    │
│     │   multi-step flows, chunk boundaries, data models, or routing)   │
│     ├── 🚀 Modern Production Reality: How Current GenAI Stacks Follow  │
│     ├── Side-by-Side Comparison Matrix (Naive RAG vs Advanced RAG)     │
│     └── Interactive Mini-Demo / Code Demonstration                     │
├────────────────────────────────────────────────────────────────────────┤
│ [G] Section K+2: Production Implementation: End-to-End Pipeline        │
│     ├── Real-world runnable Python script (Hybrid search + Reranking)  │
│     └── Type-safe error handling and verifiable citation formatting    │
├────────────────────────────────────────────────────────────────────────┤
│ [H] Section K+3: Interactive Workbench Simulator                       │
│     ├── Real-time input manipulation widgets (Query, Sliders, Filters) │
│     └── Live candidate retrieval, reranking scores, and context preview│
├────────────────────────────────────────────────────────────────────────┤
│ [I] Section K+4: Technical Interview Preparation & Production Traps    │
│     ├── Exactly 5 High-Frequency Interview Q&A Expanders               │
│     ├── Exactly 4 Production Traps (2x st.error, 2x st.warning)        │
│     └── Exactly 4 Flashcards in a 2-Column Responsive Grid             │
├────────────────────────────────────────────────────────────────────────┤
│ [J] Section K+5: Self-Assessment Checklist (Mandatory Last Section)    │
│     ├── Exactly 6 Checkboxes (All Initialized with value=True)         │
│     └── if chk1..chk6: -> st.success(...) [Strictly NO balloons]       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Structural Component Code Templates

### 3.1 Title Block & 5-Pillar Overview
```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: 6. Retrieval-Augmented Generation")
st.markdown("""
<2-4 sentence high-level narrative detailing grounded contextual retrieval, knowledge injection, 
and how production RAG eliminates hallucinations while providing verifiable citations in enterprise systems.>

In this lesson, you will master:
1. <Document Parsing and Ingestion Hierarchy Pillar 1>
2. <Hybrid Search: Combining Keyword and Semantic Match 2>
3. <Precision Cross-Encoder Reranking and Context Compression 3>
4. <Agentic Retrieval Loops: Query Rewriting and Self-Correction 4>
5. <Failure Mode Mitigation: Lost-in-the-Middle and Context Poisoning 5>
""")
st.divider()
```

### 3.2 Dark-Theme Graphviz Template for Advanced Dual-Stage RAG Pipeline (Top-to-Bottom in Expander)
```python
with st.expander("🔍 View Complete Advanced RAG Architecture (Top-to-Bottom Flow)", expanded=True):
    dot_rag_mesh = """
    digraph AdvancedRAGPipeline {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Query Ingress & Rewriter Tier";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            q [label="User Query\\n'What is the refund policy?'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            rewriter [label="Agentic Query Rewriter\\nHyDE / Sub-Query Decomposition", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            q -> rewriter [label="Ingest"];
        }

        subgraph cluster_stage1 {
            label="2. Stage 1: Parallel Hybrid Retrieval Tier";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            fontsize=12;
            sparse [label="Sparse Index (BM25)\\nExact Keyword Matching", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            dense [label="Dense Vector Store\\nSemantic Intent Matching", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            rrf [label="Reciprocal Rank Fusion\\nMerges Top-50 Candidates", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            sparse -> rrf [label="Ranked Sparse"];
            dense -> rrf [label="Ranked Dense"];
        }

        subgraph cluster_stage2 {
            label="3. Stage 2: Precision Reranking & Compression Tier";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            fontsize=12;
            rerank [label="Cross-Encoder Reranker\\nFull Cross-Attention Top-5 Filter", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            compress [label="Context Compressor\\nExtracts Relevant Sentences", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            rerank -> compress [label="Top-5 Chunks"];
        }

        subgraph cluster_synthesis {
            label="4. Grounded Synthesis Tier";
            bgcolor="#1E293B";
            fontcolor="#EC4899";
            color="#334155";
            fontsize=12;
            llm [label="LLM Generator\\nProduces Grounded Answer with Citations", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
        }

        rewriter -> sparse [label="Keywords"];
        rewriter -> dense [label="Dense Vector"];
        rrf -> rerank [label="Top-50 Candidates"];
        compress -> llm [label="Filtered Context Window"];
    }
    """
    st.graphviz_chart(dot_rag_mesh, use_container_width=True)
```

### 3.3 Pedagogical Endings Template
```python
# Section K+4: Technical Interview Preparation & Production Traps
st.header(f"{K+4}. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why is a single-stage bi-encoder vector search insufficient for enterprise RAG?"):
    st.markdown("""
    **Answer**: Bi-encoder vector search excels at broad semantic concept matching but frequently fails on exact alphanumeric tokens 
    (part numbers, error codes, legal statute numbers, and acronyms). Furthermore, vector cosine similarity ignores fine-grained token-level cross-attention. 
    Production systems therefore implement dual-stage retrieval: Stage 1 combines BM25 keyword matching with dense vectors via Reciprocal Rank Fusion, 
    and Stage 2 applies a Cross-Encoder reranker to evaluate the candidate chunks with joint attention.
    """)
# [4 more expanders: Q2 to Q5]

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Lost-in-the-Middle Phenomenon)**: LLMs attend strongly to facts at the very beginning and very end of their context window, while ignoring facts in the middle. If critical retrieved chunks sit in the middle of a 20-chunk prompt, retrieval accuracy degrades sharply. Rerank and place highest-priority chunks at top positions.")
    st.warning("**Trap 2 (Blind Fixed-Character Chunking)**: Chunking strictly by character count without sentence or recursive boundary checks splits numbers, definitions, and sentences in half across chunk borders. Always use semantic or recursive paragraph chunking with overlap.")
with col_trap2:
    st.error("**Trap 3 (Context Poisoning)**: Injecting noisy or contradictory retrieved documents into the context window causes the model to hallucinate or regurgitate conflicting claims. Implement similarity score threshold cutoffs and context compression to filter out irrelevant chunks.")
    st.warning("**Trap 4 (Stale Vector Store Drift)**: Updating or deleting source documents in an enterprise database without propagating deletions to the vector store index creates 'ghost chunks' that poison search results with outdated policies.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the architectural difference between a Bi-Encoder and a Cross-Encoder in RAG?")
        st.caption("Flip for answer:")
        st.info("**Back**: A Bi-Encoder embeds queries and documents independently into vectors for fast index lookup. A Cross-Encoder processes the query and document simultaneously through full cross-attention layers, yielding high relevance accuracy at higher computational cost.")
    # [Flashcard 2]
with col_fc2:
    # [Flashcard 3 & 4]

# Section K+5: Self-Assessment Checklist
st.header(f"{K+5}. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the difference between Naive, Advanced, and Agentic RAG architectures.", value=True)
c2 = st.checkbox("I can implement hybrid retrieval combining BM25 keyword search and dense vector search.", value=True)
c3 = st.checkbox("I know how Reciprocal Rank Fusion (RRF) merges disparate ranking lists without raw score calibration.", value=True)
c4 = st.checkbox("I can attach Cross-Encoder rerankers to filter top candidates down to high-precision context chunks.", value=True)
c5 = st.checkbox("I understand strategies to mitigate the Lost-in-the-Middle context degradation failure mode.", value=True)
c6 = st.checkbox("I can implement verifiable source citation formatting in LLM generation prompts.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to the next module.")
```
