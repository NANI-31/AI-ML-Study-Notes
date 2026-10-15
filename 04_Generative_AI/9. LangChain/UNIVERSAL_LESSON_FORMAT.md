# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for Domain: `9. LangChain` (LCEL, Runnables & Tool Calling)
### Target Curriculum: `04_Generative_AI/9. LangChain`
### Core Reference Standard: Continuous Vertical Scroll · Dark Graphviz · Zero LaTeX Math · LangChain v0.3+ LCEL

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `9. LangChain` represents an interactive, production-grade learning module covering orchestration, composition pipelines via LangChain Expression Language (LCEL), `Runnable` interfaces, prompt engineering templates, tool calling, and streaming token lifecycles.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden** because they hide educational content behind clicks.
2. **Mandatory 1:1 Section Density (Every Roadmap Bullet Gets a Dedicated Section)**: Every single numbered bullet point under a lesson in `langchain.txt` **MUST receive its own dedicated `st.header("X. <Topic Name>", divider="rainbow")` section**. Never compress or merge topics.
   - Total Sections per lesson file is locked by formula:
     `Total Sections = 1 (Architecture Blueprint) + K (Roadmap Syllabus Topics) + 1 (Production Implementation) + 1 (Interactive Workbench) + 1 (Technical Interview Prep) + 1 (Self-Assessment Checklist) = K + 5`.
3. **Clean Single Sequential Section Numbering (Strictly Zero Compound Numbering)**: Section headers MUST ALWAYS use clean, single sequential integer numbering from 1 to N (`st.header("1. <Blueprint>", divider="rainbow")`, `st.header("2. <Topic 1 Title>", divider="rainbow")`, ..., `st.header(f"{K+2}. Production Implementation: ...")`). Strictly prohibit compound, dotted, or nested section numbering (NEVER use `st.header("2. 1. ...")`, `st.header("2.1 ...")`, `st.header("3. 2. ...")`).
4. **ABSOLUTELY ZERO MATHEMATICAL EQUATIONS**:
   - This curriculum is strictly focused on Orchestration Engineering, Component Composition, and Software Integration.
   - Strictly prohibit ALL LaTeX notation (`$$...$$`, `$x$`), matrix calculus, loss equations, and theoretical math.
   - Frame all concepts in pure Systems Engineering: Ingress dictionaries, Runnable pipelines (`RunnableSequence`, `RunnableParallel`, `RunnablePassthrough`), token stream buffers, tool schema bindings, and output parsing strategies.
5. **Modern LangChain v0.3+ Production Reality Contrast**:
   - Every section covering legacy abstractions must feature a prominent callout block:
     `### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This`
   - Explicitly contrast legacy LangChain vs modern LCEL:
     - Legacy `LLMChain`, `SimpleSequentialChain`, `TransformChain` (rigid, opaque, poorly typed) vs Modern LCEL (`prompt | model | parser`, declarative syntax, native parallel execution).
     - Legacy `AgentExecutor` with regex-based text parsing (`Action: ... Action Input: ...`) vs Native Chat Model tool calling (`model.bind_tools()` with JSON Schema).
     - Synchronous batch execution vs Native Async and Streaming protocols (`.invoke()`, `.batch()`, `.stream()`, `.astream_events()`).
     - Legacy string output parsing vs `with_structured_output(PydanticModel)`.
6. **Heavy Multi-Tier Architectural & Topic Diagrams (Mandatory Whenever Required)**: Every lesson must feature Graphviz diagrams with dark theme styling (`bgcolor="#0B1120"`, cluster containers `#1E293B`, edges `#38BDF8`).
   - **Section 1 Blueprint**: The macro end-to-end orchestration chain (Input -> Prompt -> Model -> Output Parser).
   - **Contextual In-Section Diagrams (Mandatory Whenever Required)**: Diagrams must NOT be restricted to Section 1 alone. Whenever any roadmap topic section covers a multi-step composition or execution flow (e.g. `RunnableParallel` branch joins, Tool calling dispatch loops, dynamic routing via `RunnableBranch`, streaming event pipes, or RAG retriever chains), it **MUST include its own dedicated contextual Graphviz diagram directly inside that section**.
7. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
8. **Module Scope for Interactive Workbench Classes**: Any classes, Runnables, or mock tools needed by the Interactive Workbench must be defined in real Python module scope within the script. Never define executable classes solely inside string literals passed to `st.code(...)`.
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
│     │   LCEL run trees, RunnableParallel joins, or tool dispatch loops)│
│     ├── 🚀 Modern Production Reality: How Current GenAI Stacks Follow  │
│     ├── Side-by-Side Comparison Matrix (Legacy Chain vs Modern LCEL)   │
│     └── Interactive Mini-Demo / Code Demonstration                     │
├────────────────────────────────────────────────────────────────────────┤
│ [G] Section K+2: Production Implementation: End-to-End Pipeline        │
│     ├── Real-world runnable Python script (LCEL / Runnable pipeline)   │
│     └── Tool binding, streaming chunks, and structured error handling  │
├────────────────────────────────────────────────────────────────────────┤
│ [H] Section K+3: Interactive Workbench Simulator                       │
│     ├── Real-time input manipulation widgets (Prompt, Tools, Chunks)   │
│     └── Live LCEL step-by-step pipeline execution and state inspector  │
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
st.caption("Chapter <N>: <Chapter Name> | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
<2-4 sentence high-level overview detailing component composition, declarative pipeline binding, 
and how modern LangChain enables production-grade LLM applications.>

In this lesson, you will master:
1. <LCEL Composition and Runnable Architecture Pillar 1>
2. <Chat Model Parameterization and Prompt Templating 2>
3. <Tool Binding and Structured Output Schemas 3>
4. <Streaming Lifecycles and Async Event Processing 4>
5. <Production Robustness, Fallbacks, and Error Recovery 5>
""")
st.divider()
```

### 3.2 Dark-Theme Graphviz Template for LCEL Pipeline
```python
lcel_mesh_dot = """
digraph LCELPipeline {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    subgraph cluster_input {
        label="1. Ingress State";
        bgcolor="#1E293B";
        fontcolor="#38BDF8";
        color="#334155";
        in_dict [label="Input Dictionary\\n{'query': 'Analyze logs'}", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_lcel {
        label="2. LCEL Runnable Pipeline (|)";
        bgcolor="#1E293B";
        fontcolor="#F59E0B";
        color="#334155";
        prompt [label="ChatPromptTemplate\\nFormats System & Human", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        passthru [label="RunnablePassthrough\\nPreserves Metadata", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        model [label="ChatModel.bind_tools()\\nDispatches API Call", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        parser [label="JsonOutputParser\\nValidates Schema", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
    }

    subgraph cluster_output {
        label="3. Egress Outcome";
        bgcolor="#1E293B";
        fontcolor="#10B981";
        color="#334155";
        res [label="Typed Output / Action\\nTool Call Spec or Text", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
    }

    in_dict -> prompt [label="Pipes |"];
    in_dict -> passthru [label="Parallel"];
    prompt -> model [label="Pipes |"];
    passthru -> model [label="Injects Context"];
    model -> parser [label="Pipes |"];
    parser -> res [label="Delivers"];
}
"""
st.graphviz_chart(lcel_mesh_dot, use_container_width=True)
```

### 3.3 Pedagogical Endings Template
```python
# Section K+4: Technical Interview Preparation & Production Traps
st.header(f"{K+4}. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why was LCEL introduced to replace legacy LLMChain and SequentialChain?"):
    st.markdown("""
    **Answer**: Legacy chains suffered from high abstraction opacity, poor TypeScript parity, clunky debugging, 
    and brittle state dictionaries. LCEL (LangChain Expression Language) introduces the `Runnable` protocol, 
    giving every component unified `.invoke()`, `.batch()`, `.stream()`, and `.astream_events()` support out of the box, 
    along with native fallback handling (`.with_fallbacks()`) and transparent parallel execution (`RunnableParallel`).
    """)
# [4 more expanders: Q2 to Q5]

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Streaming Blockers)**: Placing a standard blocking parser (like custom string post-processors) before an output stream halts chunk delivery until completion. Always use streaming-compatible output parsers or `astream_events`.")
    st.warning("**Trap 2 (State Loss in Chains)**: Passing a dictionary into an LCEL chain where intermediate steps overwrite keys causes downstream context starvation. Use `RunnablePassthrough.assign(...)` to preserve existing keys while appending new fields.")
with col_trap2:
    st.error("**Trap 3 (Legacy AgentExecutor in 2026)**: Relying on legacy `AgentExecutor` for multi-step agent loops causes fragile parsing failures. Modern production systems build agent loops using LangGraph state machines.")
    st.warning("**Trap 4 (Unbounded Retries)**: Adding `.with_retry()` without specifying `stop_after_attempt` or backoff intervals can rapidly exhaust API rate limits during downstream provider outages.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What does `RunnablePassthrough.assign()` do in an LCEL pipeline?")
        st.caption("Flip for answer:")
        st.info("**Back**: It accepts an input dictionary, evaluates new key-value pairs (often via sub-runnables), and merges them into the original dictionary without discarding the existing keys.")
    # [Flashcard 2]
with col_fc2:
    # [Flashcard 3 & 4]

# Section K+5: Self-Assessment Checklist
st.header(f"{K+5}. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the Runnable protocol methods: invoke, batch, stream, and astream_events.", value=True)
c2 = st.checkbox("I can construct robust LCEL pipelines using the pipe (|) operator.", value=True)
c3 = st.checkbox("I know how to bind external tools using model.bind_tools().", value=True)
c4 = st.checkbox("I can use RunnableParallel to execute multiple branches concurrently.", value=True)
c5 = st.checkbox("I can configure structured outputs using with_structured_output().", value=True)
c6 = st.checkbox("I understand how to attach production fallbacks using .with_fallbacks().", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to the next module.")
```
