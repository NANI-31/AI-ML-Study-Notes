You are an expert Principal Systems Engineer, AI Systems Architect, and Curriculum Director specializing in production-grade technical education.

Your task is to design and write the complete, authoritative master specification document: `UNIVERSAL_LESSON_FORMAT.md` for the following three diverse core engineering and analytics domains:

1. **AI Agents & Agentic Systems** (`04_Generative_AI/8. AI Agent/`, Syllabus: `Ai Agent.txt`)
2. **Docker & Container Systems** (`05_Deployment/3. Docker/`, Syllabus: `docker.txt`)
3. **Spreadsheet Engineering & Modern Analytics** (`01_Data_Science_Foundations/3. Spreadsheet/`, Syllabus: `spreadsheet.txt`)

Reference Master Architecture Specification:
Adapt the structural foundation from: `04_Generative_AI/2. LLM/UNIVERSAL_LESSON_FORMAT.md` and `04_Generative_AI/6. RAG/UNIVERSAL_LESSON_FORMAT.md`.

---

## 1. TARGET DOMAIN CURRICULUM ARCHITECTURES

### Domain 1: AI Agents & Agentic Systems (`04_Generative_AI/8. AI Agent/`)
- **Syllabus Reference**: `Ai Agent.txt` (Chapters 1–36)
- **Core Engineering Pillars**:
  1. *Agent Foundations*: Cognitive architectures, Agentic AI vs Chatbots vs RAG vs Automation, Perception-Reasoning-Planning-Action-Observation-Reflection loops.
  2. *Tool Calling & Actuation*: JSON schema function declarations, tool execution engines, parallel/sequential tool calls, tool routing, error recovery.
  3. *Prompting & Cognitive Patterns*: ReAct (Thought-Action-Observation), Plan-and-Execute, Reflexion, Self-Correction, Planner/Critic prompting.
  4. *State & Memory Systems*: Stateful agent schemas, checkpoints, time-travel, short-term context window trimming, long-term episodic/semantic storage, vector DBs.
  5. *Orchestration Frameworks*: LangChain LCEL tools/agents, LangGraph cyclic state machines (Nodes, Edges, Reducers, Interrupts, Subgraphs).
  6. *Agentic RAG*: Dynamic multi-hop retrieval, query routing, source verification, self-correcting retrieval across Vector, SQL, and Web.
  7. *Specialized Agents*: Web/Browser automation, Natural Language-to-SQL data agents, Repository-level coding and debugging agents.
  8. *Multi-Agent Swarms & Protocols*: Supervisor routing, hierarchical swarms, agent handoffs, Model Context Protocol (MCP) clients, servers, tools, and resources.
  9. *Production Operations & Safety*: Human-in-the-Loop (HITL) approval gates, guardrails, prompt injection defense, agent evaluation (LLM-as-a-judge), distributed tracing, cost/latency optimization, FastAPI/Docker deployment.

### Domain 2: Docker & Container Systems (`05_Deployment/3. Docker/`)
- **Syllabus Reference**: `docker.txt` (Chapters 1–25)
- **Core Engineering Pillars**:
  1. *Container Foundations*: OCI container standards, Docker Engine architecture (Client, Daemon, containerd, runc), Containers vs Virtual Machines.
  2. *CLI & Lifecycle Operations*: Container states (Created, Running, Paused, Stopped), `docker run`, `ps`, `exec`, `inspect`, `logs`, `stats`.
  3. *Images & Layer Mechanics*: Read-only image layers, UnionFS / Overlay2, image tagging, version pinning, Alpine vs Slim vs Distroless.
  4. *Dockerfile Engineering*: Directives (`FROM`, `WORKDIR`, `COPY`, `RUN`, `CMD`, `ENTRYPOINT`, `ENV`, `ARG`, `EXPOSE`, `USER`), layer caching, `.dockerignore`.
  5. *Storage & Persistence*: Ephemeral writable layer, Named Volumes, Bind Mounts, tmpfs mounts, database persistence strategies.
  6. *Container Networking*: Bridge, Host, None, Overlay networks, container DNS discovery, port publishing (`-p`), inter-container communication.
  7. *Multi-Container Orchestration*: Docker Compose v2 specification, service dependencies (`depends_on`), environment inheritance, health checks (`HEALTHCHECK`).
  8. *Optimization & Multi-Stage Builds*: Builder vs Runtime stages, BuildKit cache mounts, layer minimization, image size reduction.
  9. *Production Security & Enterprise Ops*: Linux kernel primitives (namespaces, cgroups), non-root execution, vulnerability scanning (Trivy/Scout), CI/CD pipelines (GitHub Actions), Kubernetes Pod bridge.

### Domain 3: Spreadsheet Engineering & Modern Analytics (`01_Data_Science_Foundations/3. Spreadsheet/`)
- **Syllabus Reference**: `spreadsheet.txt` (Chapters 1–28)
- **Core Engineering Pillars**:
  1. *Grid Architecture & Evaluation Engine*: Workbook/Sheet hierarchy, cell coordinates, relative vs absolute vs mixed referencing (`$A$1`), calculation engine dependency trees.
  2. *Modern Formulas & Logic*: Operator precedence, Boolean logic (`IF`, `IFS`, `AND`, `OR`), conditional aggregations (`SUMIFS`, `COUNTIFS`), next-gen lookups (`XLOOKUP`, `INDEX+MATCH`).
  3. *Text & Temporal Engineering*: String transformation (`TEXTJOIN`, `SUBSTITUTE`), date serial numbers, business calendar calculations (`EDATE`, `EOMONTH`, `NETWORKDAYS`).
  4. *Dynamic Array Revolution*: Calculation engine spill ranges (`#`), array functions (`FILTER`, `SORT`, `SORTBY`, `UNIQUE`, `VSTACK`, `HSTACK`, `SEQUENCE`, `LAMBDA`).
  5. *Structured Data & Cleaning*: Structured References (`Table1[@Column]`), data normalization (wide vs long), Text-to-Columns, Flash Fill, Data Validation dropdowns.
  6. *Enterprise ETL with Power Query*: M Language pipelines, extract-transform-load workflows, appending, merging, unpivoting, scheduled pipeline refresh.
  7. *Relational Data Modeling & DAX*: Star Schema design, Fact vs Dimension tables, Power Pivot data models, DAX measures (`CALCULATE`, `FILTER`, `ALL`), Filter Context vs Row Context, Context Transition, Time Intelligence.
  8. *Interactive Dashboards & Visualization*: Executive KPI cards, Slicers, Timelines, chart typography, dynamic title bindings.
  9. *Programmability, Enterprise & AI*: VBA macro automation, Google Apps Script, Python in Excel (`pandas`, `openpyxl`), SQL database queries, AI-assisted formula generation, and Spreadsheet + RAG workflows.

---

## 2. CRITICAL CURRICULUM CONSTRAINTS (STRICT ENFORCEMENT)

### 1. ABSOLUTELY ZERO MATHEMATICAL EQUATIONS:
- This curriculum is strictly focused on **Software Engineering, Systems Architecture, and Practical Data Operations**.
- Strictly prohibit ALL LaTeX notation (`$$`, `$`), calculus, loss functions, gradient descent, linear algebra matrices, and academic math formulas.
- **For AI Agents**: Prohibit probability calculus or Bellman equations. Frame everything as discrete cognitive loops, JSON state contracts, tool dispatch schemas, and execution graph nodes.
- **For Docker**: Frame all container mechanics as Linux kernel isolation primitives (namespaces, cgroups), file system overlay union mounts, network bridges, and process lifecycles.
- **For Spreadsheets**: Never use abstract mathematical sigma summation notation or calculus notation. Express all calculations strictly as practical spreadsheet formula syntax (e.g. `=SUMIFS()`, `=XLOOKUP()`, `=CALCULATE()`, `=FILTER()`), business metrics (margin, variance, growth rates), and structured data tables.

### 2. STRICT 1:1 TOPIC DENSITY FORMULA:
- Every single bullet point from the target domain's curriculum syllabus file (`Ai Agent.txt`, `docker.txt`, `spreadsheet.txt`) MUST receive its own dedicated `st.header("X. <Topic Name>", divider="rainbow")`.
- Never compress, merge, or collapse multiple syllabus topics into a single generic overview.
- Total Sections per lesson file is locked by formula:
  `Total Sections = 1 (Architecture Blueprint) + K (Roadmap Syllabus Topics) + 1 (Production Implementation) + 1 (Interactive Workbench) + 1 (Technical Interview Prep) + 1 (Self-Assessment Checklist) = K + 5`.
- Clean sequential integer numbering only (1, 2, 3... K+5). Strictly prohibit compound, dotted, or nested numbering (e.g. NEVER use "2.1", "2. 1.", "3. 2.").

### 3. MANDATORY TOP-TO-BOTTOM DIAGRAM FLOW & `st.expander` ENCAPSULATION:
- **Top-to-Bottom Flow (`rankdir=TB;`) Mandatory**: Long horizontal pipelines (`rankdir=LR;` with more than 3–4 sequential nodes) are **strictly forbidden** because Streamlit scales SVGs to container width, crushing wide diagrams into flat, illegible horizontal strips. All multi-stage pipelines, container architectures, and agent loops must use **`rankdir=TB;` (Top to Bottom)** so they expand vertically with readable fonts (`fontsize=11`) and generous margins (`margin="0.2,0.1"`).
- **Mandatory `st.expander` Container for Blueprints**: Wrap macro architectural blueprints inside an expander:
  ```python
  with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
      st.graphviz_chart(dot_blueprint, use_container_width=True)
  ```
- **Contextual In-Section Diagrams (Mandatory Whenever Required)**: Diagrams must NOT be restricted to Section 1 alone. Whenever any roadmap topic section covers a multi-step workflow, data lifecycle, state machine, or container layout, it **MUST include its own dedicated contextual Graphviz diagram inside that specific section**.
  - *AI Agents*: ReAct cognitive loops, tool dispatch flows, human-in-the-loop interruption breakpoints, supervisor routing trees.
  - *Docker*: Dockerfile layer cache stacks, container namespace isolation boundaries, multi-container compose networks, volume mount topologies.
  - *Spreadsheets*: Power Query ETL pipelines, Star Schema entity relationships, Dynamic Array spill ranges, DAX filter context transition flows.
- **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).

### 4. MODERN VS. LEGACY COMPARISON CALLOUTS:
- For every legacy abstraction or classic practice taught, include a mandatory callout block:
  `### 🚀 Modern Production Reality: How Current Industry Stacks Implement This`
- Explicitly contrast:
  - **AI Agents**: Brittle regex text scrapers and rigid DAG chains vs Autonomous tool-calling agents with self-reflection, structured outputs, cyclic graph orchestration (LangGraph), and standard protocols (MCP).
  - **Docker**: Monolithic "works on my machine" virtual machines and fat 2GB images vs Minimal multi-stage builds (Distroless/Alpine), OCI standards, rootless runtimes, BuildKit cache mounts, and immutable infrastructure.
  - **Spreadsheets**: Fragile nested `VLOOKUP`, volatile formulas (`OFFSET`/`INDIRECT`), and manual copy-pasting vs Dynamic array formulas (`XLOOKUP`, `FILTER`, `UNIQUE`), automated Power Query pipelines, and relational Power Pivot data models.

### 5. PRODUCTION-READY CODE & WORKBENCH:
- Standard 5 imports: `streamlit`, `pandas`, `numpy`, `graphviz`, `typing`.
- Strictly prohibit `import os` and `from pathlib import Path`.
- All helper classes, simulator functions, or mock engines must be defined in real Python module scope (never solely inside code block strings).
- Fully runnable, realistic mock executions so lessons execute completely and reliably without external paid dependencies.

### 6. MANDATORY PEDAGOGICAL ENDINGS:
- **Technical Interview Preparation**: Exactly 5 real-world technical/systems interview Q&As + 4 Production Traps (2 error, 2 warning) + 4 Flashcards.
- **Self-Assessment Checklist**: Exactly 6 functional checkpoints + clean `st.success(...)` completion banner. Strictly prohibit `st.balloons()`.

---

## 3. UNIFIED PYTHON SKELETON REFERENCE IMPLEMENTATION

Every `.py` lesson across all three domains must implement this exact blueprint structure:

```python
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="<X.Y> <Lesson Title> - <Domain Name>",
    page_icon="⚡",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("<X.Y> <Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: <Domain Name>")
st.markdown("""
<2–4 sentence high-level narrative setting industrial context, systems motivation,
and practical operational value in production stacks.>

In this lesson, you will master:
- **Key Competency 1**: Core architectural foundations.
- **Key Competency 2**: Production implementation patterns.
- **Key Competency 3**: Operational failure modes and anti-patterns.
- **Key Competency 4**: Real-world interview and systems engineering trade-offs.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT (TOP-TO-BOTTOM) ---
st.header("1. Architectural Blueprint: <System Workflow>", divider="rainbow")

dot_blueprint = """
digraph SystemBlueprint {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_tier1 {
        label="Input & Ingestion Tier";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        node_input [label="User / Trigger Payload\\n(Request / Data Source)", fillcolor="#0F172A"];
        node_parser [label="Schema Validation / Intake\\n(Strict Typing / Contract)", fillcolor="#1E293B"];
    }

    subgraph cluster_tier2 {
        label="Execution & Processing Core";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        node_engine [label="Core Processing Engine\\n(Agent Loop / BuildKit / Calc Engine)", fillcolor="#1E293B"];
        node_state [label="State / Context Store\\n(Checkpoints / Layers / Data Model)", fillcolor="#1E293B"];
    }

    subgraph cluster_tier3 {
        label="Output & Actuation Tier";
        fontcolor="#10B981";
        style=dashed;
        color="#334155";
        node_actuator [label="Actuator / Container Runtime / UI Grid\\n(Verified Result / Output)", fillcolor="#064E3B"];
    }

    node_input -> node_parser [label="Ingest"];
    node_parser -> node_engine [label="Dispatch"];
    node_engine -> node_state [label="Sync State", dir=both];
    node_engine -> node_actuator [label="Publish / Emit"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

# --- SECTIONS 2 THROUGH K+1: ROADMAP TOPIC SECTIONS ---
# Exactly K sections corresponding 1:1 with syllabus bullets
st.header("2. <First Syllabus Topic>", divider="rainbow")
st.markdown("<Deep, rigorous technical breakdown of topic mechanics...>")

# Contextual diagram when explaining workflows or multi-node structures
with st.expander("🔍 View Flow Diagram: <Subsystem>", expanded=False):
    dot_sub = """
    digraph SubFlow {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#334155"];
        edge [color="#38BDF8"];
        s1 [label="Step 1: Init"];
        s2 [label="Step 2: Transform / Evaluate"];
        s3 [label="Step 3: Verification"];
        s1 -> s2 -> s3;
    }
    """
    st.graphviz_chart(dot_sub, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current Industry Stacks Implement This
<Explanation of how modern systems solve this compared to legacy approaches.>
""")

df_compare = pd.DataFrame({
    "Dimension": ["Execution Model", "Failure Handling", "Scale / Latency", "Industry Standard"],
    "Legacy Approach": ["Brittle / Manual", "Silent failure", "High overhead / O(N^2)", "Deprecated pattern"],
    "Modern Approach": ["Autonomous / Declarative", "Strict schema / Auto-retry", "Optimized / Zero-copy", "Current production standard"]
})
st.table(df_compare)

# Repeat for topics 3 through K+1...

# --- SECTION K+2: PRODUCTION IMPLEMENTATION ---
st.header(f"{K+2}. Production Implementation: <Domain Pipeline>", divider="rainbow")
st.code("""
# Fully runnable production implementation code
""", language="python")

# --- SECTION K+3: INTERACTIVE WORKBENCH ---
st.header(f"{K+3}. Interactive Workbench: <Simulator>", divider="rainbow")

# Helper classes and simulator logic MUST reside in real module scope
class ProductionSimulator:
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "SUCCESS", "records_processed": 100, "metrics": params}

sim = ProductionSimulator()
st.subheader("Configure Parameters")
col1, col2 = st.columns(2)
with col1:
    p1 = st.selectbox("Mode", ["Standard", "High Performance", "Strict Validation"])
with col2:
    p2 = st.slider("Concurrency / Batch Size", 1, 64, 16)

if st.button("🚀 Run Simulation", type="primary"):
    res = sim.execute({"mode": p1, "batch_size": p2})
    st.json(res)

# --- SECTION K+4: TECHNICAL INTERVIEW PREPARATION ---
st.header(f"{K+4}. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
for i, (q, a) in enumerate([
    ("Question 1", "Answer 1"),
    ("Question 2", "Answer 2"),
    ("Question 3", "Answer 3"),
    ("Question 4", "Answer 4"),
    ("Question 5", "Answer 5")
], 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Breaking Error)**: Fatal misconfiguration leading to container crashes or infinite loops.")
    st.error("**Trap 2 (Breaking Error)**: Unpinned versions or unhandled state transitions causing runtime data corruption.")
with col_trap2:
    st.warning("**Trap 3 (Performance Warning)**: High latency or cache invalidation due to poor instruction ordering.")
    st.warning("**Trap 4 (Security Warning)**: Running with excessive privileges or exposing unauthenticated endpoints.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("🎴 Flashcard 1: Core Lifecycle"):
        st.markdown("**Answer & Mechanism**")
    with st.expander("🎴 Flashcard 2: Optimization"):
        st.markdown("**Answer & Mechanism**")
with fc_col2:
    with st.expander("🎴 Flashcard 3: Security & Governance"):
        st.markdown("**Answer & Mechanism**")
    with st.expander("🎴 Flashcard 4: Failure Recovery"):
        st.markdown("**Answer & Mechanism**")

st.divider()

# --- SECTION K+5: SELF-ASSESSMENT CHECKLIST ---
st.header(f"{K+5}. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can explain the foundational architecture without notes.", value=True)
chk2 = st.checkbox("I understand the multi-stage execution and state flow.", value=True)
chk3 = st.checkbox("I can identify and avoid common production traps.", value=True)
chk4 = st.checkbox("I understand modern industry practices vs legacy techniques.", value=True)
chk5 = st.checkbox("I can implement and debug the production code snippet.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on this lesson. Ready for the next module!")
```

---

## 4. MASTER SPECIFICATION DESTINATIONS

When this prompt is executed, the complete `UNIVERSAL_LESSON_FORMAT.md` files must be generated and deployed to:
1. `d:\Data\GitHub\AI-ML-Study-Notes\04_Generative_AI\8. AI Agent\UNIVERSAL_LESSON_FORMAT.md`
2. `d:\Data\GitHub\AI-ML-Study-Notes\05_Deployment\3. Docker\UNIVERSAL_LESSON_FORMAT.md`
3. `d:\Data\GitHub\AI-ML-Study-Notes\01_Data_Science_Foundations\3. Spreadsheet\UNIVERSAL_LESSON_FORMAT.md`
