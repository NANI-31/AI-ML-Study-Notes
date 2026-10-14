# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for Domain: `8. AI Agent` (Autonomous Agents, Cognitive Loops, Multi-Agent Swarms, Tool Calling & MCP)
### Target Curriculum: `04_Generative_AI/8. AI Agent/Ai Agent.txt` (Chapters 1–36)
### Core Reference Standard: Continuous Vertical Scroll · Dark Graphviz · Top-to-Bottom Flow · Zero LaTeX Math · LangGraph & MCP Core

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `8. AI Agent` represents an interactive, production-grade learning module covering autonomous cognitive loops, tool dispatch schemas, function calling, stateful graph orchestration, long-term memory architectures, multi-agent swarms, Model Context Protocol (MCP), and production safety guardrails.

To ensure consistency, pedagogical depth, and zero UI breakage across all lessons, **every single generated `.py` file must strictly follow this single unified structure**.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden** because they hide educational content behind clicks. (Sub-tabs are only permitted inside self-contained widgets or workbench result panels).
2. **Mandatory 1:1 Section Density (Every Roadmap Bullet Gets a Dedicated Section)**: Every single numbered bullet point under a lesson in `Ai Agent.txt` **MUST receive its own dedicated `st.header("X. <Topic Name>", divider="rainbow")` section**. Never compress or merge topics.
   - Total Sections per lesson file is locked by formula:
     `Total Sections = 1 (Architecture Blueprint) + K (Roadmap Syllabus Topics) + 1 (Production Implementation) + 1 (Interactive Workbench) + 1 (Technical Interview Prep) + 1 (Self-Assessment Checklist) = K + 5`.
3. **Clean Single Sequential Section Numbering (Strictly Zero Compound Numbering)**: Section headers MUST ALWAYS use clean, single sequential integer numbering from 1 to N (`st.header("1. <Blueprint>", divider="rainbow")`, `st.header("2. <Topic 1 Title>", divider="rainbow")`, ..., `st.header(f"{K+2}. Production Implementation: ...")`). Strictly prohibit compound, dotted, or nested section numbering (e.g. NEVER use `st.header("2. 1. ...")`, `st.header("2.1 ...")`, `st.header("3. 2. ...")`).
4. **ABSOLUTELY ZERO MATHEMATICAL EQUATIONS**:
   - This curriculum is strictly focused on **Software Systems Engineering, Agentic Control Loops, and Distributed Systems**.
   - Strictly prohibit ALL LaTeX notation (`$$...$$`, `$x$`), calculus, loss functions, Bellman equations, and probability matrices.
   - Frame all concepts in pure Systems Engineering: Cognitive control loops (Perception -> Reasoning -> Planning -> Action -> Observation -> Reflection), JSON schema validation contracts, tool dispatch tables, cyclic state transitions, message reducers, and checkpoint persistence.
5. **Modern AI Agent Production Reality Contrast**:
   - Every section covering legacy or classical patterns must feature a prominent callout block:
     `### 🚀 Modern Production Reality: How Current Industry Stacks Implement This`
   - Explicitly contrast legacy vs modern implementations:
     - Brittle regex scraping & rigid sequential chains vs Autonomous tool calling with JSON Schema and self-correction.
     - Uncontrolled free-form text generations vs Type-safe structured outputs (`with_structured_output`) and Pydantic models.
     - Stateless single-turn prompts vs State-managed cyclic graphs (LangGraph) with time-travel checkpoints and Human-in-the-Loop gates.
     - Ad-hoc custom API integrations vs Standardized tool interoperability via the Model Context Protocol (MCP).
6. **Mandatory Top-to-Bottom Diagram Flow (`rankdir=TB;`) & `st.expander` Encapsulation**:
   - **Top-to-Bottom Flow (`rankdir=TB;`) Mandatory**: Long horizontal pipelines (`rankdir=LR;` with more than 3–4 sequential nodes) are **strictly forbidden** because Streamlit scales SVGs to container width, crushing wide diagrams into flat, illegible horizontal strips. All agent loops, tool dispatch trees, and multi-agent hierarchies must use **`rankdir=TB;` (Top to Bottom)** so they expand vertically with readable fonts (`fontsize=11`) and generous margins (`margin="0.2,0.1"`).
   - **Mandatory `st.expander` Container for Blueprints**: Wrap macro architectural blueprints inside an expander:
     ```python
     with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
         st.graphviz_chart(dot_blueprint, use_container_width=True)
     ```
   - **Contextual In-Section Diagrams (Mandatory Whenever Required)**: Diagrams must NOT be restricted to Section 1 alone. Whenever any roadmap topic section covers a multi-step workflow, data structure, or state machine (e.g. ReAct thought-action-observation cycles, MCP client-server negotiation, tool retry loops, memory summarization pipelines), it **MUST include its own dedicated contextual Graphviz diagram inside that specific section**.
7. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
8. **Module Scope for Interactive Workbench Classes**: Any classes, agent runners, mock tools, or dispatch pipelines needed by the Interactive Workbench must be defined in real Python module scope within the script. Never define executable classes solely inside string literals passed to `st.code(...)`.
9. **Standard Imports**: Exactly 5 standard imports at the top of the file:
   ```python
   import streamlit as st
   import pandas as pd
   import numpy as np
   import graphviz
   from typing import List, Dict, Any, Tuple, Optional
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
│ [C] Title Block: st.title + st.caption + Narrative with 4-5 Pillars    │
├────────────────────────────────────────────────────────────────────────┤
│ [D] First st.divider()                                                 │
├────────────────────────────────────────────────────────────────────────┤
│ [E] Section 1: End-to-End Architectural Blueprint (Graphviz Dark TB)   │
│     └── Wrapped in st.expander with rankdir=TB;                        │
├────────────────────────────────────────────────────────────────────────┤
│ [F] Sections 2 through K+1: Mandatory 1:1 Syllabus Roadmap Topics      │
│     ├── Deep conceptual explanation (Systems Engineering / Schemas)    │
│     ├── 📊 Contextual Diagram (Mandatory whenever workflows appear)    │
│     ├── 🚀 Modern Production Reality: Industry Standard Callout        │
│     ├── Side-by-Side Comparison Matrix (Legacy vs Modern)              │
│     └── Interactive Mini-Demo / Code Demonstration                     │
├────────────────────────────────────────────────────────────────────────┤
│ [G] Section K+2: Production Implementation: End-to-End Pipeline        │
│     ├── Real-world runnable Python agent loop / tool dispatch          │
│     └── Robust error handling, state validation & stop conditions      │
├────────────────────────────────────────────────────────────────────────┤
│ [H] Section K+3: Interactive Workbench Simulator                       │
│     ├── Real-time input manipulation widgets (Sliders, Text Areas)     │
│     └── Live Agent Loop / Tool Dispatcher Inspector                    │
├────────────────────────────────────────────────────────────────────────┤
│ [I] Section K+4: Technical Interview Preparation & Production Traps    │
│     ├── Exactly 5 High-Frequency Interview Q&A Expanders               │
│     ├── Exactly 4 Production Traps (2x st.error, 2x st.warning)        │
│     └── Exactly 4 Flashcards in a 2-Column Responsive Grid             │
├────────────────────────────────────────────────────────────────────────┤
│ [J] st.divider()                                                       │
├────────────────────────────────────────────────────────────────────────┤
│ [K] Section K+5: Self-Assessment Checklist (Mandatory Last Section)    │
│     ├── Exactly 6 Checkboxes (All Initialized with value=True)         │
│     └── if chk1..chk6: -> st.success(...) [Strictly NO balloons]       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Structural Component Code Templates

### 3.1 Title Block & 5-Pillar Overview
```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: 8. AI Agent")
st.markdown("""
<2-4 sentence high-level overview explaining how this agent concept operates,
why autonomous systems require this mechanism, and how it solves production reliability challenges.>

In this lesson, you will master:
- **Core Cognitive Architecture**: Foundational control loops and tool actuation contracts.
- **State & Memory Management**: Persistence, checkpoints, and multi-turn state propagation.
- **Resilient Tool Dispatch**: Handling tool failures, retries, and schema validation.
- **Production Guardrails**: Input/output filtering, rate limits, and Human-in-the-Loop oversight.
- **Systems Interview Readiness**: Architecture trade-offs and high-frequency technical questions.
""")
st.divider()
```

### 3.2 Dark-Theme Graphviz Template for AI Agents (Top-to-Bottom Flow)
```python
dot_blueprint = """
digraph AgentArchitecture {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_perception {
        label="Perception & Ingestion Tier";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        user_query [label="User Goal / Event Trigger\\n(Natural Language / Payload)", fillcolor="#0F172A"];
        prompt_builder [label="Prompt & Context Assembler\\n(System Prompt + Working Memory)", fillcolor="#1E293B"];
    }

    subgraph cluster_reasoning {
        label="Reasoning & Planning Core";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        llm_engine [label="Foundation LLM Core\\n(Reasoning / Decision / Plan)", fillcolor="#1E293B"];
        tool_router [label="Tool Call Decision\\n(Extract Function Schema)", fillcolor="#1E293B"];
        state_store [label="State & Checkpoint Store\\n(Conversation & Tool History)", fillcolor="#1E293B"];
    }

    subgraph cluster_actuation {
        label="Actuation & Tool Execution Tier";
        fontcolor="#10B981";
        style=dashed;
        color="#334155";
        mcp_bridge [label="Model Context Protocol (MCP)\\nClient / Server Dispatch", fillcolor="#064E3B"];
        tool_executor [label="Tool Executors / Sandbox\\n(Web, SQL, Python, APIs)", fillcolor="#064E3B"];
        eval_gate [label="Observation & Reflection Gate\\n(Validation / Guardrails)", fillcolor="#1E293B"];
    }

    user_query -> prompt_builder [label="Submit"];
    prompt_builder -> llm_engine [label="Context"];
    llm_engine -> state_store [label="Record State", dir=both];
    llm_engine -> tool_router [label="Emit Tool Call"];
    tool_router -> mcp_bridge [label="Dispatch"];
    mcp_bridge -> tool_executor [label="Execute"];
    tool_executor -> eval_gate [label="Raw Output"];
    eval_gate -> prompt_builder [label="Feedback Loop (Observe)"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)
```

---

## 4. Master Specification Reference Implementation

Below is the complete reference implementation demonstrating all required sections:

```python
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="1.1 What is an AI Agent - AI Agent Curriculum",
    page_icon="🤖",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("1.1 What is an AI Agent?")
st.caption("Chapter 1: Introduction to AI Agents | Domain: 8. AI Agent")
st.markdown("""
An AI Agent is an autonomous software entity that perceives its environment, makes informed decisions
using a foundation model reasoning core, plans multi-step execution paths, executes external tools,
and observes outcomes to achieve defined objectives.

In this lesson, you will master:
- **Core Cognitive Loop**: The Perception-Reasoning-Action-Observation lifecycle.
- **Architectural Contrasts**: How agents differ from chatbots, deterministic workflows, and passive RAG.
- **Tool Actuation Mechanics**: Function calling contracts and execution dispatch.
- **State & Feedback Control**: Managing dynamic state transitions and stopping conditions.
- **Production Systems Readiness**: Mitigating runaway loops and securing agent execution.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Autonomous Cognitive Loop", divider="rainbow")

dot_blueprint = """
digraph CognitiveLoop {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_perception {
        label="Stage 1: Perception & Ingestion";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        env_state [label="Environment / User Input\\n(Task Query, Current State)"];
        perceive [label="Perception Module\\n(Context Extraction, Schema Formatting)"];
    }

    subgraph cluster_reasoning {
        label="Stage 2: Reasoning & Planning";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        think [label="Reasoning Core (LLM)\\n(Synthesize Plan, Evaluate Goal)"];
        decide [label="Action Decision\\n(Tool Call vs Final Response)"];
    }

    subgraph cluster_actuation {
        label="Stage 3: Actuation & Observation";
        fontcolor="#10B981";
        style=dashed;
        color="#334155";
        act [label="Tool Actuation\\n(Execute API, Query DB, Browse)"];
        observe [label="Observation Collector\\n(Parse Result, Handle Errors)"];
    }

    env_state -> perceive [label="Ingest"];
    perceive -> think [label="Context"];
    think -> decide [label="Formulate"];
    decide -> act [label="Requires Tool"];
    act -> observe [label="Execute"];
    observe -> think [label="Feed Observation (Loop)"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

# --- SECTIONS 2 THROUGH K+1: ROADMAP TOPICS ---
# (Example for Topic 1: Definition of AI Agent)
st.header("2. Definition of an AI Agent", divider="rainbow")
st.markdown("""
Unlike traditional software that executes rigid deterministic logic, an **AI Agent** utilizes a foundation
model as its central reasoning engine to navigate non-deterministic problems. The agent maintains an internal
representation of its goal, decomposes goals into subtasks, selects appropriate tools, evaluates execution feedback,
and dynamically adjusts its plan until goal completion or a termination threshold is reached.
""")

with st.expander("🔍 View Flow Diagram: Agent vs Traditional Program", expanded=False):
    dot_sub = """
    digraph ComparisonFlow {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#334155"];
        edge [color="#38BDF8"];
        input [label="Input"];
        static_code [label="Hardcoded If/Else Logic"];
        llm_reasoning [label="Autonomous Reasoning Loop"];
        output [label="Result"];
        input -> static_code -> output;
        input -> llm_reasoning -> output;
    }
    """
    st.graphviz_chart(dot_sub, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current Industry Stacks Implement This
In modern enterprise stacks, agents are not built as monolithic loops with raw prompts. Instead, industry teams deploy
cyclic state machine graphs (such as LangGraph) paired with standardized protocol servers (Model Context Protocol - MCP).
This guarantees complete state persistence across server restarts, allows seamless Human-in-the-Loop interruptions, and
enforces rigorous tool sandboxing.
""")

df_compare = pd.DataFrame({
    "Dimension": ["Decision Engine", "Tool Integration", "Error Recovery", "State Handling"],
    "Legacy Chatbot / Script": ["Hardcoded intent rules / regex", "Brittle manual bindings", "Immediate fatal crash", "Stateless / in-memory only"],
    "Modern Agentic System": ["Foundation LLM reasoning loop", "MCP standard & JSON Schema", "Autonomous reflection & retry", "Persistent state checkpoints"]
})
st.table(df_compare)

# --- SECTION K+2: PRODUCTION IMPLEMENTATION ---
st.header("3. Production Implementation: Resilient Agent Loop", divider="rainbow")
st.code("""
from typing import Dict, Any, List

class ResilientAgentLoop:
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations

    def run(self, user_goal: str) -> Dict[str, Any]:
        history: List[Dict[str, str]] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            # 1. Reason: Decide action
            decision = self.decide_action(user_goal, history)
            if decision["action"] == "FINAL_ANSWER":
                return {"status": "SUCCESS", "result": decision["content"], "steps": iteration}

            # 2. Act: Execute tool
            observation = self.execute_tool(decision["action"], decision.get("args", {}))
            history.append({"action": decision["action"], "observation": observation})

        return {"status": "MAX_ITERATIONS_EXCEEDED", "result": "Goal could not be completed within budget"}

    def decide_action(self, goal: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        if len(history) >= 2:
            return {"action": "FINAL_ANSWER", "content": f"Task accomplished for: {goal}"}
        return {"action": "lookup_data", "args": {"query": goal}}

    def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        return f"Mock output from {name} with args {args}"
""", language="python")

# --- SECTION K+3: INTERACTIVE WORKBENCH ---
st.header("4. Interactive Workbench: Cognitive Loop Simulator", divider="rainbow")

class WorkbenchAgentSimulator:
    def simulate(self, goal: str, max_iter: int, fail_tool: bool) -> List[Dict[str, Any]]:
        trace = []
        for step in range(1, max_iter + 1):
            if step == 1:
                trace.append({"step": step, "thought": "Analyzing goal and formulating plan.", "action": "search_db", "status": "DISPATCHED"})
            elif step == 2 and fail_tool:
                trace.append({"step": step, "thought": "Tool returned error. Reflecting and attempting alternate tool.", "action": "retry_fallback", "status": "RECOVERED"})
            else:
                trace.append({"step": step, "thought": "Synthesizing final answer from observations.", "action": "EMIT_FINAL", "status": "COMPLETED"})
                break
        return trace

sim = WorkbenchAgentSimulator()
st.subheader("Configure Simulation Parameters")
col1, col2 = st.columns(2)
with col1:
    goal_input = st.text_input("Agent Goal", value="Analyze quarterly churn rate")
    max_steps = st.slider("Max Iteration Budget", 1, 10, 5)
with col2:
    simulate_failure = st.checkbox("Simulate Tool Failure & Recovery", value=True)

if st.button("🚀 Execute Agent Loop", type="primary"):
    logs = sim.simulate(goal_input, max_steps, simulate_failure)
    st.write("### Execution Trace")
    st.dataframe(pd.DataFrame(logs), use_container_width=True)

# --- SECTION K+4: TECHNICAL INTERVIEW PREPARATION ---
st.header("5. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qa = [
    ("What distinguishes an AI Agent from a standard LLM application?",
     "An LLM application operates primarily in a single-turn request-response pattern. An AI Agent features autonomous cognitive looping: it maintains state, chooses external tools dynamically via JSON schema function calls, observes outputs, self-corrects upon error, and iterates until goal completion."),
    ("How do you prevent an AI Agent from getting stuck in infinite execution loops?",
     "Production agents implement hard guardrails: explicit maximum iteration budgets, token expenditure caps, timeout thresholds, duplicate action detection (detecting repetitive identical tool calls), and Human-in-the-Loop escalation."),
    ("What is the Model Context Protocol (MCP) and why is it replacing custom tool wrappers?",
     "MCP is an open standard that decouples tool providers from LLM applications. Instead of building bespoke API wrappers for every service, an MCP client can connect to any standardized MCP server to dynamically discover resources, tools, and prompts with unified permissions."),
    ("Why are cyclic state machines (LangGraph) preferred over DAG chains (LangChain LCEL) for agents?",
     "Autonomous agents require cycles: after executing an action and observing an error, the agent must loop back to the reasoning node. DAGs (Directed Acyclic Graphs) cannot loop natively. LangGraph provides cyclic state machines with native checkpointing and interrupt capabilities."),
    ("How do you implement Human-in-the-Loop (HITL) safely in agent architectures?",
     "High-impact tools (financial transactions, data deletion, code deployment) are tagged with approval gates. The execution graph yields an interrupt breakpoint before tool execution, persisting state to disk. A human operator approves or rejects the action via API, and the graph resumes from that checkpoint.")
]

for i, (q, a) in enumerate(interview_qa, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Breaking Error)**: Unbounded Tool Loops. Without cycle detection and strict iteration limits, an agent will loop endlessly on ambiguous queries, draining API budgets.")
    st.error("**Trap 2 (Breaking Error)**: Hallucinated Tool Arguments. Passing unvalidated LLM JSON outputs directly to database APIs causes execution crashes. Always validate schemas via Pydantic.")
with col_trap2:
    st.warning("**Trap 3 (Performance Warning)**: High Latency from Sequential Tool Calls. When multiple independent tools are needed, sequential invocation multiplies latency. Implement parallel tool calling.")
    st.warning("**Trap 4 (Security Warning)**: Indirect Prompt Injection. Data returned by web tools or external databases can contain malicious prompt overrides that hijack the agent loop. Sanitize tool observations.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("🎴 Flashcard 1: Cognitive Loop Stages"):
        st.markdown("**Perceive -> Reason -> Plan -> Act -> Observe -> Reflect**")
    with st.expander("🎴 Flashcard 2: Stopping Conditions"):
        st.markdown("**Final Answer emitted, Max Iterations reached, Token Budget exhausted, or HITL Rejection.**")
with fc_col2:
    with st.expander("🎴 Flashcard 3: MCP Primitives"):
        st.markdown("**Resources (data/files), Tools (callable functions), Prompts (templated workflows).**")
    with st.expander("🎴 Flashcard 4: LangGraph Checkpoints"):
        st.markdown("**State snapshots stored in Postgres/Memory enabling time-travel debugging and fault recovery.**")

st.divider()

# --- SECTION K+5: SELF-ASSESSMENT CHECKLIST ---
st.header("6. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can explain the difference between an AI Agent, a Chatbot, and a Workflow.", value=True)
chk2 = st.checkbox("I understand the Perception-Reasoning-Action-Observation cognitive loop.", value=True)
chk3 = st.checkbox("I know how to prevent infinite agent execution loops using guardrails.", value=True)
chk4 = st.checkbox("I understand the Model Context Protocol (MCP) and its client-server architecture.", value=True)
chk5 = st.checkbox("I can implement an agent loop with tool error recovery in Python.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on this lesson. Ready for the next module!")
```
