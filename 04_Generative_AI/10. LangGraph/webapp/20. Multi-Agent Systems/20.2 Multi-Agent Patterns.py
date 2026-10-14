import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict, Literal

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="20.2 Multi-Agent Patterns - LangGraph Curriculum",
    page_icon="🕸️",
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
    .badge-pattern {
        background-color: #1E293B;
        color: #FBBF24;
        border: 1px solid #D97706;
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
class MultiAgentPatternsSimulator:
    """
    Simulates the core multi-agent execution patterns:
    - Supervisor, Router, Hierarchical, Peer-to-Peer, Pipeline, Debate
    - Tracks execution path, step count, and agent interaction logs
    """
    def __init__(self):
        pass

    def simulate_pattern(self, pattern: str, task: str) -> Dict[str, Any]:
        trace = []
        if pattern == "Supervisor":
            trace.append({"step": 1, "agent": "Supervisor", "action": f"Analyzed '{task}' and delegated sub-goal 1 to Researcher."})
            trace.append({"step": 2, "agent": "Researcher", "action": "Gathered external data points and returned results."})
            trace.append({"step": 3, "agent": "Supervisor", "action": "Evaluated findings; delegated sub-goal 2 to Analyst."})
            trace.append({"step": 4, "agent": "Analyst", "action": "Computed statistical metrics and returned table."})
            trace.append({"step": 5, "agent": "Supervisor", "action": "All sub-goals satisfied. Formatted final response."})

        elif pattern == "Router":
            trace.append({"step": 1, "agent": "Classifier Router", "action": f"Evaluated intent of '{task}' -> Selected 'CodeAssistant'."})
            trace.append({"step": 2, "agent": "CodeAssistant", "action": "Executed specialized script and generated code. Terminated."})

        elif pattern == "Pipeline":
            trace.append({"step": 1, "agent": "Ingest Agent", "action": f"Parsed raw documents for '{task}'."})
            trace.append({"step": 2, "agent": "Extraction Agent", "action": "Extracted structured entities from documents."})
            trace.append({"step": 3, "agent": "Synthesis Agent", "action": "Generated executive summary from extracted entities."})

        elif pattern == "Debate":
            trace.append({"step": 1, "agent": "Debater A (Bull)", "action": f"Argued bullish growth position on '{task}'."})
            trace.append({"step": 2, "agent": "Debater B (Bear)", "action": "Argued risk factors and inflation constraints."})
            trace.append({"step": 3, "agent": "Debater A (Bull)", "action": "Rebutted risk arguments with hedge data."})
            trace.append({"step": 4, "agent": "Judge / Arbiter", "action": "Synthesized balanced compromise verdict."})

        elif pattern == "Peer-to-Peer":
            trace.append({"step": 1, "agent": "Triage Specialist", "action": f"Started '{task}'. Decided SQL query needed."})
            trace.append({"step": 2, "agent": "SQL Specialist", "action": "Executed query; handed off directly to Visualizer."})
            trace.append({"step": 3, "agent": "Visualizer Specialist", "action": "Generated charts and resolved request."})

        else: # Hierarchical
            trace.append({"step": 1, "agent": "Lead Executive", "action": f"Delegated '{task}' to Engineering Department."})
            trace.append({"step": 2, "agent": "Eng Manager (Sub-Supervisor)", "action": "Divided task into Frontend and Backend tasks."})
            trace.append({"step": 3, "agent": "Frontend Worker", "action": "Built UI components."})
            trace.append({"step": 4, "agent": "Backend Worker", "action": "Constructed API endpoints."})
            trace.append({"step": 5, "agent": "Eng Manager", "action": "Integrated deliverables and returned to Lead Executive."})

        return {
            "pattern": pattern,
            "steps_count": len(trace),
            "trace": trace
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Taxonomy of Multi-Agent Collaboration Patterns in LangGraph", divider="rainbow")

st.markdown("""
Multi-agent systems are not one-size-fits-all.
Depending on the business objective, latency constraints, and reasoning requirements, engineers select from **seven fundamental multi-agent patterns**:
1. **Supervisor**: A central leader evaluates state, assigns tasks, and decides completion.
2. **Router**: Lightweight classification that routes input once to a single specialist.
3. **Worker**: Focused execution nodes performing dedicated tasks.
4. **Hierarchical**: Multi-tiered supervisor-of-supervisors for complex enterprise organizations.
5. **Peer-to-Peer**: Direct handoffs between equals without returning to a central manager.
6. **Pipeline**: Sequential assembly-line processing where outputs flow strictly forward.
7. **Debate**: Adversarial agents challenging each other to achieve balanced consensus.

The blueprint below illustrates these collaboration patterns and their flow topologies.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MultiAgentPatternsTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_supervisor {
            label="1. Supervisor Pattern (Star Topology)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            sup_node [label="Supervisor Orchestrator\\nDecides Next Step", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            sup_w1 [label="Worker A (Research)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            sup_w2 [label="Worker B (Coding)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            sup_node -> sup_w1 -> sup_node;
            sup_node -> sup_w2 -> sup_node;
        }

        subgraph cluster_pipeline {
            label="2. Pipeline Pattern (Sequential DAG)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            pip_1 [label="Extractor Node", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            pip_2 [label="Transformer Node", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            pip_3 [label="Validator Node", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            pip_1 -> pip_2 -> pip_3;
        }

        subgraph cluster_p2p {
            label="3. Peer-to-Peer / Handoffs (Decentralized Mesh)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            p2p_a [label="Triage Agent", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            p2p_b [label="Billing Specialist", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            p2p_c [label="Tech Support", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            p2p_a -> p2p_b [label="transfer_to_billing"];
            p2p_b -> p2p_c [label="transfer_to_tech"];
        }

        subgraph cluster_debate {
            label="4. Debate / Adversarial Pattern";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            deb_pro [label="Proponent (Thesis)", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            deb_con [label="Opponent (Antithesis)", fillcolor="#450A0A", fontcolor="#FECACA", color="#EF4444"];
            deb_judge [label="Judge (Synthesis)", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            deb_pro -> deb_con -> deb_pro;
            deb_pro -> deb_judge [label="Conclude"];
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: SUPERVISOR AGENT
# ==============================================================================
st.header("2. Supervisor Agent: Centralized Command, Dynamic Routing, and Convergence Control", divider="rainbow")

st.markdown("""
The **Supervisor Pattern** is the most widely deployed multi-agent pattern in enterprise AI.
A central LLM supervisor inspects the ongoing message conversation, evaluates progress against the goal, and selects which specialist to call next using structured output (e.g. `Pydantic` or OpenAI function calling).
""")

st.code('''from pydantic import BaseModel
from typing import Literal

class RouterDecision(BaseModel):
    next_agent: Literal["researcher", "coder", "FINISH"]
    reasoning: str

# Supervisor model binds schema
supervisor_llm = ChatOpenAI(model="gpt-4o").with_structured_output(RouterDecision)

def supervisor_node(state: State) -> dict:
    decision = supervisor_llm.invoke(state["messages"])
    return {"next_step": decision.next_agent}''', language="python")

# ==============================================================================
# SECTION 3: ROUTER AGENT
# ==============================================================================
st.header("3. Router Agent: Lightweight Initial Triage and Single-Turn Fast-Path Routing", divider="rainbow")

st.markdown("""
Unlike a supervisor that continually orchestrates multi-turn interactions, a **Router Agent** executes exactly once at the ingress boundary.
It inspects the user query, categorizes intent (e.g., General FAQ, Billing Dispute, Code Generation), and fast-forwards the thread to a dedicated subgraph or specialist, ending its involvement.
""")

# ==============================================================================
# SECTION 4: WORKER AGENTS
# ==============================================================================
st.header("4. Worker Agents: Sandboxed Execution, Tool Execution Loops, and Task Contracts", divider="rainbow")

st.markdown("""
**Worker Agents** are autonomous nodes with focused personas and bounded toolsets.
They receive a concrete sub-task from the supervisor or router, execute their internal reasoning loop (e.g., React loop with tools), and return a crisp deliverable back to the shared state bus.
""")

# ==============================================================================
# SECTION 5: HIERARCHICAL AGENTS
# ==============================================================================
st.header("5. Hierarchical Agents: Multi-Tier Nested Teams and Departmental Delegation", divider="rainbow")

st.markdown("""
When a system encompasses dozens of agents, a single supervisor becomes a bottleneck.
**Hierarchical Multi-Agent Systems** structure agents like an enterprise organization:
- **Executive Supervisor**: Delegates to Department Managers.
- **Department Manager**: Subgraph supervisor orchestrating 3-4 specialized team workers.
- **Team Workers**: Execute specific low-level tasks.
""")

# ==============================================================================
# SECTION 6: PEER-TO-PEER AGENTS
# ==============================================================================
st.header("6. Peer-to-Peer Agents: Direct Handoffs, Autonomous Collaboration, and Decentralized Networks", divider="rainbow")

st.markdown("""
In **Peer-to-Peer (Handoffs)** architectures, there is no centralized manager.
Agents communicate by calling explicit transfer tools: `transfer_to_support()`, `transfer_to_sales()`.
When an agent calls a transfer tool, LangGraph's conditional edge routes execution directly to the designated peer node.
""")

st.code('''def transfer_to_billing(reason: str) -> str:
    """Transfer the user conversation to the Billing Specialist."""
    return f"Transferring to billing: {reason}"

# Agent binds handoff tools
support_agent = ChatOpenAI(model="gpt-4o").bind_tools([transfer_to_billing, transfer_to_tech])''', language="python")

# ==============================================================================
# SECTION 7: PIPELINE AGENTS
# ==============================================================================
st.header("7. Pipeline Agents: Linear Assembly-Line Topologies and Incremental Transformation", divider="rainbow")

st.markdown("""
In **Pipeline Multi-Agent Systems**, data flows strictly in one direction across a deterministic assembly line:
`Ingest -> Extract -> Enrich -> Format -> Deliver`.
Each node in the pipeline is an agent specialized in a single stage of transformation. Pipelines are ideal for content generation, document indexing, and ETL operations.
""")

# ==============================================================================
# SECTION 8: DEBATE AGENTS
# ==============================================================================
st.header("8. Debate Agents: Adversarial Reflection, Consensus Voting, and Multi-Model Dialectics", divider="rainbow")

st.markdown("""
For complex judgment tasks (legal analysis, policy review, investment decisions), a single agent's reasoning can be biased.
The **Debate Pattern** pits two or more agents with conflicting mandates against each other:
- **Agent A (Advocate)**: Finds all positive arguments and growth opportunities.
- **Agent B (Skeptic)**: Stress-tests assumptions and highlights tail risks.
- **Arbiter Agent**: Evaluates the arguments from both debaters and delivers a calibrated, nuanced verdict.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-Pattern Multi-Agent Architecture", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern demonstrating a Supervisor-Worker graph with structured routing and state validation.
""")

st.code('''from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class PatternState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: Literal["research", "writer", "FINISH"]
    final_output: str

def supervisor_router_node(state: PatternState) -> dict:
    msgs = state["messages"]
    has_research = any(getattr(m, "name", "") == "researcher" for m in msgs)
    has_writer = any(getattr(m, "name", "") == "writer" for m in msgs)
    
    if not has_research:
        return {"next_step": "research"}
    elif not has_writer:
        return {"next_step": "writer"}
    else:
        return {"next_step": "FINISH", "final_output": "All deliverables approved by supervisor."}

def researcher_worker_node(state: PatternState) -> dict:
    return {"messages": [AIMessage(content="Verified facts gathered on target subject.", name="researcher")]}

def writer_worker_node(state: PatternState) -> dict:
    return {"messages": [AIMessage(content="Executive brief synthesized based on research findings.", name="writer")]}

def route_decision(state: PatternState) -> str:
    if state["next_step"] == "FINISH":
        return END
    return state["next_step"]

builder = StateGraph(PatternState)
builder.add_node("supervisor", supervisor_router_node)
builder.add_node("research", researcher_worker_node)
builder.add_node("writer", writer_worker_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_decision,
    {"research": "research", "writer": "writer", END: END}
)
builder.add_edge("research", "supervisor")
builder.add_edge("writer", "supervisor")

app = builder.compile(checkpointer=MemorySaver())
print("Multi-Pattern Multi-Agent Architecture Compiled.")''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Multi-Agent Pattern Selector & Topology Simulator", divider="rainbow")

st.markdown("""
Select any of the core multi-agent patterns and inspect the simulated execution traces, interaction steps, and convergence criteria.
""")

sim = MultiAgentPatternsSimulator()

col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    st.subheader("⚙️ Pattern Selection")
    pattern_choice = st.selectbox(
        "Choose Multi-Agent Pattern",
        ["Supervisor", "Router", "Pipeline", "Debate", "Peer-to-Peer", "Hierarchical"]
    )
    sim_task = st.text_input("Simulated Task", value="Design Cloud Migration Strategy")
    run_sim = st.button("🚀 Simulate Agent Pattern")

with col_sim2:
    st.subheader(f"📋 Execution Trace (`{pattern_choice}`)")
    if run_sim:
        res = sim.simulate_pattern(pattern_choice, sim_task)
        st.info(f"Pattern: **{res['pattern']}** | Total Steps: **{res['steps_count']}**")
        df_trace = pd.DataFrame(res["trace"])
        st.dataframe(df_trace, use_container_width=True)

st.subheader("📊 Comparative Pattern Selection Matrix")
df_pats = pd.DataFrame({
    "Pattern": ["Supervisor", "Router", "Pipeline", "Debate", "Peer-to-Peer", "Hierarchical"],
    "Coordination Topology": ["Centralized Star", "Single-Turn Fork", "Linear Sequential DAG", "Cyclic Dialectic Loop", "Decentralized Mesh", "Multi-Tier Tree"],
    "Best For": ["General Multi-Skill Teams", "Fast Ingress Intent Routing", "Document Processing / ETL", "Risk & Policy Decisions", "Customer Support Handoffs", "Large Enterprise Systems"],
    "Token Overhead": ["Moderate (Calls supervisor often)", "Low (Single check)", "Minimal (Pure forwarding)", "High (Multi-turn arguments)", "Low (Direct handoff)", "Moderate to High"]
})
st.dataframe(df_pats, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: How does the Supervisor pattern differ from the Router pattern?", expanded=False):
    st.markdown("""
    **Answer**:
    A Router acts once at the entrance of a workflow to categorize user intent and dispatch to a single target agent.
    A Supervisor remains active throughout the multi-turn lifecycle, coordinating multiple specialists, evaluating intermediate outputs, and determining when the objective is met.
    """)

with st.expander("Q2: When should an engineering team use the Debate pattern?", expanded=False):
    st.markdown("""
    **Answer**:
    When a task involves subjective trade-offs, high-stakes risk evaluation, or potential LLM bias (e.g. investment theses, regulatory compliance, architecture choices).
    Pitting opposing perspectives against each other before an arbiter yields higher fidelity decisions.
    """)

with st.expander("Q3: What mechanism enables Peer-to-Peer handoffs in LangGraph without a supervisor?", expanded=False):
    st.markdown("""
    **Answer**:
    Specialist agents are equipped with transfer tools (e.g., `transfer_to_support()`).
    When the model calls a transfer tool, conditional edges examine the tool call and route execution directly to the requested peer agent.
    """)

with st.expander("Q4: Why are Hierarchical patterns preferred over flat Supervisor patterns for large agent systems?", expanded=False):
    st.markdown("""
    **Answer**:
    A flat supervisor orchestrating 15+ agents suffers from prompt overload and token saturation.
    Hierarchical patterns group agents into departmental subgraphs, where departmental supervisors manage 3–4 workers and report to an executive supervisor.
    """)

with st.expander("Q5: How do Pipeline patterns enforce quality between sequential nodes?", expanded=False):
    st.markdown("""
    **Answer**:
    By inserting validation gate nodes between pipeline stages. If an intermediate output fails schema or quality assertions, conditional edges can loop back for revision rather than propagating bad data downstream.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Unbounded Debate Cycles Burning Infinite Tokens")
    st.markdown("""
    *Mechanism*: Configuring two debate agents without a strict maximum round counter or stopping threshold.  
    *Consequence*: Debaters continue disagreeing endlessly until the API budget or recursion limit is exhausted.  
    *Fix*: Always pass a `debate_rounds < 3` termination guard to the conditional routing edge.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Loss of Context During Peer-to-Peer Handoffs")
    st.markdown("""
    *Mechanism*: Transferring from Agent A to Agent B while wiping state channels or passing an empty message history.  
    *Consequence*: Agent B has no memory of what the user asked or what Agent A already completed.  
    *Fix*: Preserve the shared `messages` channel or pass a structured handoff payload.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Using Supervisor Pattern for Strictly Deterministic Pipelines")
    st.markdown("""
    *Mechanism*: Using an LLM supervisor to decide the next node in an ETL pipeline where the order is always Stage 1 -> Stage 2 -> Stage 3.  
    *Consequence*: Adds unnecessary LLM inference latency, cost, and hallucination risk to a deterministic task.  
    *Fix*: Use standard linear graph edges (`builder.add_edge("stage1", "stage2")`).
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Latency Warning): Deep Hierarchical Graphs Causing Extreme Time-to-First-Token")
    st.markdown("""
    *Mechanism*: 4 layers of nested supervisors before any worker executes a user request.  
    *Consequence*: 15+ seconds of latency before the user receives any feedback.  
    *Fix*: Keep hierarchies shallow (<= 2 levels) and enable streaming with `subgraphs=True`.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Supervisor Pattern**\\nA central manager evaluating state and delegating to specialist workers.")
with col_fc2:
    st.info("**Router Pattern**\\nA single-turn intent classifier dispatching directly to a specialized agent.")
with col_fc3:
    st.info("**Debate Pattern**\\nAdversarial agents advocating opposing viewpoints to reach balanced consensus.")
with col_fc4:
    st.info("**Peer Handoff**\\nDirect agent-to-agent delegation via transfer tool calls.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand all 7 core multi-agent patterns in LangGraph.", value=True)
c2 = st.checkbox("I know when to choose a Supervisor over a Router or Pipeline pattern.", value=True)
c3 = st.checkbox("I can implement peer-to-peer handoffs using transfer tools and conditional edges.", value=True)
c4 = st.checkbox("I understand how the Debate pattern mitigates single-model reasoning bias.", value=True)
c5 = st.checkbox("I know how to structure hierarchical multi-agent trees for enterprise scaling.", value=True)
c6 = st.checkbox("I understand how to prevent infinite loops in debate and peer-to-peer topologies.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 20.2 'Multi-Agent Patterns' with full architectural and practical comprehension!")
