import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="18.3 Agent Workflows | GenAI Roadmap",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .wf-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("🕸️ 18.3 Agent Workflows — ReAct, Plan-and-Execute & Multi-Agent Swarms")
st.markdown(
    """
Welcome to **Lesson 18.3: Agent Workflows**.  
A single agent with multiple tools can accomplish impressive tasks, but complex enterprise projects quickly overwhelm a monolithic agent context. Context windows fill with noisy tool outputs, and the model struggles to balance planning, execution, and quality assurance simultaneously.

Modern agent engineering divides labor across specialized workflow topologies: **ReAct loops**, **Plan-and-Execute DAGs**, and **Hierarchical Multi-Agent Systems** where specialized worker agents collaborate under an orchestrator supervisor.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Core Agent Workflow Archetypes
# ==============================================================================
st.header("1. Core Autonomous Workflow Archetypes", divider="rainbow")
st.markdown("Comparing the primary architectural patterns for agentic orchestration:")

archetypes_df = pd.DataFrame({
    "Workflow Archetype": [
        "1. ReAct (Reason + Act)",
        "2. Plan-and-Execute",
        "3. Hierarchical Supervisor-Worker",
        "4. Collaborative Peer Multi-Agent",
        "5. Debate & Self-Critique"
    ],
    "Architectural Mechanism": [
        "Interleaves a single Thought, Action, and Observation dynamically on every turn",
        "Stage 1 generates a full multi-step plan; Stage 2 executes each step using a dedicated execution agent",
        "A central router/supervisor decomposes goals, delegates to specialized agents, and merges findings",
        "Specialized agents pass messages to one another directly across a shared state graph (e.g. LangGraph)",
        "Two agents generate competing perspectives or candidate solutions, judged by an arbitrator model"
    ],
    "Best Production Use Case": [
        "Exploratory queries, dynamic troubleshooting, tasks where future steps depend on unexpected observations",
        "Deterministic multi-step engineering pipelines, long-horizon data migrations, report generators",
        "Enterprise workflows (e.g. Research Agent + Coder Agent + QA Reviewer Agent)",
        "Complex simulations, game theory modeling, collaborative document drafting",
        "Fact-checking, high-stakes legal document auditing, algorithm optimization"
    ],
    "Token & Latency Profile": [
        "Low initial latency, but token cost accumulates as loop steps grow",
        "Upfront planning latency, but highly predictable execution cost",
        "Higher token cost, but maximum isolation and highest reliability across complex domains",
        "Very high token consumption; requires strict state management to prevent runaway message chatter",
        "2x to 3x token overhead; highest accuracy on sensitive compliance decisions"
    ]
})
st.dataframe(archetypes_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: ReAct vs. Plan-and-Execute Deep Dive
# ==============================================================================
st.header("2. ReAct vs. Plan-and-Execute: Strategic Trade-Offs", divider="rainbow")

col_ap1, col_ap2 = st.columns(2)

with col_ap1:
    with st.container(border=True):
        st.subheader("🔄 ReAct (Dynamic & Adaptive)")
        st.markdown(
            """
- **How It Works:** Makes decisions one step at a time. After seeing the result of Step 1, it decides what Step 2 should be.
- **Strengths:** Excellent for volatile, unpredictable environments (e.g. diagnosing a server crash where you don't know what you'll find until you check the logs).
- **Weaknesses:** Prone to wandering, losing track of the original high-level goal, and taking suboptimal paths.
"""
        )

with col_ap2:
    with st.container(border=True):
        st.subheader("📋 Plan-and-Execute (Deterministic & Efficient)")
        st.markdown(
            """
- **How It Works:** The planner model writes out steps $1, 2, 3, 4, 5$ in advance. A lightweight execution model runs each task sequentially.
- **Strengths:** Separation of concerns: cheap models can execute individual steps while a frontier model plans. Keeps long-term goal top-of-mind.
- **Weaknesses:** Rigid: if Step 2 reveals that the planned Step 3 is impossible, the entire plan must be thrown out and re-generated.
"""
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Hierarchical Multi-Agent Supervisor Pattern", divider="rainbow")
st.markdown("How an enterprise supervisor orchestrates specialized workers across isolated context windows:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        user [label="User Complex Goal\n'Audit Q3 Churn & Email Memo'", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        supervisor [label="Supervisor Agent\n(Router & Orchestrator)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        w1 [label="Worker 1: SQL Analyst\n(Queries Database)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        w2 [label="Worker 2: Zendesk Auditor\n(Analyzes Support Tickets)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        w3 [label="Worker 3: Executive Scribe\n(Drafts Final Memo)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        final_doc [label="Final Approved Deliverable", fillcolor="#10B981", fontcolor="#FFFFFF"]

        user -> supervisor
        supervisor -> w1 [label=" Sub-task A"]
        supervisor -> w2 [label=" Sub-task B"]
        w1 -> supervisor [label=" Data Results"]
        w2 -> supervisor [label=" Ticket Findings"]
        supervisor -> w3 [label=" Synthesize Findings"]
        w3 -> final_doc
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Multi-Agent Swarm Simulator
# ==============================================================================
st.header("4. Interactive Multi-Agent Swarm Simulator", divider="rainbow")
st.markdown("Watch a 3-agent hierarchical team investigate a production customer churn spike:")

swarm_stage = st.select_slider(
    "Advance Swarm Execution State",
    options=["1. Task Ingestion", "2. Parallel Delegation", "3. Specialist Execution", "4. Synthesis & Executive Memo"],
    value="1. Task Ingestion"
)

col_sw1, col_sw2 = st.columns([1, 1])

with col_sw1:
    st.subheader("🤖 Active Agent Internal Memory")
    if swarm_stage == "1. Task Ingestion":
        st.markdown("**Supervisor Active Context:**")
        st.info("Goal received: 'Investigate Q3 enterprise churn and draft mitigation memo'. Supervisor decomposing into: (1) Quantitative DB metrics, (2) Qualitative Zendesk ticket sentiment, (3) Final executive drafting.")
    elif swarm_stage == "2. Parallel Delegation":
        st.markdown("**Supervisor Dispatching Tasks:**")
        st.write("- Delegating to **Data Analyst Agent**: `query_churn_metrics(quarter='Q3')`")
        st.write("- Delegating to **Customer Support Agent**: `search_negative_tickets(quarter='Q3', min_mrr=10000)`")
    elif swarm_stage == "3. Specialist Execution":
        st.markdown("**Specialist Worker Isolated Output:**")
        st.code("""[Data Analyst]: Found that 88% of churned accounts were in the Enterprise tier, losing $180k ARR.
[Support Agent]: Found 14 unaddressed P0 tickets citing SAML SSO timeout errors after the Oct 1 deployment.""", language="text")
    else:
        st.success("✅ Executive Scribe Finished Final Synthesis!")
        st.markdown("**Executive Scribe Context:** Combined quantitative financial loss ($180k ARR) with technical root cause (SAML SSO outage) into an approved executive briefing.")

with col_sw2:
    st.subheader("📄 Swarm Collaborative Output")
    if swarm_stage == "4. Synthesis & Executive Memo":
        st.markdown(
            """
### Multi-Agent Unified Incident Briefing
**To:** VP of Customer Success & VP of Engineering  
**From:** Autonomous Incident Analysis Swarm  

#### Findings:
1. **Financial Impact:** Q3 enterprise churn totaled **$180,000 ARR** across 4 major accounts *(analyzed by Data Analyst Agent)*.
2. **Technical Root Cause:** All 4 churned accounts experienced an unresolved SAML SSO outage spanning 48 hours following the October 1st security patch *(analyzed by Support Agent)*.

#### Recommended Interventions:
- Deploy SAML patch v4.0.2 to all staging clusters immediately.
- Issue $45,000 SLA service credits to impacted enterprise sponsors.
"""
        )
    else:
        st.caption("Swarm execution in progress...")
        st.progress((["1. Task Ingestion", "2. Parallel Delegation", "3. Specialist Execution", "4. Synthesis & Executive Memo"].index(swarm_stage)) * 33)

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Hierarchical Supervisor Agent", divider="rainbow")
st.markdown("Python implementation demonstrating a supervisor routing subtasks to specialized worker agents:")

st.code(
    '''from typing import Dict, Any, Callable
import json

class SpecializedWorkerAgent:
    def __init__(self, name: str, system_prompt: str, tool_fn: Callable):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_fn = tool_fn

    def execute(self, subtask_prompt: str) -> str:
        # Executes within its own isolated context window
        tool_result = self.tool_fn(subtask_prompt)
        return f"[{self.name} Output]: {tool_result}"

class SupervisorOrchestrator:
    def __init__(self, workers: Dict[str, SpecializedWorkerAgent]):
        self.workers = workers

    def route_and_execute(self, user_goal: str) -> Dict[str, str]:
        results = {}
        # Supervisor decomposes and calls workers sequentially or in parallel
        for worker_name, worker in self.workers.items():
            print(f"Supervisor delegating subtask to: {worker_name}")
            results[worker_name] = worker.execute(user_goal)
        return results

# Example setup
workers = {
    "data_analyst": SpecializedWorkerAgent("DataAnalyst", "You analyze SQL metrics.", lambda q: "MRR Drop: $50k"),
    "support_auditor": SpecializedWorkerAgent("SupportAuditor", "You audit tickets.", lambda q: "SSO Bug Identified")
}

supervisor = SupervisorOrchestrator(workers)
outputs = supervisor.route_and_execute("Investigate customer churn")
print("Multi-Agent Supervisor Collected Outputs:")
print(json.dumps(outputs, indent=2))
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: Why are Hierarchical Multi-Agent Systems superior to a single monolithic agent with 20 tools?", expanded=True):
        st.markdown(
            """
1. **Context Window Isolation**: A monolithic agent quickly fills its context with noisy, verbose tool outputs from multiple different tasks, leading to the 'Lost in the Middle' effect. Multi-agent systems give each worker an isolated context window for its specific subtask.
2. **Specialized Prompts & Personas**: A coding agent needs strict type-safety rules; a research agent needs citation rules; an editor agent needs narrative guidelines. You cannot effectively combine all these conflicting personas into one monolithic prompt.
3. **Model Tiering**: You can use cheap, fast models (e.g. GPT-4o-mini) for simple worker scraping/formatting tasks, and reserve expensive frontier models (e.g. GPT-4o, Claude 3.5 Sonnet) exclusively for the Supervisor orchestrator.
"""
        )
    with st.expander("Q2: What is the 'Agent Chatter' problem in collaborative multi-agent swarms?"):
        st.markdown(
            """
In peer-to-peer multi-agent systems where agents pass messages freely to one another, agents can enter an infinite conversational loop: Agent A asks Agent B for clarification, Agent B answers with an ambiguous question, and they exchange hundreds of messages without making progress towards the user goal.
**Production Defense**: Always enforce a **Hierarchical Supervisor** topology with fixed communication graphs (DAGs) and explicit step budgets rather than free-form conversational chatter.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we build an enterprise application using 10 collaborative agents chatting in a free-form loop?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, autonomous swarm emergence produces the smartest decisions."*
- **Winning Response**: *"Never in production. Free-form swarms are completely non-deterministic, impossible to debug, and suffer from runaway token costs ($50+ per query from agent chatter loops). Enterprise agentic workflows must be architected as **Deterministic State Machines** (e.g. using LangGraph) where agent transitions, state updates, and join barriers are strictly controlled by software code rather than unconstrained LLM conversation."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Hierarchical Multi-Agent**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** An architecture where a central supervisor agent decomposes goals and delegates subtasks to specialized worker agents with isolated contexts.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Context Window Isolation**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Ensuring each specialized worker agent maintains its own lean context window, preventing noisy tool outputs from polluting other tasks.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Agent Chatter**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The failure mode where peer agents in a swarm exchange infinite conversational messages without progressing toward the user's goal.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 18.3 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Choose the Right Topology**: Use ReAct for dynamic exploration, Plan-and-Execute for deterministic pipelines.
2. **Context Isolation**: Multi-agent architectures isolate context windows to prevent token bloat and confusion.
3. **Hierarchical Governance**: Always prefer supervised routing over unconstrained peer-to-peer swarms.
4. **State Machine Control**: Build agent workflows as structured state graphs (e.g. LangGraph) for auditability.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know when to select ReAct vs. Plan-and-Execute agent architectures.", value=True, key="chk_18_3_1")
    st.checkbox("I understand why context window isolation makes multi-agent systems superior to monolithic agents.", value=True, key="chk_18_3_2")
    st.checkbox("I can implement a hierarchical supervisor-worker routing pipeline in Python.", value=True, key="chk_18_3_3")
    st.checkbox("I know how to prevent the 'agent chatter' infinite loop failure mode.", value=True, key="chk_18_3_4")
