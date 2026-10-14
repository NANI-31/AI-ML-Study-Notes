import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="26.1 Multi-Agent Fundamentals - LangChain Architecture",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for agent communication cards and role indicators
st.markdown("""
<style>
    .agent-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-agent-coord {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-agent-worker {
        background-color: #065F46;
        color: #6EE7B7;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MultiAgentCollaborationEngine:
    """
    Simulation engine demonstrating single-agent cognitive overload vs
    specialized multi-agent cooperative problem solving.
    """
    def __init__(self):
        self.specialists = {
            "researcher": {
                "name": "Research Specialist",
                "focus": "Document retrieval, fact checking, citation validation",
                "tools": ["web_search", "academic_archive_retriever"],
                "token_overhead": 350
            },
            "coder": {
                "name": "Software Engineer",
                "focus": "Algorithm implementation, code execution, unit test generation",
                "tools": ["python_repl", "git_commit", "linter"],
                "token_overhead": 420
            },
            "reviewer": {
                "name": "Security & Quality Auditor",
                "focus": "Vulnerability scanning, style verification, policy adherence",
                "tools": ["static_ast_analyzer", "pii_scanner"],
                "token_overhead": 290
            }
        }

    def simulate_single_agent(self, task_complexity: int) -> Dict[str, Any]:
        """Simulates single monolithic agent attempting all roles."""
        prompt_bloat = 800 + (task_complexity * 750)
        tool_confusion_prob = min(0.85, 0.15 + (task_complexity * 0.18))
        success = tool_confusion_prob < 0.50
        
        return {
            "mode": "Single Monolithic Agent",
            "total_tokens": prompt_bloat + (task_complexity * 600),
            "tool_count": 7,
            "tool_confusion_probability": tool_confusion_prob,
            "success": success,
            "status": "COMPLETED" if success else "FAILED (Tool Hallucination / Context Saturation)"
        }

    def simulate_multi_agent(self, task_complexity: int) -> Dict[str, Any]:
        """Simulates specialized multi-agent division of labor."""
        total_tokens = sum(s["token_overhead"] for s in self.specialists.values()) + (task_complexity * 450)
        tool_confusion_prob = 0.05
        
        return {
            "mode": "Specialized Multi-Agent Collective",
            "total_tokens": total_tokens,
            "agents_deployed": len(self.specialists),
            "tool_confusion_probability": tool_confusion_prob,
            "success": True,
            "status": "COMPLETED (Modular handoffs with verified artifacts)"
        }


# --- LESSON HEADER ---
st.title("26.1 Multi-Agent Fundamentals & Cognitive Division")
st.caption("Chapter 26: Multi-Agent Systems | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
As GenAI tasks grow from simple question-answering into complex, end-to-end software development, market research,
and automated financial audits, single-agent architectures reach an inevitable breaking point. Monolithic agents
collapse under context window bloat, tool selection hallucination, and conflicting instruction sets.

In this lesson, you will master:
1. The architectural definition and core mechanisms of Multi-Agent Systems (MAS).
2. The fundamental trade-offs between Single-Agent and Multi-Agent topologies.
3. Designing Agent Specialization through tailored system prompts and localized toolsets.
4. Architecting Agent Communication buses and structured message-passing protocols.
5. Implementing Agent Coordination strategies across centralized and decentralized topologies.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Multi-Agent Collaboration Topology", divider="rainbow")
st.markdown("""
A multi-agent system divides a broad problem domain into discrete, expert personas. Each agent possesses
its own localized instruction prompt, a tightly scoped set of 2 to 4 tools, and communicates across a shared
message bus or state graph.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    mas_dot = """
    digraph MultiAgentFundamentalsBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_ingress {
            label="1. Ingress & Task Delegation";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            user_prompt [label="User Complex Objective\\n'Research competitor, write scraper, audit PII'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            coordinator [label="Supervisor Agent\\nDecomposes task, routes subtasks", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        subgraph cluster_specialists {
            label="2. Specialized Agent Collective";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            research_agent [label="Research Specialist\\nPrompt: Fact verification\\nTools: WebSearch, Docs", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            code_agent [label="Software Engineer Agent\\nPrompt: Code synthesis\\nTools: Python REPL, Git", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            audit_agent [label="Security Auditor Agent\\nPrompt: Compliance check\\nTools: AST Parser, PII Scan", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
        }

        subgraph cluster_communication {
            label="3. Inter-Agent Communication Bus";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            state_bus [label="Shared State Message Bus\\nStores structured typed artifacts & tool history", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            synthesis [label="Final Synthesis Node\\nCombines verified artifacts into final answer", fillcolor="#065F46", fontcolor="#D1FAE5", color="#34D399"];
        }

        user_prompt -> coordinator [label="Dispatches Task"];
        coordinator -> research_agent [label="Step 1: Gather Intel"];
        research_agent -> state_bus [label="Emits Research Notes"];
        state_bus -> code_agent [label="Step 2: Read Intel & Build"];
        code_agent -> audit_agent [label="Step 3: Transfer for Review"];
        audit_agent -> coordinator [label="Emits Compliance Signoff"];
        coordinator -> synthesis [label="Consolidates Result"];
    }
    """
    st.graphviz_chart(mas_dot, use_container_width=True)

# --- SECTION 2: WHAT IS A MULTI-AGENT SYSTEM? ---
st.header("2. What is a Multi-Agent System?", divider="rainbow")
st.markdown("""
A **Multi-Agent System (MAS)** is a computational architecture wherein multiple independent, autonomous LLM-powered
decision engines collaborate, negotiate, and execute actions to achieve a shared objective that exceeds the capabilities
of any single agent.

Rather than cramming twenty tool schemas, thirty policy guidelines, and fifty business rules into one massive prompt,
a multi-agent system decomposes the challenge into modular, specialized agents that act like members of an agile software team.
""")

col_mas1, col_mas2 = st.columns(2)
with col_mas1:
    st.markdown("""
    #### Core Architectural Properties of MAS:
    - **Autonomy**: Each agent operates independently within its bounded task scope.
    - **Specialization**: Agents possess distinct system prompts, personas, and localized tools.
    - **Interoperability**: Agents communicate using standardized message protocols (AIMessage, ToolMessage, JSON schemas).
    - **Decomposition**: Complex, long-running objectives are broken down into discrete, manageable subtasks.
    """)

with col_mas2:
    with st.container(border=True):
        st.markdown("🏢 **Analogy: The Modern Engineering Team**")
        st.caption("No single human performs product management, database design, frontend UX, security audits, and QA.")
        st.markdown("""
        In an engineering department:
        - The **Product Manager** defines requirements and coordinates milestones.
        - The **Software Engineer** writes implementation code.
        - The **QA Engineer** creates regression test suites.
        - The **Security Auditor** inspects vulnerabilities.
        
        Multi-agent architectures mirror this proven human organizational structure in software.
        """)

# --- SECTION 3: SINGLE AGENT VS MULTI-AGENT ---
st.header("3. Single Agent vs Multi-Agent", divider="rainbow")
st.markdown("""
Why not just build one large, capable agent? While single-agent architectures are simpler to deploy for basic workflows,
they encounter severe mathematical and operational limitations as task complexity scales.
""")

st.subheader("Architectural Comparison Matrix")
comp_mas = pd.DataFrame({
    "Dimension": ["Prompt Length & Focus", "Tool Count per Model Call", "Error Recovery", "Debugging & Observability", "Cost per Simple Task", "Scalability Ceiling"],
    "Single Monolithic Agent": ["High prompt bloat (5,000+ tokens)", "15-30 tools (High selection error rate)", "Cascading failure; one error corrupts entire trace", "Extremely opaque reasoning scratchpad", "Low (Single cheap invocation)", "Collapses at 5+ sequential reasoning steps"],
    "Multi-Agent System": ["Minimal, hyper-focused prompts (300 tokens)", "2-4 tools per specialist (Near-zero error)", "Isolated failure; supervisor can re-dispatch subtask", "Crystal-clear module traces in LangSmith", "Higher (Multiple specialized invocations)", "Scales to dozens of coordinated multi-step flows"]
})
st.dataframe(comp_mas, use_container_width=True)

# --- SECTION 4: AGENT SPECIALIZATION ---
st.header("4. Agent Specialization", divider="rainbow")
st.markdown("""
Agent specialization is the discipline of constraining an agent's reasoning scope, role persona, and available tools
to a single well-defined domain.
""")

col_spec1, col_spec2 = st.columns(2)
with col_spec1:
    st.markdown("""
    #### Why Specialization Radically Increases Reliability:
    1. **Elimination of Tool Hallucination**: When an LLM is presented with 30 tool definitions simultaneously, the probability of selecting the wrong tool or hallucinating parameter names exceeds 35%. Presenting 2 relevant tools reduces error rates to near zero.
    2. **Model Tier Optimization**: Not all tasks require expensive frontier models (e.g., Claude 3.5 Sonnet or GPT-4o). A research summarizer can run on a fast, inexpensive model (GPT-4o-mini), while an architectural coding agent uses a flagship frontier model.
    3. **Persona Consistency**: Narrow prompts prevent personality drift and ensure instructions like 'Always return valid JSON' are strictly respected without conflict.
    """)

with col_spec2:
    with st.container(border=True):
        st.markdown("🎯 **Defining Specialized Agent Personas in LCEL**")
        st.code("""
# Specialist 1: SQL Database Specialist
sql_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert SQL engineer. Your sole job is writing SELECT queries."),
    ("placeholder", "{messages}")
])
sql_agent = sql_agent_prompt | model.bind_tools([run_sql_query])

# Specialist 2: Technical Copywriter
copy_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an executive copywriter. Summarize data into clean markdown tables."),
    ("placeholder", "{messages}")
])
copy_agent = copy_agent_prompt | model # Zero tools needed!
        """, language="python")

# --- SECTION 5: AGENT COMMUNICATION ---
st.header("5. Agent Communication", divider="rainbow")
st.markdown("""
Agents must communicate information, task states, and tool results without polluting one another's reasoning traces.
In modern LangChain and LangGraph ecosystems, communication relies on typed message lists and structured state schemas.
""")

comm_dot = """
digraph CommunicationPatterns {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    agent_a [label="Agent A (Researcher)\\nGenerates Analysis", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
    msg_bus [label="Shared State / Channel\\nadd_messages reducer", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    agent_b [label="Agent B (Coder)\\nConsumes Context", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

    agent_a -> msg_bus [label="Emits AIMessage(name='researcher')"];
    msg_bus -> agent_b [label="Reads state['messages']"];
}
"""
st.graphviz_chart(comm_dot, use_container_width=True)

st.markdown("""
#### Three Core Inter-Agent Communication Patterns:
- **1. Shared State Protocol**: Agents read from and write to a common state dictionary. LangGraph channel reducers (such as `Annotated[list, add_messages]`) automatically append communications.
- **2. Direct Point-to-Point Handoff**: Agent A explicitly invokes Agent B using a handoff tool (e.g. `transfer_to_coder(code_spec=...)`).
- **3. Artifact Passing**: Agents pass typed Pydantic models containing structured data (e.g. `ResearchReport`, `CodePatch`) rather than raw unstructured chat text.
""")

# --- SECTION 6: AGENT COORDINATION ---
st.header("6. Agent Coordination", divider="rainbow")
st.markdown("""
Agent coordination determines which agent speaks next, how conflicts are resolved, and when the system determines
that the overall objective has been achieved.
""")

coord_col1, coord_col2 = st.columns(2)
with coord_col1:
    with st.container(border=True):
        st.markdown("👑 **Centralized Coordination (Supervisor Pattern)**")
        st.markdown("""
        - A central Supervisor LLM inspects the current state after every turn.
        - The supervisor decides which specialist agent to activate next.
        - **Advantage**: Deterministic, easy to debug, clear termination condition.
        - **Disadvantage**: Single bottleneck; extra LLM routing call on every step.
        """)

with coord_col2:
    with st.container(border=True):
        st.markdown("🌐 **Decentralized Coordination (Choreography / Mesh)**")
        st.markdown("""
        - Each specialist agent decides on its own who to transfer control to next.
        - Agents hand off execution dynamically via `transfer_to_*` tools.
        - **Advantage**: Highly flexible, mimics natural human peer collaboration.
        - **Disadvantage**: Risk of infinite ping-pong loops between confused agents.
        """)

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Modular Multi-Agent Protocol", divider="rainbow")
st.markdown("""
Below is a clean, runnable multi-agent routing architecture demonstrating two specialized worker agents
coordinated by a typed supervisor node.
""")

prod_mas_code = '''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, Any, List

# Define Specialized Agent Nodes
def research_specialist_node(state: dict) -> dict:
    """Specialist responsible for fact gathering and technical citations."""
    messages = state["messages"]
    # In production, invokes ChatModel.bind_tools([web_search])
    simulated_finding = "Research Finding: LCEL v0.3 introduces unified async streaming runnables."
    return {
        "messages": [AIMessage(content=simulated_finding, name="researcher")],
        "sender": "researcher"
    }

def code_specialist_node(state: dict) -> dict:
    """Specialist responsible for Python code generation and testing."""
    messages = state["messages"]
    # Reads findings emitted by researcher
    simulated_code = "```python\\nchain = prompt | model | StrOutputParser()\\n```"
    return {
        "messages": [AIMessage(content=simulated_code, name="coder")],
        "sender": "coder"
    }

def supervisor_router_node(state: dict) -> str:
    """Determines next specialist based on task progress."""
    history = [m.name for m in state.get("messages", []) if hasattr(m, "name")]
    if "researcher" not in history:
        return "researcher"
    elif "coder" not in history:
        return "coder"
    else:
        return "FINISH"

# Execution Driver
def run_multi_agent_pipeline(user_query: str) -> dict:
    state = {"messages": [HumanMessage(content=user_query)]}
    
    while True:
        next_step = supervisor_router_node(state)
        if next_step == "FINISH":
            break
        elif next_step == "researcher":
            out = research_specialist_node(state)
            state["messages"].extend(out["messages"])
        elif next_step == "coder":
            out = code_specialist_node(state)
            state["messages"].extend(out["messages"])
            
    return state
'''
st.code(prod_mas_code, language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Multi-Agent Collaboration vs Single-Agent Stress Simulator", divider="rainbow")
st.markdown("""
Compare the performance, token consumption, and failure probabilities of a single monolithic agent versus
a specialized multi-agent collective across various complexity thresholds.
""")

mas_engine = MultiAgentCollaborationEngine()

col_bench1, col_bench2 = st.columns([1, 1])

with col_bench1:
    st.subheader("⚙️ Mission Complexity Settings")
    complexity_slider = st.slider("Select Mission Task Complexity Level:", min_value=1, max_value=5, value=3)
    st.caption("1 = Simple Lookup | 3 = Multi-file Refactor | 5 = Enterprise Architecture Audit")
    
    sim_mode = st.radio("Simulation Execution Mode:", ["Side-by-Side Benchmark", "Inspect Agent Specialists"])
    run_sim_btn = st.button("🚀 Run Multi-Agent Orchestration Benchmark", type="primary", use_container_width=True)

with col_bench2:
    st.subheader("📊 Architectural Performance Telemetry")
    if run_sim_btn:
        single_res = mas_engine.simulate_single_agent(complexity_slider)
        multi_res = mas_engine.simulate_multi_agent(complexity_slider)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("##### 👤 Single Monolithic Agent")
            st.metric("Total Tokens Ingested", f"{single_res['total_tokens']:,}")
            st.metric("Tool Confusion Risk", f"{single_res['tool_confusion_probability']*100:.0f}%")
            if single_res["success"]:
                st.success(f"Status: {single_res['status']}")
            else:
                st.error(f"Status: {single_res['status']}")

        with col_m2:
            st.markdown("##### 🤖 Specialized Multi-Agent Collective")
            st.metric("Total Tokens Ingested", f"{multi_res['total_tokens']:,}")
            st.metric("Tool Confusion Risk", f"{multi_res['tool_confusion_probability']*100:.0f}%")
            st.success(f"Status: {multi_res['status']}")

        st.divider()
        st.markdown("##### 🔍 Architectural Efficiency Takeaway")
        if complexity_slider >= 3:
            st.info("💡 **Observation**: At complexity >= 3, the single agent suffers high risk of tool selection failure due to instruction overload. The multi-agent collective maintains deterministic stability by bounding each specialist's tool context.")
        else:
            st.info("💡 **Observation**: At complexity 1, single agents are cost-effective because minimal tools are required. MAS is justified when complexity rises.")
    else:
        st.info("Adjust complexity and click 'Run Multi-Agent Orchestration Benchmark' to view comparative telemetry.")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: What is the primary cause of failure in single-agent architectures as workflows expand?"):
    st.markdown("""
    **Answer**: Tool selection degradation and context saturation. As an agent is provisioned with 10+ tools and a multi-page
    system prompt, the model's attention mechanism experiences interference. The agent frequently hallucinates tool arguments,
    selects irrelevant tools, or loses track of its primary objective midway through long conversational scratchpads.
    """)

with st.expander("Q2: How does a Supervisor Agent differ from a Router Agent in LangChain?"):
    st.markdown("""
    **Answer**: A Router Agent evaluates ingress user input once and dispatches control down a single branch (one-way routing).
    A Supervisor Agent maintains an ongoing state loop: it receives results back from worker agents, decides what the next step
    is, re-dispatches tasks iteratively, and explicitly determines when the multi-step mission is complete.
    """)

with st.expander("Q3: What are the trade-offs between Centralized and Decentralized Multi-Agent Systems?"):
    st.markdown("""
    **Answer**: Centralized systems (Supervisor pattern) are predictable, easy to log and debug, and have clear termination gates,
    but the supervisor model can become an architectural and token cost bottleneck. Decentralized systems (Peer-to-peer mesh)
    allow fluid handoffs between agents, but are notoriously prone to infinite recursion loops and difficult to trace.
    """)

with st.expander("Q4: Why is artifact passing preferred over raw conversation history in multi-agent collaboration?"):
    st.markdown("""
    **Answer**: If all agents share the full raw chat history, every turn re-transmits thousands of redundant tokens to every specialist.
    Artifact passing extracts key structured outputs (e.g., a synthesized Markdown brief, a JSON schema, or code file) and passes
    only the clean artifact to the next specialist, slashing token consumption by 70% and keeping specialist context windows clean.
    """)

with st.expander("Q5: How does LangGraph facilitate Multi-Agent Systems compared to legacy LangChain?"):
    st.markdown("""
    **Answer**: Legacy LangChain attempted multi-agent flows using nested `AgentExecutor` chains, which lacked cyclic loops,
    state persistence, and clean message attribution. LangGraph introduces cyclic state machines with typed schemas, explicit
    node handoffs, checkpointer persistence, and human-in-the-loop interrupts natively designed for multi-agent choreography.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The Infinite Ping-Pong Loop)**: In decentralized peer-to-peer handoffs, Agent A asks Agent B for feedback, and Agent B hands back to Agent A indefinitely. Always enforce a hard recursion limit (e.g. `recursion_limit=25`).")
    st.warning("**Trap 2 (Full History Broadcast)**: Broadcasting the entire multi-thousand token conversational scratchpad to every specialist agent explodes API token bills and causes attention dilution.")
with col_trap2:
    st.error("**Trap 3 (Over-Engineering Simple Workflows)**: Deploying a 4-agent collective for a straightforward FAQ lookup adds latency and multi-dollar API overhead. Use multi-agent patterns only for genuinely multi-faceted objectives.")
    st.warning("**Trap 4 (Unvalidated Inter-Agent State)**: Passing raw string outputs between agents without Pydantic schema validation causes downstream parser crashes when an upstream agent emits slightly malformed text.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is a Multi-Agent System (MAS)?")
        st.caption("Flip for answer:")
        st.info("**Back**: An architecture where multiple autonomous, specialized LLM instances collaborate via message passing and division of labor to complete complex goals.")
    with st.container(border=True):
        st.markdown("**Front**: Why does Agent Specialization reduce tool errors?")
        st.caption("Flip for answer:")
        st.info("**Back**: Bounding each agent to 2-4 tools eliminates model attention confusion and prevents tool selection hallucinations.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the Supervisor Agent pattern?")
        st.caption("Flip for answer:")
        st.info("**Back**: A centralized orchestrator LLM that inspects ongoing state, assigns tasks to worker specialists, and terminates the loop when the mission is accomplished.")
    with st.container(border=True):
        st.markdown("**Front**: What is Artifact Passing in multi-agent design?")
        st.caption("Flip for answer:")
        st.info("**Back**: Transferring concise structured outputs (reports, code, schemas) between agents rather than broadcasting entire raw conversation histories.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the architectural limitations of monolithic single-agent designs.", value=True)
c2 = st.checkbox("I know how to define specialized agent personas and bounded toolsets.", value=True)
c3 = st.checkbox("I understand the distinction between Centralized and Decentralized MAS topologies.", value=True)
c4 = st.checkbox("I know how to implement inter-agent message passing and artifact handoffs.", value=True)
c5 = st.checkbox("I recognize how to prevent infinite ping-pong loops using recursion limits.", value=True)
c6 = st.checkbox("I know when to choose a multi-agent system over a simple chain or single agent.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 26.2: Multi-Agent Patterns.")
