import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="9.2 Creating Agents - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("9.2 Creating Agents: Architectures, Factory Functions, System Prompts & State")
st.caption("Chapter 9: Agents | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In early LangChain versions, creating an agent relied on high-level legacy abstractions like `initialize_agent` 
and `AgentExecutor`. In modern LangChain (v0.3+), agents are constructed via explicit, type-safe architectures 
powered by **LangGraph runtime primitives** and modular factory functions (`create_react_agent`).

In this lesson, you will master the 7 fundamental components of creating production-grade agents:
1. **Modern Agent Architectures**: Transitioning from legacy `AgentExecutor` to modern state-machine runtimes.
2. **Model Selection & Configuration**: Tuning frontier chat models for deterministic tool calling.
3. **Tools Assembly**: Compiling, validating, and binding tool registries.
4. **System Prompts**: Engineering authoritative operational guardrails and behavioral guidelines.
5. **Agent State Management**: Tracking message histories and intermediate scratchpads.
6. **Agent Invocation**: Executing agents via `.invoke()` and async streaming protocols.
7. **Agent Response Unpacking**: Extracting clean final responses, intermediate actions, and execution metadata.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockCompiledAgentResult:
    """Represents a simulated compiled agent in module scope."""
    def __init__(self, agent_name: str, model_name: str, tools_bound: List[str], system_prompt_len: int, state_schema: str):
        self.agent_name = agent_name
        self.model_name = model_name
        self.tools_bound = tools_bound
        self.system_prompt_len = system_prompt_len
        self.state_schema = state_schema

class MockAgentFactoryStudio:
    """Simulates agent assembly, state compilation, and mock invocation in module scope."""
    def __init__(self):
        pass

    def compile_agent(self, name: str, model: str, tools: List[str], sys_prompt: str) -> MockCompiledAgentResult:
        return MockCompiledAgentResult(
            agent_name=name.strip() if name.strip() else "EnterpriseSupportAgent",
            model_name=model,
            tools_bound=tools,
            system_prompt_len=len(sys_prompt),
            state_schema="TypedDict(messages: Annotated[List[BaseMessage], add_messages])"
        )

    def execute_mock_invocation(self, user_msg: str, agent_meta: MockCompiledAgentResult) -> Dict[str, Any]:
        has_tools = len(agent_meta.tools_bound) > 0
        tool_used = agent_meta.tools_bound[0] if has_tools else "None"
        
        return {
            "steps_count": 2 if has_tools else 1,
            "tool_invoked": tool_used,
            "intermediate_thought": f"Analyzed user query '{user_msg[:40]}...' using system guidelines. Invoking tool '{tool_used}' to ground facts.",
            "final_reply": f"[{agent_meta.model_name}] Handled request successfully. Verified facts using registered capabilities ({', '.join(agent_meta.tools_bound)})."
        }

# --- SECTION 1: ARCHITECTURE BLUEPRINT & AGENT ASSEMBLY ---
st.header("1. Architecture Blueprint: The Modern Agent Factory & State Assembly Pipeline", divider="rainbow")

st.markdown("""
The architecture below illustrates how modern agents are assembled. 
An agent is not a single black-box object; it is an orchestrated composite of a **Chat Model**, 
a **Tool Registry**, a **System Persona Prompt**, and an underlying **State Machine Graph**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Level 1: Building Blocks
        subgraph cluster_blocks {
            label = "1. INGREDIENTS &amp; SPECIFICATIONS";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            ModelSpec [label="Model Instance\\nChatOpenAI(model='gpt-4o', temp=0.0)"];
            ToolsSpec [label="Tool Registry List\\n[search_docs, query_db, send_slack]"];
            PromptSpec [label="System Prompt Message\\n'You are an authoritative IT Ops assistant...'"];
            StateSpec [label="State Channel Schema\\nMessagesState (Annotated with add_messages)"];
        }

        // Level 2: Modern Factory
        subgraph cluster_factory {
            label = "2. AGENT FACTORY &amp; COMPILER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            FactoryCall [label="create_react_agent(model, tools, prompt)\\nCompiles into a stateful StateGraph", color="#10B981"];
        }

        // Level 3: Compiled Runtime Graph
        subgraph cluster_runtime_graph {
            label = "3. COMPILED RUNTIME GRAPH (LangGraph Core)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            AgentNode [label="Agent Reasoning Node\\nInvokes model.bind_tools(tools)"];
            ConditionalEdge [label="Conditional Routing Edge\\nInspects: len(AIMessage.tool_calls) > 0 ?"];
            ToolNode [label="Tool Execution Node\\nRuns tools &amp; produces ToolMessages", color="#38BDF8"];
        }

        // Level 4: Execution & Observability
        subgraph cluster_execution {
            label = "4. INVOCATION &amp; CONSUMPTION";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            InvokeCall [label="agent.invoke({'messages': [HumanMessage(...)]})\\nStreams or blocks until final resolution", color="#10B981"];
        }

        // Wiring
        ModelSpec -> FactoryCall;
        ToolsSpec -> FactoryCall;
        PromptSpec -> FactoryCall;
        StateSpec -> FactoryCall;

        FactoryCall -> AgentNode;
        AgentNode -> ConditionalEdge;
        ConditionalEdge -> ToolNode [label="tool_calls present"];
        ToolNode -> AgentNode [label="append observations &amp; repeat", color="#38BDF8"];
        ConditionalEdge -> InvokeCall [label="no tool_calls (DONE)", color="#10B981"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: create_agent() / MODERN AGENT ARCHITECTURES ---
st.header("2. Modern Agent Architectures (`create_react_agent` & LangGraph Runtime)", divider="rainbow")

st.markdown("""
Historically, LangChain relied on `AgentExecutor` (a rigid, monolithic loop). 
In modern LangChain, **LangGraph** provides the standard execution engine for all agents. 
The canonical function to instantiate an agent is `create_react_agent()` from `langgraph.prebuilt`.

Why the architectural migration occurred:
- **True Cycles & State**: `AgentExecutor` struggled with complex branch loops, custom multi-agent delegation, and human-in-the-loop pauses.
- **Type-Safe State Channels**: The LangGraph runtime uses explicit typed state schemas with automatic message appending reducers (`add_messages`).
- **Persistence & Checkpointing**: Built-in support for time-travel debugging and multi-session conversational memory.
""")

st.subheader("Instantiating a Modern ReAct Agent")
st.code('''from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def calculate_uptime_percentage(total_hours: int, downtime_minutes: float) -> str:
    """Calculate server availability percentage over a billing period."""
    uptime_ratio = 1.0 - (downtime_minutes / (total_hours * 60))
    return f"Availability: {uptime_ratio * 100:.3f}%"

model = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [calculate_uptime_percentage]

# Compile the agent graph
agent = create_react_agent(
    model=model,
    tools=tools,
    state_modifier="You are a Site Reliability Engineer. Use tools to calculate exact SLA metrics."
)
''', language="python")

st.divider()

# --- SECTION 3: MODELS FOR AGENTS ---
st.header("3. Models for Agents (Selection, Temperature & Tool Binding)", divider="rainbow")

st.markdown("""
Not every LLM can function effectively as an agent controller. Agents require models with exceptional 
**instruction-following**, **context synthesis**, and **tool calling fidelity** (e.g. GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro).
""")

st.subheader("Model Configuration Best Practices")
st.markdown("""
1. **Set Temperature to 0.0**: Agents require deterministic, reproducible reasoning. Higher temperatures cause models to hallucinate tool parameters or select inappropriate tools.
2. **Verify Tool-Calling Support**: Ensure the chosen model provider supports native function calling.
3. **Set Realistic Timeouts**: Tools can take several seconds to execute; configure model client timeouts accordingly.
""")

st.code('''# Recommended Agent Model Configuration
from langchain_openai import ChatOpenAI

agent_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,       # Strict determinism
    timeout=30.0,          # Accommodates multi-step tool calls
    max_retries=3          # Built-in network fault tolerance
)
''', language="python")

st.divider()

# --- SECTION 4: TOOLS COMPILATION ---
st.header("4. Tools Compilation (Tool Registries & Binding)", divider="rainbow")

st.markdown("""
Tools passed to an agent must be organized into a clean Python list. 
When compiling the agent, the factory automatically binds the tools to the model and registers a dedicated 
`ToolNode` inside the execution graph to run the tools.
""")

st.code('''from langchain_core.tools import tool

@tool
def search_system_logs(query: str) -> str:
    """Search centralized Elasticsearch cluster for system error traces."""
    return f"Search result for '{query}': Found 3 NPE stack traces in payment-service."

@tool
def restart_microservice(service_name: str) -> str:
    """Trigger a rolling restart of an un-responsive microservice."""
    return f"Initiated rolling restart of deployment '{service_name}'."

# Assembling the tools list:
agent_tools = [search_system_logs, restart_microservice]
''', language="python")

st.divider()

# --- SECTION 5: SYSTEM PROMPTS ---
st.header("5. System Prompts (Personas & Operational Guardrails)", divider="rainbow")

st.markdown("""
The **System Prompt** (passed as `state_modifier` or `prompt` in modern agent factories) establishes the 
agent's persona, capabilities, security boundaries, and operational priorities.
""")

st.subheader("Anatomy of an Enterprise Agent System Prompt")
st.code('''SYSTEM_PROMPT = """You are a Principal DevOps Triage Agent assisting system administrators.

OPERATIONAL PRINCIPLES:
1. Always search logs before proposing disruptive remediations.
2. If an action could cause downtime (such as restarting a service), warn the user and explain the rationale.
3. Keep answers concise, factual, and backed by tool observations.
4. If you lack information to answer a question, explicitly state what is missing rather than guessing.
"""
''', language="python")

st.divider()

# --- SECTION 6: AGENT STATE ---
st.header("6. Agent State (Message Channels & Reducers)", divider="rainbow")

st.markdown("""
Under the hood, modern agents store their memory in a **State Schema**. 
The default schema is `MessagesState`, which tracks an ongoing list of messages (`HumanMessage`, `AIMessage`, `ToolMessage`).

In LangGraph-based agents, messages are accumulated using the **`add_messages` reducer**: 
instead of overwriting the previous message list, new messages are automatically appended, 
and messages with matching IDs are updated.
""")

st.code('''from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# The standard Agent State contract:
class CustomAgentState(TypedDict):
    # 'add_messages' ensures new turns are appended rather than overwritten
    messages: Annotated[List[BaseMessage], add_messages]
    user_tenant_id: str
    security_clearance_level: int
''', language="python")

st.divider()

# --- SECTION 7: AGENT INVOCATION ---
st.header("7. Agent Invocation (`agent.invoke` & Streaming Protocols)", divider="rainbow")

st.markdown("""
Invoking an agent passes an initial state dictionary containing the user's prompt. 
The agent executes the graph until it reaches the end node, returning the final state dictionary.
""")

st.code('''from langchain_core.messages import HumanMessage

# Standard Synchronous Invocation:
final_state = agent.invoke({
    "messages": [HumanMessage(content="Calculate our availability for 720 hours with 45 minutes downtime.")]
})

# Accessing the final message from the conversation history:
final_message = final_state["messages"][-1]
print(final_message.content)

# Streaming Invocation (Stream intermediate reasoning steps chunk-by-chunk):
for chunk in agent.stream(
    {"messages": [HumanMessage(content="Diagnose why node-04 is degraded.")]},
    stream_mode="updates"
):
    print(chunk)
''', language="python")

st.divider()

# --- SECTION 8: AGENT RESPONSES ---
st.header("8. Agent Responses (Unpacking Messages & Tool Trajectories)", divider="rainbow")

st.markdown("""
The return value of `agent.invoke()` is not a raw string; it is the **complete final state dictionary**. 
This state contains the entire conversation trajectory:
""")

st.subheader("Inspecting Intermediate Agent Messages")
st.code('''# Inspecting the full execution trajectory
for i, msg in enumerate(final_state["messages"]):
    role = msg.__class__.__name__
    content_preview = msg.content[:80] if msg.content else f"Tool Calls: {msg.tool_calls}"
    print(f"[{i}] {role}: {content_preview}")

# Example output:
# [0] HumanMessage: Calculate availability for 720 hours with 45 mins downtime.
# [1] AIMessage: Tool Calls: [{'name': 'calculate_uptime_percentage', 'args': {'total_hours': 720, 'downtime_minutes': 45.0}}]
# [2] ToolMessage: Availability: 99.896%
# [3] AIMessage: Based on the calculation, the system achieved 99.896% availability over 720 hours.
''', language="python")

st.divider()

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Enterprise Modular Agent Assembly Engine", divider="rainbow")

st.markdown("""
The following complete production pattern demonstrates a modular agent creation factory 
featuring system prompt injection, tool validation, deterministic model tuning, and clean execution wrapping.
""")

st.code('''import logging
from typing import List, Dict, Any, Optional
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage

logger = logging.getLogger("EnterpriseAgentFactory")

# Production Tools
@tool
def lookup_dns_record(domain: str) -> str:
    """Resolve IPv4 address records for a domain name."""
    return f"DNS A-Record for {domain}: 104.21.48.91 (TTL: 300s)"

@tool
def check_ssl_certificate(domain: str) -> str:
    """Verify validity and expiration date of TLS/SSL certificate."""
    return f"Certificate for {domain} is VALID. Issued by Let's Encrypt, expires in 68 days."

class NetworkDiagnosticsAgentFactory:
    """Factory service that constructs and executes network diagnostics agents."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            timeout=25.0
        )
        self.tools: List[BaseTool] = [lookup_dns_record, check_ssl_certificate]
        self.system_prompt = """You are an automated Network Infrastructure Diagnostic Agent.
Analyze network connectivity and security issues methodically.
Always verify DNS resolution before investigating SSL certificate health."""

        # Compile the underlying LangGraph ReAct agent
        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            state_modifier=self.system_prompt
        )

    def diagnose_host(self, host_query: str) -> Dict[str, Any]:
        logger.info(f"Dispatching network diagnosis for: {host_query}")
        try:
            state = self.agent.invoke({
                "messages": [HumanMessage(content=host_query)]
            })
            
            final_message = state["messages"][-1]
            return {
                "status": "SUCCESS",
                "final_answer": final_message.content,
                "total_messages": len(state["messages"])
            }
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            return {
                "status": "FAILED",
                "error": str(e)
            }
''', language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Live Agent Factory & Execution Studio", divider="rainbow")

st.markdown("""
Experiment with assembling an agent dynamically. Configure the agent persona, select model checkpoints, 
choose bound tools, and simulate the compiled agent's execution.
""")

studio = MockAgentFactoryStudio()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("⚙️ Agent Assembly Configuration")
    agent_name_input = st.text_input("Agent Identifier", value="DevOps-Incident-Commander")
    model_choice = st.selectbox("Frontier LLM Brain", ["gpt-4o (OpenAI)", "claude-3-5-sonnet (Anthropic)", "gemini-1.5-pro (Google)"])
    
    available_tools = [
        "query_customer_db",
        "fetch_cluster_metrics",
        "restart_service",
        "send_slack_alert"
    ]
    selected_tools = st.multiselect("Bound Tools", available_tools, default=["query_customer_db", "fetch_cluster_metrics"])
    
    system_prompt_input = st.text_area(
        "System Prompt Guidelines",
        value="You are an autonomous incident triage agent. Ground all conclusions in verified telemetry data.",
        height=100
    )

with col_wb2:
    st.subheader("📋 Compiled Agent Graph Metadata")
    compiled_agent = studio.compile_agent(agent_name_input, model_choice, selected_tools, system_prompt_input)
    st.info(f"""
    **Agent Name**: `{compiled_agent.agent_name}`  
    **Brain Model**: `{compiled_agent.model_name}`  
    **Tools Count**: `{len(compiled_agent.tools_bound)}` tools active  
    **System Prompt Length**: `{compiled_agent.system_prompt_len}` chars  
    **State Channel**: `{compiled_agent.state_schema}`
    """)

st.write("#### 🧪 Test Invocation")
test_query = st.text_input("User Message to Agent", value="Diagnose why checkout throughput dropped by 40% at 14:00 UTC.")

if st.button("🚀 Invoke Compiled Agent", type="primary"):
    res = studio.execute_mock_invocation(test_query, compiled_agent)
    st.subheader("📊 Execution Response & State")
    
    m1, m2 = st.columns(2)
    m1.metric("Reasoning Steps", res["steps_count"])
    m2.metric("Tool Invoked", res["tool_invoked"])

    st.write("**Intermediate Thought Trace**:")
    st.caption(res["intermediate_thought"])

    st.write("**Final Output**:")
    st.success(res["final_reply"])

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why did LangChain deprecate AgentExecutor in favor of LangGraph create_react_agent?",
     "AgentExecutor was a rigid, black-box Python while-loop that could not cleanly handle complex cycles, human-in-the-loop interruptions, multi-agent hierarchies, or time-travel state persistence. LangGraph treats agents as explicit, type-safe cyclic state machines, providing superior observability, checkpointing, and customization."),
    ("Why should temperature be set to 0.0 for autonomous agent controllers?",
     "Autonomous agents rely on deterministic tool selection and exact parameter adherence. Non-zero temperatures introduce stochastic sampling variance, which can cause the model to hallucinate non-existent tool names, generate malformed arguments, or trigger unexpected tool loops."),
    ("What is the purpose of the add_messages reducer in modern agent state schemas?",
     "add_messages is a state channel reducer that governs how new messages are merged into conversation history. Instead of replacing the entire message list, it automatically appends incoming messages, updates existing messages if IDs match, and preserves the full chronological trajectory across multi-turn reasoning loops."),
    ("How does streaming work when invoking a modern ReAct agent?",
     "Agents can stream in multiple modes: stream_mode='values' (yields the entire state dictionary after each node execution), stream_mode='updates' (yields only the incremental changes made by the latest node), or token streaming via .astream_events() which streams raw LLM tokens as they are generated."),
    ("How does state_modifier differ from prepending a SystemMessage manually?",
     "state_modifier in create_react_agent allows you to pass a string or a callable that dynamically modifies the prompt before model invocation on every turn. This ensures the system instructions are consistently injected into the prompt context throughout long, multi-turn tool conversations.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using Chat Models Without Native Tool-Calling Support. Passing older completion models (or models without function-calling fine-tuning) into create_react_agent will fail because the model will generate conversational text instead of structured tool_calls.")
    st.error("**Trap 2 (Breaking Error)**: State Mutation Collisions in Multi-Threaded Services. Sharing a single agent instance across concurrent user requests without passing distinct session/thread IDs causes conversation histories to cross-contaminate. Always isolate thread contexts.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Context Window Overflow in Long Tool Runs. If an agent executes 10 tool calls and each tool returns 2,000 tokens of raw data, the prompt context will balloon to 20,000+ tokens, rapidly exhausting context limits. Implement message pruning or summarization middleware.")
    st.warning("**Trap 4 (Quality Warning)**: Overloaded Tool Catalogs. Binding 30+ tools to a single agent degrades selection accuracy. Group tools into specialized domain sub-agents and use a supervisor router instead.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: create_react_agent()"):
        st.markdown("**The modern factory function creating LangGraph-powered ReAct agents.**")
    with st.expander("🎴 Flashcard 2: Agent Temperature Standard"):
        st.markdown("**Always set temperature=0.0 for deterministic tool selection and argument generation.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: add_messages Reducer"):
        st.markdown("**Appends new messages into the agent's state channel rather than overwriting.**")
    with st.expander("🎴 Flashcard 4: state_modifier Parameter"):
        st.markdown("**Configures system prompt guidelines and operational guardrails for the agent.**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand why modern LangChain uses LangGraph create_react_agent over legacy AgentExecutor.", value=True)
chk2 = st.checkbox("I know how to configure chat models with temperature=0.0 for reliable agent control.", value=True)
chk3 = st.checkbox("I can compile tool lists and bind them into an agent graph.", value=True)
chk4 = st.checkbox("I understand how to engineer effective system prompts and state modifiers.", value=True)
chk5 = st.checkbox("I know how the add_messages reducer manages state accumulation.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 agent creation interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 9.2 Creating Agents. Ready for 9.3 Agent Execution!")
