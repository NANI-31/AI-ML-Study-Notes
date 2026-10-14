import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="34.3 Agent Performance - LangChain Curriculum",
    page_icon="🤖",
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
    .agent-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-speed {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class AgentPerformanceEngine:
    """
    Simulates high-throughput autonomous agent execution:
    parallel tool dispatching, schema pruning, and observation compression.
    """
    def __init__(self):
        self.mock_tools = {
            "fetch_user_profile": {"desc": "Retrieves demographic and billing tier metadata.", "latency_ms": 120},
            "fetch_order_history": {"desc": "Queries customer past order items.", "latency_ms": 180},
            "calculate_tax_rebate": {"desc": "Applies federal tax rebate formula.", "latency_ms": 60},
            "send_notification_email": {"desc": "Dispatches confirmation email.", "latency_ms": 210}
        }

    def simulate_dispatch(self, parallel: bool, selected_tools: List[str]) -> Dict[str, Any]:
        latencies = [self.mock_tools[t]["latency_ms"] for t in selected_tools if t in self.mock_tools]
        if not latencies:
            return {"total_latency_ms": 0, "parallel": parallel, "tools_count": 0}

        if parallel:
            # Parallel execution duration is max latency + small overhead
            total_duration = max(latencies) + 25
        else:
            # Sequential execution duration is sum of latencies
            total_duration = sum(latencies) + (len(latencies) * 15)

        savings = 0.0
        if not parallel and len(latencies) > 1:
            savings = 0.0
        elif parallel and len(latencies) > 1:
            seq_time = sum(latencies) + (len(latencies) * 15)
            savings = round((1 - (total_duration / max(seq_time, 1))) * 100, 1)

        return {
            "total_latency_ms": total_duration,
            "parallel": parallel,
            "tools_count": len(latencies),
            "savings_percent": savings
        }

# Instantiate engine
agent_perf_engine = AgentPerformanceEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Autonomous Agent Acceleration, Parallel Dispatch, & Execution Tree Pruning", divider="rainbow")

st.markdown("""
Autonomous AI agents are notoriously prone to high latency and spiraling costs when poorly optimized. 
A naive agent loops sequentially through single tool calls, appends massive raw API responses to state, 
and relies on expensive frontier models for every minor formatting step. 
**Agent Performance Optimization** restructures agents to execute tools in parallel, filter observations at the boundary, 
and dynamically route simple tasks to high-speed models.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Agent_Performance_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_ingress {
            label = "Agent Planning & Routing Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            agent_planner [label="Agent Planner (Fast Model)\\n(Analyzes goal, determines necessary tool calls)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            parallel_emitter [label="Parallel Tool Call Emitter\\n(Emits 3 ToolCalls in a single LLM response turn)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        subgraph cluster_tools {
            label = "Asynchronous Concurrent Tool Execution";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            t_user [label="Tool 1: User Profile API\\n(Async HTTP: 120ms)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            t_orders [label="Tool 2: Orders Database\\n(Async SQL: 180ms)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            t_tax [label="Tool 3: Tax Calculator\\n(In-memory compute: 40ms)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_filtering {
            label = "Observation Filtering & Context Compression Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            obs_compressor [label="Boundary Observation Filter\\n(Extracts only needed fields, drops 20KB raw JSON)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
            agent_synthesizer [label="Synthesis & Action Step\\n(Produces final answer in Turn 2 rather than Turn 6)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        agent_planner -> parallel_emitter [label="bind_tools"];
        parallel_emitter -> t_user [label="async gather"];
        parallel_emitter -> t_orders [label="async gather"];
        parallel_emitter -> t_tax [label="async gather"];

        t_user -> obs_compressor;
        t_orders -> obs_compressor;
        t_tax -> obs_compressor;

        obs_compressor -> agent_synthesizer [label="clean compact state"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Parallel Execution Paradigm:</strong> By leveraging modern LLM multi-tool calling capabilities, 
    an agent can emit 3 distinct tool requests in a single generation step. Executing these tools concurrently via 
    <code>asyncio.gather()</code> reduces execution time from the sum of latencies (340ms) to the maximum single latency (180ms).
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: TOOL OPTIMIZATION
# ==============================================================================
st.header("2. Tool Optimization: Lean Pydantic Schemas and Concise Docstrings", divider="rainbow")

st.markdown("""
When an LLM is provided with tools via `bind_tools`, every tool description, parameter name, 
and Pydantic docstring is serialized into the system prompt. Bloated tool schemas degrade reasoning and inflate token costs.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Antipattern: Verbose Tool Definition
    ```python
    @tool
    def search_user(query: str, verbose: bool = True):
        \"\"\"This is a function that you should call 
        whenever you want to search for a user in the 
        internal MySQL database. Make sure to provide 
        the full name if possible. Do not pass empty strings.\"\"\"
        pass
    ```
    - **Overhead**: Long narrative docstrings consume unnecessary prompt tokens on every turn.
    """)

with col2:
    st.markdown("""
    ### Modern Standard: Lean Pydantic Tool
    ```python
    class UserSearchInput(BaseModel):
        user_id: str = Field(description="Unique 8-char ID")

    @tool(args_schema=UserSearchInput)
    def search_user(user_id: str) -> dict:
        \"\"\"Lookup account tier and status by user ID.\"\"\"
        pass
    ```
    - **Efficiency**: Minimal description, precise typed parameter constraints.
    """)

# ==============================================================================
# SECTION 3: PARALLEL TOOL CALLS
# ==============================================================================
st.header("3. Parallel Tool Calls: Native Multi-Tool Invocations and Async Gathering", divider="rainbow")

st.markdown("""
Leading models (GPT-4o, Claude 3.5 Sonnet) natively support **parallel tool calling**. 
Instead of requiring 3 separate reasoning loops to fetch profile, order history, and account balance, 
the model emits all 3 tool calls simultaneously in `tool_calls`.
""")

st.code('''import asyncio
from langchain_core.messages import AIMessage, ToolMessage

async def execute_parallel_tool_calls(ai_message: AIMessage, tool_map: dict) -> list[ToolMessage]:
    """
    Executes all tool calls emitted by the model concurrently.
    """
    async def call_single_tool(tc):
        tool_fn = tool_map[tc["name"]]
        # Execute async tool
        result = await tool_fn.ainvoke(tc["args"])
        return ToolMessage(content=str(result), tool_call_id=tc["id"])

    # Launch all tools in parallel using asyncio.gather
    tool_messages = await asyncio.gather(
        *(call_single_tool(tc) for tc in ai_message.tool_calls)
    )
    return list(tool_messages)
''', language="python")

# ==============================================================================
# SECTION 4: AGENT ROUTING
# ==============================================================================
st.header("4. Agent Routing: Intent-Based Specialization and Subagent Delegation", divider="rainbow")

st.markdown("""
Providing a single monolithic agent with 30 tools overwhelms model attention, leading to 
hallucinated tool selections and degraded accuracy. **Agent Routing** divides responsibilities across specialized subagents.
""")

st.markdown("""
### Supervisor-Worker Architecture
1. **Router / Supervisor**: Equipped with zero heavy tools; possesses only routing capability to classify intent (e.g., 'Billing', 'Technical Support', 'Compliance').
2. **Specialized Worker Agents**: Each subagent possesses only 3 to 5 domain-specific tools, maximizing tool selection accuracy and reducing system prompt sizes by 80%.
""")

# ==============================================================================
# SECTION 5: MODEL ROUTING
# ==============================================================================
st.header("5. Model Routing: Pairing Fast SLMs for Tool Selection with Frontier Models for Synthesis", divider="rainbow")

st.markdown("""
Agents frequently alternate between two phases: **Planning/Tool Selection** (which requires structured tool binding) 
and **Synthesis** (which requires nuanced natural language generation).
""")

m_router_df = pd.DataFrame([
    {"Agent Phase": "Tool Selection & Argument Parsing", "Optimal Model": "GPT-4o-mini / Claude 3.5 Haiku", "Rationale": "Extremely fast TTFT (< 200ms); highly proficient at structured function calling."},
    {"Agent Phase": "Tool Result Error Recovery", "Optimal Model": "GPT-4o-mini", "Rationale": "Simple retry logic does not require frontier reasoning."},
    {"Agent Phase": "Final Strategic Synthesis & Recommendation", "Optimal Model": "GPT-4o / Claude 3.5 Sonnet", "Rationale": "Delivers superior prose quality, nuanced trade-off analysis, and executive polish."}
])
st.dataframe(m_router_df, use_container_width=True)

# ==============================================================================
# SECTION 6: REDUCING UNNECESSARY CALLS
# ==============================================================================
st.header("6. Reducing Unnecessary Calls: Guardrail Pre-Checks and Deterministic Short-Circuits", divider="rainbow")

st.markdown("""
Every LLM call avoided represents an instant 100% latency reduction. 
Production agent architectures employ **deterministic short-circuits** before invoking reasoning loops:
""")

st.markdown("""
- **Regex & Keyword Routing**: Queries like 'What is my current balance?' can be handled directly via an authenticated SQL query without passing through an LLM reasoning loop.
- **Tool Call Deduplication**: If an agent attempts to invoke the same tool with identical arguments twice in a single thread, intercept the call and return the previous cached result.
- **Maximum Recursion Guards**: Enforce strict recursion limits (`recursion_limit=10` in LangGraph) to prevent runaway infinite loops when a tool returns ambiguous errors.
""")

# ==============================================================================
# SECTION 7: CONTEXT COMPRESSION
# ==============================================================================
st.header("7. Context Compression: State History Pruning and Observation Filtering", divider="rainbow")

st.markdown("""
A primary cause of agent degradation is **observation bloat**. 
When a tool returns a massive 50KB raw JSON payload, appending it verbatim into conversation history 
pollutes the prompt context for all subsequent turns.
""")

st.code('''def boundary_observation_filter(raw_api_response: dict) -> dict:
    """
    Extracts only essential fields before creating ToolMessage.
    Reduces observation payload from 5,000 tokens to 45 tokens.
    """
    return {
        "status": raw_api_response.get("status"),
        "account_id": raw_api_response.get("id"),
        "balance_usd": raw_api_response.get("financials", {}).get("current_balance"),
        "is_active": raw_api_response.get("account_state") == "ACTIVE"
    }
''', language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: High-Speed Async Multi-Tool Agent Pipeline in LangChain", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script demonstrating a 
**high-speed agent dispatcher with parallel execution and observation compression**.
""")

st.code('''import time
from typing import Dict, Any, List

class HighSpeedAgentDispatcher:
    def __init__(self):
        self.tool_latencies = {
            "get_balance": 0.08,
            "get_credit_score": 0.12,
            "check_kyc_status": 0.06
        }

    def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Simulates individual tool execution with observation compression."""
        latency = self.tool_latencies.get(tool_name, 0.05)
        time.sleep(latency)
        
        # Boundary compression: return only minimal facts
        if tool_name == "get_balance":
            return {"balance": 4500.00, "currency": "USD"}
        elif tool_name == "get_credit_score":
            return {"score": 780, "tier": "EXCELLENT"}
        return {"kyc_verified": True}

    def dispatch_parallel_plan(self, requested_tools: List[str]) -> Dict[str, Any]:
        """Simulates parallel execution of multiple tool calls."""
        start = time.time()
        
        # Parallel execution simulated by taking the maximum latency
        max_duration = max([self.tool_latencies.get(t, 0.05) for t in requested_tools], default=0.0)
        time.sleep(max_duration)
        
        results = {t: self.execute_tool(t, {}) for t in requested_tools}
        total_time = round((time.time() - start) * 1000, 2)
        
        return {
            "tools_executed": requested_tools,
            "results": results,
            "simulated_parallel_duration_ms": total_time
        }

agent = HighSpeedAgentDispatcher()
run_report = agent.dispatch_parallel_plan(["get_balance", "get_credit_score", "check_kyc_status"])
print("Agent Execution Completed:", run_report)
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Parallel Tool Call Dispatcher & Latency Simulator", divider="rainbow")

st.markdown("""
Compare sequential versus parallel tool dispatching latencies and observe 
how concurrency slashes execution time across multi-tool agent workflows.
""")

wb_view = st.selectbox(
    "Select Workbench Mode:",
    ["1. Parallel vs Sequential Tool Execution Benchmarker", "2. Observation Boundary Compressor Sandbox", "3. Subagent Router Visualizer"]
)

if wb_view == "1. Parallel vs Sequential Tool Execution Benchmarker":
    st.subheader("Parallel Tool Call Latency Calculator")
    selected_tools = st.multiselect(
        "Select Tools to Invoke in Turn 1:",
        list(agent_perf_engine.mock_tools.keys()),
        default=["fetch_user_profile", "fetch_order_history", "calculate_tax_rebate"]
    )
    is_parallel = st.toggle("Enable Parallel Tool Calling (asyncio.gather)", value=True)

    sim_res = agent_perf_engine.simulate_dispatch(is_parallel, selected_tools)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Execution Duration", f"{sim_res['total_latency_ms']}ms")
    with m2:
        st.metric("Tools Dispatched", f"{sim_res['tools_count']}")
    with m3:
        st.metric("Latency Reduction", f"{sim_res['savings_percent']}%")

    if is_parallel and sim_res["tools_count"] > 1:
        st.success(f"Parallel dispatch saved {sim_res['savings_percent']}% total latency by resolving tools concurrently!")
    else:
        st.info("Sequential mode executes each tool one-by-one, accumulating round-trip network delays.")

elif wb_view == "2. Observation Boundary Compressor Sandbox":
    st.subheader("Observation Payload Boundary Compressor")
    raw_payload_mock = """{
        "status": "success",
        "timestamp": 1729002931,
        "debug_trace": "db-shard-us-east-1a-prod-9921",
        "internal_flags": [true, false, false, true],
        "user_data": {
            "id": "usr_99128",
            "first_name": "Alexander",
            "last_name": "Hamilton",
            "tier": "enterprise_vip",
            "permissions": ["read_all", "write_invoices"]
        }
    }"""
    st.text_area("Raw Tool Return Payload (520 bytes):", raw_payload_mock, height=120)

    st.markdown("#### Compressed Observation for ToolMessage (58 bytes):")
    compressed_obs = """{"user_id": "usr_99128", "tier": "enterprise_vip"}"""
    st.code(compressed_obs, language="json")
    st.success("Compressed observation reduces prompt context consumption by 89% while retaining all essential decision facts.")

else:
    st.subheader("Supervisor Subagent Architecture Visualizer")
    st.markdown("Dividing 30 tools into specialized subagent domains:")
    st.info("""
    - **Supervisor**: Zero tools (Routes to Billing, Tech, or Compliance).
    - **Billing Agent**: 3 tools (`get_invoices`, `refund_charge`, `apply_coupon`).
    - **Tech Support Agent**: 4 tools (`check_status_page`, `restart_container`, `tail_logs`, `create_ticket`).
    - **Compliance Agent**: 2 tools (`verify_kyc`, `export_gdpr_data`).
    """)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How does parallel tool calling improve agent execution throughput in LangChain?"):
    st.markdown("""
    **Answer:**  
    Traditionally, an agent required N separate LLM reasoning turns to execute N tools sequentially. 
    With parallel tool calling, the LLM emits all N tool invocations in a single completion turn. 
    The runtime executes them concurrently via `asyncio.gather()`, reducing total tool latency from the sum 
    to the maximum individual tool delay, while cutting LLM reasoning passes from N to 1.
    """)

with st.expander("Q2: What is 'observation bloat', and how does boundary filtering mitigate it?"):
    st.markdown("""
    **Answer:**  
    Observation bloat occurs when raw API responses (containing verbose headers, debug fields, and nested metadata) 
    are converted directly into `ToolMessage` payloads. This inflates conversation context, slows down subsequent turns, 
    and distracts the model. Boundary filtering extracts only the essential fields needed for the agent's goal before 
    creating the `ToolMessage`.
    """)

with st.expander("Q3: Why should an agent's tool catalog be restricted to fewer than 10 tools per node?"):
    st.markdown("""
    **Answer:**  
    Large tool catalogs increase system prompt token volume, degrade tool selection accuracy, and increase the risk 
    of hallucinated arguments. Breaking a large catalog into specialized subagents coordinated by a supervisor router 
    ensures each node has high focus and low latency.
    """)

with st.expander("Q4: How do you prevent an autonomous agent from entering an infinite execution loop?"):
    st.markdown("""
    **Answer:**  
    1. Configure hard recursion limits (`recursion_limit` parameter in LangGraph or `max_iterations` in AgentExecutor).
    2. Implement tool call deduplication to detect when the agent repeats identical calls with identical arguments.
    3. Inject error guidance into `ToolMessage` indicating when repeated attempts will fail.
    """)

with st.expander("Q5: What is the benefit of pairing a distilled SLM with a frontier model in agentic workflows?"):
    st.markdown("""
    **Answer:**  
    An SLM (such as GPT-4o-mini or Claude 3.5 Haiku) handles early planning, tool selection, and tool parameter formatting 
    with ultra-low latency (< 150ms). Once all tool observations are gathered, the comprehensive state is passed to a frontier 
    model (GPT-4o or Sonnet 3.5) for final synthesis. This achieves frontier quality at a fraction of the cost and latency.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Executing Multiple Independent Tool Calls Sequentially**  
Running three 200ms database lookups one after another forces the user to wait 600ms plus 3 LLM reasoning passes. 
Always leverage parallel tool calls and `asyncio.gather()`.
""")

st.error("""
**Trap 2: Dumping Unfiltered 50KB JSON Payloads into Conversation History**  
Passing raw database dumps into `ToolMessage` consumes thousands of tokens and induces model confusion. 
Always filter payloads down to relevant fields.
""")

st.warning("""
**Trap 3: Giving a Single Agent Dozens of Overlapping Tools**  
Supplying 20+ tools with vague descriptions leads to hallucinated calls and non-deterministic routing. 
Partition tools across specialized subagents.
""")

st.warning("""
**Trap 4: Deploying Agents Without Maximum Recursion Limits**  
An unconstrained agent can loop dozens of times when encountering ambiguous API errors, exhausting API budgets 
within minutes. Always set a strict recursion limit.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Parallel Tool Calling</h4>
        <p>The capability of an LLM to emit multiple tool call requests in a single response turn for concurrent execution.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Boundary Observation Filter</h4>
        <p>A middleware layer sanitizing and compressing raw tool output before injecting it into agent state.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Supervisor Architecture</h4>
        <p>A multi-agent design pattern where a central router directs tasks to specialized subagents possessing targeted tool sets.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Recursion Guard</h4>
        <p>A hard execution threshold preventing autonomous agents from exceeding a designated number of decision turns.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to design lean tool schemas with minimal Pydantic descriptions to save prompt tokens.", value=True)
chk_2 = st.checkbox("I understand parallel tool calling mechanics and how to execute tools concurrently with asyncio.gather.", value=True)
chk_3 = st.checkbox("I can implement supervisor-worker architectures to prevent tool catalog bloat.", value=True)
chk_4 = st.checkbox("I know how to route tool selection to fast SLMs while reserving frontier models for final synthesis.", value=True)
chk_5 = st.checkbox("I can apply boundary observation filtering to eliminate 50KB JSON dumps from agent state.", value=True)
chk_6 = st.checkbox("I understand how to enforce recursion limits and short-circuit deduplicated tool invocations.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Agent Performance optimization competencies verified! Ready for Cost Optimization.")
