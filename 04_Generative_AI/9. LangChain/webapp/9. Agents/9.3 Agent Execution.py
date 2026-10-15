import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="9.3 Agent Execution - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("9.3 Agent Execution: Execution Loops, Concurrency, Termination & Resilience")
st.caption("Chapter 9: Agents | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
An agent's power emerges during runtime execution. While a chain executes a predictable linear sequence, 
an agent's execution is an **adaptive graph traversal**: responding to real-time observations, 
issuing parallel tool calls, recovering from transient API faults, and autonomously evaluating stopping conditions.

In this lesson, you will master the 8 runtime execution mechanics:
1. **The Model ➔ Tool ➔ Model Loop**: The core cyclic transition powering autonomous agents.
2. **Multiple Tool Calls**: Managing complex multi-turn problem decompositions.
3. **Sequential Tool Calls**: Orchestrating dependent multi-stage pipelines where output A feeds input B.
4. **Parallel Tool Calls**: Fanning out independent tool queries concurrently to minimize latency.
5. **Agent Termination**: Detecting goal completion and enforcing hard circuit breakers.
6. **Error Handling**: Intercepting execution exceptions and prompting the model to self-heal.
7. **Retries**: Incorporating exponential backoff to absorb transient service degradation.
8. **Dynamic Tool Selection**: Pruning and injecting tools at runtime based on context and token limits.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockExecutionEvent:
    """Represents a discrete event in an agent execution trace in module scope."""
    def __init__(self, step_idx: int, event_type: str, actor: str, payload_summary: str, duration_ms: int):
        self.step_idx = step_idx
        self.event_type = event_type
        self.actor = actor
        self.payload_summary = payload_summary
        self.duration_ms = duration_ms

class MockAgentExecutionSimulator:
    """Simulates multi-turn agent execution with sequential/parallel tool calls in module scope."""
    def __init__(self):
        pass

    def execute_pipeline(
        self,
        query: str,
        execution_style: str,
        simulate_error: bool,
        max_turns: int
    ) -> Dict[str, Any]:
        events: List[MockExecutionEvent] = []
        total_time = 0

        # Step 1: Model Initial Reasoning Turn
        t1 = 410
        total_time += t1
        events.append(MockExecutionEvent(
            step_idx=1,
            event_type="MODEL_INFERENCE",
            actor="Brain (GPT-4o)",
            payload_summary=f"Analyzed query '{query[:45]}...'. Decided tools required.",
            duration_ms=t1
        ))

        # Step 2: Tool Execution
        if execution_style == "Parallel Tool Calls (Concurrent)":
            p_time = 185
            total_time += p_time
            if simulate_error:
                events.append(MockExecutionEvent(
                    step_idx=2,
                    event_type="PARALLEL_TOOL_DISPATCH",
                    actor="ToolNode",
                    payload_summary="Executed 2 tools in parallel: Tool A (SUCCESS), Tool B (HTTP 503 Provider Error - Handled).",
                    duration_ms=p_time
                ))
            else:
                events.append(MockExecutionEvent(
                    step_idx=2,
                    event_type="PARALLEL_TOOL_DISPATCH",
                    actor="ToolNode",
                    payload_summary="Executed 2 tools in parallel: query_cluster_metrics (140ms), query_user_audit (185ms). Latency bounded by max.",
                    duration_ms=p_time
                ))
        else:  # Sequential Tool Calls
            s1_time = 150
            s2_time = 160
            total_time += (s1_time + s2_time)
            events.append(MockExecutionEvent(
                step_idx=2,
                event_type="TOOL_EXECUTION_STEP_1",
                actor="ToolNode",
                payload_summary="Executed step 1: query_cluster_metrics() ➔ Yielded cluster CPU utilization (91.4%).",
                duration_ms=s1_time
            ))
            events.append(MockExecutionEvent(
                step_idx=3,
                event_type="TOOL_EXECUTION_STEP_2",
                actor="ToolNode",
                payload_summary="Executed step 2: query_user_audit(dependent on step 1) ➔ Verified active user sessions (4,820).",
                duration_ms=s2_time
            ))

        # Step 3: Final Synthesis Turn
        t_final = 380
        total_time += t_final
        events.append(MockExecutionEvent(
            step_idx=len(events) + 1,
            event_type="TERMINATION_SYNTHESIS",
            actor="Brain (GPT-4o)",
            payload_summary="Synthesized observations into grounded final answer. Zero tool_calls emitted ➔ TERMINATED.",
            duration_ms=t_final
        ))

        return {
            "total_events": len(events),
            "total_latency_ms": total_time,
            "events": events,
            "final_answer": "Final Resolution: Analyzed cluster telemetry and active tenant traffic. Confirmed high CPU utilization (91.4%) driven by peak user load (4,820 sessions). Recommended horizontal autoscaling."
        }

# --- SECTION 1: ARCHITECTURE BLUEPRINT & EXECUTION LIFECYCLE ---
st.header("1. Architecture Blueprint: The Multi-Turn Agent Execution Lifecycle", divider="rainbow")

st.markdown("""
The diagram below details the complete execution lifecycle of a production agent. 
The agent evaluates inputs, dispatches single or parallel tool calls, ingests observations, 
and iterates through its ReAct loop until reaching a stopping condition.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Level 1: Intake & State Initialization
        subgraph cluster_init {
            label = "1. INTAKE &amp; STATE INITIALIZATION";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            UserRequest [label="HumanMessage\\n'Diagnose API gateway latency spike'"];
            StateQueue [label="MessagesState History\\n[SystemMessage, HumanMessage]"];
        }

        // Level 2: Model Turn (Reasoning Node)
        subgraph cluster_model_node {
            label = "2. MODEL REASONING &amp; DISPATCH NODE";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            ModelInference [label="LLM Autoregressive Turn\\nModel inspects messages and bound tools", color="#10B981"];
            BranchDecider [label="Conditional Routing Edge\\nInspects: len(response.tool_calls) > 0 ?"];
        }

        // Level 3: Tool Execution Node
        subgraph cluster_tool_node {
            label = "3. TOOL NODE (CONCURRENT / SEQUENTIAL)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            ToolRouter [label="Tool Dispatch Strategy\\nSequential vs Parallel (asyncio.gather)"];
            ToolA [label="Tool A: query_metrics()", fillcolor="#0F172A"];
            ToolB [label="Tool B: search_logs()", fillcolor="#0F172A"];
            ObservationAggregator [label="ToolMessage Packaging\\nAttaches matching tool_call_id", color="#38BDF8"];
        }

        // Level 4: Resolution & Termination
        subgraph cluster_termination {
            label = "4. TERMINATION &amp; CIRCUIT BREAKER TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            CircuitBreaker [label="Iteration Guard\\nturns &lt; max_iterations (default: 10)?"];
            FinalAnswer [label="Final Grounded Answer\\nAIMessage(content='...', tool_calls=[])", color="#10B981"];
            TimeoutAbort [label="Aborted with Error\\nMax iterations exceeded", color="#EF4444"];
        }

        // Wiring
        UserRequest -> StateQueue;
        StateQueue -> ModelInference;
        ModelInference -> BranchDecider;

        BranchDecider -> ToolRouter [label="tool_calls present"];
        ToolRouter -> ToolA [label="Parallel / Sequential"];
        ToolRouter -> ToolB;
        ToolA -> ObservationAggregator;
        ToolB -> ObservationAggregator;

        ObservationAggregator -> CircuitBreaker [label="Append ToolMessages"];
        CircuitBreaker -> ModelInference [label="turns &lt; max (Loop back)", color="#38BDF8", penwidth=2.0];
        CircuitBreaker -> TimeoutAbort [label="turns &gt;= max", color="#EF4444", style=dashed];

        BranchDecider -> FinalAnswer [label="tool_calls == [] (Goal Reached)", color="#10B981", penwidth=2.0];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Core Runtime Invariants:**
1. **The Model Controls Branching**: Code does not decide when to call tools; the model decides dynamically by emitting `tool_calls`.
2. **Deterministic Termination**: The loop only terminates when the model produces pure text without tool calls, or when the iteration limit circuit breaker triggers.
""")

st.divider()

# --- SECTION 2: THE MODEL -> TOOL -> MODEL LOOP ---
st.header("2. The Model ➔ Tool ➔ Model Execution Loop", divider="rainbow")

st.markdown("""
The heart of an agent's runtime is the **Model ➔ Tool ➔ Model** loop:
1. **Model Invocation (Turn 1)**: The model processes user input and emits an `AIMessage` containing `tool_calls`.
2. **Tool Execution**: The host runtime intercepts the calls, runs the functions, and creates matching `ToolMessage` objects.
3. **Model Invocation (Turn 2)**: The model ingests the updated conversation history (including the tool observations) and synthesizes a final response.
""")

st.code('''# Manual implementation of the Model -> Tool -> Model loop:
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

@tool
def get_service_latency(service: str) -> str:
    """Check p99 latency for a service."""
    return f"{service} p99 latency: 42ms (NORMAL)"

tools = [get_service_latency]
tool_map = {t.name: t for t in tools}
model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

# Turn 1: Model decides action
messages = [HumanMessage(content="Check latency for billing-service.")]
ai_msg = model.invoke(messages)
messages.append(ai_msg)

# Tool Step: Execute tool and package observation
for call in ai_msg.tool_calls:
    result = tool_map[call["name"]].invoke(call["args"])
    messages.append(ToolMessage(tool_call_id=call["id"], content=str(result)))

# Turn 2: Model grounds final answer on observation
final_msg = model.invoke(messages)
print(final_msg.content)
# "The billing-service p99 latency is currently 42ms, which is within normal parameters."
''', language="python")

st.divider()

# --- SECTION 3: MULTIPLE TOOL CALLS ---
st.header("3. Multiple Tool Calls (Problem Decomposition Across Turns)", divider="rainbow")

st.markdown("""
When solving complex objectives (such as investigating a production incident), an agent typically 
requires multiple distinct tool calls across several turns:
- **Turn 1**: Query error logs for error signatures.
- **Turn 2**: Use discovered error signature to query deployment history for recent git commits.
- **Turn 3**: Query rollback service to verify rollback safety.
- **Turn 4**: Deliver consolidated incident report.
""")

st.divider()

# --- SECTION 4: SEQUENTIAL TOOL CALLS ---
st.header("4. Sequential Tool Calls (Dependent Execution Chains)", divider="rainbow")

st.markdown("""
Tool calls are **sequential** when the parameters for Tool B depend on the observation returned by Tool A. 
For example:
`Get Customer ID by Email` ➔ returns `ID=4820` ➔ `Query Orders for Customer ID 4820`.

Sequential execution cannot be parallelized; the agent must pause after each step to allow the model 
to extract the necessary parameters from the prior observation.
""")

st.divider()

# --- SECTION 5: PARALLEL TOOL CALLS ---
st.header("5. Parallel Tool Calls (Concurrent Execution Fanning)", divider="rainbow")

st.markdown("""
When tool calls are independent (e.g. checking weather in 3 different cities or querying 2 separate databases), 
frontier models emit all tool calls in a single `AIMessage`. 
Executing these concurrently via `asyncio.gather` dramatically reduces execution latency.
""")

st.subheader("Sequential vs Parallel Latency Timeline")
st.markdown("""
- **Sequential**: `Total Time = t(DB1) + t(DB2) + t(DB3) = 150ms + 180ms + 120ms = 450ms`.
- **Parallel (`asyncio.gather`)**: `Total Time = max(150ms, 180ms, 120ms) = 180ms` (60% latency reduction).
""")

st.divider()

# --- SECTION 6: AGENT TERMINATION ---
st.header("6. Agent Termination (Stopping Conditions & Circuit Breakers)", divider="rainbow")

st.markdown("""
To prevent runaway token costs and infinite loops, enterprise agents must enforce multiple layers 
of stopping conditions:
1. **Natural Termination**: Model returns an `AIMessage` with zero `tool_calls`.
2. **Maximum Iterations (`max_iterations`)**: Hard limit on the number of reasoning turns (typically 5 to 10).
3. **Execution Timeout**: Wall-clock timeout (e.g. 30 seconds) aborting the entire request.
4. **Token Budget Caps**: Aborting if total accumulated tokens exceed a safety threshold.
""")

st.divider()

# --- SECTION 7: ERROR HANDLING ---
st.header("7. Error Handling (Graceful Degradation & Self-Healing)", divider="rainbow")

st.markdown("""
When a tool fails (e.g., throwing a `DatabaseTimeoutException` or `ValidationError`), crashing the entire agent 
destroys user experience. 

**Self-Healing Pattern**: Intercept the exception and format it into a `ToolMessage`:
`ToolMessage(content="Error: Database connection timed out after 5.0s. Please retry with a smaller date range.")`

The LLM reads this observation, understands the constraint, and adjusts its parameters in the next turn!
""")

st.divider()

# --- SECTION 8: RETRIES IN AGENT LOOPS ---
st.header("8. Retries in Agent Loops (Exponential Backoff)", divider="rainbow")

st.markdown("""
For transient infrastructure errors (HTTP 429 rate limits, socket disconnects), individual tools 
should be wrapped with `.with_retry(stop_after_attempt=3)` to absorb temporary blips before the agent 
even considers failing over.
""")

st.code('''from langchain_core.tools import tool

@tool
def fetch_telemetry_with_retry(cluster: str) -> str:
    """Fetch metrics with automated exponential backoff."""
    return f"Cluster '{cluster}': Status=HEALTHY"

# Wrap tool with 3 retry attempts before raising exception
resilient_tool = fetch_telemetry_with_retry.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)
''', language="python")

st.divider()

# --- SECTION 9: DYNAMIC TOOL SELECTION ---
st.header("9. Dynamic Tool Selection (Runtime Filtering & Relevance Pruning)", divider="rainbow")

st.markdown("""
Binding 50 tools to an agent causes two major problems:
1. **Token Bloat**: Each tool schema consumes 50 to 150 prompt tokens on every single turn.
2. **Selection Degradation**: Models become confused when choosing among dozens of overlapping tools.

**Dynamic Tool Selection**: Use an initial lightweight semantic router or vector store to retrieve only the 
top 3–5 most relevant tools for the specific user query, binding only those tools to the agent at runtime!
""")

st.code('''def select_tools_dynamically(query: str, all_tools: list) -> list:
    """Filters tools based on query intent to minimize prompt tokens."""
    query_lower = query.lower()
    if any(w in query_lower for w in ["database", "sql", "table", "revenue"]):
        return [t for t in all_tools if "db" in t.name or "sql" in t.name]
    elif any(w in query_lower for w in ["network", "dns", "ssl", "ping"]):
        return [t for t in all_tools if "network" in t.name or "dns" in t.name]
    return all_tools[:3]  # Default general tools
''', language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Enterprise Fault-Tolerant Agent Execution Runtime", divider="rainbow")

st.markdown("""
The following complete production pattern illustrates a resilient agent execution runtime 
featuring parallel tool execution, exception containment, and hard loop circuit breakers.
""")

st.code('''import asyncio
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

logger = logging.getLogger("EnterpriseAgentRuntime")

@tool
def check_node_health(node_id: str) -> str:
    """Check CPU, memory, and disk health of a compute node."""
    return f"Node {node_id}: CPU=88%, Memory=72%, Disk=45% (ELEVATED CPU)"

@tool
def restart_node_daemon(node_id: str) -> str:
    """Restart background monitoring daemon on a node."""
    return f"Daemon restarted on node {node_id}. CPU normalized to 24%."

class ResilientAgentRuntime:
    """Production agent runtime with concurrent tool dispatch and self-healing error handling."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [check_node_health, restart_node_daemon]
        self.tool_map = {t.name: t for t in self.tools}
        self.bound_model = self.model.bind_tools(self.tools)

    async def _execute_single_tool(self, call: Dict[str, Any]) -> ToolMessage:
        call_id = call["id"]
        name = call["name"]
        args = call["args"]

        if name not in self.tool_map:
            return ToolMessage(tool_call_id=call_id, content=f"Error: Unknown tool '{name}'.")

        try:
            tool_obj = self.tool_map[name]
            result = await tool_obj.ainvoke(args)
            return ToolMessage(tool_call_id=call_id, content=str(result))
        except Exception as exc:
            # Self-healing observation passed back to LLM
            logger.warning(f"Tool '{name}' failed: {exc}")
            return ToolMessage(
                tool_call_id=call_id,
                content=f"ToolExecutionError in '{name}': {str(exc)}. Please adjust arguments."
            )

    async def execute(self, user_goal: str) -> str:
        messages: List[BaseMessage] = [HumanMessage(content=user_goal)]
        
        for turn in range(1, self.max_turns + 1):
            logger.info(f"Agent Turn {turn}/{self.max_turns}")
            ai_msg: AIMessage = await self.bound_model.ainvoke(messages)
            messages.append(ai_msg)

            # Stopping Condition: Model emits no tool calls
            if not ai_msg.tool_calls:
                logger.info("Agent reached terminal goal condition.")
                return ai_msg.content

            # Dispatch all requested tools concurrently in parallel
            observations = await asyncio.gather(*[
                self._execute_single_tool(call) for call in ai_msg.tool_calls
            ])
            messages.extend(observations)

        logger.error("Agent reached maximum turn limit without resolving goal.")
        return "Agent execution terminated: Maximum turn limit reached without resolution."
''', language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Agent Execution & Dynamic Tool Routing Simulator", divider="rainbow")

st.markdown("""
Test how an agent executes multi-turn workloads. Compare **Parallel Concurrency** against **Sequential Dispatch**, 
and observe how the agent handles simulated tool errors during execution.
""")

exec_sim = MockAgentExecutionSimulator()

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("🎯 Configure Workload Scenario")
    workload_query = st.text_area(
        "User Objective / Incident Report",
        value="Investigate sudden latency spike on api-gateway and check active tenant sessions.",
        height=90
    )
    
    execution_mode = st.radio(
        "Tool Execution Concurrency",
        ["Parallel Tool Calls (Concurrent)", "Sequential Tool Calls (Blocking)"]
    )
    
    error_toggle = st.checkbox("Simulate Upstream Tool Error (Tests Self-Healing)", value=False)
    turn_limit = st.slider("Circuit Breaker Max Turns", min_value=2, max_value=8, value=5)

with col_wb2:
    st.subheader("⚙️ Configured Execution Parameters")
    st.info(f"""
    **Active Strategy:**
    - **Concurrency**: `{execution_mode}`
    - **Max Turn Bound**: `{turn_limit}` turns
    - **Fault Injection**: `{'ACTIVE (HTTP 503)' if error_toggle else 'None (Clean)'}`
    """)

if st.button("⚡ Execute Agent Run", type="primary"):
    res = exec_sim.execute_pipeline(workload_query, execution_mode, error_toggle, turn_limit)
    
    st.subheader("📊 Execution Spans & Timeline")
    m1, m2 = st.columns(2)
    m1.metric("Execution Events", res["total_events"])
    m2.metric("Total Elapsed Latency", f"{res['total_latency_ms']} ms")

    st.write("#### ⏱️ Trace Event Chronology:")
    for evt in res["events"]:
        with st.expander(f"Step {evt.step_idx}: [{evt.event_type}] by {evt.actor} ({evt.duration_ms} ms)", expanded=True):
            st.write(evt.payload_summary)

    st.write("#### 🎯 Synthesized Final Output:")
    st.success(res["final_answer"])

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does parallel tool calling reduce overall agent execution latency?",
     "When an agent needs multiple independent observations (e.g. searching 3 database shards simultaneously), modern models emit multiple tool calls in a single turn. Executing them concurrently via asyncio.gather bounds latency to max(t_i) rather than the sum of all individual execution times, cutting wall-clock wait times by 50% to 70%."),
    ("What is the difference between a hard termination condition and a natural termination condition?",
     "A natural termination condition occurs when the LLM decides it has gathered sufficient evidence to fulfill the goal and emits an AIMessage with content and zero tool_calls. A hard termination condition is a developer-enforced circuit breaker (max_iterations, wall-clock timeout) that aborts the loop if the model gets trapped in an unproductive cycle."),
    ("Why should tool exceptions be returned to the model inside ToolMessages rather than crashing the Python process?",
     "Returning the error message in the ToolMessage allows the model to perceive the failure in its conversation history. Frontier models can read error messages (e.g. 'SyntaxError in SQL query' or 'Rate limit exceeded'), diagnose the mistake, and self-correct by generating an alternative query on the next turn."),
    ("What causes tool hallucination in long-running agent loops?",
     "As conversation history grows across 10+ turns, the prompt context becomes crowded with dozens of prior observations. This attention dilution can cause models to hallucinate non-existent tool names or forget the original user goal. Mitigate this by trimming messages and dynamically pruning tool catalogs."),
    ("How does dynamic tool selection optimize both cost and selection accuracy?",
     "Instead of binding all 50 available tools to the model on every request (which consumes thousands of prompt tokens and degrades selection accuracy), dynamic tool selection retrieves only the 3 to 5 most relevant tools based on query semantic embeddings, significantly improving accuracy and reducing token costs.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Deadlocks in Nested Event Loops. Running async agent tool execution via asyncio.run() inside an existing Tornado/FastAPI async event loop raises RuntimeError: This event loop is already running. Always use native async await throughout.")
    st.error("**Trap 2 (Breaking Error)**: Swallowing Exceptions Silently in Tools. If a tool catches an exception and returns an empty string '', the model receives zero context about why the tool failed and will repeatedly retry the exact same call. Always return descriptive error text.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Unbounded ReAct Loops on Ambiguous Goals. An ambiguous goal like 'Find interesting anomalies' can cause an agent to explore logs endlessly until reaching the max_iterations cap. Provide clear stopping criteria in system prompts.")
    st.warning("**Trap 4 (Latency Warning)**: Sequential Execution of Independent Queries. Executing independent tools in a synchronous for-loop doubles or triples latency. Always use asyncio.gather for parallel tool calls.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Natural Stopping Condition"):
        st.markdown("**AIMessage with non-empty content and tool_calls=[] terminates the loop.**")
    with st.expander("🎴 Flashcard 2: Self-Healing Pattern"):
        st.markdown("**Pass tool exception strings back in ToolMessage to let the LLM self-correct.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Parallel Tool Latency Bound"):
        st.markdown("**Latency is bounded by max(t_i) rather than the sum of all tool runtimes.**")
    with st.expander("🎴 Flashcard 4: Dynamic Tool Pruning"):
        st.markdown("**Filter tool catalogs dynamically to reduce prompt token bloat and confusion.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the Model -> Tool -> Model execution loop.", value=True)
chk2 = st.checkbox("I know when to use sequential vs parallel tool execution.", value=True)
chk3 = st.checkbox("I can implement parallel concurrency using asyncio.gather().", value=True)
chk4 = st.checkbox("I understand stopping conditions and circuit breaker safeguards.", value=True)
chk5 = st.checkbox("I know how to implement self-healing error handling for failing tools.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 agent execution interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 9.3 Agent Execution. Ready for 9.4 Advanced Agents!")
