import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="17.3 Tool-Calling Workflows | GenAI Roadmap",
    page_icon="🔄",
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
    .workflow-pill {
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
st.title("🔄 17.3 Tool-Calling Workflows — Sequential, Parallel & Failure Recovery")
st.markdown(
    """
Welcome to **Lesson 17.3: Tool-Calling Workflows**.  
Invoking a single isolated function is straightforward. However, enterprise workflows frequently require complex choreography: **sequential dependencies** (where Tool B needs the output of Tool A), **parallel fan-out execution** (querying multiple APIs concurrently), and **resilient failure recovery**.

In this lesson, we explore how frontier models coordinate multi-tool execution graphs, manage parallel tool payloads in a single turn, and recover when individual microservices throw errors.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Multi-Tool Orchestration Spectrum
# ==============================================================================
st.header("1. The Multi-Tool Orchestration Spectrum", divider="rainbow")
st.markdown("Comparing sequential, parallel, and branched tool orchestration topologies:")

topologies_df = pd.DataFrame({
    "Orchestration Pattern": [
        "1. Single Shot Tool Call",
        "2. Sequential Chained Tools",
        "3. Parallel Multi-Tool Calls",
        "4. Branching Conditional Tools",
        "5. Failure Recovery & Retry"
    ],
    "Execution Topology": [
        "User Query -> LLM -> Tool 1 -> LLM -> Final Response",
        "User Query -> LLM -> Tool 1 -> Tool 2 -> Tool 3 -> Final Response",
        "User Query -> LLM -> [Tool A, Tool B, Tool C in parallel] -> Final Response",
        "User Query -> LLM -> Tool 1 -> (If status == OK -> Tool 2A else Tool 2B)",
        "User Query -> LLM -> Tool 1 (Fails) -> LLM Diagnoses Error -> Tool 1 Retry"
    ],
    "Latency & Performance Profile": [
        "Baseline (1 LLM turn + 1 Tool run + 1 LLM turn)",
        "High latency (N sequential round-trips to LLM API)",
        "Optimal latency (1 LLM turn + 3 concurrent API runs + 1 LLM turn)",
        "Dynamic routing saves unnecessary downstream API executions",
        "Resilient; 99.9% completion rate despite transient microservice glitches"
    ],
    "Enterprise Example": [
        "Look up single order tracking status",
        "Authenticate user -> Fetch bank balance -> Initiate wire transfer",
        "Compare flight prices across United, Delta, and American Airlines simultaneously",
        "Triage support ticket: if billing issue, query Stripe; if tech bug, query Jira",
        "Database query syntax error returned; LLM repairs SQL query and re-runs"
    ]
})
st.dataframe(topologies_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Parallel Tool Calling Mechanics
# ==============================================================================
st.header("2. Parallel Tool Calling: Concurrency in a Single Turn", divider="rainbow")

col_pt1, col_pt2 = st.columns(2)

with col_pt1:
    with st.container(border=True):
        st.subheader("⚡ How Parallel Tool Calling Works")
        st.markdown(
            """
- In modern models (GPT-4o, Claude 3.5), the model can emit **multiple tool calls in a single response message**.
- If a user asks: *"What is the weather in London, Paris, and Berlin?"*, the model returns an array containing 3 distinct `tool_calls` with unique IDs:
  - `call_1`: `get_weather(city="London")`
  - `call_2`: `get_weather(city="Paris")`
  - `call_3`: `get_weather(city="Berlin")`
- The client application executes all 3 API calls concurrently using `asyncio.gather()`, cutting execution time by 66%.
"""
        )

with col_pt2:
    with st.container(border=True):
        st.subheader("⚠️ Strict Message Protocol Rules")
        st.markdown(
            """
- If the model emits $K$ tool calls, the client MUST respond with **exactly $K$ tool messages**, each matching the corresponding `tool_call_id`.
- If even one tool response is omitted or has a mismatched ID, the LLM API will return a 400 Bad Request error.
- If Tool 2 fails with an exception, you must return an error payload for Tool 2 (`{"status": "error", "message": "timeout"}`) rather than dropping the message turn.
"""
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Parallel Tool Dispatch & Merging", divider="rainbow")
st.markdown("Visualizing concurrent tool execution and synthesis:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        user [label="User Multi-Action Query", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        llm [label="LLM Emits Multiple\nTool Call Array", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        t1 [label="Tool Call 1\n(Async Runner 1)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        t2 [label="Tool Call 2\n(Async Runner 2)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        t3 [label="Tool Call 3\n(Async Runner 3)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        merger [label="Async Join Barrier\n(asyncio.gather)", fillcolor="#EAB308", fontcolor="#000000"]
        final_gen [label="Final Unified Response", fillcolor="#10B981", fontcolor="#FFFFFF"]

        user -> llm
        llm -> t1
        llm -> t2
        llm -> t3
        t1 -> merger
        t2 -> merger
        t3 -> merger
        merger -> llm [label=" All K Tool Messages"]
        llm -> final_gen
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Tool Workflow Simulator
# ==============================================================================
st.header("4. Interactive Tool Workflow Simulator", divider="rainbow")
st.markdown("Observe the latency difference between sequential vs. parallel multi-tool dispatch:")

execution_mode = st.radio(
    "Select Execution Paradigm",
    ["Parallel Dispatch (asyncio.gather)", "Sequential Execution (One-by-One Loop)"],
    horizontal=True
)

simulate_failure = st.checkbox("Simulate Network Timeout on Tool 2 (Test Failure Recovery)", value=True)

col_ws1, col_ws2 = st.columns([1, 1])

with col_ws1:
    st.subheader("⏱️ Execution Latency Timeline")
    if execution_mode == "Parallel Dispatch (asyncio.gather)":
        timeline_df = pd.DataFrame({
            "Stage": ["1. LLM Planning", "2. Concurrent API Calls", "3. LLM Final Synthesis", "Total Pipeline"],
            "Duration (sec)": [0.45, 0.35, 0.50, 1.30]
        })
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        st.metric("Total Latency", "1.30s", delta="-50% vs Sequential", delta_color="inverse")
    else:
        timeline_df = pd.DataFrame({
            "Stage": ["1. LLM Turn 1", "2. Call Tool A", "3. LLM Turn 2", "4. Call Tool B", "5. LLM Turn 3", "6. Call Tool C", "7. Final LLM Turn", "Total Pipeline"],
            "Duration (sec)": [0.45, 0.35, 0.45, 0.35, 0.45, 0.35, 0.50, 2.90]
        })
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        st.metric("Total Latency", "2.90s", delta="+123% slower")

with col_ws2:
    st.subheader("📤 Handled Tool Output Array")
    sample_tools_return = [
        {"tool_call_id": "call_1", "name": "check_flight_united", "content": json.dumps({"status": "available", "price": 450})},
        {"tool_call_id": "call_2", "name": "check_flight_delta", "content": json.dumps({"status": "error", "message": "Gateway Timeout (504)"}) if simulate_failure else json.dumps({"status": "available", "price": 420})},
        {"tool_call_id": "call_3", "name": "check_flight_american", "content": json.dumps({"status": "available", "price": 480})}
    ]
    st.json(sample_tools_return)
    if simulate_failure:
        st.warning("🛡️ Graceful Failure Recovery: The pipeline did not crash. Tool 2 returned a 504 error, and the LLM synthesized results from United and American while noting Delta was temporarily unavailable.")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Asynchronous Parallel Dispatcher", divider="rainbow")
st.markdown("Python script orchestrating concurrent tool execution using `asyncio.gather()`:")

st.code(
    '''import asyncio
import json
from typing import List, Dict, Any

async def mock_tool_call(tool_name: str, args_dict: Dict[str, Any]) -> str:
    """Simulates async microservice call with network delay."""
    await asyncio.sleep(0.3)
    return json.dumps({"tool": tool_name, "status": "SUCCESS", "result": f"Data for {args_dict}"})

async def execute_parallel_tools(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Executes multiple tool calls concurrently and maps results back
    to exact tool message payloads conforming to the API specification.
    """
    async def run_single(call):
        call_id = call["id"]
        fn_name = call["function"]["name"]
        fn_args = json.loads(call["function"]["arguments"])
        
        try:
            raw_output = await mock_tool_call(fn_name, fn_args)
        except Exception as exc:
            raw_output = json.dumps({"error": str(exc)})
            
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": fn_name,
            "content": raw_output
        }

    # Execute all tools in parallel using asyncio.gather
    tool_messages = await asyncio.gather(*(run_single(call) for call in tool_calls))
    return list(tool_messages)

# Example demonstration
sample_calls = [
    {"id": "c1", "function": {"name": "fetch_metric", "arguments": '{"metric": "cpu"}'}},
    {"id": "c2", "function": {"name": "fetch_metric", "arguments": '{"metric": "memory"}'}}
]

# Run the async loop
results = asyncio.run(execute_parallel_tools(sample_calls))
print(f"Executed {len(results)} tools concurrently. All mapped with correct tool_call_id.")
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
    with st.expander("Q1: How do you handle circular dependency deadlocks in sequential tool calling?", expanded=True):
        st.markdown(
            """
In complex agent workflows, a model might get trapped in an infinite loop: calling Tool A, which returns an error, prompting it to call Tool B, which directs it back to Tool A.
**Production Mitigations**:
1. **Strict Recursion / Step Budget**: Enforce a maximum step ceiling (e.g. `max_tool_steps = 5`). If reached, abort execution and escalate to a fallback.
2. **State & Call De-duplication**: Track a history of executed `(tool_name, argument_hash)` tuples. If the exact same call is repeated consecutively, intercept the loop and inject a system warning.
"""
        )
    with st.expander("Q2: Why must tool execution messages preserve the exact `tool_call_id` emitted by the model?"):
        st.markdown(
            """
In multi-tool and parallel calling environments, the LLM engine uses the unique `tool_call_id` (e.g. `call_abc123`) to bind the returned output directly to the internal attention branch that generated the request.
If IDs are mismatched, omitted, or reordered incorrectly, the API engine cannot reconstruct the multi-modal state machine, resulting in an unrecoverable protocol error.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we drop tool calls that fail and only return the successful ones?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, if Tool 2 fails with a 500 error, just filter it out so the model only sees good data."*
- **Winning Response**: *"Never. The API specification requires an exact 1-to-1 correspondence between emitted tool call IDs and returned tool messages. Dropping a tool message causes the entire API call to reject with an HTTP 400 error. Instead, format the failure as a valid JSON error payload (`{'status': 'FAILED', 'reason': 'Gateway 504'}`) under the correct `tool_call_id`. This allows the LLM to understand what failed and formulate a graceful fallback answer."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Parallel Tool Calling**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The capability of an LLM to generate an array of multiple tool invocations in a single response turn for concurrent execution.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Tool Call Binding**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The protocol requirement that every tool response message include the exact matching `tool_call_id` emitted by the LLM.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Step Ceiling Guard**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** A hard limit capping the maximum number of sequential tool execution turns to prevent infinite looping and runaway billing.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 17.3 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Parallelism Cuts Latency**: Use parallel tool calls and `asyncio.gather()` to execute concurrent tasks in 1 turn.
2. **1-to-1 Protocol Rule**: Every emitted `tool_call_id` must have an exact matching tool message response.
3. **Graceful Error Encapsulation**: Wrap exceptions in JSON error payloads rather than dropping messages.
4. **Step Budget**: Always enforce a recursion ceiling to prevent infinite tool loops.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know how to execute parallel tool calls concurrently using asyncio.", value=True, key="chk_17_3_1")
    st.checkbox("I understand why dropping failed tool messages causes HTTP 400 protocol crashes.", value=True, key="chk_17_3_2")
    st.checkbox("I can implement a step ceiling to prevent recursive tool calling loops.", value=True, key="chk_17_3_3")
    st.checkbox("I know how to handle partial failure in multi-tool execution pipelines.", value=True, key="chk_17_3_4")
