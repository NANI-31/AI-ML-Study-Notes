import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="24.4 Testing - LangGraph Curriculum",
    page_icon="🔍",
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
    .badge-test {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE TEST RUNNER & WORKBENCH SIMULATOR
# ==============================================================================
class GraphTestSuiteSimulator:
    """
    Simulates multi-tier test execution for LangGraph agents:
    - Node unit tests
    - Conditional routing unit tests
    - Tool execution tests
    - State transition tests
    - End-to-end integration runs
    """
    def __init__(self):
        self.test_cases = [
            {
                "test_name": "test_planner_node_splits_subtasks",
                "tier": "Unit (Node)",
                "target": "planner_node",
                "inputs": {"query": "Analyze financial report and generate summary"},
                "expected": {"plan_count": 2},
                "result": "PASSED",
                "duration_ms": 4.2
            },
            {
                "test_name": "test_routing_to_tools_when_tool_call_present",
                "tier": "Unit (Routing)",
                "target": "route_agent_decision",
                "inputs": {"tool_calls": [{"name": "calculator", "args": {"expr": "14 * 2"}}]},
                "expected": {"next_node": "tools"},
                "result": "PASSED",
                "duration_ms": 1.1
            },
            {
                "test_name": "test_calculator_tool_handles_zero_division",
                "tier": "Unit (Tool)",
                "target": "calculator_tool",
                "inputs": {"expr": "100 / 0"},
                "expected": {"error": "ZeroDivisionError handled gracefully"},
                "result": "PASSED",
                "duration_ms": 2.5
            },
            {
                "test_name": "test_state_reducer_merges_messages_deterministically",
                "tier": "Unit (State Transition)",
                "target": "add_messages reducer",
                "inputs": {"existing_len": 2, "new_messages": 1},
                "expected": {"total_len": 3},
                "result": "PASSED",
                "duration_ms": 0.8
            },
            {
                "test_name": "test_e2e_rejection_loop_terminates_at_max_retries",
                "tier": "Integration",
                "target": "StateGraph full flow",
                "inputs": {"query": "Generate malicious script", "mock_reject": True},
                "expected": {"final_status": "rejected", "loop_count": 3},
                "result": "PASSED",
                "duration_ms": 88.4
            },
            {
                "test_name": "test_golden_regression_qa_dataset",
                "tier": "Regression / Eval",
                "target": "LangSmith Dataset (50 items)",
                "inputs": {"dataset_version": "v1.4"},
                "expected": {"accuracy_threshold": 0.95},
                "result": "PASSED (Score: 0.97)",
                "duration_ms": 312.0
            }
        ]

    def run_suite(self, filter_tier: Optional[str] = None) -> List[Dict[str, Any]]:
        if not filter_tier or filter_tier == "All":
            return self.test_cases
        return [tc for tc in self.test_cases if tc["tier"].startswith(filter_tier)]

test_sim = GraphTestSuiteSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Testing Pyramid, Mocking Strategies, and Evaluation Harness", divider="rainbow")

st.markdown("""
Testing LangGraph applications requires a multi-layered testing pyramid. Because agents incorporate both 
deterministic Python logic (state reducers, conditional edges, tools) and non-deterministic foundation models, 
robust production verification requires isolating deterministic components via standard unit tests while 
validating agentic workflows using mock models, integration test harnesses, and golden evaluation datasets.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph TestingPyramidBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_tier1 {
            label="Tier 1: Fast Unit Tests (Milliseconds / Deterministic / CI-CD)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            t1_nodes [label="Node Functions\\nTest pure Python state inputs & outputs\\nZero LLM or network dependency", fillcolor="#0F172A"];
            t1_routing [label="Conditional Routing Logic\\nTest branch selection for all enum/edge states", fillcolor="#0F172A"];
            t1_reducers [label="Channel Reducers\\nVerify add_messages, dict merging, and set unions", fillcolor="#0F172A"];
            t1_tools [label="Tool Unit Tests\\nMock external APIs, verify schema parsing & exceptions", fillcolor="#0F172A"];
        }

        subgraph cluster_tier2 {
            label="Tier 2: Integration & Workflow Tests (Seconds / Mock LLMs / Checkpointers)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            t2_subgraphs [label="Compiled StateGraph Execution\\nRun with MemorySaver and FakeListChatModel", fillcolor="#1E293B"];
            t2_hitl [label="HITL & Interrupt Testing\\nVerify interrupt() pausing and Command(resume=...)", fillcolor="#1E293B"];
            t2_timetravel [label="State History & Forking Tests\\nVerify get_state_history and update_state branches", fillcolor="#1E293B"];
        }

        subgraph cluster_tier3 {
            label="Tier 3: Regression & Evaluation Datasets (Minutes / LLM-as-a-Judge)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            t3_eval [label="Golden Benchmark Datasets\\nLangSmith / Pytest-eval suites", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            t3_judge [label="LLM-as-a-Judge Scorers\\nCorrectness, Groundedness, Trajectory Match", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        t1_nodes -> t2_subgraphs;
        t1_routing -> t2_subgraphs;
        t1_reducers -> t2_subgraphs;
        t1_tools -> t2_subgraphs;

        t2_subgraphs -> t3_eval;
        t2_hitl -> t3_eval;
        t2_timetravel -> t3_eval;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Unit Testing Nodes
# ==============================================================================
st.header("2. Unit Testing Nodes: Isolating Pure Functions, State Reducers, and Node Invariants", divider="rainbow")

st.markdown("""
Nodes in LangGraph are standard Python functions that accept the current state dictionary and return a state update dictionary. 
Because nodes are decoupled from the graph harness, they can be tested directly with `pytest` without invoking `compile()` or instantiating checkpointers.

### Key Node Testing Practices:
- **Mock External Services**: If a node calls an external client, inject a mock or patch the dependency.
- **Test Edge Case Inputs**: Verify how nodes handle empty strings, missing optional fields, or malformed state.
- **Assert Return Invariants**: Confirm that the returned dictionary matches the expected delta schema without mutating inputs in-place.
""")

st.code('''import pytest
from my_agent.nodes import planner_node

def test_planner_node_generates_valid_subtasks():
    # Arrange: initial input state
    initial_state = {
        "query": "Generate monthly revenue report and send email",
        "plan": [],
        "step_count": 0
    }
    
    # Act: direct function execution
    state_delta = planner_node(initial_state)
    
    # Assert: state delta invariants
    assert "plan" in state_delta
    assert len(state_delta["plan"]) == 2
    assert state_delta["plan"][0] == "Generate monthly revenue report"
    assert state_delta["step_count"] == 1
''', language="python")

# ==============================================================================
# SECTION 3: Testing Graph Routing
# ==============================================================================
st.header("3. Testing Graph Routing: Validating Conditional Edges, Routing Decisions, and Branching Logic", divider="rainbow")

st.markdown("""
Conditional edges determine execution flow based on state variables or tool call presence. 
Testing routing logic in isolation guarantees that graphs transition to the correct nodes across all possible branches.

### Testing Routing Exhaustiveness:
1. **Tool Invocation Route**: If the state contains tool calls, router must return `"tools"`.
2. **Termination Route**: If model finishes with a final answer, router must return `END`.
3. **Escalation Route**: If safety thresholds are violated, router must route to `"human_review"`.
""")

st.code('''from my_agent.routing import route_next_step
from langgraph.graph import END

def test_route_next_step_detects_tool_calls():
    state_with_tools = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"name": "search", "args": {"q": "python"}}]}
        ]
    }
    assert route_next_step(state_with_tools) == "tools"

def test_route_next_step_detects_completion():
    state_finished = {
        "messages": [
            {"role": "assistant", "content": "Here is the final answer.", "tool_calls": []}
        ]
    }
    assert route_next_step(state_finished) == END
''', language="python")

# ==============================================================================
# SECTION 4: Testing Tools
# ==============================================================================
st.header("4. Testing Tools: Mocking External APIs, Schema Validation, and Error Fallbacks", divider="rainbow")

st.markdown("""
Tools are executable capabilities called by agents. Testing tools requires verifying input schema validation (via Pydantic), 
mocking external network services (via `unittest.mock`), and confirming graceful error handling.

### Tool Test Categories:
- **Happy Path**: Valid arguments execute successfully and return formatted strings.
- **Validation Errors**: Invalid argument types or missing required fields raise descriptive validation errors.
- **Downstream Failures**: Network timeouts or database errors return clean error messages that the LLM can recover from, 
  rather than crashing the entire process with an unhandled exception.
""")

st.code('''from unittest.mock import patch
from my_agent.tools import database_lookup_tool

def test_database_lookup_tool_success():
    with patch("my_agent.tools.db_client.query") as mock_query:
        mock_query.return_value = [{"id": 1, "name": "Enterprise Deal"}]
        
        result = database_lookup_tool.invoke({"account_id": "acc-101"})
        assert "Enterprise Deal" in result

def test_database_lookup_tool_handles_network_timeout():
    with patch("my_agent.tools.db_client.query", side_effect=TimeoutError("Connection timed out")):
        result = database_lookup_tool.invoke({"account_id": "acc-101"})
        # Verify tool returns a recoverable error message string to the model
        assert "Error: Unable to reach database" in result
''', language="python")

# ==============================================================================
# SECTION 5: Testing State Transitions
# ==============================================================================
st.header("5. Testing State Transitions: Verifying Superstep State Merging and Channel Invariants", divider="rainbow")

st.markdown("""
State transitions in LangGraph rely on channel reducers. A channel reducer specifies how updates from one or more nodes 
are merged into the state during superstep barriers.

### Reducer Invariants to Verify:
- **`add_messages`**: Preserves existing messages, appends new messages, and overwrites existing messages if their IDs match.
- **Custom Reducers**: Custom accumulation functions (such as list append, integer counter increments, or dictionary merges) 
  must behave deterministically and remain side-effect free.
""")

st.code('''from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

def test_add_messages_reducer_deduplication():
    msg1 = HumanMessage(content="Hello", id="msg-001")
    msg2 = AIMessage(content="Hi there!", id="msg-002")
    
    # Initial state
    current_state = [msg1, msg2]
    
    # Update with replacement for msg-002
    updated_msg2 = AIMessage(content="Updated response", id="msg-002")
    new_state = add_messages(current_state, [updated_msg2])
    
    assert len(new_state) == 2
    assert new_state[1].content == "Updated response"
''', language="python")

# ==============================================================================
# SECTION 6: Integration Testing
# ==============================================================================
st.header("6. Integration Testing: End-to-End Workflow Execution with Mock LLMs and Checkpointers", divider="rainbow")

st.markdown("""
Integration tests verify the compiled `StateGraph` end-to-end. Rather than invoking live LLM APIs (which introduce cost, 
latency, and non-determinism into continuous integration pipelines), LangGraph tests use `FakeListChatModel` 
or recorded response cassettes.
""")

st.code('''from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver
from my_agent.graph import create_agent_graph

def test_e2e_agent_workflow_with_fake_model():
    # Prepare deterministic sequential model responses
    fake_responses = [
        "I need to query the database first.",  # Turn 1
        "The customer is active and in good standing."  # Turn 2
    ]
    mock_llm = FakeListChatModel(responses=fake_responses)
    
    checkpointer = MemorySaver()
    app = create_agent_graph(llm=mock_llm, checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test-ci-thread-1"}}
    final_state = app.invoke({"query": "Check status of account 42"}, config)
    
    assert len(final_state["messages"]) >= 3
    assert "good standing" in final_state["messages"][-1].content
''', language="python")

# ==============================================================================
# SECTION 7: Regression Testing
# ==============================================================================
st.header("7. Regression Testing: Golden Dataset Baselines and Run Comparison Harnesses", divider="rainbow")

st.markdown("""
Regression testing prevents performance degradation when updating system prompts, tool schemas, or underlying model versions. 
A golden dataset contains verified input queries paired with reference outputs or trajectories.

### Regression Testing Strategy:
1. **Maintain Golden Baselines**: Store 50-200 canonical customer queries and verified outputs in version control.
2. **Execute Automated Benchmarks**: Run the graph against the golden dataset in staging before deploying prompt changes.
3. **Compare Trajectories**: Verify that the agent selects expected tools and reaches valid termination states without 
   unnecessary looping.
""")

# ==============================================================================
# SECTION 8: Evaluation Datasets
# ==============================================================================
st.header("8. Evaluation Datasets: Constructing Ground Truth Test Sets, LLM-as-a-Judge, and LangSmith Evaluators", divider="rainbow")

st.markdown("""
For open-ended conversational tasks where exact string matching is ineffective, LangGraph applications utilize 
**LLM-as-a-Judge** evaluators and LangSmith datasets.
""")

st.code('''from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Define automated evaluation criteria
def evaluate_trajectory(run, example):
    # Verify expected tools were called
    expected_tool = example.outputs.get("expected_tool")
    called_tools = [span.name for span in run.child_spans if span.run_type == "tool"]
    is_correct = expected_tool in called_tools
    return {"key": "tool_selection_accuracy", "score": int(is_correct)}

# Run automated evaluation against LangSmith dataset
results = evaluate(
    lambda inputs: app.invoke(inputs),
    data="enterprise-support-golden-dataset",
    evaluators=[evaluate_trajectory],
    experiment_prefix="prompt-v3-migration"
)
''', language="python")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-Tiered Pytest Suite for LangGraph Systems", divider="rainbow")

st.markdown("""
The following complete test module demonstrates production testing standards, incorporating unit tests for nodes, 
conditional edges, and full mock integration execution.
""")

st.code('''import pytest
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    query: str
    decision: str
    counter: int

# System Under Test (SUT) Nodes
def analyzer_node(state: AgentState) -> dict:
    q = state.get("query", "").lower()
    decision = "escalate" if "urgent" in q else "standard"
    return {"decision": decision, "counter": state.get("counter", 0) + 1}

def routing_edge(state: AgentState) -> str:
    return "human_agent" if state.get("decision") == "escalate" else "automated_agent"

# Pytest Test Suite
class TestAgentGraphProduction:
    
    def test_analyzer_node_standard(self):
        state = {"query": "Please check balance", "decision": "", "counter": 0}
        delta = analyzer_node(state)
        assert delta["decision"] == "standard"
        assert delta["counter"] == 1

    def test_analyzer_node_escalate(self):
        state = {"query": "URGENT: system outage", "decision": "", "counter": 0}
        delta = analyzer_node(state)
        assert delta["decision"] == "escalate"
        assert delta["counter"] == 1

    def test_routing_edge_logic(self):
        assert routing_edge({"decision": "escalate", "query": "", "counter": 1}) == "human_agent"
        assert routing_edge({"decision": "standard", "query": "", "counter": 1}) == "automated_agent"

    def test_compiled_graph_integration(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyzer", analyzer_node)
        workflow.add_node("human_agent", lambda s: {"decision": "escalated_resolved"})
        workflow.add_node("automated_agent", lambda s: {"decision": "auto_resolved"})
        
        workflow.add_edge(START, "analyzer")
        workflow.add_conditional_edges("analyzer", routing_edge)
        workflow.add_edge("human_agent", END)
        workflow.add_edge("automated_agent", END)
        
        app = workflow.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-t1"}}
        
        result = app.invoke({"query": "URGENT issue", "counter": 0}, config)
        assert result["decision"] == "escalated_resolved"
        assert result["counter"] == 1
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Test Runner, Mock LLM Injection, and Assertion Matrix", divider="rainbow")

st.markdown("""
Simulate test execution across unit, routing, tool, integration, and regression tiers.
""")

col_ctrl1, col_ctrl2 = st.columns([1, 2])

with col_ctrl1:
    st.subheader("Test Runner Controls")
    tier_filter = st.selectbox(
        "Filter by Test Tier",
        ["All", "Unit", "Integration", "Regression"]
    )
    run_button = st.button("Execute Test Suite", type="primary")

with col_ctrl2:
    st.subheader("Test Execution Suite Results")
    test_results = test_sim.run_suite(tier_filter)
    
    total_count = len(test_results)
    passed_count = sum(1 for tc in test_results if "PASSED" in tc["result"])
    total_time = sum(tc["duration_ms"] for tc in test_results)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Tests", total_count)
    with m2:
        st.metric("Passed Tests", f"{passed_count} / {total_count}")
    with m3:
        st.metric("Duration", f"{total_time:.1f} ms")
        
    df_tests = pd.DataFrame([
        {
            "Test Identifier": tc["test_name"],
            "Tier": tc["tier"],
            "Target Component": tc["target"],
            "Duration (ms)": tc["duration_ms"],
            "Status": tc["result"]
        }
        for tc in test_results
    ])
    st.dataframe(df_tests, use_container_width=True)

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: Why should you avoid calling live LLMs in standard CI/CD unit and integration test pipelines?"):
    st.markdown("""
    **Answer:**  
    Calling live LLMs introduces non-determinism (prompt drift, temperature variation), high latency, rate limits, 
    and monetary cost into automated CI/CD runs. Fast, deterministic feedback requires mocking LLMs via `FakeListChatModel` 
    or cached cassettes, reserving live models for scheduled evaluation benchmarks.
    """)

with st.expander("Q2: How do you unit test conditional routing edges without executing the entire graph?"):
    st.markdown("""
    **Answer:**  
    Conditional edge functions in LangGraph are pure Python callables that take the current state dictionary and return 
    a string corresponding to the destination node name or `END`. They can be tested directly with regular unit test assertions 
    by passing pre-constructed state dictionaries and asserting the returned string.
    """)

with st.expander("Q3: What is the recommended method to test human-in-the-loop interrupt workflows?"):
    st.markdown("""
    **Answer:**  
    Test HITL flows using `MemorySaver`. Invoke the graph, verify that `snapshot.next` pauses on the expected node with 
    the expected interrupt payload, then invoke the graph with `Command(resume=...)` to confirm execution continues to completion.
    """)

with st.expander("Q4: How do you test for infinite loops in cyclic agent architectures?"):
    st.markdown("""
    **Answer:**  
    Configure the graph with a low `recursion_limit` (e.g., `config={"recursion_limit": 5}`). Verify that when an agent 
    fails to terminate or receives rejected feedback, the graph raises a `GraphRecursionError` at the threshold rather than hanging indefinitely.
    """)

with st.expander("Q5: What is the role of LLM-as-a-Judge in regression testing?"):
    st.markdown("""
    **Answer:**  
    LLM-as-a-Judge evaluates qualitative outputs (such as conversational responses, summaries, or complex reasoning) against 
    reference criteria (e.g., groundedness, tone, factual accuracy) where exact string matching fails, providing quantifiable benchmark scores.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Mutating Input State in Node Unit Tests**  
If a node function mutates `state["key"]` directly rather than returning a state update delta dictionary, tests may pass 
in isolation but fail unpredictably inside LangGraph superstep barriers when channel reducers execute.
""")

st.error("""
**Trap 2: Brittle String Equality Assertions on Generative Outputs**  
Writing unit tests that assert exact string equality (e.g., `assert response == "Hello, how can I help?"`) causes tests to break 
frequently as models update. Use semantic assertions, schema validation, or structured tool call checking instead.
""")

st.warning("""
**Trap 3: Unmocked Network Calls in Tool Tests**  
Allowing tools to perform real HTTP or database requests in unit tests leads to flaky CI builds due to network timeouts, 
transient database downtime, or credential expiry.
""")

st.warning("""
**Trap 4: Neglecting State Reducer Conflict Testing**  
Testing nodes only with single-value outputs while ignoring concurrent writes can hide race conditions where two parallel 
nodes update the same unreduced channel simultaneously.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Testing Pyramid</h4>
        <p>A testing hierarchy prioritizing fast, deterministic node and edge unit tests at the base, followed by mock integration tests and golden evaluation datasets at the top.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>FakeListChatModel</h4>
        <p>A mock LLM class providing deterministic sequential text completions without network latency or financial cost.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>GraphRecursionError</h4>
        <p>The built-in exception raised when a graph execution exceeds its configured recursion_limit, preventing runaway execution loops.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>LLM-as-a-Judge</h4>
        <p>Automated evaluation technique using an advanced foundation model to grade agent responses against factual rubrics and trajectory benchmarks.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to unit test pure node functions by passing mock state dictionaries directly.", value=True)
chk_2 = st.checkbox("I understand how to test conditional edge routing logic across all branching outcomes.", value=True)
chk_3 = st.checkbox("I can test tools with mocked external dependencies and Pydantic schema validation.", value=True)
chk_4 = st.checkbox("I understand how to verify state transition channel reducers (e.g., add_messages).", value=True)
chk_5 = st.checkbox("I know how to run end-to-end integration tests using FakeListChatModel and MemorySaver.", value=True)
chk_6 = st.checkbox("I can construct regression evaluation datasets and implement LLM-as-a-Judge benchmark scorers.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LangGraph testing and evaluation mastery confirmed! Ready for Chapter 25 Production Engineering.")
