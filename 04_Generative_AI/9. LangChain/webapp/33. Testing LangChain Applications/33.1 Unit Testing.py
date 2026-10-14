import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="33.1 Unit Testing - LangChain Applications",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for test telemetry
st.markdown("""
<style>
    .test-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .status-pass {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .status-fail {
        background-color: #7F1D1D;
        color: #FECACA;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class UnitTestingHarnessSimulator:
    """
    Simulates pytest execution against LangChain components:
    prompts, custom tools, mock LLM chains, retrievers, and agent decision loops.
    """
    def __init__(self):
        self.test_cases = [
            {"test_name": "test_prompt_template_variable_binding", "target": "ChatPromptTemplate", "status": "PASSED", "duration_ms": 1.2, "detail": "All required variables formatted without KeyError"},
            {"test_name": "test_calculator_tool_schema_validation", "target": "@tool calculator", "status": "PASSED", "duration_ms": 2.4, "detail": "Pydantic validated inputs and rejected negative values"},
            {"test_name": "test_chain_lcel_runnable_with_fake_llm", "target": "RunnableSequence", "status": "PASSED", "duration_ms": 3.1, "detail": "FakeListChatModel returned deterministic mock tokens"},
            {"test_name": "test_retriever_empty_query_fallback", "target": "VectorRetriever", "status": "PASSED", "duration_ms": 2.8, "detail": "Empty query returned fallback empty Document list"},
            {"test_name": "test_agent_termination_on_max_iterations", "target": "AgentExecutor", "status": "PASSED", "duration_ms": 5.4, "detail": "Guard halted loop at iteration limit without runaway cost"}
        ]

    def run_selected_test(self, test_name: str, inject_failure: bool = False) -> Dict[str, Any]:
        matched = next((tc for tc in self.test_cases if tc["test_name"] == test_name), self.test_cases[0])
        res = dict(matched)
        if inject_failure:
            res["status"] = "FAILED"
            res["duration_ms"] = 0.8
            res["detail"] = "AssertionError: Expected deterministic mock output but received malformed string."
        return res


# --- LESSON HEADER ---
st.title("33.1 Unit Testing in LangChain Applications")
st.markdown("""
LLM applications introduce stochasticity and latency into continuous integration pipelines.
True unit tests must remain fast, deterministic, and free of live model API calls.
This lesson examines how to write robust unit tests for prompts, tools, chains, retrievers,
and agents using `pytest`, `FakeListChatModel`, and isolated mocks.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Deterministic Unit Testing for Non-Deterministic AI", divider="rainbow")
st.markdown("""
The foundation of reliable GenAI testing is strict unit isolation.
By replacing live OpenAI or Anthropic API calls with `FakeListChatModel` and mock vector retrievers,
test suites execute in milliseconds within continuous integration (CI) runners without incurring token costs or flaky failures.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph UnitTesting_LangChain_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_pytest_runner {
            label = "Pytest Test Suite (CI/CD Pipeline)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            PytestCLI [label="pytest -v --cov\\n(Isolated, zero network, zero API keys)", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_mocks_fixtures {
            label = "Test Fixtures & Deterministic Mocks";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            MockLLM [label="FakeListChatModel / MockChatModel\\n(Pre-configured responses: FakeListChatModel(responses=[...]))", fillcolor="#1E293B", color="#60A5FA"];
            MockRetriever [label="Mock Retriever Fixture\\n(Returns deterministic [Document(page_content='...')])", fillcolor="#1E293B", color="#60A5FA"];
            MockTools [label="Mock Tool Environment\\n(Mocks external DB / API calls with pytest-mock)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_unit_targets {
            label = "LangChain Unit Targets";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            PromptTest [label="Prompt Template Unit\\n(PromptTemplate / ChatPromptTemplate variable validation)", fillcolor="#0369A1", color="#38BDF8"];
            ToolTest [label="Custom Tool Unit\\n(Pydantic input schema validation & edge case handling)", fillcolor="#0369A1", color="#38BDF8"];
            ChainTest [label="LCEL Runnable Sequence\\n(prompt | mock_llm | parser pipeline)", fillcolor="#0369A1", color="#38BDF8"];
            AgentTest [label="Agent Decision Logic\\n(Step execution, tool selection, loop limit assertions)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_assertions {
            label = "Deterministic Assertions & CI Metrics";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            Assertions [label="Deterministic Assertions\\n(assert output == expected; assert tool_call.name == ...)", fillcolor="#065F46", color="#10B981"];
        }

        PytestCLI -> PromptTest [label="Run prompt tests"];
        PytestCLI -> ToolTest [label="Run tool tests"];
        PytestCLI -> ChainTest [label="Run chain tests"];
        PytestCLI -> AgentTest [label="Run agent tests"];

        MockLLM -> ChainTest [label="Inject fake responses"];
        MockRetriever -> ChainTest [label="Inject static documents"];
        MockTools -> AgentTest [label="Inject mocked tool responses"];

        PromptTest -> Assertions [label="Verify formatted text"];
        ToolTest -> Assertions [label="Verify parsed return values"];
        ChainTest -> Assertions [label="Verify structured output"];
        AgentTest -> Assertions [label="Verify tool invocation sequence"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: TESTING PROMPTS ---
st.header("2. Unit Testing Prompts: Formatting, Variable Interpolation, and Boundary Conditions", divider="rainbow")
st.markdown("""
Prompt templates form the interface contract with generative models. Unit testing prompts ensures that required variables
are populated, edge cases (such as empty strings or Unicode characters) do not trigger runtime formatting exceptions,
and system instructions adhere to structure standards.
""")

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.markdown("### What to Test in Prompts")
    st.markdown("""
    - **Variable Completeness**: Verify that invoking `.format()` or `.format_messages()` without required keys raises a `KeyError`.
    - **Whitespace Normalization**: Assert that multiline system instructions do not contain unintended leading indentation.
    - **Injection Boundaries**: Test how the template handles user input containing curly braces (`{{` or `}}`) without breaking f-string parsers.
    """)
with col_p2:
    st.code("""
import pytest
from langchain_core.prompts import ChatPromptTemplate

def test_chat_prompt_template_variable_binding():
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise assistant for {domain}."),
        ("human", "{query}")
    ])
    
    # 1. Test correct variable interpolation
    messages = template.format_messages(domain="Healthcare", query="Analyze lab results.")
    assert len(messages) == 2
    assert "Healthcare" in messages[0].content
    assert messages[1].content == "Analyze lab results."
    
    # 2. Test missing variable raises KeyError
    with pytest.raises(KeyError):
        template.format_messages(domain="Healthcare")
    """, language="python")

# --- SECTION 3: TESTING TOOLS ---
st.header("3. Unit Testing Custom Tools: Schema Validation, Argument Coercion, and Error Handling", divider="rainbow")
st.markdown("""
Tools are executable Python functions wrapped with `@tool` or `BaseTool`. Unit testing tools verifies that
the underlying Pydantic schema correctly validates input types, handles out-of-bounds parameters, and returns
safe string error messages rather than unhandled exceptions that crash agent loops.
""")

st.code("""
import pytest
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class CurrencyTransferInput(BaseModel):
    amount: float = Field(..., gt=0, description="Transfer amount in USD")
    recipient: str = Field(..., min_length=3)

@tool(args_schema=CurrencyTransferInput)
def transfer_currency(amount: float, recipient: str) -> str:
    \"\"\"Transfers currency to a verified recipient.\"\"\"
    return f"Successfully queued ${amount} transfer to {recipient}."

def test_transfer_tool_valid_input():
    result = transfer_currency.invoke({"amount": 150.0, "recipient": "vendor_alpha"})
    assert "Successfully queued $150.0" in result

def test_transfer_tool_invalid_amount_raises_validation():
    # Pydantic validation rejects negative amount
    with pytest.raises(Exception):
        transfer_currency.invoke({"amount": -25.0, "recipient": "vendor_alpha"})
""", language="python")

# --- SECTION 4: TESTING CHAINS ---
st.header("4. Unit Testing LCEL Chains & Runnables: FakeListChatModel, Mocking, and Step Isolation", divider="rainbow")
st.markdown("""
When testing LangChain Expression Language (LCEL) chains (`prompt | model | parser`), you must never make live calls
to LLM providers in unit tests. Using `FakeListChatModel` allows you to pre-program exact model responses,
testing parsing logic, output schemas, and fallback handlers deterministically.
""")

st.code("""
import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.fake import FakeListChatModel

def test_lcel_chain_with_fake_llm():
    # Setup deterministic mock LLM with pre-defined response sequence
    fake_llm = FakeListChatModel(responses=["Calculated total revenue: $1,450,000"])
    prompt = PromptTemplate.from_template("Summarize: {topic}")
    parser = StrOutputParser()
    
    # Compose LCEL runnable pipeline
    chain = prompt | fake_llm | parser
    
    output = chain.invoke({"topic": "Q3 Financials"})
    
    assert output == "Calculated total revenue: $1,450,000"
    assert isinstance(output, str)
""", language="python")

# --- SECTION 5: TESTING RETRIEVERS ---
st.header("5. Unit Testing Retrievers & Vector Matchers: Mock Embeddings and Top-k Determinism", divider="rainbow")
st.markdown("""
Retriever unit tests verify document scoring, threshold filtering, and empty query handling.
Instead of querying production vector stores, tests use in-memory vector stores (like FAISS or InMemoryVectorStore)
populated with static mock embeddings and known document sets.
""")

st.code("""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

class StaticMockRetriever(BaseRetriever):
    docs: List[Document]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        if not query.strip():
            return []
        return [doc for doc in self.docs if query.lower() in doc.page_content.lower()]

def test_retriever_query_filtering():
    corpus = [
        Document(page_content="AWS EC2 instances provide scalable compute.", metadata={"source": "aws.md"}),
        Document(page_content="PostgreSQL is an enterprise relational database.", metadata={"source": "db.md"})
    ]
    retriever = StaticMockRetriever(docs=corpus)
    
    results = retriever.invoke("relational database")
    assert len(results) == 1
    assert results[0].metadata["source"] == "db.md"
    
    # Empty query edge case
    empty_results = retriever.invoke("")
    assert len(empty_results) == 0
""", language="python")

# --- SECTION 6: TESTING AGENTS ---
st.header("6. Unit Testing Agents: Decision Routing, Mock Tool Calls, and Max-Iteration Guards", divider="rainbow")
st.markdown("""
Unit testing agents focuses on control flow: given specific tool outputs, does the agent choose the next tool correctly?
Does it terminate when an answer is reached? Does it respect maximum iteration limits rather than entering an infinite loop?
""")

st.code("""
import pytest
from langchain_core.messages import AIMessage, ToolCall

def test_agent_terminates_on_final_answer():
    # Simulate agent state step
    agent_state = {
        "messages": [
            AIMessage(content="Final Answer: The system has verified user credentials.")
        ],
        "iteration_count": 2
    }
    
    # Assert termination condition
    last_message = agent_state["messages"][-1]
    is_terminal = "Final Answer" in last_message.content or len(last_message.tool_calls) == 0
    assert is_terminal is True

def test_agent_loop_limit_guard():
    max_iterations = 5
    current_iterations = 6
    
    # Loop guard assertion
    should_halt = current_iterations >= max_iterations
    assert should_halt is True, "Agent loop failed to trigger termination on maximum iterations."
""", language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Comprehensive Pytest Test Suite for LangChain Applications", divider="rainbow")
st.markdown("""
Here is an end-to-end production pytest file demonstrating fixtures, parameterized tests,
and deterministic assertions for a complete LangChain pipeline.
""")

st.code("""
# tests/test_enterprise_agent_units.py
import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models.fake import FakeListChatModel
from pydantic import BaseModel, Field

class AnalysisReport(BaseModel):
    risk_score: int = Field(..., ge=1, le=10)
    summary: str

@pytest.fixture
def fake_chat_model():
    # Deterministic JSON response
    mock_json = '{"risk_score": 3, "summary": "Low security risk detected."}'
    return FakeListChatModel(responses=[mock_json])

@pytest.fixture
def analysis_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are an audit agent."),
        ("human", "Audit server: {server_id}")
    ])

def test_structured_audit_pipeline(fake_chat_model, analysis_prompt):
    parser = JsonOutputParser(pydantic_object=AnalysisReport)
    chain = analysis_prompt | fake_chat_model | parser
    
    result = chain.invoke({"server_id": "srv-prod-99"})
    
    assert isinstance(result, dict)
    assert result["risk_score"] == 3
    assert "Low security risk" in result["summary"]

@pytest.mark.parametrize("invalid_server", ["", None])
def test_prompt_rejects_empty_inputs(analysis_prompt, invalid_server):
    if invalid_server is None:
        with pytest.raises(Exception):
            analysis_prompt.format_messages(server_id=invalid_server)
""", language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Pytest Runner & Mock LLM Response Simulator", divider="rainbow")
st.markdown("""
Execute unit test scenarios against mocked LangChain components.
Inspect deterministic outputs, execution durations, and assertion validations in continuous integration.
""")

harness = UnitTestingHarnessSimulator()

col_u1, col_u2 = st.columns([1, 1])

with col_u1:
    st.subheader("🧪 Pytest Runner Configuration")
    selected_test = st.selectbox(
        "Select Unit Test Target:",
        [tc["test_name"] for tc in harness.test_cases]
    )
    fail_toggle = st.checkbox("Inject Intentional Regression (Simulate Test Failure)", value=False)
    run_pytest_btn = st.button("🚀 Run Pytest Target", type="primary")

with col_u2:
    st.subheader("📋 CI/CD Test Results")
    if run_pytest_btn:
        res = harness.run_selected_test(selected_test, inject_failure=fail_toggle)
        status_badge = f"<span class='status-pass'>{res['status']}</span>" if res["status"] == "PASSED" else f"<span class='status-fail'>{res['status']}</span>"
        
        st.markdown(
            f"<div class='test-card'>"
            f"<b>Test:</b> <code>{res['test_name']}</code> | <b>Target:</b> {res['target']}<br><br>"
            f"<b>Outcome:</b> {status_badge} &nbsp;|&nbsp; <b>Duration:</b> {res['duration_ms']} ms<br>"
            f"<b>Details:</b> {res['detail']}"
            f"</div>",
            unsafe_allow_html=True
        )
        if res["status"] == "PASSED":
            st.success("All unit test assertions passed deterministically with zero network calls.")
        else:
            st.error("AssertionError triggered: Regression identified in isolated component.")
    else:
        st.info("Select a test case and execute to view pytest runner telemetry.")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why should you never use live OpenAI API calls in unit tests?"):
    st.markdown("""
    **Answer**: Live LLM calls violate the three core tenets of unit testing: speed, determinism, and isolation. Live calls take 1-5 seconds per test,
    cost money per run, fail unpredictably due to rate limits or API downtime, and vary in textual completion outputs between runs.
    Unit tests must use deterministic mocks (`FakeListChatModel`, fixtures) to test application code in milliseconds.
    """)

with st.expander("Q2: What is `FakeListChatModel` in LangChain and how is it used?"):
    st.markdown("""
    **Answer**: `FakeListChatModel` (from `langchain_community.chat_models.fake`) is a mock chat model that takes a predefined list of strings
    (`responses=["Response 1", "Response 2"]`). Each call to `.invoke()` or `.ainvoke()` pops the next string from the list. It allows developers
    to test LCEL prompt parsing, output parsers, and tool invocation triggers without an active network connection.
    """)

with st.expander("Q3: How do you unit test custom LangChain tools created with the `@tool` decorator?"):
    st.markdown("""
    **Answer**: Tools can be tested directly by calling `.invoke({"arg1": val1})` with valid and invalid inputs. Test the underlying Pydantic schema
    by asserting that invalid types (strings instead of ints) or negative values raise validation errors, and verify that the tool handles external service
    exceptions gracefully by returning formatted error strings rather than unhandled crashes.
    """)

with st.expander("Q4: How do you verify agent termination in unit tests?"):
    st.markdown("""
    **Answer**: Pass a mock LLM whose predefined responses simulate a tool call followed by a final answer. Assert that the agent state machine halts,
    the message history contains the final answer, and `max_iterations` guards prevent runaway execution if the mock continuously issues tool calls.
    """)

with st.expander("Q5: What is the difference between unit testing and evaluation testing in LangChain?"):
    st.markdown("""
    **Answer**: **Unit testing** is deterministic, mocking the LLM to verify application plumbing (prompts, tool schemas, routing, parsers) in milliseconds.
    **Evaluation testing** tests the actual probabilistic performance of the real model (faithfulness, accuracy, hallucination) against golden test datasets,
    requiring real LLM calls and evaluation metrics (e.g. Ragas, LLM-as-a-judge).
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Live Model Calls in CI Pipelines)**: Running tests against live cloud LLM APIs in GitHub Actions. Builds fail unpredictably due to network blips or rate limits, stalling deployments.")
    st.warning("**Trap 2 (Testing Only Happy Paths)**: Testing only perfect inputs for tools. In production, LLMs frequently pass malformed JSON, strings for integers, or missing arguments; tools must be tested against garbage inputs.")
with col_trap2:
    st.error("**Trap 3 (Hardcoding Exact String Matches on Live Output)**: Running `assert output == 'exact phrase'` on non-mocked LLMs. Natural language generation varies slightly with temperature or model checkpoints.")
    st.warning("**Trap 4 (Mocking Without Schema Conformance)**: Mocking an LLM output with a string when the downstream parser expects a JSON schema. The test passes superficially but fails in production.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What class in LangChain provides mock chat model responses?")
        st.caption("Flip for answer:")
        st.info("**Back**: `FakeListChatModel` from `langchain_community.chat_models.fake`.")
    with st.container(border=True):
        st.markdown("**Front**: How should custom tools handle bad inputs during agent runs?")
        st.caption("Flip for answer:")
        st.info("**Back**: Return a descriptive string error message so the LLM can self-correct, rather than crashing with unhandled exceptions.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary focus of LangChain unit tests?")
        st.caption("Flip for answer:")
        st.info("**Back**: Verifying application plumbing (prompts, schemas, parsing, routing) deterministically without calling live APIs.")
    with st.container(border=True):
        st.markdown("**Front**: How do you test that a prompt template has all required variables?")
        st.caption("Flip for answer:")
        st.info("**Back**: Call `.format()` with missing keys and assert that `KeyError` is raised.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand why live LLM API calls are prohibited in continuous integration unit tests.", value=True)
c2 = st.checkbox("I can test ChatPromptTemplate formatting and required variable bindings with pytest.", value=True)
c3 = st.checkbox("I know how to validate custom tool Pydantic schemas and error return values.", value=True)
c4 = st.checkbox("I can construct deterministic LCEL chain tests using FakeListChatModel.", value=True)
c5 = st.checkbox("I understand how to mock retrievers using static in-memory Document collections.", value=True)
c6 = st.checkbox("I know how to test agent loop iteration limits and termination conditions.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 33.2: Integration Testing.")
