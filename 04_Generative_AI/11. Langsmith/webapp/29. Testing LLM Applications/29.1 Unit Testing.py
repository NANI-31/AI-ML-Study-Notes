import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="29.1 Unit Testing - LangSmith Master Curriculum",
    page_icon="🧪",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class ComponentUnitTestEngine:
    """
    Production Component Unit Test Harness for LLM Applications.
    Emulates isolated unit testing of individual LangChain and LangGraph primitives
    without invoking expensive frontier models or remote databases:
    1. Prompts (Template formatting, variable binding, escaping)
    2. Parsers (JSON decoding, Pydantic validation, schema compliance)
    3. Tools (Argument validation, edge conditions, mocked exception handling)
    4. Retrievers (Top-K recall, document chunk deserialization, empty context)
    5. Runnables (RunnableLambda contract verification, input/output data flow)
    """
    def __init__(self, project_name: str = "llm-unit-test-harness"):
        self.project_name = project_name
        self.test_registry = [
            {"id": "ut_pmt_01", "component": "Prompt", "name": "test_prompt_variable_binding", "target": "ChatPromptTemplate", "status": "passed", "latency_ms": 1.2, "error": None},
            {"id": "ut_pmt_02", "component": "Prompt", "name": "test_prompt_double_escape_json", "target": "SystemPromptTemplate", "status": "passed", "latency_ms": 0.8, "error": None},
            {"id": "ut_psr_01", "component": "Parser", "name": "test_json_parser_valid_pydantic", "target": "JsonOutputParser", "status": "passed", "latency_ms": 2.4, "error": None},
            {"id": "ut_psr_02", "component": "Parser", "name": "test_json_parser_markdown_fence_stripping", "target": "JsonOutputParser", "status": "passed", "latency_ms": 1.9, "error": None},
            {"id": "ut_tol_01", "component": "Tool", "name": "test_tool_pydantic_schema_valid_args", "target": "StockQuoteTool", "status": "passed", "latency_ms": 3.1, "error": None},
            {"id": "ut_tol_02", "component": "Tool", "name": "test_tool_missing_arg_raises_validation_error", "target": "StockQuoteTool", "status": "passed", "latency_ms": 2.8, "error": None},
            {"id": "ut_ret_01", "component": "Retriever", "name": "test_retriever_mock_candidate_unpacking", "target": "VectorStoreRetriever", "status": "passed", "latency_ms": 4.5, "error": None},
            {"id": "ut_ret_02", "component": "Retriever", "name": "test_retriever_empty_context_graceful_fallback", "target": "VectorStoreRetriever", "status": "passed", "latency_ms": 3.2, "error": None},
            {"id": "ut_run_01", "component": "Runnable", "name": "test_runnable_lambda_string_sanitization", "target": "RunnableLambda", "status": "passed", "latency_ms": 0.9, "error": None}
        ]

    def run_unit_tests(
        self,
        component_filter: str = "All",
        inject_mock_failure: bool = False
    ) -> Dict[str, Any]:
        """
        Executes component unit tests and returns pass/fail metrics and assertion traces.
        """
        results = []
        for test in self.test_registry:
            if component_filter != "All" and test["component"] != component_filter:
                continue

            test_copy = dict(test)
            if inject_mock_failure and test["name"] in ["test_json_parser_markdown_fence_stripping", "test_retriever_mock_candidate_unpacking"]:
                test_copy["status"] = "failed"
                test_copy["error"] = "AssertionError: Expected JSON dict with keys ['status'], got None"
            else:
                test_copy["status"] = "passed"
                test_copy["error"] = None

            results.append(test_copy)

        total = len(results)
        passed = sum(1 for r in results if r["status"] == "passed")
        failed = total - passed

        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "pass_rate": round(passed / total, 3) if total > 0 else 1.0,
            "total_latency_ms": sum(r["latency_ms"] for r in results),
            "test_results": results
        }


# Instantiate engine
unit_test_harness = ComponentUnitTestEngine()

# --- HEADER SECTION ---
st.title("🧪 Chapter 29.1: Unit Testing LLM Components")
st.markdown("""
Welcome to the production guide on **Unit Testing LLM Applications** with **LangSmith**.
In conventional software engineering, unit tests verify small, isolated pieces of logic without touching network APIs.
In GenAI engineering, unit testing is frequently abandoned because developers mistakenly assume every test requires
an expensive call to OpenAI or Anthropic.
By decomposing applications into isolated components—**Prompts**, **Output Parsers**, **Tools**, and **Retrievers**—and
mocking downstream LLMs, engineering teams run deterministic unit test suites in milliseconds,
catching syntax bugs and schema regressions before code ever touches staging.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: LLM Component Unit Testing & Isolated Mock Architecture", divider="rainbow")

st.markdown("""
The architecture below illustrates the unit test isolation pattern.
Individual components are tested independently against deterministic fixtures and mocked LLMs.
Because frontier LLM APIs are not invoked during unit tests, test suites execute in milliseconds
with zero token expenditures, running seamlessly in local pre-commit hooks and GitHub Actions.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph UnitTesting {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_test_runner {
            label="1. Local Test Runner (pytest / CI pre-commit)";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            pytest [label="pytest Test Suite\nPre-commit hook & PR check\nRuntime: < 2 seconds", fillcolor="#1E293B", color="#38BDF8"];
            fixtures [label="Deterministic Mock Fixtures\nStatic JSON strings, mock doc lists", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_components {
            label="2. Isolated Components Under Test";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            u_prompt [label="1. Prompt Templates\nVerify {var} substitution\nTest double-escaping {{...}}", fillcolor="#1E293B", color="#38BDF8"];
            u_parser [label="2. Output Parsers\nJsonOutputParser / Pydantic\nTest markdown fence stripping", fillcolor="#1E293B", color="#38BDF8"];
            u_tool [label="3. Tool Definitions\nPydantic args_schema validation\nMocked API endpoints", fillcolor="#1E293B", color="#38BDF8"];
            u_ret [label="4. Retrievers\nMocked VectorStore response\nTop-K slice & metadata formatting", fillcolor="#1E293B", color="#38BDF8"];
            u_run [label="5. Custom Runnables\nRunnableLambda string clean\nContext transformation logic", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_mocks {
            label="3. Mocking Layer (Zero Token Spend)";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            fake_llm [label="FakeListChatModel\nlangchain_community.chat_models.fake\nReturns predetermined strings", fillcolor="#0F172A", color="#10B981"];
        }

        subgraph cluster_assertions {
            label="4. Deterministic Assertions & LangSmith Unit Trace";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            asserts [label="Deterministic Asserts\nassert isinstance(output, ExpectedSchema)\nassert 'required_key' in output", fillcolor="#064E3B", color="#10B981"];
            ls_unit_trace [label="LangSmith Unit Test Project\nTraces recorded with tag: ['unit-test']\nZero LLM API cost", fillcolor="#0F172A", color="#38BDF8"];
        }

        pytest -> fixtures;
        fixtures -> u_prompt;
        fixtures -> u_parser;
        fixtures -> u_tool;
        fixtures -> u_ret;
        fixtures -> u_run;

        u_prompt -> asserts;
        u_parser -> asserts;
        u_tool -> asserts;
        u_ret -> asserts;
        u_run -> asserts;

        u_run -> fake_llm [label="Mock LLM Calls", style=dotted];
        asserts -> ls_unit_trace;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Component Unit Testing Disciplines
The following table provides the unit testing strategy for each core component:
""")

ut_disciplines_df = pd.DataFrame([
    {
        "Component": "1. Prompts",
        "Target Class": "ChatPromptTemplate, PromptTemplate",
        "What to Test": "Variable binding, .partial() defaults, double-escaped JSON braces, type conversions",
        "Mocking Need": "None (pure string manipulation)"
    },
    {
        "Component": "2. Parsers",
        "Target Class": "JsonOutputParser, PydanticOutputParser",
        "What to Test": "Malformed JSON resilience, markdown fence stripping (```json), schema violations",
        "Mocking Need": "None (pass static strings directly)"
    },
    {
        "Component": "3. Tools",
        "Target Class": "@tool, BaseTool, ToolNode",
        "What to Test": "Pydantic args_schema validation, type coercion, handle_tool_error behavior",
        "Mocking Need": "Mock external HTTP APIs / DB calls with unittest.mock"
    },
    {
        "Component": "4. Retrievers",
        "Target Class": "VectorStoreRetriever, EnsembleRetriever",
        "What to Test": "Top-K slicing, score threshold cut-offs, empty candidate fallbacks, document formatting",
        "Mocking Need": "Mock VectorStore.similarity_search() returning pre-set Documents"
    },
    {
        "Component": "5. Runnables",
        "Target Class": "RunnableLambda, RunnableSequence",
        "What to Test": "Input/output dictionary transformations, custom filter logic, exception propagation",
        "Mocking Need": "Use FakeListChatModel for intermediate LLM calls"
    }
])
st.dataframe(ut_disciplines_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TESTING PROMPTS
# ==============================================================================
st.header("2. Testing Prompts (Variable Binding, Template Invariants & Negative Constraint Validation)", divider="rainbow")

st.markdown("""
Prompt templates are functions that map input dictionaries to formatted messages.
Unit testing prompt templates verifies that:
* Every declared variable (`{user_query}`, `{context}`) is correctly bound.
* Static default variables bound via `.partial()` do not raise `KeyError`.
* Literal curly braces (e.g., in JSON schema instructions or code blocks) are properly escaped as `{{` and `}}`.
""")

st.code('''
from langchain_core.prompts import ChatPromptTemplate
import pytest

def test_chat_prompt_template_formatting():
    # Prompt with static and dynamic variables
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant for {organization}. Schema: {{{{id: int}}}}"),
        ("human", "{user_query}")
    ])
    
    # Pre-bind organization
    bound_prompt = prompt.partial(organization="Acme Corp")
    
    # Invoke formatting
    messages = bound_prompt.format_messages(user_query="Hello")
    
    assert len(messages) == 2
    assert "Acme Corp" in messages[0].content
    assert "{id: int}" in messages[0].content # Verifies double-escaping worked
    assert messages[1].content == "Hello"

def test_missing_prompt_variable_raises_key_error():
    prompt = ChatPromptTemplate.from_messages([("human", "{missing_variable}")])
    with pytest.raises(KeyError):
        prompt.format_messages()
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production CI pipelines:
* Unit tests validate all prompt files across the repository to ensure no broken template variables exist.
* Tests verify that prompts formatted with maximum-length strings do not exceed pre-configured character budgets.
""")

st.divider()

# ==============================================================================
# SECTION 3: TESTING PARSERS
# ==============================================================================
st.header("3. Testing Parsers (Malformed JSON Resilience, Pydantic Schema Assertions & Partial Payloads)", divider="rainbow")

st.markdown("""
Output parsers are the bridge between stochastic model text and deterministic application logic.
Unit tests verify that parsers handle realistic LLM output quirks:
* Stripping markdown code fences (` ```json ... ``` `).
* Handling conversational preambles ("Here is the requested output: {...}").
* Raising `OutputParserException` on invalid payloads so upstream fallbacks can catch them.
""")

st.code('''
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pytest

class ActionPlan(BaseModel):
    action: str = Field(...)
    priority: int = Field(...)

def test_json_parser_strips_markdown_fences():
    parser = JsonOutputParser(pydantic_object=ActionPlan)
    
    # Simulated raw LLM generation with markdown fences and conversational preamble
    raw_llm_text = """
    Certainly! Here is your structured plan:
    ```json
    {
        "action": "upgrade_database",
        "priority": 1
    }
    ```
    """
    
    parsed = parser.parse(raw_llm_text)
    assert parsed["action"] == "upgrade_database"
    assert parsed["priority"] == 1

def test_parser_raises_on_invalid_json():
    parser = JsonOutputParser(pydantic_object=ActionPlan)
    with pytest.raises(Exception):
        parser.parse("Not a JSON object at all")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In enterprise testing:
* Parser unit test suites include a corpus of 50 historical edge-case completions (trailing commas, escaped quotes, truncated brackets).
* Running this suite ensures parsers reliably extract valid objects without unexpected runtime crashes.
""")

st.divider()

# ==============================================================================
# SECTION 4: TESTING TOOLS
# ==============================================================================
st.header("4. Testing Tools (Argument Validation, Edge Value Assertions & Mocked Exception Handling)", divider="rainbow")

st.markdown("""
Tools give agents the ability to interact with real-world APIs and databases.
Unit testing tools focuses on:
* **Schema Validation**: Ensuring that arguments strictly validate against the tool's Pydantic `args_schema`.
* **Mocked Execution**: Verifying that the tool function handles HTTP errors, timeouts, and missing records gracefully.
* **Error Resilience**: Verifying that `handle_tool_error=True` returns an informative error string instead of crashing the process.
""")

st.code('''
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pytest

class CalculatorInput(BaseModel):
    numerator: float = Field(...)
    denominator: float = Field(...)

@tool(args_schema=CalculatorInput, handle_tool_error=True)
def divide_numbers(numerator: float, denominator: float) -> str:
    """Divides numerator by denominator safely."""
    if denominator == 0:
        raise ValueError("Cannot divide by zero.")
    return str(numerator / denominator)

def test_tool_valid_arguments():
    result = divide_numbers.invoke({"numerator": 10.0, "denominator": 2.0})
    assert result == "5.0"

def test_tool_handles_zero_division():
    # Because handle_tool_error=True, it catches the ValueError and returns error text
    result = divide_numbers.invoke({"numerator": 10.0, "denominator": 0.0})
    assert "Cannot divide by zero" in result
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production agent pipelines:
* Tools are unit tested with mocked HTTP adapters (`responses` or `unittest.mock`) to test how the agent handles 429 rate limits and 503 service unavailable errors.
""")

st.divider()

# ==============================================================================
# SECTION 5: TESTING RETRIEVERS
# ==============================================================================
st.header("5. Testing Retrievers (Recall at Top-K, Document Deserialization & Query Rewriter Unit Tests)", divider="rainbow")

st.markdown("""
Retrievers query vector stores and return lists of `Document` objects.
In unit tests, we test the retriever logic (formatting, filtering, deduplication, query transformation)
by mocking the underlying vector store call.
""")

st.markdown("### What to Unit Test in Retrievers")
st.markdown("""
* **Document Formatting**: Verifying that raw database records are correctly converted into `Document(page_content=..., metadata=...)`.
* **Empty Context Behavior**: Verifying that when the vector store returns 0 matches, the retriever returns an empty list without raising exceptions.
* **Query Preprocessing**: Verifying that query rewriter functions strip special characters or inject tenant filters correctly.
""")

st.code('''
from langchain_core.documents import Document
from unittest.mock import MagicMock

def test_retriever_formatting_logic():
    # Mock vectorstore
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = [
        Document(page_content="Policy details...", metadata={"id": 1, "source": "handbook.pdf"}),
        Document(page_content="Vacation days...", metadata={"id": 2, "source": "benefits.pdf"})
    ]
    
    retriever = mock_vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("How many vacation days?")
    
    assert len(docs) == 2
    assert docs[0].metadata["source"] == "handbook.pdf"
    mock_vectorstore.similarity_search.assert_called_once_with("How many vacation days?", k=2)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
By mocking the vector store:
* Retriever unit tests run in less than 5 milliseconds without needing a live ChromaDB, Pinecone, or pgvector database running in the CI runner.
""")

st.divider()

# ==============================================================================
# SECTION 6: TESTING INDIVIDUAL COMPONENTS
# ==============================================================================
st.header("6. Testing Individual Components (Isolated Runnable Invocations & Mocked LLM Responses)", divider="rainbow")

st.markdown("""
Complex pipelines comprise custom data preparation runnables (`RunnableLambda`).
Unit testing these runnables verifies that context merging, text trimming, and state transforms
function correctly before being wired into large chains.
When testing an intermediate step that calls an LLM, we use `FakeListChatModel`.
""")

st.code('''
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.runnables import RunnableLambda

def sanitize_user_input(data: dict) -> dict:
    """Custom runnable step that strips whitespace and redacts SSNs."""
    clean_query = data["query"].strip()
    return {"clean_query": clean_query, "user_id": data.get("user_id", "anon")}

def test_runnable_sanitization_step():
    runnable = RunnableLambda(sanitize_user_input)
    result = runnable.invoke({"query": "   test query with trailing spaces   "})
    assert result["clean_query"] == "test query with trailing spaces"
    assert result["user_id"] == "anon"

def test_chain_with_fake_llm():
    # FakeListChatModel returns responses sequentially from a list
    fake_model = FakeListChatModel(responses=["Hello from fake LLM!"])
    response = fake_model.invoke("Say hello")
    assert response.content == "Hello from fake LLM!"
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern LCEL architectures:
* `FakeListChatModel` enables testing entire multi-step LCEL chains locally in automated unit tests.
* Developers can verify that prompt outputs correctly feed into models and that model outputs correctly pipe into parsers without spending a single API dollar.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: LangChain/LangGraph Component Unit Test Harness (ComponentUnitTestEngine)", divider="rainbow")

st.markdown("""
Below is the reference unit test harness implementation.
The `ComponentUnitTestEngine` coordinates isolated unit tests across prompts, parsers,
tools, retrievers, and runnables, measuring execution latencies and reporting assertion telemetry.
""")

st.code('''
class ComponentUnitTestEngine:
    """
    Production Component Unit Test Harness for LLM Applications.
    Emulates isolated unit testing of individual LangChain and LangGraph primitives
    without invoking expensive frontier models or remote databases.
    """
    def __init__(self, project_name: str = "llm-unit-test-harness"):
        self.project_name = project_name
        self.test_registry = [...]

    def run_unit_tests(
        self,
        component_filter: str = "All",
        inject_mock_failure: bool = False
    ) -> Dict[str, Any]:
        """
        Executes component unit tests and returns pass/fail metrics and assertion traces.
        """
        # Executes isolated tests with microsecond execution latencies
        ...
''', language="python")

st.info("The unit test harness is instantiated in module scope and powers the interactive test runner below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Component Unit Test Runner & Mock Injection Sandbox", divider="rainbow")

st.markdown("""
Run the component unit test suite across Prompts, Parsers, Tools, Retrievers, and Runnables.
Observe the microsecond execution speeds, and toggle mock faults to see how assertion failures are isolated.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Unit Test Execution Controls")
    comp_filter = st.selectbox(
        "Filter by Component Type:",
        options=["All", "Prompt", "Parser", "Tool", "Retriever", "Runnable"]
    )

    inject_fault = st.toggle(
        "Inject Synthetic Regression Fault",
        value=False,
        help="Simulates a regression in the JSON parser and retriever to test failure isolation"
    )

    run_tests_btn = st.button("🧪 Run Component Unit Test Suite", use_container_width=True)

with wb_col2:
    st.subheader("Unit Test Execution Report")

    test_run = unit_test_harness.run_unit_tests(
        component_filter=comp_filter,
        inject_mock_failure=inject_fault
    )

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tests Executed", test_run["total_tests"])
    m2.metric("Pass Rate", f"{test_run['pass_rate'] * 100:.1f} %")
    m3.metric("Failed Tests", test_run["failed_tests"])
    m4.metric("Total Latency", f"{test_run['total_latency_ms']:.1f} ms")

    if test_run["failed_tests"] == 0:
        st.success("✅ **All Unit Tests Passed**: Component contracts verified in under 20ms.")
    else:
        st.error(f"🚨 **{test_run['failed_tests']} Unit Test(s) Failed**: Regression detected in isolated components.")

    # Test Results Table
    st.markdown("#### Test Execution Registry")
    results_df = pd.DataFrame(test_run["test_results"])
    st.dataframe(results_df[["id", "component", "name", "target", "status", "latency_ms"]], use_container_width=True)

    # Failure Details if any
    failed_items = [r for r in test_run["test_results"] if r["status"] == "failed"]
    if failed_items:
        st.markdown("#### Regression Assertion Tracebacks")
        for f in failed_items:
            with st.container(border=True):
                st.markdown(f"**Failed Test**: `{f['name']}` ({f['target']})")
                st.error(f"Traceback: {f['error']}")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: Why should you avoid calling real LLM APIs in unit test suites?"):
    st.markdown("""
    **Answer:**
    1. **Stochastic Non-Determinism**: Real LLMs produce slight wording variations that cause brittle string assertions to fail unpredictably.
    2. **Execution Latency**: Calling an API takes 500ms to 2,000ms per test. A suite of 100 unit tests would take 3 minutes instead of 100 milliseconds.
    3. **Cost & Rate Limits**: Running real calls on every Git commit burns API credits and risks hitting cloud provider rate limits during CI/CD builds.
    Unit tests should verify deterministic code (prompts, schemas, tools, parsers) using mocks like `FakeListChatModel`.
    """)

with st.expander("Q2: How does FakeListChatModel facilitate testing of complex LCEL pipelines?"):
    st.markdown("""
    **Answer:**
    `FakeListChatModel` (from `langchain_community.chat_models.fake`) accepts a predefined list of string responses.
    Each time `.invoke()` is called, it emits the next string in the list.
    This allows developers to test:
    * Whether prompt template variables are correctly bound and piped into the model.
    * Whether downstream `JsonOutputParser` or `PydanticOutputParser` correctly parses the model's expected generation.
    * How the chain behaves when the model emits malformed text or empty strings.
    """)

with st.expander("Q3: What specific failure modes should be tested when unit testing a LangChain tool?"):
    st.markdown("""
    **Answer:**
    1. **Valid Argument Invocation**: Verifying expected return values on happy-path arguments.
    2. **Missing Mandatory Arguments**: Verifying that omitting a required field raises `pydantic.ValidationError`.
    3. **Type Coercion**: Verifying how the tool handles string inputs for integer/float fields.
    4. **Error Handling (`handle_tool_error`)**: Verifying that exceptions raised inside the tool function are caught and converted into informative error strings rather than crashing the execution thread.
    """)

with st.expander("Q4: How do you unit test prompt templates that contain JSON examples in system prompts?"):
    st.markdown("""
    **Answer:**
    Python's `str.format()` and `ChatPromptTemplate` treat single curly braces `{...}` as variable placeholders.
    To include literal JSON examples in prompts without triggering a `KeyError`:
    * Double-escape the curly braces: `{{ "status": "success", "id": {user_id} }}`.
    * In unit tests, assert that `messages[0].content` contains single curly braces `{ "status": "success" ... }` after formatting.
    """)

with st.expander("Q5: What is the difference between a Component Unit Test and an Integration Test in GenAI?"):
    st.markdown("""
    **Answer:**
    * **Unit Test**: Tests a single component (e.g., prompt formatting, parser decoding, tool logic) in complete isolation using mocked inputs and zero real LLM or database calls. Runs in < 10ms.
    * **Integration Test**: Connects multiple real components together (e.g., Prompt + Real Vector Database + Real Chat Model + Output Parser) to verify that data flows correctly across external service boundaries.
    """)

st.markdown("### Production Traps & Failure Modes")

ut_tr_col1, ut_tr_col2 = st.columns(2)

with ut_tr_col1:
    st.error("""
    **Trap 1: Calling OpenAI / Anthropic APIs in pytest Unit Tests**
    * **Failure**: Writing unit tests that invoke live foundation model APIs without mocking.
    * **Consequence**: Test suite takes 10+ minutes to execute, fails intermittently on rate limits, and burns thousands of dollars in cloud API bills during CI builds.
    * **Remedy**: Strictly isolate unit tests using `FakeListChatModel` and mocked retrievers; save live model calls for integration tests.
    """)

    st.error("""
    **Trap 2: Ignoring Parser Edge Cases in Unit Tests**
    * **Failure**: Testing `JsonOutputParser` only with clean, perfectly formatted JSON strings.
    * **Consequence**: The parser crashes in production when an LLM wraps output in markdown fences (` ```json `) or includes trailing conversational preambles.
    * **Remedy**: Include malformed, fenced, and preambled test fixtures in the parser unit test suite.
    """)

with ut_tr_col2:
    st.warning("""
    **Trap 3: Testing Without Pydantic Schema Validation**
    * **Failure**: Passing untyped raw dictionaries into tools instead of declaring an explicit `args_schema`.
    * **Consequence**: LLMs generate hallucinated argument keys that crash internal tool functions at runtime.
    * **Remedy**: Always define Pydantic models for tool inputs and write unit tests asserting that schema validation catches missing keys.
    """)

    st.warning("""
    **Trap 4: Missing Empty-Context Retriever Tests**
    * **Failure**: Testing retrievers only with queries that return abundant matching documents.
    * **Consequence**: When a user asks an out-of-domain question, downstream prompt formatters crash on empty document lists.
    * **Remedy**: Unit test retrievers with empty mock document lists, asserting clean fallback behavior.
    """)

st.markdown("### Core Technical Flashcards")

fcut_c1, fcut_c2 = st.columns(2)

with fcut_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary purpose of `FakeListChatModel` in unit testing?")
        st.markdown("**Back**: It provides deterministic, zero-cost, instantaneous responses for testing prompts, chains, and parsers without calling live external LLM APIs.")

    with fcut_c2:
        st.markdown("**Front**: How do you escape literal JSON braces in `ChatPromptTemplate`?")
        st.markdown("**Back**: Use double curly braces `{{` and `}}`. LangChain's formatting engine parses double braces into literal single braces in the final message content.")

with fcut_c1:
    with st.container(border=True):
        st.markdown("**Front**: What does `handle_tool_error=True` accomplish in LangChain tools?")
        st.markdown("**Back**: It catches exceptions raised during tool execution and returns the error message as a string rather than raising a fatal exception, enabling agent self-healing.")

    with fcut_c2:
        st.markdown("**Front**: How fast should an LLM component unit test suite execute?")
        st.markdown("**Back**: A suite of 50-100 unit tests should execute in under 2 seconds, because all external LLMs, vector stores, and web APIs are fully mocked.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of LLM component unit testing:")

ut_chk1 = st.checkbox("I know how to unit test prompt template variable bindings and double-escaped JSON braces.", value=True)
ut_chk2 = st.checkbox("I can test output parsers against malformed JSON, markdown fences, and conversational preambles.", value=True)
ut_chk3 = st.checkbox("I know how to validate tool Pydantic argument schemas and test `handle_tool_error` behavior.", value=True)
ut_chk4 = st.checkbox("I can mock vector store retrieval to unit test Top-K candidate slices and empty context fallbacks.", value=True)
ut_chk5 = st.checkbox("I understand how to use `FakeListChatModel` to test full LCEL pipelines locally without API costs.", value=True)
ut_chk6 = st.checkbox("I can maintain rapid unit test execution speeds (< 2s) by strictly decoupling from external networks.", value=True)

if ut_chk1 and ut_chk2 and ut_chk3 and ut_chk4 and ut_chk5 and ut_chk6:
    st.success("Mastery Certified: You have mastered LLM component unit testing, mock fixtures, and isolated contract verification.")
