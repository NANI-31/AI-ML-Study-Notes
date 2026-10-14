import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="29.2 Integration Testing - LangSmith Master Curriculum",
    page_icon="🔌",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class IntegrationTestingEngine:
    """
    Production Cross-Component Integration Testing Suite for LLM Applications.
    Emulates live boundary testing across five core GenAI integration architectures:
    1. Full Chains (LCEL Runnable pipes: Prompt -> Model -> Parser)
    2. RAG Pipelines (Retriever -> Context Formatter -> LLM Grounding)
    3. Autonomous Agents (Model -> ToolNode -> Self-Correction Loop)
    4. Tool Integrations (Live API sandbox endpoints, authentication tokens)
    5. Multi-Component Workflows (Hierarchical StateGraphs, subgraphs, checkpointers)
    """
    def __init__(self, project_name: str = "llm-integration-testing"):
        self.project_name = project_name
        self.integration_runs: List[Dict[str, Any]] = []

    def execute_integration_test(
        self,
        pipeline_type: str,
        sandbox_environment: str = "staging",
        inject_service_timeout: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a multi-component integration run across simulated external dependencies.
        """
        run_id = f"integ_run_{len(self.integration_runs) + 101}"

        spans = []
        overall_status = "passed"
        error_msg = None

        if pipeline_type == "full_chain":
            # Prompt -> LLM -> Parser
            spans = [
                {"step": 1, "name": "ChatPromptTemplate", "component": "Prompt", "status": "passed", "latency_ms": 1.5, "io": "{query} -> ChatPromptValue"},
                {"step": 2, "name": "ChatModel_StagingEndpoint", "component": "LLM", "status": "failed" if inject_service_timeout else "passed", "latency_ms": 1250.0 if inject_service_timeout else 480.0, "io": "ChatPromptValue -> AIMessage(content='...')"},
                {"step": 3, "name": "JsonOutputParser", "component": "Parser", "status": "skipped" if inject_service_timeout else "passed", "latency_ms": 2.1, "io": "AIMessage -> Dict[str, Any]"}
            ]
            if inject_service_timeout:
                overall_status = "failed"
                error_msg = "TimeoutError: LLM staging endpoint failed to respond within 1000ms SLA."

        elif pipeline_type == "rag_pipeline":
            # Retriever -> Formatter -> LLM
            spans = [
                {"step": 1, "name": "VectorStore_SandboxConnection", "component": "Database", "status": "failed" if inject_service_timeout else "passed", "latency_ms": 850.0 if inject_service_timeout else 45.0, "io": "Vector Query -> 4 Document Chunks"},
                {"step": 2, "name": "ContextFormatter_Lambda", "component": "Runnable", "status": "skipped" if inject_service_timeout else "passed", "latency_ms": 1.2, "io": "Documents -> Formatted String Context"},
                {"step": 3, "name": "GroundedGenerator", "component": "LLM", "status": "skipped" if inject_service_timeout else "passed", "latency_ms": 520.0, "io": "Context+Query -> Factual Grounded Response"}
            ]
            if inject_service_timeout:
                overall_status = "failed"
                error_msg = "ConnectionResetError: Vector database sandbox connection dropped."

        elif pipeline_type == "agent_workflow":
            # ReAct Agent: LLM -> ToolNode -> Reducer -> Final
            spans = [
                {"step": 1, "name": "SupervisorDecision_Node", "component": "Agent", "status": "passed", "latency_ms": 320.0, "io": "State -> AIMessage(tool_calls=[...])"},
                {"step": 2, "name": "ToolNode_Execution", "component": "Tool", "status": "failed" if inject_service_timeout else "passed", "latency_ms": 610.0 if inject_service_timeout else 115.0, "io": "ToolCalls -> ToolMessage(content='...')"},
                {"step": 3, "name": "StateReducer_add_messages", "component": "GraphState", "status": "skipped" if inject_service_timeout else "passed", "latency_ms": 0.8, "io": "State Diff Merged: +2 Messages"},
                {"step": 4, "name": "Synthesizer_Node", "component": "Agent", "status": "skipped" if inject_service_timeout else "passed", "latency_ms": 410.0, "io": "Final State -> Verified Answer"}
            ]
            if inject_service_timeout:
                overall_status = "failed"
                error_msg = "ToolException: External sandbox API 504 Gateway Timeout."

        else: # multi_component_graph
            spans = [
                {"step": 1, "name": "Checkpointer_PostgresConnection", "component": "Storage", "status": "passed", "latency_ms": 12.0, "io": "Initialized thread_id: test-thread-401"},
                {"step": 2, "name": "IngressValidation_Node", "component": "GraphNode", "status": "passed", "latency_ms": 3.4, "io": "Validated Pydantic state schema"},
                {"step": 3, "name": "SpecialistWorker_Subgraph", "component": "Subgraph", "status": "failed" if inject_service_timeout else "passed", "latency_ms": 940.0 if inject_service_timeout else 380.0, "io": "Child Graph Invocation"},
                {"step": 4, "name": "CheckpointSave_SuperStep", "component": "Storage", "status": "skipped" if inject_service_timeout else "passed", "latency_ms": 8.0, "io": "Snapshot persisted at version v2"}
            ]
            if inject_service_timeout:
                overall_status = "failed"
                error_msg = "DeadlockDetected: Subgraph state transition exceeded timeout budget."

        total_latency = sum(s["latency_ms"] for s in spans)

        record = {
            "run_id": run_id,
            "pipeline_type": pipeline_type,
            "environment": sandbox_environment,
            "status": overall_status,
            "error_message": error_msg,
            "total_latency_ms": round(total_latency, 1),
            "spans": spans
        }

        self.integration_runs.append(record)
        return record


# Instantiate engine
integration_engine = IntegrationTestingEngine()

# --- HEADER SECTION ---
st.title("🔌 Chapter 29.2: Integration Testing LLM Workflows")
st.markdown("""
Welcome to the production guide on **Integration Testing** for LLM applications traced in **LangSmith**.
While unit tests verify individual components in complete isolation with mocked data,
**Integration Tests** verify that components interact correctly across real boundaries:
testing prompt-to-model streaming, vector database chunk deserialization, live sandbox tool calling,
and stateful LangGraph subgraph transitions.
This module covers how to design robust, cost-effective integration test suites for enterprise GenAI stacks.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Cross-Component Integration Testing & Live Service Boundary Verification", divider="rainbow")

st.markdown("""
The architecture below illustrates an enterprise integration test harness.
Tests run against staging or sandboxed dependencies (staging vector database, mocked third-party webhooks,
and fast cost-effective models like `gpt-4o-mini`). LangSmith collects the entire multi-stage trace,
allowing assertion of end-to-end latency budgets, data contract compliance, and error fallback resilience.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph IntegrationTesting {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_orchestrator {
            label="1. Integration Test Runner (CI Staging Pipeline)";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            runner [label="Integration Test Harness\nNightly / Pre-Release Staging Gate\nTimeout Budget: 15s per test", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_sandboxes {
            label="2. Sandboxed Service Dependencies";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            vdb_sandbox [label="Vector Store Sandbox\nIsolated staging collection / index\nKnown seeded documents", fillcolor="#1E293B", color="#38BDF8"];
            model_sandbox [label="Fast Staging Models\ngpt-4o-mini / claude-3-5-haiku\nPredictable pricing & latency", fillcolor="#1E293B", color="#38BDF8"];
            api_sandbox [label="Third-Party Mock Server\nWireMock / Prism OpenAPI mock\nReturns deterministic payloads", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_pipelines {
            label="3. Integrated Pipeline Under Test";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            full_chain [label="Full LCEL Chain\nPrompt | ChatModel | Parser", fillcolor="#0F172A", color="#38BDF8"];
            rag_pipe [label="RAG Pipeline\nRetriever | Context | Model", fillcolor="#0F172A", color="#38BDF8"];
            agent_pipe [label="LangGraph StateGraph\nSupervisor <-> ToolNode <-> Reducer", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_langsmith_audit {
            label="4. LangSmith Integration Trace Audit";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            ls_integ_trace [label="LangSmith Integration Trace\nProject: 'staging-integration-tests'\nAudits: Parent-Child hierarchy, latencies, payloads", fillcolor="#064E3B", color="#10B981"];
            gate_verdict [label="Release Gate Verdict\nassert latency_p95 <= 3000ms\nassert error_rate == 0.0", fillcolor="#0F172A", color="#10B981"];
        }

        runner -> full_chain;
        runner -> rag_pipe;
        runner -> agent_pipe;

        full_chain -> model_sandbox [label="Inference"];
        rag_pipe -> vdb_sandbox [label="Retrieval"];
        rag_pipe -> model_sandbox [label="Generation"];
        agent_pipe -> api_sandbox [label="Tool Calls"];
        agent_pipe -> model_sandbox [label="Decisions"];

        full_chain -> ls_integ_trace;
        rag_pipe -> ls_integ_trace;
        agent_pipe -> ls_integ_trace;
        ls_integ_trace -> gate_verdict;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Integration Testing Surfaces
The following table details the scope and verification criteria for each integration target:
""")

integ_surfaces_df = pd.DataFrame([
    {
        "Surface": "1. Full Chains",
        "Architecture": "LCEL Pipeline (Prompt | Model | Parser)",
        "Boundary Verified": "Data passing from ChatPromptTemplate into ChatModel, and model string into JsonOutputParser",
        "Key Failure Detected": "Parser crashes on model token formatting; model returns truncated JSON"
    },
    {
        "Surface": "2. RAG Pipelines",
        "Architecture": "Retriever | Context Runnable | Model",
        "Boundary Verified": "Vector database network query, chunk deserialization, and grounded context injection",
        "Key Failure Detected": "Vector index schema mismatches; empty context prompt crashes; retriever timeouts"
    },
    {
        "Surface": "3. Agents",
        "Architecture": "StateGraph with cyclical tool nodes",
        "Boundary Verified": "Model emitting ToolCalls, ToolNode execution, ToolMessage state appending, agent iteration",
        "Key Failure Detected": "Tool argument hallucination; state corruption in add_messages; infinite tool loops"
    },
    {
        "Surface": "4. Tool Integrations",
        "Architecture": "BaseTool with real HTTP endpoints",
        "Boundary Verified": "OAuth authentication handshakes, network serialization, HTTP 4xx/5xx error handling",
        "Key Failure Detected": "Expired API tokens; breaking upstream schema updates; slow endpoint response"
    },
    {
        "Surface": "5. Multi-Component Workflows",
        "Architecture": "Hierarchical graphs & persistent checkpointers",
        "Boundary Verified": "Parent graph to child subgraph handoffs; checkpointer serialization in Postgres/Sqlite",
        "Key Failure Detected": "Pickling errors on custom objects; thread ID desynchronization; state race conditions"
    }
])
st.dataframe(integ_surfaces_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: FULL CHAINS
# ==============================================================================
st.header("2. Full Chains (LCEL Pipeline Integration, Inter-Stage Data Contracts & Stream Verification)", divider="rainbow")

st.markdown("""
A full chain integration test verifies that all components wired together via the LCEL pipe (`|`) operator
execute harmoniously when communicating with a real LLM endpoint.
While unit tests verify that the parser *can* parse valid JSON, an integration test verifies that the real model
actually *produces* conforming JSON in response to the specific prompt instructions.
""")

st.markdown("### Integration Test Verification Checklist")
st.markdown("""
* **Type Conformance Across Pipes**: Ensuring the model output matches what the output parser expects (`BaseMessage` into `JsonOutputParser`).
* **Streaming Protocol Integrity**: Verifying that calling `.stream()` or `.astream_events()` properly yields partial chunks without buffering or hanging mid-stream.
* **Fallback Cascade Verification**: Testing that attaching `.with_fallbacks([backup_chain])` successfully catches model rate limits or 500 errors and routes to the secondary provider.
""")

st.code('''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class SummaryOutput(BaseModel):
    key_points: list[str] = Field(description="List of 3 bullet points")
    urgency: str = Field(description="Low, Medium, High")

def test_lcel_full_chain_integration():
    # Real LCEL chain connected to staging model endpoint
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an executive assistant. Always output JSON complying with schema: {format_instructions}"),
        ("human", "{email_body}")
    ])
    
    parser = JsonOutputParser(pydantic_object=SummaryOutput)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | model | parser
    
    # Execute with realistic staging input
    result = chain.invoke({"email_body": "Server outage reported in production cluster EU-West."})
    
    assert isinstance(result, dict)
    assert "key_points" in result
    assert result["urgency"] in ["Low", "Medium", "High"]
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production CI pipelines:
* Integration tests run against `gpt-4o-mini` or equivalent fast models to verify chain dataflow at minimal cost (< $0.001 per test run).
* Traces are sent to a dedicated LangSmith project (`staging-ci-chains`) tagged with Git commit hashes to facilitate post-test inspection.
""")

st.divider()

# ==============================================================================
# SECTION 3: RAG PIPELINES
# ==============================================================================
st.header("3. RAG Pipelines (Vector DB Connectivity, Chunk Ingestion & Grounding Coherence)", divider="rainbow")

st.markdown("""
Integration testing a RAG pipeline requires validating the full lifecycle:
ingesting test documents into a temporary vector store, querying the retriever,
formatting context into prompt templates, and generating a grounded response.
""")

st.markdown("### Critical RAG Integration Failure Points")
st.markdown("""
1. **Index Schema Deserialization**: The vector database returns metadata objects that cannot be parsed by downstream formatting runnables.
2. **Context Window Saturation**: The retriever returns 10 chunks that exceed the staging model's input token capacity.
3. **Loss of Grounding**: The model ignores the retrieved context and answers based on pre-trained parametric knowledge.
""")

st.code('''
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

def test_rag_pipeline_end_to_end_integration():
    # 1. Ephemeral vectorstore for integration test
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    test_docs = [
        Document(page_content="Policy 404: Travel reimbursement per diem is $75 for domestic travel.", metadata={"doc_id": "pol_404"})
    ]
    vectorstore = Chroma.from_documents(test_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # 2. Wire into RAG pipeline
    chain = setup_rag_chain(retriever, ChatOpenAI(model="gpt-4o-mini", temperature=0))
    
    # 3. Invoke and assert factual grounding
    response = chain.invoke("What is the domestic per diem?")
    assert "$75" in response
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production test infrastructure:
* Teams deploy lightweight ephemeral vector store containers (e.g., Chroma or Qdrant in-memory mode) in GitHub Actions test workflows.
* Seeding deterministic documents guarantees that tests evaluate retrieval logic reliably without network flakiness.
""")

st.divider()

# ==============================================================================
# SECTION 4: AGENTS
# ==============================================================================
st.header("4. Agents (Multi-Step ReAct Loops, Tool Selection Fidelity & Recursion Limits)", divider="rainbow")

st.markdown("""
Agent integration tests verify multi-turn interactions.
Unlike simple chains that execute sequentially, an agent decides which tools to call,
observes tool outputs, and iterates until it resolves the task.
""")

st.markdown("### Agent Integration Testing Assertions")
st.markdown("""
* **Tool Selection Fidelity**: Verifying that the model selects the appropriate tool for a given query (e.g., selecting `SearchTool` for recent events and `CalculatorTool` for math).
* **Self-Healing Loop**: Verifying that when a tool returns an error, the agent observes the error and adjusts its arguments or attempts an alternative tool.
* **Recursion Bound Termination**: Verifying that the agent terminates within expected step budgets (e.g., completes within 4 steps and does not trigger `recursion_limit`).
""")

st.code('''
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

def test_react_agent_tool_integration():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(model, tools=[search_tool, calculator_tool])
    
    # Multi-step query requiring search followed by calculation
    inputs = {"messages": [("user", "Find the price of stock X and calculate 15% tax on it.")]}
    result = agent.invoke(inputs)
    
    messages = result["messages"]
    tool_messages = [m for m in messages if m.type == "tool"]
    
    # Assert that tools were invoked in proper sequence
    assert len(tool_messages) >= 2
    assert "final answer" in messages[-1].content.lower()
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith agent traces:
* Integration tests assert on the length and status of the trace tree.
* If an agent takes more than 5 steps on a simple task, the integration test fails for exceeding the step-efficiency SLA.
""")

st.divider()

# ==============================================================================
# SECTION 5: TOOL INTEGRATIONS
# ==============================================================================
st.header("5. Tool Integrations (Live API Sandboxes, Authentication Handshakes & Payload Ingestion)", divider="rainbow")

st.markdown("""
Tools interface with third-party APIs (Stripe, GitHub, Salesforce, Weather services).
Tool integration testing verifies that API credentials, authentication handshakes,
network payload serialization, and rate-limit backoffs function correctly against live sandbox environments.
""")

st.markdown("### Best Practices for Tool Integration Testing")
st.markdown("""
1. **Dedicated Sandbox Credentials**: Never use production API keys in automated integration tests. Always use staging sandbox keys (e.g., Stripe Test Mode `sk_test_...`).
2. **Mocking External Write Operations**: For destructive actions (e.g., deleting a database record or issuing a wire transfer), test tools against mock API endpoints (e.g., WireMock) to prevent accidental real-world side effects.
3. **Idempotency Keys**: Pass unique idempotency keys in test requests to prevent duplicate test records from accumulating in staging databases.
""")

st.code('''
def test_payment_tool_sandbox_integration():
    # Using Stripe test sandbox
    payment_tool = StripePaymentTool(api_key="sk_test_mocked_credential")
    
    result = payment_tool.invoke({
        "customer_id": "cus_test_123",
        "amount_cents": 5000,
        "currency": "usd"
    })
    
    assert result["status"] == "succeeded"
    assert "charge_id" in result
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In CI/CD environments:
* External tool integration tests are tagged with `@pytest.mark.integration` and execute on scheduled nightly builds.
* This prevents external third-party outages from blocking rapid developer pull requests during the day.
""")

st.divider()

# ==============================================================================
# SECTION 6: MULTI-COMPONENT WORKFLOWS
# ==============================================================================
st.header("6. Multi-Component Workflows (Hierarchical Graphs, Shared State Persistence & Subgraph Invocations)", divider="rainbow")

st.markdown("""
In enterprise deployments, applications are rarely flat graphs.
They consist of hierarchical state machines: top-level supervisors dispatching subgraphs
(e.g., a Research Subgraph, a Coding Subgraph, and a QA Subgraph) sharing state through persistent checkpointers.
Integration testing verifies that state channels serialize correctly across subgraph boundaries.
""")

st.markdown("### What to Test in Multi-Component StateGraphs")
st.markdown("""
* **Subgraph Input/Output Channel Mapping**: Verifying that channel values emitted by a child subgraph merge cleanly into the parent state.
* **Checkpointer Persistence Serialization**: Verifying that state snapshots successfully serialize and deserialize from the actual database engine (Postgres/SQLite) without pickling errors.
* **Human-in-the-Loop Resumption**: Verifying that pausing at `interrupt()` and resuming with `Command(resume=...)` preserves channel integrity across processes.
""")

st.code('''
from langgraph.checkpoint.sqlite import SqliteSaver

def test_hierarchical_graph_persistence_integration():
    # Real SQLite checkpointer in temporary directory
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = parent_workflow.compile(checkpointer=checkpointer)
        
        config = {"configurable": {"thread_id": "integ-thread-101"}}
        
        # Step 1: Run to interrupt gate
        app.invoke({"task": "Deploy Kubernetes Cluster"}, config=config)
        state = app.get_state(config)
        assert state.next == ("deployment_approval_gate",)
        
        # Step 2: Resume with approved signal
        resumed_result = app.invoke(Command(resume={"approved": True}), config=config)
        assert resumed_result["status"] == "deployment_complete"
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith multi-component traces:
* Engineers inspect the full hierarchy from the top-level parent graph run down into child subgraph runs.
* Integration assertions verify that metadata tags (e.g., `environment: staging`, `version: 2.4`) propagate seamlessly across all nested spans.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Integration Test Suite & Service Sandbox (IntegrationTestingEngine)", divider="rainbow")

st.markdown("""
Below is the reference integration testing engine.
The `IntegrationTestingEngine` models live cross-component testing across full chains,
RAG pipelines, agents, and multi-component state machines, auditing step-by-step telemetry in LangSmith.
""")

st.code('''
class IntegrationTestingEngine:
    """
    Production Cross-Component Integration Testing Suite for LLM Applications.
    Emulates live boundary testing across five core GenAI integration architectures:
    1. Full Chains (LCEL Runnable pipes: Prompt -> Model -> Parser)
    2. RAG Pipelines (Retriever -> Context Formatter -> LLM Grounding)
    3. Autonomous Agents (Model -> ToolNode -> Self-Correction Loop)
    4. Tool Integrations (Live API sandbox endpoints, authentication tokens)
    5. Multi-Component Workflows (Hierarchical StateGraphs, subgraphs, checkpointers)
    """
    def __init__(self, project_name: str = "llm-integration-testing"):
        self.project_name = project_name
        self.integration_runs: List[Dict[str, Any]] = []

    def execute_integration_test(
        self,
        pipeline_type: str,
        sandbox_environment: str = "staging",
        inject_service_timeout: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a multi-component integration run across simulated external dependencies.
        """
        # Emulates real cross-component data passing and latency profiling
        ...
''', language="python")

st.info("The integration engine is instantiated in module scope and powers the interactive pipeline tester below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Multi-Component Pipeline Integrator & Sandbox Tester", divider="rainbow")

st.markdown("""
Execute cross-component integration tests across different application architectures.
Observe inter-stage data contracts, execution latencies, and toggle external service timeouts
to test error boundary resilience.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Integration Pipeline Controls")

    selected_pipeline = st.selectbox(
        "Select Architecture Under Test:",
        options=["full_chain", "rag_pipeline", "agent_workflow", "multi_component_graph"],
        format_func=lambda p: {
            "full_chain": "1. Full LCEL Chain (Prompt | Model | Parser)",
            "rag_pipeline": "2. RAG Pipeline (VectorDB -> Context -> Model)",
            "agent_workflow": "3. ReAct Agent (Supervisor <-> ToolNode <-> Reducer)",
            "multi_component_graph": "4. Multi-Component StateGraph (Checkpointer & Subgraph)"
        }[p]
    )

    env_choice = st.selectbox(
        "Sandbox Environment:",
        options=["staging-sandbox", "ephemeral-pr-preview", "local-docker"]
    )

    inject_timeout = st.toggle(
        "Simulate Downstream Service Timeout",
        value=False,
        help="Simulates an external API or model endpoint timeout to inspect failure isolation"
    )

    run_integ_btn = st.button("🔌 Execute Integration Test Run", use_container_width=True)

with wb_col2:
    st.subheader("LangSmith Integration Test Trace")

    if run_integ_btn or len(integration_engine.integration_runs) == 0:
        integ_res = integration_engine.execute_integration_test(
            pipeline_type=selected_pipeline,
            sandbox_environment=env_choice,
            inject_service_timeout=inject_timeout
        )
    else:
        integ_res = integration_engine.integration_runs[-1]

    # Metrics Overview
    m1, m2, m3 = st.columns(3)
    status_label = "PASSED (GREEN)" if integ_res["status"] == "passed" else "FAILED (RED)"
    m1.metric("Integration Status", status_label)
    m2.metric("Total Pipeline Latency", f"{integ_res['total_latency_ms']:.1f} ms")
    m3.metric("Spans Audited", len(integ_res["spans"]))

    if integ_res["status"] == "passed":
        st.success("✅ **Integration Verified**: Data contracts, service boundaries, and state transitions passed successfully.")
    else:
        st.error(f"🚨 **Integration Failed**: {integ_res['error_message']}")

    # Spans Table
    st.markdown("#### Cross-Component Execution Waterfall")
    spans_df = pd.DataFrame(integ_res["spans"])
    st.dataframe(spans_df[["step", "name", "component", "status", "latency_ms", "io"]], use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What is the primary operational distinction between Unit Tests and Integration Tests in GenAI?"):
    st.markdown("""
    **Answer:**
    * **Unit Tests**: Verify isolated code components (prompts, parsers, tools) in memory with 100% mocked external dependencies. They run in milliseconds and cost zero API dollars.
    * **Integration Tests**: Verify the live communication boundaries between multiple connected components (e.g., prompt streaming into a real LLM endpoint, vector store network queries, live sandbox API tool calls). They catch serialization errors, rate limit drops, and schema mismatches that mocks hide.
    """)

with st.expander("Q2: How do you prevent integration tests from becoming flaky and expensive in CI/CD?"):
    st.markdown("""
    **Answer:**
    1. **Use Fast, Cost-Effective Models**: Run integration tests against `gpt-4o-mini` or `claude-3-5-haiku` with `temperature=0` to ensure low latency, deterministic completions, and minimal cost.
    2. **Ephemeral In-Memory Sandboxes**: Use in-memory vector stores (e.g., Chroma or Qdrant in-memory mode) pre-seeded with fixed test documents.
    3. **Tagging and Scheduling**: Tag slow tests with `@pytest.mark.integration` and run them on pre-release or nightly schedules rather than on every individual Git commit.
    """)

with st.expander("Q3: What failure modes do RAG integration tests catch that unit tests miss?"):
    st.markdown("""
    **Answer:**
    RAG integration tests catch:
    * **Vector Store Client Deserialization**: Subtle differences between local client models and cloud vector database responses.
    * **Context Overflows**: Document chunk counts that exceed the model's actual context window.
    * **Parametric vs Grounded Hallucination**: Verifying that the model answers strictly using retrieved context rather than falling back to ungrounded training knowledge.
    """)

with st.expander("Q4: How should destructive tools (e.g., database deletions) be handled during integration testing?"):
    st.markdown("""
    **Answer:**
    * **Strict Sandbox Isolation**: Execute against dedicated test sandboxes or staging environments where test data can be safely modified.
    * **Transactional Rollbacks**: Wrap database operations in transactions that automatically roll back upon test completion.
    * **Mock API Proxies (WireMock)**: For third-party SaaS tools that do not provide sandbox environments, use mock HTTP servers that simulate 200 OK responses without executing destructive operations in the external service.
    """)

with st.expander("Q5: How does LangSmith help triage failed integration test runs?"):
    st.markdown("""
    **Answer:**
    When an integration test fails:
    1. LangSmith captures the entire multi-stage trace tree, recording the exact inputs and outputs of every intermediate step.
    2. Engineers can immediately see which specific boundary failed (e.g., did the retriever return empty documents, did the model output invalid JSON, or did the tool endpoint timeout?).
    3. This eliminates the need to add debug print statements and re-run slow test suites.
    """)

st.markdown("### Production Traps & Failure Modes")

it_tr_col1, it_tr_col2 = st.columns(2)

with it_tr_col1:
    st.error("""
    **Trap 1: Running Integration Tests Against Production Databases**
    * **Failure**: Pointing test suites to production vector stores or production customer databases.
    * **Consequence**: Test runs pollute production vector indexes with fake data, or test agents accidentally mutate real customer records.
    * **Remedy**: Enforce strict environment variable isolation; integration tests must only connect to dedicated staging or ephemeral sandboxes.
    """)

    st.error("""
    **Trap 2: Non-Deterministic Model Temperature in Integration Tests**
    * **Failure**: Leaving `temperature=0.7` in integration test configurations.
    * **Consequence**: Stochastically variable completions cause test assertions to fail intermittently, creating 'flaky tests' that developers ignore.
    * **Remedy**: Always set `temperature=0` and use deterministic assertion bounds in integration test pipelines.
    """)

with it_tr_col2:
    st.warning("""
    **Trap 3: Missing Timeout Bounds on External Tool Calls**
    * **Failure**: Calling live third-party tool APIs without explicit HTTP request timeouts.
    * **Consequence**: A hanging third-party endpoint causes the entire CI pipeline to block and freeze for hours.
    * **Remedy**: Enforce strict request timeouts (e.g., `timeout=5.0s`) on all tool HTTP client calls.
    """)

    st.warning("""
    **Trap 4: Testing Full Chains Without Granular Child Spans**
    * **Failure**: Writing monolithic integration tests that only assert on the final output string without instrumenting intermediate spans.
    * **Consequence**: When a test fails, engineers cannot determine whether retrieval, prompt formatting, or model generation caused the failure.
    * **Remedy**: Ensure all integration chains are fully instrumented in LangSmith, enabling instant root-cause localization.
    """)

st.markdown("### Core Technical Flashcards")

fcit_c1, fcit_c2 = st.columns(2)

with fcit_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the key advantage of using `temperature=0` in integration tests?")
        st.markdown("**Back**: It minimizes stochastic variation in model completions, producing deterministic, repeatable outputs that eliminate flaky test failures.")

    with fcit_c2:
        st.markdown("**Front**: Why are ephemeral in-memory vector stores preferred for RAG integration testing?")
        st.markdown("**Back**: They spin up instantly in CI/CD runners, require zero external network dependencies, and can be pre-seeded with fixed test documents for reliable assertions.")

with fcit_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the role of WireMock / Prism in agent tool integration testing?")
        st.markdown("**Back**: They act as mock HTTP proxies, simulating third-party API responses, status codes, and error modes without executing destructive actions against real external systems.")

    with fcit_c2:
        st.markdown("**Front**: What does an integration test verify across an LCEL pipe (`|`)?")
        st.markdown("**Back**: It verifies that the real output type and structure emitted by upstream Component K exactly satisfies the input schema required by downstream Component K+1.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of LLM integration testing principles:")

it_chk1 = st.checkbox("I understand the difference in scope, cost, and speed between Unit and Integration tests.", value=True)
it_chk2 = st.checkbox("I can test full LCEL chains against real staging models with `temperature=0` for determinism.", value=True)
it_chk3 = st.checkbox("I know how to set up ephemeral vector stores to test RAG retrieval and factual grounding.", value=True)
it_chk4 = st.checkbox("I can integration test multi-turn ReAct agents to verify tool selection and recursion bounds.", value=True)
it_chk5 = st.checkbox("I understand how to safely test external tools using sandboxes, idempotency keys, and mock proxies.", value=True)
it_chk6 = st.checkbox("I can use LangSmith trace waterfalls to rapidly pinpoint service boundary failures in CI/CD.", value=True)

if it_chk1 and it_chk2 and it_chk3 and it_chk4 and it_chk5 and it_chk6:
    st.success("Mastery Certified: You have mastered LLM integration testing, service boundary verification, and sandboxed test execution.")
