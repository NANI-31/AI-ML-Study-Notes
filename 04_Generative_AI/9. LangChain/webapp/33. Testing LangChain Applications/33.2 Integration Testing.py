import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="33.2 Integration Testing - LangChain Applications",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for integration test telemetry
st.markdown("""
<style>
    .integ-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge-tier {
        background-color: #1E3A8A;
        color: #BFDBFE;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class IntegrationTestHarnessSimulator:
    """
    Simulates integration testing across multiple physical infrastructure tiers:
    live model endpoints, PostgreSQL databases, ephemeral Chroma vector stores,
    FastAPI HTTP routes, and end-to-end agent trajectories.
    """
    def __init__(self):
        self.tiers = {
            "Model Tier (OpenAI/Anthropic)": {
                "test": "test_live_model_connectivity_and_latency",
                "infra": "Cloud API Endpoint",
                "latency_ms": 340,
                "status": "HEALTHY",
                "detail": "Verified 200 OK, latency < 500ms, fallback provider standby active."
            },
            "Relational DB (PostgreSQL)": {
                "test": "test_sql_agent_transaction_rollback",
                "infra": "Testcontainers / Ephemeral Postgres",
                "latency_ms": 42,
                "status": "HEALTHY",
                "detail": "Created test order, verified schema query, executed rollback cleanly."
            },
            "Vector Database (Chroma / PGVector)": {
                "test": "test_ephemeral_vectorstore_similarity",
                "infra": "In-Memory Ephemeral Chroma",
                "latency_ms": 115,
                "status": "HEALTHY",
                "detail": "Ingested 10 docs, cosine distance verified top-1 match score 0.89."
            },
            "API Service (FastAPI / SSE)": {
                "test": "test_fastapi_sse_streaming_endpoint",
                "infra": "httpx.AsyncClient / TestClient",
                "latency_ms": 85,
                "status": "HEALTHY",
                "detail": "Parsed 14 SSE events (status, tool_start, token, done) successfully."
            },
            "End-to-End System Journey": {
                "test": "test_e2e_customer_support_trajectory",
                "infra": "Full Stack Staging Cluster",
                "latency_ms": 1420,
                "status": "HEALTHY",
                "detail": "User query -> Auth -> Router -> DB Tool -> Final Answer verified."
            }
        }

    def execute_tier_test(self, tier_name: str, simulate_degradation: bool = False) -> Dict[str, Any]:
        data = self.tiers.get(tier_name, self.tiers["Model Tier (OpenAI/Anthropic)"])
        res = dict(data)
        if simulate_degradation:
            res["status"] = "DEGRADED"
            res["latency_ms"] = res["latency_ms"] * 5
            res["detail"] = "Warning: SLA breached. High latency detected during integration call."
        return res


# --- LESSON HEADER ---
st.title("33.2 Integration Testing for LangChain Applications")
st.markdown("""
While unit tests isolate individual functions with mocks, integration tests verify that LangChain systems
function correctly against real infrastructure: live LLM providers, SQL databases, vector databases,
and web APIs. This lesson covers building multi-tier integration test suites, managing ephemeral test infrastructure
with Testcontainers, testing streaming API routes, and verifying end-to-end agent trajectories.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Multi-Tier Integration Testing Architecture", divider="rainbow")
st.markdown("""
Integration testing validates the integration boundaries between the LangChain application layer and external
services. By running automated tests against ephemeral Docker containers (PostgreSQL, Chroma, Redis) and controlled
staging model endpoints, teams detect breaking API changes, schema drifts, and authentication failures before production deployment.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph IntegrationTesting_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_test_runner {
            label = "Integration Test Harness (Pytest + Staging Environment)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            IntegRunner [label="Pytest Integration Suite\\n(pytest tests/integration -m 'integration')", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_ephemeral_infra {
            label = "Ephemeral Infrastructure (Testcontainers / Staging Services)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ModelStaging [label="Staging Model Provider\\n(OpenAI / Azure / Local vLLM)", fillcolor="#1E293B", color="#60A5FA"];
            PostgresContainer [label="Ephemeral Postgres Container\\n(Clean database migration & schema)", fillcolor="#1E293B", color="#60A5FA"];
            VectorContainer [label="Ephemeral VectorDB (Chroma/Qdrant)\\n(Clean temporary collection)", fillcolor="#1E293B", color="#60A5FA"];
            FastAPIClient [label="FastAPI TestClient / httpx.AsyncClient\\n(In-process ASGI HTTP requests)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_langchain_pipelines {
            label = "Live LangChain System Integrations";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            LiveModelChain [label="Model Integration\\n(Prompt -> Live LLM -> Output)", fillcolor="#0369A1", color="#38BDF8"];
            LiveSQLAgent [label="SQL Agent Integration\\n(Natural language -> Real DB execution)", fillcolor="#0369A1", color="#38BDF8"];
            LiveRAGPipeline [label="RAG Vector Integration\\n(Embedding -> Live Vector Search -> Docs)", fillcolor="#0369A1", color="#38BDF8"];
            LiveE2EJourney [label="End-to-End User Trajectory\\n(HTTP Request -> Agent Loop -> SSE Stream)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_verification {
            label = "System Verification & Teardown";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            VerificationEngine [label="Integration Assertions\\n(Latency SLA, schema fidelity, rollback cleanup)", fillcolor="#065F46", color="#10B981"];
        }

        IntegRunner -> LiveModelChain [label="Test Model Connectivity"];
        IntegRunner -> LiveSQLAgent [label="Test DB Tools"];
        IntegRunner -> LiveRAGPipeline [label="Test Vector Search"];
        IntegRunner -> LiveE2EJourney [label="Test Complete Journey"];

        LiveModelChain -> ModelStaging [label="Live API Invocation"];
        LiveSQLAgent -> PostgresContainer [label="Read/Write Transactions"];
        LiveRAGPipeline -> VectorContainer [label="Vector Indexing & KNN"];
        LiveE2EJourney -> FastAPIClient [label="HTTP Request Lifecycle"];

        ModelStaging -> VerificationEngine [label="Assert Token Latency"];
        PostgresContainer -> VerificationEngine [label="Assert DB State & Rollback"];
        VectorContainer -> VerificationEngine [label="Assert Retrieval Precision"];
        FastAPIClient -> VerificationEngine [label="Assert HTTP Status & SSE"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: MODEL INTEGRATION TESTS ---
st.header("2. Model Integration Tests: Live Provider Verification, Latency Benchmarks, and Failover", divider="rainbow")
st.markdown("""
Unlike unit tests that use mocks, model integration tests connect to real model providers (OpenAI, Anthropic, or local vLLM).
These tests verify that API credentials are valid, model versions are supported, request latency adheres to SLAs,
and fallback chains switch automatically when the primary provider encounters an outage.
""")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("### Integration Test Objectives")
    st.markdown("""
    - **Credential & Quota Check**: Confirm API keys have active quotas and permissions for the target model.
    - **Latency Benchmarking**: Ensure time-to-first-token (TTFT) and total generation time fall within acceptable thresholds.
    - **Fallback Validation**: Intentionally inject an invalid model name into the primary chain and assert that `with_fallbacks()` automatically reroutes to the secondary provider.
    """)
with col_m2:
    st.code("""
import pytest
import time
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

@pytest.mark.integration
def test_live_model_connectivity_and_latency():
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    start_time = time.time()
    response = model.invoke("Respond with the single word: PONG")
    latency = time.time() - start_time
    
    assert "PONG" in response.content.upper()
    assert latency < 3.0, f"Latency SLA breached: {latency:.2f}s"

@pytest.mark.integration
def test_model_fallback_switch():
    # Primary model with deliberately invalid model name
    primary = ChatOpenAI(model="nonexistent-gpt-model")
    fallback = ChatOpenAI(model="gpt-4o-mini")
    
    resilient_chain = primary.with_fallbacks([fallback])
    response = resilient_chain.invoke("Ping")
    assert response.content is not None
    """, language="python")

# --- SECTION 3: DATABASE TESTS ---
st.header("3. Relational Database Integration Tests: Transaction Rollbacks, Isolation, and Schema Drift", divider="rainbow")
st.markdown("""
Testing SQL agents requires a live database. Tests must never execute against production.
Instead, test suites use isolated test databases or Testcontainers. To maintain test isolation,
each test executes inside a database transaction that is rolled back at completion, ensuring zero state pollution.
""")

st.code("""
import pytest
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

@pytest.fixture
def test_db_session():
    engine = create_engine("postgresql://postgres:test@localhost:5432/test_db")
    connection = engine.connect()
    trans = connection.begin()
    
    # Seed temporary test table
    connection.execute(text("CREATE TABLE test_users (id INT, name VARCHAR, balance INT)"))
    connection.execute(text("INSERT INTO test_users VALUES (1, 'Alice', 500)"))
    
    yield connection
    
    # Rollback transaction so changes never persist
    trans.rollback()
    connection.close()

@pytest.mark.integration
def test_sql_agent_queries_live_database(test_db_session):
    db = SQLDatabase(create_engine("postgresql://postgres:test@localhost:5432/test_db"))
    agent = create_sql_agent(llm=ChatOpenAI(model="gpt-4o"), db=db, verbose=False)
    
    result = agent.invoke({"input": "What is the balance of Alice in test_users?"})
    assert "500" in result["output"]
""", language="python")

# --- SECTION 4: VECTOR DATABASE TESTS ---
st.header("4. Vector Database Integration Tests: Ephemeral Chroma/PGVector, Indexing, and Search", divider="rainbow")
st.markdown("""
Vector database integration tests verify that embeddings are properly generated, indices are updated,
and nearest-neighbor similarity searches return the correct document chunks.
Using ephemeral instances (such as Chroma in-memory or a Dockerized pgvector) allows tests to run without external dependencies.
""")

st.code("""
import pytest
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

@pytest.mark.integration
def test_vectorstore_indexing_and_similarity():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # In-memory ephemeral vector store
    vectorstore = Chroma(embedding_function=embeddings)
    
    docs = [
        Document(page_content="Kubernetes is an open-source container orchestration engine.", metadata={"category": "devops"}),
        Document(page_content="Kafka is a distributed event streaming platform.", metadata={"category": "data"})
    ]
    
    # Ingest and index
    vectorstore.add_documents(docs)
    
    # Assert semantic retrieval
    results = vectorstore.similarity_search("How to manage Docker containers in clusters?", k=1)
    assert len(results) == 1
    assert "Kubernetes" in results[0].page_content
    assert results[0].metadata["category"] == "devops"
""", language="python")

# --- SECTION 5: API TESTS ---
st.header("5. API Integration Tests: FastAPI TestClient, Header Security, and SSE Streaming Payloads", divider="rainbow")
st.markdown("""
API integration tests verify that the HTTP layer (FastAPI) properly handles authentication headers,
Pydantic payload validation, and Server-Sent Events (SSE) streaming responses.
Using `httpx.AsyncClient` or `starlette.testclient.TestClient`, tests simulate client requests in-process without launching a full web server.
""")

st.code("""
import pytest
import httpx
from fastapi.testclient import TestClient

@pytest.mark.integration
def test_fastapi_chat_stream_sse_endpoint():
    # Import the FastAPI application from the project
    from webapp.api import app
    
    client = TestClient(app)
    
    # Test unauthorized request rejected
    bad_res = client.post("/api/v1/chat/stream", json={"message": "hello", "thread_id": "t1"})
    assert bad_res.status_code == 401
    
    # Test valid streaming request
    headers = {"Authorization": "Bearer valid_jwt_token_enterprise"}
    with client.stream("POST", "/api/v1/chat/stream", json={"message": "ping", "thread_id": "t1"}, headers=headers) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        events = []
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(line)
                
        assert len(events) > 0
        assert any("[DONE]" in e or "token" in e for e in events)
""", language="python")

# --- SECTION 6: END-TO-END TESTS ---
st.header("6. End-to-End System Tests: Synthetic Multi-Step User Journeys & State Trajectories", divider="rainbow")
st.markdown("""
End-to-end (E2E) tests validate the complete user journey: a user submits an ambiguous inquiry,
the agent queries knowledge bases and external tools, requests clarification if necessary,
and produces the final grounded response. E2E tests are slower and more expensive, so they are run
on dedicated nightly or pre-release schedules rather than every pull request.
""")

st.code("""
import pytest

@pytest.mark.e2e
def test_e2e_customer_inquiry_trajectory():
    \"\"\"Simulates a multi-turn customer support journey across multiple agent turns.\"\"\"
    thread_id = "test_thread_e2e_44"
    
    # Turn 1: User asks vague question
    turn1 = customer_agent.invoke(
        {"messages": [{"role": "user", "content": "I want to return an order."}]},
        config={"configurable": {"thread_id": thread_id}}
    )
    # Assert agent asks for order number
    assert "order number" in turn1["messages"][-1].content.lower()
    
    # Turn 2: User provides order number
    turn2 = customer_agent.invoke(
        {"messages": [{"role": "user", "content": "My order number is #ORD-9912."}]},
        config={"configurable": {"thread_id": thread_id}}
    )
    # Assert agent executed lookup tool and confirmed status
    assert any("order_lookup" in str(msg) for msg in turn2["messages"])
""", language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Full-Scale Integration Test Suite with Ephemeral Testcontainers", divider="rainbow")
st.markdown("""
Below is a production-ready integration test configuration file using `pytest`, marking tests with custom markers,
and orchestrating ephemeral infrastructure setup and teardown.
""")

st.code("""
# tests/conftest.py
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def postgres_container():
    \"\"\"Spins up a lightweight ephemeral PostgreSQL container for the test session.\"\"\"
    with PostgresContainer("postgres:16-alpine") as postgres:
        db_url = postgres.get_connection_url()
        # Initialize test schema migrations using db_url
        yield postgres


# pyproject.toml marker configuration:
# [tool.pytest.ini_options]
# markers = [
#     "unit: Fast deterministic unit tests with mocks",
#     "integration: Multi-tier tests against staging services",
#     "e2e: End-to-end synthetic user journeys"
# ]
""", language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Multi-Service Integration Test Harness Simulator", divider="rainbow")
st.markdown("""
Execute live integration test simulations across the multi-tier infrastructure stack.
Evaluate response latency, tier health, and SLA conformance.
""")

harness = IntegrationTestHarnessSimulator()

col_cfg, col_out = st.columns([1, 1])

with col_cfg:
    st.subheader("🔌 Integration Target Selector")
    chosen_tier = st.selectbox(
        "Select Infrastructure Tier to Test:",
        list(harness.tiers.keys())
    )
    degrade_toggle = st.checkbox("Simulate Network Degradation / High Latency", value=False)
    run_btn = st.button("🚀 Run Tier Integration Test", type="primary")

with col_out:
    st.subheader("📊 Integration Telemetry")
    if run_btn:
        res = harness.execute_tier_test(chosen_tier, simulate_degradation=degrade_toggle)
        status_color = "#10B981" if res["status"] == "HEALTHY" else "#EF4444"
        
        st.markdown(
            f"<div class='integ-card'>"
            f"<span class='badge-tier'>{res['infra']}</span><br><br>"
            f"<b>Test Target:</b> <code>{res['test']}</code><br>"
            f"<b>Latency:</b> {res['latency_ms']} ms &nbsp;|&nbsp; <b>Status:</b> <span style='color:{status_color};font-weight:bold'>{res['status']}</span><br><br>"
            f"<b>Result Details:</b> {res['detail']}"
            f"</div>",
            unsafe_allow_html=True
        )
        if res["status"] == "HEALTHY":
            st.success("Integration boundary verified: live connection and contract upheld.")
        else:
            st.error("Integration SLA breach: High latency or service degradation detected.")
    else:
        st.info("Select a tier and execute to observe integration test telemetry.")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: What is the primary difference in role between unit tests and integration tests in LangChain systems?"):
    st.markdown("""
    **Answer**: Unit tests isolate application plumbing (prompts, schemas, parsers) using deterministic mocks, executing in milliseconds
    without external dependencies. Integration tests verify that the system works with real external services (live LLM providers, relational
    databases, vector databases, and API gateways), validating authentication, network resilience, database schemas, and protocol compliance.
    """)

with st.expander("Q2: How do you prevent integration tests from polluting database state?"):
    st.markdown("""
    **Answer**: Use transactional rollbacks in test fixtures (`trans.rollback()`). Each test runs within a transaction that inserts, updates,
    or queries data, and is rolled back when the test finishes. Alternatively, use ephemeral Docker containers via Testcontainers that are
    created and destroyed automatically with each test session.
    """)

with st.expander("Q3: How should live LLM API costs and rate limits be managed during continuous integration?"):
    st.markdown("""
    **Answer**: Separate test suites using pytest markers (`@pytest.mark.unit` vs `@pytest.mark.integration`). Unit tests run on every pull request
    or commit without live API calls. Integration and E2E tests run on a schedule (nightly or before release merges) against a dedicated staging
    API key with strict spending limits and rate-limit allocations.
    """)

with st.expander("Q4: How do you test Server-Sent Events (SSE) streaming endpoints in FastAPI integration tests?"):
    st.markdown("""
    **Answer**: Use `httpx.AsyncClient` or `TestClient.stream()` with a context manager (`with client.stream("POST", ...) as response:`).
    Iterate over the lines using `response.iter_lines()`, verify the HTTP 200 status code and `content-type: text/event-stream` header,
    and assert that events are formatted with `data:` prefixes and contain the expected JSON payload types (tokens, tool calls, and completion indicators).
    """)

with st.expander("Q5: What is Testcontainers and why is it valuable for LangChain integration testing?"):
    st.markdown("""
    **Answer**: Testcontainers is a library that allows Python test suites to programmatically launch and tear down ephemeral Docker containers
    (PostgreSQL, Redis, Qdrant, Chroma). It gives tests a completely clean, isolated, and disposable environment matching production specifications
    without needing shared, stateful staging databases.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Running Live Integration Tests on Every Git Push)**: Running live model tests on every commit causes developers to hit OpenAI rate limits and inflates cloud costs. Gate live tests to staging branches or nightlies.")
    st.warning("**Trap 2 (Testing Against Shared Databases)**: Running tests against a shared development database. Concurrent test runs overwrite each other's data, causing non-deterministic test failures.")
with col_trap2:
    st.error("**Trap 3 (Hardcoded Secrets in Test Files)**: Committing live API keys and database passwords directly into test files. Always inject credentials via environment variables or secret vaults.")
    st.warning("**Trap 4 (Ignoring Model Version Deprecations)**: Hardcoding snapshot model versions (e.g. `gpt-4-0314`) that are subsequently deprecated by providers, causing integration suites to fail unexpectedly months later.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: How do you prevent test database pollution in SQL tests?")
        st.caption("Flip for answer:")
        st.info("**Back**: Execute tests inside transactions and trigger `rollback()` at test teardown.")
    with st.container(border=True):
        st.markdown("**Front**: What pytest feature allows separating fast unit tests from live tests?")
        st.caption("Flip for answer:")
        st.info("**Back**: Pytest markers (e.g. `@pytest.mark.unit` vs `@pytest.mark.integration`).")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How do you test FastAPI streaming endpoints in integration tests?")
        st.caption("Flip for answer:")
        st.info("**Back**: Use `client.stream('POST', ...)` and iterate over `iter_lines()` to verify SSE events.")
    with st.container(border=True):
        st.markdown("**Front**: What tool spins up disposable Docker databases for tests?")
        st.caption("Flip for answer:")
        st.info("**Back**: `testcontainers-python` creates ephemeral Docker containers managed by pytest fixtures.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the distinction between unit, integration, and end-to-end testing tiers.", value=True)
c2 = st.checkbox("I can write live model integration tests verifying latency and fallback switches.", value=True)
c3 = st.checkbox("I know how to test SQL agents using database transactional rollbacks.", value=True)
c4 = st.checkbox("I can test vector database similarity search using ephemeral Chroma instances.", value=True)
c5 = st.checkbox("I know how to test FastAPI Server-Sent Events streaming routes with TestClient.", value=True)
c6 = st.checkbox("I understand how to configure pytest markers to control live API cost and rate limits.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 33.3: LLM Testing.")
