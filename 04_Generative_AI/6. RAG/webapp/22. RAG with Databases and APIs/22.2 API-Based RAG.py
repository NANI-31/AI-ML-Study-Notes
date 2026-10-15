import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="22.2 API-Based RAG",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class APIRAGDispatcherEngine:
    """Enterprise API-Based RAG Dispatcher: Orchestrates REST, GraphQL, Search & External Knowledge APIs."""

    def __init__(self):
        # Mock API Catalogs & Connectors
        self.api_catalog = {
            "REST API (Stripe Billing)": {
                "endpoint": "https://api.stripe.com/v1/customers/cus_9821/invoices",
                "protocol": "REST / JSON",
                "auth_type": "Bearer OAuth2",
                "avg_latency_ms": 68.0,
                "raw_response": {"customer_id": "cus_9821", "status": "paid", "amount_due": 0, "currency": "usd", "subscription": "enterprise_tier_v2"}
            },
            "GraphQL API (GitHub Enterprise)": {
                "endpoint": "https://api.github.com/graphql",
                "protocol": "GraphQL Query",
                "auth_type": "Personal Access Token",
                "avg_latency_ms": 95.0,
                "raw_response": {"repository": {"name": "core-engine", "openPullRequests": {"totalCount": 14}, "securityVulnerabilities": {"criticalCount": 0}}}
            },
            "Search API (Tavily Real-Time)": {
                "endpoint": "https://api.tavily.com/search",
                "protocol": "REST Search API",
                "auth_type": "API Key",
                "avg_latency_ms": 280.0,
                "raw_response": {"query": "NVIDIA Blackwell B200 availability", "results": [{"title": "NVIDIA Commences Volume Shipments", "score": 0.94, "snippet": "Datacenter systems powered by GB200 NVL72 are deploying in hyper-scale clusters."}]}
            },
            "External Knowledge API (Wolfram Alpha)": {
                "endpoint": "https://api.wolframalpha.com/v2/query",
                "protocol": "Knowledge API / Structured Pods",
                "auth_type": "AppID Token",
                "avg_latency_ms": 150.0,
                "raw_response": {"query": "power density calculation", "pods": [{"title": "Result", "plaintext": "10.2 kW / 42U rack = 242.8 W per rack unit."}]}
            }
        }

    def dispatch_api_call(self, api_name: str, query_args: str) -> Dict[str, Any]:
        """Simulates API parameterization, execution, and payload flattening."""
        api_info = self.api_catalog.get(api_name)
        if not api_info:
            return {"status": "ERROR", "message": "Unknown API endpoint."}

        # Payload Flattening & Text Serialization
        raw = api_info["raw_response"]
        flattened_text = " | ".join([f"{k}: {v}" for k, v in raw.items()])

        return {
            "api_name": api_name,
            "endpoint": api_info["endpoint"],
            "protocol": api_info["protocol"],
            "latency_ms": api_info["avg_latency_ms"],
            "raw_payload": raw,
            "flattened_text": flattened_text
        }


# --- Streamlit Layout ---

st.title("22.2 API-Based RAG Architecture")
st.caption("Structured Tool Calling: REST Endpoints, GraphQL Introspection, Real-Time Search APIs & Computational Knowledge Services")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise API-Based RAG Dispatcher & Tool Calling Hub", divider="rainbow")

st.markdown(
    """
    In modern digital enterprises, live operational state does not live in static PDF files or even in internal vector stores.
    Live customer balances, open security vulnerability tickets, real-time breaking news, and mathematical conversions
    require communicating with **External APIs** via structured tool calling.
    
    The diagram below models the complete **API-Based RAG Topology**, detailing how language models inspect OpenAPI/GraphQL schemas,
    parameterize network requests, authenticate securely, flatten nested JSON payloads into canonical prompt context, and synthesize live answers.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph API_RAG_Architecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_orchestrator {
            label = "1. TOOL CALLING REASONER (ORCHESTRATOR)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_prompt [label="User Query: 'Check customer billing status & active PR count'", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            api_dispatcher [label="API Tool Dispatcher (ReAct State Machine)\\nSelects & Parameterizes Heterogeneous APIs", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            
            user_prompt -> api_dispatcher;
        }

        subgraph cluster_heterogeneous_apis {
            label = "2. HETEROGENEOUS API ENDPOINTS";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            rest_endpoint [label="1. REST API (OpenAPI)\\nJSON / OAuth2 (Stripe Billing)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            graphql_endpoint [label="2. GraphQL API (Introspection)\\nField Pruning & Graph Queries (GitHub)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            search_endpoint [label="3. Search API (Tavily / Serper)\\nReal-Time Web Document Extraction", fillcolor="#0F172A", fontcolor="#38BDF8"];
            knowledge_endpoint [label="4. External Knowledge API\\nWolfram Alpha / Computational Engine", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            api_dispatcher -> rest_endpoint [label="HTTP GET / Invoices"];
            api_dispatcher -> graphql_endpoint [label="POST / GraphQL Query"];
            api_dispatcher -> search_endpoint [label="POST / Web Search"];
            api_dispatcher -> knowledge_endpoint [label="GET / Computation"];
        }

        subgraph cluster_payload_normalizer {
            label = "3. PAYLOAD NORMALIZATION & RATE LIMIT GATEWAY";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            normalizer_gateway [label="Payload Flattening & Circuit Breaker\\n- Strips JSON Nesting & Identifiers\\n- Enforces 429 Exponential Backoff\\n- Masks Auth Tokens & Secrets", fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            rest_endpoint -> normalizer_gateway;
            graphql_endpoint -> normalizer_gateway;
            search_endpoint -> normalizer_gateway;
            knowledge_endpoint -> normalizer_gateway;
        }

        subgraph cluster_synthesis {
            label = "4. LIVE GROUNDED SYNTHESIS";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            llm_synthesizer [label="Frontier LLM Synthesizer\\nIntegrates Live Dynamic State into Final Answer", fillcolor="#1E3A8A", fontcolor="#10B981"];
            verified_response [label="Live Operational Response\\nWith Exact API Provenance Attribution", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            normalizer_gateway -> llm_synthesizer;
            user_prompt -> llm_synthesizer;
            llm_synthesizer -> verified_response;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: REST APIS ---
st.header("2. REST APIs (OpenAPI Specifications, Path Parameterization, JSON Flattening & Status Handling)", divider="rainbow")

st.markdown(
    """
    REST represents the dominant protocol for enterprise web services:
    
    - **OpenAPI / Swagger Inlining**: Supplying the LLM with structured endpoint definitions, including path parameters (`/v1/customers/{id}`), query filters, and HTTP verbs.
    - **JSON Flattening**: Raw API responses contain deeply nested dictionaries and array wrappers. Ingestion pipelines flatten JSON hierarchies into clean key-value bullet points.
    - **Status Code Triage**: Gracefully intercepting HTTP `404 Not Found` (entity missing), `401 Unauthorized` (expired bearer token), and `429 Too Many Requests` (rate limited).
    """
)

# --- SECTION 3: GRAPHQL APIS ---
st.header("3. GraphQL APIs (Schema Introspection, Field Selection, Nested Graph Traversal & Query Pruning)", divider="rainbow")

st.markdown(
    """
    Unlike REST APIs that return static fixed payloads, **GraphQL** allows the caller to request the exact fields needed:
    
    - **Schema Introspection**: Language models inspect GraphQL type definitions to formulate syntactically valid queries.
    - **Field Pruning & Token Protection**: Rather than downloading 500 fields, the model requests only `{ name, status, openPRs }`, cutting prompt payload volume by over 80%.
    - **Multi-Hop Relational Traversal**: A single GraphQL query can traverse from a customer record to their active contracts and open support tickets without multiple network round-trips.
    """
)

# --- SECTION 4: SEARCH APIS ---
st.header("4. Search APIs (Serper, Tavily, Bing Search Connectors & Real-Time Web Scraping)", divider="rainbow")

st.markdown(
    """
    Search APIs connect RAG pipelines to the public internet:
    
    - **Engine Connectors (Tavily / Serper)**: Specialized APIs designed for LLMs that return clean, scraped article text rather than ad-laden raw HTML.
    - **Temporal Freshness Filtering**: Restricting searches to `time_range="past_24h"` or `past_week` to prevent retrieval of outdated press releases.
    - **Snippet Reranking**: Pruning the top 10 search results to the single most relevant paragraph before context insertion.
    """
)

# --- SECTION 5: EXTERNAL KNOWLEDGE APIS ---
st.header("5. External Knowledge APIs (Wolfram Alpha, Fact-Checking Engines, Financial Market Data Feeds)", divider="rainbow")

st.markdown(
    """
    Certain enterprise inquiries require **computational precision** that even frontier LLMs cannot perform reliably:
    
    - **Computational Engines (Wolfram Alpha)**: Handles complex unit conversions (*'10.2 kW per 42U rack into BTU/hr'*), differential calculus, and chemical formulas.
    - **Live Market Data Feeds**: Pulls real-time stock prices, foreign exchange rates, and bond yields from Bloomberg or Refinitiv.
    - **Fact-Checking Knowledge Bases**: Queries verified biomedical registries (PubMed, ClinVar) to audit drug contraindications.
    """
)

# --- SECTION 6: PRODUCTION IMPLEMENTATION ---
st.header("6. Production Implementation: Enterprise API Tool Calling & Dispatch Orchestrator", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `APIRAGDispatcherEngine`.
    It registers heterogeneous API connectors across REST, GraphQL, Real-Time Search, and Computational Knowledge APIs,
    simulating tool parameterization, execution, and JSON payload flattening.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class APIRAGDispatcherEngine:
    """Enterprise API-Based RAG Dispatcher."""

    def __init__(self):
        self.api_catalog = {
            "REST API (Stripe Billing)": {
                "endpoint": "https://api.stripe.com/v1/customers/cus_9821/invoices",
                "protocol": "REST / JSON",
                "avg_latency_ms": 68.0,
                "raw_response": {"customer_id": "cus_9821", "status": "paid", "amount_due": 0, "currency": "usd", "subscription": "enterprise_tier_v2"}
            },
            "GraphQL API (GitHub Enterprise)": {
                "endpoint": "https://api.github.com/graphql",
                "protocol": "GraphQL Query",
                "avg_latency_ms": 95.0,
                "raw_response": {"repository": {"name": "core-engine", "openPullRequests": {"totalCount": 14}, "securityVulnerabilities": {"criticalCount": 0}}}
            },
            "Search API (Tavily Real-Time)": {
                "endpoint": "https://api.tavily.com/search",
                "protocol": "REST Search API",
                "avg_latency_ms": 280.0,
                "raw_response": {"query": "NVIDIA Blackwell B200 availability", "results": [{"title": "NVIDIA Commences Volume Shipments", "score": 0.94, "snippet": "Datacenter systems powered by GB200 NVL72 are deploying in hyper-scale clusters."}]}
            },
            "External Knowledge API (Wolfram Alpha)": {
                "endpoint": "https://api.wolframalpha.com/v2/query",
                "protocol": "Knowledge API / Structured Pods",
                "avg_latency_ms": 150.0,
                "raw_response": {"query": "power density calculation", "pods": [{"title": "Result", "plaintext": "10.2 kW / 42U rack = 242.8 W per rack unit."}]}
            }
        }

    def dispatch_api_call(self, api_name: str, query_args: str) -> Dict[str, Any]:
        """Simulates API parameterization, execution, and payload flattening."""
        api_info = self.api_catalog.get(api_name)
        if not api_info:
            return {"status": "ERROR", "message": "Unknown API endpoint."}

        raw = api_info["raw_response"]
        flattened_text = " | ".join([f"{k}: {v}" for k, v in raw.items()])

        return {
            "api_name": api_name,
            "endpoint": api_info["endpoint"],
            "protocol": api_info["protocol"],
            "latency_ms": api_info["avg_latency_ms"],
            "raw_payload": raw,
            "flattened_text": flattened_text
        }
''',
    language="python"
)

# --- SECTION 7: INTERACTIVE WORKBENCH ---
st.header("7. Interactive Workbench: Multi-API Dynamic Dispatch & JSON Payload Transformation Sandbox", divider="rainbow")

st.markdown(
    """
    Test dynamic tool dispatch across enterprise REST, GraphQL, Search, and Knowledge APIs.
    Select an API connector, submit query parameters, and inspect the raw response payload alongside flattened prompt representations.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🔌 API Selection & Tool Calling")
    
    selected_api = st.selectbox(
        "Select Enterprise API Tool to Dispatch:",
        [
            "REST API (Stripe Billing)",
            "GraphQL API (GitHub Enterprise)",
            "Search API (Tavily Real-Time)",
            "External Knowledge API (Wolfram Alpha)"
        ],
        index=0
    )
    
    query_param = st.text_input("Simulated Tool Argument / Filter:", "target_id = 'cus_9821'")
    
    engine = APIRAGDispatcherEngine()
    result = engine.dispatch_api_call(selected_api, query_param)
    
    st.markdown("#### API Endpoint Telemetry")
    st.info(
        f"**Protocol**: `{result['protocol']}`  \n"
        f"**Endpoint**: `{result['endpoint']}`  \n"
        f"**Network Latency**: `{result['latency_ms']} ms`"
    )

with wb_col2:
    st.subheader("📦 Payload Transformation")
    
    m1, m2 = st.columns(2)
    m1.metric("Tool Called", result["api_name"].split(" (")[0])
    m2.metric("Response Latency", f"{result['latency_ms']} ms")
    
    st.markdown("#### Flattened Representation (Injected to Context)")
    st.success(result["flattened_text"])
    
    st.markdown("#### Raw API Response Payload")
    st.write(result["raw_payload"])

st.markdown("### Enterprise API Capabilities Matrix")
cat_df = pd.DataFrame([
    {
        "Tool Name": k,
        "Target Endpoint": v["endpoint"],
        "Protocol": v["protocol"],
        "Expected Latency": f"{v['avg_latency_ms']} ms",
        "Primary Use Case": "Billing & Subscriptions" if "REST" in k else ("Code & Security PRs" if "GraphQL" in k else ("Real-Time News" if "Search" in k else "Computational Math"))
    }
    for k, v in engine.api_catalog.items()
])
st.dataframe(cat_df, use_container_width=True)

# --- SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: Why is JSON payload flattening critical before feeding API responses to language models?**"):
    st.markdown(
        """
        Raw enterprise REST/GraphQL responses contain deep JSON nesting, empty arrays, redundant timestamps, and system UUIDs.
        - **Context Bloat**: A 2,000-line JSON payload might contain only 3 relevant numbers, wasting thousands of tokens.
        - **Attention Dilution**: Braces, quotation marks, and bracket punctuation degrade transformer attention weights on factual entity values.
        
        **Payload Flattening** strips syntax boilerplate, converting payloads into human-readable strings (*'Customer: Acme Corp | Status: Active | Balance: $0'*).
        """
    )

with st.expander("**Q2: How does GraphQL field pruning optimize RAG context windows compared to REST?**"):
    st.markdown(
        """
        REST endpoints suffer from **over-fetching**: calling `/v1/users/42` returns 80 fields (profile URLs, tracking preferences, legacy timestamps) when the user only asked for the user's role.
        
        **GraphQL** allows the LLM to specify the exact fields required in its generated query:
        `query { user(id: 42) { role, department } }`.
        This eliminates 90%+ of irrelevant fields before the payload traverses the network, preserving context tokens.
        """
    )

with st.expander("**Q3: How do you handle authentication securely in API-based RAG pipelines?**"):
    st.markdown(
        """
        Never expose raw API keys or access tokens directly inside LLM system prompts.
        
        Production pipelines use **Server-Side Tool Gateways**:
        1. The LLM emits only high-level tool calls and arguments (`{"tool": "get_customer_invoice", "id": "123"}`).
        2. The backend application intercepts the tool call, fetches short-lived OAuth2 bearer tokens from a secret manager (HashiCorp Vault / AWS Secrets Manager), and executes the HTTP request.
        3. Raw credentials never touch the model's context or client-facing logs.
        """
    )

with st.expander("**Q4: What circuit breaker mechanisms prevent external API downtime from crashing RAG pipelines?**"):
    st.markdown(
        """
        Third-party APIs suffer from outages, rate limits, and latency spikes.
        
        Production systems deploy **Circuit Breakers**:
        - **Timeout Bounds**: If an API fails to respond within 1,500ms, abort the request.
        - **State Tripping**: If > 5% of requests fail within 1 minute, trip the circuit to 'OPEN', bypassing the API and falling back to cached responses or a static disclaimer.
        - **Exponential Backoff**: Respecting `Retry-After` headers on HTTP `429` responses.
        """
    )

with st.expander("**Q5: When should a RAG system invoke Wolfram Alpha over standard Python code execution?**"):
    st.markdown(
        """
        While Python handles generic arithmetic, **Wolfram Alpha** provides a curated knowledge base of scientific facts, physical constants, and unit conversions.
        - Questions requiring physical unit harmonization (*'convert 10 kW into BTUs/hour'*).
        - Queries requiring symbolic algebra, differential equations, or astronomical distances.
        Wolfram Alpha computes answers with mathematically proven accuracy, preventing computational hallucinations.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Stuffing Raw 10,000-Line JSON Payloads into Context**  
        **Antipattern**: Ingesting raw JSON API dumps without pruning or flattening.  
        **Consequence**: Consumes 90% of the token window with braces and metadata, causing lost-in-the-middle context dilution.  
        **Production Fix**: Implement schema-aware JSON sanitizers that extract only requested keys.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Leaking Bearer Tokens in Prompt Traces**  
        **Antipattern**: Passing Authorization headers or API keys inside prompt templates or trace logs.  
        **Consequence**: Leaks production credentials into external observability tools and violates SOC-2 compliance.  
        **Production Fix**: Inject credentials securely at the backend gateway tier outside LLM visibility.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Synchronous Blocking on Unresponsive Third-Party APIs**  
        **Antipattern**: Waiting indefinitely for slow external search or knowledge APIs to respond.  
        **Consequence**: A single degraded API stalls the entire application, breaching user latency SLOs.  
        **Production Fix**: Enforce strict 1,200ms connection timeouts and graceful fallback degradation.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Missing HTTP 429 Rate Limit Interceptors**  
        **Antipattern**: Looping tool calls without checking rate limit headers.  
        **Consequence**: Triggers temporary IP bans from external SaaS providers during traffic surges.  
        **Production Fix**: Wrap API calls in token-bucket rate limiters with exponential backoff.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is API-Based RAG?")
        st.markdown("**Back**: An architecture where the LLM dynamically calls external REST, GraphQL, and search endpoints via structured function calling to access real-time operational data.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is Payload Flattening?")
        st.markdown("**Back**: The process of stripping syntax overhead, braces, and empty fields from raw JSON API responses to produce token-efficient, human-readable text.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: Why is GraphQL beneficial for RAG?")
        st.markdown("**Back**: Allows the language model to request only the specific fields needed, eliminating over-fetching and slashing prompt token consumption.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is a Tool Calling Gateway?")
        st.markdown("**Back**: A secure backend proxy that injects credentials, validates arguments, enforces timeouts, and executes API calls on behalf of the language model.")

# --- SECTION 9: SELF-ASSESSMENT CHECKLIST ---
st.header("9. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of API-based RAG architectures and structured tool dispatch:")

chk1 = st.checkbox("I understand the architectural workflow of API-based tool calling across REST, GraphQL, and Search APIs.", value=True)
chk2 = st.checkbox("I know how to parse OpenAPI schemas to enable automated tool parameterization.", value=True)
chk3 = st.checkbox("I can implement JSON payload flattening to strip syntax boilerplate and protect context token budgets.", value=True)
chk4 = st.checkbox("I understand how GraphQL field selection prevents over-fetching and token bloat.", value=True)
chk5 = st.checkbox("I know how to secure API credentials using server-side tool gateways and secret managers.", value=True)
chk6 = st.checkbox("I can design circuit breakers and rate-limit interceptors to handle external API latency and outages.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Enterprise API-Based RAG and Tool Calling.")
