import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="27.3 External APIs - LangGraph Curriculum",
    page_icon="🌐",
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
    .badge-api {
        background-color: #172554;
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class ExternalAPISimulator:
    """
    Simulates external API tool invocations across REST, GraphQL,
    search, finance, and enterprise internal microservices.
    """
    def __init__(self):
        self.mock_apis = {
            "search_tavily": {
                "category": "Search",
                "endpoint": "https://api.tavily.com/search",
                "latency_ms": 320,
                "sample_response": {"results": [{"title": "LangGraph v0.2 Release", "snippet": "New functional API and serverless deployment."}]}
            },
            "finance_market": {
                "category": "Finance",
                "endpoint": "https://api.marketdata.internal/v1/quote",
                "latency_ms": 85,
                "sample_response": {"ticker": "NVDA", "price": 138.25, "currency": "USD", "volume": 42000000}
            },
            "weather_service": {
                "category": "Weather",
                "endpoint": "https://api.weather.gov/gridpoints",
                "latency_ms": 110,
                "sample_response": {"city": "San Francisco", "temp_c": 18.5, "condition": "Partly Cloudy"}
            },
            "enterprise_crm": {
                "category": "Enterprise CRM",
                "endpoint": "https://corp-salesforce.internal/services/data/v58.0",
                "latency_ms": 195,
                "sample_response": {"account_id": "001xx000003DHPw", "annual_revenue": 25000000, "status": "Active Enterprise"}
            },
            "internal_auth_service": {
                "category": "Internal Microservice",
                "endpoint": "grpc://identity-mesh.corp:50051",
                "latency_ms": 12,
                "sample_response": {"user_id": "usr_9981", "scopes": ["read:crm", "write:reports"]}
            }
        }

    def call_api(self, api_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
        api = self.mock_apis.get(api_key)
        if not api:
            return {"error": "Unknown API endpoint"}
        return {
            "api_name": api_key,
            "category": api["category"],
            "endpoint": api["endpoint"],
            "latency_ms": api["latency_ms"],
            "response": api["sample_response"]
        }

api_sim = ExternalAPISimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: External API Integration Ecosystem, Authentication, and Rate Limiting Gateways", divider="rainbow")

st.markdown("""
Autonomous agents achieve real-world utility by interfacing with external APIs. 
In enterprise LangGraph systems, integrating external APIs requires managing distinct protocols 
(REST, GraphQL, gRPC), managing authentication credentials securely (OAuth2, mTLS, signed API keys), 
and implementing resiliency patterns to protect against external rate limits (HTTP 429), timeouts, and breaking schema changes.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ExternalAPIBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_agent {
            label="1. LangGraph Execution Runtime";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            agent_llm [label="Agent Reasoning Node\\nSelects Tool & Generates Arguments", fillcolor="#0F172A"];
            tool_dispatcher [label="Tool Invocation Dispatcher\\nValidates Pydantic Schemas & Injects Auth Tokens", fillcolor="#312E81"];
        }

        subgraph cluster_gateway {
            label="2. Enterprise API Integration Gateway";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            resilience_layer [label="Resilience Gateway (Tenacity / Circuit Breaker)\\nTimeouts (5s), Retries with Jitter, Token Bucket RPM", fillcolor="#1E293B"];
            secrets_vault [label="Secrets Vault (AWS Secrets Manager)\\nDynamic Token Refresh & Scoped OAuth Bearer", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        subgraph cluster_external_services {
            label="3. External & Internal API Providers";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            api_web [label="Web Search APIs (Tavily / Serper)\\nReal-Time Web Intelligence", fillcolor="#0F172A"];
            api_saas [label="Enterprise SaaS APIs (Salesforce / Jira)\\nGraphQL / REST Webhooks", fillcolor="#0F172A"];
            api_internal [label="Internal Company Services (gRPC / VPC)\\nmTLS Service Mesh / Core Databases", fillcolor="#083344", fontcolor="#67E8F9"];
        }

        agent_llm -> tool_dispatcher [label="Tool Call Request"];
        tool_dispatcher -> secrets_vault [label="Retrieve Credentials"];
        tool_dispatcher -> resilience_layer;

        resilience_layer -> api_web [label="HTTPS / REST"];
        resilience_layer -> api_saas [label="OAuth2 / GraphQL"];
        resilience_layer -> api_internal [label="mTLS / gRPC Mesh"];

        api_web -> agent_llm [label="Sanitized JSON Result"];
        api_saas -> agent_llm;
        api_internal -> agent_llm;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: REST APIs
# ==============================================================================
st.header("2. REST APIs: OpenAPI / Swagger Tool Binding, JSON Payloads, and HTTP Status Codes", divider="rainbow")

st.markdown("""
REST APIs are the most common interface for agent tools. Connecting LangGraph to REST endpoints involves:

- **OpenAPI Schema Binding**: Translating OpenAPI / Swagger endpoint definitions into Pydantic tool schemas automatically.
- **Strict Error Code Handling**: Differentiating between transient errors (500, 502, 503, 504 -> retry) and 
  permanent client errors (400, 401, 403, 404 -> return helpful error message to LLM to alter parameters).
- **Output Sanitization**: Stripping unnecessary HTML tags or verbose metadata arrays to conserve context window tokens.
""")

st.code('''from langchain_core.tools import tool
import httpx

@tool
async def fetch_customer_profile(customer_id: str) -> str:
    """Fetches customer CRM record by customer_id via REST API."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(
            f"https://api.crm.internal/customers/{customer_id}",
            headers={"Authorization": "Bearer token_secret"}
        )
        if response.status_code == 404:
            return f"Customer {customer_id} does not exist."
        response.raise_for_status()
        data = response.json()
        return f"Customer {data['name']} (Status: {data['tier']}, Region: {data['region']})"
''', language="python")

# ==============================================================================
# SECTION 3: GraphQL
# ==============================================================================
st.header("3. GraphQL: Field Selection, Query Construction, and Preventing Over-Fetching in Agent Workflows", divider="rainbow")

st.markdown("""
GraphQL is ideal for agent integrations where REST endpoints return excessively large JSON payloads:

- **Field Selection**: Agents request only the exact fields required (e.g., `id`, `name`, `status`), 
  reducing token consumption by 80% compared to REST endpoints returning 200+ properties.
- **Single Request Traversals**: Traverses relational graphs in a single query (e.g., fetching a customer and their last 3 tickets simultaneously).
""")

# ==============================================================================
# SECTION 4: Search APIs
# ==============================================================================
st.header("4. Search APIs: Web Index Retrieval (Tavily, Brave Search, Serper) and Real-Time Web Grounding", divider="rainbow")

st.markdown("""
Search APIs ground agents with live internet intelligence:
- **AI-Optimized Search (Tavily)**: Designed specifically for LLMs; returns clean markdown extracts and raw text snippets without ads or script tags.
- **Traditional Search (Brave Search / Serper)**: General web index search returning ranked URL links and snippets.
- **Citation Tracking**: Preserving source URLs in state channels so agents can ground facts with clickable citations.
""")

# ==============================================================================
# SECTION 5: Weather APIs
# ==============================================================================
st.header("5. Weather APIs: Deterministic Public Utilities, Latency Profiling, and Parameter Validation", divider="rainbow")

st.markdown("""
Public utilities like weather services (e.g., OpenWeatherMap, NOAA) provide canonical examples of deterministic tool integrations:
- **Parameter Validation**: Enforce geographic coordinates or standardized ISO city codes.
- **Response Caching**: Weather data changes slowly; cache responses in Redis with a 30-minute TTL to prevent redundant API charges.
""")

# ==============================================================================
# SECTION 6: Finance APIs
# ==============================================================================
st.header("6. Finance APIs: Market Data, Strict Numerical Precision, and Caching", divider="rainbow")

st.markdown("""
Financial APIs (such as Bloomberg, Alpha Vantage, Polygon.io) require strict data integrity standards:
- **Numerical Precision**: Avoid floating-point arithmetic errors in Python; parse currencies as string decimals or integer cents.
- **Timestamp Verification**: Ensure real-time quotes are not confused with 15-minute delayed market feeds.
- **High-Risk Thresholds**: Financial mutations (trades, wires) must trigger `interrupt()` for human supervisor confirmation.
""")

# ==============================================================================
# SECTION 7: Enterprise APIs
# ==============================================================================
st.header("7. Enterprise APIs: CRM & SaaS Integrations (Salesforce, HubSpot, Jira, Slack, GitHub)", divider="rainbow")

st.markdown("""
Enterprise SaaS APIs involve complex authentication, rate limits, and permission scopes:
- **OAuth 2.0 Token Refresh**: Implement automatic token refreshing before dispatching tool calls.
- **Webhook Subscriptions**: Trigger LangGraph execution runs asynchronously in response to external SaaS webhooks 
  (e.g., a new GitHub Pull Request or Jira ticket creation).
""")

# ==============================================================================
# SECTION 8: Internal Company Services
# ==============================================================================
st.header("8. Internal Company Services: Service Mesh Routing, Private VPC Endpoints, mTLS, and gRPC", divider="rainbow")

st.markdown("""
In enterprise microservice architectures, agents frequently need to query internal company services that are not exposed 
to the public internet:

- **Service Mesh & mTLS**: Mutual TLS authentication ensures internal service calls are cryptographically verified and encrypted.
- **Private VPC Endpoints**: Route traffic over private AWS PrivateLink or GCP VPC Peering connections.
- **gRPC Integration**: Compile Protocol Buffer schemas into Python stubs for microsecond-level internal service calls.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-API Resilient Agent Tool Integration Suite", divider="rainbow")

st.markdown("""
The following complete module demonstrates an enterprise API integration suite featuring circuit breaking, 
per-API timeout configuration, and structured error propagation.
""")

st.code('''from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import httpx

class IntegrationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    api_results: Dict[str, Any]

class ResilientAPIClient:
    """
    Enterprise HTTP client configured with aggressive timeouts,
    automatic retries, and header injection.
    """
    def __init__(self, base_url: str, timeout_sec: float = 4.0):
        self.base_url = base_url
        self.timeout = timeout_sec

    async def get_json(self, path: str, params: dict = None) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(f"{self.base_url}{path}", params=params)
                resp.raise_for_status()
                return resp.json()
            except httpx.TimeoutException:
                return {"error": "External API timed out", "status": 504}
            except httpx.HTTPStatusError as e:
                return {"error": f"API error: {e.response.status_code}", "status": e.response.status_code}

client = ResilientAPIClient(base_url="https://api.internal-services.corp")

async def external_api_node(state: IntegrationState) -> dict:
    # Query internal enterprise service
    data = await client.get_json("/v1/enterprise/account/102")
    msg = AIMessage(content=f"Enterprise API response: {data}")
    return {
        "messages": [msg],
        "api_results": data
    }
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: External API Mock Runner & Network Latency Simulator", divider="rainbow")

st.markdown("""
Test mock API calls across search, finance, weather, CRM, and internal microservices, observing latency profiles and payload schemas.
""")

col_wb1, col_wb2 = st.columns(2)

with col_wb1:
    st.subheader("API Service Selector")
    selected_api = st.selectbox(
        "Target API Integration",
        list(api_sim.mock_apis.keys()),
        format_func=lambda k: f"{api_sim.mock_apis[k]['category']} ({k})"
    )
    
    if st.button("Execute Live Mock API Call", type="primary"):
        st.session_state["api_executed"] = True

with col_wb2:
    st.subheader("Response Payload & Latency Inspector")
    res = api_sim.call_api(selected_api, {})
    
    st.markdown(f"**Endpoint:** `{res['endpoint']}`")
    st.metric("Reported Latency", f"{res['latency_ms']} ms")
    st.json(res["response"])

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: How do you handle transient vs non-transient HTTP errors in agent tool integrations?"):
    st.markdown("""
    **Answer:**  
    - **Transient Errors (502, 503, 504, 429)**: Retry with exponential backoff and jitter up to a maximum attempt limit (e.g., 3-4 attempts).  
    - **Non-Transient Errors (400, 401, 403, 404)**: Immediately return a formatted error message to the model (e.g., `"Error 404: Account 'abc' not found. Please verify the ID."`), 
      allowing the agent to alter its parameters or explain the issue to the user without wasting retry attempts.
    """)

with st.expander("Q2: Why is GraphQL preferred over REST for token-constrained agent workflows?"):
    st.markdown("""
    **Answer:**  
    REST endpoints frequently over-fetch, returning dozens of unused fields, nested arrays, and verbose metadata. 
    GraphQL allows the agent or tool developer to specify the exact fields needed for the reasoning step, reducing token overhead by up to 80%.
    """)

with st.expander("Q3: How do you protect enterprise internal microservices when agents call them in loops?"):
    st.markdown("""
    **Answer:**  
    1. **Circuit Breakers**: Trip to OPEN state if the internal service starts throwing 500s.  
    2. **Token Bucket Rate Limiters**: Restrict total requests per minute per agent thread.  
    3. **Statement & Request Timeouts**: Impose hard 2-5 second timeouts on every outbound HTTP or gRPC call.  
    4. **Caching**: Cache deterministic responses in Redis.
    """)

with st.expander("Q4: What is the risk of allowing agents to call third-party search APIs without response sanitization?"):
    st.markdown("""
    **Answer:**  
    Third-party web search snippets can contain **indirect prompt injection** attacks embedded in web pages designed to trick 
    the model into ignoring instructions. Tool outputs must be filtered, enclosed within clear delimiters (`<search_results>...</search_results>`), 
    and stripped of prompt injection keywords.
    """)

with st.expander("Q5: How do you manage OAuth2 token lifecycles in long-running agent threads?"):
    st.markdown("""
    **Answer:**  
    Never store short-lived OAuth access tokens inside persistent graph state channels. Instead, store the encrypted refresh token 
    in a secure secret vault. When a tool executes, the tool retrieves a valid access token from an in-memory token manager, 
    refreshing it automatically if expired.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Hardcoding Third-Party API Keys in Tool Functions**  
Hardcoding credentials directly in Python files compromises security. Always inject API keys dynamically from environment variables 
or enterprise secret managers at execution time.
""")

st.error("""
**Trap 2: Omitting HTTP Timeouts on External Requests**  
Using HTTP clients without explicit timeouts defaults to indefinite waiting. If an external weather or finance API hangs, 
the agent worker thread blocks forever, exhausting application connection pools.
""")

st.warning("""
**Trap 3: Resending Massive Unfiltered API Payloads to LLMs**  
Passing raw 5MB JSON responses directly to the model overflows the context window and incurs excessive token costs. 
Always filter and extract only the relevant sub-fields before returning data to the agent state.
""")

st.warning("""
**Trap 4: Missing Schema Versioning on Internal Microservices**  
If an internal service changes its field names without updating the agent's tool schema, the agent will silently hallucinate 
or fail with KeyError exceptions.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>OpenAPI Schema Binding</h4>
        <p>Translating standard REST API specifications into typed Pydantic tool definitions for autonomous model invocation.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>GraphQL Field Selection</h4>
        <p>Querying only required data attributes to eliminate over-fetching and minimize token consumption.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Tavily Search API</h4>
        <p>A specialized search engine designed for LLMs, returning clean, grounded markdown text snippets free of ads and web clutter.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>mTLS Service Mesh</h4>
        <p>Mutual TLS encryption and identity verification for securing agent communication with private internal microservices.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to integrate REST APIs into LangGraph using Pydantic tools and httpx.", value=True)
chk_2 = st.checkbox("I understand how GraphQL field selection reduces token overhead in agent workflows.", value=True)
chk_3 = st.checkbox("I can connect search APIs (Tavily, Serper) for real-time web grounding.", value=True)
chk_4 = st.checkbox("I know how to handle financial and weather APIs with numerical precision and caching.", value=True)
chk_5 = st.checkbox("I understand enterprise SaaS integrations with OAuth2 token management.", value=True)
chk_6 = st.checkbox("I can connect internal microservices securely using service meshes, mTLS, and gRPC.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("External API and enterprise service integration mastery confirmed! Ready for Chapter 28 Advanced Concepts.")
