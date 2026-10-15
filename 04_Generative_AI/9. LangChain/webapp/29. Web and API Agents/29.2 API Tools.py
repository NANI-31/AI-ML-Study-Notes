import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="29.2 API Tools - REST APIs & OpenAPIs",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for API tool telemetry
st.markdown("""
<style>
    .api-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-get {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-post {
        background-color: #1E3A8A;
        color: #BFDBFE;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class APIToolsBridgeSimulator:
    """
    Simulates REST API tool invocations (GET and POST),
    bearer token authentication, schema validation, and HTTP response handling.
    """
    def __init__(self):
        self.endpoints = {
            "GET /api/v1/orders/{order_id}": {
                "method": "GET",
                "auth_required": True,
                "description": "Fetch order status and shipping tracking details.",
                "sample_response": {"order_id": "ORD-9912", "status": "Shipped", "carrier": "FedEx", "tracking": "FX-884192"}
            },
            "POST /api/v1/tickets": {
                "method": "POST",
                "auth_required": True,
                "description": "Create a new IT support service ticket.",
                "sample_response": {"ticket_id": "TCK-4001", "status": "Created", "priority": "High", "assigned_team": "Network Engineering"}
            }
        }

    def simulate_call(self, endpoint_key: str, payload: Dict[str, Any], token: str) -> Dict[str, Any]:
        info = self.endpoints.get(endpoint_key, self.endpoints["GET /api/v1/orders/{order_id}"])
        if not token.startswith("Bearer valid_"):
            return {"status_code": 401, "error": "Unauthorized: Invalid or missing API Bearer Token"}
        return {"status_code": 200, "data": info["sample_response"], "headers": {"Content-Type": "application/json"}}


# --- LESSON HEADER ---
st.title("29.2 API Tools: REST APIs, OpenAPI Schemas & Custom Tools")
st.markdown("""
Modern enterprise agents operate as intelligent orchestrators across internal microservices and external SaaS APIs
(Stripe, GitHub, Jira, Salesforce). Equipping agents with REST API tools requires defining typed Pydantic parameter schemas,
managing Bearer tokens and OAuth2 authentication, handling HTTP status codes, and converting OpenAPI/Swagger specifications
into executable tool suites.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise REST API Agent Tool Architecture", divider="rainbow")
st.markdown("""
Connecting agents to REST APIs requires strict structural isolation.
The LLM selects an API tool based on its description and Pydantic argument schema;
the tool validates parameters, injects secure authentication headers from an external secret vault,
dispatches an asynchronous HTTP request (via `httpx`), and formats the JSON response before returning it to the agent.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph APITools_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_llm_decision {
            label = "Agent Decision Engine";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            AgentModel [label="Chat Model (ReAct / Tool Calling)\\n(Decides to invoke API tool with JSON args)", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_tool_boundary {
            label = "LangChain API Tool Wrapper Layer";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            PydanticValidator [label="Pydantic Args Schema\\n(Validates query params / JSON body structure)", fillcolor="#1E293B", color="#60A5FA"];
            AuthInjector [label="Authentication & Secret Manager\\n(Injects Authorization: Bearer <token> from Vault)", fillcolor="#1E293B", color="#60A5FA"];
            HTTPClient [label="Async HTTP Client (httpx.AsyncClient)\\n(Enforces connection timeouts & retries)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_external_services {
            label = "Target REST Microservices (FastAPI / SaaS)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            MicroserviceGET [label="GET Endpoint (/api/v1/resource/{id})\\n(Read-only resource lookup)", fillcolor="#0369A1", color="#38BDF8"];
            MicroservicePOST [label="POST Endpoint (/api/v1/resource)\\n(State mutation + Idempotency-Key)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_response_handler {
            label = "Response Pruning & Formatting";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ResponsePruner [label="JSON Response Pruner\\n(Extracts relevant fields, drops megabyte payloads)", fillcolor="#065F46", color="#10B981"];
        }

        AgentModel -> PydanticValidator [label="1. Dispatched ToolCall"];
        PydanticValidator -> AuthInjector [label="2. Validated arguments"];
        AuthInjector -> HTTPClient [label="3. Authenticated request"];
        HTTPClient -> MicroserviceGET [label="4a. GET Request"];
        HTTPClient -> MicroservicePOST [label="4b. POST Request"];
        MicroserviceGET -> HTTPClient [label="5a. HTTP 200 JSON"];
        MicroservicePOST -> HTTPClient [label="5b. HTTP 201 Created"];
        HTTPClient -> ResponsePruner [label="6. Raw response payload"];
        ResponsePruner -> AgentModel [label="7. Clean observation string"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: REST APIS ---
st.header("2. REST API Integration: Architectural Contracts & Status Codes", divider="rainbow")
st.markdown("""
REST (Representational State Transfer) APIs operate over HTTP using standard methods (`GET`, `POST`, `PUT`, `DELETE`).
When integrating with LLMs, the tool must cleanly translate HTTP status codes:
- **`200 OK / 201 Created`**: Success; return relevant JSON data.
- **`400 Bad Request / 422 Unprocessable`**: Schema error; inform the LLM of missing fields so it can self-correct.
- **`401 / 403 Unauthorized`**: Authentication error; raise security alerts.
- **`404 Not Found`**: Resource does not exist; tell the LLM to verify IDs.
""")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown("### REST Tool Design Rules")
    st.markdown("""
    - **Explicit Return Strings**: Always return human-readable strings or compact JSON summaries, never raw HTTP response objects.
    - **Never Dump Massive Payloads**: If an API returns a 5MB JSON array, prune it to top-5 entries before passing to the LLM to prevent context overflow.
    - **Strict Timeout Limits**: Set `timeout=5.0` on all requests.
    """)
with col_r2:
    st.code("""
import httpx

def handle_api_response(response: httpx.Response) -> str:
    if response.status_code in [200, 201]:
        return str(response.json())
    elif response.status_code == 404:
        return "ERROR: Requested resource ID was not found."
    elif response.status_code == 422:
        return f"VALIDATION ERROR: {response.text}. Please check required parameters."
    else:
        return f"API ERROR {response.status_code}: Service temporarily unavailable."
    """, language="python")

# --- SECTION 3: GET REQUESTS ---
st.header("3. GET Request Tools: Resource Lookups & Query Parameters", divider="rainbow")
st.markdown("""
`GET` tools retrieve data without modifying server state. They are safe and idempotent.
The tool converts the agent's Pydantic arguments into URL path variables and query parameters.
""")

st.code("""
import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class OrderLookupSchema(BaseModel):
    order_id: str = Field(..., description="Alphanumeric order ID, e.g. ORD-9912")

@tool(args_schema=OrderLookupSchema)
def lookup_order_status(order_id: str) -> str:
    \"\"\"Fetches live status and shipping details for a specific order ID.\"\"\"
    url = f"https://api.internal.enterprise/v1/orders/{order_id}"
    headers = {"Authorization": "Bearer SECURE_VAULT_TOKEN"}
    
    with httpx.Client(timeout=5.0) as client:
        res = client.get(url, headers=headers)
        if res.status_code == 200:
            data = res.json()
            return f"Order {order_id} is {data['status']}. Carrier: {data['carrier']}, Tracking: {data['tracking']}."
        return f"Order lookup failed: Status {res.status_code}"
""", language="python")

# --- SECTION 4: POST REQUESTS ---
st.header("4. POST Request Tools: Mutating State, Payload Validation & Idempotency", divider="rainbow")
st.markdown("""
`POST` tools create or mutate resources (e.g. creating tickets, charging cards, dispatching emails).
Because network failures can trigger retries, production POST tools must generate **Idempotency Keys**
(`Idempotency-Key: UUID`) to prevent duplicate transactions if an agent re-executes a call.
""")

st.code("""
import uuid
import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class CreateTicketSchema(BaseModel):
    title: str = Field(..., min_length=5, description="Brief summary of the issue")
    priority: str = Field("medium", regex="^(low|medium|high|critical)$")
    description: str = Field(..., description="Detailed issue description")

@tool(args_schema=CreateTicketSchema)
def create_support_ticket(title: str, priority: str, description: str) -> str:
    \"\"\"Creates a new engineering support ticket in Jira/ServiceNow.\"\"\"
    url = "https://api.internal.enterprise/v1/tickets"
    headers = {
        "Authorization": "Bearer SECURE_TOKEN",
        "Idempotency-Key": str(uuid.uuid4()) # Prevents duplicate ticket creation
    }
    payload = {"title": title, "priority": priority, "description": description}
    
    with httpx.Client(timeout=5.0) as client:
        res = client.post(url, json=payload, headers=headers)
        if res.status_code in [200, 201]:
            data = res.json()
            return f"Created ticket {data['ticket_id']} with priority {data['priority']}."
        return f"Ticket creation failed: {res.text}"
""", language="python")

# --- SECTION 5: AUTHENTICATION ---
st.header("5. Enterprise API Authentication: Bearer Tokens, OAuth2 & Secret Isolation", divider="rainbow")
st.markdown("""
API tools must never accept raw API tokens from the user or the LLM.
Tokens must be securely injected from environment vaults at tool runtime.
For microservices using OAuth2, the tool wrapper fetches short-lived JWT access tokens using client credentials.
""")

st.code("""
import httpx

class AuthenticatedAPIClient:
    def __init__(self, client_id: str, client_secret: str, token_url: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self._cached_token = None

    def get_token(self) -> str:
        if not self._cached_token:
            # Exchange client credentials for short-lived JWT
            res = httpx.post(self.token_url, data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret
            })
            self._cached_token = res.json()["access_token"]
        return self._cached_token
""", language="python")

# --- SECTION 6: API SCHEMAS ---
st.header("6. OpenAPI & Swagger Schemas: Automatic Tool Generation from Specifications", divider="rainbow")
st.markdown("""
Rather than manually writing Python code for 50 endpoints, LangChain provides OpenAPI toolkit integrations
that parse OpenAPI 3.0 / Swagger JSON specifications and automatically construct compliant tools.
""")

st.code("""
import json
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_openai import ChatOpenAI

# Load OpenAPI specification JSON
with open("petstore_openapi.json") as f:
    raw_spec = json.load(f)

# Reduce spec to only relevant endpoints to save context tokens
reduced_spec = reduce_openapi_spec(raw_spec)

# Construct OpenAPI agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
openapi_agent = planner.create_openapi_agent(
    reduced_spec,
    requests_wrapper=None, # Injects auth headers
    llm=llm,
    verbose=True
)
""", language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Enterprise Hardened API Tool Suite", divider="rainbow")
st.markdown("""
Below is an enterprise API tool integration featuring Pydantic validation, safe error handling,
and response pruning.
""")

st.code("""
# enterprise_api_tools.py
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import httpx

class UserSearchSchema(BaseModel):
    email: str = Field(..., regex=r"^[^@]+@[^@]+\\.[^@]+$", description="Valid user email address")

@tool(args_schema=UserSearchSchema)
def search_crm_customer(email: str) -> str:
    \"\"\"Queries the CRM system for customer subscription data by email address.\"\"\"
    try:
        with httpx.Client(timeout=4.0) as client:
            res = client.get(
                "https://crm.internal.corp/v1/customers",
                params={"email": email},
                headers={"Authorization": "Bearer VAULT_SECRET_KEY"}
            )
            if res.status_code == 200:
                data = res.json()
                # Prune response to only essential fields
                return f"Customer {data['name']}: Plan={data['plan']}, Status={data['status']}"
            elif res.status_code == 404:
                return f"No customer found with email {email}."
            return f"CRM Error {res.status_code}"
    except httpx.TimeoutException:
        return "CRM lookup timed out. Please retry later."
    except Exception as e:
        return f"CRM tool execution error: {str(e)}"
""", language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: REST API Endpoint & Tool Invocation Simulator", divider="rainbow")
st.markdown("""
Simulate executing authenticated GET and POST API tools against enterprise microservices.
Observe header validation, token authorization, and payload handling.
""")

api_sim = APIToolsBridgeSimulator()

col_req, col_resp = st.columns([1, 1])

with col_req:
    st.subheader("🔌 API Tool Selection")
    chosen_ep = st.selectbox("Select API Endpoint:", list(api_sim.endpoints.keys()))
    token_input = st.text_input("Authorization Header:", value="Bearer valid_enterprise_jwt_token")
    sim_payload = st.text_area("Simulated Tool Arguments (JSON):", value='{"order_id": "ORD-9912"}')
    dispatch_btn = st.button("🚀 Invoke REST API Tool", type="primary")

with col_resp:
    st.subheader("📊 HTTP Response & Tool Observation")
    if dispatch_btn:
        import json as py_json
        try:
            parsed_payload = py_json.loads(sim_payload)
        except:
            parsed_payload = {}
            
        res = api_sim.simulate_call(chosen_ep, parsed_payload, token_input)
        badge_cls = "badge-get" if "GET" in chosen_ep else "badge-post"
        
        st.markdown(
            f"<div class='api-card'>"
            f"<span class='{badge_cls}'>{chosen_ep}</span><br><br>"
            f"<b>HTTP Status:</b> <code>{res['status_code']}</code><br>"
            f"<b>Result:</b><br><pre>{py_json.dumps(res, indent=2)}</pre>"
            f"</div>",
            unsafe_allow_html=True
        )
        if res["status_code"] == 200:
            st.success("Tool returned valid HTTP 200 response to agent.")
        else:
            st.error(f"HTTP Error: {res.get('error', 'Request failed')}")
    else:
        st.info("Select an endpoint and invoke to observe REST tool execution.")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why must API tools validate inputs with Pydantic before issuing HTTP calls?"):
    st.markdown("""
    **Answer**: Validating inputs with Pydantic catches missing parameters, malformed email addresses, and invalid data types
    locally before wasting network latency on a doomed API call. If validation fails, Pydantic raises an error that informs the LLM
    of the exact missing or malformed parameter, allowing the model to self-correct before hitting the server.
    """)

with st.expander("Q2: What is an Idempotency Key and why is it mandatory for POST API tools?"):
    st.markdown("""
    **Answer**: In distributed networks, a POST request may succeed on the server but fail on the return trip due to a connection timeout.
    If the agent retries the POST request, it could duplicate a critical action (such as charging a customer twice or creating duplicate tickets).
    An Idempotency Key (a unique UUID header) ensures the server recognizes the retry and returns the cached result without executing the action a second time.
    """)

with st.expander("Q3: How do you prevent context window blowup when APIs return massive JSON responses?"):
    st.markdown("""
    **Answer**: Never return the raw `res.json()` directly to the model if the response contains hundreds of nested fields or large arrays.
    API tools must implement response pruning: parse the JSON, select only the 3-5 fields relevant to the agent's task,
    and return a condensed summary string.
    """)

with st.expander("Q4: How should API credentials be injected into LangChain tools?"):
    st.markdown("""
    **Answer**: Credentials must be injected programmatically within the tool's internal execution logic, loaded from secure environment variables
    or secrets vaults. Never expose API keys as parameters in the tool's Pydantic schema, as this encourages the LLM to guess or leak tokens.
    """)

with st.expander("Q5: What is the purpose of reducing OpenAPI specifications before agent creation?"):
    st.markdown("""
    **Answer**: A full OpenAPI specification for a large enterprise service can exceed 100,000 tokens, far exceeding prompt limits.
    Reducing the spec strips out verbose descriptions, unused endpoints, and extraneous schemas, distilling the spec down to only
    the specific routes and parameters required for the agent's domain.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Exposing Tokens in Tool Arguments)**: Defining `api_key: str` in the tool's Pydantic args schema. The LLM hallucinates an API key or leaks it in logs.")
    st.warning("**Trap 2 (Missing Idempotency on State-Mutating Tools)**: Executing POST tools without Idempotency-Key headers, resulting in double-billing on network timeouts.")
with col_trap2:
    st.error("**Trap 3 (Dumping 500-Item JSON Lists to Prompts)**: Returning an unpruned API response with 500 orders, consuming 40,000 tokens and causing context overflow.")
    st.warning("**Trap 4 (Missing Granular HTTP Timeouts)**: Using standard `requests.get()` without `timeout=5.0`. A stalled server hangs the entire agent workflow.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: Why are Idempotency Keys required on POST API tools?")
        st.caption("Flip for answer:")
        st.info("**Back**: They prevent duplicate mutations (e.g. double charges) when network retries occur.")
    with st.container(border=True):
        st.markdown("**Front**: What should an API tool do before passing JSON back to the agent?")
        st.caption("Flip for answer:")
        st.info("**Back**: Prune the payload to extract only the essential fields needed by the model.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How should API keys be handled in tool definitions?")
        st.caption("Flip for answer:")
        st.info("**Back**: Injected internally from environment variables or vaults; never exposed as a tool parameter.")
    with st.container(border=True):
        st.markdown("**Front**: What LangChain toolkit converts Swagger specs into tools?")
        st.caption("Flip for answer:")
        st.info("**Back**: `OpenAPIToolkit` (or `create_openapi_agent`).")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand how to map REST HTTP methods and status codes into agent tool outputs.", value=True)
c2 = st.checkbox("I can implement safe GET request tools with Pydantic argument schemas.", value=True)
c3 = st.checkbox("I know how to use Idempotency Keys to prevent duplicate mutations in POST tools.", value=True)
c4 = st.checkbox("I can securely inject Bearer tokens into tools without exposing keys in schemas.", value=True)
c5 = st.checkbox("I understand how to generate automated tools from OpenAPI/Swagger specifications.", value=True)
c6 = st.checkbox("I know how to prune large JSON payloads to prevent prompt context bloat.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 29.3: Research Agents.")
