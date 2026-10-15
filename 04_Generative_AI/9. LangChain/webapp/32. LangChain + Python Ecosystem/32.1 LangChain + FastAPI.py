import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="32.1 LangChain + FastAPI - Production Microservices",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for microservice telemetry
st.markdown("""
<style>
    .endpoint-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class FastAPIAgentServerSimulator:
    """
    Simulates high-performance FastAPI asynchronous request processing,
    JWT verification, Server-Sent Events (SSE) streaming, and agent thread management.
    """
    def __init__(self):
        self.active_threads = {
            "thr_enterprise_01": {"user_id": "usr_alpha", "messages": 6, "state": "idle"},
            "thr_support_89": {"user_id": "usr_beta", "messages": 12, "state": "streaming"},
        }
        self.registered_routes = [
            {"path": "/api/v1/chat/sync", "method": "POST", "desc": "Synchronous single-turn agent invocation"},
            {"path": "/api/v1/chat/stream", "method": "POST", "desc": "Server-Sent Events (SSE) token and tool streaming"},
            {"path": "/api/v1/agents/threads/{thread_id}/state", "method": "GET", "desc": "Inspect persistent LangGraph agent state"},
            {"path": "/api/v1/auth/token", "method": "POST", "desc": "OAuth2 password bearer JWT authentication"}
        ]

    def authenticate_request(self, token: str) -> Tuple[bool, str]:
        if token.startswith("Bearer valid_jwt_"):
            return True, "usr_authorized_99"
        return False, "Invalid or expired JWT token"

    def simulate_sse_stream(self, prompt: str) -> List[Dict[str, Any]]:
        events = []
        events.append({"event": "status", "data": json.dumps({"status": "agent_started", "step": "routing"})})
        events.append({"event": "tool_call", "data": json.dumps({"tool": "sql_query_executor", "args": {"query": "SELECT count(*) FROM orders"}})})
        events.append({"event": "tool_result", "data": json.dumps({"rows": 1420})})
        
        words = f"Based on the database records, total completed orders count is 1,420. All transactions verified.".split()
        for idx, word in enumerate(words):
            events.append({"event": "token", "data": json.dumps({"token_id": idx, "chunk": word + " "})})
        
        events.append({"event": "done", "data": json.dumps({"finish_reason": "stop", "total_tokens": len(words) + 35})})
        return events


# --- LESSON HEADER ---
st.title("32.1 LangChain + FastAPI: Production Microservices Architecture")
st.markdown("""
Deploying LangChain and LangGraph systems into enterprise production demands transitioning from synchronous,
script-based invocations to robust, asynchronous microservices. FastAPI provides native ASGI performance,
Pydantic data validation, OpenAPI specification generation, dependency injection, and asynchronous streaming
capabilities via Server-Sent Events (SSE). This module examines the architecture, endpoint design, authentication,
and async execution patterns required to serve enterprise AI workloads.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: FastAPI and LangChain Microservices", divider="rainbow")
st.markdown("""
The integration of LangChain within FastAPI decouples the client-facing HTTP/WebSocket interface from the underlying
agent execution loop. Incoming requests pass through security and rate-limiting middleware, establish persistent session
threads using Checkpointers, and stream tokens directly to the client via asynchronous generators without blocking the worker thread.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph FastAPI_LangChain_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_clients {
            label = "Client Layer (Web, Mobile, Enterprise Services)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            ClientApp [label="Client Application\\n(React / Next.js / Streamlit / Mobile)", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_fastapi_gateway {
            label = "FastAPI Asynchronous Gateway (ASGI / Uvicorn)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            CORSMiddleware [label="Middleware Pipeline\\n(CORS, Rate Limiting, Request ID)", fillcolor="#1E293B", color="#60A5FA"];
            AuthDep [label="Security Dependency\\n(OAuth2 / JWT Bearer Validation)", fillcolor="#1E293B", color="#60A5FA"];
            Router [label="APIRouter Endpoints\\n(/api/v1/chat/stream, /state, /resume)", fillcolor="#1E293B", color="#60A5FA"];
            PydanticIn [label="Pydantic Request Schemas\\n(Input Validation & Type Coercion)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_langchain_runtime {
            label = "LangChain / LangGraph Execution Core";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ThreadManager [label="Thread & Session Manager\\n(Session Context & Checkpointer Key)", fillcolor="#0369A1", color="#38BDF8"];
            AsyncAgent [label="Compiled StateGraph / AgentExecutor\\n(Async Event Streamer: astream_events)", fillcolor="#0369A1", color="#38BDF8"];
            ToolsEngine [label="Async Tool Dispatcher\\n(VectorDB, SQL, Web Search, REST APIs)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_streaming_response {
            label = "Streaming & Response Layer";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            EventSourceResp [label="StreamingResponse (SSE / text/event-stream)\\n(AsyncGenerator yielding SSE events)", fillcolor="#065F46", color="#10B981"];
            PydanticOut [label="Pydantic Response Schemas\\n(Structured JSON Output)", fillcolor="#065F46", color="#10B981"];
        }

        ClientApp -> CORSMiddleware [label="HTTP POST / GET"];
        CORSMiddleware -> AuthDep [label="Bearer Token"];
        AuthDep -> Router [label="Validated User Context"];
        Router -> PydanticIn [label="Parse Payload"];
        PydanticIn -> ThreadManager [label="Session thread_id"];
        ThreadManager -> AsyncAgent [label="Trigger Execution"];
        AsyncAgent -> ToolsEngine [label="Async Tool Calls"];
        ToolsEngine -> AsyncAgent [label="Observation Feed"];
        AsyncAgent -> EventSourceResp [label="Yield Chunks (Tokens, State)"];
        AsyncAgent -> PydanticOut [label="Final Structured Result"];
        EventSourceResp -> ClientApp [label="Server-Sent Events Stream"];
        PydanticOut -> ClientApp [label="200 OK JSON Response"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: CREATING APIS ---
st.header("2. Creating Production APIs with FastAPI and LangChain", divider="rainbow")
st.markdown("""
Building enterprise APIs requires strict contract definitions using Pydantic models. FastAPI leverages Python type hints
to enforce schema validation before any LLM logic is triggered. This shields backend model infrastructure from malformed
requests and guarantees consistent client response structures.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Architectural Principles")
    st.markdown("""
    - **Pydantic Model Separation**: Keep input payloads, output contracts, and internal agent state cleanly separated.
    - **Dependency Injection**: Inject database pools, vector stores, and LLM instances using `fastapi.Depends` to guarantee thread safety and lifecycle cleanup.
    - **Non-blocking Execution**: Always define endpoint handlers with `async def` and invoke LangChain Runnables via `.ainvoke()` or `.astream()`.
    - **Error Handling**: Intercept LLM rate limits and provider timeouts with custom `HTTPException` handlers.
    """)
with col2:
    st.markdown("### Request & Response Schemas")
    st.code("""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User input prompt")
    thread_id: str = Field(..., description="Unique conversation session identifier")
    model_override: Optional[str] = Field(None, description="Optional dynamic model selection")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    tokens_consumed: int
    tool_calls_executed: List[str]
    execution_time_seconds: float
    """, language="python")

# --- SECTION 3: AGENT ENDPOINTS ---
st.header("3. Stateful Agent Endpoints and Thread Session Management", divider="rainbow")
st.markdown("""
Enterprise conversational agents require state persistence across distinct HTTP requests. In stateless microservices,
FastAPI routes receive a `thread_id` and load the associated conversation history from a persistent checkpointer (e.g. Postgres, Redis, or DynamoDB)
before resuming agent execution.
""")

st.code("""
from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

@router.post("/invoke", response_model=ChatResponse)
async def invoke_agent_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_active_user),
    checkpointer: AsyncPostgresSaver = Depends(get_db_checkpointer)
):
    config = {"configurable": {"thread_id": request.thread_id, "user_id": current_user["id"]}}
    
    try:
        # Run agent asynchronously with persistent checkpointing
        final_state = await enterprise_graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        last_message = final_state["messages"][-1]
        return ChatResponse(
            response=last_message.content,
            thread_id=request.thread_id,
            tokens_consumed=final_state.get("total_tokens", 0),
            tool_calls_executed=final_state.get("executed_tools", []),
            execution_time_seconds=final_state.get("latency", 0.0)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failure: {str(exc)}"
        )
""", language="python")

# --- SECTION 4: STREAMING RESPONSES ---
st.header("4. Streaming Responses with Server-Sent Events (SSE) and NDJSON", divider="rainbow")
st.markdown("""
For conversational applications, waiting 10 to 30 seconds for complete agent execution introduces unacceptable latency.
FastAPI handles real-time token streaming using `StreamingResponse` configured with `media_type="text/event-stream"`.
By consuming LangChain's `.astream_events(version="v2")`, the endpoint can stream individual tokens, intermediate tool executions,
and lifecycle events to the client.
""")

st.code("""
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

@router.post("/stream")
async def stream_agent_endpoint(request: ChatRequest):
    async def event_generator():
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # astream_events exposes tokens, tool calls, and node lifecycle events
        async for event in enterprise_graph.astream_events(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
            version="v2"
        ):
            kind = event["event"]
            
            # Stream LLM tokens as they are generated
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    payload = json.dumps({"type": "token", "content": content})
                    yield f"data: {payload}\\n\\n"
            
            # Stream tool call initiation
            elif kind == "on_tool_start":
                payload = json.dumps({
                    "type": "tool_start",
                    "tool": event["name"],
                    "inputs": event["data"].get("input")
                })
                yield f"data: {payload}\\n\\n"
            
            # Stream tool call output
            elif kind == "on_tool_end":
                payload = json.dumps({
                    "type": "tool_end",
                    "tool": event["name"],
                    "output": str(event["data"].get("output"))
                })
                yield f"data: {payload}\\n\\n"

        yield "data: [DONE]\\n\\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
""", language="python")

# --- SECTION 5: AUTHENTICATION ---
st.header("5. Enterprise Authentication and Security (JWT, API Keys, RBAC)", divider="rainbow")
st.markdown("""
Exposing generative agents without strict authentication risks catastrophic financial loss due to unauthorized token usage.
FastAPI provides dependency injection hooks (`HTTPBearer`, `OAuth2PasswordBearer`, `Security`) to validate JWT tokens,
enforce role-based access control (RBAC), and isolate user contexts to prevent cross-tenant data leaks.
""")

col_sec1, col_sec2 = st.columns(2)
with col_sec1:
    st.markdown("### Security Vectors & Mitigation")
    st.markdown("""
    - **Tenant Isolation**: Bind the `user_id` and `tenant_id` from the JWT directly to the checkpointer configuration to prevent cross-session eavesdropping.
    - **API Key Rate Limiting**: Implement token-bucket rate limiters in Redis keyed by API key or user ID.
    - **Model Tool Authorizations**: Use user role claims in JWT to selectively expose sensitive tools (e.g. read-only vs write-back database tools).
    """)
with col_sec2:
    st.markdown("### Dependency Injection Implementation")
    st.code("""
from fastapi import Security, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, "SECRET_SIGNING_KEY", algorithms=["HS256"])
        user_id = payload.get("sub")
        roles = payload.get("roles", [])
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return {"id": user_id, "roles": roles}
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed")
    """, language="python")

# --- SECTION 6: ASYNC AGENTS ---
st.header("6. Asynchronous Agent Orchestration: ainvoke, astream_events, and Background Tasks", divider="rainbow")
st.markdown("""
Python's asyncio event loop must never be blocked by CPU-bound parsing or synchronous I/O.
FastAPI applications serving LangChain systems must strictly adhere to non-blocking patterns.
Long-running agent evaluations or document indexing tasks should be dispatched to FastAPI `BackgroundTasks`
or Celery/Arq task queues.
""")

st.code("""
from fastapi import BackgroundTasks

async def async_audit_and_eval(thread_id: str, prompt: str, completion: str):
    # Asynchronous background logging and LangSmith dataset creation
    await audit_logger.alog(thread_id=thread_id, prompt=prompt, completion=completion)

@router.post("/chat/async-task")
async def chat_with_background_eval(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    # Asynchronous agent invocation
    result = await enterprise_agent.ainvoke({"input": request.message})
    
    # Offload observability and metric tracking to background execution
    background_tasks.add_task(
        async_audit_and_eval,
        thread_id=request.thread_id,
        prompt=request.message,
        completion=result["output"]
    )
    
    return {"output": result["output"], "status": "processed"}
""", language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Resilient Enterprise FastAPI Agent Microservice", divider="rainbow")
st.markdown("""
Below is a complete, production-ready implementation of a FastAPI agent microservice featuring structured Pydantic models,
in-memory session state management, error boundaries, and asynchronous SSE streaming.
""")

st.code("""
import asyncio
import json
from typing import AsyncGenerator, Dict, Any, List, Optional
from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Application Initialization
app = FastAPI(
    title="LangChain Enterprise Agent Gateway",
    version="1.0.0",
    docs_url="/api/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class ChatQuery(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=3)
    user_id: Optional[str] = "usr_default"

# Simulated Async LangGraph Runtime
class AgentRuntime:
    async def stream_agent_execution(self, message: str, thread_id: str) -> AsyncGenerator[str, None]:
        # Yield status event
        yield f"event: status\\ndata: {json.dumps({'state': 'analyzing_intent'})}\\n\\n"
        await asyncio.sleep(0.05)
        
        # Yield tool call event
        yield f"event: tool_call\\ndata: {json.dumps({'tool': 'enterprise_kb_search', 'query': message})}\\n\\n"
        await asyncio.sleep(0.1)
        
        # Yield streaming token chunks
        response_text = f"Acknowledged query: '{message}'. Retrieving records for thread {thread_id} completed successfully."
        for token in response_text.split():
            yield f"event: token\\ndata: {json.dumps({'chunk': token + ' '})}\\n\\n"
            await asyncio.sleep(0.02)
            
        yield f"event: finish\\ndata: {json.dumps({'status': 'completed', 'thread_id': thread_id})}\\n\\n"

runtime = AgentRuntime()

@app.post("/api/v1/chat/stream")
async def stream_chat(query: ChatQuery):
    try:
        generator = runtime.stream_agent_execution(query.message, query.thread_id)
        return StreamingResponse(generator, media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
""", language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: FastAPI Request Lifecycle and SSE Token Simulator", divider="rainbow")
st.markdown("""
Test asynchronous endpoint execution, JWT authentication verification, and real-time Server-Sent Events (SSE) streaming.
""")

simulator = FastAPIAgentServerSimulator()

col_req, col_res = st.columns([1, 1])
with col_req:
    st.subheader("HTTP Request Simulation")
    selected_endpoint = st.selectbox("Target Endpoint", [r["path"] for r in simulator.registered_routes])
    auth_header = st.text_input("Authorization Header", value="Bearer valid_jwt_token_enterprise_99")
    prompt_input = st.text_area("Request JSON Payload (message)", value="Calculate quarterly gross revenue and verify ledger entries.")
    test_btn = st.button("Dispatch Asynchronous Request", type="primary")

with col_res:
    st.subheader("Response & Stream Inspector")
    if test_btn:
        is_valid, user_or_err = simulator.authenticate_request(auth_header)
        if not is_valid:
            st.error(f"HTTP 401 Unauthorized: {user_or_err}")
        else:
            st.success(f"HTTP 200 Authenticated — Context: {user_or_err}")
            if "stream" in selected_endpoint:
                st.caption("Listening to Server-Sent Events (SSE) Stream:")
                events = simulator.simulate_sse_stream(prompt_input)
                for ev in events:
                    with st.container(border=True):
                        st.markdown(f"**Event Type**: `{ev['event']}`")
                        st.code(ev['data'], language="json")
            else:
                st.json({
                    "status": "success",
                    "endpoint": selected_endpoint,
                    "thread_id": "thr_enterprise_01",
                    "response": f"Processed synchronous response for '{prompt_input}'"
                })

with st.expander("📋 View Microservice Registered Route Specifications", expanded=False):
    st.dataframe(pd.DataFrame(simulator.registered_routes), use_container_width=True)


# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why should you prefer Server-Sent Events (SSE) over WebSockets for LangChain streaming in FastAPI?"):
    st.markdown("""
    **Answer**: SSE operates over standard HTTP/1.1 and HTTP/2, making it compatible with enterprise firewalls, API gateways (like Kong, AWS API Gateway),
    and existing load balancers without requiring protocol switching. Since LLM streaming is strictly unidirectional (server pushing tokens to client),
    SSE provides lower protocol overhead, native browser auto-reconnection via `EventSource`, and simplified authentication via standard HTTP headers.
    """)

with st.expander("Q2: How do you prevent thread blocking when running LangChain in an async FastAPI endpoint?"):
    st.markdown("""
    **Answer**: Always use native asynchronous methods (`ainvoke()`, `astream()`, `astream_events()`). If utilizing a custom legacy tool or library
    that only provides synchronous execution, offload it to an executor pool using `asyncio.to_thread()` or `run_in_executor()`. Never call synchronous
    `invoke()` directly in an `async def` FastAPI route, as it halts the ASGI event loop and prevents all concurrent request processing.
    """)

with st.expander("Q3: How is conversational memory preserved across stateless HTTP calls in FastAPI?"):
    st.markdown("""
    **Answer**: Stateless FastAPI microservices persist state using external checkpointers (PostgreSQL, Redis, DynamoDB). Each request includes a
    `thread_id` in its payload. The agent runtime loads the persisted checkpoint from the database at the start of the request, executes the turn,
    and commits the updated state checkpoint before returning the HTTP response.
    """)

with st.expander("Q4: What is the purpose of `astream_events(version='v2')` in LangChain FastAPI endpoints?"):
    st.markdown("""
    **Answer**: Unlike simple `.astream()`, which only yields final model tokens, `.astream_events(version='v2')` is an event-driven stream yielding
    lifecycle events across the entire graph: model inputs, token chunks, tool starts, tool ends, and subgraph state transitions. This enables the
    FastAPI gateway to push real-time UI updates indicating tool progress (e.g. 'Querying database...') alongside generation tokens.
    """)

with st.expander("Q5: How do you enforce tenant isolation when serving multi-tenant LangChain agents via FastAPI?"):
    st.markdown("""
    **Answer**: Extract tenant identifiers from cryptographically verified JWT claims in FastAPI dependencies (`Depends(get_current_tenant)`).
    Inject the tenant ID into the checkpointer configuration keys and vector database metadata filters. This ensures agents can only read or write
    checkpoints and vector embeddings belonging to the authenticated tenant.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Sync Invoke in Async Handler)**: Calling `agent.invoke()` inside an `async def` endpoint. This blocks the main event loop, causing latency spikes and timeouts for all other concurrent requests.")
    st.warning("**Trap 2 (Missing Streaming Timeout Configurations)**: Failing to configure proxy and load balancer timeouts (e.g. Nginx, Cloudflare) for SSE endpoints. Proxies drop connections after 30-60 seconds of silent tool execution.")
with col_trap2:
    st.error("**Trap 3 (In-Memory Checkpointer in Distributed Cluster)**: Using `MemorySaver` in a multi-pod Kubernetes deployment. Subsequent user requests land on different pods and lose conversation history entirely.")
    st.warning("**Trap 4 (Leaking Internal Prompts via Error Handlers)**: Returning unhandled LLM provider exceptions in HTTP 500 responses, exposing sensitive system prompts and database connection strings to clients.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What media type is used for Server-Sent Events in FastAPI?")
        st.caption("Flip for answer:")
        st.info("**Back**: `text/event-stream` with `StreamingResponse` yielding lines starting with `data:`.")
    with st.container(border=True):
        st.markdown("**Front**: How does FastAPI pass conversation threads to LangGraph?")
        st.caption("Flip for answer:")
        st.info("**Back**: Via the `config={'configurable': {'thread_id': ...}}` parameter in `ainvoke` or `astream_events`.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: Why is `BackgroundTasks` used in FastAPI agent endpoints?")
        st.caption("Flip for answer:")
        st.info("**Back**: To perform non-critical post-processing (logging, audit trails, evaluation datasets) without delaying client responses.")
    with st.container(border=True):
        st.markdown("**Front**: What tool prevents blocking when wrapping synchronous tools?")
        st.caption("Flip for answer:")
        st.info("**Back**: `asyncio.to_thread()` offloads synchronous CPU or blocking I/O calls to a worker thread pool.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I can construct Pydantic request and response schemas for LangChain agent endpoints.", value=True)
c2 = st.checkbox("I understand how to implement persistent thread state using checkpointers across HTTP requests.", value=True)
c3 = st.checkbox("I can implement asynchronous Server-Sent Events (SSE) streaming via FastAPI StreamingResponse.", value=True)
c4 = st.checkbox("I know how to secure agent endpoints using JWT Bearer authentication and RBAC dependencies.", value=True)
c5 = st.checkbox("I understand the difference between .ainvoke() and .astream_events(version='v2').", value=True)
c6 = st.checkbox("I know how to avoid event-loop blocking by offloading synchronous tools to thread pools.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 32.2: LangChain + Streamlit.")
