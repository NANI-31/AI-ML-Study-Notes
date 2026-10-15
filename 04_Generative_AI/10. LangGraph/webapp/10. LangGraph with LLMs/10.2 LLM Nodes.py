import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="10.2 LLM Nodes - LangGraph Curriculum",
    page_icon="🧠",
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
    .badge-node {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-msg {
        background-color: #1E1E38;
        color: #A78BFA;
        border: 1px solid #8B5CF6;
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
class LLMNodeExecutionEngine:
    """
    Simulates node invocation, message assembly (SystemMessage, HumanMessage, AIMessage),
    reducer updates, and structured Pydantic extraction.
    """
    def __init__(self):
        self.personas = {
            "Enterprise Support Tier 1": {
                "system_prompt": "You are a Tier-1 technical support engineer for CloudScale Inc. Be concise, verify customer entitlements, and classify inquiries strictly.",
                "temperature": 0.2,
                "role_tone": "Formal, analytical, diagnostic"
            },
            "Security Incident Officer": {
                "system_prompt": "You are an automated SOC triage analyst. Identify CVE severity, assess data exfiltration risk, and recommend immediate isolation protocols.",
                "temperature": 0.0,
                "role_tone": "Urgent, compliance-driven, zero-trust"
            },
            "Financial Risk Evaluator": {
                "system_prompt": "You are an institutional credit assessment assistant. Analyze transaction velocity, verify AML red flags, and enforce lending threshold limits.",
                "temperature": 0.1,
                "role_tone": "Methodical, regulatory, quantitative"
            }
        }

    def execute_node(
        self,
        persona_name: str,
        user_query: str,
        extract_structured: bool,
        message_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        persona = self.personas.get(persona_name, self.personas["Enterprise Support Tier 1"])
        sys_msg = {"role": "system", "content": persona["system_prompt"], "id": "sys_001"}
        human_msg = {"role": "human", "content": user_query, "id": f"human_{len(message_history) + 1:03d}"}

        # Simulated message accumulation
        full_context = [sys_msg] + message_history + [human_msg]

        # Simulated LLM generation
        if extract_structured:
            # Simulated Pydantic structured output
            intent = "Urgent Support" if "down" in user_query.lower() or "error" in user_query.lower() else "General Inquiry"
            severity = "High" if "down" in user_query.lower() or "breach" in user_query.lower() else "Low"
            structured_payload = {
                "ticket_category": intent,
                "urgency_score": 9 if severity == "High" else 3,
                "requires_human_escalation": severity == "High",
                "suggested_routing": "DevOps Escalation" if severity == "High" else "Self-Service Portal",
                "extracted_entities": ["PostgreSQL Cluster" if "db" in user_query.lower() else "API Gateway"]
            }
            ai_msg = {
                "role": "ai",
                "content": f"Classification complete. Ticket tagged as [{intent}] with severity [{severity}].",
                "id": f"ai_{len(message_history) + 2:03d}",
                "structured_output": structured_payload
            }
        else:
            ai_msg = {
                "role": "ai",
                "content": f"Understood. Based on your report ('{user_query}'), our system has registered the event. Recommended next step: run health checks and review authorization logs.",
                "id": f"ai_{len(message_history) + 2:03d}",
                "structured_output": None
            }

        # State delta returned by node
        state_delta = {
            "messages": [human_msg, ai_msg],
            "node_execution_latency_ms": int(np.random.randint(280, 520)),
            "token_usage": {
                "prompt_tokens": len(persona["system_prompt"].split()) * 2 + len(user_query.split()) * 3,
                "completion_tokens": len(ai_msg["content"].split()) * 2 + (40 if extract_structured else 0)
            }
        }
        return {
            "full_context": full_context,
            "ai_msg": ai_msg,
            "state_delta": state_delta
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: LLM Invocation Lifecycle and Message Flow in LangGraph Nodes", divider="rainbow")

st.markdown("""
In LangGraph, an **LLM Node** is fundamentally a deterministic Python callable that:
1. Receives the current **Graph State** (containing message sequences and domain variables).
2. Assembles structured prompt messages (**SystemMessage**, **HumanMessage**, contextual tools).
3. Executes a model invocation (`model.invoke()` or `model.with_structured_output().invoke()`).
4. Emits a **State Delta Dictionary** containing new or updated **AIMessage** objects to be merged by channels.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    llm_node_blueprint_dot = """
    digraph LLMNodeLifecycle {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_state_in {
            label = "1. Graph State Ingestion";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            STATE_IN [label="Current State: MessagesState\\n- messages: Annotated[List[BaseMessage], add_messages]\\n- context_metadata: Dict[str, Any]", fillcolor="#0F172A", color="#38BDF8"];
        }

        subgraph cluster_node_body {
            label = "2. LLM Node Execution Function (def call_model(state))";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            PROMPT_PREP [label="Prompt Construction\\n[SystemMessage(rules) + state['messages']]", fillcolor="#1E1E38", color="#8B5CF6"];
            MODEL_INVOKE [label="Model Invocation: ChatModel.invoke(messages)\\nOr model.with_structured_output(Schema)", fillcolor="#0284C7", color="#38BDF8"];
            RESPONSE_GEN [label="Generation: AIMessage\\n- content: str\\n- tool_calls: Optional[List[Dict]]\\n- usage_metadata: Dict", fillcolor="#047857", color="#34D399"];

            PROMPT_PREP -> MODEL_INVOKE [label="Formatted Input"];
            MODEL_INVOKE -> RESPONSE_GEN [label="Emits Response"];
        }

        subgraph cluster_state_out {
            label = "3. State Channel Reducer Update";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            DELTA_DICT [label="Node Return: State Delta\\n{'messages': [response]}", fillcolor="#0F172A", color="#F59E0B"];
            CHANNEL_REDUCER [label="add_messages Reducer\\nAppends new IDs, Overwrites duplicate IDs", fillcolor="#1E293B", color="#38BDF8"];
            NEXT_SUPERSTEP [label="Persisted State Checkpoint\\nTransition to Routing / Next Node", fillcolor="#0F172A", color="#34D399"];

            DELTA_DICT -> CHANNEL_REDUCER [label="Channel merge"];
            CHANNEL_REDUCER -> NEXT_SUPERSTEP [label="Superstep committed"];
        }

        STATE_IN -> PROMPT_PREP [label="Passes state to node"];
        RESPONSE_GEN -> DELTA_DICT [label="Wraps in state dict"];
    }
    """
    st.graphviz_chart(llm_node_blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Key Architectural Takeaway:</b> Nodes in LangGraph never mutate global state directly.
They return partial state updates (state deltas) as Python dictionaries. The <code>add_messages</code> channel reducer
then handles non-destructive appending, ID collision de-duplication, and message replacement.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: CALLING AN LLM INSIDE A NODE
# ==============================================================================
st.header("2. Calling an LLM Inside a Node", divider="rainbow")

st.markdown("""
In LangGraph, an LLM call is encapsulated entirely inside a standard Python node function.
The node receives the state dictionary as its primary argument and interacts with a model instance.
""")

col_c1, col_c2 = st.columns([1, 1])
with col_c1:
    st.markdown("#### Modern LangGraph Node Pattern")
    st.code("""
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

# 1. Instantiate chat model
model = ChatOpenAI(model="gpt-4o", temperature=0.0)

# 2. Define node function
def call_agent(state: MessagesState) -> dict:
    # Read messages from incoming state channel
    messages = state["messages"]
    
    # Execute LLM call
    response = model.invoke(messages)
    
    # Return dictionary delta
    return {"messages": [response]}
""", language="python")

with col_c2:
    st.markdown("#### Critical Principles of Node Invocation")
    st.markdown("""
    1. **Function Signature**: A node accepts `state: StateType` and optionally `config: RunnableConfig`.
    2. **State Delta Return**: Never return the full state history manually; returning `{"messages": [response]}` allows the `add_messages` reducer to safely append without data duplication.
    3. **Stateless Node Functions**: Avoid caching state inside global Python variables. All contextual memory must reside within the graph state channels.
    4. **Async Equivalence**: For high-throughput services, use `async def call_agent(state)` and `await model.ainvoke(messages)` to unlock non-blocking event loops.
    """)

# ==============================================================================
# SECTION 3: PROMPT CONSTRUCTION AND DYNAMIC CONTEXT INJECTION
# ==============================================================================
st.header("3. Prompt Construction and Dynamic Context Injection", divider="rainbow")

st.markdown("""
Unlike static LLM wrappers, graph nodes often need to **construct dynamic prompts** by fusing:
1. Long-term persistent persona rules.
2. Retrieved external documents or RAG context stored in state fields.
3. Multi-turn conversation history.
""")

st.code("""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# Approach A: Direct List Composition (Fastest & Purest)
def dynamic_prompt_node(state: MessagesState) -> dict:
    domain_topic = state.get("topic", "General Computing")
    rag_docs = state.get("retrieved_chunks", [])
    
    # Synthesize contextual system prompt
    context_str = "\\n\\n".join([doc.page_content for doc in rag_docs])
    system_instruction = SystemMessage(
        content=f"You are a specialist in {domain_topic}. Answer using the context below:\\n{context_str}"
    )
    
    # Prepend dynamic system message before message history
    formatted_messages = [system_instruction] + state["messages"]
    response = model.invoke(formatted_messages)
    return {"messages": [response]}
""", language="python")

col_pr1, col_pr2 = st.columns(2)
with col_pr1:
    st.info("**Technique 1: Contextual Prepending**\\nInjecting dynamically retrieved RAG chunks directly into the head `SystemMessage` on each turn without corrupting the historical conversation thread.")
with col_pr2:
    st.info("**Technique 2: Message Trimming & Sliding Windows**\\nApplying `trim_messages()` to state history prior to invocation, ensuring tokens stay well within context window boundaries.")

# ==============================================================================
# SECTION 4: SYSTEM MESSAGES: PERSONA, RULES, AND BOUNDARIES
# ==============================================================================
st.header("4. System Messages: Persona, Rules, and Execution Boundaries", divider="rainbow")

st.markdown("""
A `SystemMessage` establishes the **behavioral boundaries**, operational tone, safety guardrails,
and operational constraints of the LLM. In multi-agent graphs, different nodes utilize distinct system messages
to assume specialized personas.
""")

sys_df = pd.DataFrame([
    {
        "Role Persona": "Router / Classifier",
        "Target Output": "Strict label or JSON enum",
        "System Prompt Guardrail": "You are a deterministic classification node. Output only the category key: [BILLING, TECHNICAL, ESCALATE]. Zero conversational text."
    },
    {
        "Role Persona": "Research Specialist",
        "Target Output": "Markdown synthesis with citations",
        "System Prompt Guardrail": "You are an exhaustive research assistant. Every factual assertion must be attributed to an explicitly provided source ID."
    },
    {
        "Role Persona": "Code Auditor",
        "Target Output": "Vulnerability assessment diff",
        "System Prompt Guardrail": "You are a senior security engineer. Analyze AST security risks, OWASP Top 10 vulnerabilities, and emit actionable remediation snippets."
    }
])
st.table(sys_df)

# ==============================================================================
# SECTION 5: HUMAN MESSAGES: USER INPUT INGESTION
# ==============================================================================
st.header("5. Human Messages: User Input Ingestion and Multi-Turn Framing", divider="rainbow")

st.markdown("""
`HumanMessage` objects represent input generated by the external user or an upstream client application.
In multi-turn chat applications, human inputs accumulate alongside AI responses.
""")

st.code("""
from langchain_core.messages import HumanMessage

# Standard text message
user_msg = HumanMessage(content="Explain how checkpointers persist state in SQLite.")

# Multi-modal input (Vision, Audio, or Documents)
multimodal_msg = HumanMessage(
    content=[
        {"type": "text", "text": "Analyze the architectural diagram shown in this image:"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."}
        }
    ]
)
""", language="python")

# ==============================================================================
# SECTION 6: AI MESSAGES: CAPTURING MODEL RESPONSES AND TOOL CALLS
# ==============================================================================
st.header("6. AI Messages: Capturing Model Responses and Tool Calls", divider="rainbow")

st.markdown("""
When a `BaseChatModel` executes, it returns an `AIMessage`. An `AIMessage` contains two critical payload types:
1. **`content`**: The generated natural language response (or empty string if solely calling tools).
2. **`tool_calls`**: A list of structured tool invocation requests with standardized schema `{name, args, id}`.
3. **`usage_metadata`**: Exact token accounting (`input_tokens`, `output_tokens`, `total_tokens`).
""")

st.code("""
# Anatomy of an AIMessage returned by ChatModel.invoke()
ai_message = model.invoke([HumanMessage(content="What is the weather in Tokyo?")])

# Inspection
print("Text Content:", ai_message.content)
print("Tool Invocations:", ai_message.tool_calls)
# Example tool_calls payload:
# [
#   {
#     "name": "get_weather",
#     "args": {"city": "Tokyo", "units": "metric"},
#     "id": "call_982b67f1a"
#   }
# ]
print("Token Usage:", ai_message.usage_metadata)
""", language="python")

# ==============================================================================
# SECTION 7: MESSAGE STATE: CHANNELS AND REDUCER ACCUMULATION
# ==============================================================================
st.header("7. Message State: Channels and Reducer Accumulation", divider="rainbow")

st.markdown("""
LangGraph's core innovation for conversational agents is the **`add_messages` reducer**.
Instead of manually concatenating lists (`state["messages"] + new_messages`), the channel manages
append operations, message replacements, and de-duplication based on message `id`.
""")

with st.expander("🔍 View Message State Channel Reducer Workflow", expanded=True):
    reducer_dot = """
    digraph MessageReducerFlow {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        OLD_STATE [label="Existing State Channel\\n[Msg1(id='a'), Msg2(id='b')]", fillcolor="#0F172A", color="#475569"];
        DELTA_IN [label="Node Output Delta\\n[Msg2_Updated(id='b'), Msg3(id='c')]", fillcolor="#1E1E38", color="#8B5CF6"];

        subgraph cluster_logic {
            label = "add_messages Reducer Logic";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#38BDF8";
            fontcolor = "#38BDF8";

            MATCH_TEST [label="Does Incoming Message ID match an existing ID?", shape=diamond, fillcolor="#0F172A", color="#38BDF8"];
            UPDATE_NODE [label="ID Match: Replace Old Message in-place with Updated Content", fillcolor="#F59E0B", color="#FDE047"];
            APPEND_NODE [label="New ID: Append to End of Conversation Sequence", fillcolor="#047857", color="#34D399"];

            MATCH_TEST -> UPDATE_NODE [label="Yes (id='b')"];
            MATCH_TEST -> APPEND_NODE [label="No (id='c')"];
        }

        RESULT_STATE [label="Resulting Channel State\\n[Msg1(id='a'), Msg2_Updated(id='b'), Msg3(id='c')]", fillcolor="#0F172A", color="#34D399"];

        OLD_STATE -> MATCH_TEST [label="Channel inspection"];
        DELTA_IN -> MATCH_TEST [label="Evaluates incoming deltas"];
        UPDATE_NODE -> RESULT_STATE [label="Commits replacement"];
        APPEND_NODE -> RESULT_STATE [label="Commits append"];
    }
    """
    st.graphviz_chart(reducer_dot, use_container_width=True)

st.markdown("""
| Feature | Standard Python List (`+`) | LangGraph `add_messages` Reducer |
| :--- | :--- | :--- |
| **Duplicate IDs** | Appends duplicate messages blindly | Updates existing message in place (ID replacement) |
| **Deletion Support** | Requires manual slicing | Passing `RemoveMessage(id='target')` prunes message |
| **State Delta** | Must copy entire list history | Only emit newly generated messages in return dict |
| **Concurrency Safety** | Risk of race conditions | Deterministic channel reconciliation per superstep |
""")

# ==============================================================================
# SECTION 8: STRUCTURED OUTPUT: TYPE-SAFE EXTRACTION WITH PYDANTIC
# ==============================================================================
st.header("8. Structured Output: Type-Safe Extraction with Pydantic", divider="rainbow")

st.markdown("""
Modern enterprise graphs rarely rely on free-form text inside intermediate reasoning nodes.
Using `model.with_structured_output(PydanticSchema)`, the node forces the LLM to emit
strictly validated, type-safe Python data structures via native tool calling.
""")

st.code("""
from pydantic import BaseModel, Field
from typing import Literal

# 1. Define strict extraction schema
class CustomerSupportTriage(BaseModel):
    category: Literal["Billing", "Technical Bug", "Security Issue", "Feature Request"] = Field(
        description="The primary classification category of the customer request."
    )
    urgency_level: int = Field(
        ge=1, le=5,
        description="Urgency scale from 1 (lowest) to 5 (production outage)."
    )
    requires_escalation: bool = Field(
        description="True if an executive engineer or manager must be paged immediately."
    )
    summary: str = Field(
        description="One-sentence distilled summary of the core issue."
    )

# 2. Bind schema to Chat Model
structured_llm = model.with_structured_output(CustomerSupportTriage)

# 3. Invoke inside LangGraph Node
def triage_node(state: MessagesState) -> dict:
    # Model returns an instantiated Pydantic object
    classification: CustomerSupportTriage = structured_llm.invoke(state["messages"])
    
    # Update domain state channel with structured attributes
    return {
        "triage_result": classification.model_dump(),
        "messages": [AIMessage(content=f"Classified as {classification.category} (Urgency: {classification.urgency_level})")]
    }
""", language="python")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full Multi-Stage Node Pipeline with Structured Output", divider="rainbow")

st.markdown("""
Below is an end-to-end production-grade implementation of a multi-stage LangGraph workflow
featuring persona steering, dynamic prompt composition, and type-safe Pydantic structured output.
""")

st.code("""
from typing import Annotated, TypedDict, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages

# ---------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------
class ProductionSupportState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tier: Literal["Standard", "Premium", "Enterprise"]
    triage_data: Dict[str, Any]

# ---------------------------------------------------------
# 2. Structured Pydantic Output Model
# ---------------------------------------------------------
class TriageSchema(BaseModel):
    issue_type: Literal["Hardware", "Cloud Infrastructure", "Authentication", "Billing"]
    severity: Literal["SEV-1", "SEV-2", "SEV-3"]
    root_cause_hypothesis: str
    action_items: List[str]

# ---------------------------------------------------------
# 3. Model Configuration
# ---------------------------------------------------------
base_model = ChatOpenAI(model="gpt-4o", temperature=0.0)
structured_extractor = base_model.with_structured_output(TriageSchema)

# ---------------------------------------------------------
# 4. Node Definitions
# ---------------------------------------------------------
def ingest_and_frame_node(state: ProductionSupportState) -> dict:
    \"\"\"Node 1: Frame the conversation with tier-aware system instructions.\"\"\"
    tier = state.get("tier", "Standard")
    system_prompt = SystemMessage(
        content=f"You are a Tier-3 Support AI for {tier} tier enterprise clients. Enforce SLA response deadlines."
    )
    # Prepend system frame to incoming messages
    return {"messages": [system_prompt]}

def structured_triage_node(state: ProductionSupportState) -> dict:
    \"\"\"Node 2: Extract structured failure taxonomy.\"\"\"
    # Model extracts Pydantic schema
    triage_result: TriageSchema = structured_extractor.invoke(state["messages"])
    
    # Emit both structured field update and explanatory AIMessage
    summary_ai_msg = AIMessage(
        content=f"Diagnostic complete. Classified as {triage_result.severity} [{triage_result.issue_type}]."
    )
    return {
        "triage_data": triage_result.model_dump(),
        "messages": [summary_ai_msg]
    }

# ---------------------------------------------------------
# 5. Graph Assembly
# ---------------------------------------------------------
builder = StateGraph(ProductionSupportState)
builder.add_node("framing", ingest_and_frame_node)
builder.add_node("triage", structured_triage_node)

builder.add_edge(START, "framing")
builder.add_edge("framing", "triage")
builder.add_edge("triage", END)

support_graph = builder.compile()
""", language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: LLM Node Execution & Message State Simulator", divider="rainbow")

st.markdown("""
Test how different system personas, user inquiries, and structured output toggles affect
node execution, prompt composition, message state channel updates, and token consumption in real time.
""")

engine = LLMNodeExecutionEngine()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Node Configuration")
    selected_persona = st.selectbox(
        "Select Node Persona / System Prompt:",
        list(engine.personas.keys()),
        index=0
    )
    user_query = st.text_area(
        "Enter Incoming User Query / Incident Description:",
        value="Our primary PostgreSQL cluster in us-east-1 is throwing connection pool exhaustion errors and API latency spiked to 4.2s.",
        height=100
    )
    enable_structured = st.toggle("Enable Pydantic with_structured_output() Extraction", value=True)
    
    execute_button = st.button("🚀 Invoke LLM Node", type="primary")

with wb_col2:
    st.subheader("📊 Execution Inspector")
    if execute_button:
        # Simulated run
        res = engine.execute_node(
            persona_name=selected_persona,
            user_query=user_query,
            extract_structured=enable_structured,
            message_history=[]
        )
        
        st.success("✅ Node Execution Succeeded!")
        
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.metric("Latency", f"{res['state_delta']['node_execution_latency_ms']} ms")
        with m_c2:
            st.metric("Prompt Tokens", res["state_delta"]["token_usage"]["prompt_tokens"])
        with m_c3:
            st.metric("Completion Tokens", res["state_delta"]["token_usage"]["completion_tokens"])
            
        st.markdown("#### Message Sequence Transmitted to Model")
        for idx, msg in enumerate(res["full_context"]):
            role = msg["role"].upper()
            badge_class = "badge-msg" if role == "SYSTEM" else "badge-node"
            st.markdown(f"<span class='{badge_class}'>{role} [ID: {msg.get('id', 'N/A')}]</span>", unsafe_allow_html=True)
            st.text(msg["content"])
            
        if enable_structured and res["ai_msg"]["structured_output"]:
            st.markdown("#### Extracted Pydantic Object (`model_dump()`)")
            st.json(res["ai_msg"]["structured_output"])
            
        st.markdown("#### State Channel Delta Returned by Node")
        st.json({
            "messages": [m["content"] for m in res["state_delta"]["messages"]],
            "token_usage": res["state_delta"]["token_usage"]
        })
    else:
        st.info("Configure the node parameters on the left and click 'Invoke LLM Node' to observe message framing and structured output.")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: Why do LangGraph nodes return a partial dictionary update rather than mutating the state object directly?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph enforces a **pure functional state machine** paradigm. State channels reconcile updates concurrently
    using deterministic reducers (such as `add_messages`). Mutating state in-place breaks time-travel debugging,
    creates non-deterministic race conditions during parallel branching (`Send` API / parallel nodes), and corrupts
    checkpointer snapshot serializability in storage backends.
    """)

with st.expander("Q2: How does the add_messages reducer handle an incoming message with an ID identical to an existing message?", expanded=False):
    st.markdown("""
    **Answer**:
    When `add_messages` encounters a message whose `id` matches a message already present in the channel,
    it **replaces the existing message in-place** rather than appending a duplicate.
    This feature is essential for:
    1. Human-in-the-loop state edits (correcting an earlier turn).
    2. Model response streaming updates.
    3. Tool re-execution retries.
    """)

with st.expander("Q3: What occurs when an LLM returns empty content with tool_calls, and how should nodes handle it?", expanded=False):
    st.markdown("""
    **Answer**:
    When an LLM chooses to invoke tools, modern frontier models (OpenAI, Anthropic) frequently emit an `AIMessage` with
    `content=""` or `content=None` and populate `tool_calls=[...]`.
    LangGraph treats this as a valid `AIMessage`. Downstream conditional routing functions inspect `bool(response.tool_calls)`
    to branch toward a `tools` execution node rather than ending the graph.
    """)

with st.expander("Q4: What is the architectural difference between model.with_structured_output() and prompting for raw JSON?", expanded=False):
    st.markdown("""
    **Answer**:
    1. **Prompting for JSON**: Relies on prompt steering ("Output valid JSON..."). The model may hallucinate markdown code fences (```json), omit required fields, or fail JSON syntax parsing, requiring manual string regex cleanup and retry loops.
    2. **with_structured_output()**: Binds the schema directly to the model's native Tool Calling or JSON Mode API (`function_call` / `tools`). The LLM generates constrained tokens conforming to the JSON schema, and LangChain automatically parses the payload directly into an instantiated Pydantic model with runtime type validation.
    """)

with st.expander("Q5: Why is it an architectural anti-pattern to inject state variables directly into string templates inside every node?", expanded=False):
    st.markdown("""
    **Answer**:
    If every node rebuilds a giant prompt string from scratch, you destroy LLM prompt caching (Anthropic prompt caching, OpenAI prefix caching).
    Prefix caching requires the initial prompt tokens (`SystemMessage` and early turns) to remain byte-for-byte identical across turns.
    Keeping static instructions in `SystemMessage` at index 0 and appending incremental turns maximizes KV cache hit rates,
    reducing latency by up to 80% and cost by up to 50%.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Omitting add_messages Reducer on Custom State Classes")
    st.markdown("""
    *Mechanism*: Defining a state class with `messages: List[BaseMessage]` instead of `Annotated[List[BaseMessage], add_messages]`.  
    *Consequence*: Nodes returning `{"messages": [response]}` completely overwrite the entire historical conversation list, wiping out all previous user and system turns.  
    *Fix*: Always use `MessagesState` or explicitly wrap message fields with `Annotated[..., add_messages]`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Missing tool_call_id Linkage in Multi-Turn Tool Executions")
    st.markdown("""
    *Mechanism*: Returning custom `ToolMessage` payloads without matching the exact `id` provided by `AIMessage.tool_calls`.  
    *Consequence*: OpenAI and Anthropic APIs reject the subsequent turn with `400 InvalidRequestError: Tool call ID not found`.  
    *Fix*: Always instantiate `ToolMessage(content=str(result), tool_call_id=tool_call['id'])`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Prompt Token Bloat from Redundant System Messages")
    st.markdown("""
    *Mechanism*: Appending a new `SystemMessage` inside every cyclic node loop rather than maintaining a single canonical system prompt at index 0.  
    *Consequence*: Context windows bloat rapidly, increasing latency and confusing LLM role attribution.  
    *Fix*: Inspect existing state messages; if a `SystemMessage` is already present, do not append duplicate system prompts.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Unhandled Pydantic ValidationError in Structured Nodes")
    st.markdown("""
    *Mechanism*: LLM generates structured output that violates strict Pydantic bounds (e.g. integer out of range).  
    *Consequence*: The node raises unhandled `ValidationError`, crashing the entire graph execution superstep.  
    *Fix*: Implement retry loops or fallback schemas using `model.with_fallbacks()` or LangGraph validation cycles.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**add_messages**\\nThe LangGraph channel reducer combining non-destructive list append with ID-based message replacement.")
with col_fc2:
    st.info("**AIMessage.tool_calls**\\nStandardized list of dictionaries `{name, args, id}` emitted when a model triggers tool executions.")
with col_fc3:
    st.info("**with_structured_output**\\nMethod binding Pydantic or JSON schemas to chat models for type-safe structured data extraction.")
with col_fc4:
    st.info("**MessagesState**\\nBuilt-in TypedDict pre-configured with `messages: Annotated[List[BaseMessage], add_messages]`.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to call an LLM inside a standard LangGraph node function.", value=True)
c2 = st.checkbox("I understand why nodes return partial state deltas instead of mutating state in place.", value=True)
c3 = st.checkbox("I know how to construct dynamic prompts by prepending SystemMessage objects.", value=True)
c4 = st.checkbox("I understand the internal mechanics of the add_messages reducer and ID de-duplication.", value=True)
c5 = st.checkbox("I can use model.with_structured_output() with Pydantic schemas for type-safe extraction.", value=True)
c6 = st.checkbox("I know how to handle tool_calls and avoid tool_call_id mismatch errors.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 10.2 'LLM Nodes' with full architectural and practical comprehension!")
