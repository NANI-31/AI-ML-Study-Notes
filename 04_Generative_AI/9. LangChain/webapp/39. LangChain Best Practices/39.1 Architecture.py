import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="39.1 Architecture Best Practices - LangChain",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for architectural scorecard
st.markdown("""
<style>
    .arch-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-clean {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-antipattern {
        background-color: #7F1D1D;
        color: #FECACA;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class ArchitecturePatternValidatorSimulator:
    """
    Simulates architecture pattern evaluation for LangChain systems:
    detecting tight coupling, monolithic prompts, hidden state mutations,
    and evaluating adherence to Clean Architecture principles.
    """
    def __init__(self):
        self.preset_evaluations = {
            "Enterprise Support CoPilot": {
                "prompt_modularity": "High (Decoupled system, persona, and domain schemas)",
                "tool_independence": "High (Pure stateless tools with Pydantic contracts)",
                "business_logic_separation": "High (Deterministic validation isolated before LLM invocation)",
                "schema_typing": "Strict (100% Pydantic models for inputs/outputs)",
                "state_explicitness": "High (LangGraph TypedDict with add_messages reducer)",
                "abstraction_fit": "Optimal (StateGraph selected over brittle chained runnables)",
                "health_score": 96,
                "diagnosis": "Production-ready clean architecture. Minimal technical debt."
            },
            "Legacy Monolithic Agent Script": {
                "prompt_modularity": "Poor (Single 4,000-token hardcoded prompt string)",
                "tool_independence": "Poor (Tools mutating global variables and direct DB writes)",
                "business_logic_separation": "Poor (Pricing and discount rules computed by LLM prompts)",
                "schema_typing": "Weak (Raw strings and untyped dictionaries)",
                "state_explicitness": "Poor (Implicit mutable dict passed across helper functions)",
                "abstraction_fit": "Poor (Nested deprecated AgentExecutor with ad-hoc while loops)",
                "health_score": 34,
                "diagnosis": "Critical architectural antipattern. High hallucination and failure risk."
            }
        }

    def evaluate_system(self, preset_key: str) -> Dict[str, Any]:
        return self.preset_evaluations.get(preset_key, self.preset_evaluations["Enterprise Support CoPilot"])


# --- LESSON HEADER ---
st.title("39.1 LangChain Architecture Best Practices")
st.markdown("""
Architecting production AI applications requires applying disciplined software engineering principles to
probabilistic systems. As systems scale, unstructured prompts, tightly coupled tools, and hidden state mutations
lead to maintenance collapse and unpredictable failures. This module covers the foundational architectural standards
for building modular, maintainable, and verifiable LangChain and LangGraph applications.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise AI Architecture Standards", divider="rainbow")
st.markdown("""
A clean LangChain architecture separates non-deterministic model generation from deterministic application rules.
Client requests pass through validation boundaries, interact with modular prompts, invoke stateless and independent
tools, and mutate explicit, type-safe state graphs rather than relying on global variables or monolithic chains.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Architecture_Best_Practices_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_presentation {
            label = "Presentation Layer (FastAPI / Streamlit / React)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            ClientReq [label="Client Request\\n(Typed Pydantic Schema)", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_application_core {
            label = "Core Application Layer (LangGraph Runtime)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            StateSchema [label="Explicit Typed State\\n(TypedDict + Pydantic + add_messages reducer)", fillcolor="#1E293B", color="#60A5FA"];
            ModularPrompts [label="Modular Prompt Engine\\n(System Persona + Context Placeholder + Safety Bounds)", fillcolor="#1E293B", color="#60A5FA"];
            DeterministicRouter [label="Deterministic Business Guard\\n(Pre-validation & Business Rules Enforcement)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_tools_services {
            label = "Independent Infrastructure & Tool Layer";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            StatelessTools [label="Stateless Composable Tools\\n(Pydantic Args + Isolated Execution + Pure Side Effects)", fillcolor="#0369A1", color="#38BDF8"];
            PersistenceStore [label="Persistent Checkpointers\\n(PostgresSaver / Redis Checkpointer per Thread)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_models_parsers {
            label = "Model & Output Parsing Layer";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ChatModel [label="Model Invocation Core\\n(temperature=0.0 + Native Structured Output)", fillcolor="#065F46", color="#10B981"];
            StructuredValidator [label="Output Schema Validator\\n(Pydantic Output Validation & Error Traps)", fillcolor="#065F46", color="#10B981"];
        }

        ClientReq -> DeterministicRouter [label="Validate Schema"];
        DeterministicRouter -> StateSchema [label="Initialize Session Thread"];
        StateSchema -> ModularPrompts [label="Inject State History"];
        ModularPrompts -> ChatModel [label="Format Messages"];
        ChatModel -> StatelessTools [label="Dispatched Tool Calls"];
        StatelessTools -> StateSchema [label="Return ToolMessage"];
        ChatModel -> StructuredValidator [label="Parse Structured Completion"];
        StructuredValidator -> PersistenceStore [label="Persist Final Checkpoint"];
        StructuredValidator -> ClientReq [label="200 OK Response"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: KEEP PROMPTS MODULAR ---
st.header("2. Modular Prompt Engineering: Decoupling Instructions, Examples, and Context", divider="rainbow")
st.markdown("""
Monolithic prompt strings that mix system persona, business constraints, formatting instructions,
and few-shot examples into one massive f-string represent a severe maintenance antipattern.
Best practice mandates breaking prompts into modular, independently testable templates combined via composition.
""")

col_pm1, col_pm2 = st.columns(2)
with col_pm1:
    st.markdown("### Modular Prompt Design Principles")
    st.markdown("""
    - **Single Responsibility**: Maintain separate templates for persona, business guardrails, formatting rules, and dynamic few-shot examples.
    - **Composition via Pipelines**: Combine modular fragments using `ChatPromptTemplate.from_messages()` or string concatenation of partial templates.
    - **Version Control & Testability**: Modular templates can be unit-tested individually with mock inputs to verify formatting boundary conditions.
    """)
with col_pm2:
    st.code("""
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

# 1. Decoupled, reusable system persona
PERSONA_PROMPT = SystemMessagePromptTemplate.from_template(
    "You are an enterprise financial analytics co-pilot for {organization}."
)

# 2. Decoupled regulatory constraints
COMPLIANCE_PROMPT = SystemMessagePromptTemplate.from_template(
    "COMPLIANCE GUARDRAIL: Never provide speculative investment advice. "
    "Always state that past performance does not guarantee future results."
)

# 3. Composed modular pipeline
chat_prompt = ChatPromptTemplate.from_messages([
    PERSONA_PROMPT,
    COMPLIANCE_PROMPT,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{user_query}")
])
    """, language="python")

# --- SECTION 3: KEEP TOOLS INDEPENDENT ---
st.header("3. Independent & Composable Tools: Pure Functions, Isolated Side Effects, and Schemas", divider="rainbow")
st.markdown("""
Tools must be designed as self-contained, independent units of functionality. A tool should do one thing well,
require explicit typed arguments via Pydantic, and never rely on hidden global application state or cross-tool dependencies.
""")

st.code("""
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class CustomerQuerySchema(BaseModel):
    customer_id: str = Field(..., description="Unique customer alphanumeric ID, e.g. CUST-9921")
    include_archived: bool = Field(False, description="Whether to include closed/archived support tickets")

@tool(args_schema=CustomerQuerySchema)
def fetch_customer_profile(customer_id: str, include_archived: bool = False) -> str:
    \"\"\"Fetches verified profile and ticket history for a designated customer ID.\"\"\"
    # Best Practice: Stateless execution with explicit parameter inputs
    # Never mutate global state, never assume preceding tool state
    return f"Retrieved profile for {customer_id}: Status Active, 0 open tickets."
""", language="python")

# --- SECTION 4: SEPARATE BUSINESS LOGIC FROM AGENTS ---
st.header("4. Separation of Concerns: Decoupling Business Logic from Probabilistic Agent Loops", divider="rainbow")
st.markdown("""
A dangerous antipattern is asking the LLM to execute deterministic mathematical calculations, financial pricing rules,
or legal compliance checks via prompt instructions. LLMs are probabilistic text predictors. Deterministic business rules
must be implemented in pure Python or dedicated microservices, with the LLM acting solely as an intent router and conversational synthesizer.
""")

col_sep1, col_sep2 = st.columns(2)
with col_sep1:
    st.markdown("### Division of Responsibility")
    st.markdown("""
    - **LLM Responsibility**: Natural language understanding, intent classification, parameter extraction, and human-friendly response synthesis.
    - **Code/Service Responsibility**: Pricing algorithms, tax calculation, authorization verification, database write validations, and financial arithmetic.
    - **The Rule**: If a requirement can be written as a Python `if/else` or SQL query, do NOT delegate it to prompt engineering.
    """)
with col_sep2:
    st.code("""
# ANTIPATTERN (Asking LLM to calculate discounts)
# Prompt: "If user spent > $1000, apply 15% discount, else 5%."

# BEST PRACTICE: Deterministic Python business logic
def calculate_enterprise_discount(total_spend: float) -> float:
    if total_spend > 1000.0:
        return total_spend * 0.85
    return total_spend * 0.95

@tool
def compute_order_total(items_cost: float) -> str:
    \"\"\"Calculates order total using enterprise tier discount rules.\"\"\"
    final_price = calculate_enterprise_discount(items_cost)
    return f"Calculated discounted total: ${final_price:.2f}"
    """, language="python")

# --- SECTION 5: USE TYPED SCHEMAS ---
st.header("5. Typed Schemas & Interface Contracts: Pydantic Validation Across All Boundaries", divider="rainbow")
st.markdown("""
Untyped dictionaries and raw string returns create brittle pipelines that fail silently.
Best-in-class LangChain architectures enforce strict Pydantic models at every interface boundary:
request payloads, tool input schemas, intermediate state definitions, and model outputs.
""")

st.code("""
from pydantic import BaseModel, Field
from typing import List, Optional

# Strict input contract
class AnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5)
    lookback_days: int = Field(30, ge=1, le=365)

# Strict structured output contract
class SentimentVerdict(BaseModel):
    sentiment: str = Field(..., regex="^(Bullish|Bearish|Neutral)$")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    primary_catalysts: List[str] = Field(..., min_items=1)
    risk_factors: Optional[List[str]] = Field(default_factory=list)

# Enforced output schema binding
structured_llm = chat_model.with_structured_output(SentimentVerdict)
""", language="python")

# --- SECTION 6: KEEP STATE EXPLICIT ---
st.header("6. Explicit State Management: Type-Safe Channel Reducers and State Transitions", divider="rainbow")
st.markdown("""
In conversational and multi-step agents, implicit state (passing arbitrary dictionaries or relying on class attributes)
causes race conditions and debugging nightmares. Using LangGraph's explicit state schemas with typed channels
and message reducers (`add_messages`) guarantees transparent, audit-ready state updates.
""")

st.code("""
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Explicit, typed agent state definition
class AgentWorkflowState(TypedDict):
    # add_messages ensures new turns append to history rather than overwriting
    messages: Annotated[Sequence[BaseMessage], add_messages]
    active_user_id: str
    security_clearance_level: int
    is_terminal_state: bool

def audit_node(state: AgentWorkflowState) -> dict:
    # Explicit state mutation: return only modified channel values
    return {"is_terminal_state": True}
""", language="python")

# --- SECTION 7: USE APPROPRIATE ABSTRACTIONS ---
st.header("7. Appropriate Abstraction Selection: Choosing LCEL Runnables vs LangGraph vs Python", divider="rainbow")
st.markdown("""
Over-engineering with the wrong abstraction introduces unnecessary latency and maintenance overhead.
Architects must select the simplest abstraction that completely fulfills the requirements:
""")

col_abs1, col_abs2 = st.columns(2)
with col_abs1:
    st.markdown("### Abstraction Decision Matrix")
    st.markdown("""
    - **Pure Python**: Use standard Python for deterministic formatting, data transforms, and business arithmetic.
    - **LCEL Runnables (`prompt | model | parser`)**: Use for straightforward linear chains without loops or complex state transitions.
    - **LangGraph (`StateGraph`)**: Mandatory for cyclic agents, multi-agent systems, human-in-the-loop pauses, or persistent thread checkpointers.
    - **Antipattern**: Deprecated `AgentExecutor` and `LLMChain` classes should never be used in modern v0.3+ LangChain systems.
    """)
with col_abs2:
    st.code("""
# 1. Simple Linear Pipeline -> Use LCEL
linear_chain = prompt | model | parser

# 2. Multi-turn Agent with State & Tools -> Use LangGraph
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentWorkflowState)
builder.add_node("agent", call_model_node)
builder.add_node("tools", execute_tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "agent")
graph = builder.compile()
    """, language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Clean Architecture Enterprise Agent Application", divider="rainbow")
st.markdown("""
Here is an enterprise-grade agent template integrating modular prompts, stateless typed tools,
deterministic pre-validation, and an explicit LangGraph state machine.
""")

st.code("""
# clean_agent_architecture.py
from typing import TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. Input Contract
class UserQuery(BaseModel):
    user_id: str = Field(..., min_length=3)
    query: str = Field(..., min_length=1)

# 2. Independent Stateless Tool
@tool
def lookup_account_tier(user_id: str) -> str:
    \"\"\"Retrieves subscription entitlement tier.\"\"\"
    return "Enterprise Platinum"

# 3. Explicit State Schema
class WorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str

# 4. Modular Prompt Composition
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enterprise account manager assistant."),
    MessagesPlaceholder("messages")
])

# 5. Compiled Architecture
# Pre-validation executed deterministically before invoking graph
def process_request(request: UserQuery):
    initial_state: WorkflowState = {
        "messages": [HumanMessage(content=request.query)],
        "user_id": request.user_id
    }
    # Invoke compiled graph with thread checkpoint configuration
    return app_graph.invoke(initial_state)
""", language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Architecture Anti-Pattern vs Clean Pattern Validator", divider="rainbow")
st.markdown("""
Evaluate architectural quality scores across sample agent systems to identify tight coupling,
untyped interfaces, and state mutation risks.
""")

validator = ArchitecturePatternValidatorSimulator()

col_sel, col_score = st.columns([1, 1])

with col_sel:
    st.subheader("🏛️ Architecture Evaluation Target")
    chosen_target = st.selectbox(
        "Select System Architecture to Audit:",
        list(validator.preset_evaluations.keys())
    )
    audit_btn = st.button("🚀 Run Architecture Scorecard Audit", type="primary")

with col_score:
    st.subheader("📋 Architectural Assessment Scorecard")
    if audit_btn:
        res = validator.evaluate_system(chosen_target)
        badge = f"<span class='badge-clean'>SCORE: {res['health_score']}/100</span>" if res["health_score"] >= 80 else f"<span class='badge-antipattern'>SCORE: {res['health_score']}/100</span>"
        
        st.markdown(
            f"<div class='arch-card'>"
            f"<b>System:</b> {chosen_target} &nbsp;|&nbsp; {badge}<br><br>"
            f"<b>Prompt Modularity:</b> {res['prompt_modularity']}<br>"
            f"<b>Tool Independence:</b> {res['tool_independence']}<br>"
            f"<b>Business Logic Separation:</b> {res['business_logic_separation']}<br>"
            f"<b>Schema Typing:</b> {res['schema_typing']}<br>"
            f"<b>State Explicitness:</b> {res['state_explicitness']}<br>"
            f"<b>Abstraction Fit:</b> {res['abstraction_fit']}<br><br>"
            f"<b>Diagnosis:</b> {res['diagnosis']}"
            f"</div>",
            unsafe_allow_html=True
        )
        if res["health_score"] >= 80:
            st.success("Clean Architecture verified: system ready for enterprise continuous deployment.")
        else:
            st.error("Architecture refactoring required: significant antipatterns detected.")
    else:
        st.info("Select a system profile and run the audit to view architectural telemetry.")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why should deterministic business logic never be delegated to LLM prompt instructions?"):
    st.markdown("""
    **Answer**: Large Language Models are probabilistic text generators that predict token probability distributions.
    Delegating mathematical calculations (e.g. pricing, discounts, tax rates) or regulatory constraints to prompts exposes the system
    to non-deterministic calculations, hallucinations, and prompt injection circumvention. Deterministic rules must be written in pure Python
    or SQL, with the LLM acting solely as an intent router and orchestrator.
    """)

with st.expander("Q2: What is the primary difference between LCEL Runnables and LangGraph in clean architecture?"):
    st.markdown("""
    **Answer**: LCEL Runnables are designed for linear Directed Acyclic Graphs (DAGs) without loops or branching state mutations (e.g. `prompt | model | parser`).
    LangGraph provides stateful, cyclic orchestration capable of loops, multi-agent communication, human-in-the-loop pauses, and persistent checkpointers.
    Using LangGraph for a simple single-turn prompt-parser pipeline introduces unnecessary complexity; conversely, using LCEL for cyclic agents requires brittle while loops.
    """)

with st.expander("Q3: How does prompt modularity enhance automated testing in continuous integration?"):
    st.markdown("""
    **Answer**: Monolithic prompts must be tested as black boxes with expensive live LLM calls. Modular prompts separate personas, compliance rules,
    and user inputs into discrete templates. Individual modules can be unit-tested deterministically with pytest for variable binding, token counts,
    and injection sanitization without making network calls to model providers.
    """)

with st.expander("Q4: Why must LangChain tools be stateless pure functions?"):
    st.markdown("""
    **Answer**: If tools mutate global class variables or depend on the order of previous tool calls, the system becomes non-reproducible and fails in
    concurrent multi-user environments. Stateless tools take explicit typed arguments, return structured responses, and communicate findings back to the
    agent through explicit state updates, preserving thread isolation.
    """)

with st.expander("Q5: What role does Pydantic play in LangChain interface boundaries?"):
    st.markdown("""
    **Answer**: Pydantic enforces runtime data validation and schema documentation. It defines input contracts for API routes, enforces type constraints
    on tool arguments, validates structured LLM completions via JSON schema reflection, and ensures internal agent states remain strictly typed,
    preventing runtime `KeyError` or attribute exceptions.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The God-Prompt Antipattern)**: Concatenating 50 business rules, tool documentation, and examples into one 5,000-token prompt. Changing one rule degrades model performance on all unrelated tasks.")
    st.warning("**Trap 2 (Hidden Global State in Tools)**: Storing user IDs or active database connections in global Python variables inside tool modules, corrupting data when serving concurrent requests.")
with col_trap2:
    st.error("**Trap 3 (Untyped Output Parsing)**: Relying on raw string slicing or unvalidated `json.loads()` on LLM completions. A single unexpected markdown fence (```json) crashes downstream services.")
    st.warning("**Trap 4 (Using Deprecated Abstractions)**: Continuing to build new projects with legacy `LLMChain` or `AgentExecutor`. These lack native async streaming, checkpoint persistence, and fine-grained state control.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the core rule for separating business logic from LLMs?")
        st.caption("Flip for answer:")
        st.info("**Back**: If a rule can be computed with deterministic code (Python/SQL), never delegate it to model prompts.")
    with st.container(border=True):
        st.markdown("**Front**: When should LangGraph be chosen over LCEL Runnables?")
        st.caption("Flip for answer:")
        st.info("**Back**: When the workflow requires loops, multi-agent collaboration, persistent state checkpointing, or human-in-the-loop pauses.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: Why must custom tools define a Pydantic `args_schema`?")
        st.caption("Flip for answer:")
        st.info("**Back**: To provide the LLM with exact argument types and descriptions, and enforce runtime input validation.")
    with st.container(border=True):
        st.markdown("**Front**: What does the `add_messages` reducer prevent in LangGraph state?")
        st.caption("Flip for answer:")
        st.info("**Back**: It prevents new turns from overwriting previous message history, automatically appending messages with ID deduplication.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I can decompose monolithic prompt strings into modular, composable templates.", value=True)
c2 = st.checkbox("I understand how to design stateless tools with explicit Pydantic args_schemas.", value=True)
c3 = st.checkbox("I know how to separate deterministic business calculations from probabilistic agent loops.", value=True)
c4 = st.checkbox("I can enforce typed schemas across request, state, and structured output boundaries.", value=True)
c5 = st.checkbox("I know how to define explicit TypedDict state schemas with add_messages reducers in LangGraph.", value=True)
c6 = st.checkbox("I can select the appropriate abstraction between pure Python, LCEL Runnables, and LangGraph.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 39.2: Performance.")
