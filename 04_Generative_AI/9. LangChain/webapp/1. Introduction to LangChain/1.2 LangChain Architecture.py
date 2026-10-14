import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="1.2 LangChain Architecture - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("1.2 LangChain Architecture: Primitives, Components & Runtimes")
st.caption("Chapter 1: Introduction to LangChain | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Building enterprise Generative AI systems requires coordinating diverse computational primitives: foundational models, 
prompt templates, token encoders, vector indexes, external tool executors, structured parsers, and telemetry spans.

In modern **LangChain v0.3+**, these disparate subsystems are decoupled into modular, interoperable layers governed by the 
universal `Runnable` protocol. By standardizing component interfaces while isolating provider implementations into dedicated 
packages, LangChain allows developers to swap embedding models, vector stores, and foundation LLMs with zero architectural friction.

In this lesson, you will master:
- **Core Component Hierarchy**: The 13 structural pillars of modern LangChain architecture.
- **Parametric vs Non-Parametric Primitives**: Models, Embeddings, Retrievers, and Vector Stores.
- **The Modern Chat Model Shift**: Why legacy string-in/string-out LLMs were superseded by structured Chat Models.
- **Runtimes & Telemetry**: How LangGraph coordinates cyclic agent states while LangSmith profiles distributed spans.
- **Systems Interview Readiness**: Component trade-offs, production failure modes, and active recall.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class ComponentTelemetry:
    def __init__(self, name: str, category: str, input_type: str, output_type: str, latency_ms: int):
        self.name = name
        self.category = category
        self.input_type = input_type
        self.output_type = output_type
        self.latency_ms = latency_ms

class ArchitecturePipelineSimulator:
    """Simulates multi-component execution across the LangChain stack in module scope."""
    def __init__(self):
        self.catalog = {
            "PromptTemplate": ComponentTelemetry("ChatPromptTemplate", "Prompt", "Dict[str, Any]", "PromptValue", 2),
            "ChatModel": ComponentTelemetry("ChatOpenAI (gpt-4o)", "Model", "List[BaseMessage]", "AIMessage", 420),
            "Embeddings": ComponentTelemetry("OpenAIEmbeddings", "Embedding", "List[str]", "List[List[float]]", 65),
            "VectorStore": ComponentTelemetry("ChromaDB / PGVector", "Index", "List[float]", "List[Document]", 18),
            "Retriever": ComponentTelemetry("VectorStoreRetriever", "Retriever", "str", "List[Document]", 22),
            "OutputParser": ComponentTelemetry("PydanticOutputParser", "Parser", "AIMessage", "PydanticModel", 4),
            "Tool": ComponentTelemetry("SQLDatabaseTool", "Actuator", "Dict[str, Any]", "ToolMessage", 85)
        }

    def simulate_pipeline(self, active_components: List[str]) -> Dict[str, Any]:
        trace = []
        total_latency = 0
        for comp_key in active_components:
            comp = self.catalog.get(comp_key)
            if comp:
                total_latency += comp.latency_ms
                trace.append({
                    "Component": comp.name,
                    "Category": comp.category,
                    "Consumes": comp.input_type,
                    "Produces": comp.output_type,
                    "Latency (ms)": comp.latency_ms
                })
        return {
            "total_latency_ms": total_latency,
            "trace": trace,
            "status": "VALID_COMPOSITION" if len(trace) >= 2 else "INCOMPLETE_PIPELINE"
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The LangChain v0.3+ Component Hierarchy", divider="rainbow")

dot_blueprint = """
digraph LangChainHierarchy {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_prompting {
        label="1. Ingestion & Prompt Layer";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        prompts [label="Prompts & Templates\\n(ChatPromptTemplate, FewShotPrompt)"];
        messages [label="Unified Messages Hierarchy\\n(SystemMessage, HumanMessage, AIMessage, ToolMessage)"];
    }

    subgraph cluster_models {
        label="2. Model & Embedding Layer";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        chat_models [label="Chat Models (BaseChatModel)\\nStructured Outputs & Tool Calling Binding"];
        embeddings [label="Embeddings Engine\\nDense Vector Projection ([D] Dimensions)"];
    }

    subgraph cluster_knowledge {
        label="3. Knowledge & Retrieval Layer";
        fontcolor="#10B981";
        style=dashed;
        color="#334155";
        vectorstores [label="Vector Stores (Index DB)\\nChromaDB, Pinecone, PGVector, Qdrant"];
        retrievers [label="Retrievers\\nHybrid Search, Contextual Compression, BM25"];
    }

    subgraph cluster_actuation {
        label="4. Tool Actuation & Parsing Layer";
        fontcolor="#EC4899";
        style=dashed;
        color="#334155";
        tools [label="Tools (@tool)\\nJSON Schema Declarations & External Actuators"];
        parsers [label="Output Parsers\\nStrOutputParser, JsonOutputParser, PydanticOutputParser"];
    }

    subgraph cluster_runtimes {
        label="5. Multi-Agent Runtimes & Observability";
        fontcolor="#A855F7";
        style=dashed;
        color="#334155";
        langgraph [label="LangGraph State Engine\\nCyclic Graphs, Memory Checkpointing & HITL"];
        langsmith [label="LangSmith Telemetry\\nDistributed Tracing, Spans & Evaluation"];
    }

    prompts -> messages [label="Renders"];
    messages -> chat_models [label="Ingests"];
    embeddings -> vectorstores [label="Populates"];
    vectorstores -> retrievers [label="Exposes"];
    retrievers -> prompts [label="Injects Context"];
    chat_models -> tools [label="Tool Calls", dir=both];
    chat_models -> parsers [label="Raw Tokens"];
    parsers -> langgraph [label="State Updates"];
    chat_models -> langsmith [style=dotted, color="#A855F7", label="Tracing"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

# --- SECTION 2: LLMS ---
st.header("2. LLMs: Pure Text-Completion Engines vs Modern Conversational Standards", divider="rainbow")

st.markdown("""
In the LangChain class hierarchy, a legacy **`LLM`** (subclass of `BaseLLM`) represents a pure text-completion model. 
It accepts a single raw string prompt as input and returns a single raw string completion:
```
String In -> String Out
```
Examples include legacy models such as `text-davinci-003` or raw base foundation checkpoints (e.g. base `Llama-3-8B` without instruction tuning).

### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Legacy `LLMs` are now virtually obsolete in production application stacks. All leading providers (OpenAI, Anthropic, Google, 
Mistral, Meta) optimize their models strictly for multi-turn structured conversations and JSON Schema tool calling. 
Modern LangChain applications mandate **`BaseChatModel`** instead of `BaseLLM`.
""")

# --- SECTION 3: CHAT MODELS ---
st.header("3. Chat Models: Multi-Turn Dialog, Tool-Calling, and Structured Output Backends", divider="rainbow")

st.markdown("""
A **Chat Model** (subclass of `BaseChatModel`) represents a foundation model explicitly trained on conversational message sequences. 
Unlike legacy LLMs, a Chat Model accepts a list of structured message objects as input and returns an `AIMessage` as output:
```
List[BaseMessage] -> AIMessage
```
Modern Chat Models natively support:
1. **Tool Binding**: Via `.bind_tools([tool1, tool2])`, extracting OpenAPI-compliant function definitions.
2. **Structured Outputs**: Via `.with_structured_output(Schema)`, guaranteeing typed Pydantic responses.
3. **Usage Telemetry**: Every returned `AIMessage` contains an explicit `usage_metadata` dictionary detailing input tokens, 
   output tokens, and total tokens.
""")

# --- SECTION 4: MESSAGES ---
st.header("4. Messages: SystemMessage, HumanMessage, AIMessage, and ToolMessage Lifecycle", divider="rainbow")

st.markdown("""
LangChain standardizes all conversational dialog into four primary message primitives that inherit from `BaseMessage`:
""")

msg_cols = st.columns(2)
with msg_cols[0]:
    st.subheader("1. `SystemMessage`")
    st.markdown("Defines model persona, operational constraints, output formatting rules, and guardrails.")
    
    st.subheader("2. `HumanMessage`")
    st.markdown("Represents end-user inputs, prompts, queries, or event triggers delivered to the assistant.")

with msg_cols[1]:
    st.subheader("3. `AIMessage`")
    st.markdown("Contains the model's generated text, thoughts, or `tool_calls` payloads requesting external function execution.")
    
    st.subheader("4. `ToolMessage`")
    st.markdown("Delivers the execution result of an external tool back to the LLM, correlated by `tool_call_id`.")

# --- SECTION 5: PROMPTS ---
st.header("5. Prompts: ChatPromptTemplate, Message Placeholders, and Few-Shot Ingestion", divider="rainbow")

st.markdown("""
A prompt in modern LangChain is rarely a static string. The **`ChatPromptTemplate`** represents a composable recipe for producing 
typed message sequences. It supports:
- **Variable Substitution**: Dynamic interpolation using `{variable_name}` without fragile f-strings.
- **`MessagesPlaceholder`**: Dynamically injecting conversation histories or lists of retrieved messages into a template.
- **Few-Shot Ingestion**: Formatting dynamic demonstrations to guide model reasoning on complex classification tasks.
""")

# --- SECTION 6: TOOLS ---
st.header("6. Tools: The @tool Decorator, JSON Schema Generation, and Sandbox Execution", divider="rainbow")

st.markdown("""
A **Tool** in LangChain encapsulates an external capability (database query, web scraper, Python sandbox, REST API call). 
Modern LangChain provides the `@tool` decorator, which inspects function signatures and docstrings to automatically generate 
the JSON Schema required by foundation models:
""")

st.code("""
from langchain_core.tools import tool

@tool
def calculate_vat(amount: float, country_code: str) -> float:
    \"\"\"Calculates applicable Value Added Tax for a given transaction amount and country.\"\"\"
    rates = {"GB": 0.20, "DE": 0.19, "FR": 0.20}
    return round(amount * rates.get(country_code.upper(), 0.15), 2)
""", language="python")

# --- SECTION 7: AGENTS ---
st.header("7. Agents: Autonomous Reasoning Loops and Dynamic Tool Actuation", divider="rainbow")

st.markdown("""
While an LCEL chain executes a predetermined sequence of steps, an **Agent** uses the model as a reasoning core to determine 
which actions to take, in what order, and when to terminate. 

In legacy LangChain, agents were executed by `AgentExecutor`. In modern LangChain, complex multi-step and cyclic agent 
workflows are orchestrated using **LangGraph**, providing checkpoint persistence and Human-in-the-Loop oversight.
""")

# --- SECTION 8: RETRIEVERS ---
st.header("8. Retrievers: Dynamic Context Ingestion and the Vector Search Interface", divider="rainbow")

st.markdown("""
A **Retriever** is a lightweight, standardized abstraction that accepts an unstructured string query and returns a list of 
`Document` objects (`str -> List[Document]`). Unlike vector stores, a retriever does not need to store documents physically; 
it can wrap external search APIs (Elasticsearch, Tavily, Wikipedia, or hybrid reranking algorithms).
""")

# --- SECTION 9: VECTOR STORES ---
st.header("9. Vector Stores: Dense Vector Indexing, Metadata Filtering, and Similarity Search", divider="rainbow")

st.markdown("""
A **Vector Store** is an index database designed to persist dense mathematical embeddings alongside raw text chunks and metadata. 
Standard methods include `.add_documents()`, `.similarity_search()`, and `.max_marginal_relevance_search()` (MMR). 
LangChain integrates natively with over 60 vector engines, including Chroma, Pinecone, Weaviate, Qdrant, and PostgreSQL (`pgvector`).
""")

# --- SECTION 10: EMBEDDINGS ---
st.header("10. Embeddings: Dense Vector Representations and Semantic Projection", divider="rainbow")

st.markdown("""
The **Embeddings** interface (`BaseEmbeddings`) defines two core methods:
- `.embed_documents(texts: List[str]) -> List[List[float]]`: Batch encodes multiple text passages for ingestion into vector stores.
- `.embed_query(text: str) -> List[float]`: Encodes a single user search query into the identical latent space for distance calculation.
""")

# --- SECTION 11: OUTPUT PARSERS ---
st.header("11. Output Parsers: Pydantic, JSON, Str, and Streaming Chunk Assembly", divider="rainbow")

st.markdown("""
An **Output Parser** transforms raw text or `AIMessage` chunks into structured data. Essential parsers include:
- `StrOutputParser`: Extracts raw string content from an `AIMessage`.
- `JsonOutputParser`: Validates and parses raw text containing JSON into a Python dictionary.
- `PydanticOutputParser`: Enforces full schema validation against a Pydantic v2 model.
""")

# --- SECTION 12: MIDDLEWARE & CALLBACKS ---
st.header("12. Middleware & Callbacks: Event Streaming, Run Trees, and Cross-Cutting Hooks", divider="rainbow")

st.markdown("""
The **Callbacks** system provides cross-cutting hooks (`on_llm_start`, `on_tool_end`, `on_chain_error`) that observe execution 
without modifying business logic. In modern LangChain, callbacks serve as the underlying transport mechanism that pushes 
telemetry spans directly into LangSmith and drives UI token streaming.
""")

# --- SECTION 13: LANGGRAPH RUNTIME ---
st.header("13. LangGraph Runtime: Cyclic Multi-Agent State Machines and Memory Checkpoints", divider="rainbow")

st.markdown("""
When applications outgrow linear DAG chains and require **loops, self-correction, memory persistence, or human intervention**, 
they transition to the **LangGraph** runtime. LangGraph coordinates stateful execution graphs where nodes represent functions/agents 
and edges define conditional routing logic.
""")

# --- SECTION 14: LANGSMITH ---
st.header("14. LangSmith: Distributed Telemetry, Tracing, and Automated Evaluation", divider="rainbow")

st.markdown("""
**LangSmith** is the enterprise observability platform purpose-built for LangChain and LangGraph. It provides:
1. **Nested Run Trees**: Visual inspection of exact prompt inputs, intermediate retriever outputs, and final tokens.
2. **Cost & Latency Accounting**: Instant attribution of dollar costs across models and teams.
3. **Continuous Evaluation**: Running unit tests against golden reference datasets to catch model regressions.
""")

# --- SECTION 15: PRODUCTION IMPLEMENTATION ---
st.header("15. Production Implementation: Multi-Component RAG and Tool Calling Pipeline", divider="rainbow")

st.markdown("""
The following complete Python snippet illustrates how prompts, models, tools, and parsers assemble into a unified 
LCEL composition:
""")

st.code("""
from typing import Dict, Any, List
from pydantic import BaseModel, Field

# 1. Structured Contract
class TicketResolution(BaseModel):
    ticket_id: str = Field(description="Unique ticket identifier")
    priority: str = Field(description="LOW, MEDIUM, HIGH, URGENT")
    suggested_fix: str = Field(description="Technical remediation plan")
    escalate_to_human: bool = Field(description="True if automated fix is insufficient")

# 2. Complete LCEL Pipeline Architecture
'''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

# Initialize model
model = init_chat_model("gpt-4o", model_provider="openai", temperature=0.0)
structured_model = model.with_structured_output(TicketResolution)

# Compose prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Enterprise Site Reliability Engineer. Triage incoming incident alerts."),
    ("human", "Alert payload: {alert_text}\\nContext: {context}")
])

# Assemble pipeline using RunnablePassthrough for context injection
triage_pipeline = (
    {"alert_text": RunnablePassthrough(), "context": lambda x: "Cluster: us-east-prod; Node: k8s-worker-04"}
    | prompt
    | structured_model
)

# Execute
result: TicketResolution = triage_pipeline.invoke("High memory pressure on pod api-gateway-7b")
print(result.priority)
'''
""", language="python")

# --- SECTION 16: INTERACTIVE WORKBENCH ---
st.header("16. Interactive Workbench: Component Assembly and Architecture Inspector", divider="rainbow")

st.markdown("""
Assemble and profile components across the LangChain stack. Toggle components below to inspect how data passes 
between layers and measure simulated cumulative latency.
""")

sim = ArchitecturePipelineSimulator()

st.subheader("Select Pipeline Components")
c_col1, c_col2 = st.columns(2)
with c_col1:
    use_prompt = st.checkbox("ChatPromptTemplate (Prompt Assembly)", value=True)
    use_model = st.checkbox("ChatModel (Foundation LLM)", value=True)
    use_parser = st.checkbox("OutputParser (Structured Parsing)", value=True)
with c_col2:
    use_retriever = st.checkbox("Retriever (Dynamic Context)", value=False)
    use_tool = st.checkbox("Tool (@tool Function Actuation)", value=False)

active_comps = []
if use_prompt: active_comps.append("PromptTemplate")
if use_retriever: active_comps.append("Retriever")
if use_model: active_comps.append("ChatModel")
if use_tool: active_comps.append("Tool")
if use_parser: active_comps.append("OutputParser")

if st.button("🚀 Analyze Pipeline Composition", type="primary"):
    res = sim.simulate_pipeline(active_comps)
    st.write(f"### Cumulative Estimated Latency: `{res['total_latency_ms']} ms`")
    st.dataframe(pd.DataFrame(res["trace"]), use_container_width=True)

# --- SECTION 17: TECHNICAL INTERVIEW PREPARATION ---
st.header("17. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why did LangChain deprecate LLM in favor of BaseChatModel?",
     "Modern foundation models are trained on conversational message roles (System, User, Assistant, Tool). Raw LLMs only support unstructured strings, whereas BaseChatModel natively supports multi-turn message arrays, token usage telemetry, and JSON Schema tool calling."),
    ("What is the difference between a Retriever and a Vector Store?",
     "A Vector Store physically stores embeddings and document chunks on disk, exposing low-level similarity search methods. A Retriever is a lightweight functional abstraction (str -> List[Document]) that may wrap a vector store, an external search API, or a hybrid reranking algorithm."),
    ("How does LangChain handle tool calling behind the scenes?",
     "When tools decorated with @tool are bound via .bind_tools(), LangChain parses their type hints and docstrings into an OpenAPI JSON Schema. When the model emits a tool call request, LangChain captures the function arguments in an AIMessage tool_calls array."),
    ("What is the role of MessagesPlaceholder in ChatPromptTemplate?",
     "MessagesPlaceholder reserves a dynamic slot within a prompt template where an arbitrary list of BaseMessage objects (such as multi-turn conversation memory or retrieved chat history) can be injected at runtime."),
    ("How does the Callbacks system decouple telemetry from core pipeline logic?",
     "Callbacks act as an event-driven observer pattern. Standard handlers intercept lifecycle events (on_llm_start, on_tool_end) to stream tokens or record trace spans to LangSmith without injecting logging code into business logic.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Mixing Legacy LLM and ChatModel APIs. Passing raw string prompts to methods expecting List[BaseMessage] causes validation crashes. Always structure inputs as messages.")
    st.error("**Trap 2 (Breaking Error)**: Incomplete Tool Docstrings. The LLM decides whether to invoke a tool based entirely on its description. Vague docstrings result in hallucinated tool selections.")
with trap_c2:
    st.warning("**Trap 3 (Performance Warning)**: High Latency from Over-Chaining. Adding multiple sequential LLM calls in one pipeline compounds latency (e.g. 3 x 800ms = 2.4s). Use RunnableParallel where steps are independent.")
    st.warning("**Trap 4 (Cost Warning)**: Unbounded Retrieval Context. Feeding large top-k documents into prompt templates without reranking inflates input token costs and causes the model to miss relevant facts in the middle.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: BaseMessage Subtypes"):
        st.markdown("**SystemMessage (rules), HumanMessage (user), AIMessage (model/tools), ToolMessage (result).**")
    with st.expander("🎴 Flashcard 2: Retriever Contract"):
        st.markdown("**A Runnable accepting a query string and returning a list of Document objects (str -> List[Document]).**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: @tool Decorator"):
        st.markdown("**Inspects function type hints and Google-style docstrings to generate OpenAPI JSON schemas.**")
    with st.expander("🎴 Flashcard 4: LangGraph Distinction"):
        st.markdown("**Extends LangChain with cyclic graphs, state persistence, and human-in-the-loop interruption gates.**")

st.divider()

# --- SECTION 18: SELF-ASSESSMENT CHECKLIST ---
st.header("18. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can differentiate between BaseLLM and BaseChatModel.", value=True)
chk2 = st.checkbox("I understand the lifecycle of the 4 core BaseMessage primitives.", value=True)
chk3 = st.checkbox("I know how the @tool decorator generates JSON schemas from docstrings.", value=True)
chk4 = st.checkbox("I can articulate the architectural difference between a Retriever and a Vector Store.", value=True)
chk5 = st.checkbox("I understand when to transition from linear LCEL to cyclic LangGraph state machines.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 1.2 LangChain Architecture. Ready for 1.3 Installing LangChain!")
