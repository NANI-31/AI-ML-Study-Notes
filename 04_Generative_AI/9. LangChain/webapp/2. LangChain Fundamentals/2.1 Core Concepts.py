import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="2.1 Core Concepts - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("2.1 LangChain Core Concepts: The Ten Architectural Pillars")
st.caption("Chapter 2: LangChain Fundamentals | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Modern Generative AI engineering requires decomposing complex application workflows into composable, 
reusable, and observable primitives. Rather than writing brittle glue code that directly ties business logic 
to specific model API endpoints, LangChain (v0.3+) establishes a unified mental model built upon **ten foundational concepts**.

Understanding these primitives and how they compose through the **LangChain Expression Language (LCEL)** 
is essential for building enterprise cognitive systems. In this comprehensive lesson, you will master:

- **The Model Layer**: Chat models versus legacy text completion engines.
- **The Prompt Layer**: Static templates, dynamic parameters, and zero-shot/few-shot framing.
- **The Message Layer**: Type-safe conversational turns (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`).
- **The Tool & Agent Layer**: Deterministic tool contracts and autonomous ReAct reasoning loops.
- **The Knowledge Layer**: Documents, embeddings, vector stores, and retrievers forming the RAG backbone.
- **The Output Contract**: Guaranteed type-safe schema enforcement via `.with_structured_output()`.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockEmbeddingEngine:
    """Simulates high-dimensional semantic vector projections in module scope."""
    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def embed_query(self, text: str) -> List[float]:
        np.random.seed(abs(hash(text)) % (2**32))
        raw_vec = np.random.normal(loc=0.0, scale=1.0, size=self.dimension)
        norm = np.linalg.norm(raw_vec)
        normalized = (raw_vec / norm).round(4).tolist()
        return normalized

class MockVectorRecord:
    """Represents a document chunk with vector embedding and metadata."""
    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any], vector: List[float]):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata
        self.vector = vector

class MockConceptPipeline:
    """Simulates end-to-end execution of the 10 LangChain core concepts."""
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.embedder = MockEmbeddingEngine(dimension=6)
        
        # Knowledge Base Seed
        raw_docs = [
            ("doc-1", "Enterprise RAG requires hybrid search combining sparse BM25 and dense vectors.", {"source": "rag_handbook.pdf", "tier": "advanced"}),
            ("doc-2", "LangGraph uses StateGraph to coordinate cyclical multi-agent workflows with memory.", {"source": "langgraph_specs.md", "tier": "expert"}),
            ("doc-3", "Pydantic v2 powers LangChain structured output schema validation with Rust speed.", {"source": "pydantic_guide.txt", "tier": "core"}),
            ("doc-4", "ChatModel exposes invoke, ainvoke, batch, and stream through unified Runnables.", {"source": "runnables_primer.md", "tier": "core"})
        ]
        self.vector_store: List[MockVectorRecord] = [
            MockVectorRecord(did, content, meta, self.embedder.embed_query(content))
            for did, content, meta in raw_docs
        ]

    def retrieve(self, query: str, top_k: int = 2) -> List[MockVectorRecord]:
        query_vec = np.array(self.embedder.embed_query(query))
        scored = []
        for record in self.vector_store:
            rec_vec = np.array(record.vector)
            sim = float(np.dot(query_vec, rec_vec))
            scored.append((sim, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def execute_pipeline(self, user_query: str, system_prompt: str, use_tools: bool) -> Dict[str, Any]:
        # 1. Embed query
        q_vec = self.embedder.embed_query(user_query)
        
        # 2. Retrieve relevant documents
        retrieved_docs = self.retrieve(user_query, top_k=2)
        context_text = "\n".join([f"[{d.doc_id}] {d.content}" for d in retrieved_docs])
        
        # 3. Format message sequence
        messages = [
            {"type": "SystemMessage", "content": system_prompt},
            {"type": "HumanMessage", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
        ]
        
        # 4. Tool execution simulation if enabled
        tool_executed = None
        if use_tools and ("calculate" in user_query.lower() or "speed" in user_query.lower() or "search" in user_query.lower()):
            tool_executed = {
                "name": "enterprise_kb_search",
                "arguments": {"query": user_query, "scope": "production_indexes"},
                "status": "COMPLETED",
                "output": "Found 4 indexed clusters across active availability zones."
            }
            messages.append({"type": "AIMessage", "content": "", "tool_calls": [tool_executed]})
            messages.append({"type": "ToolMessage", "content": str(tool_executed["output"]), "name": "enterprise_kb_search"})
        
        # 5. Synthesize Model output
        answer_text = (
            f"Based on retrieved context from {retrieved_docs[0].metadata['source']}, "
            f"LangChain integrates {self.model_name} with declarative LCEL Runnables. "
            f"Retrieved primary match score: {round(float(np.dot(np.array(q_vec), np.array(retrieved_docs[0].vector))), 3)}."
        )
        
        structured_output = {
            "query": user_query,
            "synthesized_answer": answer_text,
            "grounding_sources": [d.metadata["source"] for d in retrieved_docs],
            "confidence_score": round(max(0.70, 0.98 - (self.temperature * 0.15)), 2),
            "schema_valid": True
        }
        
        return {
            "query_vector": q_vec,
            "retrieved_records": retrieved_docs,
            "message_chain": messages,
            "tool_call": tool_executed,
            "structured_result": structured_output
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: LangChain Core Conceptual Topology & Data Flow", divider="rainbow")

st.markdown("""
The diagram below illustrates how all ten core concepts interconnect within a modern production application. 
Data enters via **Prompts** and **Messages**, is processed by **Models** and governed by **Structured Output**, 
while grounding is provided by the **Retrieval** chain and action is driven by **Tools** and **Agents**.
""")

dot_blueprint = """
digraph CoreConceptsTopology {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;
    
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_input {
        label="1. Ingestion and Prompt Engineering Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_user [label="User / Upstream Service\n(Raw Query Text)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_prompt [label="Prompt (ChatPromptTemplate)\nDynamic Variables and System Directives", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_messages [label="Messages (Sequence)\nSystemMessage + HumanMessage", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        
        node_user -> node_prompt [label="raw query", color="#38BDF8"];
        node_prompt -> node_messages [label="formats into", color="#38BDF8"];
    }

    subgraph cluster_knowledge {
        label="2. Grounding and Knowledge Retrieval Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_docs [label="Documents\n(page_content + metadata)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_embed [label="Embeddings Engine\n(Dense Vector Projections)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_vstore [label="Vector Store (Index)\n(HNSW / ANN Similarity Search)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_retriever [label="Retriever Interface\n(.invoke(query) -> List[Document])", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_docs -> node_embed [label="vectorized by"];
        node_embed -> node_vstore [label="indexed in"];
        node_vstore -> node_retriever [label="searched via"];
    }

    subgraph cluster_cognition {
        label="3. Cognitive Engine and Agentic Execution Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_model [label="Chat Model (BaseChatModel)\n(OpenAI, Anthropic, Gemini, Ollama)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_tools [label="Tools (@tool contracts)\n(API Call, SQL, Python REPL)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_agent [label="Agent Loop (ReAct)\n(Observation -> Thought -> Action)", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_model -> node_tools [label="bind_tools()", color="#38BDF8"];
        node_tools -> node_agent [label="tool results"];
        node_agent -> node_model [label="state feedback"];
    }

    subgraph cluster_output {
        label="4. Output Serialization and Validation Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_struct [label="Structured Output (Pydantic v2)\n(.with_structured_output(Schema))", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_app [label="Downstream Production App\n(Type-Safe DTO / API Response)", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_struct -> node_app [label="validated JSON DTO", color="#34D399"];
    }

    // Inter-cluster connections
    node_retriever -> node_prompt [label="context injection", color="#F59E0B", style="dashed"];
    node_messages -> node_model [label="passed to", color="#38BDF8"];
    node_model -> node_struct [label="raw generation", color="#38BDF8"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: LangChain is not a monolithic application framework; it is an interconnected 
> matrix of modular abstractions. Notice how **Documents** become searchable via **Embeddings** and **Vector Stores**, 
> which feed **Retrievers**, which inject context into **Prompts**, which format **Messages** for **Chat Models**, 
> which bind to **Tools** and enforce type-safety through **Structured Output**.
""")

# --- SECTION 2: MODELS ---
st.header("2. Models: Foundation Chat Engines & Unified Model Abstraction", divider="rainbow")

st.markdown("""
In early Generative AI libraries, the primary interface was a legacy `LLM` (taking a raw text string and returning a raw text string). 
Modern systems standardize entirely on **Chat Models** (`BaseChatModel`).

### Legacy LLMs vs Modern Chat Models

| Architectural Dimension | Legacy LLMs (`langchain_community.llms`) | Modern Chat Models (`langchain_core.language_models.chat_models`) |
| :--- | :--- | :--- |
| **Input Signature** | Raw text string (`str`) | Sequence of typed messages (`List[BaseMessage]`) |
| **Output Signature** | Raw text string (`str`) | Rich message object (`AIMessage`) containing metadata |
| **Tool Calling Support** | Brittle regex parsing from raw text | Native JSON schema binding via API protocol (`bind_tools`) |
| **Multimodal Capabilities** | None (pure text only) | Native support for images, audio, PDF payloads |
| **LCEL Compatibility** | Partial / Deprecated | 100% native first-class `Runnable` |
""")

st.code('''from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Modern Chat Model instantiation
gpt4o = ChatOpenAI(model="gpt-4o", temperature=0.2)
claude = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.2)

# Unified invocation signature
response = gpt4o.invoke([HumanMessage(content="Explain LCEL in one sentence.")])
print(type(response))  # <class 'langchain_core.messages.ai.AIMessage'>
print(response.content)
''', language="python")

# --- SECTION 3: PROMPTS ---
st.header("3. Prompts: Dynamic Templates & Context Framing", divider="rainbow")

st.markdown("""
A prompt in production is rarely a static string. It is a parametrized template that injects runtime parameters, 
formats system instructions, and enforces role boundaries. LangChain provides `ChatPromptTemplate` to construct 
reusable prompt blueprints.
""")

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.markdown("""
    #### Key Characteristics of `ChatPromptTemplate`
    - **Tuple Syntax**: Clean definition using `("role", "template_text")` tuples.
    - **Variable Interpolation**: Named braces `{variable_name}` validated at runtime.
    - **Partial Formatting**: Pre-binding static parameters while leaving dynamic parameters open.
    - **Messages Placeholder**: Dynamically injecting conversation histories (`MessagesPlaceholder`).
    """)
with col_p2:
    st.code('''from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_blueprint = ChatPromptTemplate.from_messages([
    ("system", "You are an enterprise architect specializing in {domain}."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_query}")
])

# Formatting prompt into PromptValue
prompt_val = prompt_blueprint.invoke({
    "domain": "Distributed Database Systems",
    "chat_history": [],
    "user_query": "Compare Raft vs Paxos consensus latency."
})
''', language="python")

# --- SECTION 4: MESSAGES ---
st.header("4. Messages: Structured Dialogue Primitives", divider="rainbow")

st.markdown("""
Messages are the universal communication currency in modern GenAI pipelines. Every conversational turn 
is encapsulated as an instance of a subclass of `BaseMessage`:

1. **`SystemMessage`**: Sets the model persona, tone, guardrails, and behavioral boundaries.
2. **`HumanMessage`**: Represents raw user input, queries, or client-side multimodal payloads.
3. **`AIMessage`**: The response generated by the model, containing generated text, tool call requests, and usage metadata.
4. **`ToolMessage`**: The execution result returned from an external function/tool back to the model.
""")

st.code('''from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

conversation_sequence = [
    SystemMessage(content="You are a financial analyst assistant."),
    HumanMessage(content="What is the current stock price of MSFT?"),
    AIMessage(
        content="",
        tool_calls=[{"name": "fetch_ticker", "args": {"ticker": "MSFT"}, "id": "call_123"}]
    ),
    ToolMessage(
        content='{"ticker": "MSFT", "price": 428.50, "currency": "USD"}',
        tool_call_id="call_123"
    )
]
''', language="python")

# --- SECTION 5: TOOLS ---
st.header("5. Tools: External Capability Interfaces & Functional Contracts", divider="rainbow")

st.markdown("""
Foundation models cannot write to disk, run shell commands, or query production databases on their own. 
**Tools** provide clean functional boundaries allowing models to invoke external computational resources.

In LangChain, any Python function can be decorated with `@tool`. LangChain automatically extracts the 
function signature, docstring, and type annotations into an OpenAI-compatible JSON Schema.
""")

st.code('''from langchain_core.tools import tool

@tool
def calculate_vat(subtotal: float, country_code: str) -> float:
    """Calculate the final Value Added Tax based on national tax policies.
    
    Args:
        subtotal: The base monetary amount before tax.
        country_code: Two-letter ISO country code (e.g., 'DE', 'FR', 'UK').
    """
    tax_rates = {"DE": 0.19, "FR": 0.20, "UK": 0.20}
    rate = tax_rates.get(country_code.upper(), 0.15)
    return round(subtotal * rate, 2)

# Inspecting auto-generated JSON schema contract
print(calculate_vat.name)         # 'calculate_vat'
print(calculate_vat.description)  # 'Calculate the final Value Added Tax...'
print(calculate_vat.args)         # {'subtotal': {'title': 'Subtotal', 'type': 'number'}, ...}
''', language="python")

# --- SECTION 6: AGENTS ---
st.header("6. Agents: Autonomous Reasoning Loops & Decision Trees", divider="rainbow")

st.markdown("""
Unlike static sequential chains where execution flows along a predetermined path, an **Agent** uses 
a Chat Model as a reasoning engine to dynamically decide which actions to take, observe the outcome, 
and iterate until the goal is achieved.
""")

dot_react = """
digraph ReActLoop {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_start [label="User Task Input", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_thought [label="Reasoning / Thought\n(Analyze context & missing info)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_decision [label="Action Needed?", shape=diamond, fillcolor="#0F172A", fontcolor="#38BDF8"];
    node_tool [label="Execute Tool\n(API, Database, Calculator)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_obs [label="Observation\n(Tool Output / Environment Feedback)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_finish [label="Final Response Synthesis", fillcolor="#1E293B", fontcolor="#34D399"];

    node_start -> node_thought;
    node_thought -> node_decision;
    node_decision -> node_tool [label="Yes (tool_calls)", color="#F59E0B"];
    node_tool -> node_obs;
    node_obs -> node_thought [label="Feed observation back\ninto context loop", color="#38BDF8"];
    node_decision -> node_finish [label="No (direct answer)", color="#34D399"];
}
"""

with st.expander("🔍 Contextual Diagram: The ReAct (Reason + Act) Loop Mechanism", expanded=True):
    st.graphviz_chart(dot_react, use_container_width=True)

st.markdown("""
> **Modern Agentic Evolution**: While LangChain historically featured `AgentExecutor`, modern production systems 
> build stateful cyclic agent architectures using **LangGraph**, where state transitions and human-in-the-loop checkpoints 
> are strictly deterministic.
""")

# --- SECTION 7: RETRIEVERS ---
st.header("7. Retrievers: Semantic Querying & Knowledge Fetching Contracts", divider="rainbow")

st.markdown("""
A **Retriever** is a generic interface that accepts an unstructured query string and returns a list of matching `Document` objects:

```text
Retriever Input:  str (e.g., "What are the latency SLA guarantees?")
Retriever Output: List[Document]
```

Retrievers decouple the retrieval algorithm from downstream model chains. A retriever can be backed by:
- A Vector Store performing Approximate Nearest Neighbor (ANN) search.
- An ElasticSearch or BM25 index performing keyword search.
- A Hybrid Search pipeline combining dense vectors and sparse keywords via Reciprocal Rank Fusion (RRF).
- A Multi-Query or Contextual Compression re-ranking pipeline.
""")

st.code('''from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# Creating a retriever directly from a vector store
# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Every Retriever is an LCEL Runnable:
# matching_docs: List[Document] = retriever.invoke("enterprise deployment checklist")
''', language="python")

# --- SECTION 8: DOCUMENTS ---
st.header("8. Documents: Information Containers & Metadata Schema", divider="rainbow")

st.markdown("""
In LangChain, all external knowledge chunks are standardized as `Document` objects. 
A `Document` consists of two distinct fields:
1. **`page_content`** (`str`): The raw textual content of the chunk.
2. **`metadata`** (`dict`): Arbitrary contextual attributes (source file, author, timestamp, chunk index, security classification).
""")

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.markdown("""
    #### Document Schema Best Practices
    - Always include unambiguous source references (`source: "contracts/2026_q1_sla.pdf"`).
    - Store chunk coordinates (`page: 14`, `chunk_id: 104`).
    - Use metadata filters during retrieval to partition multi-tenant data securely.
    """)
with col_d2:
    st.code('''from langchain_core.documents import Document

doc = Document(
    page_content="All production API endpoints must maintain 99.99% uptime under 250ms p95 latency.",
    metadata={
        "source": "infrastructure_sla.md",
        "department": "Platform Engineering",
        "confidentiality": "Internal",
        "updated_at": "2026-03-15"
    }
)

print(doc.page_content)
print(doc.metadata["department"])
''', language="python")

# --- SECTION 9: EMBEDDINGS ---
st.header("9. Embeddings: High-Dimensional Semantic Vector Projections", divider="rainbow")

st.markdown("""
An **Embedding** model maps human language into continuous high-dimensional vector spaces. 
Texts with similar semantic meanings are placed close together in this vector space, measured by metrics 
such as **Cosine Similarity** or **Euclidean Distance**.

LangChain's `Embeddings` base class defines two essential methods:
- `embed_documents(texts: List[str]) -> List[List[float]]`: Optimized for bulk document indexing.
- `embed_query(text: str) -> List[float]`: Optimized for single runtime user search queries.
""")

st.code('''from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Vectorizing a query into high-dimensional space (e.g., 1536 dimensions)
vector = embeddings_model.embed_query("How does vector indexing accelerate similarity search?")
print(f"Vector length: {len(vector)}")  # 1536
print(f"Sample dimensions: {vector[:3]}")
''', language="python")

# --- SECTION 10: VECTOR STORES ---
st.header("10. Vector Stores: High-Performance Approximate Nearest Neighbor Search", divider="rainbow")

st.markdown("""
Storing millions of embeddings and scanning them linearly (O(N * d)) is computationally prohibitive in production. 
**Vector Stores** index high-dimensional vectors using specialized data structures (such as **HNSW** — Hierarchical Navigable Small World, 
or **IVF** — Inverted File Index) to perform Approximate Nearest Neighbor (ANN) search in sub-linear logarithmic time (O(log N)).
""")

dot_vstore = """
digraph VectorStoreWorkflow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_index {
        label="Indexing Phase (Offline / Ingestion Pipeline)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        
        node_raw [label="Source Documents (PDF, MD, SQL)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_split [label="Text Splitter (RecursiveCharacterTextSplitter)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_embed_bulk [label="Embeddings Engine (Batch embed_documents)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_storage [label="Vector Store Index (Chroma, Pinecone, Qdrant, PGVector)", fillcolor="#1E293B", fontcolor="#34D399"];
        
        node_raw -> node_split -> node_embed_bulk -> node_storage;
    }

    subgraph cluster_query {
        label="Query Phase (Online / Real-time Request)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        
        node_user_q [label="User Query String", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_embed_q [label="Embeddings Engine (embed_query)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_ann [label="ANN Search Algorithm (Cosine / Dot Product)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_topk [label="Top-K Retrieved Document Chunks", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        
        node_user_q -> node_embed_q -> node_ann;
        node_storage -> node_ann [label="similarity search", style="dashed", color="#F59E0B"];
        node_ann -> node_topk;
    }
}
"""

with st.expander("🔍 Contextual Diagram: Vector Store Indexing and Query Pipeline", expanded=True):
    st.graphviz_chart(dot_vstore, use_container_width=True)

st.code('''from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# In-memory vector store demonstration
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

# Ingesting documents
# vector_store.add_documents([doc1, doc2, doc3])

# Querying via similarity search
# results = vector_store.similarity_search("infrastructure latency SLA", k=2)
''', language="python")

# --- SECTION 11: STRUCTURED OUTPUT ---
st.header("11. Structured Output: Schema Enforcement & Pydantic Validation", divider="rainbow")

st.markdown("""
In enterprise workflows, LLM responses cannot be freeform text; they must feed relational databases, microservices, 
and message queues. Relying on raw string parsing or instructions like "respond only in valid JSON" fails at production scale.

Modern LangChain uses `.with_structured_output()` to bind **Pydantic v2 schemas** directly into the provider's API.
""")

st.code('''from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI

class EnterpriseIncidentAudit(BaseModel):
    """Pydantic schema enforcing structured model extraction."""
    incident_id: str = Field(description="Unique system incident ticket ID")
    severity: str = Field(description="Severity grade: LOW, MEDIUM, HIGH, CRITICAL")
    affected_services: List[str] = Field(description="Names of impacted microservices")
    root_cause_summary: str = Field(description="Concise 1-sentence technical root cause")
    estimated_recovery_minutes: int = Field(description="Estimated resolution time in minutes")

# Binding schema to ChatModel
model = ChatOpenAI(model="gpt-4o", temperature=0.0)
structured_model = model.with_structured_output(EnterpriseIncidentAudit)

# Result is an instantiated Pydantic object with guaranteed types
# audit_data: EnterpriseIncidentAudit = structured_model.invoke("Incident INC-901: Payment gateway timed out...")
# print(audit_data.severity)  # Guaranteed string matching schema
''', language="python")

# --- SECTION 12: PRODUCTION IMPLEMENTATION ---
st.header("12. Production Implementation: Unified Core Concept Pipeline", divider="rainbow")

st.markdown("""
Below is an end-to-end, production-grade Python implementation showcasing how all ten core concepts 
integrate into a clean, testable, and maintainable module.
""")

st.code('''import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. Structured Output Schema Definition
class RetrievalSynthesisResponse(BaseModel):
    summary: str = Field(description="Synthesized technical answer")
    key_findings: List[str] = Field(description="Extracted facts directly grounded in context")
    confidence_rating: float = Field(description="Confidence score between 0.0 and 1.0")
    source_attribution: List[str] = Field(description="List of document source identifiers")

# 2. Tool Definition
@tool
def calculate_system_throughput(qps: float, latency_seconds: float) -> Dict[str, float]:
    """Calculate Little's Law concurrency footprint for distributed systems."""
    concurrency = qps * latency_seconds
    return {"calculated_concurrency": round(concurrency, 2), "buffer_headroom": round(concurrency * 1.5, 2)}

# 3. Knowledge Base Assembly (Documents + Embeddings + Vector Store)
raw_knowledge = [
    Document(
        page_content="Aurora PostgreSQL read replicas provide sub-100ms replica lag across multi-AZ regions.",
        metadata={"source": "db_specs.md", "domain": "Database"}
    ),
    Document(
        page_content="Kafka message retention is set to 7 days across production payment ingestion topics.",
        metadata={"source": "streaming_specs.md", "domain": "Messaging"}
    )
]

def build_production_pipeline(api_key: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(raw_knowledge)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 4. Model and Structured Output Binding
    base_model = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
    model_with_tools = base_model.bind_tools([calculate_system_throughput])
    structured_extractor = base_model.with_structured_output(RetrievalSynthesisResponse)

    # 5. Declarative LCEL Composition
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise systems auditor. Answer strictly based on provided context.\\nContext:\\n{context}"),
        ("human", "{question}")
    ])

    # Composite execution function
    def execute_audit(query: str) -> RetrievalSynthesisResponse:
        retrieved_docs = retriever.invoke(query)
        formatted_context = "\\n\\n".join([f"[{d.metadata['source']}] {d.page_content}" for d in retrieved_docs])
        
        # Invoke LCEL chain
        audit_chain = prompt | structured_extractor
        return audit_chain.invoke({"context": formatted_context, "question": query})

    return execute_audit
''', language="python")

# --- SECTION 13: INTERACTIVE WORKBENCH ---
st.header("13. Interactive Workbench: LangChain Concept Pipeline Simulator", divider="rainbow")

st.markdown("""
Interact with this live simulator to observe how the **10 Core Concepts** collaborate in real-time. 
Adjust generation temperature, enable or disable external tools, inspect vector similarities, and view the 
strongly-typed synthesized output.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    model_choice = st.selectbox(
        "Chat Model Engine",
        ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro", "llama-3.1-70b"],
        index=0
    )
    temp_choice = st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    enable_tools = st.toggle("Enable External Tool Calling", value=True)

with col_wb2:
    sys_prompt = st.text_input(
        "System Message Directive",
        value="You are a Principal AI Architect analyzing LangChain orchestration patterns."
    )
    user_query = st.text_area(
        "User Query (Try keywords like 'RAG', 'LangGraph', 'Pydantic', or 'calculate')",
        value="How does LangGraph coordinate cyclic workflows and can we calculate system speed?"
    )

if st.button("🚀 Execute 10-Concept Pipeline", type="primary"):
    pipeline_sim = MockConceptPipeline(model_name=model_choice, temperature=temp_choice)
    execution_data = pipeline_sim.execute_pipeline(user_query, sys_prompt, enable_tools)
    
    st.subheader("📊 Pipeline Execution Telemetry")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Selected Model", model_choice)
    t2.metric("Vector Dimensions", len(execution_data["query_vector"]))
    t3.metric("Retrieved Documents", len(execution_data["retrieved_records"]))
    t4.metric("Tool Triggered", "Yes" if execution_data["tool_call"] else "No")

    # Step-by-step breakdown
    tab_docs, tab_msgs, tab_tools, tab_output = st.tabs(["📚 1. Retrieved Documents", "💬 2. Message Sequence", "🛠️ 3. Tool Execution", "📋 4. Structured Output"])
    
    with tab_docs:
        st.markdown("**Vector Store Match Results:**")
        for rec in execution_data["retrieved_records"]:
            st.info(f"**ID**: `{rec.doc_id}` | **Source**: `{rec.metadata['source']}` | **Tier**: `{rec.metadata['tier']}`\n\nContent: *\"{rec.content}\"*")
        
        st.caption(f"Query Vector Representation (First 6 dimensions): `{execution_data['query_vector']}`")

    with tab_docs:
        pass  # Tab handled

    with tab_msgs:
        st.markdown("**Conversation Messages Ingested by Model:**")
        for msg in execution_data["message_chain"]:
            if msg["type"] == "SystemMessage":
                st.code(f"SystemMessage: {msg['content']}", language="markdown")
            elif msg["type"] == "HumanMessage":
                st.code(f"HumanMessage: {msg['content']}", language="markdown")
            elif msg["type"] == "AIMessage":
                st.code(f"AIMessage: Tool Calls requested: {msg.get('tool_calls', [])}", language="json")
            elif msg["type"] == "ToolMessage":
                st.code(f"ToolMessage ({msg.get('name')}): {msg['content']}", language="json")

    with tab_tools:
        if execution_data["tool_call"]:
            st.success("Tool Invocation Successful:")
            st.json(execution_data["tool_call"])
        else:
            st.write("No external tool calls required for this query.")

    with tab_output:
        st.markdown("**Validated Pydantic DTO Contract:**")
        st.json(execution_data["structured_result"])

# --- SECTION 14: TECHNICAL INTERVIEW PREPARATION ---
st.header("14. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the fundamental difference between an LLM and a ChatModel in LangChain?",
     "An LLM takes a plain string and returns a plain string. A ChatModel accepts a list of BaseMessage objects (SystemMessage, HumanMessage, AIMessage, ToolMessage) and returns an AIMessage. ChatModels support native tool calling, multimodal inputs, and rich response metadata. In LangChain v0.3+, legacy LLMs are deprecated in favor of ChatModels."),
    ("How does LangChain's .with_structured_output() guarantee schema conformity?",
     "It binds a Pydantic model or JSON Schema to the ChatModel's tool-calling or response-format endpoint (e.g., OpenAI's function calling or JSON mode). The model outputs JSON matching the schema, which LangChain automatically deserializes and validates against Pydantic validators. If validation fails, an exception is raised or a retry loop is initiated."),
    ("What is the difference between a Vector Store and a Retriever in LangChain?",
     "A Vector Store is a storage backend that holds embeddings and document chunks (e.g., Chroma, Pinecone) and provides low-level search methods. A Retriever is a higher-level, generic LCEL Runnable that accepts a string query and returns a list of Documents. All Vector Stores can become Retrievers via .as_retriever(), but Retrievers can also implement BM25, hybrid search, or multi-stage re-ranking without direct Vector Store coupling."),
    ("Why should you avoid using Python's eval() in custom tools created with @tool?",
     "Using eval() or exec() inside tools creates critical Remote Code Execution (RCE) vulnerabilities. If an LLM is manipulated via prompt injection, an attacker could force the agent to execute arbitrary system commands. Custom tools must perform strict argument validation and use sandboxed execution environments."),
    ("What is the difference between embed_documents and embed_query in the Embeddings interface?",
     "embed_documents accepts a list of strings and is optimized for batch indexing with bulk API requests. embed_query accepts a single string query and is called at runtime for user searches. Some embedding models (like Cohere or instructor-embeddings) also use different prefix instructions or projection weights for queries vs documents.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using Deprecated LLMChain Instead of LCEL. Copying older tutorials using `from langchain.chains import LLMChain` breaks modern async streaming and lacks Pydantic v2 type safety. Always use `prompt | model`.")
    st.error("**Trap 2 (Breaking Error)**: Mismatched Tool Output Types. Returning raw objects or non-serializable datatypes from `@tool` functions causes serialization crashes when building `ToolMessage`. Always return strings or JSON-serializable dictionaries.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Redundant Embedding Generation. Calling `embed_query` repeatedly inside loops without a local cache or vector store causes unnecessary API costs and introduces high network latency. Always cache or store indexed embeddings.")
    st.warning("**Trap 4 (Latency Warning)**: Giant Top-K in Retrievers. Passing `k=20` to `.as_retriever()` injects excessive context into the prompt, diluting relevant facts (lost-in-the-middle phenomenon) and multiplying token latency and cost. Keep `k` between 3 and 5, using a reranker if needed.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: BaseChatModel Return Type"):
        st.markdown("**AIMessage containing .content (string), .tool_calls (list of dicts), and .response_metadata (dict).**")
    with st.expander("🎴 Flashcard 2: Tool Contract Definition"):
        st.markdown("**The @tool decorator extracts function name, type annotations, and docstring to build an OpenAPI/JSON Schema.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Document Primitives"):
        st.markdown("**A Document contains exactly two attributes: page_content (str) and metadata (dict).**")
    with st.expander("🎴 Flashcard 4: Retriever Contract"):
        st.markdown("**A Runnable conforming to the signature .invoke(query: str) -> List[Document].**")

st.divider()

# --- SECTION 15: SELF-ASSESSMENT CHECKLIST ---
st.header("15. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can name and describe all 10 core concepts of LangChain without looking at notes.", value=True)
chk2 = st.checkbox("I understand the architectural differences between legacy LLMs and modern Chat Models.", value=True)
chk3 = st.checkbox("I know how @tool auto-generates JSON Schema contracts for tool calling.", value=True)
chk4 = st.checkbox("I can explain the difference between a Vector Store and a Retriever.", value=True)
chk5 = st.checkbox("I understand why .with_structured_output() is required for enterprise API reliability.", value=True)
chk6 = st.checkbox("I am ready to answer all 5 technical interview questions confidently.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 2.1 Core Concepts. Ready for 2.2 LangChain Message Types!")
