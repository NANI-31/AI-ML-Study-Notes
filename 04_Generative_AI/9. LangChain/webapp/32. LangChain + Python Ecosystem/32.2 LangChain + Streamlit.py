import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="32.2 LangChain + Streamlit - Interactive Agent Applications",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Streamlit agent telemetry
st.markdown("""
<style>
    .metric-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-rag {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 2px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class StreamlitAgentSessionSimulator:
    """
    Simulates reactive session state, chat history persistence,
    document uploads, in-memory RAG embeddings, and intermediate agent trace rendering.
    """
    def __init__(self):
        self.preset_documents = {
            "quarterly_financial_report.pdf": "Q3 Revenue reached $42.5M, a 18% YoY increase. Cloud infrastructure expenses totaled $12.3M.",
            "api_security_guidelines.md": "All API endpoints must enforce RS256 JWT tokens. Rate limiting is capped at 100 requests per minute per IP.",
            "employee_handbook.docx": "Standard remote work policy permits up to 3 days remote per week. Core collaboration hours are 10 AM to 4 PM EST."
        }

    def process_file_upload(self, filename: str) -> Dict[str, Any]:
        content = self.preset_documents.get(filename, "Standard document payload parsed into 3 chunks.")
        words = len(content.split())
        return {
            "filename": filename,
            "status": "Indexed into Vector Store",
            "chunks_count": 3,
            "tokens": words * 2,
            "preview": content[:120] + "..."
        }

    def simulate_chat_response(self, prompt: str, doc_name: Optional[str]) -> Tuple[str, List[Dict[str, str]]]:
        trajectory = [
            {"step": "Query Analysis", "details": f"Analyzing semantic intent for '{prompt}'"},
            {"step": "Retriever Query", "details": f"Filtering vector store with active document: {doc_name if doc_name else 'Global KB'}"},
            {"step": "Synthesizer", "details": "Grounding response using retrieved top-2 chunks."}
        ]
        
        if doc_name and doc_name in self.preset_documents:
            doc_context = self.preset_documents[doc_name]
            reply = f"Based on '{doc_name}': {doc_context} (Retrieved with confidence score 0.94)"
        else:
            reply = f"I evaluated your inquiry regarding '{prompt}'. In the general knowledge store, enterprise guidelines prioritize strict compliance, automated checks, and audit logging."
            
        return reply, trajectory


# --- LESSON HEADER ---
st.title("32.2 LangChain + Streamlit: Interactive AI Applications and Dashboards")
st.markdown("""
Streamlit provides the premier rapid-prototyping and operational dashboard framework for Python-based generative AI systems.
Integrating LangChain and LangGraph with Streamlit requires mastering its unique reactive execution model: every user interaction
re-executes the script from top to bottom. This guide details how to leverage `st.session_state` for message history,
`st.cache_resource` for expensive vector store and model singletons, `st.chat_message` for modern conversational interfaces,
and real-time agent observability dashboards.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Reactive Streamlit UI Architecture for LangChain", divider="rainbow")
st.markdown("""
Unlike traditional event-driven frontend frameworks (React, Vue), Streamlit operates on a top-to-bottom re-run loop.
To prevent reloading models or clearing conversation history on every keystroke, LangChain state is persisted in
`st.session_state`, while long-lived objects (embedding models, vector stores, compiled graphs) are cached in `st.cache_resource`.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Streamlit_LangChain_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_browser {
            label = "Browser / User Interaction";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            UserAction [label="User Input / File Drop\\n(st.chat_input / st.file_uploader)", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_streamlit_runtime {
            label = "Streamlit Reactive Engine (Top-to-Bottom Execution)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            SessionState [label="st.session_state Storage\\n(Messages, Active Thread ID, UI Flags)", fillcolor="#1E293B", color="#60A5FA"];
            CacheResource [label="@st.cache_resource Singletons\\n(ChatOpenAI, Embeddings, Chroma VectorDB)", fillcolor="#1E293B", color="#60A5FA"];
            UIRenderer [label="Chat Stream Renderer\\n(st.chat_message, st.write_stream)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_langchain_core {
            label = "LangChain / LangGraph Engine";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            GraphRunner [label="Compiled StateGraph / LCEL Chain\\n(astream / stream generator)", fillcolor="#0369A1", color="#38BDF8"];
            DocPipeline [label="Ephemeral Ingestion Pipeline\\n(PyPDFLoader -> RecursiveSplitter -> VectorDB)", fillcolor="#0369A1", color="#38BDF8"];
            ToolsRuntime [label="Agent Tools & Retrivers\\n(SQL, Search, RAG Retriver)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_dashboard_telemetry {
            label = "Observability & State Inspector Dashboard";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            StateInspector [label="Real-time State & Trajectory Inspector\\n(Expander, JSON state, Token counter)", fillcolor="#065F46", color="#10B981"];
        }

        UserAction -> SessionState [label="Triggers Script Re-run"];
        SessionState -> GraphRunner [label="Load History & Messages"];
        CacheResource -> GraphRunner [label="Inject Shared LLM & Vector Store"];
        UserAction -> DocPipeline [label="Uploaded Bytes"];
        DocPipeline -> CacheResource [label="Update In-Memory Index"];
        GraphRunner -> ToolsRuntime [label="Invoke Intermediate Tools"];
        ToolsRuntime -> GraphRunner [label="Tool Results"];
        GraphRunner -> UIRenderer [label="Yield Streaming Chunks"];
        GraphRunner -> StateInspector [label="Log Intermediate Steps"];
        UIRenderer -> SessionState [label="Persist Assistant Message"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: CHAT INTERFACES ---
st.header("2. Modern Conversational Interfaces: st.chat_message, st.chat_input, and State", divider="rainbow")
st.markdown("""
Streamlit provides native primitives `st.chat_message` and `st.chat_input` designed specifically for LLM conversational workflows.
Because Streamlit wipes local variables upon every re-run, chat histories must be stored in `st.session_state.messages`.
When the user submits a prompt, the message is immediately appended to state, rendered to the screen, and passed to LangChain.
""")

col_code1, col_code2 = st.columns(2)
with col_code1:
    st.markdown("### Core Chat Lifecycle Pattern")
    st.markdown("""
    1. **Initialize State**: Check if `"messages"` exists in `st.session_state`; if not, set to an empty list.
    2. **Render Past History**: Loop through `st.session_state.messages` and draw each using `st.chat_message(role)`.
    3. **Capture User Input**: Use `prompt = st.chat_input(...)`. If submitted, append to state and render.
    4. **Stream Generation**: Use `st.write_stream()` to consume LangChain's `.stream()` generator.
    5. **Save Completion**: Append the full assistant response to `st.session_state.messages`.
    """)
with col_code2:
    st.code("""
import streamlit as st
from langchain_openai import ChatChatOpenAI

llm = ChatOpenAI(model="gpt-4o", streaming=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Render historical turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Capture and process user input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Stream model tokens directly to UI
    with st.chat_message("assistant"):
        stream = llm.stream(prompt)
        response_text = st.write_stream(chunk.content for chunk in stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    """, language="python")

# --- SECTION 3: FILE UPLOADS ---
st.header("3. Dynamic File Uploads and Ephemeral Document Ingestion: st.file_uploader", divider="rainbow")
st.markdown("""
Streamlit's `st.file_uploader` allows end users to drag and drop PDFs, DOCX, CSV, or text files directly into the UI.
Uploaded files are stored as in-memory byte buffers (`io.BytesIO`). To pass these files into LangChain loaders,
the bytes can either be parsed directly via memory-aware loaders or written to a temporary file path before chunking.
""")

st.code("""
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

uploaded_file = st.file_uploader("Upload Policy PDF for Ingestion", type=["pdf"])

if uploaded_file is not None:
    # Ephemeral file handling
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Ingest with LangChain
    loader = PyPDFLoader(tmp_path)
    raw_docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunked_docs = splitter.split_documents(raw_docs)
    
    st.success(f"Successfully processed {uploaded_file.name}: {len(chunked_docs)} semantic chunks created.")
""", language="python")

# --- SECTION 4: RAG APPLICATIONS ---
st.header("4. Cached RAG Workflows: st.cache_resource, In-Memory Retrieval, and Vector Indices", divider="rainbow")
st.markdown("""
Creating embedding models and vector databases is computationally heavy. If initialized inside the main script body,
the vector database and model weights would reload on every single user keystroke or UI interaction.
Decorating the vector store factory with `@st.cache_resource` ensures the index is constructed once and shared across all user re-runs.
""")

st.code("""
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

@st.cache_resource(show_spinner="Embedding and indexing knowledge base...")
def build_cached_vectorstore(documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Vector store persists in memory across all Streamlit script re-runs
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Usage in app
if "chunked_docs" in st.session_state:
    vstore = build_cached_vectorstore(st.session_state.chunked_docs)
    retriever = vstore.as_retriever(search_kwargs={"k": 4})
""", language="python")

# --- SECTION 5: AGENT DASHBOARDS ---
st.header("5. Real-Time Agent Observability Dashboards: State Inspection, Trajectories, and Tools", divider="rainbow")
st.markdown("""
Streamlit enables building powerful internal monitoring dashboards for multi-step agents.
By listening to intermediate agent steps or streaming callbacks (`StreamlitCallbackHandler`),
developers and enterprise operators can inspect tool inputs, query executions, and internal reasoning scratchpads.
""")

col_db1, col_db2 = st.columns(2)
with col_db1:
    st.markdown("### Telemetry Panels to Include")
    st.markdown("""
    - **Step-by-step Trajectory**: Display each node or tool invocation with status badges.
    - **State JSON Inspector**: Use `st.json(state)` in an expander to reveal full checkpoint contents.
    - **Token & Cost Metrics**: Track cumulative tokens using `st.metric()` in the sidebar.
    - **Human-in-the-Loop Overrides**: Add button triggers (`st.button('Approve Tool Call')`) to interrupt and resume graph steps.
    """)
with col_db2:
    st.code("""
from langchain_community.callbacks import StreamlitCallbackHandler
import streamlit as st

st_callback = StreamlitCallbackHandler(
    st.container(),
    expand_new_thoughts=True,
    collapse_completed_thoughts=True
)

# Pass callback to agent execution
result = agent_executor.invoke(
    {"input": prompt},
    {"callbacks": [st_callback]}
)
    """, language="python")

# --- SECTION 6: PRODUCTION IMPLEMENTATION ---
st.header("6. Production Implementation: Full-Featured Enterprise LangChain Streamlit App", divider="rainbow")
st.markdown("""
Here is a complete, production-grade template integrating `st.session_state`, cached vector retrieval,
streaming responses, and intermediate tool inspection into a single cohesive script.
""")

st.code("""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Enterprise AI Copilot", layout="wide")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "token_count" not in st.session_state:
    st.session_state.token_count = 0

# Sidebar Telemetry
with st.sidebar:
    st.header("📊 Session Telemetry")
    st.metric("Total Turns", len(st.session_state.chat_history))
    st.metric("Estimated Tokens", st.session_state.token_count)
    if st.button("Clear Conversation History", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.token_count = 0
        st.rerun()

# Main Chat View
st.subheader("💬 Operational Intelligence Assistant")

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_input := st.chat_input("Enter your command or question..."):
    # Append & display user prompt
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process and Stream Assistant Turn
    with st.chat_message("assistant"):
        with st.status("Executing Agent Plan...", expanded=False) as status_box:
            st.write("Querying knowledge base...")
            st.write("Synthesizing context...")
            status_box.update(label="Complete", state="complete")
            
        simulated_response = f"Processed request: '{user_input}'. Analysis complete with 0 alerts."
        st.markdown(simulated_response)
        
    st.session_state.chat_history.append(AIMessage(content=simulated_response))
    st.session_state.token_count += len(user_input.split()) + len(simulated_response.split())
""", language="python")

# --- SECTION 7: INTERACTIVE WORKBENCH ---
st.header("7. Interactive Workbench: Reactive State & Streaming Chat Simulator", divider="rainbow")
st.markdown("""
Test interactive file ingestion, session state message persistence, and real-time agent trajectory inspection.
""")

session_sim = StreamlitAgentSessionSimulator()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📁 Ingestion & Query Panel")
    selected_doc = st.selectbox(
        "Simulate Ingesting Enterprise Document:",
        list(session_sim.preset_documents.keys())
    )
    ingest_btn = st.button("📥 Parse and Cache Document into VectorDB", type="secondary")
    
    if ingest_btn:
        doc_info = session_sim.process_file_upload(selected_doc)
        st.success(f"Indexed `{doc_info['filename']}`: {doc_info['chunks_count']} Chunks ({doc_info['tokens']} tokens).")
        st.caption(f"Preview: {doc_info['preview']}")

    query_input = st.text_input("User Prompt (st.chat_input):", value="What are the remote work collaboration hours?")
    send_query = st.button("💬 Submit Query to Agent", type="primary")

with col_right:
    st.subheader("🖥️ Simulated UI & Trajectory Output")
    if send_query:
        ans, trajectory = session_sim.simulate_chat_response(query_input, selected_doc)
        
        with st.chat_message("user"):
            st.markdown(query_input)
            
        with st.chat_message("assistant"):
            with st.status("Agent Execution Trajectory (StreamlitCallbackHandler)", expanded=True):
                for step in trajectory:
                    st.write(f"**{step['step']}**: {step['details']}")
            st.markdown(f"**Response**: {ans}")
    else:
        st.info("Submit a query or ingest a document to observe reactive state rendering.")

# --- SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why is `@st.cache_resource` required for LangChain LLM and VectorStore instances?"):
    st.markdown("""
    **Answer**: Streamlit's runtime re-executes the entire Python script from top to bottom on every user interaction. If an LLM or vector
    database is created in the global scope without caching, the model weights, connections, and vector embeddings are re-initialized on every
    click or keystroke, creating massive latency and GPU/memory leaks. `@st.cache_resource` keeps singletons alive across runs.
    """)

with st.expander("Q2: How does `st.write_stream` interact with LangChain's `.stream()` output?"):
    st.markdown("""
    **Answer**: `st.write_stream` accepts any standard Python generator or iterator yielding string chunks or token objects. When fed a generator
    yielding `chunk.content` from LangChain's `.stream()` or `.astream()`, Streamlit renders characters in real-time with a typewriter effect
    and automatically aggregates the individual chunks into a final string return value that can be saved directly to `st.session_state`.
    """)

with st.expander("Q3: What causes `st.session_state` key errors when handling chat input?"):
    st.markdown("""
    **Answer**: Accessing `st.session_state.messages` before verifying its existence. Because Streamlit state is ephemeral per browser session,
    the script must always begin with an initialization guard: `if "messages" not in st.session_state: st.session_state.messages = []`. Failing
    to check initialization raises `KeyError` or `AttributeError` on initial page load.
    """)

with st.expander("Q4: How do you implement file uploads for LangChain in Streamlit without filling disk space?"):
    st.markdown("""
    **Answer**: Use Python's `tempfile.NamedTemporaryFile` with context managers to create ephemeral files that are discarded after loading,
    or read the file bytes in memory via `io.BytesIO` and parse directly with byte-aware parsers (e.g., `PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))`).
    Never write uploaded files to static paths without automatic cleanup logic.
    """)

with st.expander("Q5: What is the purpose of `StreamlitCallbackHandler` in LangChain?"):
    st.markdown("""
    **Answer**: `StreamlitCallbackHandler` is a specialized LangChain callback that intercepts agent tool calls, thought processes, and chain runs,
    automatically rendering collapsible accordion containers (`st.status`) inside the Streamlit UI. This provides transparent observability into
    the agent's reasoning trajectory without requiring custom manual logging code.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Global Variables Instead of Session State)**: Storing user messages in a global Python list (`messages = []`). In Streamlit, global module variables are shared across ALL concurrent browser tabs, causing users to see each other's chat messages.")
    st.warning("**Trap 2 (Re-running Vector Embeddings on Every Keystroke)**: Splitting documents and computing vector embeddings directly inside the main script flow without `@st.cache_resource`, locking the server on every interaction.")
with col_trap2:
    st.error("**Trap 3 (Blocking the Event Loop with Long Agent Loops)**: Executing a 2-minute agent loop synchronously without streaming or status indicators. The browser displays a frozen spinner, leading users to believe the application crashed.")
    st.warning("**Trap 4 (Unbounded Session History Memory Growth)**: Appending thousands of turns and high-resolution uploaded images into `st.session_state` without truncation, triggering browser tab crashes and high server RAM usage.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What decorator ensures vector stores are loaded only once in Streamlit?")
        st.caption("Flip for answer:")
        st.info("**Back**: `@st.cache_resource` (used for unpicklable, long-lived objects like models and DB pools).")
    with st.container(border=True):
        st.markdown("**Front**: How does `st.chat_input` trigger a rerun?")
        st.caption("Flip for answer:")
        st.info("**Back**: When a user submits text, `st.chat_input` returns the string and triggers an automatic script re-run.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How do you render streaming tokens with a typing effect in Streamlit?")
        st.caption("Flip for answer:")
        st.info("**Back**: Pass the LangChain stream generator to `st.write_stream(chunk.content for chunk in stream)`.")
    with st.container(border=True):
        st.markdown("**Front**: What callback visualizes LangChain agent thoughts in Streamlit?")
        st.caption("Flip for answer:")
        st.info("**Back**: `StreamlitCallbackHandler` from `langchain_community.callbacks`.")

# --- SECTION 9: SELF-ASSESSMENT CHECKLIST ---
st.header("9. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the Streamlit reactive re-run model and why session state is required for chat history.", value=True)
c2 = st.checkbox("I can implement conversational streaming using st.chat_message and st.write_stream.", value=True)
c3 = st.checkbox("I know how to handle uploaded document byte streams using st.file_uploader and tempfile.", value=True)
c4 = st.checkbox("I know why @st.cache_resource must be applied to vector store and embedding initializations.", value=True)
c5 = st.checkbox("I can monitor agent tool trajectories using StreamlitCallbackHandler and st.status containers.", value=True)
c6 = st.checkbox("I understand why global Python lists cause cross-user data leakage in multi-user Streamlit deployments.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 32.3: LangChain + React.")
