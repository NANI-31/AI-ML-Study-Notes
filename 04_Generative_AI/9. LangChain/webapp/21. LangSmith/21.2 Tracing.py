import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="21.2 Tracing - LangChain Architecture",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling run tree nodes and tags
st.markdown("""
<style>
    .run-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .tag-badge {
        display: inline-block;
        background-color: #0369A1;
        color: #E0F2FE;
        padding: 2px 8px;
        border-radius: 12px;
        font-family: monospace;
        font-size: 0.75rem;
        margin-right: 4px;
    }
    .meta-badge {
        display: inline-block;
        background-color: #374151;
        color: #F3F4F6;
        padding: 2px 8px;
        border-radius: 12px;
        font-family: monospace;
        font-size: 0.75rem;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockHierarchicalRunTreeExplorer:
    """
    Simulates a rich hierarchical run tree produced by LangSmith tracing.
    Exposes root agent runs, nested chains, tool calls, model traces,
    custom metadata, and queryable tags.
    """
    def __init__(self):
        self.runs = [
            {
                "run_id": "root-run-101",
                "name": "FinancialResearchAgent",
                "run_type": "chain",
                "parent_id": None,
                "latency_ms": 1840,
                "tags": ["prod", "finance", "q1-2026"],
                "metadata": {"user_id": "usr-889", "session_id": "sess-441", "tier": "enterprise"},
                "inputs": {"query": "Analyze TSLA earnings and summarize risk factors."},
                "outputs": {"summary": "TSLA reported Q4 gross margin contraction..."},
                "children": ["child-llm-102", "child-tool-103", "child-llm-104"]
            },
            {
                "run_id": "child-llm-102",
                "name": "ChatOpenAI (Planner)",
                "run_type": "llm",
                "parent_id": "root-run-101",
                "latency_ms": 420,
                "tags": ["model", "planner"],
                "metadata": {"model_name": "gpt-4o", "temperature": 0.0},
                "inputs": {"prompt": "Identify required tools for TSLA earnings analysis."},
                "outputs": {"tool_calls": [{"name": "fetch_sec_filings", "args": {"ticker": "TSLA", "form": "10-K"}}]},
                "children": []
            },
            {
                "run_id": "child-tool-103",
                "name": "fetch_sec_filings",
                "run_type": "tool",
                "parent_id": "root-run-101",
                "latency_ms": 610,
                "tags": ["sec-edgar", "external-api"],
                "metadata": {"endpoint": "https://api.sec.gov/edgar"},
                "inputs": {"ticker": "TSLA", "form": "10-K"},
                "outputs": {"filing_text": "Item 1A Risk Factors: Automotive regulatory credits..."},
                "children": []
            },
            {
                "run_id": "child-llm-104",
                "name": "ChatOpenAI (Synthesizer)",
                "run_type": "llm",
                "parent_id": "root-run-101",
                "latency_ms": 810,
                "tags": ["model", "synthesizer"],
                "metadata": {"model_name": "gpt-4o", "temperature": 0.2},
                "inputs": {"prompt": "Synthesize risk factors based on SEC filing excerpts."},
                "outputs": {"summary": "TSLA reported Q4 gross margin contraction..."},
                "children": []
            }
        ]

    def query_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        if not tag or tag == "ALL":
            return self.runs
        return [r for r in self.runs if tag in r["tags"]]

# Main Application Title & Objective
st.title("🌳 21.2 Tracing in LangSmith")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
Tracing is the core data primitive of LangSmith. Every user interaction decomposes into a **Hierarchical Run Tree** 
linking parent chains, model calls, vector retrievers, and tool executions. This deep dive covers **run identifiers**, 
**nested execution lineages**, **specialized trace types**, **metadata enrichment**, and **queryable tags**.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Hierarchical Run Tree Topology & Execution Lineage", divider="rainbow")
st.markdown("""
The architecture below illustrates the topological structure of a LangSmith trace run tree. 
Execution begins at the root parent run and branches into child runs, with every node recording exact inputs, outputs, 
timing boundaries, token allocations, and parent lineage pointers.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    tree_dot = graphviz.Digraph(comment="Run Tree Hierarchy")
    tree_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    tree_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    tree_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    tree_dot.node("root", "Root Run: Agent Pipeline\n[run_type: chain | ID: run-101]\nTags: ['prod', 'finance']", fillcolor="#0F172A", shape="ellipse")
    
    # Children of Root
    tree_dot.node("step1_llm", "Child 1: ChatOpenAI (Planner)\n[run_type: llm | parent: run-101]\nPrompt & Tool Formulation", fillcolor="#1E3A8A")
    tree_dot.node("step2_tool", "Child 2: fetch_sec_filings\n[run_type: tool | parent: run-101]\nExternal API Fetch", fillcolor="#0E7490")
    tree_dot.node("step3_retriever", "Child 3: VectorRetriever\n[run_type: retriever | parent: run-101]\nTop-K Chunk Similarity", fillcolor="#701A75")
    tree_dot.node("step4_synth", "Child 4: ChatOpenAI (Synthesizer)\n[run_type: llm | parent: run-101]\nFinal Text Synthesis", fillcolor="#047857")

    tree_dot.node("end_trace", "Trace Closed & Flushed\n[Total Duration: 1,840ms | Status: SUCCESS]", fillcolor="#065F46", shape="ellipse")

    tree_dot.edge("root", "step1_llm", label=" Step 1")
    tree_dot.edge("root", "step2_tool", label=" Step 2")
    tree_dot.edge("root", "step3_retriever", label=" Step 3")
    tree_dot.edge("root", "step4_synth", label=" Step 4")
    
    tree_dot.edge("step1_llm", "end_trace")
    tree_dot.edge("step2_tool", "end_trace")
    tree_dot.edge("step3_retriever", "end_trace")
    tree_dot.edge("step4_synth", "end_trace")

    st.graphviz_chart(tree_dot, use_container_width=True)

# --- SECTION 2: TRACE RUNS ---
st.header("2. Trace Runs & Distributed Trace Identifiers", divider="rainbow")
st.markdown("""
A **Run** represents a single unit of work in LangSmith. Every run is assigned a globally unique UUID (`run_id`). 
A **Trace** is a collection of runs sharing the same root identifier, forming a complete transactional DAG.
""")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.subheader("Core Run Properties")
    st.markdown("""
    * `id`: Unique UUID4 identifying the individual run.
    * `name`: Human-readable identifier (e.g. `ChatOpenAI`, `RetrievalQA`).
    * `run_type`: Enum string (`chain`, `llm`, `tool`, `retriever`, `prompt`).
    * `start_time` / `end_time`: Microsecond timestamps used to calculate execution latency.
    * `status`: `SUCCESS` or `ERROR`.
    """)
with col_r2:
    st.subheader("Root vs. Child Runs")
    st.markdown("""
    * **Root Run (`parent_run_id = None`)**: Represents the top-level invocation initiated by the user. 
      The trace ID of the entire tree is the `id` of this root run.
    * **Child Run (`parent_run_id = UUID`)**: Represents an internal step nested inside a parent chain or agent.
    """)

# --- SECTION 3: NESTED RUNS ---
st.header("3. Nested Runs & Execution Hierarchy", divider="rainbow")
st.markdown("""
Complex agents generate deeply nested execution hierarchies. For example, an agent may invoke a subgraph, which calls 
a RAG chain, which calls a vector retriever, which invokes an embedding model.
""")

col_nest1, col_nest2 = st.columns(2)
with col_nest1:
    st.subheader("Lineage Tracking Mechanisms")
    st.markdown("""
    * **Trace Context Propagation**: LangChain automatically passes parent run IDs down through thread-local storage or `asyncio.ContextVar`.
    * **Distributed Tracing**: For multi-service architectures, trace IDs can be propagated across microservices using W3C TraceContext HTTP headers.
    * **Flame Graph Representation**: In the LangSmith UI, nested runs render as an interactive flame graph / waterfall chart.
    """)
with col_nest2:
    st.subheader("Programmatic Nested Run Context")
    st.code("""
from langsmith.run_trees import RunTree

# Create Root Run
root_run = RunTree(name="MasterAgent", run_type="chain", inputs={"q": "Analyze"})
root_run.post()

# Create Nested Child Run linked to Parent
child_run = root_run.create_child(
    name="SubQueryExecutor",
    run_type="chain",
    inputs={"step": 1}
)
child_run.post()
child_run.end(outputs={"status": "done"})
child_run.patch()

root_run.end(outputs={"final": "result"})
root_run.patch()
    """, language="python")

# --- SECTION 4: TOOL TRACES ---
st.header("4. Tool Traces & Argument Inspection", divider="rainbow")
st.markdown("""
Tool traces record the discrete execution of external integrations (SQL queries, calculators, web search APIs, Python interpreters). 
Inspecting tool traces is critical for identifying **hallucinated arguments** and **API serialization failures**.
""")

col_tl1, col_tl2 = st.columns(2)
with col_tl1:
    st.subheader("What Tool Traces Record")
    st.markdown("""
    * **Tool Name**: The `@tool` identifier selected by the model.
    * **Input Arguments**: The exact JSON dictionary decoded from the model's function calling response.
    * **Return Payload**: The raw string or structured object returned to the agent.
    * **Execution Duration**: Network latency vs internal computation time.
    """)
with col_tl2:
    st.subheader("Automatic Tool Tracing")
    st.code("""
from langchain_core.tools import tool

@tool
def calculate_vat(amount: float, country_code: str) -> float:
    \"\"\"Calculates VAT based on country standard rates.\"\"\"
    rates = {"DE": 0.19, "FR": 0.20, "UK": 0.20}
    return amount * rates.get(country_code.upper(), 0.15)

# When invoked inside an agent, inputs, outputs,
# and exceptions are automatically logged to LangSmith!
    """, language="python")

# --- SECTION 5: MODEL TRACES ---
st.header("5. Model Traces & Raw Token Telemetry", divider="rainbow")
st.markdown("""
Model runs (`run_type="llm"`) record the raw interactions with foundational models. 
Unlike application logs that show formatted strings, model traces preserve the exact wire payloads exchanged with LLM providers.
""")

col_mod1, col_mod2 = st.columns(2)
with col_mod1:
    st.subheader("Model Telemetry Metrics")
    st.markdown("""
    * **Full Formatted Prompts**: The rendered prompt template with all variables populated.
    * **Sampling Hyperparameters**: Temperature, top_p, frequency_penalty, and max_tokens.
    * **Token Usage Metrics**: `prompt_tokens`, `completion_tokens`, and `total_tokens`.
    * **Response Metadata**: Stop reasons (`stop`, `length`, `tool_calls`) and system fingerprints.
    """)
with col_mod2:
    st.subheader("Model Run Schema Excerpt")
    st.code("""
{
  "name": "ChatOpenAI",
  "run_type": "llm",
  "inputs": {
    "messages": [[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]]
  },
  "outputs": {
    "generations": [{"text": "...", "message": {"content": "..."}}]
  },
  "prompt_tokens": 842,
  "completion_tokens": 128
}
    """, language="python")

# --- SECTION 6: CHAIN TRACES ---
st.header("6. Chain Traces & LCEL Pipeline Steps", divider="rainbow")
st.markdown("""
In LCEL pipelines (`prompt | model | parser`), the composite pipeline constitutes a `chain` run type. 
LangSmith decomposes the pipeline into its constituent Runnable steps, showing intermediate data transformations.
""")

st.markdown("""
### LCEL Step Decomposition in LangSmith
* **`PromptTemplate` Run**: Shows input dictionary and rendered prompt string.
* **`ChatOpenAI` Run**: Shows rendered prompt in, `AIMessage` out.
* **`StrOutputParser` Run**: Shows `AIMessage` in, clean Python string out.
By inspecting chain traces, engineers can verify exactly where data formatting breaks along the pipe.
""")

# --- SECTION 7: AGENT TRACES ---
st.header("7. Agent Traces & ReAct Decision Trajectories", divider="rainbow")
st.markdown("""
Agent traces capture multi-step cyclic reasoning trajectories. A single user query may prompt an agent to execute 
a loop across reasoning, tool dispatch, observation ingestion, and final answer synthesis.
""")

col_ag1, col_ag2 = st.columns(2)
with col_ag1:
    st.subheader("Trajectory Step Audit")
    st.markdown("""
    * **Iteration Index**: Tracks each cycle in the agent loop.
    * **Reasoning Trace**: Inspects why the agent chose tool A over tool B.
    * **Multi-Turn Context**: Preserves the complete message buffer as it grows across iterations.
    * **Loop Termination**: Verifies whether the agent terminated cleanly via `final_answer` or hit recursion limits.
    """)
with col_ag2:
    st.subheader("Agent Trajectory Graph")
    st.code("""
# Trajectory Sequence in LangSmith Trace:
1. [chain] AgentExecutor (Root)
   ├── [llm] ChatOpenAI (Iteration 1: Decides Tool)
   ├── [tool] DatabaseSearch (Executes query)
   ├── [llm] ChatOpenAI (Iteration 2: Decides Tool)
   ├── [tool] Calculator (Calculates ratio)
   └── [llm] ChatOpenAI (Iteration 3: Synthesizes Final Answer)
    """, language="text")

# --- SECTION 8: METADATA ---
st.header("8. Metadata Annotation & Run Context", divider="rainbow")
st.markdown("""
**Metadata** consists of structured key-value pairs attached to a run. 
Unlike tags (which are simple strings), metadata supports arbitrary JSON-serializable dictionaries for deep attribution.
""")

col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    st.subheader("Enterprise Metadata Dimensions")
    st.markdown("""
    * **User Attribution**: `user_id`, `organization_id`, `tenant_id`.
    * **Session Context**: `session_id`, `conversation_id`, `client_ip`.
    * **System Architecture**: `app_version`, `git_commit`, `environment` (`staging`, `prod`).
    * **Business Metrics**: `subscription_tier`, `department_code`.
    """)
with col_meta2:
    st.subheader("Attaching Runtime Metadata")
    st.code("""
# Pass metadata via RunnableConfig:
config = {
    "metadata": {
        "tenant_id": "corp_acme",
        "user_email": "alice@acme.com",
        "environment": "production",
        "model_routing_tier": "fast"
    }
}

response = chain.invoke(inputs, config=config)
    """, language="python")

# --- SECTION 9: TAGS ---
st.header("9. Tags & Queryable Telemetry Labels", divider="rainbow")
st.markdown("""
**Tags** are simple string labels attached to runs to enable fast filtering and categorization in the LangSmith console. 
While metadata is used for structured context, tags are designed for quick cohort queries.
""")

col_tag1, col_tag2 = st.columns(2)
with col_tag1:
    st.subheader("Tagging Strategies")
    st.markdown("""
    * **Lifecycle Stage**: `dev`, `staging`, `prod`, `canary`.
    * **Experiment Cohort**: `prompt-v2-ab-test`, `temperature-0-test`.
    * **Feature Domain**: `billing`, `support`, `legal-search`.
    * **Failure Category**: `timeout`, `schema-error`, `prompt-injection`.
    """)
with col_tag2:
    st.subheader("Applying Tags in Invocations")
    st.code("""
# Attach tags via RunnableConfig:
config = {
    "tags": ["production", "experiment-b", "eu-west-1"]
}

# Tags are indexed for instant sub-second search in LangSmith!
response = chain.invoke(inputs, config=config)
    """, language="python")

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Hierarchical Run Tree Tracer with Custom Metadata and Tags Decorator", divider="rainbow")
st.markdown("""
The following complete production pattern illustrates how to build a custom traceable decorator that wraps 
agent functions, generates hierarchical run trees, and enriches traces with enterprise metadata and tags.
""")

st.code('''
import functools
import time
from typing import Callable, Any, Dict, List
from langsmith import Client
from langsmith.run_trees import RunTree

client = Client()

def enterprise_traced_node(
    run_name: str,
    run_type: str = "chain",
    tags: List[str] = None
) -> Callable:
    """
    Decorator that instruments any business logic function with
    LangSmith RunTree tracing, attaching metadata and timing telemetry.
    """
    tags = tags or []

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract optional parent run context or create new root
            parent_run: RunTree = kwargs.pop("_parent_run", None)
            inputs = {"args": str(args), "kwargs": kwargs}

            if parent_run:
                run = parent_run.create_child(
                    name=run_name,
                    run_type=run_type,
                    inputs=inputs,
                    tags=tags
                )
            else:
                run = RunTree(
                    name=run_name,
                    run_type=run_type,
                    inputs=inputs,
                    tags=tags,
                    client=client
                )

            run.post()
            start_epoch = time.time()

            try:
                result = func(*args, _current_run=run, **kwargs)
                duration = time.time() - start_epoch
                run.end(outputs={"output": result})
                run.patch()
                return result
            except Exception as exc:
                run.end(error=str(exc))
                run.patch()
                raise exc

        return wrapper
    return decorator
''', language="python")

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Hierarchical Run Tree Explorer & Tag Query Sandbox", divider="rainbow")
st.markdown("""
Explore a simulated LangSmith hierarchical run tree. Query runs by **Tag** or select an individual **Run ID** 
to inspect parent-child relationships, inputs, outputs, and attached metadata.
""")

# Instantiate Module Scope Simulator
if "tree_sim" not in st.session_state:
    st.session_state.tree_sim = MockHierarchicalRunTreeExplorer()

tree_sim = st.session_state.tree_sim

col_qb1, col_qb2 = st.columns([1, 1])

with col_qb1:
    st.subheader("1. Run Tree Tag Filter")
    tag_choices = ["ALL", "prod", "finance", "model", "planner", "synthesizer", "sec-edgar"]
    selected_tag = st.selectbox("Filter Runs by Tag:", options=tag_choices)
    matched_runs = tree_sim.query_by_tag(selected_tag)

    st.markdown(f"**Matched Runs Count**: `{len(matched_runs)}`")
    for r in matched_runs:
        st.markdown(f"""
        <div class="run-card">
            <h4>{r['name']} <code>({r['run_type']})</code></h4>
            <p><strong>Run ID:</strong> <code>{r['run_id']}</code> | <strong>Parent:</strong> <code>{r['parent_id'] or 'None (ROOT)'}</code></p>
            <p><strong>Latency:</strong> <code>{r['latency_ms']} ms</code></p>
            <div>
                {''.join([f'<span class="tag-badge">#{t}</span>' for t in r['tags']])}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_qb2:
    st.subheader("2. Detailed Run Payload Inspector")
    inspected_id = st.selectbox("Select Run ID to Inspect:", options=[r["run_id"] for r in tree_sim.runs])
    inspected_run = next(r for r in tree_sim.runs if r["run_id"] == inspected_id)

    st.markdown(f"**Name**: `{inspected_run['name']}`")
    st.markdown(f"**Run Type**: `{inspected_run['run_type']}`")
    st.markdown(f"**Latency**: `{inspected_run['latency_ms']} ms`")

    st.markdown("#### Metadata Annotations:")
    st.json(inspected_run["metadata"])

    st.markdown("#### Inputs & Outputs:")
    st.json({"inputs": inspected_run["inputs"], "outputs": inspected_run["outputs"]})

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: How does LangSmith represent parent-child relationships across nested runs?"):
    st.markdown("""
    Each run contains an `id` (its own UUID) and a `parent_run_id` pointing to the UUID of its parent run. 
    The root run has `parent_run_id = None`. In LangChain, the callback manager injects the current run ID into 
    thread-local or `ContextVar` storage, ensuring that any sub-run created during execution inherits the parent ID automatically.
    """)

with st.expander("Q2: What is the semantic difference between `metadata` and `tags` in LangSmith?"):
    st.markdown("""
    * **`tags`**: Flat list of strings used for categorical grouping and fast UI filtering (e.g., `["prod", "v2-canary"]`).
    * **`metadata`**: Structured JSON dictionary used for granular context and programmatic analytical slicing 
      (e.g., `{"tenant_id": "corp_402", "tokens": 120, "tier": "enterprise"}`).
    """)

with st.expander("Q3: How do tool traces assist in detecting LLM hallucination bugs?"):
    st.markdown("""
    When an LLM hallucinates an invalid tool argument (e.g. `query_database(table="non_existent_table")`), 
    the model run may look successful (it generated text), but the subsequent tool run records an error or empty result. 
    Inspecting the tool trace exposes the exact malformed argument passed by the model.
    """)

with st.expander("Q4: Can you dynamically update metadata on a run after it has started?"):
    st.markdown("""
    Yes. Using `RunTree.patch()` or the `client.update_run()` API, developers can attach dynamic metadata 
    (such as user feedback ratings, post-processed evaluation scores, or downstream transaction status) to an existing run.
    """)

with st.expander("Q5: What are the best practices for tracing long-running asynchronous background agents?"):
    st.markdown("""
    For long-running workflows spanning multiple asynchronous jobs, use explicit `trace_id` injection. 
    Pass the root `trace_id` in message queues (RabbitMQ, SQS, Celery) so worker pods create child runs linked 
    to the original root trace, maintaining unified lineage across distributed workers.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_tr_t1, col_tr_t2 = st.columns(2)
with col_tr_t1:
    st.error("""
    **Production Trap 1 (Fatal Error): High-Cardinality Tags Causing Index Degeneration**  
    Using unique values (such as `user_id`, UUIDs, or timestamps) as `tags` instead of `metadata`. 
    High-cardinality tags degrade search index performance in database systems. Always use `tags` for broad cohorts and `metadata` for unique IDs.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): Orphaned Child Runs from Lost Context**  
    Spawning background tasks with `asyncio.create_task()` without propagating the parent `CallbackManager` or `ContextVar`. 
    This severs the parent-child link, creating disconnected root runs instead of a unified run tree.
    """)
with col_tr_t2:
    st.error("""
    **Production Trap 3 (Fatal Error): Plaintext Password Logging in Tool Traces**  
    Allowing tools that accept credentials (e.g. database connection strings or API passwords) to trace inputs unredacted. 
    These credentials are saved to cloud trace storage. Always mark sensitive parameters for redaction.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): Massive Metadata Payloads**  
    Attaching large multi-megabyte payloads (such as raw 50MB PDF binaries) to run metadata. 
    Metadata should contain identifiers and summary statistics, not raw bulk binary blobs.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_f_c1, col_f_c2 = st.columns(2)
with col_f_c1:
    st.markdown("""
    <div class="run-card">
        <h4>Flashcard 1: parent_run_id</h4>
        <p><strong>Property:</strong> Lineage Pointer</p>
        <p><strong>Definition:</strong> The UUID field linking a child model or tool execution to its enclosing parent chain in the LangSmith run tree.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="run-card">
        <h4>Flashcard 2: RunType</h4>
        <p><strong>Enum:</strong> Component Classification</p>
        <p><strong>Definition:</strong> The discriminator defining whether a run represents a <code>chain</code>, <code>llm</code>, <code>tool</code>, <code>retriever</code>, or <code>prompt</code>.</p>
    </div>
    """, unsafe_allow_html=True)
with col_f_c2:
    st.markdown("""
    <div class="run-card">
        <h4>Flashcard 3: RunnableConfig Metadata</h4>
        <p><strong>Parameter:</strong> <code>config={"metadata": {...}}</code></p>
        <p><strong>Definition:</strong> The standard dictionary used to inject runtime user, tenant, and environment dimensions into LangChain traces.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="run-card">
        <h4>Flashcard 4: Tags</h4>
        <p><strong>Parameter:</strong> <code>config={"tags": [...]}</code></p>
        <p><strong>Definition:</strong> Flat list of indexed string labels attached to runs for fast categorizing, cohort filtering, and experiment tracking.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of LangSmith tracing and hierarchical run tree architectures:")

chk_tr1 = st.checkbox("1. I understand how trace runs and parent-child execution lineages are structured in LangSmith.")
chk_tr2 = st.checkbox("2. I know how nested runs capture internal steps within chains, agents, and subgraphs.")
chk_tr3 = st.checkbox("3. I understand what tool traces record and how to inspect tool arguments to debug hallucinations.")
chk_tr4 = st.checkbox("4. I know what model traces record, including prompt tokens, completion tokens, and stop reasons.")
chk_tr5 = st.checkbox("5. I understand the architectural difference between structured `metadata` and queryable `tags`.")
chk_tr6 = st.checkbox("6. I know why high-cardinality values must be stored in `metadata` rather than `tags`.")

if chk_tr1 and chk_tr2 and chk_tr3 and chk_tr4 and chk_tr5 and chk_tr6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of Tracing in LangSmith.")
