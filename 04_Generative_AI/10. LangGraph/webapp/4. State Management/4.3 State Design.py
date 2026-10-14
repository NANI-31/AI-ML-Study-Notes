import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="4.3 State Design - LangGraph Curriculum",
    page_icon="📐",
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
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE STATE DESIGN AUDITOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class StateDesignAuditor:
    """
    Audits state architectures for production readiness:
    Evaluates bloat risk, normalization status, channel naming hygiene,
    and ephemeral scratchpad lifecycle.
    """
    def __init__(self):
        self.best_practice_rules = [
            "Store pointers/URIs instead of raw heavy binary blobs (PDFs, images).",
            "Normalize entities by ID dictionary instead of repeating objects in lists.",
            "Use clear semantic prefixes: is_* for booleans, *_count for integers.",
            "Reset ephemeral scratchpad keys before exiting to prevent state pollution.",
            "Separate public API contracts from internal working memory."
        ]

    def audit_schema_channels(self, channels: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Audits a dictionary of channel specifications.
        Each channel: {'type': str, 'scope': str, 'estimated_size_kb': float, 'is_normalized': bool}
        """
        total_kb = sum(c["estimated_size_kb"] for c in channels.values())
        warnings = []
        recommendations = []

        # Rule 1: Size bloat
        if total_kb > 500:
            warnings.append(f"High Memory Footprint: Total estimated state size ({total_kb:.1f} KB) exceeds the 500 KB threshold. High risk of checkpointer I/O bottlenecks.")
            recommendations.append("Offload large objects to external storage (S3/blob) and retain only object URIs.")

        # Rule 2: Normalization
        denormalized = [k for k, c in channels.items() if not c.get("is_normalized", True) and "list" in c.get("type", "").lower()]
        if denormalized:
            warnings.append(f"Denormalized Lists Detected: Channels {denormalized} may duplicate records across supersteps.")
            recommendations.append("Adopt relational normalization: store a dict of {id: item} and a list of active_ids.")

        # Rule 3: Naming hygiene
        bad_names = [k for k in channels.keys() if " " in k or k.isupper() or len(k) < 2]
        if bad_names:
            warnings.append(f"Non-Standard Channel Names: {bad_names} violate PEP8 snake_case naming conventions.")
            recommendations.append("Rename channels to clean lowercase snake_case (e.g., query_context, user_session_id).")

        # Compute architecture score (0-100)
        penalty = min(70, len(warnings) * 20 + max(0, (total_kb - 100) / 20))
        score = max(30, int(100 - penalty))

        return {
            "score": score,
            "total_size_kb": total_kb,
            "warnings": warnings,
            "recommendations": recommendations,
            "channel_count": len(channels)
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: State Scoping, Normalization, and Clean Topology", divider="rainbow")

st.markdown("""
Designing state for enterprise LangGraph agents is an exercise in **data architecture**.
State is not a haphazard junk drawer; it is the persistent backbone that dictates thread durability, checkpointer performance,
and sub-agent modularity. A production state architecture segregates fields into four distinct scopes:
**Public State**, **Private State**, **Shared Coordination Channels**, and **Temporary Ephemeral Scratchpads**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    design_blueprint_dot = """
    digraph StateDesignArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_public_scope {
            label="1. Public State Scope (API / Client Surface)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            
            Public_Box [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#0369A1"><b>Public Channels (Egress Visible)</b></td></tr>
                <tr><td>query</td><td>str (User input)</td></tr>
                <tr><td>messages</td><td>list[BaseMessage] (Conversation history)</td></tr>
                <tr><td>final_answer</td><td>str (Synthesized response)</td></tr>
            </table>>, shape=plaintext];
        }

        subgraph cluster_shared_scope {
            label="2. Shared State Scope (Cross-Node Coordination)";
            style="filled";
            color="#1E293B";
            fontcolor="#A855F7";
            
            Shared_Box [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#6B21A8"><b>Shared Normalized Stores</b></td></tr>
                <tr><td>documents_by_id</td><td>Dict[str, DocMetadata] (Normalized store)</td></tr>
                <tr><td>active_doc_ids</td><td>List[str] (Relational pointer list)</td></tr>
                <tr><td>execution_status</td><td>str ('RETRIEVING' | 'REFINING')</td></tr>
            </table>>, shape=plaintext];
        }

        subgraph cluster_private_scope {
            label="3. Private State Scope (Internal Node Mechanics)";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            
            Private_Box [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#065F46"><b>Private Subgraph / Node Channels</b></td></tr>
                <tr><td>_internal_retry_count</td><td>int (Loop guard)</td></tr>
                <tr><td>_vector_raw_scores</td><td>List[float] (Internal heuristic)</td></tr>
                <tr><td>_auth_token_ref</td><td>str (Vault pointer reference)</td></tr>
            </table>>, shape=plaintext];
        }

        subgraph cluster_temporary_scope {
            label="4. Temporary Scratchpad Scope (Ephemeral)";
            style="dashed";
            color="#EAB308";
            fontcolor="#FACC15";
            
            Temp_Box [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#854D0E"><b>Ephemeral Working Memory</b></td></tr>
                <tr><td>_is_tool_call_pending</td><td>bool (Transient flag)</td></tr>
                <tr><td>_raw_llm_json_buffer</td><td>str (Cleared after parse)</td></tr>
            </table>>, shape=plaintext];
        }

        Public_Box -> Shared_Box [label="Ingest to Normalized Store"];
        Shared_Box -> Private_Box [label="Passed to Specialized Worker Nodes"];
        Private_Box -> Temp_Box [label="Allocates Ephemeral Buffers"];
        Temp_Box -> Shared_Box [label="Flushes Clean Output (Resets Temp)"];
        Shared_Box -> Public_Box [label="Projects Final Egress Response"];
    }
    """
    st.graphviz_chart(design_blueprint_dot, use_container_width=True)

st.markdown("""
### Core Architectural Axiom: State Minimization Principle
Every single byte placed in state is serialized to storage at the conclusion of every superstep.
If an agent graph takes 15 supersteps to complete a task, a 2 MB state payload results in **30 MB of database writes** and serialized network I/O.
**Golden Rule**: *Compute locally inside node functions; store in state only what is required to coordinate across supersteps or persist across threads.*
""")

# ==============================================================================
# SECTION 2: PUBLIC STATE
# ==============================================================================
st.header("2. Public State: External Interfaces and API Visibility", divider="rainbow")

st.markdown("""
**Public State** constitutes the contract between your LangGraph application and the external world (REST APIs, WebSockets, or UI widgets).
It defines the fields that external callers provide during initiation and consume upon conclusion.
""")

col_pub1, col_pub2 = st.columns(2)
with col_pub1:
    st.markdown("""
    #### Characteristics of Public State
    - **Stable & Versioned**: Public state should rarely change breaking keys. Treat it like a public REST API.
    - **Customer-Facing**: Contains user queries, chat message history, completed summaries, and user feedback.
    - **Clean Types**: Avoid internal LangGraph-specific types where possible; prefer standard strings, lists, and dicts.
    """)
with col_pub2:
    st.code("""class PublicConversationState(TypedDict):
    # Public Ingress & Egress Channels
    session_id: str
    user_query: str
    messages: list[BaseMessage]
    final_response: Optional[str]
    citations: list[dict]
""", language="python")

# ==============================================================================
# SECTION 3: PRIVATE STATE
# ==============================================================================
st.header("3. Private State: Internal Node Mechanics and Subgraph Encapsulation", divider="rainbow")

st.markdown("""
**Private State** encompasses data needed solely by internal reasoning nodes or nested subgraphs.
Private channels should never be exposed to clients, as they may contain sensitive telemetry, prompt engineering tokens, or raw tool logs.
""")

col_priv1, col_priv2 = st.columns(2)
with col_priv1:
    st.markdown("""
    #### Architectural Patterns for Private State
    - **Underscore Prefix Convention**: Prefix internal channels with an underscore (`_loop_count`, `_raw_tokens`) to signal private scope.
    - **Subgraph Encapsulation**: Modern LangGraph supports **subgraphs** that possess their own isolated private state schema, completely hidden from the parent graph.
    - **Security Isolation**: Keep API keys, session tokens, and database connection pools completely out of state. Pass them via `RunnableConfig["configurable"]`.
    """)
with col_priv2:
    st.code("""class InternalWorkerState(TypedDict):
    # Public Channels
    query: str
    
    # Private Working Channels (Filtered at Egress)
    _internal_retry_count: int
    _raw_vector_scores: list[float]
    _intermediate_tool_outputs: list[str]
""", language="python")

# ==============================================================================
# SECTION 4: SHARED STATE
# ==============================================================================
st.header("4. Shared State: Global Coordination Channels Across Parallel Branches", divider="rainbow")

st.markdown("""
**Shared State** comprises the central channels that coordinate communication between parallel nodes or autonomous agents.
When two or more parallel nodes branch out during a superstep, the shared state channels allow both nodes to synchronize their findings.
""")

# Contextual diagram for shared coordination
coord_dot = """
digraph SharedCoordination {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", margin="0.15,0.08"];
    edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

    Supervisor [label="Supervisor Agent Node\\nDispatches parallel tasks", fillcolor="#1E293B"];
    Worker_Web [label="Web Researcher Node\\nAppends findings", fillcolor="#334155"];
    Worker_SQL [label="Database Analyst Node\\nAppends findings", fillcolor="#334155"];
    Shared_Channel [label="Shared Channel: Annotated[List[Fact], operator.add]\\nSynchronizes concurrent findings", fillcolor="#6B21A8"];
    Synthesizer [label="Synthesizer Node\\nReads all collected facts", fillcolor="#065F46"];

    Supervisor -> Worker_Web;
    Supervisor -> Worker_SQL;
    Worker_Web -> Shared_Channel [label="Emit Fact A"];
    Worker_SQL -> Shared_Channel [label="Emit Fact B"];
    Shared_Channel -> Synthesizer [label="Consolidated Knowledge"];
}
"""
with st.expander("🤝 View Shared Parallel Coordination Channel Flow", expanded=True):
    st.graphviz_chart(coord_dot, use_container_width=True)

# ==============================================================================
# SECTION 5: TEMPORARY STATE
# ==============================================================================
st.header("5. Temporary State: Ephemeral Scratchpads and Transient Flags", divider="rainbow")

st.markdown("""
**Temporary (Ephemeral) State** channels are allocated for short-lived operational flags (e.g. `is_tool_call_pending`, `raw_llm_chunk_buffer`).
If temporary state is not cleaned up after its purpose is fulfilled, it persists into subsequent supersteps, leading to stale state bugs.
""")

col_temp1, col_temp2 = st.columns(2)
with col_temp1:
    st.markdown("""
    #### The Ephemeral Cleanup Rule
    Whenever a node consumes temporary state, it should emit an explicit reset in its partial delta dictionary:
    ```python
    def consumer_node(state: AgentState) -> dict:
        pending_flag = state.get("is_tool_call_pending", False)
        # Process the tool call...
        
        # Reset the ephemeral flag!
        return {
            "tool_result": "Success",
            "is_tool_call_pending": False # Cleaned
        }
    ```
    """)
with col_temp2:
    st.markdown("""
    #### Why Ephemeral Hygiene is Crucial
    - **Prevents Infinite Loops**: Stale `has_error = True` flags will trigger fallback routers indefinitely.
    - **Reduces Memory Footprint**: Large JSON buffers generated by LLMs are discarded once parsed into structured records.
    - **Clean Graph Telemetry**: LangSmith traces remain focused on meaningful domain transitions rather than lingering transient variables.
    """)

# ==============================================================================
# SECTION 6: STATE NORMALIZATION
# ==============================================================================
st.header("6. State Normalization: Relational and Deduplicated Structures", divider="rainbow")

st.markdown("""
In naive agent design, documents, tools, and message records are often duplicated across multiple lists in state.
This leads to exponential state bloat and synchronization inconsistencies.
Production systems use **Normalized State Architecture**, storing entities in a central dictionary by unique ID.
""")

col_norm_bad, col_norm_good = st.columns(2)
with col_norm_bad:
    st.error("❌ Denormalized State (Bloated & Inconsistent)")
    st.code("""class BadState(TypedDict):
    all_retrieved_docs: list[dict] # 50 KB
    selected_docs: list[dict]      # Duplicates same 50 KB!
    reranked_docs: list[dict]      # Duplicates same 50 KB!
    # Total: 150 KB serialized per superstep!
""", language="python")

with col_norm_good:
    st.success("✅ Normalized State (Lean & Relational)")
    st.code("""class NormalizedState(TypedDict):
    # Store documents once in an ID-keyed lookup table
    document_store: dict[str, dict] # 50 KB
    
    # Store only lightweight ID references elsewhere:
    selected_doc_ids: list[str]     # 20 bytes!
    reranked_doc_ids: list[str]     # 20 bytes!
    # Total: ~50 KB serialized per superstep!
""", language="python")

# ==============================================================================
# SECTION 7: STATE NAMING CONVENTIONS
# ==============================================================================
st.header("7. State Naming Conventions: Semantic Clarity and Namespace Standards", divider="rainbow")

st.markdown("""
Adopt a consistent naming taxonomy across your engineering team to ensure readability, maintainability, and robust self-documenting graphs.
""")

st.markdown("""
| Channel Type | Naming Convention | Examples | Anti-Patterns to Avoid |
| :--- | :--- | :--- | :--- |
| **Booleans / Flags** | `is_*`, `has_*`, `should_*` | `is_authorized`, `has_failed`, `should_route_tool` | `flag`, `status_bool`, `check` |
| **Counters / Limits** | `*_count`, `*_index`, `max_*` | `retry_count`, `current_step_index`, `max_loops` | `c`, `i`, `num`, `loops` |
| **Collections** | Plural nouns or `*_list` | `messages`, `document_ids`, `retrieved_chunks` | `message_data`, `docs_obj` |
| **Mappings / Registries** | `*_by_id`, `*_map`, `*_lookup` | `documents_by_id`, `user_metadata_map` | `dict1`, `hash_table` |
| **Private Channels** | Leading underscore `_*` | `_raw_token_count`, `_internal_scratchpad` | Mixing private and public names |
| **Timestamps / Dates** | `*_at`, `*_timestamp` | `created_at`, `last_evaluated_timestamp` | `time`, `date_val` |
""")

# ==============================================================================
# SECTION 8: AVOIDING UNNECESSARY STATE
# ==============================================================================
st.header("8. Avoiding Unnecessary State: Memory Bloat and Serialization Pitfalls", divider="rainbow")

st.markdown("""
A common beginner mistake is placing everything into LangGraph state.
The table below specifies what belongs in graph state versus external runtime environments.
""")

col_what_in, col_what_out = st.columns(2)
with col_what_in:
    st.markdown("""
    #### ✅ What Belongs in Graph State
    - **Conversation Context**: Messages (`BaseMessage`) requiring LLM prompt inclusion.
    - **Routing Variables**: Classification labels, branch choices, loop counters.
    - **Entity IDs**: Document IDs, user IDs, session identifiers.
    - **Domain Outputs**: Extracted entities, final answers, customer-facing tables.
    """)
with col_what_out:
    st.markdown("""
    #### ❌ What NEVER Belongs in Graph State
    - **Database Connections & Clients**: Unpicklable objects (e.g. `psycopg2.connect`, `boto3.client`).
    - **Heavy Binary Blobs**: Raw PDFs, full video files, high-res images (store in S3; store S3 URI in state).
    - **Embedding Matrices**: Raw tensor floats (store in Vector Database; store query in state).
    - **API Keys & Secrets**: Pass via `RunnableConfig` to avoid storing secrets in disk checkpoints.
    """)

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Clean Architected Production Agent State", divider="rainbow")

st.markdown("""
Below is a complete, production-tested implementation demonstrating clean state scoping,
normalized document stores, ephemeral scratchpad handling, and public output projection.
""")

clean_state_prod_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Step 1: Define Normalized Domain State
class NormalizedEnterpriseState(TypedDict):
    # Public Core Channels
    session_id: str
    user_query: str
    
    # Normalized Data Storage (Entities stored once by ID)
    document_store: Dict[str, Dict[str, Any]]
    relevant_doc_ids: List[str]
    
    # Shared Append-Reducer Channel
    audit_trail: Annotated[List[str], operator.add]
    
    # Private Working Channels
    _retry_counter: int
    _is_scratchpad_dirty: bool
    
    # Public Egress Channel
    synthesized_summary: Optional[str]

# Step 2: Implement Functional Nodes with Normalized State
def retrieve_docs_node(state: NormalizedEnterpriseState) -> Dict[str, Any]:
    # Simulate fetching knowledge docs
    doc_a = {"id": "doc_001", "title": "LangGraph Best Practices", "size_kb": 12.4}
    doc_b = {"id": "doc_002", "title": "State Normalization Guide", "size_kb": 8.1}
    
    # Store normalized entities by ID
    updated_store = {doc_a["id"]: doc_a, doc_b["id"]: doc_b}
    
    return {
        "document_store": updated_store,
        "relevant_doc_ids": ["doc_001", "doc_002"],
        "audit_trail": ["Retrieved and normalized 2 knowledge documents."],
        "_is_scratchpad_dirty": True
    }

def synthesize_summary_node(state: NormalizedEnterpriseState) -> Dict[str, Any]:
    # Resolve IDs against normalized store
    doc_ids = state["relevant_doc_ids"]
    store = state["document_store"]
    titles = [store[did]["title"] for did in doc_ids if did in store]
    
    summary = f"Summary compiled from: {', '.join(titles)}"
    
    # Clean up ephemeral flag upon completing synthesis
    return {
        "synthesized_summary": summary,
        "audit_trail": ["Synthesis completed successfully."],
        "_is_scratchpad_dirty": False # Ephemeral reset
    }

# Step 3: Public Projection Schema (Strips private & raw stores)
class PublicReportOutput(TypedDict):
    session_id: str
    synthesized_summary: str
    audit_trail: List[str]

# Step 4: Assemble Clean Graph Pipeline
def build_clean_architecture_graph():
    builder = StateGraph(NormalizedEnterpriseState, output=PublicReportOutput)
    
    builder.add_node("retrieve", retrieve_docs_node)
    builder.add_node("synthesize", synthesize_summary_node)
    
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = build_clean_architecture_graph()
    initial_payload = {
        "session_id": "sess_prod_404",
        "user_query": "How do I structure large enterprise agent state?",
        "document_store": {},
        "relevant_doc_ids": [],
        "audit_trail": ["Graph session initiated."],
        "_retry_counter": 0,
        "_is_scratchpad_dirty": False,
        "synthesized_summary": None
    }
    output = app.invoke(initial_payload)
    print("Public Clean Egress Keys:", list(output.keys()))
    print("Summary:", output["synthesized_summary"])
'''
st.code(clean_state_prod_code, language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: State Bloat Analyzer and Architecture Inspector", divider="rainbow")

st.markdown("""
Evaluate the health of your state design in real-time. Add channels, configure estimated payloads,
and run the **State Design Auditor** to inspect serialization overhead, normalization status, and naming hygiene.
""")

auditor = StateDesignAuditor()

col_wb_l, col_wb_r = st.columns([1, 1.2])

with col_wb_l:
    st.subheader("Configure State Channels to Audit")
    
    sample_channels = {
        "user_query": {"type": "str", "scope": "Public", "estimated_size_kb": 0.5, "is_normalized": True},
        "messages": {"type": "list[BaseMessage]", "scope": "Public", "estimated_size_kb": 12.0, "is_normalized": True},
        "raw_pdf_bytes": {"type": "bytes", "scope": "Private", "estimated_size_kb": 450.0, "is_normalized": False},
        "all_candidate_documents": {"type": "list[dict]", "scope": "Shared", "estimated_size_kb": 120.0, "is_normalized": False},
        "active_doc_ids": {"type": "list[str]", "scope": "Shared", "estimated_size_kb": 0.2, "is_normalized": True},
        "_is_tool_pending": {"type": "bool", "scope": "Temporary", "estimated_size_kb": 0.01, "is_normalized": True}
    }

    include_heavy_blob = st.checkbox("Include Heavy Binary Blob (`raw_pdf_bytes`: 450 KB)", value=True)
    include_denorm = st.checkbox("Include Denormalized Document List (`all_candidate_documents`: 120 KB)", value=True)
    fix_naming = st.checkbox("Apply Standard snake_case Naming Conventions", value=False)

    test_channels = dict(sample_channels)
    if not include_heavy_blob:
        del test_channels["raw_pdf_bytes"]
        test_channels["pdf_s3_uri"] = {"type": "str", "scope": "Private", "estimated_size_kb": 0.1, "is_normalized": True}
        
    if not include_denorm:
        del test_channels["all_candidate_documents"]
        test_channels["document_store_by_id"] = {"type": "dict[str, dict]", "scope": "Shared", "estimated_size_kb": 25.0, "is_normalized": True}

    if not fix_naming:
        test_channels["BAD Channel NAME"] = {"type": "str", "scope": "Public", "estimated_size_kb": 0.1, "is_normalized": True}

    audit_result = auditor.audit_schema_channels(test_channels)

with col_wb_r:
    st.subheader("State Architecture Health Report")
    
    c_s1, c_s2, c_s3 = st.columns(3)
    c_s1.metric("Design Health Score", f"{audit_result['score']} / 100")
    c_s2.metric("Total Size (Per Superstep)", f"{audit_result['total_size_kb']:.1f} KB")
    c_s3.metric("Channel Count", audit_result["channel_count"])

    if audit_result["warnings"]:
        st.error(f"⚠️ {len(audit_result['warnings'])} Design Violations Detected:")
        for w in audit_result["warnings"]:
            st.markdown(f"- 🔴 {w}")
    else:
        st.success("🎉 Zero Critical Violations Detected! State is lean and normalized.")

    st.markdown("**Architectural Recommendations:**")
    for r in audit_result["recommendations"]:
        st.markdown(f"- 💡 {r}")

st.markdown("### Configured Channels Telemetry")
df_channels = pd.DataFrame([
    {"Channel": k, "Type": v["type"], "Scope": v["scope"], "Size (KB)": v["estimated_size_kb"], "Normalized": v["is_normalized"]}
    for k, v in test_channels.items()
])
st.dataframe(df_channels, use_container_width=True)

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: What is the Pointer Pattern in LangGraph state design and why is it critical?", expanded=False):
    st.markdown("""
    **Answer**:
    The Pointer Pattern replaces heavy data objects (raw PDFs, images, high-dimensional tensors, extensive SQL dumps) with lightweight references or URIs (such as `s3://bucket/doc_102.pdf` or database row IDs).
    Because LangGraph serializes the entire state snapshot at every superstep, storing a 10 MB PDF inside state would trigger 100+ MB of serialized writes across a 10-step agent loop. Using pointers keeps superstep serialization under 10 KB and eliminates database latency spikes.
    """)

with st.expander("Q2: How do you design state for a multi-agent system to prevent sub-agents from overwriting each other?", expanded=False):
    st.markdown("""
    **Answer**:
    1. **Namespaced Keys**: Assign sub-agent private channels using prefixed namespaces (e.g. `researcher_findings`, `coder_code_snippet`).
    2. **Reducer Channels**: For shared dialogue or task backlogs, use `Annotated[list, operator.add]` so agents append contributions without overwriting.
    3. **Subgraphs with Isolated Schemas**: Encapsulate each specialized sub-agent as a child `CompiledStateGraph` with its own private schema. The parent graph only maps inputs and outputs at the boundary.
    """)

with st.expander("Q3: What are Ephemeral Scratchpad channels and how should their lifecycle be managed?", expanded=False):
    st.markdown("""
    **Answer**:
    Ephemeral channels store transient flags or intermediate buffers (such as `is_waiting_for_user_approval` or `raw_llm_json`).
    Their lifecycle must include an explicit **cleanup phase**: the node that completes processing the ephemeral data must return `{'_is_scratchpad_active': False}` or `{'_raw_buffer': None}` in its update delta. Otherwise, stale values cause subsequent cycles to misfire.
    """)

with st.expander("Q4: Why is normalized state preferred over nested denormalized lists in production graphs?", expanded=False):
    st.markdown("""
    **Answer**:
    In denormalized state, if five different nodes each maintain lists of documents, the same document dictionary is duplicated five times in memory and serialized storage.
    In normalized state, documents are stored once in an ID-keyed dictionary (`documents_by_id: dict[str, dict]`), and nodes pass lightweight ID strings (`['doc_1', 'doc_2']`). This reduces memory by 80% and ensures that updating a document's metadata in one place updates it everywhere.
    """)

with st.expander("Q5: Can LangGraph state hold unpicklable objects like active database connection pools or socket handles?", expanded=False):
    st.markdown("""
    **Answer**:
    **No**. LangGraph state must be strictly serializable (via JSON, pickle, or msgpack) to support checkpointers (`MemorySaver`, `SqliteSaver`, `AsyncPostgresSaver`).
    Active database connections, thread locks, and network sockets cannot be serialized and will cause runtime `PicklingError` or `TypeError`. Non-serializable clients should be instantiated globally or injected via `config["configurable"]`.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr_e1, col_tr_e2 = st.columns(2)
with col_tr_e1:
    st.error("🚨 Trap 1 (Fatal Error): Serializing Database Connections or Thread Locks in State")
    st.markdown("""
    *Mechanism*: Placing an instantiated `psycopg2.connection` or `threading.Lock` object inside the graph state dictionary.  
    *Consequence*: The checkpointer crashes immediately with `TypeError: cannot pickle '_thread.lock' object`.  
    *Fix*: Inject active connection clients via `RunnableConfig` or application-level dependency injection.
    """)

with col_tr_e2:
    st.error("🚨 Trap 2 (Fatal Error): Lingering Stale Flags in Cyclic Agent Loops")
    st.markdown("""
    *Mechanism*: Setting `{"has_tool_error": True}` in an error node without clearing it when the tool retries successfully.  
    *Consequence*: Subsequent iterations read the stale error flag, causing the agent to get trapped in an infinite error-handling loop.  
    *Fix*: Always emit explicit resets `{"has_tool_error": False}` on successful recovery transitions.
    """)

col_tr_w1, col_tr_w2 = st.columns(2)
with col_tr_w1:
    st.warning("⚠️ Trap 3 (Performance Warning): Checkpointer I/O Saturation from Heavy State Bloat")
    st.markdown("""
    *Mechanism*: Storing full-text articles and uncompressed image frames in central state across 20+ supersteps.  
    *Consequence*: Postgres or SQLite checkpointer experiences high write latency, slowing down agent response times by multiple seconds.  
    *Fix*: Apply the Pointer Pattern: store heavy assets in cloud object storage and store lightweight URIs in state.
    """)

with col_tr_w2:
    st.warning("⚠️ Trap 4 (Architectural Warning): High Coupling Across Un-Namespaced Multi-Agent Channels")
    st.markdown("""
    *Mechanism*: Allowing 5 parallel agents to write to generic scalar keys like `status` or `data` without reducers or namespaces.  
    *Consequence*: Race conditions, un-deterministic overwrites, and untraceable bugs in LangSmith logs.  
    *Fix*: Use namespaced keys (`agent_a_status`, `agent_b_status`) or wrap agents in isolated subgraphs.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Public State**\\nStable, client-facing channels exposed through ingress input schemas and egress output projections.")
with col_fc2:
    st.info("**Private State**\\nInternal working memory and intermediate telemetry hidden from external callers.")
with col_fc3:
    st.info("**State Normalization**\\nStoring data entities once in an ID-keyed dictionary and referencing them via lightweight IDs.")
with col_fc4:
    st.info("**Pointer Pattern**\\nStoring external object storage URIs instead of heavy raw binary payloads in graph state.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to divide state into Public, Private, Shared, and Temporary scopes.", value=True)
c2 = st.checkbox("I understand the Pointer Pattern to prevent heavy binary bloat from choking checkpointer storage.", value=True)
c3 = st.checkbox("I can implement relational normalization for documents and entities in agent state.", value=True)
c4 = st.checkbox("I know how to manage the lifecycle of ephemeral scratchpads to prevent stale flag loops.", value=True)
c5 = st.checkbox("I adhere to standardized naming conventions for booleans, counters, and mappings.", value=True)
c6 = st.checkbox("I understand why unpicklable objects like database connections must never be stored in state.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 4.3 'State Design' with full architectural and practical comprehension!")
