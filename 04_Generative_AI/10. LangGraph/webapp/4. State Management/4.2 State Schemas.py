import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="4.2 State Schemas - LangGraph Curriculum",
    page_icon="📋",
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
# MODULE-SCOPE SCHEMA VALIDATION ENGINE FOR WORKBENCH & PIPELINE
# ==============================================================================
class SchemaValidationEngine:
    """
    Simulates schema validation, nested state parsing, and input/output
    projections across TypedDict, Dataclass, and Pydantic schema paradigms.
    """
    def __init__(self):
        self.schema_definitions = {
            "TypedDict": {
                "overhead": "0.00 ms (Zero runtime check in CPython)",
                "validation": "Static linting only (mypy / pyright)",
                "mutability": "Standard Python Dictionary",
                "recommended_for": "High-throughput production pipelines, fast prototyping"
            },
            "Dataclass": {
                "overhead": "~0.05 ms (Object instantiation)",
                "validation": "Shallow type hinting, custom __post_init__",
                "mutability": "Mutable by default (frozen=True supported)",
                "recommended_for": "OOP agent design, internal domain logic"
            },
            "Pydantic": {
                "overhead": "~0.85 ms (Deep runtime coercion and validation)",
                "validation": "Strict runtime regex, range, and type parsing",
                "mutability": "Immutable (frozen=True) or validated mutable",
                "recommended_for": "Ingress API boundaries, untrusted external LLM JSON"
            }
        }

    def validate_payload(self, schema_type: str, payload: Dict[str, Any], input_only: bool = False) -> Dict[str, Any]:
        """
        Evaluates a candidate state dictionary against schema invariants.
        """
        required_fields = ["query", "session_id"] if input_only else ["query", "session_id", "status"]
        errors = []
        parsed = {}

        for req in required_fields:
            if req not in payload or payload[req] is None or payload[req] == "":
                errors.append(f"Missing required field: '{req}'")

        if "confidence" in payload and payload["confidence"] is not None:
            try:
                val = float(payload["confidence"])
                if not (0.0 <= val <= 1.0):
                    errors.append(f"Range validation failed: 'confidence' ({val}) must be between 0.0 and 1.0.")
                parsed["confidence"] = val
            except (ValueError, TypeError):
                errors.append(f"Type validation failed: 'confidence' must be float-compatible.")

        if "nested_metadata" in payload:
            meta = payload["nested_metadata"]
            if not isinstance(meta, dict):
                errors.append(f"Nested structure error: 'nested_metadata' must be a dictionary.")
            elif "source" not in meta:
                errors.append(f"Nested schema error: 'nested_metadata' missing mandatory key 'source'.")
            else:
                parsed["nested_metadata"] = meta

        is_valid = len(errors) == 0
        return {
            "valid": is_valid,
            "schema_used": schema_type,
            "errors": errors,
            "validated_data": payload if is_valid else None
        }

    def project_output_state(self, full_state: Dict[str, Any], output_fields: List[str]) -> Dict[str, Any]:
        """
        Simulates LangGraph's output_schema projection.
        Filters internal scratchpads and returns only customer-facing keys.
        """
        return {k: full_state[k] for k in output_fields if k in full_state}


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Schema Paradigms and Type Systems in LangGraph", divider="rainbow")

st.markdown("""
In LangGraph, the **State Schema** represents the formal data contract governing all communication within the graph.
LangGraph supports three primary schema paradigms: **TypedDict**, **Dataclasses**, and **Pydantic Models**.
Furthermore, LangGraph allows splitting state into distinct **Input Schemas**, **Internal Overall Schemas**, and **Output Schemas**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    schema_blueprint_dot = """
    digraph SchemaArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Ingress Boundary (Input Schema)";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            
            Input_Contract [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#1E293B"><b>InputState (TypedDict / Pydantic)</b></td></tr>
                <tr><td>query</td><td>str (Required)</td></tr>
                <tr><td>user_id</td><td>str (Required)</td></tr>
            </table>>, shape=plaintext];
        }

        subgraph cluster_runtime_state {
            label="2. Graph Core Execution (Overall State Schema)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            
            Overall_State [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#0284C7"><b>OverallState (Full Internal Schema)</b></td></tr>
                <tr><td>query</td><td>str</td></tr>
                <tr><td>user_id</td><td>str</td></tr>
                <tr><td>messages</td><td>Annotated[list, add_messages]</td></tr>
                <tr><td>raw_tool_tokens</td><td>int (Private Scratchpad)</td></tr>
                <tr><td>internal_score</td><td>float (Private Metric)</td></tr>
                <tr><td>final_answer</td><td>str (Public Response)</td></tr>
            </table>>, shape=plaintext];

            Node_Worker1 [label="Worker Node 1\\nReads overall state\\nAppends internal scratchpad", fillcolor="#334155"];
            Node_Worker2 [label="Worker Node 2\\nReads internal scratchpad\\nGenerates final_answer", fillcolor="#334155"];
        }

        subgraph cluster_egress {
            label="3. Egress Boundary (Output Schema Projection)";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            
            Output_Contract [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#059669"><b>OutputState (Filtered Projection)</b></td></tr>
                <tr><td>final_answer</td><td>str</td></tr>
                <tr><td>messages</td><td>list[BaseMessage]</td></tr>
            </table>>, shape=plaintext];
            
            END_CLIENT [label="Client / API Response\\nScratchpads stripped automatically", fillcolor="#1E293B", shape=ellipse];
        }

        Input_Contract -> Overall_State [label="Runtime Validation & Injection"];
        Overall_State -> Node_Worker1 [label="Full State Access"];
        Node_Worker1 -> Node_Worker2 [label="Internal State Passing"];
        Node_Worker2 -> Overall_State [label="Write final_answer"];
        Overall_State -> Output_Contract [label="LangGraph Filters Output Schema"];
        Output_Contract -> END_CLIENT [label="Clean Payload Returned"];
    }
    """
    st.graphviz_chart(schema_blueprint_dot, use_container_width=True)

st.markdown("""
### The Three-Tier Schema Separation Paradigm
Modern LangGraph systems strictly decouple what enters the graph from what lives in the graph and what exits:
1. **Input Schema (`input_schema`)**: Minimal payload required to initiate the graph. Enforces that clients don't provide internal runtime keys.
2. **Overall State (`state_schema`)**: The complete superset of all channels, containing internal logs, tool buffers, and agent scratchpads.
3. **Output Schema (`output_schema`)**: The clean, filtered projection exposed back to callers. Protects private internal reasoning tokens.
""")

# ==============================================================================
# SECTION 2: TYPEDDICT
# ==============================================================================
st.header("2. TypedDict: Lightweight High-Performance Schemas", divider="rainbow")

st.markdown("""
`TypedDict` from Python's standard `typing` module is the official, default, and most performant schema format for LangGraph.
Under the hood, a `TypedDict` instance is just a standard Python dictionary at runtime, incurring **zero performance overhead**.
""")

col_td1, col_td2 = st.columns(2)
with col_td1:
    st.markdown("""
    #### Why TypedDict is the Default Choice
    - **Zero Serialization Penalty**: Standard dictionaries serialize directly into JSON or msgpack in milliseconds.
    - **Native Reducer Support**: Seamless integration with `typing.Annotated[list, operator.add]`.
    - **Full IDE Type Hinting**: Autocomplete and type linting in VS Code and PyCharm via static analysis.
    - **Partial Dict Returns**: Nodes can naturally return `{}` or partial dicts without instantiating class objects.
    """)
with col_td2:
    code_typeddict = """from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage

class FastAgentState(TypedDict):
    session_id: str
    query: str
    step_count: int
    messages: Annotated[List[BaseMessage], operator.add]

# Usage in Graph Compilation
from langgraph.graph import StateGraph
builder = StateGraph(FastAgentState)
"""
    st.code(code_typeddict, language="python")

# ==============================================================================
# SECTION 3: DATACLASSES
# ==============================================================================
st.header("3. Dataclasses: Object-Oriented Mutable and Immutable Schemas", divider="rainbow")

st.markdown("""
Python `@dataclass` classes can also serve as LangGraph state schemas. They are particularly popular in domain-driven
codebases that prefer object-oriented syntax or require default attribute factories.
""")

col_dc1, col_dc2 = st.columns(2)
with col_dc1:
    st.markdown("""
    #### Dataclass Considerations
    - **Field Defaults**: Easy specification of default values via `field(default_factory=list)`.
    - **Method Support**: Can host helper methods for domain computations.
    - **Immutability Option**: Can be defined with `@dataclass(frozen=True)` to prevent in-place accidental modifications.
    - **Note on Updates**: Even with dataclasses, LangGraph node functions still return **dictionaries** of field updates!
    """)
with col_dc2:
    code_dataclass = """from dataclasses import dataclass, field
from typing import Annotated, List
import operator

@dataclass
class DomainAgentState:
    session_id: str
    query: str
    iteration: int = 0
    # Reducers work via Annotated on dataclass fields
    logs: Annotated[List[str], operator.add] = field(default_factory=list)

# Nodes still return partial dictionaries:
def analyze_node(state: DomainAgentState) -> dict:
    return {"iteration": state.iteration + 1, "logs": ["Analyzed query"]}
"""
    st.code(code_dataclass, language="python")

# ==============================================================================
# SECTION 4: PYDANTIC MODELS
# ==============================================================================
st.header("4. Pydantic Models: Runtime Validation and Coercion", divider="rainbow")

st.markdown("""
Pydantic `BaseModel` (v2) provides strict runtime data parsing, schema coercion, and attribute validation.
It is the gold standard when ingesting untrusted user inputs or parsing structured outputs from LLM function calls.
""")

col_pyd1, col_pyd2 = st.columns(2)
with col_pyd1:
    st.markdown("""
    #### When to Use Pydantic vs TypedDict
    - **API Gateways**: Use Pydantic as the `input_schema` to validate request payloads before graph invocation.
    - **Strict RegEx & Ranges**: Validate regex patterns, email formats, or bounded numbers (`Field(ge=0, le=100)`).
    - **Runtime Coercion**: Automatically convert incoming string integers (`"42"`) into real Python integers (`42`).
    - **Trade-off**: Incur ~0.5ms - 1ms overhead per node execution due to runtime schema re-validation.
    """)
with col_pyd2:
    code_pydantic = """from pydantic import BaseModel, Field
from typing import Annotated, List
import operator

class ValidatedAgentState(BaseModel):
    user_query: str = Field(min_length=3, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    retry_count: int = Field(ge=0, default=0)
    
    # Reducers on Pydantic fields
    tags: Annotated[List[str], operator.add] = Field(default_factory=list)

# Initialized with runtime guarantees:
# ValidatedAgentState(user_query="Hi", confidence=1.5) -> Raises ValidationError!
"""
    st.code(code_pydantic, language="python")

# ==============================================================================
# SECTION 5: OPTIONAL FIELDS
# ==============================================================================
st.header("5. Optional Fields and total=False Configurations", divider="rainbow")

st.markdown("""
In many workflows, certain state channels are not populated at the beginning of the run.
LangGraph provides two ways to accommodate optional and unpopulated keys:
""")

col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    st.markdown("""
    #### 1. typing.Optional (Nullable Channels)
    All keys are required to exist in the dictionary, but their values can be `None`.
    ```python
    class AgentState(TypedDict):
        query: str                 # Mandatory
        retrieved_docs: Optional[list] # Mandatory key, can be None
        final_answer: Optional[str]   # Mandatory key, can be None
    ```
    """)
with col_opt2:
    st.markdown("""
    #### 2. total=False (Omittable Channels)
    Keys do not even need to exist in the initial dictionary.
    ```python
    class FlexibleState(TypedDict, total=False):
        query: str          # May be present or absent
        error_flag: bool    # Only populated if error occurs
        debug_trace: list   # Only present in debug mode
    ```
    """)

# ==============================================================================
# SECTION 6: NESTED STATE STRUCTURES
# ==============================================================================
st.header("6. Nested State Structures and Hierarchical Payloads", divider="rainbow")

st.markdown("""
When building complex enterprise agent graphs, flat state schemas quickly become unreadable.
You can nest `TypedDict` or Pydantic models to group related channels logically.
""")

nested_code = """from typing import TypedDict, Optional, List

class SecurityContext(TypedDict):
    user_role: str
    tenant_id: str
    authorized_tools: List[str]

class GenerationMetrics(TypedDict):
    input_tokens: int
    output_tokens: int
    latency_ms: float

class EnterpriseAgentState(TypedDict):
    # Top-level state hosting nested schemas
    session_id: str
    query: str
    security: SecurityContext
    metrics: GenerationMetrics
    response: Optional[str]

def node_with_nested_access(state: EnterpriseAgentState) -> dict:
    role = state["security"]["user_role"]
    tokens = state["metrics"]["input_tokens"]
    return {
        "metrics": {**state["metrics"], "latency_ms": 142.5}
    }
"""
st.code(nested_code, language="python")

# ==============================================================================
# SECTION 7: STATE VALIDATION AND SCHEMA ENFORCEMENT
# ==============================================================================
st.header("7. State Validation and Schema Enforcement at Runtime", divider="rainbow")

st.markdown("""
LangGraph validates that incoming inputs conform to the schema during graph invocation.
If required fields are missing or type coercion fails, LangGraph raises clear runtime exceptions before running any node.
""")

# Contextual diagram for schema validation
val_diagram_dot = """
digraph ValidationPipeline {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", margin="0.15,0.08"];
    edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

    Payload [label="Incoming Client Payload\\n{'query': 'Search Docs'}", fillcolor="#1E293B"];
    Validator [label="LangGraph Ingress Validator\\nChecks: required fields, types, reducers", fillcolor="#0284C7"];
    Decision [label="Validation Passed?", shape=diamond, fillcolor="#334155"];
    Success [label="Initialize Superstep 0\\nDispatch active entry nodes", fillcolor="#059669"];
    Failure [label="Raise ValidationError\\nAbort before node execution", fillcolor="#DC2626"];

    Payload -> Validator;
    Validator -> Decision;
    Decision -> Success [label="Yes: All types match"];
    Decision -> Failure [label="No: Missing keys or bad types"];
}
"""
with st.expander("🛡️ View Schema Validation Ingress Pipeline", expanded=True):
    st.graphviz_chart(val_diagram_dot, use_container_width=True)

# ==============================================================================
# SECTION 8: INPUT STATE SCHEMAS
# ==============================================================================
st.header("8. Input State Schemas for Graph Ingestion", divider="rainbow")

st.markdown("""
By passing an explicit `input_schema` to `StateGraph`, you define the exact contract that external callers must satisfy.
This prevents callers from injecting unexpected internal keys.
""")

code_input_schema = """from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. External Callers only supply this:
class UserInputState(TypedDict):
    raw_prompt: str
    user_id: str

# 2. Internal Graph holds this:
class InternalState(TypedDict):
    raw_prompt: str
    user_id: str
    sanitized_prompt: str
    internal_tokens: int
    final_output: str

# 3. Bind input schema explicitly
builder = StateGraph(InternalState, input=UserInputState)
"""
st.code(code_input_schema, language="python")

# ==============================================================================
# SECTION 9: OUTPUT STATE SCHEMAS
# ==============================================================================
st.header("9. Output State Schemas and Filtered Projections", divider="rainbow")

st.markdown("""
Similarly, specifying an `output_schema` ensures that only designated fields are returned to the caller upon completion.
Private scratchpads, token counts, and internal intermediate steps are cleanly stripped out.
""")

code_output_schema = """# 4. Caller only sees this clean response:
class PublicOutputState(TypedDict):
    final_output: str

# Bind both input and output schemas
builder = StateGraph(InternalState, input=UserInputState, output=PublicOutputState)
graph = builder.compile()

# When invoked, graph only returns {'final_output': '...'}
result = graph.invoke({"raw_prompt": "Hello", "user_id": "usr_99"})
print(result.keys()) # Output: dict_keys(['final_output'])
"""
st.code(code_output_schema, language="python")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Multi-Schema Pipeline with Input and Output Separation", divider="rainbow")

st.markdown("""
Here is a complete production pipeline showing multi-schema definition, Pydantic ingress validation,
and clean output projection.
""")

full_schema_prod_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# Tier 1: Input Schema (Strict Pydantic Validation for Public API)
class PublicRequest(BaseModel):
    query: str = Field(min_length=5, description="Search query")
    max_results: int = Field(default=3, ge=1, le=10)

# Tier 2: Internal State Schema (TypedDict for high-speed execution)
class InternalGraphState(TypedDict):
    query: str
    max_results: int
    raw_database_records: List[Dict[str, Any]]
    execution_trace: Annotated[List[str], operator.add]
    summary_report: str

# Tier 3: Output Schema (Clean Customer Response)
class PublicResponse(TypedDict):
    summary_report: str
    execution_trace: List[str]

# Node 1: Fetch
def fetch_records_node(state: InternalGraphState) -> Dict[str, Any]:
    limit = state["max_results"]
    records = [{"id": i, "title": f"Doc {i}"} for i in range(1, limit + 1)]
    return {
        "raw_database_records": records,
        "execution_trace": [f"Fetched {len(records)} database records"]
    }

# Node 2: Synthesize
def synthesize_node(state: InternalGraphState) -> Dict[str, Any]:
    records = state["raw_database_records"]
    report = f"Synthesized report across {len(records)} verified sources."
    return {
        "summary_report": report,
        "execution_trace": ["Report synthesis completed successfully"]
    }

# Build Pipeline with Input and Output Separation
def create_multischema_agent():
    # Pass input and output schema contracts
    builder = StateGraph(InternalGraphState, input=PublicRequest, output=PublicResponse)
    
    builder.add_node("fetch", fetch_records_node)
    builder.add_node("synthesize", synthesize_node)
    
    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "synthesize")
    builder.add_edge("synthesize", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = create_multischema_agent()
    # Invoke with minimal input payload
    clean_output = app.invoke({"query": "AI Agents in 2026", "max_results": 2})
    print("Filtered Output Keys:", list(clean_output.keys()))
    print("Report:", clean_output["summary_report"])
'''
st.code(full_schema_prod_code, language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Schema Comparison and Runtime Validation Simulator", divider="rainbow")

st.markdown("""
Test how different schema strategies (TypedDict, Dataclass, Pydantic) validate inputs, handle optional fields,
and project filtered outputs.
""")

validator_engine = SchemaValidationEngine()

col_sc_left, col_sc_right = st.columns([1, 1.2])

with col_sc_left:
    st.subheader("Payload Configuration")
    chosen_paradigm = st.selectbox("Select Schema Paradigm", ["TypedDict", "Dataclass", "Pydantic"])
    
    input_query = st.text_input("Payload 'query'", value="Analyze portfolio risk")
    input_session = st.text_input("Payload 'session_id'", value="sess_tx_901")
    input_conf = st.text_input("Payload 'confidence' (Float 0.0 - 1.0)", value="0.85")
    include_meta = st.checkbox("Include Nested Metadata Object", value=True)
    meta_src = st.text_input("Nested 'source' key", value="Bloomberg API" if include_meta else "")

    test_input_only = st.checkbox("Evaluate as Input Schema (Ignore internal keys)", value=True)

    btn_eval = st.button("🧪 Validate Against Schema Contract", use_container_width=True)

with col_sc_right:
    st.subheader("Validation & Projection Results")
    
    payload_to_test = {
        "query": input_query,
        "session_id": input_session,
        "confidence": input_conf,
        "status": "IN_PROGRESS" if not test_input_only else None
    }
    if include_meta:
        payload_to_test["nested_metadata"] = {"source": meta_src}

    res = validator_engine.validate_payload(chosen_paradigm, payload_to_test, input_only=test_input_only)
    
    if res["valid"]:
        st.success(f"✅ Schema Contract Satisfied ({chosen_paradigm})")
        st.json(res["validated_data"])
        
        st.markdown("**Simulated Output Schema Projection (`output_schema`):**")
        projected = validator_engine.project_output_state(
            payload_to_test,
            output_fields=["query", "confidence"]
        )
        st.info(f"Projected Egress Payload: {projected}")
    else:
        st.error(f"❌ Validation Errors Detected ({len(res['errors'])} errors):")
        for err in res["errors"]:
            st.markdown(f"- 🔴 `{err}`")

st.markdown("### Paradigm Benchmark & Architectural Trade-offs")
df_schemas = pd.DataFrame([
    {"Paradigm": k, **v} for k, v in validator_engine.schema_definitions.items()
])
st.dataframe(df_schemas, use_container_width=True)

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: What are the primary trade-offs between TypedDict and Pydantic as a LangGraph state schema?", expanded=False):
    st.markdown("""
    **Answer**:
    - **TypedDict**: Provides zero runtime overhead (pure CPython dictionaries), instantaneous serialization, and native compatibility with partial dictionary returns. However, it offers zero runtime type enforcement or coercion.
    - **Pydantic**: Guarantees strict runtime validation, automatic type conversion, and nested field constraints. However, it incurs roughly 0.5ms–1.0ms computational overhead per node and requires more careful handling when returning partial dictionaries.
    """)

with st.expander("Q2: How does LangGraph enforce separation between input_schema, overall state, and output_schema?", expanded=False):
    st.markdown("""
    **Answer**:
    When creating a `StateGraph`, you can provide:
    ```python
    builder = StateGraph(OverallState, input=InputState, output=OutputState)
    ```
    1. During `.invoke()` or `.stream()`, LangGraph validates the incoming payload exclusively against `InputState`.
    2. During graph execution, the state channels expand to accommodate `OverallState`.
    3. At terminal nodes (`END`), LangGraph projects the final state through `OutputState`, discarding all internal channels that are not part of `OutputState`.
    """)

with st.expander("Q3: Can you use total=False with Annotated reducers in TypedDict?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes. You can declare `class State(TypedDict, total=False):` where some fields use `Annotated[list, operator.add]`.
    However, if an `Annotated` channel with a reducer is not initialized at superstep 0, LangGraph requires that the first node populating the channel emits an initial valid collection. If a reducer receives `None` or an uninitialized key, it may raise a `TypeError` unless the reducer function explicitly handles `None` defaults.
    """)

with st.expander("Q4: How does LangGraph handle schema evolution when migrating checkpoints across application versions?", expanded=False):
    st.markdown("""
    **Answer**:
    When persisted checkpoints contain fields that were modified or removed in a newer codebase version:
    1. With `TypedDict`, extra keys in old checkpoints are simply preserved in the dictionary without breaking execution.
    2. With Pydantic models, schema evolution requires setting `model_config = ConfigDict(extra="ignore")` or providing default values for newly added fields to prevent `ValidationError` on historical thread resumption.
    """)

with st.expander("Q5: Why should you avoid deeply nested dictionaries inside state channels?", expanded=False):
    st.markdown("""
    **Answer**:
    In LangGraph, channel reducers operate at the top-level keys of the state dictionary.
    If you nest fields deeply (`state["agent"]["memory"]["tools"]["history"]`), LangGraph cannot apply partial updates to individual subkeys. A node updating one subkey must overwrite the entire parent object, destroying concurrency and increasing the risk of race conditions between parallel nodes.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Missing Fields in Strict TypedDict Input Payloads")
    st.markdown("""
    *Mechanism*: Using a default `TypedDict` (which has `total=True`) and passing an input dictionary that omits optional internal channels.  
    *Consequence*: Runtime KeyError or missing key exceptions inside nodes that assume all declared keys are populated.  
    *Fix*: Use separate `InputState` schemas or annotate optional keys with `Optional[T]` and provide default initialization in START.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Mutating Pydantic Model Attributes In-Place")
    st.markdown("""
    *Mechanism*: Executing `state.messages.append(...)` or `state.counter += 1` when using Pydantic models.  
    *Consequence*: Pydantic objects are mutated in-memory, bypassing LangGraph's delta tracking and corrupting checkpoint rollbacks.  
    *Fix*: Always return new partial dictionaries: `return {"counter": state.counter + 1}`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Pydantic Re-Validation Overhead in High-Throughput Loops")
    st.markdown("""
    *Mechanism*: Using complex Pydantic models with hundreds of regex validators in agent graphs that cycle 50+ times per query.  
    *Consequence*: Latency increases by 50ms - 100ms purely from Python object instantiation and validation.  
    *Fix*: Use Pydantic only at the `input_schema` boundary; use lightweight `TypedDict` for internal graph states.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Leaking Internal Tool Scratchpads to Public APIs")
    st.markdown("""
    *Mechanism*: Failing to specify an `output_schema`, causing internal raw tool responses, SQL credentials, or reasoning tokens to be returned to clients.  
    *Consequence*: Security vulnerabilities and massive egress payload bloat.  
    *Fix*: Always define a dedicated `output_schema` to project only authorized public response fields.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**TypedDict**\\nZero-overhead static type dictionary; the standard recommended schema for LangGraph internal state.")
with col_fc2:
    st.info("**Pydantic Schema**\\nStrict runtime validation and coercion schema; ideal for input validation and public API contracts.")
with col_fc3:
    st.info("**Input Schema**\\nA restricted schema passed to `StateGraph(input=...)` defining external ingress parameters.")
with col_fc4:
    st.info("**Output Schema**\\nA filtered projection passed to `StateGraph(output=...)` that strips internal scratchpads upon exit.")

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know when to choose TypedDict versus Dataclasses versus Pydantic models.", value=True)
c2 = st.checkbox("I understand the architectural purpose of separating Input, Overall, and Output schemas.", value=True)
c3 = st.checkbox("I know how to handle unpopulated keys using Optional and total=False configurations.", value=True)
c4 = st.checkbox("I understand the performance implications of runtime Pydantic validation in cyclic graphs.", value=True)
c5 = st.checkbox("I can structure nested state schemas without destroying top-level channel reducer granularity.", value=True)
c6 = st.checkbox("I know how to prevent private scratchpads and reasoning tokens from leaking to external API clients.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 4.2 'State Schemas' with full architectural and practical comprehension!")
