import streamlit as st
import json
import time

st.set_page_config(
    page_title="22.3 Debugging Complex Systems - Prompt Engineering",
    page_icon="🕸️",
    layout="wide"
)

# Header Section
st.title("22.3 Debugging Complex Multi-Component Prompt Systems")
st.caption("Chapter 22: Prompt Debugging and Troubleshooting | Domain: 1. Prompt Engineering")
st.markdown("""
In compound AI systems—such as **multi-step prompt chains, tool-augmented models, RAG pipelines, and autonomous agent loops**—debugging 
becomes an distributed tracing challenge. A failure in step 4 is rarely caused by prompt 4; it is almost always the result of 
subtle error compounding, context poisoning, schema misalignment, or unhandled tool exceptions in earlier steps.
""")

st.divider()

# Section 1: Complex System Failure Topologies
st.header("1. Failure Modes Across Compound Architecture Layers", divider="rainbow")
st.markdown("""
Compound systems combine multiple non-deterministic model calls with deterministic software components. 
Isolating failure points requires layer-specific diagnostics:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Prompt Chains: Error Compounding")
    st.markdown("""
    * **Failure Pattern**: A slight formatting defect or mild hallucination in Prompt 1 (e.g., classifying a question slightly off-target) 
    is accepted by Prompt 2 as authoritative ground truth, leading to catastrophic degradation by Prompt 3.
    * **Diagnostic**: Unit-test each chain link independently with pre-canned, golden intermediate outputs.
    """)

    st.subheader("2. Tool Calls: Schema & Execution Drift")
    st.markdown("""
    * **Failure Pattern**: The LLM outputs hallucinated parameters (`{"user_id": "all"}` instead of an integer), 
    or the external API throws a 500 error that gets unhandled, causing the agent to loop endlessly.
    * **Diagnostic**: Inspect raw tool invocation payload before calling the tool; mock tool execution returns.
    """)

with col2:
    st.subheader("3. RAG: Retrieval Poisoning & Mismatch")
    st.markdown("""
    * **Failure Pattern**: Top-k chunks contain irrelevant or outdated corporate documentation that directly contradicts 
    the system instructions, poisoning the synthesis prompt.
    * **Diagnostic**: Log the exact injected context chunks alongside cosine similarity distances.
    """)

    st.subheader("4. Autonomous Agents: Infinite Loops & Goal Amnesia")
    st.markdown("""
    * **Failure Pattern**: The agent executes the same tool repeatedly with identical failing parameters, 
    or forgets its original objective after 5 turns as conversational context rolls over.
    * **Diagnostic**: Set hard step limits (`max_iterations=5`), cycle detection heuristics, and persistent scratchpad memory.
    """)

st.divider()

# Section 2: Distributed Tracing & Observability Architecture
st.header("2. Distributed Tracing Architecture for Compound LLMs", divider="rainbow")
st.markdown("""
Without end-to-end tracing, diagnosing multi-component failures is impossible. 
Modern architectures wrap every prompt, tool execution, and vector search in distributed trace spans:
""")

st.graphviz_chart('''
digraph TraceArchitecture {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    UserReq [label="User Query\\nTrace ID: #tr-891", fillcolor="#E1BEE7"];
    Rewriter [label="Step 1: Query Rewriter\\n(Prompt Call)", fillcolor="#BBDEFB"];
    Retriever [label="Step 2: Vector DB\\n(Semantic Search)", fillcolor="#C8E6C9"];
    ToolRunner [label="Step 3: Tool Execution\\n(Python SQL Sandbox)", fillcolor="#FFF9C4"];
    Synthesizer [label="Step 4: Final Synthesizer\\n(Generation LLM)", fillcolor="#BBDEFB"];
    TraceCollector [label="OpenTelemetry / Langfuse\\nTrace Dashboard", fillcolor="#FFCDD2"];
    
    UserReq -> Rewriter;
    Rewriter -> Retriever [label="rewritten_query"];
    Retriever -> ToolRunner [label="context_docs"];
    ToolRunner -> Synthesizer [label="tool_result"];
    
    Rewriter -> TraceCollector [style=dashed, label="Span 1"];
    Retriever -> TraceCollector [style=dashed, label="Span 2"];
    ToolRunner -> TraceCollector [style=dashed, label="Span 3"];
    Synthesizer -> TraceCollector [style=dashed, label="Span 4"];
}
''', use_container_width=False)

st.divider()

# Section 3: Trace Inspection Code Pattern
st.header("3. Implementing Trace Spans & Failure Point Detection", divider="rainbow")
st.markdown("""
The following Python pattern illustrates how to decorate prompt chains and tool executions with structured tracing 
to pinpoint the exact node where execution fails:
""")

st.code('''# Structured Observability & Trace Span Pattern
import time
import uuid

class TraceLogger:
    def __init__(self, trace_id=None):
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.spans = []

    def log_span(self, component: str, inputs: dict, outputs: dict, latency_ms: float, status: str = "SUCCESS", error: str = None):
        span = {
            "trace_id": self.trace_id,
            "component": component,
            "inputs": inputs,
            "outputs": outputs,
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "error": error
        }
        self.spans.append(span)
        return span

def execute_compound_pipeline(user_query: str, tracer: TraceLogger):
    # Step 1: Query Expansion
    t0 = time.time()
    try:
        expanded_query = f"{user_query} [relevance expanded]"
        tracer.log_span("QueryExpansion", {"query": user_query}, {"expanded": expanded_query}, (time.time()-t0)*1000)
    except Exception as e:
        tracer.log_span("QueryExpansion", {"query": user_query}, {}, (time.time()-t0)*1000, status="FAILED", error=str(e))
        raise

    # Step 2: Vector Retrieval
    t0 = time.time()
    retrieved_chunks = ["Docs: Return policy allows 30 days refund."]
    tracer.log_span("VectorRetriever", {"search_query": expanded_query}, {"chunks_found": len(retrieved_chunks)}, (time.time()-t0)*1000)

    # Step 3: Synthesis Generation
    t0 = time.time()
    synthesis = "Based on our return policy, refunds are permitted within 30 days."
    tracer.log_span("LLMSynthesizer", {"context_count": len(retrieved_chunks)}, {"completion_chars": len(synthesis)}, (time.time()-t0)*1000)
    
    return synthesis, tracer.spans''', language="python")

st.divider()

# Section 4: Interactive Trace Diagnostic Workbench
st.header("4. Interactive System Trace Diagnostic Workbench", divider="rainbow")
st.markdown("Inspect simulated execution traces to locate the exact point of system breakdown.")

selected_trace = st.selectbox(
    "Select Compound Trace Log to Diagnose:",
    [
        "Trace #101: RAG Context Poisoning leading to False Refusal",
        "Trace #102: Agent Tool Schema Parameter Type Mismatch",
        "Trace #103: Agent Infinite ReAct Loop (Cycle Detection)"
    ]
)

if selected_trace == "Trace #101: RAG Context Poisoning leading to False Refusal":
    st.subheader("Trace #101 Analysis")
    st.markdown("**User Query**: *'How do I reset my API key?'*")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.info("Span 1: Query Expansion (Success, 120ms)")
        st.code('Input: "How do I reset my API key?"\nOutput: "API key reset procedure token generate credentials"', language="text")
        
        st.error("Span 2: Vector Retrieval (Defective Context Retrieved, 45ms)")
        st.code('Retrieved Chunk (Score 0.41): "Legacy Deprecation Notice: On Dec 2021, all manual API keys were revoked."\nMissing: Current 2025 SSO Key management doc.', language="text")
        
    with col_t2:
        st.error("Span 3: Final Synthesis (False Refusal, 340ms)")
        st.code('Response: "API keys are deprecated and cannot be reset according to corporate policies."', language="text")
        st.markdown("**Root Cause Identified**: Vector database indexed stale deprecation notes with higher lexical overlap than current SSO docs.")
        st.success("Remediation: Add metadata filters (`doc_version >= 2024`) or semantic reranker.")

elif selected_trace == "Trace #102: Agent Tool Schema Parameter Type Mismatch":
    st.subheader("Trace #102 Analysis")
    st.markdown("**User Query**: *'Look up flight UA342 arriving at SFO on May 12'*")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.info("Span 1: Agent Reasoning (Success, 450ms)")
        st.code('Thought: I should use the flight_status tool.\nAction: flight_status\nAction Input: {"flight_num": 342, "date": "May 12th"}', language="text")
        
        st.error("Span 2: Tool Execution (FAILED: 422 Unprocessable Entity)")
        st.code('API Error: date must be in ISO format YYYY-MM-DD. flight_num must be string.', language="text")
        
    with col_t2:
        st.warning("Span 3: Agent Error Recovery (Failed)")
        st.code('Agent repeated identical input: {"flight_num": 342, "date": "May 12th"}', language="text")
        st.markdown(r"**Root Cause Identified**: Tool schema lacked Pydantic regex patterns (`pattern=r'^\d{4}-\d{2}-\d{2}$'`) and few-shot formatting.")
        st.success("Remediation: Enforce JSON Schema format constraints in tool definition.")

else:
    st.subheader("Trace #103 Analysis")
    st.markdown("**User Query**: *'What is the stock price of PrivateCo Ltd?'*")
    
    st.error("Cycle Detected: Agent executed `search_web('PrivateCo Ltd stock price')` 5 consecutive times without state change.")
    st.code('''Turn 1: search_web("PrivateCo Ltd stock price") -> "No public ticker found"
Turn 2: search_web("PrivateCo stock ticker") -> "No public ticker found"
Turn 3: search_web("PrivateCo Ltd stock market") -> "No public ticker found"
Turn 4: search_web("PrivateCo quotation") -> "No public ticker found"
Turn 5: Timeout exceeded.''', language="text")
    st.markdown("**Root Cause Identified**: The agent prompt had no explicit stop condition or awareness of private companies.")
    st.success("Remediation: Inject rule: *'If a tool returns no matches twice, stop searching and inform the user that the entity may be private or unlisted.'*")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: How do you prevent compounding errors in long prompt chains?"):
        st.write("""
        1. **Contract Isolation**: Treat each prompt as an independent microservice with typed Pydantic inputs and outputs.
        2. **Validator Interceptors**: Place deterministic validation logic between chain steps. If step 2 output fails schema validation, 
        trigger a localized single-step retry rather than re-running the entire pipeline.
        3. **Confidence Scoring**: Have intermediate steps output confidence ratings. If confidence falls below 0.7, 
        divert to fallback logic or human-in-the-loop escalation.
        """)

    with st.expander("Q: What is 'Context Poisoning' in RAG systems and how do you trace it?"):
        st.write("""
        Context poisoning occurs when a retrieved chunk contains misleading, outdated, or adversarial text 
        that overrides the system instructions or corrupts generation. 
        It is traced by recording the exact chunk IDs and relevance scores in distributed trace spans, 
        and running automated claim-attribution evaluations that map generated claims back to individual chunk IDs.
        """)

with qa2:
    with st.expander("Trap to Avoid: Debugging Agents Without Step Execution Limits"):
        st.write("""
        Never deploy an autonomous agent workflow without a strict hard ceiling on iterations (`max_iterations = 6`) 
        and total execution cost (`max_cost = $0.50`). Without these safety breakers, a minor formatting bug in a tool schema 
        can trap the agent in an infinite retry loop, burning thousands of dollars in API credits within minutes.
        """)

    with st.expander("Flashcards: Complex System Debugging"):
        fc_sel = st.selectbox(
            "Select Architecture Term:",
            ["Compounding Drift", "Tool Schema Drift", "Context Poisoning", "Agent Cycle Breaker"]
        )
        if fc_sel == "Compounding Drift":
            st.info("Small errors in early pipeline links that snowball into massive hallucinations downstream.")
        elif fc_sel == "Tool Schema Drift":
            st.info("When an LLM produces parameters that violate the tool function's expected data types.")
        elif fc_sel == "Context Poisoning":
            st.info("Injected RAG chunks that mislead or contradict the system instructions.")
        else:
            st.info("Heuristic logic that terminates agent execution if identical tool calls are executed repeatedly.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I know how to trace errors across compound RAG and agent architectures.")
    st.checkbox("I can implement distributed trace span logging for LLM applications.")
with c2:
    st.checkbox("I understand how to protect agent loops from infinite cycles with step ceilings.")
    st.checkbox("I can isolate whether a bug originated in prompt formulation, tool schema, or vector retrieval.")
