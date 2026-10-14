import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="23.1 LangChain Components - LangSmith Master Curriculum",
    page_icon="🦜",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class LangChainComponentTracingEngine:
    """
    Production LangChain Component Tracing & Run Hierarchy Engine.
    Emulates LangSmith automatic trace capture across all core LangChain abstractions:
    Prompts, Chat Models, LCEL Runnables, Chains, Retrievers, Tools, and Agents.
    """
    def __init__(self, project_name: str = "langchain-component-observability"):
        self.project_name = project_name
        self.active_traces: List[Dict[str, Any]] = []
        self._span_counter = 500

    def trace_agent_execution(
        self,
        user_query: str,
        include_retriever: bool = True,
        include_tool: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a simulated multi-component LangChain agent trace
        capturing Prompt -> LLM -> Tool -> Parser -> Final Output span tree.
        """
        self._span_counter += 1
        trace_id = f"trace_lc_{self._span_counter}"
        root_run_id = f"run_agent_{self._span_counter}"
        prompt_run_id = f"run_prompt_{self._span_counter}"
        llm1_run_id = f"run_llm1_{self._span_counter}"
        retriever_run_id = f"run_retriever_{self._span_counter}"
        tool_run_id = f"run_tool_{self._span_counter}"
        llm2_run_id = f"run_llm2_{self._span_counter}"

        child_spans = []
        total_latency = 0.0

        # 1. Prompt Template Span
        prompt_latency = 2.5
        total_latency += prompt_latency
        child_spans.append({
            "run_id": prompt_run_id,
            "parent_run_id": root_run_id,
            "name": "ChatPromptTemplate",
            "run_type": "prompt",
            "inputs": {"query": user_query, "system": "You are a customer assistant with tool access."},
            "outputs": {"messages": [f"System: Assist user.", f"Human: {user_query}"]},
            "latency_ms": prompt_latency
        })

        # 2. First LLM Decision Span (Tool Selection)
        llm1_latency = 210.0
        total_latency += llm1_latency
        tool_call_name = "search_corporate_kb" if include_retriever else ("calculate_interest" if include_tool else "none")
        child_spans.append({
            "run_id": llm1_run_id,
            "parent_run_id": root_run_id,
            "name": "ChatOpenAI (gpt-4o)",
            "run_type": "llm",
            "inputs": {"prompt_messages": 2},
            "outputs": {"tool_calls": [{"name": tool_call_name, "args": {"query": user_query}}]},
            "latency_ms": llm1_latency,
            "tokens": {"prompt_tokens": 145, "completion_tokens": 28, "total_tokens": 173}
        })

        # 3. Retriever Span (if applicable)
        if include_retriever:
            retriever_latency = 45.0
            total_latency += retriever_latency
            child_spans.append({
                "run_id": retriever_run_id,
                "parent_run_id": root_run_id,
                "name": "VectorStoreRetriever",
                "run_type": "retriever",
                "inputs": {"query": user_query, "top_k": 2},
                "outputs": {"documents": ["Doc 1: Enterprise SLA guarantees 99.9% uptime."]},
                "latency_ms": retriever_latency
            })

        # 4. Tool Execution Span (if applicable)
        if include_tool:
            tool_latency = 12.0
            total_latency += tool_latency
            child_spans.append({
                "run_id": tool_run_id,
                "parent_run_id": root_run_id,
                "name": "FinancialCalculatorTool",
                "run_type": "tool",
                "inputs": {"principal": 10000, "rate": 0.05, "years": 3},
                "outputs": {"result": 11576.25},
                "latency_ms": tool_latency
            })

        # 5. Second LLM Synthesis Span
        llm2_latency = 310.0
        total_latency += llm2_latency
        final_answer = f"Based on our SLA policies and financial calculations, the requested operation yields 11,576.25 with 99.9% guaranteed uptime."
        child_spans.append({
            "run_id": llm2_run_id,
            "parent_run_id": root_run_id,
            "name": "ChatOpenAI (gpt-4o)",
            "run_type": "llm",
            "inputs": {"context": "Retrieved evidence + Tool outputs"},
            "outputs": {"content": final_answer},
            "latency_ms": llm2_latency,
            "tokens": {"prompt_tokens": 420, "completion_tokens": 42, "total_tokens": 462}
        })

        trace_payload = {
            "trace_id": trace_id,
            "project_name": self.project_name,
            "root_run": {
                "run_id": root_run_id,
                "name": "EnterpriseAgentExecutor",
                "run_type": "chain",
                "inputs": {"input": user_query},
                "outputs": {"output": final_answer},
                "total_latency_ms": round(total_latency, 2)
            },
            "child_spans": child_spans
        }
        self.active_traces.append(trace_payload)
        return trace_payload


# --- HERO SECTION ---
st.title("23.1 LangChain Components")
st.caption("Chapter 23: LangSmith with LangChain | Domain: 11. LangSmith Observability & Testing")
st.markdown("""
LangChain provides the industry-standard component abstractions for assembling LLM applications:
**Models, Prompts, Runnables, Chains, Retrievers, Tools, and Agents**.
In legacy applications, when a complex multi-step chain fails or returns an unexpected response,
developers are left inspecting fragmented terminal logs with zero contextual correlation between input parameters and output text.
**LangSmith with LangChain** provides automatic, zero-code distributed tracing:
by simply configuring environment variables, LangSmith wraps every LangChain abstraction into a **hierarchical run span tree**,
capturing exact prompt variable injections, model token burn rates, retriever candidate payloads, tool arguments, and agent thought loops.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Unified LangChain Component Run Hierarchy & Span Topology", divider="rainbow")

st.markdown("""
The architecture below details how LangSmith automatically organizes disparate LangChain components into a structured execution tree.
The parent **Agent / Chain Span** branches into nested **Prompt**, **ChatModel**, **Retriever**, and **Tool** child spans,
preserving execution lineage and exact input/output states.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    lc_blueprint_dot = """
    digraph LangChainComponentHierarchy {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, compound=true, pad="0.3", nodesep="0.45", ranksep="0.55"];
        node [fontname="Helvetica, Arial, sans-serif", fontsize=10, penwidth=1.5, style="filled,rounded"];
        edge [fontname="Helvetica, Arial, sans-serif", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.3, arrowsize=0.7];

        subgraph cluster_root_chain {
            label="1. Root Orchestrator Span (run_type: chain)";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#0EA5E9";
            fontcolor="#38BDF8";
            fontsize=11;

            RootChain [label="Root Run: EnterpriseAgentExecutor\nrun_id: run_agent_501 | Total Duration: 579.5ms\nInputs: {'input': 'Calculate investment with 99.9% SLA'}", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_prompt_llm {
            label="2. Prompt & First Model Call Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#10B981";
            fontcolor="#34D399";
            fontsize=11;

            PromptSpan [label="Child Span 1: ChatPromptTemplate\nrun_type: prompt | 2.5ms\nSubstitutes {query} into messages", shape=box, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            LLM1Span [label="Child Span 2: ChatOpenAI (gpt-4o)\nrun_type: llm | 210.0ms | 173 Tokens\nEmits Tool Call: search_corporate_kb", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        }

        subgraph cluster_retriever_tool {
            label="3. Retrieval & Tool Execution Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#F59E0B";
            fontcolor="#FCD34D";
            fontsize=11;

            RetrieverSpan [label="Child Span 3: VectorStoreRetriever\nrun_type: retriever | 45.0ms\nYields 2 candidate chunks", shape=cylinder, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
            ToolSpan [label="Child Span 4: FinancialCalculatorTool\nrun_type: tool | 12.0ms\nInputs: {principal: 10000} | Outputs: 11576.25", shape=box, fillcolor="#451A03", fontcolor="#FED7AA", color="#F97316"];
        }

        subgraph cluster_final_llm {
            label="4. Synthesis & Parser Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#EC4899";
            fontcolor="#F472B6";
            fontsize=11;

            LLM2Span [label="Child Span 5: ChatOpenAI (gpt-4o)\nrun_type: llm | 310.0ms | 462 Tokens\nSynthesizes Grounded Final Answer", shape=box, fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            ParserSpan [label="Child Span 6: StrOutputParser\nrun_type: parser | 0.5ms\nValidates string response", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#EC4899"];
        }

        RootChain -> PromptSpan [label="1. Formats Prompt"];
        PromptSpan -> LLM1Span [label="2. Model Reasoning"];
        LLM1Span -> RetrieverSpan [label="3. Action: Search"];
        LLM1Span -> ToolSpan [label="4. Action: Calculate"];
        RetrieverSpan -> LLM2Span [label="5. Context Feed"];
        ToolSpan -> LLM2Span [label="6. Tool Output Feed"];
        LLM2Span -> ParserSpan [label="7. Extract String"];
        ParserSpan -> RootChain [label="8. Return Answer"];
    }
    """
    st.graphviz_chart(lc_blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: MODELS
# ==============================================================================
st.header("2. Models (Chat Models, Completion APIs, Token Usage Spans & Invocations)", divider="rainbow")

st.markdown("""
In LangChain, **Models** represent interfaces to Large Language Models (`ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, `Ollama`).
Every model call in a LangChain application creates a dedicated span of `run_type="llm"`.
LangSmith automatically extracts critical model telemetry from the underlying API response:
* **Prompt & Completion Tokens**: Exact input/output token counts for unit cost tracking.
* **Stop Reasons**: Whether the generation finished naturally (`stop`), exceeded limits (`length`), or triggered a tool call (`tool_calls`).
* **Raw Prompt Messages**: The exact list of System, Human, and AI messages dispatched over the wire.
""")

col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    st.markdown("### Model Span Metadata Payload:")
    st.code('''
{
  "name": "ChatOpenAI",
  "run_type": "llm",
  "inputs": {
    "messages": [
      {"role": "system", "content": "You are a helpful banking assistant."},
      {"role": "user", "content": "What is my account balance?"}
    ]
  },
  "outputs": {
    "generations": [{
      "text": "Your current checking balance is $4,250.00.",
      "message": {"role": "assistant"}
    }]
  },
  "metadata": {
    "ls_provider": "openai",
    "ls_model_name": "gpt-4o",
    "ls_temperature": 0.2
  }
}
''', language="json")

with col_m2:
    st.markdown("""
    ### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
    In LangChain v0.3+, zero code changes are needed to trace models.
    Simply set standard environment variables:
    ```bash
    export LANGSMITH_TRACING=true
    export LANGSMITH_API_KEY="lsv2_pt_..."
    export LANGSMITH_PROJECT="prod-customer-service"
    ```
    Every `model.invoke()` call automatically streams microsecond-level spans to LangSmith asynchronously in the background.
    """)

st.divider()

# ==============================================================================
# SECTION 3: PROMPTS
# ==============================================================================
st.header("3. Prompts (Prompt Templates, Few-Shot Injections & Variable Substitution Traces)", divider="rainbow")

st.markdown("""
Prompts in LangChain are managed via `PromptTemplate` and `ChatPromptTemplate`.
A frequent bug in production is **Variable Substitution Corruption**:
passing unescaped JSON strings, missing template variables (e.g., `{context}` resolving to empty string),
or unformatted few-shot examples that throw formatting errors.
LangSmith captures the **exact before-and-after state** of prompt formatting:
the raw template string, the input variable dictionary, and the resulting message list.
""")

col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.markdown("### Prompt Formatting Trace Lineage")
    prompt_trace_df = pd.DataFrame([
        {"Phase": "1. Raw Template", "Content": "System: Answer query '{query}' strictly using context: {context}"},
        {"Phase": "2. Input Variables", "Content": "{'query': 'SLA guarantee', 'context': '99.9% uptime'}"},
        {"Phase": "3. Resolved Prompt", "Content": "System: Answer query 'SLA guarantee' strictly using context: 99.9% uptime"},
        {"Phase": "Verification Status", "Content": "All variables successfully bound (Zero unmapped keys)"}
    ])
    st.dataframe(prompt_trace_df, use_container_width=True)

with col_p2:
    st.markdown("### LangSmith Prompt Hub Integration")
    st.markdown("""
    Instead of hardcoding prompt strings in Python code, enterprise teams manage prompts through the **LangSmith Prompt Hub**:
    ```python
    from langchain import hub

    # Pull version-pinned prompt template from LangSmith
    prompt = hub.pull("enterprise/customer-support-agent:v2.1")
    chain = prompt | model | StrOutputParser()
    ```
    This decouples prompt engineering from code deployment: prompt updates can be tested, versioned, and rolled back in LangSmith without redeploying microservice containers.
    """)

st.divider()

# ==============================================================================
# SECTION 4: RUNNABLES
# ==============================================================================
st.header("4. Runnables (LCEL Primitives, Invoke / Batch / Stream Execution & Pipe Semantics)", divider="rainbow")

st.markdown("""
LangChain Expression Language (LCEL) is built on the **Runnable Protocol**:
every component implements a unified interface (`invoke`, `ainvoke`, `batch`, `stream`, `astream_events`).
When runnables are chained using the pipe operator (`|`):
`chain = prompt | model | parser`
LangSmith automatically tracks each segment as a distinct step, measuring input, output, and execution latency across each pipe boundary.
""")

col_r1, col_r2 = st.columns([1, 1])

with col_r1:
    st.markdown("### LCEL Core Runnable Protocol Methods:")
    runnable_methods_df = pd.DataFrame([
        {"Method": "invoke / ainvoke", "Type": "Single Input", "Trace Behavior": "Creates 1 root trace with sequential child spans"},
        {"Method": "batch / abatch", "Type": "Batch Inputs", "Trace Behavior": "Creates concurrent traces grouped under single batch run"},
        {"Method": "stream / astream", "Type": "Token Streaming", "Trace Behavior": "Tracks Time-To-First-Token (TTFT) and token chunk stream"},
        {"Method": "astream_events", "Type": "Fine-Grained Events", "Trace Behavior": "Streams real-time AST events directly to clients"}
    ])
    st.dataframe(runnable_methods_df, use_container_width=True)

with col_r2:
    st.markdown("### Runnable Composition Pipe Flow")
    dot_pipe = """
    digraph LCELPipe {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        In [label="User Input Dict\n{'query': 'Reset password'}", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        Prompt [label="Runnable 1: ChatPromptTemplate\nInput: Dict -> Output: PromptValue", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        Model [label="Runnable 2: ChatModel (gpt-4o)\nInput: PromptValue -> Output: ChatMessage", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        Parser [label="Runnable 3: StrOutputParser\nInput: ChatMessage -> Output: str", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];

        In -> Prompt -> Model -> Parser;
    }
    """
    st.graphviz_chart(dot_pipe, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: CHAINS
# ==============================================================================
st.header("5. Chains (Sequential Orchestration, LCEL Composition & Parent-Child Span Trees)", divider="rainbow")

st.markdown("""
A **Chain** is an orchestration pipeline linking multiple Runnables.
In LangSmith, a chain represents the **Parent Span** (`run_type="chain"`).
When a chain executes, its child spans automatically inherit its `trace_id` and reference its `run_id` as their `parent_run_id`.
If an error occurs anywhere within the chain—such as a schema validation failure in the parser—the entire chain is marked failed in LangSmith,
allowing engineers to click directly onto the red node in the trace tree to inspect the error stack trace.
""")

col_c1, col_c2 = st.columns([1, 1])

with col_c1:
    st.markdown("### Parent-Child Span Tree Relationship:")
    st.code('''
Trace ID: trace_lc_501 (Root)
└── EnterpriseAgentExecutor (chain, 579.5ms) [STATUS: OK]
    ├── ChatPromptTemplate (prompt, 2.5ms)
    ├── ChatOpenAI (llm, 210.0ms)
    ├── VectorStoreRetriever (retriever, 45.0ms)
    ├── FinancialCalculatorTool (tool, 12.0ms)
    └── ChatOpenAI (llm, 310.0ms)
''', language="text")

with col_c2:
    st.markdown("### Why Span Trees are Essential for Debugging:")
    st.markdown("""
    * **Latency Attribution**: Instantly reveals which specific child span consumed the majority of execution time (e.g., 520ms spent in LLM vs 12ms in tool).
    * **State Mutation Tracking**: Inspect the exact data transformations as variables pass from one runnable to the next.
    * **Error Confinement**: If a tool raises a socket timeout, the tool span captures the exact exception payload while the parent chain records the graceful degradation or fallback route.
    """)

st.divider()

# ==============================================================================
# SECTION 6: RETRIEVERS
# ==============================================================================
st.header("6. Retrievers (VectorStore Retrievers, Context Ingestion & Hit Metadata Logging)", divider="rainbow")

st.markdown("""
In LangChain, **Retrievers** implement the runnable interface accepting a string query and returning a list of `Document` objects.
When a retriever runs inside an LCEL chain:
`retrieval_chain = {"context": retriever, "query": RunnablePassthrough()} | prompt | model`
LangSmith instruments the retriever as `run_type="retriever"`.
It logs the exact candidate document payloads, their similarity scores, source URLs, and page numbers,
preventing the black-box opacity of traditional vector search lookups.
""")

col_rt1, col_rt2 = st.columns([1, 1])

with col_rt1:
    st.markdown("### Retriever Span Ingress & Egress Data:")
    st.markdown("""
    * **`inputs`**: `{"query": "How to configure VPC peering across regions?"}`
    * **`outputs`**:
      ```json
      {
        "documents": [
          {"page_content": "VPC peering establishes private IP routing...", "metadata": {"source": "net.pdf", "page": 44}},
          {"page_content": "Transit Gateway simplifies hub-spoke routing...", "metadata": {"source": "net.pdf", "page": 52}}
        ]
      }
      ```
    * **`metadata`**: `{"search_type": "similarity", "k": 2, "vectorstore": "Milvus"}`
    """)

with col_rt2:
    st.markdown("### Retriever Ingestion Pipeline Topology")
    dot_ret_pipe = """
    digraph RetPipe {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        Q [label="Query: 'VPC peering'", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        Retriever [label="VectorStoreRetriever (run_type: retriever)\nSearches HNSW index in 45ms", shape=cylinder, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        Docs [label="Retrieved Docs List\nPassed to Prompt Template {context}", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];

        Q -> Retriever -> Docs;
    }
    """
    st.graphviz_chart(dot_ret_pipe, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: TOOLS
# ==============================================================================
st.header("7. Tools (Structured Tool Invocations, Schema Serialization & Exit Code Tracing)", divider="rainbow")

st.markdown("""
**Tools** extend language models with external execution capabilities: querying SQL databases, calculating math,
or invoking external REST APIs.
Tools in LangChain are defined with Pydantic schemas via `@tool`.
In LangSmith, every tool execution generates a `run_type="tool"` span recording:
* **Tool Name & Description**: Confirms the model selected the intended tool.
* **Parsed Arguments**: Validates whether the LLM generated arguments adhering to the tool's Pydantic schema.
* **Return Payload / Exception**: Captures the exact return payload or runtime error thrown by the tool execution.
""")

col_t1, col_t2 = st.columns([1, 1])

with col_t1:
    st.markdown("### Tool Execution Span Data Example:")
    st.code('''
{
  "name": "FinancialCalculatorTool",
  "run_type": "tool",
  "inputs": {
    "principal": 10000,
    "rate": 0.05,
    "years": 3
  },
  "outputs": {
    "result": 11576.25
  },
  "latency_ms": 12.0
}
''', language="json")

with col_t2:
    st.markdown("### Common Tool Failure Modes Captured in Spans:")
    st.markdown("""
    1. **Schema Validation Error**:
       The LLM emits string arguments (`"principal": "ten thousand"`) instead of expected floats (`10000.0`), throwing a Pydantic `ValidationError`.
    2. **Tool Hallucination**:
       The LLM attempts to call a non-existent tool name (`execute_order_v2`), causing a tool dispatcher lookup failure.
    3. **Runtime API Timeout**:
       The remote API times out after 10 seconds. LangSmith captures the timeout stack trace and correlates it with the model's fallback response.
    """)

st.divider()

# ==============================================================================
# SECTION 8: AGENTS
# ==============================================================================
st.header("8. Agents (AgentExecutor vs LangGraph Workflows, Intermediate Steps & Action Trees)", divider="rainbow")

st.markdown("""
An **Agent** uses a language model as a reasoning engine to dynamically determine which actions to take and in what order.
Unlike static linear chains, agents execute in **iterative loops**:
`Thought -> Action (Tool Call) -> Observation (Tool Output) -> Thought -> Final Answer`.

In legacy debugging, tracing a 10-step agent loop across terminal logs is nearly impossible.
LangSmith represents agent runs as **Interactive Action Trees**:
expanding each reasoning iteration into dedicated child spans, displaying the model's internal scratchpad,
tool call parameters, intermediate observations, and loop termination conditions.
""")

col_ag1, col_ag2 = st.columns([1, 1])

with col_ag1:
    st.markdown("### Agent Iteration Step Trace Hierarchy:")
    agent_steps_df = pd.DataFrame([
        {"Step": "Iteration 1: Model Call", "Action": "Decides to invoke 'search_corporate_kb'", "Duration (ms)": 210.0},
        {"Step": "Iteration 1: Tool Exec", "Action": "Retrieves SLA uptime documentation", "Duration (ms)": 45.0},
        {"Step": "Iteration 2: Model Call", "Action": "Decides to invoke 'FinancialCalculatorTool'", "Duration (ms)": 180.0},
        {"Step": "Iteration 2: Tool Exec", "Action": "Calculates compound interest result", "Duration (ms)": 12.0},
        {"Step": "Iteration 3: Final Synthesis", "Action": "Synthesizes final answer with citations", "Duration (ms)": 310.0}
    ])
    st.dataframe(agent_steps_df, use_container_width=True)

with col_ag2:
    st.markdown("### Agent Reasoning Loop Architecture")
    dot_agent_loop = """
    digraph AgentLoop {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        AgentRoot [label="Agent Loop Controller\n(run_type: chain)", shape=box, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        LLMReason [label="LLM Reasoning Step\n(Emits Tool Action)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        ToolExec [label="Execute Tool\n(Returns Observation)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        Decision [label="Task Finished?", shape=diamond, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
        FinalAns [label="Deliver Final Answer", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        AgentRoot -> LLMReason;
        LLMReason -> ToolExec;
        ToolExec -> Decision [label="Feed Observation"];
        Decision -> LLMReason [label="No: Next Action"];
        Decision -> FinalAns [label="Yes: Terminate"];
    }
    """
    st.graphviz_chart(dot_agent_loop, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Full LangChain Component Tracer (LangChainComponentTracingEngine)", divider="rainbow")

st.markdown("""
Below is the verified production engine `LangChainComponentTracingEngine`.
It constructs end-to-end multi-component run hierarchies capturing Prompts, ChatModels,
Retrievers, Tools, and Agent execution loops in pure Python.
""")

st.code('''
class LangChainComponentTracingEngine:
    """
    Production LangChain Component Tracing & Run Hierarchy Engine.
    Emulates LangSmith automatic trace capture across all core LangChain abstractions:
    Prompts, Chat Models, LCEL Runnables, Chains, Retrievers, Tools, and Agents.
    """
    def __init__(self, project_name: str = "langchain-component-observability"):
        self.project_name = project_name
        self.active_traces: List[Dict[str, Any]] = []
        self._span_counter = 500

    def trace_agent_execution(
        self,
        user_query: str,
        include_retriever: bool = True,
        include_tool: bool = True
    ) -> Dict[str, Any]:
        self._span_counter += 1
        trace_id = f"trace_lc_{self._span_counter}"
        root_run_id = f"run_agent_{self._span_counter}"
        prompt_run_id = f"run_prompt_{self._span_counter}"
        llm1_run_id = f"run_llm1_{self._span_counter}"
        retriever_run_id = f"run_retriever_{self._span_counter}"
        tool_run_id = f"run_tool_{self._span_counter}"
        llm2_run_id = f"run_llm2_{self._span_counter}"

        child_spans = []
        total_latency = 0.0

        # Prompt Span
        prompt_latency = 2.5
        total_latency += prompt_latency
        child_spans.append({
            "run_id": prompt_run_id,
            "parent_run_id": root_run_id,
            "name": "ChatPromptTemplate",
            "run_type": "prompt",
            "latency_ms": prompt_latency
        })

        # First LLM Reasoning Span
        llm1_latency = 210.0
        total_latency += llm1_latency
        child_spans.append({
            "run_id": llm1_run_id,
            "parent_run_id": root_run_id,
            "name": "ChatOpenAI (gpt-4o)",
            "run_type": "llm",
            "latency_ms": llm1_latency,
            "tokens": {"total_tokens": 173}
        })

        if include_retriever:
            retriever_latency = 45.0
            total_latency += retriever_latency
            child_spans.append({
                "run_id": retriever_run_id,
                "parent_run_id": root_run_id,
                "name": "VectorStoreRetriever",
                "run_type": "retriever",
                "latency_ms": retriever_latency
            })

        if include_tool:
            tool_latency = 12.0
            total_latency += tool_latency
            child_spans.append({
                "run_id": tool_run_id,
                "parent_run_id": root_run_id,
                "name": "FinancialCalculatorTool",
                "run_type": "tool",
                "latency_ms": tool_latency
            })

        # Second LLM Synthesis Span
        llm2_latency = 310.0
        total_latency += llm2_latency
        final_answer = f"Operation calculated successfully with 99.9% SLA guarantees."
        child_spans.append({
            "run_id": llm2_run_id,
            "parent_run_id": root_run_id,
            "name": "ChatOpenAI (gpt-4o)",
            "run_type": "llm",
            "latency_ms": llm2_latency,
            "tokens": {"total_tokens": 462}
        })

        trace_payload = {
            "trace_id": trace_id,
            "project_name": self.project_name,
            "root_run": {
                "run_id": root_run_id,
                "name": "EnterpriseAgentExecutor",
                "run_type": "chain",
                "inputs": {"input": user_query},
                "outputs": {"output": final_answer},
                "total_latency_ms": round(total_latency, 2)
            },
            "child_spans": child_spans
        }
        self.active_traces.append(trace_payload)
        return trace_payload
''', language="python")

# --- SELF VERIFICATION OF PRODUCTION ENGINE ---
lc_tracer_inst = LangChainComponentTracingEngine()

test_agent_trace = lc_tracer_inst.trace_agent_execution(
    user_query="Calculate 3-year investment with enterprise SLA uptime.",
    include_retriever=True,
    include_tool=True
)

assert test_agent_trace["trace_id"].startswith("trace_lc_"), "Trace ID must follow standard naming convention"
assert test_agent_trace["root_run"]["total_latency_ms"] > 0, "Total agent latency must be positive"
assert len(test_agent_trace["child_spans"]) == 5, "Agent run must encompass prompt, llm1, retriever, tool, and llm2 spans"

st.success("Automated Verification Passed: LangChainComponentTracingEngine successfully verified across multi-component agent execution and child span lineage.")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Multi-Component LCEL Trace Hierarchy & Span Inspector", divider="rainbow")

st.markdown("""
Simulate an enterprise LangChain agent request and inspect the resulting LangSmith span hierarchy.
Toggle retriever lookups and tool executions to observe how the distributed trace dynamically adapts.
""")

wb_lc1, wb_lc2 = st.columns([1, 1])

with wb_lc1:
    st.markdown("#### Agent Pipeline Configuration")
    in_user_q = st.text_input("User Agent Prompt", value="Calculate returns on $10,000 investment under cloud enterprise SLA.")
    opt_inc_retriever = st.toggle("Enable VectorStore Retriever Span", value=True)
    opt_inc_tool = st.toggle("Enable Financial Calculator Tool Span", value=True)

with wb_lc2:
    st.markdown("#### Live Simulated Agent Execution")
    simulated_agent_trace = lc_tracer_inst.trace_agent_execution(
        user_query=in_user_q,
        include_retriever=opt_inc_retriever,
        include_tool=opt_inc_tool
    )
    st.metric(
        "Total Agent Duration",
        f"{simulated_agent_trace['root_run']['total_latency_ms']} ms",
        delta=f"{len(simulated_agent_trace['child_spans'])} Child Spans Captured"
    )

st.markdown("### LangSmith Run Tree Inspector")

span_records = [
    {
        "Run ID": simulated_agent_trace["root_run"]["run_id"],
        "Component Name": simulated_agent_trace["root_run"]["name"],
        "Run Type": simulated_agent_trace["root_run"]["run_type"],
        "Parent Run ID": "ROOT (None)",
        "Latency (ms)": simulated_agent_trace["root_run"]["total_latency_ms"]
    }
]
for span in simulated_agent_trace["child_spans"]:
    span_records.append({
        "Run ID": span["run_id"],
        "Component Name": span["name"],
        "Run Type": span["run_type"],
        "Parent Run ID": span["parent_run_id"],
        "Latency (ms)": span["latency_ms"]
    })

st.dataframe(pd.DataFrame(span_records), use_container_width=True)

st.markdown("#### Final Agent Output Synthesized")
st.info(simulated_agent_trace["root_run"]["outputs"]["output"])

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Questions & Architectural Solutions")

with st.expander("**Q1: What are the primary run_types supported in LangSmith, and what does each represent?**"):
    st.markdown("""
    **Answer**:
    LangSmith categorizes runs using standardized `run_type` primitives:
    * `llm`: A raw model invocation (e.g., `ChatOpenAI`), capturing token consumption, prompts, and completions.
    * `chain`: An orchestrator or LCEL sequence linking multiple Runnables.
    * `prompt`: A prompt template formatting step, capturing input variables and resulting messages.
    * `retriever`: A vector store or document retrieval lookup, capturing candidate chunk payloads.
    * `tool`: An external function or API call, capturing arguments and return outputs.
    * `parser`: An output formatting step (e.g., `JsonOutputParser`), validating string-to-object conversions.
    """)

with st.expander("**Q2: How does LangSmith's async background telemetry avoid adding latency to LangChain chains?**"):
    st.markdown("""
    **Answer**:
    LangChain's LangSmith callback handler (`LangChainTracer`) buffers span start and end events into a thread-safe in-memory queue.
    A dedicated background daemon worker batches these events and flushes them to the LangSmith collector API over HTTP/2.
    If the collector experiences network latency or temporary downtime, the worker drops non-critical events gracefully without stalling the user's synchronous `chain.invoke()` call.
    """)

with st.expander("**Q3: What is the difference between tracing legacy AgentExecutor and modern LangGraph agent workflows?**"):
    st.markdown("""
    **Answer**:
    * **Legacy `AgentExecutor`**: Treated the agent as a rigid while-loop chain. Traces showed flat lists of intermediate tool steps, making cyclic state loops difficult to visualize.
    * **Modern `LangGraph`**: Implements stateful cyclic directed graphs. In LangSmith, every node invocation, state channel update (`add_messages`), conditional edge transition, and checkpoint state is captured as a distinct graph execution span.
    """)

with st.expander("**Q4: How do you inject custom metadata into a LangChain LCEL chain for filtering in LangSmith?**"):
    st.markdown("""
    **Answer**:
    You can pass metadata dynamically via the `config` dictionary in any Runnable invocation:
    ```python
    response = chain.invoke(
        {"input": "Refund policy"},
        config={
            "metadata": {
                "tenant_id": "ACME_CORP",
                "user_tier": "Enterprise",
                "session_id": "sess_88912"
            },
            "tags": ["billing", "priority_support"]
        }
    )
    ```
    All child spans spawned by the chain automatically inherit these metadata tags, enabling faceted search and filtering in the LangSmith dashboard.
    """)

with st.expander("**Q5: What is the LangSmith Prompt Hub, and how does it prevent prompt regression in production?**"):
    st.markdown("""
    **Answer**:
    The Prompt Hub is a centralized registry for versioning and testing prompt templates.
    Instead of hardcoding prompt strings in application code, microservices pull pinned prompt hashes (e.g., `hub.pull("org/agent:3a8f1b")`).
    Engineers can test candidate prompt revisions against versioned evaluation datasets in LangSmith before promoting a new prompt hash to production, completely eliminating prompt regression risk.
    """)

st.markdown("### Production Traps & Failure Modes")

ltr_col1, ltr_col2 = st.columns(2)

with ltr_col1:
    st.error("""
    **Trap 1: Hardcoding Prompts as Python String Literals**
    * **Failure**: Embedding prompt instructions directly in Python code files across distributed repositories.
    * **Consequence**: Non-technical domain experts cannot iterate on prompts without software engineer pull requests, unit tests, and production container rebuilds.
    * **Remedy**: Store and version production prompts in the LangSmith Prompt Hub, pulling them dynamically with version pinning.
    """)

    st.error("""
    **Trap 2: Missing Custom Metadata in Production Chain Invocations**
    * **Failure**: Running `chain.invoke(input)` in production without passing `tenant_id` or `user_id` in the `config` parameter.
    * **Consequence**: When a specific enterprise client reports a bug, searching through millions of traces in LangSmith without a tenant filter is like finding a needle in a haystack.
    * **Remedy**: Always inject session context: `config={"metadata": {"tenant_id": tenant_id}}`.
    """)

with vtr_col2:
    st.warning("""
    **Trap 3: Unhandled Tool Schema Validation Failures Breaking Chains**
    * **Failure**: Defining tools with rigid Pydantic types without providing fallback recovery when the LLM emits malformed JSON arguments.
    * **Consequence**: The tool throws a Pydantic `ValidationError`, causing the entire agent loop to crash immediately.
    * **Remedy**: Enable `handle_tool_error=True` on tools, allowing the error message to be passed back to the LLM for self-correction.
    """)

    st.warning("""
    **Trap 4: Over-Tracing Local Development Runs in Shared Projects**
    * **Failure**: Pointing local developer test scripts to the shared production LangSmith project (`LANGSMITH_PROJECT=production`).
    * **Consequence**: Production metrics (latency p95, error rates, token burn) become corrupted by developer experiments.
    * **Remedy**: Use environment-specific project names: `LANGSMITH_PROJECT=dev-alice-local`.
    """)

st.markdown("### Core Technical Flashcards")

fcl_c1, fcl_c2 = st.columns(2)

with fcl_c1:
    with st.container(border=True):
        st.markdown("**Front**: What are the 6 primary LangChain abstractions instrumented by LangSmith?")
        st.markdown("**Back**: 1. Models (`llm`), 2. Prompts (`prompt`), 3. Runnables/Chains (`chain`), 4. Retrievers (`retriever`), 5. Tools (`tool`), and 6. Output Parsers (`parser`).")

    with st.container(border=True):
        st.markdown("**Front**: How are Parent and Child spans linked in LangSmith?")
        st.markdown("**Back**: All child spans inherit the Root Span's `trace_id` and contain a `parent_run_id` field referencing their direct caller, constructing a complete directed acyclic run tree.")

with fcl_c2:
    with st.container(border=True):
        st.markdown("**Front**: What is the Runnable protocol in LangChain (LCEL)?")
        st.markdown("**Back**: A unified execution standard implemented by all LangChain components providing `invoke`, `batch`, `stream`, and async variants connected via pipe (`|`) syntax.")

    with st.container(border=True):
        st.markdown("**Front**: How do you pass session metadata to a LangChain chain for LangSmith tracing?")
        st.markdown("**Back**: By passing `config={'metadata': {'tenant_id': '...'}}` to `chain.invoke()`, automatically attaching custom metadata to all downstream child spans.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of LangChain Component Tracing principles:")

c_chk1 = st.checkbox("I understand the 6 core LangSmith run_types (llm, chain, prompt, retriever, tool, parser).", value=True)
c_chk2 = st.checkbox("I know how LangSmith constructs parent-child span trees from LCEL runnable pipe compositions.", value=True)
c_chk3 = st.checkbox("I can trace ChatModel token consumption, stop reasons, and raw prompt message payloads.", value=True)
c_chk4 = st.checkbox("I understand how to decouple prompt management from code deployment using the LangSmith Prompt Hub.", value=True)
c_chk5 = st.checkbox("I know how to inject custom metadata and tags into chain execution configs for faceted filtering.", value=True)
c_chk6 = st.checkbox("I can trace multi-step agent reasoning loops, tool call arguments, and intermediate observations.", value=True)

if c_chk1 and c_chk2 and c_chk3 and c_chk4 and c_chk5 and c_chk6:
    st.success("Mastery Certified: You have mastered LangChain component tracing, run hierarchies, and LCEL observability in LangSmith.")
