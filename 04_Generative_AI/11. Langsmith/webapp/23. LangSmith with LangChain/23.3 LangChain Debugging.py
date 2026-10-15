import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="23.3 LangChain Debugging - LangSmith Master Curriculum",
    page_icon="🛠️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class LangChainDebuggingEngine:
    """
    Production LangChain Triage & Exception Diagnostic Engine.
    Emulates LangSmith distributed trace capture and automated root cause analysis
    across five critical failure domains:
    1. Chain Failures (Context overflow, Recursion limits, Pipe disconnects)
    2. Tool Failures (Pydantic schema validation, Timeouts, Execution crashes)
    3. Retrieval Failures (Empty candidates, Deserialization, Vector store connection drops)
    4. Prompt Failures (Missing variables, Formatting type mismatches, Invalid escaping)
    5. Output Parser Failures (Malformed JSON, Schema non-compliance, Truncated payloads)
    """
    def __init__(self, project_name: str = "langchain-exception-triage"):
        self.project_name = project_name
        self.simulated_runs: List[Dict[str, Any]] = []
        self._run_counter = 800

    def simulate_failure_trace(
        self,
        failure_type: str,
        query: str,
        auto_remediation: bool = False
    ) -> Dict[str, Any]:
        """
        Simulates an execution trace with a targeted failure mode and generates
        diagnostic metadata and LangSmith span trees.
        """
        self._run_counter += 1
        trace_id = f"trace_err_{self._run_counter}"
        root_run_id = f"run_root_{self._run_counter}"

        span_tree = []
        root_status = "error" if not auto_remediation else "remediated"
        root_error_msg = ""
        root_cause_analysis = {}

        if failure_type == "chain_failure":
            root_error_msg = "RecursionError: Maximum runnable recursion depth of 25 exceeded in cyclic branch routing"
            span_tree = [
                {
                    "run_id": f"span_seq_1_{self._run_counter}",
                    "name": "RunnableBranch_DynamicRouter",
                    "type": "chain",
                    "status": "success",
                    "latency_ms": 14.2,
                    "error": None
                },
                {
                    "run_id": f"span_seq_2_{self._run_counter}",
                    "name": "FallbackLoop_Branch",
                    "type": "chain",
                    "status": "error" if not auto_remediation else "success",
                    "latency_ms": 320.5,
                    "error": root_error_msg if not auto_remediation else None
                }
            ]
            root_cause_analysis = {
                "category": "Chain Execution",
                "failure_mode": "Cyclic Recursion / Stack Exhaustion",
                "trigger_component": "RunnableBranch",
                "detected_in_langsmith": "Run tree shows 25 identical child spans without terminating condition",
                "remediation": "Configure max_depth or migration to LangGraph StateGraph with explicit cycle limits"
            }

        elif failure_type == "tool_failure":
            root_error_msg = "ValidationError: Field 'symbol' required; received {'ticker_symbol': 'GOOG'} in ToolCall"
            span_tree = [
                {
                    "run_id": f"span_tool_1_{self._run_counter}",
                    "name": "StockQuoteTool",
                    "type": "tool",
                    "status": "error" if not auto_remediation else "success",
                    "latency_ms": 42.1,
                    "error": root_error_msg if not auto_remediation else None
                }
            ]
            root_cause_analysis = {
                "category": "Tool Calling",
                "failure_mode": "Pydantic Schema Argument Mismatch",
                "trigger_component": "StockQuoteTool.invoke()",
                "detected_in_langsmith": "Input payload keys deviated from JSON Schema bound via bind_tools()",
                "remediation": "Add aliases to Pydantic schema or implement handle_tool_error fallback callable"
            }

        elif failure_type == "retrieval_failure":
            root_error_msg = "RetrievalError: VectorStore returned 0 document chunks for similarity threshold >= 0.85"
            span_tree = [
                {
                    "run_id": f"span_ret_1_{self._run_counter}",
                    "name": "EmbeddingInference",
                    "type": "embedding",
                    "status": "success",
                    "latency_ms": 65.4,
                    "error": None
                },
                {
                    "run_id": f"span_ret_2_{self._run_counter}",
                    "name": "VectorStoreRetriever",
                    "type": "retriever",
                    "status": "error" if not auto_remediation else "success",
                    "latency_ms": 120.8,
                    "error": root_error_msg if not auto_remediation else None
                }
            ]
            root_cause_analysis = {
                "category": "Retrieval Pipeline",
                "failure_mode": "Zero Document Recall / Over-Constrained Filter",
                "trigger_component": "VectorStoreRetriever",
                "detected_in_langsmith": "Output payload contains empty Document list []; downstream prompt lacks grounding context",
                "remediation": "Implement hybrid dense-sparse search or relax similarity cut-off with fallback default doc"
            }

        elif failure_type == "prompt_failure":
            root_error_msg = "KeyError: Missing variable 'chat_history' in ChatPromptTemplate input dictionary"
            span_tree = [
                {
                    "run_id": f"span_pmt_1_{self._run_counter}",
                    "name": "ChatPromptTemplate_Formatter",
                    "type": "prompt",
                    "status": "error" if not auto_remediation else "success",
                    "latency_ms": 2.1,
                    "error": root_error_msg if not auto_remediation else None
                }
            ]
            root_cause_analysis = {
                "category": "Prompt Construction",
                "failure_mode": "Unbound Template Variable Injection",
                "trigger_component": "ChatPromptTemplate",
                "detected_in_langsmith": "Prompt step crashed before generating ChatPromptValue; parent run aborted",
                "remediation": "Inject optional defaults via .partial(chat_history=[]) or validate inputs upstream"
            }

        else: # output_parser_failure
            root_error_msg = "OutputParserException: Failed to parse LLM generation to JSON. Generation was truncated."
            span_tree = [
                {
                    "run_id": f"span_llm_1_{self._run_counter}",
                    "name": "ChatModel_Generation",
                    "type": "llm",
                    "status": "success",
                    "latency_ms": 780.0,
                    "error": None
                },
                {
                    "run_id": f"span_psr_1_{self._run_counter}",
                    "name": "JsonOutputParser",
                    "type": "parser",
                    "status": "error" if not auto_remediation else "success",
                    "latency_ms": 5.4,
                    "error": root_error_msg if not auto_remediation else None
                }
            ]
            root_cause_analysis = {
                "category": "Structured Parsing",
                "failure_mode": "Malformed JSON Output / Truncated Stream",
                "trigger_component": "JsonOutputParser",
                "detected_in_langsmith": "LLM output span contains invalid closing brackets; parser child span threw exception",
                "remediation": "Deploy OutputFixingParser with secondary LLM call or switch to model-native structured outputs"
            }

        trace_record = {
            "trace_id": trace_id,
            "root_run_id": root_run_id,
            "query": query,
            "failure_type": failure_type,
            "status": root_status,
            "error_message": root_error_msg if not auto_remediation else "None (Auto-Remediated)",
            "spans": span_tree,
            "triage": root_cause_analysis,
            "total_latency_ms": sum(s["latency_ms"] for s in span_tree)
        }

        self.simulated_runs.append(trace_record)
        return trace_record


# Instantiate engine
langchain_debugger = LangChainDebuggingEngine()

# --- HEADER SECTION ---
st.title("🛠️ Chapter 23.3: LangChain Debugging & Exception Triage")
st.markdown("""
Welcome to the production debugging handbook for **LangChain applications** monitored through **LangSmith**.
In distributed GenAI architectures, failures rarely surface as simple line-by-line tracebacks.
Instead, they propagate silently through pipeline stages—causing truncated prompts, empty candidate sets,
hallucinated tool schemas, and unparsed JSON blocks.
This module equips you with systematic observability methods to inspect, isolate, and remediate every failure mode.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: LangChain Error Hierarchy & Exception Propagation Diagnostics", divider="rainbow")

st.markdown("""
Production LangChain applications comprise interconnected runnables: prompt templates, retrievers, chat models,
tools, and output parsers. When an exception occurs at any tier of the DAG, LangSmith captures the exact execution
context, preserving input states, intermediate outputs, child stack traces, and timing profiles.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LangChainErrorTriage {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, nodesep=0.6, ranksep=0.7];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_input {
            label="1. Ingestion & Request Boundary";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            req [label="Incoming User Request\n{query, session_id, config}", fillcolor="#1E293B", color="#38BDF8"];
            cfg [label="RunnableConfig Injection\nTags: ['prod', 'v2'], Metadata: {tenant_id}", fillcolor="#1E293B", color="#38BDF8"];
        }

        subgraph cluster_prompt_tier {
            label="2. Prompt Formatting Tier";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            pmt [label="ChatPromptTemplate\nInput Key Validation", fillcolor="#1E293B", color="#38BDF8"];
            pmt_err [label="Prompt Failure Point\nKeyError / Missing Variables", fillcolor="#7F1D1D", color="#EF4444"];
        }

        subgraph cluster_retrieval_tier {
            label="3. Vector Retrieval Tier";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            ret [label="Retriever Pipeline\nBi-Encoder + Vector DB Query", fillcolor="#1E293B", color="#38BDF8"];
            ret_err [label="Retrieval Failure Point\nEmpty Candidates / Similarity Cut-off", fillcolor="#7F1D1D", color="#EF4444"];
        }

        subgraph cluster_model_tier {
            label="4. Chat Model & Tool Invocation";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            llm [label="ChatModel Inference\nFunction Call / bind_tools", fillcolor="#1E293B", color="#38BDF8"];
            tool [label="ToolNode Execution\nPydantic Input Validation", fillcolor="#1E293B", color="#38BDF8"];
            tool_err [label="Tool Failure Point\nSchema Mismatch / Timeout", fillcolor="#7F1D1D", color="#EF4444"];
        }

        subgraph cluster_parser_tier {
            label="5. Structured Output Parsing";
            style=dashed;
            color="#334155";
            fontcolor="#38BDF8";
            psr [label="Json / Pydantic OutputParser\nOutput Conformance Check", fillcolor="#1E293B", color="#38BDF8"];
            psr_err [label="Parser Failure Point\nMalformed JSON / Truncated Tokens", fillcolor="#7F1D1D", color="#EF4444"];
        }

        subgraph cluster_langsmith {
            label="6. LangSmith Observability Platform";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            ls_trace [label="LangSmith Trace Hierarchy\nRun Tree + Captured Error Payload", fillcolor="#0F172A", color="#10B981"];
            ls_alert [label="Automated Triage & Sentry Bridge\nTags: error=True, retry_count=N", fillcolor="#0F172A", color="#F59E0B"];
        }

        req -> cfg -> pmt;
        pmt -> pmt_err [label="Unbound Key", color="#EF4444"];
        pmt -> ret [label="Valid Prompt"];
        ret -> ret_err [label="Top-K = 0", color="#EF4444"];
        ret -> llm [label="Grounded Context"];
        llm -> tool [label="ToolCall Issued"];
        tool -> tool_err [label="Validation Fail", color="#EF4444"];
        tool -> psr [label="Tool Output"];
        psr -> psr_err [label="Non-JSON", color="#EF4444"];

        pmt_err -> ls_trace [label="Trace Span Crash", color="#EF4444"];
        ret_err -> ls_trace [label="Empty Recall", color="#EF4444"];
        tool_err -> ls_trace [label="Stack Trace", color="#EF4444"];
        psr_err -> ls_trace [label="Parser Traceback", color="#EF4444"];
        ls_trace -> ls_alert [label="Trigger Alert"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Five Core Failure Vectors in LangChain Observability
The following table outlines the diagnostic signatures and root-cause indicators for each of the five failure domains:
""")

err_overview_df = pd.DataFrame([
    {
        "Failure Domain": "1. Chain Failures",
        "Common Exception": "RecursionError, ContextOverflowError",
        "LangSmith Diagnostic Signature": "Run tree contains excessive child spans or parent span terminates with 500 error",
        "Production Triage Action": "Check recursion_limit or inspect infinite cycle in RunnableBranch / StateGraph"
    },
    {
        "Failure Domain": "2. Tool Failures",
        "Common Exception": "pydantic.ValidationError, ToolException",
        "LangSmith Diagnostic Signature": "Tool run marked 'error'; inputs contain unexpected argument keys or bad datatypes",
        "Production Triage Action": "Inspect LLM tool_call arguments against target Pydantic model; add schema aliases"
    },
    {
        "Failure Domain": "3. Retrieval Failures",
        "Common Exception": "RetrievalError, DeserializationError",
        "LangSmith Diagnostic Signature": "Retriever run returns empty list []; downstream prompt lacks context; score thresholding failed",
        "Production Triage Action": "Verify vector store connectivity, index schema consistency, and score cutoff thresholds"
    },
    {
        "Failure Domain": "4. Prompt Failures",
        "Common Exception": "KeyError, ValueError, FormatterSyntaxError",
        "LangSmith Diagnostic Signature": "Prompt child span fails before ChatPromptValue is created; error shows missing input keys",
        "Production Triage Action": "Audit dictionary payload passed to invoke(); verify .partial() default bindings"
    },
    {
        "Failure Domain": "5. Parser Failures",
        "Common Exception": "OutputParserException, JSONDecodeError",
        "LangSmith Diagnostic Signature": "LLM run succeeds, but parser child span fails with invalid JSON token at character index",
        "Production Triage Action": "Inspect raw generation span; configure OutputFixingParser or switch to structured outputs"
    }
])
st.dataframe(err_overview_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: CHAIN FAILURES
# ==============================================================================
st.header("2. Chain Failures (Unhandled Exceptions, Broken Pipes & Context Window Overflows)", divider="rainbow")

st.markdown("""
Chain failures occur when the composition pipeline breaks during execution.
Unlike single-function calls, chains link together disparate runnables. If an unhandled exception occurs in an intermediate
pipe, the entire runnable chain halts unless explicit fallback runnables (`with_fallbacks`) are attached.
""")

st.markdown("### Top Root Causes of Chain Failures")
st.markdown("""
1. **Unbounded Dynamic Branching**: Using `RunnableBranch` where routing functions cycle between steps without reaching an end node, resulting in `RecursionError`.
2. **Context Window Token Overflow**: Passing accumulated message histories and document contexts that exceed the model's token capacity (`maximum_context_length_exceeded`).
3. **Broken Pipe / Unhandled Subprocess Crashes**: Custom `RunnableLambda` steps terminating prematurely or encountering non-picklable object serializations.
""")

st.code('''
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Production Resilience Pattern: Using with_fallbacks to survive chain failures
primary_chain = complex_lcel_pipeline
fallback_chain = simplified_rule_based_pipeline

resilient_pipeline = primary_chain.with_fallbacks([fallback_chain])

# Trace shows primary_chain marked 'error' and fallback_chain marked 'success',
# allowing the end-to-end user request to complete gracefully while alerting oncall.
result = resilient_pipeline.invoke({"query": "analyze quarter earnings"})
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production environments, legacy nested chains (`SequentialChain`, `LLMChain`) have been universally replaced by
**LangChain LCEL Runnables** and **LangGraph StateGraphs**.
When debugging chain crashes in LangSmith:
* Trace trees display distinct status badges (`COMPLETED`, `FAILED`, `CANCELLED`).
* Error spans display the full traceback and the exact input dictionary that caused the failure.
* By attaching `.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)`, transient network failures are automatically retried before raising fatal trace exceptions.
""")

st.divider()

# ==============================================================================
# SECTION 3: TOOL FAILURES
# ==============================================================================
st.header("3. Tool Failures (Schema Validation Errors, Missing Arguments & Tool Timeouts)", divider="rainbow")

st.markdown("""
Tool calling is the primary mechanism for autonomous agents to interact with external environments.
However, because LLMs generate text representations of function calls, tool invocation is vulnerable to:
* **Hallucinated Arguments**: The model invents parameters not present in the tool's Pydantic schema.
* **Type Incompatibilities**: The model supplies strings instead of integers or floats.
* **Downstream API Timeouts**: The external tool endpoint hangs, causing the agent thread to block indefinitely.
""")

st.markdown("### Diagnostic Architecture: Tool Execution Spans")

with st.expander("🔍 View Tool Exception Flow Diagram", expanded=True):
    tool_flow_dot = """
    digraph ToolFlow {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        llm_out [label="LLM Emits ToolCall\nname: 'weather_api'\nargs: {'city': 'NYC', 'units': 'kelvin'}", fillcolor="#1E293B", color="#38BDF8"];
        pydantic [label="Pydantic Schema Validation\nExpected: {'location': str, 'unit': Literal['c','f']}", fillcolor="#1E293B", color="#38BDF8"];
        val_error [label="ValidationError Raised\nInvalid keys: 'city', 'units'", fillcolor="#7F1D1D", color="#EF4444"];
        ls_tool_span [label="LangSmith Tool Run\nStatus: ERROR\nInputs: args dictionary\nTraceback recorded", fillcolor="#0F172A", color="#EF4444"];
        agent_heal [label="Self-Healing Loop\nError feedback passed as ToolMessage\nLLM regenerates conforming call", fillcolor="#1E293B", color="#10B981"];

        llm_out -> pydantic;
        pydantic -> val_error [label="Schema Mismatch", color="#EF4444"];
        val_error -> ls_tool_span;
        ls_tool_span -> agent_heal [label="handle_tool_error=True", color="#10B981"];
    }
    """
    st.graphviz_chart(tool_flow_dot, use_container_width=True)

st.code('''
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class StockSchema(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g., AAPL")
    period: str = Field(default="1y", description="Analysis time window")

# Handling tool errors gracefully in LangChain v0.3+
@tool(args_schema=StockSchema, handle_tool_error=True)
def get_stock_data(ticker: str, period: str) -> str:
    """Fetch financial summary for a given equity ticker."""
    if ticker not in ["AAPL", "GOOG", "MSFT"]:
        raise ValueError(f"Ticker symbol '{ticker}' not found in active database.")
    return f"Financial data for {ticker} over {period}: Normalized return +14.2%"
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern tool architectures, tool errors are not fatal crashes. Instead, systems use **self-correcting agent loops**:
When a tool throws an error, the exception message is formatted into a `ToolMessage` and appended to the conversational state.
The LLM inspects the error message on its next iteration and issues a corrected function call.
In LangSmith, this appears as an initial failed tool span followed immediately by a successful retry span, providing complete transparency into the agent's self-healing capabilities.
""")

st.divider()

# ==============================================================================
# SECTION 4: RETRIEVAL FAILURES
# ==============================================================================
st.header("4. Retrieval Failures (Empty Candidate Sets, Connection Resets & Document Deserialization Errors)", divider="rainbow")

st.markdown("""
Retrievers bridge external vector stores and conversational prompts.
When retrieval fails, models either hallucinate or state that they do not possess sufficient information.
LangSmith enables engineers to inspect the exact query vector, search parameters, retrieved document payloads,
and vector database responses.
""")

st.markdown("### Common Retrieval Failure Modes")
st.markdown("""
* **Zero Recall (Empty Candidate Set)**: Overly aggressive similarity cut-offs (e.g., threshold >= 0.90) or restrictive metadata filters (`tenant_id == 'alpha'`) return zero documents.
* **Vector Store Connectivity Drop**: Database connection timeouts, client pool exhaustion, or unauthorized API tokens.
* **Corrupted Metadata / Deserialization Error**: Vector database records containing non-string or malformed metadata fields that crash downstream prompt formatters.
""")

st.code('''
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

def safe_retriever_wrapper(retriever, query: str) -> List[Document]:
    """
    Wraps standard retriever to log warning metadata in LangSmith
    and provide fallback knowledge when recall is 0.
    """
    docs = retriever.invoke(query)
    if not docs:
        # Fallback doc prevents downstream empty-context crashes
        return [Document(page_content="No specific internal documents matched your query.", metadata={"source": "fallback"})]
    return docs
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production RAG systems instrument retrievers with **fallback pipelines** and **diagnostic metadata**:
* Every retriever span in LangSmith logs the candidate count (`len(docs)`), query string, and similarity scores.
* If dense retrieval returns fewer than K candidates, hybrid search automatically falls back to sparse BM25 indexing.
* Observability monitors trigger Slack/PagerDuty alerts when the empty recall rate exceeds 2% over a 15-minute sliding window.
""")

st.divider()

# ==============================================================================
# SECTION 5: PROMPT FAILURES
# ==============================================================================
st.header("5. Prompt Failures (Unbound Template Keys, Formatting Type Errors & Escaping Bugs)", divider="rainbow")

st.markdown("""
Prompt formatting errors occur before an LLM is even invoked.
Because prompt templates rely on variable substitution (e.g., `{query}`, `{context}`, `{chat_history}`),
any discrepancy between the expected dictionary keys and the runtime dictionary causes a runtime exception.
""")

st.markdown("### Prompt Template Failure Signatures")
st.markdown("""
1. **KeyError (Unbound Variable)**: The template specifies `{user_role}` but the input dictionary only provides `{query}`.
2. **Type Error in Value Injection**: Passing complex Python objects (e.g., SQLAlchemy models or numpy arrays) that cannot be coerced to strings.
3. **Escaping Collisions**: Prompts containing JSON code blocks or curly braces (`{}`) that confuse Python's standard `f-string` or `format` engine.
""")

st.code('''
from langchain_core.prompts import ChatPromptTemplate

# Best Practice: Explicit input validation and partial variable binding
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enterprise AI assistant for {organization}."),
    ("user", "{query}")
])

# Use .partial() to bind static variables early, preventing runtime KeyError
bound_prompt = prompt.partial(organization="Acme Corporation")

# Safe invocation
formatted = bound_prompt.invoke({"query": "What are our compliance guidelines?"})
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production LangSmith workflows:
* The **LangSmith Prompt Hub** manages versioned prompt templates with explicit input schema definitions.
* If an unbound variable is detected during unit testing or CI/CD regression runs, LangSmith flags the experiment run with a `PromptSchemaError`.
* Prompts containing code or JSON templates use double-escaped braces (`{{` and `}}`) to prevent format engine syntax errors.
""")

st.divider()

# ==============================================================================
# SECTION 6: OUTPUT PARSER FAILURES
# ==============================================================================
st.header("6. Output Parser Failures (Malformed JSON, Schema Non-Compliance & Pydantic Validation Errors)", divider="rainbow")

st.markdown("""
Output parsers transform unstructured text generated by LLMs into structured objects (JSON, Pydantic models, Lists).
Output parsing is notoriously fragile because LLMs can output conversational preambles ("Sure, here is your JSON:"),
trailing markdown fences (` ```json `), or truncate output mid-generation due to token limits.
""")

st.markdown("### Parser Exception Diagnostic Anatomy")

with st.expander("🔍 View Output Parsing Exception & Repair Flow", expanded=True):
    parser_flow_dot = """
    digraph ParserFlow {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        raw_llm [label="LLM Raw Output\n```json\n{'user_id': 101, 'status': 'active'\n(Truncated without closing brace)", fillcolor="#1E293B", color="#38BDF8"];
        json_parser [label="JsonOutputParser\njson.loads() Attempt", fillcolor="#1E293B", color="#38BDF8"];
        parse_err [label="OutputParserException\nJSONDecodeError: Expecting ',' delimiter", fillcolor="#7F1D1D", color="#EF4444"];
        ls_span [label="LangSmith Parser Span\nStatus: ERROR\nRecords raw LLM text + parse error", fillcolor="#0F172A", color="#EF4444"];
        fixer [label="OutputFixingParser / Retry\nPrompts LLM to repair invalid JSON string", fillcolor="#1E293B", color="#10B981"];
        repaired [label="Valid Pydantic Object\nSuccessfully emitted to client", fillcolor="#1E293B", color="#10B981"];

        raw_llm -> json_parser;
        json_parser -> parse_err [label="Invalid Syntax", color="#EF4444"];
        parse_err -> ls_span;
        ls_span -> fixer [label="Auto-Fix Triggered", color="#10B981"];
        fixer -> repaired;
    }
    """
    st.graphviz_chart(parser_flow_dot, use_container_width=True)

st.code('''
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class AnalysisReport(BaseModel):
    summary: str = Field(description="Summary of findings")
    risk_score: float = Field(description="Risk assessment from 0.0 to 1.0")

base_parser = JsonOutputParser(pydantic_object=AnalysisReport)

# Production Remediation: Wrap parser with OutputFixingParser
# If the initial generation is malformed, OutputFixingParser invokes a fast model
# passing the malformed text and schema instructions to repair the JSON.
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern production stacks have largely phased out string-based regex parsers in favor of **native model structured outputs**:
* Methods like `.with_structured_output(AnalysisReport)` instruct OpenAI/Anthropic/Gemini APIs to enforce JSON schema constraints at the token-generation level via constrained decoding.
* In LangSmith, structured outputs eliminate parsing child spans entirely, turning prompt-to-model into a single deterministic schema-validated execution.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: LangChain Triage & Exception Diagnostic Engine (LangChainDebuggingEngine)", divider="rainbow")

st.markdown("""
Below is the reference production diagnostic implementation.
The `LangChainDebuggingEngine` models trace capture, span categorization, error injection,
and automated remediation synthesis for enterprise LangSmith workflows.
""")

st.code('''
class LangChainDebuggingEngine:
    """
    Production LangChain Triage & Exception Diagnostic Engine.
    Emulates LangSmith distributed trace capture and automated root cause analysis.
    """
    def __init__(self, project_name: str = "langchain-exception-triage"):
        self.project_name = project_name
        self.simulated_runs: List[Dict[str, Any]] = []
        self._run_counter = 800

    def simulate_failure_trace(
        self,
        failure_type: str,
        query: str,
        auto_remediation: bool = False
    ) -> Dict[str, Any]:
        """
        Simulates an execution trace with a targeted failure mode and generates
        diagnostic metadata and LangSmith span trees.
        """
        # Emulates span creation, error attachment, and root-cause analysis
        ...
''', language="python")

st.info("The engine is fully instantiated in module scope and powers the interactive debugging workbench below.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: LangChain Failure Mode Injector & Trace Debugger", divider="rainbow")

st.markdown("""
Experiment with different failure modes across the LangChain execution lifecycle.
Inject simulated errors into the pipeline, inspect how LangSmith captures the failure trace tree,
and observe the automated root-cause analysis and remediation strategy.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.subheader("Trace Injection Controls")
    selected_failure = st.selectbox(
        "Select Failure Domain to Inject:",
        options=[
            "chain_failure",
            "tool_failure",
            "retrieval_failure",
            "prompt_failure",
            "output_parser_failure"
        ],
        format_func=lambda x: {
            "chain_failure": "1. Chain Failure (Cyclic Recursion)",
            "tool_failure": "2. Tool Failure (Pydantic Schema Mismatch)",
            "retrieval_failure": "3. Retrieval Failure (Zero Candidate Recall)",
            "prompt_failure": "4. Prompt Failure (Unbound Template KeyError)",
            "output_parser_failure": "5. Output Parser Failure (Malformed JSON)"
        }[x]
    )

    test_query = st.text_input(
        "Test User Query:",
        value="Summarize quarterly balance sheet and calculate net EBITDA margin"
    )

    enable_remediation = st.toggle(
        "Enable Self-Healing Remediation",
        value=False,
        help="Applies fallbacks, OutputFixingParser, or schema adapters to resolve the error"
    )

    run_sim_btn = st.button("🚀 Execute Trace with Injected Fault", use_container_width=True)

with wb_col2:
    st.subheader("LangSmith Trace & Root Cause Inspector")

    if run_sim_btn or len(langchain_debugger.simulated_runs) == 0:
        result_trace = langchain_debugger.simulate_failure_trace(
            failure_type=selected_failure,
            query=test_query,
            auto_remediation=enable_remediation
        )
    else:
        result_trace = langchain_debugger.simulated_runs[-1]

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    status_label = "REMEDIATED (SUCCESS)" if result_trace["status"] == "remediated" else "FAILED (ERROR)"
    m1.metric("Run Status", status_label)
    m2.metric("Total Latency", f"{result_trace['total_latency_ms']:.1f} ms")
    m3.metric("Spans Captured", len(result_trace["spans"]))
    m4.metric("Trace ID", result_trace["trace_id"][-12:])

    # Span Tree Display
    st.markdown("#### Execution Span Hierarchy")
    spans_df = pd.DataFrame(result_trace["spans"])
    st.dataframe(spans_df, use_container_width=True)

    # Diagnostic Card
    triage = result_trace["triage"]
    st.markdown("#### Automated Root-Cause Triage Report")

    if result_trace["status"] == "error":
        st.error(f"**Fatal Error Caught**: {result_trace['error_message']}")
    else:
        st.success(f"**Execution Healed**: {result_trace['error_message']}")

    with st.container(border=True):
        st.markdown(f"**Failure Category**: {triage['category']}")
        st.markdown(f"**Failure Mode**: {triage['failure_mode']}")
        st.markdown(f"**Triggering Component**: `{triage['trigger_component']}`")
        st.markdown(f"**LangSmith Trace Evidence**: {triage['detected_in_langsmith']}")
        st.markdown(f"**Recommended Production Fix**: {triage['remediation']}")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How do you debug silent failures in LangChain RAG pipelines using LangSmith?"):
    st.markdown("""
    **Answer:**
    Silent failures in RAG occur when retrieval returns irrelevant or empty documents, yet the chat model generates an hallucinated or non-committal answer without throwing an exception.
    In LangSmith:
    1. Inspect the `Retriever` child span to audit input queries, vector similarity scores, and the exact chunks returned.
    2. Check the candidate count (`len(docs)`). If it equals 0, the threshold or metadata filter was overly restrictive.
    3. Inspect the `ChatPromptTemplate` output span to verify how context was injected into the prompt.
    4. Attach automated LangSmith evaluators (e.g., Ragas Faithfulness, Context Recall) to catch hallucinated outputs programmatically.
    """)

with st.expander("Q2: What is the architectural difference between with_fallbacks and handle_tool_error?"):
    st.markdown("""
    **Answer:**
    * **`with_fallbacks([fallback_runnable])`**: Operates at the **Runnable/Chain** level. If the primary runnable raises an unhandled exception (e.g., model rate limit, API 500 error), the entire runnable fails over to an alternative execution pipeline.
    * **`handle_tool_error=True`**: Operates specifically at the **Tool** level. Instead of crashing the execution thread when a tool function fails, it catches the exception and returns the error message as a `ToolMessage` string to the model, giving the agent a chance to self-correct its arguments.
    """)

with st.expander("Q3: How does OutputFixingParser work under the hood, and what are its trade-offs?"):
    st.markdown("""
    **Answer:**
    `OutputFixingParser` catches `OutputParserException` from an underlying parser (e.g., `JsonOutputParser`). When an error occurs, it constructs a new prompt containing:
    1. The target schema instructions.
    2. The original malformed LLM completion.
    3. The specific parser error message.
    It passes these to an LLM (typically a fast model like `gpt-4o-mini`) to synthesize valid JSON.
    * **Trade-off**: Increases system resilience at the expense of an extra LLM call (additional latency and token cost).
    """)

with st.expander("Q4: Why does LangChain throw a KeyError during prompt formatting, and how do you prevent it in production?"):
    st.markdown("""
    **Answer:**
    `ChatPromptTemplate` parses `{variable}` strings as required template variables. If the input dictionary passed to `.invoke()` is missing any variable, Python raises a `KeyError`.
    **Prevention strategies**:
    1. Use `.partial(var="default")` to bind known or static variables early.
    2. Wrap inputs in a Pydantic model to guarantee schema validation before invoking the prompt.
    3. If literal curly braces are needed (e.g., code snippets or JSON examples in system prompts), double-escape them (`{{` and `}}`).
    """)

with st.expander("Q5: How do you diagnose infinite loops or recursion limits in cyclic chains using LangSmith?"):
    st.markdown("""
    **Answer:**
    Cyclic chains or multi-agent loops can get trapped in repetitive reasoning cycles, hitting LangChain's `recursion_limit` (default 25).
    In LangSmith:
    1. Inspect the trace run tree: an infinite loop appears as an endlessly repeating sequence of identical child spans.
    2. Analyze the input/output state diff between iterations to see why the termination condition failed to trigger.
    3. Migrate from unbounded `RunnableBranch` chains to LangGraph `StateGraph`, which provides explicit step bounds and cycle telemetry.
    """)

st.markdown("### Production Traps & Failure Modes")

deb_col1, deb_col2 = st.columns(2)

with deb_col1:
    st.error("""
    **Trap 1: Silent Swallowing of Tool Exceptions**
    * **Failure**: Writing custom tool functions wrapped in bare `try...except: return ""` blocks without logging.
    * **Consequence**: The tool returns an empty string without raising an exception, masking the failure in LangSmith and causing the LLM to hallucinate tool results.
    * **Remedy**: Always raise a `ToolException` with descriptive error details so LangSmith marks the tool span as failed.
    """)

    st.error("""
    **Trap 2: Context Token Overflow in Appended Histories**
    * **Failure**: Appending raw chat history strings indefinitely without trimming or windowing.
    * **Consequence**: The prompt span crashes when passing 130,000 tokens to a model with a 128,000 token context window.
    * **Remedy**: Use `trim_messages` or LangGraph memory checkpointers with message token budgets before invoking prompts.
    """)

with deb_col2:
    st.warning("""
    **Trap 3: Unhandled Regex Failures in Custom Parsers**
    * **Failure**: Relying on fragile regular expressions to parse JSON blocks out of raw LLM outputs.
    * **Consequence**: Models frequently format markdown fences with slight whitespace deviations, causing regex parsers to return `None`.
    * **Remedy**: Use `JsonOutputParser` or native structured outputs (`with_structured_output`) with JSON Schema constraints.
    """)

    st.warning("""
    **Trap 4: Missing Request Metadata in Error Traces**
    * **Failure**: Invoking chains without passing `tenant_id`, `user_id`, or `session_id` in `RunnableConfig`.
    * **Consequence**: When an incident occurs in production, on-call engineers cannot isolate which user or customer experienced the failure.
    * **Remedy**: Always inject session and tenant metadata via `config={"metadata": {...}}` at the invocation boundary.
    """)

st.markdown("### Core Technical Flashcards")

fcd_c1, fcd_c2 = st.columns(2)

with fcd_c1:
    with st.container(border=True):
        st.markdown("**Front**: What happens to a LangSmith trace when a child span raises an unhandled exception?")
        st.markdown("**Back**: The child span is marked with status `ERROR`, records the error string and traceback, and propagates the error up the tree, marking parent spans as `FAILED` unless caught by fallbacks.")

    with fcd_c2:
        st.markdown("**Front**: What is the purpose of `with_fallbacks()` in LangChain LCEL?")
        st.markdown("**Back**: It provides declarative disaster recovery by executing secondary fallback runnables if the primary runnable throws an exception (e.g., rate limit, 500 server error).")

with fcd_c1:
    with st.container(border=True):
        st.markdown("**Front**: How does `handle_tool_error=True` facilitate agent self-correction?")
        st.markdown("**Back**: It converts tool exceptions into text strings returned to the LLM as `ToolMessage` outputs, enabling the agent to read the error and correct its arguments on the next step.")

    with fcd_c2:
        st.markdown("**Front**: What is the most robust modern replacement for `JsonOutputParser`?")
        st.markdown("**Back**: `.with_structured_output(Schema)`, which leverages provider-level JSON schema constraints and grammar-guided decoding to prevent syntax parsing errors entirely.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of LangChain debugging and exception triage:")

ch_k1 = st.checkbox("I can isolate and diagnose the 5 core LangChain failure domains in LangSmith trace trees.", value=True)
ch_k2 = st.checkbox("I understand how unhandled exceptions propagate up the run hierarchy and how to attach `with_fallbacks`.", value=True)
ch_k3 = st.checkbox("I can implement `handle_tool_error` and Pydantic schema validation for self-healing agent loops.", value=True)
ch_k4 = st.checkbox("I know how to identify empty retrieval candidate sets and score cut-off issues in vector retriever spans.", value=True)
ch_k5 = st.checkbox("I know how to prevent prompt `KeyError` crashes using `.partial()` bindings and double-escaped braces.", value=True)
ch_k6 = st.checkbox("I understand how `OutputFixingParser` and native structured outputs eliminate output parsing crashes.", value=True)

if ch_k1 and ch_k2 and ch_k3 and ch_k4 and ch_k5 and ch_k6:
    st.success("Mastery Certified: You have mastered LangChain debugging, distributed error triage, and automated remediation patterns in LangSmith.")
