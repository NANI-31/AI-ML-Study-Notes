import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="5.1 Runnable Fundamentals - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("5.1 Runnable Fundamentals: The Unified Protocol of Modern LangChain")
st.caption("Chapter 5: LangChain Expression Language and Runnables | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In legacy software architectures, connecting prompt templates, foundation models, document retrievers, 
and output parsers required brittle, non-standardized wrapper classes (`LLMChain`, `TransformChain`, `SequentialChain`). 
Each component had unique execution methods, obscure state dictionaries, and inconsistent asynchronous support.

**LangChain v0.3+** solves this architectural fragmentation through a single, elegant abstraction: **The `Runnable`**. 
Every component in the ecosystem—whether a prompt, a chat model, a vector store retriever, a tool, or an output parser—implements 
the exact same `Runnable` interface.

In this lesson, you will master:
- **The Core Runnable Contract**: Why a single unified protocol powers the entire LangChain v0.3+ ecosystem.
- **The Core Primitive Hierarchy**: `RunnableSequence`, `RunnableParallel`, `RunnableLambda`, and `RunnablePassthrough`.
- **Branching & Concurrency**: Executing parallel analytical paths concurrently with zero threading boilerplate.
- **Data Flow Plumbing**: Passing unmodified inputs down-pipeline using `RunnablePassthrough.assign()`.
- **Runtime Configuration Context**: Injecting thread-local metadata, callbacks, and tags via `RunnableConfig`.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockRunnableNode:
    """Represents an atomic Runnable node in module scope."""
    def __init__(self, name: str, node_type: str, latency_ms: int):
        self.name = name
        self.node_type = node_type
        self.latency_ms = latency_ms

class MockRunnableDAGSimulator:
    """Simulates execution of a multi-branch Runnable graph in module scope."""
    def __init__(self):
        self.nodes = [
            MockRunnableNode("Input Passthrough", "RunnablePassthrough", 2),
            MockRunnableNode("Security Auditor", "RunnableLambda", 85),
            MockRunnableNode("Cost Estimator", "RunnableLambda", 45),
            MockRunnableNode("Compliance Verifier", "RunnableLambda", 120),
            MockRunnableNode("Synthesis Model", "ChatModel", 320)
        ]

    def execute_dag(self, input_payload: str, branches: List[str]) -> Dict[str, Any]:
        """Simulates parallel branch execution followed by synthesis."""
        branch_latencies = {
            "Security Auditor": 85,
            "Cost Estimator": 45,
            "Compliance Verifier": 120
        }
        
        # Parallel execution latency is max(latencies)
        active_latencies = [branch_latencies[b] for b in branches if b in branch_latencies]
        parallel_time = max(active_latencies) if active_latencies else 0
        total_time = 2 + parallel_time + 320  # passthrough + max(branches) + synthesis model

        branch_results = {}
        if "Security Auditor" in branches:
            branch_results["security"] = "PASSED: Zero open ingress ports detected."
        if "Cost Estimator" in branches:
            branch_results["cost"] = "ESTIMATE: 14.50 USD per month based on t4g.medium instance."
        if "Compliance Verifier" in branches:
            branch_results["compliance"] = "VERIFIED: Compliant with SOC2-Type-II controls."

        synthesis_text = (
            f"Consolidated Analysis for '{input_payload[:30]}...': "
            f"Evaluated {len(branches)} parallel branches concurrently in {parallel_time} ms. "
            f"All pipeline criteria met under unified Runnable protocol."
        )

        return {
            "branches_evaluated": len(branches),
            "parallel_latency_ms": parallel_time,
            "total_latency_ms": total_time,
            "branch_outputs": branch_results,
            "final_synthesis": synthesis_text
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Core Runnable Abstraction Hierarchy & Data Flow", divider="rainbow")

st.markdown("""
The diagram below illustrates the structural hierarchy of LangChain's `Runnable` primitives 
and how data flows through sequential and parallel pipelines.
""")

dot_blueprint = """
digraph RunnableHierarchy {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_base {
        label="1. The Universal Runnable Base Contract (langchain_core.runnables)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_runnable [label="Runnable[InputType, OutputType]\nCore Methods: .invoke(), .ainvoke(), .stream(), .astream(), .batch(), .abatch()", fillcolor="#1E293B", fontcolor="#38BDF8"];
    }

    subgraph cluster_primitives {
        label="2. Foundational Composition Primitives";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_seq [label="RunnableSequence (A | B | C)\nSequential data flow pipeline\nOutput(A) -> Input(B)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_par [label="RunnableParallel ({'k1': A, 'k2': B})\nConcurrent multi-branch execution\nInput -> [Branch 1, Branch 2] -> Dict", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_lambda [label="RunnableLambda(func)\nWraps any Python function into a Runnable", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_pass [label="RunnablePassthrough\nForwards unmodified input\n.assign() adds new computed keys", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_runnable -> node_seq [dir=back, arrowtail=empty, color="#38BDF8"];
        node_runnable -> node_par [dir=back, arrowtail=empty, color="#38BDF8"];
        node_runnable -> node_lambda [dir=back, arrowtail=empty, color="#38BDF8"];
        node_runnable -> node_pass [dir=back, arrowtail=empty, color="#38BDF8"];
    }

    subgraph cluster_concrete {
        label="3. Concrete Component Implementations";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_prompt [label="ChatPromptTemplate\nInput: dict -> Output: PromptValue", fillcolor="#1E293B", fontcolor="#34D399"];
        node_model [label="BaseChatModel (GPT-4o, Claude)\nInput: PromptValue -> Output: AIMessage", fillcolor="#1E293B", fontcolor="#34D399"];
        node_parser [label="StrOutputParser / PydanticParser\nInput: AIMessage -> Output: str / DTO", fillcolor="#1E293B", fontcolor="#34D399"];

        node_seq -> node_prompt [style="dashed", color="#64748B"];
        node_prompt -> node_model [label="piped into (|)", color="#38BDF8"];
        node_model -> node_parser [label="piped into (|)", color="#38BDF8"];
    }
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: Because every single component in modern LangChain inherits from `Runnable`, 
> they all share the exact same execution lifecycle. You can chain a prompt, a custom Python lambda, 
> an external REST API, an LLM, and a database writer using a single uniform syntax with zero glue code.
""")

# --- SECTION 2: WHAT IS A RUNNABLE? ---
st.header("2. What is a Runnable? The Unified Contract Across LLMs, Prompts & Parsers", divider="rainbow")

st.markdown("""
In computer science, a **protocol** or **interface** defines a standard contract for interaction. 

A **`Runnable`** is a generic unit of computational work that transforms an input of type `Input` 
into an output of type `Output`.

Every `Runnable` guarantees six fundamental execution methods:
1. **`invoke(input, config=None)`**: Synchronous single input to output.
2. **`ainvoke(input, config=None)`**: Asynchronous single input to output.
3. **`stream(input, config=None)`**: Synchronous progressive token/chunk generator.
4. **`astream(input, config=None)`**: Asynchronous progressive token/chunk generator.
5. **`batch(inputs, config=None)`**: Concurrent processing of a list of inputs.
6. **`abatch(inputs, config=None)`**: Asynchronous concurrent processing of inputs.
""")

# --- SECTION 3: RUNNABLE INTERFACE ---
st.header("3. Runnable Interface: Input/Output Schemas, Polymorphic Signatures & Type Constraints", divider="rainbow")

st.markdown("""
Runnables are strongly-typed Python objects. You can programmatically inspect the expected input schema 
and output schema of any Runnable using **`get_input_schema()`** and **`get_output_schema()`**, which return 
auto-generated **Pydantic models**.
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Translate {text} into {language}.")
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Composing a standard LCEL sequence
chain = prompt | model | parser

# Programmatic Schema Inspection:
print(chain.get_input_schema().model_json_schema())
# Shows JSON Schema requiring 'text' (string) and 'language' (string)

print(chain.get_output_schema().model_json_schema())
# Shows JSON Schema producing a string output
''', language="python")

# --- SECTION 4: RUNNABLE SEQUENCE ---
st.header("4. RunnableSequence: Linear Sequential Composition & The Pipe Protocol", divider="rainbow")

st.markdown("""
A **`RunnableSequence`** chains multiple Runnables sequentially: the output of component i is passed 
directly as the input to component i+1.

In LangChain, a `RunnableSequence` is created either:
1. Automatically using the **pipe operator (`|`)**: `chain = prompt | model | parser`
2. Explicitly using the class constructor: `RunnableSequence(first=prompt, middle=[model], last=parser)`
""")

st.code('''from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Summarize: {topic}")
model = ChatOpenAI(model="gpt-4o", temperature=0.0)
parser = StrOutputParser()

# Equivalent approaches:
# Approach 1: Pipe Operator
chain_pipe = prompt | model | parser

# Approach 2: Explicit Class Constructor
chain_class = RunnableSequence(first=prompt, middle=[model], last=parser)

result = chain_pipe.invoke({"topic": "Distributed Consensus in Raft"})
print(result)
''', language="python")

# --- SECTION 5: RUNNABLE PARALLEL ---
st.header("5. RunnableParallel: Concurrent Multi-Branch Execution & Dictionary Syntax", divider="rainbow")

st.markdown("""
A **`RunnableParallel`** executes multiple Runnables concurrently on the exact same input, returning 
a single dictionary where keys correspond to each branch's output.

In LCEL, any Python dictionary containing Runnables is automatically coerced into a `RunnableParallel`!
""")

dot_parallel = """
digraph ParallelBranching {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_input [label="Input: {'topic': 'Quantum Computing'}", fillcolor="#1E293B", fontcolor="#F8FAFC"];

    subgraph cluster_branches {
        label="RunnableParallel Execution (Concurrent ThreadPool)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";

        node_b1 [label="Branch 'pros':\nprompt_pros | model", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_b2 [label="Branch 'cons':\nprompt_cons | model", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_b3 [label="Branch 'history':\nprompt_hist | model", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    node_output [label="Output Dictionary:\n{\n  'pros': '...', \n  'cons': '...', \n  'history': '...'\n}", fillcolor="#1E293B", fontcolor="#34D399"];

    node_input -> node_b1;
    node_input -> node_b2;
    node_input -> node_b3;

    node_b1 -> node_output;
    node_b2 -> node_output;
    node_b3 -> node_output;
}
"""

with st.expander("🔍 Contextual Diagram: RunnableParallel Concurrent Execution Flow", expanded=True):
    st.graphviz_chart(dot_parallel, use_container_width=True)

st.code('''from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o", temperature=0.2)
parser = StrOutputParser()

# Define two analytical branches
pros_chain = PromptTemplate.from_template("List 2 pros of {tech}") | model | parser
cons_chain = PromptTemplate.from_template("List 2 cons of {tech}") | model | parser

# Compose into RunnableParallel (executes concurrently via threadpool)
parallel_evaluator = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain
)

# Output is a single merged dictionary:
# {'pros': '1. Speed\n2. Scalability', 'cons': '1. Cost\n2. Complexity'}
results = parallel_evaluator.invoke({"tech": "Serverless Functions"})
''', language="python")

# --- SECTION 6: RUNNABLE LAMBDA ---
st.header("6. RunnableLambda: Wrapping Pure Python Functions into Runnables", divider="rainbow")

st.markdown("""
Not all transformations require an LLM. Applications frequently require data munging, regex parsing, 
filtering, or external REST calls. 

**`RunnableLambda`** converts any standard Python function or lambda into a first-class `Runnable` 
with full support for synchronous, asynchronous, and batch execution.
""")

st.code('''from langchain_core.runnables import RunnableLambda

def calculate_token_estimate(text: str) -> int:
    """Standard Python transformation."""
    return len(text.split()) * 2

# Wrap in RunnableLambda (or simply pass function into LCEL pipe)
token_counter = RunnableLambda(calculate_token_estimate)

# Fully conforms to Runnable protocol:
print(token_counter.invoke("Explain distributed vector indexing"))  # 8
print(token_counter.batch(["Hello world", "Artificial intelligence"])) # [4, 4]
''', language="python")

# --- SECTION 7: RUNNABLE PASSTHROUGH ---
st.header("7. RunnablePassthrough: Forwarding Unmodified Inputs & Dynamic Key Assignment", divider="rainbow")

st.markdown("""
In many chains, an input variable must be preserved for later stages while an intermediate step computes a new value.

**`RunnablePassthrough`** provides two critical functions:
1. **`RunnablePassthrough()`**: Forwards the input untouched to the next step.
2. **`RunnablePassthrough.assign(**kwargs)`**: Appends new computed fields to an existing dictionary input without overwriting the original keys.
""")

st.code('''from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def fetch_weather(inputs: dict) -> str:
    return "72F and Sunny in " + inputs["city"]

# RunnablePassthrough.assign takes existing dict and appends 'weather'
chain = (
    RunnablePassthrough.assign(weather=RunnableLambda(fetch_weather))
    | PromptTemplate.from_template("City: {city}\\nWeather: {weather}\\nSuggest activities:")
    | ChatOpenAI(model="gpt-4o")
)

# Input: {'city': 'Seattle'}
# After assign: {'city': 'Seattle', 'weather': '72F and Sunny in Seattle'}
''', language="python")

# --- SECTION 8: RUNNABLE CONFIGURATION ---
st.header("8. Runnable Configuration: RunnableConfig, Context Propagation & Thread Locals", divider="rainbow")

st.markdown("""
Every invocation method accepts an optional **`config`** argument (`RunnableConfig`). 
This dictionary propagates execution context down the entire call tree:
- **`tags`**: List of string tags for filtering in LangSmith.
- **`metadata`**: Dictionary of tracking metadata (e.g., `user_id`, `environment`).
- **`callbacks`**: Custom tracer or logging event handlers.
- **`max_concurrency`**: Threadpool limit for batch and parallel steps.
""")

st.code('''# Passing configuration during invocation:
config_payload = {
    "tags": ["production-v2", "customer-facing"],
    "metadata": {"user_id": "usr-8921", "tenant": "enterprise-acme"},
    "max_concurrency": 4
}

# response = chain.invoke({"query": "System status"}, config=config_payload)
''', language="python")

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Enterprise Multi-Branch Analytical Pipeline", divider="rainbow")

st.markdown("""
Below is an enterprise-grade analytical pipeline implementing `RunnablePassthrough.assign()`, 
`RunnableParallel`, `RunnableLambda`, and LCEL chaining.
""")

st.code('''from typing import Dict, Any
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def calculate_complexity_score(inputs: Dict[str, Any]) -> str:
    code = inputs["code_snippet"]
    lines = len(code.splitlines())
    if lines > 50:
        return "HIGH_COMPLEXITY"
    return "STANDARD_COMPLEXITY"

class CodeReviewPipeline:
    """Production multi-branch code auditor executing concurrent security and style reviews."""

    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
        self.parser = StrOutputParser()

        # Branch 1: Security Audit
        security_prompt = ChatPromptTemplate.from_template(
            "Audit this code for security vulnerabilities (SQL injection, XSS, RCE):\\n{code_snippet}"
        )
        self.security_branch = security_prompt | self.model | self.parser

        # Branch 2: Performance Audit
        perf_prompt = ChatPromptTemplate.from_template(
            "Audit this code for performance bottlenecks and algorithmic complexity:\\n{code_snippet}"
        )
        self.perf_branch = perf_prompt | self.model | self.parser

        # Final Synthesis Prompt
        synthesis_prompt = ChatPromptTemplate.from_template("""### CODE AUDIT SYNTHESIS ###
Complexity Rating: {complexity}

[SECURITY FINDINGS]
{security_report}

[PERFORMANCE FINDINGS]
{perf_report}

Provide final executive recommendation:""")

        # Assemble Master LCEL Pipeline
        self.pipeline = (
            RunnablePassthrough.assign(complexity=RunnableLambda(calculate_complexity_score))
            | RunnablePassthrough.assign(
                security_report=self.security_branch,
                perf_report=self.perf_branch
            )
            | synthesis_prompt
            | self.model
            | self.parser
        )

    def analyze(self, code: str) -> str:
        return self.pipeline.invoke({"code_snippet": code})
''', language="python")

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Live Runnable Graph Simulator & Parallel Execution Engine", divider="rainbow")

st.markdown("""
Simulate the concurrent execution of a `RunnableParallel` pipeline below. 
Select which analytical branches to activate, submit a mock workload, and inspect 
the concurrency latency savings and consolidated synthesis.
""")

sim_dag = MockRunnableDAGSimulator()

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    st.subheader("⚡ Active Parallel Branches")
    active_branches = []
    if st.checkbox("Branch 1: Security Auditor (85ms)", value=True):
        active_branches.append("Security Auditor")
    if st.checkbox("Branch 2: Cost Estimator (45ms)", value=True):
        active_branches.append("Cost Estimator")
    if st.checkbox("Branch 3: Compliance Verifier (120ms)", value=True):
        active_branches.append("Compliance Verifier")

with col_wb2:
    st.subheader("Input Workload")
    workload_input = st.text_area(
        "Infrastructure Spec",
        value="AWS EKS cluster with 3 t4g.medium worker nodes and encrypted EBS volumes."
    )

if st.button("🚀 Execute Runnable DAG", type="primary"):
    res = sim_dag.execute_dag(workload_input, active_branches)
    
    st.subheader("📊 Execution Diagnostics & Latency")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Branches", res["branches_evaluated"])
    m2.metric("Parallel Latency", f"{res['parallel_latency_ms']} ms")
    m3.metric("Total End-to-End Latency", f"{res['total_latency_ms']} ms")
    m4.metric("Concurrency Efficiency", f"~{int((sum([85, 45, 120]) - res['parallel_latency_ms']))} ms saved")

    st.write("### Branch Output Dictionary (`RunnableParallel` Output)")
    st.json(res["branch_outputs"])

    st.write("### Consolidated Synthesis Output")
    st.code(res["final_synthesis"], language="markdown")

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the core architectural purpose of the Runnable protocol in LangChain v0.3+?",
     "The Runnable protocol standardizes the interface for every component in the ecosystem. By enforcing standard methods (.invoke, .ainvoke, .stream, .batch), LangChain eliminated disparate legacy chain abstractions. This guarantees that all components natively support asynchronous I/O, streaming, batching, distributed tracing, and LCEL pipe composition."),
    ("How does RunnableParallel achieve concurrent execution in Python?",
     "RunnableParallel utilizes an internal ThreadPoolExecutor. When invoked with a dictionary of Runnables, each branch is dispatched concurrently onto worker threads, gathering results into a single merged dictionary. In async contexts (.ainvoke), it uses asyncio.gather() for non-blocking concurrency."),
    ("What is the difference between RunnablePassthrough() and RunnablePassthrough.assign()?",
     "RunnablePassthrough() simply returns its input unmodified without altering structure. RunnablePassthrough.assign(**kwargs) takes an existing dictionary input, computes the specified keys using downstream Runnables, and merges those new keys into the original dictionary without mutating or dropping existing fields."),
    ("Why should developers avoid putting blocking synchronous I/O inside RunnableLambda in async pipelines?",
     "RunnableLambda wraps Python functions. If a synchronous blocking function (e.g., requests.get()) is called inside a RunnableLambda and invoked via .ainvoke(), it will block the entire Python asyncio event loop unless wrapped with run_in_executor. In async workflows, use async def functions for RunnableLambda."),
    ("How does RunnableConfig propagate context down an LCEL execution tree?",
     "RunnableConfig is an optional dictionary passed to .invoke(input, config=...). LangChain automatically forwards this configuration across every child Runnable in the DAG, allowing developers to set global callbacks, session metadata, tags, and concurrency limits without passing parameters manually through each function signature.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Using Plain Dicts Instead of RunnablePassthrough.assign(). Overwriting inputs by passing a dictionary without `.assign()` drops original keys needed by downstream prompts, raising instant `KeyError` exceptions.")
    st.error("**Trap 2 (Breaking Error)**: Mutable State Leakage in RunnableLambda. Modifying external global variables inside a `RunnableLambda` causes race conditions under concurrent `.batch()` or multi-threaded execution. Keep lambdas purely functional.")
with trap_c2:
    st.warning("**Trap 3 (Latency Warning)**: Giant Parallel Fan-Out Exhausting Connections. Creating a `RunnableParallel` with 50 branches against a rate-limited API without setting `max_concurrency` triggers HTTP 429 errors. Bound concurrency via `config={'max_concurrency': 5}`.")
    st.warning("**Trap 4 (Memory Warning)**: Unbounded Batch Sizes in .batch(). Passing a list of 1,000 items to `.batch()` spawns hundreds of threads simultaneously, exhausting memory. Process large batches in manageable chunks.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Universal Execution Methods"):
        st.markdown("**.invoke(), .ainvoke(), .stream(), .astream(), .batch(), .abatch()**")
    with st.expander("🎴 Flashcard 2: RunnableParallel Coercion"):
        st.markdown("**Any Python dict of Runnables in LCEL is automatically coerced into a RunnableParallel.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Input Key Preservation"):
        st.markdown("**RunnablePassthrough.assign(**kwargs) appends new keys while preserving existing input dict.**")
    with st.expander("🎴 Flashcard 4: Python Function Wrapper"):
        st.markdown("**RunnableLambda(func) converts standard Python functions into first-class Runnables.**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can define the core Runnable protocol and its 6 execution methods.", value=True)
chk2 = st.checkbox("I know how to construct sequential pipelines using RunnableSequence and pipe (|).", value=True)
chk3 = st.checkbox("I understand how RunnableParallel executes multi-branch tasks concurrently.", value=True)
chk4 = st.checkbox("I know how to use RunnablePassthrough.assign() to preserve dictionary context.", value=True)
chk5 = st.checkbox("I understand how to wrap custom Python logic using RunnableLambda.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 5.1 Runnable Fundamentals. Ready for 5.2 LCEL!")
