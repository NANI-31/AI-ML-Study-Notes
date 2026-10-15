import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="6.1 Understanding Chains - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("6.1 Understanding Chains: Architecture, Evolution & Core Topologies")
st.caption("Chapter 6: Chains | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In Generative AI application engineering, a single foundation model invocation is rarely sufficient to solve 
an enterprise task. Real-world workflows require multi-step reasoning: retrieving domain context, formatting prompts, 
calling language models, validating structured payloads, branching conditionally, and synthesizing final answers.

A **Chain** is the architectural construct that links these discrete computational steps into a cohesive pipeline. 
While early LangChain versions relied on monolithic wrapper classes (`LLMChain`, `SequentialChain`, `TransformChain`), 
modern LangChain (v0.3+) implements chains as **declarative compositions of `Runnable` primitives** through LCEL.

In this lesson, you will master:
- **The Concept of a Chain**: From legacy class wrappers to modern functional `RunnableSequence` pipelines.
- **The Core Anatomy of a Chain**: Source inputs, prompt engineering, cognitive models, parsers, and sinks.
- **Sequential Topologies**: Passing state linearly and accumulating intermediate variables.
- **Parallel Topologies**: Executing independent analytical branches concurrently via `RunnableParallel`.
- **Conditional Topologies**: Routing execution paths dynamically based on runtime classification.
- **Hierarchical Nesting**: Encapsulating complete sub-chains as single, reusable nodes in larger workflows.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockChainTopologyNode:
    """Represents a node in a simulated chain topology in module scope."""
    def __init__(self, node_id: str, name: str, node_type: str, latency_ms: int):
        self.node_id = node_id
        self.name = name
        self.node_type = node_type
        self.latency_ms = latency_ms

class MockChainTopologySimulator:
    """Simulates sequential, parallel, conditional, and nested chain execution in module scope."""
    def __init__(self):
        self.topologies = {
            "Sequential": ["Input Dict", "ChatPromptTemplate", "BaseChatModel", "StrOutputParser"],
            "Parallel": ["Input Broadcast", "Security Branch (Parallel)", "Cost Branch (Parallel)", "Summary Synthesizer"],
            "Conditional": ["Input Query", "Intent Classifier", "Specialized Sub-Chain (Routed)", "Final Formatter"],
            "Nested": ["User Request", "Drafting Sub-Chain", "Verification Sub-Chain", "Approval Output"]
        }

    def simulate_execution(self, topology_name: str, input_text: str) -> Dict[str, Any]:
        stages = self.topologies.get(topology_name, self.topologies["Sequential"])
        
        simulated_latencies = {
            "Sequential": 420,
            "Parallel": 280,
            "Conditional": 340,
            "Nested": 560
        }
        
        simulated_outputs = {
            "Sequential": f"Sequential chain executed linearly: '{input_text[:30]}...' processed through Prompt -> Model -> Parser.",
            "Parallel": f"Parallel branches executed concurrently: Security audit passed (0 CVEs) and Cost model estimated at $0.04.",
            "Conditional": f"Conditional router directed '{input_text[:30]}...' to specialized tier. High-priority path executed.",
            "Nested": f"Nested pipeline executed: Sub-chain A drafted initial response; Sub-chain B audited and approved artifact."
        }

        return {
            "topology": topology_name,
            "stages_executed": stages,
            "total_latency_ms": simulated_latencies.get(topology_name, 350),
            "output_summary": simulated_outputs.get(topology_name, "Chain completed successfully.")
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Evolution of Chains (Legacy vs Modern LCEL)", divider="rainbow")

st.markdown("""
The diagram below contrasts the architectural evolution of Chains: from legacy monolithic class wrappers 
(`LLMChain`, `SequentialChain`) to modern declarative `RunnableSequence` and `RunnableParallel` DAGs.
""")

dot_blueprint = """
digraph ChainEvolution {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_legacy {
        label="1. Legacy Chains (v0.0.x - Deprecated Monolithic Wrappers)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#EF4444";
        fontsize=12;

        node_llmchain [label="LLMChain(llm=..., prompt=...)\nOpaque wrapper class; hidden intermediate state", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_seqchain [label="SimpleSequentialChain([chain1, chain2])\nBrittle string-only piping; no async/streaming", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        
        node_llmchain -> node_seqchain [color="#EF4444"];
    }

    subgraph cluster_modern {
        label="2. Modern LCEL Chains (v0.3+ Declarative Runnable DAGs)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#34D399";
        fontsize=12;

        node_prompt [label="ChatPromptTemplate\nInput: dict -> Output: PromptValue", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_model [label="BaseChatModel (GPT-4o, Claude)\nInput: PromptValue -> Output: AIMessage", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_parallel [label="RunnableParallel ({'k1': branch1, 'k2': branch2})\nConcurrent multi-branch execution", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_parser [label="StrOutputParser / PydanticOutputParser\nInput: AIMessage -> Output: DTO / str", fillcolor="#1E293B", fontcolor="#34D399"];

        node_prompt -> node_model [label="pipe (|)", color="#34D399"];
        node_model -> node_parallel [label="pipe (|)", color="#34D399"];
        node_parallel -> node_parser [label="pipe (|)", color="#34D399"];
    }

    // Connect evolution
    node_seqchain -> node_prompt [label="architectural shift\nto pure Runnables", color="#38BDF8", style="dashed"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: Modern chains are not distinct wrapper classes. A chain is simply 
> any composition of Runnables assembled with the pipe operator (`|`) or instantiated via 
> `RunnableSequence` and `RunnableParallel`.
""")

# --- SECTION 2: WHAT IS A CHAIN? ---
st.header("2. What is a Chain? From Monolithic Class Wrappers to Pure Functional Pipelines", divider="rainbow")

st.markdown("""
At its core, a **Chain** is a directed graph of computational units where data flows from source inputs 
through transformation steps to produce an output:

```text
Input --> [Step 1: Prompt] --> [Step 2: Model] --> [Step 3: Parser] --> Output
```

In modern LangChain, any object that implements the `Runnable` protocol can serve as a link in the chain. 
Because components adhere to a single standard, chains are **stateless**, **thread-safe**, and **reusable**.
""")

# --- SECTION 3: CHAIN COMPONENTS ---
st.header("3. Chain Components: Inputs, Prompts, Models, Adapters, Parsers & State Contexts", divider="rainbow")

st.markdown("""
Every modern chain is composed of five standard architectural tiers:

1. **Input Payload**: Typically a Python dictionary supplying dynamic variables.
2. **Prompt Template**: Formats variables into system, human, and message histories (`ChatPromptTemplate`).
3. **Cognitive Engine**: The foundation model generating reasoning or tool calls (`BaseChatModel`).
4. **Intermediate Adapters**: Data transformations, custom functions, or retrievers (`RunnableLambda`, `RunnablePassthrough`).
5. **Output Parser**: Deserializes raw responses into strings, JSON schemas, or Pydantic models (`BaseOutputParser`).
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. Input: {"cloud_provider": "AWS"}
# 2. Prompt:
prompt = PromptTemplate.from_template("What is the primary managed container service on {cloud_provider}?")
# 3. Model:
model = ChatOpenAI(model="gpt-4o", temperature=0.0)
# 4. Parser:
parser = StrOutputParser()

# The Chain:
chain = prompt | model | parser
response = chain.invoke({"cloud_provider": "AWS"})
print(response)  # "Amazon Elastic Kubernetes Service (EKS) and Amazon ECS."
''', language="python")

# --- SECTION 4: SEQUENTIAL CHAINS ---
st.header("4. Sequential Chains: Linear Pipelines, Unidirectional Data Flow & State Accumulation", divider="rainbow")

st.markdown("""
In a **Sequential Chain**, execution moves in a strict linear order. Each stage consumes the output 
of the previous stage.

When subsequent stages need access to earlier variables (e.g., preserving the original user query alongside 
the model's intermediate answer), LangChain uses **`RunnablePassthrough.assign()`** to accumulate state 
into a growing context dictionary.
""")

dot_sequential = """
digraph SequentialStateAccumulation {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_in [label="Step 1: Initial Input\n{'query': 'Analyze incident INC-901'}", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_step2 [label="Step 2: RunnablePassthrough.assign(triage=...)\nOutput: {'query': '...', 'triage': 'SEVERITY_1'}", fillcolor="#1E293B", fontcolor="#F59E0B"];
    node_step3 [label="Step 3: RunnablePassthrough.assign(action=...)\nOutput: {'query': '...', 'triage': '...', 'action': 'Page On-Call'}", fillcolor="#1E293B", fontcolor="#34D399"];

    node_in -> node_step2 [label="pipe (|)", color="#38BDF8"];
    node_step2 -> node_step3 [label="pipe (|)", color="#34D399"];
}
"""

with st.expander("🔍 Contextual Diagram: Sequential State Accumulation Lifecycle", expanded=True):
    st.graphviz_chart(dot_sequential, use_container_width=True)

st.code('''from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o", temperature=0.1)
parser = StrOutputParser()

# Stage 1: Generate technical hypothesis
hypothesis_chain = PromptTemplate.from_template("Hypothesize root cause for alert: {alert}") | model | parser

# Stage 2: Preserving 'alert' while adding 'hypothesis'
full_chain = (
    RunnablePassthrough.assign(hypothesis=hypothesis_chain)
    | PromptTemplate.from_template("Alert: {alert}\\nHypothesis: {hypothesis}\\nDraft remediation command:")
    | model
    | parser
)

result = full_chain.invoke({"alert": "High CPU throttling on Kubernetes ingress"})
''', language="python")

# --- SECTION 5: PARALLEL CHAINS ---
st.header("5. Parallel Chains: Concurrent Multi-Head Processing & Map-Reduce Topologies", divider="rainbow")

st.markdown("""
A **Parallel Chain** splits an incoming input into multiple branches, evaluates them simultaneously 
across worker threads, and aggregates their outputs into a single dictionary.

This is essential for:
- Evaluating multiple perspectives on the same text (e.g., security review AND cost review).
- Generating multiple alternative candidate answers concurrently.
- Implementing Map-Reduce summarization pipelines.
""")

st.code('''from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o", temperature=0.1)
parser = StrOutputParser()

# Branch A: Security Perspective
sec_branch = PromptTemplate.from_template("Security analysis for: {tech}") | model | parser

# Branch B: Financial Perspective
fin_branch = PromptTemplate.from_template("Cost/financial analysis for: {tech}") | model | parser

# Concurrent Multi-Head Pipeline
evaluation_chain = RunnableParallel(
    security_verdict=sec_branch,
    financial_verdict=fin_branch
)

# Output is a single aggregated dictionary:
# {'security_verdict': '...', 'financial_verdict': '...'}
res = evaluation_chain.invoke({"tech": "Multi-region active-active database replication"})
''', language="python")

# --- SECTION 6: CONDITIONAL CHAINS ---
st.header("6. Conditional Chains: Dynamic Branching, Intent-Based Routing & Context Guardrails", divider="rainbow")

st.markdown("""
In real-world applications, different user queries require completely different processing logic. 
A **Conditional Chain** dynamically inspects the input or intermediate output and selects the appropriate 
sub-chain to execute using `RunnableBranch` or a routing lambda.
""")

st.code('''from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Sub-Chain 1: Code Review
code_chain = PromptTemplate.from_template("Review code: {input}") | model | parser

# Sub-Chain 2: General Chat
chat_chain = PromptTemplate.from_template("Answer question: {input}") | model | parser

# Conditional Router
router = RunnableBranch(
    (lambda x: "def " in x["input"] or "class " in x["input"], code_chain),
    chat_chain  # Default branch
)

print(router.invoke({"input": "def calculate_hash(data): return sha256(data)"}))
''', language="python")

# --- SECTION 7: NESTED CHAINS ---
st.header("7. Nested Chains: Encapsulating Sub-Pipelines, Hierarchical Workflows & Modular Abstraction", divider="rainbow")

st.markdown("""
Because modern chains are pure Runnables, any chain can be embedded as an atomic step inside another chain. 
This enables **hierarchical architecture**: decomposing massive complex enterprise workflows into 
modular, independently testable sub-chains.
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o", temperature=0.0)
parser = StrOutputParser()

# Modular Sub-Chain 1: Fact Extractor
extractor_subchain = (
    PromptTemplate.from_template("Extract technical facts from: {document}")
    | model
    | parser
)

# Modular Sub-Chain 2: Executive Summarizer
summarizer_subchain = (
    PromptTemplate.from_template("Create executive briefing from facts:\\n{facts}")
    | model
    | parser
)

# Hierarchical Master Pipeline:
# Extractor subchain output feeds directly into Summarizer subchain
master_hierarchical_chain = (
    {"facts": extractor_subchain}
    | summarizer_subchain
)
''', language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Multi-Tier Chain Architecture", divider="rainbow")

st.markdown("""
Below is an enterprise-grade multi-tier chain implementation. It demonstrates hierarchical nesting, 
concurrent parallel auditing branches, and state accumulation via `RunnablePassthrough.assign()`.
""")

st.code('''from typing import Dict, Any
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class EnterpriseArchitectureAuditor:
    """Production multi-stage chain auditing system architecture documents."""

    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
        self.parser = StrOutputParser()

        # Tier 1: Security Audit Sub-Chain
        sec_prompt = ChatPromptTemplate.from_template("Identify security vulnerabilities in architecture:\\n{doc}")
        self.security_chain = sec_prompt | self.model | self.parser

        # Tier 2: Reliability Audit Sub-Chain
        rel_prompt = ChatPromptTemplate.from_template("Identify single-points-of-failure in architecture:\\n{doc}")
        self.reliability_chain = rel_prompt | self.model | self.parser

        # Tier 3: Consolidated Executive Synthesis Sub-Chain
        synthesis_prompt = ChatPromptTemplate.from_template("""### ENTERPRISE ARCHITECTURE REVIEW ###
Target Architecture: {doc_title}

[SECURITY AUDIT]
{security_report}

[RELIABILITY AUDIT]
{reliability_report}

Provide final go/no-go production deployment decision:""")
        self.synthesis_chain = synthesis_prompt | self.model | self.parser

        # Assemble Master Hierarchical Chain
        self.master_audit_chain = (
            RunnablePassthrough.assign(
                security_report=self.security_chain,
                reliability_report=self.reliability_chain
            )
            | self.synthesis_chain
        )

    def review_architecture(self, title: str, document_text: str) -> str:
        return self.master_audit_chain.invoke({
            "doc_title": title,
            "doc": document_text
        })
''', language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Live Chain Topology Inspector & Multi-Pattern Execution Lab", divider="rainbow")

st.markdown("""
Explore different chain topologies in the interactive lab below. 
Compare **Sequential**, **Parallel**, **Conditional**, and **Nested** topologies, 
inspect their execution stages, and observe latency characteristics.
""")

sim_engine = MockChainTopologySimulator()

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    selected_topo = st.selectbox(
        "Select Chain Topology to Test",
        ["Sequential", "Parallel", "Conditional", "Nested"],
        index=0
    )
    user_input_wb = st.text_area(
        "Chain Input Payload",
        value="Database cluster primary node experienced network partition during peak traffic."
    )

with col_wb2:
    st.subheader("Topology Execution Stages")
    stages = sim_engine.topologies[selected_topo]
    for idx, stage in enumerate(stages):
        st.caption(f"Stage {idx + 1}: `{stage}`")

if st.button("🚀 Execute Chain Topology", type="primary"):
    res = sim_engine.simulate_execution(selected_topo, user_input_wb)
    
    st.subheader("📊 Execution Telemetry")
    m1, m2, m3 = st.columns(3)
    m1.metric("Topology", res["topology"])
    m2.metric("Total Stages", len(res["stages_executed"]))
    m3.metric("Latency", f"{res['total_latency_ms']} ms")

    st.write("### Execution Trace")
    for idx, stg in enumerate(res["stages_executed"]):
        st.write(f"**Step {idx+1}**: `{stg}` completed.")

    st.write("### Final Chain Output")
    st.code(res["output_summary"], language="markdown")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why did LangChain deprecate legacy chain classes like LLMChain in favor of LCEL?",
     "LLMChain and related classes were monolithic black boxes. They made it difficult to inspect intermediate state, lacked unified type hints, required custom callbacks for streaming, and suffered from inconsistent async support. Modern LCEL replaces them with composable Runnables using the pipe operator (|), guaranteeing native streaming, batching, async, and schema validation across every link in the chain."),
    ("How do you preserve intermediate variables across multi-step sequential chains in LCEL?",
     "By using `RunnablePassthrough.assign(**kwargs)`. Instead of replacing the input dictionary with the output of an intermediate step, `.assign()` computes new fields and appends them to the existing dictionary, ensuring all upstream inputs remain accessible to subsequent prompt templates."),
    ("What is the primary difference between a Sequential Chain and a Parallel Chain?",
     "A Sequential Chain executes steps serially, passing the output of stage N as input to stage N+1. A Parallel Chain broadcasts the exact same input to multiple independent branches simultaneously, evaluating them concurrently across worker threads and aggregating the results into a merged dictionary."),
    ("How does RunnableParallel improve throughput in RAG and multi-agent pipelines?",
     "In complex RAG pipelines, you often need to generate multiple search queries, retrieve from different vector collections, or perform independent safety checks. RunnableParallel executes these network I/O operations simultaneously instead of serially, reducing total latency from the sum of all branches to the duration of the slowest branch."),
    ("How does LangChain handle nested chains from a memory and execution perspective?",
     "Because any composite chain (like RunnableSequence) is itself a Runnable, nested chains are treated as single execution nodes. When invoked, the outer chain passes inputs to the inner chain, which executes its internal DAG and returns its output, with LangSmith preserving the full hierarchical parent-child trace structure.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Accidental State Overwrite in Sequential Pipes. Chaining `step1 | step2` where `step1` returns a string and `step2` expects a dictionary raises an immediate `TypeError`. Always use `RunnablePassthrough.assign()` when state needs to be retained.")
    st.error("**Trap 2 (Breaking Error)**: Importing Deprecated `langchain.chains.LLMChain`. Older tutorials recommend `LLMChain`. In LangChain v0.3+, importing this pulls deprecated modules lacking Pydantic v2 support. Always construct chains using `prompt | model`.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Redundant LLM Invocations in Parallel Chains. Creating 5 parallel branches that all invoke frontier models (GPT-4o) when simpler extraction rules would suffice multiplies your API cost by 5x. Use specialized, lightweight models for parallel checks.")
    st.warning("**Trap 4 (Quality Warning)**: Deeply Nested Chains Without Timeouts. If a sub-chain 4 levels deep stalls on a network call without a configured timeout, the entire parent pipeline hangs indefinitely. Always set `timeout` on individual models within sub-chains.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Modern Chain Definition"):
        st.markdown("**Any composition of Runnables assembled with pipe (|) or RunnableSequence.**")
    with st.expander("🎴 Flashcard 2: State Accumulation"):
        st.markdown("**RunnablePassthrough.assign() appends new fields while preserving existing keys.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Parallel Evaluation"):
        st.markdown("**RunnableParallel evaluates multi-branch Runnables concurrently.**")
    with st.expander("🎴 Flashcard 4: Hierarchical Nesting"):
        st.markdown("**Any RunnableSequence can be embedded as a single node inside another chain.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural differences between legacy chains and modern LCEL.", value=True)
chk2 = st.checkbox("I can identify the 5 core components of a modern chain.", value=True)
chk3 = st.checkbox("I know how to preserve state in sequential chains using RunnablePassthrough.assign().", value=True)
chk4 = st.checkbox("I can implement concurrent multi-branch workflows using RunnableParallel.", value=True)
chk5 = st.checkbox("I understand how to encapsulate modular sub-chains into hierarchical pipelines.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 6.1 Understanding Chains. Ready for 6.2 Building Chains!")
