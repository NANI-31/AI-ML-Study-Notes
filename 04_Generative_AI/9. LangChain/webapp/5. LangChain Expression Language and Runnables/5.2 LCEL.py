import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="5.2 LCEL - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("5.2 LangChain Expression Language (LCEL): Declarative Composition & Branching")
st.caption("Chapter 5: LangChain Expression Language and Runnables | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
The **LangChain Expression Language (LCEL)** is a declarative domain-specific language designed for orchestrating 
complex cognitive pipelines in Python. By leveraging Python's native pipe operator (`|`), LCEL allows developers 
to compose modular Runnables into Directed Acyclic Graphs (DAGs) without writing imperative glue code.

LCEL is not merely syntactic sugar: it natively compiles pipelines with **automatic streaming**, **transparent 
parallel execution**, **asynchronous non-blocking I/O**, **distributed tracing hooks**, and **built-in schema validation**.

In this lesson, you will master:
- **Declarative Composition**: How Python's `__or__` dunder method compiles declarative pipelines.
- **Component Chaining**: Connecting prompts, chat models, tools, and output parsers.
- **Sequential vs Parallel Flow**: Combining linear sequences with concurrent dictionary branches.
- **Conditional Branching**: Routing execution paths dynamically using `RunnableBranch` or custom lambda classifiers.
- **Sub-DAG Encapsulation**: Creating reusable, nested pipelines that function as standalone atomic units.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockLCELBranchRouter:
    """Simulates dynamic conditional routing in module scope."""
    def __init__(self):
        self.routes = {
            "Billing Query": "Billing Specialist Chain (Stripe API + Invoice DB)",
            "Technical Support": "Technical Support Chain (Codebase RAG + Prometheus)",
            "Sales Inquiry": "Sales Lead Qualification Chain (HubSpot + CRM)"
        }

    def route_query(self, query: str) -> Tuple[str, str, int]:
        q_lower = query.lower()
        if any(w in q_lower for w in ["invoice", "credit", "card", "refund", "billing", "charge"]):
            category = "Billing Query"
            latency = 140
        elif any(w in q_lower for w in ["error", "crash", "cluster", "node", "bug", "latency", "timeout"]):
            category = "Technical Support"
            latency = 310
        else:
            category = "Sales Inquiry"
            latency = 190

        resolved_chain = self.routes[category]
        return category, resolved_chain, latency

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: LangChain Expression Language Declarative Pipeline Topology", divider="rainbow")

st.markdown("""
The diagram below illustrates the complete execution topology of an enterprise LCEL pipeline: 
from incoming request through parallel feature extraction, conditional routing, model inference, and output parsing.
""")

dot_blueprint = """
digraph LCELExecutionTopology {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_input {
        label="1. Ingestion Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_raw [label="User Input: {'question': 'Refund order #9821'}", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    subgraph cluster_parallel {
        label="2. Parallel Enrichment Tier (RunnableParallel)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_passthrough [label="question: RunnablePassthrough()", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_intent [label="intent: IntentClassificationChain", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_context [label="context: VectorRetriever", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_raw -> node_passthrough [color="#38BDF8"];
        node_raw -> node_intent [color="#38BDF8"];
        node_raw -> node_context [color="#38BDF8"];
    }

    subgraph cluster_branch {
        label="3. Conditional Routing Tier (RunnableBranch / Lambda)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_router [label="Branch Router (Condition on intent)", shape=diamond, fillcolor="#0F172A", fontcolor="#F59E0B"];
        node_billing [label="Billing Chain\n(Prompt + Model + Tool)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_support [label="Support Chain\n(RAG Context + Model)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_general [label="Default Chain\n(Standard FAQ ChatModel)", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_passthrough -> node_router [style="invis"];
        node_intent -> node_router [color="#F59E0B"];
        
        node_router -> node_billing [label="intent == 'billing'", color="#34D399"];
        node_router -> node_support [label="intent == 'support'", color="#38BDF8"];
        node_router -> node_general [label="default fallback", color="#64748B"];
    }

    subgraph cluster_output {
        label="4. Parsing Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_parser [label="StrOutputParser / PydanticOutputParser\nExtracts validated response string or DTO", fillcolor="#1E293B", fontcolor="#34D399"];

        node_billing -> node_parser [color="#34D399"];
        node_support -> node_parser [color="#34D399"];
        node_general -> node_parser [color="#34D399"];
    }
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: In LCEL, pipelines are declared as pure directed acyclic graphs. 
> Notice how the incoming request splits into concurrent parallel branches, converges at the router, 
> executes the specialized sub-chain, and finally streams through the output parser.
""")

# --- SECTION 2: LANGCHAIN EXPRESSION LANGUAGE ---
st.header("2. LangChain Expression Language: Foundations, Declarative Syntax & Functional Composition", divider="rainbow")

st.markdown("""
Prior to LCEL, chaining components in LangChain required writing imperative code or relying on 
monolithic wrapper classes like `LLMChain` and `SequentialChain`. 

### Why LCEL Replaced Legacy Chains

| Dimension | Legacy Chains (`langchain.chains`) | Modern LCEL (`langchain_core.runnables`) |
| :--- | :--- | :--- |
| **Composition Syntax** | Nested classes with dictionary arguments | Clean declarative pipe operator (`|`) |
| **Streaming Support** | Opaque; required custom callback handlers | Native first-class token streaming (`.stream()`) |
| **Async Execution** | Inconsistent across classes; high risk of blocking | 100% native asynchronous event loop support (`.ainvoke()`) |
| **Intermediate State** | Black-box hidden state inside class wrappers | Transparent inspection of every DAG node |
| **Type Safety** | Dynamic string keys without compile-time types | Auto-generated Pydantic schemas via `get_input_schema()` |
""")

# --- SECTION 3: PIPE OPERATOR ---
st.header("3. Pipe Operator (|): Python __or__ Overloading, Pipeline Assembly & DAG Compilation", divider="rainbow")

st.markdown("""
In Python, the pipe symbol (`|`) is the bitwise OR operator. In LCEL, `Runnable` overloads the 
dunder methods `__or__` and `__ror__`.

When you write `A | B`:
1. If `A` is a `Runnable` and `B` is a `Runnable`, it compiles a `RunnableSequence(first=A, last=B)`.
2. If `A` is a `Runnable` and `B` is a standard Python function, `B` is automatically coerced into a `RunnableLambda(B)`.
3. If `A` is a `Runnable` and `B` is a Python dictionary, `B` is automatically coerced into a `RunnableParallel(B)`.
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Component definition
prompt = PromptTemplate.from_template("Explain {concept} in 1 sentence.")
model = ChatOpenAI(model="gpt-4o", temperature=0.0)
parser = StrOutputParser()

# The pipe operator creates a RunnableSequence:
chain = prompt | model | parser

# Equivalently:
# chain = prompt.__or__(model).__or__(parser)
print(type(chain))  # <class 'langchain_core.runnables.base.RunnableSequence'>
''', language="python")

# --- SECTION 4: CHAINING COMPONENTS ---
st.header("4. Chaining Components: Prompt, Model, Parser & Type Contract Validation", divider="rainbow")

st.markdown("""
The canonical LCEL pattern is **Prompt + Model + Output Parser**. 
Data flows seamlessly through three type boundaries:

```text
Input: dict 
  --> ChatPromptTemplate (produces ChatPromptValue)
  --> BaseChatModel (produces AIMessage)
  --> StrOutputParser (produces str)
Output: str
```
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Canonical 3-stage LCEL chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enterprise Kubernetes SRE."),
    ("human", "How do I diagnose {incident_type}?")
])
model = ChatOpenAI(model="gpt-4o", temperature=0.1)
parser = StrOutputParser()

chain = prompt | model | parser

# Invocations are strongly typed end-to-end
response_text = chain.invoke({"incident_type": "CrashLoopBackOff"})
print(response_text)
''', language="python")

# --- SECTION 5: SEQUENTIAL EXECUTION ---
st.header("5. Sequential Execution: Linear Data Transformation & Streaming Propagation", divider="rainbow")

st.markdown("""
In a sequential chain, each step executes in order. The critical advantage of LCEL is that 
**streaming propagates across sequential boundaries automatically**. If the model streams token chunks, 
downstream parsers yield string chunks immediately without waiting for the full response to finish generating.
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    PromptTemplate.from_template("List 3 advantages of asynchronous microservices:")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

# Streaming progressive chunks directly to stdout:
for token in chain.stream({}):
    print(token, end="", flush=True)
''', language="python")

# --- SECTION 6: PARALLEL EXECUTION ---
st.header("6. Parallel Execution: Dict-Coerced Multi-Branch Concurrency", divider="rainbow")

st.markdown("""
When you pipe a `Runnable` into a Python dictionary, LangChain automatically runs each dictionary value 
in parallel worker threads.
""")

st.code('''from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o", temperature=0.1)
parser = StrOutputParser()

# Two parallel chains evaluated simultaneously
branch_a = PromptTemplate.from_template("Summarize in 5 words: {text}") | model | parser
branch_b = PromptTemplate.from_template("Detect sentiment (POSITIVE/NEGATIVE): {text}") | model | parser

# Parallel pipeline: input dictionary is broadcast to both branches
parallel_chain = {
    "summary": branch_a,
    "sentiment": branch_b,
    "original": RunnablePassthrough()  # Preserves raw input
}

results = parallel_chain.invoke({"text": "The database cluster experienced high memory pressure."})
# {'summary': 'Database had high memory pressure.', 'sentiment': 'NEGATIVE', 'original': {'text': '...'}}
''', language="python")

# --- SECTION 7: BRANCHING ---
st.header("7. Branching: Conditional Routing (RunnableBranch), Dynamic Selectors & Fallback Paths", divider="rainbow")

st.markdown("""
Not all tasks follow a single linear trajectory. In intelligent systems, execution paths must branch 
dynamically based on classification results or query characteristics.

LangChain provides two ways to implement branching:
1. **`RunnableBranch`**: The declarative condition-action branching primitive.
2. **Custom routing functions via `RunnableLambda`**: Returning different sub-chains dynamically based on input.
""")

dot_branch_flow = """
digraph RunnableBranchFlow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_query [label="Incoming Request Dictionary\n{'query': 'How to optimize PostgreSQL query plans?'}", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_classifier [label="Classifier Lambda / LLM Router", fillcolor="#1E293B", fontcolor="#F59E0B"];
    node_branch [label="RunnableBranch Evaluation", shape=diamond, fillcolor="#0F172A", fontcolor="#38BDF8"];
    
    node_sql [label="SQL Optimization Chain", fillcolor="#1E293B", fontcolor="#34D399"];
    node_code [label="Python Code Review Chain", fillcolor="#1E293B", fontcolor="#38BDF8"];
    node_general [label="General Engineering Chain", fillcolor="#1E293B", fontcolor="#64748B"];

    node_query -> node_classifier -> node_branch;
    node_branch -> node_sql [label="condition 1: 'sql' in query", color="#34D399"];
    node_branch -> node_code [label="condition 2: 'code' in query", color="#38BDF8"];
    node_branch -> node_general [label="default fallback", color="#64748B"];
}
"""

with st.expander("🔍 Contextual Diagram: Conditional Branching & Dynamic Routing", expanded=True):
    st.graphviz_chart(dot_branch_flow, use_container_width=True)

st.code('''from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o", temperature=0.0)
parser = StrOutputParser()

# Specialized Sub-Chains
sql_chain = PromptTemplate.from_template("Provide PostgreSQL SQL optimization for: {query}") | model | parser
python_chain = PromptTemplate.from_template("Provide Python performance review for: {query}") | model | parser
general_chain = PromptTemplate.from_template("Answer engineering question: {query}") | model | parser

# Declarative RunnableBranch
branch = RunnableBranch(
    (lambda x: "sql" in x["query"].lower(), sql_chain),
    (lambda x: "python" in x["query"].lower(), python_chain),
    general_chain  # Default branch if no conditions match
)

res = branch.invoke({"query": "Explain SQL index creation."})
''', language="python")

# --- SECTION 8: COMPOSITION ---
st.header("8. Composition: Nested Chains, Higher-Order Runnables & Sub-DAG Encapsulation", divider="rainbow")

st.markdown("""
A primary tenet of software engineering is **composability**: complex systems are built by composing simpler, 
reusable parts. Because any `RunnableSequence` or `RunnableParallel` is itself a `Runnable`, you can nest 
sub-chains inside parent chains with zero impedance mismatch.
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Sub-chain: Generate draft
draft_chain = PromptTemplate.from_template("Draft a technical memo on: {topic}") | ChatOpenAI(model="gpt-4o") | StrOutputParser()

# Sub-chain: Polish and audit
audit_chain = PromptTemplate.from_template("Critique and improve this technical memo:\\n{memo}") | ChatOpenAI(model="gpt-4o") | StrOutputParser()

# Nested Master Pipeline:
master_pipeline = (
    {"memo": draft_chain}  # Draft chain executes first, producing {'memo': str}
    | audit_chain          # Audited memo produced from draft
)
''', language="python")

# --- SECTION 9: REUSABLE CHAINS ---
st.header("9. Reusable Chains: Component Modularization, Factory Patterns & Cross-Service Reuse", divider="rainbow")

st.markdown("""
In enterprise microservices, chains should be packaged as reusable factory functions. 
A chain factory encapsulates model hyperparameters, system prompts, and parsers, exposing 
a clean, type-safe API to the rest of your organization.
""")

st.code('''from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def create_summarization_chain(model_name: str = "gpt-4o", max_bullets: int = 3) -> RunnableSequence:
    """Factory creating an enterprise-standardized summarization Runnable."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a technical editor. Summarize all inputs in at most {max_bullets} bullet points."),
        ("human", "{text}")
    ])
    model = ChatOpenAI(model=model_name, temperature=0.0)
    return prompt | model | StrOutputParser()

# Shared across internal teams:
# summarizer = create_summarization_chain(max_bullets=5)
# output = summarizer.invoke({"text": "Production system logs..."})
''', language="python")

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Enterprise Multi-Stage LCEL Pipeline with Branching", divider="rainbow")

st.markdown("""
Below is an enterprise-grade multi-stage LCEL application. It combines input classification, 
parallel feature enrichment, dynamic conditional branching, and centralized output parsing.
""")

st.code('''from typing import Dict, Any
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class EnterpriseSupportRouter:
    """Production cognitive router directing support inquiries to specialized engineering chains."""

    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
        self.parser = StrOutputParser()

        # 1. Specialized Sub-Chains
        billing_prompt = PromptTemplate.from_template("You are a billing lead. Resolve: {query}")
        self.billing_chain = billing_prompt | self.model | self.parser

        tech_prompt = PromptTemplate.from_template("You are a Staff SRE. Resolve infrastructure alert: {query}")
        self.tech_chain = tech_prompt | self.model | self.parser

        general_prompt = PromptTemplate.from_template("You are a general support representative. Answer: {query}")
        self.general_chain = general_prompt | self.model | self.parser

        # 2. Classifier Lambda
        def classify_intent(inputs: Dict[str, Any]) -> str:
            text = inputs["query"].lower()
            if any(w in text for w in ["invoice", "refund", "billing", "credit"]):
                return "BILLING"
            elif any(w in text for w in ["crash", "latency", "error", "pod", "database"]):
                return "TECH"
            return "GENERAL"

        # 3. Dynamic RunnableBranch
        self.router_branch = RunnableBranch(
            (lambda x: x["category"] == "BILLING", self.billing_chain),
            (lambda x: x["category"] == "TECH", self.tech_chain),
            self.general_chain
        )

        # 4. Master Composite LCEL Pipeline
        self.master_pipeline: RunnableSequence = (
            RunnablePassthrough.assign(category=RunnableLambda(classify_intent))
            | self.router_branch
        )

    def process_ticket(self, user_query: str) -> str:
        return self.master_pipeline.invoke({"query": user_query})
''', language="python")

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Interactive LCEL Pipeline Graph Explorer & Live Runner", divider="rainbow")

st.markdown("""
Test the dynamic LCEL conditional routing mechanism below. Submit queries across different domains 
(Billing, Infrastructure SRE, or Sales) and observe how the classifier dynamically routes execution 
to the appropriate specialized chain.
""")

router_engine = MockLCELBranchRouter()

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    user_query_wb = st.text_area(
        "User Ticket Query",
        value="Database cluster latency spiked to 450ms and worker pods are restarting."
    )

with col_wb2:
    st.subheader("Available Domain Sub-Chains")
    for category, chain_name in router_engine.routes.items():
        st.caption(f"• **{category}**: `{chain_name}`")

if st.button("🚀 Execute LCEL Routing Pipeline", type="primary"):
    category_resolved, chain_assigned, latency_est = router_engine.route_query(user_query_wb)
    
    st.subheader("📊 Dynamic Execution Diagnostics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Resolved Intent", category_resolved)
    m2.metric("Target Sub-Chain", chain_assigned.split(" (")[0])
    m3.metric("Execution Latency", f"{latency_est} ms")

    st.write("### Active Route Topology")
    st.info(f"Route selected: **{category_resolved}** ➔ Dispatched to `{chain_assigned}`")

    st.write("### Synthesized Response Output")
    st.code(
        f"[{category_resolved.upper()} SPECIALIST RESPONSE]\n"
        f"Analyzed query: '{user_query_wb}'\n"
        f"Dispatched via LCEL branch to {chain_assigned}. "
        f"Remediation steps generated under unified Runnable contract.",
        language="markdown"
    )

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does LCEL's pipe operator (|) differ from the Unix command-line pipe?",
     "While both pipe data from one stage to the next, Unix pipes operate strictly on byte streams. LCEL pipes operate on typed Python objects (dictionaries, BaseMessages, PromptValues). Furthermore, LCEL automatically enables bi-directional context propagation (RunnableConfig), concurrency via threadpools, and progressive streaming token generation across the pipe."),
    ("What happens under the hood when a dictionary is passed into an LCEL pipe sequence?",
     "LangChain automatically coerces the dictionary into a RunnableParallel instance. Each key's corresponding value (which is a Runnable) is executed concurrently across internal worker threads, and the results are aggregated into a single dictionary matching the input keys."),
    ("Why was RunnableBranch created and how does it compare to a standard if-else statement?",
     "RunnableBranch allows conditional routing to be defined declaratively inside an LCEL chain without breaking the pipe syntax. It takes tuples of (condition_callable, runnable_branch) followed by a default fallback. It ensures that whichever branch is selected still inherits the unified Runnable execution interface, streaming, and tracing."),
    ("How does LCEL optimize Time-To-First-Token (TTFT) during streaming?",
     "In a sequential chain like `prompt | model | parser`, streaming starts as soon as the model yields its first token chunk. The StrOutputParser yields that chunk immediately to the client application without waiting for the full response to finish generating, achieving sub-second TTFT even on long completions."),
    ("What are the debugging advantages of LCEL over legacy LLMChain?",
     "Legacy chains hid internal states inside opaque class methods. LCEL decomposes execution into discrete, observable Runnables. In tools like LangSmith, every pipe stage (prompt, model, parser, lambda) produces a dedicated span showing exact inputs, outputs, execution latencies, and token metrics.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Type Mismatch Between Pipe Stages. Piping a component that outputs a string into a component that expects a dictionary (like `ChatPromptTemplate`) raises an immediate runtime `TypeError`. Verify intermediate schemas.")
    st.error("**Trap 2 (Breaking Error)**: Missing Default in RunnableBranch. Failing to supply a final fallback branch to `RunnableBranch` causes an exception if none of the explicit conditional lambdas evaluate to True.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Redundant Model Invocations in Nested Pipes. Calling two independent chains that both invoke full GPT-4o models when a single shared context would suffice doubles API costs and latency. Use `RunnableParallel` with shared context.")
    st.warning("**Trap 4 (Quality Warning)**: Complex Conditionals Inside Lambdas. Writing 50-line nested conditional logic inside a single `RunnableLambda` breaks observability. Decompose complex routing into discrete, named functions.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Pipe Operator Behavior"):
        st.markdown("**Overloads __or__ to compose Runnables into a unified RunnableSequence.**")
    with st.expander("🎴 Flashcard 2: Dict Coercion"):
        st.markdown("**Dictionaries in LCEL are automatically transformed into RunnableParallel.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Declarative Routing"):
        st.markdown("**RunnableBranch((condition, branch), ..., default_branch) provides conditional execution.**")
    with st.expander("🎴 Flashcard 4: Streaming Across Chains"):
        st.markdown("**Calling .stream() on an LCEL chain progressively streams token chunks from model through parsers.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can construct LCEL pipelines using the pipe operator (|).", value=True)
chk2 = st.checkbox("I understand how Python's __or__ dunder method powers RunnableSequence.", value=True)
chk3 = st.checkbox("I know how dictionaries automatically convert into concurrent RunnableParallel branches.", value=True)
chk4 = st.checkbox("I can implement dynamic routing using RunnableBranch and conditional lambdas.", value=True)
chk5 = st.checkbox("I understand how streaming propagates across sequential LCEL boundaries.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 5.2 LCEL. Ready for 5.3 Runnable Methods!")
