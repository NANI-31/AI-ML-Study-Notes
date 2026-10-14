import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="6.2 Building Chains - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("6.2 Building Chains: The Canonical Triad, Multi-Step Workflows & Composition")
st.caption("Chapter 6: Chains | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Building production-ready Generative AI systems requires mastering the practical assembly of **Chains**. 
From simple two-element pairings (`Prompt + Model` or `Model + Parser`) to the universal industry standard 
**Canonical Triad (`Prompt + Model + Parser`)**, modern LCEL provides an intuitive, type-safe syntax for assembling 
robust cognitive pipelines.

In this lesson, you will master:
- **Foundational Pairings**: Connecting prompts to chat models and models to output parsers.
- **The Canonical Triad**: Assembling the `Prompt | Model | Parser` pattern powering 90% of GenAI applications.
- **Multi-Step Orchestration**: Feeding the synthesized output of one model turn as the contextual input to the next.
- **Reusable Chain Factories**: Encapsulating chains into standalone, parametrizeable factory functions.
- **Higher-Order Composition**: Nesting and merging specialized sub-chains into a master workflow.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockChainStepRecord:
    """Represents an execution step within a chain simulator in module scope."""
    def __init__(self, step_name: str, input_type: str, output_type: str, latency_ms: int, payload_preview: str):
        self.step_name = step_name
        self.input_type = input_type
        self.output_type = output_type
        self.latency_ms = latency_ms
        self.payload_preview = payload_preview

class MockChainBuilderSimulator:
    """Simulates multi-step chain execution with intermediate state inspection in module scope."""
    def __init__(self):
        pass

    def run_multi_step_audit(self, service_name: str, code_snippet: str) -> Dict[str, Any]:
        step1 = MockChainStepRecord(
            "Step 1: Prompt Formatting",
            "Dict[str, str]",
            "ChatPromptValue",
            2,
            f"Formatted prompt for service '{service_name}' with code block length {len(code_snippet)} chars."
        )
        step2 = MockChainStepRecord(
            "Step 2: Model Inference (Audit)",
            "ChatPromptValue",
            "AIMessage",
            340,
            "AIMessage: Identified potential resource leak in database connection pool handling."
        )
        step3 = MockChainStepRecord(
            "Step 3: Parser Transformation",
            "AIMessage",
            "str",
            1,
            "Parsed raw text into normalized vulnerability finding string."
        )
        step4 = MockChainStepRecord(
            "Step 4: Remediation Sub-Chain (Model)",
            "str",
            "AIMessage",
            290,
            "AIMessage: Generated safe context-managed 'with db_pool.acquire():' code refactor."
        )

        total_latency = step1.latency_ms + step2.latency_ms + step3.latency_ms + step4.latency_ms
        final_summary = (
            f"Multi-step chain completed for '{service_name}'. "
            f"Detected connection leak in audit phase; synthesized context-managed fix in remediation phase."
        )

        return {
            "service": service_name,
            "steps": [step1, step2, step3, step4],
            "total_latency_ms": total_latency,
            "final_remediation": final_summary
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Core Pipeline Construction & Composition Patterns", divider="rainbow")

st.markdown("""
The diagram below illustrates the progressive composition of chains: starting from foundational pairings, 
advancing through the **Canonical Triad (`Prompt | Model | Parser`)**, and scaling to multi-step pipelines.
""")

dot_blueprint = """
digraph ChainConstructionPatterns {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_triad {
        label="1. The Canonical Triad Pattern (Prompt | Model | Parser)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_p [label="ChatPromptTemplate\nInput: dict\nOutput: ChatPromptValue", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_m [label="BaseChatModel (GPT-4o)\nInput: ChatPromptValue\nOutput: AIMessage", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_o [label="StrOutputParser / PydanticParser\nInput: AIMessage\nOutput: str / DTO", fillcolor="#1E293B", fontcolor="#34D399"];

        node_p -> node_m [label="pipe (|)", color="#38BDF8"];
        node_m -> node_o [label="pipe (|)", color="#38BDF8"];
    }

    subgraph cluster_multistep {
        label="2. Multi-Step Chaining (Pipeline A | Pipeline B)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_stage1 [label="Stage 1: Diagnostic Chain\nAnalyzes raw log -> Produces root cause", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_adapter [label="RunnablePassthrough.assign()\nMerges log + root cause into context", fillcolor="#1E293B", fontcolor="#F59E0B"];
        node_stage2 [label="Stage 2: Remediation Chain\nConsumes context -> Produces CLI command", fillcolor="#1E293B", fontcolor="#34D399"];

        node_stage1 -> node_adapter [label="pipe (|)", color="#38BDF8"];
        node_adapter -> node_stage2 [label="pipe (|)", color="#34D399"];
    }

    node_o -> node_stage1 [style="dashed", color="#64748B"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: Notice how every composition is homogeneous. Whether you are chaining three 
> elementary components (`Prompt | Model | Parser`) or two massive multi-stage enterprise pipelines, 
> the interface remains identical. Each link adheres to the same standard `Runnable` contract.
""")

# --- SECTION 2: PROMPT + MODEL ---
st.header("2. Prompt + Model: The Primary Cognitive Bridge & Message Serialization", divider="rainbow")

st.markdown("""
The simplest composite chain pairs a `ChatPromptTemplate` directly with a `ChatModel`. 

```text
chain = prompt | model
Input: dict --> Output: AIMessage
```

This pattern is ideal when you need to inspect the raw `AIMessage` returned by the model—such as accessing 
`tool_calls`, reading raw token usage in `usage_metadata`, or inspecting provider-specific flags in `response_metadata`.
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a cloud architect."),
    ("human", "Recommend a primary database for {workload_type}.")
])
model = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Prompt + Model Chain
chain = prompt | model

# Output is a rich AIMessage object
ai_msg = chain.invoke({"workload_type": "time-series IoT telemetry"})
print(type(ai_msg))  # <class 'langchain_core.messages.ai.AIMessage'>
print(ai_msg.content)
print(ai_msg.usage_metadata)  # {'input_tokens': 24, 'output_tokens': 56, ...}
''', language="python")

# --- SECTION 3: MODEL + PARSER ---
st.header("3. Model + Parser: Transforming Generation Payloads into Typed Outputs", divider="rainbow")

st.markdown("""
Conversely, you can pair a `ChatModel` directly with an output parser.

```text
chain = model | parser
Input: List[BaseMessage] (or str) --> Output: str (or Pydantic DTO)
```

This pattern is common when the upstream input is already an explicit list of `BaseMessage` objects 
(such as in multi-turn chat applications or agent memory buffers) and you simply want to deserialize the model's response.
""")

st.code('''from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o", temperature=0.0)
parser = StrOutputParser()

# Model + Parser Chain
chain = model | parser

# Direct invocation with message array
response_text = chain.invoke([
    HumanMessage(content="Explain Little's Law in 10 words.")
])
print(type(response_text))  # <class 'str'>
print(response_text)
''', language="python")

# --- SECTION 4: PROMPT + MODEL + PARSER ---
st.header("4. Prompt + Model + Parser: The Canonical Triad Pattern of Modern LCEL", divider="rainbow")

st.markdown("""
The **Canonical Triad (`prompt | model | parser`)** is the defining architectural pattern of modern LangChain. 
It cleanly separates:
1. **Input Preparation**: Handled declaratively by the prompt template.
2. **Cognitive Reasoning**: Handled by the model.
3. **Data Deserialization**: Handled by the output parser.
""")

dot_triad = """
digraph CanonicalTriadFlow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_in [label="Incoming Request Dict\n{'service': 'auth-service', 'qps': 14000}", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_prompt [label="ChatPromptTemplate\nFormats variables into system and user turns", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_model [label="BaseChatModel (GPT-4o)\nAutoregressively generates response tokens", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_parser [label="JsonOutputParser / StrOutputParser\nParses JSON text into validated Python dict", fillcolor="#1E293B", fontcolor="#34D399"];
    node_out [label="Downstream API Response\nType-safe payload ready for client consumption", fillcolor="#1E293B", fontcolor="#34D399"];

    node_in -> node_prompt [label="1. dict input", color="#38BDF8"];
    node_prompt -> node_model [label="2. ChatPromptValue", color="#38BDF8"];
    node_model -> node_parser [label="3. AIMessage", color="#38BDF8"];
    node_parser -> node_out [label="4. validated output", color="#34D399"];
}
"""

with st.expander("🔍 Contextual Diagram: The Canonical Triad Data Flow", expanded=True):
    st.graphviz_chart(dot_triad, use_container_width=True)

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# The Canonical Triad in action
triad_chain = (
    PromptTemplate.from_template("Provide 1 technical reason why {tech_a} outperforms {tech_b}.")
    | ChatOpenAI(model="gpt-4o", temperature=0.0)
    | StrOutputParser()
)

result = triad_chain.invoke({"tech_a": "PostgreSQL", "tech_b": "MySQL"})
print(result)
''', language="python")

# --- SECTION 5: MULTI-STEP CHAINS ---
st.header("5. Multi-Step Chains: Sequential Pipeline Orchestration & Intermediate Data Flow", divider="rainbow")

st.markdown("""
In multi-step chains, the output of an initial LLM generation is fed as the input into a second LLM generation. 
Common examples include:
- **Draft then Critique**: Model 1 drafts a technical document; Model 2 reviews it for security flaws.
- **Translate then Summarize**: Model 1 translates foreign text; Model 2 summarizes the translated text.
- **Query Expansion then Answer**: Model 1 expands an ambiguous query; Model 2 generates the grounded answer.
""")

st.code('''from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o", temperature=0.1)
parser = StrOutputParser()

# Step 1: Generate technical hypothesis
hypothesis_chain = PromptTemplate.from_template("Hypothesize root cause for alert: {alert}") | model | parser

# Step 2: Multi-step chain accumulating hypothesis and drafting remediation
two_step_chain = (
    RunnablePassthrough.assign(hypothesis=hypothesis_chain)
    | PromptTemplate.from_template("Alert: {alert}\\nHypothesis: {hypothesis}\\nDraft CLI fix:")
    | model
    | parser
)

fix = two_step_chain.invoke({"alert": "PostgreSQL read replica lag exceeded 2500ms"})
print(fix)
''', language="python")

# --- SECTION 6: REUSABLE CHAINS ---
st.header("6. Reusable Chains: Component Encapsulation, Factory Functions & Standardized Contracts", divider="rainbow")

st.markdown("""
To prevent code duplication across enterprise teams, chains should be encapsulated inside 
**factory functions**. A factory function accepts configuration options (such as model checkpoint or temperature) 
and returns a standardized `RunnableSequence`.
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

def make_code_review_chain(strict_mode: bool = True) -> RunnableSequence:
    """Factory creating an enterprise code review chain."""
    directive = "Be extremely strict and flag all potential edge cases." if strict_mode else "Focus only on critical errors."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a Principal Software Engineer. {directive}"),
        ("human", "Review this code snippet:\\n{code}")
    ])
    model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    return prompt | model | StrOutputParser()

# Reusable instantiation:
# strict_reviewer = make_code_review_chain(strict_mode=True)
# audit_result = strict_reviewer.invoke({"code": "def run(): eval(user_input)"})
''', language="python")

# --- SECTION 7: CHAIN COMPOSITION ---
st.header("7. Chain Composition: Hierarchical Merging, Pipe Operators & Nested Runnables", divider="rainbow")

st.markdown("""
Chains can be composed hierarchically. When you have two independent chains `Chain A` and `Chain B`, 
you can compose them into a higher-order pipeline using dictionaries and the pipe operator:

```python
master_chain = (
    {"audit": chain_a, "metadata": chain_b}
    | synthesis_chain
)
```
""")

st.code('''from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Sub-Chain A: Performance Audit
perf_chain = PromptTemplate.from_template("Evaluate performance of {code}") | model | parser

# Sub-Chain B: Security Audit
sec_chain = PromptTemplate.from_template("Evaluate security vulnerabilities in {code}") | model | parser

# Synthesis Chain
synthesis_prompt = PromptTemplate.from_template("""### CONSOLIDATED CODE REVIEW ###
Performance Findings: {perf}
Security Findings: {sec}
Final Verdict:""")
synthesis_chain = synthesis_prompt | model | parser

# Composite Master Pipeline:
master_review_chain = (
    {"perf": perf_chain, "sec": sec_chain}
    | synthesis_chain
)
''', language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise Multi-Stage Code Review and Remediation Chain", divider="rainbow")

st.markdown("""
Below is an enterprise-grade multi-stage pipeline demonstrating the Canonical Triad, intermediate data flow, 
and sub-chain composition.
""")

st.code('''from typing import Dict, Any
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class EnterpriseCodeReviewOrchestrator:
    """Production orchestrator combining automated defect detection and patch generation."""

    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
        self.parser = StrOutputParser()

        # Stage 1: Defect Detection Sub-Chain
        detect_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an automated static security analyzer. Identify high-risk bugs in the code."),
            ("human", "Target Service: {service}\\nCode:\\n{code}")
        ])
        self.detection_chain = detect_prompt | self.model | self.parser

        # Stage 2: Remediation Patch Sub-Chain
        patch_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Principal Software Engineer. Provide a minimal, secure git diff patch to fix the defect."),
            ("human", "Service: {service}\\nOriginal Code:\\n{code}\\nDefect Analysis:\\n{defect_analysis}")
        ])
        self.patch_chain = patch_prompt | self.model | self.parser

        # Master End-to-End Pipeline
        self.master_pipeline: RunnableSequence = (
            RunnablePassthrough.assign(defect_analysis=self.detection_chain)
            | RunnablePassthrough.assign(patch_diff=self.patch_chain)
        )

    def execute_audit(self, service_name: str, code_snippet: str) -> Dict[str, str]:
        return self.master_pipeline.invoke({
            "service": service_name,
            "code": code_snippet
        })
''', language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Interactive Chain Builder & Step-by-Step Execution Inspector", divider="rainbow")

st.markdown("""
Inspect the multi-step chain execution lifecycle below. 
Submit a sample service name and code snippet, and inspect each intermediate stage 
(Prompt formatting, Defect detection inference, Parser transformation, and Patch generation).
""")

builder_sim = MockChainBuilderSimulator()

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    service_input = st.text_input("Microservice Identifier", value="payment-auth-service")
    code_input = st.text_area(
        "Code Snippet for Review",
        value="""def process_payment(account_id, amount):
    conn = get_raw_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE accounts SET balance = balance - {amount} WHERE id = '{account_id}'")
    conn.commit()"""
    )

with col_wb2:
    st.subheader("Configured Pipeline Topology")
    st.info("""
    **Multi-Stage LCEL Pipeline:**
    1. `Prompt | Model | StrOutputParser` ➔ Analyzes SQL Injection & Connection Leak
    2. `RunnablePassthrough.assign()` ➔ Merges Findings
    3. `Remediation Prompt | Model | Parser` ➔ Synthesizes Parametrized Fix
    """)

if st.button("🚀 Execute Multi-Stage Review Chain", type="primary"):
    res = builder_sim.run_multi_step_audit(service_input, code_input)
    
    st.subheader("📊 Execution Diagnostics")
    m1, m2 = st.columns(2)
    m1.metric("Stages Executed", len(res["steps"]))
    m2.metric("Total Pipeline Latency", f"{res['total_latency_ms']} ms")

    st.write("### Step-by-Step Intermediate Execution Spans")
    for s in res["steps"]:
        with st.expander(f"⏱️ {s.step_name} ({s.latency_ms} ms)", expanded=True):
            st.caption(f"Input Type: `{s.input_type}` ➔ Output Type: `{s.output_type}`")
            st.code(s.payload_preview, language="markdown")

    st.write("### Final Consolidated Outcome")
    st.success(res["final_remediation"])

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why is the Canonical Triad (Prompt | Model | Parser) considered the fundamental building block in LangChain?",
     "The Canonical Triad represents the complete lifecycle of a single cognitive task: 1) Prompt transforms structured variables into formatted messages, 2) Model performs autoregressive generation, and 3) Parser deserializes the raw response into a clean string or strongly-typed Pydantic model. Chaining these three guarantees type contracts, transparent streaming, and zero glue code."),
    ("When would you intentionally use a `Prompt | Model` chain without an output parser?",
     "When you need access to the rich metadata and attributes of the raw AIMessage object—such as inspecting tool_calls for agentic dispatching, extracting token counts from usage_metadata, checking finish_reason, or inspecting logprobs."),
    ("How does `RunnablePassthrough.assign()` prevent data loss in multi-step chains?",
     "In a multi-step chain, subsequent steps often need both the original user query and the output of intermediate steps. Without .assign(), piping directly into a new step replaces the input with only the intermediate output. .assign() merges new computed fields into the original dictionary, preserving all upstream state."),
    ("What are the advantages of writing Chain Factories instead of global singleton chain instances?",
     "Chain factories allow parameterized configuration (adjusting model checkpoint, temperature, or system prompts dynamically). Furthermore, in multi-threaded web applications, factories ensure clean instance encapsulation and prevent accidental parameter mutation across concurrent requests."),
    ("How does streaming work when combining two sequential chains via LCEL?",
     "Streaming tokens flow progressively through the first chain into the consumer. However, if Step 2 requires the complete completed output of Step 1 to format its prompt, Step 2 cannot begin generating until Step 1 finishes. LangChain handles this dependency automatically without deadlocking.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Incompatible Input/Output Signatures in Chains. Chaining `chain_a | chain_b` where `chain_a` outputs a plain string and `chain_b` expects a dictionary with `{query}` raises an immediate `TypeError`. Always use adapters or dictionaries.")
    st.error("**Trap 2 (Breaking Error)**: Dropping System Prompts in Composed Sub-Chains. When composing multiple sub-chains, failing to specify system prompts on intermediate model calls causes downstream models to lose context and tone consistency.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Redundant Model Calls in Multi-Step Chains. Splitting a single task into 4 sequential model calls when a single well-structured prompt could achieve the same result in 1 call multiplies token cost and adds 3–5 seconds of latency.")
    st.warning("**Trap 4 (Quality Warning)**: Hallucination Compounding in Sequential Chains. If Step 1 generates a hallucinated fact and Step 2 accepts it as ground truth, the final output will compound the error. Add validation or assertion checks between stages.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: The Canonical Triad"):
        st.markdown("**Prompt | Model | Parser represents the fundamental cognitive pipeline unit.**")
    with st.expander("🎴 Flashcard 2: Prompt | Model Return Type"):
        st.markdown("**Returns an AIMessage containing content, tool_calls, and usage_metadata.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: State Accumulation"):
        st.markdown("**RunnablePassthrough.assign(key=subchain) preserves upstream dictionary variables.**")
    with st.expander("🎴 Flashcard 4: Chain Factory Pattern"):
        st.markdown("**Functions returning RunnableSequence instances encapsulate configurable parameters.**")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can construct the Canonical Triad (Prompt | Model | Parser) using LCEL.", value=True)
chk2 = st.checkbox("I know when to omit the output parser to inspect AIMessage metadata.", value=True)
chk3 = st.checkbox("I can orchestrate multi-step chains using RunnablePassthrough.assign().", value=True)
chk4 = st.checkbox("I understand how to build parameterized chain factories.", value=True)
chk5 = st.checkbox("I know how to hierarchically compose multiple sub-chains into a master workflow.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 6.2 Building Chains. Ready for 6.3 Advanced Chains!")
