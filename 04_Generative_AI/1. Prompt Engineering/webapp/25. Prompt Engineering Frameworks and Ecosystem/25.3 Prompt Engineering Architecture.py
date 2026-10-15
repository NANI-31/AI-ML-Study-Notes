import streamlit as st
import json

st.set_page_config(
    page_title="25.3 Prompt Engineering Architecture - Prompt Engineering",
    page_icon="🏛️",
    layout="wide"
)

# Header Section
st.title("25.3 The 7-Layer Prompt Engineering Architecture")
st.caption("Chapter 25: Prompt Engineering Frameworks and Ecosystem | Domain: 1. Prompt Engineering")
st.markdown("""
Enterprise generative AI systems are not monolithic API calls; they are **layered software architectures**. 
Designing a robust production system requires separating concerns across seven distinct tiers: 
from raw compute (LLM Layer) and semantic grounding (Retrieval Layer) up to state management (Memory Layer) 
and continuous quality assurance (Evaluation Layer).
""")

st.divider()

# Section 1: The 7-Layer Architecture Blueprint
st.header("1. The 7-Layer Architecture Blueprint", divider="rainbow")
st.markdown("""
Each layer in the 7-tier stack has an explicit contract and boundary:
""")

st.graphviz_chart('''
digraph SevenLayerArchitecture {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    L7 [label="Layer 7: Evaluation & Observability\\n(CI/CD Benchmarks, G-Eval, Langfuse Tracing, Cost Governance)", fillcolor="#FFCDD2"];
    L6 [label="Layer 6: Agent & Orchestration\\n(Goal Decomposition, Plan & Execute, ReAct loops, LangGraph)", fillcolor="#FFE0B2"];
    L5 [label="Layer 5: Memory & State Layer\\n(Short-Term Rolling Window, Long-Term Semantic Episodic Memory)", fillcolor="#FFF9C4"];
    L4 [label="Layer 4: Tool & Action Execution\\n(Pydantic Schemas, API Gateways, Sandboxed Python Runners)", fillcolor="#E1BEE7"];
    L3 [label="Layer 3: Retrieval & Knowledge Grounding\\n(Vector DBs, Hybrid BM25, Neural Cross-Encoder Rerankers)", fillcolor="#C8E6C9"];
    L2 [label="Layer 2: Prompt Layer\\n(Templates, Few-Shot Exemplars, System Guardrails, Dynamic Injectors)", fillcolor="#BBDEFB"];
    L1 [label="Layer 1: LLM & Inference Engine\\n(Model Routing, vLLM, OpenAI, Quantized Snapshots, Logit Processors)", fillcolor="#CFD8DC"];
    
    L7 -> L6 -> L5 -> L4 -> L3 -> L2 -> L1;
}
''', use_container_width=False)

st.divider()

# Section 2: Deep Dive into the 7 Layers
st.header("2. Architectural Responsibilities by Layer", divider="rainbow")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Layer 1: Foundation LLM Layer")
    st.markdown("""
    * **Responsibilities**: Host inference, manage KV attention caches, apply token decoders, and handle model failovers (e.g. falling back from GPT-4o to Claude 3.5 Sonnet).
    * **Key Concern**: Throughput (tokens/sec), concurrency, and GPU VRAM economics.
    """)

    st.subheader("Layer 2: Prompt Template Layer")
    st.markdown("""
    * **Responsibilities**: Manage versioned prompt templates, dynamically inject user variables, format few-shot exemplars, and enforce structural delimiters.
    * **Key Concern**: Prompt version control, preventing delimiter collision, and prompt injection hardening.
    """)

    st.subheader("Layer 3: Retrieval & Grounding Layer")
    st.markdown("""
    * **Responsibilities**: Index domain documents, compute dense embeddings, execute hybrid lexical + vector search, and rerank candidates.
    * **Key Concern**: Semantic relevance, latency, and context window compression.
    """)

    st.subheader("Layer 4: Tool & Action Layer")
    st.markdown("""
    * **Responsibilities**: Register typed API functions, validate LLM tool payloads with Pydantic, execute external calls safely, and return results.
    * **Key Concern**: Parameter typing, privilege separation, timeouts, and sandboxed security.
    """)

with col2:
    st.subheader("Layer 5: Memory & State Layer")
    st.markdown("""
    * **Responsibilities**: Maintain conversational context across multi-turn sessions using sliding windows, token summarizers, or vector-based episodic recall.
    * **Key Concern**: Preventing context window saturation and preserving user preferences over time.
    """)

    st.subheader("Layer 6: Agent & Orchestration Layer")
    st.markdown("""
    * **Responsibilities**: Control flow logic, goal planning, reflection, loop termination heuristics, and coordinating multi-agent handoffs.
    * **Key Concern**: Cycle detection, preventing infinite loops, and task progress monitoring.
    """)

    st.subheader("Layer 7: Evaluation & Observability Layer")
    st.markdown("""
    * **Responsibilities**: Log end-to-end spans with trace IDs, compute real-time quality metrics (faithfulness, latency), and alert on regressions.
    * **Key Concern**: Data privacy, PII masking, cost budgeting, and production SLAs.
    """)

st.divider()

# Section 3: Clean Separation of Concerns Pattern
st.header("3. Clean Architecture Implementation Pattern", divider="rainbow")
st.markdown("""
Below is an enterprise clean architecture implementation illustrating strict layer separation:
""")

st.code('''# Clean Architecture 7-Layer Orchestration Pattern
from dataclasses import dataclass
from typing import List, Dict

# Layer 2 & 4: Typed Contracts
@dataclass
class PromptContext:
    system_template: str
    user_query: str
    retrieved_docs: List[str]
    session_history: List[Dict[str, str]]

class EnterpriseAIApplication:
    def __init__(self, llm_gateway, retriever, memory_store, tracer):
        self.llm = llm_gateway        # Layer 1
        self.retriever = retriever    # Layer 3
        self.memory = memory_store    # Layer 5
        self.tracer = tracer          # Layer 7

    def handle_request(self, session_id: str, user_query: str) -> str:
        # Layer 7: Start Trace Span
        with self.tracer.start_span("UserRequest", session_id=session_id):
            
            # Layer 5: Load Session History
            history = self.memory.get_recent_history(session_id, last_k=5)
            
            # Layer 3: Retrieve Grounded Documents
            docs = self.retriever.retrieve(user_query, top_k=3)
            
            # Layer 2: Compile Structured Prompt
            formatted_prompt = self._compile_prompt(user_query, docs, history)
            
            # Layer 1: Execute LLM Call with Fallback Routing
            response = self.llm.generate(formatted_prompt)
            
            # Layer 5: Commit Turn to Memory
            self.memory.append_turn(session_id, user_query, response)
            
            return response

    def _compile_prompt(self, query, docs, history) -> str:
        # Pure string assembly in Layer 2
        joined_docs = "\\n".join(f"- {d}" for d in docs)
        return f"<context>{joined_docs}</context>\\n<query>{query}</query>"''', language="python")

st.divider()

# Section 4: Interactive Architecture Health Checker
st.header("4. Interactive Architecture Health Checker", divider="rainbow")
st.markdown("Audit your application's architectural maturity against the 7-Layer enterprise benchmark.")

st.subheader("Maturity Assessment Questionnaire")
q1 = st.checkbox("Layer 1: Do you have automatic model routing or fallback logic for provider outages?", value=True)
q2 = st.checkbox("Layer 2: Are your prompt templates version-controlled in Git rather than hardcoded strings?", value=True)
q3 = st.checkbox("Layer 3: Does your retrieval layer employ two-stage search (hybrid + reranker)?", value=False)
q4 = st.checkbox("Layer 4: Are all tool inputs strictly validated using Pydantic schemas?", value=True)
q5 = st.checkbox("Layer 5: Does your memory layer summarize long histories to prevent token bloat?", value=False)
q6 = st.checkbox("Layer 6: Do your agent workflows enforce hard step limits to prevent infinite loops?", value=True)
q7 = st.checkbox("Layer 7: Do you track distributed traces and cost per query in CI/CD and production?", value=True)

score = sum([q1, q2, q3, q4, q5, q6, q7])
st.divider()

col_sc1, col_sc2 = st.columns(2)
with col_sc1:
    st.metric("Architecture Maturity Score", f"{score} / 7")
with col_sc2:
    if score >= 6:
        st.success("🏆 Tier 3 Enterprise Grade: Exceptional decoupling and resilience across all layers.")
    elif score >= 4:
        st.info("⚙️ Tier 2 Production Grade: Solid foundation; consider hardening retrieval and memory summarization.")
    else:
        st.warning("⚠️ Tier 1 Prototype: High architectural coupling. Refactor into modular layers before enterprise launch.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: Why is separating the Prompt Layer from the LLM Layer critical for enterprise engineering?"):
        st.write("""
        Decoupling the Prompt Layer (templates, schema specifications, few-shot injection) from the LLM Layer (model inference, 
        API endpoints, connection pools) allows teams to:
        1. Swap inference providers (e.g., migrating from OpenAI to self-hosted vLLM) without rewriting application prompts.
        2. Version-control prompts independently in CI/CD pipelines.
        3. Run A/B test variations of prompt templates against the exact same underlying model checkpoints.
        """)

    with st.expander("Q: What failure modes emerge when the Memory Layer lacks token summarization?"):
        st.write("""
        Without summarization or sliding window truncation, conversation history grows linearly with every user turn. 
        Eventually:
        1. The prompt exceeds the context window, causing 400 Bad Request errors.
        2. Input token costs explode quadratically.
        3. Attention degradation ('Lost in the Middle') occurs, causing the model to forget instructions from earlier in the session.
        """)

with qa2:
    with st.expander("Trap to Avoid: Coupling Tool Execution Directly into the Prompt Layer"):
        st.write("""
        Never write prompt generation functions that directly invoke external database queries or APIs inline. 
        Tool execution belongs strictly in Layer 4 (Actions). Keeping the Prompt Layer purely functional (taking data in, 
        outputting formatted strings) ensures prompts can be unit-tested deterministically without mocking 10 external services.
        """)

    with st.expander("Flashcards: 7-Layer Architecture"):
        fc_sel = st.selectbox(
            "Select Architectural Layer:",
            ["Layer 1: LLM Engine", "Layer 3: Retrieval", "Layer 5: Memory", "Layer 7: Observability"]
        )
        if fc_sel == "Layer 1: LLM Engine":
            st.info("Manages model endpoints, retry policies, and hardware inference backends.")
        elif fc_sel == "Layer 3: Retrieval":
            st.info("Fetches grounded reference knowledge via vector search, hybrid indices, and rerankers.")
        elif fc_sel == "Layer 5: Memory":
            st.info("Maintains session state, entity memory, and summary condensation across conversation turns.")
        else:
            st.info("Captures distributed traces, latency SLAs, quality scores, and cost economics.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I can enumerate all 7 layers of modern prompt engineering architecture.")
    st.checkbox("I know how to decouple prompt template generation from model inference.")
with c2:
    st.checkbox("I understand the role of the Memory Layer in preventing context window exhaustion.")
    st.checkbox("I can audit an LLM system for layer separation and enterprise resilience.")
