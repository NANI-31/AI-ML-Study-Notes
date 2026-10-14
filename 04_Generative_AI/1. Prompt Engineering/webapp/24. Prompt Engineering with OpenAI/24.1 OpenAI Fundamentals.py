import streamlit as st
import json

st.set_page_config(
    page_title="24.1 OpenAI Fundamentals - Prompt Engineering",
    page_icon="🤖",
    layout="wide"
)

# Header Section
st.title("24.1 OpenAI Platform Architecture & Model Foundations")
st.caption("Chapter 24: Prompt Engineering with OpenAI | Domain: 1. Prompt Engineering")
st.markdown("""
OpenAI is the industry benchmark for commercial generative AI inference. 
Mastering prompt engineering on the OpenAI platform requires an intimate understanding of its **model families 
(GPT-4o, GPT-4o-mini, o1/o3 reasoning models), message abstraction hierarchies (system/developer vs user vs assistant), 
and tokenization mechanisms**.
""")

st.divider()

# Section 1: OpenAI Model Families & Capabilities
st.header("1. OpenAI Model Families and Capability Matrix", divider="rainbow")
st.markdown("""
Modern OpenAI models are optimized for distinct engineering workloads:
""")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.subheader("GPT-4o (Omni)")
    st.markdown("""
    * **Role**: Flagship multimodal foundation model.
    * **Capabilities**: Text, vision, audio natively. High-speed reasoning with 128k context window.
    * **Best For**: Complex synthesis, vision analysis, general agentic tool calling.
    """)

with col_m2:
    st.subheader("GPT-4o-mini")
    st.markdown("""
    * **Role**: Lightweight, ultra-fast, cost-effective workhorse.
    * **Economics**: Over 60% cheaper than GPT-3.5 Turbo with superior MMLU scores.
    * **Best For**: High-throughput extraction, classification, query rewriting, and intent routing.
    """)

with col_m3:
    st.subheader("o1 & o3 (Reasoning Series)")
    st.markdown("""
    * **Role**: Test-time compute reasoning models.
    * **Mechanics**: Employs internal hidden Chain-of-Thought (RL-trained) before emitting final tokens.
    * **Best For**: Competitive programming, complex math, PhD-level chemistry, and multi-step theorem proving.
    """)

st.divider()

# Section 2: Messages Hierarchy & Developer Roles
st.header("2. The Message Abstraction: System, Developer, User, & Assistant", divider="rainbow")
st.markdown("""
In the OpenAI API, prompts are structured as an ordered array of message objects. 
Recent models introduce the specialized `developer` role alongside traditional `system` instructions:
""")

st.graphviz_chart('''
digraph MessageHierarchy {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    DevRole [label="developer / system Message\\n'Top-Level Governance'\\nSets core persona, constraints, safety rules & schemas", fillcolor="#C8E6C9"];
    UserRole [label="user Message\\n'Untrusted Dynamic Input'\\nContains current user question, document payload, or query", fillcolor="#BBDEFB"];
    AsstRole [label="assistant Message\\n'Model Generation / Exemplar'\\nProvides prior model turns or few-shot demonstration history", fillcolor="#FFF9C4"];
    ToolRole [label="tool Message\\n'Deterministic Feedback'\\nReturns raw JSON responses from executed function tools", fillcolor="#E1BEE7"];
    
    DevRole -> UserRole -> AsstRole -> ToolRole;
}
''', use_container_width=False)

st.markdown("""
* **`developer` role**: In reasoning models (like `o1`), OpenAI replaced `system` with `developer` to make clear that instructions 
come from the application creator, possessing higher instruction hierarchy than untrusted user inputs.
* **`user` role**: Carries user inquiries and dynamic document context.
* **`assistant` role**: Represents model completions or few-shot demonstration targets.
* **`tool` role**: Carries function execution returns back to the model for synthesis.
""")

st.divider()

# Section 3: Official OpenAI Python SDK Setup
st.header("3. Production OpenAI SDK Implementation Pattern", divider="rainbow")
st.markdown("""
Production applications interact with OpenAI via the official typed `openai` Python SDK:
""")

st.code('''# Official OpenAI Python SDK Pattern (v1.x+)
from openai import OpenAI
import os

# Client initialization reads OPENAI_API_KEY from environment automatically
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_retries=3,
    timeout=20.0
)

def analyze_customer_feedback(feedback_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a customer sentiment analyst. Extract sentiment and actionable recommendations."
            },
            {
                "role": "user",
                "content": f"Customer Feedback: {feedback_text}"
            }
        ],
        temperature=0.1,
        max_tokens=300
    )
    
    # Extract completion text and token telemetry
    content = response.choices[0].message.content
    usage = response.usage
    return {
        "analysis": content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }''', language="python")

st.divider()

# Section 4: Interactive OpenAI Model Selector Workbench
st.header("4. Interactive OpenAI Model Selector Workbench", divider="rainbow")
st.markdown("Select a task type to determine the optimal OpenAI model family, estimated latency, and cost profile.")

task_type = st.selectbox(
    "Choose Production Workload:",
    [
        "High-Volume E-Commerce Intent Classification (1M req/day)",
        "Complex Legal Contract Multi-Clause Analysis",
        "Autonomous Multi-Step Python Code Generation & Debugging",
        "Real-Time Live Chat Customer Support Bot"
    ]
)

col_s1, col_s2 = st.columns(2)

with col_s1:
    if task_type == "High-Volume E-Commerce Intent Classification (1M req/day)":
        rec_model = "gpt-4o-mini"
        reason = "Classification requires fast inference and minimal token cost. GPT-4o-mini achieves >95% accuracy at 1/15th the price of GPT-4o."
        est_cost = "$0.15 per 1,000 queries"
        est_lat = "~200ms"
    elif task_type == "Complex Legal Contract Multi-Clause Analysis":
        rec_model = "gpt-4o"
        reason = "Dense multi-page legal reasoning requires high attention capacity and low hallucination across long contexts."
        est_cost = "$2.50 per 1,000 queries"
        est_lat = "~800ms"
    elif task_type == "Autonomous Multi-Step Python Code Generation & Debugging":
        rec_model = "o1 / o3-mini"
        reason = "Test-time compute reasoning excels at algorithmic synthesis, boundary edge-cases, and debugging deep logic chains."
        est_cost = "$6.00 per 1,000 queries"
        est_lat = "~2,500ms (Internal CoT reasoning)"
    else:
        rec_model = "gpt-4o-mini (with streaming)"
        reason = "User-facing chat demands sub-300ms TTFT to feel natural. GPT-4o-mini provides conversational fluency with minimal latency."
        est_cost = "$0.20 per 1,000 queries"
        est_lat = "~150ms TTFT"

    st.success(f"**Recommended Model**: `{rec_model}`")
    st.write(f"**Architecture Rationale**: {reason}")

with col_s2:
    st.subheader("Workload Economics & SLA")
    st.metric("Estimated Cost Profile", est_cost)
    st.metric("Estimated P50 Latency", est_lat)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: How do OpenAI's o1/o3 reasoning models fundamentally differ in prompt design compared to GPT-4o?"):
        st.write("""
        * **No Chain-of-Thought Prompting Needed**: You should NOT prompt o1 with 'Think step by step' because the model already has 
        an internal RL-trained hidden reasoning mechanism. Explicit CoT instructions can degrade performance or cause redundant tokens.
        * **Developer vs System Role**: o1 uses the `developer` role instead of `system`.
        * **No Temperature Tuning**: Temperature is fixed at 1.0 (or restricted) because sampling affects internal reasoning traces.
        * **Output Limits**: Max tokens include both internal reasoning tokens (hidden) and visible completion tokens.
        """)

    with st.expander("Q: What is the purpose of the `seed` parameter in the OpenAI API?"):
        st.write("""
        The `seed` parameter (integer) requests deterministic sampling from OpenAI's inference clusters. 
        When paired with identical `messages`, `temperature=0`, and model version, the provider makes a best-effort attempt 
        to return identical completions, logging a `system_fingerprint` in the response header to verify cluster backend consistency.
        """)

with qa2:
    with st.expander("Trap to Avoid: Pinning to Floating Model Aliases in Production"):
        st.write("""
        Calling `model="gpt-4o"` points to a dynamic pointer that OpenAI updates periodically. 
        When OpenAI deploys a checkpoint update, prompt behavior, formatting, or refusal boundaries can subtly shift overnight. 
        In enterprise production, always pin to immutable dated snapshots (e.g., `model="gpt-4o-2024-08-06"`) 
        and run automated benchmark suites before migrating to new versions.
        """)

    with st.expander("Flashcards: OpenAI Ecosystem"):
        fc_sel = st.selectbox(
            "Select OpenAI Term:",
            ["system_fingerprint", "Developer Role", "o1 Test-Time Compute", "Model Snapshots"]
        )
        if fc_sel == "system_fingerprint":
            st.info("Identifier returned by OpenAI indicating the specific backend configuration that served the request.")
        elif fc_sel == "Developer Role":
            st.info("Top-level governance role in reasoning models that supersedes user inputs.")
        elif fc_sel == "o1 Test-Time Compute":
            st.info("Models that generate hidden internal reasoning tokens before emitting the final visible answer.")
        else:
            st.info("Immutable dated model releases (e.g. gpt-4o-2024-08-06) that protect pipelines from unexpected drift.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the trade-offs between GPT-4o, GPT-4o-mini, and o1.")
    st.checkbox("I know why prompt engineering for reasoning models (o1) forbids manual CoT prompting.")
with c2:
    st.checkbox("I can structure multi-turn messages across developer, user, assistant, and tool roles.")
    st.checkbox("I understand why enterprise systems pin to dated model snapshot versions.")
