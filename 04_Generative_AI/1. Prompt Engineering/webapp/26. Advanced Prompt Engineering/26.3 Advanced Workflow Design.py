import streamlit as st
import json

st.set_page_config(
    page_title="26.3 Advanced Workflow Design - Prompt Engineering",
    page_icon="🔄",
    layout="wide"
)

# Header Section
st.title("26.3 Advanced Workflow Design: Pipelines, Fan-Out, & Human Gates")
st.caption("Chapter 26: Advanced Prompt Engineering | Domain: 1. Prompt Engineering")
st.markdown("""
Mission-critical generative AI systems require orchestration patterns borrowed from distributed systems engineering. 
Building resilient systems involves **sequential prompt pipelines, conditional branch routing, parallel map-reduce fan-outs, 
circuit breaker fallbacks, and human-in-the-loop (HITL) approval gates**.
""")

st.divider()

# Section 1: Workflow Topology Blueprint
st.header("1. Core Orchestration Topologies", divider="rainbow")
st.markdown("""
Modern compound systems combine three foundational control-flow topologies:
""")

col_top1, col_top2, col_top3 = st.columns(3)
with col_top1:
    st.subheader("1. Sequential Pipeline")
    st.markdown("""
    Output of stage A becomes input to stage B:
    * `Raw Input` -> `De-identification` -> `Extraction` -> `Translation` -> `Final Format`.
    * Unit-testable at each discrete boundary.
    """)

with col_top2:
    st.subheader("2. Parallel Fan-Out / Fan-In")
    st.markdown("""
    Execute multiple independent prompts concurrently:
    * **Map-Reduce**: 10 document sections summarized in parallel, then combined into 1 executive memo.
    * Slashes end-to-end latency by up to 80%.
    """)

with col_top3:
    st.subheader("3. Human-in-the-Loop Gate")
    st.markdown("""
    Autonomous execution pauses at critical decision nodes:
    * Refunds > $500, legal actions, or public email dispatch require explicit human admin sign-off before proceeding.
    """)

st.divider()

# Section 2: Circuit Breaker & Fallback Architecture
st.header("2. Circuit Breakers and Graceful Degradation", divider="rainbow")
st.markdown("""
When primary cloud inference providers suffer outages or rate-limit saturation, a circuit breaker pattern 
diverts traffic instantly to secondary backup engines or cached heuristics:
""")

st.graphviz_chart('''
digraph CircuitBreaker {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    UserReq [label="Production Request", fillcolor="#E1BEE7"];
    Breaker [label="Circuit Breaker Check\\n(Failure rate < 5%?)", shape=diamond, fillcolor="#FFF9C4"];
    Primary [label="Primary Provider\\n(OpenAI GPT-4o)", fillcolor="#BBDEFB"];
    Secondary [label="Secondary Fallback\\n(Anthropic Claude 3.5)", fillcolor="#FFE0B2"];
    LocalFallback [label="Local Heuristic Fallback\\n(Cached rules / Default response)", fillcolor="#FFCDD2"];
    
    UserReq -> Breaker;
    Breaker -> Primary [label="CLOSED (Healthy)"];
    Breaker -> Secondary [label="OPEN (Tripped)"];
    Secondary -> LocalFallback [label="If Secondary fails"];
}
''', use_container_width=False)

st.code('''# Production Fallback Pattern in Python
def robust_model_dispatch(prompt_payload: dict, client_pool: dict) -> str:
    """
    Attempts Primary provider; automatically cascades to Secondary on 5xx or timeout.
    """
    try:
        # 1. Attempt Primary (e.g. OpenAI GPT-4o)
        return client_pool["openai"].invoke(prompt_payload, timeout=5.0)
    except Exception as e_primary:
        # Log warning to telemetry
        print(f"Primary provider failed: {e_primary}. Cascading to Secondary...")
        
        try:
            # 2. Attempt Secondary Fallback (e.g. Anthropic Claude 3.5 Sonnet)
            return client_pool["anthropic"].invoke(prompt_payload, timeout=5.0)
        except Exception as e_secondary:
            # 3. Emergency Graceful Degradation
            print(f"CRITICAL: All providers failed: {e_secondary}. Engaging safe fallback.")
            return "Our AI assistant is temporarily undergoing maintenance. Please reach out to support@domain.com."''', language="python")

st.divider()

# Section 3: Parallel Map-Reduce Implementation
st.header("3. Parallel Map-Reduce Execution Pattern", divider="rainbow")
st.markdown("""
When processing a 100-page regulatory report, sequential prompt execution would take 5 minutes. 
Using `concurrent.futures.ThreadPoolExecutor`, chunks are summarized simultaneously in under 8 seconds:
""")

st.code('''# Parallel Fan-Out Summarization
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

def parallel_map_reduce_summary(document_sections: List[str], max_workers: int = 5) -> str:
    # Stage 1: Parallel Map (Fan-Out)
    section_summaries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(summarize_single_section, sec): idx 
            for idx, sec in enumerate(document_sections)
        }
        for future in as_completed(future_to_idx):
            section_summaries.append(future.result())
            
    # Stage 2: Reduce (Fan-In Synthesis)
    joined_summaries = "\\n\\n".join(section_summaries)
    final_memo = synthesize_master_memo(joined_summaries)
    return final_memo''', language="python")

st.divider()

# Section 4: Interactive Workflow Simulator
st.header("4. Interactive HITL Workflow Simulator", divider="rainbow")
st.markdown("Simulate an enterprise financial transaction pipeline with an automated Human Approval Gate.")

trans_amount = st.number_input("Transaction Amount ($)", value=1250, step=250)
trans_risk = st.selectbox("Risk Classification:", ["LOW (Known Vendor)", "MEDIUM (International Vendor)", "HIGH (Unverified Account)"])

st.markdown("#### Workflow Execution State:")

if trans_amount > 1000 or trans_risk == "HIGH (Unverified Account)":
    st.warning("⚠️ **Human Approval Gate Tripped**: Transaction exceeds $1,000 threshold or is flagged as HIGH risk.")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.write("AI Evaluation: Transaction requires manual compliance manager review.")
        approval = st.radio("Compliance Officer Verdict:", ["Pending Review", "Approved by Human Admin", "Rejected by Human Admin"])
        if approval == "Approved by Human Admin":
            st.success("✅ Workflow Resumed: Payment processed following authorized human override.")
        elif approval == "Rejected by Human Admin":
            st.error("🛑 Workflow Terminated: Payment halted by compliance policy.")
        else:
            st.info("⏳ Workflow Paused: Awaiting human reviewer sign-off.")
    with col_a2:
        st.code('''{
  "transaction_id": "TXN-9021",
  "status": "PAUSED_FOR_APPROVAL",
  "reason": "Amount > $1000 or High Risk",
  "requires_role": "Compliance_Officer"
}''', language="json")
else:
    st.success("⚡ **Autonomous Execution**: Transaction cleared all safety parameters and was processed automatically in 240ms.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: What is the Circuit Breaker pattern in LLM architectures and what are its states?"):
        st.write("""
        The Circuit Breaker pattern prevents an application from repeatedly executing an operation that is failing:
        * **CLOSED (Normal)**: Requests pass through to primary LLM provider. Failures are counted.
        * **OPEN (Tripped)**: If failure rate exceeds a threshold (e.g. 5 errors in 10s), the breaker trips OPEN. All requests bypass primary provider immediately and route to fallback model or cache without waiting for timeouts.
        * **HALF-OPEN (Testing)**: After a cooldown (e.g. 30s), limited trial requests test if primary provider has recovered. If successful, breaker resets to CLOSED.
        """)

    with st.expander("Q: Why is Map-Reduce preferred over long-context ingestion for large multi-document synthesis?"):
        st.write("""
        * **Latency**: Map-Reduce runs $N$ chunks in parallel, reducing latency from $O(N)$ to $O(1)$ wall-clock time.
        * **Attention Quality**: Prevents 'Lost-in-the-Middle' degradation by forcing the model to attend to each isolated section with 100% focused attention.
        * **Cost Predictability**: Errors in one section only require retrying that single chunk rather than re-ingesting 200,000 tokens.
        """)

with qa2:
    with st.expander("Trap to Avoid: Blocking User Threads on Long Asynchronous Chains"):
        st.write("""
        Never run multi-step agent chains or map-reduce jobs synchronously on the user's HTTP request thread. 
        HTTP connections time out after 30-60 seconds. Always offload complex workflows to an asynchronous background worker 
        (e.g., Celery, Temporal, or AWS SQS) and return a task ID so the frontend can poll or receive a WebSocket push.
        """)

    with st.expander("Flashcards: Advanced Workflows"):
        fc_sel = st.selectbox(
            "Select Workflow Concept:",
            ["Parallel Fan-Out", "Circuit Breaker", "Human-in-the-Loop Gate", "Graceful Degradation"]
        )
        if fc_sel == "Parallel Fan-Out":
            st.info("Dispatching multiple prompt executions simultaneously across worker threads.")
        elif fc_sel == "Circuit Breaker":
            st.info("A mechanism that halts traffic to an overloaded provider to allow recovery.")
        elif fc_sel == "Human-in-the-Loop Gate":
            st.info("A checkpoint in an autonomous workflow where execution blocks pending human sign-off.")
        else:
            st.info("Providing a simplified or cached response when primary AI inference clusters are offline.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I can implement parallel Map-Reduce prompt execution using ThreadPoolExecutor.")
    st.checkbox("I know how to design multi-provider fallback cascades with Circuit Breakers.")
with c2:
    st.checkbox("I understand how to design Human-in-the-Loop approval gates for high-stakes actions.")
    st.checkbox("I know when to offload long prompt chains to asynchronous background task queues.")
