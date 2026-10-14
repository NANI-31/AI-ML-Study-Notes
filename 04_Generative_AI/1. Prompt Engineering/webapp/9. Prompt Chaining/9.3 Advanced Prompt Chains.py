import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="9.3 Advanced Prompt Chains | GenAI Roadmap",
    page_icon="⚡",
    layout="wide",
)

# Custom Styling for GPU-accelerated sidebar & Glassmorphic cards
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .badge-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("⚡ 9.3 Advanced Prompt Chains — Parallelism, Branching & HITL")
st.markdown(
    """
Welcome to **Lesson 9.3: Advanced Prompt Chains**.  
Linear pipelines (`A -> B -> C`) are only the beginning of prompt chain architecture. 
Production enterprise systems frequently require **asynchronous parallel execution (fan-out/fan-in)**, **dynamic conditional branching**, **circuit-breaker fallbacks**, and **Human-in-the-Loop (HITL) approval gates**.

Below, we master the **6 Advanced Orchestration Patterns** that transform isolated prompts into resilient, enterprise-grade AI workflow graphs.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Advanced Orchestration Patterns Matrix
# ==============================================================================
st.header("1. The 6 Advanced Orchestration Patterns", divider="rainbow")
st.markdown("Advanced chaining patterns navigate concurrency, failure recovery, and governance:")

adv_patterns_df = pd.DataFrame({
    "Pattern": [
        "1. Conditional Chains",
        "2. Parallel Chains (Fan-Out/Fan-In)",
        "3. Branching DAG Workflows",
        "4. Fallback Prompts",
        "5. Exponential Backoff Retries",
        "6. Human-in-the-Loop (HITL)"
    ],
    "Architectural Mechanism": [
        "Dynamic routing based on upstream boolean outputs or confidence thresholds",
        "Concurrently executing independent prompts using `asyncio.gather()`",
        "Directed Acyclic Graph (DAG) state machines with join and merge points",
        "Degrading gracefully to lightweight models or rule-based heuristics on error",
        "Automated self-healing retry loops with jitter for rate limits and syntax slips",
        "Pausing execution for human approval if confidence drops below SLA threshold"
    ],
    "Performance & Reliability Impact": [
        "Executes only required pipeline stages, avoiding wasted compute",
        "Cuts total multi-step latency from 3x to 1x by running tasks in parallel",
        "Enables complex multi-agent reasoning and cross-functional coordination",
        "Prevents user-facing 500 crashes; ensures high availability",
        "Provides 99.9% pipeline completion rate despite intermittent network glitches",
        "Guarantees compliance and safety in high-stakes legal, medical, or financial tasks"
    ]
})
st.dataframe(adv_patterns_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Detailed Technical Breakdown & Code Blueprints
# ==============================================================================
st.header("2. Detailed Technical Breakdown & Code Blueprints", divider="rainbow")

# 2. Parallel Chains (Fan-Out / Fan-In)
with st.container(border=True):
    st.subheader("2. Parallel Chains (Fan-Out / Fan-In with asyncio)")
    st.markdown(
        """
When evaluating a document across multiple independent criteria (e.g. Security, Performance, and Legal Compliance), **do not run them sequentially**. 
Execute all three prompts concurrently using `asyncio.gather()`, then aggregate their results in a final synthesis prompt.
"""
    )
    st.code(
        '''import asyncio
import openai

client = openai.AsyncOpenAI()

async def analyze_security(code: str) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Audit security vulnerabilities: {code}"}]
    )
    return resp.choices[0].message.content

async def analyze_performance(code: str) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Audit time/space bottlenecks: {code}"}]
    )
    return resp.choices[0].message.content

async def run_parallel_audit_pipeline(code_snippet: str) -> str:
    # Fan-Out: Run independent audits simultaneously in parallel!
    sec_task = analyze_security(code_snippet)
    perf_task = analyze_performance(code_snippet)
    
    sec_report, perf_report = await asyncio.gather(sec_task, perf_task)
    
    # Fan-In: Synthesize aggregated findings into final report
    summary = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Synthesize final PR review:\\nSECURITY:\\n{sec_report}\\n\\nPERF:\\n{perf_report}"
        }]
    )
    return summary.choices[0].message.content''',
        language="python"
    )

# 6. Human-in-the-Loop (HITL)
with st.container(border=True):
    st.subheader("6. Human-in-the-Loop (HITL) Gating")
    st.markdown(
        """
In enterprise compliance and finance, automated actions exceeding risk thresholds must pause and await human manager authorization.
"""
    )
    st.code(
        '''def process_refund_with_hitl_gate(claim_text: str):
    # Step 1: Automated extraction & fraud scoring
    assessment = call_fraud_evaluation_prompt(claim_text)
    
    # Step 2: HITL Threshold Check
    if assessment.refund_amount_usd > 1000.0 or assessment.confidence_score < 0.85:
        # Route to human review queue (pause execution)
        ticket_id = create_zendesk_manager_ticket(assessment)
        return {
            "status": "AWAITING_HUMAN_REVIEW",
            "ticket_id": ticket_id,
            "message": "Your refund claim has been forwarded to a manager for signoff."
        }
    else:
        # Automated execution path
        execute_automated_stripe_refund(assessment)
        return {"status": "AUTO_APPROVED", "refund_id": assessment.claim_id}''',
        language="python"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Parallel Fan-Out Workbench
# ==============================================================================
st.header("3. Interactive Workbench: Parallel Multi-Expert Audit", divider="rainbow")
st.markdown(
    """
Witness the speed and quality advantage of **Parallel Fan-Out Execution**:  
A codebase pull request is audited simultaneously by 3 specialized mini-models (Security, Scalability, Cost) and merged into a unified report.
"""
)

sample_code_pr = """@app.post("/upload")
async def handle_upload(file: UploadFile):
    content = await file.read()
    # Save directly to /tmp without size limit
    with open(f"/tmp/{file.filename}", "wb") as f:
        f.write(content)
    # Query database synchronously inside async route
    user = db.query(User).filter(User.id == 1).first()
    return {"status": "ok"}"""

st.code(sample_code_pr, language="python")

col_fan1, col_fan2, col_fan3 = st.columns(3)

with col_fan1:
    with st.container(border=True):
        st.markdown("#### 🛡️ Security Audit (Parallel)")
        st.error("P0 Risk: Arbitrary file write / Path traversal via unsanitized `file.filename`.")

with col_fan2:
    with st.container(border=True):
        st.markdown("#### ⚡ Scalability Audit (Parallel)")
        st.warning("P1 Bottleneck: Blocking synchronous DB call inside async event loop; no upload byte cap (OOM risk).")

with col_fan3:
    with st.container(border=True):
        st.markdown("#### 💰 Cost / Cloud Audit (Parallel)")
        st.info("P2 Finding: Unbounded local disk writes on container ephemeral storage.")

st.markdown("#### Synthesized Executive Review (Fan-In Aggregation):")
st.success(
    """
**PR Status: CHANGES_REQUESTED (Confidence: 99.2%)**
1. **Critical Security**: Sanitize filenames using `uuid.uuid4()` to prevent path traversal exploits.
2. **Architecture**: Implement `file.size` streaming checks with a 10MB ceiling.
3. **Async Performance**: Replace blocking `db.query` with asynchronous SQLAlchemy session `await session.execute()`.
"""
)

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("4. Asynchronous Fan-Out & HITL Topology", divider="rainbow")

col_par1, col_par2 = st.columns([1, 1])

with col_par1:
    st.markdown("#### Fan-Out / Fan-In + HITL Gate Diagram")
    dot_hitl = """
    digraph AdvancedChaining {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Input [label="User / Document Input", fillcolor="#334155", fontcolor="#ffffff"];

        subgraph cluster_parallel {
            label = "Concurrent Fan-Out (asyncio.gather)";
            color = "#38bdf8";
            style = "dashed";
            P1 [label="Task 1: Security Audit", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
            P2 [label="Task 2: Performance Audit", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
            P3 [label="Task 3: Cost Audit", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        }

        FanIn [label="Fan-In Aggregator\\n(Synthesize Findings)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Gate [label="Confidence >= 90% AND\\nRisk < $1000?", shape=diamond, fillcolor="#1e293b", fontcolor="#facc15", color="#facc15"];

        AutoDeploy [label="Automated Action Execution", fillcolor="#047857", fontcolor="#ffffff"];
        HumanReview [label="Pause -> Human Approval Queue", fillcolor="#991b1b", fontcolor="#ffffff"];

        Input -> P1;
        Input -> P2;
        Input -> P3;
        P1 -> FanIn;
        P2 -> FanIn;
        P3 -> FanIn;
        FanIn -> Gate;
        Gate -> AutoDeploy [label="Yes"];
        Gate -> HumanReview [label="No (High Risk)"];
    }
    """
    st.graphviz_chart(dot_hitl, use_container_width=False)

with col_par2:
    st.markdown("#### Latency Math: Sequential vs Parallel")
    st.markdown(
        """
- **Sequential Execution**:  
  `Task 1 (600ms) + Task 2 (550ms) + Task 3 (620ms) + Merge (400ms)`  
  = **2,170 ms Total Latency**
- **Parallel Fan-Out Execution (`asyncio.gather`)**:  
  `max(600ms, 550ms, 620ms) + Merge (400ms)`  
  = **1,020 ms Total Latency (53% faster!)**
"""
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

ac_tab1, ac_tab2, ac_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with ac_tab1:
    with st.expander("Q1: How do you implement automated retry logic in prompt chains without causing cascading rate limit failures?", expanded=True):
        st.markdown(
            """
1. **Exponential Backoff with Full Jitter**: Calculate sleep delay as $2^{\\text{attempt}} + \\text{random}(0, 1)$, preventing 'thundering herd' spikes against LLM provider rate limits.
2. **Specific Error Categorization**: Retry on transient network errors (HTTP 429, 502, 503) and schema parse validation failures; do NOT retry on deterministic client errors (HTTP 401 Unauthorized, 400 Bad Request).
3. **Strict Circuit Breaker Ceiling**: Cap maximum retries at 3 before falling back to a deterministic rule or human review queue.
"""
        )

    with ac_tab2:
        with st.expander("Trap 1: 'Can we parallelize prompts that have step-by-step data dependencies?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, asyncio can run any list of functions at the same time."*
- **Winning Response**: *"No. Parallel execution (fan-out) requires that tasks be mutually independent—meaning Task B does not require the output of Task A as an input parameter. If Task B depends on Task A (e.g. you cannot draft a resolution email until the claim is extracted and verified), those links MUST execute sequentially. In complex DAGs, parallelize only the independent sub-branches (e.g. concurrent document audits), and join them at synchronization merge barriers."*
"""
            )

    with ac_tab3:
        c_ac1, c_ac2, c_ac3 = st.columns(3)
        with c_ac1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Fan-Out / Fan-In**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Running independent prompts concurrently in parallel, then aggregating results in a single synthesis prompt.")

        with c_ac2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Human-in-the-Loop (HITL)**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Pausing automated pipeline execution when confidence scores drop below a threshold to require human manager signoff.")

        with c_ac3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Fallback Prompt**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Gracefully degrading to a simpler zero-shot model or deterministic heuristic when the primary pipeline fails.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 9.3 Summary & Self-Assessment", divider="rainbow")

col_acs1, col_acs2 = st.columns(2)
with col_acs1:
    st.markdown(
        """
### Key Takeaways:
1. **Parallelism Cuts Latency**: Use `asyncio.gather()` for independent evaluations to cut latency in half.
2. **HITL Governance**: Gate high-risk actions behind human manager review queues.
3. **Resilient Retries**: Implement exponential backoff with jitter on transient failures.
4. **Fallback Degradation**: Have a backup plan when frontier APIs experience downtime.
"""
    )

with col_acs2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can implement asynchronous fan-out/fan-in prompt execution using `asyncio.gather()`.", value=True, key="chk_9_3_1")
    st.checkbox("I know how to design a Human-in-the-Loop threshold gate for automated actions.", value=True, key="chk_9_3_2")
    st.checkbox("I understand the difference between parallelizable subtasks and sequential dependencies.", value=True, key="chk_9_3_3")
    st.checkbox("I can implement exponential backoff retry logic for LLM API calls.", value=True, key="chk_9_3_4")
