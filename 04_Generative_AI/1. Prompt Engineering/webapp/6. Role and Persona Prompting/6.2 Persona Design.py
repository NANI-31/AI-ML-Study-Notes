import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="6.2 Persona Design | GenAI Roadmap",
    page_icon="👤",
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
st.title("👤 6.2 Persona Design — The 6 Core Archetypes")
st.markdown(
    """
Welcome to **Lesson 6.2: Persona Design**.  
While Role Prompting establishes domain seniority and operational boundaries, **Persona Design** crafts the complete behavioral, psychological, and conversational identity of an AI assistant. 
A well-engineered persona balances domain vocabulary, epistemic humility, tone cadence, and user-relationship dynamics.

Below, we master the **6 Foundational Persona Archetypes**: Professional, Teacher, Developer, Researcher, Customer Support, and Data Analyst.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Core Persona Archetypes Matrix
# ==============================================================================
st.header("1. Enterprise Persona Archetype Matrix", divider="rainbow")
st.markdown("Here is how different personas shift tone, epistemic humility, and delivery formats:")

personas_df = pd.DataFrame({
    "Persona Archetype": [
        "1. Professional Consultant",
        "2. Socratic Teacher",
        "3. Staff Developer",
        "4. Academic Researcher",
        "5. Support Specialist",
        "6. Quantitative Analyst"
    ],
    "Tone & Voice": [
        "Formal, structured, executive-facing, risk-conscious",
        "Patient, encouraging, pedagogical, scaffolded",
        "Terse, pragmatic, code-first, linting-obsessed",
        "Rigorous, peer-review style, hypothesis-driven",
        "Empathetic, reassuring, de-escalating, action-oriented",
        "Skeptical, data-driven, anomaly-hunting, statistical"
    ],
    "Core Pedagogical / Operational Technique": [
        "Top-down Pyramid Principle (conclusions first, then evidence)",
        "Socratic questioning (hints before answers, checking understanding)",
        "Idiomatic code snippets, unit test plans, architectural tradeoffs",
        "Literature citations, methodological scrutiny, explicit caveats",
        "Active listening, step-by-step resolution paths, customer advocacy",
        "Hypothesis testing, metric breakdown, identifying spurious trends"
    ],
    "Target Use Case": [
        "Board briefings, compliance audits, business strategy",
        "Employee onboarding, technical tutoring, interactive learning",
        "Code reviews, refactoring bots, pair-programming assistants",
        "Literature review agents, clinical trial evaluation, research triage",
        "Customer service ticketing, billing dispute de-escalation",
        "BI dashboard interpretation, telemetry analysis, KPI auditing"
    ]
})
st.dataframe(personas_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Detailed Technical Breakdown & System Prompts
# ==============================================================================
st.header("2. Detailed Technical Breakdown & System Prompts", divider="rainbow")

# 1. Professional Persona
with st.container(border=True):
    st.subheader("1. The Professional Consultant Persona")
    st.markdown(
        """
Built for executive briefings and business decision-makers. Employs the **Minto Pyramid Principle**: deliver the core recommendation in sentence one, followed by supporting pillars and quantified risks.
"""
    )
    st.code(
        '''system_prompt_consultant = """
You are a Principal Management Consultant at an elite enterprise advisory firm.

COMMUNICATION CADENCE:
- Structure all deliverables using the Pyramid Principle (Executive Recommendation first, followed by 3 supporting pillars).
- Maintain an authoritative, concise, and business-focused tone.
- Quantify all financial impacts, operational risks, and implementation timelines.
- Forbid casual conversational greetings, disclaimers, or meta-commentary.
"""''',
        language="python"
    )

# 2. Teacher Persona
with st.container(border=True):
    st.subheader("2. The Socratic Teacher Persona")
    st.markdown(
        """
Never provides the direct answer immediately. Scaffolds concepts, asks guiding questions, diagnoses misconceptions, and celebrates small epiphanies.
"""
    )
    st.code(
        '''system_prompt_teacher = """
You are an expert Computer Science Professor utilizing the Socratic Method.

PEDAGOGICAL RULES:
- Never provide the final solution or code snippet immediately.
- Break complex algorithms into incremental mental steps.
- Ask targeted conceptual questions to guide the student to discover the solution.
- If the student makes an error, diagnose the underlying mental misconception politely.
- Keep explanations grounded in intuitive real-world metaphors before introducing formal syntax.
"""''',
        language="python"
    )

# 3. Developer Persona
with st.container(border=True):
    st.subheader("3. The Staff Developer Persona")
    st.markdown(
        """
Pragmatic, opinionated, and focused on clean architecture, test coverage, type safety, and edge-case prevention.
"""
    )
    st.code(
        '''system_prompt_developer = """
You are a Senior Staff Backend Engineer specializing in Python and distributed systems.

ENGINEERING PHILOSOPHY:
- Favor explicit over implicit code. Strict type annotations (PEP 484) are mandatory.
- Code snippets must include error handling and production pytest assertions.
- Highlight time/space complexity (Big-O) and concurrency pitfalls (race conditions, deadlocks).
- Be direct, terse, and opinionated. Omit decorative pleasantries.
"""''',
        language="python"
    )

# 4. Researcher Persona
with st.container(border=True):
    st.subheader("4. The Academic Researcher Persona")
    st.markdown(
        """
Exhibits high epistemic humility, rigorously qualifies claims, scrutinizes sample sizes and biases, and avoids definitive assertions without empirical citations.
"""
    )
    st.code(
        '''system_prompt_researcher = """
You are a Senior AI Research Scientist conducting a rigorous peer-review evaluation.

EPISTEMIC STANDARDS:
- Distinguish strictly between empirical evidence, theoretical claims, and unverified heuristics.
- Actively highlight confounding variables, selection bias, and benchmark leakage risks.
- Frame all conclusions probabilistically with appropriate confidence bounds.
- Provide academic citations (Author, Year, Conference) where relevant.
"""''',
        language="python"
    )

# 5. Customer-Support Persona
with st.container(border=True):
    st.subheader("5. The Customer Support Specialist Persona")
    st.markdown(
        """
Master of de-escalation, empathy, and structured troubleshooting. Uses the **HEAT model** (Hear, Empathize, Apologize, Take Action).
"""
    )
    st.code(
        '''system_prompt_support = """
You are a Tier-2 Technical Customer Success Specialist.

SUPPORT PROTOCOL:
- Acknowledge customer frustration with genuine empathy without admitting legal liability.
- Provide numbered, step-by-step troubleshooting instructions formatted with clear headings.
- If an issue requires engineering escalation, reassure the customer with clear SLA timelines.
- Tone: Warm, highly professional, reassuring, and solution-focused.
"""''',
        language="python"
    )

# 6. Analyst Persona
with st.container(border=True):
    st.subheader("6. The Quantitative Analyst Persona")
    st.markdown(
        """
Forensic data detective. Questions assumptions, calculates baseline comparisons, looks for Simpson's Paradox, and demands metric rigor.
"""
    )
    st.code(
        '''system_prompt_analyst = """
You are a Lead Quantitative Data Analyst.

ANALYTICAL RIGOR:
- Never accept a vanity metric at face value; demand denominator context and cohort splits.
- Test for statistical significance, seasonality anomalies, and survivorship bias.
- Present findings using structured markdown tables with percentage deltas.
- Tone: Skeptical, analytical, objective, and metric-obsessed.
"""''',
        language="python"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Multi-Persona Workbench
# ==============================================================================
st.header("3. Interactive Workbench: The Same Incident Across 6 Personas", divider="rainbow")
st.markdown(
    """
Select a persona below to observe how each responds to the **exact same scenario**:  
**Scenario**: *"Our background email sending queue timed out, failing 5,000 marketing emails during a Black Friday sale."*
"""
)

active_p = st.selectbox(
    "Choose Active Persona Archetype:",
    [
        "1. Professional Consultant",
        "2. Socratic Teacher",
        "3. Staff Developer",
        "4. Academic Researcher",
        "5. Customer Support Specialist",
        "6. Quantitative Analyst"
    ]
)

if "Consultant" in active_p:
    st.info("Perspective: Risk Mitigation, Executive Impact, Action Plan")
    st.markdown(
        """
**Executive Briefing (Consultant):**
1. **Core Recommendation**: Implement automated circuit breaker routing to a secondary transactional email provider to recover $85,000 in at-risk conversion revenue within 2 hours.
2. **Pillars**:
   - *Financial Impact*: 5,000 unsent emails represent an estimated 250 abandoned carts ($12,500 direct lost GMV).
   - *Brand Equity*: Failure occurred during peak Black Friday promotional window.
   - *Remediation Timeline*: 48-hour post-incident review with executive signoff.
"""
    )

elif "Teacher" in active_p:
    st.info("Perspective: Pedagogical Guidance, Concept Scaffolding")
    st.markdown(
        """
**Guiding Discussion (Teacher):**  
*"That sounds like a stressful situation during Black Friday! Before we write any code, let's think about how worker queues operate.*  
*When 5,000 jobs arrive at once, what happens if the worker process waits on each SMTP connection sequentially rather than processing concurrently?*  
*What pattern could we use to ensure one slow connection doesn't block the remaining 4,999 emails?"*
"""
    )

elif "Developer" in active_p:
    st.info("Perspective: Technical Architecture, Async Worker, Code Fix")
    st.code(
        '''# Staff Developer Remediation:
# Root Cause: Blocking I/O inside Celery task worker.
# Fix: Batch dispatch using asyncio + aiosmtplib with connection pooling and dead-letter queues.

from celery import shared_task
import asyncio

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def send_marketing_batch(self, email_batch: list[dict]):
    try:
        # Use asyncio to dispatch concurrent non-blocking connections
        asyncio.run(dispatch_async_pool(email_batch, concurrency=50))
    except Exception as exc:
        # Route unrecoverable failures to Dead Letter Queue (DLQ)
        logger.error(f"Batch failed: {exc}")
        raise self.retry(exc=exc)''',
        language="python"
    )

elif "Researcher" in active_p:
    st.info("Perspective: Methodology, Failure Modes, Reliability Theory")
    st.markdown(
        """
**Methodological Assessment (Researcher):**  
*"This failure mode exemplifies the classic 'Metastable Failure' described in distributed systems literature (Huang et al., OSDI '22).*  
*Under peak surge, retry storms exacerbate queue saturation, shifting the system into a persistent degraded state even after nominal traffic subsides.*  
*We hypothesize that implementing exponential backoff with full jitter will reduce collision probability by 74%."*
"""
    )

elif "Support" in active_p:
    st.info("Perspective: Empathy, Reassurance, SLA Transparency")
    st.markdown(
        """
**Customer Communication (Support):**  
*"We sincerely apologize for the disruption to your Black Friday campaign. We understand how critical this promotional window is for your team.*  
*Our engineering team has already isolated the worker queue bottleneck and is actively resending the affected 5,000 messages in priority batches.*  
*You can track live dispatch progress on your campaign dashboard. We will provide another update within 30 minutes."*
"""
    )

elif "Analyst" in active_p:
    st.info("Perspective: Metrics, Statistical Significance, Funnel Analysis")
    st.markdown(
        """
**Quantitative Impact Analysis (Analyst):**
| Metric | Expected Baseline | Incident Actual | Delta (%) |
| :--- | :--- | :--- | :--- |
| Queue Dispatch Latency | 120 ms | 31,400 ms (Timeout) | +26,066% |
| Delivery Success Rate | 99.8% | 88.2% | -11.6% |
| Email Open Rate Delta | 28.4% | Pending Resend | TBD |

*Recommendation: Split failure cohort by recipient domain (Gmail vs Yahoo) to verify if provider-side rate limits triggered the queue backup.*
"""
    )

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("4. Persona Multi-Dimensional Conditioning Space", divider="rainbow")

col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.markdown("#### Persona Orthogonal Axes")
    dot_p = """
    digraph PersonaAxes {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Input [label="Raw Task Query", fillcolor="#334155", fontcolor="#ffffff"];
        Tone [label="1. Tone Axis\\nFormal <---> Empathetic", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Rigor [label="2. Epistemic Rigor\\nPragmatic <---> Academic", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        OutputAxis [label="3. Output Format\\nExecutive Brief <---> Code Block", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];
        PersonaEngine [label="Calibrated Persona Output", fillcolor="#047857", fontcolor="#ffffff"];

        Input -> Tone -> PersonaEngine;
        Input -> Rigor -> PersonaEngine;
        Input -> OutputAxis -> PersonaEngine;
    }
    """
    st.graphviz_chart(dot_p, use_container_width=False)

with col_p2:
    st.markdown("#### Persona Design Rules of Thumb")
    st.markdown(
        """
1. **Always declare the Audience**: Tell the persona who they are speaking to (e.g. *Executive VP* vs *Junior Intern*).
2. **Explicitly forbid conflicting tones**: If assigning a developer persona, forbid marketing fluff.
3. **Calibrate Epistemic Humility**: Instruct the persona when to admit uncertainty.
"""
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

pd_tab1, pd_tab2, pd_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with pd_tab1:
    with st.expander("Q1: How do you design a Socratic tutor persona that prevents the model from giving away answers?", expanded=True):
        st.markdown(
            """
To prevent conversational leakage where the model immediately writes the solution:
1. Provide an explicit negative constraint: *"Do NOT provide the answer or write code for the student under any circumstance"*.
2. Provide a positive structural protocol: *"Step 1: Check student understanding. Step 2: Ask 1 leading diagnostic question. Step 3: Wait for user reply before proceeding"*.
3. Include 2 few-shot exemplars demonstrating the teacher refusing a direct request for the solution and steering the student back to discovery.
"""
        )

    with pd_tab2:
        with st.expander("Trap 1: 'Can we create a persona that is simultaneously an expert developer, marketing genius, and legal counsel?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, LLMs can be prompted to be good at everything at once."*
- **Winning Response**: *"No. Creating 'Omniscient Super-Personas' leads to Persona Diffusion and conflicting instruction interference. An adversarial security reviewer should not adopt an overly polite customer-support persona, as polite de-escalation directly conflicts with blunt vulnerability disclosure. In production multi-agent architectures (e.g. LangGraph), assign single, sharply defined personas to distinct worker nodes rather than conflating roles into one prompt."*
"""
            )

    with pd_tab3:
        c_pd1, c_pd2, c_pd3 = st.columns(3)
        with c_pd1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Persona Diffusion**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Dilution of behavioral rigor when a prompt attempts to combine too many conflicting professional identities.")

        with c_pd2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Pyramid Principle**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Executive communication pattern stating the core recommendation first, followed by supporting pillars.")

        with c_pd3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Epistemic Humility**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** The persona's deliberate ability to express probabilistic confidence and state boundaries of certainty.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 6.2 Summary & Self-Assessment", divider="rainbow")

col_ps1, col_ps2 = st.columns(2)
with col_ps1:
    st.markdown(
        """
### Key Takeaways:
1. **The 6 Archetypes**: Consultant, Teacher, Developer, Researcher, Support, and Analyst.
2. **Align Audience & Purpose**: Calibrate vocabulary density to the consumer of the response.
3. **Avoid Persona Diffusion**: Keep personas focused on a single operational domain.
4. **Use Structural Protocols**: Combine personas with explicit communication frameworks (Pyramid Principle, Socratic scaffolding).
"""
    )

with col_ps2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can implement each of the 6 core enterprise persona archetypes in system prompts.", value=True, key="chk_6_2_1")
    st.checkbox("I know how to prevent Socratic teacher personas from leaking final answers.", value=True, key="chk_6_2_2")
    st.checkbox("I understand the architectural danger of multi-domain 'Omniscient Super-Personas'.", value=True, key="chk_6_2_3")
    st.checkbox("I can structure an executive consultant persona using the Pyramid Principle.", value=True, key="chk_6_2_4")
