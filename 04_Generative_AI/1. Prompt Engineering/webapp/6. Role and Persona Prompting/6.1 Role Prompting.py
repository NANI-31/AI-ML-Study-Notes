import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="6.1 Role Prompting | GenAI Roadmap",
    page_icon="🎭",
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
st.title("🎭 6.1 Role Prompting — Conditioning System Priors & Expert Perspectives")
st.markdown(
    """
Welcome to **Lesson 6.1: Role Prompting**.  
In natural language generation, an LLM possesses a vast, unconditioned latent probability space covering everything from casual Reddit comments to peer-reviewed academic literature. 
Without guidance, default completions tend toward generic, middle-of-the-road conversational prose.

**Role Prompting** is the deliberate engineering technique of instructing an LLM to adopt a specific professional identity, responsibility boundary, domain expertise, and communication style.
Below, we dissect how role conditioning operates in Transformer latent spaces and build production-grade expert personas.
"""
)

st.divider()

# ==============================================================================
# Section 1: What is Role Prompting & How It Operates Internally
# ==============================================================================
st.header("1. What is Role Prompting & Attention Conditioning", divider="rainbow")

col_rp1, col_rp2 = st.columns([1, 1])

with col_rp1:
    st.markdown(
        """
### Definition
**Role Prompting** is the practice of framing an LLM's operational context with a designated professional persona (*"You are a Principal Staff Cloud Security Architect"*), establishing its domain vantage point, vocabulary register, and analytical rigor.

### The Mechanism of Semantic Conditioning:
1. **Latent Space Steering**: Pre-training data contains distinct linguistic clusters (legal corpora, medical trials, code review threads). Specifying a role shifts the initial state vector toward the corresponding domain manifold.
2. **Vocabulary Calibration**: Activating an expert persona increases the logit probability of domain-specific technical terminology (e.g. *idempotency*, *asymmetric routing*, *P99 latency*) while suppressing colloquial fluff.
3. **Perspective Alignment**: Directs the model's analytical focus toward domain-specific priorities (a security role prioritizes threat vectors; a finance role prioritizes margin impact).
"""
    )

with col_rp2:
    st.markdown("### Attention Latent Steering (ASCII)")
    st.code(
        """UNCONDITIONED PROMPT:
"Explain microservices"
  -> High probability: General consumer overview, simple metaphors ("like a restaurant kitchen").

ROLE-CONDITIONED PROMPT:
"You are a Distributed Systems Architect specializing in zero-trust banking."
  -> High probability: RPC serialization overhead, mTLS mesh, consensus algorithms (Raft/Paxos).

LATENT MANIFOLD SHIFT:
[Casual Chat Space] ───(Role Tokens: 'Staff Architect')───> [High-Rigor Tech Space]""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 2: The 4 Core Anatomy Pillars of an Expert Role
# ==============================================================================
st.header("2. The 4 Pillars of an Enterprise Role Definition", divider="rainbow")
st.markdown("A vague role (*'You are an expert'*) fails. Production roles must define 4 specific structural dimensions:")

role_pillars_df = pd.DataFrame({
    "Pillar": [
        "1. Expert Identity & Title",
        "2. Scope of Responsibilities",
        "3. Technical Expertise & Standards",
        "4. Communication Style & Register"
    ],
    "Architectural Purpose": [
        "Specifies exact professional seniority, domain authority, and organizational tier",
        "Declares what the persona is responsible for AND explicitly forbidden from doing",
        "Declares familiarity with specific industry standards, frameworks, and acronyms",
        "Calibrates tone (direct, adversarial, pedagogical), conciseness, and formatting"
    ],
    "Example Specification": [
        "'Principal Staff Site Reliability Engineer for High-Frequency Trading'",
        "'Conduct root-cause analysis; do NOT write marketing copy or business fluff'",
        "'Expert in Linux kernel tracing (eBPF), TCP socket tuning, and P99.9 latency'",
        "'Terse, direct, evidence-based; format recommendations as prioritized action tickets'"
    ]
})
st.dataframe(role_pillars_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Production Code Blueprints for Enterprise Roles
# ==============================================================================
st.header("3. Production Role System Prompts", divider="rainbow")
st.markdown("Here is how production systems craft fully qualified role prompts across diverse enterprise domains:")

st.code(
    '''# =============================================================================
# Role 1: Principal Security Architect (Adversarial Threat Modeler)
# =============================================================================
system_security_role = """
You are a Principal Cloud Security Architect conducting an adversarial STRIDE threat assessment.

EXPERTISE & STANDARDS:
- Zero-Trust Architecture (NIST SP 800-207), OAuth 2.1, OIDC, AWS IAM least-privilege.
- Deep familiarity with OWASP Top 10 API Security Risks.

RESPONSIBILITIES:
- Identify unmitigated attack surfaces and privilege escalation vectors.
- Assign severity ratings (P0-P3) according to CVSS v3.1 impact guidelines.
- Do NOT validate business viability or UX convenience; focus strictly on exploitability.

COMMUNICATION STYLE:
- Direct, clinical, and precise.
- Format every identified vulnerability as: [SEVERITY | ATTACK VECTOR | IMPACT | MITIGATION].
"""

# =============================================================================
# Role 2: Staff Pediatric Triage Nurse (Clinical Grounding)
# =============================================================================
system_medical_triage_role = """
You are a Staff Pediatric Emergency Triage Nurse with 15 years of trauma center experience.

EXPERTISE:
- Pediatric Advanced Life Support (PALS) protocols, Emergency Severity Index (ESI 1-5).

RESPONSIBILITIES:
- Screen parent inquiries for red-flag pediatric emergencies (stridor, cyanosis, lethargy).
- Provide immediate first-aid guidance pending physician consultation.
- Mandatory Rule: If emergency indicators are present, output immediate 911 alert instructions first.

COMMUNICATION STYLE:
- Calm, empathetic, yet urgent and unambiguous. No medical jargon without plain-English explanation.
"""''',
    language="python"
)

st.divider()

# ==============================================================================
# Section 4: Interactive Role Calibration Workbench
# ==============================================================================
st.header("4. Interactive Workbench: The Multi-Perspective Lens", divider="rainbow")
st.markdown(
    """
Witness how the **exact same incident** is analyzed completely differently depending on the role assigned to the LLM:  
**Incident**: A critical database connection pool exhaustion outage during peak checkout.
"""
)

role_choice = st.selectbox(
    "Choose Active Role Persona:",
    [
        "1. Principal Staff Site Reliability Engineer (SRE)",
        "2. Chief Information Security Officer (CISO)",
        "3. Chief Financial Officer (CFO)",
        "4. VP of Product Management"
    ]
)

if "SRE" in role_choice:
    st.info("Perspective: Infrastructure, Concurrency, Kernel Limits, Pool Metrics")
    st.code(
        """ANALYSIS (Principal SRE):
- Root Cause: Connection starvation due to idle database transactions holding row locks beyond client read timeout.
- Immediate Action:
  1. Reduce `idle_in_transaction_session_timeout` from 300s to 15s.
  2. Implement PgBouncer transaction-mode connection pooling.
  3. Configure TCP keepalives to drop stale dead sockets.
- Reliability Metric: Maintain connection saturation < 70% during 3x burst traffic.""",
        language="text"
    )

elif "CISO" in role_choice:
    st.warning("Perspective: Threat Surface, Denial-of-Service, Access Abuse, Audit Logs")
    st.code(
        """ANALYSIS (CISO):
- Threat Vector: Denial-of-Service (DoS) via resource exhaustion (CWE-400).
- Security Audit:
  1. Verify if anomalous IP clusters flooded the checkout endpoint with unauthenticated requests.
  2. Check WAF rate-limiting telemetry on `/v1/checkout/pay`.
  3. Ensure database error stack traces did not leak connection strings to the public client.
- Compliance Action: Document incident for annual SOC2 Type II availability criteria.""",
        language="text"
    )

elif "CFO" in role_choice:
    st.error("Perspective: Revenue Loss, Customer Churn, SLA Penalties, Budget Allocation")
    st.code(
        """ANALYSIS (CFO):
- Financial Impact:
  1. Direct Gross Merchandise Value (GMV) loss: $42,000 during 18-minute outage window.
  2. Enterprise SLA breach credits: $8,500 owed to tier-1 enterprise merchants.
- Capital Allocation:
  - Approve $15,000/quarter budget increase for multi-AZ database read replica auto-scaling.
  - Require ROI payback justification based on transaction abandonment recovery.""",
        language="text"
    )

elif "Product" in role_choice:
    st.success("Perspective: User Experience, Customer Trust, Checkout Flow, Brand Sentiment")
    st.code(
        """ANALYSIS (VP of Product):
- User Impact:
  1. 850 shoppers encountered generic 504 error screens; high risk of permanent brand churn.
  2. Cart abandonment rate spiked to 68%.
- Product Remediation:
  - Replace ugly 504 server error page with a friendly "High Demand - Saving Your Cart" waiting room.
  - Implement automated customer recovery email offering 10% coupon for affected carts.""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 5: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("5. Role Prior Conditioning Topology", divider="rainbow")

col_rt1, col_rt2 = st.columns([1, 1])

with col_rt1:
    st.markdown("#### Prior Probability Steering Diagram")
    dot_role = """
    digraph RoleConditioning {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Role [label="System Role Anchor\\n'Senior Penetration Tester'", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Standards [label="Domain Knowledge\\n'OWASP API Top 10'", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        Style [label="Style Calibration\\n'Adversarial, CVE Citing'", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];

        Attention [label="Multi-Head Cross-Attention Layer\\n(Shapes Logits Prior Distribution)", fillcolor="#047857", fontcolor="#ffffff"];
        Output [label="Domain-Aligned Expert Output", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];

        Role -> Attention;
        Standards -> Attention;
        Style -> Attention;
        Attention -> Output;
    }
    """
    st.graphviz_chart(dot_role, use_container_width=False)

with col_rt2:
    st.markdown("#### The Role Calibration Spectrum")
    st.markdown(
        """
- **Weak Role**: *"You are an expert programmer"* -> Generic, verbose, uses beginner explanations.
- **Strong Role**: *"You are a Rust Compiler Engineer optimizing SIMD intrinsics"* -> High-density, zero fluff, uses inline assembly and benchmark profiling data.
"""
    )

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

r_tab1, r_tab2, r_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with r_tab1:
    with st.expander("Q1: What are the mathematical/representational mechanics behind role prompting?", expanded=True):
        st.markdown(
            """
Transformers predict tokens autoregressively by computing attention weights over the prompt context. 
When tokens like *"Senior Site Reliability Engineer"* or *"Patent Attorney"* are in the prefix, the early self-attention layers activate high attention weights around domain-specific semantic sub-networks in the feed-forward layers. 
This acts as a soft Bayesian prior, conditioning the output probabilities toward vocabulary, syntactic patterns, and reasoning templates characteristic of that profession in the training corpus.
"""
        )

    with r_tab2:
        with st.expander("Trap 1: 'Does assigning a persona endow the model with factual knowledge it never learned?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, telling the model it is a cardiologist unlocks medical expertise."*
- **Winning Response**: *"No. Role prompting only modulates tone, vocabulary register, and analytical framing; it cannot create facts or reasoning pathways absent from the pre-training weights. If the model does not possess medical knowledge, role prompting may actually induce more confident, convincing hallucinations. Factual accuracy requires grounding via RAG or fine-tuning; role prompting only calibrates the persona."*
"""
            )

    with r_tab3:
        c_rp1, c_rp2, c_rp3 = st.columns(3)
        with c_rp1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Role Prompting**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Conditioning the model's vocabulary, perspective, and analytical rigor using a designated professional identity.")

        with c_rp2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Tone vs Capability**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Roles calibrate communicative style and viewpoint; they do NOT inject new factual knowledge.")

        with c_rp3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Role Specificity**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Concrete titles ('Staff SRE for HFT') vastly outperform vague adjectives ('expert programmer').")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 6.1 Summary & Self-Assessment", divider="rainbow")

col_rs1, col_rs2 = st.columns(2)
with col_rs1:
    st.markdown(
        """
### Key Takeaways:
1. **Semantic Prior Steering**: Roles condition vocabulary registers and perspective in latent space.
2. **The 4 Pillars**: Title, Responsibilities, Expertise Standards, and Communication Style.
3. **Calibrates Tone, Not Truth**: Roles do not replace factual grounding (RAG).
4. **Be Specific**: Concrete seniority and domain frameworks yield the highest analytical rigor.
"""
    )

with col_rs2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can explain how role prompts shift token probability distributions in Transformer models.", value=True, key="chk_6_1_1")
    st.checkbox("I know how to define an enterprise role using the 4 structural pillars.", value=True, key="chk_6_1_2")
    st.checkbox("I understand the distinction between role calibration and factual knowledge injection.", value=True, key="chk_6_1_3")
    st.checkbox("I can structure specialized system prompts for security, medical, and financial roles.", value=True, key="chk_6_1_4")
