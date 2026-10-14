import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="7.2 Providing Context | GenAI Roadmap",
    page_icon="📚",
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
st.title("📚 7.2 Providing Context — The 6 Enterprise Information Layers")
st.markdown(
    """
Welcome to **Lesson 7.2: Providing Context**.  
An LLM without context is merely an educated guesser relying on broad, frozen web training. 
To build reliable enterprise AI systems, developers must inject **rich, structured, multi-layer contextual payloads** directly into the model's working memory at runtime.

Below, we master the **6 Essential Contextual Information Layers**: Background domain information, Reference documentation, User history, Business rules, Demonstrative examples, and Real-time external knowledge.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Information Layers Framework
# ==============================================================================
st.header("1. The 6 Contextual Information Layers", divider="rainbow")
st.markdown("Enterprise applications layer distinct operational data streams into the context window:")

layers_df = pd.DataFrame({
    "Context Layer": [
        "1. Background Information",
        "2. Reference Documents",
        "3. User & Session History",
        "4. Business Rules & Logic",
        "5. Exemplars & Demonstrations",
        "6. External Tool & API Knowledge"
    ],
    "Data Source": [
        "Static application metadata, tech stack versions, corporate identity",
        "RAG vector chunks (PDFs, Confluence pages, API specifications)",
        "Chat history, recent shopping cart actions, authenticated user ID/tier",
        "Regulatory compliance mandates, operational SOPs, discount rules",
        "Curated input-output pairs illustrating complex edge cases",
        "Live SQL query results, weather APIs, real-time stock quotes"
    ],
    "Attention Function": [
        "Sets macro operational scope and establishes vocabulary standards",
        "Provides the factual source of truth to eliminate hallucinations",
        "Maintains multi-turn conversational coherence and personalization",
        "Enforces deterministic decision constraints on ambiguous queries",
        "Calibrates output syntax, schema keys, and reasoning style",
        "Supplies transient facts that did not exist at model training time"
    ]
})
st.dataframe(layers_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Detailed Technical Breakdown & Code Blueprints
# ==============================================================================
st.header("2. Detailed Technical Breakdown & Code Blueprints", divider="rainbow")

# 1 & 2: Background & Reference Documents
with st.container(border=True):
    st.subheader("1 & 2. Background Information & Reference Documents (RAG)")
    st.markdown(
        """
Background info establishes the macro scope, while reference documents ground specific answers. 
Always encapsulate reference documents within explicit `<reference_docs>` tags accompanied by source identifiers.
"""
    )
    st.code(
        '''prompt = """
### BACKGROUND:
You are an internal IT assistant for Acme Financial Services. 
Our infrastructure runs AWS EKS Kubernetes clusters with PostgreSQL 16 databases.

### REFERENCE DOCUMENTS:
<reference_docs>
[DOC_ID: RUNBOOK-402 | TITLE: Database Failover Protocol]
In the event of an automated primary database failover, standby read-replicas 
promote within 30 seconds. Application pods must drain existing connection pools 
and establish fresh TCP handshakes to the new primary endpoint.
</reference_docs>

### USER INQUIRY:
Why did our backend pods throw connection reset errors after the 03:00 UTC database maintenance?
"""''',
        language="text"
    )

# 3 & 4: User History & Business Rules
with st.container(border=True):
    st.subheader("3 & 4. Session History & Deterministic Business Rules")
    st.markdown(
        """
Business rules must be framed as **hard logical predicates** that override conversational defaults. 
Combine user session state (e.g. loyalty tier) directly with explicit conditional policy.
"""
    )
    st.code(
        '''prompt = """
### BUSINESS RULES:
- Tier Gold/Platinum: Free shipping on all orders; return window 60 days.
- Tier Standard: Free shipping on orders >= $50; return window 30 days.
- Final Sale items (SKU starting with "FS-"): Non-refundable under all circumstances.

### USER SESSION CONTEXT:
Customer Name: Robert Chen
Loyalty Tier: Standard
Cart Subtotal: $42.50
Item in Question: Linen Shirt (SKU: FS-9012)

QUESTION:
Can I return this shirt if it doesn't fit, and do I get free shipping?
"""''',
        language="text"
    )

# 6: External Knowledge & Live API Injection
with st.container(border=True):
    st.subheader("6. Real-Time External Knowledge & API Results")
    st.markdown(
        """
Inject live real-time API responses directly into the prompt context. 
This bridges the gap between static training cutoffs and live operational telemetry.
"""
    )
    st.code(
        '''import requests
import json

# Fetch live operational metrics from Datadog / Prometheus API
live_telemetry = {
    "cluster_cpu_utilization_pct": 94.2,
    "active_worker_pods": 18,
    "pending_jobs_queue": 1420,
    "hpa_target_reached": True
}

prompt = f"""
### REAL-TIME CLUSTER TELEMETRY (Timestamp: 14:02:18 UTC):
```json
{json.dumps(live_telemetry, indent=2)}
```

TASK:
Based on this live telemetry snapshot, explain why new background workers 
are not auto-scaling and provide the kubectl command to inspect pod resource limits.
"""''',
        language="python"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Context Layering Simulator
# ==============================================================================
st.header("3. Interactive Workbench: The Context Layering Simulator", divider="rainbow")
st.markdown("Toggle each of the 6 contextual layers on and off to observe how prompt completeness and precision evolve:")

col_l1, col_l2, col_l3 = st.columns(3)
with col_l1:
    inc_bg = st.checkbox("Layer 1: Background Info", value=True)
    inc_docs = st.checkbox("Layer 2: Reference Docs", value=True)
with col_l2:
    inc_hist = st.checkbox("Layer 3: User Session State", value=True)
    inc_rules = st.checkbox("Layer 4: Business Rules", value=True)
with col_l3:
    inc_ex = st.checkbox("Layer 5: Few-Shot Examples", value=False)
    inc_api = st.checkbox("Layer 6: Live API Telemetry", value=True)

st.markdown("#### Dynamically Assembled Context Payload:")

prompt_layers = []

if inc_bg:
    prompt_layers.append("### [LAYER 1: BACKGROUND]\nYou are an automated logistics triage assistant for GlobalFreight Express (Europe Region).")

if inc_docs:
    prompt_layers.append("""### [LAYER 2: REFERENCE DOCUMENTATION]
<kb_articles>
[KB-102]: Customs clearance for UK-to-EU freight requires an electronic EORI number and commercial invoice. Missing EORI triggers automated customs holding at Port of Dover.
</kb_articles>""")

if inc_hist:
    prompt_layers.append("""### [LAYER 3: USER SESSION STATE]
User: Acme Exporters Ltd (Account ID: ACME-991)
Active Shipment: Waybill #GB-88219-EU (Origin: Manchester, Destination: Lyon)""")

if inc_rules:
    prompt_layers.append("""### [LAYER 4: BUSINESS RULES]
1. If customs hold exceeds 48 hours, issue a priority customs broker ticket.
2. If EORI is missing, prompt user with the direct HMRC lookup portal link.
3. Fallback: If waybill status is UNKNOWN, output: "ESCALATE_TO_DISPATCH".""")

if inc_ex:
    prompt_layers.append("""### [LAYER 5: FEW-SHOT EXAMPLES]
<example>
Query: Shipment held at Calais due to missing invoice.
Action: ISSUE_INVOICE_UPLOAD_LINK
</example>""")

if inc_api:
    prompt_layers.append("""### [LAYER 6: REAL-TIME API TELEMETRY]
Live Waybill Status: {"status": "CUSTOMS_HOLD", "port": "Dover", "hold_hours": 54, "missing_documents": ["EORI_NUMBER"]}""")

prompt_layers.append("### USER INQUIRY:\nWhy is our Lyon shipment delayed and what is the immediate resolution?")

assembled_context_str = "\n\n".join(prompt_layers)
st.code(assembled_context_str, language="text")

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz & ASCII)
# ==============================================================================
st.header("4. Context Orchestration Architecture", divider="rainbow")

col_co1, col_co2 = st.columns([1, 1])

with col_co1:
    st.markdown("#### The Multi-Source Ingestion Pipeline")
    dot_ctx_layers = """
    digraph ContextIngestion {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        DB [label="Relational DB\\n(User Profile & Rules)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Vector [label="Vector Store (ChromaDB)\\n(RAG Policy Docs)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        LiveAPI [label="Live API Endpoints\\n(Telemetry / Status)", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];

        Aggregator [label="Context Aggregator & Sanitizer\\n(Validate types, truncate, delimit)", fillcolor="#047857", fontcolor="#ffffff"];
        Prompt [label="Unified Prompt Payload\\n(Structured XML Sections)", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];

        DB -> Aggregator;
        Vector -> Aggregator;
        LiveAPI -> Aggregator;
        Aggregator -> Prompt;
    }
    """
    st.graphviz_chart(dot_ctx_layers, use_container_width=False)

with col_co2:
    st.markdown("#### Context Layer Priority Hierarchy")
    st.code(
        """┌────────────────────────────────────────────────────────┐
│ [LAYER 4] Hard Business Rules & Negative Guardrails    │  <-- Absolute Priority
├────────────────────────────────────────────────────────┤
│ [LAYER 2] Reference Documents & Verified RAG Chunks    │  <-- Ground Truth
├────────────────────────────────────────────────────────┤
│ [LAYER 6] Real-Time API Telemetry & External State     │  <-- Transient Truth
├────────────────────────────────────────────────────────┤
│ [LAYER 3] User History & Session State (JWT Token)     │  <-- Personalization
├────────────────────────────────────────────────────────┤
│ [LAYER 1] General Background & Identity Defaults       │  <-- Macro Tone
└────────────────────────────────────────────────────────┘""",
        language="text"
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

pc_tab1, pc_tab2, pc_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with pc_tab1:
    with st.expander("Q1: How do you prioritize conflicting information across different context layers?", expanded=True):
        st.markdown(
            """
In enterprise prompts, establish an explicit **Priority Hierarchy Rule**:
1. **Business Rules & Safety Directives** supersede all other context.
2. **Real-Time API Telemetry** supersedes historical user notes.
3. **Verified Reference Documents (RAG)** supersede the model's pre-trained parametric knowledge.
Explicitly command the model in the system prompt: *"In any conflict between reference documents and background general knowledge, the reference documents are authoritative"*.
"""
        )

    with pc_tab2:
        with st.expander("Trap 1: 'Can we just dump entire 50-page PDF documents into the context window since context is cheap?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Modern models have large context windows, so sending entire PDFs avoids chunking complexity."*
- **Winning Response**: *"No. Loading 50-page PDFs injects massive amounts of irrelevant noise (legal boilerplate, headers, appendices), increasing hallucination risk and triggering 'Lost in the Middle' retrieval failures. Furthermore, it creates unnecessary latency (10-30 seconds TTFT) and multiplies API bills by 20x. Production systems always use intelligent chunking, semantic vector search, and reranking to inject only the top 3-5 relevant paragraphs into the context."*
"""
            )

    with pc_tab3:
        c_pc1, c_pc2, c_pc3 = st.columns(3)
        with c_pc1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Context Layering**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Structuring multiple operational data streams (background, docs, session, rules, telemetry) into distinct prompt sections.")

        with c_pc2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Authority Hierarchy**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Explicit system rule dictating which context source wins when two documents contain contradictory data.")

        with c_pc3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Live API Context**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Injecting real-time JSON payloads (telemetry, stock prices) to ground the model in current external state.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 7.2 Summary & Self-Assessment", divider="rainbow")

col_psa, col_psb = st.columns(2)
with col_psa:
    st.markdown(
        """
### Key Takeaways:
1. **The 6 Layers**: Background, Reference Docs, User Session, Business Rules, Examples, Live APIs.
2. **Explicit Scoping**: Enclose each information stream in dedicated XML tags (`<reference_docs>`).
3. **Declare Precedence**: State which context layer takes priority in the event of contradictions.
4. **Lean & Targeted**: Inject only high-signal passages rather than raw document dumps.
"""
    )

with col_psb:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can identify and structure the 6 essential context layers in an enterprise prompt.", value=True, key="chk_7_2_1")
    st.checkbox("I know how to resolve contradictory information between context layers using priority rules.", value=True, key="chk_7_2_2")
    st.checkbox("I understand the architectural cost of dumping unparsed PDFs into context windows.", value=True, key="chk_7_2_3")
    st.checkbox("I can implement live API telemetry injection into a Python prompt template.", value=True, key="chk_7_2_4")
