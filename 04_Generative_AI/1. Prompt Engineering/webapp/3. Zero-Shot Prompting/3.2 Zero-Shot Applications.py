import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="3.2 Zero-Shot Applications | GenAI Roadmap",
    page_icon="🚀",
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
st.title("🚀 3.2 Zero-Shot Applications — The 7 Enterprise Workloads")
st.markdown(
    """
Welcome to **Lesson 3.2: Zero-Shot Applications**.  
Zero-shot prompting is not a theoretical novelty; it powers the majority of real-world enterprise LLM workloads today. 
From ingesting raw support tickets to extracting structured entities from PDF OCR streams, zero-shot pipelines handle high-throughput operations at minimal token cost.

Below, we master the **7 Primary Enterprise Applications of Zero-Shot Prompting**, complete with production code implementations, comparative schema blueprints, and an interactive testbench.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 7 Core Applications Matrix
# ==============================================================================
st.header("1. Enterprise Zero-Shot Workload Matrix", divider="rainbow")
st.markdown("Here is the architectural comparison across all 7 major enterprise application categories:")

apps_data = {
    "Application": [
        "1. Text Classification",
        "2. Sentiment Analysis",
        "3. Machine Translation",
        "4. Text Summarization",
        "5. Question Answering (QA)",
        "6. Information Extraction",
        "7. Content Generation"
    ],
    "Enterprise Use Case": [
        "Routing incoming support tickets to engineering vs billing tiers",
        "Social media listening and customer NPS feedback polarity scoring",
        "Localizing API documentation and technical runbooks across regions",
        "Generating executive digests and daily standup summaries from Slack",
        "Internal HR knowledge base lookup and policy query answering",
        "Parsing invoices, resume PDFs, and legal contracts into JSON",
        "Automating changelogs, customer notification emails, and boilerplate"
    ],
    "Key Prompting Strategy": [
        "Enforce closed categorical enums with fallback tokens",
        "Aspect-based extraction (separate product, shipping, pricing)",
        "Inject target language glossary rules and preserve markdown tags",
        "Define strict length limits and quantitative bullet requirements",
        "Context-bound prompts with deterministic 'NOT_IN_CONTEXT' gates",
        "Strict Pydantic / JSON schema contracts with null fallbacks",
        "Target audience framing, brand tone guidelines, and negative constraints"
    ],
    "Success Metric": [
        "F1-Score / Categorization Accuracy",
        "Aspect Resolution Precision",
        "BLEU / Terminology Fidelity",
        "Compression Ratio & ROUGE Score",
        "Hallucination Rate (Target: 0%)",
        "Schema Parse Validity (Target: 100%)",
        "Tone Alignment & Acceptance Rate"
    ]
}
st.dataframe(pd.DataFrame(apps_data), use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Deep-Dive Into the 7 Applications with Code
# ==============================================================================
st.header("2. Deep Technical Breakdown & Production Code", divider="rainbow")

# 1. Text Classification
with st.container(border=True):
    st.subheader("1. Zero-Shot Text Classification")
    st.markdown(
        """
Assigning an arbitrary text snippet into one or more predefined categories.  
- **Best Practice**: Never ask the model to *"categorize this"* without providing the closed set of allowed values. Always provide an explicit fallback.
"""
    )
    st.code(
        '''prompt = """
INSTRUCTION:
Classify the incoming bug report into exactly ONE category from the list below:
- FRONTEND_UI
- BACKEND_API
- DATABASE_INFRA
- AUTH_SECURITY
- OTHER

CONSTRAINTS:
Output strictly the chosen category name. Do not output explanations or quotes.

INPUT:
"Users in the EU region receive HTTP 403 Forbidden when attempting to refresh their OAuth bearer tokens."

CATEGORY:"""''',
        language="python"
    )

# 2. Sentiment Analysis
with st.container(border=True):
    st.subheader("2. Zero-Shot Sentiment Analysis (Aspect-Based)")
    st.markdown(
        """
Simple binary sentiment (*positive/negative*) is rarely sufficient in enterprise settings. 
**Aspect-Based Sentiment Analysis (ABSA)** breaks down sentiment across distinct business dimensions (e.g. product quality vs shipping speed vs pricing).
"""
    )
    st.code(
        '''prompt = """
Analyze the customer review below and evaluate sentiment across three distinct aspects:
1. Product Quality: [POSITIVE, NEUTRAL, NEGATIVE]
2. Shipping Speed: [POSITIVE, NEUTRAL, NEGATIVE]
3. Price / Value: [POSITIVE, NEUTRAL, NEGATIVE]

Format as valid JSON:
{
  "product_quality": string,
  "shipping_speed": string,
  "price_value": string,
  "overall_sentiment": string,
  "customer_churn_risk": "LOW" | "MEDIUM" | "HIGH"
}

REVIEW:
"The noise cancellation on these headphones is sublime, but it took three weeks to arrive from Shenzhen and $350 feels steep."
"""''',
        language="python"
    )

# 3. Translation
with st.container(border=True):
    st.subheader("3. Zero-Shot Technical Translation")
    st.markdown(
        """
Zero-shot translation excels at preserving technical terminology and formatting (code blocks, markdown headings, variable names) that traditional machine translation engines corrupt.
"""
    )
    st.code(
        '''prompt = """
Translate the technical documentation below from English to German.

RULES:
- Preserve all markdown code blocks, parameter names (e.g. `timeout_ms`), and JSON keys in English.
- Use professional engineering terminology (Sie-Form).
- Do not translate the product brand name "HyperDB".

ENGLISH SOURCE:
### Configuration
To configure connection retry intervals, pass `timeout_ms=5000` to the `HyperDB.connect()` factory method.

GERMAN TRANSLATION:"""''',
        language="python"
    )

# 4. Summarization
with st.container(border=True):
    st.subheader("4. Zero-Shot Text Summarization")
    st.markdown(
        """
Condensing long documents into actionable digests.  
- **Engineering Tip**: Quantify length explicitly (e.g. *"Under 80 words"*, *"Exactly 3 bullet points"*) and specify the operational purpose (Executive Briefing vs Engineering Handoff).
"""
    )
    st.code(
        '''prompt = """
TASK: Summarize the post-mortem report below for an executive engineering audience.

REQUIREMENTS:
- Maximum 3 bullet points.
- Bullet 1: Root Cause Analysis (technical breakdown).
- Bullet 2: Customer Impact (downtime duration & transaction loss).
- Bullet 3: Preventive Action Item (infrastructure change with SLA).
- Total word count must not exceed 75 words.

REPORT:
[Post-Mortem Logs...]"""''',
        language="python"
    )

# 5. Question Answering
with st.container(border=True):
    st.subheader("5. Zero-Shot Context-Grounded QA (Closed-Domain)")
    st.markdown(
        """
Answering queries based strictly on provided documentation without relying on external pre-training assumptions.
"""
    )
    st.code(
        '''prompt = """
You are an internal corporate policy assistant.
Answer the employee query using ONLY the policy snippet inside <policy_text>.
If the answer cannot be determined strictly from the text, reply EXACTLY:
"POLICY_NOT_FOUND: Please escalate to hr-helpdesk@company.com"

<policy_text>
Parental Leave Policy 2026: Full-time employees who have completed at least 12 continuous months of service are eligible for up to 16 weeks of fully paid parental leave. Part-time employees receive prorated leave.
</policy_text>

QUESTION: Can a contractor take 8 weeks of paid parental leave?
ANSWER:"""''',
        language="python"
    )

# 6. Information Extraction
with st.container(border=True):
    st.subheader("6. Zero-Shot Information Extraction (NER & JSON)")
    st.markdown(
        """
Transforming unstructured natural language strings (emails, medical notes, chat logs) into strongly typed, machine-readable JSON schemas.
"""
    )
    st.code(
        '''prompt = """
Extract invoice metadata from the OCR text into raw parseable JSON.

JSON SCHEMA:
{
  "vendor_name": string,
  "invoice_number": string,
  "invoice_date": string (YYYY-MM-DD),
  "line_items": [
    {"description": string, "quantity": integer, "unit_price": float, "total": float}
  ],
  "total_amount": float,
  "currency": string
}

OCR TEXT:
"ACME Cloud Services Inc. - Invoice #INV-2026-9918 - Date: Oct 12, 2026
Items: Compute Pod 4x (qty 2 @ $120.00 = $240.00), Storage S3 500GB (qty 1 @ $50.00 = $50.00). Total: $290.00 USD"
"""''',
        language="python"
    )

# 7. Content Generation
with st.container(border=True):
    st.subheader("7. Zero-Shot Structured Content Generation")
    st.markdown(
        """
Generating domain-specific communications, release announcements, or unit tests from structured inputs.
"""
    )
    st.code(
        '''prompt = """
ROLE: Senior Technical Writer.
TASK: Write a customer-facing product release note for version 2.4.0.

INPUT CHANGELOG:
- Added support for PostgreSQL 16 connection pooling.
- Fixed memory leak in websocket event dispatcher.
- Deprecated legacy /v1/auth endpoint (removal date: Jan 2027).

GUIDELINES:
- Tone: Confident, professional, developer-focused.
- Sections: "What's New", "Performance Improvements", "Important Deprecations".
- Word count: 100-140 words total.
"""''',
        language="python"
    )

st.divider()

# ==============================================================================
# Section 3: Interactive Application Workbench
# ==============================================================================
st.header("3. Interactive Enterprise Application Workbench", divider="rainbow")
st.markdown("Select an application below to inspect live prompt schemas, token budgets, and output validation contracts:")

selected_app = st.selectbox(
    "Choose Zero-Shot Application to explore:",
    [
        "1. Text Classification (Support Tickets)",
        "2. Aspect-Based Sentiment Analysis",
        "3. Technical Translation with Glossary Rules",
        "4. Executive Summarization",
        "5. Closed-Domain Policy QA",
        "6. Invoice Information Extraction",
        "7. Developer Release Note Generation"
    ]
)

if "Classification" in selected_app:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Sample Incoming Payload:**")
        sample_ticket = st.text_area(
            "Ticket Text:",
            "Our automated billing job failed with credit card gateway timeout error 504. Subscriptions are not renewing.",
            height=100
        )
    with col_c2:
        st.markdown("**Live Zero-Shot Prompt Constructed:**")
        st.code(f"""Classify into: [BILLING, INFRASTRUCTURE, AUTH, SALES]
Fallback: UNKNOWN

TICKET:
"{sample_ticket}"

CATEGORY:""", language="text")
        st.success("Target Output: `BILLING` (Tokens consumed: ~35)")

elif "Sentiment" in selected_app:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sample_rev = st.text_area(
            "Review Text:",
            "The camera sensor captures phenomenal dynamic range in daylight, but the battery drains in 3 hours and customer service was unhelpful.",
            height=100
        )
    with col_s2:
        st.markdown("**Expected Aspect JSON Output:**")
        st.json({
            "camera_hardware": "POSITIVE",
            "battery_performance": "NEGATIVE",
            "customer_service": "NEGATIVE",
            "overall_sentiment": "MIXED"
        })

elif "Translation" in selected_app:
    st.info("Demonstrating preservation of code variables and markdown links during translation.")
    st.code(
        """Source Text:
"Ensure the `redis_cluster_nodes` array contains at least 3 master nodes before deploying."

Target (Japanese):
"`redis_cluster_nodes` 配列に、デプロイ前に少なくとも3つのマスターノードが含まれていることを確認してください。"
""",
        language="text"
    )

elif "Summarization" in selected_app:
    st.info("3-Tier Multi-Level Compression Strategy")
    comp_df = pd.DataFrame({
        "Tier": ["TL;DR (1-Sentence)", "Executive (3 Bullets)", "Full Technical Digest"],
        "Target Word Count": ["15 - 25 words", "60 - 80 words", "180 - 250 words"],
        "Target Audience": ["Slack notification / Mobile", "VP of Engineering / CTO", "On-Call Incident Responders"]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

elif "Policy QA" in selected_app:
    st.warning("Hallucination Defense Test: Asking an out-of-scope query.")
    st.code(
        """Query: "What is the pet policy in the office?"
Context: [Only covers Health Insurance & 401k Plans]
Expected Model Output: "POLICY_NOT_FOUND: Please escalate to hr-helpdesk@company.com" """,
        language="text"
    )

elif "Extraction" in selected_app:
    st.info("Zero-Shot Schema Validation via Pydantic Contract")
    st.code(
        """from pydantic import BaseModel, Field

class InvoiceSchema(BaseModel):
    vendor: str
    total_usd: float = Field(gt=0, description="Gross total in USD")
    tax_exempt: bool = False
    due_date: str = Field(regex=r"^\\d{4}-\\d{2}-\\d{2}$")""",
        language="python"
    )

elif "Release Note" in selected_app:
    st.info("Tone Calibration: From Git Commits to Marketing Copy")
    st.code(
        """Git Commit: "fix(auth): resolve race condition in redis session store (PR #402)"
Zero-Shot Release Note: "⚡ Stability Update: Improved session reliability during peak traffic surges." """,
        language="text"
    )

st.divider()

# ==============================================================================
# Section 4: Architectural Flowchart (Graphviz)
# ==============================================================================
st.header("4. Enterprise Task Routing Topology", divider="rainbow")

col_topo1, col_topo2 = st.columns([1, 1])

with col_topo1:
    st.markdown("#### Dynamic Workload Router")
    dot_apps = """
    digraph ZeroShotRouter {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Input [label="Raw Text Ingestion\\n(Tickets / PDFs / Web)", fillcolor="#334155", fontcolor="#ffffff"];
        Router [label="Task Type Classifier\\n(Zero-Shot Dispatcher)", fillcolor="#1e293b", fontcolor="#facc15", color="#facc15"];
        Classify [label="1. Text Classification\\n(Enum Enforced)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        Extract [label="6. Structured Extraction\\n(Pydantic JSON)", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        Summarize [label="4. Summarization\\n(Length Bounded)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        QA [label="5. Context QA\\n(Zero Hallucination)", fillcolor="#1e293b", fontcolor="#ec4899", color="#ec4899"];

        Input -> Router;
        Router -> Classify;
        Router -> Extract;
        Router -> Summarize;
        Router -> QA;
    }
    """
    st.graphviz_chart(dot_apps, use_container_width=False)

with col_topo2:
    st.markdown("#### Enterprise Deployment Metrics")
    st.markdown(
        """
- **Latency Advantage**: Zero-shot bypasses example embedding retrieval, dropping P95 latency by ~120ms.
- **Cost Reduction**: Eliminates 300–800 input tokens per call compared to few-shot prompts.
- **Microservice Integration**: Structured outputs feed directly into message queues (Kafka / RabbitMQ) or SQL tables.
"""
    )

st.divider()

# ==============================================================================
# Section 5: Technical Interview Preparation
# ==============================================================================
st.header("5. Technical Interview Preparation", divider="rainbow")

a_tab1, a_tab2, a_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with a_tab1:
    with st.expander("Q1: How do you perform Aspect-Based Sentiment Analysis (ABSA) zero-shot?", expanded=True):
        st.markdown(
            """
To perform zero-shot ABSA, decompose the prompt into explicitly enumerated dimensions (e.g. *Usability*, *Pricing*, *Customer Service*, *Reliability*). 
Provide a rigid JSON schema requiring a rating for each aspect independently, plus an overall sentiment score. 
This forces the attention heads to evaluate the context across orthogonal semantic axes rather than collapsing the entire review into a single generic sentiment label.
"""
        )

    with st.expander("Q2: In zero-shot information extraction, how do you handle missing fields?", expanded=False):
        st.markdown(
            """
Always explicitly specify in the system prompt: *"If a field is not explicitly present in the input text, set its value to null or 'NOT_SPECIFIED'—do NOT invent or guess values"*. 
In typed frameworks like Pydantic, define optional attributes with default values (`email: str | None = None`).
"""
        )

    with st.expander("Q3: Why does zero-shot translation often outperform traditional statistical MT tools for software docs?", expanded=False):
        st.markdown(
            """
Statistical MT engines translate words in local ngram windows, frequently translating reserved code keywords (e.g. translating Python `def` into foreign language verbs). 
LLMs possess contextual syntax awareness: their multi-head attention recognizes markdown code blocks, backtick spans, and variable identifiers, leaving technical tokens intact while localizing surrounding prose.
"""
        )

with a_tab2:
    with st.expander("Trap 1: 'Can zero-shot summarization replace fine-tuned domain summarizers in medical or legal domains?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, GPT-4 is smart enough to summarize any legal contract or clinical trial."*
- **Winning Response**: *"No. While zero-shot summarization produces fluent text, highly regulated domains like medical pathology or corporate law require strict adherence to regulatory taxonomies, legal liability thresholds, and domain-specific acronyms. Zero-shot models frequently omit critical nuances or misinterpret statutory definitions. Production medical/legal summarization requires domain fine-tuning, RAG grounding, and human-in-the-loop expert review."*
"""
        )

with a_tab3:
    c_ac1, c_ac2, c_ac3 = st.columns(3)
    with c_ac1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Aspect-Based Sentiment**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Evaluating sentiment separately across distinct operational dimensions (speed, quality, price) rather than a single global label.")

    with c_ac2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Technical Translation Guard**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Protecting code blocks, variable names, and JSON keys from improper linguistic translation.")

    with c_ac3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Null Handling Contract**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Mandatory rule commanding `null` for missing extraction fields, halting hallucinated data.")

st.divider()

# ==============================================================================
# Section 6: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("6. Lesson 3.2 Summary & Self-Assessment", divider="rainbow")

col_as1, col_as2 = st.columns(2)
with col_as1:
    st.markdown(
        """
### Key Takeaways:
1. **The 7 Workloads**: Classification, Sentiment (ABSA), Translation, Summarization, QA, Extraction, and Generation.
2. **Enum Boundaries**: Always constrain classification tasks to closed sets with fallback tokens.
3. **Structured Schemas**: Use Pydantic and JSON schemas for zero-shot information extraction.
4. **Protect Code Tokens**: Explicitly forbid translation of variable names and markdown code fences.
"""
    )

with col_as2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can implement zero-shot classification with categorical enums and fallback tokens.", value=True, key="chk_3_2_1")
    st.checkbox("I know how to design an Aspect-Based Sentiment Analysis prompt returning JSON.", value=True, key="chk_3_2_2")
    st.checkbox("I can structure an extraction prompt with null safety rules for missing data.", value=True, key="chk_3_2_3")
    st.checkbox("I understand the architectural tradeoffs of zero-shot vs fine-tuning in regulated domains.", value=True, key="chk_3_2_4")
