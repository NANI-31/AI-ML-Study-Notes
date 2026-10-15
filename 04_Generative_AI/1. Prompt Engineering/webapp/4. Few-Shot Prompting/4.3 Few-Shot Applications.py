import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="4.3 Few-Shot Applications | GenAI Roadmap",
    page_icon="⚙️",
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
st.title("⚙️ 4.3 Few-Shot Applications — The 7 Enterprise Implementations")
st.markdown(
    """
Welcome to **Lesson 4.3: Few-Shot Applications**.  
When tasks require strict adherence to proprietary schemas, custom domain taxonomies, or idiosyncratic coding patterns that did not exist during model pre-training, Zero-Shot prompting breaks down. 
**Few-Shot Prompting bridges this capability gap**, acting as a rapid, in-memory micro-adaptation layer.

Below, we master the **7 Major Enterprise Workloads of Few-Shot Prompting**, complete with production code blueprints, dynamic exemplar retrieval patterns, and an interactive exploration workbench.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 7 Core Enterprise Workloads Matrix
# ==============================================================================
st.header("1. Enterprise Few-Shot Workload Matrix", divider="rainbow")
st.markdown("Here is how few-shot exemplars unlock mission-critical capabilities across 7 application domains:")

workload_df = pd.DataFrame({
    "Application": [
        "1. Classification",
        "2. Data Extraction",
        "3. Custom Formatting",
        "4. Text Transformation",
        "5. Style Imitation",
        "6. Code Generation",
        "7. Structured Outputs"
    ],
    "Why Zero-Shot Struggles": [
        "Proprietary class boundaries differ from common public web usage",
        "Ambiguous nested structures; model guesses key names",
        "Bespoke serialization formats (custom markup, TOML, non-standard delimiters)",
        "Dialect normalization, corporate acronym translation, domain rewriting",
        "Subtle brand voice, tone nuances, and stylistic cadences",
        "Internal proprietary SDKs, custom company ORMs, non-public APIs",
        "Strict multi-level nested JSON schemas requiring exact type matching"
    ],
    "Few-Shot Solution": [
        "3-5 balanced exemplars mapping edge-case inputs to target class tokens",
        "Exemplars demonstrate extraction of tricky optional and null fields",
        "Exemplars demonstrate exact custom syntax with 100% delimiter fidelity",
        "Paired before-and-after samples demonstrate exact transformation rules",
        "Exemplars provide direct stylistic priors that prime the voice",
        "Exemplars demonstrate idiomatic usage of internal SDK methods",
        "Exemplars anchor opening braces, key casing, and nesting structures"
    ]
})
st.dataframe(workload_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Deep Technical Breakdown & Code Blueprints
# ==============================================================================
st.header("2. Detailed Technical Breakdown & Code Blueprints", divider="rainbow")

# 1. Classification
with st.container(border=True):
    st.subheader("1. Few-Shot Proprietary Classification")
    st.markdown(
        """
Classifying items according to an internal corporate taxonomy (e.g., Jira issue escalation levels) where standard web models would misinterpret definitions.
"""
    )
    st.code(
        '''prompt = """
INSTRUCTION:
Classify incident reports into internal severity tiers: [SEV_ALPHA, SEV_BETA, SEV_GAMMA].

<example>
<incident>Customer checkout gateway latency spike to 450ms. Zero dropped transactions.</incident>
<tier>SEV_BETA</tier>
</example>

<example>
<incident>Automated database failover succeeded; read replicas caught up in 8 seconds.</incident>
<tier>SEV_GAMMA</tier>
</example>

<example>
<incident>Payment ledger reconciliation mismatch > $10,000 detected during batch run.</incident>
<tier>SEV_ALPHA</tier>
</example>

<incident>Auth service Redis cluster memory at 98%; cache eviction active.</incident>
<tier>"""''',
        language="python"
    )

# 2. Data Extraction
with st.container(border=True):
    st.subheader("2. Few-Shot Unstructured Data Extraction")
    st.markdown(
        """
Extracting clinical, legal, or log data with nested entities where schema keys must adhere strictly to internal data warehouse column names.
"""
    )
    st.code(
        '''prompt = """
Extract lab results from unstructured physician notes into JSON.

<example>
<note>Pt presented with BP 142/90, fasting glucose 118 mg/dL, HbA1c 6.2%.</note>
<json>
{"systolic_bp": 142, "diastolic_bp": 90, "glucose_mg_dl": 118, "hba1c_pct": 6.2, "status": "PREDIABETIC_ELEVATED"}
</json>
</example>

<example>
<note>Vitals normal. Glucose not tested today.</note>
<json>
{"systolic_bp": null, "diastolic_bp": null, "glucose_mg_dl": null, "hba1c_pct": null, "status": "INCOMPLETE_LABS"}
</json>
</example>

<note>Patient recorded fasting glucose 92 mg/dL. BP was 118/76 mmHg.</note>
<json>"""''',
        language="python"
    )

# 3. Custom Formatting
with st.container(border=True):
    st.subheader("3. Few-Shot Custom Formatting & Delimiters")
    st.markdown(
        """
Generating bespoke configuration formats (e.g. Envoy proxy filters, proprietary DSLs) where zero-shot models default to standard JSON or YAML.
"""
    )
    st.code(
        '''prompt = """
Convert routing rules into custom internal router notation (RULE: [SRC] >> [DEST] | FLAGS).

EXAMPLE 1:
Input: Send /api/v1/auth traffic to auth-service on port 8080 with SSL verification.
Output: RULE: [/api/v1/auth] >> [auth-service:8080] | VERIFY_SSL=1

EXAMPLE 2:
Input: Forward /metrics to prometheus-collector on port 9090 without logging.
Output: RULE: [/metrics] >> [prometheus-collector:9090] | LOG=0

TARGET:
Input: Route /v2/checkout to billing-pod on port 443 with SSL and retry limit 3.
Output:"""''',
        language="text"
    )

# 4. Text Transformation
with st.container(border=True):
    st.subheader("4. Few-Shot Domain Text Transformation")
    st.markdown(
        """
Normalizing non-standard technical error stack traces into customer-facing plain English explanations.
"""
    )
    st.code(
        '''prompt = """
Translate raw PostgreSQL database exceptions into clear user-facing error messages.

<ex>
<raw>psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint "users_email_key"</raw>
<user_msg>An account with this email address already exists. Please log in or reset your password.</user_msg>
</ex>

<ex>
<raw>psycopg2.OperationalError: server closed the connection unexpectedly</raw>
<user_msg>Our database service is temporarily unavailable. Please retry your request in a few moments.</user_msg>
</ex>

<raw>psycopg2.errors.ForeignKeyViolation: insert or update on table "orders" violates foreign key constraint "fk_customer"</raw>
<user_msg>"""''',
        language="text"
    )

# 5. Style Imitation
with st.container(border=True):
    st.subheader("5. Few-Shot Style & Voice Imitation")
    st.markdown(
        """
Calibrating tone to perfectly match executive communication, corporate PR guidelines, or legal counsel drafting.
"""
    )
    st.code(
        '''prompt = """
Draft executive updates in the concise, direct communication style of the VP of Infrastructure.

<sample 1>
Update: S3 migration finished 2 days ahead of schedule. Compute costs down 14% ($18k/mo run-rate). Next milestone: DynamoDB TTL cleanup by Friday.
</sample 1>

<sample 2>
Update: Blocked on Datadog API rate limit negotiation. Pushed deployment to Tuesday 09:00 UTC to prevent alerting blackout. Risk is low.
</sample 2>

DRAFT UPDATE FOR:
We completed the Kubernetes 1.30 upgrade across all 8 clusters without any customer downtime. Memory usage decreased by 8%."""''',
        language="text"
    )

# 6. Code Generation
with st.container(border=True):
    st.subheader("6. Few-Shot Code Generation for Proprietary Frameworks")
    st.markdown(
        """
Teaching the model how to write code for an internal proprietary SDK or in-house Python library that does not exist in public training data.
"""
    )
    st.code(
        '''prompt = """
Write API endpoints using our internal microservice framework (NexusSDK).

<example>
Task: Create user
Code:
@nexus.route("/users", method="POST")
def create_user(ctx: NexusContext, payload: UserSchema) -> NexusResult:
    ctx.audit_log(action="USER_CREATED", actor=ctx.auth.user_id)
    return ctx.db.insert(payload).to_result()
</example>

<example>
Task: Delete product
Code:
@nexus.route("/products/{id}", method="DELETE")
def delete_product(ctx: NexusContext, id: str) -> NexusResult:
    ctx.audit_log(action="PRODUCT_DELETED", target_id=id)
    return ctx.db.soft_delete("products", id).to_result()
</example>

TASK: Write an endpoint to update user email at /users/{id}/email (PUT).
CODE:"""''',
        language="python"
    )

# 7. Structured Outputs
with st.container(border=True):
    st.subheader("7. Few-Shot Nested Structured Outputs")
    st.markdown(
        """
Guaranteed JSON generation for deeply nested structures where zero-shot models omit secondary fields or produce syntax anomalies.
"""
    )
    st.code(
        '''prompt = """
Parse meeting transcripts into nested action item objects.

<example>
<transcript>"Alice will refactor the Redis cache by Wednesday, and Bob will update the API docs."</transcript>
<json>
{
  "action_items": [
    {"owner": "Alice", "task": "Refactor Redis cache", "deadline": "Wednesday", "priority": "MEDIUM"},
    {"owner": "Bob", "task": "Update API docs", "deadline": null, "priority": "LOW"}
  ]
}
</json>
</example>

<transcript>"David needs to hotfix the auth token leak ASAP today; Carol to review the PR."</transcript>
<json>"""''',
        language="text"
    )

st.divider()

# ==============================================================================
# Section 3: Dynamic Few-Shot Architecture (RAG for Examples)
# ==============================================================================
st.header("3. Dynamic Few-Shot Architecture (k-NN Retrieval)", divider="rainbow")
st.markdown(
    """
In production, you cannot hardcode the same 3 examples into every prompt. 
If you have 10,000 historical support tickets, which 3 should you include?  
**Solution: Dynamic Few-Shot Retrieval (k-NN Few-Shot)**:
1. Embed a bank of 500+ curated exemplars into a Vector Database (e.g. ChromaDB).
2. When a runtime query arrives, embed the query and retrieve the **Top-3 most semantically similar exemplars**.
3. Dynamically inject those 3 targeted exemplars into the prompt context!
"""
    )

col_dyn1, col_dyn2 = st.columns([1, 1])

with col_dyn1:
    st.markdown("#### Dynamic Retrieval Pipeline Diagram")
    dot_dyn = """
    digraph DynamicFewShot {
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, margin="0.1,0.08"];
        edge [color="#94a3b8", fontsize=8, color="white", fontcolor="white"];

        Query [label="Runtime User Query\\n'Need to cancel order'", fillcolor="#334155", fontcolor="#ffffff"];
        Embed [label="Embedding Model\\n(text-embedding-3-small)", fillcolor="#1e293b", fontcolor="#38bdf8", color="#38bdf8"];
        VectorDB [label="Exemplar Vector Store\\n(ChromaDB Collection)", fillcolor="#1e293b", fontcolor="#a855f7", color="#a855f7"];
        TopK [label="Retrieve Top-3 Similar\\nOrder-Cancellation Exemplars", fillcolor="#1e293b", fontcolor="#f59e0b", color="#f59e0b"];
        Assemble [label="Dynamic Prompt Assembly\\n(Instruction + Retrieved Exemplars)", fillcolor="#1e293b", fontcolor="#22c55e", color="#22c55e"];
        LLM [label="LLM In-Context Completion", fillcolor="#047857", fontcolor="#ffffff"];

        Query -> Embed -> VectorDB -> TopK -> Assemble -> LLM;
        Query -> Assemble;
    }
    """
    st.graphviz_chart(dot_dyn, use_container_width=False)

with col_dyn2:
    st.markdown("#### Production Python Blueprint (ChromaDB + OpenAI)")
    st.code(
        '''import chromadb

# Initialize ChromaDB exemplar collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("prompt_exemplars")

# Retrieve top 3 most semantically relevant examples
def get_dynamic_few_shot_prompt(user_query: str) -> str:
    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )
    
    exemplar_block = "\\n\\n".join(results["documents"][0])
    
    return f"""INSTRUCTION: Classify the user query.
    
### RELEVANT EXAMPLES:
{exemplar_block}

### TARGET QUERY:
{user_query}
CATEGORY:"""''',
        language="python"
    )

st.divider()

# ==============================================================================
# Section 4: Technical Interview Preparation
# ==============================================================================
st.header("4. Technical Interview Preparation", divider="rainbow")

f_tab1, f_tab2, f_tab3 = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with f_tab1:
    with st.expander("Q1: What is Dynamic Few-Shot Prompting and when is it required over static few-shot?", expanded=True):
        st.markdown(
            """
**Static Few-Shot** hardcodes the same 3 examples for every single query. It works well when the task is narrow and homogeneous.  
**Dynamic Few-Shot** uses semantic search (vector embeddings) to retrieve the $k$ most relevant exemplars from an exemplar bank for the specific user query at runtime. 
It is required when the input domain is broad, complex, or multi-faceted (e.g. enterprise legal contracts covering hundreds of distinct clauses), where static examples cannot cover all possible sub-domains.
"""
        )

    with f_tab2:
        with st.expander("Trap 1: 'Can Dynamic Few-Shot replace RAG completely?'", expanded=True):
            st.markdown(
                """
- **Naive Response**: *"Yes, if you retrieve examples from ChromaDB, you are doing RAG."*
- **Winning Response**: *"No. Dynamic Few-Shot retrieves input-output task demonstrations to teach the model HOW to perform a task or format. RAG retrieves factual source documents (articles, policies, PDFs) to provide the factual knowledge WHAT to answer. In production, enterprise pipelines often combine both: retrieving 2 dynamic exemplars for syntax calibration and 3 context chunks for factual grounding."*
"""
            )

    with f_tab3:
        c_fa1, c_fa2, c_fa3 = st.columns(3)
        with c_fa1:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Dynamic Few-Shot**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Querying a vector database to retrieve the most semantically relevant exemplars at runtime.")

        with c_fa2:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Proprietary SDK Adaptation**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Using 2-3 code exemplars to teach the model custom in-house library methods absent from pre-training data.")

        with c_fa3:
            with st.container(border=True):
                st.markdown("#### 📇 Front")
                st.markdown("**Task vs Fact Retrieval**")
                with st.popover("Reveal Back"):
                    st.markdown("**Back:** Dynamic Few-Shot retrieves task demonstrations; RAG retrieves factual reference knowledge.")

st.divider()

# ==============================================================================
# Section 5: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("5. Lesson 4.3 Summary & Self-Assessment", divider="rainbow")

col_ws1, col_ws2 = st.columns(2)
with col_ws1:
    st.markdown(
        """
### Key Takeaways:
1. **The 7 Workloads**: Classification, Extraction, Formatting, Transformation, Style, Code, and Structured Outputs.
2. **Proprietary Bridge**: Few-shot is the fastest way to teach models internal corporate schemas and SDKs.
3. **Dynamic Few-Shot (k-NN)**: Scale few-shot prompts to thousands of scenarios using vector search exemplar banks.
4. **Distinguish from RAG**: Exemplars teach execution syntax; RAG supplies reference truth.
"""
    )

with col_ws2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I can identify when a workload requires few-shot exemplars over zero-shot instructions.", value=True, key="chk_4_3_1")
    st.checkbox("I know how to implement few-shot code generation for proprietary frameworks.", value=True, key="chk_4_3_2")
    st.checkbox("I understand the architectural workflow of Dynamic Few-Shot retrieval using ChromaDB.", value=True, key="chk_4_3_3")
    st.checkbox("I can differentiate Dynamic Few-Shot exemplar retrieval from factual RAG document retrieval.", value=True, key="chk_4_3_4")
