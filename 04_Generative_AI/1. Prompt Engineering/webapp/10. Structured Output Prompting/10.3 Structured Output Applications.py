import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="10.3 Structured Output Applications | GenAI Roadmap",
    page_icon="🚀",
    layout="wide",
)

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
    .app-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
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
st.title("🚀 10.3 Structured Output Applications — Enterprise Integration & Automation")
st.markdown(
    """
Welcome to **Lesson 10.3: Structured Output Applications**.  
Generating valid JSON is not an end in itself—it is the bridge that allows generative intelligence to drive **enterprise software systems**.

In this lesson, we explore how structured outputs power **6 Core Production Workflows**: from automated database insertion and CRM ingestion to microservice API routing and autonomous agent action execution.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 6 Enterprise Application Archetypes
# ==============================================================================
st.header("1. The 6 Production Application Archetypes", divider="rainbow")
st.markdown("Structured outputs serve as the translation layer between messy real-world text and rigid software backends:")

apps_df = pd.DataFrame({
    "Application Domain": [
        "1. Information Extraction",
        "2. API Response Synthesis",
        "3. Database / ORM Insertion",
        "4. Multi-Label Classification",
        "5. Automation / RPA Workflows",
        "6. Autonomous Agent Tool Outputs"
    ],
    "Input (Unstructured)": [
        "PDF invoices, clinical trial notes, resumes, scanned receipts",
        "Customer natural language queries across support chat",
        "Free-text user feedback, product reviews, audit logs",
        "Customer emails, support tickets, regulatory filings",
        "Process instructions, bug reports, migration guides",
        "Agent observations, environment states, reasoning traces"
    ],
    "Output (Structured)": [
        "Normalized JSON entity maps with ISO timestamps and numeric quantities",
        "Standardized REST API payloads conforming to OpenAPI specifications",
        "Parameterized SQL row dictionaries matching database table columns",
        "Categorical tags with confidence probabilities and severity enums",
        "Deterministic step-by-step DAG task definitions for orchestration runners",
        "Tool invocation payloads: `tool_name` + typed keyword arguments"
    ],
    "Downstream Target System": [
        "ERP (SAP, NetSuite), HRMS (Workday), Document Repositories",
        "Mobile apps, web frontends, internal search engines",
        "PostgreSQL, Snowflake, BigQuery, DynamoDB",
        "Zendesk, Salesforce CRM, PagerDuty incident routers",
        "Temporal, Celery, Airflow, Zapier webhooks",
        "LangGraph, AutoGen, CrewAI agent execution engines"
    ]
})
st.dataframe(apps_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Real-World Ingestion Pipelines
# ==============================================================================
st.header("2. Real-World Case Studies: Ingestion Deep Dives", divider="rainbow")

col_cs1, col_cs2 = st.columns(2)

with col_cs1:
    with st.container(border=True):
        st.subheader("🏥 Healthcare Clinical Notes Extraction")
        st.markdown("**Unstructured Doctor's Note:**")
        st.text("""Patient Maria Gonzales, 48F, presents with acute migraine.
Prescribed Sumatriptan 50mg PO PRN (max 200mg/day).
History of controlled hypertension (Lisinopril 10mg daily).
No known drug allergies. Follow-up in 4 weeks.""")
        st.markdown("**Validated FHIR-Compliant JSON Extraction:**")
        st.code(
            '''{
  "patient": {
    "name": "Maria Gonzales",
    "age": 48,
    "gender": "FEMALE"
  },
  "chief_complaint": "Acute migraine",
  "prescriptions": [
    {
      "medication": "Sumatriptan",
      "dosage_mg": 50,
      "route": "PO",
      "frequency": "PRN",
      "max_daily_mg": 200
    }
  ],
  "comorbidities": ["Hypertension"],
  "allergies": [],
  "follow_up_weeks": 4
}''',
            language="json",
        )

with col_cs2:
    with st.container(border=True):
        st.subheader("💳 Fintech Fraud & Transaction Classifier")
        st.markdown("**Unstructured Transaction Log:**")
        st.text("""Alert: Card ending in 4102 attempted $4,850.00 withdrawal at 03:15 AM
from ATM in Kiev, Ukraine. Previous transaction was $14.20 coffee
in San Francisco, CA at 01:45 AM (90 minutes prior).""")
        st.markdown("**Validated Fraud Detection Action Payload:**")
        st.code(
            '''{
  "card_last4": "4102",
  "anomaly_detected": true,
  "risk_score": 0.98,
  "reasons": [
    "Impossible travel velocity (SF to Kiev in 90 min)",
    "Abnormal withdrawal volume ($4,850 vs $200 historical avg)",
    "High-risk geographic zone and unusual hour"
  ],
  "automated_action": "FREEZE_CARD_IMMEDIATELY",
  "notify_channels": ["SMS_CHALLENGE", "PUSH_NOTIFICATION"]
}''',
            language="json",
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Unstructured Ingestion to Enterprise Storage", divider="rainbow")
st.markdown("End-to-end data flow showing how structured LLM output populates downstream transactional databases:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        doc [label="Unstructured Doc\n(Email / PDF / Note)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        llm [label="LLM Ingestion\n(Strict JSON Schema)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        parser [label="Pydantic\nType Validation", fillcolor="#EAB308", fontcolor="#000000"]
        sql_builder [label="Parameterized\nQuery Builder", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        db [label="PostgreSQL / BigQuery\n(Transactional Commit)", fillcolor="#10B981", fontcolor="#FFFFFF"]
        dlq [label="Dead-Letter Queue (DLQ)\n(Audit & Human Review)", fillcolor="#EF4444", fontcolor="#FFFFFF"]

        doc -> llm -> parser
        parser -> sql_builder [label=" Schema Valid", color="#10B981"]
        parser -> dlq [label=" Parse Failure", color="#EF4444"]
        sql_builder -> db [label=" INSERT / UPSERT", color="#10B981"]
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Production Application Simulator
# ==============================================================================
st.header("4. Interactive Production Application Simulator", divider="rainbow")
st.markdown("Simulate how different enterprise business functions transform unstructured content into actionable payloads:")

selected_app = st.selectbox(
    "Select Enterprise Domain Archetype",
    [
        "Fintech Transaction Auditing",
        "Healthcare Patient Intake",
        "E-Commerce Catalog Auto-Tagging",
        "Autonomous Customer Support Routing"
    ]
)

sim_col1, sim_col2 = st.columns([1, 1])

with sim_col1:
    st.subheader("📥 Inbound Raw Data Feed")
    if selected_app == "Fintech Transaction Auditing":
        inbound_text = st.text_area(
            "Transaction Telemetry",
            value="Merchant: Quantum Cloud Inc. Amount: $18,400.00. Categorized as Software Subscription by user dev-ops-9. Paid via Corporate AMEX. Requires secondary VP approval if over $10,000.",
            height=140
        )
    elif selected_app == "Healthcare Patient Intake":
        inbound_text = st.text_area(
            "Patient Triage Form",
            value="Patient James Kowalski, DOB 1985-04-12. Reports severe chest pain radiating to left arm. Blood pressure 165/105. SpO2 93%. Triage priority red. Assigned to trauma bay 3.",
            height=140
        )
    elif selected_app == "E-Commerce Catalog Auto-Tagging":
        inbound_text = st.text_area(
            "Product Description Text",
            value="Titanium Wireless Noise-Cancelling Headphones with 40-hour battery life, USB-C fast charging, Bluetooth 5.3, water resistance rating IPX4. Category: Audio Equipment. Price: $299.99.",
            height=140
        )
    else:
        inbound_text = st.text_area(
            "Support Ticket Raw Content",
            value="I was double charged on my annual subscription yesterday! Transaction IDs are TX-881 and TX-882. Please refund TX-882 immediately or I will contact my bank. Tier: Enterprise VIP.",
            height=140
        )

with sim_col2:
    st.subheader("📤 Downstream Execution Payload")
    if selected_app == "Fintech Transaction Auditing":
        payload = {
            "vendor": "Quantum Cloud Inc",
            "amount_usd": 18400.00,
            "cost_center": "Software Subscription",
            "payment_method": "CORPORATE_AMEX",
            "requires_vp_signoff": True,
            "approval_routing": "vp_finance_queue"
        }
        st.json(payload)
        st.code("""-- Automated SQL Ingestion
INSERT INTO corporate_expenses (vendor, amount, cost_center, payment_method, requires_vp_signoff)
VALUES ('Quantum Cloud Inc', 18400.00, 'Software Subscription', 'CORPORATE_AMEX', TRUE);""", language="sql")

    elif selected_app == "Healthcare Patient Intake":
        payload = {
            "patient_name": "James Kowalski",
            "dob": "1985-04-12",
            "systolic_bp": 165,
            "diastolic_bp": 105,
            "spo2_percent": 93,
            "triage_level": "LEVEL_1_CRITICAL",
            "assigned_location": "TRAUMA_BAY_3"
        }
        st.json(payload)
        st.code("""-- EHR System Event Dispatch
EMR_QUEUE.publish("EMERGENCY_TRIAGE_ALERT", payload)""", language="python")

    elif selected_app == "E-Commerce Catalog Auto-Tagging":
        payload = {
            "product_name": "Titanium Wireless Headphones",
            "category": "Consumer Electronics > Audio",
            "price_usd": 299.99,
            "specs": {
                "battery_hours": 40,
                "connectivity": ["Bluetooth 5.3", "USB-C"],
                "ip_rating": "IPX4"
            },
            "seo_tags": ["wireless", "noise-cancelling", "40hr-battery", "ipx4"]
        }
        st.json(payload)
        st.code("""-- Elasticsearch Product Indexing Document
es.index(index="product_catalog", id="prod_titanium_hp", document=payload)""", language="python")

    else:
        payload = {
            "ticket_type": "BILLING_DISPUTE",
            "urgency": "HIGH",
            "user_tier": "ENTERPRISE_VIP",
            "action_required": "INITIATE_REFUND",
            "reference_ids": ["TX-881", "TX-882"],
            "refund_target": "TX-882",
            "assignee_group": "tier3_billing_specialists"
        }
        st.json(payload)
        st.code("""-- Zendesk API Dispatch
zendesk.tickets.create(priority="urgent", tags=["billing", "vip"], custom_fields=payload)""", language="python")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: SQL Database Ingestion", divider="rainbow")
st.markdown("Safely transform validated LLM structured output into parameterized SQL inserts to prevent injection attacks:")

st.code(
    '''import sqlite3
import json
from pydantic import BaseModel, Field

# 1. Pydantic Extraction Schema
class ExpenseRecord(BaseModel):
    vendor: str
    amount_usd: float = Field(ge=0.0)
    category: str
    is_billable: bool = False

# 2. Database Ingestion Pipeline
def insert_expense_to_db(db_conn: sqlite3.Connection, record: ExpenseRecord) -> int:
    """
    Safely inserts validated structured LLM output into an SQL database.
    Uses parameterized queries to prevent SQL injection vulnerabilities.
    """
    sql = """
    INSERT INTO company_expenses (vendor, amount_usd, category, is_billable)
    VALUES (?, ?, ?, ?);
    """
    cursor = db_conn.cursor()
    cursor.execute(sql, (record.vendor, record.amount_usd, record.category, record.is_billable))
    db_conn.commit()
    return cursor.lastrowid

# 3. Execution Example
# Simulating validated output from LLM:
llm_dict = {
    "vendor": "Datadog Cloud Monitoring",
    "amount_usd": 3450.75,
    "category": "Infrastructure Observability",
    "is_billable": False
}

validated_record = ExpenseRecord(**llm_dict)
conn = sqlite3.connect(":memory:")
conn.execute("""
CREATE TABLE company_expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor TEXT,
    amount_usd REAL,
    category TEXT,
    is_billable BOOLEAN
);
""")

inserted_id = insert_expense_to_db(conn, validated_record)
print(f"Successfully committed expense record with DB ID: {inserted_id}")
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: Why should you never construct SQL queries by asking an LLM to generate raw SQL strings directly from user prompts?", expanded=True):
        st.markdown(
            """
1. **Severe SQL Injection Vulnerabilities**: If untrusted user input is passed into an LLM asked to generate SQL, prompt injection can trick the model into emitting `DROP TABLE` or data exfiltration queries (`UNION SELECT credit_cards...`).
2. **Schema Hallucinations**: LLMs frequently hallucinate column names, join tables incorrectly, or use dialect-incompatible functions.
3. **Best Practice Architecture**: Ask the LLM to output **strictly typed JSON parameters** (e.g. `{"filter_field": "status", "filter_value": "active"}`), validate that JSON with Pydantic, and construct the SQL query programmatically in your backend using parameterized queries or an ORM.
"""
        )
    with st.expander("Q2: How do you design a Dead-Letter Queue (DLQ) for failed structured extractions in enterprise production?"):
        st.markdown(
            """
When a document fails schema validation after 2 retry attempts:
1. **Preserve Raw Inputs**: Capture the original document payload, the failed LLM output, and the exact validation error trace.
2. **Push to DLQ Topic**: Publish the error packet to a Kafka topic or AWS SQS Dead-Letter Queue.
3. **Human Triage Workflow**: Surface the failed record on an internal dashboard where human operators can either manually correct the fields or update the schema prompt if a new edge case was discovered.
4. **Zero Data Loss**: This guarantees that corrupt inputs never poison downstream databases while ensuring 100% of inputs are accounted for.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we directly parse LLM JSON output and execute database updates without validation?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"If we use OpenAI's JSON mode, it's already guaranteed to be valid JSON, so we can run `db.execute(insert_stmt)` right away."*
- **Winning Response**: *"Never. JSON Mode only guarantees valid JSON syntax—it does not validate semantic bounds, business logic, or type ranges (e.g., negative prices, invalid foreign keys, or hallucinated enum values). You must always pass the parsed dictionary through a Pydantic or schema validation layer to verify business constraints before running transactional database commits."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Structured Ingestion**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The automated pipeline of extracting unstructured text into validated JSON and committing it to databases or microservices.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Dead-Letter Queue (DLQ)**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** A storage queue holding extraction payloads that failed validation retries, routing them for human review without stalling the pipeline.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Parameterized DB Commit**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Inserting validated structured LLM outputs into database tables using parameterized placeholders (`?` or `:val`) to prevent SQL injection.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 10.3 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Integration Layer**: Structured outputs transform text into typed inputs for databases, APIs, and agents.
2. **Security First**: Never let LLMs generate unparameterized raw SQL strings directly.
3. **Dead-Letter Queues**: Route irrecoverable validation errors to DLQs for audit and human review.
4. **End-to-End Type Safety**: Pydantic models ensure strict typing between LLM inference and database ORMs.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know how to map unstructured documents to relational database schemas.", value=True, key="chk_10_3_1")
    st.checkbox("I understand why parameterized queries are required when inserting LLM data into SQL.", value=True, key="chk_10_3_2")
    st.checkbox("I can architect a Dead-Letter Queue (DLQ) pattern for failed extraction payloads.", value=True, key="chk_10_3_3")
    st.checkbox("I can design tool-calling outputs for autonomous agent action execution.", value=True, key="chk_10_3_4")
