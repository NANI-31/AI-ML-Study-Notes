import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="15.2 Structured Data | GenAI Roadmap",
    page_icon="🗂️",
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
    .schema-pill {
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
st.title("🗂️ 15.2 Structured Data — CSV, JSON, Relational SQL & Schema Prompting")
st.markdown(
    """
Welcome to **Lesson 15.2: Structured Data**.  
Enterprise enterprise repositories do not store data as free-form essays—they operate on **CSVs**, **deeply nested JSON blobs**, and **relational SQL databases**.

Prompting models to reason over structured data requires choosing the right serialization representation, communicating **relational schemas without token bloat**, and preventing the hallucination of non-existent tables or columns.
"""
)

st.divider()

# ==============================================================================
# Section 1: Structured Formats Trade-Off Matrix
# ==============================================================================
st.header("1. Structured Data Serialization Trade-Offs", divider="rainbow")
st.markdown("Comparing how different data formats consume tokens and interact with LLM attention:")

formats_df = pd.DataFrame({
    "Data Format": [
        "1. Compact CSV",
        "2. Nested JSON",
        "3. Markdown Table",
        "4. SQL DDL (`CREATE TABLE`)",
        "5. YAML Dictionary"
    ],
    "Token Overhead": [
        "Lowest (Minimal delimiters; 1 header line + raw values)",
        "High (Repeats key names on every individual row/object)",
        "Moderate-High (Pipes `|` and alignment spaces add tokens)",
        "Very Low (Represents schema and constraints without raw data)",
        "Moderate (Indentation-based; no closing braces)"
    ],
    "Nesting & Hierarchy": [
        "None (Flat 2D tabular data only)",
        "Highest (Arbitrarily deep object trees and array lists)",
        "None (Flat 2D tabular data only)",
        "Relational (Foreign keys, joins, primary key constraints)",
        "High (Readable hierarchical configuration maps)"
    ],
    "Best Production Use Case": [
        "Feeding 10-50 sample tabular rows for few-shot in-context learning",
        "Document-store extraction, telemetry events, API payloads",
        "Displaying comparative results for human review in chat interfaces",
        "Text-to-SQL generation prompts, database query synthesis",
        "Configuration settings, prompt metadata, rule definitions"
    ]
})
st.dataframe(formats_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Schema Representation Strategies for Text-to-SQL
# ==============================================================================
st.header("2. Production Schema Prompting for Text-to-SQL", divider="rainbow")

col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    with st.container(border=True):
        st.subheader("❌ Naive Schema Prompt")
        st.code(
            '''Write a SQL query to find top customers from the database.
Tables: customers, orders, order_items.''',
            language="text",
        )
        st.error("🚨 Why It Breaks:")
        st.markdown(
            """
- The LLM hallucinates column names (e.g. `customer_name` vs `full_name` vs `first_name, last_name`).
- It has no idea which foreign key joins `orders` to `customers` (`customer_id` vs `user_id` vs `client_fk`).
- It produces invalid SQL that crashes the database engine immediately.
"""
        )

with col_sc2:
    with st.container(border=True):
        st.subheader("✅ Production DDL Schema Prompt")
        st.code(
            '''<database_schema dialect="PostgreSQL 16">
CREATE TABLE customers (
    id UUID PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE orders (
    id UUID PRIMARY KEY,
    customer_id UUID REFERENCES customers(id),
    total_usd NUMERIC(10, 2) NOT NULL,
    status VARCHAR(50) -- 'PENDING', 'COMPLETED', 'REFUNDED'
);
</database_schema>

<instructions>
Write a parameterized PostgreSQL query for active customers with over $10k completed orders.
Join using customers.id = orders.customer_id.
</instructions>''',
            language="xml",
        )
        st.success("🟢 Why It Succeeds:")
        st.markdown(
            """
- Explicit types (`UUID`, `NUMERIC`, `VARCHAR`)
- Foreign key relations mapped clearly (`REFERENCES customers(id)`)
- Valid enum values documented directly in column comments
"""
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: Schema Serialization to Query Execution", divider="rainbow")
st.markdown("End-to-end data flow for enterprise Text-to-SQL and structured querying:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        db [label="Enterprise Relational DB\n(PostgreSQL / Snowflake)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        schema_extractor [label="Schema Introspector\n(DDL + Enums + Sample Rows)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        prompt [label="Structured Prompt\nwith Target Dialect", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        llm [label="LLM SQL Reasoner\n(Generates Query AST)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        validator [label="SQL Validator / EXPLAIN\n(AST & Syntax Check)", fillcolor="#EAB308", fontcolor="#000000"]
        exec_results [label="Query Execution\n& Data Visualization", fillcolor="#10B981", fontcolor="#FFFFFF"]

        db -> schema_extractor -> prompt -> llm -> validator -> exec_results
        exec_results -> db [label=" Read-Only Query", style="dashed"]
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Structured Data Workbench
# ==============================================================================
st.header("4. Interactive Structured Data Workbench", divider="rainbow")
st.markdown("Test how different source schemas are interpreted to synthesize precise SQL and analytics queries:")

source_format = st.selectbox(
    "Select Source Data Representation",
    [
        "Relational Schema (PostgreSQL DDL)",
        "Nested E-Commerce Event (JSON Payload)",
        "Flat Tabular Telemetry (CSV Sample)"
    ]
)

col_sw1, col_sw2 = st.columns([1, 1])

with col_sw1:
    st.subheader("📥 Input Schema / Payload")
    if source_format == "Relational Schema (PostgreSQL DDL)":
        sample_input = """CREATE TABLE subscriptions (
    sub_id VARCHAR(64) PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    plan_tier VARCHAR(20), -- 'FREE', 'PRO', 'ENTERPRISE'
    monthly_price NUMERIC(8,2),
    created_at TIMESTAMP WITH TIME ZONE,
    is_cancelled BOOLEAN DEFAULT FALSE
);"""
    elif source_format == "Nested E-Commerce Event (JSON Payload)":
        sample_input = """{
  "event_id": "evt_99812",
  "user": {"id": "usr_402", "tier": "ENTERPRISE"},
  "cart": {
    "currency": "USD",
    "line_items": [
      {"sku": "SKU-A", "qty": 2, "price": 49.99},
      {"sku": "SKU-B", "qty": 1, "price": 120.00}
    ]
  }
}"""
    else:
        sample_input = """timestamp,device_id,cpu_pct,ram_mb,error_count
2026-10-12T12:00:00Z,srv-01,78.2,14200,0
2026-10-12T12:01:00Z,srv-01,98.5,15800,12
2026-10-12T12:02:00Z,srv-01,99.1,16100,45"""

    st.code(sample_input, language="sql" if "SQL" in source_format else ("json" if "JSON" in source_format else "text"))

with col_sw2:
    st.subheader("📤 Synthesized Analytical Query / Code")
    if source_format == "Relational Schema (PostgreSQL DDL)":
        st.markdown("**User Intent:** *Calculate Monthly Recurring Revenue (MRR) for active Pro and Enterprise tiers.*")
        st.code(
            '''SELECT 
    plan_tier,
    COUNT(sub_id) AS active_subscribers,
    SUM(monthly_price) AS total_mrr
FROM subscriptions
WHERE is_cancelled = FALSE 
  AND plan_tier IN ('PRO', 'ENTERPRISE')
GROUP BY plan_tier
ORDER BY total_mrr DESC;''',
            language="sql",
        )
        st.success("✅ Clean query matching exact column names and enum values.")

    elif source_format == "Nested E-Commerce Event (JSON Payload)":
        st.markdown("**User Intent:** *Extract total cart value and item count using Python Pandas.*")
        st.code(
            '''import json

event = json.loads(payload_str)
cart = event['cart']
total_value = sum(item['qty'] * item['price'] for item in cart['line_items'])
total_items = sum(item['qty'] for item in cart['line_items'])
print(f"User {event['user']['id']} Cart Total: ${total_value:.2f} across {total_items} items")''',
            language="python",
        )
        st.success("✅ Safely navigates nested dictionary keys without KeyError.")

    else:
        st.markdown("**User Intent:** *Detect server memory saturation threshold in Pandas.*")
        st.code(
            '''import pandas as pd
df = pd.read_csv("telemetry.csv")
# Flag records exceeding 95% CPU and non-zero errors
critical_spikes = df[(df['cpu_pct'] > 95.0) & (df['error_count'] > 0)]
print(f"Identified {len(critical_spikes)} saturation incident windows.")''',
            language="python",
        )
        st.success("✅ Fast vectorized boolean filtering across CSV columns.")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Dynamic DDL Schema Introspector", divider="rainbow")
st.markdown("Python utility to extract relational schema DDL and format token-efficient Text-to-SQL prompts:")

st.code(
    '''import sqlite3
from typing import List

def extract_schema_ddl_for_prompt(db_path: str, table_names: List[str]) -> str:
    """
    Introspects a database and generates clean, minimal DDL statements
    for inclusion in Text-to-SQL system prompts.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    ddl_statements = []
    
    for table in table_names:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,))
        result = cursor.fetchone()
        if result and result[0]:
            ddl_statements.append(result[0] + ";")
            
    conn.close()
    
    schema_block = "\\n\\n".join(ddl_statements)
    prompt = f"""<database_schema>
{schema_block}
</database_schema>

<instructions>
Write a SQLite query based on the schema above.
Return ONLY the raw SQL query. Do not wrap in markdown or conversational pleasantries.
</instructions>
"""
    return prompt

print("Schema Introspector generates production-ready <database_schema> blocks.")
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
    with st.expander("Q1: What is 'Schema Linking' in Text-to-SQL systems, and why is it essential?", expanded=True):
        st.markdown(
            """
**Schema Linking** is the process of mapping natural language tokens in the user's question directly to the specific tables, columns, and foreign keys in the database schema:
- User says: *"Show me all VIP buyers who ordered in October"*
- Schema linker maps:
  - *"VIP buyers"* -> `customers.tier = 'VIP'`
  - *"ordered"* -> `orders.created_at`
  - Table join -> `customers.id = orders.customer_id`
In production architectures, schema linking runs as an explicit preprocessing step to prune irrelevant tables before generating the final SQL query, reducing prompt tokens and eliminating hallucinated joins.
"""
        )
    with st.expander("Q2: How do you prevent an LLM from generating dangerous DDL/DML statements like `DROP TABLE` or `DELETE`?"):
        st.markdown(
            """
1. **Database Role Constraints (Principle of Least Privilege)**: Connect to the database using a **Read-Only Database User** (`SELECT` permissions only). Even if an injection attacks occurs, the database engine will reject `DROP` or `UPDATE` commands.
2. **AST SQL Parser & Blocklist**: Pass the generated SQL string through a parser (like `sqlglot`) before execution to assert that the statement type is strictly `Select` and contains no write operations.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we dump all 200 tables of our enterprise database schema into the system prompt?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Modern models have huge context windows, so we can paste the entire schema."*
- **Winning Response**: *"Never. Dumping 200 tables (10,000+ columns) causes severe attention dilution, high token cost, and catastrophic hallucination of unrelated joins. Production Text-to-SQL systems use **Dynamic Schema Pruning**: they use semantic vector search or BM25 keyword matching over table descriptions to retrieve only the top 3-6 relevant tables needed for that specific query."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Schema Linking**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Mapping natural language query entities directly to corresponding database tables, columns, and join foreign keys.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Dynamic Schema Pruning**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Retrieving only the subset of database tables relevant to a user query rather than passing the entire enterprise database schema.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Read-Only Connection Role**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Executing LLM-generated SQL queries using a database role restricted to SELECT privileges to eliminate data corruption risks.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 15.2 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Format Trade-offs**: CSV is most token-efficient for sample rows; DDL is optimal for schema definition.
2. **DDL Representation**: Always provide foreign key constraints and column comments in SQL prompts.
3. **Dynamic Schema Pruning**: Never dump hundreds of tables into context; retrieve only relevant tables.
4. **Read-Only Database Roles**: Enforce security at the database connection level, not through prompt instructions alone.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I know the token efficiency trade-offs between CSV, JSON, and DDL schemas.", value=True, key="chk_15_2_1")
    st.checkbox("I can structure production Text-to-SQL prompts with DDL schemas and foreign keys.", value=True, key="chk_15_2_2")
    st.checkbox("I understand why Schema Linking and Dynamic Pruning are mandatory for enterprise databases.", value=True, key="chk_15_2_3")
    st.checkbox("I know how to enforce database security using read-only connection credentials.", value=True, key="chk_15_2_4")
