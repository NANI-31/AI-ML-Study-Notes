import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="20.2 Multiple Data Sources RAG",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class FederatedMultiSourceEngine:
    """Enterprise Ingestion & Retrieval Hub connecting heterogeneous data sources: PDFs, Web, SQL, APIs, Cloud & Wikis."""

    def __init__(self):
        # Heterogeneous Enterprise Source Store
        self.sources = {
            "PDFs": [
                {"id": "pdf_sec_10k", "name": "Annual_Report_2024.pdf", "type": "PDF", "content": "Section 4.1: Net income reached $18.4 billion with R&D allocation totaling $4.2B.", "latency_ms": 42.0}
            ],
            "Websites": [
                {"id": "web_corp_portal", "name": "https://corp.example.com/press", "type": "Website", "content": "Press Release: Corporation announces expansion of AI compute clusters in Ireland.", "latency_ms": 110.0}
            ],
            "Databases": [
                {"id": "sql_orders_db", "name": "PostgreSQL: analytics.quarterly_orders", "type": "SQL Database", "content": "SELECT region, SUM(amount) FROM orders GROUP BY region -> NA: $12.1B, EMEA: $8.4B, APAC: $5.2B", "latency_ms": 18.0}
            ],
            "APIs": [
                {"id": "api_stripe_v1", "name": "REST API: api.stripe.com/v1/balance", "type": "REST API", "content": "JSON Payload: {'available': [{'currency': 'usd', 'amount': 452000000}]}", "latency_ms": 85.0}
            ],
            "Cloud Storage": [
                {"id": "s3_lake_telemetry", "name": "s3://corp-lake/prod/telemetry_manifest.parquet", "type": "S3 Cloud Bucket", "content": "Parquet Schema: 99.98% cluster uptime recorded across 4,200 GPU instances.", "latency_ms": 65.0}
            ],
            "Knowledge Bases": [
                {"id": "notion_arch_wiki", "name": "Confluence / Notion: Architecture Codex", "type": "Enterprise Wiki", "content": "Article 88: Microservices communicate via gRPC with mTLS authentication strictly enforced.", "latency_ms": 35.0}
            ]
        }

    def query_source(self, source_name: str, query: str) -> List[Dict[str, Any]]:
        """Queries an individual data connector and standardizes output to a unified manifest."""
        records = self.sources.get(source_name, [])
        normalized = []
        for r in records:
            normalized.append({
                "record_id": r["id"],
                "source_category": source_name,
                "asset_name": r["name"],
                "protocol": r["type"],
                "normalized_text": r["content"],
                "connector_latency_ms": r["latency_ms"]
            })
        return normalized

    def federated_scatter_gather(self, query: str, active_sources: List[str]) -> Dict[str, Any]:
        """Dispatches federated query in parallel across enabled sources and normalizes schemas."""
        all_records = []
        total_latency = 0.0
        
        for src in active_sources:
            res = self.query_source(src, query)
            all_records.extend(res)
            # Parallel fan-out: max latency across sources
            if res:
                total_latency = max(total_latency, res[0]["connector_latency_ms"])

        unified_context = "\n\n".join([
            f"[{r['source_category']} | {r['asset_name']}]: {r['normalized_text']}"
            for r in all_records
        ])

        return {
            "query": query,
            "records": all_records,
            "total_records": len(all_records),
            "critical_path_latency_ms": total_latency + 12.0,  # 12ms normalization overhead
            "unified_context": unified_context
        }


# --- Streamlit Layout ---

st.title("20.2 Multiple Data Sources in RAG")
st.caption("Heterogeneous Connectors: PDFs, Websites, SQL Databases, REST APIs, Cloud Object Storage & Enterprise Wikis")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Federated Multi-Source Enterprise Ingestion & Retrieval Hub", divider="rainbow")

st.markdown(
    """
    Enterprise data is fragmented across distinct storage systems and protocol boundaries.
    Engineering specifications reside in Notion wikis, transactional data lives in relational SQL databases,
    annual earnings are locked in scanned PDF reports, real-time balances require live REST API calls,
    and bulk analytical logs are stored in cloud object buckets (Amazon S3 / GCS).
    
    A production **Multi-Source RAG Architecture** acts as a **Federated Query Engine**, routing queries to specialized
    source connectors, standardizing diverse schemas into canonical document chunks, and assembling unified context for LLM synthesis.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Federated_Multi_Source {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_ingress {
            label = "1. FEDERATED QUERY DISPATCH";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_query [label="Unified User Query\\n'Audit European cluster revenue & infrastructure'", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            federated_hub [label="Federated Ingestion & Retrieval Hub\\n(Parallel Async Scatter-Gather Dispatcher)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            
            user_query -> federated_hub;
        }

        subgraph cluster_heterogeneous_sources {
            label = "2. HETEROGENEOUS ENTERPRISE DATA SOURCES";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            src_pdf [label="1. PDF Documents\\nLayout OCR & PyMuPDF\\n(Tables & Scanned Text)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            src_web [label="2. Corporate Websites\\nPlaywright / Trafilatura\\n(DOM Boilerplate Stripping)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            src_db [label="3. SQL / NoSQL Databases\\nText-to-SQL & JDBC\\n(Postgres / Snowflake / Mongo)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            src_api [label="4. External REST / GraphQL\\nOpenAPI & OAuth2\\n(Stripe / Salesforce / Workday)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            src_cloud [label="5. Cloud Object Buckets\\nS3 / GCS / Azure Blob\\n(Parquet / Avro / Delta Lake)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            src_wiki [label="6. Enterprise Knowledge Base\\nConfluence / Notion API\\n(Hierarchical Markdown Trees)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            federated_hub -> src_pdf [label="Async RPC"];
            federated_hub -> src_web [label="Crawl Fetch"];
            federated_hub -> src_db [label="SQL Query"];
            federated_hub -> src_api [label="HTTP GET"];
            federated_hub -> src_cloud [label="S3 Boto3"];
            federated_hub -> src_wiki [label="REST Pull"];
        }

        subgraph cluster_normalization {
            label = "3. NORMALIZATION & CANONICAL CHUNK SCHEMA";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            canonical_normalizer [label="Canonical Normalization Layer\\n- Pydantic Document Schema\\n- Strip HTML / Markdown Encodings\\n- Universal Provenance Metadata", fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            src_pdf -> canonical_normalizer;
            src_web -> canonical_normalizer;
            src_db -> canonical_normalizer;
            src_api -> canonical_normalizer;
            src_cloud -> canonical_normalizer;
            src_wiki -> canonical_normalizer;
        }

        subgraph cluster_synthesis {
            label = "4. CROSS-SOURCE SYNTHESIS & GENERATION";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            rerank_fusion [label="Cross-Source Fusion & Reranker\\n(Balancing SQL Precision vs PDF Narrative)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            frontier_llm [label="Frontier LLM Synthesizer\\n(Joint Multi-Source Generation & Provenance)", fillcolor="#0F172A", fontcolor="#10B981"];
            verified_output [label="Enterprise Multi-Source Response\\nWith Inline Attribution [SQL_1, S3_Parquet]", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            canonical_normalizer -> rerank_fusion -> frontier_llm -> verified_output;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: PDFS ---
st.header("2. PDFs (Layout Analysis, OCR Scans, Multi-Column Text & Embedded Tables)", divider="rainbow")

st.markdown(
    """
    PDFs represent over 60% of enterprise knowledge assets, yet they are the most challenging format to ingest:
    - **Visual Reading Order & Multi-Column Layouts**: Slicing PDF text naively by character coordinates reads across columns horizontally, scrambling paragraphs into gibberish.
    - **Embedded Tables**: Tabular financial reports require specialized table extractors (such as Table Transformers or OCR-based bounding boxes) to convert cells into structured markdown tables.
    - **Scanned Image Documents**: Historical documents lack text layers entirely, requiring high-accuracy OCR pipelines (Tesseract, AWS Textract, or vision models).
    """
)

# --- SECTION 3: WEBSITES ---
st.header("3. Websites (DOM Extraction, Dynamic JavaScript Rendering, Boilerplate Stripping & Crawl Policies)", divider="rainbow")

st.markdown(
    """
    Ingesting web pages is far more complex than executing a simple HTTP `requests.get()`:
    - **Dynamic Client-Side JavaScript**: Modern single-page applications (React, Next.js) return blank HTML templates; scrapers must execute headless Chromium browsers (Playwright, Puppeteer) to hydrate the DOM.
    - **Boilerplate & Noise Stripping**: 70% of a raw HTML page consists of navigation menus, header bars, footer legal notices, and tracking scripts. Libraries like `Trafilatura` or `Readability` extract the pure article body.
    - **Politeness & Rate Limits**: Enterprise web ingestion engines must respect `robots.txt`, adhere to crawl budgets, and cache unchanged pages.
    """
)

# --- SECTION 4: DATABASES ---
st.header("4. Databases (Relational SQL Catalogs, Schema Embeddings & Hybrid Vector-Relational Joins)", divider="rainbow")

st.markdown(
    """
    Databases contain ground-truth structured transactions (orders, inventory, financial ledgers).
    Text embeddings cannot accurately compute mathematical aggregations like `SUM(revenue)`.
    
    ### Bridging Databases and RAG:
    1. **Text-to-SQL Pipelines**: The agent translates natural language queries into parameterized SQL statements executed directly on PostgreSQL, Snowflake, or BigQuery.
    2. **Schema Embedding**: Database catalog tables and column descriptions are indexed in vector databases, allowing the router to discover relevant tables dynamically.
    3. **Hybrid Joins**: The pipeline joins unstructured customer support tickets with structured order history tables using foreign keys (`customer_id`).
    """
)

# --- SECTION 5: APIS ---
st.header("5. APIs (REST/GraphQL Payloads, JSON Normalization, Rate Limiting & Auth Tokens)", divider="rainbow")

st.markdown(
    """
    Live business operations require interacting with SaaS REST and GraphQL endpoints (Salesforce, Stripe, Jira, ServiceNow):
    - **Schema Normalization**: API payloads return complex nested JSON structures filled with system IDs and timestamps. The ingestion layer must flatten and summarize payloads into human-readable text chunks.
    - **Authentication & Secret Management**: Vault-managed OAuth2 access tokens and API keys must be injected securely without leaking credentials into prompt logs.
    - **Rate Limit Throttling**: API gateways enforce exponential backoff and circuit breaking when rate limits (HTTP `429 Too Many Requests`) are reached.
    """
)

# --- SECTION 6: CLOUD STORAGE ---
st.header("6. Cloud Storage (S3/GCS/Azure Blob Connectors, Object Lifecycle & Asynchronous Ingestion)", divider="rainbow")

st.markdown(
    """
    Data lakes store petabytes of analytical data in object storage (Amazon S3, Google Cloud Storage, Azure Blob):
    - **Analytical Formats**: Parsing column-oriented files (Apache Parquet, ORC, Delta Lake) requires specialized pyarrow engines rather than raw text parsers.
    - **Event-Driven Ingestion**: Using AWS S3 event notifications (SNS / SQS) to trigger asynchronous Lambda or Kubernetes ingestion workers whenever a new file is uploaded.
    - **Cold Storage Latency**: Handling object retrieval delays when documents reside in glacier or cold storage archive tiers.
    """
)

# --- SECTION 7: KNOWLEDGE BASES ---
st.header("7. Knowledge Bases (Structured Ontologies, Enterprise Wikis, Confluence & Notion Sync)", divider="rainbow")

st.markdown(
    """
    Enterprise wikis (Confluence, Notion, SharePoint) contain team collaboration knowledge:
    - **Hierarchical Tree Structures**: Wiki pages exist in nested parent-child hierarchies. Ingestion must capture breadcrumb paths (`Engineering > Infrastructure > Kubernetes > Storage`) to provide contextual anchoring.
    - **Real-Time Incremental Sync**: Webhook-based change detection syncs edited pages within seconds, preventing knowledge bases from serving stale internal documentation.
    - **Access Control & ACL Filtering**: Ensuring search results respect user-specific document permissions and enterprise role-based access control (RBAC).
    """
)

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Unified Federated Multi-Source Ingestion & Retrieval Engine", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `FederatedMultiSourceEngine`.
    It orchestrates heterogeneous connectors across PDFs, web portals, SQL databases, REST APIs, cloud object stores, and wikis,
    executing parallel scatter-gather queries and normalizing diverse schemas into a canonical manifest.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class FederatedMultiSourceEngine:
    """Enterprise Ingestion & Retrieval Hub connecting heterogeneous data sources."""

    def __init__(self):
        self.sources = {
            "PDFs": [
                {"id": "pdf_sec_10k", "name": "Annual_Report_2024.pdf", "type": "PDF", "content": "Section 4.1: Net income reached $18.4 billion with R&D allocation totaling $4.2B.", "latency_ms": 42.0}
            ],
            "Websites": [
                {"id": "web_corp_portal", "name": "https://corp.example.com/press", "type": "Website", "content": "Press Release: Corporation announces expansion of AI compute clusters in Ireland.", "latency_ms": 110.0}
            ],
            "Databases": [
                {"id": "sql_orders_db", "name": "PostgreSQL: analytics.quarterly_orders", "type": "SQL Database", "content": "SELECT region, SUM(amount) FROM orders GROUP BY region -> NA: $12.1B, EMEA: $8.4B, APAC: $5.2B", "latency_ms": 18.0}
            ],
            "APIs": [
                {"id": "api_stripe_v1", "name": "REST API: api.stripe.com/v1/balance", "type": "REST API", "content": "JSON Payload: {'available': [{'currency': 'usd', 'amount': 452000000}]}", "latency_ms": 85.0}
            ],
            "Cloud Storage": [
                {"id": "s3_lake_telemetry", "name": "s3://corp-lake/prod/telemetry_manifest.parquet", "type": "S3 Cloud Bucket", "content": "Parquet Schema: 99.98% cluster uptime recorded across 4,200 GPU instances.", "latency_ms": 65.0}
            ],
            "Knowledge Bases": [
                {"id": "notion_arch_wiki", "name": "Confluence / Notion: Architecture Codex", "type": "Enterprise Wiki", "content": "Article 88: Microservices communicate via gRPC with mTLS authentication strictly enforced.", "latency_ms": 35.0}
            ]
        }

    def query_source(self, source_name: str, query: str) -> List[Dict[str, Any]]:
        """Queries an individual data connector and standardizes output to a unified manifest."""
        records = self.sources.get(source_name, [])
        normalized = []
        for r in records:
            normalized.append({
                "record_id": r["id"],
                "source_category": source_name,
                "asset_name": r["name"],
                "protocol": r["type"],
                "normalized_text": r["content"],
                "connector_latency_ms": r["latency_ms"]
            })
        return normalized

    def federated_scatter_gather(self, query: str, active_sources: List[str]) -> Dict[str, Any]:
        """Dispatches federated query in parallel across enabled sources and normalizes schemas."""
        all_records = []
        total_latency = 0.0
        
        for src in active_sources:
            res = self.query_source(src, query)
            all_records.extend(res)
            if res:
                total_latency = max(total_latency, res[0]["connector_latency_ms"])

        unified_context = "\\n\\n".join([
            f"[{r['source_category']} | {r['asset_name']}]: {r['normalized_text']}"
            for r in all_records
        ])

        return {
            "query": query,
            "records": all_records,
            "total_records": len(all_records),
            "critical_path_latency_ms": total_latency + 12.0,
            "unified_context": unified_context
        }
''',
    language="python"
)

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Heterogeneous Source Connector & Normalization Sandbox", divider="rainbow")

st.markdown(
    """
    Configure federated scatter-gather queries across enterprise data sources.
    Select which data connectors to activate, submit cross-system questions, and inspect schema normalization and latency metrics.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🔌 Heterogeneous Source Connectors")
    
    query_input = st.text_input(
        "Enter Federated Query:",
        "Audit global infrastructure uptime, regional revenue, and Ireland datacenter expansion"
    )
    
    st.markdown("**Select Active Data Sources:**")
    include_pdf = st.toggle("Include PDF Reports (SEC 10-K)", value=True)
    include_web = st.toggle("Include Corporate Web Portals", value=True)
    include_sql = st.toggle("Include SQL Relational Databases", value=True)
    include_api = st.toggle("Include REST / GraphQL APIs", value=True)
    include_cloud = st.toggle("Include S3 Cloud Object Storage", value=True)
    include_wiki = st.toggle("Include Enterprise Wikis (Confluence/Notion)", value=True)

    active_sources = []
    if include_pdf: active_sources.append("PDFs")
    if include_web: active_sources.append("Websites")
    if include_sql: active_sources.append("Databases")
    if include_api: active_sources.append("APIs")
    if include_cloud: active_sources.append("Cloud Storage")
    if include_wiki: active_sources.append("Knowledge Bases")

with wb_col2:
    st.subheader("📊 Federated Ingestion Telemetry")
    
    engine = FederatedMultiSourceEngine()
    results = engine.federated_scatter_gather(query_input, active_sources)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Active Connectors", len(active_sources))
    m2.metric("Normalized Records", results["total_records"])
    m3.metric("Critical-Path Latency", f"{results['critical_path_latency_ms']} ms")
    
    st.markdown("#### Unified Normalized Prompt Context")
    if results["unified_context"]:
        st.info(results["unified_context"])
    else:
        st.warning("No data sources enabled. Please toggle at least one connector.")

st.markdown("### Normalized Multi-Source Manifest")
if results["records"]:
    records_df = pd.DataFrame(results["records"])
    st.dataframe(records_df, use_container_width=True)

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: Why is parallel scatter-gather query execution mandatory for multi-source RAG?**"):
    st.markdown(
        """
        If a pipeline queries 6 separate enterprise data sources sequentially:
        `Total_Latency = Latency_PDF (40ms) + Latency_Web (120ms) + Latency_SQL (20ms) + Latency_API (90ms) + Latency_S3 (65ms) + Latency_Wiki (35ms) = 370ms`.
        
        Under synchronous execution, latency degrades linearly with every new connector.
        **Parallel Scatter-Gather** dispatches all 6 queries concurrently using `asyncio` or worker threads.
        Total latency is bounded by the slowest single source plus minimal gathering overhead:
        `Total_Latency = max(Latencies) + Overhead = 120ms + 12ms = 132ms`, saving over 60% of critical-path runtime.
        """
    )

with st.expander("**Q2: How do you handle schema drift when ingesting third-party REST APIs into RAG?**"):
    st.markdown(
        """
        External SaaS APIs frequently add, rename, or deprecate JSON attributes without notice.
        Production architectures implement:
        1. **Strict Pydantic Contract Validation**: Ingestion parsers validate payloads against versioned schemas.
        2. **Semantic Flattening**: Converting nested JSON key-value pairs into human-readable synthetic sentences (*'Customer status is ACTIVE with 12 open tickets'*).
        3. **Dead-Letter Queues (DLQ)**: Malformed payloads that fail schema validation are routed to a DLQ for offline inspection rather than crashing the real-time search pipeline.
        """
    )

with st.expander("**Q3: What is the optimal ingestion architecture for parsing multi-column PDF reports?**"):
    st.markdown(
        """
        Standard PDF text extractors (such as basic `pypdf`) sort characters by raw stream order, which reads across columns horizontally, corrupting multi-column text.
        
        Production pipelines use **Vision-Based Document Layout Parsers** (e.g., LayoutLMv3, Surya, or PyMuPDF with block detection):
        1. Group bounding boxes into structural visual columns.
        2. Detect reading order vectors top-to-bottom per column.
        3. Isolate table bounding boxes and extract cells using dedicated table transformer models into Markdown format before vector embedding.
        """
    )

with st.expander("**Q4: How do you secure Multi-Source RAG against cross-tenant data leaks and permission violations?**"):
    st.markdown(
        """
        In multi-source systems, different users have different read permissions across Confluence, SQL databases, and cloud storage.
        
        To prevent unauthorized data exposure:
        1. **Pre-Filtering (ACL Gating)**: The user's Active Directory / Okta groups and roles are passed to each data connector. Documents are filtered by access control lists (ACLs) *before* retrieval.
        2. **Post-Retrieval Verification**: Any retrieved chunk whose security tags exceed user permissions is discarded before prompt injection.
        3. **Never rely on LLM system prompts for authorization**: LLMs cannot be trusted to self-enforce permissions.
        """
    )

with st.expander("**Q5: How do you bridge structured SQL databases with unstructured vector embeddings in a single pipeline?**"):
    st.markdown(
        """
        Vector search cannot perform mathematical operations like sums, averages, or exact joins.
        
        Production architectures use **Hybrid Text-to-SQL + Vector Routing**:
        1. A router classifies whether the query requires numerical aggregation or semantic narrative.
        2. If aggregation is required, an LLM generates a SQL query executed against the relational engine.
        3. The resulting SQL summary table is injected as a verified context chunk alongside unstructured narrative chunks.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Synchronous Sequential Source Polling**  
        **Antipattern**: Querying databases, APIs, and web scrapers sequentially in a single synchronous loop.  
        **Consequence**: Adds up the latency of every connector; a single slow API call causes the entire user interaction to hang or timeout.  
        **Production Fix**: Execute all connector lookups concurrently using asynchronous scatter-gather patterns with per-source timeout budgets.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Feeding Raw Nested JSON Directly to LLMs**  
        **Antipattern**: Ingesting entire 5,000-line JSON API responses with deep nesting and metadata into the context window.  
        **Consequence**: Exhausts token budgets on braces, null keys, and system IDs, inducing heavy attention distraction.  
        **Production Fix**: Flatten and summarize JSON payloads into structured markdown tables or concise natural language clauses before injection.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Scraping Web Pages Without Headless DOM Execution**  
        **Antipattern**: Relying strictly on basic HTTP GET requests to scrape dynamic corporate web portals.  
        **Consequence**: Returns blank HTML wrappers missing client-rendered React/Next.js content, causing zero-hit retrieval failures.  
        **Production Fix**: Deploy headless browser runners (Playwright) to hydrate JavaScript DOMs before content extraction.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Neglecting Rate Limits and OAuth Expiry on Enterprise APIs**  
        **Antipattern**: Invoking third-party SaaS APIs on every user search without caching or token renewal handling.  
        **Consequence**: Triggers HTTP 429 rate limit bans and authentication failures during peak traffic hours.  
        **Production Fix**: Implement token caching in Redis and wrap API calls in circuit breakers with exponential backoff.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Federated Multi-Source RAG?")
        st.markdown("**Back**: An architecture that queries heterogeneous enterprise storage systems (PDFs, SQL DBs, REST APIs, Object Stores) in parallel and normalizes their outputs into a unified context.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Why use Scatter-Gather Querying?")
        st.markdown("**Back**: Dispatches queries concurrently across all active data connectors, ensuring total latency is bounded by the slowest single connector rather than the sum of all connectors.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: Why is Layout Analysis needed for PDFs?")
        st.markdown("**Back**: PDF text is stored without semantic reading order. Layout analysis groups bounding boxes into multi-column paragraphs and tables to prevent text scrambling.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Why can't Vector Search replace SQL?")
        st.markdown("**Back**: Vector embeddings compute geometric similarity, not mathematical aggregations. Exact metrics, sums, and filters require structured relational SQL engines.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of multi-source RAG architectures and heterogeneous data connectors:")

chk1 = st.checkbox("I understand the end-to-end architecture of a federated multi-source scatter-gather retrieval hub.", value=True)
chk2 = st.checkbox("I can identify the unique extraction challenges of PDFs, dynamic websites, SQL databases, and REST APIs.", value=True)
chk3 = st.checkbox("I know how to design asynchronous parallel dispatchers to protect critical-path latency.", value=True)
chk4 = st.checkbox("I understand why JSON payloads must be normalized and flattened before vectorization and prompt injection.", value=True)
chk5 = st.checkbox("I know how to combine structured Text-to-SQL aggregations with unstructured semantic document chunks.", value=True)
chk6 = st.checkbox("I can implement pre-filtering ACL gating and authentication safeguards across enterprise data connectors.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Enterprise Multi-Source RAG and Heterogeneous Connectors.")
