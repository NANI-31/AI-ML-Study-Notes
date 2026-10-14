import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="20.3 Source Routing in RAG",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class GatewaySourceRouterEngine:
    """Enterprise Gateway Source Router: Classifies intent and dynamically routes queries to SQL, Web Search, or Vector DBs."""

    def __init__(self):
        self.routes = {
            "SQL_DATABASE": {
                "name": "Relational Data Warehouse (Snowflake / PostgreSQL)",
                "type": "Structured Analytical Engine",
                "avg_latency_ms": 25.0,
                "cost_tier": "Low ($0.0001 / query)",
                "description": "Handles exact numerical aggregations, sums, counts, filters, and historical metrics."
            },
            "WEB_SEARCH": {
                "name": "Live Web Search (Tavily / Google Serper)",
                "type": "Real-Time External Engine",
                "avg_latency_ms": 320.0,
                "cost_tier": "Moderate ($0.0050 / query)",
                "description": "Handles real-time news, breaking events, stock prices, and queries with temporal freshness constraints."
            },
            "VECTOR_DATABASE": {
                "name": "Enterprise Vector DB (Milvus / Qdrant HNSW)",
                "type": "Dense Semantic Document Engine",
                "avg_latency_ms": 35.0,
                "cost_tier": "Low ($0.0004 / query)",
                "description": "Handles unstructured policies, engineering manuals, legal contracts, and product documentation."
            }
        }

    def classify_query(self, query: str) -> Dict[str, Any]:
        """Analyzes query intent, temporal markers, and numerical calculation demands."""
        q_lower = query.lower()
        
        # Check for numerical / SQL aggregation cues
        sql_keywords = ["how many", "total", "sum", "average", "revenue", "count", "metrics", "percentage", "highest"]
        # Check for web freshness / real-time cues
        web_keywords = ["latest", "today", "yesterday", "stock price", "breaking", "news", "current weather", "2026", "ceo of"]
        
        has_sql = any(k in q_lower for k in sql_keywords)
        has_web = any(k in q_lower for k in web_keywords)
        
        if has_web and not has_sql:
            primary_route = "WEB_SEARCH"
            confidence = 0.94
            rationale = "Temporal freshness keywords detected requiring real-time external indexing."
        elif has_sql:
            primary_route = "SQL_DATABASE"
            confidence = 0.91
            rationale = "Structured numerical aggregation query detected. Text-to-SQL required for mathematical precision."
        else:
            primary_route = "VECTOR_DATABASE"
            confidence = 0.88
            rationale = "Unstructured semantic knowledge query detected. Routing to enterprise document vector index."
            
        return {
            "query": query,
            "primary_route": primary_route,
            "confidence": confidence,
            "rationale": rationale,
            "route_metadata": self.routes[primary_route]
        }

    def execute_route_simulation(self, query: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates dispatch and returns execution telemetry."""
        route = classification["primary_route"]
        
        if route == "SQL_DATABASE":
            simulated_action = "Generated Parameterized SQL: SELECT region, SUM(arr) FROM enterprise_billing GROUP BY region"
            result_snippet = "SQL Execution Succeeded: North America ($45.2M), EMEA ($31.8M), APAC ($18.4M)."
        elif route == "WEB_SEARCH":
            simulated_action = "Invoked Tavily Real-Time Search API with temporal filter 'past_24h'"
            result_snippet = "Web Search Returned 3 Fresh Articles: 'Tech Conference 2026 Keynote Highlights Next-Gen AI Silicon'."
        else:
            simulated_action = "Probed Qdrant HNSW Collection 'enterprise_policies' (ef_search=64, top_k=4)"
            result_snippet = "Retrieved 4 Policy Chunks: 'SOC-2 Compliance Handbook Section 9.2 (Access Reviews)'."
            
        return {
            "dispatched_route": route,
            "simulated_action": simulated_action,
            "result_snippet": result_snippet,
            "latency_ms": classification["route_metadata"]["avg_latency_ms"],
            "cost_tier": classification["route_metadata"]["cost_tier"]
        }


# --- Streamlit Layout ---

st.title("20.3 Source Routing in RAG")
st.caption("Dynamic Query Classification, Intent Triage, and Dispatch across SQL Databases, Real-Time Web & Vector Stores")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Dynamic Multi-Source Triage & Gateway Routing Architecture", divider="rainbow")

st.markdown(
    """
    Submitting every user query to every data source simultaneously is commercially unviable and technologically inefficient.
    Querying live web APIs for internal policy questions adds unnecessary latency and cost, while searching vector databases for
    aggregate financial sums leads to mathematical hallucinations.
    
    A production **Gateway Source Router** inspects query syntax, intent semantics, and temporal markers within milliseconds.
    It dispatches each request exclusively to the specialized storage engine engineered to answer it, falling back to secondary sources
    only when primary retrieval confidence collapses.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Gateway_Source_Router {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_ingress {
            label = "1. INGRESS & INTENT CLASSIFICATION";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_query [label="Incoming User Query", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            intent_classifier [label="Zero-Shot Query Classifier\\n- Semantic Centroid Matching\\n- Temporal Marker Detection\\n- Numerical / Aggregation Parser", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            
            user_query -> intent_classifier;
        }

        subgraph cluster_decision_routing {
            label = "2. SOURCE DISPATCH LOGIC & BUDGET TRIAGE";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            triage_switch [label="Dynamic Dispatch Switchboard\\n(Cost, Latency & Domain Constraint Evaluation)", shape=diamond, fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            intent_classifier -> triage_switch;
        }

        subgraph cluster_specialized_engines {
            label = "3. SPECIALIZED DATA DESTINATIONS";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            route_sql [label="Route A: Relational Database (SQL)\\nText-to-SQL Engine\\n(Aggregations, Counts, Exact Sums)", fillcolor="#0F172A", fontcolor="#10B981"];
            route_web [label="Route B: Live Web Search API\\nReal-Time External Corpus\\n(Breaking News, Stock Tickers, Freshness)", fillcolor="#0F172A", fontcolor="#F59E0B"];
            route_vector [label="Route C: Enterprise Vector Database\\nHNSW Dense Vector Search\\n(Policies, Engineering Docs, Legal PDFs)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            triage_switch -> route_sql [label="Needs Aggregation"];
            triage_switch -> route_web [label="Needs Real-Time"];
            triage_switch -> route_vector [label="Needs Semantic Docs"];
        }

        subgraph cluster_fallback {
            label = "4. FALLBACK & AUDIT TIER";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            confidence_check [label="Confidence Evaluation Gate\\n(Did Primary Source Return Signal?)", shape=diamond, fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            fallback_cascade [label="Fallback Cascade Policy\\n(Zero-Hits -> Escalate to Web / Broader Vector)", fillcolor="#7F1D1D", fontcolor="#FCA5A5"];
            synthesis_gateway [label="Prompt Synthesis Gateway\\n(Pass Unified Clean Context to LLM)", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            route_sql -> confidence_check;
            route_web -> confidence_check;
            route_vector -> confidence_check;
            
            confidence_check -> synthesis_gateway [label="High Confidence"];
            confidence_check -> fallback_cascade [label="Low / Zero-Hits"];
            fallback_cascade -> route_web [label="Retry via Web", color="#F43F5E", style=dashed];
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: QUERY CLASSIFICATION ---
st.header("2. Query Classification (Zero-Shot Intent Triage, Named Entity Extraction & Complexity Scoring)", divider="rainbow")

st.markdown(
    """
    The router begins with **Query Classification**, executing in under 20 milliseconds:
    
    1. **Structural / Keyword Fast-Path**: Checks for numerical operators, mathematical queries (*"sum"*, *"average"*, *"total"*), or known entity IDs.
    2. **Temporal Marker Detection**: Identifies freshness constraints (*"today"*, *"latest"*, *"Q1 2026"*). If temporal markers are present, static document indices are downgraded in favor of live web or streaming sources.
    3. **Embedding Centroid Classification**: Projects the query into vector space and computes cosine proximity against pre-computed domain cluster centroids (`SQL_CENTROID`, `LEGAL_CENTROID`, `ENGINEERING_CENTROID`).
    """
)

# --- SECTION 3: SELECTING THE CORRECT SOURCE ---
st.header("3. Selecting the Correct Source (Multi-Class Dispatch, Fallback Priority Cascades & Cost/Latency Budgets)", divider="rainbow")

st.markdown(
    """
    Once intent is classified, the router balances three trade-offs to select the optimal destination:
    
    - **Precision Suitability**: Relational SQL engines are mathematically exact (100% precision for numbers), whereas vector search is approximate.
    - **Latency Constraints**: SQL and Vector lookups complete in $15\text{ms} - 40\text{ms}$, while live Web Search requires $300\text{ms} - 600\text{ms}$.
    - **Financial Budgets**: Internal vector lookups cost fractions of a cent, while external search APIs charge per-query API fees.
    """
)

# --- SECTION 4: DATABASE ROUTING ---
st.header("4. Database Routing (Text-to-SQL Triage, Analytical Schemas & Aggregation Query Paths)", divider="rainbow")

st.markdown(
    """
    Queries routed to relational databases execute along a specialized **Text-to-SQL Path**:
    
    1. **Catalog & Schema Injection**: The router passes table definitions (`CREATE TABLE orders (id INT, amount DECIMAL, region VARCHAR)`) to a code-optimized LLM.
    2. **Parameterized Query Generation**: The model emits a strictly read-only, sanitized SQL query.
    3. **Database Execution & Tabular Packaging**: The query executes against PostgreSQL or Snowflake, and the resulting table rows are formatted into a markdown table injected as verified ground-truth context.
    """
)

# --- SECTION 5: WEB SEARCH ROUTING ---
st.header("5. Web Search Routing (Temporal Freshness Triggers, Live Entity Lookups & Serper/Tavily Connectors)", divider="rainbow")

st.markdown(
    """
    Internal enterprise documents cannot answer questions about external competitors, live stock movements, or recent industry announcements.
    
    When the query router detects:
    - Real-time events occurring after the vector index build timestamp.
    - Explicit external entity research (*"What did NVIDIA announce yesterday at GTC?"*).
    
    The request is dispatched to specialized search APIs (Tavily, Serper, Bing Search).
    The router prunes advertising boilerplate and injects the top 3 extracted web snippets directly into the prompt.
    """
)

# --- SECTION 6: VECTOR DATABASE ROUTING ---
st.header("6. Vector Database Routing (Unstructured Semantic Search, Domain Shard Targeting & Partition Pruning)", divider="rainbow")

st.markdown(
    """
    When the user seeks explanatory, conceptual, or procedural knowledge (*"How do I configure OAuth2 in our microservices?"*),
    the router directs the query to the **Enterprise Vector Database**:
    
    - **Partition Pruning**: The router uses metadata tags to query only relevant collection partitions (e.g., querying `collection="engineering"` rather than searching millions of unrelated HR documents).
    - **Dense Semantic Retrieval**: Employs HNSW approximate nearest neighbor search to retrieve the top `K` document chunks.
    - **High Recall**: Bridges vocabulary differences and synonyms that exact SQL queries would miss.
    """
)

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Enterprise Gateway Source Router", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `GatewaySourceRouterEngine`.
    It analyzes query syntax, classifies intent, selects the optimal target system (SQL Database, Live Web, or Vector Store),
    and executes simulated data retrieval with latency tracking.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class GatewaySourceRouterEngine:
    """Enterprise Gateway Source Router: Classifies intent and dynamically routes queries."""

    def __init__(self):
        self.routes = {
            "SQL_DATABASE": {
                "name": "Relational Data Warehouse (Snowflake / PostgreSQL)",
                "type": "Structured Analytical Engine",
                "avg_latency_ms": 25.0,
                "cost_tier": "Low ($0.0001 / query)",
                "description": "Handles exact numerical aggregations, sums, counts, filters, and historical metrics."
            },
            "WEB_SEARCH": {
                "name": "Live Web Search (Tavily / Google Serper)",
                "type": "Real-Time External Engine",
                "avg_latency_ms": 320.0,
                "cost_tier": "Moderate ($0.0050 / query)",
                "description": "Handles real-time news, breaking events, stock prices, and queries with temporal freshness constraints."
            },
            "VECTOR_DATABASE": {
                "name": "Enterprise Vector DB (Milvus / Qdrant HNSW)",
                "type": "Dense Semantic Document Engine",
                "avg_latency_ms": 35.0,
                "cost_tier": "Low ($0.0004 / query)",
                "description": "Handles unstructured policies, engineering manuals, legal contracts, and product documentation."
            }
        }

    def classify_query(self, query: str) -> Dict[str, Any]:
        """Analyzes query intent, temporal markers, and numerical calculation demands."""
        q_lower = query.lower()
        sql_keywords = ["how many", "total", "sum", "average", "revenue", "count", "metrics", "percentage", "highest"]
        web_keywords = ["latest", "today", "yesterday", "stock price", "breaking", "news", "current weather", "2026", "ceo of"]
        
        has_sql = any(k in q_lower for k in sql_keywords)
        has_web = any(k in q_lower for k in web_keywords)
        
        if has_web and not has_sql:
            primary_route = "WEB_SEARCH"
            confidence = 0.94
            rationale = "Temporal freshness keywords detected requiring real-time external indexing."
        elif has_sql:
            primary_route = "SQL_DATABASE"
            confidence = 0.91
            rationale = "Structured numerical aggregation query detected. Text-to-SQL required for mathematical precision."
        else:
            primary_route = "VECTOR_DATABASE"
            confidence = 0.88
            rationale = "Unstructured semantic knowledge query detected. Routing to enterprise document vector index."
            
        return {
            "query": query,
            "primary_route": primary_route,
            "confidence": confidence,
            "rationale": rationale,
            "route_metadata": self.routes[primary_route]
        }

    def execute_route_simulation(self, query: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates dispatch and returns execution telemetry."""
        route = classification["primary_route"]
        if route == "SQL_DATABASE":
            simulated_action = "Generated Parameterized SQL: SELECT region, SUM(arr) FROM enterprise_billing GROUP BY region"
            result_snippet = "SQL Execution Succeeded: North America ($45.2M), EMEA ($31.8M), APAC ($18.4M)."
        elif route == "WEB_SEARCH":
            simulated_action = "Invoked Tavily Real-Time Search API with temporal filter 'past_24h'"
            result_snippet = "Web Search Returned 3 Fresh Articles: 'Tech Conference 2026 Keynote Highlights Next-Gen AI Silicon'."
        else:
            simulated_action = "Probed Qdrant HNSW Collection 'enterprise_policies' (ef_search=64, top_k=4)"
            result_snippet = "Retrieved 4 Policy Chunks: 'SOC-2 Compliance Handbook Section 9.2 (Access Reviews)'."
            
        return {
            "dispatched_route": route,
            "simulated_action": simulated_action,
            "result_snippet": result_snippet,
            "latency_ms": classification["route_metadata"]["avg_latency_ms"],
            "cost_tier": classification["route_metadata"]["cost_tier"]
        }
''',
    language="python"
)

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Source Routing Switchboard & Triage Dispatch Console", divider="rainbow")

st.markdown(
    """
    Test dynamic intent triage and source dispatch across heterogeneous storage engines.
    Select presets representing numerical SQL lookups, live web searches, or policy vector queries to inspect automated routing decisions.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🔀 Query Triage Presets")
    
    preset_choice = st.selectbox(
        "Select Query Preset:",
        [
            "Preset 1 (SQL): What was the total cloud revenue by sales region for Q4?",
            "Preset 2 (Web): What are the latest news announcements regarding AI hardware in 2026?",
            "Preset 3 (Vector): What is the enterprise password rotation and SOC-2 access policy?",
            "Preset 4 (Custom Query): Enter your own query below"
        ],
        index=0
    )
    
    if "Custom" in preset_choice:
        active_query = st.text_input("Active Query:", "Calculate the average customer tenure in the enterprise database")
    else:
        active_query = preset_choice.split(": ", 1)[1]
        st.text_input("Active Query:", active_query, disabled=True)

    engine = GatewaySourceRouterEngine()
    classification = engine.classify_query(active_query)
    
    st.markdown("#### Automated Triage Classification")
    st.info(
        f"**Target Destination**: `{classification['primary_route']}`  \n"
        f"**Confidence**: `{classification['confidence'] * 100:.1f}%`  \n"
        f"**Engine**: {classification['route_metadata']['name']}  \n"
        f"**Rationale**: {classification['rationale']}"
    )

with wb_col2:
    st.subheader("⚡ Dispatched Execution Telemetry")
    
    telemetry = engine.execute_route_simulation(active_query, classification)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Route", telemetry["dispatched_route"])
    m2.metric("Engine Latency", f"{telemetry['latency_ms']} ms")
    m3.metric("Cost Tier", telemetry["cost_tier"].split(" (")[0])
    
    st.markdown("#### Simulated Engine Execution Action")
    st.code(telemetry["simulated_action"], language="text")
    
    st.markdown("#### Returned Execution Result")
    st.success(telemetry["result_snippet"])

st.markdown("### Source Capabilities Comparison Matrix")
comp_df = pd.DataFrame([
    {
        "Engine Route": k,
        "System Name": v["name"],
        "Architecture Paradigm": v["type"],
        "Expected Latency": f"{v['avg_latency_ms']} ms",
        "Cost Profile": v["cost_tier"],
        "Core Strength": v["description"]
    }
    for k, v in engine.routes.items()
])
st.dataframe(comp_df, use_container_width=True)

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: Why is intent routing superior to sending every query to all available data sources?**"):
    st.markdown(
        """
        Broadcasting every user question to all sources (SQL DBs, live Web APIs, and Vector DBs) creates three critical production problems:
        1. **Unnecessary Financial Cost**: Web search APIs incur per-request billing, and heavy vector embeddings consume GPU resources.
        2. **Latency Inflation**: A simple policy lookup that could resolve in 20ms in a local vector DB is forced to wait 500ms for external web APIs to respond.
        3. **Context Pollution**: Querying all sources returns contradictory or irrelevant snippets that confuse the synthesizer LLM.
        """
    )

with st.expander("**Q2: How do you design an ultra-fast Query Classifier that does not add hundreds of milliseconds of latency?**"):
    st.markdown(
        """
        Prompting a frontier LLM to classify query intent adds 400ms - 800ms of latency before retrieval even starts.
        Production architectures implement a **Hierarchical Classifier**:
        - **Tier 1 (Regex & Keyword Rules)**: Checks for explicit SQL terms (*'sum'*, *'average'*) or temporal markers (*'today'*, *'latest'*) in $< 1\text{ ms}$.
        - **Tier 2 (Embedding Centroid Projections)**: Projects the query into a fast embedding space (e.g., BGE-Small) and computes cosine distance against intent cluster centroids in $< 10\text{ ms}$.
        - **Tier 3 (Distilled SLM)**: For ambiguous queries, a local 1B parameter quantized model makes the routing decision in $< 40\text{ ms}$.
        """
    )

with st.expander("**Q3: What triggers an automatic fallback from an internal Vector DB to external Web Search?**"):
    st.markdown(
        """
        An enterprise fallback cascade trips when:
        1. **Zero-Hit Retrival**: The vector database returns 0 chunks within the maximum distance radius.
        2. **Similarity Floor Collapse**: The Top-1 cosine similarity score is below the relevance threshold (e.g., `score < 0.65`), indicating an out-of-domain query.
        3. **Temporal Freshness Mismatch**: The query references events occurring after the latest document timestamp in the vector index.
        """
    )

with st.expander("**Q4: What security safeguards are necessary when routing natural language queries to SQL databases?**"):
    st.markdown(
        """
        Executing raw LLM-generated SQL directly against production databases is extremely dangerous (SQL injection, accidental drops).
        
        Production Text-to-SQL safeguards require:
        1. **Read-Only Database Roles**: The connection user must strictly possess `SELECT` permissions with zero `INSERT`, `UPDATE`, `DELETE`, or `DROP` grants.
        2. **AST Query Validation**: Parsing the generated SQL abstract syntax tree (AST) to ensure no DDL statements or malicious table joins exist.
        3. **Strict Query Timeouts & Row Limits**: Enforcing `LIMIT 100` and a hard 3-second database statement timeout to prevent denial-of-service cartesian joins.
        """
    )

with st.expander("**Q5: How does Partition Pruning improve retrieval efficiency in enterprise vector databases?**"):
    st.markdown(
        """
        In massive vector stores containing tens of millions of documents, searching the global index requires extensive HNSW graph traversal.
        
        **Partition Pruning** leverages the router's classification to restrict the search exclusively to a specific metadata partition (e.g., `tenant_id = 'org_42'` and `domain = 'legal'`).
        This reduces the candidate search space by 95%+, cutting search latency from 60ms to 8ms while preventing cross-department context pollution.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Synchronous Frontier LLM Router Bottleneck**  
        **Antipattern**: Using GPT-4o to classify every user query before deciding where to retrieve.  
        **Consequence**: Adds 800ms of latency and $0.02 of cost to every single request, eliminating any latency advantage.  
        **Production Fix**: Employ fast embedding centroid classifiers or local distilled SLMs with latency $< 20\text{ ms}$.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Routing Numerical Aggregations to Vector Embeddings**  
        **Antipattern**: Attempting to answer queries like *"What was total Q3 revenue across all 50 stores?"* using vector similarity search over financial PDFs.  
        **Consequence**: Vector search retrieves random paragraphs; the LLM hallucinates an inaccurate total sum.  
        **Production Fix**: Route aggregation queries to structured SQL databases equipped with exact arithmetic calculation engines.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Unbounded Web Search Fan-Out**  
        **Antipattern**: Falling back to live web search without query rewriting or search depth limits.  
        **Consequence**: Ingests noisy web search advertisements, blog commentary, and irrelevant links that dilute grounded context.  
        **Production Fix**: Restrict web retrieval to the top 3 verified domains and enforce strict domain whitelists.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Missing Fallback Telemetry on Router Misclassifications**  
        **Antipattern**: Routing silently fails with zero hits without logging the misclassification event.  
        **Consequence**: Engineering teams have zero visibility into queries that were directed to the wrong storage engine.  
        **Production Fix**: Emit OpenTelemetry classification audit events to continuously fine-tune router centroids.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Source Routing in RAG?")
        st.markdown("**Back**: An architectural mechanism that evaluates query intent to dynamically dispatch requests to the optimal data engine (SQL DB, Live Web, or Vector Store) rather than querying all sources.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Why route to SQL instead of Vector DBs?")
        st.markdown("**Back**: SQL engines deliver 100% mathematical precision for numerical sums, averages, and exact counts, whereas vector embeddings are purely semantic and prone to math hallucinations.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is an Embedding Centroid Classifier?")
        st.markdown("**Back**: A low-latency router that measures cosine similarity between the query embedding and pre-calculated intent cluster centroids in under 10 milliseconds.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is Partition Pruning?")
        st.markdown("**Back**: Restricting vector search to a specific collection shard or metadata filter based on router classification, drastically speeding up search and eliminating cross-domain noise.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of source routing architectures, intent classification, and multi-source dispatch:")

chk1 = st.checkbox("I understand the architectural advantages of intelligent source routing over blind multi-source broadcasting.", value=True)
chk2 = st.checkbox("I can implement low-latency query classification using regex fast-paths and embedding centroid projections.", value=True)
chk3 = st.checkbox("I know when to route queries to structured Text-to-SQL databases versus unstructured semantic vector stores.", value=True)
chk4 = st.checkbox("I understand how temporal markers trigger live web search routing for real-time information freshness.", value=True)
chk5 = st.checkbox("I can architect fallback cascade policies that gracefully escalate when primary source confidence is low.", value=True)
chk6 = st.checkbox("I know how to implement read-only access controls and AST sanitization on Text-to-SQL database routes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Enterprise Source Routing and Query Triage.")
