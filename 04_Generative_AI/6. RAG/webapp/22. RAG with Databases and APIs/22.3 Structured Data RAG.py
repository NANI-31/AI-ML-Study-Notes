import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="22.3 Structured Data RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class StructuredDataRAGEngine:
    """Enterprise Structured Data & Self-Querying Retrieval Engine across CSV, JSON, and Databases."""

    def __init__(self):
        # Heterogeneous structured catalog
        self.catalog = pd.DataFrame([
            {"id": "rec_01", "name": "NVIDIA H100 SXM5", "format": "CSV", "category": "Hardware", "price_usd": 32000.0, "rating": 4.9, "release_year": 2023, "description": "80GB HBM3 memory with 3.35 TB/s bandwidth, transformer engine optimized for FP8 training."},
            {"id": "rec_02", "name": "Apple M3 Max", "format": "JSON", "category": "Hardware", "price_usd": 3500.0, "rating": 4.8, "release_year": 2024, "description": "16-core CPU, 40-core GPU with 128GB unified memory for on-device local model inference."},
            {"id": "rec_03", "name": "Enterprise PostgreSQL 16", "format": "Database", "category": "Software", "price_usd": 0.0, "rating": 4.9, "release_year": 2023, "description": "Open source relational database with native pgvector extension support for high-throughput similarity search."},
            {"id": "rec_04", "name": "Pinecone Serverless", "format": "JSON", "category": "Software", "price_usd": 1200.0, "rating": 4.6, "release_year": 2024, "description": "Managed cloud vector database with metadata filtering and decoupled compute and storage pricing."},
            {"id": "rec_05", "name": "Google TPU v5p", "format": "CSV", "category": "Hardware", "price_usd": 28000.0, "rating": 4.7, "release_year": 2024, "description": "Cloud TPU accelerator with 95GB HBM2e memory designed for large-scale multi-host Gemini training."}
        ])

    def compile_self_query(self, natural_query: str) -> Dict[str, Any]:
        """Translates natural language into a semantic query and structured metadata filter dictionary."""
        q_lower = natural_query.lower()
        
        # Self-querying extraction heuristics
        category_filter = None
        max_price_filter = None
        min_year_filter = None
        
        if "hardware" in q_lower:
            category_filter = "Hardware"
        elif "software" in q_lower:
            category_filter = "Software"
            
        if "under" in q_lower and ("5000" in q_lower or "5k" in q_lower):
            max_price_filter = 5000.0
        elif "under" in q_lower and "30000" in q_lower:
            max_price_filter = 30000.0
            
        if "2024" in q_lower or "recent" in q_lower:
            min_year_filter = 2024

        semantic_part = natural_query
        for word in ["find", "hardware", "software", "under", "5000", "5k", "30000", "in", "2024", "recent"]:
            semantic_part = semantic_part.replace(word, "").replace(word.capitalize(), "")
        semantic_query = " ".join(semantic_part.split())
        if not semantic_query:
            semantic_query = natural_query

        return {
            "original_query": natural_query,
            "semantic_query": semantic_query,
            "filters": {
                "category": category_filter,
                "max_price": max_price_filter,
                "min_year": min_year_filter
            }
        }

    def execute_self_query_retrieval(self, query_plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Applies hard metadata filters, then scores semantic relevance on filtered subset."""
        df = self.catalog.copy()
        filters = query_plan["filters"]
        
        # 1. Hard Boolean Metadata Filtering
        if filters["category"]:
            df = df[df["category"] == filters["category"]]
        if filters["max_price"] is not None:
            df = df[df["price_usd"] <= filters["max_price"]]
        if filters["min_year"] is not None:
            df = df[df["release_year"] >= filters["min_year"]]

        if df.empty:
            return df, {"matched_count": 0, "filter_pruned_pct": 100.0}

        # 2. Semantic Similarity Scoring on Filtered Candidates
        q_tokens = set(query_plan["semantic_query"].lower().split())
        scores = []
        for _, row in df.iterrows():
            text_tokens = set((row["name"] + " " + row["description"]).lower().split())
            overlap = len(q_tokens.intersection(text_tokens))
            score = float(overlap / (len(q_tokens) + 1e-5)) + 0.35
            scores.append(round(min(1.0, score), 4))
            
        df["semantic_score"] = scores
        df = df.sort_values(by="semantic_score", ascending=False)
        pruned_pct = round(((len(self.catalog) - len(df)) / len(self.catalog)) * 100.0, 1)

        return df, {"matched_count": len(df), "filter_pruned_pct": pruned_pct}


# --- Streamlit Layout ---

st.title("22.3 Structured Data RAG & Self-Querying")
st.caption("Hybrid Ingestion across CSVs, JSON & Relational Catalogs with Automated Self-Querying Metadata Filtering")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Structured Ingestion & Self-Querying Filter Architecture", divider="rainbow")

st.markdown(
    """
    Standard vector databases struggle with **structured numerical and categorical constraints**.
    A user query like *"Find hardware accelerators under $5,000 released in 2024"* requires two distinct mechanisms:
    1. **Hard Boolean Metadata Filtering**: Eliminating records that do not strictly satisfy `price <= 5000` and `release_year >= 2024`.
    2. **Fuzzy Semantic Vector Search**: Ranking the filtered records based on description relevance (*"accelerator"*).
    
    The diagram below models the complete **Self-Querying RAG Architecture**, showing how natural language is decomposed
    into a semantic search vector and a structured Boolean filter dictionary before probing multi-format data stores.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Structured_Data_Architecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_query_decomp {
            label = "1. QUERY DECOMPOSITION & SELF-QUERY COMPILER";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_input [label="User Query: 'Find hardware under $5,000 in 2024'", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            self_query_parser [label="Self-Querying Language Model\\nDecomposes into Semantic Query & Metadata Filters", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            semantic_query [label="Semantic Vector String: 'accelerator'", fillcolor="#0F172A", fontcolor="#38BDF8"];
            metadata_filters [label="Structured Filter Dict:\\n{category: 'Hardware', price <= 5000, year >= 2024}", fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            user_input -> self_query_parser;
            self_query_parser -> semantic_query;
            self_query_parser -> metadata_filters;
        }

        subgraph cluster_ingestion_formats {
            label = "2. MULTI-FORMAT STRUCTURED CATALOG";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            fmt_csv [label="1. CSV Catalogs (Pandas / Polars)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            fmt_json [label="2. JSON Payloads (Nested Dictionaries)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            fmt_sql [label="3. Relational Database Tables", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        subgraph cluster_two_stage_filtering {
            label = "3. HARD METADATA PRUNING & SEMANTIC SCORING";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            pre_filter_gate [label="Stage 1: Hard Boolean Metadata Filter\\nPrunes 80% of candidates instantly", fillcolor="#78350F", fontcolor="#FDE68A"];
            ann_scorer [label="Stage 2: Dense Semantic Scorer\\nScores similarity only on pre-filtered records", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            
            metadata_filters -> pre_filter_gate;
            fmt_csv -> pre_filter_gate;
            fmt_json -> pre_filter_gate;
            fmt_sql -> pre_filter_gate;
            
            pre_filter_gate -> ann_scorer [label="Valid Candidates Only"];
            semantic_query -> ann_scorer;
        }

        subgraph cluster_output {
            label = "4. GROUNDED OUTPUT GENERATION";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            final_response [label="Synthesized Grounded Response\\nWith Verified Structured Attributes", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            ann_scorer -> final_response;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: CSV RETRIEVAL ---
st.header("2. CSV Retrieval (Columnar Parsing, Header Normalization & In-Memory DataFrames)", divider="rainbow")

st.markdown(
    """
    Comma-Separated Values (CSV) are widely used for product catalogs, pricing sheets, and metrics tables:
    
    - **Header Normalization**: Column headers often contain spaces, casing differences, or abbreviations (*'price_usd'*, *'cost'*, *'MSRP'*). Normalization maps synonyms to standardized schema keys.
    - **High-Performance In-Memory DataFrames**: Libraries like Pandas, Polars, or DuckDB execute filtering, sorting, and aggregations on millions of rows in sub-millisecond times.
    - **Row-to-Document Serialization**: Converting CSV rows into key-value formatted text snippets before vector embedding.
    """
)

# --- SECTION 3: JSON RETRIEVAL ---
st.header("3. JSON Retrieval (Hierarchical JSON Paths, Schema Normalization & Object Flattening)", divider="rainbow")

st.markdown(
    """
    JSON is the native data format of modern APIs and document databases (MongoDB, DynamoDB):
    
    - **JSONPath Expressions**: Extracting specific nested fields (`$.customer.orders[*].amount`) rather than loading unneeded hierarchies.
    - **Schema Normalization**: Harmonizing heterogeneous JSON structures from different systems into a consistent schema dictionary.
    - **Recursive Object Flattening**: Converting nested objects into dotted single-level key-value pairs for seamless metadata filtering.
    """
)

# --- SECTION 4: DATABASE RETRIEVAL ---
st.header("4. Database Retrieval (Hybrid Relational-Vector Joins & Catalog Alignment)", divider="rainbow")

st.markdown(
    """
    Modern vector databases (Qdrant, pgvector, Milvus) support **Hybrid Relational-Vector Retrieval**:
    
    - Storing relational metadata payloads alongside dense vector embeddings.
    - Executing SQL joins between unstructured customer feedback embeddings and relational customer order tables using common keys (`customer_id`).
    - Eliminating the need to query two separate database clusters for hybrid text-and-tabular inquiries.
    """
)

# --- SECTION 5: METADATA FILTERING ---
st.header("5. Metadata Filtering (Hard Boolean Constraints, Range Queries & Partition Pruning)", divider="rainbow")

st.markdown(
    """
    Metadata filtering enforces deterministic business rules that vector similarity cannot guarantee:
    
    - **Exact Equality**: `tenant_id == 'corp_42'`.
    - **Numerical Ranges**: `price >= 1000 AND price <= 5000`.
    - **Date Filters**: `created_at >= '2025-01-01'`.
    - **Partition Pruning**: Directing the search to specific vector collection partitions, slashing search latency by 90%+.
    """
)

# --- SECTION 6: STRUCTURED QUERY GENERATION ---
st.header("6. Structured Query Generation (Self-Querying Extractors, Logical Filter Compilation & Pydantic Translation)", divider="rainbow")

st.markdown(
    """
    **Self-Querying** (LangChain / LlamaIndex) prompts a language model to inspect user queries and emit structured Pydantic objects:
    
    ```json
    {
      "query": "accelerator",
      "filter": {
        "and": [
          {"category": {"eq": "Hardware"}},
          {"price_usd": {"lte": 5000}},
          {"release_year": {"gte": 2024}}
        ]
      }
    }
    ```
    
    The compiler translates this logical tree directly into the vector database's native filter syntax (e.g., Qdrant Filter, Pinecone Filter, or SQL WHERE clause).
    """
)

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Enterprise Structured Data Self-Querying Engine", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `StructuredDataRAGEngine`.
    It implements self-querying query decomposition, multi-format catalog indexing across CSV, JSON, and databases,
    hard Boolean metadata filtering, and semantic similarity scoring on pre-filtered records.
    """
)

st.code(
    '''
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

class StructuredDataRAGEngine:
    """Enterprise Structured Data & Self-Querying Retrieval Engine."""

    def __init__(self):
        self.catalog = pd.DataFrame([
            {"id": "rec_01", "name": "NVIDIA H100 SXM5", "format": "CSV", "category": "Hardware", "price_usd": 32000.0, "rating": 4.9, "release_year": 2023, "description": "80GB HBM3 memory with 3.35 TB/s bandwidth, transformer engine optimized for FP8 training."},
            {"id": "rec_02", "name": "Apple M3 Max", "format": "JSON", "category": "Hardware", "price_usd": 3500.0, "rating": 4.8, "release_year": 2024, "description": "16-core CPU, 40-core GPU with 128GB unified memory for on-device local model inference."},
            {"id": "rec_03", "name": "Enterprise PostgreSQL 16", "format": "Database", "category": "Software", "price_usd": 0.0, "rating": 4.9, "release_year": 2023, "description": "Open source relational database with native pgvector extension support for high-throughput similarity search."},
            {"id": "rec_04", "name": "Pinecone Serverless", "format": "JSON", "category": "Software", "price_usd": 1200.0, "rating": 4.6, "release_year": 2024, "description": "Managed cloud vector database with metadata filtering and decoupled compute and storage pricing."},
            {"id": "rec_05", "name": "Google TPU v5p", "format": "CSV", "category": "Hardware", "price_usd": 28000.0, "rating": 4.7, "release_year": 2024, "description": "Cloud TPU accelerator with 95GB HBM2e memory designed for large-scale multi-host Gemini training."}
        ])

    def compile_self_query(self, natural_query: str) -> Dict[str, Any]:
        """Translates natural language into a semantic query and structured metadata filter dictionary."""
        q_lower = natural_query.lower()
        category_filter = None
        max_price_filter = None
        min_year_filter = None
        
        if "hardware" in q_lower:
            category_filter = "Hardware"
        elif "software" in q_lower:
            category_filter = "Software"
            
        if "under" in q_lower and ("5000" in q_lower or "5k" in q_lower):
            max_price_filter = 5000.0
        elif "under" in q_lower and "30000" in q_lower:
            max_price_filter = 30000.0
            
        if "2024" in q_lower or "recent" in q_lower:
            min_year_filter = 2024

        semantic_part = natural_query
        for word in ["find", "hardware", "software", "under", "5000", "5k", "30000", "in", "2024", "recent"]:
            semantic_part = semantic_part.replace(word, "").replace(word.capitalize(), "")
        semantic_query = " ".join(semantic_part.split())
        if not semantic_query:
            semantic_query = natural_query

        return {
            "original_query": natural_query,
            "semantic_query": semantic_query,
            "filters": {
                "category": category_filter,
                "max_price": max_price_filter,
                "min_year": min_year_filter
            }
        }

    def execute_self_query_retrieval(self, query_plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Applies hard metadata filters, then scores semantic relevance on filtered subset."""
        df = self.catalog.copy()
        filters = query_plan["filters"]
        
        if filters["category"]:
            df = df[df["category"] == filters["category"]]
        if filters["max_price"] is not None:
            df = df[df["price_usd"] <= filters["max_price"]]
        if filters["min_year"] is not None:
            df = df[df["release_year"] >= filters["min_year"]]

        if df.empty:
            return df, {"matched_count": 0, "filter_pruned_pct": 100.0}

        q_tokens = set(query_plan["semantic_query"].lower().split())
        scores = []
        for _, row in df.iterrows():
            text_tokens = set((row["name"] + " " + row["description"]).lower().split())
            overlap = len(q_tokens.intersection(text_tokens))
            score = float(overlap / (len(q_tokens) + 1e-5)) + 0.35
            scores.append(round(min(1.0, score), 4))
            
        df["semantic_score"] = scores
        df = df.sort_values(by="semantic_score", ascending=False)
        pruned_pct = round(((len(self.catalog) - len(df)) / len(self.catalog)) * 100.0, 1)

        return df, {"matched_count": len(df), "filter_pruned_pct": pruned_pct}
''',
    language="python"
)

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Self-Query Filter Compiler & Multi-Format Ingestion Sandbox", divider="rainbow")

st.markdown(
    """
    Test self-querying query decomposition and metadata filtering across CSV, JSON, and database records.
    Select presets with varying price, category, and date constraints to observe automated filter compilation and candidate pruning.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🔍 Natural Query Input")
    
    preset_choice = st.selectbox(
        "Select Query Preset:",
        [
            "Preset 1 (Hard Filters): Find hardware under 5000 in 2024 for on-device inference",
            "Preset 2 (Category Filter): Find software databases for pgvector similarity search",
            "Preset 3 (Price Constraint): Find hardware under 30000 with HBM memory",
            "Preset 4 (Custom Query): Enter your own query below"
        ],
        index=0
    )
    
    if "Custom" in preset_choice:
        active_query = st.text_input("Active Query:", "Find hardware accelerators in 2024")
    else:
        active_query = preset_choice.split(": ", 1)[1]
        st.text_input("Active Query:", active_query, disabled=True)

    engine = StructuredDataRAGEngine()
    query_plan = engine.compile_self_query(active_query)
    
    st.markdown("#### Compiled Self-Query Plan")
    st.info(
        f"**Semantic Search String**: `{query_plan['semantic_query']}`  \n"
        f"**Category Filter**: `{query_plan['filters']['category']}`  \n"
        f"**Max Price Filter**: `${query_plan['filters']['max_price']}`  \n"
        f"**Min Year Filter**: `{query_plan['filters']['min_year']}`"
    )

with wb_col2:
    st.subheader("⚡ Execution & Pruning Telemetry")
    
    results_df, meta = engine.execute_self_query_retrieval(query_plan)
    
    m1, m2 = st.columns(2)
    m1.metric("Filtered Matches", meta["matched_count"])
    m2.metric("Catalog Pruned", f"{meta['filter_pruned_pct']}%")
    
    st.markdown("#### Filtered & Scored Candidate Records")
    if not results_df.empty:
        st.dataframe(results_df[["name", "format", "category", "price_usd", "release_year", "semantic_score"]], use_container_width=True)
    else:
        st.warning("No records satisfied all hard Boolean metadata filter constraints.")

st.markdown("### Multi-Format Catalog Schema")
st.dataframe(engine.catalog, use_container_width=True)

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: Why does post-filtering fail in high-selectivity vector search?**"):
    st.markdown(
        """
        In **Post-Filtering**:
        1. The vector index retrieves the top 100 nearest neighbors purely by cosine distance.
        2. The application checks the metadata filter (e.g., `tenant_id == 'corp_42'`) on those 100 items.
        
        If tenant `corp_42` owns only 0.1% of the total documents in the database (high selectivity), all 100 retrieved neighbors will belong to other tenants.
        The post-filter drops 100% of candidates, returning zero results to the user even though valid documents exist in the corpus.
        Production systems require **Pre-Filtering** or **Single-Stage HNSW Filtered Traversal**.
        """
    )

with st.expander("**Q2: How does Pre-Filtering differ from In-Filter graph traversal in vector databases?**"):
    st.markdown(
        """
        - **Pre-Filtering**: The database creates a boolean bitset of all matching document IDs first, then restricts the nearest-neighbor search strictly to those IDs.
        - **In-Filter (Single-Stage) HNSW**: The HNSW graph traversal checks the metadata filter condition at every hop across graph nodes.
        If a node does not satisfy the filter, its distance is ignored, but its connections are still traversed to avoid disconnected graph dead-ends.
        """
    )

with st.expander("**Q3: What is the Self-Querying paradigm and how does it compile logical filter ASTs?**"):
    st.markdown(
        """
        Self-Querying prompts an LLM with:
        1. A list of metadata attribute names, types, and descriptions.
        2. Allowed comparison operators (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `and`, `or`).
        The LLM emits a structured JSON AST representing both the semantic search string and the boolean filter logic, which is compiled directly into the target database's query syntax.
        """
    )

with st.expander("**Q4: How do you handle schema drift in structured JSON document collections?**"):
    st.markdown(
        """
        JSON documents from evolving systems often have inconsistent schemas (e.g., `price` vs `cost`, or missing keys).
        Production architectures enforce:
        1. **Pydantic Data Contracts**: Ingestion workers validate and coerce documents into unified schemas.
        2. **Default Fallback Values**: Missing attributes are populated with neutral defaults (`None` or `0.0`) so metadata index filters do not raise null-pointer exceptions.
        """
    )

with st.expander("**Q5: What are the performance trade-offs of storing payloads inside vector databases vs separate relational DBs?**"):
    st.markdown(
        """
        - **In-Database Payloads (Single-Cluster)**: Extremely low latency ($< 5\text{ms}$); filters and vectors evaluate together in memory. However, large text blobs bloat RAM usage.
        - **Decoupled Architecture**: Vector DB stores only vectors and primary keys (`id`); the application fetches full records from PostgreSQL upon match. More memory-efficient, but requires a secondary network round-trip.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Relying on Post-Filtering for Multi-Tenant SaaS**  
        **Antipattern**: Retrieving Top-50 vector matches and filtering by `tenant_id` afterwards.  
        **Consequence**: Returns zero hits when a small tenant's documents are crowded out by large tenants, causing silent search failure.  
        **Production Fix**: Enforce pre-filtering or separate isolated vector collection partitions per tenant.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Embedding Numbers as Plain Text for Range Queries**  
        **Antipattern**: Concatenating numbers like `price: $3500` into chunk text expecting cosine similarity to handle `price < 5000`.  
        **Consequence**: Vector cosine similarity has zero comprehension of numeric ordering ($3500$ does not mathematically order between $1000$ and $5000$).  
        **Production Fix**: Store numeric attributes in structured scalar metadata fields for exact range filtering.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Unindexed Metadata Attribute Filtering**  
        **Antipattern**: Filtering on metadata fields without creating scalar indexes in the vector database.  
        **Consequence**: Forces full-table scans across millions of payload records, ballooning latency from 15ms to 1,500ms.  
        **Production Fix**: Create explicit payload indexes (e.g., Qdrant payload index, Milvus scalar index) on all filtered keys.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Overly Restrictive Automated Self-Queries**  
        **Antipattern**: Compiling conversational nuances into hard `eq` filters that eliminate valid candidate documents.  
        **Consequence**: A query like *"recent laptops"* compiles into `release_year == 2026`, dropping all valid 2025 laptops.  
        **Production Fix**: Use soft range filters (`gte`) and fallback broadening loops when filters return zero hits.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Self-Querying RAG?")
        st.markdown("**Back**: An architecture where an LLM decomposes a natural language query into a semantic search string and a structured boolean metadata filter dictionary before vector search.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Why does Post-Filtering fail?")
        st.markdown("**Back**: When a filter is highly selective, the top-K nearest neighbors retrieved without filters belong to other categories, causing post-filtering to discard all results.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is Single-Stage HNSW Filtering?")
        st.markdown("**Back**: Evaluating metadata filter conditions at every hop during HNSW graph traversal, ensuring search strictly discovers valid filtered nodes without dead-ends.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: Why are scalar metadata indexes needed?")
        st.markdown("**Back**: Without scalar payload indexing on filtered fields, the database must scan millions of records sequentially, degrading search latency.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of structured data RAG, metadata filtering, and self-query compilation:")

chk1 = st.checkbox("I understand why numerical range and boolean constraints must be handled via metadata filters rather than vector similarity.", value=True)
chk2 = st.checkbox("I can implement the Self-Querying paradigm to decompose queries into semantic strings and filter dictionaries.", value=True)
chk3 = st.checkbox("I know why post-filtering fails in high-selectivity multi-tenant environments and how pre-filtering resolves it.", value=True)
chk4 = st.checkbox("I understand how single-stage HNSW graph traversal integrates metadata constraints into index probing.", value=True)
chk5 = st.checkbox("I know how to index scalar metadata payloads to prevent full-table scan bottlenecks.", value=True)
chk6 = st.checkbox("I can integrate heterogeneous data formats (CSV, JSON, SQL) into a unified hybrid retrieval catalog.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Structured Data RAG and Self-Querying Architecture.")
