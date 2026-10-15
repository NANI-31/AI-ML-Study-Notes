import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class AdvancedQueryTechniquesEngine:
    """
    Simulates advanced query transformation paradigms:
    1. Step-Back Prompting (Abstracting specific queries into high-level concepts)
    2. HyDE - Hypothetical Document Embeddings (Generating hypothetical answer text to embed)
    3. Multi-Hop Query Decomposition (Synthesizing facts across multiple steps)
    4. Sub-Question Generation (Extracting atomic sub-tasks)
    5. Query Routing (Directing queries to specialized vector indices)
    """
    TECHNIQUE_DEMOS = {
        "Step-Back Prompting": {
            "query": "Did Acme Corp violate SEC Rule 10b-5 by delaying its Q3 2024 earnings announcement by 14 days?",
            "step_back_query": "What are the legal standards, reporting timelines, and violation thresholds for SEC Rule 10b-5?",
            "rationale": "Steps back from specific corporate entities and dates to retrieve the governing legal principles, which are then used to evaluate the specific case facts.",
            "target_retrieval": "SEC Compliance & Statutory Law Corpus"
        },
        "HyDE (Hypothetical Document Embeddings)": {
            "query": "How does PostgreSQL handle transaction isolation anomalies under Read Committed?",
            "hypothetical_doc": "Under Read Committed isolation in PostgreSQL, each query in a transaction sees a new snapshot of data committed before that query began. Row-level locks prevent dirty writes, but non-repeatable reads and phantom reads can still occur. Concurrent updates lock the target row until the conflicting transaction commits.",
            "rationale": "Generates a hypothetical technical answer before retrieval. Embedding this answer matches authoritative documentation in vector space far better than embedding an open question.",
            "target_retrieval": "PostgreSQL Internals Vector Store"
        },
        "Multi-Hop Query Reasoning": {
            "query": "Which microservices deployed in the EU cluster communicate with databases that had fatal connection pool timeouts last month?",
            "sub_questions": [
                "1. Which databases experienced fatal connection pool timeouts in the EU cluster last month?",
                "2. Which microservices have network routes or connection configs to those specific identified databases?"
            ],
            "rationale": "Multi-hop dependency where the output of Sub-Question 1 feeds as an entity constraint into Sub-Question 2.",
            "target_retrieval": "SRE Incident Postmortems + Kubernetes Service Topology"
        },
        "Sub-Question Generation": {
            "query": "Compare Apache Iceberg and Delta Lake on ACID transactions, time travel queries, and storage layout.",
            "sub_questions": [
                "1. How does Apache Iceberg implement ACID transactions, metadata snapshots, and storage layout?",
                "2. How does Delta Lake implement ACID transactions, transaction logs, and Parquet storage?",
                "3. Comparison of time travel query syntax and performance between Apache Iceberg and Delta Lake."
            ],
            "rationale": "Breaks an expansive comparative question into atomic, independently retrievable facets.",
            "target_retrieval": "Data Lakehouse Architecture Docs"
        },
        "Intelligent Query Routing": {
            "query": "SELECT * FROM billing_invoices WHERE amount > 5000 AND status = 'unpaid';",
            "route_decision": "SQL / Text-to-SQL Relational Engine",
            "fallback_route": "Unstructured Invoice OCR Vector Store",
            "rationale": "Detects structured SQL syntax and routes directly to the relational database engine, bypassing vector search completely.",
            "target_retrieval": "PostgreSQL Operational Billing DB"
        }
    }

    @classmethod
    def process_technique(cls, technique_name: str) -> Dict[str, Any]:
        return cls.TECHNIQUE_DEMOS[technique_name]

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="12.4 Advanced Query Techniques - RAG Master Curriculum",
    page_icon="🚀",
    layout="wide"
)

st.title("12.4 Advanced Query Techniques: Step-Back Prompting, HyDE, Multi-Hop, & Routing")
st.caption("Chapter 12: Query Transformation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Standard query transformations (like pronoun rewriting and synonym expansion) optimize retrieval for straightforward, 
single-intent questions. However, enterprise systems routinely encounter complex inquiries that require 
deeper cognitive pre-processing:
* **Abstract Reasoning Questions**: Inquiries where the answer requires understanding underlying principles before analyzing specific facts (**Step-Back Prompting**).
* **Asymmetric Vocabulary Gaps**: Questions where queries are short questions but target documents are detailed technical paragraphs (**HyDE - Hypothetical Document Embeddings**).
* **Chained Dependencies**: Questions where identifying Entity B depends on the search results for Entity A (**Multi-Hop Queries**).
* **Multi-Facet Comparisons**: Questions with several orthogonal facets requiring discrete retrieval passes (**Sub-Question Generation**).
* **Poly-Store Routing**: Directing inquiries to Vector Stores, Knowledge Graphs, or Relational SQL databases (**Query Routing**).

This lesson explores these advanced pre-retrieval techniques that power state-of-the-art Agentic RAG systems.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Advanced Query Transformation Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    adv_bp_dot = """
    digraph AdvQueryBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_ingress {
            label = "1. Query Ingress & Strategy Selection Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            user_query [label="Complex User Query\\ne.g. 'Compare Iceberg vs Delta Lake ACID & Time Travel'", fillcolor="#0F172A", color="#38BDF8"];
            strategy_router [label="Cognitive Strategy Router\\nEvaluates Abstraction, Multi-Hop, & Engine Needs", fillcolor="#334155", color="#10B981"];

            user_query -> strategy_router [label="Evaluate Intent"];
        }

        subgraph cluster_strategies {
            label = "2. Advanced Transformation Modules Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            step_back [label="Step-Back Engine\\nGenerates High-Level Conceptual Query", fillcolor="#0F172A", color="#F59E0B"];
            hyde_gen [label="HyDE Generator\\nSynthesizes Hypothetical Document Chunk", fillcolor="#0F172A", color="#8B5CF6"];
            sub_q_gen [label="Sub-Question Decomposer\\nGenerates Atomic Independent Sub-Tasks", fillcolor="#0F172A", color="#EC4899"];
            multi_hop_chain [label="Multi-Hop Planner\\nChains Sequential Dependent Steps", fillcolor="#0F172A", color="#06B6D4"];

            strategy_router -> step_back [label="Broad Principles Needed"];
            strategy_router -> hyde_gen [label="Asymmetric Vocab Gap"];
            strategy_router -> sub_q_gen [label="Multi-Facet Comparison"];
            strategy_router -> multi_hop_chain [label="Chained Dependency"];
        }

        subgraph cluster_retrieval {
            label = "3. Multi-Store Retrieval & Aggregation Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            vdb [label="Vector Database\\nHNSW Proximity Embeddings", fillcolor="#334155", color="#38BDF8"];
            sql_engine [label="Relational Database\\nStructured SQL Query Execution", fillcolor="#334155", color="#10B981"];
            graph_db [label="Knowledge Graph\\nEntity Relationship Traversals", fillcolor="#334155", color="#EC4899"];

            aggregator [label="Context Synthesizer LLM\\nCombines Evidence into Final Grounded Answer", fillcolor="#0F172A", color="#10B981"];

            step_back -> vdb;
            hyde_gen -> vdb [label="Embeds Hypo Doc"];
            sub_q_gen -> vdb;
            multi_hop_chain -> graph_db [label="Hop 1"];
            graph_db -> vdb [label="Hop 2"];
            strategy_router -> sql_engine [label="Structured Query"];

            vdb -> aggregator;
            sql_engine -> aggregator;
        }
    }
    """
    st.graphviz_chart(adv_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production Agentic RAG systems (such as **LangGraph**, **LlamaIndex Workflows**, or **DSPy**), 
these advanced query transformations are structured as conditional routing nodes. 
An orchestration agent evaluates the query state: if the query is conceptual, it executes a Step-Back retrieval; 
if the question is sparse, it generates a HyDE embedding; if the query involves structured data, 
it routes to a Text-to-SQL tool.
""")

st.divider()

# ==============================================================================
# SECTION 2: STEP-BACK PROMPTING
# ==============================================================================

st.header("2. Step-Back Prompting: Abstracting Specific Queries into Core Principles", divider="rainbow")

st.markdown("""
**Step-Back Prompting** (introduced by Google DeepMind in 2023) is a technique that grounds complex reasoning 
by asking a broader, more generic question before tackling specific details.
""")

col_sb1, col_sb2 = st.columns([1, 1])

with col_sb1:
    st.subheader("How Step-Back Prompting Operates")
    st.markdown("""
    * **The Specificity Trap in Naive Search**:
      * *Original Specific Query*: *"Did Acme Corp violate SEC Rule 10b-5 by delaying its Q3 2024 earnings announcement by 14 days?"*
      * If this specific query is searched directly, the retriever searches for *"Acme Corp Q3 2024"*. 
        It retrieves press releases and financial tables, but fails to retrieve the governing legal framework of SEC Rule 10b-5!
    * **The Step-Back Transformation**:
      * The model generates a **Step-Back Query**: 
        `"What are the reporting timelines, materiality standards, and legal requirements for SEC Rule 10b-5?"`
    * **Dual Context Retrieval**:
      1. Search 1 retrieves the legal principles (via the Step-Back query).
      2. Search 2 retrieves the company facts (via the specific query).
      3. The LLM synthesizes the answer by applying the retrieved legal principles to the specific corporate facts.
    """)

with col_sb2:
    st.subheader("Step-Back Prompt Pattern")
    st.code('''# Step-Back Question Generation Prompt
STEP_BACK_PROMPT = """You are an expert at information retrieval and abstraction.
Your task is to step back and paraphrase a high-level, generic question 
about the underlying physics, law, mathematics, or core concepts 
that would provide the foundational knowledge to answer the specific question.

Specific Question: {question}
Step-Back High-Level Question:"""
''', language="python")

st.divider()

# ==============================================================================
# SECTION 3: HYDE (HYPOTHETICAL DOCUMENT EMBEDDINGS)
# ==============================================================================

st.header("3. HyDE: Hypothetical Document Embeddings for Asymmetric Retrieval", divider="rainbow")

st.markdown("""
**HyDE (Hypothetical Document Embeddings)** (Gao et al., 2022) solves the **Query-Document Asymmetry** problem 
by generating a hallucinated passage and embedding that passage rather than the user's question.
""")

col_hyde1, col_hyde2 = st.columns([1, 1])

with col_hyde1:
    st.subheader("The Query-Document Asymmetry Problem")
    st.markdown("""
    * **The Geometric Gap**:
      * User queries are typically short, interrogative, and speculative (*"How does Postgres prevent lost updates?"*).
      * Knowledge base documents are long, expository, and declarative (*"PostgreSQL enforces snapshot isolation using multi-version concurrency control..."*).
      * Because questions and declarative answers inhabit different stylistic subspaces, their cosine similarity can be surprisingly low.
    * **How HyDE Solves It**:
      1. An LLM generates a **Hypothetical Document** answering the question (even if some facts are hallucinated).
      2. The hypothetical document is written in the **expository, declarative style** of the target corpus.
      3. The system generates an embedding of the **hypothetical document** (`vector = embed(hypo_doc)`).
      4. This vector searches the database, retrieving real documents written in the exact same stylistic neighborhood!
    """)

with col_hyde2:
    st.subheader("HyDE Workflow Diagram")
    hyde_dot = """
    digraph HyDEWorkflow {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.35, ranksep=0.45];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin="0.15,0.08"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=8, fontcolor="#94A3B8"];

        query [label="User Query: 'How does Postgres MVCC work?'", fillcolor="#0F172A", color="#38BDF8"];
        gen_doc [label="Hypothetical Expository Document\\n(Generated by LLM without search)", fillcolor="#334155", color="#EC4899"];
        embed_step [label="Embedding Model (f_doc)\\nEncodes Declarative Text", fillcolor="#0F172A", color="#10B981"];
        vdb [label="Vector Store (Target Corpus)\\nFinds Stylistically Similar Real Chunks", fillcolor="#334155", color="#38BDF8"];
        final_doc [label="Real Authoritative Chunks", fillcolor="#0F172A", color="#10B981"];

        query -> gen_doc [label="Generate"];
        gen_doc -> embed_step [label="Encode"];
        embed_step -> vdb [label="ANN Search"];
        vdb -> final_doc [label="Retrieve Real"];
    }
    """
    st.graphviz_chart(hyde_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: MULTI-HOP QUERIES
# ==============================================================================

st.header("4. Multi-Hop Queries: Sequential Dependency Chains and Iterative Retrieval", divider="rainbow")

st.markdown("""
In **Multi-Hop Queries**, the system cannot retrieve all required information in a single parallel step because 
the search parameters for Hop 2 depend directly on the entities discovered in Hop 1.
""")

col_mh1, col_mh2 = st.columns([1, 1])

with col_mh1:
    st.subheader("Sequential Dependency Execution")
    st.markdown("""
    **Example Multi-Hop Query**:
    > *"What was the revenue of the company that acquired Figma's competitor in 2023?"*

    * **Hop 1 (Entity Discovery)**:
      * Search: *"Figma competitors acquired in 2023"*
      * Result: Discovers entity *"InVision"* was acquired by *"Mural"*.
    * **Hop 2 (Fact Retrieval)**:
      * Search: *"Mural revenue 2023 financial report"*
      * Result: Retrieves the revenue numbers.
    * **Why Standard RAG Fails**:
      * Searching for the entire question directly yields zero results because the name *"Mural"* does not appear in the original query! 
      * Multi-hop pipelines must execute iteratively, feeding intermediate entities into subsequent query templates.
    """)

with col_mh2:
    st.subheader("Multi-Hop Pipeline Patterns")
    mh_patterns_df = pd.DataFrame([
        {
            "Pattern": "Iterative Vector RAG (Self-RAG)",
            "Mechanism": "Retriever fetches Hop 1 -> LLM inspects candidates and reformulates query -> Retriever fetches Hop 2.",
            "Latency": "200 - 450 ms",
            "Best For": "Unstructured document archives"
        },
        {
            "Pattern": "Graph RAG (Knowledge Graph Walk)",
            "Mechanism": "Extracts entity from query -> Walks relationship edges in Neo4j -> Extracts connected entity.",
            "Latency": "20 - 50 ms",
            "Best For": "Corporate org charts, ownership networks, supply chains"
        }
    ])
    st.dataframe(mh_patterns_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: SUB-QUESTION GENERATION
# ==============================================================================

st.header("5. Sub-Question Generation: Orthogonal Deconstruction of Multi-Topic Queries", divider="rainbow")

st.markdown("""
**Sub-Question Generation** deconstructs a broad, multi-topic question into a set of discrete, orthogonal sub-questions 
that can be executed concurrently.
""")

col_sq1, col_sq2 = st.columns([1, 1])

with col_sq1:
    st.subheader("Deconstruction Principles")
    st.markdown("""
    * **Atomic Independence**:
      Each generated sub-question must be fully self-contained, specifying all entities, technologies, and metrics explicitly.
    * **Exhaustive Coverage**:
      The combined answers of the sub-questions must completely cover the original inquiry without gaps.
    * **Parallel Dispatch**:
      Unlike Multi-Hop queries (which have sequential dependencies), sub-questions are orthogonal and execute concurrently:
      ```python
      # Concurrent sub-question execution
      tasks = [retriever.ainvoke(sq) for sq in sub_questions]
      results = await asyncio.gather(*tasks)
      ```
    """)

with col_sq2:
    st.subheader("Comparative Sub-Question Example")
    st.markdown("""
    **User Query**:
    *"Compare Apache Iceberg and Delta Lake on ACID transactions, time travel, and storage format."*

    **Generated Sub-Questions**:
    1. *Sub-Q 1*: *"How does Apache Iceberg implement ACID transactions, metadata snapshots, and storage layout?"*
    2. *Sub-Q 2*: *"How does Delta Lake implement ACID transactions, transaction logs, and Parquet storage?"*
    3. *Sub-Q 3*: *"How does time travel querying work in Apache Iceberg versus Delta Lake?"*
    """)

st.divider()

# ==============================================================================
# SECTION 6: QUERY ROUTING
# ==============================================================================

st.header("6. Query Routing: Directing Queries Across Vector Stores, SQL, and Knowledge Graphs", divider="rainbow")

st.markdown("""
In enterprise data platforms, information lives across diverse storage engines. 
**Query Routing** directs questions to the optimal storage paradigm.
""")

route_cards = st.columns(3)

with route_cards[0]:
    st.markdown("""
    **1. Unstructured Vector Stores**
    * **Target**: PDF manuals, customer reviews, blog posts, chat transcripts.
    * **Query Style**: Conversational natural language, conceptual how-to.
    * **Engine**: Qdrant, Pinecone, Chroma, Milvus.
    """)

with route_cards[1]:
    st.markdown("""
    **2. Structured Relational SQL**
    * **Target**: Exact financial ledger entries, inventory counts, user orders.
    * **Query Style**: Aggregations, exact sums, counts, date filters.
    * **Engine**: PostgreSQL, Snowflake, BigQuery.
    """)

with route_cards[2]:
    st.markdown("""
    **3. Knowledge Graphs**
    * **Target**: Enterprise hierarchies, fraud networks, dependency trees.
    * **Query Style**: Multi-hop entity relationship paths.
    * **Engine**: Neo4j, Amazon Neptune, Memgraph.
    """)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Advanced Query Strategy Router", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python module implementing Step-Back abstraction, HyDE document generation, 
sub-question generation, and poly-store routing.
""")

st.code(r'''
from typing import List, Dict, Any, Optional

class EnterpriseAdvancedQueryRouter:
    """
    Production-grade query router and advanced transformer:
    - Step-Back prompting for principle grounding
    - HyDE hypothetical passage synthesis
    - Multi-hop query plan generation
    - Multi-engine storage routing (Vector vs SQL vs Graph)
    """
    @classmethod
    def generate_step_back(cls, query: str) -> str:
        """
        Abstracts specific question into governing principle query.
        """
        # In production, invoked via fast LLM chain
        return f"What are the general rules, principles, and architectural standards governing: {query}?"

    @classmethod
    def generate_hyde_doc(cls, query: str) -> str:
        """
        Generates an expository declarative passage matching target corpus style.
        """
        return (
            f"This authoritative technical passage explains {query}. "
            f"The architecture utilizes standard industry protocols, enforcing deterministic "
            f"data integrity, low latency execution, and comprehensive error handling."
        )

    @classmethod
    def route_query(cls, query: str) -> Dict[str, Any]:
        """
        Directs query to optimal storage engine: SQL vs Graph vs Vector.
        """
        q_lower = query.lower()
        if any(term in q_lower for term in ["how many", "total count", "sum of", "average", "where amount"]):
            return {
                "route": "STRUCTURED_SQL",
                "engine": "PostgreSQL / Snowflake",
                "pipeline": "text_to_sql_execution"
            }
        elif any(term in q_lower for term in ["relationship between", "who owns", "reports to", "connected to"]):
            return {
                "route": "KNOWLEDGE_GRAPH",
                "engine": "Neo4j Graph Database",
                "pipeline": "cypher_graph_walk"
            }
        else:
            return {
                "route": "UNSTRUCTURED_VECTOR",
                "engine": "Qdrant Vector DB",
                "pipeline": "dense_hybrid_retrieval"
            }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Advanced Query Techniques Lab", divider="rainbow")

st.markdown("""
Select any of the advanced query techniques below to inspect how the system transforms queries, 
generates hypothetical embeddings, and routes across enterprise database engines.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    selected_technique = st.selectbox(
        "Select Advanced Technique to Demonstrate:",
        list(AdvancedQueryTechniquesEngine.TECHNIQUE_DEMOS.keys())
    )
    demo_info = AdvancedQueryTechniquesEngine.process_technique(selected_technique)

    st.markdown("#### 📥 Input Query")
    st.info(f"*\"{demo_info['query']}\"*")
    st.write(f"**Target Storage Engine**: `{demo_info['target_retrieval']}`")

with wb_col2:
    st.markdown("### 🔬 Execution Output & Cognitive Transformation")

    if selected_technique == "Step-Back Prompting":
        st.markdown("**Abstracted Step-Back Query**:")
        st.success(f"*\"{demo_info['step_back_query']}\"*")
        st.caption(f"💡 {demo_info['rationale']}")

    elif selected_technique == "HyDE (Hypothetical Document Embeddings)":
        st.markdown("**Synthesized Hypothetical Document (For Embedding)**:")
        st.success(f"*\"{demo_info['hypothetical_doc']}\"*")
        st.caption(f"💡 {demo_info['rationale']}")

    elif selected_technique == "Multi-Hop Query Reasoning":
        st.markdown("**Iterative Multi-Hop Dependency Plan**:")
        for step in demo_info["sub_questions"]:
            st.markdown(f"* {step}")
        st.caption(f"💡 {demo_info['rationale']}")

    elif selected_technique == "Sub-Question Generation":
        st.markdown("**Orthogonal Sub-Questions (Parallel Dispatch)**:")
        for sq in demo_info["sub_questions"]:
            st.markdown(f"* {sq}")
        st.caption(f"💡 {demo_info['rationale']}")

    else:  # Intelligent Query Routing
        st.markdown(f"**Selected Route**: `{demo_info['route_decision']}`")
        st.markdown(f"**Fallback Route**: `{demo_info['fallback_route']}`")
        st.caption(f"💡 {demo_info['rationale']}")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: How does Step-Back Prompting improve reasoning accuracy on complex domain questions?"):
    st.markdown("""
    **Answer**: Complex questions often require applying a general law, mathematical theorem, or architectural principle 
    to a specific set of facts. When specific questions are searched directly, retrieval models frequently retrieve surface-level 
    coincidences while failing to retrieve the governing principle. 
    Step-Back Prompting abstracts the query to retrieve the high-level principle first, then retrieves the case-specific facts. 
    Downstream LLM synthesis receives both contexts simultaneously, allowing it to ground specific deductions in verified core principles.
    """)

with st.expander("Q2: In what scenarios does HyDE (Hypothetical Document Embeddings) fail or cause severe regressions?"):
    st.markdown("""
    **Answer**: HyDE fails severely in **fact-seeking queries about unknown, private, or novel entities** where the LLM has zero training data. 
    If a user asks *"What was Project Apex's secret Q4 margin?"*, the LLM will generate a plausible-sounding but completely fabricated hypothetical document. 
    Embedding this hallucinated document drives the vector search towards documents matching the hallucination rather than the true secret project, 
    cementing catastrophic false-positive retrieval. HyDE should only be used for open-domain conceptual questions, never for exact fact lookup.
    """)

with st.expander("Q3: What is the architectural difference between Sub-Question Generation and Multi-Hop Retrieval?"):
    st.markdown("""
    **Answer**:
    * **Sub-Question Generation**: Produces **independent, orthogonal** sub-queries that can all be executed concurrently in parallel 
      via `asyncio.gather` (e.g., comparing feature X of tool A with feature X of tool B).
    * **Multi-Hop Retrieval**: Involves **sequential, dependent** queries where the search parameters of Hop 2 cannot be formulated 
      until the output of Hop 1 is retrieved and parsed (e.g., *"Find the author of book X, then find all awards won by that author"*).
    """)

with st.expander("Q4: How do modern systems route queries across Vector Databases versus Relational SQL databases?"):
    st.markdown("""
    **Answer**: Systems use semantic intent routers (via structured function calling or shallow classifiers). 
    If the query involves quantitative aggregations (*"How many customers churned in May?"*), arithmetic operations, 
    or strict relational joins, the router translates the query into SQL via Text-to-SQL. 
    If the query involves unstructured semantic search (*"What are customer complaints regarding billing?"*), 
    the router directs the inquiry to the vector store.
    """)

with st.expander("Q5: What are the latency and cost overheads of running HyDE in production?"):
    st.markdown("""
    **Answer**: HyDE introduces an additional LLM generation step (typically 150–350ms and 150 output tokens) *before* vector retrieval can begin. 
    This roughly doubles total search latency. 
    Production systems mitigate this by:
    1. Using lightweight, high-speed distilled models (e.g., Gemini 1.5 Flash or Claude 3 Haiku).
    2. Generating short hypothetical passages (< 60 words).
    3. Caching hypothetical documents for popular queries.
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Deploying HyDE on Private Enterprise Codebases and Internal Entities**
    * **Symptom**: System retrieves completely wrong internal documents with high confidence.
    * **Root Cause**: The LLM hallucinates an internal API method or internal team name, embedding the hallucination and pulling irrelevant matches.
    * **Remedy**: Disable HyDE for queries containing proprietary acronyms, internal error codes, or private entities.
    """)

    st.error("""
    **Trap 2: Sequential Execution of Independent Sub-Questions**
    * **Symptom**: Comparative search requests take 1.5+ seconds to return.
    * **Root Cause**: Executing 3 generated sub-questions serially in a synchronous loop.
    * **Remedy**: Always dispatch independent sub-questions concurrently using `asyncio.gather`.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Unbounded Multi-Hop Loops (Infinite Reasoning Loops)**
    * **Symptom**: Autonomous agent gets stuck executing 10+ retrieval hops on ambiguous queries, draining API budgets.
    * **Root Cause**: Multi-hop planning loop lacks a hard termination condition or max-hop depth budget.
    * **Remedy**: Enforce a strict maximum hop limit (e.g., `max_hops = 3`) and trigger human-in-the-loop clarification if unresolved.
    """)

    st.warning("""
    **Trap 4: Rigid Single-Engine Routing Without Hybrid Fallback**
    * **Symptom**: Text-to-SQL router attempts to query an unstructured text column with SQL LIKE operators, returning zero results.
    * **Root Cause**: Routing decisions are binary without multi-store hybrid fallbacks.
    * **Remedy**: Implement dual-routing with confidence thresholds: if SQL returns empty results, automatically fall back to vector search.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Step-Back Prompting?")
        st.markdown("**Back**: An abstraction technique where the system generates a high-level conceptual question to retrieve governing principles before tackling the specific factual case.")

    with st.container(border=True):
        st.markdown("**Front**: How does HyDE (Hypothetical Document Embeddings) operate?")
        st.markdown("**Back**: An LLM generates a hypothetical declarative answer, and the system embeds that answer instead of the query, matching the expository style of target documents in vector space.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: Why is HyDE dangerous for private internal data lookup?")
        st.markdown("**Back**: Because the LLM lacks knowledge of private entities and hallucinates realistic details, steering vector search towards documents matching the hallucination.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: What distinguishes Sub-Question Generation from Multi-Hop Queries?")
            st.markdown("**Back**: Sub-questions are orthogonal and execute in parallel; Multi-Hop queries have sequential dependencies where Hop 2 requires the output entity of Hop 1.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Advanced Query Techniques, Step-Back Prompting, HyDE, Multi-Hop, and Routing:
""")

chk1 = st.checkbox("I understand how Step-Back Prompting retrieves governing principles to ground factual deductions.", value=True)
chk2 = st.checkbox("I can articulate how HyDE resolves Query-Document stylistic asymmetry by embedding declarative text.", value=True)
chk3 = st.checkbox("I understand the critical failure mode of HyDE when applied to private or proprietary entities.", value=True)
chk4 = st.checkbox("I know how to execute Multi-Hop queries iteratively when subsequent searches depend on intermediate entities.", value=True)
chk5 = st.checkbox("I understand how Sub-Question Generation breaks complex comparisons into parallel execution tasks.", value=True)
chk6 = st.checkbox("I know how to design a multi-store query router directing traffic across Vector, SQL, and Graph engines.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Magnificent! You have mastered Advanced Query Techniques and Cognitive Transformation pipelines.")
