import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class QueryUnderstandingEngine:
    """
    Cognitive pre-retrieval query analyzer. Evaluates incoming user queries
    before touching vector indices: performs intent detection, domain classification,
    complexity scoring (single-hop vs. multi-hop), and ambiguity validation.
    """
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy)\b",
        r"^(thanks|thank you|appreciate it)\b",
        r"^how are you"
    ]

    COMPARISON_MARKERS = [
        r"\bcompare\b", r"\bversus\b", r"\bvs\b", r"\bdifference between\b",
        r"\bboth\b", r"\band also\b", r"\bas well as\b"
    ]

    PROCEDURAL_MARKERS = [
        r"\bhow (do I|to|can I)\b", r"\bstep by step\b", r"\bconfigure\b",
        r"\binstall\b", r"\bdeploy\b", r"\btroubleshoot\b"
    ]

    PRESET_QUERIES = {
        "Casual Greeting (Chitchat)": "Hello there! Hope you are having a wonderful productive day.",
        "Simple Factoid (Single-Hop)": "What is the standard parental leave duration for full-time employees?",
        "Multi-Hop Comparison": "Compare our Q3 2024 gross margin against the 2022 forecast and list key variances.",
        "Technical Procedural (How-To)": "How do I configure PgBouncer transaction pooling with client SSL certificates?",
        "Underspecified / Ambiguous": "What is the policy?"
    }

    @classmethod
    def analyze_query(cls, query_str: str) -> Dict[str, Any]:
        q = query_str.strip()
        tokens = [t.lower() for t in re.findall(r"\b\w+\b", q)]
        token_count = len(tokens)

        # 1. Chitchat check
        for pattern in cls.GREETING_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE) and token_count <= 12:
                return {
                    "query": q,
                    "intent": "Casual Chitchat / Social Greeting",
                    "requires_retrieval": False,
                    "complexity": "Zero-Hop (No Knowledge Base Needed)",
                    "complexity_score": 0.05,
                    "target_domain": "Direct Conversational LLM",
                    "route": "Bypass Vector DB -> Direct Social Response",
                    "rationale": "Casual greeting detected. Passing this to vector store causes hallucinated corporate document retrieval."
                }

        # 2. Underspecified check
        if token_count <= 4 and not any(re.search(p, q, re.IGNORECASE) for p in cls.PROCEDURAL_MARKERS):
            return {
                "query": q,
                "intent": "Underspecified / Ambiguous Request",
                "requires_retrieval": False,
                "complexity": "Ambiguous (Insufficient Information)",
                "complexity_score": 0.30,
                "target_domain": "Clarification Engine",
                "route": "Halt Search -> Interactive Clarification Prompt",
                "rationale": "Query has only a few generic tokens without entity identifiers. Blind search returns noisy mixed documents."
            }

        # 3. Multi-hop check
        is_comparison = any(re.search(p, q, re.IGNORECASE) for p in cls.COMPARISON_MARKERS)
        has_multiple_dates_or_entities = len(re.findall(r"\b(19\d\d|20\d\d|Q[1-4])\b", q, re.IGNORECASE)) >= 2

        if is_comparison or has_multiple_dates_or_entities or (" and " in q and token_count > 14):
            return {
                "query": q,
                "intent": "Multi-Hop Analytical Synthesis",
                "requires_retrieval": True,
                "complexity": "Multi-Hop (Multi-Document Reasoning)",
                "complexity_score": 0.90,
                "target_domain": "Financial & Strategy Archives",
                "route": "Decompose Query -> Parallel Sub-Query Retrieval -> Cross-Document Synthesis",
                "rationale": "Inquiry requires synthesizing multiple temporal periods or distinct entities. A single vector lookup fails."
            }

        # 4. Procedural check
        if any(re.search(p, q, re.IGNORECASE) for p in cls.PROCEDURAL_MARKERS):
            return {
                "query": q,
                "intent": "Procedural How-To / Configuration Guide",
                "requires_retrieval": True,
                "complexity": "Single-Hop Procedural",
                "complexity_score": 0.55,
                "target_domain": "Engineering Runbooks & Tech Docs",
                "route": "Hybrid BM25 + Dense Search on Runbooks Index",
                "rationale": "Actionable technical instructions requested. Routes to technical documentation with exact syntax matching."
            }

        # 5. Default Factoid
        return {
            "query": q,
            "intent": "Direct Factoid Knowledge Lookup",
            "requires_retrieval": True,
            "complexity": "Single-Hop Direct",
            "complexity_score": 0.40,
            "target_domain": "General Corporate Knowledge Base",
            "route": "Standard Dense Vector Search (Top-K = 3)",
            "rationale": "Focused singular informational request. Single vector proximity lookup is optimal."
        }

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="12.1 Query Understanding - RAG Master Curriculum",
    page_icon="🧐",
    layout="wide"
)

st.title("12.1 Query Understanding: Intent Detection, Query Classification, & Complexity Scoring")
st.caption("Chapter 12: Query Transformation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In naive RAG architectures, user query strings are piped directly into an embedding model:
`documents = vector_store.search(user_raw_string)`.

In real-world enterprise applications, this pattern fails systematically because human queries are fraught with conversational artifacts:
* **Conversational Greetings**: *"Hello, hope you're having a great day!"* (Passing this to a vector database forces an ANN lookup, returning irrelevant corporate memos).
* **Pronoun Ambiguity**: *"What did she say about his contract?"* (Lacks explicit entities and context).
* **Multi-Part Questions**: *"Compare our 2024 gross margin against the 2022 forecast and list key variances."* (A single vector lookup cannot retrieve evidence across disparate historical files).
* **Underspecified Queries**: *"What is the policy?"* (Which department, region, or product line?).

**Query Understanding** serves as the cognitive inspection gate before retrieval. It analyzes incoming text, 
detects user intent, classifies query complexity, filters out chitchat, and routes inquiries to specialized retrieval pipelines.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Query Understanding Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    qu_bp_dot = """
    digraph QueryUnderstandingBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_ingress {
            label = "1. Raw Query Ingress & Tokenizer Analysis Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            raw_query [label="Raw User Input String\\ne.g. 'Compare 2024 margins with 2022 strategic plan'", fillcolor="#0F172A", color="#38BDF8"];
            fast_regex [label="Linguistic Feature Extractor\\nToken Count, Entity Density, Temporal Markers", fillcolor="#334155", color="#10B981"];

            raw_query -> fast_regex [label="Inspect Syntax"];
        }

        subgraph cluster_classification {
            label = "2. Cognitive Intent & Complexity Classification Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            chitchat_gate [label="Chitchat & Social Filter\\nZero Retrieval Required (Bypass DB)", fillcolor="#0F172A", color="#F59E0B"];
            ambiguity_gate [label="Ambiguity / Specificity Check\\nTriggers Clarification Dialogue", fillcolor="#0F172A", color="#EC4899"];
            complexity_eval [label="Complexity Scorer\\nSingle-Hop (Score < 0.6) vs Multi-Hop (Score >= 0.6)", fillcolor="#0F172A", color="#8B5CF6"];

            fast_regex -> chitchat_gate [label="Greeting Pattern"];
            fast_regex -> ambiguity_gate [label="Token Count <= 4"];
            fast_regex -> complexity_eval [label="Information Query"];
        }

        subgraph cluster_routing {
            label = "3. Downstream Execution Routing Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            route_bypass [label="Direct Generator LLM\\nFriendly Social Response", fillcolor="#334155", color="#F59E0B"];
            route_clarify [label="Clarification Prompt Generator\\n'Which policy do you mean?'", fillcolor="#334155", color="#EC4899"];
            route_single [label="Standard Single-Hop Retriever\\nVector Store Top-K Lookups", fillcolor="#334155", color="#10B981"];
            route_decomp [label="Query Decomposition Pipeline\\nGenerates Independent Sub-Queries", fillcolor="#334155", color="#8B5CF6"];

            chitchat_gate -> route_bypass [label="Bypass Search"];
            ambiguity_gate -> route_clarify [label="Request Context"];
            complexity_eval -> route_single [label="Single-Hop"];
            complexity_eval -> route_decomp [label="Multi-Hop"];
        }
    }
    """
    st.graphviz_chart(qu_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production RAG systems, query understanding executes through a two-stage filter. 
First, ultra-low latency regex heuristics and Trie token dictionaries evaluate the query in under 1 millisecond on CPU. 
If the query is identified as chitchat, the system bypasses retrieval completely. 
Second, complex queries are passed to a lightweight, instruction-tuned classifier (such as **Gemini 1.5 Flash** or a local **DistilBERT** classifier) 
to output structured JSON specifying intent category, entity slots, target domain partitions, and multi-hop complexity flags.
""")

st.divider()

# ==============================================================================
# SECTION 2: USER QUERY ANALYSIS
# ==============================================================================

st.header("2. User Query Analysis: Syntactic, Semantic, and Conversational Parsing", divider="rainbow")

st.markdown("""
User query analysis deconstructs raw human input into computational features required by retrieval engines.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Core Query Deconstruction Pillars")
    st.markdown("""
    * **Surface Form vs. Semantic Goal**:
      * *Surface Form*: The exact literal tokens typed by the user.
      * *Semantic Goal*: The underlying information state the user wants to achieve.
    * **Entity & Slot Extraction**:
      * Identifying specific named entities (e.g., product names, person names, dates, financial quarters).
      * Extracted entities are mapped directly to metadata filters (`year == 2024`, `category == 'finance'`).
    * **Conversational Co-reference Resolution**:
      * In multi-turn dialogs, users use pronouns (*"What was its price?"*). Query analysis replaces pronouns 
        with referenced antecedent entities from conversation memory before search.
    * **Noise Pruning**:
      * Stripping conversational filler (*"Can you please tell me if..."* -> *"vacation rollover policy"*).
    """)

with col2:
    st.subheader("Query Analysis Transformation Matrix")
    analysis_df = pd.DataFrame([
        {
            "Raw User Input": "Hi! Can you tell me what the PTO rollover policy is for 2026?",
            "Extracted Entities": "Entity: PTO Rollover; Year: 2026",
            "Filtered Noise": "Removed 'Hi!', 'Can you tell me'",
            "Optimized Search String": "PTO rollover policy 2026"
        },
        {
            "Raw User Input": "How do I fix KafkaConsumer poll crashing with NullPointer?",
            "Extracted Entities": "Class: KafkaConsumer; Method: poll; Error: NullPointer",
            "Filtered Noise": "Removed 'How do I fix', 'crashing with'",
            "Optimized Search String": "KafkaConsumer.poll NullPointerException fix"
        },
        {
            "Raw User Input": "Thanks so much, that answered everything!",
            "Extracted Entities": "None (Conversational closure)",
            "Filtered Noise": "Complete chitchat string",
            "Optimized Search String": "[BYPASS_RETRIEVAL]"
        }
    ])
    st.dataframe(analysis_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: INTENT DETECTION
# ==============================================================================

st.header("3. Intent Detection: Categorizing User Goals and Action Paths", divider="rainbow")

st.markdown("""
Intent detection identifies what type of cognitive action the user expects the system to perform.
""")

intent_cols = st.columns(4)

with intent_cols[0]:
    st.markdown("""
    **1. Factoid Lookup**
    * **Definition**: Seeks a specific, singular data point or definition.
    * **Example**: *"What is the capital expenditure budget for Q2?"*
    * **Downstream Action**: High-precision dense vector search with low Top-K (`K = 3`).
    """)

with intent_cols[1]:
    st.markdown("""
    **2. Procedural How-To**
    * **Definition**: Seeks step-by-step instructions or troubleshooting runbooks.
    * **Example**: *"How do I rotate database credentials in AWS KMS?"*
    * **Downstream Action**: Hybrid BM25 + dense search restricted to engineering documentation.
    """)

with intent_cols[2]:
    st.markdown("""
    **3. Comparative Synthesis**
    * **Definition**: Evaluates trade-offs or changes across two or more entities.
    * **Example**: *"Compare Postgres vs. Mongo for JSON document storage."*
    * **Downstream Action**: Query decomposition into multiple parallel vector queries.
    """)

with intent_cols[3]:
    st.markdown("""
    **4. Conversational Chitchat**
    * **Definition**: Social greetings, compliments, or casual interaction.
    * **Example**: *"Good morning, Antigravity!"*
    * **Downstream Action**: Bypasses vector storage entirely to prevent context pollution.
    """)

st.divider()

# ==============================================================================
# SECTION 4: QUERY CLASSIFICATION
# ==============================================================================

st.header("4. Query Classification: Domain Partitioning and Security Scoping", divider="rainbow")

st.markdown("""
Query classification routes questions into specialized enterprise knowledge silos, enforcing security boundaries 
and preventing cross-domain search contamination.
""")

col_cls1, col_cls2 = st.columns([1, 1])

with col_cls1:
    st.subheader("Domain-Partitioned Knowledge Routing")
    st.markdown("""
    Large enterprises maintain hundreds of document collections spanning different business verticals:
    * **Human Resources & Benefits**: Policies, payroll, medical insurance, PTO.
    * **Legal & Regulatory**: Contracts, compliance briefs, NDAs, litigation precedent.
    * **Software & Site Reliability**: API docs, architecture diagrams, runbooks, GitHub issues.
    * **Finance & Accounting**: SEC filings, quarterly earnings, audit reports, tax schedules.

    **The Danger of Flat Searches**:
    Executing a search against a single unpartitioned vector store containing all corporate files frequently 
    returns legal disclaimers for HR questions, or internal HR documents for external customer inquiries. 
    Query classification restricts the search space via partition filters (`collection_scope == 'HR'`).
    """)

with col_cls2:
    st.subheader("Security & Access Control Enforcement")
    st.markdown("""
    * **Role-Based Access Control (RBAC)**:
      * Query classification verifies whether the authenticated user has permissions to search the classified domain.
      * If an employee queries *"Executive compensation packages"*, classification tags the domain as `Executive_HR`. 
        If the user lacks `HR_EXEC_READ` credentials, the query is blocked before any vector embedding lookup executes.
    * **PII & Sensitive Token Detection**:
      * Queries containing Social Security Numbers or credit card digits are flagged and sanitized.
    """)

st.divider()

# ==============================================================================
# SECTION 5: QUERY COMPLEXITY DETECTION
# ==============================================================================

st.header("5. Query Complexity Detection: Single-Hop vs. Multi-Hop Reasoning", divider="rainbow")

st.markdown("""
Determining whether a question requires a **single document** or **multi-document synthesis** is the most 
critical factor for choosing retrieval pipelines.
""")

col_comp1, col_comp2 = st.columns([1, 1])

with col_comp1:
    st.subheader("Single-Hop vs. Multi-Hop Architectures")
    st.markdown("""
    * **Single-Hop Query**:
      * *Characteristics*: The complete answer resides within a single coherent text chunk.
      * *Execution*: Single vector lookup -> Top-K candidates -> LLM generation.
      * *Latency*: 15ms - 30ms.
    * **Multi-Hop Query**:
      * *Characteristics*: Answering the question requires bridging facts across multiple disparate documents.
      * *Example*: *"Did the CEO who signed the 2018 merger agreement also approve the 2024 divestiture?"*
      * *Why Naive RAG Fails*: A single query vector represents the average semantic point between 2018 and 2024. 
        It retrieves documents about one event or the other, but rarely both!
      * *Execution*: Query Decomposition -> Sub-query 1: *"Who signed 2018 merger agreement?"* -> 
        Sub-query 2: *"Who approved 2024 divestiture?"* -> Cross-document join.
    """)

with col_comp2:
    st.subheader("Complexity Scoring Matrix")
    comp_df = pd.DataFrame([
        {
            "Query Attribute": "Singular Subject & Predicate",
            "Example": "What is the deductible for Dental Plan B?",
            "Complexity Score": "0.20 (Low)",
            "Recommended Execution": "Single-Hop Vector Search"
        },
        {
            "Query Attribute": "Temporal Comparison",
            "Example": "Compare Q1 2023 revenue vs. Q1 2024 revenue",
            "Complexity Score": "0.75 (High)",
            "Recommended Execution": "Multi-Hop Query Decomposition"
        },
        {
            "Query Attribute": "Conditional Multi-Entity Reasoning",
            "Example": "Which vendors approved in 2023 have active SOC2 compliance in 2026?",
            "Complexity Score": "0.95 (Very High)",
            "Recommended Execution": "Agentic Iterative Graph RAG"
        }
    ])
    st.dataframe(comp_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("6. Production Implementation: Enterprise Query Understanding Engine", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python module implementing upfront query understanding with regex pre-filtering, 
complexity scoring, domain classification, and structured execution plan output.
""")

st.code(r'''
import re
from typing import Dict, Any, List

class EnterpriseQueryUnderstandingEngine:
    """
    Cognitive pre-retrieval query analyzer. Evaluates incoming user queries
    before touching vector indices: performs intent detection, domain classification,
    complexity scoring (single-hop vs. multi-hop), and ambiguity validation.
    """
    GREETINGS = [
        r"^(hi|hello|hey|good morning|good afternoon|good evening)\b",
        r"^(thanks|thank you|appreciate it)\b"
    ]
    COMPARISON_REGEX = re.compile(r"\b(compare|versus|vs|difference between|both|and also)\b", re.IGNORECASE)
    PROCEDURAL_REGEX = re.compile(r"\b(how do i|how to|configure|install|deploy|troubleshoot)\b", re.IGNORECASE)
    YEAR_REGEX = re.compile(r"\b(19\d\d|20\d\d|Q[1-4])\b", re.IGNORECASE)

    @classmethod
    def analyze(cls, query_string: str) -> Dict[str, Any]:
        cleaned = query_string.strip()
        tokens = re.findall(r"\b\w+\b", cleaned.lower())
        token_count = len(tokens)

        # 1. Social Chitchat Guard
        for pattern in cls.GREETINGS:
            if re.search(pattern, cleaned, re.IGNORECASE) and token_count <= 8:
                return {
                    "query": cleaned,
                    "action": "BYPASS_RETRIEVAL",
                    "intent": "CHITCHAT",
                    "complexity": "ZERO_HOP",
                    "requires_retrieval": False,
                    "target_domain": "CONVERSATIONAL",
                    "route": "direct_llm_social_response"
                }

        # 2. Ambiguity Guard
        if token_count <= 3 and not cls.PROCEDURAL_REGEX.search(cleaned):
            return {
                "query": cleaned,
                "action": "CLARIFICATION_REQUIRED",
                "intent": "UNDERSPECIFIED",
                "complexity": "AMBIGUOUS",
                "requires_retrieval": False,
                "target_domain": "CLARIFICATION_AGENT",
                "route": "prompt_user_for_context"
            }

        # 3. Multi-Hop Reasoning Check
        has_comparison = bool(cls.COMPARISON_REGEX.search(cleaned))
        temporal_matches = cls.YEAR_REGEX.findall(cleaned)
        if has_comparison or len(temporal_matches) >= 2 or (token_count > 16 and " and " in cleaned):
            return {
                "query": cleaned,
                "action": "DECOMPOSE_QUERY",
                "intent": "COMPARATIVE_SYNTHESIS",
                "complexity": "MULTI_HOP",
                "requires_retrieval": True,
                "target_domain": "MULTI_COLLECTION",
                "route": "sub_query_decomposition_pipeline"
            }

        # 4. Procedural How-To Check
        if cls.PROCEDURAL_REGEX.search(cleaned):
            return {
                "query": cleaned,
                "action": "EXECUTE_HYBRID_SEARCH",
                "intent": "PROCEDURAL_HOW_TO",
                "complexity": "SINGLE_HOP_PROCEDURAL",
                "requires_retrieval": True,
                "target_domain": "TECH_DOCS_RUNBOOKS",
                "route": "hybrid_bm25_dense_search"
            }

        # 5. Default Factoid Search
        return {
            "query": cleaned,
            "action": "EXECUTE_VECTOR_SEARCH",
            "intent": "FACTOID_LOOKUP",
            "complexity": "SINGLE_HOP_DIRECT",
            "requires_retrieval": True,
            "target_domain": "GENERAL_KNOWLEDGE_BASE",
            "route": "dense_vector_search_top_k"
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 7: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("7. Interactive Workbench: Query Understanding Analyzer", divider="rainbow")

st.markdown("""
Test how the Query Understanding Engine inspects real-world queries, identifies intent, 
computes complexity scores, and selects optimal downstream execution routes.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    selected_preset = st.selectbox(
        "Select Example Query Preset:",
        list(QueryUnderstandingEngine.PRESET_QUERIES.keys())
    )
    user_custom_input = st.text_area(
        "Or Enter Custom User Query String:",
        value=QueryUnderstandingEngine.PRESET_QUERIES[selected_preset]
    )
    strict_mode = st.toggle("Enforce Strict Ambiguity Cutoff", value=True)

with wb_col2:
    st.markdown("### 🔍 Live Cognitive Inspection")
    analysis_res = QueryUnderstandingEngine.analyze_query(user_custom_input)

    st.write(f"**Detected Intent**: `{analysis_res['intent']}`")
    st.write(f"**Complexity Classification**: `{analysis_res['complexity']}`")
    st.write(f"**Downstream Route**: `{analysis_res['route']}`")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Requires Retrieval?", "Yes" if analysis_res["requires_retrieval"] else "No")
    metric_cols[1].metric("Complexity Score", f"{analysis_res['complexity_score']:.2f}")
    metric_cols[2].metric("Target Domain", analysis_res["target_domain"])

    if not analysis_res["requires_retrieval"]:
        st.warning(f"⚠️ **Action**: {analysis_res['rationale']}")
    else:
        st.success(f"✅ **Action**: {analysis_res['rationale']}")

st.divider()

# ==============================================================================
# SECTION 8: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("8. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why is upfront Chitchat Detection essential in enterprise RAG systems?"):
    st.markdown("""
    **Answer**: If a user submits a casual greeting like *"Hello, how are you today?"*, passing this string into a vector database 
    forces an approximate nearest neighbor search. The vector database compares the greeting vector against corporate document vectors 
    and returns the closest mathematical matches (e.g., an email sign-off or employee handbook introduction). 
    The context prompt is polluted with irrelevant corporate text, and the LLM responds awkwardly citing internal manuals! 
    An upfront chitchat detector intercepts greetings in under 1 millisecond on CPU, bypassing the database entirely and answering cordially.
    """)

with st.expander("Q2: What is the difference between a Single-Hop and a Multi-Hop query, and why do multi-hop queries break naive RAG?"):
    st.markdown("""
    **Answer**:
    * **Single-Hop**: The question can be answered completely from a single coherent document chunk (e.g., *"What is the standard vacation allowance?"*).
    * **Multi-Hop**: The question requires gathering discrete facts from multiple disparate documents and reasoning across them (e.g., *"Did our 2024 operating margin improve compared to the target set in the 2022 strategic plan?"*).
    * **Why Naive RAG Fails**: A single dense embedding represents the geometric average of the entire query. It will retrieve documents discussing 2024 margins OR the 2022 strategic plan, but rarely both, starving the LLM of the evidence needed to make the comparison. Multi-hop queries require **Query Decomposition** into independent sub-queries.
    """)

with st.expander("Q3: How should an enterprise RAG system handle underspecified or ambiguous queries before retrieval?"):
    st.markdown("""
    **Answer**: When a user submits an ambiguous query (e.g., *"What is the warranty policy?"* in an enterprise with 50 distinct hardware products), 
    an intelligent query understanding layer detects low specificity. Instead of executing a blind search that returns mixed, noisy warranty documents, 
    the system triggers a **Clarification Dialogue**: *"We have warranties for Laptops, Mobile Devices, and Server Appliances. Which product line would you like to review?"* 
    This eliminates wasted tokens, reduces embedding latency, and guarantees high-precision answers.
    """)

with st.expander("Q4: What latency budget should be allocated to the query understanding stage in production search?"):
    st.markdown("""
    **Answer**: In high-throughput production search APIs with an overall budget of 150–200ms, query understanding must complete in 
    **under 10 to 20 milliseconds**. Invoking an external heavyweight LLM (e.g., GPT-4) adds 300–600ms of latency, which is unacceptable. 
    Production systems achieve low latency using a tiered cascade: deterministic regex heuristics and Trie dictionaries (< 1ms) 
    for 70% of standard queries, followed by a fast lightweight local model (such as DistilBERT, DeBERTa, or Gemini 1.5 Flash) 
    only when query intent is ambiguous.
    """)

with st.expander("Q5: How does query understanding enhance security and Role-Based Access Control (RBAC)?"):
    st.markdown("""
    **Answer**: Query understanding parses the semantic domain of the query (e.g., `Financial_Audits`, `Executive_Payroll`, `Customer_PII`) 
    before dispatching database calls. The orchestration layer checks the authenticated user's access tokens against the identified domain scope. 
    If a junior developer queries *"Q4 Executive Bonus Allocations"*, the query is intercepted at the classification gate and denied, 
    preventing sensitive document chunks from even entering the retriever's similarity candidate heap.
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Synchronous Heavy LLM Calls on Every Single User Turn**
    * **Symptom**: Search API latency balloons from 25ms to 650ms after deploying query intent detection.
    * **Root Cause**: Calling an external 70B+ parameter model synchronously to classify simple 3-word queries and social pleasantries.
    * **Remedy**: Implement a tiered cascade: use compiled regex rules and in-memory Trie lookups first (< 1ms); only call lightweight sub-second models when rule-based confidence is low.
    """)

    st.error("""
    **Trap 2: Hard Partition Locking Without Fallback Routing**
    * **Symptom**: User receives a "No documents found" error even though the relevant knowledge exists in the company knowledge base.
    * **Root Cause**: The classifier routed the query strictly into `HR_Benefits`, but the specific document was filed under `Operations_Payroll`.
    * **Remedy**: Never hard-lock to a single partition when classification confidence is below 90%. Implement multi-partition secondary fallbacks.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Preserving Raw Conversational Pronouns in Vector Lookups**
    * **Symptom**: In a multi-turn chat, asking *"Can you explain how to configure it?"* returns irrelevant documents about unrelated technologies.
    * **Root Cause**: The raw pronoun "it" is embedded directly without resolving the antecedent entity from previous conversational turns.
    * **Remedy**: Run conversational queries through Co-reference Resolution before embedding, substituting pronouns with explicit entities.
    """)

    st.warning("""
    **Trap 4: Blind Multi-Hop Decomposition on Simple Questions**
    * **Symptom**: System generates 3 redundant sub-queries for a straightforward question, tripling database load and synthesis latency.
    * **Root Cause**: An overly sensitive multi-hop detector triggered decomposition simply because the query contained the word "and".
    * **Remedy**: Enforce strict complexity thresholds requiring multiple distinct temporal dates or entity clauses before triggering multi-query decomposition.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Query Understanding in modern RAG architectures?")
        st.markdown("**Back**: The pre-retrieval cognitive inspection stage that analyzes user queries to detect intent, classify complexity, prune conversational noise, and route to specialized retrieval pipelines.")

    with st.container(border=True):
        st.markdown("**Front**: Why must social chitchat bypass vector database retrieval?")
        st.markdown("**Back**: Casual greetings ('hello', 'thank you') force vector databases to return nearest-neighbor corporate documents, polluting LLM context with irrelevant memos and causing awkward hallucinations.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: What defines a Multi-Hop query and why does naive RAG fail on it?")
        st.markdown("**Back**: A query requiring synthesis across multiple disparate documents. Naive RAG fails because a single dense vector averages concepts, retrieving documents about one part of the question while missing the others.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: What is the recommended latency budget for the Query Understanding stage?")
            st.markdown("**Back**: Under 10 to 20 milliseconds, achieved via rule-based regex and Trie dictionaries for the fast path, reserving lightweight models only for ambiguous cases.")

st.divider()

# ==============================================================================
# SECTION 9: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("9. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Query Understanding, Intent Detection, and Complexity Scoring:
""")

chk1 = st.checkbox("I understand why piping raw user input directly into vector search is a critical architectural vulnerability.", value=True)
chk2 = st.checkbox("I can articulate how upfront chitchat filtering protects LLM context prompts from irrelevant document pollution.", value=True)
chk3 = st.checkbox("I understand the architectural distinction between Single-Hop and Multi-Hop queries.", value=True)
chk4 = st.checkbox("I know how to design a tiered query analysis cascade using sub-millisecond regex heuristics and lightweight models.", value=True)
chk5 = st.checkbox("I understand how query classification enforces Role-Based Access Control (RBAC) and prevents cross-domain contamination.", value=True)
chk6 = st.checkbox("I know how to intercept underspecified queries with interactive clarification dialogs.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Phenomenal! You have mastered Query Understanding and Cognitive Pre-Retrieval architectures.")
