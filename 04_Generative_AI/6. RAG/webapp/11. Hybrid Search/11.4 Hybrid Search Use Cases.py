import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class DomainAdaptiveHybridRouter:
    """
    Simulates domain-specific hybrid search routing across 5 enterprise verticals:
    Product Search, Enterprise Search, Legal Search, Technical Docs, and E-Commerce.
    Demonstrates dynamic Alpha calibration, metadata gating, and domain failure modes.
    """
    DOMAIN_CONFIGS = {
        "Product Search (Catalog & Hardware)": {
            "default_alpha": 0.25,
            "lexical_role": "Primary (Matches exact component SKUs, dimensions, and tolerances)",
            "dense_role": "Secondary (Captures functional equivalence and category tags)",
            "failure_pure_dense": "Recommends incompatible hardware components due to shared topical category."
        },
        "Enterprise Search (Intranet & HR)": {
            "default_alpha": 0.80,
            "lexical_role": "Secondary (Matches specific department codes, acronyms, and names)",
            "dense_role": "Primary (Resolves conversational intent, synonymy, and ambiguous policies)",
            "failure_pure_dense": "Misses specific form IDs or exact employee names."
        },
        "Legal Search (Statutes & Case Law)": {
            "default_alpha": 0.50,
            "lexical_role": "Equal Partner (Mandatory for exact section numbers, e.g. 42 U.S.C. 1983)",
            "dense_role": "Equal Partner (Mandatory for finding precedent analogies across court opinions)",
            "failure_pure_dense": "Hallucinates or omits mandatory statutory citations."
        },
        "Technical Documentation (APIs & DevOps)": {
            "default_alpha": 0.40,
            "lexical_role": "Primary (Pinpoints exact function names, stack traces, and error codes)",
            "dense_role": "Secondary (Identifies high-level debugging tutorials and architecture patterns)",
            "failure_pure_dense": "Returns generic networking tutorials instead of the specific bug fix."
        },
        "E-Commerce Search (Consumer Retail)": {
            "default_alpha": 0.35,
            "lexical_role": "Primary (Locks onto brand names, model lines, sizes, and attributes)",
            "dense_role": "Secondary (Handles subjective discovery, e.g. 'comfy lightweight summer shoes')",
            "failure_pure_dense": "Displays competitor brands when the user explicitly queried a specific brand."
        }
    }

    MULTI_DOMAIN_CORPUS = [
        {
            "id": "prod_01",
            "domain": "Product Search (Catalog & Hardware)",
            "title": "Industrial High-Torque Stepper Motor NEMA-23-01",
            "text": "Part SKU NEMA-23-01. 2.8A current rating, 1.8 degree step angle, dual-shaft CNC motor.",
            "lex_score": 19.8,
            "dense_score": 0.72
        },
        {
            "id": "prod_02",
            "domain": "Product Search (Catalog & Hardware)",
            "title": "Universal CNC Servo Motor Driver Module",
            "text": "Brushless motor driver controller compatible with NEMA frames and digital pulse control.",
            "lex_score": 4.5,
            "dense_score": 0.85
        },
        {
            "id": "ent_01",
            "domain": "Enterprise Search (Intranet & HR)",
            "title": "Parental Leave & Family Bonding Policy 2026",
            "text": "Full-time employees receive 16 weeks of fully paid family leave following birth or adoption.",
            "lex_score": 3.1,
            "dense_score": 0.94
        },
        {
            "id": "ent_02",
            "domain": "Enterprise Search (Intranet & HR)",
            "title": "Form HR-401B: Paid Time Off Carryover Request",
            "text": "Submit Form HR-401B before December 1st to roll over up to 5 unused PTO days into Q1.",
            "lex_score": 15.2,
            "dense_score": 0.75
        },
        {
            "id": "leg_01",
            "domain": "Legal Search (Statutes & Case Law)",
            "title": "42 U.S.C. Section 1983 - Civil Rights Deprivation",
            "text": "Statutory cause of action for deprivation of rights under color of state law, qualified immunity standards.",
            "lex_score": 18.9,
            "dense_score": 0.81
        },
        {
            "id": "leg_02",
            "domain": "Legal Search (Statutes & Case Law)",
            "title": "Harlow v. Fitzgerald Precedent Analysis",
            "text": "Supreme Court doctrine establishing objective reasonableness test for government official immunity.",
            "lex_score": 5.4,
            "dense_score": 0.89
        },
        {
            "id": "tech_01",
            "domain": "Technical Documentation (APIs & DevOps)",
            "title": "Kafka Consumer Poll NullPointerException Bugfix",
            "text": "Fixing NullPointerException when KafkaConsumer.poll() is called after heartbeater thread exit.",
            "lex_score": 21.4,
            "dense_score": 0.78
        },
        {
            "id": "tech_02",
            "domain": "Technical Documentation (APIs & DevOps)",
            "title": "Event-Driven Microservices with Distributed Streaming",
            "text": "Best practices for building fault-tolerant stream processing architectures with consumer groups.",
            "lex_score": 3.8,
            "dense_score": 0.91
        },
        {
            "id": "ecom_01",
            "domain": "E-Commerce Search (Consumer Retail)",
            "title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
            "text": "Flagship over-ear Bluetooth headphones with Auto NC Optimizer, 30-hour battery, silver edition.",
            "lex_score": 17.6,
            "dense_score": 0.80
        },
        {
            "id": "ecom_02",
            "domain": "E-Commerce Search (Consumer Retail)",
            "title": "Bose QuietComfort Ultra Wireless Headphones",
            "text": "Premium active noise cancellation over-ear headphones with immersive spatial audio, black.",
            "lex_score": 2.2,
            "dense_score": 0.88
        }
    ]

    @classmethod
    def execute_domain_search(
        cls,
        domain_name: str,
        alpha_override: Optional[float] = None,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        config = cls.DOMAIN_CONFIGS[domain_name]
        alpha = config["default_alpha"] if alpha_override is None else alpha_override

        candidates = [d for d in cls.MULTI_DOMAIN_CORPUS if d["domain"] == domain_name]

        # Min-max normalization for lexical
        lex_scores = [d["lex_score"] for d in candidates]
        min_lex, max_lex = min(lex_scores), max(lex_scores)
        span_lex = max_lex - min_lex if max_lex > min_lex else 1.0

        # Min-max normalization for dense
        dense_scores = [d["dense_score"] for d in candidates]
        min_dense, max_dense = min(dense_scores), max(dense_scores)
        span_dense = max_dense - min_dense if max_dense > min_dense else 1.0

        results = []
        for d in candidates:
            norm_lex = (d["lex_score"] - min_lex) / span_lex
            norm_dense = (d["dense_score"] - min_dense) / span_dense
            fused_score = ((1.0 - alpha) * norm_lex) + (alpha * norm_dense)
            results.append({
                "id": d["id"],
                "title": d["title"],
                "text": d["text"],
                "fused_score": round(fused_score, 4),
                "norm_lexical": round(norm_lex, 4),
                "norm_dense": round(norm_dense, 4),
                "raw_lex": d["lex_score"],
                "raw_dense": d["dense_score"]
            })

        results.sort(key=lambda x: x["fused_score"], reverse=True)
        return results[:top_k]

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="11.4 Hybrid Search Use Cases - RAG Master Curriculum",
    page_icon="🏢",
    layout="wide"
)

st.title("11.4 Hybrid Search Use Cases: Product, Enterprise, Legal, Tech Docs, & E-Commerce")
st.caption("Chapter 11: Hybrid Search | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Hybrid search is not a static one-size-fits-all formula. 
Enforcing a single fixed weighting across an entire enterprise platform fails because different domains—and 
distinct query intents within the same application—operate on vastly different information retrieval principles:

* In **Product Catalog & E-Commerce Search**, buyers search with exact part SKUs (*"NEMA-23-01"*), brands (*"Sony WH-1000XM5"*), 
  and dimensions. Lexical exact matching must lead; an overly aggressive semantic search risks showing competitor brands.
* In **Legal Discovery**, lawyers demand exact statutory citations (*"42 U.S.C. 1983"*) combined with conceptual precedent reasoning.
* In **Technical Documentation**, developers paste raw stack traces (*"KafkaConsumer.poll NullPointerException"*), 
  where missing the exact method name renders the answer useless.
* In **Enterprise Intranet Search**, employees ask fuzzy conversational questions (*"Can I take time off to care for my newborn?"*), 
  where dense semantic understanding must dominate.

This lesson explores the operational realities of hybrid search across all five core business verticals, 
highlighting why domain-adaptive routing is mandatory for enterprise RAG.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Domain-Specific Hybrid Search Routing Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    domain_bp_dot = """
    digraph DomainHybridBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_ingress {
            label = "1. Query Ingress & Domain Intent Classifier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            user_query [label="Incoming Search Query\\nRegex + Token Inspection + Domain Context", fillcolor="#0F172A", color="#38BDF8"];
            classifier [label="Adaptive Query Intent Router\\nClassifies: SKU, Statute, Stack Trace, or Natural Lang", fillcolor="#334155", color="#10B981"];

            user_query -> classifier [label="Inspect Tokens"];
        }

        subgraph cluster_domains {
            label = "2. Domain Routing & Dynamic Alpha Calibration";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            route_prod [label="Product Catalog\\nAlpha = 0.25 (Lexical Heavy)\\nExact SKU & Hardware Gating", fillcolor="#0F172A", color="#F59E0B"];
            route_ent [label="Enterprise Intranet\\nAlpha = 0.80 (Dense Heavy)\\nHigh Synonym & Intent Tolerance", fillcolor="#0F172A", color="#8B5CF6"];
            route_leg [label="Legal Discovery\\nAlpha = 0.50 (Balanced)\\nExact Statute + Precedent Analogy", fillcolor="#0F172A", color="#EC4899"];
            route_tech [label="Technical Documentation\\nAlpha = 0.40 (Lexical Tilt)\\nMethod Signatures + Stack Traces", fillcolor="#0F172A", color="#06B6D4"];
            route_ecom [label="E-Commerce Retail\\nAlpha = 0.35 (Lexical Tilt)\\nBrand Protection + Fuzzy Discovery", fillcolor="#0F172A", color="#10B981"];

            classifier -> route_prod [label="SKU / Spec Match"];
            classifier -> route_ent [label="Conversational HR"];
            classifier -> route_leg [label="Citation Pattern"];
            classifier -> route_tech [label="Code / Error Stack"];
            classifier -> route_ecom [label="Brand / Shopping"];
        }

        subgraph cluster_execution {
            label = "3. Parallel Execution & Domain-Constrained Fusion";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            exec_engines [label="Concurrent Inverted Index + Vector DB\\nMetadata Filter: domain_scope == target", fillcolor="#334155", color="#38BDF8"];
            fusion_node [label="Domain-Weighted Fusion Layer\\nScore = (1 - alpha)*Norm_BM25 + alpha*Norm_Dense", fillcolor="#1E293B", color="#EC4899"];
            rag_output [label="Domain-Accurate Top-K Candidates\\nOptimized Context Payload for Generator", fillcolor="#0F172A", color="#10B981"];

            route_prod -> exec_engines;
            route_ent -> exec_engines;
            route_leg -> exec_engines;
            route_tech -> exec_engines;
            route_ecom -> exec_engines;

            exec_engines -> fusion_node [label="Dual Candidate Streams"];
            fusion_node -> rag_output [label="Ranked Fusion"];
        }
    }
    """
    st.graphviz_chart(domain_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production RAG systems (e.g., enterprise portals spanning Jira, GitHub, Slack, Confluence, and ERP databases), 
a unified hybrid pipeline dynamically inspects query characteristics before assigning alpha weights. 
If an exact regular expression matches an issue key (`PROD-1024`) or legal code (`18 U.S.C. 1030`), 
the router shifts weight to lexical BM25 (`alpha <= 0.20`). If the query is an open-ended natural language sentence, 
the router shifts weight to dense vector embeddings (`alpha >= 0.75`).
""")

st.divider()

# ==============================================================================
# SECTION 2: PRODUCT SEARCH
# ==============================================================================

st.header("2. Product Search: Industrial Catalogs, Hardware Specifications, and SKU Precision", divider="rainbow")

st.markdown("""
In enterprise product search (e.g., industrial machinery, electronic components, aerospace supply chains), 
queries prioritize exact specifications, model tolerances, and part identifiers over fuzzy topical meaning.
""")

col_prod1, col_prod2 = st.columns([1, 1])

with col_prod1:
    st.subheader("Operational Realities of Product Search")
    st.markdown("""
    * **SKU and Model Specificity**:
      * A technician searching for *"NEMA-23-01 2.8A stepper motor"* must not be shown a *"NEMA-17"* motor, 
        even though they share 90% semantic embedding proximity as stepper motors!
    * **Recommended Hybrid Configuration**:
      * **Alpha (Dense Weight)**: `0.20 - 0.30` (Heavily Lexical Dominant).
      * **Filtering**: Strict hard metadata filters on dimensions, voltage, and manufacturer IDs.
    * **The Pure Semantic Disaster**:
      * Pure vector search maps all electric motors into an undifferentiated geometric cluster. 
        It regularly recommends incompatible voltage ratings or wrong frame dimensions, causing hardware damage.
    """)

with col_prod2:
    st.subheader("Product Search Engineering Matrix")
    prod_matrix = pd.DataFrame([
        {"Attribute": "Query Style", "Specification": "Exact Part Number + Specification (e.g. 'M8x1.25 titanium bolt')"},
        {"Attribute": "Lexical Role", "Specification": "Hard keyword match on part numbers, thread pitches, and SKU codes"},
        {"Attribute": "Dense Role", "Specification": "Fallback for generic functional categories (e.g. 'anti-vibration fastener')"},
        {"Attribute": "Critical Failure", "Specification": "Returning structurally incompatible parts with identical descriptions"}
    ])
    st.dataframe(prod_matrix, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: ENTERPRISE SEARCH
# ==============================================================================

st.header("3. Enterprise Search: Intranet Portals, HR Policies, and Internal Knowledge Bases", divider="rainbow")

st.markdown("""
Enterprise internal search spans Confluence pages, Google Drive docs, Slack threads, and HR benefit guides. 
Here, users rarely know the official administrative titles of documents, requiring maximum semantic tolerance.
""")

col_ent1, col_ent2 = st.columns([1, 1])

with col_ent1:
    st.subheader("Operational Realities of Enterprise Search")
    st.markdown("""
    * **The Vocabulary Disconnect**:
      * Employees search with informal phrases (*"taking days off for my new baby"*, *"how do I roll over vacation days?"*).
      * Official policy manuals use formal legalistic terminology (*"Parental Leave and Primary Caregiver Benefit"*, *"Form HR-401B Paid Time Off Carryover"*).
    * **Recommended Hybrid Configuration**:
      * **Alpha (Dense Weight)**: `0.75 - 0.85` (Heavily Semantic Dominant).
      * **Lexical Reserve (15% - 25%)**: Retained to ensure exact matches when employees search for specific form IDs (*"Form W-2"*, *"Form HR-401B"*), team acronyms (*"SRE"*, *"GTM"*), or colleague names.
    * **The Pure Lexical Disaster**:
      * Pure BM25 returns zero results for *"maternity leave"* if the company's document is titled *"Parental Benefits"*.
    """)

with col_ent2:
    st.subheader("Enterprise Search Engineering Matrix")
    ent_matrix = pd.DataFrame([
        {"Attribute": "Query Style", "Specification": "Conversational Natural Language (e.g. 'Can I expense my home monitor?')"},
        {"Attribute": "Lexical Role", "Specification": "Exact matching for policy form IDs, employee names, and corporate acronyms"},
        {"Attribute": "Dense Role", "Specification": "Bridging informal conversational intent to formal legal HR guidelines"},
        {"Attribute": "Critical Failure", "Specification": "Zero results returned due to minor phrasing differences with policy title"}
    ])
    st.dataframe(ent_matrix, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: LEGAL SEARCH
# ==============================================================================

st.header("4. Legal Search: Statutory Precision, Citation Codes, and Precedent Analogies", divider="rainbow")

st.markdown("""
Legal discovery and case law retrieval represent the most demanding test of hybrid retrieval. 
Attorneys require literal statutory citation precision alongside conceptual reasoning across legal opinions.
""")

col_leg1, col_leg2 = st.columns([1, 1])

with col_leg1:
    st.subheader("Operational Realities of Legal Search")
    st.markdown("""
    * **Dual Information Requirements**:
      * **Lexical Mandate**: Exact statutory citations (*"42 U.S.C. 1983"*, *"Fed. R. Civ. P. 12(b)(6)"*). 
        If a search engine misses the exact statute number, the retrieval is completely invalid.
      * **Semantic Mandate**: Finding case opinions that share abstract legal reasoning 
        (*"government official acting with deliberate indifference"*) without sharing identical case facts.
    * **Recommended Hybrid Configuration**:
      * **Alpha (Dense Weight)**: `0.45 - 0.55` (Equal Balance).
      * **Two-Stage Execution**: Lexical filters isolate cases citing the governing statute; 
        dense vectors rank those cases by factual and conceptual analogy to the current brief.
    * **The Pure Semantic Disaster**:
      * Pure dense embeddings hallucinate adjacent statutory sections (e.g., confusing Section 1983 with Section 1985 
        because both discuss civil rights conspiracies in similar prose).
    """)

with col_leg2:
    st.subheader("Legal Search Engineering Matrix")
    leg_matrix = pd.DataFrame([
        {"Attribute": "Query Style", "Specification": "Statute Code + Conceptual Precedent (e.g. '18 U.S.C. 1030 exceeding authorization')"},
        {"Attribute": "Lexical Role", "Specification": "Hard gating on specific case citations, statutory codes, and docket numbers"},
        {"Attribute": "Dense Role", "Specification": "Mapping abstract legal arguments, procedural postures, and factual analogies"},
        {"Attribute": "Critical Failure", "Specification": "Retrieving cases from irrelevant statutes that share general legal jargon"}
    ])
    st.dataframe(leg_matrix, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: TECHNICAL DOCUMENTATION SEARCH
# ==============================================================================

st.header("5. Technical Documentation Search: APIs, Error Stack Traces, and Runbooks", divider="rainbow")

st.markdown("""
Software documentation and site reliability runbooks combine structured alphanumeric syntax 
(method names, HTTP status codes, error messages) with architectural concepts.
""")

col_tech1, col_tech2 = st.columns([1, 1])

with col_tech1:
    st.subheader("Operational Realities of Technical Docs Search")
    st.markdown("""
    * **Alphanumeric Code Matching**:
      * Engineers search using raw compiler errors (*"NullPointerException in KafkaConsumer.poll"*), 
        hexadecimal memory offsets, or CLI flags (*"--max-request-size"*).
    * **Recommended Hybrid Configuration**:
      * **Alpha (Dense Weight)**: `0.35 - 0.45` (Balanced with Lexical Tilt).
      * **Tokenizer Customization**: Inverted index tokenizers must preserve underscores, periods, and camelCase 
        (`KafkaConsumer.poll` must not be stripped into separate words).
    * **The Pure Semantic Disaster**:
      * A developer searching for a specific Kafka crash gets general tutorials on *"Introduction to Event Streaming"* 
        because the embedding model clusters all Kafka documents together without isolating the specific crashed method.
    """)

with col_tech2:
    st.subheader("Technical Documentation Engineering Matrix")
    tech_matrix = pd.DataFrame([
        {"Attribute": "Query Style", "Specification": "Code Identifier + Error String (e.g. 'Postgres deadlock PgBouncer pool')"},
        {"Attribute": "Lexical Role", "Specification": "Exact match on method signatures, stack traces, library versions, and CLI flags"},
        {"Attribute": "Dense Role", "Specification": "Retrieving conceptual troubleshooting flows, architecture guides, and root causes"},
        {"Attribute": "Critical Failure", "Specification": "Returning 101-level intro tutorials instead of the specific bug fix"}
    ])
    st.dataframe(tech_matrix, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: E-COMMERCE SEARCH
# ==============================================================================

st.header("6. E-Commerce Search: Consumer Retail, Brand Integrity, and Fuzzy Intent", divider="rainbow")

st.markdown("""
In consumer e-commerce (e.g., Amazon, Shopify, retail marketplaces), search directly impacts revenue. 
Search engines must protect strict brand intent while accommodating fuzzy consumer inquiries.
""")

col_ecom1, col_ecom2 = st.columns([1, 1])

with col_ecom1:
    st.subheader("Operational Realities of E-Commerce Search")
    st.markdown("""
    * **The Brand Cannibalization Risk**:
      * When a consumer searches for *"Sony WH-1000XM5 headphones"*, they must see Sony headphones. 
        If pure semantic search displays Bose QuietComfort headphones at the top because they share 
        92% conceptual similarity as noise-canceling headphones, consumer trust evaporates.
    * **Fuzzy Subjective Discovery**:
      * When a consumer searches for *"comfortable lightweight shoes for summer walking"*, 
        no specific brand or model is requested. Here, dense semantic embeddings must lead the ranking.
    * **Recommended Hybrid Configuration**:
      * **Dynamic Dual-Mode Weighting**:
        * **Identified Brand / Product Line**: Set `alpha = 0.20` (Strict Lexical Dominant).
        * **Open Discovery Query**: Set `alpha = 0.70` (Semantic Dominant).
    """)

with col_ecom2:
    st.subheader("E-Commerce Search Engineering Matrix")
    ecom_matrix = pd.DataFrame([
        {"Attribute": "Query Style", "Specification": "Brand + Product Class + Attribute (e.g. 'Apple iPhone 15 Pro Max 256GB')"},
        {"Attribute": "Lexical Role", "Specification": "Strict protection of brand names, product generations, colors, and storage tiers"},
        {"Attribute": "Dense Role", "Specification": "Resolving fuzzy subjective descriptors (e.g. 'breathable gym apparel')"},
        {"Attribute": "Critical Failure", "Specification": "Recommending competitor brands when an exact trademark was searched"}
    ])
    st.dataframe(ecom_matrix, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Domain-Adaptive Hybrid Retrieval Router", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python module that inspects incoming query patterns, automatically identifies 
the domain intent, calibrates alpha weights dynamically, and executes domain-constrained hybrid retrieval.
""")

st.code(r'''
import re
from typing import List, Dict, Any, Optional

class DomainAdaptiveHybridRouter:
    """
    Production-grade query router that inspects query tokens,
    infers the target business domain, and dynamically calibrates
    the hybrid alpha weighting between lexical BM25 and dense vectors.
    """
    # Regex patterns for high-precision entity detection
    SKU_PATTERN = re.compile(r"^[A-Z0-9]{3,}-[A-Z0-9]{2,}", re.IGNORECASE)
    STATUTE_PATTERN = re.compile(r"\b\d+\s+U\.S\.C\.\s+\d+", re.IGNORECASE)
    STACK_TRACE_PATTERN = re.compile(r"([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+|\b(Exception|Error|Fatal)\b)")
    BRAND_LIST = {"sony", "apple", "bose", "nike", "samsung", "dell"}

    @classmethod
    def analyze_query_intent(cls, query: str) -> Dict[str, Any]:
        """
        Analyzes query string to determine domain classification and optimal alpha.
        """
        q_clean = query.strip()
        tokens = [t.lower() for t in q_clean.split()]

        # 1. Check for statutory citation
        if cls.STATUTE_PATTERN.search(q_clean):
            return {
                "inferred_domain": "Legal Search",
                "recommended_alpha": 0.50,
                "reason": "Statutory citation detected. Requires 50/50 balance between citation and precedent."
            }

        # 2. Check for technical stack trace or method signature
        if cls.STACK_TRACE_PATTERN.search(q_clean):
            return {
                "inferred_domain": "Technical Docs",
                "recommended_alpha": 0.40,
                "reason": "Code identifier or error trace detected. Lexical BM25 prioritized for exact token match."
            }

        # 3. Check for industrial part SKU
        for token in tokens:
            if cls.SKU_PATTERN.match(token):
                return {
                    "inferred_domain": "Product Search",
                    "recommended_alpha": 0.25,
                    "reason": "Part SKU detected. Heavily prioritizing BM25 exact match to prevent hardware mismatch."
                }

        # 4. Check for retail brand name
        if any(b in tokens for b in cls.BRAND_LIST):
            return {
                "inferred_domain": "E-Commerce",
                "recommended_alpha": 0.30,
                "reason": "Specific brand detected. Enforcing lexical dominance to protect brand integrity."
            }

        # 5. Default to conversational enterprise search
        return {
            "inferred_domain": "Enterprise Search",
            "recommended_alpha": 0.80,
            "reason": "Natural language inquiry detected. Prioritizing dense semantic understanding."
        }

    @classmethod
    def execute_adaptive_search(
        cls,
        query: str,
        lexical_hits: List[Dict[str, Any]],
        dense_hits: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Calculates domain-adaptive hybrid scores using dynamically selected alpha.
        """
        intent = cls.analyze_query_intent(query)
        alpha = intent["recommended_alpha"]

        # Extract doc IDs across both pools
        all_ids = set([x["id"] for x in lexical_hits] + [x["id"] for x in dense_hits])
        lex_map = {x["id"]: x.get("score", 0.0) for x in lexical_hits}
        dense_map = {x["id"]: x.get("score", 0.0) for x in dense_hits}

        # Min-max scaling
        min_l, max_l = min(lex_map.values(), default=0.0), max(lex_map.values(), default=1.0)
        span_l = max_l - min_l if max_l > min_l else 1.0

        min_d, max_d = min(dense_map.values(), default=0.0), max(dense_map.values(), default=1.0)
        span_d = max_d - min_d if max_d > min_d else 1.0

        scored_candidates = []
        for d_id in all_ids:
            norm_l = (lex_map.get(d_id, 0.0) - min_l) / span_l
            norm_d = (dense_map.get(d_id, 0.0) - min_d) / span_d
            fused = ((1.0 - alpha) * norm_l) + (alpha * norm_d)
            scored_candidates.append({
                "id": d_id,
                "fused_score": round(fused, 4),
                "norm_lexical": round(norm_l, 4),
                "norm_dense": round(norm_d, 4)
            })

        scored_candidates.sort(key=lambda x: x["fused_score"], reverse=True)
        return {
            "intent": intent,
            "top_candidates": scored_candidates[:top_k]
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Cross-Domain Hybrid Search Simulator", divider="rainbow")

st.markdown("""
Select any of the five enterprise domains below. Observe how adjusting the **Dense Alpha Weight** 
dramatically alters the champion document retrieved for that specific industry problem.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    selected_domain = st.selectbox(
        "Select Enterprise Domain Vertical:",
        list(DomainAdaptiveHybridRouter.DOMAIN_CONFIGS.keys())
    )
    domain_meta = DomainAdaptiveHybridRouter.DOMAIN_CONFIGS[selected_domain]
    default_a = domain_meta["default_alpha"]

    use_manual_alpha = st.toggle("Override Recommended Alpha", value=False)
    if use_manual_alpha:
        active_alpha = st.slider(
            "Custom Alpha (0.0 = Pure BM25, 1.0 = Pure Dense):",
            min_value=0.0,
            max_value=1.0,
            value=default_a,
            step=0.05
        )
    else:
        active_alpha = default_a
        st.info(f"**Recommended Production Alpha**: `{default_a}`")

    top_k_domain = st.slider("Candidates to Return:", min_value=1, max_value=2, value=2)

with wb_col2:
    st.markdown("### 🏢 Domain Characteristics & Failure Warning")
    st.markdown(f"* **Lexical Role**: {domain_meta['lexical_role']}")
    st.markdown(f"* **Dense Role**: {domain_meta['dense_role']}")
    st.error(f"**Critical Failure Mode if Pure Semantic (Alpha = 1.0)**:\n{domain_meta['failure_pure_dense']}")

domain_results = DomainAdaptiveHybridRouter.execute_domain_search(
    domain_name=selected_domain,
    alpha_override=active_alpha,
    top_k=top_k_domain
)

st.markdown("### 📊 Live Ranked Results for Selected Domain")
if domain_results:
    res_df = pd.DataFrame(domain_results)
    st.dataframe(res_df[["id", "title", "fused_score", "norm_lexical", "norm_dense", "raw_lex", "raw_dense", "text"]], use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why does using a static global alpha parameter fail across multi-tenant enterprise RAG platforms?"):
    st.markdown("""
    **Answer**: Enterprise platforms ingest disparate knowledge sources simultaneously, including GitHub repositories, 
    Zendesk tickets, HR portals, and financial accounting databases. A global alpha of `0.60` is suboptimal everywhere: 
    it is too semantic for developer bug searches (where exact method names like `KafkaConsumer.poll` require `alpha <= 0.35`), 
    yet too lexical for conversational HR inquiries (where fuzzy questions like *"maternity benefits"* require `alpha >= 0.80`). 
    Production platforms must classify query intent dynamically and assign domain-specific alpha weights on a per-query basis.
    """)

with st.expander("Q2: In consumer e-commerce search, what is brand cannibalization and how does hybrid retrieval prevent it?"):
    st.markdown("""
    **Answer**: Brand cannibalization occurs when a user queries a specific trademarked brand (e.g., *"Sony noise cancelling headphones"*), 
    but a pure dense semantic vector engine returns a competitor brand (e.g., *"Bose QuietComfort"*) at rank #1 because both products share 
    95% topical vector proximity in the audio category. 
    Hybrid retrieval prevents this by using entity recognition to detect brand tokens, subsequently enforcing strict lexical BM25 weighting 
    (`alpha = 0.20`) or applying hard metadata filters (`brand == 'Sony'`) to preserve commercial integrity.
    """)

with st.expander("Q3: How should a hybrid retrieval system handle alphanumeric codes containing punctuation (e.g. 42 U.S.C. 1983)?"):
    st.markdown("""
    **Answer**: Standard NLP tokenizers often strip punctuation, breaking statutory codes like `42 U.S.C. 1983` into four isolated tokens 
    (`"42"`, `"u"`, `"s"`, `"c"`, `"1983"`), which destroys inverse document frequency specificity. 
    Production legal hybrid systems employ specialized tokenizers (such as Lucene PatternCapture or regex token filters) that preserve 
    the full citation as a single inviolable token while concurrently generating sub-tokens, ensuring 100% precision in the lexical index.
    """)

with st.expander("Q4: Why is hard metadata filtering superior to soft semantic score weighting in industrial product catalogs?"):
    st.markdown("""
    **Answer**: In industrial supply catalogs, engineering constraints are non-negotiable boolean conditions: a motor is either 24V or 110V, 
    and a bolt is either M8 or M10. Soft semantic score weighting allows a 110V motor with a high semantic similarity score to outrank 
    a 24V motor if the user's phrasing matches closely. 
    Therefore, industrial systems employ **hard pre-filtering** (`WHERE voltage == 24 AND thread == 'M8'`) to constrain the candidate pool 
    before executing hybrid lexical-semantic ranking.
    """)

with st.expander("Q5: What are the latency implications of implementing query intent classification before hybrid retrieval?"):
    st.markdown("""
    **Answer**: Introducing an LLM forward pass to classify query intent adds 150–400ms of latency, which is unacceptable for sub-50ms search APIs. 
    Production systems solve this by using ultra-fast, deterministic rule engines: compiled regular expressions (for SKUs, error codes, and statutes), 
    in-memory Trie dictionaries (for brand and product entity matching), or lightweight shallow classifiers (FastText / Logistic Regression) 
    that execute in under 1 millisecond on CPU before dispatching the hybrid query.
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Stripping Punctuation in Technical Stack Trace Indexing**
    * **Symptom**: Searching for `KafkaConsumer.poll()` or `ERR_FATAL_007` returns thousands of irrelevant documents mentioning the word 'poll'.
    * **Root Cause**: The lexical inverted index applies standard whitespace and punctuation stripping, destroying the method signature and error token.
    * **Remedy**: Configure custom Lucene / Tantivy analyzers with code-aware tokenization that preserves dots, underscores, colons, and camelCase.
    """)

    st.error("""
    **Trap 2: Relying on Dense Semantic Search for Legal Citation Matching**
    * **Symptom**: Legal RAG assistant cites Section 1985 instead of Section 1983 in legal briefs.
    * **Root Cause**: Subword tokenizers fragment statutory numbers into generic digits. Dense vectors treat Section 1983 and 1985 as virtually identical civil rights topics.
    * **Remedy**: Never rely on dense vectors alone for legal citations. Enforce 50% BM25 weighting with exact string citation boosting.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Permitting Brand Cannibalization in Consumer Retail Search**
    * **Symptom**: Searching for a specific luxury brand displays competitor brand products on the first page, resulting in merchant complaints and lost sales.
    * **Root Cause**: Overly dense semantic search (`alpha >= 0.70`) prioritizes product category over explicit brand trademark intent.
    * **Remedy**: Detect brand entities during query ingress and dynamically suppress competitor products using hard brand filtering or lowering alpha to `<= 0.25`.
    """)

    st.warning("""
    **Trap 4: Using Heavyweight LLMs for Real-Time Query Classification**
    * **Symptom**: User-perceived search latency balloons from 30ms to 500ms after deploying dynamic query routing.
    * **Root Cause**: Calling an external LLM API to determine whether the query is a SKU, an HR question, or a legal statute on every keystroke.
    * **Remedy**: Replace LLM classifiers with sub-millisecond regex rules, in-memory brand Tries, and lightweight FastText classifiers.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: Why does Product Catalog Search require a heavily lexical hybrid weighting (alpha = 0.25)?")
        st.markdown("**Back**: Because industrial parts rely on exact model numbers, dimensions, and electrical tolerances. Dense vectors cluster all motors together, risking fatal hardware mismatch.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: Why does Enterprise HR Search require a heavily semantic hybrid weighting (alpha = 0.80)?")
            st.markdown("**Back**: Employees search using conversational terms ('days off for baby') that share zero vocabulary with official policy titles ('Parental Caregiver Leave'). Semantic models bridge this lexical divide.")

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Brand Cannibalization in E-Commerce Search?")
        st.markdown("**Back**: A failure mode where dense semantic search returns competitor products instead of the exact brand queried by the shopper due to shared product category vector proximity.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: How should query intent classification be implemented to maintain sub-50ms search latency?")
        st.markdown("**Back**: Using compiled regular expressions, in-memory Trie brand lookups, or sub-millisecond FastText models on CPU rather than slow external LLM inference.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Domain-Specific Hybrid Search, Vertical Calibration, and Dynamic Intent Routing:
""")

chk1 = st.checkbox("I understand why static global alpha weighting fails across multi-tenant enterprise platforms.", value=True)
chk2 = st.checkbox("I can explain why industrial product search requires heavy lexical dominance (alpha <= 0.30) to prevent spec mismatch.", value=True)
chk3 = st.checkbox("I understand how enterprise HR search requires heavy semantic dominance (alpha >= 0.75) to bridge vocabulary mismatches.", value=True)
chk4 = st.checkbox("I know how brand cannibalization occurs in e-commerce search and how hybrid routing preserves brand integrity.", value=True)
chk5 = st.checkbox("I recognize why legal search demands balanced 50/50 hybrid retrieval to satisfy both statutory citations and case analogies.", value=True)
chk6 = st.checkbox("I understand why query intent routing must use fast regex and Trie rules rather than high-latency LLM calls.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Magnificent! You have mastered Domain-Specific Hybrid Search architectures across all major industry verticals.")
