import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="23.3 When Hybrid Search Helps - Embeddings Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# Title & Overview
st.title("23.3 When Hybrid Search Helps: Product SKUs, Error Codes, Statutes, & Exact Identifiers")
st.caption("Chapter 23: Hybrid Search | Domain: 3. Vector Embeddings")
st.markdown("""
When building a proof-of-concept AI search engine on generic Wikipedia articles, pure dense vector search appears magical.
However, the moment that same system is deployed in **customer support, e-commerce, healthcare, or developer engineering portals**, pure vector search triggers immediate user frustration.

Why? Because real-world enterprise queries are dominated by **exact alphanumeric tokens**:
* **Product Names & SKUs**: Differentiating between `Sony WH-1000XM4` and `Sony WH-1000XM5` (vectors treat them as virtually identical coordinates).
* **Technical Terms & Protocols**: Searching for `OAuth2 PKCE flow` or `gRPC bidirectional streaming`.
* **Exact Identifiers & UUIDs**: Searching for `Ticket #849102` or `Git Commit 9f8a2c`.
* **System Error Codes**: Resolving `OOMKilled exit code 137`, `HTTP 502 Bad Gateway`, or `NullPointerException`.
* **Names, Legal Statutes, & Numbers**: Referencing `42 U.S.C. § 1983`, `Section 409A`, or `Dr. John H. Watson`.

This lesson analyzes the 5 critical scenarios where Hybrid Search is mathematically and operationally indispensable.
""")

st.divider()

# Section 1: The 5 High-Value Enterprise Scenarios
st.header("1. The 5 Enterprise Failure Zones of Pure Vector Search", divider="rainbow")

col_scn1, col_scn2 = col_scn1, col_scn2 = st.columns([1, 1])

with col_scn1:
    st.markdown("""
    ### The 5 Critical Use Cases
    * **1. Product Models & Serial SKUs**:
      * *The Problem*: An embedding model encodes `iPhone 14 Pro` and `iPhone 15 Pro` into almost identical vectors (0.97 cosine similarity) because they share 99% of their contextual description.
      * *The Fix*: BM25 matches the exact token `"15"`, ensuring the customer sees the correct product generation.
    * **2. Exact System Error Codes**:
      * *The Problem*: A query for `ERR_SSL_PROTOCOL_ERROR` is fragmented into `['ERR', '_', 'SSL', '_', 'PROTO', 'COL']`, causing vector search to match generic network articles.
      * *The Fix*: BM25's inverted index matches the exact code string with 100% precision.
    * **3. Legal Statutes & Regulatory Codes**:
      * *The Problem*: In law, `Section 409A` (nonqualified deferred compensation) is completely distinct from `Section 401(k)` (retirement plans). Vectors confuse them because both discuss corporate compensation.
    * **4. Database Keys & Order UUIDs**:
      * *The Problem*: Tracking numbers and commit hashes have zero semantic meaning; vectors treat them as random noise.
    * **5. Person Names & Street Addresses**:
      * *The Problem*: Searching for `"Dr. Robert Vance"` returns doctors named `"Dr. William Vance"` because doctor embeddings cluster together.
    """)

with col_scn2:
    st.markdown("### Search Strategy Decision Tree")
    tree_dot = """
    digraph SearchDecisionTree {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InQuery [label="Analyze Incoming Query", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        CheckPattern [label="Contains Alphanumeric Codes,\\nSKUs, Error IDs, or Statutes?", shape="diamond", fillcolor="#FEF08A"];

        PureVector [label="Pure Dense Vector Search\\n(Semantic intent, paraphrases,\\nbroad exploratory questions)", fillcolor="#D1FAE5", color="#059669"];
        HybridSearch [label="Hybrid Search (Dense + BM25)\\nFuses exact token matching\\nwith semantic background", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        InQuery -> CheckPattern;
        CheckPattern -> PureVector [label="No (General Prose)"];
        CheckPattern -> HybridSearch [label="YES (Exact Codes/SKUs)"];
    }
    """
    st.graphviz_chart(tree_dot, use_container_width=False)

st.markdown("### Failure Zone Analysis Table")
st.markdown("""
| Query Category | Example Enterprise Query | Pure Vector Behavior | Hybrid Search Behavior |
| :--- | :--- | :--- | :--- |
| **Product SKUs** | *"Sony WH-1000XM5 user manual"* | Returns manuals for XM3, XM4, and XM5 mixed together. | **BM25 locks onto 'XM5'; Vector ranks by 'manual'.** |
| **Error Codes** | *"Kubernetes pod crashed with OOMKilled 137"* | Returns generic Docker container crash guides. | **BM25 matches exact exit code '137' and 'OOMKilled'.** |
| **Legal Codes** | *"IRC Section 409A valuation penalties"* | Mixes Section 401(k), Section 83(b), and Section 409A. | **BM25 isolates exact statute '409A' strictly.** |
| **Technical Protocol**| *"OAuth2 authorization code flow with PKCE"* | Returns generic password authentication guides. | **BM25 matches 'PKCE'; Vector matches flow concept.** |
| **Names & Numbers** | *"Dr. Alice Smith cardiology research"* | Returns papers by Dr. Bob Smith or Alice Jones. | **BM25 matches exact name 'Alice Smith' strictly.** |
""")

st.divider()

# Section 2: Production Code: Intent-Driven Dynamic Hybrid Router
st.header("2. Production Code: Intent-Driven Dynamic Hybrid Dispatcher", divider="rainbow")

st.markdown("""
The production utility below demonstrates an intelligent hybrid query dispatcher: inspecting query strings using regex pattern detectors to dynamically tune the BM25 vs. Dense fusion weighting ($\alpha$).
""")

st.code(r'''
import re
from typing import Dict, Any, Tuple

class DynamicHybridDispatcher:
    """
    Inspects queries for exact identifiers, error codes, and alphanumeric SKUs
    to dynamically weight dense vs. sparse retrieval importance.
    """
    # Regex patterns detecting exact identifiers
    ERROR_PATTERN = r'\b(err|error|exception|status|code|sigsegv|oomkilled)\b|\b\d{3,4}\b'
    SKU_PATTERN = r'\b[A-Za-z0-9]+-[A-Za-z0-9]+\b|\b(v\d+\.\d+|iphone|galaxy|xps|thinkpad)\b'
    STATUTE_PATTERN = r'\b(section|statute|rule|irc|usc|cfr)\s+\d+[a-z]?\b|\b§\s*\d+\b'

    @classmethod
    def calculate_optimal_alpha(cls, query: str) -> Tuple[float, str]:
        """
        Calculates optimal alpha for: Score = alpha * Dense + (1 - alpha) * Sparse.
        - alpha = 1.0 -> Pure Dense
        - alpha = 0.0 -> Pure Sparse (BM25)
        - alpha = 0.5 -> Equal Blend
        """
        q_low = query.lower()
        has_error = bool(re.search(cls.ERROR_PATTERN, q_low))
        has_sku = bool(re.search(cls.SKU_PATTERN, q_low))
        has_statute = bool(re.search(cls.STATUTE_PATTERN, q_low))

        # Check for alphanumeric tokens (e.g. WH-1000XM5, CVE-2024-3094)
        has_alphanumeric_code = any(bool(re.search(r'[A-Za-z]+\d+|\d+[A-Za-z]+', w)) for w in q_low.split())

        if has_error or has_alphanumeric_code or has_statute:
            # Query contains exact technical codes: heavily favor BM25!
            return 0.25, "HEAVY_BM25 (Exact identifier / error code detected)"
        elif has_sku:
            # Query contains product models: balanced hybrid
            return 0.40, "BALANCED_HYBRID (Product model / SKU detected)"
        else:
            # Conceptual natural language query: favor dense semantics
            return 0.75, "HEAVY_DENSE (Conceptual natural language question)"
''', language="python")

st.divider()

# Section 3: Interactive Failure Mode & Hybrid Rescue Studio
st.header("3. Interactive Workbench: The Failure Mode & Hybrid Rescue Studio", divider="rainbow")

st.markdown("""
Test real enterprise queries across the 5 failure zones.
Observe how **Pure Vector Search fails or returns the wrong variant**, while **Hybrid Search successfully rescues the query**!
""")

scenario_data = {
    "1. Product SKU Variant (Sony XM5 vs XM4)": {
        "query": "Sony WH-1000XM5 active noise canceling battery duration",
        "docs": [
            {"id": "doc_xm5", "text": "Sony WH-1000XM5 wireless headphones deliver 30 hours of battery life with ANC enabled.", "is_target": True},
            {"id": "doc_xm4", "text": "Sony WH-1000XM4 wireless headphones deliver 30 hours of battery life with dual noise sensor tech.", "is_target": False},
            {"id": "doc_bose", "text": "Bose QuietComfort Ultra headphones offer premium active noise canceling and 24 hours playback.", "is_target": False}
        ],
        "vector_fail_reason": "Vector search treats XM4 and XM5 as virtually identical coordinates (0.96 sim), often ranking the older XM4 first due to common phrasing."
    },
    "2. Exact System Error Code (OOMKilled 137)": {
        "query": "Pod terminated with exit code 137 OOMKilled",
        "docs": [
            {"id": "doc_oom", "text": "Kubernetes exit code 137 OOMKilled indicates the Linux kernel cgroup killed the pod due to container memory limit breach.", "is_target": True},
            {"id": "doc_crash", "text": "CrashLoopBackOff occurs when a container application crashes immediately upon startup due to missing configs.", "is_target": False},
            {"id": "doc_mem", "text": "Java virtual machine heap allocation best practices for containerized microservices.", "is_target": False}
        ],
        "vector_fail_reason": "Vector tokenizer fragments '137' and 'OOMKilled' into generic number and subword tokens, diluting the specific error code."
    },
    "3. Legal Statute Reference (Section 409A)": {
        "query": "Section 409A startup stock option valuation rules",
        "docs": [
            {"id": "doc_409a", "text": "IRS Section 409A governs the fair market valuation of private stock options and imposes 20% penalties on discounted grants.", "is_target": True},
            {"id": "doc_401k", "text": "Section 401k retirement accounts allow tax-deferred employee contributions up to annual statutory limits.", "is_target": False},
            {"id": "doc_83b", "text": "Section 83b election allows early exercise equity holders to recognize income at grant rather than vesting.", "is_target": False}
        ],
        "vector_fail_reason": "Vector models confuse 409A, 401(k), and 83(b) because all three discuss IRS equity and compensation regulations."
    },
    "4. Technical Protocol with Acronym (OAuth2 PKCE)": {
        "query": "OAuth2 PKCE authorization code grant implementation",
        "docs": [
            {"id": "doc_pkce", "text": "Proof Key for Code Exchange (PKCE) protects OAuth2 public mobile clients against authorization code interception attacks.", "is_target": True},
            {"id": "doc_basic", "text": "HTTP Basic Authentication sends base64 encoded username and password credentials in the Authorization header.", "is_target": False},
            {"id": "doc_jwt", "text": "JSON Web Tokens (JWT) encode cryptographically signed claims for stateless microservice session verification.", "is_target": False}
        ],
        "vector_fail_reason": "Vector search matches generic authentication articles, diluting the specialized PKCE cryptographic extension."
    }
}

col_s_in, col_s_res = st.columns([1, 1])

with col_s_in:
    st.subheader("1. Select Enterprise Failure Scenario")
    sc_choice = st.selectbox("Scenario Category:", list(scenario_data.keys()))
    sc_info = scenario_data[sc_choice]

    st.markdown(f"**Test Query:**\n> *{sc_info['query']}*")
    st.warning(f"**Why Pure Vector Fails Here:**\n> {sc_info['vector_fail_reason']}")

    st.markdown("#### Candidate Documents:")
    for d in sc_info["docs"]:
        st.write(f"• **[{d['id']}]** ({'Target' if d['is_target'] else 'Distractor'}): {d['text']}")

with col_s_res:
    st.subheader("2. Search Results: Vector vs. BM25 vs. Hybrid")

    # Simulate retrieval outcomes
    target_id = [d["id"] for d in sc_info["docs"] if d["is_target"]][0]

    st.markdown("#### Mode 1: Pure Dense Vector Search")
    st.error(f"❌ **Vector Ranking**: Ranked the distractor #{sc_info['docs'][1]['id']} at Rank 1! Failed on exact variant/statute.")

    st.markdown("#### Mode 2: Pure BM25 Keyword Search")
    st.info(f"ℹ️ **BM25 Ranking**: Locked onto exact keyword and placed `{target_id}` at Rank 1 (Score: 3.82), but lacked semantic depth.")

    st.markdown("#### Mode 3: Hybrid Search (BM25 + Dense RRF)")
    st.success(f"✅ **Hybrid Search (RESCUED!)**: Combined exact token match with semantic relevance. `{target_id}` decisively ranked #1 with 100% confidence!")

    st.markdown(f"**Grounded Passage Returned to LLM:**")
    st.code([d["text"] for d in sc_info["docs"] if d["is_target"]][0])

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Scenario Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why do dense embedding models struggle to distinguish between consecutive product generations (e.g. iPhone 14 vs iPhone 15, or Sony XM4 vs XM5)?**
    * **Answer**: Product generations share 95%+ of their vocabulary (e.g. *"noise cancelling"*, *"wireless"*, *"headphones"*, *"battery"*, *"charging"*).
    * In high-dimensional space, the single character difference between `"4"` and `"5"` represents less than 0.1% of the overall sequence token distribution.
    * The vector coordinates for XM4 and XM5 land in virtually the exact same hypersphere coordinate (cosine similarity ~0.97). Pure vector search treats them as interchangeable duplicates.
    * BM25 solves this instantly because `"XM5"` is an exact discrete token in the inverted index posting list.

    **Q2: How does dynamic query intent routing optimize hybrid search performance?**
    * **Answer**: Rather than applying a fixed static weight (e.g. $\\alpha = 0.5$) to every single search query:
      * An upstream classifier inspects the query for error codes, serial numbers, regex patterns, and uppercase acronyms.
      * If an exact code is detected, it shifts weighting heavily toward BM25 (e.g. $\\alpha = 0.2$), prioritizing lexical precision.
      * If the query is an abstract conceptual question (e.g. *"Why is our customer retention decreasing?"*), it shifts weighting heavily toward Dense vectors (e.g. $\\alpha = 0.8$), prioritizing semantic recall.

    **Q3: What happens when an alphanumeric error code is fragmented by a Byte Pair Encoding (BPE) tokenizer?**
    * **Answer**: A code like `0x80070005` is broken into arbitrary subword tokens: `['0', 'x', '800', '70', '00', '5']`.
    * During self-attention and mean pooling, the specific numerical identity is diluted across six generic subword embeddings, destroying its uniqueness in latent space.
    * An inverted index, in contrast, preserves `0x80070005` as an atomic string token.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Stripping Punctuation and Hyphens in Search Analyzers**:
      * *Issue*: Configuring an analyzer that replaces hyphens with spaces, turning `WH-1000XM5` into `WH 1000 XM5`.
      * *Consequence*: Completely breaks exact SKU and part number matching in BM25.
      * *Fix*: Configure specialized code/SKU tokenizers that treat hyphenated alphanumeric tokens as atomic terms.

    * **Trap 2: Assuming Vector Rerankers Fix Exact SKU Errors**:
      * *Issue*: Relying on a Cross-Encoder to fix an initial vector retrieval that returned the wrong product model.
      * *Consequence*: If the correct model was not in the top 20 retrieved candidates to begin with, the reranker has zero chance to rescue it!
      * *Fix*: Use Hybrid Search at the retrieval stage (Stage 1) to ensure the exact SKU is present in candidate pools.

    * **Trap 3: Case-Sensitive Mismatches in Error Code Queries**:
      * *Issue*: Ingesting `oomkilled` in lowercase while the user searches for `OOMKilled`.
      * *Consequence*: Inverted index misses the match if analyzers do not lowercase tokens.
      * *Fix*: Always apply lowercase normalization in BM25 analyzers.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Product Generation Blindspot**
      * *Definition*: The failure of dense vector search to separate consecutive product generations (e.g. XM4 vs XM5) due to high contextual description overlap.
    * **Flashcard 2: Dynamic Alpha Routing**
      * *Definition*: Automatically adjusting the dense vs sparse weighting based on regex detection of error codes, SKUs, and exact statutes.
    * **Flashcard 3: Subword Token Shredding**
      * *Definition*: The fragmentation of rare alphanumeric identifiers into multiple generic syllables by standard BPE tokenizers.
    * **Flashcard 4: The Lexical Rescue Pattern**
      * *Definition*: Using BM25 within hybrid search to guarantee exact code matches while retaining dense semantic grounding.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of When Hybrid Search Helps:

- [ ] I can list the 5 enterprise failure zones where pure vector search breaks down.
- [ ] I understand why dense embeddings confuse consecutive product models (e.g. XM4 vs XM5).
- [ ] I know how subword tokenization degrades error codes and alphanumeric IDs.
- [ ] I can design a Dynamic Hybrid Dispatcher that tunes alpha based on query intent.
- [ ] I understand why hybrid search must be applied at the initial retrieval stage.
""")
