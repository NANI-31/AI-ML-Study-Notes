import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="22.2 Vector Database Debugging - LangSmith Master Curriculum",
    page_icon="🐛",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class VectorDBDebuggingEngine:
    """
    Production Vector Database Anomaly Diagnostic & Root-Cause Analysis Engine.
    Diagnoses dimension mismatches, un-normalized vector drifts, metric inversions,
    over-constrained metadata filtering, and Top-K context dilution via LangSmith traces.
    """
    def __init__(self):
        self.diagnostic_rules = {
            "DIMENSION_MISMATCH": "Query vector dimensions do not match index dimensions.",
            "UNNORMALIZED_DOT_PRODUCT": "Dot product metric used on vectors with magnitude != 1.0.",
            "SEMANTIC_WASHING": "All candidate chunks have low cosine similarity (< 0.60).",
            "OVERCONSTRAINED_FILTER": "Metadata filter eliminated 100% of candidate chunks.",
            "CONTEXT_DILUTION": "Top-K > 10 without reranking introduces severe noise."
        }

    def diagnose_retrieval_anomaly(
        self,
        query_dims: int,
        index_dims: int,
        metric: str,
        vector_magnitude: float,
        filter_match_count: int,
        top_k: int,
        max_similarity: float
    ) -> Dict[str, Any]:
        """
        Evaluates retrieval parameters against diagnostic rules and returns
        root-cause classification, severity, and LangSmith diagnostic trace attributes.
        """
        anomalies_detected = []
        severity = "HEALTHY"

        # 1. Dimension Mismatch Check
        if query_dims != index_dims:
            anomalies_detected.append({
                "code": "DIMENSION_MISMATCH",
                "severity": "FATAL",
                "detail": f"Query dimension ({query_dims}d) != Index dimension ({index_dims}d).",
                "remediation": "Align query bi-encoder model checkpoint with ingestion model checkpoint."
            })
            severity = "CRITICAL_FAILURE"

        # 2. Metric Inversion / Un-normalized Vectors Check
        if metric == "dot_product" and abs(vector_magnitude - 1.0) > 0.05:
            anomalies_detected.append({
                "code": "UNNORMALIZED_DOT_PRODUCT",
                "severity": "ERROR",
                "detail": f"Vector norm is {round(vector_magnitude, 2)}, but Dot Product requires unit vectors.",
                "remediation": "Apply L2 normalization prior to index querying, or switch index metric to Cosine."
            })
            if severity != "CRITICAL_FAILURE":
                severity = "DEGRADED"

        # 3. Over-Constrained Metadata Filtering Check
        if filter_match_count == 0:
            anomalies_detected.append({
                "code": "OVERCONSTRAINED_FILTER",
                "severity": "WARNING",
                "detail": "Metadata filter expression matched 0 candidate documents in collection.",
                "remediation": "Audit filter keys and values; inspect tenant_id and department tags."
            })
            if severity == "HEALTHY":
                severity = "EMPTY_RETRIEVAL"

        # 4. Semantic Washing / Poor Retrieval
        if max_similarity < 0.60 and filter_match_count > 0:
            anomalies_detected.append({
                "code": "SEMANTIC_WASHING",
                "severity": "WARNING",
                "detail": f"Max chunk similarity is {round(max_similarity, 3)} (< 0.60 threshold).",
                "remediation": "Adjust text chunking size (chunking too large averages diverse topics)."
            })
            if severity == "HEALTHY":
                severity = "POOR_RECALL"

        # 5. Top-K Context Dilution Check
        if top_k > 8:
            anomalies_detected.append({
                "code": "CONTEXT_DILUTION",
                "severity": "ADVISORY",
                "detail": f"Top-K is set to {top_k}. Large candidate pools dilute LLM attention without reranking.",
                "remediation": "Reduce Top-K to 3-5, or introduce a Cross-Encoder reranking span."
            })

        return {
            "status": severity,
            "anomaly_count": len(anomalies_detected),
            "anomalies": anomalies_detected,
            "langsmith_tags": [a["code"] for a in anomalies_detected] if anomalies_detected else ["retrieval_healthy"]
        }


# --- HERO SECTION ---
st.title("22.2 Vector Database Debugging")
st.caption("Chapter 22: LangSmith with Vector Databases | Domain: 11. LangSmith Observability & Testing")
st.markdown("""
A traditional SQL database either succeeds or throws a descriptive syntax error; **a vector database often fails silently**.
When retrieval degrades, the system continues to return HTTP 200 responses with candidate documents—except the documents are
**semantically irrelevant, inverted by wrong distance metrics, or completely empty due to typos in metadata filter tags**.
**Vector Database Debugging** with LangSmith provides automated root-cause diagnostics across the five most common vector failure modes:
1. **Incorrect Embeddings**: Detecting model version desynchronization, dimension drift, and un-normalized vector magnitudes.
2. **Poor Retrieval & Semantic Washing**: Identifying chunks with low similarity ceilings caused by oversized text splitting.
3. **Wrong Similarity Metrics**: Diagnosing Cosine vs Dot Product vs Euclidean distance rank inversions.
4. **Incorrect Metadata Filtering**: Catching over-constrained boolean filters that starve the context window.
5. **Top-K Problems**: Balancing candidate pool starvation against context dilution and lost-in-the-middle degradation.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Root-Cause Diagnostic Architecture for Vector DB Anomalies", divider="rainbow")

st.markdown("""
The architecture below outlines the automated diagnostic pipeline within LangSmith.
Every retrieval trace is analyzed across **Dimensionality Checkers**, **Normalization Verifiers**,
**Filter Cardinality Scanners**, and **Similarity Threshold Monitors** to tag anomalous runs automatically.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    vdb_debug_blueprint = """
    digraph VectorDBDiagnostics {
        graph [bgcolor="#0B1120", rankdir=TB, splines=spline, compound=true, pad="0.3", nodesep="0.45", ranksep="0.55"];
        node [fontname="Helvetica, Arial, sans-serif", fontsize=10, penwidth=1.5, style="filled,rounded"];
        edge [fontname="Helvetica, Arial, sans-serif", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.3, arrowsize=0.7];

        subgraph cluster_telemetry_in {
            label="1. Vector Search Run Telemetry Ingress";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#0EA5E9";
            fontcolor="#38BDF8";
            fontsize=11;

            RetrieverRun [label="Retriever Span Telemetry\nInputs: query, vector, top_k, filter\nOutputs: candidates, scores, latency", shape=box, fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_diagnostic_engine {
            label="2. LangSmith Diagnostic & Rule Evaluation Engine";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#10B981";
            fontcolor="#34D399";
            fontsize=11;

            DimCheck [label="1. Dimension Validator\nquery_dims == index_dims?", shape=diamond, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            NormCheck [label="2. Normalization Validator\nmetric == dot_product AND norm == 1.0?", shape=diamond, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            FilterCheck [label="3. Filter Cardinality Validator\nmatch_count > 0?", shape=diamond, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
            ThresholdCheck [label="4. Similarity Floor Validator\nmax_score >= 0.60?", shape=diamond, fillcolor="#1E1B4B", fontcolor="#C7D2FE", color="#6366F1"];
        }

        subgraph cluster_triage_actions {
            label="3. Automated Triage, Tagging & Remediation Tier";
            style="filled,rounded";
            fillcolor="#1E293B";
            color="#F59E0B";
            fontcolor="#FCD34D";
            fontsize=11;

            TagFatal [label="Tag: #dimension_mismatch\nRaise Fatal Exception in Trace", shape=box, fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];
            TagDegraded [label="Tag: #metric_inversion\nAlert on Ranking Distortion", shape=box, fillcolor="#451A03", fontcolor="#FED7AA", color="#F97316"];
            TagEmpty [label="Tag: #empty_filter_starvation\nAudit Metadata Predicate", shape=box, fillcolor="#451A03", fontcolor="#FED7AA", color="#F97316"];
            TagHealthy [label="Tag: #retrieval_healthy\nPass Chunks to LLM Generator", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        }

        RetrieverRun -> DimCheck;
        DimCheck -> TagFatal [label="No (Mismatch)", color="#EF4444"];
        DimCheck -> NormCheck [label="Yes"];

        NormCheck -> TagDegraded [label="No (Unnormalized)", color="#F97316"];
        NormCheck -> FilterCheck [label="Yes"];

        FilterCheck -> TagEmpty [label="No (0 Matches)", color="#F97316"];
        FilterCheck -> ThresholdCheck [label="Yes"];

        ThresholdCheck -> TagHealthy [label="Yes (Score >= 0.60)", color="#10B981"];
        ThresholdCheck -> TagDegraded [label="No (Semantic Wash)", color="#F97316"];
    }
    """
    st.graphviz_chart(vdb_debug_blueprint, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: INCORRECT EMBEDDINGS
# ==============================================================================
st.header("2. Incorrect Embeddings (Dimension Mismatch, Normalization Drift & Model Desynchronization)", divider="rainbow")

st.markdown("""
A primary cause of total retrieval failure is **Model Desynchronization**:
* The document corpus was embedded during offline ingestion using `text-embedding-3-small` (1536 dimensions).
* A developer deploys a new query service using an open-source model like `bge-small-en-v1.5` (384 dimensions) or `text-embedding-3-large` (3072 dimensions).

When the bi-encoders mismatch, the vector database throws an unhandled dimension exception, or worse, if dimensions happen to match, the semantic spaces are completely orthogonal, producing random garbage rankings.
""")

col_ie1, col_ie2 = st.columns([1, 1])

with col_ie1:
    st.markdown("### Common Embedding Failure Profiles:")
    st.markdown("""
    * **Dimension Mismatch**: Query vector has 768 dimensions while the index was constructed on 1536 dimensions.
    * **Normalization Drift**: Computing Dot Product similarity without dividing by vector magnitudes. If vector lengths vary, longer chunks receive artificially inflated similarity scores regardless of relevance.
    * **Truncated Text Bloat**: Passing 8,000 tokens of raw text into an embedding model whose maximum context length is 512 tokens. The model silently truncates the latter 93% of the document.
    """)

with col_ie2:
    st.markdown("""
    ### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
    In LangSmith, you configure **Custom Evaluators** to assert dimension equality on every retrieval span:
    ```python
    def validate_embedding_dimensions(run, example):
        query_dims = run.outputs.get("dimensions")
        index_dims = 1536
        return {
            "key": "dimension_match",
            "score": 1.0 if query_dims == index_dims else 0.0,
            "comment": "Aligned" if query_dims == index_dims else "Dimension Mismatch"
        }
    ```
    Runs that fail this check are tagged `#dimension_mismatch` and trigger automated Slack/PagerDuty alerts.
    """)

st.divider()

# ==============================================================================
# SECTION 3: POOR RETRIEVAL
# ==============================================================================
st.header("3. Poor Retrieval (Semantic Washing, Granularity Mismatch & Query-Document Lexical Gaps)", divider="rainbow")

st.markdown("""
Even with identical embedding models, retrieval can fail due to **Semantic Washing**:
When text chunks are chunked too coarsely (e.g., 2,000 words per chunk containing diverse topics),
the bi-encoder averages all topics into a single blurred vector representation.
Specific factual needles get washed out, causing maximum cosine similarity to drop below 0.60.
""")

col_pr1, col_pr2 = st.columns([1, 1])

with col_pr1:
    st.markdown("### Diagnosing Semantic Washing in LangSmith Traces:")
    st.markdown("""
    * **Symptom**: Max retrieved score across all Top-K candidates is consistently below 0.60.
    * **Root Cause 1**: Chunk size too large (averaging multiple concepts together).
    * **Root Cause 2**: Query-Document asymmetry (short 3-word keyword queries compared to long expository documents).
    * **Remedy**: Implement Parent-Document Chunking (retrieve small 128-token child chunks for vector matching, but deliver full 512-token parent chunks to the LLM generator).
    """)

with col_pr2:
    st.markdown("### Similarity Distribution Comparison")
    dist_df = pd.DataFrame({
        "Chunking Strategy": ["Optimal (256 tokens)", "Coarse / Washed (2,048 tokens)", "Tiny / Fractured (32 tokens)"],
        "Top-1 Cosine Score": [0.892, 0.541, 0.612],
        "Recall Accuracy": ["96.4%", "58.1% (Washed)", "62.5% (Missing Context)"],
        "Diagnostic Status": ["HEALTHY", "FLAGGED: SEMANTIC_WASHING", "FLAGGED: FRACTURED_CONTEXT"]
    })
    st.dataframe(dist_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: WRONG SIMILARITY METRIC
# ==============================================================================
st.header("4. Wrong Similarity Metric (Cosine vs Dot Product vs Euclidean Distance Inversion)", divider="rainbow")

st.markdown("""
Vector databases support three standard distance metrics, each with contrasting mathematical properties:
1. **Cosine Similarity**: Measures angular difference between vectors, independent of magnitude (`range: -1.0 to 1.0`).
2. **Dot Product (Inner Product / IP)**: Multiplies element-wise. Equivalent to Cosine *only* if vectors are normalized to unit length (`||v|| = 1.0`).
3. **Euclidean Distance (L2)**: Measures straight-line geometric distance. Lower distance means higher similarity.

Mismatches cause catastrophic candidate ranking inversion:
""")

col_sm1, col_sm2 = st.columns([1, 1])

with col_sm1:
    st.markdown("### Metric Configuration Inversion Trap:")
    st.markdown("""
    * If an index is configured for **Euclidean Distance (L2)**, the closest vector has distance near `0.0`.
    * If application code mistakenly treats the score as **Cosine Similarity** and sorts candidates descending (`ORDER BY score DESC`), the application will literally select the **most distant, least relevant document** in the entire database!
    * LangSmith span inspection reveals the raw scores: seeing scores of `142.8` indicates L2 distance, not cosine similarity.
    """)

with col_sm2:
    st.markdown("### Metric Behavior Summary")
    metric_matrix_df = pd.DataFrame({
        "Metric": ["Cosine Similarity", "Dot Product (IP)", "Euclidean Distance (L2)"],
        "Formula / Behavior": ["Angular alignment (ignores norm)", "Length-dependent dot product", "Geometric distance between points"],
        "Requires Normalization?": ["No (Self-normalizing)", "Strictly Required (Norm = 1.0)", "Recommended"],
        "Sorting Order": ["Descending (Higher is better)", "Descending (Higher is better)", "Ascending (Lower is better)"]
    })
    st.dataframe(metric_matrix_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: INCORRECT METADATA FILTERING
# ==============================================================================
st.header("5. Incorrect Metadata Filtering (Over-Constrained Predicates, Type Mismatches & Empty Result Sets)", divider="rainbow")

st.markdown("""
Metadata filtering restricts vector search to specific partitions (e.g., `tenant_id == 'acme' AND year == 2026`).
A widespread bug in enterprise RAG is **Over-Constrained Filtering**:
* Typos in filter keys (e.g., `"dept"` vs `"department"`).
* Type mismatches (e.g., passing `"year": "2026"` as a string when the index stored it as an integer `2026`).
* When an over-constrained filter matches zero records, the vector store returns an empty candidate list, starving the context window.
""")

col_mf1, col_mf2 = st.columns([1, 1])

with col_mf1:
    st.markdown("### Trace Diagnostic for Filter Starvation:")
    st.code('''
{
  "run_name": "QdrantRetriever",
  "inputs": {
    "query": "Q3 engineering budget",
    "filter": {
      "must": [
        {"key": "tenant_id", "match": {"value": "acme"}},
        {"key": "department", "match": {"value": "Engineering"}},
        {"key": "quarter", "match": {"value": "q3"}}  // Index has "Q3" uppercase!
      ]
    }
  },
  "outputs": {
    "hits_count": 0,
    "documents": []  // EMPTY! Context starvation!
  }
}
''', language="json")

with col_mf2:
    st.markdown("### Best Practices for Metadata Debugging:")
    st.markdown("""
    1. **Case-Insensitive Tag Normalization**: Always lowercase string attributes before indexing and filtering.
    2. **Log Pre-Filter Cardinality**: Inspect how many documents matched the scalar filter before vector distance calculation.
    3. **Automated Fallback**: If a restrictive filter returns 0 hits, log a LangSmith warning and execute an automated relaxation strategy (e.g., search across all quarters for that department).
    """)

st.divider()

# ==============================================================================
# SECTION 6: TOP-K PROBLEMS
# ==============================================================================
st.header("6. Top-K Problems (Candidate Pool Starvation, Context Dilution & Reranking Deficits)", divider="rainbow")

st.markdown("""
Configuring the `Top-K` retrieval parameter is a delicate engineering balance:
* **Top-K Too Small (K = 1 to 2)**: Triggers **Candidate Pool Starvation**. If the true factual answer requires synthesizing information across 3 different documents, the retriever fails to provide sufficient evidence.
* **Top-K Too Large (K = 15 to 30)**: Triggers **Context Dilution**. Irrelevant distracting chunks crowd the prompt, increasing latency, inflating token costs, and causing the LLM to suffer from the **Lost-in-the-Middle** hallucination effect.
""")

col_tk1, col_tk2 = st.columns([1, 1])

with col_tk1:
    st.markdown("### Top-K Sensitivity Trade-Offs:")
    topk_df = pd.DataFrame({
        "Top-K Setting": ["K = 1 to 2 (Under-Retrieved)", "K = 3 to 5 (Empirical Sweet Spot)", "K = 15 to 30 (Over-Retrieved)"],
        "Recall": ["Low (Misses multi-hop facts)", "High (Captures primary facts)", "Maximum (High coverage)"],
        "Precision": ["High (Zero filler)", "Balanced", "Very Low (Excessive noise)"],
        "LLM Attention Impact": ["Context Starvation", "Optimal Focus", "Lost-in-the-Middle Dilution"],
        "Recommended Architecture": ["Add Two-Stage Reranker", "Standard Direct RAG", "Mandatory Cross-Encoder Reranker"]
    })
    st.dataframe(topk_df, use_container_width=True)

with col_tk2:
    st.markdown("### The Two-Stage Solution: Retrieve Broad -> Rerank Narrow")
    dot_twostage = """
    digraph TwoStageRAG {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [fontname="Helvetica", fontsize=9, style="filled,rounded", penwidth=1.2];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8"];

        Stage1 [label="Stage 1: High-Recall Vector Search\nRetrieve Top-25 Candidates\n(Fast HNSW, ensures coverage)", shape=box, fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        Rerank [label="Stage 2: Cross-Encoder Reranker\nScore Candidate Relevancy\n(Prunes 25 chunks down to Top-4)", shape=ellipse, fillcolor="#0284C7", fontcolor="#FFFFFF"];
        LLM [label="Generator Context Window\nReceives exactly 4 High-Precision Chunks\n(Zero Dilution, Optimal Synthesis)", shape=box, fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

        Stage1 -> Rerank -> LLM;
    }
    """
    st.graphviz_chart(dot_twostage, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Vector Database Diagnostic Debugger (VectorDBDebuggingEngine)", divider="rainbow")

st.markdown("""
Below is the complete production diagnostic engine `VectorDBDebuggingEngine`.
It implements rule-based anomaly detection across dimensionality mismatches, normalization drifts,
empty filter sets, semantic washing, and Top-K context dilution.
""")

st.code('''
class VectorDBDebuggingEngine:
    """
    Production Vector Database Anomaly Diagnostic & Root-Cause Analysis Engine.
    Diagnoses dimension mismatches, un-normalized vector drifts, metric inversions,
    over-constrained metadata filtering, and Top-K context dilution via LangSmith traces.
    """
    def __init__(self):
        self.diagnostic_rules = {
            "DIMENSION_MISMATCH": "Query vector dimensions do not match index dimensions.",
            "UNNORMALIZED_DOT_PRODUCT": "Dot product metric used on vectors with magnitude != 1.0.",
            "SEMANTIC_WASHING": "All candidate chunks have low cosine similarity (< 0.60).",
            "OVERCONSTRAINED_FILTER": "Metadata filter eliminated 100% of candidate chunks.",
            "CONTEXT_DILUTION": "Top-K > 10 without reranking introduces severe noise."
        }

    def diagnose_retrieval_anomaly(
        self,
        query_dims: int,
        index_dims: int,
        metric: str,
        vector_magnitude: float,
        filter_match_count: int,
        top_k: int,
        max_similarity: float
    ) -> Dict[str, Any]:
        anomalies_detected = []
        severity = "HEALTHY"

        if query_dims != index_dims:
            anomalies_detected.append({
                "code": "DIMENSION_MISMATCH",
                "severity": "FATAL",
                "detail": f"Query dimension ({query_dims}d) != Index dimension ({index_dims}d)."
            })
            severity = "CRITICAL_FAILURE"

        if metric == "dot_product" and abs(vector_magnitude - 1.0) > 0.05:
            anomalies_detected.append({
                "code": "UNNORMALIZED_DOT_PRODUCT",
                "severity": "ERROR",
                "detail": f"Vector norm is {round(vector_magnitude, 2)}, but Dot Product requires unit vectors."
            })
            if severity != "CRITICAL_FAILURE":
                severity = "DEGRADED"

        if filter_match_count == 0:
            anomalies_detected.append({
                "code": "OVERCONSTRAINED_FILTER",
                "severity": "WARNING",
                "detail": "Metadata filter matched 0 candidate documents."
            })
            if severity == "HEALTHY":
                severity = "EMPTY_RETRIEVAL"

        if max_similarity < 0.60 and filter_match_count > 0:
            anomalies_detected.append({
                "code": "SEMANTIC_WASHING",
                "severity": "WARNING",
                "detail": f"Max chunk similarity is {round(max_similarity, 3)} (< 0.60 threshold)."
            })
            if severity == "HEALTHY":
                severity = "POOR_RECALL"

        if top_k > 8:
            anomalies_detected.append({
                "code": "CONTEXT_DILUTION",
                "severity": "ADVISORY",
                "detail": f"Top-K is {top_k}. Large pools cause context dilution without reranking."
            })

        return {
            "status": severity,
            "anomaly_count": len(anomalies_detected),
            "anomalies": anomalies_detected,
            "langsmith_tags": [a["code"] for a in anomalies_detected] if anomalies_detected else ["retrieval_healthy"]
        }
''', language="python")

# --- SELF VERIFICATION OF PRODUCTION ENGINE ---
debug_engine_inst = VectorDBDebuggingEngine()

# Test 1: Dimension Mismatch Detection
dim_test = debug_engine_inst.diagnose_retrieval_anomaly(
    query_dims=384, index_dims=1536, metric="cosine", vector_magnitude=1.0, filter_match_count=10, top_k=3, max_similarity=0.85
)
assert dim_test["status"] == "CRITICAL_FAILURE", "Must catch dimension mismatch as CRITICAL_FAILURE"
assert "DIMENSION_MISMATCH" in dim_test["langsmith_tags"]

# Test 2: Unnormalized Dot Product Detection
norm_test = debug_engine_inst.diagnose_retrieval_anomaly(
    query_dims=1536, index_dims=1536, metric="dot_product", vector_magnitude=4.2, filter_match_count=10, top_k=3, max_similarity=0.85
)
assert "UNNORMALIZED_DOT_PRODUCT" in norm_test["langsmith_tags"]

# Test 3: Over-constrained Filter Detection
filter_test = debug_engine_inst.diagnose_retrieval_anomaly(
    query_dims=1536, index_dims=1536, metric="cosine", vector_magnitude=1.0, filter_match_count=0, top_k=3, max_similarity=0.0
)
assert filter_test["status"] == "EMPTY_RETRIEVAL"
assert "OVERCONSTRAINED_FILTER" in filter_test["langsmith_tags"]

st.success("Automated Verification Passed: VectorDBDebuggingEngine successfully verified across dimension checks, normalization drift, and filter starvation.")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Vector Database Failure Mode Simulator & Trace Diagnostic Sandbox", divider="rainbow")

st.markdown("""
Inject synthetic vector database anomalies and inspect how LangSmith flags and diagnoses root causes in real time.
Adjust dimensions, distance metrics, vector magnitudes, filter hit counts, and Top-K settings.
""")

wb_d1, wb_d2 = st.columns([1, 1])

with wb_d1:
    st.markdown("#### Retrieval Parameter Controls")
    in_q_dims = st.selectbox("Query Embedding Dimensions", [384, 768, 1536, 3072], index=2)
    in_idx_dims = st.selectbox("Index Embedding Dimensions", [384, 768, 1536, 3072], index=2)
    in_metric = st.selectbox("Distance Metric", ["cosine", "dot_product", "euclidean"])
    in_norm = st.slider("Vector Magnitude (Norm)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

with wb_d2:
    st.markdown("#### Corpus & Filter Controls")
    in_filter_hits = st.slider("Metadata Filter Matching Documents", min_value=0, max_value=50, value=12, step=1)
    in_top_k = st.slider("Top-K Candidates Requested", min_value=1, max_value=25, value=4, step=1)
    in_max_sim = st.slider("Max Retrieved Cosine Similarity", min_value=0.20, max_value=0.99, value=0.88, step=0.02)

diag_res = debug_engine_inst.diagnose_retrieval_anomaly(
    query_dims=in_q_dims,
    index_dims=in_idx_dims,
    metric=in_metric,
    vector_magnitude=in_norm,
    filter_match_count=in_filter_hits,
    top_k=in_top_k,
    max_similarity=in_max_sim
)

st.markdown("### LangSmith Run Diagnostic Telemetry")

dm_1, dm_2, dm_3 = st.columns(3)
dm_1.metric("Run Diagnostic Status", diag_res["status"], delta="Issues Found" if diag_res["anomaly_count"] > 0 else "Clean")
dm_2.metric("Total Anomalies Detected", f"{diag_res['anomaly_count']} Issues")
dm_3.metric("LangSmith Automated Tags", ", ".join(diag_res["langsmith_tags"]))

if diag_res["anomalies"]:
    st.markdown("#### Detected Anomalies & Recommended Remediation")
    anom_table = [
        {
            "Bug Code": a["code"],
            "Severity": a["severity"],
            "Diagnostic Detail": a["detail"],
            "Remediation Action": a.get("remediation", "Inspect configuration")
        }
        for a in diag_res["anomalies"]
    ]
    st.dataframe(pd.DataFrame(anom_table), use_container_width=True)
else:
    st.success("All retrieval parameters are fully aligned. Vector search pipeline is healthy and ready for production serving.")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Questions & Architectural Solutions")

with st.expander("**Q1: What happens if you execute Dot Product similarity search on un-normalized embeddings?**"):
    st.markdown("""
    **Answer**:
    The Dot Product formula is `dot(u, v) = ||u|| * ||v|| * cos(theta)`.
    * If vectors are normalized to unit length (`||u|| = 1`, `||v|| = 1`), Dot Product is mathematically identical to Cosine Similarity.
    * If vectors are un-normalized, candidate documents with large magnitudes (e.g., longer chunks or chunks containing repetitive high-entropy words) receive artificially inflated dot product scores, completely dominating the search results regardless of topical relevance.
    * In LangSmith, this manifests as low-relevance documents ranking at Top-1 with abnormally high inner product scores.
    """)

with st.expander("**Q2: How does Semantic Washing degrade retrieval, and what chunking strategy resolves it?**"):
    st.markdown("""
    **Answer**:
    Semantic Washing occurs when document splitters create chunks that are too large (e.g., 2,000+ tokens encompassing diverse concepts).
    Because embedding models project the entire text into a single fixed-dimensional point, diverse concepts pull the vector in competing directions, diluting specific factual signal.
    * **The Resolution**:
      Implement **Parent-Document (Hierarchical) Retrieval**. Text is chunked into small 128-token child segments for bi-encoder indexing. When a child segment matches the user's query, its broader 512-token parent chunk is retrieved and delivered to the LLM.
    """)

with st.expander("**Q3: Why can metadata filtering cause HNSW graph disconnection and search failure?**"):
    st.markdown("""
    **Answer**:
    In HNSW graphs, nearest neighbor searches traverse proximity edges between vectors.
    If a metadata filter is highly restrictive (e.g., matching only 0.1% of the vectors), the majority of graph nodes are inaccessible.
    During greedy routing, the search algorithm can hit dead ends where none of the current node's neighbors satisfy the filter, terminating early and returning empty results even though valid matching vectors exist elsewhere in the graph.
    * **Remedy**: Use modern vector engines with **Iterative Filtering** or hybrid inverted-index fallback routines.
    """)

with st.expander("**Q4: What is the Lost-in-the-Middle phenomenon, and how is it triggered by Top-K misconfiguration?**"):
    st.markdown("""
    **Answer**:
    Research shows that transformer language models attend heavily to tokens at the very beginning and very end of their input prompts, while attention weights sag significantly in the middle 60% of the context.
    If Top-K is set too high (e.g., K = 20), the primary evidence chunk often ends up placed in the middle of a massive context block.
    The LLM fails to attend to it and hallucinates, despite the fact that the answer was technically present in the prompt.
    """)

with st.expander("**Q5: How does LangSmith automate the detection of empty retrieval result sets in production?**"):
    st.markdown("""
    **Answer**:
    LangSmith allows setting up **Automated Rules & Alerting Filters** on project traces:
    ```
    FILTER: run_type = "retriever" AND outputs.documents.length = 0
    ACTION: Tag run "#empty_retrieval" AND send PagerDuty alert
    ```
    This catches over-constrained metadata filters or database indexing dropouts immediately, before end users submit complaints.
    """)

st.markdown("### Production Traps & Failure Modes")

dtr_col1, dtr_col2 = st.columns(2)

with dtr_col1:
    st.error("""
    **Trap 1: Silent Vector Metric Inversion**
    * **Failure**: Creating an index using Euclidean (L2) distance but sorting retrieved candidates in descending order (`ORDER BY score DESC`).
    * **Consequence**: The application retrieves the most diametrically opposed, least relevant chunks in the database, producing garbage answers.
    * **Remedy**: Always verify metric sorting in LangSmith: L2 distances must be sorted ascending, whereas Cosine and Dot Product must be sorted descending.
    """)

    st.error("""
    **Trap 2: Case-Sensitive String Mismatches in Metadata Filters**
    * **Failure**: Filtering on `{"department": "finance"}` when source files were ingested with `{"department": "Finance"}`.
    * **Consequence**: The filter matches 0 documents, returning an empty context and causing the bot to reply "I do not know".
    * **Remedy**: Enforce strict lowercase normalization across all metadata keys and values during both ingestion and retrieval.
    """)

with dtr_col2:
    st.warning("""
    **Trap 3: Oversized Top-K without Cross-Encoder Reranking**
    * **Failure**: Setting Top-K = 25 directly into the LLM context to 'guarantee high recall'.
    * **Consequence**: Token costs skyrocket 5x, TTFT latency triples, and the LLM suffers from lost-in-the-middle hallucinations.
    * **Remedy**: Keep Top-K at 3-5 for direct generation, or insert a lightweight Cross-Encoder reranker span to prune 25 candidates down to 4.
    """)

    st.warning("""
    **Trap 4: Upgrading Embedding Checkpoints without Corpus Re-Indexing**
    * **Failure**: Upgrading query embedding code from `text-embedding-ada-002` to `text-embedding-3-small` without re-indexing historical documents.
    * **Consequence**: Cosine similarity scores between queries and chunks collapse to near zero because the models map to completely incompatible semantic coordinate spaces.
    * **Remedy**: Always version vector collections and enforce automated embedding model parity assertions in CI/CD pipelines.
    """)

st.markdown("### Core Technical Flashcards")

fcd_c1, fcd_c2 = st.columns(2)

with fcd_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the mathematical condition required for Dot Product to equal Cosine Similarity?")
        st.markdown("**Back**: Both vectors must be normalized to unit length (`||u|| = 1.0` and `||v|| = 1.0`). If un-normalized, Dot Product is heavily skewed by vector magnitude.")

    with st.container(border=True):
        st.markdown("**Front**: What is Semantic Washing in RAG chunking?")
        st.markdown("**Back**: When text chunks are too large (2,000+ tokens), the embedding model averages diverse topics into a single blurred vector, washing out specific factual needles and dropping cosine similarity.")

with fcd_c2:
    with st.container(border=True):
        st.markdown("**Front**: What causes Context Dilution when Top-K is set too high?")
        st.markdown("**Back**: Too many tangential chunks crowd the prompt, inflating token costs and causing the LLM to overlook the primary evidence due to the Lost-in-the-Middle attention sag.")

    with st.container(border=True):
        st.markdown("**Front**: How does LangSmith detect over-constrained metadata filtering in production?")
        st.markdown("**Back**: By monitoring retriever span outputs for `hits_count == 0` or empty document arrays, allowing automated alerting when filters eliminate 100% of candidate chunks.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Vector Database Debugging principles:")

d_chk1 = st.checkbox("I can identify and diagnose dimension mismatches and model desynchronization between bi-encoders.", value=True)
d_chk2 = st.checkbox("I understand why un-normalized vectors distort Dot Product rankings and how to verify vector norms in traces.", value=True)
d_chk3 = st.checkbox("I know how to identify Semantic Washing from similarity score floors and resolve it with hierarchical chunking.", value=True)
d_chk4 = st.checkbox("I understand the sorting direction differences between Cosine (descending) and Euclidean L2 (ascending) metrics.", value=True)
d_chk5 = st.checkbox("I can debug over-constrained metadata filters and type mismatches that starve the context window.", value=True)
d_chk6 = st.checkbox("I understand how two-stage retrieval (Broad Top-25 -> Narrow Top-4 Reranking) resolves Top-K dilution.", value=True)

if d_chk1 and d_chk2 and d_chk3 and d_chk4 and d_chk5 and d_chk6:
    st.success("Mastery Certified: You have mastered vector database debugging, metric verification, and automated trace diagnostics in LangSmith.")
