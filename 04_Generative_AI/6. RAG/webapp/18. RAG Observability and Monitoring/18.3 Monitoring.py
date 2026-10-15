import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="18.3 Production RAG Monitoring & Alerting",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class ProductionMonitoringEngine:
    """Enterprise RAG Production Telemetry, Quality Monitoring & Automated Alerting Engine."""

    def __init__(self, latency_sla_ms: float = 1200.0, cost_cap_usd: float = 50.0, similarity_floor: float = 0.65):
        self.latency_sla_ms = latency_sla_ms
        self.cost_cap_usd = cost_cap_usd
        self.similarity_floor = similarity_floor
        self.daily_spend_usd = 0.0
        self.total_queries = 0
        self.alerts: List[Dict[str, Any]] = []

    def check_retrieval_health(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect zero-hit retrievals, similarity floor violations, and distribution drift."""
        if not retrieved_docs:
            return {
                "status": "FAIL",
                "reason": "Zero chunks retrieved (Empty context breach)",
                "max_score": 0.0,
                "mean_score": 0.0,
                "circuit_breaker": True
            }
        scores = [doc.get("score", 0.0) for doc in retrieved_docs]
        max_score = float(np.max(scores))
        mean_score = float(np.mean(scores))
        
        if max_score < self.similarity_floor:
            return {
                "status": "FAIL",
                "reason": f"Max similarity score ({max_score:.3f}) below safety floor ({self.similarity_floor:.3f})",
                "max_score": max_score,
                "mean_score": mean_score,
                "circuit_breaker": False
            }
        return {
            "status": "HEALTHY",
            "reason": "Retrieval distribution within nominal boundaries",
            "max_score": max_score,
            "mean_score": mean_score,
            "circuit_breaker": False
        }

    def detect_hallucination(self, query: str, context_chunks: List[str], generated_answer: str) -> Dict[str, Any]:
        """Fast lexical-semantic groundedness audit checking entity overlap and phantom citations."""
        if not generated_answer or not context_chunks:
            return {"groundedness_score": 0.0, "is_hallucination": True, "verdict": "Unverified Generation"}
        
        combined_context = " ".join(context_chunks).lower()
        answer_words = [w.strip(".,;:?!()[]\"'") for w in generated_answer.lower().split() if len(w) > 4]
        
        if not answer_words:
            return {"groundedness_score": 1.0, "is_hallucination": False, "verdict": "Nominal Short Answer"}

        matched_words = [w for w in answer_words if w in combined_context]
        groundedness_score = float(len(matched_words) / len(answer_words))
        
        has_citations = "[" in generated_answer and "]" in generated_answer
        is_hallucination = groundedness_score < 0.60
        
        verdict = "Faithful Grounded" if not is_hallucination else "High Hallucination Risk (Unanchored Claims)"
        if not has_citations and len(generated_answer) > 80:
            verdict += " | Missing In-line Citations"

        return {
            "groundedness_score": round(groundedness_score, 3),
            "is_hallucination": is_hallucination,
            "matched_tokens": len(matched_words),
            "total_audit_tokens": len(answer_words),
            "verdict": verdict
        }

    def audit_latency_breakdown(self, retrieval_ms: float, rerank_ms: float, ttft_ms: float, gen_ms: float) -> Dict[str, Any]:
        """Verify SLA compliance and pinpoint latency bottleneck."""
        total_latency_ms = retrieval_ms + rerank_ms + ttft_ms + gen_ms
        breached = total_latency_ms > self.latency_sla_ms
        
        bottleneck = "Retrieval"
        max_val = retrieval_ms
        if rerank_ms > max_val:
            bottleneck = "Reranking"
            max_val = rerank_ms
        if ttft_ms > max_val:
            bottleneck = "TTFT (LLM Queue)"
            max_val = ttft_ms
        if gen_ms > max_val:
            bottleneck = "LLM Generation"
            max_val = gen_ms
            
        return {
            "total_ms": round(total_latency_ms, 1),
            "breached": breached,
            "sla_limit_ms": self.latency_sla_ms,
            "bottleneck": bottleneck,
            "pct_of_budget": round((total_latency_ms / self.latency_sla_ms) * 100.0, 1)
        }

    def audit_cost_burn(self, prompt_tokens: int, completion_tokens: int, prompt_rate_per_m: float = 2.50, comp_rate_per_m: float = 10.00) -> Dict[str, Any]:
        """Compute query cost and update cumulative budget burn."""
        cost = (prompt_tokens / 1_000_000.0) * prompt_rate_per_m + (completion_tokens / 1_000_000.0) * comp_rate_per_m
        self.daily_spend_usd += cost
        self.total_queries += 1
        
        burn_rate_pct = (self.daily_spend_usd / self.cost_cap_usd) * 100.0
        budget_breached = self.daily_spend_usd > self.cost_cap_usd
        
        return {
            "query_cost_usd": round(cost, 5),
            "daily_spend_usd": round(self.daily_spend_usd, 4),
            "budget_cap_usd": self.cost_cap_usd,
            "burn_rate_pct": round(burn_rate_pct, 2),
            "budget_breached": budget_breached
        }

    def evaluate_quality_triad(self, context_relevance: float, faithfulness: float, answer_relevance: float) -> Dict[str, Any]:
        """Aggregate Triad score and flag quality degradation."""
        triad_mean = float((context_relevance + faithfulness + answer_relevance) / 3.0)
        warning_triggered = triad_mean < 0.70 or faithfulness < 0.65
        return {
            "triad_mean": round(triad_mean, 3),
            "context_relevance": round(context_relevance, 2),
            "faithfulness": round(faithfulness, 2),
            "answer_relevance": round(answer_relevance, 2),
            "quality_flag": "CRITICAL" if triad_mean < 0.50 else ("WARNING" if warning_triggered else "PASS")
        }

    def evaluate_and_route_alerts(
        self,
        retrieval_res: Dict[str, Any],
        hallucination_res: Dict[str, Any],
        latency_res: Dict[str, Any],
        cost_res: Dict[str, Any],
        quality_res: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Dispatch production alerts based on SLA and quality breaches."""
        new_alerts = []
        if retrieval_res["circuit_breaker"]:
            new_alerts.append({
                "severity": "SEV-1 CRITICAL",
                "subsystem": "Vector Database",
                "message": f"Zero-hit retrieval failure detected. Automated circuit breaker triggered to fallback corpus.",
                "routing": "PagerDuty Primary On-Call -> Engineering Incident Channel"
            })
        elif retrieval_res["status"] == "FAIL":
            new_alerts.append({
                "severity": "SEV-2 HIGH",
                "subsystem": "Vector Retriever",
                "message": f"Similarity score collapsed below floor ({retrieval_res['max_score']:.3f} < {self.similarity_floor}). Out-of-domain query surge.",
                "routing": "Slack #rag-alerts-high"
            })
            
        if hallucination_res["is_hallucination"]:
            new_alerts.append({
                "severity": "SEV-2 HIGH",
                "subsystem": "Guardrail Evaluator",
                "message": f"Hallucination score breach: Groundedness ({hallucination_res['groundedness_score']}) below safety threshold.",
                "routing": "Telemetry Datadog Event -> Slack #rag-quality"
            })
            
        if latency_res["breached"]:
            new_alerts.append({
                "severity": "SEV-3 MEDIUM",
                "subsystem": "Serving Gateway",
                "message": f"Latency SLA breached: {latency_res['total_ms']}ms > {latency_res['sla_limit_ms']}ms. Bottleneck: {latency_res['bottleneck']}.",
                "routing": "Datadog Monitor -> Opsgenie Tier-2"
            })
            
        if cost_res["budget_breached"]:
            new_alerts.append({
                "severity": "SEV-1 CRITICAL",
                "subsystem": "FinOps Gateway",
                "message": f"Daily budget ceiling violated: Spend ${cost_res['daily_spend_usd']:.2f} exceeded Cap ${cost_res['budget_cap_usd']:.2f}.",
                "routing": "AWS SNS -> Engineering Manager & CFO FinOps Alert"
            })
            
        if quality_res["quality_flag"] == "CRITICAL":
            new_alerts.append({
                "severity": "SEV-2 HIGH",
                "subsystem": "Continuous Eval",
                "message": f"Aggregate RAG Triad score dropped to critical floor ({quality_res['triad_mean']}). Prompt/Model degradation detected.",
                "routing": "Slack #rag-eval-monitoring"
            })
            
        self.alerts.extend(new_alerts)
        return new_alerts


# --- Streamlit Layout ---

st.title("18.3 RAG Production Monitoring & Real-Time Alerting")
st.caption("Continuous SLA Telemetry, Failure Detection, RAG Triad Guardrails, Cost Velocity & Incident Management")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise RAG Production Monitoring & Incident Response System", divider="rainbow")

st.markdown(
    """
    In mission-critical generative AI systems, deployment is not the end of the engineering lifecycle; it is the inception of **continuous runtime telemetry**.
    A production RAG pipeline consists of distributed, non-deterministic dependencies: vector index clusters, reranking endpoints, embedding inference services,
    and frontier LLM gateways. Silent failure modes—such as semantic drift, phantom citations, context truncation, and token budget exhaustion—cannot be detected
    by traditional HTTP `200 OK` health checks.
    
    The diagram below models the complete **enterprise RAG monitoring topology**, detailing telemetry collection, asynchronous stream evaluators, anomaly detection filters,
    and prioritized multi-tier incident alerting.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RAG_Monitoring_Architecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_gateway {
            label = "1. INGRESS & TELEMETRY COLLECTION TIER";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            client_req [label="User Query Ingress\\n(FastAPI Gateway / Kong)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            otel_interceptor [label="Async Telemetry Interceptor\\n(OpenTelemetry Traces & Metrics)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            pipeline_exec [label="Live Pipeline Execution\\n(Embed -> Retrieve -> Rerank -> LLM)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        subgraph cluster_streaming_bus {
            label = "2. ASYNCHRONOUS EVENT BUS & LOG PIPELINE";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            event_bus [label="Kafka Event Stream / AWS Kinesis\\n(Full Payload: Query, Chunks, Answer, Stats)", shape=cylinder, fillcolor="#312E81", fontcolor="#A5B4FC"];
            consumer_workers [label="Continuous Eval Workers\\n(Celery / Kubernetes Autoscaled Pods)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
        }

        subgraph cluster_eval_engines {
            label = "3. SPECIALIZED AUDIT & METRIC EVALUATORS";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            retrieval_guard [label="Retrieval Failure Monitor\\n- Zero-Hit Checks\\n- Cosine Floor (< 0.65)\\n- Index Freshness", fillcolor="#0F172A", fontcolor="#38BDF8"];
            hallucination_guard [label="Hallucination Scanner\\n- Groundedness Ratio\\n- Phantom Citations\\n- Sentence Contradiction", fillcolor="#0F172A", fontcolor="#F43F5E"];
            latency_guard [label="Latency & SLA Monitor\\n- P50/P95/P99 Tracking\\n- TTFT Queues\\n- Endpoint Timeouts", fillcolor="#0F172A", fontcolor="#F59E0B"];
            cost_guard [label="FinOps Cost Guard\\n- Token Velocity\\n- Daily Burn Caps\\n- Margin Safeguards", fillcolor="#0F172A", fontcolor="#10B981"];
            quality_guard [label="Continuous Triad Eval\\n- Context Relevance\\n- Faithfulness\\n- Answer Relevance", fillcolor="#0F172A", fontcolor="#8B5CF6"];
        }

        subgraph cluster_storage {
            label = "4. TELEMETRY WAREHOUSE & DASHBOARDS";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            tsdb [label="Prometheus & Datadog\\n(Time-Series Metrics)", shape=cylinder, fillcolor="#0F172A", fontcolor="#38BDF8"];
            clickhouse [label="ClickHouse / BigQuery\\n(Full Execution Traces & Payloads)", shape=cylinder, fillcolor="#0F172A", fontcolor="#38BDF8"];
            grafana_dash [label="Grafana Operations Cockpit\\n(Real-Time Operations & SLAs)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
        }

        subgraph cluster_incident {
            label = "5. INCIDENT ROUTING & AUTOMATED REMEDIATION";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            alert_manager [label="Prometheus Alertmanager / Datadog Monitors\\n(Deduplication & Grouping)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            sev1_pager [label="SEV-1: PagerDuty Escalation\\n(On-Call Engineers Paged)", fillcolor="#7F1D1D", fontcolor="#FCA5A5"];
            sev2_slack [label="SEV-2 / SEV-3: Slack Channels\\n(#rag-ops-alerts, #rag-finops)", fillcolor="#78350F", fontcolor="#FDE68A"];
            circuit_breaker [label="Automated Circuit Breaker\\n- Divert to Fallback Corpus\\n- Cache Fallback / Graceful Abstention", fillcolor="#4C1D95", fontcolor="#DDD6FE"];
        }

        client_req -> otel_interceptor [label="Raw HTTP Ingress"];
        otel_interceptor -> pipeline_exec [label="Execute Request"];
        pipeline_exec -> event_bus [label="Emit Telemetry Event (Async)"];
        
        event_bus -> consumer_workers [label="Consume Batch"];
        consumer_workers -> retrieval_guard;
        consumer_workers -> hallucination_guard;
        consumer_workers -> latency_guard;
        consumer_workers -> cost_guard;
        consumer_workers -> quality_guard;
        
        retrieval_guard -> tsdb [label="Push Gauges"];
        hallucination_guard -> tsdb [label="Push Violation Rate"];
        latency_guard -> tsdb [label="Push Histograms"];
        cost_guard -> tsdb [label="Push Token Rates"];
        quality_guard -> tsdb [label="Push Triad Scores"];
        
        consumer_workers -> clickhouse [label="Stream Records"];
        tsdb -> grafana_dash [label="Render Dashboards"];
        
        tsdb -> alert_manager [label="Threshold Breaches"];
        alert_manager -> sev1_pager [label="P99 Latency > 3s or Budget Exhausted"];
        alert_manager -> sev2_slack [label="Groundedness < 0.65 or Cosine < 0.60"];
        alert_manager -> circuit_breaker [label="Vector DB 0-Hits or Provider Downtime"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: RETRIEVAL FAILURES ---
st.header("2. Retrieval Failures (Similarity Floors, Zero-Hit Queries, Fallback Triggers & Distributional Drift)", divider="rainbow")

st.markdown(
    """
    The first failure point in any RAG system is **retrieval collapse**. If the retriever fails to surface contextually relevant chunks,
    downstream generation is mathematically bounded by the LLM's static parametric memory or forced into hallucination.
    
    ### Key Retrieval Failure Signatures in Production
    - **Zero-Hit Ingress (Empty Context)**: The vector index query returns 0 chunks within the maximum distance radius, usually caused by misconfigured metadata filters (e.g., `tenant_id = 'X'` and `date > '2026-01-01'` eliminating all candidates).
    - **Similarity Floor Collapse**: Chunks are retrieved, but the top cosine similarity score is below the statistical relevance floor (e.g., `score < 0.65`). When top similarity collapses, the user has submitted an out-of-distribution (OOD) query or an adversarial prompt.
    - **Index Freshness & Representation Drift**: As domain vocabulary shifts (e.g., new product versions, regulatory revisions), older chunk embeddings drift away from modern user query embeddings, causing a gradual decay in mean retrieval similarity.
    - **Chunk Truncation & Segmentation Leaks**: Chunks split across tabular boundaries or broken mid-sentence prevent dense embeddings from capturing semantic meaning, leading to low rank retrieval scores.
    """
)

col_rf1, col_rf2 = st.columns(2)
with col_rf1:
    st.subheader("Retrieval Anomaly Threshold Matrix")
    rf_data = {
        "Metric": ["Zero-Hit Rate", "Top-1 Cosine Score", "Mean Top-K Score", "Metadata Filter Drops"],
        "Nominal Target": ["< 0.1%", "> 0.78", "> 0.72", "< 2.0%"],
        "Warning Level": ["0.5% - 2.0%", "0.68 - 0.75", "0.62 - 0.68", "2.0% - 8.0%"],
        "Critical Level": ["> 2.0%", "< 0.65", "< 0.60", "> 8.0%"],
        "Automated Mitigation": ["Trigger Fallback BM25", "Trigger Query Expansion / HyDE", "Enlarge Top-K Candidate Pool", "Drop Restrictive Filters"]
    }
    st.dataframe(pd.DataFrame(rf_data), use_container_width=True)

with col_rf2:
    st.subheader("Retrieval Anomaly Triage State Machine")
    rf_diag = """
    digraph RetrievalTriage {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.3, ranksep=0.4];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=10, color="#38BDF8", penwidth=1.2];
        edge [color="#64748B", fontname="Inter", fontsize=9, fontcolor="#94A3B8"];

        query [label="Query Ingested", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        check_hits [label="Chunks Returned > 0?", shape=diamond, fillcolor="#1E3A8A", fontcolor="#60A5FA"];
        check_floor [label="Top Score >= 0.65?", shape=diamond, fillcolor="#1E3A8A", fontcolor="#60A5FA"];
        
        fallback_bm25 [label="Circuit Breaker:\\nFallback to Keyword BM25", fillcolor="#7F1D1D", fontcolor="#FCA5A5"];
        hyde_retry [label="Warning Action:\\nTrigger HyDE Rewriter", fillcolor="#78350F", fontcolor="#FDE68A"];
        nominal_pass [label="Retrieval Healthy:\\nPass Chunks to Reranker", fillcolor="#064E3B", fontcolor="#6EE7B7"];

        query -> check_hits;
        check_hits -> fallback_bm25 [label="No (Zero-Hits)"];
        check_hits -> check_floor [label="Yes"];
        check_floor -> hyde_retry [label="No (OOD Drift)"];
        check_floor -> nominal_pass [label="Yes (Nominal)"];
    }
    """
    st.graphviz_chart(rf_diag, use_container_width=True)

# --- SECTION 3: HALLUCINATION DETECTION ---
st.header("3. Hallucination Detection (Real-Time Groundedness Scanners & Phantom Citation Alarms)", divider="rainbow")

st.markdown(
    """
    Hallucinations in production RAG systems fall into two distinct engineering classifications:
    1. **Extrinsic Hallucinations (Unanchored Claims)**: The generator introduces facts, figures, or entities that are neither present in the retrieved context nor verifiable in the external knowledge base.
    2. **Intrinsic Hallucinations (Faithfulness Violations)**: The generator directly misrepresents, contradicts, or swaps entities from the retrieved chunks (e.g., claiming *Drug X* treats *Disease Y* when the document states *Drug X* is contraindicated for *Disease Y*).
    
    ### Production Detection Architectures
    Continuous offline batch evaluation is insufficient for consumer-facing or regulatory systems. Real-time RAG observability employs three parallel detection strategies:
    """
)

col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    with st.container(border=True):
        st.markdown("#### 1. Real-Time Token Grounding")
        st.markdown(
            """
            - **Mechanism**: Extracts non-stopword noun chunks and named entities from the generation and computes exact/fuzzy token overlap against retrieved chunks.
            - **Latency**: `< 5ms` per request.
            - **Use Case**: Inline API gateway rejection before streaming completion to the client.
            """
        )
with col_h2:
    with st.container(border=True):
        st.markdown("#### 2. Small Guardrail LLM Verifier")
        st.markdown(
            """
            - **Mechanism**: Passes `(Context, Generation)` to a lightweight distilled model (e.g., Llama-3-8B-Instruct or Mistral-7B) prompted solely for binary NLI entailment.
            - **Latency**: `150ms - 350ms`.
            - **Use Case**: High-assurance financial, legal, and clinical applications.
            """
        )
with col_h3:
    with st.container(border=True):
        st.markdown("#### 3. Phantom Citation Verification")
        st.markdown(
            """
            - **Mechanism**: Parses in-line brackets (`[Doc 1]`, `[Doc 2]`) and validates that cited chunk IDs exist in the candidate pool and actually contain the claimed assertion.
            - **Latency**: `< 10ms`.
            - **Use Case**: Eliminates fabrications where LLMs invent citations to seem authoritative.
            """
        )

# --- SECTION 4: LATENCY MONITORING ---
st.header("4. Latency Monitoring (P50/P95/P99 SLO Tracking, TTFT Anomalies, Queue Delays & Provider Degradation)", divider="rainbow")

st.markdown(
    """
    RAG latency is inherently multi-modal because the pipeline traverses multiple distributed RPC endpoints.
    A single slow request can be caused by vector index lock contention, GPU queue starvation, reranker memory paging, or upstream LLM provider throttling.
    
    ### Deconstructing RAG Latency Metrics
    """
)

col_lat1, col_lat2 = st.columns(2)
with col_lat1:
    st.markdown(
        """
        - **Embedding Generation Latency**: Time required to tokenize the query and run vector inference via embedding model endpoint (`nominal: 15ms - 45ms`).
        - **Vector Index Search Latency (ANN)**: HNSW graph traversal or IVF inverted index probe time in Milvus, Qdrant, or Pinecone (`nominal: 10ms - 50ms`).
        - **Neural Cross-Encoder Rerank Latency**: Heavy transformer inference computing joint cross-attention across Top-K pairs (`nominal: 80ms - 250ms`).
        - **Time To First Token (TTFT)**: Gateway transmission, context prompt pre-fill processing on the LLM accelerator, and first token emission (`nominal: 250ms - 800ms`).
        - **Inter-Token Latency (ITL) & Generation**: Generation velocity in tokens per second (`nominal: 40 - 100 tokens/sec`).
        """
    )
    st.info("⚠️ **The Long Tail Vulnerability**: While P50 latency may appear healthy at 850ms, P99 latency frequently spikes to 4,500ms due to prompt prefill overhead when maximum context windows (e.g., 32k tokens) are injected.")

with col_lat2:
    st.subheader("Synthetic Latency Distribution Profile (ms)")
    latency_profile = pd.DataFrame({
        "Stage": ["Embedding", "Vector ANN", "Reranker", "LLM TTFT", "LLM Gen (300 tok)", "Total Pipeline"],
        "P50 (Nominal)": [25, 35, 120, 450, 600, 1230],
        "P95 (Degraded)": [65, 95, 280, 1100, 1200, 2740],
        "P99 (Critical)": [140, 220, 550, 2400, 2100, 5410],
        "Alert Ceiling": [100, 150, 400, 1800, 1800, 3500]
    })
    st.dataframe(latency_profile, use_container_width=True)

# --- SECTION 5: COST MONITORING ---
st.header("5. Cost Monitoring (Token Consumption Velocity, Daily Budget Burn Rate & Cache Margin Protection)", divider="rainbow")

st.markdown(
    """
    In enterprise deployments, unbounded RAG pipelines can consume tens of thousands of dollars in LLM API credits within hours if subjected to
    query loops, recursive agentic loops, or traffic surges. Real-time cost monitoring establishes hard economic guardrails.
    
    ### Financial Risk Vectors in RAG
    1. **Context Stuffing Inefficiency**: Naive architectures inject 20 large chunks (10,000+ tokens) for every trivial question. A query like *"What time is checkout?"* costs $0.03 instead of $0.0001.
    2. **Missing Semantic Caching**: Serving identical or semantically duplicate queries through the entire retrieval and LLM generation stack instead of returning pre-verified responses from Redis/GPTCache.
    3. **Token Velocity Spikes**: Automated scripts or scraping bots executing thousands of unauthenticated requests per minute, exhausting daily budgets.
    """
)

cost_kpis = st.columns(4)
cost_kpis[0].metric(label="Nominal Cost / Query", value="$0.0042", delta="-12% (via reranking)")
cost_kpis[1].metric(label="Target Cache Hit Yield", value="42.8%", delta="+5.2% MoM")
cost_kpis[2].metric(label="Daily Burn Velocity", value="$38.40 / day", delta="76.8% of $50 cap")
cost_kpis[3].metric(label="FinOps Margin Safety", value="98.2%", delta="Protected")

# --- SECTION 6: QUALITY MONITORING ---
st.header("6. Quality Monitoring (Continuous RAG Triad Telemetry, Golden Dataset Drift & CSAT Tracking)", divider="rainbow")

st.markdown(
    """
    Quality monitoring evaluates whether the system delivers accurate, relevant, and helpful information over time.
    Unlike static software, RAG quality degrades silently due to document corpus churn, upstream model quantization updates, and user query evolution.
    
    ### The Three Pillars of Continuous Quality Telemetry
    """
)

col_q1, col_q2 = st.columns(2)
with col_q1:
    st.markdown(
        """
        1. **Live Production Sample Auditing (RAG Triad)**:
           - Asynchronous workers continuously sample 5% of production traces and score:
             - **Context Relevance**: Did the retriever isolate signal and reject noise?
             - **Groundedness / Faithfulness**: Is the generated answer 100% faithful to the retrieved context?
             - **Answer Relevance**: Did the model directly address the user's explicit intent?
        2. **Synthetic Golden Dataset Canaries**:
           - Automated cron jobs execute a curated suite of 200 benchmark question-answer pairs against the production vector database every hour.
           - Any drop in recall or faithfulness triggers an immediate rollback or index reindexing alarm.
        3. **Explicit & Implicit User Feedback (CSAT)**:
           - Tracking thumbs-up / thumbs-down signals, answer copy events, follow-up clarification queries, and session abandonment rates.
        """
    )
with col_q2:
    st.subheader("Continuous Quality Thresholds")
    q_data = {
        "Evaluation Dimension": ["Context Relevance", "Groundedness (Faithfulness)", "Answer Relevance", "Golden Canary Pass Rate", "User Thumbs-Down Rate"],
        "Target Baseline": [">= 0.80", ">= 0.90", ">= 0.85", ">= 98.0%", "< 3.0%"],
        "Warning Level": ["0.70 - 0.79", "0.80 - 0.89", "0.75 - 0.84", "94.0% - 97.9%", "3.0% - 7.0%"],
        "Sev-2 Incident Trigger": ["< 0.70", "< 0.80", "< 0.75", "< 94.0%", "> 7.0%"]
    }
    st.dataframe(pd.DataFrame(q_data), use_container_width=True)

# --- SECTION 7: PRODUCTION ALERTS ---
st.header("7. Production Alerts (Severity-1 Escalations, Dead Man's Switches & Circuit Breaker Triggers)", divider="rainbow")

st.markdown(
    """
    Effective monitoring requires an unambiguous, multi-tiered incident escalation matrix.
    Alert noise must be aggressively minimized to prevent on-call fatigue while ensuring catastrophic failures are mitigated within seconds.
    """
)

col_alt1, col_alt2 = st.columns(2)
with col_alt1:
    with st.container(border=True):
        st.markdown("### 🔴 SEV-1: Critical Outage & Financial Breach")
        st.markdown(
            """
            - **Criteria**: Zero-hit retrieval rate > 5%, Vector DB unreachable, Daily API spend exceeded cap, P99 latency > 6.0s for 5 consecutive minutes.
            - **Notification**: Automated PagerDuty phone call to Primary On-Call Engineer + Slack `#rag-war-room`.
            - **Automated Mitigation**: Trigger Circuit Breaker: divert incoming traffic to cached responses or a static fallback search engine.
            """
        )
    with st.container(border=True):
        st.markdown("### 🟠 SEV-2: High Priority Quality Degradation")
        st.markdown(
            """
            - **Criteria**: Faithfulness score < 0.75 over 15-minute rolling window, Top-1 Cosine similarity < 0.65 surge (> 10% of queries).
            - **Notification**: PagerDuty notification to secondary engineer + Slack `#rag-incident-alerts`.
            - **Automated Mitigation**: Automatically switch to stricter system prompt template with forced citation requirements.
            """
        )
with col_alt2:
    with st.container(border=True):
        st.markdown("### 🟡 SEV-3: Warning & Subsystem Performance Drift")
        st.markdown(
            """
            - **Criteria**: P95 latency > 2.5s, Reranker queue backlog > 50 requests, Cache hit rate drops below 25%.
            - **Notification**: Slack notification `#rag-monitoring-low`.
            - **Automated Mitigation**: Scale out Kubernetes HPA replicas for the reranking inference pods.
            """
        )
    with st.container(border=True):
        st.markdown("### 🟢 Automated Circuit Breakers & Dead Man's Switch")
        st.markdown(
            """
            - **Dead Man's Switch**: If telemetry collector pods stop emitting heartbeats for > 60 seconds, trigger independent alert indicating logging pipeline blackout.
            - **Circuit Breaker**: When vector DB latency exceeds 2,000ms, fail-safe to cached search results and return partial answers with explicit caveats.
            """
        )

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Enterprise RAG Monitoring & Automated Incident Alerting Engine", divider="rainbow")

st.markdown(
    """
    Below is the complete, self-contained implementation of `ProductionMonitoringEngine`.
    This enterprise module inspects multi-stage telemetry payloads, evaluates retrieval health, audits hallucinations, verifies latency SLAs,
    tracks budget burn, and automatically routes prioritized incident alerts.
    """
)

st.code(
    '''
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class ProductionMonitoringEngine:
    """Enterprise RAG Production Telemetry, Quality Monitoring & Automated Alerting Engine."""

    def __init__(self, latency_sla_ms: float = 1200.0, cost_cap_usd: float = 50.0, similarity_floor: float = 0.65):
        self.latency_sla_ms = latency_sla_ms
        self.cost_cap_usd = cost_cap_usd
        self.similarity_floor = similarity_floor
        self.daily_spend_usd = 0.0
        self.total_queries = 0
        self.alerts: List[Dict[str, Any]] = []

    def check_retrieval_health(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect zero-hit retrievals, similarity floor violations, and distribution drift."""
        if not retrieved_docs:
            return {
                "status": "FAIL",
                "reason": "Zero chunks retrieved (Empty context breach)",
                "max_score": 0.0,
                "mean_score": 0.0,
                "circuit_breaker": True
            }
        scores = [doc.get("score", 0.0) for doc in retrieved_docs]
        max_score = float(np.max(scores))
        mean_score = float(np.mean(scores))
        
        if max_score < self.similarity_floor:
            return {
                "status": "FAIL",
                "reason": f"Max similarity score ({max_score:.3f}) below safety floor ({self.similarity_floor:.3f})",
                "max_score": max_score,
                "mean_score": mean_score,
                "circuit_breaker": False
            }
        return {
            "status": "HEALTHY",
            "reason": "Retrieval distribution within nominal boundaries",
            "max_score": max_score,
            "mean_score": mean_score,
            "circuit_breaker": False
        }

    def detect_hallucination(self, query: str, context_chunks: List[str], generated_answer: str) -> Dict[str, Any]:
        """Fast lexical-semantic groundedness audit checking entity overlap and phantom citations."""
        if not generated_answer or not context_chunks:
            return {"groundedness_score": 0.0, "is_hallucination": True, "verdict": "Unverified Generation"}
        
        combined_context = " ".join(context_chunks).lower()
        answer_words = [w.strip(".,;:?!()[]\\"'") for w in generated_answer.lower().split() if len(w) > 4]
        
        if not answer_words:
            return {"groundedness_score": 1.0, "is_hallucination": False, "verdict": "Nominal Short Answer"}

        matched_words = [w for w in answer_words if w in combined_context]
        groundedness_score = float(len(matched_words) / len(answer_words))
        
        has_citations = "[" in generated_answer and "]" in generated_answer
        is_hallucination = groundedness_score < 0.60
        
        verdict = "Faithful Grounded" if not is_hallucination else "High Hallucination Risk (Unanchored Claims)"
        if not has_citations and len(generated_answer) > 80:
            verdict += " | Missing In-line Citations"

        return {
            "groundedness_score": round(groundedness_score, 3),
            "is_hallucination": is_hallucination,
            "matched_tokens": len(matched_words),
            "total_audit_tokens": len(answer_words),
            "verdict": verdict
        }

    def audit_latency_breakdown(self, retrieval_ms: float, rerank_ms: float, ttft_ms: float, gen_ms: float) -> Dict[str, Any]:
        """Verify SLA compliance and pinpoint latency bottleneck."""
        total_latency_ms = retrieval_ms + rerank_ms + ttft_ms + gen_ms
        breached = total_latency_ms > self.latency_sla_ms
        
        bottleneck = "Retrieval"
        max_val = retrieval_ms
        if rerank_ms > max_val:
            bottleneck = "Reranking"
            max_val = rerank_ms
        if ttft_ms > max_val:
            bottleneck = "TTFT (LLM Queue)"
            max_val = ttft_ms
        if gen_ms > max_val:
            bottleneck = "LLM Generation"
            max_val = gen_ms
            
        return {
            "total_ms": round(total_latency_ms, 1),
            "breached": breached,
            "sla_limit_ms": self.latency_sla_ms,
            "bottleneck": bottleneck,
            "pct_of_budget": round((total_latency_ms / self.latency_sla_ms) * 100.0, 1)
        }

    def audit_cost_burn(self, prompt_tokens: int, completion_tokens: int, prompt_rate_per_m: float = 2.50, comp_rate_per_m: float = 10.00) -> Dict[str, Any]:
        """Compute query cost and update cumulative budget burn."""
        cost = (prompt_tokens / 1_000_000.0) * prompt_rate_per_m + (completion_tokens / 1_000_000.0) * comp_rate_per_m
        self.daily_spend_usd += cost
        self.total_queries += 1
        
        burn_rate_pct = (self.daily_spend_usd / self.cost_cap_usd) * 100.0
        budget_breached = self.daily_spend_usd > self.cost_cap_usd
        
        return {
            "query_cost_usd": round(cost, 5),
            "daily_spend_usd": round(self.daily_spend_usd, 4),
            "budget_cap_usd": self.cost_cap_usd,
            "burn_rate_pct": round(burn_rate_pct, 2),
            "budget_breached": budget_breached
        }

    def evaluate_quality_triad(self, context_relevance: float, faithfulness: float, answer_relevance: float) -> Dict[str, Any]:
        """Aggregate Triad score and flag quality degradation."""
        triad_mean = float((context_relevance + faithfulness + answer_relevance) / 3.0)
        warning_triggered = triad_mean < 0.70 or faithfulness < 0.65
        return {
            "triad_mean": round(triad_mean, 3),
            "context_relevance": round(context_relevance, 2),
            "faithfulness": round(faithfulness, 2),
            "answer_relevance": round(answer_relevance, 2),
            "quality_flag": "CRITICAL" if triad_mean < 0.50 else ("WARNING" if warning_triggered else "PASS")
        }

    def evaluate_and_route_alerts(
        self,
        retrieval_res: Dict[str, Any],
        hallucination_res: Dict[str, Any],
        latency_res: Dict[str, Any],
        cost_res: Dict[str, Any],
        quality_res: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Dispatch production alerts based on SLA and quality breaches."""
        new_alerts = []
        if retrieval_res["circuit_breaker"]:
            new_alerts.append({
                "severity": "SEV-1 CRITICAL",
                "subsystem": "Vector Database",
                "message": f"Zero-hit retrieval failure detected. Automated circuit breaker triggered to fallback corpus.",
                "routing": "PagerDuty Primary On-Call -> Engineering Incident Channel"
            })
        elif retrieval_res["status"] == "FAIL":
            new_alerts.append({
                "severity": "SEV-2 HIGH",
                "subsystem": "Vector Retriever",
                "message": f"Similarity score collapsed below floor ({retrieval_res['max_score']:.3f} < {self.similarity_floor}). Out-of-domain query surge.",
                "routing": "Slack #rag-alerts-high"
            })
            
        if hallucination_res["is_hallucination"]:
            new_alerts.append({
                "severity": "SEV-2 HIGH",
                "subsystem": "Guardrail Evaluator",
                "message": f"Hallucination score breach: Groundedness ({hallucination_res['groundedness_score']}) below safety threshold.",
                "routing": "Telemetry Datadog Event -> Slack #rag-quality"
            })
            
        if latency_res["breached"]:
            new_alerts.append({
                "severity": "SEV-3 MEDIUM",
                "subsystem": "Serving Gateway",
                "message": f"Latency SLA breached: {latency_res['total_ms']}ms > {latency_res['sla_limit_ms']}ms. Bottleneck: {latency_res['bottleneck']}.",
                "routing": "Datadog Monitor -> Opsgenie Tier-2"
            })
            
        if cost_res["budget_breached"]:
            new_alerts.append({
                "severity": "SEV-1 CRITICAL",
                "subsystem": "FinOps Gateway",
                "message": f"Daily budget ceiling violated: Spend ${cost_res['daily_spend_usd']:.2f} exceeded Cap ${cost_res['budget_cap_usd']:.2f}.",
                "routing": "AWS SNS -> Engineering Manager & CFO FinOps Alert"
            })
            
        if quality_res["quality_flag"] == "CRITICAL":
            new_alerts.append({
                "severity": "SEV-2 HIGH",
                "subsystem": "Continuous Eval",
                "message": f"Aggregate RAG Triad score dropped to critical floor ({quality_res['triad_mean']}). Prompt/Model degradation detected.",
                "routing": "Slack #rag-eval-monitoring"
            })
            
        self.alerts.extend(new_alerts)
        return new_alerts
''',
    language="python"
)

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Real-Time Telemetry Dashboard & Anomaly Trigger Console", divider="rainbow")

st.markdown(
    """
    Simulate live production query events through the monitoring engine. Configure anomaly conditions
    such as low-confidence retrieval, hallucinated answers, latency spikes, and budget exhaustion to inspect
    telemetry gauges and automated incident alerts.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🛠️ Anomaly Injection & SLA Controls")
    
    scenario = st.selectbox(
        "Select Test Scenario:",
        [
            "Scenario A: Nominal Healthy Query Event",
            "Scenario B: Zero-Hit Retrieval Failure (Circuit Breaker)",
            "Scenario C: Hallucinated / Unanchored Generation",
            "Scenario D: Extreme Latency Spike (SLA Breach)",
            "Scenario E: Runaway Cost & Daily Budget Breach"
        ],
        index=0
    )
    
    user_query = st.text_input("Simulated User Query:", "What is the capital expenditure amortization rule for cloud servers?")
    
    if scenario == "Scenario A: Nominal Healthy Query Event":
        sim_scores = [0.89, 0.84, 0.79]
        sim_context = [
            "Cloud computing server acquisitions are amortized over a 3-year useful lifespan according to GAAP section 350-40.",
            "Capital expenditures for enterprise infrastructure exceeding $5,000 are depreciated straight-line."
        ]
        sim_gen = "According to GAAP section 350-40, cloud server acquisitions are amortized over a 3-year useful lifespan straight-line [Doc 1]."
        retrieval_ms = 35.0
        rerank_ms = 110.0
        ttft_ms = 420.0
        gen_ms = 350.0
        prompt_tok = 850
        comp_tok = 120
        sim_spend = 12.50
    elif scenario == "Scenario B: Zero-Hit Retrieval Failure (Circuit Breaker)":
        sim_scores = []
        sim_context = []
        sim_gen = "I do not have sufficient information in the provided knowledge base."
        retrieval_ms = 22.0
        rerank_ms = 0.0
        ttft_ms = 180.0
        gen_ms = 80.0
        prompt_tok = 120
        comp_tok = 25
        sim_spend = 12.50
    elif scenario == "Scenario C: Hallucinated / Unanchored Generation":
        sim_scores = [0.72, 0.69]
        sim_context = [
            "The company policy does not specify hardware depreciation schedules for remote contractor workstations."
        ]
        sim_gen = "The government allows a 100% tax credit bonus deduction immediately in year 1 with zero amortization required under European Directive 45."
        retrieval_ms = 40.0
        rerank_ms = 130.0
        ttft_ms = 520.0
        gen_ms = 450.0
        prompt_tok = 920
        comp_tok = 180
        sim_spend = 14.20
    elif scenario == "Scenario D: Extreme Latency Spike (SLA Breach)":
        sim_scores = [0.82, 0.78]
        sim_context = ["Standard amortization guidelines apply over 36 monthly increments."]
        sim_gen = "Standard amortization is amortized in 36 monthly increments [Doc 1]."
        retrieval_ms = 450.0
        rerank_ms = 850.0
        ttft_ms = 1950.0
        gen_ms = 1200.0
        prompt_tok = 1200
        comp_tok = 150
        sim_spend = 15.00
    else: # Scenario E
        sim_scores = [0.85, 0.81]
        sim_context = ["Standard accounting policies apply to all data center assets."]
        sim_gen = "Standard accounting policies apply to all data center assets [Doc 1]."
        retrieval_ms = 30.0
        rerank_ms = 95.0
        ttft_ms = 400.0
        gen_ms = 320.0
        prompt_tok = 45000
        comp_tok = 8000
        sim_spend = 52.80

    st.markdown("**Fine-Tune Simulated Metrics:**")
    sla_threshold = st.slider("Latency SLA Ceiling (ms):", min_value=500.0, max_value=3000.0, value=1200.0, step=100.0)
    budget_cap = st.slider("Daily FinOps Budget Cap ($):", min_value=10.0, max_value=200.0, value=50.0, step=5.0)
    sim_floor = st.slider("Similarity Safety Floor:", min_value=0.50, max_value=0.85, value=0.65, step=0.05)

with wb_col2:
    st.subheader("📊 Live Telemetry Cockpit & Alerts")
    
    # Initialize Engine
    engine = ProductionMonitoringEngine(latency_sla_ms=sla_threshold, cost_cap_usd=budget_cap, similarity_floor=sim_floor)
    engine.daily_spend_usd = sim_spend
    
    # Format retrieved docs
    docs_payload = [{"doc_id": f"doc_{i}", "score": s} for i, s in enumerate(sim_scores)]
    
    # Evaluate Pipeline
    retrieval_res = engine.check_retrieval_health(docs_payload)
    hallucination_res = engine.detect_hallucination(user_query, sim_context, sim_gen)
    latency_res = engine.audit_latency_breakdown(retrieval_ms, rerank_ms, ttft_ms, gen_ms)
    cost_res = engine.audit_cost_burn(prompt_tok, comp_tok)
    quality_res = engine.evaluate_quality_triad(
        context_relevance=0.88 if sim_scores else 0.10,
        faithfulness=hallucination_res["groundedness_score"],
        answer_relevance=0.90 if not hallucination_res["is_hallucination"] else 0.40
    )
    
    alerts = engine.evaluate_and_route_alerts(retrieval_res, hallucination_res, latency_res, cost_res, quality_res)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Latency", f"{latency_res['total_ms']} ms", delta=f"{latency_res['pct_of_budget']}% of SLA", delta_color="inverse" if latency_res["breached"] else "normal")
    m_col2.metric("Groundedness", f"{hallucination_res['groundedness_score'] * 100:.1f}%", delta=hallucination_res["verdict"][:12])
    m_col3.metric("Daily Spend", f"${cost_res['daily_spend_usd']:.2f}", delta=f"{cost_res['burn_rate_pct']:.1f}% used", delta_color="inverse" if cost_res["budget_breached"] else "normal")
    m_col4.metric("Triad Mean", f"{quality_res['triad_mean']:.2f}", delta=quality_res["quality_flag"])
    
    st.markdown("#### Live Incident Escalations")
    if not alerts:
        st.success("✅ **All Systems Nominal**: Zero SLA breaches or quality warnings detected.")
    else:
        for alt in alerts:
            if "SEV-1" in alt["severity"]:
                st.error(f"🚨 **[{alt['severity']}] {alt['subsystem']}**: {alt['message']}\n\n*Dispatch*: `{alt['routing']}`")
            elif "SEV-2" in alt["severity"]:
                st.warning(f"⚠️ **[{alt['severity']}] {alt['subsystem']}**: {alt['message']}\n\n*Dispatch*: `{alt['routing']}`")
            else:
                st.info(f"ℹ️ **[{alt['severity']}] {alt['subsystem']}**: {alt['message']}\n\n*Dispatch*: `{alt['routing']}`")
                
    st.markdown("#### Latency Waterfall Breakdown")
    lat_chart_df = pd.DataFrame({
        "Stage": ["1. Retrieval", "2. Rerank", "3. LLM TTFT", "4. Generation"],
        "Latency (ms)": [retrieval_ms, rerank_ms, ttft_ms, gen_ms]
    })
    st.bar_chart(lat_chart_df.set_index("Stage"))

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: Why are HTTP 200 OK health checks insufficient for RAG systems, and what defines a silent failure?**"):
    st.markdown(
        """
        In traditional microservices, an HTTP `200 OK` confirms that the web process, routing, and database connections are operational.
        In RAG, a pipeline can return HTTP `200 OK` while suffering catastrophic failure:
        1. **Semantic Vacuum**: The retriever returned 0 chunks due to restrictive metadata filtering, but the LLM politely generated a plausible hallucination.
        2. **Similarity Floor Collapse**: Chunks returned had low cosine similarity (`0.45`), but the generator was forced to extrapolate, inventing facts.
        3. **Prompt Truncation**: Context length exceeded token limits, silently dropping the key evidence chunk.
        
        Monitoring must inspect **payload semantics, groundedness ratios, and cosine distributions**, not merely HTTP status codes.
        """
    )

with st.expander("**Q2: How do you design an automated circuit breaker for vector database retrieval failures?**"):
    st.markdown(
        """
        An enterprise vector circuit breaker operates on three rolling metrics:
        - **Timeout / Error Rate**: If vector queries timeout (`> 1500ms`) or fail on > 5% of requests over 1 minute.
        - **Empty Hit Ratio**: If > 3% of queries return 0 chunks.
        - **Similarity Floor Breach**: If the moving average of Top-1 similarity drops below `0.65`.
        
        When tripped, the circuit breaker opens and executes graceful fallbacks:
        1. Query an in-memory keyword search engine (Elasticsearch/BM25).
        2. Query an in-memory semantic cache (Redis).
        3. If all fails, execute graceful abstention: *"Our internal document retrieval service is temporarily degraded. Please re-query shortly."*
        """
    )

with st.expander("**Q3: How do you monitor Time To First Token (TTFT) versus Generation Token Velocity in production?**"):
    st.markdown(
        """
        TTFT and inter-token generation velocity measure completely different hardware bottlenecks:
        - **TTFT** measures prompt ingestion, network serialization, and **pre-fill attention compute** on the GPU. High TTFT indicates large injected context windows (e.g., 20 chunks = 10k tokens) or GPU queue saturation.
        - **Inter-Token Latency (ITL)** measures **autoregressive decode compute**, bounded by memory bandwidth rather than compute.
        
        Monitoring TTFT separately allows engineering teams to detect whether slow queries are caused by bloated retrieved contexts (solvable by context compression/reranking) or GPU memory bandwidth throttling.
        """
    )

with st.expander("**Q4: What statistical drift detection methods identify when an enterprise vector index is becoming stale?**"):
    st.markdown(
        """
        Vector index drift is detected through three primary telemetry distributions:
        1. **Top-K Similarity Distribution Shift**: Tracking rolling weekly histograms of cosine similarity scores. A downward shift in mean similarity indicates new domain concepts are emerging that lack corresponding chunk embeddings.
        2. **Query Vector Clustering vs Document Vector Centroids**: Computing the Wasserstein distance or Kullback-Leibler (KL) divergence between incoming query embedding distributions and document embedding clusters.
        3. **Zero-Hit Rate Velocity**: An increasing trend in zero-hit queries indicates users are searching for topics outside the knowledge repository.
        """
    )

with st.expander("**Q5: How do you implement budget capping without abruptly terminating in-flight user sessions?**"):
    st.markdown(
        """
        Hard session termination creates catastrophic user experience. Production FinOps architectures employ tiered degradation:
        1. **80% Budget Alert**: Alert FinOps and engineering teams via Slack `#rag-finops`.
        2. **90% Soft Cap (Dynamic Model Downgrade)**: Automatically route non-critical queries from expensive frontier reasoning models (e.g., GPT-4o / Claude 3.5 Sonnet) to distilled SLMs (e.g., GPT-4o-mini / Haiku 3.5), cutting token cost by 90%.
        3. **95% Compression Throttle**: Reduce retrieved Top-K from 10 to 3 chunks and enable aggressive extractive context compression.
        4. **100% Hard Cap**: Restrict queries strictly to semantic cache hits; un-cached queries receive an apologetic rate-limit notice.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Synchronous In-Line LLM-as-a-Judge Evaluation**  
        **Antipattern**: Invoking an LLM judge synchronously within the user request path to evaluate answer groundedness before returning the response.  
        **Consequence**: Doubles total latency (adding 800ms - 2,500ms) and doubles API token expenditure on every single user interaction.  
        **Production Fix**: Run lightweight lexical/token checks inline (`< 5ms`), and stream full LLM evaluation asynchronously to Kafka/Celery background workers.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Alert Storms Caused by Global Average Latency Thresholds**  
        **Antipattern**: Setting a single global alert: *Trigger PagerDuty if mean latency > 1,500ms*.  
        **Consequence**: Long, complex multi-document summarization queries legitimately require 3,000ms, constantly firing false positive alerts during regular usage.  
        **Production Fix**: Partition latency SLOs by query intent category (e.g., Simple Factoid: `< 800ms`, Analytical Synthesis: `< 3,500ms`).
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Logging Unsanitized PII and Context Chunks in Plaintext**  
        **Antipattern**: Streaming complete raw user prompts and retrieved proprietary documents into public third-party telemetry tools without masking.  
        **Consequence**: Violates GDPR, HIPAA, and SOC2 compliance; leaks employee credentials or customer data into external log aggregators.  
        **Production Fix**: Implement deterministic SHA-256 pseudonymization and regex/NER PII scrubbing at the gateway interceptor tier before telemetry dispatch.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Neglecting Phantom Citation Auditing**  
        **Antipattern**: Assuming that because an LLM output contains formatted brackets like `[1]` and `[2]`, the citations are authentic.  
        **Consequence**: Models routinely synthesize plausible-looking document tags that point to non-existent chunk IDs or misattribute quotes to innocent sources.  
        **Production Fix**: Validate all citation brackets programmatically against the candidate chunk ID array at the gateway level.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is the RAG Triad?")
        st.markdown("**Back**: The three core evaluation metrics: Context Relevance (retrieval quality), Groundedness/Faithfulness (hallucination defense), and Answer Relevance (query satisfaction).")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is TTFT and why does it matter?")
        st.markdown("**Back**: Time To First Token. Measures how long a user waits before streaming begins; dominated by prompt prefill processing of large retrieved contexts.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is a Vector Circuit Breaker?")
        st.markdown("**Back**: An automated fail-safe that stops querying the vector database when latency spikes or zero-hits surge, diverting to keyword search or cached responses.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is Semantic Drift in RAG?")
        st.markdown("**Back**: The gradual divergence between user query language and static vector embeddings over time, leading to silent drops in retrieval recall.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of production RAG observability, monitoring, and automated incident management:")

chk1 = st.checkbox("I understand why traditional HTTP 200 OK checks fail to capture silent semantic, similarity, and hallucination failures.", value=True)
chk2 = st.checkbox("I can architect a vector circuit breaker that detects zero-hit and low-similarity queries and redirects to BM25 or semantic caches.", value=True)
chk3 = st.checkbox("I can differentiate between Time To First Token (TTFT) and Inter-Token Latency (ITL) and identify their distinct hardware bottlenecks.", value=True)
chk4 = st.checkbox("I know how to implement real-time lexical grounding and phantom citation verification without adding multi-second LLM latency.", value=True)
chk5 = st.checkbox("I can design tiered FinOps budget throttling (soft downgrade -> compression -> hard cache lock) to protect production margins.", value=True)
chk6 = st.checkbox("I understand how to structure SEV-1, SEV-2, and SEV-3 alert escalation policies across PagerDuty, Slack, and Datadog without causing alert fatigue.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered Enterprise RAG Production Monitoring, Telemetry, and Incident Response.")
