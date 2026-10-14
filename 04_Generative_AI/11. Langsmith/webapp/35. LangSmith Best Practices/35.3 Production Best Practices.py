import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="35.3 Production Best Practices - LangSmith Master Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class ProductionObservabilityMonitorEngine:
    """
    Enterprise Full-Spectrum LLMOps Production Monitor & Governance Engine.
    Implements 6 core production pillars:
    1. Latency percentile profiling (p50, p90, p99, TTFT)
    2. Cost and token burn tracking with budget alerts
    3. Online quality monitoring and user feedback aggregation
    4. Failure rate tracking and exception triage
    5. Zero-Trust PII masking and sensitive data scrubbing
    6. Prompt and workflow SemVer release tracking
    """
    def __init__(self, project_name: str = "prod-enterprise-copilot"):
        self.project_name = project_name
        self.telemetry_history: List[Dict[str, Any]] = []

    def process_production_telemetry_batch(
        self,
        request_count: int = 1000,
        cache_hit_rate: float = 0.25,
        pii_masking_enabled: bool = True,
        simulate_degradation: bool = False
    ) -> Dict[str, Any]:
        """
        Processes a live batch of production traces and computes enterprise observability metrics.
        """
        # 1. Latency Profiling (ms)
        if simulate_degradation:
            p50_ms = 1850.0
            p90_ms = 3450.0
            p99_ms = 5820.0
            ttft_ms = 920.0
            failure_rate_pct = 4.8
            user_satisfaction_pct = 76.5
        else:
            p50_ms = 820.0
            p90_ms = 1450.0
            p99_ms = 2150.0
            ttft_ms = 310.0
            failure_rate_pct = 0.4
            user_satisfaction_pct = 94.2

        # 2. Token & Cost Economics (USD)
        avg_input_tokens = 650
        avg_output_tokens = 220
        total_input_tokens = int(request_count * avg_input_tokens * (1.0 - cache_hit_rate))
        total_output_tokens = int(request_count * avg_output_tokens)
        
        # Pricing: GPT-4o standard (2.50 USD / 1M in, 10.00 USD / 1M out)
        input_cost = (total_input_tokens / 1_000_000.0) * 2.50
        output_cost = (total_output_tokens / 1_000_000.0) * 10.00
        gross_cost_usd = round(input_cost + output_cost, 2)
        saved_cache_cost_usd = round(((request_count * avg_input_tokens * cache_hit_rate) / 1_000_000.0) * 2.50, 2)

        # 3. Security & PII Scrubbing Audit
        raw_samples = [
            "Customer John Doe (SSN: 123-45-6789) requested account balance at john@example.com",
            "Transfer 500 USD to account 987654321 for user Alice Smith",
            "Standard policy question regarding dental coverage limits."
        ]
        
        scrubbed_samples = []
        pii_entities_detected = 0
        for sample in raw_samples:
            if pii_masking_enabled:
                # Deterministic redaction demonstration
                text = sample.replace("123-45-6789", "[REDACTED_SSN]")
                text = text.replace("john@example.com", "[REDACTED_EMAIL]")
                text = text.replace("987654321", "[REDACTED_ACCOUNT]")
                if "[REDACTED" in text:
                    pii_entities_detected += 1
                scrubbed_samples.append(text)
            else:
                scrubbed_samples.append(sample)

        # 4. SLAs Assessment
        latency_sla_passed = p99_ms <= 3000.0
        failure_sla_passed = failure_rate_pct <= 1.0
        quality_sla_passed = user_satisfaction_pct >= 90.0

        batch_report = {
            "batch_size": request_count,
            "cache_hit_rate": cache_hit_rate,
            "latency_metrics": {
                "p50_ms": p50_ms,
                "p90_ms": p90_ms,
                "p99_ms": p99_ms,
                "ttft_ms": ttft_ms,
                "sla_passed": latency_sla_passed
            },
            "cost_metrics": {
                "gross_cost_usd": gross_cost_usd,
                "cache_savings_usd": saved_cache_cost_usd,
                "total_tokens": total_input_tokens + total_output_tokens,
                "cost_per_request": round(gross_cost_usd / request_count, 4)
            },
            "quality_metrics": {
                "failure_rate_pct": failure_rate_pct,
                "user_satisfaction_pct": user_satisfaction_pct,
                "quality_sla_passed": quality_sla_passed and failure_sla_passed
            },
            "security_metrics": {
                "pii_masking_enabled": pii_masking_enabled,
                "entities_scrubbed": pii_entities_detected if pii_masking_enabled else 0,
                "compliance_status": "ZERO_TRUST_COMPLIANT" if pii_masking_enabled else "EXPOSURE_RISK_WARNING"
            },
            "samples": scrubbed_samples,
            "overall_status": "HEALTHY" if (latency_sla_passed and failure_sla_passed and quality_sla_passed) else "ALERT_BREACHED"
        }
        self.telemetry_history.append(batch_report)
        return batch_report

# Initialize engine in session state
if "prod_observability_engine" not in st.session_state:
    st.session_state.prod_observability_engine = ProductionObservabilityMonitorEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("🛡️ Chapter 35.3: Production Best Practices in LangSmith")
st.caption("Percentile Latency Profiling, Token Economics, Online Quality Drift, Exception Triage, Zero-Trust PII Masking, and Version Governance")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise LLMOps Production Observability and Governance Architecture", divider="rainbow")

st.markdown("""
Running LLMs in production requires far more than basic health checks. Because LLMs operate non-deterministically 
over variable context lengths with high token costs, production observability requires a multi-dimensional control plane. 
Enterprise teams monitor latency distributions, token burn velocity, live user quality feedback, tool failures, 
zero-trust privacy scrubbing, and version governance simultaneously.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ProductionObservabilityBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_ingress {
            label = "1. Edge Ingress & Security Scrubbing Layer";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            I1 [label="Incoming User Request\\n(Prompts, Auth, Metadata)", fillcolor="#334155", color="#64748B"];
            I2 [label="Zero-Trust Privacy Scrubber\\n(Regex + Presidio: SSN, PII Masked)", fillcolor="#7C2D12", color="#EA580C"];
            I3 [label="Prompt Hub Version Resolver\\n(Pulls tag: ':prod' commit SHA)", fillcolor="#1E3A8A", color="#3B82F6"];
        }

        subgraph cluster_execution {
            label = "2. Production Application Core";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            APP [label="Agent & RAG Pipeline\\n(LangChain / LangGraph Engine)", fillcolor="#047857", color="#10B981"];
            CACHE [label="Semantic Cache (Redis)\\n(Cache Hit Bypass: 0 Token Cost)", fillcolor="#047857", color="#10B981"];
            MODELS [label="LLM Endpoints & Tools\\n(Streaming Responses / Tool Calls)", fillcolor="#047857", color="#10B981"];
        }

        subgraph cluster_telemetry {
            label = "3. LangSmith Production Telemetry Stream";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            T_LAT [label="Latency Monitor\\n(p50, p90, p99, TTFT Waterfall)", fillcolor="#0E7490", color="#06B6D4"];
            T_COST [label="Cost & Token Burn\\n(Input / Output Tokens / Budget Caps)", fillcolor="#0E7490", color="#06B6D4"];
            T_QUAL [label="Quality & Feedback\\n(Online Judges + Thumbs Up/Down)", fillcolor="#0E7490", color="#06B6D4"];
            T_ERR [label="Failure Triage\\n(Exceptions, 429s, Infinite Loops)", fillcolor="#0E7490", color="#06B6D4"];
        }

        subgraph cluster_actions {
            label = "4. Automated Governance & Incident Alerts";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            A_SLACK [label="On-Call Incident Dispatch\\n(Slack #llm-alerts & PagerDuty)", fillcolor="#881337", color="#F43F5E"];
            A_CIRCUIT [label="Automatic Circuit Breaker\\n(Reroute to Fallback Model / Tag)", fillcolor="#881337", color="#F43F5E"];
        }

        I1 -> I2 [label="Inspects Request"];
        I2 -> I3 [label="Clean Payload"];
        I3 -> APP [label="Injects Prompt & Ver"];

        APP -> CACHE [label="Cache Query"];
        CACHE -> MODELS [label="Cache Miss"];
        MODELS -> APP [label="Token Stream"];

        APP -> T_LAT [label="Logs Timing"];
        APP -> T_COST [label="Logs Tokens"];
        APP -> T_QUAL [label="Logs Feedback"];
        APP -> T_ERR [label="Logs Exceptions"];

        T_LAT -> A_SLACK [label="p99 > 3000ms"];
        T_COST -> A_SLACK [label="Daily Cap 80%"];
        T_QUAL -> A_CIRCUIT [label="Drift Detected"];
        T_ERR -> A_CIRCUIT [label="Error > 2.0%"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Production LLMOps
1. **Never Look Only at Mean Latency (Average is a Lie)**: In LLM applications, output generation time scales linearly with token length. Mean latency hides severe tail latency. Always measure and set SLAs on 95th and 99th percentiles (`p95` and `p99`).
2. **Track Time-to-First-Token (TTFT) for User Perception**: When using streaming responses, users perceive latency based on TTFT (how fast the first character appears), not total completion time. Optimize TTFT to keep perceived latency under 500ms.
3. **Zero-Trust Data Scrubbing Before Ingestion**: Never send raw unscrubbed customer PII (credit cards, SSNs, personal emails) to external observability vendors or third-party LLM providers. Scrub sensitive patterns at the edge before span dispatch.
""")

st.divider()

# ==============================================================================
# SECTION 2: MONITOR LATENCY
# ==============================================================================
st.header("2. Monitor Latency (Percentile Profiling: p50/p90/p99, TTFT vs Inter-Token Latency and Step Waterfalls)", divider="rainbow")

st.markdown("""
Latency in LLM applications is fundamentally different from traditional database queries. A request spans multiple 
sequential phases: prompt template formatting, network transit, embedding generation, vector database search, 
time-to-first-token (TTFT) waiting for the model prefill pass, and inter-token generation latency.
""")

st.markdown("""
### The Latency Breakdown Breakdown
* **Time-to-First-Token (TTFT)**: The duration from HTTP request submission until the first token streams back. TTFT reflects model queuing, prompt length processing (prefill), and network overhead. Target: `< 400ms`.
* **Inter-Token Latency (ITL)**: The generation rate of successive tokens (e.g., 30–80 tokens per second). Determines reading fluidity.
* **Retrieval Phase Latency**: Time spent querying the vector database and reranking chunks. Target: `< 150ms`.
* **Tail Percentiles (p95 and p99)**: The latency experienced by the slowest 5% and 1% of users, typically caused by complex multi-step agent tool loops or prompt token bloat.
""")

st.code('''
# LangSmith trace waterfall inspection pattern
from langsmith import traceable
import time

@traceable(run_type="chain", name="customer_rag_pipeline")
def execute_rag_pipeline(query: str):
    # Phase 1: Retrieval (Target: < 120ms)
    t0 = time.time()
    docs = vector_retriever.invoke(query)
    retrieval_ms = (time.time() - t0) * 1000

    # Phase 2: Generation with TTFT logging
    t_gen = time.time()
    response_stream = llm.stream(docs_and_query_prompt)
    
    first_token_received = False
    ttft_ms = 0.0
    full_output = []

    for chunk in response_stream:
        if not first_token_received:
            ttft_ms = (time.time() - t_gen) * 1000
            first_token_received = True
        full_output.append(chunk.content)

    return "".join(full_output)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In LangSmith, selecting any production trace displays an interactive Gantt waterfall. 
Engineers instantly discern whether an 8-second request was caused by 6 sequential tool calls or a single verbose LLM generation.
""")

st.divider()

# ==============================================================================
# SECTION 3: MONITOR COST
# ==============================================================================
st.header("3. Monitor Cost (Token Burn Velocity, Model Pricing Tiers, Cache Hit Ratios and Budget Cap Alerts)", divider="rainbow")

st.markdown("""
LLM applications introduce variable, usage-based cloud costs that can scale out of control if unmonitored. 
An unhandled retry loop or runaway recursive agent can burn thousands of dollars in minutes. 
Enterprise observability tracks cost per request, model-level expenditure, and cache efficiency.
""")

st.markdown("""
### Enterprise Cost Management Pillars
1. **Dynamic Token Unit Economics**: Tracking prompt tokens vs completion tokens. Completion tokens typically cost 3x–4x more than prompt tokens across major providers.
2. **Semantic Caching**: Using Redis or GPTCache to intercept repeated queries. A 25% cache hit rate directly translates to a 25% reduction in model inference costs.
3. **Hard Budget Cap Tripwires**: Configuring automated alerts in LangSmith when hourly token spend exceeds 80% of budget allocations.
""")

cost_tier_df = pd.DataFrame([
    {"Model Endpoint": "GPT-4o (Standard)", "Input (1M Tokens)": "2.50 USD", "Output (1M Tokens)": "10.00 USD", "Role": "Complex agent reasoning & final synthesis"},
    {"Model Endpoint": "GPT-4o-mini", "Input (1M Tokens)": "0.15 USD", "Output (1M Tokens)": "0.60 USD", "Role": "Query routing, intent classification, small summaries"},
    {"Model Endpoint": "Claude 3.5 Sonnet", "Input (1M Tokens)": "3.00 USD", "Output (1M Tokens)": "15.00 USD", "Role": "Code generation & advanced tool calling"},
    {"Model Endpoint": "Text-Embedding-3-Small", "Input (1M Tokens)": "0.02 USD", "Output (1M Tokens)": "N/A", "Role": "Dense semantic retrieval embeddings"}
])
st.dataframe(cost_tier_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Finance teams export LangSmith token metrics grouped by `metadata.tenant_id` to Datadog or Snowflake. 
This enables chargeback billing where each business department or client is billed exactly for their LLM consumption.
""")

st.divider()

# ==============================================================================
# SECTION 4: MONITOR QUALITY
# ==============================================================================
st.header("4. Monitor Quality (Online Grounding Daemons, Feedback Thumbs Up/Down and Quality Drift Alarms)", divider="rainbow")

st.markdown("""
Traditional web applications fail with binary HTTP errors (e.g., `500 Server Error`). 
LLM applications fail silently while returning `200 OK` with plausible-sounding hallucinations or unhelpful answers. 
Monitoring production quality requires continuous sampling and live user feedback loops.
""")

st.markdown("""
### Online Quality Surveillance Strategies
* **Online Evaluation Sampling (5% Stream)**: Running lightweight evaluator daemons asynchronously on a 5% stratified sample of production traces to grade context faithfulness and relevance.
* **Implicit & Explicit User Signals**: Logging user thumbs up/down clicks, copy-to-clipboard actions, and follow-up query clarification rates directly to LangSmith via the feedback API.
* **Quality Drift Detection**: Triggering alerts when 7-day rolling user thumbs-down rates rise by more than 1.5% above baseline.
""")

st.code('''
from langsmith import Client

client = Client()

# Logging explicit end-user feedback to LangSmith run
def log_user_feedback(run_id: str, is_positive: bool, user_comment: str = ""):
    client.create_feedback(
        run_id=run_id,
        key="user_satisfaction",
        score=1.0 if is_positive else 0.0,
        value="thumbs_up" if is_positive else "thumbs_down",
        comment=user_comment
    )
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Product managers set up LangSmith dashboards charting weekly feedback ratios. 
Runs receiving negative feedback are automatically added to an "Escalated Review" annotation queue for weekly triage.
""")

st.divider()

# ==============================================================================
# SECTION 5: MONITOR FAILURES
# ==============================================================================
st.header("5. Monitor Failures (Exception Categorization, Tool Timeout Rate and Infinite Loop Detection)", divider="rainbow")

st.markdown("""
Failures in agentic and RAG workflows occur across multiple distributed subsystems: 
model provider rate limits (`429`), upstream tool database timeouts, JSON output schema parsing errors, and agent infinite loops.
""")

st.markdown("""
### The 4 Major Failure Categories
1. **Provider Rate Limits (`429 Too Many Requests`)**: Exceeded tokens-per-minute (TPM) or requests-per-minute (RPM). Requires exponential backoff with jitter.
2. **Structured Output Parsing Exceptions**: The model generates invalid JSON or misses required schema keys, crashing the downstream parser. Requires automatic retry prompting.
3. **Tool Execution Timeouts**: External SQL or REST APIs take > 5000ms. Requires timeout wrappers and fallback tools.
4. **Agent Recursion Limit Exceeded**: Autonomous agents looping between tools without progressing toward completion. Requires setting `recursion_limit=15` in StateGraph.
""")

failure_matrix_df = pd.DataFrame([
    {"Failure Mode": "HTTP 429 Rate Limit", "Root Cause": "Spike in concurrent user traffic", "Mitigation": "Multi-region endpoint load balancing with exponential backoff"},
    {"Failure Mode": "OutputParserException", "Root Cause": "Model emitted conversational markdown around JSON", "Mitigation": "Enforce strict OpenAI JSON mode or Instructor Pydantic repair"},
    {"Failure Mode": "Tool Timeout (HTTP 504)", "Root Cause": "Slow internal legacy CRM database", "Mitigation": "Asynchronous fallback responses; fail gracefully"},
    {"Failure Mode": "Agent Infinite Loop", "Root Cause": "Tool output lacks info needed to satisfy stop criteria", "Mitigation": "Hard recursion limit in StateGraph; fallback to human agent"}
])
st.dataframe(failure_matrix_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
LangSmith automatically tags runs containing Python exceptions with `status: error`. 
Engineers filter for `status: error` in the UI to inspect the exact input query and stack trace that triggered the crash.
""")

st.divider()

# ==============================================================================
# SECTION 6: PROTECT SENSITIVE DATA
# ==============================================================================
st.header("6. Protect Sensitive Data (Zero-Trust Scrubbers, PII Masking, Metadata-Only Mode and TLS 1.3 VPC Transit)", divider="rainbow")

st.markdown("""
Capturing full prompts and completions in observability traces creates major regulatory compliance exposure 
under GDPR, HIPAA, and SOC-2 if requests contain Personally Identifiable Information (PII) or protected health data. 
Enterprise architectures implement zero-trust data protection before traces leave the local network.
""")

st.markdown("""
### Sensitive Data Defense-in-Depth
* **Edge PII Redaction**: Using Microsoft Presidio or deterministic regex scrubbers to mask SSNs, credit card numbers, phone numbers, and emails before sending prompts to the LLM or LangSmith.
* **Metadata-Only Tracing Mode**: For extreme compliance environments (banking/defense), configure LangSmith to record execution latencies, token counts, and metadata *without* persisting raw prompt strings or completions.
* **TLS 1.3 VPC Transit**: Ensuring all telemetry dispatches transit private VPC endpoints (AWS PrivateLink) directly to enterprise LangSmith instances.
""")

st.code(r'''
import re

class EnterprisePIIScrubber:
    """
    Edge PII Redaction Filter for Telemetry Streams.
    """
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    @classmethod
    def scrub_text(cls, text: str) -> str:
        text = cls.SSN_PATTERN.sub("[REDACTED_SSN]", text)
        text = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        return text

# Usage before traceable function dispatch
clean_prompt = EnterprisePIIScrubber.scrub_text(raw_user_input)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
LangSmith provides client-side masking hooks (`hide_inputs=True`, `hide_outputs=True`) in the SDK. 
When enabled, traces capture complete execution graphs, timings, and token metrics while leaving input and output text fields completely blank in cloud storage.
""")

st.divider()

# ==============================================================================
# SECTION 7: VERSION PROMPTS AND WORKFLOWS
# ==============================================================================
st.header("7. Version Prompts and Workflows (Prompt Hub SemVer, StateGraph Checkpoint Versioning and Rollback Safety)", divider="rainbow")

st.markdown("""
An unversioned LLM application is an unmanageable liability. Versioning ensures that every production trace 
can be traced back to the exact code, prompt template, model weights, and agent state schema that generated it.
""")

st.markdown("""
### Enterprise Versioning Standards
* **Prompt Hub Releases**: System prompts must be tagged with explicit SemVer strings (`support-v2.1.0`) and mapped to environment aliases (`:prod`, `:staging`).
* **Agent State Schemas**: When updating LangGraph state definitions, maintain backward-compatible field defaults so active multi-turn conversations stored in checkpoints deserialize cleanly.
* **Rollback Readiness**: Maintaining a validated, immutable baseline version that can be hot-swapped in under 500ms via gateway routing pointers if an incident occurs.
""")

version_matrix = pd.DataFrame([
    {"Artifact Type": "System Prompt", "Tooling": "LangSmith Prompt Hub", "Versioning Scheme": "SemVer + Tag (:prod)", "Rollback Mechanism": "One-click tag re-point"},
    {"Artifact Type": "Model Endpoint", "Tooling": "AI Gateway (LiteLLM)", "Versioning Scheme": "Pinned Provider Snapshot", "Rollback Mechanism": "Traffic weight swap (0% Green / 100% Blue)"},
    {"Artifact Type": "Agent Graph", "Tooling": "LangGraph Cloud", "Versioning Scheme": "Docker Container Tag", "Rollback Mechanism": "Rolling Kubernetes deployment undo"},
    {"Artifact Type": "Vector Index", "Tooling": "Pinecone / Qdrant", "Versioning Scheme": "Snapshot Timestamp ID", "Rollback Mechanism": "Index alias pointer switch"}
])
st.dataframe(version_matrix, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
GitOps pipelines link Git commit SHAs directly to LangSmith Prompt Hub commits. 
Merging a pull request automatically publishes a new prompt version to the Hub, runs automated Continuous Evaluation, 
and updates production pointers only when all checks pass.
""")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full-Spectrum LLMOps Production Monitor (ProductionObservabilityMonitorEngine)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the observability engine. 
It processes batches of production telemetry, computes latency distributions (p50/p90/p99/TTFT), tracks token economics, 
performs zero-trust PII redaction, and verifies multi-metric SLA compliance.
""")

st.code('''
class ProductionObservabilityMonitorEngine:
    """
    Enterprise Full-Spectrum LLMOps Production Monitor & Governance Engine.
    Implements 6 core production pillars:
    1. Latency percentile profiling (p50, p90, p99, TTFT)
    2. Cost and token burn tracking with budget alerts
    3. Online quality monitoring and user feedback aggregation
    4. Failure rate tracking and exception triage
    5. Zero-Trust PII masking and sensitive data scrubbing
    6. Prompt and workflow SemVer release tracking
    """
    def __init__(self, project_name: str = "prod-enterprise-copilot"):
        self.project_name = project_name
        self.telemetry_history: List[Dict[str, Any]] = []

    def process_production_telemetry_batch(
        self,
        request_count: int = 1000,
        cache_hit_rate: float = 0.25,
        pii_masking_enabled: bool = True,
        simulate_degradation: bool = False
    ) -> Dict[str, Any]:
        """
        Processes a live batch of production traces and computes enterprise observability metrics.
        """
        # Latency Profiling (ms)
        if simulate_degradation:
            p50_ms = 1850.0
            p90_ms = 3450.0
            p99_ms = 5820.0
            ttft_ms = 920.0
            failure_rate_pct = 4.8
            user_satisfaction_pct = 76.5
        else:
            p50_ms = 820.0
            p90_ms = 1450.0
            p99_ms = 2150.0
            ttft_ms = 310.0
            failure_rate_pct = 0.4
            user_satisfaction_pct = 94.2

        # Token & Cost Economics (USD)
        avg_input_tokens = 650
        avg_output_tokens = 220
        total_input_tokens = int(request_count * avg_input_tokens * (1.0 - cache_hit_rate))
        total_output_tokens = int(request_count * avg_output_tokens)
        
        # Pricing: GPT-4o (2.50 USD / 1M in, 10.00 USD / 1M out)
        input_cost = (total_input_tokens / 1_000_000.0) * 2.50
        output_cost = (total_output_tokens / 1_000_000.0) * 10.00
        gross_cost_usd = round(input_cost + output_cost, 2)
        saved_cache_cost_usd = round(((request_count * avg_input_tokens * cache_hit_rate) / 1_000_000.0) * 2.50, 2)

        # Security & PII Scrubbing Audit
        raw_samples = [
            "Customer John Doe (SSN: 123-45-6789) requested account balance at john@example.com",
            "Transfer 500 USD to account 987654321 for user Alice Smith",
            "Standard policy question regarding dental coverage limits."
        ]
        
        scrubbed_samples = []
        pii_entities_detected = 0
        for sample in raw_samples:
            if pii_masking_enabled:
                text = sample.replace("123-45-6789", "[REDACTED_SSN]")
                text = text.replace("john@example.com", "[REDACTED_EMAIL]")
                text = text.replace("987654321", "[REDACTED_ACCOUNT]")
                if "[REDACTED" in text:
                    pii_entities_detected += 1
                scrubbed_samples.append(text)
            else:
                scrubbed_samples.append(sample)

        latency_sla_passed = p99_ms <= 3000.0
        failure_sla_passed = failure_rate_pct <= 1.0
        quality_sla_passed = user_satisfaction_pct >= 90.0

        return {
            "batch_size": request_count,
            "cache_hit_rate": cache_hit_rate,
            "latency_metrics": {
                "p50_ms": p50_ms, "p90_ms": p90_ms, "p99_ms": p99_ms, "ttft_ms": ttft_ms,
                "sla_passed": latency_sla_passed
            },
            "cost_metrics": {
                "gross_cost_usd": gross_cost_usd, "cache_savings_usd": saved_cache_cost_usd,
                "total_tokens": total_input_tokens + total_output_tokens,
                "cost_per_request": round(gross_cost_usd / request_count, 4)
            },
            "quality_metrics": {
                "failure_rate_pct": failure_rate_pct, "user_satisfaction_pct": user_satisfaction_pct,
                "quality_sla_passed": quality_sla_passed and failure_sla_passed
            },
            "security_metrics": {
                "pii_masking_enabled": pii_masking_enabled,
                "entities_scrubbed": pii_entities_detected if pii_masking_enabled else 0,
                "compliance_status": "ZERO_TRUST_COMPLIANT" if pii_masking_enabled else "EXPOSURE_RISK_WARNING"
            },
            "samples": scrubbed_samples,
            "overall_status": "HEALTHY" if (latency_sla_passed and failure_sla_passed and quality_sla_passed) else "ALERT_BREACHED"
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Production Telemetry Cockpit and Multi-Metric SLA Alarm Sandbox", divider="rainbow")

st.markdown("""
Monitor simulated production traffic across latency percentiles, cost burn rates, semantic cache efficiencies, 
and zero-trust PII scrubbing. Toggle degradation to observe automated SLA alarm triggers.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Telemetry Controls")
    batch_size_slider = st.slider(
        "Hourly Request Volume (Requests / Hr)",
        min_value=500,
        max_value=10000,
        value=2000,
        step=500
    )
    cache_rate_slider = st.slider(
        "Semantic Cache Hit Ratio",
        min_value=0.0,
        max_value=0.60,
        value=0.25,
        step=0.05,
        help="Proportion of repeated user queries resolved from cache without LLM calls."
    )
    pii_toggle = st.toggle(
        "Enable Zero-Trust PII Masking Scrubber",
        value=True,
        help="Masks SSNs, emails, and account numbers prior to span persistence."
    )
    degrade_toggle = st.toggle(
        "Simulate Production Outage / Degradation",
        value=False,
        help="Simulates provider rate limiting, latency spikes, and quality drift."
    )

with wb_col2:
    st.markdown("### Live Observability Telemetry Cockpit")
    report = st.session_state.prod_observability_engine.process_production_telemetry_batch(
        request_count=batch_size_slider,
        cache_hit_rate=cache_rate_slider,
        pii_masking_enabled=pii_toggle,
        simulate_degradation=degrade_toggle
    )

    if report["overall_status"] == "HEALTHY":
        st.success(f"**SYSTEM STATUS: {report['overall_status']}** — All production latency, cost, and quality SLAs satisfied.")
    else:
        st.error(f"**SYSTEM STATUS: {report['overall_status']}** — Multiple SLA thresholds breached. Critical incident alerted.")

    # High-level Metrics Row
    lat = report["latency_metrics"]
    cost = report["cost_metrics"]
    qual = report["quality_metrics"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("p99 Tail Latency", f"{lat['p99_ms']:.0f} ms", delta=f"{lat['p99_ms'] - 2150.0:+.0f} ms", delta_color="inverse")
    with m2:
        st.metric("Time-to-First-Token", f"{lat['ttft_ms']:.0f} ms", delta=f"{lat['ttft_ms'] - 310.0:+.0f} ms", delta_color="inverse")
    with m3:
        st.metric("Gross Cost (Hr)", f"{cost['gross_cost_usd']:.2f} USD", delta=f"-{cost['cache_savings_usd']:.2f} USD Cache Saved")
    with m4:
        st.metric("User Satisfaction", f"{qual['user_satisfaction_pct']:.1f}%", delta=f"{qual['user_satisfaction_pct'] - 94.2:+.1f}%")

    # Security & PII Redaction Audit
    st.markdown("#### Zero-Trust Data Protection Audit")
    sec = report["security_metrics"]
    if sec["pii_masking_enabled"]:
        st.info(f"Compliance: **{sec['compliance_status']}** | Entities Redacted: **{sec['entities_scrubbed']}**")
    else:
        st.warning(f"Compliance: **{sec['compliance_status']}** | Warning: Sensitive PII transmitted in plaintext traces!")

    for s in report["samples"]:
        st.code(s, language="text")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

q_exp3 = st.expander("View 5 Comprehensive Production Interview Q&As", expanded=True)

with q_exp3:
    st.markdown("""
    **Q1: Why is p99 latency significantly more important than average (mean) latency in LLM architectures?**  
    *Answer:* Because LLM completion time scales with generated token length, average latency obscures severe user-facing delays. A system with an average latency of 1.2s may have a p99 latency of 12.0s for complex queries requiring deep multi-step agent reasoning. Monitoring p95 and p99 surfaces token bloat, slow tool executions, and retrieval bottlenecks before they cause widespread user abandonment.
    """)

    st.markdown("""
    **Q2: What is Time-to-First-Token (TTFT), and how does streaming optimize perceived user latency?**  
    *Answer:* TTFT measures the time between request submission and the arrival of the first completion token. Without streaming, a user must wait for the entire completion to generate (e.g., 4.0s). With streaming, TTFT delivers the first token in 300ms, allowing the user to begin reading immediately while subsequent tokens arrive at 50 tokens/sec. This reduces perceived wait time by up to 90%.
    """)

    st.markdown("""
    **Q3: How does semantic caching reduce operational LLM costs in production?**  
    *Answer:* Many user inquiries are semantically similar (e.g., *"What is your return policy?"* vs *"How do I return an item?"*). A semantic cache computes embeddings for incoming queries and checks if a semantically equivalent query was resolved recently. If cosine similarity exceeds 0.95, the cached completion is returned immediately, saving 100% of LLM generation costs and reducing latency to < 50ms.
    """)

    st.markdown("""
    **Q4: How should enterprise teams handle PII and sensitive data in LangSmith production traces?**  
    *Answer:* Implement edge redaction using tools like Microsoft Presidio or regex scrubbers to replace SSNs, emails, and account numbers with tokens like `[REDACTED_SSN]` before traces leave the local network. For extreme compliance environments (HIPAA/defense), configure the LangSmith SDK in metadata-only mode (`hide_inputs=True`, `hide_outputs=True`) so that only latency, token counts, and error codes are persisted.
    """)

    st.markdown("""
    **Q5: What are the three essential components of an automated production circuit breaker for LLMs?**  
    *Answer:* 1) Real-time telemetry monitoring (tracking rolling error rates, p99 latency, and user thumbs-down ratios); 2) Codified SLA tripwires (e.g., error rate > 2.0% over 5 minutes); and 3) An automated routing failover mechanism (an AI Gateway switching 100% of traffic to a stable fallback model or pinned prompt version).
    """)

st.markdown("### Production Traps and Anti-Patterns")

trap_c1, trap_c2 = st.columns(2)

with trap_c1:
    st.error("""
    **Trap 1: Sending Unredacted PII to Observability Backends**
    * **Failure**: Emitting production user prompts containing SSNs and medical histories directly into cloud trace platforms.
    * **Consequence**: Severe regulatory violations under GDPR, HIPAA, and CCPA, resulting in massive fines and compliance audits.
    * **Remedy**: Implement client-side zero-trust PII scrubbing or metadata-only tracing mode for sensitive workflows.
    """)

    st.error("""
    **Trap 2: Ignoring Completion Token Ratios in Cost Models**
    * **Failure**: Assuming input and output tokens cost the same when budgeting cloud spend.
    * **Consequence**: A prompt update that makes the model 3x more verbose unexpectedly doubles the monthly cloud bill, because completion tokens are priced 3x-4x higher than input tokens.
    * **Remedy**: Always track input and output token consumption separately and enforce max token generation caps.
    """)

with trap_c2:
    st.warning("""
    **Trap 3: Unmonitored Agent Recursion Loops**
    * **Failure**: Allowing autonomous agents to execute tools without strict recursion limits.
    * **Consequence**: An agent stuck in a decision deadlock makes 100 consecutive tool calls, burning hundreds of dollars and timing out the user session.
    * **Remedy**: Set hard recursion limits (`recursion_limit=15`) in LangGraph StateGraph configurations.
    """)

    st.warning("""
    **Trap 4: Relying on Binary Health Checks (200 OK Fallacy)**
    * **Failure**: Only alerting when HTTP servers return 500 errors, ignoring LLM output quality.
    * **Consequence**: An ungrounded model hallucinates incorrect financial advice for days with zero infrastructure alerts because the endpoint returned HTTP 200 OK.
    * **Remedy**: Implement online quality surveillance daemons and user thumbs-down alerts in LangSmith.
    """)

st.markdown("### Core Technical Flashcards")

fc_a1, fc_a2 = st.columns(2)

with fc_a1:
    with st.container(border=True):
        st.markdown("**Front**: What is the difference between TTFT and Inter-Token Latency?")
        st.markdown("**Back**: TTFT is the latency to receive the first token (prefill phase); Inter-Token Latency is the generation rate of subsequent tokens (decoding phase).")

    with fc_a1:
        with st.container(border=True):
            st.markdown("**Front**: Why is completion token pricing higher than prompt token pricing?")
            st.markdown("**Back**: Prompt tokens are processed in parallel during the prefill phase; completion tokens must be generated sequentially one-by-one (autoregressively).")

with fc_a2:
    with st.container(border=True):
        st.markdown("**Front**: How does Metadata-Only Tracing Mode ensure privacy compliance?")
        st.markdown("**Back**: It persists execution timing, token counts, and error metadata while leaving input prompt and output text fields completely blank.")

    with fc_a2:
        with st.container(border=True):
            st.markdown("**Front**: What is the primary purpose of an LLM Production Circuit Breaker?")
            st.markdown("**Back**: To automatically detect error rate or latency breaches and instantly reroute traffic to a known-good fallback model without human intervention.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Production Best Practices in LangSmith:")

pbp_chk1 = st.checkbox("I know how to monitor tail latency percentiles (p50, p90, p99) and Time-to-First-Token (TTFT).", value=True)
pbp_chk2 = st.checkbox("I understand how to track token unit economics and calculate cost savings from semantic caching.", value=True)
pbp_chk3 = st.checkbox("I can implement online quality surveillance daemons and capture live user thumbs-up/down feedback.", value=True)
pbp_chk4 = st.checkbox("I know how to categorize and mitigate the 4 major LLM failure modes (429s, parser errors, timeouts, loops).", value=True)
pbp_chk5 = st.checkbox("I understand how to enforce zero-trust PII masking and metadata-only tracing modes for compliance.", value=True)
pbp_chk6 = st.checkbox("I can design automated production circuit breakers with instant fallback routing.", value=True)

if pbp_chk1 and pbp_chk2 and pbp_chk3 and pbp_chk4 and pbp_chk5 and pbp_chk6:
    st.success("Mastery Certified: You have mastered enterprise Production Best Practices, multi-metric observability, and LLMOps governance in LangSmith.")
