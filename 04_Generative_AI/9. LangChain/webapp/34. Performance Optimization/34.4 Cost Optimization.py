import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="34.4 Cost Optimization - LangChain Curriculum",
    page_icon="💰",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .cost-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-save {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class CostOptimizationEngine:
    """
    Simulates enterprise LLM cost modeling: input/output pricing disparity,
    Batch API 50% discounts, Anthropic prompt cache savings, and budget guards.
    """
    def __init__(self):
        self.pricing_table = {
            "gpt-4o": {"input_per_m": 2.50, "output_per_m": 10.00, "batch_discount": 0.50},
            "gpt-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60, "batch_discount": 0.50},
            "claude-3-5-sonnet": {"input_per_m": 3.00, "output_per_m": 15.00, "batch_discount": 0.50},
            "claude-3-5-haiku": {"input_per_m": 0.80, "output_per_m": 4.00, "batch_discount": 0.50}
        }

    def calculate_spend(self, model: str, prompt_toks: int, comp_toks: int, is_batch: bool = False, cache_hit_pct: float = 0.0) -> Dict[str, Any]:
        info = self.pricing_table.get(model, self.pricing_table["gpt-4o-mini"])
        
        # Calculate standard input cost
        cached_input = prompt_toks * cache_hit_pct
        uncached_input = prompt_toks * (1 - cache_hit_pct)
        
        # Anthropic/OpenAI prompt cache hits are typically discounted by 90%
        base_input_cost = ((uncached_input / 1_000_000) * info["input_per_m"]) + ((cached_input / 1_000_000) * info["input_per_m"] * 0.10)
        base_output_cost = (comp_toks / 1_000_000) * info["output_per_m"]
        
        total_standard = base_input_cost + base_output_cost
        
        if is_batch:
            final_cost = total_standard * (1 - info["batch_discount"])
        else:
            final_cost = total_standard
            
        return {
            "model": model,
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "total_tokens": prompt_toks + comp_toks,
            "final_cost_usd": round(final_cost, 5),
            "is_batch": is_batch,
            "cache_hit_pct": round(cache_hit_pct * 100, 1)
        }

# Instantiate engine
cost_engine = CostOptimizationEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise LLM Cost Governance, Tiered Economics, & Spend Attribution", divider="rainbow")

st.markdown("""
In enterprise deployments, LLM API expenditures can spiral out of control without stringent governance. 
**Cost Optimization** is an architectural discipline combining **token accounting**, **Batch API offloading**, 
**prefix prompt caching**, **dynamic tiered model routing**, and **circuit-breaker budget caps**.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Cost_Optimization_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_ingress {
            label = "Traffic Ingress & SLA Classification Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            in_request [label="Incoming Business Request\\n(Tenant ID + Payload + SLA)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            sla_classifier [label="SLA & Urgency Classifier\\n(Real-Time Interactive vs Asynchronous Batch)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        subgraph cluster_routing {
            label = "Cost-Aware Routing & Optimization Engine";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            prompt_cache [label="Prompt Cache Layer (Anthropic / OpenAI)\\n(90% input discount on cached prefix)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            batch_queue [label="Asynchronous Batch API Queue (24h SLA)\\n(Guaranteed 50% price reduction across all tokens)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
            model_router [label="Complexity Model Router\\n(Routes 80% extraction to GPT-4o-mini / Haiku)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_governance {
            label = "Governance & Budget Enforcement Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            budget_guard [label="Circuit Breaker Spend Cap\\n(Halts execution if tenant exceeds daily limit)", fillcolor="#0F172A", fontcolor="#F43F5E", color="#F43F5E"];
            cost_attribution [label="LangSmith Cost Attribution Dashboard\\n(Per-tenant dollar breakdown and daily burn rate)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        in_request -> sla_classifier;
        sla_classifier -> batch_queue [label="Async offline (50% off)"];
        sla_classifier -> model_router [label="Real-time interactive"];

        model_router -> prompt_cache;
        prompt_cache -> budget_guard;
        batch_queue -> budget_guard;

        budget_guard -> cost_attribution [label="records exact spend per tenant"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Blended Architecture Strategy:</strong> Routing 70% of standard traffic to distilled models, 
    running non-urgent analytics through Batch APIs (50% discount), and caching common prompt prefixes (90% discount) 
    achieves a cumulative 75% to 85% reduction in overall enterprise API costs without degrading user experience.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: TOKEN COST
# ==============================================================================
st.header("2. Token Cost: Input vs Output Pricing Asymmetry and Token Accounting", divider="rainbow")

st.markdown("""
A critical principle of LLM economics is **pricing asymmetry**: across virtually all commercial model APIs, 
**output (completion) tokens are 3x to 5x more expensive than input (prompt) tokens**.
""")

st.markdown("""
### Pricing Disparity Across Leading Models
- **GPT-4o**: 2.50 dollars / 1M input tokens vs 10.00 dollars / 1M output tokens (4x higher).
- **Claude 3.5 Sonnet**: 3.00 dollars / 1M input tokens vs 15.00 dollars / 1M output tokens (5x higher).
- **GPT-4o-mini**: 0.15 dollars / 1M input tokens vs 0.60 dollars / 1M output tokens (4x higher).
""")

st.markdown("""
### Cost Optimization Implication
Instructing a model to be concise ('Answer in under 30 words') or constraining responses to compact JSON 
saves vastly more dollars than trimming input prompts, because every output token saved is worth 4 to 5 input tokens.
""")

# ==============================================================================
# SECTION 3: MODEL COST
# ==============================================================================
st.header("3. Model Cost: Pricing Tiers and Open-Source Self-Hosting Unit Economics", divider="rainbow")

st.markdown("""
Organizations often assume commercial APIs are always more expensive than self-hosting. 
In reality, the economic crossover point depends strictly on request concurrency and GPU saturation.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Managed Serverless APIs
    - **Characteristics**: Zero idle cost, pay per token.
    - **Best For**: Variable traffic, bursty workloads, prototyping, low-to-medium volume (< 10M tokens/day).
    - **Unit Economics**: Extremely cost-efficient for distilled SLMs ($0.15/1M tokens).
    """)

with col2:
    st.markdown("""
    ### Self-Hosted vLLM on Dedicated GPUs
    - **Characteristics**: Fixed infrastructure cost (e.g., $1,500/month for an A100 instance).
    - **Best For**: High, predictable 24/7 volume (> 50M tokens/day) with constant GPU utilization.
    - **Warning**: Under-utilized GPU clusters (running at 15% load) cost significantly more per token than managed APIs.
    """)

# ==============================================================================
# SECTION 4: CACHING
# ==============================================================================
st.header("4. Caching: Prompt Caching and In-Memory Exact Caching", divider="rainbow")

st.markdown("""
**Prompt Caching** (pioneered by Anthropic and adopted by OpenAI) fundamentally changes RAG and multi-turn agent economics. 
When prompts share a common prefix of at least 1,024 tokens (system prompt, large documentation context, or few-shot examples), 
the provider caches the key-value activations.
""")

st.markdown("""
### Prompt Cache Economics (Anthropic Standard)
- **Base Input Price**: 3.00 dollars per 1M tokens.
- **Cache Creation Price**: 3.75 dollars per 1M tokens (one-time 25% write premium).
- **Cache Read (Hit) Price**: **0.30 dollars per 1M tokens (90% discount!)**.
- **Time to First Token**: Slashed by up to 80% on cache hits.
""")

# ==============================================================================
# SECTION 5: BATCH PROCESSING
# ==============================================================================
st.header("5. Batch Processing: OpenAI and Anthropic Batch APIs for Asynchronous Workloads", divider="rainbow")

st.markdown("""
Not all LLM requests require synchronous sub-second responses. Offline tasks—such as overnight document summarization, 
dataset labeling, synthetic evaluation runs, and bulk categorization—should always be routed to **Batch APIs**.
""")

st.markdown("""
### Key Benefits of Batch APIs
- **Flat 50% Discount**: Both OpenAI and Anthropic offer an unconditional 50% discount on both input and output tokens for batch requests.
- **Dedicated Quota Pools**: Batch requests do not consume synchronous TPM (Tokens Per Minute) quotas, preventing live production outages during bulk jobs.
- **Standard 24-Hour SLA**: Jobs are typically returned within 1 to 4 hours.
""")

# ==============================================================================
# SECTION 6: MODEL SELECTION
# ==============================================================================
st.header("6. Model Selection: Dynamic Capability-Based Tier Routing and Downgrade Cascades", divider="rainbow")

st.markdown("""
A **Downgrade Cascade** policy attempts execution with a cheap distilled model first. 
If and only if the cheap model fails schema validation or confidence thresholds, the system escalates to a frontier model.
""")

st.code('''from pydantic import BaseModel, ValidationError

class ExtractedEntities(BaseModel):
    user_id: str
    amount_usd: float

def execute_downgrade_cascade(user_text: str) -> ExtractedEntities:
    # Tier 1: Attempt extraction with fast, ultra-cheap SLM (GPT-4o-mini / Haiku)
    try:
        # raw_json = fast_slm.invoke(user_text)
        # return ExtractedEntities.model_validate_json(raw_json)
        return ExtractedEntities(user_id="usr_123", amount_usd=49.99)
    except ValidationError:
        # Tier 2: Escalate only upon schema failure to Frontier model (GPT-4o)
        # raw_json = frontier_model.invoke(user_text)
        # return ExtractedEntities.model_validate_json(raw_json)
        pass
''', language="python")

# ==============================================================================
# SECTION 7: COST MONITORING
# ==============================================================================
st.header("7. Cost Monitoring: Per-Tenant Attribution and Circuit-Breaker Spend Caps", divider="rainbow")

st.markdown("""
In SaaS multi-tenant architectures, individual users can inadvertently trigger runaway loops. 
Production systems require **per-tenant cost attribution** and **automated circuit-breaker spend caps**.
""")

st.markdown("""
### Cost Guardrail Implementation Checklist
1. **Metadata Tagging**: Attach `tenant_id`, `environment`, and `feature_name` to every LangChain run metadata.
2. **Real-Time Token Aggregation**: Maintain a Redis counter tracking cumulative token expenditures per tenant per day.
3. **Circuit Breaker Trip**: If a tenant exceeds 80% of their daily spend cap, trigger alerts; if they hit 100%, return HTTP 429 quota exceeded.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Cost-Optimized Batch and Cached Chain with Budget Circuit Breaker", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**budget-guarded execution engine with prompt caching and tenant attribution**.
""")

st.code('''from typing import Dict, Any

class EnterpriseCostGuardEngine:
    def __init__(self, daily_budget_usd: float = 5.00):
        self.daily_budget_usd = daily_budget_usd
        self.current_spend_usd = 0.00
        self.cost_catalog = {
            "gpt-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
            "gpt-4o": {"input_per_m": 2.50, "output_per_m": 10.00}
        }

    def execute_guarded_run(self, tenant_id: str, prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Calculates token cost and halts execution if daily spend cap is breached."""
        model_rates = self.cost_catalog.get(model, self.cost_catalog["gpt-4o-mini"])
        run_cost = ((prompt_tokens / 1_000_000) * model_rates["input_per_m"]) + ((completion_tokens / 1_000_000) * model_rates["output_per_m"])
        
        # Circuit Breaker Check
        if self.current_spend_usd + run_cost > self.daily_budget_usd:
            return {
                "tenant_id": tenant_id,
                "status": "BLOCKED_BUDGET_EXCEEDED",
                "current_spend_usd": round(self.current_spend_usd, 4),
                "budget_limit_usd": self.daily_budget_usd
            }
            
        # Update spend
        self.current_spend_usd += run_cost
        
        return {
            "tenant_id": tenant_id,
            "status": "SUCCESS",
            "model_used": model,
            "run_cost_usd": round(run_cost, 6),
            "remaining_budget_usd": round(self.daily_budget_usd - self.current_spend_usd, 4)
        }

guard = EnterpriseCostGuardEngine(daily_budget_usd=1.00)
# Normal invocation with distilled SLM
res1 = guard.execute_guarded_run("tenant_alpha", prompt_tokens=2000, completion_tokens=300, model="gpt-4o-mini")
print("Run 1 Result:", res1)
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Enterprise Cost Projection Calculator & Batch API Savings Estimator", divider="rainbow")

st.markdown("""
Estimate operational costs, model expenditure curves, and simulate cumulative savings 
achieved via Batch APIs and Anthropic prompt caching.
""")

wb_view = st.selectbox(
    "Select Cost Analysis Tool:",
    ["1. Interactive Model Cost Calculator", "2. Batch API & Prompt Caching Savings Simulator", "3. Enterprise Monthly Burn Projection"]
)

if wb_view == "1. Interactive Model Cost Calculator":
    st.subheader("Model Token Cost Calculator")
    sel_m = st.selectbox("Select Model Checkpoint:", list(cost_engine.pricing_table.keys()))

    c1, c2 = st.columns(2)
    with c1:
        p_tok = st.slider("Prompt (Input) Tokens:", 500, 50000, 4000, 500)
    with c2:
        c_tok = st.slider("Completion (Output) Tokens:", 50, 4000, 400, 50)

    use_batch = st.toggle("Execute via Batch API (50% Off)", value=False)
    cache_pct = st.slider("Prompt Cache Hit Ratio:", 0.0, 1.0, 0.5, 0.05)

    calc_res = cost_engine.calculate_spend(sel_m, p_tok, c_tok, use_batch, cache_pct)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Incurred Cost", f"${calc_res['final_cost_usd']}")
    with m2:
        st.metric("Cache Hit Ratio", f"{calc_res['cache_hit_pct']}%")
    with m3:
        st.metric("Batch API Active", f"{calc_res['is_batch']}")

    st.info(f"Using {sel_m}: Output tokens represent a higher cost per unit than input tokens.")

elif wb_view == "2. Batch API & Prompt Caching Savings Simulator":
    st.subheader("Optimization Technique Savings Comparison")
    st.markdown("Comparing 100,000 requests of 2,000 prompt tokens and 300 completion tokens on GPT-4o:")

    base_spend = (100_000 * 2000 / 1_000_000 * 2.50) + (100_000 * 300 / 1_000_000 * 10.00) # $800
    batch_spend = base_spend * 0.50 # $400
    slm_spend = (100_000 * 2000 / 1_000_000 * 0.15) + (100_000 * 300 / 1_000_000 * 0.60) # $48

    comp_data = pd.DataFrame([
        {"Architecture": "Unoptimized Frontier Model (GPT-4o)", "Total Cost": f"${base_spend:.2f}", "Savings": "0%"},
        {"Architecture": "Batch API (50% Discount)", "Total Cost": f"${batch_spend:.2f}", "Savings": "50%"},
        {"Architecture": "Distilled SLM (GPT-4o-mini)", "Total Cost": f"${slm_spend:.2f}", "Savings": "94%"}
    ])
    st.dataframe(comp_data, use_container_width=True)

else:
    st.subheader("Enterprise Monthly Burn Projection")
    daily_reqs = st.slider("Daily Request Volume:", 1000, 200000, 25000, 5000)
    slm_ratio = st.slider("Percentage of Traffic Routed to SLM:", 0.0, 1.0, 0.8, 0.05)

    # Calculate blended monthly cost
    reqs_month = daily_reqs * 30
    slm_reqs = reqs_month * slm_ratio
    frontier_reqs = reqs_month * (1 - slm_ratio)

    cost_slm = (slm_reqs * 2000 / 1_000_000 * 0.15) + (slm_reqs * 300 / 1_000_000 * 0.60)
    cost_frontier = (frontier_reqs * 2000 / 1_000_000 * 2.50) + (frontier_reqs * 300 / 1_000_000 * 10.00)

    total_monthly = cost_slm + cost_frontier
    st.metric("Projected Monthly API Spend", f"${total_monthly:,.2f}")
    st.caption("Routing 80% of traffic to SLMs cuts blended monthly spend by over 75% compared to all-frontier deployments.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: Why are completion tokens substantially more expensive than prompt tokens?"):
    st.markdown("""
    **Answer:**  
    Input tokens are processed in parallel across transformer attention layers in a single matrix multiplication pass (prefill). 
    Completion tokens must be generated sequentially one-by-one (autoregression), requiring repeated GPU memory reads 
    and cache lookups for every generated token. This makes generation significantly more compute- and memory-bandwidth-intensive.
    """)

with st.expander("Q2: How do Batch APIs provide a 50% discount without compromising quality?"):
    st.markdown("""
    **Answer:**  
    Synchronous APIs require providers to reserve dedicated GPU capacity to guarantee sub-second Time to First Token. 
    Batch APIs allow providers to run jobs opportunistically during off-peak hours (e.g., overnight) on idle cluster 
    capacity, maximizing hardware utilization. Providers pass this operational efficiency to customers as a 50% discount.
    """)

with st.expander("Q3: What architectural conditions are required to achieve maximum savings from Anthropic prompt caching?"):
    st.markdown("""
    **Answer:**  
    1. The static prefix must exceed 1,024 tokens (e.g., large system prompt or documentation context).
    2. The prefix must be placed at the very beginning of the prompt without dynamic user insertions.
    3. Requests must arrive within the 5-minute cache TTL window (refreshed on each cache hit).
    """)

with st.expander("Q4: What is a Downgrade Cascade, and how does it safeguard generation quality?"):
    st.markdown("""
    **Answer:**  
    A downgrade cascade executes tasks on cheap distilled models by default. The system validates the output 
    using Pydantic schema validation or confidence scores. If validation fails, the task automatically escalates 
    to a frontier model. This ensures 90% of requests run at minimal cost while preserving high quality for edge cases.
    """)

with st.expander("Q5: How do you design a per-tenant budget guard in LangChain microservices?"):
    st.markdown("""
    **Answer:**  
    Attach `tenant_id` to the run metadata. Use a fast distributed datastore (like Redis) to maintain atomic daily spend 
    counters. Before invoking the LLM, evaluate the tenant's current spend against their daily quota. If the limit is reached, 
    short-circuit execution with an HTTP 429 quota exception before calling the model API.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Placing Dynamic User Variables at the Beginning of Prompts**  
Inserting a dynamic timestamp or user name at character 0 invalidates the entire prefix cache for subsequent requests, 
destroying prompt caching cost savings. Keep dynamic variables at the end.
""")

st.error("""
**Trap 2: Allowing Unconstrained Verbose LLM Completions**  
Allowing a model to emit 1,000 tokens of apologetic conversational filler when a 20-token JSON object suffices 
wastes 98% of your completion token budget.
""")

st.warning("""
**Trap 3: Running Asynchronous Overnight Batch Workloads on Synchronous APIs**  
Executing bulk dataset labeling or offline document indexing against standard synchronous endpoints pays 
double what the Batch API charges and risks triggering rate limits.
""")

st.warning("""
**Trap 4: Deploying Multi-Tenant Systems Without Individual Tenant Spend Caps**  
A single tenant running a runaway loop or automated script can exhaust an entire company's monthly API budget 
overnight. Always enforce per-tenant quotas.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Pricing Asymmetry</h4>
        <p>The economic reality where output tokens cost 3x to 5x more than input tokens due to autoregressive generation overhead.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Batch API</h4>
        <p>An asynchronous execution endpoint offering an unconditional 50% discount with a 24-hour turnaround SLA.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Prompt Cache Hit</h4>
        <p>Reusing precomputed key-value transformer states for identical prompt prefixes, yielding a 90% input cost reduction.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Circuit Breaker Spend Cap</h4>
        <p>An automated governance guard halting API invocations once a tenant reaches their predefined dollar expenditure limit.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand input vs output token pricing asymmetry and why constraining output tokens is vital.", value=True)
chk_2 = st.checkbox("I know when managed serverless APIs are cheaper than self-hosted GPU clusters.", value=True)
chk_3 = st.checkbox("I understand prompt caching economics and how to structure prompts to maximize cache hits.", value=True)
chk_4 = st.checkbox("I know how to route asynchronous workloads to Batch APIs to achieve 50% cost reductions.", value=True)
chk_5 = st.checkbox("I can implement a downgrade cascade that escalates to frontier models only on validation errors.", value=True)
chk_6 = st.checkbox("I know how to enforce per-tenant spend caps and circuit breakers to prevent budget exhaustion.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Cost Optimization competencies verified! Chapter 34 fully completed.")
