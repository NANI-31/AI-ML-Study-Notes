import streamlit as st
import json

st.set_page_config(
    page_title="27.3 Cost Optimization - Prompt Engineering",
    page_icon="💰",
    layout="wide"
)

# Header Section
st.title("27.3 Enterprise Cost Optimization & FinOps Governance")
st.caption("Chapter 27: Performance and Cost Optimization | Domain: 1. Prompt Engineering")
st.markdown("""
Without disciplined architectural governance, generative AI costs scale linearly with user traffic, rapidly eroding product margins. 
Enterprise LLM FinOps focuses on **token budgeting, tiered model routing, automated prompt prefix caching, 
asynchronous batch discounts, and continuous Cost-per-Successful-Query (CPSQ) monitoring**.
""")

st.divider()

# Section 1: The Five Pillars of LLM Cost Optimization
st.header("1. The Five Pillars of Enterprise FinOps", divider="rainbow")
st.markdown("""
Achieving an 80%+ reduction in inference bills without sacrificing output quality requires combining all five pillars:
""")

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.subheader("1. Tiered Model Right-Sizing")
    st.markdown("""
    Never use a frontier reasoning model (e.g. GPT-4o, Claude 3.5 Sonnet at $2.50-$3.00/M tokens) 
    for simple tasks that an edge model (GPT-4o-mini, Claude 3 Haiku at $0.15-$0.25/M tokens) can execute with identical accuracy.
    * Route 75% of volume to mini models.
    * Reserve frontier models for complex multi-page synthesis.
    """)

    st.subheader("2. Provider Prompt Prefix Caching")
    st.markdown("""
    Structure all prompts so that static system instructions, schemas, and few-shot exemplars appear 
    at the start of the prompt (1,024+ tokens). 
    Providers automatically bill cached tokens at a **50% to 90% discount**!
    """)

    st.subheader("3. Asynchronous Batch API (50% Off)")
    st.markdown("""
    Offload non-real-time jobs (backfill labeling, nocturnal evaluations, document indexing) 
    to the Batch API, slashing token costs by exactly half across both input and output tokens.
    """)

with col_p2:
    st.subheader("4. Semantic Application Caching (Redis)")
    st.markdown("""
    Store question-and-answer pairs in an in-memory vector cache. 
    If a user asks a question with 96%+ cosine similarity to an existing record, serve the answer from Redis at $0.00 model cost.
    """)

    st.subheader("5. Strict Token Budgets & Hard Ceilings")
    st.markdown("""
    Enforce tenant-level and user-level token quotas. 
    Wrap all completion requests with sensible `max_tokens` boundaries (e.g. 350 tokens) to prevent rogue rambling loops from draining credits.
    """)

st.divider()

# Section 2: Cost per Successful Query (CPSQ) Metric
st.header("2. The True Metric: Cost per Successful Query (CPSQ)", divider="rainbow")
st.markdown("""
A common engineering pitfall is choosing the cheapest model, only to realize that 30% of its outputs fail JSON validation 
or hallucinate, requiring multiple expensive retries or human customer support intervention.
""")

st.code('''# Cost per Successful Query (CPSQ) Formula & Computation
def calculate_cpsq(total_api_spend_usd: float, total_queries: int, success_rate: float) -> dict:
    successful_queries = total_queries * success_rate
    if successful_queries <= 0:
        return {"cpsq": float('inf'), "effective_overhead": 0.0}
        
    nominal_cost_per_query = total_api_spend_usd / total_queries
    actual_cpsq = total_api_spend_usd / successful_queries
    waste_percentage = (1.0 - success_rate) * 100
    
    return {
        "nominal_cost_per_query": round(nominal_cost_per_query, 5),
        "true_cpsq": round(actual_cpsq, 5),
        "waste_percentage": round(waste_percentage, 2),
        "wasted_dollars": round(total_api_spend_usd * (1.0 - success_rate), 2)
    }''', language="python")

st.divider()

# Section 3: Tiered Model Router Implementation Pattern
st.header("3. Tiered Model Routing Architecture Pattern", divider="rainbow")
st.markdown("""
The code pattern below illustrates how an intelligent gateway routes queries to save 75% on inference spend:
""")

st.code('''# Dynamic Tiered Cost-Saving Router
def route_and_execute_task(user_prompt: str, client):
    # Rule 1: Fast Heuristic Classifier (Lightweight)
    word_count = len(user_prompt.split())
    is_complex_coding = any(k in user_prompt.lower() for k in ["refactor", "algorithm", "asyncio", "memory leak"])
    
    if is_complex_coding or word_count > 400:
        # Route to Frontier Tier (GPT-4o)
        model_target = "gpt-4o"
    else:
        # Route to High-Speed Lightweight Tier (GPT-4o-mini: 93% cheaper!)
        model_target = "gpt-4o-mini"
        
    response = client.chat.completions.create(
        model=model_target,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=400,
        temperature=0.1
    )
    return response, model_target''', language="python")

st.divider()

# Section 4: Interactive FinOps Master Calculator
st.header("4. Interactive FinOps Master Budget Simulator", divider="rainbow")
st.markdown("Model the financial impact of applying the 5 cost optimization pillars across monthly volume.")

col_f1, col_f2 = st.columns(2)

with col_f1:
    monthly_reqs = st.number_input("Monthly Production Requests", value=1000000, step=250000)
    in_toks = st.slider("Average Input Tokens", 500, 5000, 1500, step=100)
    out_toks = st.slider("Average Output Tokens", 100, 1000, 300, step=50)
    pct_mini = st.slider("Percentage Routed to Mini Model (%)", 0, 95, 75, step=5)
    pct_cache = st.slider("Semantic / KV Cache Hit Rate (%)", 0, 80, 40, step=5)
    pct_batch = st.slider("Percentage Handled via Async Batch (50% Off) (%)", 0, 80, 20, step=5)

with col_f2:
    # Pricing benchmarks ($ per 1M tokens)
    # Frontier: $2.50 in, $10.00 out
    # Mini: $0.15 in, $0.60 out
    
    # 1. Baseline: 100% Frontier, zero cache, zero batch
    base_in = (monthly_reqs * in_toks / 1e6) * 2.50
    base_out = (monthly_reqs * out_toks / 1e6) * 10.00
    baseline_cost = base_in + base_out
    
    # 2. Optimized calculation
    frontier_share = (100 - pct_mini) / 100.0
    mini_share = pct_mini / 100.0
    
    # Blended token rates
    blended_in_rate = (frontier_share * 2.50) + (mini_share * 0.15)
    blended_out_rate = (frontier_share * 10.00) + (mini_share * 0.60)
    
    raw_opt_in = (monthly_reqs * in_toks / 1e6) * blended_in_rate
    raw_opt_out = (monthly_reqs * out_toks / 1e6) * blended_out_rate
    
    # Apply cache discount (free for cached output, 50% discount on cached input)
    cached_share = pct_cache / 100.0
    opt_in = raw_opt_in * (1.0 - (cached_share * 0.5))
    opt_out = raw_opt_out * (1.0 - cached_share)
    
    # Apply batch discount (50% on batch share)
    batch_share = pct_batch / 100.0
    final_cost = (opt_in + opt_out) * (1.0 - (batch_share * 0.5))
    
    monthly_savings = baseline_cost - final_cost
    pct_saved = (monthly_savings / baseline_cost) * 100
    
    st.subheader("FinOps Optimization Impact")
    m1, m2 = st.columns(2)
    m1.metric("Unoptimized Baseline Spend", f"${baseline_cost:,.2f}")
    m2.metric("Optimized Production Spend", f"${final_cost:,.2f}")
    st.metric("Total Monthly Net Savings", f"${monthly_savings:,.2f}", delta=f"-{pct_saved:.1f}%")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: What is CPSQ (Cost Per Successful Query) and why is it superior to Cost Per Token?"):
        st.write("""
        Cost Per Token measures only input expenditure without accounting for outcome quality. 
        If a cheap model produces incorrect formatting or hallucinations on 25% of requests, those queries 
        must either be retried (doubling cost) or escalated to human support agents (costing $5-$15 per ticket). 
        **CPSR divides total spend strictly by queries that successfully pass automated validation rules**, 
        providing the true economic efficiency of the prompt architecture.
        """)

    with st.expander("Q: How does structured Prompt Prefix Alignment maximize provider KV cache discounts?"):
        st.write("""
        Inference providers like OpenAI and Anthropic check for prefix matches starting from token 0. 
        If you inject dynamic timestamps or user IDs at the very top of your prompt, the entire prefix hash changes, 
        invalidating the cache for every subsequent token. 
        By keeping the first 1,024+ tokens strictly static (system rules, few-shot examples, tools), 
        providers reuse the GPU KV cache across all requests, automatically saving 50% on input costs.
        """)

with qa2:
    with st.expander("Trap to Avoid: Failing to Implement Per-User Rate Quotas"):
        st.write("""
        Without hard user-level rate limits, a single automated script, malicious user, or testing engineer 
        can send 10,000 recursive requests to your API in minutes, exhausting your organization's entire monthly budget. 
        Always enforce token bucket rate limits per user/IP address at your API gateway layer.
        """)

    with st.expander("Flashcards: FinOps Terminology"):
        fc_sel = st.selectbox(
            "Select FinOps Term:",
            ["CPSQ", "KV Prefix Alignment", "Tiered Routing", "Token Bucket Rate Limiting"]
        )
        if fc_sel == "CPSQ":
            st.info("Total infrastructure spend divided by the count of requests that successfully meet quality SLAs.")
        elif fc_sel == "KV Prefix Alignment":
            st.info("Placing static prompt tokens at the beginning to maximize 50% KV cache discounts.")
        elif fc_sel == "Tiered Routing":
            st.info("Dispatching standard queries to cheap models and reserving frontier models for hard tasks.")
        else:
            st.info("An algorithm that enforces request and token consumption limits per user/tenant.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I know how to compute Cost per Successful Query (CPSQ).")
    st.checkbox("I can implement a tiered model router in Python.")
with c2:
    st.checkbox("I understand how to structure prompts to trigger automatic KV cache discounts.")
    st.checkbox("I know how to leverage the asynchronous Batch API for 50% cost reductions.")
