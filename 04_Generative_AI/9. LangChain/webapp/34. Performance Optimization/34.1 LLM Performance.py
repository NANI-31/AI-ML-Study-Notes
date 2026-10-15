import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="34.1 LLM Performance - LangChain Curriculum",
    page_icon="⚡",
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
    .perf-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-gain {
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
class LLMPerformanceEngine:
    """
    Simulates LLM throughput acceleration, context reduction algorithms,
    semantic caching, and tiered model latency profiling.
    """
    def __init__(self):
        self.cache_store = {
            "what is langchain expression language": "LCEL is a declarative syntax for composing streaming, async-ready Runnable chains.",
            "how do you trace an agent in langsmith": "Set LANGSMITH_TRACING=true and inject project metadata into the active run tree."
        }
        
        self.model_catalog = {
            "gpt-4o": {"type": "Frontier Flagship", "tokens_per_sec": 45, "ttft_ms": 380, "relative_cost": 1.00},
            "gpt-4o-mini": {"type": "Distilled SLM", "tokens_per_sec": 130, "ttft_ms": 190, "relative_cost": 0.06},
            "claude-3-5-haiku": {"type": "High-Throughput SLM", "tokens_per_sec": 145, "ttft_ms": 160, "relative_cost": 0.08},
            "llama-3.2-3b-q4": {"type": "Quantized Edge Model", "tokens_per_sec": 180, "ttft_ms": 110, "relative_cost": 0.01}
        }

    def check_exact_cache(self, query: str) -> Tuple[bool, Optional[str]]:
        normalized = query.strip().lower()
        if normalized in self.cache_store:
            return True, self.cache_store[normalized]
        return False, None

    def prune_context(self, raw_text: str, remove_stopwords: bool = True) -> Dict[str, Any]:
        words = raw_text.split()
        original_count = len(words)
        
        if remove_stopwords:
            stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
            filtered = [w for w in words if w.lower() not in stopwords]
        else:
            filtered = words
            
        pruned_count = len(filtered)
        savings_percent = round((1 - (pruned_count / max(original_count, 1))) * 100, 1)
        
        return {
            "original_tokens": original_count,
            "pruned_tokens": pruned_count,
            "reduction_percent": savings_percent,
            "pruned_text": " ".join(filtered)
        }

# Instantiate engine
perf_engine = LLMPerformanceEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: LLM Throughput Acceleration, Context Trimming, & Semantic Caching", divider="rainbow")

st.markdown("""
In enterprise generative AI, raw model quality is meaningless if an endpoint incurs a 4-second latency 
or exhausts operational cloud budgets. **LLM Performance Optimization** in LangChain encompasses a multi-tiered 
pipeline: intercepting repetitive requests via in-memory and semantic caches, aggressively pruning prompt and conversation 
contexts, routing queries to high-speed Small Language Models (SLMs), and optimizing token throughput.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LLM_Performance_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_ingress {
            label = "Request Ingress & Cache Interception Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            incoming_query [label="Incoming User Prompt\\n(Raw Text + Chat History)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            exact_cache [label="Exact In-Memory Cache\\n(Hash Lookup: O(1) latency < 2ms)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            semantic_cache [label="Semantic Vector Cache (Redis/GPTCache)\\n(Cosine distance threshold >= 0.94)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_optimization {
            label = "Context Compression & Prompt Tuning Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            context_pruner [label="Context Reduction & Pruner\\n(Sliding window k=8, summarization pass)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            prompt_compressor [label="Directive & Few-Shot Optimizer\\n(Replaces verbose text with concise Pydantic schemas)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            model_router [label="Complexity Model Router\\n(Simple queries -> SLM, Complex logic -> Frontier LLM)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        subgraph cluster_inference {
            label = "Tiered Execution Endpoints";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            frontier_llm [label="Frontier Model (GPT-4o / Claude 3.5 Sonnet)\\n(High reasoning, higher TTFT ~350ms)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            distilled_slm [label="High-Speed SLM (GPT-4o-mini / Haiku / Llama-3.2)\\n(Fast inference ~140 tok/s, TTFT ~120ms)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
        }

        incoming_query -> exact_cache;
        exact_cache -> semantic_cache [label="cache miss"];
        semantic_cache -> context_pruner [label="cache miss"];

        context_pruner -> prompt_compressor;
        prompt_compressor -> model_router;

        model_router -> distilled_slm [label="simple / extraction (80% traffic)"];
        model_router -> frontier_llm [label="complex reasoning / code (20% traffic)"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Zero-Compute Principle:</strong> The fastest, cheapest LLM call is the one that is never made. 
    A well-architected semantic cache satisfies 20% to 40% of production enterprise queries in under 15ms, 
    eliminating downstream provider rate limits and compute spend.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: TOKEN OPTIMIZATION
# ==============================================================================
st.header("2. Token Optimization: Tokenizer Encoding Mechanics and Whitespace Trimming", divider="rainbow")

st.markdown("""
Every byte passed into an LLM prompt consumes token budget. Understanding how tokenizers 
(such as Byte-Pair Encoding via `tiktoken`) split text is fundamental to optimizing throughput.
""")

st.markdown("""
### Tokenizer Encoding Traps
1. **Trailing Whitespace Inflation**: In Byte-Pair Encoding (BPE), trailing spaces or repeated newlines (`\\n\\n\\n`) often tokenize into individual inefficient single-byte tokens rather than merged multi-character words.
2. **Punctuation and Formatting Overhead**: Excessive ASCII decoration lines (e.g., `====================`) can consume 20 to 50 tokens each.
3. **JSON Schema Indentation**: Pretty-printing JSON with 4-space indentations multiplies whitespace tokens. Stripping indentation (`json.dumps(obj, separators=(',', ':'))`) reduces token overhead by up to 35%.
""")

st.code('''import json

raw_data = {"user_id": "usr_9981", "action": "checkout", "items": ["sku_1", "sku_2"]}

# Inefficient verbose formatting (4-space indentation)
verbose_json = json.dumps(raw_data, indent=4)

# Minified compact formatting (eliminates redundant whitespace tokens)
compact_json = json.dumps(raw_data, separators=(',', ':'))

print(f"Verbose Length: {len(verbose_json)} characters")
print(f"Compact Length: {len(compact_json)} characters (Saves ~30% token overhead)")
''', language="python")

# ==============================================================================
# SECTION 3: PROMPT OPTIMIZATION
# ==============================================================================
st.header("3. Prompt Optimization: Few-Shot Compression and Directive Pruning", divider="rainbow")

st.markdown("""
Over time, developer prompts suffer from **prompt bloat**—layers of redundant instructions, 
apologetic phrasing, and verbose few-shot examples accumulate, degrading both latency and model adherence.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Antipattern: Verbose Defensive Prompt
    ```markdown
    You are an extremely helpful and polite AI assistant. 
    Please make sure to always be courteous and friendly. 
    When the user asks you a question about their account, 
    please look up the information and provide a very detailed, 
    polite response explaining everything step by step. 
    Do not hallucinate and please do not make up false facts.
    ```
    - **Token Count**: ~75 tokens.
    - **Instruction Entropy**: High (vague directives like 'be courteous' confuse reasoning).
    """)

with col2:
    st.markdown("""
    ### Modern Standard: Compressed Directive Prompt
    ```markdown
    Role: Account Support Specialist.
    Task: Answer billing inquiries using provided context.
    Constraints:
    - Factual and concise (max 3 sentences).
    - If answer is not in context, reply 'Information not found.'
    ```
    - **Token Count**: ~28 tokens (62% reduction).
    - **Instruction Entropy**: Minimal (precise, machine-enforceable rules).
    """)

# ==============================================================================
# SECTION 4: CONTEXT REDUCTION
# ==============================================================================
st.header("4. Context Reduction: Sliding Window Buffers and Semantic Summarizers", divider="rainbow")

st.markdown("""
In multi-turn chat applications, blindly appending every message into conversation history causes prompt 
tokens to scale linearly with turn count. This leads to **quadratic token growth over time**, degrading Time to First Token.
""")

st.markdown("""
### Strategies for Conversation Context Pruning
- **Sliding Window Buffer (`k=6`)**: Keep only the most recent k interaction pairs. Older history is dropped.
- **Summary Window Memory**: When conversation history exceeds a token threshold (e.g., 2,000 tokens), an asynchronous background model compresses turns 1 through N - 2 into a 100-word persistent summary block.
- **Dynamic Entity Memory**: Instead of saving raw conversation text, extract and persist structured user profile attributes (`{name: 'Alice', plan: 'Enterprise'}`).
""")

# ==============================================================================
# SECTION 5: MODEL SELECTION
# ==============================================================================
st.header("5. Model Selection: Task-Specific Routing and Latency-Accuracy Pareto Analysis", divider="rainbow")

st.markdown("""
Defaulting to a frontier model (GPT-4o or Claude 3.5 Sonnet) for every trivial task is the leading cause 
of excessive latency and cloud bills. **Semantic Model Routing** classifies incoming queries and dispatches 
them to the optimal model tier.
""")

model_tier_df = pd.DataFrame([
    {"Task Category": "Intent Classification & Routing", "Recommended Model Tier": "Distilled SLM (Llama-3.2-1B / GPT-4o-mini)", "Typical Latency": "80ms - 150ms", "Cost Multiplier": "0.05x"},
    {"Task Category": "Structured JSON Entity Extraction", "Recommended Model Tier": "Fast SLM (GPT-4o-mini / Haiku)", "Typical Latency": "200ms - 400ms", "Cost Multiplier": "0.10x"},
    {"Task Category": "Document Summarization & Q&A", "Recommended Model Tier": "Mid-tier LLM (Gemini 1.5 Flash)", "Typical Latency": "300ms - 600ms", "Cost Multiplier": "0.15x"},
    {"Task Category": "Complex Multi-Hop Agentic Planning", "Recommended Model Tier": "Frontier Flagship (GPT-4o / Sonnet 3.5)", "Typical Latency": "900ms - 2500ms", "Cost Multiplier": "1.00x"}
])
st.dataframe(model_tier_df, use_container_width=True)

# ==============================================================================
# SECTION 6: SMALLER MODELS
# ==============================================================================
st.header("6. Smaller Models: Quantized Checkpoints and Distillation for Edge & Local Inference", divider="rainbow")

st.markdown("""
**Small Language Models (SLMs)**—models with 1B to 8B parameters—have achieved remarkable reasoning density 
due to high-quality synthetic data distillation. When quantized to 4-bit (AWQ / GGUF) or 8-bit precision, 
these models execute with microsecond token generation on commodity hardware.
""")

st.markdown("""
### Why SLMs Excel in Production Microservices
1. **Low Memory Footprint**: A 3B parameter model quantized to 4-bit occupies only ~2.2 GB of VRAM, fitting easily onto edge devices or cheap T4 GPU instances.
2. **Massive Tokens-Per-Second (TPS)**: SLMs generate 150+ tokens per second, making user-facing streaming feel instantaneous.
3. **Zero Data Egress & Fixed Cost**: Self-hosting quantized SLMs on internal infrastructure provides fixed monthly compute costs independent of request volume.
""")

# ==============================================================================
# SECTION 7: CACHING
# ==============================================================================
st.header("7. Caching: Exact Match InMemoryCache vs Semantic Vector Caching", divider="rainbow")

st.markdown("""
LangChain provides native caching hooks directly at the `set_llm_cache` layer, allowing applications 
to intercept identical prompts before dispatching provider network requests.
""")

c_c1, c_c2 = st.columns(2)
with c_c1:
    st.markdown("""
    ### 1. Exact Match Caching (`InMemoryCache` / `SQLiteCache`)
    - **Mechanism**: Computes a SHA-256 hash of the exact prompt string and model parameters.
    - **Pros**: Sub-millisecond lookup latency, zero embedding cost, 100% deterministic.
    - **Cons**: Rigid. A single differing space or punctuation mark results in a cache miss.
    """)

with c_c2:
    st.markdown("""
    ### 2. Semantic Caching (`RedisCache` / `GPTCache`)
    - **Mechanism**: Computes dense vector embeddings of the user query; hits cache if cosine similarity exceeds threshold (e.g., 0.95).
    - **Pros**: Captures semantically equivalent phrasing (e.g., "Reset password" vs "How do I change my password?").
    - **Cons**: Incurs small embedding latency (~15ms) and requires setting an appropriate distance threshold.
    """)

st.code('''from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_openai import ChatOpenAI

# 1. Enable Global Exact-Match In-Memory Cache
set_llm_cache(InMemoryCache())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# First call: executes network request to OpenAI (Latency ~400ms)
# res1 = llm.invoke("What is LangChain?")

# Second call with identical input: intercepted by InMemoryCache (Latency < 1ms!)
# res2 = llm.invoke("What is LangChain?")
''', language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: High-Throughput Cached and Compressed LCEL Chain", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script demonstrating a complete 
**high-throughput pipeline** incorporating in-memory caching, whitespace compression, and SLM model routing.
""")

st.code('''import time
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. High-Performance Local Cache Layer
CACHE_STORE: Dict[str, str] = {}

def get_cached_response(prompt_hash: str) -> str | None:
    return CACHE_STORE.get(prompt_hash)

def set_cached_response(prompt_hash: str, response: str):
    CACHE_STORE[prompt_hash] = response

# 2. Token Compression Preprocessor
def compress_prompt_text(text: str) -> str:
    """Strips redundant whitespace and cleans linebreaks."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return " ".join(lines)

# 3. Fast Execution Pipeline
def optimized_query_pipeline(raw_query: str) -> Dict[str, Any]:
    start_time = time.time()
    
    # Step A: Minify query text
    clean_query = compress_prompt_text(raw_query)
    query_key = clean_query.lower()
    
    # Step B: Cache Interception
    cached_val = get_cached_response(query_key)
    if cached_val:
        return {
            "query": clean_query,
            "response": cached_val,
            "cache_hit": True,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    # Step C: Simulated Distilled SLM Inference (GPT-4o-mini / Haiku)
    time.sleep(0.12) # Simulates ultra-fast 120ms SLM generation
    simulated_answer = f"Optimized SLM answer for: {clean_query}"
    
    # Update cache
    set_cached_response(query_key, simulated_answer)
    
    return {
        "query": clean_query,
        "response": simulated_answer,
        "cache_hit": False,
        "latency_ms": round((time.time() - start_time) * 1000, 2)
    }

# Execution Run 1 (Cache Miss)
r1 = optimized_query_pipeline("   What are the core advantages of     using an SLM?   ")
print(f"Run 1 (Cache Miss): Latency = {r1['latency_ms']}ms | Cache Hit = {r1['cache_hit']}")

# Execution Run 2 (Cache Hit)
r2 = optimized_query_pipeline("What are the core advantages of using an SLM?")
print(f"Run 2 (Cache Hit):  Latency = {r2['latency_ms']}ms | Cache Hit = {r2['cache_hit']}")
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Token Pruner, Semantic Cache Simulator, & Model Latency Benchmarker", divider="rainbow")

st.markdown("""
Experiment with real-time context reduction algorithms, test in-memory cache lookups, 
and compare latency-throughput characteristics across model checkpoints.
""")

wb_view = st.selectbox(
    "Select Optimization Tool:",
    ["1. Context Reduction & Token Pruner Sandbox", "2. In-Memory Cache Lookup Tester", "3. Model Throughput & Latency Matrix"]
)

if wb_view == "1. Context Reduction & Token Pruner Sandbox":
    st.subheader("Prompt Token Compression Sandbox")
    sample_prompt = st.text_area(
        "Enter Raw Input Prompt:",
        "Please be advised that the system is going to be undergoing scheduled maintenance on the weekend for all users in the organization."
    )
    strip_stops = st.toggle("Strip Syntactic Stopwords & Filler Words", value=True)

    prune_res = perf_engine.prune_context(sample_prompt, strip_stops)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Original Word Count", f"{prune_res['original_tokens']}")
    with m2:
        st.metric("Pruned Word Count", f"{prune_res['pruned_tokens']}")
    with m3:
        st.metric("Token Reduction", f"{prune_res['reduction_percent']}%")

    st.markdown("#### Compressed Result:")
    st.info(prune_res["pruned_text"])

elif wb_view == "2. In-Memory Cache Lookup Tester":
    st.subheader("In-Memory Exact Cache Simulator")
    test_q = st.text_input("Enter Query to Test Against Cache:", "what is langchain expression language")

    if st.button("Query Pipeline"):
        hit, resp = perf_engine.check_exact_cache(test_q)
        if hit:
            st.success(f"CACHE HIT (Latency < 2ms): {resp}")
        else:
            st.warning("CACHE MISS (Would dispatch to model provider, Latency ~350ms)")
            st.caption("Tip: Try querying 'what is langchain expression language' or 'how do you trace an agent in langsmith'.")

else:
    st.subheader("Model Throughput & Latency Matrix")
    m_df = pd.DataFrame([
        {
            "Model Checkpoint": k,
            "Architecture Tier": v["type"],
            "Generation Speed (Tokens/Sec)": f"{v['tokens_per_sec']} tok/s",
            "Avg TTFT": f"{v['ttft_ms']}ms",
            "Relative Cost Multiplier": f"{v['relative_cost']}x"
        }
        for k, v in perf_engine.model_catalog.items()
    ])
    st.dataframe(m_df, use_container_width=True)
    st.info("SLMs (GPT-4o-mini, Haiku, Llama-3.2) deliver 3x higher token generation throughput at 90% lower operational cost.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: How does Time to First Token (TTFT) differ from completion generation speed, and how do you optimize each?"):
    st.markdown("""
    **Answer:**  
    - **TTFT** is the delay before the first token byte arrives at the client, dominated by prompt context length and network queueing. Optimize TTFT via prompt compression, prefix prompt caching (Anthropic), and semantic caching.
    - **Completion Generation Speed** is the rate (tokens/sec) at which the model autoregressively synthesizes tokens. Optimize generation speed by using smaller models, 4-bit quantization, or high-throughput inference engines like vLLM / Groq.
    """)

with st.expander("Q2: What is the primary operational trade-off between Exact Match Caching and Semantic Vector Caching?"):
    st.markdown("""
    **Answer:**  
    Exact match caching has zero false-positive risk and sub-millisecond hash lookup latency, but misses minor paraphrasing. 
    Semantic caching captures variable phrasing, but requires calculating a query embedding vector (~15ms latency), 
    and introduces the risk of false-positive cache hits if the cosine similarity threshold is set too leniently.
    """)

with st.expander("Q3: Why can whitespace and formatting cause unexpected token cost spikes in BPE tokenizers?"):
    st.markdown("""
    **Answer:**  
    Byte-Pair Encoding merges common multi-character words into single tokens, but repeated spaces, tabs, or rare 
    newline combinations often cannot be merged and must be encoded as individual byte tokens. Minifying JSON payloads 
    and stripping trailing whitespace prevents this inflation.
    """)

with st.expander("Q4: How does dynamic model routing reduce production cloud costs while preserving reasoning accuracy?"):
    st.markdown("""
    **Answer:**  
    In typical production workloads, 70% to 80% of incoming queries are simple tasks (extraction, classification, greeting). 
    A lightweight classifier routes simple queries to ultra-cheap SLMs (costing ~0.06x), reserving costly frontier models 
    only for complex multi-hop reasoning. This reduces blended system costs by 70%+.
    """)

with st.expander("Q5: What are the risks of using sliding-window memory (k=6) in long multi-turn conversations?"):
    st.markdown("""
    **Answer:**  
    A pure sliding-window memory drops older messages completely. If a user mentioned critical constraints in turn 1 
    (e.g., 'My name is Bob and I am allergic to peanuts'), the model will forget this by turn 8. Mitigate this by pairing 
    sliding windows with a persistent summary buffer or structured entity store.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Using Frontier Flagship Models for Simple Extraction and Routing**  
Dispatching simple text classification or JSON extraction to GPT-4o multiplies latency and costs by 15x. 
Always benchmark small distilled models first.
""")

st.error("""
**Trap 2: Setting Semantic Cache Similarity Thresholds Too Low (< 0.90)**  
Setting a loose cosine similarity threshold causes semantically distinct queries (e.g., 'Delete account' vs 'Upgrade account') 
to return cached responses from one another, resulting in severe bugs.
""")

st.warning("""
**Trap 3: Unbounded Conversation History Appending**  
Failing to implement a sliding window or summarizer pass allows conversation context to explode into tens of thousands of tokens, 
causing quadratic cost spikes and severe TTFT degradation.
""")

st.warning("""
**Trap 4: Pretty-Printing JSON Payloads in Production Prompts**  
Passing 4-space formatted JSON into prompts wastes 30% of your token budget on whitespace characters alone. 
Always minify JSON strings.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Semantic Caching</h4>
        <p>A technique that checks query vector proximity against historical prompts, returning cached answers when cosine similarity exceeds a strict threshold.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Time to First Token (TTFT)</h4>
        <p>The duration required for the model to process the prompt and emit the first streaming token, dictated by prompt size and prefill compute.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Small Language Model (SLM)</h4>
        <p>High-density 1B to 8B parameter models optimized for rapid inference, low memory footprint, and low operational cost.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Model Router</h4>
        <p>An architectural component that classifies query complexity to dynamically route tasks to the cheapest viable model tier.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how tokenizer encoding mechanics affect token counts and how to minify whitespace.", value=True)
chk_2 = st.checkbox("I understand prompt directive pruning and how to eliminate apologetic prompt bloat.", value=True)
chk_3 = st.checkbox("I can implement sliding window buffers and summary memory to prevent quadratic context inflation.", value=True)
chk_4 = st.checkbox("I know how to design a tiered model routing policy to balance latency, cost, and reasoning power.", value=True)
chk_5 = st.checkbox("I understand the trade-offs between exact in-memory caching and semantic vector caching.", value=True)
chk_6 = st.checkbox("I can differentiate Time to First Token (TTFT) from generation speed and apply targeted optimizations.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LLM Performance optimization competencies verified! Ready for RAG Performance.")
