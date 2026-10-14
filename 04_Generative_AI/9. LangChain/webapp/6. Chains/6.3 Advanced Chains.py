import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="6.3 Advanced Chains - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("6.3 Advanced Chains: Concurrency, Routing, Resilience & Fault-Tolerant Orchestration")
st.caption("Chapter 6: Chains | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In enterprise AI architectures, linear chains (`Prompt | Model | Parser`) represent only the initial baseline. 
Real-world systems must handle multi-branch fan-outs, dynamic runtime routing, unpredictable API rate limits, 
upstream model outages, and conditional decision logic.

LangChain Expression Language (LCEL) elevates chains from static directed graphs to **dynamic, resilient, fault-tolerant state machines**.
In this lesson, you will master the 8 foundational pillars of advanced chain orchestration:
1. **Parallel Chains (`RunnableParallel`)**: Concurrent execution of decoupled cognitive tasks with latency bounded by `max(t_i)`.
2. **Branching (`RunnableBranch`)**: Declarative conditional branching based on schema inspections and predicate functions.
3. **Routing**: Semantic, rule-based, and embedding-driven query dispatch to specialized sub-chains.
4. **Conditional Execution**: Dynamic boolean guards, assertion checks, and cost-saving short-circuit pathways.
5. **Dynamic Chains**: Runtime chain construction, schema adaptation, and prompt injection based on session parameters.
6. **Error Handling**: Graceful degradation, exception suppression, and contextual rescue wrappers.
7. **Retry Mechanisms (`with_retry`)**: Exponential backoff with random jitter to absorb transient network hiccups and HTTP 429 rate limits.
8. **Fallback Models (`with_fallbacks`)**: High-availability multi-provider failover chains ensuring 99.99% uptime.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockExecutionSpan:
    """Represents a simulated execution span within an advanced chain pipeline."""
    def __init__(self, step_name: str, status: str, duration_ms: int, details: str, model_used: str = "N/A"):
        self.step_name = step_name
        self.status = status  # 'SUCCESS', 'RETRY', 'FALLBACK', 'SHORT_CIRCUIT'
        self.duration_ms = duration_ms
        self.details = details
        self.model_used = model_used

class MockAdvancedChainEngine:
    """Simulates an enterprise resilient chain with parallel branches, routing, retries, and fallbacks."""
    def __init__(self):
        pass

    def execute_request(
        self,
        query: str,
        simulate_primary_outage: bool,
        simulate_rate_limit: bool,
        enable_parallel_analysis: bool,
        max_retries: int
    ) -> Dict[str, Any]:
        spans: List[MockExecutionSpan] = []
        total_time = 0

        # Step 1: Routing / Intent Classification
        query_lower = query.lower()
        if any(w in query_lower for w in ["bug", "error", "traceback", "exception", "code", "function"]):
            route = "Engineering / Code Review"
            confidence = 0.94
        elif any(w in query_lower for w in ["billing", "invoice", "refund", "credit", "subscription", "price"]):
            route = "Finance & Billing"
            confidence = 0.98
        elif any(w in query_lower for w in ["legal", "terms", "gdpr", "compliance", "privacy"]):
            route = "Compliance & Legal"
            confidence = 0.91
        else:
            route = "General Support"
            confidence = 0.82

        spans.append(MockExecutionSpan(
            step_name="1. Dynamic Intent Router",
            status="SUCCESS",
            duration_ms=25,
            details=f"Classified query into route '{route}' with confidence {confidence * 100:.1f}%.",
            model_used="Semantic Router Rule"
        ))
        total_time += 25

        # Step 2: Parallel Analysis (if enabled)
        parallel_results = {}
        if enable_parallel_analysis:
            # Three concurrent branches: Sentiment, Key Entities, Urgency
            p1_time = 140
            p2_time = 195
            p3_time = 160
            parallel_latency = max(p1_time, p2_time, p3_time)
            
            sentiment = "Frustrated / High Friction" if any(w in query_lower for w in ["urgent", "broken", "fail", "angry", "terrible", "cancel"]) else "Neutral / Informative"
            entities = ["payment_gateway", "user_auth"] if "login" in query_lower or "pay" in query_lower else ["platform_core"]
            urgency = "P1-CRITICAL" if "urgent" in query_lower or "immediately" in query_lower else "P3-NORMAL"

            parallel_results = {
                "sentiment": sentiment,
                "entities": entities,
                "urgency_level": urgency
            }

            spans.append(MockExecutionSpan(
                step_name="2. Parallel Fan-Out (RunnableParallel)",
                status="SUCCESS",
                duration_ms=parallel_latency,
                details=f"Ran 3 branches concurrently. Latency bounded by max({p1_time}, {p2_time}, {p3_time}) = {parallel_latency} ms. Merged metadata into state.",
                model_used="Lightweight Embedding / Extractor"
            ))
            total_time += parallel_latency

        # Step 3: LLM Synthesis with Retry and Fallback logic
        primary_model = "Claude-3-5-Sonnet"
        secondary_model = "GPT-4o-mini"
        tertiary_model = "Llama-3-70b-Local"

        # Simulate Retry Logic
        if simulate_rate_limit:
            retry_count = 0
            while retry_count < max_retries and retry_count < 2:
                retry_count += 1
                backoff_ms = int(50 * (2 ** retry_count) + np.random.randint(10, 30))
                total_time += backoff_ms
                spans.append(MockExecutionSpan(
                    step_name=f"3.{retry_count} Rate Limit Retry (HTTP 429)",
                    status="RETRY",
                    duration_ms=backoff_ms,
                    details=f"Primary model hit 429 Too Many Requests. Applied exponential backoff with jitter: delayed {backoff_ms} ms.",
                    model_used=primary_model
                ))

        # Simulate Fallback Model Execution
        if simulate_primary_outage:
            spans.append(MockExecutionSpan(
                step_name="3. Primary Model Invocation",
                status="FALLBACK",
                duration_ms=120,
                details="Primary model timed out with HTTP 503 Service Unavailable. with_fallbacks interceptor caught exception.",
                model_used=primary_model
            ))
            total_time += 120

            # Execute Secondary Fallback
            spans.append(MockExecutionSpan(
                step_name="4. Secondary Fallback Invocation",
                status="SUCCESS",
                duration_ms=310,
                details=f"Secondary model responded successfully without downtime. Synthesized response for '{route}'.",
                model_used=secondary_model
            ))
            total_time += 310
            active_model = secondary_model
            response_text = f"[{secondary_model} - Fallback Active] Successfully handled inquiry regarding '{route}'. Context and safety checks validated."
        else:
            spans.append(MockExecutionSpan(
                step_name="3. Primary Model Invocation",
                status="SUCCESS",
                duration_ms=450,
                details=f"Primary model executed with high fidelity response. Prompt routed to specialized '{route}' pipeline.",
                model_used=primary_model
            ))
            total_time += 450
            active_model = primary_model
            response_text = f"[{primary_model} - Primary Engine] Full authoritative diagnosis for query under '{route}' domain."

        return {
            "route": route,
            "confidence": confidence,
            "parallel_results": parallel_results,
            "active_model": active_model,
            "response_text": response_text,
            "total_latency_ms": total_time,
            "spans": spans
        }

# --- SECTION 1: ARCHITECTURE BLUEPRINT & SYSTEM TOPOLOGY ---
st.header("1. Architecture Blueprint: Resilient Multi-Branch LCEL Topology", divider="rainbow")

st.markdown("""
Modern enterprise chains operate as **directed execution graphs** equipped with dynamic conditional branching, 
concurrency primitives, and automated resilience barriers. The architecture below details the complete lifecycle 
of an advanced LangChain LCEL pipeline from raw query intake to fault-tolerant delivery.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph G {
        bgcolor="#0B1120";
        rankdir=TB;
        splines=spline;
        node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B", arrowsize=0.8, penwidth=1.2];

        // Level 1: Input & Sanitization
        subgraph cluster_input {
            label = "1. INGESTION &amp; PRECONDITION TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            UserQuery [label="User Query Payload\\n{query, auth_tier, session_id}"];
            GuardValidator [label="RunnableLambda(validate_schema)\\nBoolean Guard &amp; Rate Check"];
        }

        // Level 2: Router & Branching
        subgraph cluster_routing {
            label = "2. ROUTING &amp; CONDITIONAL DISPATCH TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            IntentRouter [label="Dynamic Router\\n(Semantic / Heuristic Classifier)"];
            BranchDecider [label="RunnableBranch / Dynamic Dispatch\\nPredicate Matching"];
        }

        // Level 3: Concurrency Fan-out
        subgraph cluster_parallel {
            label = "3. CONCURRENT FAN-OUT TIER (RunnableParallel)";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#0F172A"];
            BranchDocRetrieval [label="Branch A: Vector Retriever\\nHybrid Search (BM25 + Dense)"];
            BranchSentiment [label="Branch B: Sentiment Analyzer\\nZero-Shot Classifier"];
            BranchSecurity [label="Branch C: PII &amp; Security Guard\\nRegex + Guardrail Model"];
        }

        // Level 4: Synthesis & Resilience
        subgraph cluster_resilience {
            label = "4. RESILIENT SYNTHESIS TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            StateMerge [label="RunnablePassthrough.assign()\\nState Consolidation Dict"];
            PrimaryLLM [label="Primary Model (Claude 3.5 Sonnet)\\nWrapped with .with_retry()", color="#10B981"];
            FallbackLLM [label="Fallback Model (GPT-4o-mini)\\nActivated via .with_fallbacks()", color="#F59E0B"];
            RescueHandler [label="Rescue / Degraded Response Node\\nwith_exception_handling()", color="#EF4444"];
        }

        // Level 5: Delivery
        subgraph cluster_output {
            label = "5. STRUCTURED DELIVERY TIER";
            style = "dashed,rounded";
            color = "#334155";
            fontcolor = "#94A3B8";
            node [fillcolor="#1E293B"];
            OutputParser [label="PydanticOutputParser\\nTyped Output Schema"];
            FinalPayload [label="Validated API Response Object\\n{status, route, payload, telemetry}"];
        }

        // Wiring
        UserQuery -> GuardValidator;
        GuardValidator -> IntentRouter [label="Passed Validation"];
        GuardValidator -> RescueHandler [label="Validation Failure (400)", style=dotted, color="#EF4444"];
        
        IntentRouter -> BranchDecider;
        BranchDecider -> BranchDocRetrieval [label="Thread 1 (Parallel)"];
        BranchDecider -> BranchSentiment [label="Thread 2 (Parallel)"];
        BranchDecider -> BranchSecurity [label="Thread 3 (Parallel)"];

        BranchDocRetrieval -> StateMerge;
        BranchSentiment -> StateMerge;
        BranchSecurity -> StateMerge;

        StateMerge -> PrimaryLLM [label="Invoke Primary"];
        PrimaryLLM -> OutputParser [label="HTTP 200 OK"];
        PrimaryLLM -> FallbackLLM [label="HTTP 503 / Timeout (Fallback)", style=dotted, color="#F59E0B"];
        FallbackLLM -> OutputParser [label="Fallback Success"];
        FallbackLLM -> RescueHandler [label="Secondary Failure", style=dotted, color="#EF4444"];

        RescueHandler -> FinalPayload [label="Degraded Payload"];
        OutputParser -> FinalPayload [label="Clean Response"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
**Key Architectural Guarantees:**
- **Zero-Block Concurrency**: Branches A, B, and C execute in parallel; total latency is determined by the slowest branch rather than their sum.
- **Failover Redundancy**: If the primary frontier model experiences a regional network partition or service degradation, the chain switches to secondary models without crashing.
- **Circuit Breakers**: Upstream payload guards discard invalid requests prior to token consumption.
""")

st.divider()

# --- SECTION 2: PARALLEL CHAINS ---
st.header("2. Parallel Chains (`RunnableParallel` & Concurrent Execution)", divider="rainbow")

st.markdown("""
In many LLM workflows, a single user input requires multiple independent cognitive or data retrieval tasks. 
For instance, a customer support agent might need to simultaneously:
1. Search the company knowledge base for relevant policy documents.
2. Analyze the sentiment and emotional escalation of the user's message.
3. Query an internal SQL database for customer purchase history.

Executing these sequentially creates cumulative latency: `total_latency = t_retrieval + t_sentiment + t_sql` (often 3 to 6 seconds). 
With `RunnableParallel`, all branches run concurrently in parallel threads or asynchronous tasks, bounding latency to:
`total_latency = max(t_retrieval, t_sentiment, t_sql) + overhead` (typically 1 to 2 seconds).
""")

# Contextual Graphviz Diagram for Parallel Execution
parallel_dot = """
digraph G {
    bgcolor="#0B1120";
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", fillcolor="#1E293B"];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B"];

    InputQuery [label="Input: {'topic': 'Quantum Computing'}"];
    
    subgraph cluster_parallel_box {
        label = "RunnableParallel(summary=..., code_example=..., citations=...)";
        style = "dashed";
        color = "#0284C7";
        fontcolor = "#38BDF8";

        Branch1 [label="Branch 'summary':\\nPrompt | Model | StrParser\\n[Latency: 1.2s]", fillcolor="#0F172A"];
        Branch2 [label="Branch 'code_example':\\nPrompt | Model | StrParser\\n[Latency: 1.8s]", fillcolor="#0F172A"];
        Branch3 [label="Branch 'citations':\\nRetriever | FormatDocs\\n[Latency: 0.6s]", fillcolor="#0F172A"];
    }

    JoinedState [label="Joined Dictionary Output:\\n{'summary': ..., 'code_example': ..., 'citations': ...}\\n[Total Wall Time: ~1.8s]"];

    InputQuery -> Branch1;
    InputQuery -> Branch2;
    InputQuery -> Branch3;

    Branch1 -> JoinedState;
    Branch2 -> JoinedState;
    Branch3 -> JoinedState;
}
"""
st.graphviz_chart(parallel_dot, use_container_width=True)

st.subheader("Implementing `RunnableParallel` in LCEL")
st.code('''from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

# Define Branch 1: High-level summary
summary_chain = (
    ChatPromptTemplate.from_template("Summarize the significance of {topic} in 2 sentences.")
    | model
    | parser
)

# Define Branch 2: Technical code snippet
code_chain = (
    ChatPromptTemplate.from_template("Provide a concise Python snippet illustrating {topic}.")
    | model
    | parser
)

# Define Branch 3: Industry use-cases
use_case_chain = (
    ChatPromptTemplate.from_template("List 3 practical enterprise applications of {topic}.")
    | model
    | parser
)

# Combine into a single parallel runnable
concurrent_research_chain = RunnableParallel({
    "topic": RunnablePassthrough(),  # Preserves the original input
    "summary": summary_chain,
    "code": code_chain,
    "use_cases": use_case_chain
})

# Invocation: all 3 branches run concurrently across thread pools
result = concurrent_research_chain.invoke("Retrieval-Augmented Generation")
print(result.keys())
# Output: dict_keys(['topic', 'summary', 'code', 'use_cases'])
''', language="python")

st.divider()

# --- SECTION 3: BRANCHING ---
st.header("3. Branching (`RunnableBranch` & Dynamic Path Selection)", divider="rainbow")

st.markdown("""
While parallel chains run multiple branches simultaneously, **Branching** selects **exactly one execution path** 
out of several candidates based on runtime conditions.

Historically, LangChain used the `RunnableBranch` construct. In modern LCEL, branching can be accomplished either via 
`RunnableBranch((condition_1, runnable_1), (condition_2, runnable_2), default_runnable)` or via a dynamic routing function 
wrapped in `RunnableLambda`.
""")

st.subheader("Declarative Branching Syntax")
st.code('''from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

# Specialized Sub-Chains
sql_chain = ChatPromptTemplate.from_template("Write a PostgreSQL query for: {query}") | model | StrOutputParser()
python_chain = ChatPromptTemplate.from_template("Write a clean Python script for: {query}") | model | StrOutputParser()
general_chain = ChatPromptTemplate.from_template("Answer this technical query concisely: {query}") | model | StrOutputParser()

# Condition Predicates
def is_sql_intent(x: dict) -> bool:
    q = x.get("query", "").lower()
    return any(k in q for k in ["select", "table", "sql", "database", "schema", "join"])

def is_python_intent(x: dict) -> bool:
    q = x.get("query", "").lower()
    return any(k in q for k in ["python", "def", "class", "pandas", "numpy", "list comprehension"])

# Assemble RunnableBranch:
# Format: RunnableBranch((predicate_1, runnable_1), (predicate_2, runnable_2), ..., default_runnable)
branching_chain = RunnableBranch(
    (is_sql_intent, sql_chain),
    (is_python_intent, python_chain),
    general_chain  # Default fallback path
)

# Invocation
sql_res = branching_chain.invoke({"query": "Find top 5 customers with highest revenue in orders table"})
py_res = branching_chain.invoke({"query": "Write a python function to invert a binary tree"})
''', language="python")

st.info("""
💡 **Best Practice Note**: When conditional logic involves more than 3 branches or requires complex semantic classification, 
prefer dedicated **Semantic Routing** or dynamic dispatch functions over deeply nested `RunnableBranch` structures.
""")

st.divider()

# --- SECTION 4: ROUTING ---
st.header("4. Routing (Intent Classifiers & Semantic Dispatching)", divider="rainbow")

st.markdown("""
**Routing** is the strategic separation of query understanding from query execution. Rather than relying on simple 
substring keywords, enterprise architectures employ dedicated intent classifiers to dispatch incoming requests to the 
most cost-effective and specialized models.
""")

routing_dot = """
digraph G {
    bgcolor="#0B1120";
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", fillcolor="#1E293B"];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B"];

    UserQuery [label="User Prompt: 'My credit card was charged twice'"];
    
    Classifier [label="Intent Router Model (Fast/Cheap)\\nPrompt: 'Classify into [Billing, Tech, General]'\\nResult: 'Billing' (0.98 confidence)", fillcolor="#0284C7"];

    subgraph cluster_routes {
        label = "Specialized Domain Sub-Chains";
        style = "dashed";
        color = "#334155";
        fontcolor = "#94A3B8";

        BillingChain [label="Billing Chain:\\nPrompt + Stripe Tool + GPT-4o", fillcolor="#0F172A", color="#10B981"];
        TechChain [label="Tech Support Chain:\\nPrompt + Jira Retriever + Claude", fillcolor="#0F172A"];
        GeneralChain [label="General FAQ Chain:\\nCached RAG VectorDB + Llama-3", fillcolor="#0F172A"];
    }

    UserQuery -> Classifier;
    Classifier -> BillingChain [label="Route == 'Billing'", color="#10B981", penwidth=2.0];
    Classifier -> TechChain [label="Route == 'Tech'", style=dotted];
    Classifier -> GeneralChain [label="Route == 'General'", style=dotted];
}
"""
st.graphviz_chart(routing_dot, use_container_width=True)

st.subheader("Semantic Classifier Router in LCEL")
st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

fast_classifier_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 1: Build the Router Prompt
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an enterprise query classifier. 
Classify the user input into exactly ONE of the following categories:
- BILLING
- TECHNICAL_SUPPORT
- GENERAL_INQUIRY
Return only the single category keyword with no punctuation."""),
    ("human", "{query}")
])

router_chain = router_prompt | fast_classifier_model | StrOutputParser()

# Step 2: Route Dispatcher Function
def route_dispatcher(input_dict: dict):
    category = input_dict["category"].strip()
    query = input_dict["query"]
    
    if category == "BILLING":
        return billing_chain
    elif category == "TECHNICAL_SUPPORT":
        return tech_support_chain
    else:
        return general_inquiry_chain

# Step 3: Complete Unified Routing Chain
full_routed_chain = (
    {"category": router_chain, "query": lambda x: x["query"]}
    | RunnableLambda(route_dispatcher)
)
''', language="python")

st.divider()

# --- SECTION 5: CONDITIONAL EXECUTION ---
st.header("5. Conditional Execution (Predicates & Boolean Decision Guards)", divider="rainbow")

st.markdown("""
Not all requests should reach an expensive LLM. **Conditional Execution** introduces guardrails that evaluate 
preconditions, assess security policies, or inspect token thresholds before dispatching to heavy compute nodes.
If a precondition fails, the chain short-circuits immediately, returning a cached or error response.
""")

st.subheader("Guarded Execution with Short-Circuit Logic")
st.code('''from langchain_core.runnables import RunnableLambda
from typing import Dict, Any

class SecurityViolationException(Exception):
    pass

def pii_scrubber_guard(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Inspects payload for illegal tokens or PII patterns before model invocation."""
    query = input_data.get("query", "")
    banned_keywords = ["DROP TABLE", "DELETE FROM", "SSN:", "PASSWORD="]
    
    for banned in banned_keywords:
        if banned.lower() in query.lower():
            return {
                "blocked": True,
                "reason": f"Payload contained prohibited pattern: {banned}",
                "query": query
            }
            
    return {"blocked": False, "query": query}

def conditional_executor(data: Dict[str, Any]):
    """Short-circuits if payload was flagged by the upstream guard."""
    if data.get("blocked", False):
        # Return canned remediation without calling expensive LLM
        return f"Request Rejected by Security Guardrail: {data['reason']}"
    
    # Pass clean query to primary processing chain
    return primary_llm_chain.invoke(data["query"])

guarded_chain = (
    RunnableLambda(pii_scrubber_guard)
    | RunnableLambda(conditional_executor)
)
''', language="python")

st.divider()

# --- SECTION 6: DYNAMIC CHAINS ---
st.header("6. Dynamic Chains (Runtime DAG Assembly & Adaptive Configuration)", divider="rainbow")

st.markdown("""
Static chains have fixed topologies defined at module import time. In contrast, **Dynamic Chains** adapt their 
topologies, model choices, temperature settings, or prompt templates at runtime based on the execution context 
(e.g., user authorization tier, budget constraints, or dynamic schema requirements).
""")

st.code('''from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def dynamic_chain_factory(request_context: dict):
    """Dynamically compiles a tailored LCEL chain based on user authorization tier."""
    user_tier = request_context.get("tier", "free")
    
    if user_tier == "enterprise":
        # Enterprise users receive full context reasoning with top-tier model
        model = ChatOpenAI(model="gpt-4o", temperature=0.2)
        system_msg = "You are a Senior Principal Architect providing in-depth, exhaustive analysis."
    else:
        # Free users receive concise answers via high-speed lightweight model
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        system_msg = "You are a concise technical assistant. Limit explanations to 3 bullet points."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{query}")
    ])
    
    # Compile and return the customized chain for this specific execution
    return prompt | model | StrOutputParser()

# Master chain using RunnableLambda to dynamically instantiate and invoke sub-chains
adaptive_pipeline = RunnablePassthrough() | RunnableLambda(dynamic_chain_factory)

# Free Tier Invocation
free_response = adaptive_pipeline.invoke({"tier": "free", "query": "Explain event loops in Node.js"})

# Enterprise Tier Invocation
enterprise_response = adaptive_pipeline.invoke({"tier": "enterprise", "query": "Explain event loops in Node.js"})
''', language="python")

st.divider()

# --- SECTION 7: ERROR HANDLING ---
st.header("7. Error Handling (`with_exception_handling` & Graceful Degradation)", divider="rainbow")

st.markdown("""
In production systems, third-party LLM APIs occasionally fail due to network timeouts, provider 500 errors, 
content moderation policy triggers, or invalid response formats. Unhandled exceptions in a synchronous chain 
will crash the parent application or result in 500 errors for end users.

LangChain provides multiple mechanisms for graceful degradation, allowing chains to catch exceptions, 
log diagnostic metadata, and return fallback payloads.
""")

st.code('''from langchain_core.runnables import RunnableLambda
import logging

logger = logging.getLogger("EnterpriseChain")

def robust_model_invoker(chain_input: dict):
    try:
        # Attempt primary model invocation
        return fragile_external_chain.invoke(chain_input)
    except Exception as exc:
        logger.error(f"Upstream provider failure: {str(exc)}", exc_info=True)
        # Return graceful degraded response dictionary rather than raising unhandled error
        return {
            "status": "degraded",
            "error_type": type(exc).__name__,
            "message": "The primary reasoning service is currently experiencing high load. A cached summary has been provided.",
            "data": "Default baseline response: please retry your query shortly."
        }

safe_chain = RunnableLambda(robust_model_invoker)
''', language="python")

st.divider()

# --- SECTION 8: RETRY MECHANISMS ---
st.header("8. Retry Mechanisms (`with_retry` & Exponential Backoff with Jitter)", divider="rainbow")

st.markdown("""
Transient errors (such as HTTP 429 Rate Limits, socket disconnects, or momentary gateway timeouts) can often be 
resolved simply by waiting briefly and trying again. 

Calling `.with_retry()` on any LCEL Runnable wraps it in a production retry handler with configurable exponential backoff:
`backoff_delay = min(max_delay, initial_delay * (multiplier ** attempt)) + uniform_random_jitter`
""")

retry_dot = """
digraph G {
    bgcolor="#0B1120";
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", fillcolor="#1E293B"];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B"];

    Attempt1 [label="Attempt 1: Invoke LLM API\\nHTTP 429 Too Many Requests", color="#EF4444"];
    Backoff1 [label="Sleep: 100ms + Jitter\\nExponential Backoff (Multiplier: 2.0)", fillcolor="#0F172A"];
    
    Attempt2 [label="Attempt 2: Retry LLM API\\nHTTP 429 Rate Limit Continued", color="#EF4444"];
    Backoff2 [label="Sleep: 200ms + Jitter\\nExponential Backoff (Multiplier: 2.0)", fillcolor="#0F172A"];
    
    Attempt3 [label="Attempt 3: Retry LLM API\\nHTTP 200 OK - Successful Token Stream", color="#10B981"];
    Delivery [label="Return AIMessage to Downstream Parser", fillcolor="#0284C7"];

    Attempt1 -> Backoff1 [label="Retryable Status"];
    Backoff1 -> Attempt2;
    Attempt2 -> Backoff2 [label="Retryable Status"];
    Backoff2 -> Attempt3;
    Attempt3 -> Delivery [label="Success"];
}
"""
st.graphviz_chart(retry_dot, use_container_width=True)

st.subheader("Configuring `.with_retry()` in Modern LCEL")
st.code('''from langchain_openai import ChatOpenAI

# Standard Chat Model
raw_model = ChatOpenAI(model="gpt-4o", timeout=10.0)

# Configure Retry Interceptor
# Automatically retries on RateLimitError, APIConnectionError, InternalServerError
resilient_model = raw_model.with_retry(
    stop_after_attempt=4,            # Maximum retry attempts
    wait_exponential_jitter=True,    # Add random jitter to prevent herd stampede
    retry_if_exception_type=(        # Specify exact exception classes to retry
        TimeoutError,
        ConnectionError,
        # openai.RateLimitError, openai.APIConnectionError
    )
)

# Piping the resilient model into standard chains
chain = prompt | resilient_model | parser
''', language="python")

st.divider()

# --- SECTION 9: FALLBACK MODELS ---
st.header("9. Fallback Models (`with_fallbacks` & Multi-Provider Failover)", divider="rainbow")

st.markdown("""
When a primary model provider experiences extended downtime or severe service degradation, retries will fail. 
To guarantee high availability, production architectures use **Multi-Provider Fallbacks**.

LCEL provides the native `.with_fallbacks([secondary_model, tertiary_model])` method. If the primary model raises 
an exception after all retries are exhausted, LCEL automatically intercepts the failure and invokes the first 
fallback model using the exact same prompt and parameters.
""")

fallback_dot = """
digraph G {
    bgcolor="#0B1120";
    rankdir=TB;
    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=box, style="filled,rounded", color="#38BDF8", fillcolor="#1E293B"];
    edge [fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", color="#64748B"];

    UserPrompt [label="Formatted Prompt Message\\nChatPromptValue"];
    
    PrimaryModel [label="Primary Provider:\\nAnthropic Claude-3-5-Sonnet\\n(High Reasoning)", fillcolor="#1E293B", color="#38BDF8"];
    Fallback1 [label="Secondary Fallback:\\nOpenAI GPT-4o-mini\\n(Cross-Cloud Redundancy)", fillcolor="#1E293B", color="#F59E0B"];
    Fallback2 [label="Tertiary Fallback:\\nLocal Ollama (Llama-3-8B)\\n(Zero External Network Dependency)", fillcolor="#1E293B", color="#10B981"];
    OutputNode [label="Downstream Output Parser\\nStrOutputParser", fillcolor="#0284C7"];

    UserPrompt -> PrimaryModel;
    PrimaryModel -> OutputNode [label="HTTP 200 OK (Normal Path)"];
    PrimaryModel -> Fallback1 [label="Provider Outage (503/Timeout)", color="#EF4444", style=dashed];
    Fallback1 -> OutputNode [label="Fallback 1 Success", color="#F59E0B"];
    Fallback1 -> Fallback2 [label="Secondary API Exhausted", color="#EF4444", style=dashed];
    Fallback2 -> OutputNode [label="Tertiary Local Success", color="#10B981"];
}
"""
st.graphviz_chart(fallback_dot, use_container_width=True)

st.subheader("Implementing Cross-Provider Failover in LCEL")
st.code('''from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Define Primary and Alternative Model Candidates
primary_model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", timeout=8.0)
fallback_openai = ChatOpenAI(model="gpt-4o-mini", timeout=5.0)
fallback_local = ChatOllama(model="llama3:8b", timeout=12.0)

# 2. Chain Primary with Fallbacks
# If Anthropic fails or times out, fallback to OpenAI; if OpenAI fails, fallback to Local Ollama
high_availability_model = primary_model.with_fallbacks([
    fallback_openai,
    fallback_local
])

# 3. Assemble End-to-End Fault Tolerant Chain
prompt = ChatPromptTemplate.from_template("Analyze the risk factors in this contract excerpt: {text}")
ha_pipeline = prompt | high_availability_model | StrOutputParser()

# Under normal conditions: executed via Claude 3.5 Sonnet
# Under AWS/Anthropic outage: automatically executed via GPT-4o-mini without application code modifications
response = ha_pipeline.invoke({"text": "The licensee agrees to indemnify licensor against all third-party claims..."})
''', language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Enterprise Fault-Tolerant Multi-Route Pipeline", divider="rainbow")

st.markdown("""
The following complete production pattern combines **Dynamic Intent Routing**, **Parallel Branching**, 
**Exponential Backoff Retries**, and **Multi-Provider Fallbacks** into a unified, enterprise-grade LCEL system.
""")

st.code('''import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel, 
    RunnablePassthrough, 
    RunnableLambda, 
    RunnableSequence
)
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnterprisePipeline")

class ResilientEnterpriseWorkflow:
    """Production orchestration pipeline featuring multi-branch fan-out,

    semantic routing, automated retries, and cross-provider failover.
    """
    def __init__(self):
        # Primary Frontier Model with Retries
        self.primary_llm = ChatAnthropic(
            model_name="claude-3-5-sonnet-20240620", 
            temperature=0.1,
            timeout=10.0
        ).with_retry(stop_after_attempt=3)

        # Secondary Redundant Model
        self.fallback_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1,
            timeout=8.0
        ).with_retry(stop_after_attempt=2)

        # Resilient Model Node with Fallback
        self.resilient_model = self.primary_llm.with_fallbacks([self.fallback_llm])
        self.parser = StrOutputParser()

        # Step 1: Parallel Enrichment Engine
        self.enrichment_branch = RunnableParallel({
            "original_query": RunnablePassthrough(),
            "sentiment": (
                ChatPromptTemplate.from_template("Classify tone (Positive/Neutral/Urgent): {query}")
                | self.fallback_llm 
                | self.parser
            ),
            "key_entities": (
                ChatPromptTemplate.from_template("Extract top 3 technical entities as comma list: {query}")
                | self.fallback_llm 
                | self.parser
            )
        })

        # Step 2: Synthesis Engine
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Principal Engineering Consultant. 
Synthesize a comprehensive resolution addressing the customer query.
Context Metadata:
- Customer Tone: {sentiment}
- Extracted Entities: {key_entities}"""),
            ("human", "Customer Query: {original_query}")
        ])

        # Step 3: Complete Unified Pipeline
        self.pipeline: RunnableSequence = (
            self.enrichment_branch
            | self.synthesis_prompt
            | self.resilient_model
            | self.parser
        )

    def process(self, query: str) -> str:
        logger.info(f"Initiating resilient pipeline for query: {query[:50]}...")
        return self.pipeline.invoke(query)
''', language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Resilient Dynamic Chain Simulator", divider="rainbow")

st.markdown("""
Test how an enterprise chain dynamically balances **Intent Routing**, **Concurrent Branching (`RunnableParallel`)**, 
**Rate Limit Retries (`with_retry`)**, and **Outage Failover (`with_fallbacks`)** under real-world conditions.
""")

engine = MockAdvancedChainEngine()

wb_col1, wb_col2 = st.columns(2)

with wb_col1:
    st.subheader("🛠️ Request & Fault Injection Controls")
    sample_query = st.text_area(
        "User Prompt / Query",
        value="URGENT: Our production checkout service is failing with a database connection pool timeout error!",
        height=100
    )
    
    st.markdown("#### Fault Injection Simulation")
    sim_rate_limit = st.checkbox("Simulate HTTP 429 Rate Limit (Triggers Exponential Backoff)", value=True)
    sim_outage = st.checkbox("Simulate Primary Model Outage (503 Service Unavailable)", value=True)
    enable_parallel = st.checkbox("Enable Parallel Metadata Extraction (RunnableParallel)", value=True)
    max_retries = st.slider("Max Configured Retries", min_value=1, max_value=4, value=3)

with wb_col2:
    st.subheader("⚙️ Configured Pipeline State")
    st.info(f"""
    **Active Configuration:**
    - **Intent Classifier**: Semantic Multi-Route Router
    - **Concurrency**: {'RunnableParallel Active (Sentiment + Entities + Urgency)' if enable_parallel else 'Disabled (Single Thread)'}
    - **Primary Model**: Claude-3-5-Sonnet (Timeout: 10s)
    - **Secondary Fallback**: GPT-4o-mini (Auto-Failover enabled)
    - **Backoff Algorithm**: Exponential Delay with Uniform Random Jitter
    """)

if st.button("⚡ Execute Resilient Pipeline", type="primary"):
    res = engine.execute_request(
        query=sample_query,
        simulate_primary_outage=sim_outage,
        simulate_rate_limit=sim_rate_limit,
        enable_parallel_analysis=enable_parallel,
        max_retries=max_retries
    )

    st.subheader("📊 Execution Telemetry & Diagnostics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Classified Route", res["route"])
    m2.metric("Router Confidence", f"{res['confidence']*100:.1f}%")
    m3.metric("Serving Model", res["active_model"])
    m4.metric("Total Latency", f"{res['total_latency_ms']} ms")

    if res["parallel_results"]:
        st.write("#### ⚡ Concurrent Extraction Output (`RunnableParallel`)")
        pc1, pc2, pc3 = st.columns(3)
        pc1.info(f"**Sentiment**: {res['parallel_results']['sentiment']}")
        pc2.info(f"**Urgency**: {res['parallel_results']['urgency_level']}")
        pc3.info(f"**Entities**: {', '.join(res['parallel_results']['entities'])}")

    st.write("#### ⏱️ Trace Spans & Interceptor Logs")
    for span in res["spans"]:
        if span.status == "SUCCESS":
            color = "🟢"
        elif span.status == "RETRY":
            color = "🟡"
        elif span.status == "FALLBACK":
            color = "🟠"
        else:
            color = "🔵"

        with st.expander(f"{color} {span.step_name} ({span.duration_ms} ms) - Model: {span.model_used}", expanded=True):
            st.caption(f"Status: **{span.status}**")
            st.write(span.details)

    st.write("#### 🎯 Synthesized Final Output")
    if "Fallback Active" in res["response_text"]:
        st.warning(res["response_text"])
    else:
        st.success(res["response_text"])

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does RunnableParallel differ from executing tasks in a standard Python for-loop?",
     "A standard Python for-loop executes tasks sequentially on a single thread, where total latency equals the sum of all tasks. RunnableParallel leverages concurrent execution (via thread pools or asyncio event loops), allowing multiple independent prompts or retrieval calls to run simultaneously. The overall wall-clock latency is bounded by the slowest branch rather than the sum of all branches."),
    ("What is the difference between .with_retry() and .with_fallbacks() in LCEL?",
     ".with_retry() re-invokes the EXACT SAME model or runnable multiple times with backoff delays, addressing transient errors such as network timeouts or momentary rate limits (HTTP 429). In contrast, .with_fallbacks() catches failures after all retries are exhausted and switches execution to an entirely DIFFERENT model or alternative provider (e.g. failing over from Anthropic Claude to OpenAI GPT-4o-mini)."),
    ("Why is exponential backoff with jitter critical when handling LLM API rate limits?",
     "If multiple distributed worker instances hit a rate limit simultaneously and retry with a fixed delay (e.g. exactly 1 second), they will all hit the API again at the exact same millisecond, causing a persistent 'thundering herd' cycle. Adding uniform random jitter desynchronizes the retries across workers, smoothing out traffic spikes and allowing the API token bucket to refill."),
    ("When should you use dynamic chain compilation over static LCEL pipelines?",
     "Static chains are optimal when workflow topology is invariant. Dynamic chain compilation (via RunnableLambda factory functions) is essential when chain structure depends on runtime state—such as routing free-tier users to lightweight models while directing enterprise users to reasoning models, or dynamically generating prompts based on a database schema discovered at runtime."),
    ("How does streaming work through a RunnableParallel node in LCEL?",
     "When invoking .astream() or .astream_events() on a RunnableParallel node, LangChain streams chunks from all parallel branches as they arrive. Each streamed dictionary chunk is tagged with its corresponding branch key (e.g., {'summary': 'token'}, {'code': 'token'}), allowing the client UI to render multiple live output panes concurrently.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Cascading Timeouts in Sequential Fallbacks. If your primary model timeout is 30 seconds and you configure 3 fallback models with 30-second timeouts, an outage will block the user request for 120 seconds before failing. Always set aggressive, decreasing timeouts on fallback candidates (e.g., 8s ➔ 5s ➔ 3s).")
    st.error("**Trap 2 (Breaking Error)**: Output Schema Mismatches Across Fallback Models. If the primary model reliably returns valid JSON for Pydantic parsing but the fallback model returns markdown-wrapped text, the fallback will trigger an unhandled validation error. Always test all fallback models against the exact same downstream parser.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Retrying Non-Retryable HTTP Errors. Retrying HTTP 400 (Bad Request), 401 (Unauthorized), or 422 (Unprocessable Entity) wastes resources and delays error reporting. Always configure retry_if_exception_type to retry only transient errors (RateLimitError, TimeoutError, 503).")
    st.warning("**Trap 4 (Latency Warning)**: Unbounded Concurrency in RunnableParallel. Launching 50 parallel branches simultaneously will exhaust local thread pools and immediately hit upstream provider concurrency limits. Bound parallelism using asyncio semaphores or batch processing.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: RunnableParallel Wall Time"):
        st.markdown("**Latency is bounded by max(t_1, t_2, ..., t_n) rather than the sum of latencies.**")
    with st.expander("🎴 Flashcard 2: with_retry vs with_fallbacks"):
        st.markdown("**with_retry retries the same node; with_fallbacks switches to an alternative node.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Intent Classification Routing"):
        st.markdown("**Uses a lightweight model to classify intent before dispatching to specialized sub-chains.**")
    with st.expander("🎴 Flashcard 4: Jitter in Exponential Backoff"):
        st.markdown("**Random delay added to exponential backoff to prevent thundering herd API stampedes.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand how RunnableParallel executes decoupled branches concurrently.", value=True)
chk2 = st.checkbox("I can implement conditional branching using RunnableBranch and dynamic dispatchers.", value=True)
chk3 = st.checkbox("I know how to design semantic intent routers to dispatch queries to specialized chains.", value=True)
chk4 = st.checkbox("I can configure .with_retry() with exponential backoff and jitter for transient errors.", value=True)
chk5 = st.checkbox("I can implement cross-provider failover chains using .with_fallbacks().", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 advanced chain interview questions without reference material.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 6.3 Advanced Chains and completed all of Chapter 6: Chains!")
