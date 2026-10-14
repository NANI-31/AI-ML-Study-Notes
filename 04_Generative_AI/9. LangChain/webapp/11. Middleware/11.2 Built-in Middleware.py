import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class PolicyEvaluationResult:
    """Holds the granular audit trail of built-in middleware evaluations."""
    def __init__(self):
        self.stage_logs: List[Dict[str, Any]] = []
        self.redacted_prompt: str = ""
        self.selected_model: str = ""
        self.hitl_paused: bool = False
        self.retries_attempted: int = 0
        self.tool_error_recovered: bool = False
        self.final_output: str = ""
        self.total_cost_usd: float = 0.0

class MockBuiltInMiddlewareStudio:
    """Simulates an enterprise agent runtime configured with all 8 built-in middleware policies."""
    def __init__(self,
                 enable_summarization: bool = True,
                 enable_hitl: bool = True,
                 enable_retries: bool = True,
                 enable_tool_error_recovery: bool = True,
                 enable_model_router: bool = True,
                 enable_guardrails: bool = True,
                 enable_pii_redaction: bool = True,
                 enable_rate_limiter: bool = True):
        self.enable_summarization = enable_summarization
        self.enable_hitl = enable_hitl
        self.enable_retries = enable_retries
        self.enable_tool_error_recovery = enable_tool_error_recovery
        self.enable_model_router = enable_model_router
        self.enable_guardrails = enable_guardrails
        self.enable_pii_redaction = enable_pii_redaction
        self.enable_rate_limiter = enable_rate_limiter

    def process(self, prompt: str, target_tool: str, hitl_approved: bool, simulated_failure: bool) -> PolicyEvaluationResult:
        res = PolicyEvaluationResult()
        current_prompt = prompt

        # 1. Rate Limiting Check
        if self.enable_rate_limiter:
            res.stage_logs.append({
                "stage": "8. Rate Limiting",
                "status": "PASS",
                "detail": "Client quota verified: 42 requests remaining in current minute window."
            })

        # 2. Guardrails (Input Safety)
        if self.enable_guardrails:
            if "jailbreak" in current_prompt.lower() or "override system" in current_prompt.lower():
                res.stage_logs.append({
                    "stage": "6. Guardrails",
                    "status": "BLOCKED",
                    "detail": "Input violation: Prompt injection or jailbreak attempt flagged."
                })
                res.final_output = "[EXECUTION BLOCKED]: Query violates Enterprise Safety & Alignment Guidelines."
                return res
            res.stage_logs.append({
                "stage": "6. Guardrails",
                "status": "PASS",
                "detail": "Prompt safety scored 0.02 (Safety Threshold: < 0.20)."
            })

        # 3. PII Detection & Redaction
        if self.enable_pii_redaction:
            scrubbed = current_prompt
            if "987-65-4321" in scrubbed:
                scrubbed = scrubbed.replace("987-65-4321", "[REDACTED-SSN-TOKEN-1]")
            if "admin@megacorp.internal" in scrubbed:
                scrubbed = scrubbed.replace("admin@megacorp.internal", "[REDACTED-EMAIL-TOKEN-2]")
            res.redacted_prompt = scrubbed
            current_prompt = scrubbed
            res.stage_logs.append({
                "stage": "7. PII Redaction",
                "status": "MUTATED",
                "detail": f"Sanitized payload: Real entity markers mapped to encrypted vault tokens."
            })
        else:
            res.redacted_prompt = current_prompt

        # 4. Summarization Middleware
        if self.enable_summarization:
            word_count = len(current_prompt.split())
            if word_count > 50:
                res.stage_logs.append({
                    "stage": "1. Summarization",
                    "status": "COMPACTED",
                    "detail": f"Context exceeded 50 words ({word_count} words). Compacted past dialogue into 20-word rolling memo."
                })
            else:
                res.stage_logs.append({
                    "stage": "1. Summarization",
                    "status": "BYPASS",
                    "detail": f"Prompt length within window ({word_count} words <= 50 words ceiling)."
                })

        # 5. Dynamic Model Selection
        if self.enable_model_router:
            if "complex" in prompt.lower() or "audit" in prompt.lower() or len(prompt.split()) > 35:
                res.selected_model = "claude-3-5-sonnet (High Reasoning Tier)"
                res.total_cost_usd += 0.015
            else:
                res.selected_model = "gpt-4o-mini (High Speed / Cost-Efficient Tier)"
                res.total_cost_usd += 0.0008
            res.stage_logs.append({
                "stage": "5. Model Selection",
                "status": "ROUTED",
                "detail": f"Routed query to {res.selected_model} based on prompt complexity heuristics."
            })
        else:
            res.selected_model = "default-gpt-4o"
            res.total_cost_usd += 0.005

        # 6. Human-In-The-Loop (Tool Dispatch Guard)
        if self.enable_hitl and target_tool in ["transfer_funds", "drop_table", "execute_terminal"]:
            if not hitl_approved:
                res.hitl_paused = True
                res.stage_logs.append({
                    "stage": "2. Human-In-The-Loop",
                    "status": "INTERRUPT",
                    "detail": f"Action '{target_tool}' requires explicit human verification. Execution paused on checkpoint token."
                })
                res.final_output = f"[PAUSED AT CHECKPOINT]: Action '{target_tool}' awaits human supervisor approval."
                return res
            res.stage_logs.append({
                "stage": "2. Human-In-The-Loop",
                "status": "APPROVED",
                "detail": f"Human supervisor signature verified for '{target_tool}'."
            })

        # 7. Retry Handling & Tool Execution
        if simulated_failure:
            if self.enable_retries:
                res.retries_attempted = 2
                res.stage_logs.append({
                    "stage": "3. Retry Handling",
                    "status": "RETRIED",
                    "detail": "Tool API returned 503 Service Unavailable. Exponential backoff triggered (attempt 1: 100ms, attempt 2: 200ms)."
                })
            else:
                res.stage_logs.append({
                    "stage": "3. Retry Handling",
                    "status": "FAILED",
                    "detail": "Tool API failed without retries configured."
                })

            if self.enable_tool_error_recovery:
                res.tool_error_recovered = True
                res.stage_logs.append({
                    "stage": "4. Tool Error Recovery",
                    "status": "RECOVERED",
                    "detail": "Caught tool exception and fed synthesized observation back to agent: 'External API degraded, switched to local cache'."
                })
                res.final_output = f"Completed request via fallback cache after intercepting tool error in '{target_tool}'."
                return res
            else:
                res.final_output = f"[TOOL EXECUTION ERROR]: Fatal unhandled exception in '{target_tool}'."
                return res

        res.final_output = f"Successfully processed request using {res.selected_model}. Tool '{target_tool}' executed with sanitized payload."
        return res


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="11.2 Built-in Middleware",
    page_icon="🧩",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">11.2 Built-in Middleware</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pre-packaged enterprise interceptors: summarization, human-in-the-loop, retries, error recovery, model routing, guardrails, PII redaction, and rate limiting.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise Built-in Middleware Stack", divider="rainbow")

st.markdown("""
Rather than requiring engineers to author custom interceptor logic from scratch for common production needs, 
modern agent runtimes provide a suite of **built-in middleware components**. These standardized modules form a robust 
protective pipeline covering safety, cost control, fault tolerance, and regulatory compliance.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph BuiltinMiddlewareArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_gateway {
            label="1. Inbound Enterprise Perimeter";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            ClientReq [label="Client Inbound Message\\n(Raw User Prompt + Session Token)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            RateLimitMW [label="Rate Limiting Middleware\\n(Token Bucket Quota, 429 Throttling)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            GuardrailsMW [label="Content Guardrails Middleware\\n(Jailbreak Detection, Llama Guard)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            PIIMW [label="PII Redaction Middleware\\n(NER Masking, Cryptographic Token Vault)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
        }

        subgraph cluster_optimization {
            label="2. Context & Model Orchestration";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            fontsize=12;

            SummarizerMW [label="Summarization Middleware\\n(Rolling Dialogue Compaction)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            ModelRouterMW [label="Model Selection Middleware\\n(Complexity Router: Fast vs Frontier)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        }

        subgraph cluster_execution {
            label="3. Execution & Fault Resilience";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#A855F7";
            fontsize=12;

            HITLMW [label="Human-In-The-Loop Middleware\\n(Privilege Escalation Gate, Interrupt)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            CoreModel [label="Model / Agent Execution Loop\\n(LLM Planning & Tool Selection)", fillcolor="#3B0764", fontcolor="#FAF5FF", color="#D8B4FE", penwidth=2];
            RetryMW [label="Retry Handling Middleware\\n(Exponential Backoff + Jitter)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            ToolRecoveryMW [label="Tool Error Recovery Middleware\\n(Exception Wrapping, Fallback Prompting)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
        }

        ClientReq -> RateLimitMW [label="1. Ingress Request"];
        RateLimitMW -> GuardrailsMW [label="2. Quota OK"];
        GuardrailsMW -> PIIMW [label="3. Content Safe"];
        PIIMW -> SummarizerMW [label="4. Clean State"];
        SummarizerMW -> ModelRouterMW [label="5. Context Bounded"];
        ModelRouterMW -> HITLMW [label="6. Model Picked"];
        HITLMW -> CoreModel [label="7. Action Approved"];
        CoreModel -> RetryMW [label="8. API Invocation"];
        RetryMW -> ToolRecoveryMW [label="9. Tool Dispatched"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: SUMMARIZATION MIDDLEWARE ---
st.header("2. Summarization Middleware", divider="rainbow")

st.markdown("""
Conversational agents accumulate dialogue history across turns. Left unchecked, message histories quickly exceed model 
context windows, resulting in exorbitant token costs and degraded cognitive precision (the "lost-in-the-middle" effect).

**Summarization Middleware** monitors the accumulated token count or message list length. When a predefined threshold is crossed, 
it triggers an asynchronous summarization run, condensing older turns into a concise executive summary while preserving the most recent 
interactive turns intact.
""")

st.code("""# Summarization Middleware Configuration
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class SummarizationMiddleware:
    def __init__(self, max_tokens: int = 4000, keep_recent_turns: int = 4, summarizer_model = None):
        self.max_tokens = max_tokens
        self.keep_recent_turns = keep_recent_turns
        self.summarizer = summarizer_model

    def on_ingress(self, state: dict) -> dict:
        messages = state.get("messages", [])
        total_tokens = sum(len(m.content.split()) * 4 for m in messages)
        
        if total_tokens > self.max_tokens and len(messages) > self.keep_recent_turns:
            to_summarize = messages[:-self.keep_recent_turns]
            preserved = messages[-self.keep_recent_turns:]
            
            # Condense older messages
            summary_text = self.summarizer.invoke(
                f"Condense the following conversation history into 3 bullet points:\\n{to_summarize}"
            ).content
            
            # Inject consolidated memory header
            state["messages"] = [
                SystemMessage(content=f"Prior Conversation Summary: {summary_text}")
            ] + preserved
        return state""", language="python")

st.divider()

# --- SECTION 3: HUMAN-IN-THE-LOOP MIDDLEWARE ---
st.header("3. Human-in-the-Loop Middleware", divider="rainbow")

st.markdown("""
Autonomous agents equipped with database execution, payment gateways, or system shell tools represent severe financial 
and operational risks if allowed to execute unrestrained.

**Human-in-the-Loop (HITL) Middleware** inspects the tool call intentions generated by the model. 
If the targeted tool belongs to a designated high-risk classification, the middleware intercepts the call, generates a 
checkpoint token, suspends execution, and alerts a human operator for approval, argument amendment, or rejection.
""")

hitl_df = pd.DataFrame([
    {"Tool Category": "Read-Only Queries (e.g. search_faq, get_weather)", "Risk Profile": "Low", "HITL Policy": "Autonomous Execution", "Latency Impact": "None (0s)"},
    {"Tool Category": "Transactional Queries (e.g. book_seat, send_draft)", "Risk Profile": "Medium", "HITL Policy": "Async User Notification", "Latency Impact": "Non-blocking"},
    {"Tool Category": "Destructive / Financial (e.g. wire_money, delete_account)", "Risk Profile": "Critical", "HITL Policy": "Mandatory Synchronous Approval Gate", "Latency Impact": "Suspended until Human Review"}
])
st.dataframe(hitl_df, use_container_width=True)

st.divider()

# --- SECTION 4: RETRY HANDLING MIDDLEWARE ---
st.header("4. Retry Handling Middleware", divider="rainbow")

st.markdown("""
Cloud LLM APIs and third-party SaaS tools routinely encounter transient disruptions: HTTP 429 rate limit throttles, 
HTTP 502/503 service outages, and network socket timeouts.

**Retry Handling Middleware** wraps invocations in exponential backoff algorithms with randomized jitter. 
This prevents the "thundering herd" problem while shielding agent logic from temporary infrastructure hiccups.
""")

st.code("""# Retry Handling Middleware with Exponential Backoff + Jitter
import time, random

class RetryMiddleware:
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0, backoff_multiplier: float = 2.0):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.multiplier = backoff_multiplier

    def execute_with_retry(self, fn, *args, **kwargs):
        delay = self.initial_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (TimeoutError, ConnectionError) as exc:
                if attempt == self.max_retries:
                    raise exc
                # Exponential Backoff with Full Jitter
                sleep_duration = random.uniform(0, delay)
                time.sleep(sleep_duration)
                delay *= self.multiplier""", language="python")

st.divider()

# --- SECTION 5: TOOL ERROR HANDLING MIDDLEWARE ---
st.header("5. Tool Error Handling Middleware", divider="rainbow")

st.markdown("""
When a tool throws an unhandled exception (e.g. `psycopg2.OperationalError`, `json.JSONDecodeError`, or a 404 REST response), 
raw agent implementations crash immediately.

**Tool Error Handling Middleware** intercepts tool exceptions at the execution boundary. 
Instead of terminating the agent process, it transforms the Python stack trace into an informative, natural-language 
`ToolMessage` observation. This gives the agent cognitive awareness of the error, allowing it to reformulate its plan, 
fix syntax mistakes, or choose an alternative tool autonomously.
""")

st.divider()

# --- SECTION 6: DYNAMIC MODEL SELECTION MIDDLEWARE ---
st.header("6. Dynamic Model Selection Middleware", divider="rainbow")

st.markdown("""
Sending simple conversational greetings to 70B+ or frontier models costs 20x to 50x more than necessary. Conversely, 
delegating multi-step algorithmic planning or complex code generation to lightweight 8B models leads to hallucinations.

**Dynamic Model Selection Middleware** acts as an intelligent router. It analyzes query complexity, token volume, 
and reasoning requirements to dynamically bind the optimal LLM backend.
""")

col_mod1, col_mod2 = st.columns(2)
with col_mod1:
    st.info("""
    **🚀 Fast Tier (e.g. gpt-4o-mini / gemini-1.5-flash):**
    - Single-turn fact retrieval
    - Basic categorization and formatting
    - Latency: < 400ms | Cost: ~0.00015 USD / 1K tokens
    """)
with col_mod2:
    st.success("""
    **🧠 Frontier Reasoning Tier (e.g. o1 / claude-3-5-sonnet):**
    - Multi-table SQL synthesis and database schema joins
    - Multi-file code refactoring and security auditing
    - Latency: 2s - 8s | Cost: ~0.015 USD / 1K tokens
    """)

st.divider()

# --- SECTION 7: GUARDRAILS & CONTENT SAFETY MIDDLEWARE ---
st.header("7. Guardrails & Content Safety Middleware", divider="rainbow")

st.markdown("""
Guardrails enforce corporate compliance, brand safety, and cybersecurity policies across both ingress and egress channels:
- **Input Guardrails**: Screen prompts for adversarial prompt injection, system prompt extraction, and policy violations before tokenization.
- **Output Guardrails**: Screen synthesized answers for toxic language, hallucinated financial guarantees, or unauthorized advice.
""")

st.divider()

# --- SECTION 8: PII DETECTION & REDACTION MIDDLEWARE ---
st.header("8. PII Detection & Redaction Middleware", divider="rainbow")

st.markdown("""
Under GDPR, HIPAA, and PCI-DSS, sending plain-text Personally Identifiable Information (PII) to external cloud AI providers 
constitutes a severe compliance breach.

**PII Detection & Redaction Middleware** performs **bidirectional token vaulting**:
1. **Ingress**: Recognizes credit cards, social security numbers, and emails using NER and regex, replacing them with synthetic tokens (e.g. `[ENTITY_TOKEN_42]`).
2. **Egress**: Re-maps the synthetic tokens back to the original entities when delivering the final response to the authorized end user, ensuring third-party LLMs never observe raw customer identifiers.
""")

st.divider()

# --- SECTION 9: RATE LIMITING & QUOTA MANAGEMENT MIDDLEWARE ---
st.header("9. Rate Limiting & Quota Management Middleware", divider="rainbow")

st.markdown("""
Without active rate limiting, buggy client scripts or malicious actors can saturate backend worker pools, 
causing Denial-of-Service (DoS) and catastrophic cloud billing spikes.

**Rate Limiting Middleware** enforces **Token Bucket** and **Sliding Window** algorithms, throttling accounts based on:
- **Requests Per Minute (RPM)**
- **Tokens Per Minute (TPM)**
- **Concurrent Execution Slots**
""")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: End-to-End Enterprise Built-in Middleware Stack", divider="rainbow")

st.markdown("""
The following script demonstrates an enterprise middleware coordinator orchestrating all 8 built-in middleware capabilities 
into a unified execution pipeline.
""")

st.code("""from typing import Dict, List, Any, Optional

class EnterpriseBuiltinStack:
    \"\"\"Production orchestration harness for all 8 built-in middleware modules.\"\"\"
    def __init__(self,
                 rate_limiter,
                 guardrail_checker,
                 pii_redactor,
                 summarizer,
                 model_router,
                 hitl_manager,
                 retry_handler,
                 tool_error_handler):
        self.rate_limiter = rate_limiter
        self.guardrails = guardrail_checker
        self.pii = pii_redactor
        self.summarizer = summarizer
        self.router = model_router
        self.hitl = hitl_manager
        self.retry = retry_handler
        self.tool_recovery = tool_error_handler

    def execute(self, user_query: str, session_id: str, tools: List[Any]) -> Dict[str, Any]:
        # 1. Rate Limiting
        self.rate_limiter.check_quota(session_id)

        # 2. Ingress Guardrails
        if self.guardrails.detect_violation(user_query):
            return {"status": "BLOCKED", "message": "Query flagged by Safety Guardrail."}

        # 3. PII Token Vaulting
        clean_query, vault_map = self.pii.redact(user_query)

        # 4. Context Summarization
        compacted_context = self.summarizer.compact_if_needed(session_id, clean_query)

        # 5. Model Routing
        selected_llm = self.router.select_model(compacted_context)

        # 6. HITL Action Inspection
        plan = selected_llm.generate_plan(compacted_context, tools)
        if self.hitl.requires_human_approval(plan.targeted_tool):
            checkpoint = self.hitl.suspend_execution(plan)
            return {"status": "PAUSED_HITL", "checkpoint_id": checkpoint.id}

        # 7 & 8. Tool Execution with Retries & Error Recovery
        tool_result = self.retry.run_with_backoff(
            lambda: self.tool_recovery.safe_call(plan.targeted_tool, plan.arguments)
        )

        # Egress PII De-tokenization
        final_answer = self.pii.restore(tool_result.content, vault_map)
        return {"status": "SUCCESS", "response": final_answer, "model_used": selected_llm.name}""", language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Multi-Stage Policy Guardrails & Interceptor Studio", divider="rainbow")

st.markdown("""
Test all 8 built-in middleware components simultaneously. Toggle policies, simulate tool API server failures, 
and inspect the complete policy evaluation audit trail in real time.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Scenario & Interceptor Configuration")
    sim_prompt = st.text_area(
        "User Prompt Input",
        value="Please perform a complex financial audit for admin@megacorp.internal with SSN 987-65-4321."
    )
    sim_tool = st.selectbox(
        "Targeted Action / Tool",
        options=["get_account_balance", "transfer_funds", "drop_table", "search_documentation"]
    )

    st.markdown("#### Policy Activations:")
    t_rate = st.toggle("Enable Rate Limiting", value=True)
    t_guard = st.toggle("Enable Content Guardrails", value=True)
    t_pii = st.toggle("Enable PII Redaction", value=True)
    t_sum = st.toggle("Enable Context Summarization", value=True)
    t_route = st.toggle("Enable Dynamic Model Router", value=True)
    t_hitl = st.toggle("Enable Human-In-The-Loop", value=True)
    t_retry = st.toggle("Enable Retry Handling", value=True)
    t_recover = st.toggle("Enable Tool Error Recovery", value=True)

    st.markdown("#### Simulation Overrides:")
    sim_failure = st.toggle("Simulate Downstream Tool Server Failure (503)", value=False)
    sim_approval = st.toggle("Human Supervisor Pre-Approval Granted", value=False)

    btn_eval = st.button("🚀 Run Interceptor Studio Evaluation", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Multi-Stage Evaluation Dashboard")

    if btn_eval:
        studio = MockBuiltInMiddlewareStudio(
            enable_summarization=t_sum,
            enable_hitl=t_hitl,
            enable_retries=t_retry,
            enable_tool_error_recovery=t_recover,
            enable_model_router=t_route,
            enable_guardrails=t_guard,
            enable_pii_redaction=t_pii,
            enable_rate_limiter=t_rate
        )

        res = studio.process(sim_prompt, sim_tool, sim_approval, sim_failure)

        # Metrics display
        m1, m2, m3 = st.columns(3)
        m1.metric("Selected Model", res.selected_model or "N/A")
        m2.metric("Incurred Cost", f"${res.total_cost_usd:.4f}" if res.total_cost_usd > 0 else "0.00 USD")
        m3.metric("HITL Status", "⏸️ Suspended" if res.hitl_paused else "✅ Clear")

        st.markdown("#### 💬 Final Synthesized Output:")
        if "BLOCKED" in res.final_output or "PAUSED" in res.final_output:
            st.warning(res.final_output)
        else:
            st.success(res.final_output)

        st.markdown("#### 🔍 Granular Interceptor Audit Trail:")
        for log in res.stage_logs:
            st.write(f"- **{log['stage']}** `[{log['status']}]`: {log['detail']}")
    else:
        st.info("Configure scenario parameters and click **Run Interceptor Studio Evaluation** to trace policy enforcement.")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does Human-in-the-Loop (HITL) middleware pause and resume execution without losing state?",
     "HITL middleware uses persistence checkpointers (such as SqliteSaver or PostgresSaver). When a sensitive tool call is identified, the middleware persists the entire graph state into storage, emits an interrupt signal, and returns an execution checkpoint token. The client or supervisor can review the proposed action and call graph.resume() with the checkpoint token to continue."),
    ("Why is exponential backoff with jitter superior to fixed delay retries for LLM APIs?",
     "When an API provider experiences a temporary outage, thousands of clients fail simultaneously. If all clients retry after a fixed delay (e.g. exactly 2.0 seconds), they hit the API at the exact same moment, causing a 'thundering herd' shock. Exponential backoff increases the delay exponentially, while jitter introduces random variance, spreading retries evenly over time."),
    ("Explain the difference between input guardrails and output guardrails.",
     "Input guardrails run on the user's prompt before tokenization, scanning for prompt injections, jailbreaks, and PII. Output guardrails run on the model's generated text before delivery, inspecting for hallucinations, toxic language, competitor mentions, and unintended leaks of system instructions."),
    ("How does PII redaction middleware maintain contextual understanding in LLMs without leaking sensitive data?",
     "PII middleware uses entity token replacement (e.g. replacing 'John Doe' with '[PERSON_1]' and '123 Fake St' with '[ADDRESS_1]'). The LLM understands the grammatical structure and relationship between the placeholders while remaining completely unaware of the actual customer's identity. On egress, the middleware performs token substitution to restore real names."),
    ("What are the latency tradeoffs of chaining multiple built-in middleware layers?",
     "Every synchronous middleware layer adds latency. Guardrails and PII scanners using auxiliary neural models can each add 100-300ms. In production, latency is managed by executing read-only checks asynchronously in parallel (fan-out), utilizing compiled regex engines for fast-path PII checks, and reserving heavy neural scanners only for high-risk prompts.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Retrying Non-Retryable Client Errors (400 / 401 / 403). Configuring retry middleware without filtering HTTP error codes causes the agent to waste 30+ seconds retrying invalid API keys or malformed JSON payloads that will never succeed. Only retry 429, 502, 503, and network timeouts.")
    st.error("**Trap 2 (Breaking Error)**: Unhandled HITL Timeouts in Synchronous Workflows. When a tool call is suspended for human approval, holding open an HTTP socket connection causes gateway timeouts (504). HITL workflows must always be asynchronous with polling or webhook callbacks.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Static Model Routing for Simple Tasks. Defaulting all agent turns to expensive frontier models drains budgets. Always implement dynamic routing middleware.")
    st.warning("**Trap 4 (Quality Warning)**: Aggressive Summarization Losing Tool Execution Arguments. Over-summarizing past dialogue can discard critical IDs or parameters generated in turn 1 that are needed in turn 6. Keep structured state separate from natural language message history.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: HITL Interrupt"):
        st.markdown("**Suspends execution at checkpoint token before high-risk tool dispatch, awaiting human supervisor signoff.**")
    with st.expander("🎴 Flashcard 2: Exponential Backoff + Jitter"):
        st.markdown("**Randomized, exponentially increasing delays between retries to prevent thundering herd crashes.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Token Vaulting (PII)"):
        st.markdown("**Bidirectional substitution replacing real customer identities with encrypted synthetic placeholders.**")
    with st.expander("🎴 Flashcard 4: Tool Error Recovery"):
        st.markdown("**Catches tool exceptions and translates stack traces into model observations so the agent can self-correct.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural role and operation of all 8 built-in middleware modules.", value=True)
chk2 = st.checkbox("I can implement Human-in-the-Loop checkpoints and execution interruption patterns.", value=True)
chk3 = st.checkbox("I know how to design exponential backoff algorithms with jitter to handle API 429 and 503 errors.", value=True)
chk4 = st.checkbox("I can configure bidirectional PII token vaulting for enterprise privacy compliance.", value=True)
chk5 = st.checkbox("I understand how dynamic model routing optimizes both operating costs and response latency.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 built-in middleware interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 11.2 Built-in Middleware!")
