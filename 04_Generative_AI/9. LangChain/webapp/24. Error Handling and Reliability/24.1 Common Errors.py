import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="24.1 Common Errors - LangChain Master Curriculum",
    page_icon="⚠️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class LCELCommonErrorDiagnosticEngine:
    """
    Enterprise Error Diagnostic and Triage Engine for LangChain applications.
    Simulates triage and automated remediation across the 8 common LLM failure categories:
    1. API Errors (Provider downtimes, HTTP 502/503 bad gateways)
    2. Authentication Errors (Missing OPENAI_API_KEY, expired tokens)
    3. Rate Limits (HTTP 429 TPM/RPM quota exhaustion)
    4. Timeout Errors (Slow tool responses, gateway 504s)
    5. Tool Errors (Schema mismatches, SQL execution exceptions)
    6. Parsing Errors (OutputParserException, markdown backtick pollution)
    7. Validation Errors (Pydantic field type mismatches, missing required keys)
    8. Context-Length Errors (Exceeding max model context window)
    """
    def __init__(self, tenant_env: str = "production"):
        self.tenant_env = tenant_env

    def diagnose_and_triage(
        self,
        error_category: str,
        input_token_count: int = 1500,
        parser_payload_format: str = "Valid Raw JSON",
        has_api_key: bool = True
    ) -> Dict[str, Any]:
        """
        Diagnoses an application failure and returns structural remediation instructions.
        """
        # Diagnostic catalog
        if not has_api_key:
            return {
                "category": "Authentication Error",
                "http_status": 401,
                "error_name": "AuthenticationError",
                "severity": "CRITICAL",
                "root_cause": "The runtime environment variable OPENAI_API_KEY is unset or invalid.",
                "remediation_strategy": "Inject validated secrets via secure secret managers (AWS Secrets Manager, Vault, or .env files).",
                "lcel_resilience_pattern": "Fail fast at application startup using configuration validation before initializing LLM instances.",
                "can_auto_recover": False,
                "recommended_retry": "Do NOT retry automatically; human intervention or credential reload required."
            }

        if input_token_count > 128000:
            return {
                "category": "Context-Length Error",
                "http_status": 400,
                "error_name": "ContextWindowExceededError",
                "severity": "HIGH",
                "root_cause": f"Input prompt size ({input_token_count:,} tokens) exceeds model context window limit (128,000 tokens).",
                "remediation_strategy": "Implement semantic chunking, recursive map-reduce summarization, or context pruning filters.",
                "lcel_resilience_pattern": "Chain a token counter runnable before the LLM to dynamically trim or summarize conversation history.",
                "can_auto_recover": True,
                "recommended_retry": "Retry immediately after applying token truncation or context compression."
            }

        if error_category == "Rate Limits (HTTP 429)":
            return {
                "category": "Rate Limit",
                "http_status": 429,
                "error_name": "RateLimitError",
                "severity": "HIGH",
                "root_cause": "Provider TPM (Tokens Per Minute) or RPM (Requests Per Minute) tier quota exhausted.",
                "remediation_strategy": "Implement exponential backoff with full jitter, or distribute traffic across multiple API keys/accounts.",
                "lcel_resilience_pattern": "model.with_retry(stop_after_attempt=5, wait_exponential_jitter=True)",
                "can_auto_recover": True,
                "recommended_retry": "Retry after exponential backoff sleep (e.g., 2s, 4s, 8s with random jitter)."
            }

        if error_category == "API Errors (502/503)":
            return {
                "category": "API Error",
                "http_status": 503,
                "error_name": "ServiceUnavailableError",
                "severity": "CRITICAL",
                "root_cause": "Upstream LLM provider infrastructure outage or transient gateway failure.",
                "remediation_strategy": "Configure multi-provider fallbacks (failover from OpenAI to Anthropic or Azure OpenAI).",
                "lcel_resilience_pattern": "primary_model.with_fallbacks([backup_model_claude, backup_model_azure])",
                "can_auto_recover": True,
                "recommended_retry": "Reroute traffic immediately to the fallback model runnable."
            }

        if error_category == "Timeout Errors (504)":
            return {
                "category": "Timeout Error",
                "http_status": 504,
                "error_name": "GatewayTimeoutError",
                "severity": "MEDIUM",
                "root_cause": "The model generation or external tool execution exceeded client request timeout limits.",
                "remediation_strategy": "Set explicit timeouts per runnable and employ asynchronous streaming to prevent socket drops.",
                "lcel_resilience_pattern": "model.with_config(timeout=10.0)",
                "can_auto_recover": True,
                "recommended_retry": "Retry with reduced generation max_tokens or fall back to cached tool output."
            }

        if error_category == "Tool Errors":
            return {
                "category": "Tool Error",
                "http_status": 500,
                "error_name": "ToolExecutionException",
                "severity": "MEDIUM",
                "root_cause": "External tool failed during execution (e.g., SQL syntax error, database connection refused).",
                "remediation_strategy": "Pass tool errors back to the model as observations so the agent can self-correct arguments.",
                "lcel_resilience_pattern": "Tool(name='sql_query', func=run_sql, handle_tool_error=True)",
                "can_auto_recover": True,
                "recommended_retry": "Feed error description back into the LLM conversational loop for next-step reasoning."
            }

        if error_category == "Parsing Errors":
            if parser_payload_format == "Markdown Wrapped JSON":
                return {
                    "category": "Parsing Error",
                    "http_status": 422,
                    "error_name": "OutputParserException",
                    "severity": "MEDIUM",
                    "root_cause": "Model wrapped JSON payload in markdown code fences (```json ... ```), breaking strict JSON parsers.",
                    "remediation_strategy": "Use LangChain's OutputFixingParser or JsonOutputParser which handles markdown strips automatically.",
                    "lcel_resilience_pattern": "JsonOutputParser() or OutputFixingParser.from_llm(parser=parser, llm=llm)",
                    "can_auto_recover": True,
                    "recommended_retry": "Strip code fence delimiters using regex pre-processing before deserializing."
                }
            elif parser_payload_format == "Corrupted JSON String":
                return {
                    "category": "Parsing Error",
                    "http_status": 422,
                    "error_name": "OutputParserException",
                    "severity": "HIGH",
                    "root_cause": "Model truncated output before completing JSON braces or produced unquoted keys.",
                    "remediation_strategy": "Invoke an auto-repair LLM chain to reconstruct valid JSON from the corrupted completion.",
                    "lcel_resilience_pattern": "OutputFixingParser.from_llm(parser=base_parser, llm=fast_repair_llm)",
                    "can_auto_recover": True,
                    "recommended_retry": "Invoke self-healing repair prompt with the malformed snippet."
                }
            else:
                return {
                    "category": "Parsing Success",
                    "http_status": 200,
                    "error_name": "NoError",
                    "severity": "INFO",
                    "root_cause": "JSON string conforms cleanly to parser schema.",
                    "remediation_strategy": "None required.",
                    "lcel_resilience_pattern": "prompt | model | JsonOutputParser()",
                    "can_auto_recover": True,
                    "recommended_retry": "N/A"
                }

        # Validation Errors
        return {
            "category": "Validation Error",
            "http_status": 422,
            "error_name": "ValidationError",
            "severity": "HIGH",
            "root_cause": "The generated JSON failed Pydantic schema validation (e.g., missing required field, type mismatch).",
            "remediation_strategy": "Inject exact Pydantic schema validation errors back into the LLM prompt to guide self-correction.",
            "lcel_resilience_pattern": "PydanticOutputParser(pydantic_object=TargetSchema)",
            "can_auto_recover": True,
            "recommended_retry": "Prompt the LLM: 'Your response failed validation with: {error}. Please regenerate valid data.'"
        }


# ==============================================================================
# HEADER BLOCK
# ==============================================================================
st.title("24.1 Common Errors: Failure Taxonomies and Exception Triage")
st.caption("LangChain v0.3 Enterprise Reliability • API Outages, Rate Limits, Timeouts, Tool Failures, Parser Crashes, and Context Window Exceeded")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Common LLM Failure Taxonomies and Exception Triage Architecture", divider="rainbow")

st.markdown("""
Unlike traditional deterministic software, LLM applications operate in an environment fraught with 
multi-dimensional failure modes. Failures arise across four distinct architectural domains: 
network infrastructure, security authentication, model reasoning non-determinism, and external tool execution.

An enterprise LangChain application implements an automated exception triage architecture that classifies 
errors immediately, isolates transient failures from fatal misconfigurations, and applies targeted remediation 
tactics (retries, fallbacks, self-healing parsers) without bubbling raw exceptions to end users.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ErrorTriageBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Arial", fontsize=11, fontcolor="#F8FAFC", style="filled", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_input_domain {
            label = "Tier 1: Request Ingestion & Pre-Execution Checks";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            client_req [label="Client Invocations\\n(Prompt & Context)", shape=box, fillcolor="#1E3A8A", color="#3B82F6"];
            auth_validator [label="Credential & Token Pre-Check\\n(Check OPENAI_API_KEY)", shape=diamond, fillcolor="#B45309", color="#F59E0B"];
            context_guard [label="Context Window Guard\\n(Token Count <= Context Limit)", shape=diamond, fillcolor="#B45309", color="#F59E0B"];
        }

        subgraph cluster_runtime_domain {
            label = "Tier 2: LCEL Runnable Execution & Provider Interaction";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            lcel_executor [label="LCEL Pipeline Runner\\n(model.invoke / astream)", shape=box, fillcolor="#1E3A8A", color="#3B82F6"];
            provider_gateway [label="Provider API Call\\n(OpenAI / Anthropic)", shape=box, fillcolor="#4C1D95", color="#8B5CF6"];
            tool_runner [label="External Tool Execution\\n(SQL / REST / Vector DB)", shape=box, fillcolor="#047857", color="#10B981"];
        }

        subgraph cluster_triage_domain {
            label = "Tier 3: Exception Interceptor & Classification Matrix";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            exception_router [label="Exception Router\\n(try / except Exception as e)", shape=hexagon, fillcolor="#991B1B", color="#EF4444"];
            rate_limit_handler [label="HTTP 429 Handler\\n(Exponential Backoff + Jitter)", shape=box, fillcolor="#0E7490", color="#06B6D4"];
            api_outage_handler [label="HTTP 502/503 Handler\\n(Model Fallback Routing)", shape=box, fillcolor="#0E7490", color="#06B6D4"];
            tool_error_handler [label="Tool Exception Handler\\n(Pass Error to Model Agent)", shape=box, fillcolor="#0E7490", color="#06B6D4"];
            parse_fixer [label="Parser Repair Handler\\n(OutputFixingParser / JSON Strip)", shape=box, fillcolor="#0E7490", color="#06B6D4"];
        }

        subgraph cluster_output_domain {
            label = "Tier 4: Successful Delivery or Graceful Degradation";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            clean_response [label="Validated Output Delivered\\n(HTTP 200 OK)", shape=box, fillcolor="#047857", color="#10B981"];
            degraded_response [label="Graceful Degradation Notice\\n(Cached Result / Polite Fallback)", shape=box, fillcolor="#B45309", color="#F59E0B"];
        }

        client_req -> auth_validator;
        auth_validator -> context_guard [label="Credentials Valid"];
        context_guard -> lcel_executor [label="Tokens in Range"];
        lcel_executor -> provider_gateway;
        provider_gateway -> tool_runner [label="Tool Call Emitted"];
        tool_runner -> clean_response [label="Tool & Model Success"];

        provider_gateway -> exception_router [label="Network/Provider Exception"];
        tool_runner -> exception_router [label="Tool Execution Crash"];
        lcel_executor -> exception_router [label="Parser / Schema Rejection"];

        exception_router -> rate_limit_handler [label="429 RateLimit"];
        exception_router -> api_outage_handler [label="503 Outage"];
        exception_router -> tool_error_handler [label="Tool Crash"];
        exception_router -> parse_fixer [label="JSON Syntax Error"];

        rate_limit_handler -> lcel_executor [label="Retry with Sleep"];
        api_outage_handler -> lcel_executor [label="Invoke Fallback Model"];
        parse_fixer -> clean_response [label="Fixed JSON"];
        exception_router -> degraded_response [label="Fatal Failure"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: API ERRORS
# ==============================================================================
st.header("2. API Errors (Provider Outages, Bad Gateway HTTP 502/503 and Network Dropouts)", divider="rainbow")

st.markdown("""
LLM applications rely on remote SaaS endpoints managed by third-party model vendors. 
Even the largest providers experience localized network drops, transient gateway timeouts (HTTP 504), 
and service unavailability (HTTP 503):

- **Symptoms**: `openai.InternalServerError`, `anthropic.APIConnectionError`, or HTTP 502 Bad Gateway responses.
- **Root Cause**: Upstream GPU cluster saturation, cloud zone outages, or transit network drops.
- **Anti-pattern**: Retrying the same failed endpoint indefinitely, freezing the client application.
- **Production Standard**: Implementing cross-provider multi-model fallbacks using LangChain's `.with_fallbacks()` mechanism.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In mission-critical production systems, pipelines are rarely hardcoded to a single vendor. 
A standard LCEL pipeline binds `ChatOpenAI(model="gpt-4o")` as the primary engine and pairs it with 
`ChatAnthropic(model="claude-3-5-sonnet")` as a secondary fallback. When OpenAI returns an HTTP 503, 
LangChain automatically switches execution to Anthropic within milliseconds.
""")

st.divider()

# ==============================================================================
# SECTION 3: AUTHENTICATION ERRORS
# ==============================================================================
st.header("3. Authentication Errors (Missing Environment Variables, Expired API Keys and RBAC Failures)", divider="rainbow")

st.markdown("""
Authentication errors occur when credentials fail authorization checks:

- **Symptoms**: `openai.AuthenticationError: 401 Unauthorized` or `anthropic.AuthenticationError`.
- **Root Cause**: Missing environment variable (`OPENAI_API_KEY` not found in runtime container), 
  expired billing payment cards causing provider account locks, or fine-grained RBAC project key revocation.
- **Fatal Nature**: Authentication errors are non-transient. Retrying an authentication failure is useless 
  and wastes compute.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production services perform **Pre-Flight Startup Audits**. Before spinning up HTTP listeners, the application 
verifies that credentials exist, are well-formed, and can successfully ping a lightweight endpoint (`models.list()`). 
If the audit fails, the container crashes early with exit code 1 to alert DevOps immediately.
""")

st.divider()

# ==============================================================================
# SECTION 4: RATE LIMITS
# ==============================================================================
st.header("4. Rate Limits (HTTP 429 TPM/RPM Exhaustion, Tier Quotas and Jittered Retries)", divider="rainbow")

st.markdown("""
Every LLM provider enforces strict usage limits partitioned into two metrics:
1. **RPM (Requests Per Minute)**: Total number of API invocations permitted in a 60-second window.
2. **TPM (Tokens Per Minute)**: Combined sum of prompt and completion tokens processed in a 60-second window.

When batch jobs or concurrent user traffic exceed either ceiling, the API returns `HTTP 429 Too Many Requests`.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Simple fixed-delay retries cause **Thundering Herd** problems, where all blocked worker threads sleep for the same duration 
and hit the provider API simultaneously, triggering another 429 cascade. 
Production stacks implement **Exponential Backoff with Full Jitter**:
`sleep_duration = random.uniform(0, min(max_backoff, base_backoff * (2 ** attempt_number)))`.
""")

st.divider()

# ==============================================================================
# SECTION 5: TIMEOUT ERRORS
# ==============================================================================
st.header("5. Timeout Errors (Slow Tool Invocations, Model Generation Hangs and Gateway 504s)", divider="rainbow")

st.markdown("""
Timeouts in LLM architectures occur in two places:
1. **Model Generation Timeouts**: During peak hours, large models generating long responses can take 30+ seconds, causing reverse proxies (NGINX, Cloudflare, AWS ALB) to drop HTTP connections with a 504 Gateway Timeout.
2. **Tool Execution Timeouts**: An agent invokes an external REST API or database query that hangs indefinitely waiting for locks.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Every LCEL runnable must specify explicit step-level timeouts using `.with_config(timeout=15.0)`. 
In addition, applications stream tokens to client browsers using `astream()`. Streaming keeps HTTP sockets alive 
and resets reverse proxy idle timeouts with every token received.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOOL ERRORS
# ==============================================================================
st.header("6. Tool Errors (SQL Syntax Crashes, Schema Mismatches and Swallowed Failure Antipatterns)", divider="rainbow")

st.markdown("""
When autonomous agents interact with tools, failures frequently occur:
- **Schema Mismatches**: The model passes arguments as strings instead of required integers or dictionaries.
- **Query Syntax Errors**: The agent generates invalid SQL syntax or referencing non-existent table columns.
- **Swallowed Error Antipattern**: Catching the exception and returning an empty string or `"Error occurred"` to the model. 
  The LLM then hallucinates plausible data, turning an infrastructure failure into a silent business error.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern LangChain tools enable `handle_tool_error=True`. When a tool crashes, the exception message is converted into 
a structured observation and returned directly to the agent's scratchpad. The LLM sees the error message 
(e.g., `column 'rev' does not exist in table 'orders'`) and autonomously corrects its query on the next turn.
""")

st.divider()

# ==============================================================================
# SECTION 7: PARSING ERRORS
# ==============================================================================
st.header("7. Parsing Errors (OutputParserException, Malformed JSON and Conversational Markdown Wrapper)", divider="rainbow")

st.markdown("""
Standard models are trained for conversational dialogue, not strict JSON serialization. Even when instructed 
to produce pure JSON, models often output markdown wrappers:
`Here is the result: ```json {"status": "ok"} ``` Hope this helps!`

When passed to standard `json.loads()`, this raises `OutputParserException`. Other common parsing failures include:
- Unquoted object keys (`{name: "Alice"}`).
- Trailing commas (`{"items": [1, 2, ],}`).
- Truncated JSON caused by reaching max generation tokens.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production pipelines utilize LangChain's `JsonOutputParser` (which extracts JSON from markdown fences automatically) 
or `OutputFixingParser`. When parsing fails, `OutputFixingParser` sends the malformed text back to a fast, cheap model 
(e.g., GPT-4o-mini) with instructions to repair the syntax without altering semantic fields.
""")

st.divider()

# ==============================================================================
# SECTION 8: VALIDATION ERRORS
# ==============================================================================
st.header("8. Validation Errors (Pydantic Schema Rejections, Missing Required Fields and Type Coercions)", divider="rainbow")

st.markdown("""
While parsing errors represent syntactical JSON problems, **Validation Errors** represent semantic contract breaches:
- The JSON is syntactically valid, but lacks mandatory fields defined in the Pydantic model.
- Field types do not match (e.g., string `"twenty"` provided for an integer field).
- Range constraints are violated (e.g., confidence score of `1.4` where schema requires `0.0 <= score <= 1.0`).
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern LCEL pipelines, Pydantic validation errors are formatted into descriptive feedback strings. 
If validation fails, an automatic self-healing loop passes the Pydantic error trace back to the model:
`"Your previous output failed validation: field 'age' must be an integer, but got 'unknown'. Please regenerate."`
""")

st.divider()

# ==============================================================================
# SECTION 9: CONTEXT-LENGTH ERRORS
# ==============================================================================
st.header("9. Context-Length Errors (ContextWindowExceeded, Max Token Truncation and Document Pruning)", divider="rainbow")

st.markdown("""
Every LLM architecture has a strict upper limit on input context length (e.g., 128k tokens for GPT-4o, 200k for Claude 3.5 Sonnet). 
When an application injects excessive chat history, large vector retrieval chunks, or massive documents, the API returns:
`ContextWindowExceededError: This model's maximum context length is 128000 tokens. However, your messages resulted in 134210 tokens.`

When this error occurs, the provider immediately aborts request execution.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production RAG systems enforce token budgeting prior to model dispatch:
1. **Dynamic Context Pruning**: Trimming older conversational messages while preserving system prompts.
2. **Context Compaction**: Summarizing long transcripts into bullet points before injecting them into prompts.
3. **Smart Retriever Filters**: Limiting retrieved vector chunks dynamically based on remaining available token capacity.
""")

st.divider()

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Common Error Diagnostic Engine (LCELCommonErrorDiagnosticEngine)", divider="rainbow")

st.markdown("""
The following complete Python implementation demonstrates `LCELCommonErrorDiagnosticEngine`, an enterprise-grade 
exception triage engine that categorizes failures and provides automated remediation plans.
""")

st.code(r'''
from typing import Dict, Any

class LCELCommonErrorDiagnosticEngine:
    """
    Enterprise Error Diagnostic and Triage Engine for LangChain applications.
    """
    def __init__(self, tenant_env: str = "production"):
        self.tenant_env = tenant_env

    def diagnose_and_triage(
        self,
        error_category: str,
        input_token_count: int = 1500,
        parser_payload_format: str = "Valid Raw JSON",
        has_api_key: bool = True
    ) -> Dict[str, Any]:
        if not has_api_key:
            return {
                "category": "Authentication Error",
                "http_status": 401,
                "severity": "CRITICAL",
                "root_cause": "OPENAI_API_KEY is missing or invalid.",
                "remediation_strategy": "Inject validated secrets from secure key store.",
                "can_auto_recover": False
            }

        if input_token_count > 128000:
            return {
                "category": "Context-Length Error",
                "http_status": 400,
                "severity": "HIGH",
                "root_cause": f"Input ({input_token_count:,} tokens) exceeds 128k limit.",
                "remediation_strategy": "Apply token pruning and summarize chat history.",
                "can_auto_recover": True
            }

        if error_category == "Rate Limits (HTTP 429)":
            return {
                "category": "Rate Limit",
                "http_status": 429,
                "severity": "HIGH",
                "root_cause": "TPM/RPM quota exhausted.",
                "remediation_strategy": "Retry with exponential backoff and random jitter.",
                "can_auto_recover": True
            }

        if error_category == "API Errors (502/503)":
            return {
                "category": "API Error",
                "http_status": 503,
                "severity": "CRITICAL",
                "root_cause": "Upstream LLM provider outage.",
                "remediation_strategy": "Reroute to secondary fallback model (Claude / Azure).",
                "can_auto_recover": True
            }

        return {
            "category": "Generic Diagnostic",
            "http_status": 200,
            "severity": "INFO",
            "root_cause": "Standard execution path.",
            "remediation_strategy": "Normal pipeline progression.",
            "can_auto_recover": True
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: LLM Error Diagnostic Simulator and Exception Recovery Sandbox", divider="rainbow")

st.markdown("""
Select failure parameters below to simulate how the enterprise exception triage engine analyzes 
errors and generates structured remediation recommendations.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    selected_category = st.selectbox(
        "Simulated Failure Scenario",
        options=[
            "API Errors (502/503)",
            "Authentication Errors (401)",
            "Rate Limits (HTTP 429)",
            "Timeout Errors (504)",
            "Tool Errors",
            "Parsing Errors",
            "Validation Errors",
            "Context-Length Errors"
        ],
        index=0
    )
    simulated_tokens = st.slider("Prompt Token Volume", min_value=1000, max_value=145000, value=3500, step=2500)

with col_wb2:
    parser_format = st.radio(
        "Model Output Payload Format",
        options=["Valid Raw JSON", "Markdown Wrapped JSON", "Corrupted JSON String"],
        index=0
    )
    api_key_valid = st.toggle("API Key Present in Runtime Environment", value=True)

# Run diagnostic triage
diag_engine = LCELCommonErrorDiagnosticEngine()
diag_result = diag_engine.diagnose_and_triage(
    error_category=selected_category,
    input_token_count=simulated_tokens,
    parser_payload_format=parser_format,
    has_api_key=api_key_valid
)

st.subheader("🔍 Triage Diagnostic Report")

col_res1, col_res2, col_res3 = st.columns(3)
col_res1.metric("Category", diag_result["category"])
col_res2.metric("HTTP Status", f"{diag_result['http_status']}")
col_res3.metric("Severity Level", diag_result["severity"])

if diag_result["severity"] == "CRITICAL":
    st.error(f"🚨 **Root Cause**: {diag_result['root_cause']}")
elif diag_result["severity"] == "HIGH":
    st.warning(f"⚠️ **Root Cause**: {diag_result['root_cause']}")
else:
    st.info(f"ℹ️ **Root Cause**: {diag_result['root_cause']}")

st.markdown(f"**Recommended Remediation**: {diag_result['remediation_strategy']}")
st.markdown(f"**LCEL Architectural Pattern**: `{diag_result.get('lcel_resilience_pattern', 'N/A')}`")
st.markdown(f"**Auto-Recovery Possible**: {'✅ Yes' if diag_result['can_auto_recover'] else '❌ No (Human Intervention Needed)'}")

st.divider()

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("12. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions on LLM Errors")

with st.expander("Q1: What is the Thundering Herd problem in LLM rate limit handling, and how is it mitigated?"):
    st.markdown("""
    **Answer**: When a batch of concurrent worker threads receives HTTP 429 rate limit responses, a naive fixed-retry 
    strategy causes all workers to sleep for the exact same duration and retry simultaneously. This creates a recurring wave 
    of synchronized requests that re-exhausts provider quotas immediately. It is mitigated by adding **Full Jitter**: 
    introducing a random sleep interval on top of exponential backoff to scatter request retries evenly over time.
    """)

with st.expander("Q2: Why is swallowing tool exceptions considered a dangerous antipattern in autonomous agents?"):
    st.markdown("""
    **Answer**: If a tool crash (e.g., database connection timeout) is caught and replaced with a generic fallback 
    (such as `"No data found"`), the LLM agent is denied the opportunity to understand why the tool failed. 
    Believing the query completed successfully with zero records, the model often fabricates answers from its pre-training weights, 
    transforming a recoverable infrastructure timeout into an unverified customer-facing hallucination.
    """)

with st.expander("Q3: How does OutputFixingParser resolve JSON formatting failures without human intervention?"):
    st.markdown("""
    **Answer**: When an initial output parser throws `OutputParserException` due to syntax issues (such as unquoted keys or markdown backticks), 
    `OutputFixingParser` intercepts the exception. It constructs a new prompt containing the original corrupted text and the exact 
    parser error message, directing a fast model (such as GPT-4o-mini) to repair the syntax while preserving the schema.
    """)

with st.expander("Q4: What is the architectural difference between a 400 ContextWindowExceeded error and a 429 RateLimit error?"):
    st.markdown("""
    **Answer**: A 429 RateLimit error is **temporal**: the request is valid, but the rate of requests or tokens per minute 
    has exceeded current tier limits; retrying after backoff will eventually succeed. A 400 ContextWindowExceeded error is **structural**: 
    the prompt payload itself is fundamentally too large for the model's context window; retrying without pruning or compressing 
    the prompt will fail every single time.
    """)

with st.expander("Q5: What is the recommended strategy for handling authentication errors in automated retry loops?"):
    st.markdown("""
    **Answer**: Authentication errors (HTTP 401/403) are non-transient and must be explicitly excluded from retry loops. 
    Retrying an invalid API key burns CPU cycles and delays user feedback. The system must fail fast, log a critical security alert, 
    and stop execution immediately.
    """)

st.markdown("### Critical Production Traps & Antipatterns")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("**Trap 1 (Retrying Non-Transient 401/403 Errors)**: Configuring blanket retry loops on all exceptions. Retrying invalid API keys or malformed prompts wastes worker threads and floods logs without any possibility of success.")
    st.warning("**Trap 2 (Swallowing Tool Crashes)**: Returning empty strings when tools fail. The LLM assumes the search returned no records and hallucinates answers from imagination instead of reporting the service outage.")
with col_tr2:
    st.error("**Trap 3 (Unjittered Exponential Retries)**: Using standard exponential backoff without jitter. Creates severe thundering herd spikes on LLM provider gateways during traffic surges.")
    st.warning("**Trap 4 (Blind Prompt Injection in Context Truncation)**: Truncating prompts from the top down without preserving the system message. Cutting the system prompt strips security guardrails and instructions from the model.")

st.markdown("### Core Technical Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the difference between TPM and RPM in LLM rate limits?")
        st.caption("Flip for answer:")
        st.info("**Back**: RPM governs request volume per minute; TPM governs cumulative token consumption (input + output) per minute.")

    with st.container(border=True):
        st.markdown("**Front**: How does full jitter prevent thundering herd retries?")
        st.caption("Flip for answer:")
        st.info("**Back**: By randomizing backoff intervals between 0 and 2^attempt * base_backoff, spreading retries evenly across time.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the role of handle_tool_error=True in LangChain tools?")
        st.caption("Flip for answer:")
        st.info("**Back**: It converts tool crashes into agent observations, enabling the model to inspect the error and self-correct.")

    with st.container(border=True):
        st.markdown("**Front**: Why is a 400 ContextWindowExceeded error considered non-transient?")
        st.caption("Flip for answer:")
        st.info("**Back**: The prompt is physically larger than model capacity. Retrying without pruning or compressing tokens will always fail.")

st.divider()

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Common Errors in LangChain:")

c1 = st.checkbox("I understand the 4 failure domains of LLM systems (network, auth, model, tools).", value=True)
c2 = st.checkbox("I know how to distinguish transient errors (429, 503) from non-transient errors (401, 400).", value=True)
c3 = st.checkbox("I can implement exponential backoff with full jitter to eliminate thundering herd retries.", value=True)
c4 = st.checkbox("I understand why swallowing tool exceptions creates silent customer hallucinations.", value=True)
c5 = st.checkbox("I know how to use OutputFixingParser and JsonOutputParser to heal malformed JSON payloads.", value=True)
c6 = st.checkbox("I know how to manage context-length budgeting and pruning to prevent ContextWindowExceeded errors.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Certified: You have mastered Common Error triage, root-cause taxonomy, and automated remediation in LangChain.")
