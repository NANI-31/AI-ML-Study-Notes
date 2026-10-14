import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="2.2 Environment Configuration - LangGraph Curriculum",
    page_icon="⚙️",
    layout="wide"
)

# Custom Theme CSS
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
    .badge-key {
        background-color: #1E293B;
        color: #10B981;
        border: 1px solid #10B981;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE CONFIGURATION VALIDATOR CLASS FOR WORKBENCH & PIPELINE
# ==============================================================================
class EnvironmentConfigValidator:
    """
    Validates enterprise environment variables and configuration settings
    required for secure, observable LangGraph execution.
    """
    def __init__(self):
        self.required_keys = [
            ("OPENAI_API_KEY", "Model Provider Credential", True),
            ("ANTHROPIC_API_KEY", "Secondary Provider Credential", False),
            ("LANGSMITH_API_KEY", "Observability Access Token", True),
            ("LANGSMITH_TRACING", "Observability Boolean Toggle", True),
            ("LANGSMITH_PROJECT", "Tracing Project Workspace", True),
            ("DATABASE_URL", "PostgreSQL Checkpointer URI", True)
        ]

    def validate_environment(self, env_dict: Dict[str, str]) -> Tuple[List[Dict[str, Any]], bool]:
        audit_results = []
        all_passed = True

        for key, description, is_mandatory in self.required_keys:
            val = env_dict.get(key, "").strip()
            present = bool(val)
            
            # Mask value
            masked = f"{val[:3]}...{val[-3:]}" if len(val) > 8 else ("*****" if val else "MISSING")
            
            valid = present or (not is_mandatory)
            if is_mandatory and not present:
                all_passed = False

            audit_results.append({
                "Variable Name": key,
                "Description": description,
                "Mandatory": "Yes" if is_mandatory else "Optional",
                "Status": "Configured" if present else ("Missing (Optional)" if not is_mandatory else "MISSING (CRITICAL)"),
                "Masked Value": masked,
                "Valid": valid
            })

        return audit_results, all_passed


# ==============================================================================
# TITLE & HERO OVERVIEW
# ==============================================================================
st.title("2.2 Environment Configuration: Secrets, .env Hygiene, & Runtime Configs")
st.caption("Chapter 2: LangGraph Installation and Environment | Domain: 10. LangGraph Multi-Agent Systems")

st.markdown("""
A stateful AI agent platform interacts with commercial LLM APIs, vector databases, checkpointer relational databases, 
and observability tracing hubs. Hardcoding keys or treating configuration as an afterthought leads to catastrophic 
consequences: **leaked API keys on GitHub**, unmasked credentials in logs, and divergence between development and production.

In this lesson, you will master:
* Managing local environment defaults with `.env` files without risking git leaks.
* Configuring the 12-factor application environment variables for LangGraph.
* Passing dynamic per-request parameters via LangChain's **`RunnableConfig`** and `configurable` namespaces.
* Integrating enterprise Secrets Managers (AWS Secrets Manager, HashiCorp Vault, Kubernetes Secrets).
* Enforcing strict parity and safety policies between development, staging, and production environments.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Multi-Tier Environment & Secrets Management Lifecycle", divider="rainbow")

st.markdown("""
The architecture below illustrates how environment configurations and credentials flow into a production LangGraph runtime: 
from encrypted secrets storage to container injection, schema validation, and dynamic per-request thread scoping.
""")

blueprint_dot = """
digraph ConfigLifecycle {
    graph [rankdir=TB, bgcolor="#0B1120", pad="0.3", nodesep="0.4", ranksep="0.5"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.2,0.1", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#64748B", fontcolor="#94A3B8", penwidth=1.2];

    subgraph cluster_vault {
        label = "1. Enterprise Secret Storage (Encrypted at Rest)";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#38BDF8"; fontsize = 11;
        CloudVault [label="AWS Secrets Manager / Vault / K8s Secrets\\n(KMS Encrypted API Keys, DB Passwords)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
        LocalEnv [label="Local Developer .env File\\n(Excluded from Git via .gitignore)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_injection {
        label = "2. Runtime Environment Ingestion";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#10B981"; fontsize = 11;
        EnvLoader [label="Pydantic BaseSettings / python-dotenv\\n(Strict Type Validation on Boot)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        Sanitizer [label="Memory & Log Sanitizer\\n(Redacts secret patterns before logging)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    }

    subgraph cluster_graph {
        label = "3. Dynamic Graph Invocations (RunnableConfig)";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#F59E0B"; fontsize = 11;
        GlobalConfig [label="Global System Config\\n(DB connection pools, LangSmith project)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        PerRequestConfig [label="Dynamic Request Config (RunnableConfig)\\n{'configurable': {'thread_id': '...', 'user_id': '...'}}", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
    }

    subgraph cluster_runtime {
        label = "4. LangGraph Execution Engine";
        color = "#1E293B"; style = "filled"; fillcolor = "#0F172A"; fontcolor = "#EC4899"; fontsize = 11;
        StateGraphExec [label="StateGraph Superstep Execution\\n(Nodes access config safely without globals)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#EC4899"];
    }

    CloudVault -> EnvLoader [label="Pod Startup Mount"];
    LocalEnv -> EnvLoader [label="Local Dev Load"];
    EnvLoader -> Sanitizer [label="Validate Schema"];
    Sanitizer -> GlobalConfig;

    GlobalConfig -> StateGraphExec [label="Inject Static Clients"];
    PerRequestConfig -> StateGraphExec [label="Per-Invocation Context", color="#EC4899", penwidth=1.8];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - .ENV FILES
# ==============================================================================
st.header("2. .env Files: Local Development Defaults, Dotenv Parsers, and Git Hygiene", divider="rainbow")

st.markdown("""
During local development, API keys and database credentials reside in a root `.env` file. 
However, improper `.env` management is the #1 cause of catastrophic credential leakage.
""")

col_env1, col_env2 = st.columns([1, 1])

with col_env1:
    st.markdown("""
    ### Canonical `.env.example` Pattern
    * **Never commit `.env` to Git**: Always maintain an unpopulated `.env.example` in version control:
    """)
    st.code("""
# .env.example (Safe to commit)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=default
DATABASE_URL=postgresql://user:pass@localhost:5432/langgraph_db
    """, language="bash")

with col_env2:
    st.markdown("""
    ### Git Hygiene Rule
    Ensure your root `.gitignore` includes:
    """)
    st.code("""
# .gitignore
.env
.env.*
!.env.example
*.pem
secrets/
    """, language="gitignore")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - API KEYS
# ==============================================================================
st.header("3. API Keys: Provider Credentials, Token Scrubbing, and Rotation Protocols", divider="rainbow")

st.markdown("""
API keys grant administrative access to billable cloud compute. In LangGraph production pipelines:
* **Token Scrubbing**: Never log API keys. Middleware must automatically redact strings matching provider regex patterns (e.g. `sk-[A-Za-z0-9]{32,}`).
* **Automated Rotation**: Production environments must support dual-key rotation: while Key A is active, inject Key B, transition traffic, and revoke Key A without restarting services.
* **Per-Tenant Service Accounts**: Rather than using a single shared company API key, enterprise deployments scope API keys per department or enterprise client.
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - ENVIRONMENT VARIABLES
# ==============================================================================
st.header("4. Environment Variables: 12-Factor App Principles & System-Level Variables", divider="rainbow")

st.markdown("""
According to the **12-Factor App methodology**, application configuration must be strictly separated from code. 
LangGraph and LangChain recognize standard system environment variables natively:
""")

var_table = [
    {"Variable": "OPENAI_API_KEY", "Purpose": "Automatically read by `ChatOpenAI` if not passed explicitly in Python constructor."},
    {"Variable": "ANTHROPIC_API_KEY", "Purpose": "Automatically read by `ChatAnthropic`."},
    {"Variable": "LANGSMITH_TRACING", "Purpose": "Enables distributed background tracing when set to 'true'."},
    {"Variable": "LANGSMITH_ENDPOINT", "Purpose": "Specifies LangSmith cloud or self-hosted enterprise endpoint URI."},
    {"Variable": "LANGSMITH_API_KEY", "Purpose": "Authentication token for LangSmith trace ingestion."},
    {"Variable": "LANGSMITH_PROJECT", "Purpose": "Organizes run traces into named projects (e.g., 'staging-support-agent')."}
]
st.dataframe(pd.DataFrame(var_table), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - CONFIGURATION MANAGEMENT
# ==============================================================================
st.header("5. Configuration Management: Dynamic Graph Configs via RunnableConfig", divider="rainbow")

st.markdown("""
One of the most powerful features of LangGraph is that nodes do not need to rely on static environment variables. 
You can pass dynamic runtime configuration into **`graph.invoke()`** using **`RunnableConfig`**:
""")

col_rc1, col_rc2 = st.columns([1, 1])

with col_rc1:
    st.markdown("""
    ### How `configurable` Works
    * When invoking a graph, provide a config dictionary with a `configurable` sub-key.
    * Every node function in the graph can declare a second parameter: `def my_node(state, config): ...`.
    * Nodes read user IDs, model overrides, or tenant feature flags directly from `config["configurable"]`!
    """)

with col_rc2:
    st.markdown("""
    ### Code Example
    """)
    st.code("""
from langchain_core.runnables import RunnableConfig

def agent_node(state, config: RunnableConfig):
    # Read dynamic configuration safely
    tenant_id = config["configurable"].get("tenant_id")
    model_name = config["configurable"].get("model_name", "gpt-4o")
    print(f"Executing for tenant {tenant_id} with model {model_name}")
    return {"status": "DONE"}

# Invocation with dynamic runtime parameters
config = {
    "configurable": {
        "thread_id": "session-4412",
        "tenant_id": "acme-corp",
        "model_name": "claude-3-5-sonnet"
    }
}
graph.invoke({"input": "Hello"}, config=config)
    """, language="python")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - SECRETS MANAGEMENT
# ==============================================================================
st.header("6. Secrets Management: Vault, AWS Secrets Manager, & Kubernetes Secrets", divider="rainbow")

st.markdown("""
In enterprise Kubernetes environments, `.env` files are strictly prohibited. Production deployments rely on 
centralized, hardware-encrypted secrets managers:
""")

sec_comp = [
    {
        "Secrets Manager": "AWS Secrets Manager",
        "Mechanism": "KMS-encrypted JSON secrets fetched via IAM roles on container boot",
        "Key Advantage": "Automatic key rotation with AWS Lambda integration"
    },
    {
        "Secrets Manager": "HashiCorp Vault",
        "Mechanism": "Dynamic ephemeral secrets with short time-to-live (TTL) lease tokens",
        "Key Advantage": "Multi-cloud and on-premise datacenter standardization"
    },
    {
        "Secrets Manager": "Kubernetes Secrets",
        "Mechanism": "Base64 encoded secrets injected into pods as environment variables or volume mounts",
        "Key Advantage": "Native integration with Helm and ArgoCD GitOps pipelines"
    }
]
st.dataframe(pd.DataFrame(sec_comp), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - DEVELOPMENT VS PRODUCTION CONFIGURATION
# ==============================================================================
st.header("7. Development vs Production Configuration: Parity, Staging, & Enterprise Profiles", divider="rainbow")

st.markdown("""
Enterprise AI architectures require strict separation between Development, Staging, and Production profiles:
""")

profile_matrix = [
    {
        "Dimension": "Checkpointer Storage",
        "Local Development": "`MemorySaver()` (Process RAM)",
        "Staging Environment": "`SqliteSaver()` / Dev Postgres",
        "Production Environment": "`PostgresSaver()` with PgBouncer connection pool"
    },
    {
        "Dimension": "LangSmith Project",
        "Local Development": "`LANGSMITH_PROJECT=local-dev`",
        "Staging Environment": "`LANGSMITH_PROJECT=staging-evals`",
        "Production Environment": "`LANGSMITH_PROJECT=prod-customer-facing`"
    },
    {
        "Dimension": "Model Endpoints",
        "Local Development": "Ollama / Mock models / mini models",
        "Staging Environment": "Staging rate-limited commercial APIs",
        "Production Environment": "Provisioned Throughput (PTUs) + Multi-region fallbacks"
    },
    {
        "Dimension": "Log Redaction",
        "Local Development": "Full verbose prompt output in console",
        "Staging Environment": "Partial PII masking",
        "Production Environment": "Zero prompt retention / Mandatory Presidio masking"
    }
]
st.dataframe(pd.DataFrame(profile_matrix), use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Secure Enterprise Configuration Manager", divider="rainbow")

st.markdown("""
Below is a type-safe, production-ready configuration module using Pydantic Settings. 
It automatically validates required credentials on application startup and fails fast if any critical key is missing:
""")

prod_code = """
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class EnterpriseLangGraphSettings(BaseSettings):
    \"\"\"
    Type-safe configuration loader with automatic .env fallback
    and validation on startup.
    \"\"\"
    # Model Credentials
    openai_api_key: str = Field(..., description="OpenAI API Key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API Key")
    
    # LangSmith Tracing
    langsmith_tracing: bool = Field(True, description="Enable LangSmith distributed tracing")
    langsmith_api_key: str = Field(..., description="LangSmith API Key")
    langsmith_project: str = Field("production-agent", description="LangSmith project name")
    
    # Checkpointer Database
    database_url: str = Field(..., description="PostgreSQL checkpointer connection URI")
    
    # Runtime Tuning
    recursion_limit: int = Field(25, description="Max graph supersteps before error")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Bootstrap validation at startup
def load_app_settings() -> EnterpriseLangGraphSettings:
    try:
        settings = EnterpriseLangGraphSettings()
        print("Application configuration validated successfully.")
        return settings
    except Exception as e:
        print(f"FATAL: Configuration validation failed on startup: {e}")
        raise SystemExit(1)
"""
st.code(prod_code, language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Environment Configuration Inspector & Validator", divider="rainbow")

st.markdown("""
Simulate and audit your application environment configuration. Enter sample keys (or leave them empty) 
to test startup validation, secret masking, and compliance checks.
""")

validator = EnvironmentConfigValidator()

col_wb1, col_wb2 = st.columns([1.1, 0.9])

with col_wb1:
    st.subheader("1. Input Environment Settings")
    
    input_openai = st.text_input("OPENAI_API_KEY", value="sk-proj-9941a8b27c6d5e4f3a2b1c", type="password")
    input_anthropic = st.text_input("ANTHROPIC_API_KEY (Optional)", value="", type="password")
    input_ls_key = st.text_input("LANGSMITH_API_KEY", value="lsv2_pt_882194a0b1c2", type="password")
    input_ls_tracing = st.selectbox("LANGSMITH_TRACING", ["true", "false"], index=0)
    input_ls_proj = st.text_input("LANGSMITH_PROJECT", value="prod-customer-service")
    input_db_url = st.text_input("DATABASE_URL", value="postgresql://agent_admin:secret@localhost:5432/checkpoints")

    audit_btn = st.button("Audit Environment Configuration 🔍", use_container_width=True)

with col_wb2:
    st.subheader("2. Compliance Audit & Redaction Report")
    
    env_candidate = {
        "OPENAI_API_KEY": input_openai,
        "ANTHROPIC_API_KEY": input_anthropic,
        "LANGSMITH_API_KEY": input_ls_key,
        "LANGSMITH_TRACING": input_ls_tracing,
        "LANGSMITH_PROJECT": input_ls_proj,
        "DATABASE_URL": input_db_url
    }
    
    report, is_valid = validator.validate_environment(env_candidate)
    
    m1, m2 = st.columns(2)
    m1.metric("Mandatory Keys Check", "PASS" if is_valid else "FAIL")
    m2.metric("Environment Status", "Production Ready" if is_valid else "Incomplete")

    st.markdown("#### Variable Audit Table")
    st.dataframe(pd.DataFrame(report), use_container_width=True)

    if is_valid:
        st.success("✅ Configuration Audit Passed: All mandatory secrets and database URIs are present.")
    else:
        st.error("❌ Configuration Audit Failed: Critical required variables are missing. Startup would halt.")

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Interviewers frequently test senior engineers on secrets management, credential leakage prevention, 
and runtime thread configuration.
""")

st.subheader("5 Core Architectural Interview Questions")

with st.expander("Q1: How do you pass user-specific authentication tokens into LangGraph nodes without global variables?"):
    st.markdown("""
    **Answer**: Global variables are strictly prohibited in multi-tenant systems because concurrent async requests will collide 
    and overwrite each other's credentials. Instead, pass user tokens through **`RunnableConfig`**:
    `graph.invoke(input, config={"configurable": {"user_jwt": token, "thread_id": id}})`. 
    Inside any node function, declare `def my_node(state, config: RunnableConfig):` and extract the token directly 
    via `config["configurable"]["user_jwt"]`.
    """)

with st.expander("Q2: What is the risk of using `.env` files in production Docker containers?"):
    st.markdown("""
    **Answer**: Baking a `.env` file into a Docker image during build time exposes raw secrets in the image layers. 
    Anyone with read access to the container registry can extract the plain-text credentials. In production, Docker images 
    must be completely stateless and secret-free; credentials should be injected at container runtime via Kubernetes Secrets 
    or AWS Secrets Manager environment variable mounts.
    """)

with st.expander("Q3: How does LangSmith tracing handle sensitive PII in logged prompts?"):
    st.markdown("""
    **Answer**: LangSmith provides **Hide Input / Output** settings and custom client-side masking hooks. 
    Developers can implement a client filter function in `langsmith.Client` or configure Presidio redactors 
    to scrub credit card numbers, emails, and passwords before traces leave the local network boundary.
    """)

with st.expander("Q4: What happens if `DATABASE_URL` for `PostgresSaver` experiences transient network partition?"):
    st.markdown("""
    **Answer**: Without connection pooling, checkpointer commit writes fail and throw unhandled connection exceptions, 
    terminating the active superstep. In production, wrap `PostgresSaver` with **`psycopg_pool.ConnectionPool`** and 
    configure retry policies on checkpointer writes to gracefully absorb momentary network hiccups.
    """)

with st.expander("Q5: Why should `recursion_limit` be configured explicitly rather than relying on defaults?"):
    st.markdown("""
    **Answer**: The default `recursion_limit` (typically 25) may be too generous for simple query-response workflows 
    (wasting money if an agent gets stuck in a 25-turn loop) or too restrictive for deep multi-agent research tasks 
    that legitimately require 40 supersteps. Setting `recursion_limit` per workflow via `RunnableConfig` aligns loop 
    thresholds with business requirements.
    """)

st.subheader("4 Production Traps & Antipatterns")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.error("""
    **Trap 1: Accidental Git Commit of Real API Keys (CRITICAL)**
    * *Error*: Pushing a `.env` file containing live OpenAI or Anthropic production keys to GitHub.
    * *Impact*: Web crawlers harvest keys within seconds, running up thousands in unauthorized API usage.
    * *Fix*: Mandate pre-commit hooks (`git-secrets` / `trufflehog`) to block commits containing key signatures.
    """)

    st.error("""
    **Trap 2: Logging Unredacted `RunnableConfig` in Production Traces (CRITICAL)**
    * *Error*: Printing `print(f"Config: {config}")` when `config` contains user JWTs or customer PII.
    * *Impact*: Leaks authentication tokens into centralized log aggregators (Datadog/CloudWatch).
    * *Fix*: Implement structured loggers that explicitly sanitize the `configurable` dictionary.
    """)

with col_t2:
    st.warning("""
    **Trap 3: Forgetting `LANGSMITH_PROJECT` Causing Trace Contamination (WARNING)**
    * *Warning*: Leaving `LANGSMITH_PROJECT` set to 'default' across dev, staging, and prod.
    * *Impact*: Mixes dirty developer test queries with real production customer evaluation metrics.
    * *Fix*: Set distinct project names per environment profile (`prod-app`, `dev-app`).
    """)

    st.warning("""
    **Trap 4: Modifying Environment Variables at Runtime via `os.environ` (WARNING)**
    * *Warning*: Setting `os.environ['OPENAI_API_KEY'] = key` inside a web request handler.
    * *Impact*: Mutates process-wide state, causing race conditions across concurrent multi-threaded requests.
    * *Fix*: Pass keys explicitly via `ChatOpenAI(api_key=...)` or `RunnableConfig`.
    """)

st.subheader("Technical Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 1: RunnableConfig</strong><br><br>
        <em>Front:</em> How do nodes receive dynamic per-request parameters?<br><br>
        <em>Back:</em> Through the <code>RunnableConfig</code> parameter via <code>config['configurable']</code>.
    </div>
    """, unsafe_allow_html=True)

with fc_col2:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 2: .env.example</strong><br><br>
        <em>Front:</em> What is the purpose of .env.example?<br><br>
        <em>Back:</em> Documents all required keys with dummy values and is safely committed to Git.
    </div>
    """, unsafe_allow_html=True)

with fc_col3:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 3: 12-Factor</strong><br><br>
        <em>Front:</em> What does the 12-Factor App rule mandate for configs?<br><br>
        <em>Back:</em> Strict separation of configuration from code; injection via environment variables.
    </div>
    """, unsafe_allow_html=True)

with fc_col4:
    st.markdown("""
    <div class="metric-card">
        <strong>Card 4: Recursion Limit</strong><br><br>
        <em>Front:</em> What happens when recursion_limit is exceeded?<br><br>
        <em>Back:</em> LangGraph raises <code>GraphRecursionError</code> to prevent infinite billing loops.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of LangGraph environment configuration:")

chk1 = st.checkbox("I know how to manage `.env` files safely using `.env.example` and `.gitignore`.", value=True)
chk2 = st.checkbox("I understand the standard environment variables recognized by LangGraph and LangSmith.", value=True)
chk3 = st.checkbox("I know how to pass dynamic per-request context into graph nodes using `RunnableConfig`.", value=True)
chk4 = st.checkbox("I understand how enterprise Secrets Managers inject credentials into Kubernetes pods.", value=True)
chk5 = st.checkbox("I can implement type-safe configuration validation using Pydantic Settings.", value=True)
chk6 = st.checkbox("I know how to prevent credential leakage and avoid runtime mutation of global environment state.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Lesson 2.2: Environment Configuration! Proceed to Lesson 2.3: Verifying Installation.")
