import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="35.1 Tracing Best Practices - LangSmith Master Curriculum",
    page_icon="📐",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class TracingTaxonomyEngine:
    """
    Enterprise Tracing Taxonomy & Metadata Hygiene Auditor for LangSmith.
    Enforces production tracing standards: meaningful project naming conventions,
    bounded low-cardinality tags, structured metadata schemas, operation span scoping,
    and release version lineage tracking.
    """
    def __init__(self):
        self.validation_records: List[Dict[str, Any]] = []

    def audit_trace_configuration(
        self,
        project_name: str,
        tags: List[str],
        metadata: Dict[str, Any],
        operation_type: str,
        app_version: str
    ) -> Dict[str, Any]:
        """
        Audits a proposed run span configuration against enterprise tracing best practices.
        """
        violations = []
        warnings = []

        # 1. Project Naming Convention Check
        valid_env_prefixes = ["prod-", "staging-", "dev-", "eval-"]
        if not any(project_name.startswith(p) for p in valid_env_prefixes):
            violations.append(f"Project '{project_name}' lacks required environment prefix ({', '.join(valid_env_prefixes)}).")
        if " " in project_name or any(c.isupper() for c in project_name):
            violations.append("Project name must be strictly lowercase-kebab-case without spaces.")

        # 2. Tag Cardinality Check (Tags must be categorical and low-cardinality)
        for tag in tags:
            if any(char.isdigit() for char in tag) and len(tag) > 10:
                violations.append(f"Tag '{tag}' looks like a high-cardinality ID (UUID/hash). Move unique identifiers to metadata!")
        if len(tags) > 8:
            warnings.append(f"Span has {len(tags)} tags. Limit tags to <= 8 high-level categorical labels.")

        # 3. Metadata Completeness Check
        required_meta_keys = ["environment", "tenant_id", "session_id"]
        for rk in required_meta_keys:
            if rk not in metadata:
                violations.append(f"Missing mandatory metadata key: '{rk}'.")

        # 4. Version Format Check
        if not (app_version.startswith("v") or len(app_version) == 7):  # SemVer or 7-char Git SHA
            warnings.append(f"Application version '{app_version}' should follow SemVer (v1.2.3) or short Git SHA format.")

        # 5. Operation Scope Check
        valid_operations = ["retrieval", "llm_call", "tool_execution", "agent_node", "guardrail_check"]
        if operation_type not in valid_operations:
            warnings.append(f"Operation type '{operation_type}' is unconventional. Recommended: {valid_operations}.")

        is_compliant = len(violations) == 0

        audit_result = {
            "is_compliant": is_compliant,
            "violations": violations,
            "warnings": warnings,
            "sanitized_project": project_name.lower().replace(" ", "-"),
            "total_tags": len(tags),
            "metadata_key_count": len(metadata),
            "status": "APPROVED_FOR_PRODUCTION" if is_compliant else "REJECTED_HYGIENE_VIOLATION"
        }
        self.validation_records.append(audit_result)
        return audit_result

# Initialize engine in session state
if "tracing_taxonomy_engine" not in st.session_state:
    st.session_state.tracing_taxonomy_engine = TracingTaxonomyEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("📐 Chapter 35.1: Tracing Best Practices in LangSmith")
st.caption("Granular Span Scoping, Environment-Partitioned Projects, Low-Cardinality Tagging, Structured Metadata, and Version Auditing")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Tracing Taxonomy and Best Practices Architecture", divider="rainbow")

st.markdown("""
In enterprise LLMOps, tracing is not merely a debugging toggle; it is the fundamental telemetry backbone powering 
system observability, cost accounting, security auditing, and continuous improvement. Without strict taxonomies and standards, 
LangSmith instances rapidly devolve into chaotic trace dumps filled with anonymous runs, high-cardinality tag explosions, 
and unsearchable errors.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph EnterpriseTracingArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_hierarchy {
            label = "1. Enterprise Project Namespace Partitioning";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            P1 [label="prod-customer-service-rag\\n(Zero Noise / Strict Retention 90d)", fillcolor="#1E3A8A", color="#3B82F6"];
            P2 [label="staging-customer-service-rag\\n(Pre-Release Canary / Retention 30d)", fillcolor="#1E3A8A", color="#3B82F6"];
            P3 [label="dev-team-sandbox\\n(Local Debugging / Retention 7d)", fillcolor="#334155", color="#64748B"];
        }

        subgraph cluster_spans {
            label = "2. Granular Operation Span Hierarchy";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            S_ROOT [label="Root Run: User Workflow\\nname='customer_inquiry_agent'\\ntype='chain'", fillcolor="#047857", color="#10B981"];
            S_RET [label="Child Span: Hybrid Search\\nname='hybrid_vector_retriever'\\ntype='retriever'", fillcolor="#0E7490", color="#06B6D4"];
            S_LLM [label="Child Span: Model Synthesis\\nname='gpt_4o_chat_completion'\\ntype='llm'", fillcolor="#0E7490", color="#06B6D4"];
            S_TOOL [label="Child Span: Account Balance\\nname='sql_crm_tool'\\ntype='tool'", fillcolor="#0E7490", color="#06B6D4"];
        }

        subgraph cluster_context {
            label = "3. Telemetry Metadata & Tagging Contract";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            TAGS [label="Low-Cardinality Tags (Filter Facets)\\n['channel:web', 'tier:enterprise', 'geo:us-east']", fillcolor="#7C2D12", color="#EA580C"];
            META [label="Structured High-Cardinality Metadata\\n{'tenant_id': 'cust_982', 'session_id': 'sess_b4', 'git_sha': '8f12a'}", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_downstream {
            label = "4. Downstream Observability & Triage";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            OUT_FILTER [label="Sub-Second Faceted Filtering\\n(Query by Tenant, Model, Error)", fillcolor="#065F46", color="#059669"];
            OUT_BILL [label="Accurate Cost Allocation\\n(Bill-Back by Customer Tenant)", fillcolor="#065F46", color="#059669"];
            OUT_AUDIT [label="Version Lineage Auditing\\n(Correlate Incident to Commit)", fillcolor="#065F46", color="#059669"];
        }

        P1 -> S_ROOT [label="Scopes Execution"];
        S_ROOT -> S_RET [label="Dispatches Step"];
        S_ROOT -> S_LLM [label="Synthesizes Answer"];
        S_ROOT -> S_TOOL [label="Executes Tool"];

        S_ROOT -> TAGS [label="Attaches Tags"];
        S_ROOT -> META [label="Attaches Metadata"];

        TAGS -> OUT_FILTER [label="Enables UI Facets"];
        META -> OUT_BILL [label="Groups Cost by Tenant"];
        META -> OUT_AUDIT [label="Traces Git SHA"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Tracing Hygiene
1. **Never Let Local Development Traces Pollute Production Projects**: Mixing test scripts with production traffic ruins latency percentiles, corrupts token usage metrics, and causes false alarms. Partition strictly by environment (`prod-`, `staging-`, `dev-`).
2. **Never Put High-Cardinality Identifiers in Tags**: Tags power dashboard dropdown filters. Putting `user_id` or `trace_id` in tags creates millions of unique filter options, rendering UI filtering unusable. Put unique IDs in `metadata`; reserve `tags` for low-cardinality facets (e.g., `channel:slack`, `tier:enterprise`).
3. **Always Bind Version Lineage at the Root Run**: Every trace must be tagged with the application version (`app_version`) and prompt commit hash (`prompt_commit`) to enable instant correlation between software deployments and quality shifts.
""")

st.divider()

# ==============================================================================
# SECTION 2: TRACE IMPORTANT OPERATIONS
# ==============================================================================
st.header("2. Trace Important Operations (Granular Span Scoping, Tool Nodes and Filtering Noise)", divider="rainbow")

st.markdown("""
A common tracing failure is either tracing too little (only seeing the final LLM string) or tracing too much 
(wrapping every trivial internal string manipulation in a LangSmith span, creating massive overhead and noisy trees). 
Production tracing focuses on the boundaries where latency, cost, and errors occur.
""")

st.markdown("""
### What to Trace vs What to Omit
* **Always Trace (Crucial Spans)**:
  * **LLM Calls**: Captures prompt tokens, completion tokens, latency, model parameters, and raw JSON output.
  * **Vector Database Retrievals**: Captures top-k queried documents, cosine similarity scores, and search latency.
  * **External Tool & API Invocations**: Captures tool inputs, external API execution times, and HTTP failure codes.
  * **Guardrail Checks**: Captures input classification scores, PII redaction actions, and safety tripwires.
* **Never Trace as Separate Spans (Noisy Operations)**:
  * In-memory regex parsing or dictionary lookups taking < 0.1ms.
  * Internal loop iterations of string formatting.
  * Health check pings and heartbeat endpoints.
""")

st.code('''
from langsmith import traceable

# Best Practice: Granular span scoping with proper run types
@traceable(run_type="retriever", name="qdrant_hybrid_retriever")
def retrieve_relevant_docs(query: str, top_k: int = 5):
    # Vector DB network call
    return client.search(collection="kb", query=query, limit=top_k)

@traceable(run_type="tool", name="crm_customer_lookup")
def execute_crm_tool(customer_id: str):
    # External API HTTP call
    return crm_api.get_customer(customer_id)

@traceable(run_type="chain", name="customer_support_pipeline")
def run_support_pipeline(user_query: str, customer_id: str):
    # Root workflow orchestrating children
    docs = retrieve_relevant_docs(user_query)
    cust_info = execute_crm_tool(customer_id)
    # Synthesize with model
    return synthesize_response(user_query, docs, cust_info)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
By using `@traceable(run_type="...")`, developers assign semantic meanings to spans. 
In the LangSmith waterfall view, this renders dedicated icons for retrievers, tools, and models, allowing on-call engineers 
to immediately identify whether a 3.5s latency spike was caused by the database retriever (2.8s) or the LLM call (0.7s).
""")

st.divider()

# ==============================================================================
# SECTION 3: USE MEANINGFUL PROJECT NAMES
# ==============================================================================
st.header("3. Use Meaningful Project Names (Environment Partitioning: prod vs staging vs dev and Domain Topologies)", divider="rainbow")

st.markdown("""
By default, unconfigured LangSmith SDK clients send all traces to a project named `default`. 
In enterprise environments, the `default` project quickly becomes an unmanageable dumping ground where developer 
laptop tests mingle with live paying customer requests.
""")

st.markdown("""
### Enterprise Project Naming Taxonomy
Adopt a structured three-tier naming convention: `<environment>-<service-name>-<subsystem>`.
* `prod-banking-customer-rag`: Production customer-facing RAG traffic.
* `staging-banking-customer-rag`: Pre-production candidate verification and QA testing.
* `dev-banking-sandbox`: Developer experimentation and feature branch work.
* `eval-banking-benchmark-nightly`: Automated continuous evaluation runs.
""")

naming_table = pd.DataFrame([
    {"Project Name Example": "prod-ecom-recommender", "Environment": "Production", "Retention Policy": "90 Days", "Sampling Rate": "100% Errors / 10% Success", "Purpose": "Live revenue-generating customer traffic"},
    {"Project Name Example": "staging-ecom-recommender", "Environment": "Staging", "Retention Policy": "30 Days", "Sampling Rate": "100% All Spans", "Purpose": "Pre-deployment canary and QA sign-off"},
    {"Project Name Example": "eval-ecom-nightly-benchmark", "Environment": "CI / Evaluation", "Retention Policy": "60 Days", "Sampling Rate": "100% All Spans", "Purpose": "Scheduled Golden Dataset regression tracking"},
    {"Project Name Example": "dev-alice-agent-sandbox", "Environment": "Development", "Retention Policy": "7 Days", "Sampling Rate": "100% All Spans", "Purpose": "Local laptop debugging and prototype experiments"}
])
st.dataframe(naming_table, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production deployment manifests set the project name dynamically via environment variables in Kubernetes pods:
`LANGCHAIN_PROJECT=prod-customer-service-rag`. This ensures code remains environment-agnostic while traces automatically 
route to the appropriate LangSmith project container based on deployment manifests.
""")

st.divider()

# ==============================================================================
# SECTION 4: USE USEFUL METADATA
# ==============================================================================
st.header("4. Use Useful Metadata (Tenant Hashes, Session IDs, Feature Flags and User Personas)", divider="rainbow")

st.markdown("""
Metadata in LangSmith represents structured, queryable key-value pairs attached to a run. Unlike tags (which are flat strings), 
metadata accepts nested JSON structures, integers, floats, and booleans. High-cardinality values that are unique to a 
request or user must always live in metadata.
""")

st.markdown("""
### Core Metadata Taxonomy
* **Multi-Tenant Attribution**: `{"tenant_id": "org_710a", "subscription_tier": "enterprise"}`. Crucial for calculating per-tenant token usage and charging back costs.
* **Session & Conversation State**: `{"session_id": "sess_8b21", "conversation_turn": 3}`. Enables reconstructing full multi-turn dialogs across disparate traces.
* **Feature Flags & Routing**: `{"experiment_variant": "prompt_v2", "use_reranker": True}`. Enables grouping traces by feature flag during A/B tests.
* **Infrastructure Context**: `{"region": "us-east-1", "pod_name": "rag-worker-6b9f"}`. Enables debugging localized cloud infrastructure latency spikes.
""")

st.code('''
from langsmith import traceable

@traceable(
    name="customer_rag_chat",
    # Metadata contains structured, high-cardinality queryable fields
    metadata={
        "environment": "production",
        "tenant_id": "cust_acme_corp_819",
        "session_id": "sess_f812-4091-ba92",
        "user_tier": "enterprise_sla",
        "feature_flag_rag_v2": True,
        "region": "us-east-1",
        "app_commit": "7a91bf2"
    }
)
def handle_customer_request(query: str):
    return execute_pipeline(query)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In multi-tenant SaaS platforms, finance teams use LangSmith metadata filters (`metadata.tenant_id = "org_91"`) 
to export exact monthly token consumption per enterprise client, automating customer billing and gross margin auditing.
""")

st.divider()

# ==============================================================================
# SECTION 5: USE MEANINGFUL TAGS
# ==============================================================================
st.header("5. Use Meaningful Tags (Categorical Filtering, A/B Test Buckets and High-Cardinality Avoidance)", divider="rainbow")

st.markdown("""
Tags in LangSmith are indexed as low-cardinality categorical facets. They populate the filter dropdowns in the LangSmith UI. 
If used correctly, tags allow engineers to filter 100,000 production traces down to specific slices in two clicks.
""")

st.markdown("""
### Tagging Best Practices vs Anti-Patterns
* **Correct Use (Low Cardinality, Categorical)**:
  * `channel:slack`, `channel:web`, `channel:mobile`
  * `language:en`, `language:es`, `language:de`
  * `tier:enterprise`, `tier:free`
  * `variant:control`, `variant:challenger`
* **Severe Anti-Pattern (High Cardinality)**:
  * `tag:user_12948194` (Breaks UI filter dropdowns with 100,000 unique values!)
  * `tag:session_7b9a1024` (Must be in metadata!)
  * `tag:timestamp_171928401` (Redundant; runs already store timestamps!)
""")

tag_rules = pd.DataFrame([
    {"Rule": "Low Cardinality (< 50 unique values)", "Correct Example": "tags=['channel:slack', 'geo:eu']", "Violating Example": "tags=['user:alice_9410']", "Impact": "Preserves UI performance and clean filter menus"},
    {"Rule": "Namespace with Colons", "Correct Example": "tags=['model:gpt-4o', 'intent:billing']", "Violating Example": "tags=['billing'] (ambiguous)", "Impact": "Prevents collision between intent, model, and channel"},
    {"Rule": "Bake A/B Test Buckets into Tags", "Correct Example": "tags=['ab:canary_prompt_v2']", "Violating Example": "Hidden in deep JSON only", "Impact": "Enables one-click UI comparison of test cohorts"}
])
st.dataframe(tag_rules, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Engineers create saved views in LangSmith pinned to specific tag combinations (e.g., `tags CONTAINS "channel:slack" AND tags CONTAINS "status:error"`). 
This creates instant on-call dashboards for monitoring specific integration channels.
""")

st.divider()

# ==============================================================================
# SECTION 6: TRACK APPLICATION VERSIONS
# ==============================================================================
st.header("6. Track Application Versions (Git SHAs, SemVer Tags, Prompt Hub Commit Pointers and Release Lineage)", divider="rainbow")

st.markdown("""
LLM applications evolve rapidly across code, prompts, and models. If an application begins hallucinating on a Wednesday afternoon, 
observability must immediately clarify which deployment caused the regression. 
Tracking application versions across every trace establishes unbroken lineage.
""")

st.markdown("""
### The Four Coordinates of Application Versioning
1. **Application Binary Version (`app_version`)**: SemVer release tag (e.g., `v2.4.1`) or container image SHA.
2. **Git Commit SHA (`git_commit`)**: Exact 7-character commit hash representing the executed source code (`git-sha-7e81b2a`).
3. **Prompt Hub Commit (`prompt_commit`)**: Exact commit hash of the prompt pulled from LangSmith Prompt Hub.
4. **Environment Release Pointer (`release_pointer`)**: The active deployment channel (`prod-blue`, `canary-green`).
""")

st.code('''
import os

# Automatically extract version coordinates from runtime environment
APPLICATION_VERSION = os.getenv("APP_VERSION", "v2.4.1")
GIT_COMMIT_SHA = os.getenv("GIT_COMMIT_SHA", "c89f1a2")
PROMPT_VERSION = "prompt-hub-7b0a"

@traceable(
    name="production_rag_agent",
    extra={
        "metadata": {
            "app_version": APPLICATION_VERSION,
            "git_commit": GIT_COMMIT_SHA,
            "prompt_commit": PROMPT_VERSION,
            "deployment_slot": "canary-5-percent"
        }
    },
    tags=[f"version:{APPLICATION_VERSION}"]
)
def run_agent(query: str):
    return agent_executor.invoke({"input": query})
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In CI/CD build scripts, GitHub Actions bakes `APP_VERSION` and `GIT_COMMIT_SHA` into container environment variables. 
When the container initializes, every LangSmith trace automatically inherits these variables, enabling instant 
regression comparison across releases in LangSmith.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Enterprise Tracing Taxonomy and Metadata Validator (TracingTaxonomyEngine)", divider="rainbow")

st.markdown("""
Below is the complete production-grade implementation of the Tracing Taxonomy Engine. 
It validates trace configurations before emission, auditing project naming conventions, checking tag cardinality, 
verifying required metadata schemas, and ensuring version lineage compliance.
""")

st.code('''
class TracingTaxonomyEngine:
    """
    Enterprise Tracing Taxonomy & Metadata Hygiene Auditor for LangSmith.
    Enforces production tracing standards: meaningful project naming conventions,
    bounded low-cardinality tags, structured metadata schemas, operation span scoping,
    and release version lineage tracking.
    """
    def __init__(self):
        self.validation_records: List[Dict[str, Any]] = []

    def audit_trace_configuration(
        self,
        project_name: str,
        tags: List[str],
        metadata: Dict[str, Any],
        operation_type: str,
        app_version: str
    ) -> Dict[str, Any]:
        """
        Audits a proposed run span configuration against enterprise tracing best practices.
        """
        violations = []
        warnings = []

        # 1. Project Naming Convention Check
        valid_env_prefixes = ["prod-", "staging-", "dev-", "eval-"]
        if not any(project_name.startswith(p) for p in valid_env_prefixes):
            violations.append(f"Project '{project_name}' lacks required environment prefix ({', '.join(valid_env_prefixes)}).")
        if " " in project_name or any(c.isupper() for c in project_name):
            violations.append("Project name must be strictly lowercase-kebab-case without spaces.")

        # 2. Tag Cardinality Check (Tags must be categorical and low-cardinality)
        for tag in tags:
            if any(char.isdigit() for char in tag) and len(tag) > 10:
                violations.append(f"Tag '{tag}' looks like a high-cardinality ID (UUID/hash). Move unique identifiers to metadata!")
        if len(tags) > 8:
            warnings.append(f"Span has {len(tags)} tags. Limit tags to <= 8 high-level categorical labels.")

        # 3. Metadata Completeness Check
        required_meta_keys = ["environment", "tenant_id", "session_id"]
        for rk in required_meta_keys:
            if rk not in metadata:
                violations.append(f"Missing mandatory metadata key: '{rk}'.")

        # 4. Version Format Check
        if not (app_version.startswith("v") or len(app_version) == 7):  # SemVer or 7-char Git SHA
            warnings.append(f"Application version '{app_version}' should follow SemVer (v1.2.3) or short Git SHA format.")

        # 5. Operation Scope Check
        valid_operations = ["retrieval", "llm_call", "tool_execution", "agent_node", "guardrail_check"]
        if operation_type not in valid_operations:
            warnings.append(f"Operation type '{operation_type}' is unconventional. Recommended: {valid_operations}.")

        is_compliant = len(violations) == 0

        audit_result = {
            "is_compliant": is_compliant,
            "violations": violations,
            "warnings": warnings,
            "sanitized_project": project_name.lower().replace(" ", "-"),
            "total_tags": len(tags),
            "metadata_key_count": len(metadata),
            "status": "APPROVED_FOR_PRODUCTION" if is_compliant else "REJECTED_HYGIENE_VIOLATION"
        }
        return audit_result
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Tracing Taxonomy Validator and Metadata Hygiene Inspector", divider="rainbow")

st.markdown("""
Test how tracing configurations are audited against enterprise best practices. 
Experiment with project names, tag cardinality violations, metadata schemas, and version coordinates to verify compliance.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Span Configuration Inputs")
    proj_preset = st.selectbox(
        "Project Name Preset",
        ["prod-customer-service-rag", "default", "My Experimental Project", "dev-team-sandbox", "staging-rag-v2"],
        index=0
    )
    op_type_select = st.selectbox(
        "Operation Span Type",
        ["retrieval", "llm_call", "tool_execution", "agent_node", "misc_string_split"],
        index=0
    )
    tag_mode = st.radio(
        "Tagging Strategy Preset",
        ["Best Practice (Low-Cardinality Facets)", "Anti-Pattern (UUID High-Cardinality Explosion)"],
        index=0
    )
    include_metadata_toggle = st.toggle(
        "Attach Mandatory Metadata (Tenant, Session, Env)",
        value=True,
        help="Controls whether mandatory enterprise metadata keys are bound to the trace."
    )
    ver_input = st.selectbox(
        "Release Version String",
        ["v2.4.1", "7e81b2a", "latest", "unversioned-local"],
        index=0
    )

with wb_col2:
    st.markdown("### Live Tracing Taxonomy Audit Report")

    # Construct test payloads
    if tag_mode.startswith("Best"):
        test_tags = ["channel:web", "tier:enterprise", "geo:us-east", "env:prod"]
    else:
        test_tags = ["channel:web", "user_uuid_941a80b721894a", "sess_token_8829471928410a", "req_991823"]

    test_meta = {}
    if include_metadata_toggle:
        test_meta = {
            "environment": "production",
            "tenant_id": "cust_acme_corp_819",
            "session_id": "sess_8109-ba21",
            "git_sha": "7e81b2a",
            "prompt_commit": "hub:support-v1.9"
        }
    else:
        test_meta = {"misc_note": "quick test"}

    audit = st.session_state.tracing_taxonomy_engine.audit_trace_configuration(
        project_name=proj_preset,
        tags=test_tags,
        metadata=test_meta,
        operation_type=op_type_select,
        app_version=ver_input
    )

    if audit["is_compliant"]:
        st.success(f"**TAXONOMY STATUS: {audit['status']}** — Span satisfies all enterprise tracing guidelines.")
    else:
        st.error(f"**TAXONOMY STATUS: {audit['status']}** — Span rejected due to hygiene violations.")

    # Violations & Warnings Lists
    if audit["violations"]:
        st.markdown("#### Critical Hygiene Violations (Blocking)")
        for v in audit["violations"]:
            st.error(f"• {v}")

    if audit["warnings"]:
        st.markdown("#### Hygiene Warnings (Non-Blocking)")
        for w in audit["warnings"]:
            st.warning(f"• {w}")

    # Configuration Inspection Card
    st.markdown("#### Audited Span Inspection Card")
    insp_df = pd.DataFrame([
        {"Field": "Project Name", "Configured Value": proj_preset, "Compliant": "✅" if not any("Project" in v for v in audit["violations"]) else "❌"},
        {"Field": "Tag Count & Hygiene", "Configured Value": f"{len(test_tags)} tags ({', '.join(test_tags[:3])}...)", "Compliant": "✅" if not any("Tag" in v for v in audit["violations"]) else "❌"},
        {"Field": "Mandatory Metadata", "Configured Value": f"{len(test_meta)} keys attached", "Compliant": "✅" if not any("metadata" in v for v in audit["violations"]) else "❌"},
        {"Field": "Operation Type", "Configured Value": op_type_select, "Compliant": "✅" if not any("Operation" in w for w in audit["warnings"]) else "⚠️"},
        {"Field": "Version Lineage", "Configured Value": ver_input, "Compliant": "✅" if not any("version" in w for w in audit["warnings"]) else "⚠️"}
    ])
    st.dataframe(insp_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

t_exp = st.expander("View 5 Comprehensive Production Interview Q&As", expanded=True)

with t_exp:
    st.markdown("""
    **Q1: What is the primary operational difference between LangSmith Tags and LangSmith Metadata?**  
    *Answer:* Tags are indexed as flat, low-cardinality categorical strings designed for UI filter dropdowns and fast facet exploration (e.g., `channel:slack`, `tier:free`). Metadata is a structured key-value JSON schema designed for high-cardinality queryable data, cost accounting, and deep diagnostics (e.g., `user_id`, `tenant_id`, `session_id`, `git_commit`). Putting high-cardinality UUIDs into tags destroys UI filter dropdown usability.
    """)

    st.markdown("""
    **Q2: Why is using the 'default' LangSmith project strictly prohibited in enterprise production?**  
    *Answer:* The `default` project becomes a dumping ground where local developer experiments, CI runs, and production traffic collide. This pollutes latency p95 percentiles, invalidates token cost tracking, triggers false alerting, and prevents granular access control (RBAC). Production architectures require strict environment-prefixed project namespaces (e.g., `prod-service-rag`, `staging-service-rag`).
    """)

    st.markdown("""
    **Q3: How do you prevent tracing overhead from inflating latency in high-throughput production services?**  
    *Answer:* Tracing spans should only wrap meaningful boundaries (LLMs, vector retrievals, external tools, guardrails) rather than trivial internal loops. In Python, the LangSmith SDK dispatches traces asynchronously in background worker threads, minimizing main-thread blocking. For extreme high-throughput services, configure deterministic sampling (e.g., log 100% of errors and 10% of successful runs).
    """)

    st.markdown("""
    **Q4: How should multi-tenant SaaS applications structure metadata for cost attribution?**  
    *Answer:* Every root span must include a mandatory `tenant_id` and `subscription_tier` metadata key. Downstream, LangSmith filters and API exports group token counts and dollar costs by `metadata.tenant_id`, allowing finance teams to calculate accurate customer unit economics and charge-backs.
    """)

    st.markdown("""
    **Q5: Why is tracking application and prompt versions on every trace essential for incident response?**  
    *Answer:* When response quality degrades or error rates spike, version tracking allows engineers to immediately determine whether the issue is correlated with a specific software commit (`git_commit`), prompt update (`prompt_commit`), or third-party model update. Without version lineage, identifying the root cause of regressions requires manual code archaeology.
    """)

st.markdown("### Production Traps and Anti-Patterns")

tr1, tr2 = st.columns(2)

with tr1:
    st.error("""
    **Trap 1: High-Cardinality Tag Explosion**
    * **Failure**: Tagging spans with user UUIDs or request IDs (`tags=[f"req_{request_id}"]`).
    * **Consequence**: Generates hundreds of thousands of unique tags in LangSmith, crashing the tag filter UI and making categorical search impossible.
    * **Remedy**: Always place unique request, session, and user identifiers in `metadata`, reserving `tags` strictly for bounded categories (< 50 unique values).
    """)

    st.error("""
    **Trap 2: Tracing Internal Trivial String Operations**
    * **Failure**: Placing `@traceable` on small regex helpers or dictionary lookup functions.
    * **Consequence**: Clutters trace waterfalls with hundreds of 0.05ms spans, bloating ingestion payloads and hiding actual model and retrieval bottlenecks.
    * **Remedy**: Restrict `@traceable` to significant network boundaries: LLMs, retrievers, tools, and top-level workflow chains.
    """)

with tr2:
    st.warning("""
    **Trap 3: Anonymous Production Traces**
    * **Failure**: Emitting production traces without `tenant_id`, `session_id`, or `app_version`.
    * **Consequence**: When a user reports a bug, support engineers cannot locate the user's trace, making debugging nearly impossible.
    * **Remedy**: Enforce an enterprise middleware that injects mandatory context keys into `metadata` on every incoming request.
    """)

    st.warning("""
    **Trap 4: Unpartitioned Staging and Production Traces**
    * **Failure**: Pointing both staging and production deployments to the same LangSmith project name.
    * **Consequence**: Synthetic stress-testing in staging corrupts production SLA metrics and skews real customer usage dashboards.
    * **Remedy**: Enforce strict environment prefixes: `prod-<app>` vs `staging-<app>`.
    """)

st.markdown("### Core Technical Flashcards")

fc1, fc2 = st.columns(2)

with fc1:
    with st.container(border=True):
        st.markdown("**Front**: When should an identifier be placed in `metadata` versus `tags`?")
        st.markdown("**Back**: High-cardinality unique IDs (user_id, session_id, request_id) go in `metadata`; low-cardinality categorical facets (channel, tier, language) go in `tags`.")

    with st.container(border=True):
        st.markdown("**Front**: What are the four recommended environment prefixes for LangSmith project names?")
        st.markdown("**Back**: `prod-`, `staging-`, `dev-`, and `eval-` (e.g., `prod-customer-service-rag`).")

with fc2:
    with st.container(border=True):
        st.markdown("**Front**: What are the four semantic run types assigned via `@traceable(run_type=...)`?")
        st.markdown("**Back**: `llm` (model inference), `retriever` (vector DB search), `tool` (external API/function execution), and `chain` (workflow orchestration).")

    with st.container(border=True):
        st.markdown("**Front**: Why must every production trace bind `app_version` and `git_commit`?")
        st.markdown("**Back**: To enable instant correlation between quality shifts or error spikes and specific code/prompt deployments during post-mortems.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Tracing Best Practices in LangSmith:")

tbp_chk1 = st.checkbox("I know how to partition LangSmith projects using environment prefixes (prod-, staging-, dev-).", value=True)
tbp_chk2 = st.checkbox("I understand the strict boundary between low-cardinality tags and high-cardinality metadata.", value=True)
tbp_chk3 = st.checkbox("I know which operations to trace (LLMs, retrievers, tools) and which internal helpers to omit.", value=True)
tbp_chk4 = st.checkbox("I can structure multi-tenant metadata schemas for accurate client cost allocation.", value=True)
tbp_chk5 = st.checkbox("I understand how to bind application SemVer and Git commit SHAs into every root trace.", value=True)
tbp_chk6 = st.checkbox("I can implement an automated taxonomy auditor to reject non-compliant span telemetry.", value=True)

if tbp_chk1 and tbp_chk2 and tbp_chk3 and tbp_chk4 and tbp_chk5 and tbp_chk6:
    st.success("Mastery Certified: You have mastered enterprise Tracing Best Practices, metadata hygiene, and observability taxonomy in LangSmith.")
