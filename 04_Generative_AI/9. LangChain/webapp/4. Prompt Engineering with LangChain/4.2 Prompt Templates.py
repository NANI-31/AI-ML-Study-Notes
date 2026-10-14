import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="4.2 Prompt Templates - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("4.2 Prompt Templates: Dynamic Interpolation, Partial Binding & Serialization")
st.caption("Chapter 4: Prompt Engineering with LangChain | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In production applications, prompts cannot remain hardcoded string literals. They must be parametrized, 
modular blueprints capable of safely binding dynamic user parameters, database records, and runtime timestamps.

**`PromptTemplate`** is LangChain's fundamental string template primitive. Built on top of Python's `Runnable` 
architecture, `PromptTemplate` provides automated variable extraction, strict schema validation, 
support for multiple templating engines (f-string and Jinja2), partial variable pre-binding, 
and seamless composition into downstream LCEL pipelines.

In this lesson, you will master:
- **`PromptTemplate` Fundamentals**: Initializing templates from raw strings and enforcing strict schemas.
- **Input Variables & Invariants**: Inspecting extracted variables and catching missing inputs before model execution.
- **Variable Injection Techniques**: Handling multiple dynamic parameters and conditional fallbacks.
- **Partial Variable Pre-Binding**: Freezing static or runtime parameters (e.g., timestamps) via `.partial()`.
- **Template Reusability & Serialization**: Saving and loading templates across microservice boundaries.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockPromptTemplateEngine:
    """Simulates PromptTemplate formatting, partial variable binding, and error validation in module scope."""
    def __init__(self, raw_template: str, partial_vars: Dict[str, Any]):
        self.raw_template = raw_template
        self.partial_vars = partial_vars

    def extract_variables(self) -> List[str]:
        import re
        found = re.findall(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', self.raw_template)
        return list(dict.fromkeys(found))

    def format_template(self, runtime_vars: Dict[str, Any]) -> Dict[str, Any]:
        all_vars = {**self.partial_vars, **runtime_vars}
        required_vars = self.extract_variables()
        missing = [v for v in required_vars if v not in all_vars or str(all_vars[v]).strip() == ""]
        
        if missing:
            return {
                "success": False,
                "error": f"KeyError: Missing required input variables: {missing}",
                "formatted_text": "",
                "variables_bound": all_vars
            }

        try:
            formatted = self.raw_template.format(**all_vars)
            return {
                "success": True,
                "error": None,
                "formatted_text": formatted,
                "variables_bound": all_vars
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Formatting Error: {str(e)}",
                "formatted_text": "",
                "variables_bound": all_vars
            }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: PromptTemplate Lifecycle & String Interpolation Engine", divider="rainbow")

st.markdown("""
The diagram below illustrates the complete lifecycle of a `PromptTemplate`: from template compilation 
and partial variable pre-binding to runtime variable injection, schema validation, and `PromptValue` emission.
""")

dot_blueprint = """
digraph PromptTemplateLifecycle {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_compile {
        label="1. Template Definition & Compilation Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_raw [label="Raw Template String\n'Audit {target} for compliance with {standard}'", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_factory [label="PromptTemplate.from_template()\nAuto-extracts input_variables=['target', 'standard']", fillcolor="#1E293B", fontcolor="#38BDF8"];

        node_raw -> node_factory [label="compiles into", color="#38BDF8"];
    }

    subgraph cluster_partial {
        label="2. Partial Variable Pre-Binding Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_partial [label="template.partial(standard='SOC2-Type-II')\nPre-binds static environment config", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_lambda [label="Callable Lambdas (e.g., datetime.utcnow)\nEvaluated lazily per invocation", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_factory -> node_partial [label=".partial()", color="#38BDF8"];
        node_partial -> node_lambda [style="dashed", color="#94A3B8"];
    }

    subgraph cluster_runtime {
        label="3. Runtime Validation & Invocation Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_runtime_vars [label="Runtime Input Dictionary\n{'target': 'production-kubernetes-cluster'}", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_validate [label="Variable Invariant Check\nEnsures all non-partial variables present", fillcolor="#1E293B", fontcolor="#F59E0B"];
        node_pvalue [label="StringPromptValue / Formatted String\nEmitted to ChatModel or LLM in LCEL", fillcolor="#1E293B", fontcolor="#34D399"];

        node_partial -> node_validate [label="invoked with", color="#38BDF8"];
        node_runtime_vars -> node_validate;
        node_validate -> node_pvalue [label="valid interpolation", color="#34D399"];
    }
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: `PromptTemplate` is an LCEL `Runnable`. When invoked via `.invoke(dict)`, 
> it outputs a `PromptValue` (specifically a `StringPromptValue`). This object implements `.to_string()` 
> and `.to_messages()`, making it seamlessly compatible with both legacy text completion models and modern Chat Models.
""")

# --- SECTION 2: PROMPTTEMPLATE ---
st.header("2. PromptTemplate: String Templates, Format Styles & Schema Validation", divider="rainbow")

st.markdown("""
The simplest and most common way to create a template is using `PromptTemplate.from_template()`. 
By default, LangChain uses Python's standard `f-string` syntax (matching `{variable}` braces).
""")

col_pt1, col_pt2 = st.columns(2)
with col_pt1:
    st.markdown("""
    #### Supported Formatting Engines
    - **`f-string` (Default)**: Fast, native Python syntax. Braces must be escaped by doubling (`{{` and `}}`) if literal JSON or curly braces are needed.
    - **`jinja2`**: Advanced templating supporting loops (`{% for x in list %}`), conditionals (`{% if %}`), and filters. Requires `pip install jinja2`.
    """)
with col_pt2:
    st.code('''from langchain_core.prompts import PromptTemplate

# Standard f-string template instantiation
template = PromptTemplate.from_template(
    "Analyze microservice {service_name} deployed in region {aws_region}."
)

print(template.input_variables)  # ['service_name', 'aws_region']
formatted_str = template.format(service_name="payment-gateway", aws_region="eu-central-1")
print(formatted_str)
''', language="python")

# --- SECTION 3: INPUT VARIABLES ---
st.header("3. Input Variables: Extraction, Type Checking & input_variables Invariants", divider="rainbow")

st.markdown("""
When a `PromptTemplate` is instantiated, LangChain automatically parses the template string 
and populates the `input_variables` attribute.

If you invoke a template and fail to provide any of the declared `input_variables`, LangChain raises 
a `KeyError` before calling the model API. This fail-fast validation saves network latency and prevents invalid API requests.
""")

st.code('''from langchain_core.prompts import PromptTemplate

# Explicit input_variables declaration (strict schema validation)
strict_template = PromptTemplate(
    template="Audit {hostname} under policy {policy_id}.",
    input_variables=["hostname", "policy_id"]
)

# Calling with missing variable raises KeyError:
# strict_template.format(hostname="db-primary.internal")  # KeyError: 'policy_id'
''', language="python")

# --- SECTION 4: TEMPLATE FORMATTING ---
st.header("4. Template Formatting: Synchronous .format(), .invoke() & PromptValue Normalization", divider="rainbow")

st.markdown("""
In LangChain, you can format a template in two ways:
1. **`.format(**kwargs) -> str`**: Legacy helper returning a raw Python string.
2. **`.invoke(dict) -> PromptValue`**: Modern LCEL Runnable protocol returning a polymorphic `PromptValue`.
""")

st.code('''from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("Translate from {source_lang} to {target_lang}: {text}")

# 1. Legacy format method
raw_string = template.format(source_lang="German", target_lang="English", text="Guten Tag")

# 2. Modern LCEL Runnable invoke method
prompt_value = template.invoke({
    "source_lang": "German",
    "target_lang": "English",
    "text": "Guten Tag"
})

print(type(prompt_value))       # <class 'langchain_core.prompt_values.StringPromptValue'>
print(prompt_value.to_string()) # 'Translate from German to English: Guten Tag'
print(prompt_value.to_messages()) # [HumanMessage(content='Translate from German to English: Guten Tag')]
''', language="python")

# --- SECTION 5: MULTIPLE VARIABLES ---
st.header("5. Multiple Variables: Multi-Parameter Injection & Hierarchical Context Merging", divider="rainbow")

st.markdown("""
Enterprise prompts frequently integrate 4 to 8 variables across different tiers of the system 
(user identity, tenant ID, database rows, security flags, and the raw query).
""")

st.code('''from langchain_core.prompts import PromptTemplate

multi_var_template = PromptTemplate.from_template("""
SYSTEM POLICY: {compliance_framework}
TENANT: {tenant_id} (Tier: {tier_level})
AUTHORIZATION: {user_role}

AUDIT TARGET: {resource_arn}
QUERY: {audit_query}
""")

payload = {
    "compliance_framework": "FedRAMP High",
    "tenant_id": "tenant-corp-42",
    "tier_level": "Enterprise Platinum",
    "user_role": "SecOps Lead",
    "resource_arn": "arn:aws:iam::123456789012:role/ProductionAdmin",
    "audit_query": "Verify privilege escalation pathways."
}

audit_text = multi_var_template.format(**payload)
''', language="python")

# --- SECTION 6: OPTIONAL VARIABLES ---
st.header("6. Optional Variables: Default Fallbacks, Conditional Rendering & Graceful Null Handling", divider="rainbow")

st.markdown("""
Standard Python f-strings do not natively support optional variables; if a placeholder `{notes}` is in the string, 
it must be provided. In LangChain, handling optional variables is achieved through:

1. **Jinja2 Templating**: Using Jinja's `{% if variable %}` conditional blocks.
2. **Partial Variables with Defaults**: Pre-binding sensible default fallback values.
""")

st.code('''from langchain_core.prompts import PromptTemplate

# Approach: Jinja2 conditional template
jinja_template = PromptTemplate.from_template(
    """Analyze database host: {{ db_host }}
{% if custom_port %}Port: {{ custom_port }}{% else %}Port: 5432 (Default){% endif %}
{% if remarks %}Remarks: {{ remarks }}{% endif %}""",
    template_format="jinja2"
)

# Rendering with custom_port omitted
print(jinja_template.format(db_host="postgres-cluster.internal"))
''', language="python")

# --- SECTION 7: PARTIAL VARIABLES ---
st.header("7. Partial Variables: Static Pre-Binding, Runtime Lambdas & Dynamic Timestamps", divider="rainbow")

st.markdown("""
A **Partial Template** is a `PromptTemplate` where a subset of the required variables have already been bound. 
This is useful when certain parameters are known at application startup (such as corporate policy or environment) 
while others arrive at runtime.

LangChain supports two types of partial variables:
1. **Static Values**: Pre-bound strings or dictionaries.
2. **Callable Functions**: Dynamic lambdas or functions evaluated freshly each time the template is formatted.
""")

dot_partial_flow = """
digraph PartialBindingFlow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_base [label="Base Template\n'Date: {current_time} | Host: {hostname} | Query: {query}'", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_part [label="Partial Pre-Binding (.partial)\n- current_time = lambda: datetime.now().isoformat()\n- hostname = 'prod-us-east-1'", fillcolor="#1E293B", fontcolor="#F59E0B"];
    node_exec [label="Runtime Execution (.invoke)\nOnly 'query' must be provided by user", fillcolor="#1E293B", fontcolor="#34D399"];

    node_base -> node_part [label="static & lambda binding", color="#38BDF8"];
    node_part -> node_exec [label="simplified input signature", color="#34D399"];
}
"""

with st.expander("🔍 Contextual Diagram: Static & Callable Partial Variable Binding", expanded=True):
    st.graphviz_chart(dot_partial_flow, use_container_width=True)

st.code('''from langchain_core.prompts import PromptTemplate
from datetime import datetime

# Function evaluated dynamically on every format call
def get_current_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

base_template = PromptTemplate.from_template(
    "Report Generated: {timestamp}\\nEnvironment: {env}\\nAudit Target: {target}"
)

# Pre-binding static environment and dynamic timestamp function
partial_template = base_template.partial(
    env="Production-VPC",
    timestamp=get_current_timestamp
)

# Now only 'target' needs to be supplied at runtime!
result = partial_template.format(target="Elasticsearch-Cluster-01")
print(result)
''', language="python")

# --- SECTION 8: REUSABLE PROMPTS ---
st.header("8. Reusable Prompts: Modular Serialization, JSON/YAML Saving & Hub Loading", divider="rainbow")

st.markdown("""
Hardcoding prompt strings inside Python source files violates separation of concerns. 
Prompt text should be versioned, audited, and loaded as configuration artifacts.

LangChain supports **serializing prompts to JSON and YAML files**, as well as loading public and private 
templates from the **LangChain Hub** (`langchain hub pull`).
""")

st.code('''from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("Summarize {document_text} in exactly {bullet_count} bullets.")

# Serializing prompt to local JSON/YAML configuration file
# template.save("prompts/summarization_v1.json")

# Loading serialized prompt in production microservice
# from langchain_core.prompts import load_prompt
# loaded_template = load_prompt("prompts/summarization_v1.json")

# Pulling standardized prompts from LangChain Hub
# from langchain import hub
# hub_prompt = hub.pull("rlm/rag-prompt")
''', language="python")

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Enterprise Modular Template Engine with Partial Binding", divider="rainbow")

st.markdown("""
Below is an enterprise-grade prompt management module demonstrating strict schema definitions, 
dynamic callable partial binding, and safe LCEL execution.
""")

st.code('''from typing import Dict, Any, Callable
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

class EnterprisePromptRegistry:
    """Manages versioned, pre-bound prompt templates for production pipelines."""

    @staticmethod
    def get_security_audit_template(environment_name: str) -> PromptTemplate:
        raw_text = """### AUDIT TELEMETRY ###
TIMESTAMP: {timestamp}
ENVIRONMENT: {environment}
CLUSTER: {cluster_id}

### INCIDENT LOG ###
{incident_log}

TASK: Identify root cause and propose zero-downtime remediation."""

        # Base template definition
        base = PromptTemplate.from_template(raw_text)

        # Pre-bind environment statically and timestamp dynamically
        def dynamic_timestamp() -> str:
            return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        bound_template = base.partial(
            environment=environment_name,
            timestamp=dynamic_timestamp
        )
        return bound_template

# Production Assembly Pipeline
def build_audit_chain(api_key: str, env_name: str) -> RunnableSequence:
    prompt_template = EnterprisePromptRegistry.get_security_audit_template(env_name)
    model = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
    
    # Declarative LCEL Pipeline
    return prompt_template | model
''', language="python")

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Interactive Template Compiler & Partial Variable Playground", divider="rainbow")

st.markdown("""
Test dynamic prompt interpolation and partial variable binding below. 
Observe how pre-bound partial variables simplify the runtime input contract and how missing variables 
are flagged instantly by validation checks.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    template_str = st.text_area(
        "Template Blueprint String",
        value="System: {company} Cloud Audit\nHost: {hostname}\nRegion: {region}\nTarget Service: {service}\nAudit Directive: {directive}"
    )
    partial_company = st.text_input("Pre-Bound Partial 'company'", value="Acme Enterprise Security")
    partial_region = st.text_input("Pre-Bound Partial 'region'", value="us-west-2")

with col_wb2:
    st.subheader("Runtime Inputs (Caller Supplied)")
    runtime_hostname = st.text_input("Runtime 'hostname'", value="k8s-control-plane-01")
    runtime_service = st.text_input("Runtime 'service'", value="Vault Secrets Manager")
    runtime_directive = st.text_area("Runtime 'directive'", value="Verify mutual TLS certificates expiration dates.")

if st.button("🚀 Compile & Format PromptTemplate", type="primary"):
    partial_dict = {
        "company": partial_company,
        "region": partial_region
    }
    runtime_dict = {
        "hostname": runtime_hostname,
        "service": runtime_service,
        "directive": runtime_directive
    }

    compiler = MockPromptTemplateEngine(template_str, partial_dict)
    extracted = compiler.extract_variables()
    res = compiler.format_template(runtime_dict)

    st.subheader("📊 Compilation Telemetry")
    m1, m2, m3 = st.columns(3)
    m1.metric("Variables Declared", len(extracted))
    m2.metric("Pre-Bound (Partial)", len(partial_dict))
    m3.metric("Required at Runtime", len(extracted) - len(partial_dict))

    if res["success"]:
        st.success("✅ Prompt compilation succeeded without variable errors!")
        st.write("### Formatted Output Buffer")
        st.code(res["formatted_text"], language="markdown")
    else:
        st.error(f"❌ {res['error']}")

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the primary difference between PromptTemplate.format() and PromptTemplate.invoke()?",
     ".format() is a synchronous string helper returning a raw Python str. .invoke() conforms to the unified LCEL Runnable interface, accepting a dictionary and returning a PromptValue. A PromptValue can be cast to a string via to_string() or a list of messages via to_messages(), allowing seamless composition with both legacy LLMs and ChatModels."),
    ("How does PromptTemplate.partial() improve system design in multi-tenant applications?",
     "partial() pre-binds static or shared configuration (tenant ID, company name, security guidelines) or callable functions (timestamps, UUIDs) at pipeline initialization. Downstream API route handlers only need to supply the remaining runtime variables (e.g., user query), simplifying API contracts and preventing parameter duplication."),
    ("Why must curly braces be doubled in PromptTemplates containing JSON schemas?",
     "Because LangChain's default f-string formatter treats single curly braces {field} as variable placeholders. If your prompt contains literal JSON (e.g., {'status': 200}), Python's string parser will attempt to substitute 'status' as an input variable and crash with a KeyError. Escaping as {{'status': 200}} preserves literal braces."),
    ("How does Jinja2 templating in LangChain handle advanced formatting requirements?",
     "By passing template_format='jinja2', PromptTemplate leverages Jinja2's templating engine. This enables programmatic logic within prompts, such as iterating over lists with for-loops ({% for item in items %}), evaluating boolean conditionals ({% if flag %}), and applying filters, which standard f-strings cannot perform."),
    ("What are the advantages of serializing prompt templates to JSON/YAML in production?",
     "Serializing prompts to configuration files decouples prompt engineering from Python codebase deployments. Prompts can be updated, reviewed by domain experts, version-controlled in Git, or fetched dynamically from prompt registries (like LangChain Hub) without requiring service redeployment.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Unescaped JSON Braces in f-string Templates. Placing `{'key': '{val}'}` in a template string causes Python to treat `'key'` as an input variable, raising an immediate `KeyError`. Always double non-variable braces: `{{'key': '{val}'}}`.")
    st.error("**Trap 2 (Breaking Error)**: Mutable Static Default Dictionaries in Partial. Passing a mutable dictionary or object reference to `.partial()` across threads can cause race conditions where one thread alters defaults seen by others. Always pass immutable strings or callable lambdas.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Callable Lambdas with Expensive Database Calls. Passing a lambda to `.partial()` that performs a blocking, un-cached SQL query will execute on every single prompt format, degrading latency. Cache dynamic parameters.")
    st.warning("**Trap 4 (Quality Warning)**: Jinja2 Dependency Bloat. Using `template_format='jinja2'` requires installing `jinja2`. If deploying in minimal Docker scratch containers without Jinja2 installed, application startup will crash with `ImportError`.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: PromptTemplate Invocation"):
        st.markdown("**template.invoke(dict) returns a PromptValue conforming to the Runnable protocol.**")
    with st.expander("🎴 Flashcard 2: Partial Variable Binding"):
        st.markdown("**template.partial(key=val) pre-binds static values or callable functions.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Curly Brace Escaping"):
        st.markdown("**Double curly braces {{ and }} to include literal JSON brackets in f-string templates.**")
    with st.expander("🎴 Flashcard 4: Jinja2 Support"):
        st.markdown("**PromptTemplate.from_template(..., template_format='jinja2') enables loops and conditionals.**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can construct PromptTemplates using PromptTemplate.from_template().", value=True)
chk2 = st.checkbox("I understand the difference between .format() and .invoke().", value=True)
chk3 = st.checkbox("I know how to double-escape literal curly braces in f-string templates.", value=True)
chk4 = st.checkbox("I can pre-bind static and dynamic callable parameters using .partial().", value=True)
chk5 = st.checkbox("I understand how to serialize and deserialize prompt templates to JSON/YAML.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 4.2 Prompt Templates. Ready for 4.3 Chat Prompt Templates!")
