import streamlit as st
import json

st.set_page_config(
    page_title="28.2 Production Best Practices - Prompt Engineering",
    page_icon="🛡️",
    layout="wide"
)

# Header Section
st.title("28.2 Production Best Practices: CI/CD, Observability, & Safety")
st.caption("Chapter 28: Prompt Engineering Best Practices | Domain: 1. Prompt Engineering")
st.markdown("""
Moving from prototype to mission-critical enterprise production demands rigorous engineering disciplines: 
**semantic versioning of prompt templates in Git, automated CI/CD regression test suites, runtime Pydantic validation, 
PII masking before network dispatch, distributed dead-letter logging, and graceful degradation fail-safes**.
""")

st.divider()

# Section 1: The Production Lifecycle Flywheel
st.header("1. The Enterprise Prompt Engineering Lifecycle", divider="rainbow")
st.markdown("""
Production prompts follow a continuous, automated software development lifecycle:
""")

st.graphviz_chart('''
digraph ProductionLifecycle {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    
    Ver [label="1. Version Control\\n(Git repo / YAML registry\\nv1.2.0 SemVer)", fillcolor="#BBDEFB"];
    Test [label="2. CI/CD Testing\\n(Promptfoo / DeepEval\\nPasses 100 Golden Evals)", fillcolor="#C8E6C9"];
    Deploy [label="3. Canary Rollout\\n(10% traffic to v1.2.0,\\n90% on v1.1.0)", fillcolor="#FFF9C4"];
    Guard [label="4. Runtime Guards\\n(PII Masking & Pydantic\\nSchema Verification)", fillcolor="#E1BEE7"];
    Monitor [label="5. Observability\\n(Langfuse Tracing, Latency,\\nDead-Letter Logging)", fillcolor="#FFCDD2"];
    
    Ver -> Test -> Deploy -> Guard -> Monitor;
    Monitor -> Ver [label="Iterate on Failure Traces", style=dashed];
}
''', use_container_width=False)

st.divider()

# Section 2: Deep Dive into the 7 Production Pillars
st.header("2. The Seven Pillars of Enterprise Production", divider="rainbow")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Semantic Prompt Versioning")
    st.markdown("""
    * Store prompt templates in versioned YAML or JSON files alongside application source code.
    * Use Semantic Versioning (`vMAJOR.MINOR.PATCH`):
      * **PATCH** (`v1.0.1`): Minor wording tweaks, fixing typos.
      * **MINOR** (`v1.1.0`): Adding few-shot exemplars or optional fields.
      * **MAJOR** (`v2.0.0`): Changing output schema or migrating underlying model families.
    """)

    st.subheader("2. Automated CI/CD Regression Tests")
    st.markdown("""
    * Never push a prompt change directly to production.
    * Run automated test suites (e.g. Promptfoo or pytest + DeepEval) against golden datasets in GitHub Actions.
    * Require 98%+ pass rates on safety and format compliance before merging pull requests.
    """)

    st.subheader("3. Runtime Response Validation")
    st.markdown("""
    * Enforce strict Pydantic parsing at runtime.
    * If a response fails schema validation, catch the exception and trigger an automated retry with the error trace, or safely route to human review.
    """)

with col2:
    st.subheader("4. PII Masking & Data Sanitization")
    st.markdown("""
    * Pre-process all user inputs through a local regex or lightweight NER model (e.g. Presidio) to redact Social Security Numbers, credit cards, and customer names before dispatching to external APIs.
    """)

    st.subheader("5. Failure Logging & Dead-Letter Queues (DLQ)")
    st.markdown("""
    * When an LLM call fails, hallucinated, or throws a 500 error, serialize the entire request envelope (prompt, system version, input payload, error trace) to a Dead-Letter Queue (DLQ) in Kafka/S3 for developer inspection.
    """)

    st.subheader("6. Graceful Edge-Case Degradation")
    st.markdown("""
    * Define deterministic fallbacks for timeout scenarios, empty vector retrievals, and unparseable responses.
    """)

st.divider()

# Section 3: Code Implementation: Secure Production Wrapper
st.header("3. Secure Production Execution Wrapper Pattern", divider="rainbow")
st.markdown("""
Below is an enterprise-ready Python wrapper combining PII redaction, versioning, Pydantic validation, and dead-letter logging:
""")

st.code('''# Enterprise Production Execution Wrapper Pattern
import re
import json
from pydantic import BaseModel, ValidationError

PROMPT_VERSION = "v2.1.0"

class SupportTicketResponse(BaseModel):
    category: str
    priority: int
    resolution_steps: list[str]

def redact_pii(text: str) -> str:
    # Redact credit cards and emails
    text = re.sub(r'\\b(?:\\d[ -]*?){13,16}\\b', '[REDACTED_CC]', text)
    text = re.sub(r'[\\w\\.-]+@[\\w\\.-]+\\.\\w+', '[REDACTED_EMAIL]', text)
    return text

def execute_production_pipeline(raw_user_input: str, llm_client, dlq_logger) -> SupportTicketResponse:
    # 1. Redact sensitive user data
    sanitized_input = redact_pii(raw_user_input)
    
    # 2. Compile versioned prompt
    prompt = f"<system_version>{PROMPT_VERSION}</system_version>\\n<input>{sanitized_input}</input>"
    
    # 3. Call LLM with timeout
    try:
        raw_output = llm_client.generate(prompt, timeout=10.0)
        parsed_data = json.loads(raw_output)
        validated_obj = SupportTicketResponse(**parsed_data)
        return validated_obj
    except (ValidationError, json.JSONDecodeError, Exception) as err:
        # 4. Log failure payload to Dead-Letter Queue (DLQ)
        dlq_logger.log({
            "version": PROMPT_VERSION,
            "sanitized_input": sanitized_input,
            "error": str(err),
            "timestamp": "2026-10-12T12:00:00Z"
        })
        # 5. Return deterministic safe fallback
        return SupportTicketResponse(
            category="MANUAL_REVIEW",
            priority=1,
            resolution_steps=["Escalate to human tier-2 support."]
        )''', language="python")

st.divider()

# Section 4: Interactive Production Readiness Audit
st.header("4. Interactive Production Readiness Audit", divider="rainbow")
st.markdown("Assess whether your prompt deployment meets enterprise production standards.")

b1 = st.checkbox("Prompts stored in versioned YAML/Git files rather than string literals in Python code", value=True)
b2 = st.checkbox("CI/CD pipeline runs automated evaluations (Promptfoo/pytest) on every PR", value=True)
b3 = st.checkbox("Runtime output wrapped in strict Pydantic schema validation contracts", value=True)
b4 = st.checkbox("PII masking active on user inputs prior to sending across external APIs", value=False)
b5 = st.checkbox("Dead-Letter Queue (DLQ) captures failed inferences for root-cause analysis", value=False)
b6 = st.checkbox("Hard token budgeting limits configured per user and per API call", value=True)

prod_score = sum([b1, b2, b3, b4, b5, b6])

m_col1, m_col2 = st.columns(2)
with m_col1:
    st.metric("Production Readiness Rating", f"{prod_score} / 6")
with m_col2:
    if prod_score >= 5:
        st.success("🚀 Enterprise Ready: Hardened architecture capable of mission-critical scale.")
    elif prod_score >= 3:
        st.info("⚙️ Staging Grade: Address PII masking and dead-letter queues before launching to production.")
    else:
        st.warning("⚠️ Prototype Grade: High risk of silent regressions and data leaks. Implement versioning and CI/CD tests.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa1, qa2 = st.columns(2)

with qa1:
    with st.expander("Q: How do you design an effective prompt regression testing pipeline in CI/CD?"):
        st.write("""
        1. **Curate Golden Dataset**: Maintain 50-200 representative test cases representing happy paths, tricky edge cases, and known historical bugs.
        2. **Define Deterministic & LLM-as-a-Judge Assertions**: Assert that JSON schemas parse 100%, key regexes match, and semantic faithfulness exceeds 0.85.
        3. **Run on PR Hooks**: Trigger evaluation runs via CLI tools (Promptfoo or DeepEval) whenever a prompt YAML file changes in a pull request.
        4. **Block Merges on Regressions**: Set hard gating thresholds; if pass rate drops below baseline (e.g. 96%), block the merge.
        """)

    with st.expander("Q: What is a Dead-Letter Queue (DLQ) in generative AI pipelines?"):
        st.write("""
        A Dead-Letter Queue is an asynchronous buffer (like an SQS queue or Kafka topic) where failed, 
        malformed, or timed-out inference requests are stored along with their complete context (system prompt version, 
        user input, and stack trace). It prevents failed requests from vanishing into thin air, 
        allowing engineers to replay failures locally and add them to regression test suites.
        """)

with qa2:
    with st.expander("Trap to Avoid: Deploying Prompts Without Rollback Capabilities"):
        st.write("""
        Treating prompts like configuration that can be live-edited in a web console without versioning 
        is catastrophic. If a prompt edit causes an unforeseen regression, there is no quick rollback. 
        Always couple prompt deployments with Git commits and canary traffic routing (e.g. 10% traffic to new prompt version).
        """)

    with st.expander("Flashcards: Production Best Practices"):
        fc_sel = st.selectbox(
            "Select Practice Flashcard:",
            ["Semantic Versioning (SemVer)", "Dead-Letter Queue (DLQ)", "PII Redaction", "Golden Evaluation Suite"]
        )
        if fc_sel == "Semantic Versioning (SemVer)":
            st.info("Tracking prompt template changes using MAJOR.MINOR.PATCH to manage breaking schema updates.")
        elif fc_sel == "Dead-Letter Queue (DLQ)":
            st.info("An isolated logging queue capturing failed inference payloads for developer debugging.")
        elif fc_sel == "PII Redaction":
            st.info("Sanitizing sensitive customer data (SSNs, emails) before sending payloads to external cloud LLMs.")
        else:
            st.info("A curated collection of verified test cases used in CI/CD to prevent prompt performance regressions.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I know how to implement semantic versioning for prompt templates.")
    st.checkbox("I understand how to build CI/CD regression test pipelines with Promptfoo/DeepEval.")
with c2:
    st.checkbox("I can implement runtime PII sanitization and Dead-Letter Queue logging.")
    st.checkbox("I can design deterministic graceful degradation fallbacks for production failures.")
