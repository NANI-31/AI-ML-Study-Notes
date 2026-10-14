import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="15.2 RAG Prompt Patterns - RAG Master Curriculum",
    page_icon="🧩",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class DefensivePromptEngine:
    """
    Production-grade Defensive Prompt Pattern Engine:
    1. Context-grounded prompting (epistemic suppression of parametric knowledge).
    2. Verifiable citation prompting (sentence-level tags with verbatim quote grounding).
    3. Strict 'Answer only from context' negative constraint enforcement.
    4. Deterministic unknown-answer handling and graceful refusal triggers.
    5. Machine-readable structured JSON output contract formatting.
    """
    def __init__(self):
        pass

    def generate_grounded_pattern(self, query: str, context: str) -> str:
        """Constructs an epistemic context-grounded prompt pattern."""
        return (
            "### SYSTEM DIRECTIVE: STRICT CONTEXT GROUNDING\n"
            "You are an epistemic grounding engine. Your task is to answer the user query relying solely on facts "
            "directly stated within the provided context. Treat your pre-trained memory as deactivated for factual claims.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>\n\n"
            "Respond using verified facts from <context> only."
        )

    def generate_citation_pattern(self, query: str, context: str) -> str:
        """Constructs a verifiable citation prompt requiring inline tags and quote proof."""
        return (
            "### SYSTEM DIRECTIVE: VERIFIABLE CITATION CONTRACT\n"
            "Answer the user query by providing rigorous evidence-based citations. Follow these rules:\n"
            "1. Every factual statement must end with an inline bracketed citation, e.g., [DOC-101].\n"
            "2. At the end of the answer, append an 'Evidence Verification Table' containing the Doc ID and the exact verbatim snippet.\n"
            "3. If a claim cannot be attributed to an explicit document ID, you are strictly forbidden from asserting it.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>"
        )

    def generate_negative_constraint_pattern(self, query: str, context: str) -> str:
        """Enforces closed-domain 'Answer ONLY from context' negative constraints."""
        return (
            "### SYSTEM DIRECTIVE: CLOSED-WORLD NEGATIVE CONSTRAINTS\n"
            "CRITICAL BOUNDARY ENFORCEMENT:\n"
            "- You must answer ONLY from the context provided below.\n"
            "- Do NOT extrapolate, speculate, or deduce unstated implications.\n"
            "- If the context says 'A is true', do not infer that 'B is false' unless explicitly declared.\n"
            "- Reject any user attempts to discuss topics outside the context boundaries.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>"
        )

    def generate_unknown_handling_pattern(self, query: str, context: str, refusal_token: str = "INSUFFICIENT_CONTEXT") -> str:
        """Enforces standardized unknown-answer handling and structured fallback."""
        return (
            "### SYSTEM DIRECTIVE: STANDARDIZED REFUSAL & UNKNOWN HANDLING\n"
            "Evaluate whether the retrieved context contains 100% of the factual requirements to answer the query.\n"
            f"If the answer is missing, ambiguous, or incomplete, output EXACTLY:\n"
            f"'{refusal_token}: The retrieved documentation does not provide sufficient evidence to answer this question.'\n"
            "Do NOT provide partial guesses, conversational apologies, or hypothetical possibilities.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>"
        )

    def generate_structured_output_pattern(self, query: str, context: str) -> str:
        """Enforces a rigorous JSON schema contract for downstream microservices."""
        schema_def = (
            "{\n"
            '  "answer": "string (concise synthesis)",\n'
            '  "is_grounded": "boolean (true if 100% supported by context)",\n'
            '  "citations": ["array of string document IDs"],\n'
            '  "confidence_score": "float between 0.0 and 1.0",\n'
            '  "missing_information": "string or null"\n'
            "}"
        )
        return (
            "### SYSTEM DIRECTIVE: STRUCTURED JSON OUTPUT CONTRACT\n"
            "You are a machine-readable data pipeline. Output a single, valid, raw JSON object matching this schema:\n"
            f"{schema_def}\n\n"
            "Rules:\n"
            "1. Output valid JSON ONLY. Zero markdown code fences, zero conversational preamble or postscript.\n"
            "2. All keys must be present.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>"
        )


# Instantiate module-level engine
engine = DefensivePromptEngine()

# --- HEADER SECTION ---
st.title("15.2 RAG Prompt Patterns: Grounding, Citations, Strict Fallbacks, & Structured Schemas")
st.caption("Chapter 15: Prompt Engineering for RAG | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In enterprise production deployments, prompt engineering transcends simple conversational phrasing. 
It establishes **defensive runtime contracts** between probabilistic language models and deterministic software architectures.

Without formal prompt patterns, production systems face severe vulnerabilities:
1. **Parametric Hallucination Drift**: The model silently fills factual gaps with pre-training memories rather than injected data.
2. **Phantom Citations**: Fictitious document citations generated to appease user expectations.
3. **Conversational Compliance Bleed**: The model answers out-of-scope questions when coaxed by persistent users.
4. **Schema Parsing Crashes**: Natural language output contaminating JSON parsing in downstream microservice queues.

This lesson analyzes the **five canonical defensive prompt patterns** deployed across mission-critical RAG pipelines.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Defensive Prompt Patterns & Output Contracts", divider="rainbow")

st.markdown("""
Defensive prompt patterns route the user query and context payload through strict behavioral filters, 
producing audited, structured responses ready for automated downstream consumption.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph DefensivePromptBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // INCOMING ASSETS
        subgraph cluster_inputs {
            label="1. Ingestion of Retrieval Artifacts";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            RawQuery [label="User Natural Query\\n('What is the refund window?')", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            RetrievedContext [label="Enveloped Context Blocks\\n(<document id='DOC-1'>...)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
        }

        // DEFENSIVE PATTERN SELECTION
        subgraph cluster_patterns {
            label="2. Defensive Pattern Orchestration";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            PatternGrounded [label="Pattern 1: Context Grounding\\n(Suppresses parametric memory)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            PatternCitation [label="Pattern 2: Verifiable Citation\\n(Sentence-level tags and quotes)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            PatternNegative [label="Pattern 3: 'Answer Only from Context'\\n(Closed-world negative constraints)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            PatternRefusal [label="Pattern 4: Unknown-Answer Protocol\\n(Deterministic abstention trigger)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            PatternStructured [label="Pattern 5: Structured JSON Output\\n(Pydantic / Typed schema contracts)", fillcolor="#0F172A", fontcolor="#10B981", color="#059669"];
        }

        // LLM GENERATION STAGE
        subgraph cluster_execution {
            label="3. Model Execution & Evaluation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            ModelExecution [label="Inference Generation\\n(Temperature = 0.0, Top_P = 1.0)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            OutputEvaluator [label="Deterministic Guardrail Checker\\n(JSON schema validation and citation audit)", fillcolor="#0F172A", fontcolor="#A855F7", color="#7C3AED"];
        }

        // OUTPUT PATHS
        VerifiedOutput [label="Audited Structured Payload\\n(Consumable by downstream APIs)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];
        RefusalPayload [label="Standardized Refusal Contract\\n(Triggering fallback ticket / human handoff)", fillcolor="#450A0A", fontcolor="#FCA5A5", color="#DC2626", shape=doubleoctagon];

        // FLOW EDGES
        RawQuery -> PatternGrounded;
        RetrievedContext -> PatternGrounded;
        PatternGrounded -> PatternCitation;
        PatternCitation -> PatternNegative;
        PatternNegative -> PatternRefusal;
        PatternRefusal -> PatternStructured;
        PatternStructured -> ModelExecution [label="Compiled Prompt Contract"];
        ModelExecution -> OutputEvaluator;
        OutputEvaluator -> VerifiedOutput [label="Passes Grounding & Schema"];
        OutputEvaluator -> RefusalPayload [label="Missing Context / Insufficient Evidence"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Core Architectural Guarantees:
* **Zero Parametric Leaks**: Pre-trained priors are explicitly demoted below retrieved evidence.
* **Deterministic Traceability**: Every claim maps to an attested document identifier.
* **Predictable Abstention**: Missing facts trigger clean, machine-parsable refusals rather than persuasive fabrications.
* **Type-Safe Contract Enforcement**: Output conforms to programmatic JSON schemas, eliminating string parsing errors.
""")

st.divider()

# ==============================================================================
# SECTION 2: CONTEXT-GROUNDED PROMPTING
# ==============================================================================
st.header("2. Context-Grounded Prompting (Parametric Suppression & Factual Anchoring)", divider="rainbow")

st.markdown("""
By default, foundation models blend information from their multi-billion parameter pre-training weights with 
prompt context. In enterprise RAG, this creates subtle factual contamination (e.g., answering with standard 
California labor laws instead of an employer's proprietary policy).
""")

gp_col1, gp_col2 = st.columns([1, 1])

with gp_col1:
    st.markdown("""
    ### Parametric Memory vs Injected Evidence:
    * **The Phenomenon**: When an LLM detects a query on a topic it studied during pre-training, its internal attention weights compete against the external context tokens.
    * **Prompt Inoculation Technique**:
      1. Explicitly designate pre-trained memory as *deactivated* or *untrusted*.
      2. Direct attention heads to treat `<context>` as the sole repository of truth.
      3. Instruct the model that if an enterprise policy contradicts common world knowledge, the context takes absolute precedence.
    """)

with gp_col2:
    st.markdown("### Epistemic Grounding Directive Template")
    st.code(r'''
GROUNDING DIRECTIVE:
You are an evidence-grounded reasoning engine.
Your operational space is strictly bounded by the information inside <context>.
Assume you have no prior world knowledge regarding the entities, dates, 
or rules discussed. If the context states that gravity pulls upward, 
you must state that gravity pulls upward within this domain.
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 3: CITATION PROMPTING
# ==============================================================================
st.header("3. Citation Prompting (Inline Bracket Tags, Verbatim Evidence Quotes & Traceability)", divider="rainbow")

st.markdown("""
Citations transform unverified prose into an auditable document trail. Citation prompting instructs the model 
to pair each factual assertion with its exact source ID and verbatim evidence quotes.
""")

cite_col1, cite_col2 = st.columns([1, 1])

with cite_col1:
    st.markdown("""
    ### The 3 Levels of Citation Rigor:
    1. **Document-Level Attribution**: Appending `[DOC-1]` at the paragraph end. Weakest rigor; permits loose paraphrasing.
    2. **Sentence-Level Bracketed Tags**: Appending `[DOC-1]` to each discrete sentence. Standard enterprise practice.
    3. **Verbatim Quotation Anchoring**: Requiring the model to quote exact sentence substrings alongside the claim. Highest rigor; eliminates hallucinated interpretations.
    """)

with cite_col2:
    st.markdown("### Production Citation Prompt Directive")
    st.code(r'''
CITATION RULES:
1. Every factual sentence MUST conclude with an inline citation tag, e.g., [DOC-ID].
2. Following your synthesis, construct an 'Evidence Table' in this markdown format:
   | Claim | Source ID | Verbatim Quote from Context |
3. Fabricating document IDs not present in the context is strictly prohibited.
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 4: "ANSWER ONLY FROM CONTEXT" BOUNDARY ENFORCEMENT
# ==============================================================================
st.header("4. Answer Only from Context Boundary Enforcement (Closed-World Contracts)", divider="rainbow")

st.markdown("""
The *"Answer only from context"* directive establishes a **Closed-World Assumption (CWA)**. 
Anything not explicitly asserted in the context is assumed to be either false or unknowable.
""")

cwa_col1, cwa_col2 = st.columns([1, 1])

with cwa_col1:
    st.markdown("""
    ### Why Negative Constraints Are Mandatory:
    * **Conversational Drift**: Users frequently ask follow-ups that drift away from indexed documents (e.g., asking for recipe ideas in a cloud architecture assistant).
    * **Speculative Deduction**: Models naturally attempt to deduce missing steps. For example, if a document states *"Servers require patch 2.4"*, the model might extrapolate that *"Servers running 2.3 will crash"*, which may be factually incorrect.
    * **The Boundary Rule**: Explicit negative directives instruct the model to stop at literal assertions and refuse logical extrapolations.
    """)

with cwa_col2:
    st.markdown("### Closed-World Directive Example")
    st.info("""
    **Production Negative Constraint Directive:**
    * Answer ONLY questions whose direct answers are contained within the text.
    * If a user asks a general question (e.g., *"What is machine learning?"*) and the context covers only database replication, refuse the query immediately.
    * Do NOT attempt to be helpful by offering external knowledge.
    """)

st.divider()

# ==============================================================================
# SECTION 5: UNKNOWN-ANSWER HANDLING
# ==============================================================================
st.header("5. Unknown-Answer Handling (Explicit Fallback Triggers, Zero Speculation & Refusals)", divider="rainbow")

st.markdown("""
When retrieved documents do not contain the answer, standard LLMs tend to generate apologetic, speculative responses. 
Production RAG systems require **deterministic, machine-parsable refusal protocols**.
""")

unk_col1, unk_col2 = st.columns([1, 1])

with unk_col1:
    st.markdown("""
    ### Anatomy of a High-Quality Refusal:
    1. **Deterministic Sentinel Token**: Begins with a recognized token (e.g., `[REFUSAL: INSUFFICIENT_CONTEXT]`) allowing regex triage.
    2. **Concise Explanation**: States what piece of information was missing.
    3. **Actionable Remediation**: Suggests search keywords or alternative documentation repositories.
    4. **Zero Apologies**: Omits *"I apologize, as an AI language model..."*, reducing token consumption.
    """)

with unk_col2:
    st.markdown("### Standard Refusal Template Comparison")
    refusal_df = pd.DataFrame([
        {"Approach": "Default Conversational", "Response Snippet": "'I am sorry, but I do not have access to that info...'", "Downstream Parsability": "Very Poor", "Risk": "May still hallucinate"},
        {"Approach": "Standardized Sentinel", "Response Snippet": "'[STATUS: MISSING_CONTEXT] Uptime SLA details absent.'", "Downstream Parsability": "Excellent (Regex)", "Risk": "Zero"},
        {"Approach": "Triage & Escalation", "Response Snippet": "'Refusal: Missing Auth Docs. Route to IT Helpdesk.'", "Downstream Parsability": "High (Workflow hook)", "Risk": "Zero"}
    ])
    st.dataframe(refusal_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 6: STRUCTURED OUTPUT
# ==============================================================================
st.header("6. Structured Output (JSON Schemas, Pydantic Alignment & Function Calling Envelopes)", divider="rainbow")

st.markdown("""
When RAG powers automated agents, customer support ticketing bots, or data extraction workflows, 
natural language prose is unusable. The prompt must enforce a **strict JSON schema contract**.
""")

json_col1, json_col2 = st.columns([1, 1])

with json_col1:
    st.markdown("""
    ### Prompting for Reliable JSON:
    1. **TypeScript / Pydantic Schema Specification**: Providing the exact type schema inside the prompt yields higher structural compliance than vague descriptions.
    2. **Constraint Directives**: Mandating *"Output raw JSON only. Do NOT enclose in markdown backticks or commentary."*
    3. **Confidence Self-Scoring**: Requiring a numerical `confidence_score` (between 0.0 and 1.0) based purely on context support.
    """)

with json_col2:
    st.markdown("### Target JSON Production Schema")
    st.code(r'''
{
  "answer": "string",
  "is_grounded": true,
  "sources": ["DOC-101", "DOC-102"],
  "confidence_score": 0.95,
  "refusal_reason": null
}
''', language="json")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Defensive RAG Prompt Engine", divider="rainbow")

st.markdown("""
The production `DefensivePromptEngine` formats and compiles all five prompt patterns in real module scope.
""")

st.code(r'''
class DefensivePromptEngine:
    """
    Enterprise Defensive Prompt Pattern Engine:
    - Context-grounded prompting (suppressing parametric pre-training memory).
    - Verifiable citation prompting with sentence-level tags and quotes.
    - 'Answer only from context' closed-world negative constraints.
    - Standardized unknown-answer handling and sentinel token refusals.
    - Typed JSON schema contract generation.
    """
    def generate_grounded_pattern(self, query: str, context: str) -> str:
        return (
            "### SYSTEM DIRECTIVE: STRICT CONTEXT GROUNDING\n"
            "Rely solely on facts stated within <context>. "
            "Deactivate parametric memory for factual claims.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>"
        )

    def generate_citation_pattern(self, query: str, context: str) -> str:
        return (
            "### SYSTEM DIRECTIVE: VERIFIABLE CITATION CONTRACT\n"
            "Rules:\n"
            "1. Append an inline bracketed citation [DOC-ID] to every factual assertion.\n"
            "2. Append an 'Evidence Table' mapping each claim to a verbatim quote.\n"
            "3. Do not assert claims without explicit document citations.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>"
        )

    def generate_negative_constraint_pattern(self, query: str, context: str) -> str:
        return (
            "### SYSTEM DIRECTIVE: CLOSED-WORLD NEGATIVE CONSTRAINTS\n"
            "Rules:\n"
            "- Answer ONLY from the context below.\n"
            "- Do NOT speculate or infer unstated implications.\n"
            "- Reject questions falling outside context boundaries.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>"
        )

    def generate_unknown_handling_pattern(self, query: str, context: str, sentinel: str = "INSUFFICIENT_CONTEXT") -> str:
        return (
            "### SYSTEM DIRECTIVE: STANDARDIZED REFUSAL\n"
            f"If context is insufficient, output EXACTLY:\n"
            f"'{sentinel}: The retrieved documentation does not provide sufficient evidence.'\n"
            "Zero guessing, apologies, or speculations.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>"
        )

    def generate_structured_output_pattern(self, query: str, context: str) -> str:
        schema = '{"answer": "string", "is_grounded": bool, "citations": ["doc_id"], "confidence": float}'
        return (
            "### SYSTEM DIRECTIVE: STRUCTURED JSON CONTRACT\n"
            f"Output a single valid JSON object adhering strictly to:\n{schema}\n"
            "No markdown code fences, no extra text.\n\n"
            f"<context>\n{context.strip()}\n</context>\n\n"
            f"<query>\n{query.strip()}\n</query>"
        )
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Prompt Pattern Simulator & Refusal Benchmark", divider="rainbow")

st.markdown("""
Select and test different defensive prompt patterns against test scenarios. 
Inspect how each pattern constrains the output payload.
""")

sample_context_text = """
<document id="SLA-101" source="Cloud Infrastructure Manual" date="2024-03-01">
High-availability clusters guarantee 99.99% uptime. Automated database failovers complete within 120 seconds.
Planned maintenance occurs only during the third Sunday of each calendar quarter.
</document>
<document id="SEC-202" source="Data Protection Policy" date="2024-02-15">
All backups are encrypted at rest using AES-256-GCM. Off-site replica synchronization occurs hourly.
Retention period for financial audit logs is 7 years.
</document>
"""

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.markdown("### Pattern Configuration")
    selected_pattern = st.selectbox(
        "Choose Defensive Prompt Pattern:",
        options=[
            "1. Strict Context-Grounded Pattern",
            "2. Verifiable Citation Pattern",
            "3. Closed-World Negative Constraint Pattern",
            "4. Standardized Unknown-Answer Pattern",
            "5. Structured JSON Output Pattern"
        ],
        index=0
    )

    test_scenario = st.selectbox(
        "Select Test Query Scenario:",
        options=[
            "Standard In-Scope Query (SLA & Uptime)",
            "Out-of-Scope Query (Employee Vacation Policy)",
            "Speculative Query (Will the database survive an asteroid strike?)"
        ],
        index=0
    )

    if test_scenario == "Standard In-Scope Query (SLA & Uptime)":
        active_query = "What is the guaranteed uptime and how long does automated failover take?"
    elif test_scenario == "Out-of-Scope Query (Employee Vacation Policy)":
        active_query = "How many days of paid vacation do employees receive annually?"
    else:
        active_query = "If an entire AWS data center is destroyed by a natural disaster, how long until servers reboot?"

    user_prompt_input = st.text_input("Active Query:", value=active_query)

with wb_c2:
    st.markdown("### Assembled Defensive Prompt Preview")
    if "1. Strict Context-Grounded" in selected_pattern:
        compiled_output = engine.generate_grounded_pattern(user_prompt_input, sample_context_text)
    elif "2. Verifiable Citation" in selected_pattern:
        compiled_output = engine.generate_citation_pattern(user_prompt_input, sample_context_text)
    elif "3. Closed-World Negative" in selected_pattern:
        compiled_output = engine.generate_negative_constraint_pattern(user_prompt_input, sample_context_text)
    elif "4. Standardized Unknown-Answer" in selected_pattern:
        compiled_output = engine.generate_unknown_handling_pattern(user_prompt_input, sample_context_text)
    else:
        compiled_output = engine.generate_structured_output_pattern(user_prompt_input, sample_context_text)

    st.text_area("Compiled Prompt Template:", value=compiled_output, height=300)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: What is the Closed-World Assumption (CWA) in RAG prompt engineering, and why is it essential?**"):
    st.markdown("""
    **Answer**:
    The Closed-World Assumption dictates that the model treats the injected context as the complete universe of truth. 
    Any proposition not explicitly affirmed within the context is treated as either false or unknown.
    In enterprise systems (legal, healthcare, finance), CWA is essential because permitting open-world reasoning allows 
    the model to extrapolate plausible-sounding but legally catastrophic assumptions that were never verified by enterprise custodians.
    """)

with st.expander("**Q2: How does citation prompting mitigate hallucination risks?**"):
    st.markdown("""
    **Answer**:
    Citation prompting forces the LLM's autoregressive generation process to ground each factual assertion in a specific token sequence. 
    By requiring inline brackets (`[DOC-1]`) and verbatim quote proof, the model's self-attention must align the generated tokens 
    with corresponding key-value vectors in the context block. 
    Furthermore, sentence-level citations make hallucination easily detectable by downstream verification scripts.
    """)

with st.expander("**Q3: What are the engineering tradeoffs between requesting raw JSON vs using provider-native Function Calling / Structured Outputs?**"):
    st.markdown("""
    **Answer**:
    * **Prompted Raw JSON**: Works uniformly across all LLM providers and open-source models, but risks occasional markdown backtick wrapping, missing commas, or truncated braces requiring retry loops.
    * **Provider-Native Structured Outputs (e.g., OpenAI JSON Schema mode)**: Guarantees 100% schema compliance at the tokenizer decoding level by using constrained decoding (grammar masks), but introduces vendor lock-in and slight latency overhead during schema compilation.
    """)

with st.expander("**Q4: How should a production system structure refusal prompts to enable automated downstream escalation?**"):
    st.markdown("""
    **Answer**:
    A production refusal prompt must mandate a **machine-readable sentinel token** (e.g., `[INSUFFICIENT_CONTEXT_ERROR]`). 
    When the gateway receives this sentinel, it bypasses user presentation and triggers automated workflows:
    1. Filing a high-priority ticket in Jira for knowledge base documentation gaps.
    2. Routing the customer inquiry to a human tier-2 support agent.
    3. Triggering an expanded search across secondary fallback corpora.
    """)

with st.expander("**Q5: Why does 'Answer only from context' sometimes fail on frontier models, and how can it be fortified?**"):
    st.markdown("""
    **Answer**:
    Models are heavily RLHF-tuned to be helpful and conversational. When faced with a simple instruction like *"Answer only from context"*, 
    the model's helpfulness objective often overrides the negative constraint if it perceives that answering from general knowledge will please the user.
    To fortify this:
    1. Frame the constraint positively: *"Your sole operational capability is extracting facts from <context>."*
    2. Provide 2 to 3 few-shot examples demonstrating clean refusals on common queries.
    3. Impose explicit penalties: *"Answering outside context constitutes a system compliance violation."*
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Free-Text Markdown Citations**
    * **Failure Mode**: Asking the LLM to *"cite sources"* without an exact format yields inconsistent outputs like *"According to document one..."* or *"(see HR guide)"*, breaking regex scrapers.
    * **Solution**: Standardize on rigid brackets: `[DOC-ID]`.
    """)

    st.error("""
    **Production Trap 2: Politeness in Refusals**
    * **Failure Mode**: Permitting models to output conversational apologies (*"I'm sorry, but as an AI..."*) creates inconsistent text that breaks automated test suites.
    * **Solution**: Enforce exact sentinel tokens for all refusals.
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Permissive JSON Prompts**
    * **Failure Mode**: Asking for JSON without explicit prohibition of markdown backticks causes responses wrapped in ```` ```json ... ``` ````, crashing standard `json.loads()`.
    * **Solution**: Explicitly forbid markdown wrappers and strip backticks programmatically before parsing.
    """)

    st.warning("""
    **Production Trap 4: Over-Constraining Factual Synthesis**
    * **Failure Mode**: Excessive negative constraints can cause models to refuse valid questions whose answers require simple multi-sentence synthesis across the context.
    * **Solution**: Clarify that synthesizing facts across multiple documents in the context is explicitly permitted.
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: Closed-World Assumption (CWA)")
        st.caption("What does CWA enforce in enterprise RAG?")
        st.markdown("**Back**: Treats unstated facts as unknowable, suppressing parametric knowledge and preventing speculative leaps.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: Sentinel Refusal Token")
        st.caption("Why use sentinel tokens in refusal prompts?")
        st.markdown("**Back**: Enables deterministic regex detection for automated triage and ticketing without parsing apologetic prose.")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: Verbatim Quote Anchoring")
        st.caption("How does quote anchoring elevate citation rigor?")
        st.markdown("**Back**: Mandates extracting literal context substrings alongside claims, eliminating hallucinated interpretations.")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Schema Constrained Output")
        st.caption("How to prevent LLM responses from crashing microservices?")
        st.markdown("**Back**: Enforce rigid typed JSON contracts specifying required keys and value types.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of defensive prompt patterns before proceeding:")

c1 = st.checkbox("I understand the mechanics of suppressing parametric memory using context-grounded prompt directives.", value=True)
c2 = st.checkbox("I can implement verifiable citation patterns combining inline bracketed tags with verbatim quote anchors.", value=True)
c3 = st.checkbox("I know how to formulate Closed-World Assumption negative constraints to eliminate speculative inferences.", value=True)
c4 = st.checkbox("I can design deterministic sentinel-based refusal protocols that trigger automated downstream workflows.", value=True)
c5 = st.checkbox("I can construct Pydantic-aligned JSON prompt contracts for reliable microservice integration.", value=True)
c6 = st.checkbox("I understand why conversational politeness in refusals compromises automated evaluation pipelines.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **RAG Prompt Patterns Mastery Achieved!** You are ready to deploy enterprise-grade defensive prompt contracts.")
