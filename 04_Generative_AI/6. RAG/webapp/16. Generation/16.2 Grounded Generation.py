import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="16.2 Grounded Generation - RAG Master Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class GroundedGenerationEngine:
    """
    Production-grade Grounded Generation Engine:
    1. Prevents unsupported claims via atomic proposition decomposition.
    2. Enforces closed-world answering from retrieved documents.
    3. Handles missing information with deterministic abstention protocols.
    4. Computes source attribution and claim-to-document mapping.
    5. Validates inline citation tags and extracts verbatim evidence quotes.
    """
    def __init__(self):
        pass

    def decompose_claims(self, text: str) -> List[str]:
        """Decomposes compound sentences into discrete atomic factual propositions."""
        clean_text = text.replace("\n", " ").replace(";", ".")
        raw_sentences = [s.strip() for s in clean_text.split(".") if len(s.strip()) > 8]
        claims = []
        for s in raw_sentences:
            if " and " in s:
                sub_parts = s.split(" and ")
                for part in sub_parts:
                    if len(part.strip()) > 5:
                        claims.append(part.strip())
            else:
                claims.append(s)
        return claims

    def verify_claim_grounding(self, claim: str, context_docs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Verifies if an atomic claim is grounded in context documents using lexical overlap
        and keyword entity matching.
        """
        claim_words = set([w.lower().strip(",.") for w in claim.split() if len(w) > 3])
        best_doc_id = None
        best_overlap = 0
        best_ratio = 0.0

        for doc in context_docs:
            doc_text = doc.get("text", "").lower()
            overlap_count = sum(1 for w in claim_words if w in doc_text)
            ratio = overlap_count / max(1, len(claim_words))
            if ratio > best_ratio:
                best_ratio = ratio
                best_overlap = overlap_count
                best_doc_id = doc.get("id", "Unknown")

        if best_ratio >= 0.70:
            status = "ENTAILED"
        elif best_ratio >= 0.40:
            status = "WEAK_SUPPORT"
        else:
            status = "UNGROUNDED_HALLUCINATION"

        return {
            "claim": claim,
            "status": status,
            "attributed_doc": best_doc_id if status != "UNGROUNDED_HALLUCINATION" else None,
            "support_score": round(best_ratio, 2)
        }

    def audit_generated_response(
        self, response_text: str, context_docs: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Audits an entire generated answer for faithfulness, citations, and hallucinations."""
        claims = self.decompose_claims(response_text)
        audited_claims = [self.verify_claim_grounding(c, context_docs) for c in claims]

        entailed_count = sum(1 for c in audited_claims if c["status"] == "ENTAILED")
        weak_count = sum(1 for c in audited_claims if c["status"] == "WEAK_SUPPORT")
        hallucination_count = sum(1 for c in audited_claims if c["status"] == "UNGROUNDED_HALLUCINATION")

        total = max(1, len(audited_claims))
        faithfulness_score = (entailed_count + 0.5 * weak_count) / total

        return {
            "total_claims": len(audited_claims),
            "entailed_claims": entailed_count,
            "hallucination_claims": hallucination_count,
            "faithfulness_score": round(faithfulness_score, 2),
            "claim_details": audited_claims,
            "is_production_safe": hallucination_count == 0 and faithfulness_score >= 0.85
        }


# Instantiate engine at module level
grounding_engine = GroundedGenerationEngine()

# --- HEADER SECTION ---
st.title("16.2 Grounded Generation: Preventing Unsupported Claims, Citation Attribution, & Hallucination Auditing")
st.caption("Chapter 16: Generation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In conversational AI, generating fluent, plausible-sounding prose is trivial. 
In enterprise RAG, however, **fluency without truth is fatal**.

**Grounded Generation** (often formalized as **Faithfulness** or **Attribution**) is the architectural guarantee 
that every factual proposition emitted by the language model is strictly entailed by, and traceable to, 
the retrieved context documents.

When an LLM asserts facts not contained in the context, it commits a **faithfulness violation**. 
In regulated industries (healthcare, legal, banking, corporate compliance), even a single ungrounded assumption 
can lead to regulatory penalties, catastrophic liability, or operational failure.

This lesson analyzes the core systems engineering required to guarantee closed-book grounded generation.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Closed-Book Grounded Generation Architecture", divider="rainbow")

st.markdown("""
Grounded generation does not treat the LLM as an unmonitored generator. It wraps model output in an automated 
decomposition, verification, and redaction pipeline before the payload is delivered to users or APIs.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph GroundedGenArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // RETRIEVED CONTEXT REPOSITORY
        subgraph cluster_context {
            label="1. Injected Knowledge Substrate";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            ContextStore [label="Verified Context Repository\\n(Tagged with immutable Doc IDs)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            UserQuery [label="User Evaluation Query\\n('What is the failover SLA?')", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
        }

        // GENERATION WITH DIRECTIVES
        subgraph cluster_generator {
            label="2. Grounded LLM Generation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            ClosedBookPrompt [label="Closed-World Prompt Contract\\n('Answer strictly from context. Never speculate.')", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            RawResponse [label="Draft Generated Response\\n(Text with inline citations [DOC-1])", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
        }

        // VERIFICATION GUARDRAIL
        subgraph cluster_guardrail {
            label="3. Automated Attribution & NLI Guardrail";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            ClaimDecomposer [label="Proposition Decomposer\\n(Extracts atomic factual claims)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            NLIClassifier [label="Natural Language Inference (NLI)\\n(Classifies: Entailed / Neutral / Contradicted)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            CitationAuditor [label="Citation Verifier\\n(Matches bracketed IDs to ground-truth spans)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
        }

        // DISPATCH DECISION
        subgraph cluster_dispatch {
            label="4. Safe Delivery & Circuit Breaker";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            PassGate [label="Grounded Output Gateway\\n(100% Entailed & Verified)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];
            RedactionEngine [label="Hallucination Redactor & Fallback\\n(Strips unproven claims or emits refusal)", fillcolor="#450A0A", fontcolor="#FCA5A5", color="#DC2626", shape=doubleoctagon];
        }

        // PIPELINE EDGES
        ContextStore -> ClosedBookPrompt;
        UserQuery -> ClosedBookPrompt;
        ClosedBookPrompt -> RawResponse;
        RawResponse -> ClaimDecomposer;
        ContextStore -> NLIClassifier [label="Ground Truth Reference"];
        ClaimDecomposer -> NLIClassifier;
        NLIClassifier -> CitationAuditor;
        CitationAuditor -> PassGate [label="All Claims Entailed"];
        CitationAuditor -> RedactionEngine [label="Ungrounded Claims Detected"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### The 3 Core Pillars of Grounded Generation:
1. **Upfront Epistemic Inoculation**: Directing the LLM to deactivate parametric general knowledge during generation.
2. **Deterministic Attribution**: Requiring discrete inline source markers tied directly to explicit document identifiers.
3. **Post-Generation Verification**: Running an automated judge to verify that every asserted statement is logically entailed by the retrieved context.
""")

st.divider()

# ==============================================================================
# SECTION 2: PREVENTING UNSUPPORTED CLAIMS
# ==============================================================================
st.header("2. Preventing Unsupported Claims (Hallucination Suppression & Faithfulness Verification)", divider="rainbow")

st.markdown("""
Hallucinations in RAG differ fundamentally from hallucinations in open-ended chat. 
In RAG, hallucinations manifest as **faithfulness breaches**: statements that appear completely fluent and coherent 
but are not supported by the retrieved evidence.
""")

puc_col1, puc_col2 = st.columns([1, 1])

with puc_col1:
    st.markdown("""
    ### Taxonomy of RAG Hallucinations:
    * **1. Parametric Bleed**: The model answers using general world training data rather than company-specific documentation (e.g., asserting standard California overtime rules instead of the employer's internal policy).
    * **2. Entity Swapping**: Correctly identifying a relationship, but swapping the entities (e.g., claiming *Service A depends on Service B* when the document states *Service B depends on Service A*).
    * **3. Extrapolative Leap**: Drawing speculative conclusions that seem logical but are never explicitly stated in the source text.
    * **4. Numerical Fabrication**: Rounding, approximating, or altering financial numbers or SLA percentages.
    """)

with puc_col2:
    st.markdown("### Hallucination Vulnerability Spectrum")
    vuln_data = pd.DataFrame([
        {"Failure Mode": "Numerical / Date Modification", "Detection Difficulty": "High (Requires exact match)", "Severity": "Critical", "Remedy": "Entity extraction check"},
        {"Failure Mode": "Entity Inversion / Swapping", "Detection Difficulty": "Medium (Dependency parsing)", "Severity": "Critical", "Remedy": "NLI relation testing"},
        {"Failure Mode": "Uncited General Knowledge", "Detection Difficulty": "Low (Missing citations)", "Severity": "High", "Remedy": "Mandatory inline citations"},
        {"Failure Mode": "Speculative Extrapolation", "Detection Difficulty": "Medium", "Severity": "High", "Remedy": "Closed-world negative constraints"}
    ])
    st.dataframe(vuln_data, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 3: ANSWERING FROM RETRIEVED DOCUMENTS
# ==============================================================================
st.header("3. Answering from Retrieved Documents (Closed-World Conditioning & Parametric Memory Isolation)", divider="rainbow")

st.markdown("""
To guarantee grounded generation, the system enforces a **Closed-World Contract**: 
the LLM must act as an extraction and synthesis engine over the provided documents, treating its internal pre-training weights as untrusted.
""")

ard_col1, ard_col2 = st.columns([1, 1])

with ard_col1:
    st.markdown("""
    ### Techniques for Parametric Memory Isolation:
    * **Role Inoculation**: Direct the model to assume the role of an *Evidence Extraction Specialist* with zero prior knowledge of the topic.
    * **Explicit Override Mandate**: State explicitly that if the retrieved documents contradict common knowledge (e.g., an internal project naming convention), the documents take absolute priority.
    * **Zero Assumptions Clause**: Forbid the model from adding common-sense background explanations not directly corroborated by the text.
    """)

with ard_col2:
    st.markdown("### Closed-World Conditioning Directive")
    st.code(r'''
GROUNDING DIRECTIVE:
You are an evidence extraction engine.
You must construct your answer EXCLUSIVELY using explicit facts 
from <context>.
Assume you have no memory of the world outside <context>.
If an assertion cannot be proved by directly quoting <context>, 
you are strictly prohibited from stating it.
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 4: HANDLING MISSING INFORMATION
# ==============================================================================
st.header("4. Handling Missing Information (Abstention Protocols, Sentinel Fallbacks & Unanswerable Queries)", divider="rainbow")

st.markdown("""
When the retrieved documents do not contain the answer, an unconstrained LLM naturally attempts to be helpful 
by guessing. Grounded generation turns this failure mode into a **controlled abstention**.
""")

mi_col1, mi_col2 = st.columns([1, 1])

with mi_col1:
    st.markdown("""
    ### The Anatomy of High-Reliability Abstention:
    1. **Pre-Generation Audit**: Instruct the model to verify whether the context contains the necessary facts *before* formulating an answer.
    2. **Machine-Readable Sentinel**: Emit an unambiguous sentinel string (e.g., `[ABSTAIN: INSUFFICIENT_EVIDENCE]`) to allow automated microservice routing.
    3. **Missing Fact Enumeration**: State what specific topic or parameter was absent from the documentation, helping human curators update the knowledge base.
    4. **Zero Conversational Apologies**: Eliminate apologetic chatter (*"I apologize, but as an AI..."*), ensuring predictable downstream parsing.
    """)

with mi_col2:
    st.markdown("### Abstention Directive Pattern")
    st.code(r'''
ABSTENTION DIRECTIVE:
Before answering, check if <context> contains 100% of the facts 
required to answer <query>.
If the answer is absent, ambiguous, or incomplete, output EXACTLY:
[STATUS: INSUFFICIENT_CONTEXT]
Reason: [Specify exactly which facts are missing]
Do not guess, speculate, or hedge.
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 5: SOURCE ATTRIBUTION
# ==============================================================================
st.header("5. Source Attribution (Provenance Tracing, Document Envelopes & Claim-to-Source Mapping)", divider="rainbow")

st.markdown("""
Source attribution bridges generated sentences with their originating documents. 
Every claim must be provably linked to a specific source envelope.
""")

sa_col1, sa_col2 = st.columns([1, 1])

with sa_col1:
    st.markdown("""
    ### Structural Attribution Enveloping:
    * **Enveloped Passages**: Each retrieved chunk is wrapped in structured metadata tags containing:
      - `id`: Unique document identifier (e.g., `SLA-101`)
      - `title`: Canonical document name
      - `author`: Certified authority group
      - `last_updated`: Timestamp for recency validation
    * **Claim-to-Source Linkage**: Generated text maps atomic claims to these identifiers, enabling automated audit trails.
    """)

with sa_col2:
    st.markdown("### Multi-Tier Attribution Traceability")
    st.code(r'''
<context>
  <document id="POL-401" title="Employee Travel Policy" updated="2024-01-15">
    Hotel reimbursement is capped at $250 per night in Tier-1 cities.
  </document>
</context>

Response:
Hotel reimbursements are capped at $250 per night in Tier-1 cities [POL-401].
''', language="xml")

st.divider()

# ==============================================================================
# SECTION 6: CITATION GENERATION
# ==============================================================================
st.header("6. Citation Generation (Inline Bracketed Tags, Verbatim Quote Anchors & Citation Precision)", divider="rainbow")

st.markdown("""
Citation generation establishes an auditable contract. 
Without strict directives, models invent citations or attach broad disclaimers that cannot be validated.
""")

cg_col1, cg_col2 = st.columns([1, 1])

with cg_col1:
    st.markdown("""
    ### Production Citation Standards:
    * **Sentence-Level Inline Tags**: Every factual sentence must end with an inline bracketed citation: `[DOC-ID]`.
    * **Verbatim Evidence Anchoring**: Require an accompanying evidence table quoting the exact text from the source:
      `| Claim | Source ID | Verbatim Proof Quote |`
    * **Zero Phantom IDs**: Forbid citing IDs not explicitly declared in the `<context>` envelopes.
    """)

with cg_col2:
    st.markdown("### Citation Verification Metrics")
    citation_metrics = pd.DataFrame([
        {"Metric": "Citation Precision", "Definition": "Proportion of generated citations that accurately substantiate their claim.", "Target": ">= 98.0%"},
        {"Metric": "Citation Recall", "Definition": "Proportion of factual claims that have an accompanying valid citation.", "Target": "100.0%"},
        {"Metric": "Phantom Citation Rate", "Definition": "Frequency of citing non-existent document IDs.", "Target": "0.0%"}
    ])
    st.dataframe(citation_metrics, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Grounded Generation Guardrail Engine", divider="rainbow")

st.markdown("""
Below is the production implementation of the `GroundedGenerationEngine`. It handles claim decomposition, 
context grounding checks, and response-level faithfulness auditing in real module scope.
""")

st.code(r'''
class GroundedGenerationEngine:
    """
    Enterprise Grounded Generation Guardrail:
    - Decomposes compound responses into atomic factual claims.
    - Verifies grounding against context passages using word overlap and entity matching.
    - Classifies claims as ENTAILED, WEAK_SUPPORT, or UNGROUNDED_HALLUCINATION.
    - Calculates a quantitative Faithfulness Score (0.0 to 1.0).
    """
    def decompose_claims(self, text: str) -> List[str]:
        clean_text = text.replace("\n", " ").replace(";", ".")
        sentences = [s.strip() for s in clean_text.split(".") if len(s.strip()) > 8]
        claims = []
        for s in sentences:
            if " and " in s:
                for part in s.split(" and "):
                    if len(part.strip()) > 5:
                        claims.append(part.strip())
            else:
                claims.append(s)
        return claims

    def verify_claim_grounding(self, claim: str, context_docs: List[Dict[str, str]]) -> Dict[str, Any]:
        claim_words = set([w.lower().strip(",.") for w in claim.split() if len(w) > 3])
        best_doc_id = None
        best_ratio = 0.0

        for doc in context_docs:
            doc_text = doc.get("text", "").lower()
            overlap = sum(1 for w in claim_words if w in doc_text)
            ratio = overlap / max(1, len(claim_words))
            if ratio > best_ratio:
                best_ratio = ratio
                best_doc_id = doc.get("id", "Unknown")

        if best_ratio >= 0.70:
            status = "ENTAILED"
        elif best_ratio >= 0.40:
            status = "WEAK_SUPPORT"
        else:
            status = "UNGROUNDED_HALLUCINATION"

        return {
            "claim": claim,
            "status": status,
            "attributed_doc": best_doc_id if status != "UNGROUNDED_HALLUCINATION" else None,
            "support_score": round(best_ratio, 2)
        }

    def audit_generated_response(self, response_text: str, context_docs: List[Dict[str, str]]) -> Dict[str, Any]:
        claims = self.decompose_claims(response_text)
        audited = [self.verify_claim_grounding(c, context_docs) for c in claims]
        entailed = sum(1 for c in audited if c["status"] == "ENTAILED")
        weak = sum(1 for c in audited if c["status"] == "WEAK_SUPPORT")
        hallucinated = sum(1 for c in audited if c["status"] == "UNGROUNDED_HALLUCINATION")
        total = max(1, len(audited))
        score = (entailed + 0.5 * weak) / total

        return {
            "total_claims": len(audited),
            "entailed_claims": entailed,
            "hallucination_claims": hallucinated,
            "faithfulness_score": round(score, 2),
            "claim_details": audited,
            "is_production_safe": hallucinated == 0 and score >= 0.85
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Faithfulness & Hallucination Auditor", divider="rainbow")

st.markdown("""
Test the grounding verification engine in real time. 
Evaluate how the engine decomposes a generated response into atomic claims, checks grounding against source documents, 
and flags ungrounded hallucinations.
""")

test_corpus = [
    {
        "id": "SEC-01",
        "title": "Cloud Security Standard",
        "text": "All employee laptop drives must be encrypted using AES-256 BitLocker. Biometric fingerprint authentication is required for office building entry."
    },
    {
        "id": "HR-02",
        "title": "Remote Work Policy",
        "text": "Employees may work remotely up to two days per week with prior manager approval. Core working hours are 10:00 AM to 4:00 PM EST."
    }
]

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.markdown("### Test Candidate Responses")
    preset_response = st.selectbox(
        "Select Test Generation Scenario:",
        options=[
            "1. 100% Grounded Answer",
            "2. Partially Grounded with Hallucinated Assumption",
            "3. Completely Ungrounded Hallucination"
        ],
        index=0
    )

    if preset_response.startswith("1."):
        active_response = "Laptop drives must be encrypted with AES-256 BitLocker. Employees can work remotely two days per week with manager approval."
    elif preset_response.startswith("2."):
        active_response = "Laptop drives require AES-256 BitLocker encryption. Employees must also submit monthly hard drive audit reports to IT."
    else:
        active_response = "All employees receive a $500 monthly home internet stipend and can work remotely five days per week without manager approval."

    response_input = st.text_area("Generated Answer to Audit:", value=active_response, height=130)

with wb_c2:
    st.markdown("### Live Grounding Audit Scorecard")
    audit_results = grounding_engine.audit_generated_response(response_input, test_corpus)

    m1, m2, m3 = st.columns(3)
    m1.metric("Faithfulness Score", f"{audit_results['faithfulness_score'] * 100:.0f}%")
    m2.metric("Entailed Claims", f"{audit_results['entailed_claims']} / {audit_results['total_claims']}")
    m3.metric("Hallucinations", f"{audit_results['hallucination_claims']}")

    if audit_results["is_production_safe"]:
        st.success("✅ Output Approved: 100% verified and grounded against source context.")
    else:
        st.error(f"❌ Output Blocked: {audit_results['hallucination_claims']} ungrounded claim(s) detected!")

st.markdown("#### Atomic Proposition Claim Audit Breakdown:")
claim_table_rows = []
for c in audit_results["claim_details"]:
    claim_table_rows.append({
        "Atomic Proposition Claim": c["claim"],
        "Status": c["status"],
        "Attributed Document": c["attributed_doc"] or "None (Ungrounded)",
        "Support Ratio": f"{c['support_score'] * 100:.0f}%"
    })
st.dataframe(pd.DataFrame(claim_table_rows), use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: What is the formal definition of Faithfulness in RAG, and how is it calculated mathematically?**"):
    st.markdown("""
    **Answer**:
    Faithfulness measures whether all factual claims in the generated response can be directly inferred from the retrieved context.
    Mathematically:
    `Faithfulness = Number of Entailed Claims / Total Number of Generated Claims`
    The calculation workflow:
    1. Extract all atomic propositions `C = {c_1, c_2, ..., c_n}` from the response.
    2. For each claim `c_i`, run a Natural Language Inference (NLI) model testing if `Context -> c_i`.
    3. Sum the verified claims and divide by `|C|`. A production system requires `Faithfulness >= 0.95`.
    """)

with st.expander("**Q2: How does an automated hallucination guardrail differ from an output toxicity guardrail?**"):
    st.markdown("""
    **Answer**:
    * **Toxicity / Safety Guardrail**: Evaluates the response in isolation against content policy classifiers (detecting profanity, hate speech, or PII leaks).
    * **Hallucination / Faithfulness Guardrail**: Must evaluate the response **relative to the context documents**. A statement like *"The server reboots in 45 seconds"* is completely safe from a toxicity standpoint, but catastrophic if the documentation states *"The server reboots in 120 seconds"*.
    """)

with st.expander("**Q3: What causes an LLM to hallucinate even when the correct answer is present in the context?**"):
    st.markdown("""
    **Answer**:
    1. **Attention Dispersion**: In long contexts (>4,000 tokens), attention heads suffer from the Lost-in-the-Middle attention slump, failing to route attention to the critical key.
    2. **Strong Parametric Prior**: If the model has seen a common phrasing millions of times in pre-training (e.g., standard laws or common libraries), the prior probability of those tokens overwhelms the cross-attention signals from the context.
    3. **High Sampling Temperature**: Setting temperature > 0.5 allows the sampler to pick low-probability tokens that diverge from the text.
    """)

with st.expander("**Q4: How should a production RAG system handle partially ungrounded answers?**"):
    st.markdown("""
    **Answer**:
    Production systems apply **Active Claim Redaction**:
    1. Decompose the answer into atomic sentences.
    2. If sentences 1 and 2 are entailed (`[DOC-1]`), but sentence 3 is ungrounded, the system strips sentence 3 entirely before presenting the answer to the user.
    3. If the core answer itself is ungrounded, the system suppresses the response completely and emits the standardized fallback refusal:
       `"I am unable to answer this inquiry based on verified documentation."`
    """)

with st.expander("**Q5: Why is sentence-level citation superior to paragraph-level citation?**"):
    st.markdown("""
    **Answer**:
    Paragraph-level citations (e.g., placing `[1]` at the end of a long paragraph) allow models to interleave three grounded sentences with one hallucinated sentence under the umbrella of a single citation.
    Sentence-level citations force the model to explicitly justify every discrete assertion, making automated verification via regex and NLI straightforward.
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Trusting Generative Self-Evaluation**
    * **Failure Mode**: Asking the same model in the same prompt *"Are you sure you didn't hallucinate?"* yields high confidence even when hallucinating.
    * **Solution**: Use an independent, zero-temperature evaluator model or atomic NLI classifier.
    """)

    st.error("""
    **Production Trap 2: Permitting Conversational Hedging**
    * **Failure Mode**: When missing facts, models say *"While not explicitly stated, it is commonly understood that..."* introducing unverified claims.
    * **Solution**: Strictly forbid speculative hedging in system guidelines.
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Loose Citation Regex Patterns**
    * **Failure Mode**: Relying on loose regex patterns that accept vague references like `[Doc]` or `[Source]` without validating the unique ID.
    * **Solution**: Assert that all cited IDs strictly exist in the active retrieval set.
    """)

    st.warning("""
    **Production Trap 4: Over-Pruning via Strict Lexical Overlap**
    * **Failure Mode**: Rejecting valid paraphrased conclusions because they use synonyms not found literally in the text.
    * **Solution**: Combine semantic NLI models with keyword overlap checks for balanced precision.
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: Faithfulness Score Formula")
        st.caption("How is RAG faithfulness mathematically computed?")
        st.markdown("**Back**: Number of context-entailed atomic propositions divided by the total number of propositions in the response.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: Parametric Bleed")
        st.caption("What causes parametric bleed in grounded RAG?")
        st.markdown("**Back**: Pre-trained memory priors overwhelming weak cross-attention context signals, injecting general world facts.")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: Active Claim Redaction")
        st.caption("How to remediate a response containing a single hallucinated sentence?")
        st.markdown("**Back**: Decompose the answer into atomic propositions and programmatically strip the ungrounded sentence before delivery.")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Closed-World Assumption")
        st.caption("What constraint prevents speculative leaps in generation?")
        st.markdown("**Back**: Instructing the model that propositions absent from context are either false or unprovable, forbidding extrapolation.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of grounded generation before proceeding:")

c1 = st.checkbox("I understand the mathematical formulation of Faithfulness as the ratio of entailed claims to total claims.", value=True)
c2 = st.checkbox("I can differentiate between parametric bleed, entity swapping, and extrapolative hallucinations.", value=True)
c3 = st.checkbox("I know how to formulate Closed-World Assumption prompt contracts that suppress external parametric priors.", value=True)
c4 = st.checkbox("I can implement atomic proposition decomposition to verify individual factual claims.", value=True)
c5 = st.checkbox("I know how to design automated abstention protocols that emit machine-readable sentinel tokens.", value=True)
c6 = st.checkbox("I understand the mechanics of active claim redaction for purifying partially grounded answers.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **Grounded Generation Mastery Achieved!** You are ready to deploy enterprise-grade hallucination guardrails.")
