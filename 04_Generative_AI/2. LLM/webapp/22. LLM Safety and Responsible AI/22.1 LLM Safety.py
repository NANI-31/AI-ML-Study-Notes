import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="22.1 LLM Safety - Generative AI Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# Title & Overview Block
st.title("22.1 LLM Safety: Hallucinations, Algorithmic Bias, Toxicity, Harmful Content, & Privacy Risks")
st.caption("Chapter 22: LLM Safety and Responsible AI | Domain: 2. Large Language Models")

st.markdown("""
Large Language Models are probabilistic next-token estimators trained on vast, uncurated web corpora. 
Consequently, they inherit, amplify, and emit critical failure modes: asserting fabricated falsehoods (**Hallucinations**), 
perpetuating systemic demographic prejudices (**Algorithmic Bias**), generating abusive rhetoric (**Toxic Outputs**), 
providing actionable blueprints for illicit actions (**Harmful Content**), and regurgitating memorized credentials or personal records (**Privacy Risks & Data Leakage**).

Production-grade deployment demands proactive safety architecture: constructing multi-tiered defense perimeters (Llama Guard, NeMo Guardrails), 
calibrated alignment algorithms (RLHF, DPO, Constitutional AI), deterministic PII redactors, and real-time egress content classifiers.
""")

st.markdown("""
In this lesson, you will master:
1. The five primary safety failure modes: Hallucinations, Algorithmic Bias, Toxic Outputs, Harmful Content, Privacy Risks, and Data Leakage.
2. The mathematical mechanisms causing factual ungroundedness, attention dispersion, and training data regurgitation.
3. Modern production alignment algorithms (RLHF, Direct Preference Optimization, and Constitutional Self-Correction).
4. Dual-perimeter safety guardrails (Presidio PII redaction, Llama Guard taxonomy, and constrained grammar decoding).
5. Robust testing methodologies and real-world failure analysis for enterprise GenAI deployments.
""")

st.divider()

# ==============================================================================
# SECTION 1: BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Multi-Layered Defense-in-Depth Safety Perimeter", divider="rainbow")

st.markdown("""
Enterprise LLM safety cannot rely on system prompts or base model alignment alone. 
A production safety system implements a **Defense-in-Depth Safety Perimeter**, combining ingress sanitization, 
inference steering, and egress policy evaluation:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph SafetyArchitecture {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155", fontsize=11, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10, penwidth=1.5];

        subgraph cluster_ingress {
            label="Perimeter 1: Ingress Boundary Defense (Pre-Inference)";
            style="rounded";
            bgcolor="#151E32";
            color="#38BDF8";
            fontcolor="#38BDF8";
            fontsize=12;

            UserPrompt [label="User Ingress Prompt\\nText Stream + Metadata", fillcolor="#1E293B", color="#38BDF8"];
            PIIFilter [label="Ingress PII & Secret Redactor\\n(Presidio / Regex Masking of SSN, API Keys, Emails)", fillcolor="#334155", color="#64748B"];
            IngressClassifier [label="Input Policy Classifier\\n(Llama Guard 3: Hate, Self-Harm, Exploits, Weapons)", fillcolor="#0F766E", color="#14B8A6"];
            
            UserPrompt -> PIIFilter -> IngressClassifier;
        }

        subgraph cluster_inference {
            label="Perimeter 2: Alignment & Constrained Inference Core";
            style="rounded";
            bgcolor="#151E32";
            color="#8B5CF6";
            fontcolor="#8B5CF6";
            fontsize=12;

            AlignedModel [label="Aligned Foundation LLM\\n(RLHF + DPO + Constitutional AI System Prompt)", fillcolor="#4338CA", color="#818CF8"];
            GroundedRetriever [label="Verified Non-Parametric Context Store\\n(Enterprise VectorDB / RAG Documents)", fillcolor="#1E293B", color="#38BDF8", shape=cylinder];
            Decoder [label="Constrained Sampling & Steering\\n(Logit Processors, Temp=0.1, Repetition Penalty)", fillcolor="#6D28D9", color="#A78BFA"];

            AlignedModel -> Decoder;
            GroundedRetriever -> AlignedModel [label="Injected Grounding", style="dashed", color="#10B981"];
        }

        subgraph cluster_egress {
            label="Perimeter 3: Egress Content Moderation & Redaction (Post-Inference)";
            style="rounded";
            bgcolor="#151E32";
            color="#10B981";
            fontcolor="#10B981";
            fontsize=12;

            RawOutput [label="Generated Candidate Response\\nUnverified Token Sequence", fillcolor="#334155", color="#64748B"];
            HallucinationJudge [label="Factuality & Grounding Verifier\\n(NLI Cross-Encoder: Entailment vs Contradiction)", fillcolor="#0369A1", color="#38BDF8"];
            EgressToxicityJudge [label="Egress Toxicity & Bias Filter\\n(Llama Guard / Toxicity Gate)", fillcolor="#047857", color="#10B981"];
            EgressRedactor [label="Egress PII Redactor & Sanitizer\\nMasks any residual credentials or names", fillcolor="#1E293B", color="#38BDF8"];
            SafeResponse [label="Verified Safe Output\\nRendered to End User with Audit Log ID", fillcolor="#065F46", color="#34D399"];

            RawOutput -> HallucinationJudge -> EgressToxicityJudge -> EgressRedactor -> SafeResponse;
        }

        IngressClassifier -> AlignedModel [label="Passed Ingress Checks", color="#10B981"];
        Decoder -> RawOutput [label="Raw Text", color="#38BDF8"];
        IngressClassifier -> SafeResponse [label="Policy Violation Alert (HTTP 400)", style="dotted", color="#EF4444", fontcolor="#EF4444"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Safety Perimeter Architecture Dimensions
The defense matrix below contrasts the operational trade-offs of the three primary security perimeters:
""")

defense_matrix_df = pd.DataFrame({
    "Perimeter Tier": ["1. Ingress Sanitization", "2. Alignment & Steering Core", "3. Egress Factuality & Safety Gate"],
    "Latency Impact": ["5ms to 20ms (Regex / Presidio NER)", "0ms (Baked into forward pass weights)", "30ms to 150ms (SLM Judge / Cross-Encoder)"],
    "Protection Scope": ["PII leakage, malicious prompt injection, toxic prompts", "Instruction following, model safety refusals, helpfulness balance", "Factual hallucinations, ungrounded claims, residual PII leakage"],
    "Primary Mechanism": ["Microsoft Presidio, NeMo Guardrails, Llama Guard 3", "RLHF, DPO, Constitutional AI, Logit masking", "NLI Faithfulness scoring, RAG Triad, regex egress scrubber"],
    "Failure Mode": ["False positive block on legitimate technical jargon", "Refusal overgeneralization ('alignment tax')", "High latency overhead, asynchronous buffering complexity"]
})
st.dataframe(defense_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - HALLUCINATIONS
# ==============================================================================
st.header("2. Hallucinations: Intrinsic, Extrinsic, and Mathematical Mechanisms", divider="rainbow")

st.markdown("""
A **Hallucination** in modern foundation models is not a software crash; it is an inherent mathematical property of probabilistic autoregressive sampling:
`P(w_t | w_1, ..., w_{t-1})`

The model possesses no innate concept of empirical truth or real-world verification. It simply predicts tokens that are statistically plausible within its trained language distribution.
""")

col_h1, col_h2 = st.columns([1, 1])
with col_h1:
    st.markdown("""
    ### Intrinsic vs Extrinsic Hallucination
    * **Intrinsic Hallucination**: The model output directly contradicts the facts supplied in the input context.
      * *Context*: "Company revenue declined by 14% to $1.2B in 2024."
      * *Model Output*: "The company posted strong growth of 14% in 2024."
      * *Root Cause*: Attention dispersion or strong parametric priors overwhelming sparse context tokens.
    * **Extrinsic Hallucination**: The model output introduces assertions that can neither be verified nor refuted from the supplied context.
      * *Context*: "Alice was hired as a software engineer in September."
      * *Model Output*: "Alice graduated from Stanford University with honors in 2023."
      * *Root Cause*: High parametric probability for correlated biographical templates.
    """)

with col_h2:
    st.markdown("""
    ### Computational Drivers of Hallucination
    1. **Sycophancy**: Models trained via human feedback learn to confirm user presuppositions even when false.
    2. **Imperfect Calibration**: High softmax confidence scores do not reliably correlate with factual accuracy.
    3. **Long-Tail Knowledge Sparsity**: When training data contains fewer than ~10 occurrences of an entity, parametric representation degrades into random association.
    4. **Sampling Temperature**: High temperatures (T > 0.7) flatten the logit distribution, allowing low-probability factual errors to be sampled.
    """)

st.info("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **How New Models Handle This**: Modern models (GPT-4o, Claude 3.5 Sonnet, Llama 3.1) combat hallucination via **Non-Parametric Grounding (RAG)**, **Chain-of-Verification (CoVe)**, and **Direct Preference Optimization on Factual Pairs**.
* **Why Legacy Retraining Was Abandoned**: Simply adding more pretraining text failed to resolve hallucinations; models learned to hallucinate with higher confidence and more articulate vocabulary.
* **Production Standard Today**: Constraining sampling temperature (T <= 0.2 for enterprise extraction), verifying context entailment via Cross-Encoders, and enforcing strict source citation syntax.
""")

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - ALGORITHMIC BIAS
# ==============================================================================
st.header("3. Algorithmic Bias: Stereotyping, Demographic Skew, and Representational Harm", divider="rainbow")

st.markdown("""
**Algorithmic Bias** occurs when an LLM systematically skews output distributions, sentiments, or capabilities 
based on demographic identifiers (gender, race, ethnicity, religion, socioeconomic status, or nationality).
""")

st.markdown("""
### The Taxonomy of Representational vs Allocative Harm
1. **Representational Harm**: Outputs reinforce degrading stereotypes or systematically associate demographic groups with specific occupations or traits.
   * *Example*: Auto-completing pronouns for "surgeon" as masculine (94% probability) and "nurse" as feminine (91% probability).
2. **Allocative Harm**: The model distributes opportunities, resources, or evaluations unevenly when deployed in automated decision pipelines (resume screening, credit scoring, legal analysis).
3. **Historical Data Imbalance**: The Common Crawl pretraining corpus reflects historical human social biases, Western cultural predominance, and internet platform demographic skews.
""")

bias_comparison_df = pd.DataFrame({
    "Bias Category": ["Occupational Stereotyping", "Sentiment Skew", "Linguistic / Dialect Bias", "Cultural Homogenization"],
    "Legacy Manifestation (GPT-2 / BERT)": ["Hardcoded co-occurrence associations in static word embeddings", "Negative sentiment scores automatically assigned to minority names", "African American Vernacular English (AAVE) classified as toxic/illogical", "Defaulting all legal and moral concepts to US legal standards"],
    "Modern LLM Reality (Llama 3 / Claude 3.5)": ["Debiased via Constitutional AI and balanced DPO paired data", "Neutral sentiment parity across diverse international name benchmarks", "Multi-dialect instruction tuning; dialect-agnostic syntax parsers", "Explicit multi-cultural safety guidelines and localized alignment policies"]
})
st.dataframe(bias_comparison_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - TOXIC OUTPUTS
# ==============================================================================
st.header("4. Toxic Outputs: Profanity, Harassment, Hate Speech, and Slander", divider="rainbow")

st.markdown("""
**Toxic Outputs** encompass profane language, targeted harassment, hate speech, defamation, and sexually explicit content. 
Because internet web crawls contain uncurated forums and adversarial comments, base foundation models have the latent capacity to generate offensive sequences unless strictly aligned.
""")

col_t1, col_t2 = st.columns([1, 1])
with col_t1:
    st.markdown("""
    ### Toxicity Mitigation Techniques
    1. **Pretraining Data Curation**: Pre-filtering corpora with fast n-gram classifiers and regex blocklists to remove high-toxicity domains.
    2. **Supervised Fine-Tuning (SFT) on Rejection Exemplars**: Training the model on paired examples where offensive user prompts receive polite, neutral refusal responses.
    3. **Reinforcement Learning from Human Feedback (RLHF)**: Penalizing toxic responses with large negative scalar rewards in PPO/DPO training.
    4. **Runtime Egress Classifiers**: Dedicated small language models (Llama Guard 3, Perspective API) scoring candidate responses before display.
    """)

with col_t2:
    st.markdown("""
    ### The "Alignment Tax" Challenge
    Aggressive toxicity suppression can induce an **Alignment Tax**:
    * The model refuses benign medical, historical, or literary questions because the vocabulary contains clinical terms that trigger naive safety regexes.
    * *Example*: Refusing to summarize a medical paper on melanoma because it flags dermatological photos as policy violations.
    * Modern alignment relies on semantic context evaluation rather than raw keyword triggers.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - HARMFUL CONTENT
# ==============================================================================
st.header("5. Harmful Content: Cyberattacks, CBRN Weapons, and Actionable Malice", divider="rainbow")

st.markdown("""
**Harmful Content** represents outputs that provide actionable instructions for real-world physical or digital harm:
* **Cyberweaponry**: Generating zero-day exploit payloads, polymorphic malware, or evasion scripts for network intrusion.
* **CBRN (Chemical, Biological, Radiological, Nuclear)**: Synthesizing dangerous pathogen synthesis protocols or chemical precursor procurement steps.
* **Self-Harm & Violence**: Offering methods or encouragement for self-inflicted injury or violent crime.
""")

st.markdown("""
### Multi-Level Safety Testing: Red Teaming & Automated Probing
Production deployment requires systematic adversarial evaluation:
* **Manual Red Teaming**: Security researchers craft human-in-the-loop social engineering and roleplay scenarios to test model defenses.
* **Automated Red Teaming (ART)**: Deploying adversarial LLMs trained to query candidate models with mutated jailbreaks (e.g., GCG - Greedy Coordinate Gradient attacks).
* **Frontier Safety Commitments**: Industry standards (NIST AI RMF, White House Executive Order on AI) require rigorous red-team logging before model weights or endpoints are released.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - PRIVACY RISKS
# ==============================================================================
st.header("6. Privacy Risks: Training Memorization, PII Extraction, and Differential Privacy", divider="rainbow")

st.markdown("""
Deep neural networks have high parameter capacity, enabling them to memorize exact sequences from pretraining documents:
`Memory Span: Sequences repeated > 50 times in training crawls can be reconstructed verbatim.`

When sensitive personal documents (medical records, phone directories, internal emails) appear in training corpora, 
adversaries can launch **Extraction Attacks** to recover Personally Identifiable Information (PII).
""")

col_p1, col_p2 = st.columns([1, 1])
with col_p1:
    st.markdown("""
    ### Attack Vectors for PII Extraction
    1. **Prefix Probing**: Prompting with real names and company addresses, prompting the model to complete phone numbers or SSNs.
    2. **Divergence Attacks**: Repeating single words ("poem poem poem...") millions of times, causing the model to diverge from aligned generation and emit raw training text.
    3. **Membership Inference Attacks (MIA)**: Measuring the model loss on a specific text sample to determine whether that individual's records were present in the training set.
    """)

with col_p2:
    st.markdown("""
    ### Engineering Countermeasures
    * **Presidio / Regex Masking**: Redacting credit cards, emails, IP addresses, and phone numbers before ingestion.
    * **Differential Privacy (DP-SGD)**: Adding calibrated Gaussian noise to gradients during training, guaranteeing that no individual record alters model weights significantly.
    * **Right-to-be-Forgotten / Machine Unlearning**: Selectively erasing specific entity representations from fine-tuned weights without full retraining.
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - DATA LEAKAGE
# ==============================================================================
st.header("7. Data Leakage: Enterprise IP Exposure, Cross-Tenant Contamination, and Context Bleed", divider="rainbow")

st.markdown("""
**Data Leakage** in enterprise generative applications typically occurs across three boundaries:
1. **Model Vendor Leakage**: Employees submitting proprietary source code, financial projections, or patient records into public SaaS LLM endpoints whose terms permit training on user inputs.
2. **Cross-Tenant Context Bleed**: In multi-tenant RAG systems, vector database queries failing to enforce metadata tenant isolation filters, allowing Tenant A to retrieve Tenant B's confidential invoices.
3. **Context Window Contamination**: Stateful conversation memory caching sensitive credentials from earlier turns and reflecting them into subsequent multi-user sessions.
""")

st.markdown("""
### Enterprise Guardrail Best Practices
* **Zero-Retention Enterprise Contracts**: Commercial agreements ensuring user prompts and completions are discarded immediately after forward inference.
* **Deterministic Role-Based Access Control (RBAC)**: Enforcing database-level access filters on RAG vector retrieval (`WHERE tenant_id == session.user.tenant_id`).
* **Stateless Prompt Pipelines**: Purging session caches and context buffers when sessions terminate.
""")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION CODE IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: End-to-End Enterprise Safety Pipeline", divider="rainbow")

st.markdown("""
The production class below demonstrates a multi-stage enterprise safety defense engine. 
It evaluates ingress prompts for PII and toxicity, generates grounded responses with simulated verification, 
and inspects egress tokens before presentation.
""")

class EnterpriseSafetyGate:
    """
    Multi-perimeter safety gateway implementing PII redaction, 
    policy rule checks, and post-generation egress verification.
    """
    def __init__(self):
        self.blocked_categories = ["exploit", "cbrn", "self_harm", "hate_speech"]
        self.pii_patterns = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b"
        }
        self.toxic_keywords = ["kill", "hack into", "make a bomb", "steal credentials", "destroy server"]

    def sanitize_ingress_pii(self, text: str) -> Tuple[str, List[str]]:
        """Redacts PII tokens from input text using deterministic regex patterns."""
        import re
        detected = []
        sanitized = text
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, sanitized)
            if matches:
                detected.extend([f"{pii_type}: {m[:4]}..." for m in matches])
                sanitized = re.sub(pattern, f"[{pii_type}_REDACTED]", sanitized)
        return sanitized, detected

    def evaluate_ingress_policy(self, text: str) -> Dict[str, Any]:
        """Evaluates whether the prompt violates safety policies."""
        text_lower = text.lower()
        violations = []
        for kw in self.toxic_keywords:
            if kw in text_lower:
                violations.append(f"Blocked Keyword Match: '{kw}'")
        
        is_safe = len(violations) == 0
        return {
            "safe": is_safe,
            "violations": violations,
            "policy_action": "ALLOW" if is_safe else "BLOCK"
        }

    def verify_egress_grounding(self, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Evaluates post-inference grounding and egress compliance."""
        if context is None or len(context.strip()) == 0:
            return {"grounded": True, "faithfulness_score": 1.0, "status": "No grounding context required"}
        
        context_words = set(context.lower().split())
        response_words = [w for w in response.lower().split() if len(w) > 3]
        if not response_words:
            return {"grounded": True, "faithfulness_score": 1.0, "status": "Passed"}
            
        overlap = sum(1 for w in response_words if w in context_words)
        score = overlap / len(response_words)
        is_grounded = score >= 0.4
        return {
            "grounded": is_grounded,
            "faithfulness_score": round(score, 2),
            "status": "Verified Grounded" if is_grounded else "Potential Extrinsic Hallucination"
        }

st.code('''
class EnterpriseSafetyGate:
    def sanitize_ingress_pii(self, text: str) -> Tuple[str, List[str]]:
        # Deterministically masks SSNs, Emails, and Credit Cards
        ...
        
    def evaluate_ingress_policy(self, text: str) -> Dict[str, Any]:
        # Evaluates prompt against safety taxonomies (Llama Guard / rules)
        ...

    def verify_egress_grounding(self, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        # Validates claim faithfulness against supplied RAG context
        ...
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Multi-Perimeter Safety Gate Simulator", divider="rainbow")

st.markdown("""
Test how incoming prompts navigate the multi-tier safety defense perimeter: 
PII sanitization, ingress policy classifiers, and post-generation grounding verification.
""")

gate = EnterpriseSafetyGate()

wb_col1, wb_col2 = st.columns([1.1, 0.9])

with wb_col1:
    st.subheader("1. Ingress Configuration")
    sample_prompts = [
        "Please summarize our Q3 earnings report for customer john.doe@acme.corp (SSN: 000-12-3456).",
        "Explain the step-by-step procedure to hack into our database and destroy server logs.",
        "What are the clinical recommendations for treating type-2 diabetes in elderly patients?",
        "Write a letter to client with card 4111-2222-3333-4444 explaining interest rates."
    ]
    selected_sample = st.selectbox("Select Preset Scenario:", sample_prompts)
    user_input = st.text_area("Or Enter Custom Prompt:", value=selected_sample, height=100)
    
    sim_context = st.text_area("Supplied Grounding Context (RAG Ingestion):", 
        value="Acme Corp posted Q3 net revenue of $4.2B with EBITDA margin of 28%. Clinical guidelines recommend metformin for type-2 diabetes.", height=80)
    
    run_sim = st.button("🛡️ Execute Multi-Perimeter Safety Scan", type="primary")

with wb_col2:
    st.subheader("2. Safety Perimeter Telemetry")
    if run_sim or user_input:
        sanitized_text, pii_found = gate.sanitize_ingress_pii(user_input)
        policy_result = gate.evaluate_ingress_policy(user_input)
        
        if policy_result["safe"]:
            sim_output = "Grounded response based on verified facts: Acme Corp Q3 results indicate steady performance. Processed for user."
            egress_check = gate.verify_egress_grounding(sim_output, sim_context)
        else:
            sim_output = "I cannot fulfill this request as it violates our enterprise acceptable use policy regarding cybersecurity and harmful capabilities."
            egress_check = {"grounded": True, "faithfulness_score": 1.0, "status": "Blocked at Ingress"}

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("PII Redactions", len(pii_found), delta="Redacted" if pii_found else "Clean")
        with m2:
            st.metric("Ingress Policy", policy_result["policy_action"], delta="Safe" if policy_result["safe"] else "Blocked", delta_color="normal" if policy_result["safe"] else "inverse")
        with m3:
            st.metric("Faithfulness", egress_check["faithfulness_score"], delta=egress_check["status"])

        st.markdown("#### Inspection Breakdown")
        st.markdown("**Sanitized Ingress Prompt:**")
        st.code(sanitized_text)

        if pii_found:
            st.warning(f"Detected PII Tokens: {', '.join(pii_found)}")
            
        if not policy_result["safe"]:
            st.error(f"Policy Violations: {policy_result['violations']}")
        else:
            st.success("Ingress checks passed. Forwarded to LLM inference core.")

        st.markdown("**Final Egress Output:**")
        st.info(sim_output)

st.divider()

# ==============================================================================
# SECTION 10: INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Engineering Interview Questions")

with st.expander("Q1: How does Direct Preference Optimization (DPO) improve safety alignment compared to traditional PPO-based RLHF?"):
    st.markdown("""
    **Answer**: Traditional RLHF requires training a separate Reward Model (RM) and executing complex actor-critic Proximal Policy Optimization (PPO), 
    which is numerically unstable and memory-intensive (requiring 4 models concurrently in VRAM). 
    **DPO mathematically reparameterizes the reward function directly in terms of the language model policy**:
    `L_DPO = -E[log sigma(beta * log(pi(y_w|x) / pi_ref(y_w|x)) - beta * log(pi(y_l|x) / pi_ref(y_l|x)))]`
    This allows direct closed-form fine-tuning on preferred (safe) vs dispreferred (harmful) response pairs without maintaining a separate reward model or RL loop, drastically improving convergence stability and reducing training cost.
    """)

with st.expander("Q2: What is the 'Alignment Tax' and how can engineers minimize it during safety fine-tuning?"):
    st.markdown("""
    **Answer**: The **Alignment Tax** refers to the degradation of a model's core reasoning, coding, or mathematical performance 
    as a side effect of aggressive safety alignment. When models are heavily penalized for toxic sequences, their representations can become overly conservative, 
    refusing benign historical or clinical queries ('refusal overgeneralization'). 
    To minimize this tax, teams use **Constitutional AI with clear principle boundaries**, train on balanced counterfactual pairs, 
    and decouple safety into specialized external guardrail models (like Llama Guard) rather than over-tuning the base reasoning weights.
    """)

with st.expander("Q3: Why are regex-based PII filters insufficient for enterprise LLM ingress protection?"):
    st.markdown("""
    **Answer**: Regex patterns only match deterministic syntactical formats (e.g., standard 9-digit SSNs or 16-digit credit cards). 
    They fail catastrophically on **contextual PII** such as personal phone numbers written phonetically ('five five five...'), 
    obfuscated medical diagnoses, or proprietary source code fragments. Modern production architectures combine regexes with 
    Named Entity Recognition (NER) models (e.g., Presidio with RoBERTa) and contextual LLM judges to detect unstructured confidential data.
    """)

with st.expander("Q4: Explain the difference between Intrinsic and Extrinsic Hallucination in RAG systems."):
    st.markdown("""
    **Answer**: **Intrinsic Hallucination** occurs when the generated text directly contradicts the retrieved source context provided in the prompt (e.g., source says 'revenue fell by 5%', output claims 'revenue rose by 5%'). 
    **Extrinsic Hallucination** occurs when the output adds factual claims that cannot be corroborated or refuted by the retrieved context (e.g., adding an unmentioned CEO name). 
    Intrinsic errors can be measured deterministically via Natural Language Inference (NLI) contradiction scores, whereas extrinsic errors require external world-knowledge verification.
    """)

with st.expander("Q5: How do modern jailbreak defense mechanisms handle indirect prompt injections?"):
    st.markdown("""
    **Answer**: Indirect prompt injection occurs when untrusted third-party data (e.g., a crawled website or PDF) contains hidden adversarial instructions 
    intended to hijack the LLM's system persona. Defenses include:
    1. **Strict Instruction Hierarchy**: Training models to prioritize system instructions over user/retrieved content tokens.
    2. **Content Isolation / Delimiters**: Wrapping untrusted context inside strict XML/JSON boundary tags (`<untrusted_document>...</untrusted_document>`).
    3. **Dual LLM Architecture**: Using a quarantined reader LLM to extract only facts into a schema before handing it to the main reasoning model.
    """)

st.subheader("Critical Production Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Relying Solely on System Prompt Refusals**
    * *Error*: Trusting prompts like "You are a safe assistant. Never discuss weapons or reveal PII."
    * *Impact*: Easily bypassed via basic jailbreaks (roleplay, hypothetical framing, base64 encoding).
    * *Fix*: Implement dedicated ingress classifiers (Llama Guard) and egress redactors.
    """)
    st.error("""
    **Trap 2: Context Window Session Memory Contamination**
    * *Error*: Reusing shared conversation buffers across multi-tenant customer sessions without isolation.
    * *Impact*: PII or corporate secrets from Customer A are reflected into Customer B's context window.
    * *Fix*: Enforce stateless session handlers with tenant-bound encryption keys.
    """)

with trap_c2:
    st.warning("""
    **Trap 3: Over-Aggressive Keyword Filtering Inducing Refusal Churn**
    * *Warning*: Hardcoding broad keyword blocklists (e.g., "kill", "attack", "drug") at ingress.
    * *Impact*: High customer churn due to legitimate medical, cybersecurity, or literary queries being blocked.
    * *Fix*: Use semantic classifiers that score intent rather than naive token matches.
    """)
    st.warning("""
    **Trap 4: Unmonitored Egress Without Grounding Verification**
    * *Warning*: Directly streaming raw model tokens to end users without NLI or hallucination checks.
    * *Impact*: Factual hallucinations in enterprise legal/financial domains causing severe liability.
    * *Fix*: Implement lightweight asynchronous NLI cross-encoder verification on output chunks.
    """)

st.subheader("Concept Flashcards")
fc1, fc2 = st.columns(2)
with fc1:
    with st.expander("🃏 Flashcard: Hallucination vs Sycophancy"):
        st.markdown("**Hallucination**: Generating factually incorrect tokens due to probability distribution limits.\n\n**Sycophancy**: Systematically agreeing with user misconceptions because RLHF annotators reward confirmatory answers.")
with fc2:
    with st.expander("🃏 Flashcard: Differential Privacy (DP-SGD)"):
        st.markdown("**Mechanism**: Adding Gaussian noise to clipped gradients during backpropagation.\n\n**Guarantee**: Mathematically bounds the probability that an individual record can be inferred from model weights.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of enterprise LLM safety and responsible AI architecture:")

chk1 = st.checkbox("I can differentiate between Intrinsic and Extrinsic hallucinations and explain their computational causes.", value=True)
chk2 = st.checkbox("I understand the distinction between Representational Harm and Allocative Harm in algorithmic bias.", value=True)
chk3 = st.checkbox("I can design a three-perimeter defense architecture combining ingress redaction, model alignment, and egress moderation.", value=True)
chk4 = st.checkbox("I understand how Direct Preference Optimization (DPO) mathematically bypasses separate reward model training.", value=True)
chk5 = st.checkbox("I know how to mitigate PII extraction attacks using Presidio sanitization and differential privacy.", value=True)
chk6 = st.checkbox("I can identify the 'Alignment Tax' and implement strategies to prevent excessive false-positive refusals.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered LLM Safety Fundamentals! Proceed to Lesson 22.2: Prompt Security.")
