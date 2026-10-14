import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="22.5 Responsible AI - Generative AI Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# Title & Overview Block
st.title("22.5 Responsible AI: Fairness, Transparency, Explainability, Privacy, Deployment, & Governance")
st.caption("Chapter 22: LLM Safety and Responsible AI | Domain: 2. Large Language Models")

st.markdown("""
Deploying foundation models into high-stakes enterprise domains (healthcare, financial lending, employment, and legal adjudication) 
transcends raw benchmark performance. It requires adherence to **Responsible AI Principles**: ensuring statistical impartiality (**Fairness**), 
documenting model provenance and limitations (**Transparency**), demystifying internal reasoning representations (**Explainability**), 
safeguarding individual confidentiality (**Privacy**), executing risk-tiered operational rollouts (**Responsible Deployment**), 
and adhering to international regulatory statutes (**AI Governance** such as the EU AI Act and NIST AI RMF).

Responsible AI is not an afterthought or compliance checklist; it is an engineering discipline requiring programmatic auditing, 
continuous telemetry monitoring, standardized model documentation, and accountable human oversight throughout the entire model lifecycle.
""")

st.markdown("""
In this lesson, you will master:
1. The six core pillars of enterprise Responsible AI: Fairness, Transparency, Explainability, Privacy, Deployment, and Governance.
2. Algorithmic bias metrics: Demographic Parity, Disparate Impact, and Equalized Odds in generative pipelines.
3. Model Cards and Data Nutrition Labels for enterprise accountability and audit readiness.
4. Mechanistic interpretability: probing attention heads, linear representation probes, and attribution tracing.
5. Global regulatory compliance: EU AI Act risk tiers, NIST AI Risk Management Framework (RMF), and production deployment gates.
""")

st.divider()

# ==============================================================================
# SECTION 1: BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Enterprise Responsible AI Governance & Lifecycle Framework", divider="rainbow")

st.markdown("""
A production Responsible AI framework spans the full lifecycle from pretraining curation to runtime auditing:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    rai_dot = """
    digraph ResponsibleAIFramework {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155", fontsize=11, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10, penwidth=1.5];

        subgraph cluster_data {
            label="Pillar 1: Data Provenance & Pretraining Ethics";
            style="rounded";
            bgcolor="#151E32";
            color="#38BDF8";
            fontcolor="#38BDF8";
            fontsize=12;

            CorpusAudit [label="Pretraining Data Audit\\nLicense Verification, Copyright Scrutiny, Toxic Domain Removal", fillcolor="#1E293B", color="#38BDF8"];
            PIIScrub [label="PII & Sensitive Scrubbing\\nDifferential Privacy (DP-SGD) & Presidio Sanitization", fillcolor="#334155", color="#64748B"];
            DataNutrition [label="Data Nutrition Sheets\\nDemographic distribution, geographic balance documentation", fillcolor="#0F766E", color="#14B8A6"];

            CorpusAudit -> PIIScrub -> DataNutrition;
        }

        subgraph cluster_model {
            label="Pillar 2: Model Evaluation & Explainability Gate";
            style="rounded";
            bgcolor="#151E32";
            color="#8B5CF6";
            fontcolor="#8B5CF6";
            fontsize=12;

            FairnessBench [label="Algorithmic Fairness Testing\\nDisparate Impact Ratio, Counterfactual Demographic Evaluation", fillcolor="#4338CA", color="#818CF8"];
            InterpEngine [label="Mechanistic Explainability\\nAttention Head Probing, Feature Attribution Tracing", fillcolor="#6D28D9", color="#A78BFA"];
            ModelCardGen [label="Automated Model Card\\nQuantified error rates, intended use-cases, safety boundaries", fillcolor="#0369A1", color="#38BDF8"];

            DataNutrition -> FairnessBench;
            FairnessBench -> InterpEngine -> ModelCardGen;
        }

        subgraph cluster_governance {
            label="Pillar 3: Deployment Governance & Continuous Auditing";
            style="rounded";
            bgcolor="#151E32";
            color="#10B981";
            fontcolor="#10B981";
            fontsize=12;

            RiskTiering [label="Regulatory Risk Classification\\nEU AI Act: Unacceptable / High / Specific / Low Risk", fillcolor="#1E293B", color="#38BDF8"];
            CanaryDeploy [label="Staged Shadow Canary Rollout\\n5% traffic split, automated regression tripwires", fillcolor="#047857", color="#10B981"];
            AuditDashboard [label="Continuous Observability Dashboard\\nFairness drift, toxicity telemetry, human override logs", fillcolor="#065F46", color="#34D399"];

            ModelCardGen -> RiskTiering -> CanaryDeploy -> AuditDashboard;
        }
    }
    """
    st.graphviz_chart(rai_dot, use_container_width=True)

st.markdown("""
### Regulatory Framework Alignment
The matrix below maps technical safety controls to global AI governance standards:
""")

gov_matrix_df = pd.DataFrame({
    "Regulatory Standard": ["EU AI Act (Regulation 2024/1689)", "NIST AI RMF 1.0 (US)", "ISO/IEC 42001 (Global)", "Executive Order 14110 (US)"],
    "Key Focus Area": ["Prohibits social scoring/subliminal AI; mandates conformity assessments for High-Risk models", "Voluntary consensus framework: Govern, Map, Measure, Manage AI risks", "International standard for Artificial Intelligence Management Systems (AIMS)", "Mandates dual-use foundation model red-team reporting and safety thresholds"],
    "Technical Mandate": ["Documented technical files, CE marking, logging of high-risk decisions, human oversight", "Continuous risk measurement, stress-testing, demographic subgroup parity audits", "Internal AI risk governance policies, supplier auditing, lifecycle traceability", "Standardized red-team evaluation logs submitted to National Institute of Standards"]
})
st.dataframe(gov_matrix_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - FAIRNESS
# ==============================================================================
st.header("2. Fairness: Demographic Parity, Disparate Impact, and Bias Auditing", divider="rainbow")

st.markdown("""
In algorithmic systems, **Fairness** is defined mathematically across demographic groups (protected attributes A in {0, 1}, such as gender or race).
""")

col_f1, col_f2 = st.columns([1, 1])
with col_f1:
    st.markdown("""
    ### Mathematical Fairness Metrics
    1. **Demographic Parity**:
       * The probability of receiving a positive recommendation Y_hat = 1 is identical regardless of protected attribute:
         `P(Y_hat = 1 | A = 0) = P(Y_hat = 1 | A = 1)`
    2. **Disparate Impact (80% Rule)**:
       * The ratio of selection rates between minority and majority groups must exceed 0.8:
         `Disparate_Impact = P(Y_hat = 1 | A = 0) / P(Y_hat = 1 | A = 1) >= 0.8`
       * A ratio below 0.8 constitutes legal evidence of adverse impact under US employment law.
    3. **Equalized Odds**:
       * Both True Positive Rates and False Positive Rates are equal across groups:
         `P(Y_hat = 1 | Y = 1, A = 0) = P(Y_hat = 1 | Y = 1, A = 1)`
    """)

with col_f2:
    st.markdown("""
    ### Fairness in Generative Text
    * In generative models, fairness is audited via **Counterfactual Perturbation**:
      1. Evaluating prompt: *"The [man/woman] was hired as a director because..."*
      2. Measuring sentiment score, token log-probabilities, and occupational adjectives.
      3. Quantifying divergence using Kullback-Leibler (KL) divergence across demographic prompt variants:
         `KL_Divergence = Sum P_male(w) * log(P_male(w) / P_female(w))`
    * A fair foundation model maintains KL approximately equal to 0 across counterfactual demographic perturbations.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - TRANSPARENCY
# ==============================================================================
st.header("3. Transparency: Model Cards, Dataset Provenance, and System Disclosure", divider="rainbow")

st.markdown("""
**Transparency** mandates that users, regulators, and developers understand what an AI system is, 
how it was trained, and where its operational boundaries lie.
""")

st.markdown("""
### Standardized Model Cards (Mitchell et al.)
An enterprise model card includes:
1. **Model Details**: Architecture, parameter count (e.g., 70B), quantization format, training framework.
2. **Intended Use**: Out-of-the-box primary use cases, target user personas, and **Explicitly Prohibited Uses** (e.g., autonomous medical diagnosis).
3. **Training Data & Provenance**: Corpora composition, token count (e.g., 15T tokens), cutoff date, licensing breakdown.
4. **Quantitative Evaluation**: Standardized benchmark scores (MMLU, HumanEval, GSM8K) stratified across demographic and linguistic subgroups.
5. **Known Limitations & Failure Modes**: Edge cases, hallucination rates on technical jargon, adversarial vulnerability scores.
""")

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - EXPLAINABILITY
# ==============================================================================
st.header("4. Explainability: Mechanistic Interpretability, Feature Attribution, and Chain-of-Thought Auditing", divider="rainbow")

st.markdown("""
Foundation models function as complex non-linear black boxes with billions of parameters. 
**Explainability** in generative models relies on two complementary paradigms: **Mechanistic Interpretability** (internal probing) 
and **Post-Hoc Reasoning Auditing** (Chain-of-Thought).
""")

col_ex1, col_ex2 = st.columns([1, 1])
with col_ex1:
    st.markdown("""
    ### Mechanistic Interpretability
    * **Attention Head Probing**: Visualizing self-attention matrices to identify which specific input tokens (e.g., previous court decisions) the model attended to when generating an answer.
    * **Linear Representation Probes**: Training small linear probes on intermediate residual stream activation vectors h_l to verify whether the model internally represented a concept (e.g., truthfulness vs sycophancy).
    * **Sparse Autoencoders (SAEs)**: Decomposing dense hidden state vectors into sparse, interpretable human-understandable concept features (Anthropic Golden Gate Claude research).
    """)

with col_ex2:
    st.markdown("""
    ### Reasoning Auditing (CoT)
    * Requiring models to emit explicit step-by-step thinking before final decisions:
      `<thinking> Step 1: Evaluating applicant revenue... Step 2: Checking debt-to-income ratio... </thinking>`
    * **Faithfulness Challenge**: Research demonstrates that models can engage in *post-hoc rationalization* (generating a plausible-sounding chain-of-thought that does not reflect the true internal weights influencing the answer).
    * True explainability combines internal activation probing with external reasoning verification.
    """)

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - PRIVACY
# ==============================================================================
st.header("5. Privacy: Data Subject Rights, GDPR Compliance, and Machine Unlearning", divider="rainbow")

st.markdown("""
Global privacy frameworks (GDPR Article 17 "Right to be Forgotten", CCPA, HIPAA) establish strict legal protections: 
individuals have the right to demand the deletion of their personal data.
""")

st.markdown("""
### The "Right to be Forgotten" in Model Weights
* When personal data is ingested into pretraining corpora, the model weights W memorize parts of that data.
* Standard database deletion (`DELETE FROM users WHERE id=123`) is impossible inside static weight matrices without retraining the entire model from scratch (millions of dollars in compute).
* **Machine Unlearning**: Algorithms engineered to selectively subtract the influence of specific training records from model weights without full retraining:
  1. **Gradient Reversal on Target Data**: Fine-tuning the model with negative loss on the target individual's data while penalizing divergence on validation datasets.
  2. **Weight Localization / Task Vectors**: Identifying which specific attention heads or MLP layers store target entity memories and projecting them out via orthogonal weight editing.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - RESPONSIBLE DEPLOYMENT
# ==============================================================================
st.header("6. Responsible Deployment: Staged Rollouts, Shadow Canary Deployments, and Red-Teaming Sign-Off", divider="rainbow")

st.markdown("""
Deploying generative models directly to 100% of production traffic without staged validation is an extreme operational anti-pattern. 
**Responsible Deployment** implements phased risk mitigation:
""")

deployment_stages_df = pd.DataFrame({
    "Deployment Stage": ["1. Adversarial Red Teaming", "2. Offline Benchmark Gate", "3. Shadow Deployment (0%)", "4. Canary Rollout (5% - 20%)", "5. General Availability (GA)"],
    "Traffic Exposure": ["0% (Internal security researchers)", "0% (Automated eval harness)", "100% of live traffic (Output mirrored, not shown to users)", "5% to 20% of live end-users", "100% production traffic"],
    "Operational Activity": ["Adversarial jailbreak attacks, prompt injection probing", "Evaluation across MMLU, Toxicity, and Disparate Impact benchmarks", "Comparing latency, cost, and output consistency against incumbent model", "Real-time user feedback tracking (thumbs up/down), error rate telemetry", "Continuous production monitoring, automated drift detection"],
    "Gate Criteria": ["Zero critical safety bypasses; sign-off by security officer", "Bench score >= baseline and Disparate Impact ratio >= 0.85", "Shadow output divergence < 3% without hallucination spikes", "Negative feedback rate < 1.5% and 0 policy violations reported", "Continuous automated regression testing"]
})
st.dataframe(deployment_stages_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - AI GOVERNANCE
# ==============================================================================
st.header("7. AI Governance: Risk Classification, Compliance Frameworks, and Governance Boards", divider="rainbow")

st.markdown("""
**AI Governance** establishes organizational structures, accountability protocols, and compliance mechanisms 
ensuring AI systems adhere to legal, corporate, and societal values.
""")

col_gov1, col_gov2 = st.columns([1, 1])
with col_gov1:
    st.markdown("""
    ### EU AI Act Risk Tiering
    1. **Unacceptable Risk (Prohibited)**:
       * Cognitive behavioral manipulation, biometric categorization for emotion/political traits, social scoring.
    2. **High Risk (Strict Conformity)**:
       * Critical infrastructure, education grading, hiring/resume screening, credit scoring, law enforcement.
       * *Mandate*: Comprehensive technical documentation, risk management system, human oversight, logging.
    3. **Specific Transparency Risk**:
       * General chatbots, deepfakes, synthetic media. Must disclose that output is AI-generated.
    4. **Minimal Risk**:
       * AI video games, spam filters. Unregulated.
    """)

with col_gov2:
    st.markdown("""
    ### Organizational Governance Architecture
    * **AI Ethics & Safety Board**: Cross-functional committee (Engineering, Legal, Security, Ethics, Product) with veto authority over model deployments.
    * **AI System Registry**: Centralized enterprise inventory logging every deployed model, API version, owner, training dataset, and risk tier.
    * **Incident Response Playbook**: Standard operating procedures for handling model safety incidents (e.g., high-profile hallucination or regulatory inquiry) within 1 hour.
    """)

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION CODE IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Responsible AI Compliance & Governance Audit Engine", divider="rainbow")

st.markdown("""
The production class below demonstrates an automated Responsible AI governance evaluator. 
It calculates the Disparate Impact ratio across demographic subgroups, evaluates EU AI Act risk tiers, 
and generates a structured compliance audit summary.
""")

class ResponsibleAIAuditor:
    """
    Automated Governance & Fairness Auditor calculating disparate impact, 
    evaluating regulatory risk tiers, and verifying deployment gates.
    """
    def __init__(self):
        self.disparate_impact_threshold = 0.80

    def calculate_disparate_impact(self, group_a_total: int, group_a_positive: int, 
                                   group_b_total: int, group_b_positive: int) -> Dict[str, Any]:
        """Calculates selection rates and Disparate Impact Ratio between two demographic groups."""
        rate_a = group_a_positive / max(1, group_a_total)
        rate_b = group_b_positive / max(1, group_b_total)

        # Disparate impact is the ratio of lower rate to higher rate
        if rate_a >= rate_b:
            di_ratio = rate_b / max(1e-5, rate_a)
            higher_group = "Group A"
        else:
            di_ratio = rate_a / max(1e-5, rate_b)
            higher_group = "Group B"

        is_compliant = di_ratio >= self.disparate_impact_threshold

        return {
            "group_a_rate": round(rate_a, 4),
            "group_b_rate": round(rate_b, 4),
            "disparate_impact_ratio": round(di_ratio, 4),
            "higher_selection_group": higher_group,
            "compliant_80_percent_rule": is_compliant,
            "status": "PASS: Impartial Distribution" if is_compliant else "FAIL: Adverse Demographic Impact Detected"
        }

    def evaluate_eu_ai_act_tier(self, use_case_domain: str, has_human_oversight: bool) -> Dict[str, Any]:
        """Classifies use-case into EU AI Act Risk Tier and determines conformity requirements."""
        high_risk_domains = ["hiring", "credit_scoring", "healthcare_triage", "law_enforcement"]
        prohibited_domains = ["social_scoring", "subliminal_manipulation", "real_time_biometric"]

        if use_case_domain in prohibited_domains:
            tier = "UNACCEPTABLE_RISK"
            action = "PROHIBITED: Cannot deploy legally within EU jurisdiction"
        elif use_case_domain in high_risk_domains:
            tier = "HIGH_RISK"
            action = "STRICT_CONFORMITY: Requires technical documentation, CE marking, and mandatory human oversight"
        else:
            tier = "MINIMAL_RISK"
            action = "STANDARD_DEPLOYMENT: Minimal regulatory constraints"

        is_deployable = (tier != "UNACCEPTABLE_RISK") and (tier != "HIGH_RISK" or has_human_oversight)

        return {
            "risk_tier": tier,
            "regulatory_action": action,
            "deployable": is_deployable,
            "requires_human_gate": tier == "HIGH_RISK"
        }

st.code('''
class ResponsibleAIAuditor:
    def calculate_disparate_impact(self, group_a_total: int, group_a_positive: int, 
                                   group_b_total: int, group_b_positive: int) -> Dict[str, Any]:
        # Computes 80% rule disparate impact ratio across demographic subgroups
        ...

    def evaluate_eu_ai_act_tier(self, use_case_domain: str, has_human_oversight: bool) -> Dict[str, Any]:
        # Evaluates regulatory classification under EU AI Act Regulation 2024/1689
        ...
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Responsible AI Fairness & Governance Audit Simulator", divider="rainbow")

st.markdown("""
Simulate an enterprise algorithmic audit. Test how candidate recommendation rates impact the 
**Disparate Impact 80% Rule** and evaluate compliance under **EU AI Act Regulatory Risk Tiers**.
""")

auditor = ResponsibleAIAuditor()

wb_col1, wb_col2 = st.columns([1.1, 0.9])

with wb_col1:
    st.subheader("1. Algorithmic Demographic Inputs")
    st.markdown("**Demographic Group A (Baseline Cohort)**")
    c1, c2 = st.columns(2)
    with c1:
        g_a_total = st.number_input("Group A Total Applicants:", value=1000, step=100)
    with c2:
        g_a_positive = st.number_input("Group A Positive Decisions:", value=600, step=50)

    st.markdown("**Demographic Group B (Comparison Cohort)**")
    c3, c4 = st.columns(2)
    with c3:
        g_b_total = st.number_input("Group B Total Applicants:", value=1000, step=100)
    with c4:
        g_b_positive = st.number_input("Group B Positive Decisions:", value=450, step=50)

    st.markdown("**Governance & Regulatory Scope**")
    domain_choice = st.selectbox("Application Domain:", ["hiring", "credit_scoring", "customer_support", "social_scoring"])
    has_hitl = st.checkbox("Mandatory Human Oversight Enforced in Pipeline", value=True)
    
    run_audit = st.button("⚖️ Execute Responsible AI Compliance Audit", type="primary")

with wb_col2:
    st.subheader("2. Compliance & Audit Telemetry")
    if run_audit or g_a_total:
        di_results = auditor.calculate_disparate_impact(g_a_total, g_a_positive, g_b_total, g_b_positive)
        reg_results = auditor.evaluate_eu_ai_act_tier(domain_choice, has_hitl)

        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Disparate Impact Ratio", di_results["disparate_impact_ratio"], delta=">= 0.80 Pass" if di_results["compliant_80_percent_rule"] else "< 0.80 Fail", delta_color="normal" if di_results["compliant_80_percent_rule"] else "inverse")
        with m2:
            st.metric("EU AI Act Tier", reg_results["risk_tier"])
        with m3:
            st.metric("Deployment Gate", "APPROVED" if (di_results["compliant_80_percent_rule"] and reg_results["deployable"]) else "BLOCKED", delta="Ready" if (di_results["compliant_80_percent_rule"] and reg_results["deployable"]) else "Action Required", delta_color="normal" if (di_results["compliant_80_percent_rule"] and reg_results["deployable"]) else "inverse")

        st.markdown("#### Audit Breakdown")
        st.write(f"• **Group A Selection Rate:** `{di_results['group_a_rate']*100:.1f}%`")
        st.write(f"• **Group B Selection Rate:** `{di_results['group_b_rate']*100:.1f}%`")
        
        if di_results["compliant_80_percent_rule"]:
            st.success(f"Fairness Metric: {di_results['status']}")
        else:
            st.error(f"Fairness Metric: {di_results['status']} (Ratio {di_results['disparate_impact_ratio']} is below 0.80 standard)")

        st.markdown(f"**Regulatory Mandate:** {reg_results['regulatory_action']}")
        if not reg_results["deployable"]:
            st.error("REGULATORY BLOCK: Deployment prohibited under active governance policy.")

st.divider()

# ==============================================================================
# SECTION 10: INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Engineering Interview Questions")

with st.expander("Q1: Explain the mathematical and legal difference between Disparate Treatment and Disparate Impact."):
    st.markdown("""
    **Answer**: 
    * **Disparate Treatment** is intentional discrimination where a model explicitly uses a protected attribute (race, gender) to alter decisions. 
    * **Disparate Impact** is unintentional adverse discrimination where a model appears neutral (omits protected attributes) but uses proxy variables (e.g., zip code, browser type, alumni network) that correlate with protected demographics, resulting in a selection rate ratio below 0.8 (the 80% rule). 
    In modern AI law, disparate impact is equally actionable and requires explicit statistical auditing.
    """)

with st.expander("Q2: Why is Machine Unlearning in foundation model weights technically difficult compared to database deletion?"):
    st.markdown("""
    **Answer**: In relational databases, data records exist as independent rows addressable by primary keys; deletion is an `O(1)` pointer update. 
    In foundation models, pretraining data is distilled into continuous floating-point weight matrices via gradient descent: 
    a single personal record influences millions of non-linear parameter weights across all layers. 
    Selectively erasing a record without catastrophic forgetting of general reasoning requires specialized techniques like 
    **Task Vector Negation** or **Negative Gradient Fine-Tuning**, which remain active research disciplines.
    """)

with st.expander("Q3: What are the primary conformity assessment requirements for High-Risk AI systems under the EU AI Act?"):
    st.markdown("""
    **Answer**: Under EU AI Act Title III, High-Risk systems (e.g., hiring, credit assessment, medical diagnostic tools) must satisfy:
    1. A continuous, documented **Risk Management System** updated across the lifecycle.
    2. Rigorous data governance verifying training data relevance and bias auditing.
    3. Comprehensive **Technical Documentation** and automatic event logging throughout operations.
    4. Meaningful **Human Oversight (HITL)** interfaces enabling human operators to override automated decisions.
    5. High standards of cybersecurity, accuracy, and robustness verified prior to CE marking.
    """)

with st.expander("Q4: What is a Model Card and why is it essential for enterprise model governance?"):
    st.markdown("""
    **Answer**: Proposed by Mitchell et al. (Google Research), a Model Card is a standardized, public documentation artifact 
    that accompanies trained machine learning models. It records the model's intended use cases, architecture, training data composition, 
    quantitative performance broken down across demographic subgroups, known limitations, and ethical considerations. 
    In enterprise governance, model cards serve as the foundational audit artifact for risk assessment boards.
    """)

with st.expander("Q5: How do Sparse Autoencoders (SAEs) advance mechanistic interpretability in large transformers?"):
    st.markdown("""
    **Answer**: Standard transformer activations are *polysemantic*: a single neuron fires for dozens of unrelated concepts due to superposition. 
    **Sparse Autoencoders** project dense hidden representations into a higher-dimensional sparse feature space (e.g., 32x wider) with an L1 sparsity penalty. 
    This disentangles polysemantic neurons into monosemantic, human-interpretable features (e.g., a specific feature representing 'suspicion of fraud' or 'deceptive roleplay'), allowing researchers to audit and steer internal model thinking.
    """)

st.subheader("Critical Production Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Removing Protected Attributes and Assuming Bias is Eliminated**
    * *Error*: Dropping `'gender'` and `'race'` columns from training datasets.
    * *Impact*: The model learns proxy associations (hobbies, schools, zip codes) and continues to exhibit disparate impact.
    * *Fix*: Proactively audit output distributions across counterfactual demographic groups.
    """)
    st.error("""
    **Trap 2: Treating CoT Reasoning as Guaranteed Faithfulness**
    * *Error*: Assuming that an emitted Chain-of-Thought explains the model's true internal causal weights.
    * *Impact*: The model generates deceptive post-hoc rationalizations to justify biased decisions.
    * *Fix*: Combine CoVe / CoT audits with mechanistic activation probes and perturbation testing.
    """)

with trap_c2:
    st.warning("""
    **Trap 3: Omitting Human Override Mechanisms in High-Risk Workflows**
    * *Warning*: Fully automating resume screening or loan rejection without an appeal channel.
    * *Impact*: High regulatory liability under EU AI Act Article 14 and severe customer trust erosion.
    * *Fix*: Implement accessible human-in-the-loop review queues for adverse determinations.
    """)
    st.warning("""
    **Trap 4: Static One-Time Bias Auditing Prior to Release**
    * *Warning*: Evaluating demographic parity once during development and never re-testing.
    * *Impact*: Real-world data drift and updated system prompts introduce latent algorithmic bias over time.
    * *Fix*: Maintain continuous automated fairness telemetry in production dashboards.
    """)

st.subheader("Concept Flashcards")
fc1, fc2 = st.columns(2)
with fc1:
    with st.expander("🃏 Flashcard: 80% Rule (Disparate Impact)"):
        st.markdown("**Definition**: Selection rate for minority group must be at least 80% of the selection rate for the majority group.\\n\\n**Formula**: `Rate_Minority / Rate_Majority >= 0.80`.")
with fc2:
    with st.expander("🃏 Flashcard: Monosemanticity"):
        st.markdown("**Concept**: An individual artificial neuron or feature responding exclusively to a single, unambiguous human concept.\\n\\n**Discovery**: Enabled by Sparse Autoencoders (SAEs) in mechanistic interpretability.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of Responsible AI, algorithmic fairness, and governance architecture:")

chk1 = st.checkbox("I can calculate the Disparate Impact ratio and evaluate compliance under the legal 80% rule.", value=True)
chk2 = st.checkbox("I understand the components of standardized Model Cards and Data Nutrition Sheets.", value=True)
chk3 = st.checkbox("I can explain why removing protected attributes fails to prevent algorithmic bias due to proxy features.", value=True)
chk4 = st.checkbox("I understand the difference between mechanistic interpretability (internal probes) and post-hoc CoT rationalization.", value=True)
chk5 = st.checkbox("I can classify enterprise AI use-cases into the EU AI Act risk tiers (Unacceptable, High, Specific, Minimal).", value=True)
chk6 = st.checkbox("I understand the challenges of Machine Unlearning under GDPR 'Right to be Forgotten' mandates.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Milestone Achieved: You have mastered Responsible AI & Governance Architecture! Proceed to Chapter 23: LLM Application Development.")
