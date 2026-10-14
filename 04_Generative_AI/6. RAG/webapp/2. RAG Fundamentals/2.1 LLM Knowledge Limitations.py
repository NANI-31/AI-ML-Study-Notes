import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="2.1 LLM Knowledge Limitations - RAG Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title Block & 5-Pillar Overview
st.title("2.1 LLM Knowledge Limitations: Cutoffs, Hallucinations, & The Parametric Frontier")
st.caption("Chapter 2: RAG Fundamentals | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
A foundational understanding of Retrieval-Augmented Generation begins with an unvarnished audit of Large Language Model 
structural limitations. An LLM is not an infallible enterprise database—it is a statistical next-token predictor 
trained on a snapshot of public internet text. 

When organizations deploy raw foundation models for mission-critical operations, they inevitably collide with six 
inherent parametric limitations: **knowledge cutoff**, **blindness to private data**, **stochastic hallucinations**, 
**finite context windows**, **inability to handle dynamic streaming data**, and **shallow long-tail domain expertise**. 
RAG is the architectural cure for every single one of these vulnerabilities.

In this lesson, you will master:
1. The computational origins of parametric memory failure modes in deep neural networks.
2. Why knowledge cutoff and temporal drift invalidate standalone LLMs in fast-moving industries.
3. The mechanics of hallucination: semantic plausibility vs. factual grounding.
4. Attention dilution, token window limits, and the Needle-in-a-Haystack benchmark.
5. Production-grade guardrails and evaluation strategies to detect and neutralize parametric failures.
""")

st.divider()

# Module-Scope Helper Classes for Interactive Workbench & Production Pipeline
class ParametricFailureSimulator:
    def __init__(self):
        self.scenarios = {
            "Temporal Knowledge Cutoff": {
                "query": "What are the key provisions of the 2026 Global AI Safety Treaty?",
                "parametric_response": "As of my knowledge cutoff in 2024, no global AI treaty exists. Major frameworks include the EU AI Act of 2024 and executive orders.",
                "rag_response": "The 2026 Global AI Safety Treaty signed in Geneva mandates real-time watermarking on models above 10^26 FLOPs and establishes an international incident response bureau [Citation: UN_Treaty_2026_Art4].",
                "vulnerability": "Model weights frozen at training time cannot know post-cutoff legal statutes."
            },
            "Private Enterprise Data": {
                "query": "What is our internal Q3 2026 customer churn rate for Tier-1 accounts?",
                "parametric_response": "SaaS industry benchmark churn rates typically range from 3% to 7% annually for enterprise tiers.",
                "rag_response": "According to the Q3 2026 Customer Operations Ledger, Tier-1 customer churn was 1.4%, outperforming target guidance [Citation: RevOps_Q3_2026.xlsx, Row 44].",
                "vulnerability": "Proprietary internal metrics do not exist in the public pretraining corpus."
            },
            "Plausible Hallucination": {
                "query": "What was the ruling in the landmark 2025 cyber-tort case Mitchell v. DataCore?",
                "parametric_response": "In Mitchell v. DataCore (2025), the 9th Circuit held that cloud hosting providers bear strict liability for zero-day data exfiltrations.",
                "rag_response": "Search across Federal and State legal databases returned ZERO records matching 'Mitchell v. DataCore'. The case appears to be a fabricated hallucination.",
                "vulnerability": "Self-attention generates fluent legal citations that do not exist in reality."
            },
            "Dynamic Live Data": {
                "query": "What is the current server status of the European API gateway cluster?",
                "parametric_response": "All public cloud European regions operate with standard 99.99% uptime availability.",
                "rag_response": "Live telemetry at 15:20 UTC: European cluster gateway node EU-CENTRAL-3 is experiencing a degraded network packet drop of 4.2% [Citation: Prometheus_Live_Metrics].",
                "vulnerability": "Real-time infrastructure health cannot be captured in static weights."
            }
        }

    def evaluate_query(self, scenario_name: str) -> Dict[str, str]:
        return self.scenarios.get(scenario_name, {})

# ==============================================================================
# 1. Architecture Blueprint
# ==============================================================================
st.header("1. Architecture Blueprint: The Parametric vs. Non-Parametric Knowledge Frontier", divider="rainbow")
st.markdown("""
The diagram below maps how incoming queries trigger either **Parametric Failure Modes** when relying on raw weights, 
or **Grounded Resolution** when augmented by external Non-Parametric memory in a clean top-to-bottom layout:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_limitations = """
    digraph KnowledgeFrontier {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_input {
            label="1. Query Ingress Tier";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            q [label="User Enterprise Query\\n(Private / Dynamic / Post-Cutoff)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_raw {
            label="2. Unaugmented Path (Parametric Weights Alone)";
            bgcolor="#1E293B";
            fontcolor="#EF4444";
            color="#7F1D1D";
            fontsize=12;
            raw_llm [label="Frozen Weights Alone\\nStatic Pretraining Snapshot", fillcolor="#450A0A", fontcolor="#FCA5A5", color="#EF4444"];
            f1 [label="Knowledge Cutoff Failure\\n(Temporal Blindness)", fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];
            f2 [label="Blind to Private Enterprise Data\\n(Proprietary Ignorance)", fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];
            f3 [label="Confident Hallucination\\n(Plausible Fabrications)", fillcolor="#7F1D1D", fontcolor="#FEE2E2", color="#EF4444"];
            raw_llm -> f1;
            raw_llm -> f2;
            raw_llm -> f3;
        }

        subgraph cluster_rag {
            label="3. RAG Grounding Path (Non-Parametric Storage)";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#064E3B";
            fontsize=12;
            retriever [label="Dynamic Search & Retrieval\\nVector DB, BM25 Index, Live Operational APIs", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            grounded_llm [label="Grounded Synthesis Engine\\nConstrained Prompt Contract with Source Citations", fillcolor="#047857", fontcolor="#ECFDF5", color="#10B981"];
            resp [label="Factually Grounded Response\\n100% Verifiable Provenance Citations", fillcolor="#022C22", fontcolor="#A7F3D0", color="#10B981"];
            retriever -> grounded_llm -> resp;
        }

        q -> raw_llm [label="Without Retrieval", color="#EF4444"];
        q -> retriever [label="With Retrieval", color="#10B981"];
    }
    """
    st.graphviz_chart(dot_limitations, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Even frontier models with massive reasoning capacities (Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro) explicitly declare: 
**reasoning cannot substitute for missing information**. Frontier AI platforms now integrate native search tools 
and grounding layers so that whenever a prompt involves temporal references or proprietary domains, the model 
autonomously triggers a retrieval tool rather than guessing from its internal weights.
""")

st.divider()

# ==============================================================================
# 2. Knowledge Cutoff
# ==============================================================================
st.header("2. Knowledge Cutoff & Temporal Invalidation", divider="rainbow")
st.markdown("""
Every LLM pretraining run has an immutable **training cutoff timestamp**. From the microsecond pretraining completes, 
the model begins aging:
* **The Decay of Factual Relevance**: Laws are amended, stock prices fluctuate, API interfaces deprecate, and new security vulnerabilities emerge daily.
* **Why Retraining Fails**: Full pretraining of an enterprise model costs hundreds of thousands of dollars and weeks of compute. Organizations cannot retrain a foundation model daily to track corporate memos.
* **The RAG Solution**: RAG decouples knowledge freshness from compute cycles. When a new policy is published at 09:00 AM, it is embedded and searchable across the entire enterprise by 09:01 AM.
""")

st.divider()

# ==============================================================================
# 3. Missing Private Information
# ==============================================================================
st.header("3. Missing Private & Proprietary Enterprise Information", divider="rainbow")
st.markdown("""
Public foundation models are trained on public datasets (Common Crawl, Wikipedia, GitHub, arXiv). They have **zero exposure** 
to private enterprise information:
""")

df_private_info = pd.DataFrame([
    {"Information Class": "Proprietary IP", "Why Missing from LLM": "Trade secrets, proprietary algorithms, CAD files.", "RAG Grounding Strategy": "Encrypted vector indices inside enterprise VPC."},
    {"Information Class": "Customer Telemetry", "Why Missing from LLM": "Protected by GDPR, CCPA, and strict privacy boundaries.", "RAG Grounding Strategy": "Metadata-filtered vector search with tenant isolation."},
    {"Information Class": "Internal HR & Policies", "Why Missing from LLM": "Stored behind intranet firewalls (SharePoint, Notion).", "RAG Grounding Strategy": "Role-based access control (RBAC) vector connectors."},
    {"Information Class": "Financial & Operational Ledgers", "Why Missing from LLM": "Sensitive quarterly financials, tax ledgers, ERP records.", "RAG Grounding Strategy": "Structured Text-to-SQL + Hybrid Vector RAG."}
])
st.dataframe(df_private_info, use_container_width=True)

st.divider()

# ==============================================================================
# 4. Hallucinations
# ==============================================================================
st.header("4. Hallucinations: Plausible Stochastic Fabrication", divider="rainbow")
st.markdown("""
**Hallucination** is not an accidental defect in neural networks—it is the direct computational consequence of 
**statistical next-token prediction**. 

When an LLM lacks exact factual evidence in its active context, the attention mechanism selects whichever token 
maximizes linguistic fluency:
* In legal tasks, it fabricates realistic court names, docket numbers, and judicial precedents.
* In medical tasks, it generates plausible drug dosages that have never been clinically validated.
* In engineering, it creates imaginary library functions that fail at runtime.

RAG replaces statistical guesswork with **strict in-context evidence bounding**.
""")

with st.expander("🔍 View Hallucination Elimination Mechanism Flowchart", expanded=True):
    dot_hallucination = """
    digraph HallucinationMitigation {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        q [label="Query on Niche Topic\\n(e.g., Internal Microservice Config)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        check [label="Retriever Context Check\\nIs evidence found in index?", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        found [label="Evidence Chunks Present\\nCondition prompt strictly on chunks", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        missing [label="Zero Matching Chunks\\nTrigger Negative Constraint Abstention", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        
        grounded_out [label="Grounded Synthesis with Exact Citations", fillcolor="#047857", fontcolor="#ECFDF5", color="#10B981"];
        abstain_out [label="Honest Abstention:\\n'Information not found in enterprise records'", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];

        q -> check;
        check -> found [label="Retrieved Chunks >= 1"];
        check -> missing [label="No Chunks Found"];
        found -> grounded_out;
        missing -> abstain_out;
    }
    """
    st.graphviz_chart(dot_hallucination, use_container_width=True)

st.divider()

# ==============================================================================
# 5. Context Limitations
# ==============================================================================
st.header("5. Context Limitations & The Attention Needle-in-a-Haystack Trap", divider="rainbow")
st.markdown("""
Modern models boast context windows ranging from 128k to 2M tokens. However, stuffing entire document archives 
into a massive prompt introduces severe architectural penalties:
* **The Lost-in-the-Middle Phenomenon**: Decoder-only attention maps concentrate probability mass on prompt headers and endings. Evidence buried in the middle of 200,000 tokens suffers from severe attention degradation.
* **Cost Scaling**: Pricing scales linearly with every single token ingested. Submitting 500,000 tokens for every user query costs substantial fees per API call.
* **Latency Penalties**: Time-to-First-Token (TTFT) scales with prompt length, pushing response latency from 600ms to 15+ seconds.
""")

st.divider()

# ==============================================================================
# 6. Dynamic Information
# ==============================================================================
st.header("6. Dynamic & Continuously Changing Information", divider="rainbow")
st.markdown("""
Enterprise operations depend on live, high-velocity streaming data: inventory counts, stock prices, flight schedules, 
and server metrics. 
* Parametric models cannot represent data that changes from minute to minute.
* **RAG with Live Ingestion**: By combining vector search with real-time API tools (Function Calling / Tool Use), RAG seamlessly blends historical reference documents with live operational metrics.
""")

st.divider()

# ==============================================================================
# 7. Domain-Specific Knowledge
# ==============================================================================
st.header("7. Domain-Specific Jargon & Long-Tail Knowledge", divider="rainbow")
st.markdown("""
General pretraining datasets suffer from severe power-law distribution skew:
""")

col_skew1, col_skew2 = st.columns(2)
with col_skew1:
    st.subheader("The Head: Well-Represented Public Data")
    st.markdown("""
    * Millions of pages discussing Python syntax, World War II, and pop culture.
    * The LLM's parametric recall is strong, accurate, and fluent on these topics.
    """)
with col_skew2:
    st.subheader("The Long Tail: Specialized Domain Data")
    st.markdown("""
    * Niche aerospace stress-test formulas, specialized biotech enzyme binding protocols, localized tax codes.
    * The LLM contains few training tokens for these topics, leading to high error rates and subtle hallucinations.
    """)

st.markdown("""
RAG democratizes domain specialization: an enterprise can inject millions of specialized proprietary research papers 
into a vector database, giving a generic foundation model world-class domain precision without fine-tuning.
""")

st.divider()

# ==============================================================================
# 8. Production Implementation
# ==============================================================================
st.header("8. Production Implementation: Hallucination Detection & Temporal Guardrail Engine", divider="rainbow")
st.markdown("""
Below is a runnable Python production implementation of a **Hallucination Detection & Guardrail Engine** that 
verifies whether generated response assertions are grounded in the retrieved context:
""")

prod_guardrail_code = """
from typing import List, Dict, Any

class GroundednessVerificationEngine:
    def __init__(self, token_overlap_threshold: float = 0.55):
        self.token_overlap_threshold = token_overlap_threshold

    def extract_claims(self, generated_text: str) -> List[str]:
        \"\"\"Splits generated output into atomic factual sentences.\"\"\"
        sentences = [s.strip() for s in generated_text.split(".") if len(s.strip()) > 10]
        return sentences

    def verify_groundedness(self, context_chunks: List[str], generated_text: str) -> Dict[str, Any]:
        claims = self.extract_claims(generated_text)
        all_context_tokens = set(" ".join(context_chunks).lower().split())
        
        claim_scores = []
        for claim in claims:
            claim_tokens = set(claim.lower().split())
            overlap = len(claim_tokens.intersection(all_context_tokens))
            grounding_ratio = overlap / (len(claim_tokens) + 1e-6)
            is_grounded = grounding_ratio >= self.token_overlap_threshold
            claim_scores.append({
                "claim": claim,
                "overlap_score": round(grounding_ratio, 3),
                "status": "PASS" if is_grounded else "POTENTIAL_HALLUCINATION"
            })

        overall_grounded = all(c["status"] == "PASS" for c in claim_scores)
        return {
            "is_fully_grounded": overall_grounded,
            "total_claims": len(claims),
            "claim_breakdown": claim_scores
        }
"""
st.code(prod_guardrail_code, language="python")

st.divider()

# ==============================================================================
# 9. Interactive Workbench
# ==============================================================================
st.header("9. Interactive Workbench: Parametric Failure & Knowledge Augmentation Studio", divider="rainbow")
st.markdown("""
Test real enterprise scenarios comparing raw **Parametric Model Output** against a **Grounded RAG Pipeline**:
""")

simulator = ParametricFailureSimulator()

selected_scenario = st.selectbox(
    "Select Failure Scenario to Simulate:",
    list(simulator.scenarios.keys())
)

scenario_data = simulator.evaluate_query(selected_scenario)

st.markdown(f"**Enterprise Prompt**: `{scenario_data['query']}`")
st.caption(f"Underlying Vulnerability: {scenario_data['vulnerability']}")

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.subheader("❌ Raw Parametric Model (Unaugmented)")
    st.error(scenario_data["parametric_response"])
    st.caption("Result: Severe Hallucination / Missing Fact / Stale Information")

with col_v2:
    st.subheader("✅ Grounded RAG Pipeline (Non-Parametric)")
    st.success(scenario_data["rag_response"])
    st.caption("Result: 100% Factually Grounded with Exact Provenance Citation")

st.divider()

# ==============================================================================
# 10. Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Why is hallucination an intrinsic mathematical property of autoregressive LLMs?"):
    st.markdown("""
    **Answer**: Autoregressive models compute token probabilities conditioned on preceding tokens. They optimize 
    for likelihood over training data, not factual truth. In the absence of direct factual evidence in the prompt, 
    the model naturally predicts tokens that maintain linguistic smoothness and stylistic plausibility.
    """)

with st.expander("Q2: How does RAG solve the 'Needle-in-a-Haystack' problem in large context models?"):
    st.markdown("""
    **Answer**: While a model may accept 1M tokens, dense empirical evaluations prove retrieval accuracy degrades when 
    critical facts are surrounded by irrelevant text. RAG uses vector similarity and cross-encoder reranking to extract 
    only the 3-5 relevant 'needles', presenting a compact, high-signal prompt that maximizes attention fidelity.
    """)

with st.expander("Q3: What is the difference between Extrinsic and Intrinsic hallucination?"):
    st.markdown("""
    **Answer**: **Intrinsic Hallucination** occurs when the generated text directly contradicts the provided source context. 
    **Extrinsic Hallucination** occurs when the generated text introduces claims that can neither be confirmed nor contradicted 
    by the context (speculative additions). RAG prompt contracts are engineered to prevent both.
    """)

with st.expander("Q4: Why can't continual fine-tuning eliminate knowledge cutoff issues?"):
    st.markdown("""
    **Answer**: Continual fine-tuning suffers from **Catastrophic Forgetting** (overwriting previously learned capabilities), 
    is computationally prohibitive for daily updates, and cannot produce deterministic document page citations.
    """)

with st.expander("Q5: What is the role of an NLI (Natural Language Inference) model in RAG evaluation?"):
    st.markdown("""
    **Answer**: NLI models evaluate factual entailment by treating the retrieved chunk as the Premise and each generated sentence 
    as the Hypothesis. If the premise entails the hypothesis, the claim is verified as grounded; if neutral or contradicted, 
    a hallucination alert is triggered.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The Sycophantic Grounding Trap)**: When retrieved chunks contain incorrect or spoofed information, models tend to uncritically regurgitate the error, believing the context unconditionally over truth.")
    st.warning("**Trap 2 (Over-Reliance on Large Context Windows as a Substitute for RAG)**: Stuffing full 500-page manuals into every query increases latency by 10x and API token costs by 50x compared to precision chunk retrieval.")
with col_trap2:
    st.error("**Trap 3 (Temporal Inconsistency in Ingestion)**: Failing to timestamp document chunks, causing the retriever to pull an old 2021 HR policy chunk alongside a 2026 update, confusing the LLM with contradictory instructions.")
    st.warning("**Trap 4 (Silent Failure to Abstain)**: Failing to test whether the LLM will hallucinate when the retriever returns zero relevant chunks. Without explicit negative constraints, the model will guess rather than say 'I do not know'.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is Parametric Memory?")
        st.caption("Flip for answer:")
        st.info("**Back**: Knowledge encoded in billions of static neural network parameters during pretraining. It cannot cite sources and is frozen at training cutoff.")
    with st.container(border=True):
        st.markdown("**Front**: What is the difference between Intrinsic and Extrinsic Hallucination?")
        st.caption("Flip for answer:")
        st.info("**Back**: Intrinsic directly contradicts the provided prompt context. Extrinsic introduces ungrounded external claims that cannot be verified from the context.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is Non-Parametric Memory?")
        st.caption("Flip for answer:")
        st.info("**Back**: External, real-time indexable data stores (Vector DB, BM25, SQL) that are auditable, dynamic, and updatable without retraining.")
    with st.container(border=True):
        st.markdown("**Front**: What is the Attention Needle-in-a-Haystack Trap?")
        st.caption("Flip for answer:")
        st.info("**Back**: The degradation of factual recall when critical evidence chunks are buried in the middle of massive 100k+ token context windows.")

st.divider()

# ==============================================================================
# 11. Self-Assessment Checklist
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I can enumerate the six core parametric limitations of standalone LLMs.", value=True)
c2 = st.checkbox("I understand why next-token prediction causes hallucinations when facts are missing.", value=True)
c3 = st.checkbox("I know why large context windows do not eliminate the need for precision retrieval (Lost-in-the-Middle).", value=True)
c4 = st.checkbox("I understand the danger of temporal drift and why document chunks must include timestamp metadata.", value=True)
c5 = st.checkbox("I can implement an automated groundedness check to detect ungrounded factual claims.", value=True)
c6 = st.checkbox("I know how to write strict prompt contracts that instruct the model to abstain when context is missing.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Module 2.1 Complete! You are ready to advance to Lesson 2.2: RAG Workflow.")
