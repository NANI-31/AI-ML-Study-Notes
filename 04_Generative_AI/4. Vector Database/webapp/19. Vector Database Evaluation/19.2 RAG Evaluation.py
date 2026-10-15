import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="19.2 RAG Evaluation - Vector DB Curriculum",
    page_icon="🧪",
    layout="wide"
)

# Title & Overview
st.title("19.2 RAG Evaluation: The RAG Triad, Ragas Framework, Faithfulness & Groundedness")
st.caption("Chapter 19: Vector Database Evaluation | Domain: 4. Vector Database")
st.markdown("""
Traditional software testing relies on unit tests with deterministic assertions (`assert response == expected`). 
However, in Retrieval-Augmented Generation (RAG) systems powered by vector databases, answers are probabilistic, 
non-deterministic, and synthesized from unstructured text.

To ensure production safety without relying on expensive human review, the industry has standardized on **The RAG Triad** 
and automated evaluation frameworks like **Ragas** and **TruLens**.

The RAG Triad evaluates the three fundamental legs of information flow:
1. **Context Relevance**: Did the vector database retrieve passages that directly address the user's question?
2. **Faithfulness / Groundedness**: Is every factual claim in the generated answer strictly traceable to the retrieved context? (Zero Hallucinations).
3. **Answer Relevance**: Does the final synthesized text directly answer what the user asked without conversational tangents?

In this lesson, we explore the mathematics and LLM-as-a-Judge heuristics behind Ragas metrics, examine context precision 
and recall, and test an interactive Live RAG Triad Diagnostic Workbench.
""")

st.divider()

# Section 1: The RAG Triad Architecture
st.header("1. The RAG Triad Architecture", divider="rainbow")

col_tri1, col_tri2 = col_tri1, col_tri2 = st.columns([1.1, 0.9])

with col_tri1:
    st.markdown("""
    ### The Three Invariant Pillars of RAG Quality
    
    * **1. Context Relevance (Question -> Retrieved Context)**:
      * *Question Evaluated*: *"Is the retrieved context signal-dense and free of irrelevant noise?"*
      * *Failure Mode*: Retrieving 5 chunks about general corporate history when the user asked about database failover.
      * *Component Responsible*: Vector Database & Retriever.
    
    * **2. Groundedness / Faithfulness (Retrieved Context -> Answer)**:
      * *Question Evaluated*: *"Can every statement in the answer be proven using only the retrieved context?"*
      * *Failure Mode*: The LLM hallucinates an unstated fact or pulls knowledge from its pre-training weights that contradicts the company manual.
      * *Component Responsible*: Large Language Model (LLM).
    
    * **3. Answer Relevance (Question -> Answer)**:
      * *Question Evaluated*: *"Does the response directly satisfy the user's specific request?"*
      * *Failure Mode*: Answering a completely different question or generating evasive pleasantries.
      * *Component Responsible*: LLM System Prompt & Guardrails.
    """)

with col_tri2:
    st.markdown("### The RAG Triad of Trust")
    triad_dot = """
    digraph RAGTriad {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="User Query", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Context [label="Retrieved Context\\n(Vector DB Chunks)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        Answer [label="Generated Answer\\n(LLM Response)", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];

        UserQ -> Context [label="1. Context Relevance\\n(Is context useful?)", color="#D97706", style="bold"];
        Context -> Answer [label="2. Faithfulness / Groundedness\\n(Are facts grounded?)", color="#059669", style="bold"];
        UserQ -> Answer [label="3. Answer Relevance\\n(Did we answer user?)", color="#2563EB", style="bold", dir=both];
    }
    """
    st.graphviz_chart(triad_dot, use_container_width=False)

st.divider()

# Section 2: Deep-Dive into the Ragas Framework Metric Suite
st.header("2. Deep-Dive into the Ragas Framework Metric Suite", divider="rainbow")

st.markdown("""
The **Ragas (Retrieval Augmented Generation Assessment)** framework expands the RAG Triad into six rigorous, 
reproducible automated metrics:
""")

RAGAS_METRICS = [
    {
        "Metric": "Faithfulness",
        "Target Component": "LLM Generator",
        "Mathematical Definition": "Faithfulness = (Number of Claims Supported by Context) / (Total Claims in Answer)",
        "Range": "0.0 - 1.0",
        "Production Target": ">= 0.95 (Zero tolerance for hallucinations)"
    },
    {
        "Metric": "Answer Relevance",
        "Target Component": "LLM Generator",
        "Mathematical Definition": "Cosine similarity between original query embedding and synthetic questions generated from the answer.",
        "Range": "0.0 - 1.0",
        "Production Target": ">= 0.85 (Direct, focused responses)"
    },
    {
        "Metric": "Context Precision",
        "Target Component": "Vector Store / Reranker",
        "Mathematical Definition": "Evaluates whether all ground-truth relevant chunks are ranked at the top of context: Context_Precision@K = sum(Precision@k * v_k) / Total_Relevant_Chunks.",
        "Range": "0.0 - 1.0",
        "Production Target": ">= 0.80 (Signal before noise)"
    },
    {
        "Metric": "Context Recall",
        "Target Component": "Vector Store / Retriever",
        "Mathematical Definition": "Fraction of ground-truth reference sentences that can be attributed to the retrieved context chunks.",
        "Range": "0.0 - 1.0",
        "Production Target": ">= 0.90 (Complete knowledge capture)"
    },
    {
        "Metric": "Context Relevance",
        "Target Component": "Vector Store / Retriever",
        "Mathematical Definition": "(Number of Sentence Chunks Relevant to Question) / (Total Sentence Chunks in Context)",
        "Range": "0.0 - 1.0",
        "Production Target": ">= 0.70 (Low context dilution)"
    }
]

st.dataframe(pd.DataFrame(RAGAS_METRICS), use_container_width=True, hide_index=True)

st.divider()

# Section 3: LLM-as-a-Judge Prompting Architecture
st.header("3. LLM-as-a-Judge Prompting Architecture", divider="rainbow")

col_judge1, col_judge2 = col_judge1, col_judge2 = st.columns([1, 1])

with col_judge1:
    st.markdown("""
    ### How Automated Judges Verify Groundedness
    To calculate **Faithfulness** programmatically without humans:
    
    1. **Step 1 (Claim Extraction)**: An evaluator model decomposes the generated answer into atomic factual claims:
       * *Answer*: *"Envoy enforces mTLS using SPIFFE certs with a 1-hour TTL."*
       * *Claim 1*: Envoy enforces mutual TLS.
       * *Claim 2*: Envoy uses SPIFFE certificates.
       * *Claim 3*: Certificates have a 1-hour TTL.
    2. **Step 2 (Verifiable Alignment)**: For each claim, the evaluator verifies whether the claim is directly supported by the retrieved context.
    3. **Step 3 (Mathematical Score)**: If 3 out of 3 claims are supported, `Faithfulness = 3 / 3 = 1.0`.
    """)

with col_judge2:
    st.markdown("### The Production Faithfulness Evaluation Prompt")
    st.code("""EVAL_PROMPT = \"\"\"
You are an expert AI judge evaluating RAG faithfulness.
Analyze the following retrieved context and generated answer.

Retrieved Context:
<context>
{context}
</context>

Generated Answer:
<answer>
{answer}
</answer>

Instructions:
1. Break down the generated answer into atomic factual statements.
2. For each statement, determine whether it is strictly supported by the context above (YES or NO).
3. If unsupported, explain the exact discrepancy.
4. Output JSON:
{
  "statements": [
    {"statement": "...", "supported": true, "reason": "..."}
  ],
  "faithfulness_score": 1.0
}
\"\"\"
""", language="text")

st.divider()

# Section 4: Interactive Workbench: Live RAG Triad Diagnostic Workbench
st.header("4. Interactive Workbench: Live RAG Triad Diagnostic Workbench", divider="rainbow")

st.markdown("""
Inspect a simulated RAG interaction. Test scenarios where the system suffers from **Hallucinations**, 
**Missing Context**, or **Irrelevant Tangents**, and observe how the RAG Triad diagnoses the exact faulty component.
""")

SCENARIOS = {
    "1. Perfectly Grounded & Relevant (All High)": {
        "q": "What is the continuous WAL archiving retention period in our database?",
        "context": "Continuous write-ahead logging (WAL) streams database changes to Amazon S3. Point-In-Time Recovery (PITR) is maintained for exactly 35 days under company SLA.",
        "answer": "Our database WAL archives are streamed to Amazon S3 with Point-In-Time Recovery maintained for exactly 35 days.",
        "unsupported_claim": None,
        "ctx_rel": 0.95, "faith": 1.00, "ans_rel": 0.98
    },
    "2. Severe Hallucination / Unfaithful (Low Groundedness)": {
        "q": "What is the continuous WAL archiving retention period in our database?",
        "context": "Continuous write-ahead logging (WAL) streams database changes to Amazon S3. Point-In-Time Recovery (PITR) is maintained for exactly 35 days under company SLA.",
        "answer": "Our database WAL archives are retained for 35 days, and all archives are automatically duplicated to Google Cloud Storage for redundancy.",
        "unsupported_claim": "Archives are automatically duplicated to Google Cloud Storage (Not mentioned in context!)",
        "ctx_rel": 0.95, "faith": 0.50, "ans_rel": 0.92
    },
    "3. Irrelevant Context / Low Recall (Retriever Failure)": {
        "q": "What is the continuous WAL archiving retention period in our database?",
        "context": "PostgreSQL connection pooling is managed via PgBouncer with a maximum pool limit of 200 concurrent connections.",
        "answer": "Connection pooling is managed by PgBouncer with up to 200 concurrent connections.",
        "unsupported_claim": "Did not answer the question about WAL retention!",
        "ctx_rel": 0.15, "faith": 1.00, "ans_rel": 0.20
    }
}

col_scen_s, col_scen_disp = col_scen_s, col_scen_disp = st.columns([1, 1.2])

with col_scen_s:
    st.subheader("1. Select Diagnostic Scenario")
    active_scen_name = st.selectbox("Scenario:", list(SCENARIOS.keys()))
    scen_data = SCENARIOS[active_scen_name]

with col_scen_disp:
    st.subheader("2. Interaction Artifacts")
    st.info(f"**User Question**: *\"{scen_data['q']}\"*")
    st.text_area("Retrieved Context Chunks (Vector DB):", value=scen_data["context"], height=80)
    st.text_area("Generated Answer (LLM Output):", value=scen_data["answer"], height=80)

st.divider()

col_diag_scores, col_diag_verdict = st.columns([1, 1])

with col_diag_scores:
    st.subheader("Calculated RAG Triad Scores")
    
    c1, c2, c3 = st.columns(3)
    c1.metric(label="Context Relevance", value=f"{scen_data['ctx_rel']*100:.0f}%")
    c2.metric(label="Faithfulness", value=f"{scen_data['faith']*100:.0f}%")
    c3.metric(label="Answer Relevance", value=f"{scen_data['ans_rel']*100:.0f}%")

with col_diag_verdict:
    st.subheader("System Diagnostic Root Cause")
    
    if scen_data["faith"] < 0.8:
        st.error(f"🚨 **HALLUCINATION DETECTED!** Faithfulness dropped to {scen_data['faith']*100:.0f}%.")
        st.warning(f"Unsupported Claim: **\"{scen_data['unsupported_claim']}\"**")
        st.markdown("**Remedy**: Strengthen LLM system prompt: *'Answer strictly using the text. If not stated, say you do not know.'*")
    elif scen_data["ctx_rel"] < 0.5:
        st.error(f"🚨 **RETRIEVAL FAILURE!** Context Relevance collapsed to {scen_data['ctx_rel']*100:.0f}%.")
        st.warning("The vector database retrieved completely irrelevant connection pooling chunks instead of WAL retention chunks.")
        st.markdown("**Remedy**: Check embedding model alignment, tune HNSW `efSearch`, or add Hybrid BM25 keyword search.")
    else:
        st.success("🎉 **VERIFIED SYSTEM HEALTH**: All RAG Triad metrics exceed production thresholds. 100% grounded and focused.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the RAG Triad and what distinct failure mode does each metric isolate?"):
    st.markdown("""
    **Answer**:
    The RAG Triad isolates the exact architectural layer responsible for an erroneous response:
    1. **Context Relevance**: Evaluates whether the **Retriever / Vector DB** pulled signal instead of noise. If low, the retriever is broken.
    2. **Faithfulness (Groundedness)**: Evaluates whether the **LLM is hallucinating**. If low, the LLM is adding claims not supported by the context.
    3. **Answer Relevance**: Evaluates whether the **LLM followed user instructions**. If low, the LLM answered an unrelated question or went off on a conversational tangent.
    """)

with st.expander("Q2: How does Ragas compute the Answer Relevance metric mathematically?"):
    st.markdown("""
    **Answer**:
    Ragas uses the **Reverse Question Generation Heuristic**:
    1. It prompts an LLM to generate `N` potential questions (e.g. 3 questions) that the generated answer would naturally resolve.
    2. It embeds each synthetic question and the user's original question using an embedding model.
    3. It calculates the mean cosine similarity between the original question vector and the synthetic question vectors:
       `Answer_Relevance = (1 / N) * sum( Cosine_Sim(vec(Q_original), vec(Q_synthetic_i)) )`.
    * If the answer went off on a tangent, the synthetic questions will diverge from the original query, producing a low cosine score.
    """)

with st.expander("Q3: What is the difference between Context Precision and Context Recall in Ragas?"):
    st.markdown("""
    **Answer**:
    * **Context Precision**: Evaluates **signal-to-noise and rank order**. Did the retriever rank the truly relevant chunks at positions 1 and 2, or did it bury them at positions 4 and 5 beneath irrelevant chunks?
    * **Context Recall**: Evaluates **knowledge completeness**. Did the retrieved chunks capture all key facts present in the ground-truth reference answer?
    """)

with st.expander("Q4: Can a RAG system have 100% Faithfulness but 0% Answer Relevance? Give an example."):
    st.markdown("""
    **Answer**:
    **Yes!**
    * **User Question**: *"What are our password complexity requirements?"*
    * **Retrieved Context**: *"Company travel meal stipends are capped at $85 per day."*
    * **Generated Answer**: *"Our company travel meal stipends are capped at $85 per day."*
    * **Evaluation**:
      * *Faithfulness*: **100%** (The statement is 100% true and directly grounded in the context!).
      * *Answer Relevance*: **0%** (It completely failed to answer the user's question about passwords!).
    """)

with st.expander("Q5: Why should LLM-as-a-Judge evaluators use small temperature (temperature = 0)?"):
    st.markdown("""
    **Answer**:
    Evaluation frameworks must be **deterministic and reproducible**:
    * Setting `temperature > 0` introduces random sampling variance, meaning the exact same RAG output could receive a Faithfulness score of 0.90 on Monday and 0.65 on Tuesday.
    * Setting `temperature = 0` forces greedy argmax token selection, guaranteeing consistent, auditable benchmark tracking across CI/CD pull request runs.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Evaluating Groundedness with the Same Model Used for Generation**  
    Using a weak 8B model to evaluate its own responses results in circular agreement bias. Use a frontier model (Claude 3.5 Sonnet / GPT-4o) as the evaluator.
    """)
    st.warning("""
    **Trap 2: Ignoring Claim Extraction in Faithfulness**  
    Asking an LLM simply *"Is this answer grounded? Rate 1-5"* produces noisy, uncalibrated numbers. Always require extracting atomic claims and grading each claim individually.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Evaluating in Production Without Sampling**  
    Running LLM-as-a-Judge evaluations on 100% of live production traffic doubles LLM API costs. Sample 2% to 5% of traffic for real-time monitoring.
    """)
    st.warning("""
    **Trap 4: Missing CI/CD Regression Gates**  
    Shipping prompt updates or switching vector databases without automated RAG Triad regression tests allows silent hallucination regressions into production.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Faithfulness?"):
        st.markdown("""
        **Groundedness Metric**:  
        Fraction of generated answer claims directly supported by retrieved context chunks (zero hallucinations).
        """)
    with st.expander("🗂️ Flashcard: What does Context Relevance measure?"):
        st.markdown("""
        **Retriever Quality**:  
        Evaluates whether retrieved passages are useful and topical relative to the user's inquiry.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How is Answer Relevance measured?"):
        st.markdown("""
        **Question Generation**:  
        Generates synthetic questions from the answer and computes cosine similarity with original query.
        """)
    with st.expander("🗂️ Flashcard: What is Context Precision?"):
        st.markdown("""
        **Signal Density & Rank**:  
        Measures whether relevant chunks are placed at the top of the context window.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the RAG Triad of Trust: Context Relevance, Faithfulness, and Answer Relevance.", value=True)
chk2 = st.checkbox("I know how the Ragas framework decomposes answers into atomic claims to evaluate Faithfulness.", value=True)
chk3 = st.checkbox("I understand the difference between Context Precision (rank order) and Context Recall (completeness).", value=True)
chk4 = st.checkbox("I can implement an automated LLM-as-a-Judge prompt pipeline with deterministic output.", value=True)
chk5 = st.checkbox("I know how to identify whether a RAG failure is caused by the Vector Store or the LLM Generator.", value=True)
chk6 = st.checkbox("I can establish automated CI/CD quality gates for continuous RAG regression benchmarking.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered RAG Evaluation and are ready for Chapter 19.3: RAG Experimentation & A/B Benchmarking!")
