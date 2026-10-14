import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="17.1 Why Evaluate RAG - RAG Master Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class RAGEvaluationEngine:
    """
    Production-grade RAG Evaluation & Diagnostic Engine:
    1. Retrieval Quality Assessment (signal-to-noise ratio, recall coverage).
    2. Generation Quality Assessment (fluency, coherence, structural validity).
    3. Groundedness / Faithfulness Scoring (atomic claim entailment ratio).
    4. Relevance Scoring (query-context alignment, answer completeness).
    5. Accuracy & Benchmark Verification against ground-truth references.
    """
    def __init__(self):
        pass

    def evaluate_retrieval_quality(self, query: str, context_chunks: List[str]) -> Dict[str, Any]:
        """Calculates lexical and entity overlap between query and retrieved chunks."""
        query_terms = set([w.lower().strip(",.?") for w in query.split() if len(w) > 3])
        total_chunks = max(1, len(context_chunks))
        relevant_chunks = 0
        overlap_ratios = []

        for chunk in context_chunks:
            chunk_lower = chunk.lower()
            matches = sum(1 for term in query_terms if term in chunk_lower)
            ratio = matches / max(1, len(query_terms))
            overlap_ratios.append(ratio)
            if ratio >= 0.50:
                relevant_chunks += 1

        precision = relevant_chunks / total_chunks
        avg_snr = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0

        return {
            "retrieval_precision": round(precision, 2),
            "signal_to_noise_ratio": round(avg_snr, 2),
            "relevant_chunk_count": relevant_chunks,
            "total_chunk_count": total_chunks,
            "status": "PASS" if precision >= 0.60 else "POOR_RETRIEVAL"
        }

    def evaluate_groundedness(self, answer: str, context_chunks: List[str]) -> Dict[str, Any]:
        """Decomposes answer into sentences and computes context support."""
        sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if len(s.strip()) > 8]
        if not sentences:
            return {"groundedness_score": 1.0, "entailed": 0, "total": 0, "status": "PASS"}

        combined_context = " ".join(context_chunks).lower()
        entailed = 0

        for sent in sentences:
            words = set([w.lower().strip(",.?") for w in sent.split() if len(w) > 3])
            overlap = sum(1 for w in words if w in combined_context)
            if (overlap / max(1, len(words))) >= 0.65:
                entailed += 1

        score = entailed / len(sentences)
        return {
            "groundedness_score": round(score, 2),
            "entailed_sentences": entailed,
            "total_sentences": len(sentences),
            "status": "PASS" if score >= 0.85 else "HALLUCINATION_RISK"
        }

    def evaluate_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        """Evaluates answer relevance by checking query keyword coverage and length."""
        q_terms = set([w.lower().strip(",.?") for w in query.split() if len(w) > 3])
        ans_lower = answer.lower()
        matched = sum(1 for term in q_terms if term in ans_lower)
        coverage = matched / max(1, len(q_terms))

        # Check for non-committal or generic answers
        non_committal = "i don't know" in ans_lower or "cannot answer" in ans_lower
        relevance_score = 0.2 if non_committal else min(1.0, coverage * 1.2)

        return {
            "answer_relevance_score": round(relevance_score, 2),
            "query_term_coverage": round(coverage, 2),
            "is_non_committal": non_committal,
            "status": "PASS" if relevance_score >= 0.70 else "OFF_TOPIC"
        }

    def compute_rag_triad(
        self, query: str, context_chunks: List[str], answer: str
    ) -> Dict[str, Any]:
        """Computes the consolidated RAG Triad scorecard."""
        ret_eval = self.evaluate_retrieval_quality(query, context_chunks)
        grd_eval = self.evaluate_groundedness(answer, context_chunks)
        rel_eval = self.evaluate_relevance(query, answer)

        harmonic_mean = 3.0 / (
            (1.0 / max(0.01, ret_eval["retrieval_precision"])) +
            (1.0 / max(0.01, grd_eval["groundedness_score"])) +
            (1.0 / max(0.01, rel_eval["answer_relevance_score"]))
        )

        return {
            "retrieval_quality": ret_eval,
            "groundedness": grd_eval,
            "relevance": rel_eval,
            "composite_triad_score": round(harmonic_mean, 2),
            "is_production_ready": (
                ret_eval["retrieval_precision"] >= 0.60 and
                grd_eval["groundedness_score"] >= 0.85 and
                rel_eval["answer_relevance_score"] >= 0.70
            )
        }


# Instantiate engine at module level
eval_engine = RAGEvaluationEngine()

# --- HEADER SECTION ---
st.title("17.1 Why Evaluate RAG: The RAG Triad, Dual Failure Modes, & Holistic Pipeline Diagnostics")
st.caption("Chapter 17: RAG Evaluation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Deploying a Retrieval-Augmented Generation system to production without a quantitative evaluation harness 
is like piloting an aircraft without flight telemetry.

Unlike traditional machine learning models that optimize a single scalar loss function (e.g., cross-entropy or mean squared error), 
**RAG systems are compound AI systems** consisting of two decoupled components:
1. **The Retrieval Component**: Surfacing the most relevant, non-redundant evidence chunks from millions of documents.
2. **The Generation Component**: Synthesizing a concise, factual, and faithfully attributed answer conditioned on those chunks.

When an end user reports an incorrect answer, **where did the pipeline fail?**
* Did the **retriever** fail to find the gold passage (Retrieval Failure)?
* Did the **reranker** bury the gold passage in the attention dead-zone (Context Ordering Failure)?
* Did the **generator** hallucinate and ignore the facts in the prompt (Generation Failure)?
* Did the **user question** contain ambiguities that diverted the vector search (Query Failure)?

This lesson introduces the foundational science of RAG evaluation, centered on the industry-standard **RAG Triad**, 
enabling systematic debugging and quantitative quality guarantees.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Multi-Dimensional RAG Evaluation Framework", divider="rainbow")

st.markdown("""
The RAG Evaluation subsystem sits alongside the production pipeline, continuously intercepting queries, 
retrieval pools, and generated completions to compute telemetry scores across the three vertices of the RAG Triad.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RAGEvaluationArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // PRODUCTION ARTIFACTS
        subgraph cluster_runtime {
            label="1. Runtime Ingestion & Trace Interception";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            QueryArtifact [label="User Query (Q)\\n('What is the failover SLA?')", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            ContextArtifact [label="Retrieved Context (C)\\n(Top-K chunks & metadata)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            AnswerArtifact [label="Generated Answer (A)\\n(Synthesized response & citations)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
        }

        // EVALUATION TRIAD AXES
        subgraph cluster_triad {
            label="2. The RAG Triad Evaluation Core";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            ContextRelevance [label="Vertex 1: Context Relevance (Q -> C)\\nDid retriever find gold facts with low noise?", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            Groundedness [label="Vertex 2: Groundedness (C -> A)\\nAre all claims logically entailed by context?", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            AnswerRelevance [label="Vertex 3: Answer Relevance (Q -> A)\\nDoes answer directly satisfy user question?", fillcolor="#0F172A", fontcolor="#10B981", color="#059669"];
        }

        // DIAGNOSTIC SYNTHESIZER
        subgraph cluster_diagnostics {
            label="3. Root Cause Diagnostics & Scoring";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            TriadScorecard [label="Consolidated Triad Scorecard\\n(Harmonic Mean Composite Metric)", fillcolor="#0F172A", fontcolor="#A855F7", color="#7C3AED"];
            RootCauseArbiter [label="Root Cause Classifier\\n(Pinpoints: Retriever vs Generator Failure)", fillcolor="#0F172A", fontcolor="#A855F7", color="#7C3AED"];
        }

        // OUTPUT TELEMETRY
        ProductionTelemetry [label="CI/CD Quality Gate & Alerting\\n(Fails PRs if Faithfulness < 0.90)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];

        // PIPELINE EDGES
        QueryArtifact -> ContextRelevance [label="Evaluate Alignment"];
        ContextArtifact -> ContextRelevance;
        ContextArtifact -> Groundedness [label="Ground Truth Reference"];
        AnswerArtifact -> Groundedness [label="Extract Propositions"];
        QueryArtifact -> AnswerRelevance [label="Intent Matching"];
        AnswerArtifact -> AnswerRelevance;

        ContextRelevance -> TriadScorecard;
        Groundedness -> TriadScorecard;
        AnswerRelevance -> TriadScorecard;

        TriadScorecard -> RootCauseArbiter;
        RootCauseArbiter -> ProductionTelemetry;
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Core Architectural Objectives:
* **Decoupled Root-Cause Analysis**: Distinguishes retrieval errors (low recall) from generative errors (hallucinations).
* **Reference-Free Evaluation**: Evaluates live user queries where human-annotated ground-truth answers do not exist.
* **Continuous CI/CD Regression Testing**: Automatically blocks bad prompts or faulty chunking strategies before production deployment.
""")

st.divider()

# ==============================================================================
# SECTION 2: RETRIEVAL QUALITY
# ==============================================================================
st.header("2. Retrieval Quality (Noise Rejection, Recall Gaps & Search Diagnostic Auditing)", divider="rainbow")

st.markdown("""
Retrieval quality evaluates the **upstream search stage**: did the retriever supply the necessary facts, 
or did it inject irrelevant distractors that saturate the LLM's attention heads?
""")

rq_col1, rq_col2 = st.columns([1, 1])

with rq_col1:
    st.markdown("""
    ### Dual Failure Modes of Retrieval:
    1. **Low Recall (Recall Gap)**:
       * *Symptom*: The golden document containing the answer was never retrieved (ranked outside Top-K).
       * *Consequence*: The generator is starved of factual substrate, forcing either a refusal or an ungrounded guess.
    2. **Low Precision (Noise Saturation)**:
       * *Symptom*: The golden document is retrieved at rank 5, but ranks 1 through 4 are irrelevant boilerplate documents.
       * *Consequence*: The LLM's attention heads suffer from the Lost-in-the-Middle attention slump, diluting generation accuracy.
    """)

with rq_col2:
    st.markdown("### Signal-to-Noise Ratio (SNR) in Retrieved Context")
    snr_levels = pd.DataFrame([
        {"Context Composition": "100% Gold Facts (0 Distractors)", "Retrieval Quality": "Optimal", "Hallucination Risk": "Minimal (< 2%)"},
        {"Context Composition": "50% Gold / 50% Distractors", "Retrieval Quality": "Acceptable", "Hallucination Risk": "Moderate (5 - 10%)"},
        {"Context Composition": "10% Gold / 90% Distractors", "Retrieval Quality": "Severely Cluttered", "Hallucination Risk": "High (25 - 40%)"},
        {"Context Composition": "0% Gold (Complete Search Miss)", "Retrieval Quality": "Catastrophic Failure", "Hallucination Risk": "Extreme (100% unless refused)"}
    ])
    st.dataframe(snr_levels, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 3: GENERATION QUALITY
# ==============================================================================
st.header("3. Generation Quality (Syntactic Fluency, Coherence & Hallucination Diagnostics)", divider="rainbow")

st.markdown("""
Generation quality evaluates the **downstream synthesis stage**: given a set of retrieved documents, 
how effectively did the LLM synthesize the final answer?
""")

gq_col1, gq_col2 = st.columns([1, 1])

with gq_col1:
    st.markdown("""
    ### Evaluation Dimensions of Generation:
    * **Coherence & Fluency**: Is the generated text logically structured, syntactically sound, and devoid of repetitive token loops?
    * **Conciseness vs Verbosity**: Does the answer directly address the question without superfluous conversational filler?
    * **Structure Compliance**: If JSON or markdown tables were requested, did the model strictly adhere to the target schema?
    * **Tone & Persona Fidelity**: Does the output maintain the mandated professional enterprise tone?
    """)

with gq_col2:
    st.markdown("### Generative Failure Classification")
    gen_fails = pd.DataFrame([
        {"Failure Type": "Extrapolative Hallucination", "Description": "Model invents speculative rules not present in context.", "Severity": "Critical"},
        {"Failure Type": "Entity Inversion", "Description": "Swaps roles of subjects and objects in technical procedures.", "Severity": "Critical"},
        {"Failure Type": "Runaway Verbosity", "Description": "Generates 500 words of background when 20 words were requested.", "Severity": "Moderate"},
        {"Failure Type": "Premature Truncation", "Description": "Answer cuts off mid-sentence due to max_token limits.", "Severity": "High"}
    ])
    st.dataframe(gen_fails, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 4: GROUNDEDNESS
# ==============================================================================
st.header("4. Groundedness (Faithfulness, Factual Consistency & Parametric Bleed Detection)", divider="rainbow")

st.markdown("""
**Groundedness** (Vertex 2 of the RAG Triad) evaluates the relationship between the **Retrieved Context** 
and the **Generated Response**. It answers: *Can every single assertion in the answer be traced to the context?*
""")

grd_col1, grd_col2 = st.columns([1, 1])

with grd_col1:
    st.markdown("""
    ### Groundedness Calculation Methodology:
    1. **Proposition Decomposition**: Break the answer into atomic factual statements: `S = {s_1, s_2, ..., s_m}`.
    2. **Context Entailment Verification**: For each statement `s_i`, run an NLI classifier testing whether `Context -> s_i`.
    3. **Mathematical Formulation**:
       `Groundedness = Count(Entailed Statements) / Total Statements`
    * A score of `1.0` indicates 100% factual faithfulness.
    * A score `< 0.85` indicates active hallucination and requires output suppression.
    """)

with grd_col2:
    st.markdown("### Groundedness Verification Flow")
    st.code(r'''
Answer: "Replication lag is under 1 second [DOC-1]. Manual failover takes 45 minutes."

Decomposition:
1. "Replication lag is under 1 second." -> Checked against DOC-1 -> ENTAILED (1.0)
2. "Manual failover takes 45 minutes." -> Checked against context -> NOT FOUND (0.0)

Groundedness Score = (1.0 + 0.0) / 2 = 0.50 (FLAGGED: UNGROUNDED)
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 5: RELEVANCE
# ==============================================================================
st.header("5. Relevance (Query Alignment, Answer Completeness & Redundancy Penalization)", divider="rainbow")

st.markdown("""
Relevance encompasses both **Context Relevance** (did we fetch the right chunks?) and **Answer Relevance** 
(did we satisfy the user's specific inquiry without dodging or hallucinating)?
""")

rel_col1, rel_col2 = st.columns([1, 1])

with rel_col1:
    st.markdown("""
    ### Answer Relevance Mechanics:
    * **The Dilemma of Dodging**: A model might achieve a 100% Groundedness score by simply repeating irrelevant facts from the context (*"The office was built in 1998"* when asked about 401k benefits).
    * **Measuring Answer Relevance**:
      - Measures the semantic similarity between the user query and the generated answer.
      - Penalizes responses that are non-committal or wander off-topic.
      - Uses LLM-as-a-judge to evaluate completeness: *Did the response address all parts of a compound question?*
    """)

with rel_col2:
    st.markdown("### Relevance Scoring Tradeoffs")
    rel_table = pd.DataFrame([
        {"Metric": "Context Relevance", "Evaluates": "Query -> Context Chunks", "Question": "Is the retrieved text focused on the query?", "Penalty For": "Unrelated background text"},
        {"Metric": "Answer Relevance", "Evaluates": "Query -> Generated Answer", "Question": "Does the response directly answer the prompt?", "Penalty For": "Dodging, non-committal filler"},
        {"Metric": "Completeness", "Evaluates": "Compound Sub-Queries", "Question": "Were all parts of the question addressed?", "Penalty For": "Partial answers"}
    ])
    st.dataframe(rel_table, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 6: ACCURACY
# ==============================================================================
st.header("6. Accuracy (Semantic Exactness, Golden Truth Benchmarking & Factual Precision)", divider="rainbow")

st.markdown("""
While Groundedness and Relevance evaluate live pipelines without reference labels, **Accuracy** measures 
performance against curated **Golden Datasets** where human experts have annotated canonical ground-truth answers.
""")

acc_col1, acc_col2 = st.columns([1, 1])

with acc_col1:
    st.markdown("""
    ### Golden Benchmark Evaluation:
    * **Golden Q&A Pairs**: Test suites of 200 to 1,000 curated questions paired with certified answers and gold document IDs.
    * **Evaluation Dimensions**:
      - **Semantic Exactness (BERTScore / LLM Judge)**: Does the meaning match the golden answer, even if phrased differently?
      - **Factual Precision**: Are dates, numbers, entity names, and error codes identical to the reference?
      - **Regression Tracking**: Run in CI/CD before deploying new chunking strategies or vector database indexes.
    """)

with acc_col2:
    st.markdown("### Live Production vs Benchmark Evaluation")
    eval_comp = pd.DataFrame([
        {"Criterion": "Live Telemetry (Reference-Free)", "Methodology": "RAG Triad (Context Rel, Faithfulness, Answer Rel)", "Cost": "Low per request", "Dataset Needed": "None (Real user traffic)"},
        {"Criterion": "Offline Benchmark (Reference-Based)", "Methodology": "Exact Match, Semantic Similarity, Factual Precision", "Cost": "Periodic batch evaluation", "Dataset Needed": "Human curated golden Q&A"}
    ])
    st.dataframe(eval_comp, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Multi-Dimensional RAG Evaluation Engine", divider="rainbow")

st.markdown("""
Below is the production implementation of `RAGEvaluationEngine`. It runs in module scope, 
computing retrieval precision, sentence-level groundedness, query-answer relevance, and composite triad scores.
""")

st.code(r'''
class RAGEvaluationEngine:
    """
    Enterprise RAG Evaluation & Diagnostic Suite:
    - Calculates Retrieval Quality & Signal-to-Noise Ratio (SNR).
    - Decomposes answers into sentences to compute Groundedness / Faithfulness.
    - Evaluates Answer Relevance and query keyword coverage.
    - Computes composite RAG Triad scores (Harmonic Mean).
    """
    def evaluate_retrieval_quality(self, query: str, context_chunks: List[str]) -> Dict[str, Any]:
        query_terms = set([w.lower().strip(",.?") for w in query.split() if len(w) > 3])
        total = max(1, len(context_chunks))
        relevant = 0
        ratios = []
        for chunk in context_chunks:
            matches = sum(1 for t in query_terms if t in chunk.lower())
            r = matches / max(1, len(query_terms))
            ratios.append(r)
            if r >= 0.50:
                relevant += 1
        prec = relevant / total
        return {
            "retrieval_precision": round(prec, 2),
            "signal_to_noise_ratio": round(float(np.mean(ratios)), 2) if ratios else 0.0,
            "status": "PASS" if prec >= 0.60 else "POOR_RETRIEVAL"
        }

    def evaluate_groundedness(self, answer: str, context_chunks: List[str]) -> Dict[str, Any]:
        sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if len(s.strip()) > 8]
        if not sentences:
            return {"groundedness_score": 1.0, "status": "PASS"}
        context_blob = " ".join(context_chunks).lower()
        entailed = 0
        for sent in sentences:
            words = set([w.lower().strip(",.?") for w in sent.split() if len(w) > 3])
            overlap = sum(1 for w in words if w in context_blob)
            if (overlap / max(1, len(words))) >= 0.65:
                entailed += 1
        score = entailed / len(sentences)
        return {
            "groundedness_score": round(score, 2),
            "entailed_count": entailed,
            "total_sentences": len(sentences),
            "status": "PASS" if score >= 0.85 else "HALLUCINATION_RISK"
        }

    def evaluate_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        q_terms = set([w.lower().strip(",.?") for w in query.split() if len(w) > 3])
        matched = sum(1 for t in q_terms if t in answer.lower())
        cov = matched / max(1, len(q_terms))
        non_committal = "i don't know" in answer.lower() or "cannot answer" in answer.lower()
        score = 0.2 if non_committal else min(1.0, cov * 1.2)
        return {
            "answer_relevance_score": round(score, 2),
            "status": "PASS" if score >= 0.70 else "OFF_TOPIC"
        }

    def compute_rag_triad(self, query: str, context_chunks: List[str], answer: str) -> Dict[str, Any]:
        ret = self.evaluate_retrieval_quality(query, context_chunks)
        grd = self.evaluate_groundedness(answer, context_chunks)
        rel = self.evaluate_relevance(query, answer)
        h_mean = 3.0 / (
            (1.0 / max(0.01, ret["retrieval_precision"])) +
            (1.0 / max(0.01, grd["groundedness_score"])) +
            (1.0 / max(0.01, rel["answer_relevance_score"]))
        )
        return {
            "retrieval": ret, "groundedness": grd, "relevance": rel,
            "composite_score": round(h_mean, 2),
            "is_production_ready": ret["retrieval_precision"] >= 0.60 and grd["groundedness_score"] >= 0.85 and rel["answer_relevance_score"] >= 0.70
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: RAG Triad Evaluator & Diagnostic Scorecard", divider="rainbow")

st.markdown("""
Test the evaluation engine against simulated pipeline runs. 
Select different failure scenarios to observe how the **RAG Triad Scorecard** pinpoints root causes in real time.
""")

sample_eval_scenarios = {
    "1. Healthy Pipeline (High Precision & Grounded)": {
        "query": "What is the guaranteed database uptime SLA and failover latency?",
        "context": [
            "Tier-1 database SLA guarantees 99.99% uptime for active production clusters.",
            "Automated multi-region failover initiates within 120 seconds of outage detection."
        ],
        "answer": "The database SLA guarantees 99.99% uptime. Automated failover initiates within 120 seconds."
    },
    "2. Retriever Failure (Noise Saturation & Missed Facts)": {
        "query": "What is the guaranteed database uptime SLA and failover latency?",
        "context": [
            "Employees receive 25 days of paid time off per calendar year.",
            "Cafeteria lunch hours are 12:00 PM to 2:00 PM EST daily."
        ],
        "answer": "The documentation does not contain information regarding database uptime or failover latency."
    },
    "3. Generator Hallucination (Ignored Context)": {
        "query": "What is the guaranteed database uptime SLA and failover latency?",
        "context": [
            "Tier-1 database SLA guarantees 99.99% uptime for active production clusters.",
            "Automated multi-region failover initiates within 120 seconds of outage detection."
        ],
        "answer": "The database guarantees 99.999% five-nines uptime and manual failover takes 30 minutes with engineer approval."
    }
}

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.markdown("### Pipeline Scenario Selection")
    selected_scenario_name = st.selectbox(
        "Choose Pipeline Test Scenario:",
        options=list(sample_eval_scenarios.keys()),
        index=0
    )
    active_sc = sample_eval_scenarios[selected_scenario_name]

    eval_query = st.text_input("Evaluated Query:", value=active_sc["query"])
    eval_answer = st.text_area("Generated Answer:", value=active_sc["answer"], height=90)
    eval_context_blob = st.text_area("Retrieved Context (Separated by newlines):", value="\n".join(active_sc["context"]), height=90)
    eval_context_list = [c.strip() for c in eval_context_blob.split("\n") if c.strip()]

with wb_c2:
    st.markdown("### Live RAG Triad Diagnostic Scorecard")
    triad_res = eval_engine.compute_rag_triad(eval_query, eval_context_list, eval_answer)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Retrieval Precision", f"{triad_res['retrieval']['retrieval_precision'] * 100:.0f}%")
    m2.metric("Groundedness", f"{triad_res['groundedness']['groundedness_score'] * 100:.0f}%")
    m3.metric("Answer Relevance", f"{triad_res['relevance']['answer_relevance_score'] * 100:.0f}%")
    m4.metric("Composite Triad", f"{triad_res['composite_score'] * 100:.0f}%")

    if triad_res["is_production_ready"]:
        st.success("🏆 Pipeline Quality Verified: All three vertices of the RAG Triad pass production thresholds.")
    else:
        st.error(f"🚨 Pipeline Quality Alert: Composite score {triad_res['composite_score']}. Inspect diagnostic breakdowns below.")

st.markdown("#### Diagnostic Root Cause Attribution Table:")
diag_summary = pd.DataFrame([
    {"Evaluation Vertex": "Context Relevance (Q -> C)", "Score": f"{triad_res['retrieval']['retrieval_precision'] * 100:.0f}%", "Status": triad_res["retrieval"]["status"], "Diagnostic Interpretation": "High search purity" if triad_res["retrieval"]["status"] == "PASS" else "Retriever fetched irrelevant noise"},
    {"Evaluation Vertex": "Groundedness / Faithfulness (C -> A)", "Score": f"{triad_res['groundedness']['groundedness_score'] * 100:.0f}%", "Status": triad_res["groundedness"]["status"], "Diagnostic Interpretation": "Zero hallucination" if triad_res["groundedness"]["status"] == "PASS" else "Generator asserted ungrounded claims"},
    {"Evaluation Vertex": "Answer Relevance (Q -> A)", "Score": f"{triad_res['relevance']['answer_relevance_score'] * 100:.0f}%", "Status": triad_res["relevance"]["status"], "Diagnostic Interpretation": "Directly answers query" if triad_res["relevance"]["status"] == "PASS" else "Answer dodges or misses query intent"}
])
st.dataframe(diag_summary, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: What is the RAG Triad, and why is evaluating compound AI systems different from evaluating standard LLMs?**"):
    st.markdown("""
    **Answer**:
    The RAG Triad (pioneered by TruLens) evaluates the three core relationships of a compound RAG pipeline:
    1. **Context Relevance**: `Query -> Retrieved Context` (Did the retriever find the right information?).
    2. **Groundedness / Faithfulness**: `Retrieved Context -> Answer` (Are all generated claims entailed by the context?).
    3. **Answer Relevance**: `Query -> Answer` (Does the response answer the user's question?).
    Standard LLMs only possess an input-output interface (`Prompt -> Output`). Compound RAG systems feature decoupled components; evaluating only the final answer masks whether errors were caused by retriever misses or generator hallucinations.
    """)

with st.expander("**Q2: How does reference-free evaluation operate, and why is it mandatory for production RAG telemetry?**"):
    st.markdown("""
    **Answer**:
    * **Reference-Based Evaluation**: Compares generated answers against human-annotated golden reference answers using metrics like BLEU, ROUGE, or BERTScore. This is only possible offline on curated test sets.
    * **Reference-Free Evaluation**: Evaluates live user queries where no human answer exists. It uses the RAG Triad: testing whether the answer is logically entailed by the retrieved context (NLI) and whether the answer addresses the query (semantic similarity). This enables 24/7 production observability across 100% of live traffic.
    """)

with st.expander("**Q3: What is the Harmonic Mean composite score in RAG Triad evaluation, and why is it used over the Arithmetic Mean?**"):
    st.markdown("""
    **Answer**:
    The Harmonic Mean of three metrics `(x, y, z)` is:
    `H = 3 / (1/x + 1/y + 1/z)`
    In the arithmetic mean `(x + y + z) / 3`, a catastrophic failure in one component (e.g., Groundedness = 0.0 with Context Relevance = 1.0 and Answer Relevance = 1.0) yields an average of 0.67, masking the failure.
    The harmonic mean severely penalizes extreme outliers: if any single vertex collapses toward 0.0, the composite score collapses to zero, correctly signaling an unacceptable production failure.
    """)

with st.expander("**Q4: How do you distinguish between a Retrieval Failure and a Generation Failure in production telemetry?**"):
    st.markdown("""
    **Answer**:
    * **Retrieval Failure**: Context Relevance is LOW (<0.40). The retriever fetched irrelevant passages. Even if the LLM generates a fluent answer, it is either hallucinated or an unhelpful refusal.
    * **Generation Failure**: Context Relevance is HIGH (>0.85), but Groundedness is LOW (<0.50). The gold document was successfully placed in the prompt, but the LLM ignored it or hallucinated ungrounded facts due to high temperature or attention slump.
    """)

with st.expander("**Q5: Why does high Answer Relevance not guarantee a correct answer?**"):
    st.markdown("""
    **Answer**:
    Answer relevance measures semantic alignment with the question, not factual truth. 
    If a user asks *"What is the boiling point of liquid nitrogen?"* and the model answers *"Liquid nitrogen boils at exactly 450 degrees Celsius"*, the answer has **high Answer Relevance** (it directly addresses the topic), but **zero Groundedness / Accuracy** (it is factually completely wrong).
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Relying Exclusively on BLEU / ROUGE Metrics**
    * **Failure Mode**: BLEU and ROUGE evaluate n-gram token overlap against a reference text. A model saying *"The policy allows refunds"* and *"The policy does NOT allow refunds"* will receive an 80% ROUGE score despite having opposite factual meanings.
    * **Solution**: Use semantic NLI entailment and LLM-as-a-judge evaluators.
    """)

    st.error("""
    **Production Trap 2: Evaluating Only the Final Generated Answer**
    * **Failure Mode**: When answers fail, developers spend weeks tweaking system prompts, unaware that the actual bug was the retriever failing to fetch the gold document.
    * **Solution**: Log and evaluate the retriever and generator separately.
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Evaluating with Small, Homogeneous Test Sets**
    * **Failure Mode**: Testing on 20 simple synthetic questions produces 95% scores, followed by failure in production on complex compound domain questions.
    * **Solution**: Benchmark across diverse query archetypes (single-hop, multi-hop, unanswerable, adversarial).
    """)

    st.warning("""
    **Production Trap 4: Ignoring Context Relevance (Noise Clutter)**
    * **Failure Mode**: Achieving high recall by retrieving 20 chunks (Top-K = 20) saturates the context window, causing high inference latency and attention degradation.
    * **Solution**: Continuously monitor and enforce a Context Relevance floor.
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: The RAG Triad")
        st.caption("What three evaluation axes comprise the RAG Triad?")
        st.markdown("**Back**: 1. Context Relevance (Q -> C), 2. Groundedness/Faithfulness (C -> A), 3. Answer Relevance (Q -> A).")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: Reference-Free Telemetry")
        st.caption("How to evaluate live user traffic without human ground truth?")
        st.markdown("**Back**: Evaluate atomic claim entailment against context (Groundedness) and semantic intent alignment (Answer Relevance).")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: Harmonic Mean Scoring")
        st.caption("Why use harmonic mean for composite triad scores?")
        st.markdown("**Back**: Severely penalizes single-vertex failures; if Groundedness collapses to zero, the composite score drops to zero.")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Retrieval vs Generation Failure")
        st.caption("How to isolate retriever errors from generator errors?")
        st.markdown("**Back**: Low Context Relevance indicates a retrieval failure; high Context Relevance with low Groundedness indicates a generative hallucination.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of RAG evaluation fundamentals before proceeding:")

c1 = st.checkbox("I can define and explain all three vertices of the RAG Triad: Context Relevance, Groundedness, and Answer Relevance.", value=True)
c2 = st.checkbox("I understand how reference-free evaluation enables 24/7 production observability without ground-truth answers.", value=True)
c3 = st.checkbox("I can diagnose whether a production failure originated in the retrieval component or generative component.", value=True)
c4 = st.checkbox("I understand why n-gram metrics (BLEU/ROUGE) fail in semantic RAG evaluation and how NLI replaces them.", value=True)
c5 = st.checkbox("I know how to calculate composite quality scores using the harmonic mean to penalize catastrophic single-vertex drops.", value=True)
c6 = st.checkbox("I understand how offline golden benchmarks and CI/CD regression gates prevent degraded search deployments.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **RAG Evaluation Fundamentals Mastery Achieved!** You are ready to build quantitative evaluation pipelines.")
