import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="20.3 RAG Evaluation - Embeddings Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# Title & Overview
st.title("20.3 RAG Evaluation: Context Relevance, Precision, Recall, Faithfulness, & RAGAS")
st.caption("Chapter 20: Embedding Evaluation | Domain: 3. Vector Embeddings")
st.markdown("""
Evaluating a standard machine learning model is straightforward: you compare predicted numbers or classes against labeled targets.
Evaluating a **Retrieval-Augmented Generation (RAG) system** is notoriously complex because it spans **two disconnected phases**:
1. **The Retrieval Phase**: Handled by embedding models and vector databases.
2. **The Generation Phase**: Handled by large language models.

If the final answer is wrong, where did the pipeline fail? Did the embedding model retrieve bad chunks, or did the LLM hallucinate despite having good chunks?

To isolate and measure each component independently, the industry relies on **The RAG Triad** and frameworks like **RAGAS (Retrieval Augmented Generation Assessment)**:
* **Context Relevance**: Does the retrieved context contain signal, or is it mostly distractor noise?
* **Context Precision**: Are the most relevant chunks ranked at the top of the context window?
* **Context Recall**: Did the retrieved context contain all facts needed to answer the question?
* **Faithfulness (Groundedness)**: Can every single claim in the generated answer be directly inferred from the retrieved context (zero hallucinations)?
* **Answer Relevance**: Does the generated answer directly address the user's prompt?

This lesson explores the quantitative mechanics of end-to-end RAG pipeline evaluation.
""")

st.divider()

# Section 1: The RAG Triad & Metric Architecture
st.header("1. The RAG Triad & The RAGAS Evaluation Framework", divider="rainbow")

col_tri1, col_tri2 = col_tri1, col_tri2 = st.columns([1, 1])

with col_tri1:
    st.markdown("""
    ### The 3 Core Evaluation Axes
    * **1. Query $\\leftrightarrow$ Context (Context Relevance & Recall)**:
      * Evaluates the **Embedding Retriever**.
      * Did the vector search engine locate the necessary facts (**Recall**), and is the retrieved context free of irrelevant noise (**Relevance**)?
    * **2. Context $\\leftrightarrow$ Answer (Faithfulness / Groundedness)**:
      * Evaluates **Hallucination Rate**.
      * Every factual claim made in the LLM's answer is checked against the retrieved passages. If the LLM invents a statistic or date not present in the context, Faithfulness plummets!
    * **3. Query $\\leftrightarrow$ Answer (Answer Relevance)**:
      * Evaluates **Helpfulness & Directness**.
      * Does the final answer directly resolve the user's question, or does it ramble off-topic?
    """)

with col_tri2:
    st.markdown("### The RAG Triad Architecture")
    triad_dot = """
    digraph RAGTriad {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        Query [label="User Query\\n'What is the refund window?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        Context [label="Retrieved Context\\n'Full refund within 30 days...'", fillcolor="#ECFDF5", color="#10B981", style="filled,bold"];
        Answer [label="Generated Answer\\n'Customers have 30 days...'", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        Query -> Context [dir="both", label="Context Relevance\\n& Context Recall", color="#059669"];
        Context -> Answer [dir="both", label="Faithfulness\\n(Hallucination Check)", color="#DC2626"];
        Query -> Answer [dir="both", label="Answer Relevance\\n(Directness)", color="#2563EB"];
    }
    """
    st.graphviz_chart(triad_dot, use_container_width=False)

st.markdown("### The 5 Standard RAGAS Evaluation Metrics")
st.markdown("""
| Metric Name | Evaluates | Method of Measurement | What a Low Score Indicates |
| :--- | :--- | :--- | :--- |
| **Context Relevance** | Retriever | Sentence-level semantic filtering. | Prompt context is flooded with useless distractor text. |
| **Context Precision** | Retriever | Rank-weighted recall of ground-truth sentences. | Key facts are buried at the bottom of the prompt context. |
| **Context Recall** | Retriever | Comparison against reference ground-truth answers. | The embedding model failed to retrieve critical facts. |
| **Faithfulness** | LLM Generator | Claim-level entailment verification. | **Hallucination!** The LLM invented claims not in the context. |
| **Answer Relevance** | LLM Generator | Reverse query generation and embedding cosine similarity. | The LLM rambled, repeated disclaimers, or missed the point. |
""")

st.divider()

# Section 2: Production Code: Automated RAGAS Evaluation Suite
st.header("2. Production Code: Pure-Python RAGAS Evaluation Engine", divider="rainbow")

st.markdown("""
The production utility below demonstrates how automated RAG evaluation works: extracting atomic claims from an answer, verifying entailment against retrieved context sentences, and calculating quantitative Faithfulness and Context Precision scores.
""")

st.code(r'''
from typing import List, Dict, Any, Set

class RAGEvaluationEngine:
    """
    Simulates automated RAGAS metrics:
    - Faithfulness (Claim-level entailment against context)
    - Context Relevance (Signal sentence ratio)
    - Context Precision (Rank-weighted hit positions)
    - Answer Relevance (Cosine query alignment)
    """
    @staticmethod
    def compute_faithfulness(
        answer_claims: List[str],
        retrieved_context_sentences: List[str]
    ) -> Dict[str, Any]:
        """
        Faithfulness = (Count of supported claims) / (Total claims in answer)
        A claim is supported if it can be directly verified from the context.
        """
        if not answer_claims:
            return {"faithfulness": 1.0, "supported_claims": 0, "total_claims": 0}

        supported_count = 0
        verified_details = []

        context_corpus = " ".join(retrieved_context_sentences).lower()

        for claim in answer_claims:
            # Simulated entailment check: keyword/phrase presence in context
            claim_words = [w.lower() for w in claim.split() if len(w) > 3]
            match_ratio = sum(1 for w in claim_words if w in context_corpus) / max(1, len(claim_words))
            
            is_supported = match_ratio >= 0.70
            if is_supported:
                supported_count += 1
            
            verified_details.append({
                "claim": claim,
                "is_supported": is_supported,
                "verdict": "VERIFIED" if is_supported else "HALLUCINATION"
            })

        faithfulness = supported_count / float(len(answer_claims))
        return {
            "faithfulness": faithfulness,
            "supported_count": supported_count,
            "total_claims": len(answer_claims),
            "details": verified_details
        }

    @staticmethod
    def compute_context_precision(
        retrieved_chunks_relevance: List[bool]
    ) -> float:
        """
        Context Precision = Average Precision over retrieved ranked chunks.
        Rewards ranking relevant chunks at the very top of the prompt.
        """
        if not retrieved_chunks_relevance:
            return 0.0

        precisions = []
        hits = 0
        for rank, is_rel in enumerate(retrieved_chunks_relevance, 1):
            if is_rel:
                hits += 1
                precisions.append(hits / float(rank))

        return sum(precisions) / float(len(precisions)) if precisions else 0.0
''', language="python")

st.divider()

# Section 3: Interactive RAGAS Evaluation Workbench
st.header("3. Interactive Workbench: Live RAGAS Diagnostic Studio", divider="rainbow")

st.markdown("""
Diagnose three distinct production RAG failure modes. Observe how **Faithfulness**, **Answer Relevance**, and **Context Recall** pinpoint the exact breakdown point in the pipeline.
""")

eval_scenarios = {
    "Scenario 1: High-Fidelity Ideal RAG (All Systems Healthy)": {
        "query": "What is the warranty policy for battery replacements?",
        "context_chunks": [
            "[Chunk 1 - High Rel]: Replacement batteries are covered under a 2-year limited warranty from the date of professional installation.",
            "[Chunk 2 - Low Rel]: General maintenance inspections are recommended every 12 months for optimal battery health."
        ],
        "chunks_relevant": [True, False],
        "answer": "Replacement batteries carry a 2-year limited warranty starting from the date they are professionally installed.",
        "claims": [
            ("Replacement batteries carry a 2-year limited warranty", True),
            ("The warranty starts from the date of professional installation", True)
        ],
        "context_recall": 1.0,
        "answer_relevance": 0.96,
        "verdict": "PERFECT EXECUTION: Retriever found the needle at Rank 1; LLM generated an accurate, grounded answer with zero hallucination."
    },
    "Scenario 2: Severe Hallucination Bug (Generation Failure)": {
        "query": "What is the warranty policy for battery replacements?",
        "context_chunks": [
            "[Chunk 1 - High Rel]: Replacement batteries are covered under a 2-year limited warranty from the date of professional installation.",
            "[Chunk 2 - Low Rel]: In-store diagnostic tests cost twenty-five dollars."
        ],
        "chunks_relevant": [True, False],
        "answer": "Replacement batteries are covered for 2 years, and customers receive free lifetime roadside towing assistance and a free loaner vehicle.",
        "claims": [
            ("Replacement batteries are covered for 2 years", True),
            ("Customers receive free lifetime roadside towing assistance", False),
            ("Customers receive a free loaner vehicle", False)
        ],
        "context_recall": 1.0,
        "answer_relevance": 0.88,
        "verdict": "GENERATION FAILURE (HALLUCINATION): Retriever succeeded, but the LLM hallucinated free roadside towing and loaner vehicles!"
    },
    "Scenario 3: Retrieval Failure / Low Context Recall (Retriever Failure)": {
        "query": "What is the warranty policy for battery replacements?",
        "context_chunks": [
            "[Chunk 1 - Distractor]: Brake pads must be inspected every twenty thousand miles.",
            "[Chunk 2 - Distractor]: Windshield wiper replacement fluid is included during annual checkups."
        ],
        "chunks_relevant": [False, False],
        "answer": "I do not have information in the provided documents regarding battery warranty policies.",
        "claims": [
            ("No information regarding battery warranty was found in documents", True)
        ],
        "context_recall": 0.0,
        "answer_relevance": 0.40,
        "verdict": "RETRIEVAL FAILURE: Embedding model failed to retrieve any battery chunks into the top 2. Context Recall is 0.0."
    }
}

col_sc1, col_sc2 = st.columns([1, 1])

with col_sc1:
    st.subheader("1. Select Production Pipeline Run")
    selected_sc_name = st.selectbox("Choose Evaluation Scenario:", list(eval_scenarios.keys()))
    scenario = eval_scenarios[selected_sc_name]

    st.markdown(f"**User Query:**\n> *{scenario['query']}*")
    
    st.markdown("#### Retrieved Prompt Context:")
    for c in scenario["context_chunks"]:
        st.code(c)

    st.markdown("#### Generated LLM Answer:")
    st.info(scenario["answer"])

with col_sc2:
    st.subheader("2. RAGAS Metric Scorecard")

    # Compute metrics
    supported_claims_count = sum(1 for c in scenario["claims"] if c[1])
    total_claims = len(scenario["claims"])
    faithfulness_score = supported_claims_count / float(total_claims)

    hits = sum(1 for r in scenario["chunks_relevant"] if r)
    precisions = [hits / (i + 1) for i, r in enumerate(scenario["chunks_relevant"]) if r]
    context_precision_score = precisions[0] if precisions else 0.0
    context_recall_score = scenario["context_recall"]
    answer_relevance_score = scenario["answer_relevance"]

    m1, m2 = st.columns(2)
    m1.metric("Faithfulness (Groundedness)", f"{faithfulness_score * 100:.0f}%", help="Supported claims / Total claims")
    m2.metric("Answer Relevance", f"{answer_relevance_score * 100:.0f}%", help="Directness to user query")

    m3, m4 = st.columns(2)
    m3.metric("Context Precision", f"{context_precision_score * 100:.0f}%", help="Are relevant chunks ranked at Rank 1?")
    m4.metric("Context Recall", f"{context_recall_score * 100:.0f}%", help="Did context contain the needed facts?")

    if faithfulness_score < 0.60:
        st.error(f"🚨 {scenario['verdict']}")
    elif context_recall_score == 0.0:
        st.warning(f"⚠️ {scenario['verdict']}")
    else:
        st.success(f"✅ {scenario['verdict']}")

    st.markdown("#### Claim-by-Claim Verification Audit")
    claim_audit = []
    for claim_text, is_sup in scenario["claims"]:
        claim_audit.append({
            "Atomic Claim": claim_text,
            "Entailed by Context?": "✅ Verified" if is_sup else "❌ HALLUCINATED",
            "Penalty": "0.0" if is_sup else "-1.0"
        })
    st.dataframe(pd.DataFrame(claim_audit), use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "RAGAS Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How do modern automated evaluation frameworks (like RAGAS) calculate Faithfulness without human judges?**
    * **Answer**: RAGAS uses a two-step LLM-as-a-Judge prompting procedure:
      1. **Claim Decomposition**: An evaluator model breaks the generated answer down into a list of standalone atomic factual claims (e.g. *"Claim 1: The warranty is 2 years."*, *"Claim 2: It includes free roadside towing."*).
      2. **Entailment Verification**: For each claim, the evaluator performs Natural Language Inference (NLI) against the retrieved context to classify whether the claim is `Entailed (Supported)` or `Not Entailed (Hallucinated)`.
      `Faithfulness = Number of Supported Claims / Total Generated Claims`.

    **Q2: What is the exact difference between Context Relevance and Context Precision?**
    * **Answer**:
      * **Context Relevance**: Measures the *density of signal*. If an entire 500-word retrieved chunk contains only one useful sentence and 450 words of fluff, Context Relevance is low.
      * **Context Precision**: Measures *ranking position*. It evaluates whether the relevant chunks were placed at the very top of the prompt (Rank 1) rather than buried at the bottom (Rank 5), penalizing pipelines that present distractor chunks first.

    **Q3: If your RAG pipeline has 100% Faithfulness but 30% Answer Relevance, what is going wrong?**
    * **Answer**: The LLM is generating answers that are 100% strictly true according to the context, but it is **not answering the user's specific question**.
    * Typical causes: The LLM got confused by contradictory context, became overly cautious and replied with boilerplate disclaimers, or focused on an irrelevant background detail mentioned in the passage instead of the target query.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Using the Generator LLM as Its Own Judge**:
      * *Issue*: Prompting GPT-4 to generate an answer, and then asking the exact same GPT-4 instance to grade its own faithfulness.
      * *Consequence*: Self-enhancement bias; LLMs rate their own hallucinations as faithful up to 40% more often than independent models.
      * *Fix*: Use a separate evaluator model family or dedicated NLI cross-encoder models.

    * **Trap 2: Grading RAG Systems Purely on Final Answer BLEU / ROUGE Scores**:
      * *Issue*: Comparing LLM output to reference answers using n-gram overlap.
      * *Consequence*: An answer can be 100% factually correct while using different phrasing, receiving a low BLEU score.
      * *Fix*: Use semantic claim verification and LLM-as-a-Judge metrics (RAGAS).

    * **Trap 3: Conflating Retrieval Failure with LLM Hallucination**:
      * *Issue*: Firing a prompt engineer because the LLM answered incorrectly, when the embedding model never retrieved the answer chunk.
      * *Consequence*: Weeks wasted rewording system prompts when the fix required tuning chunk size or embedding models.
      * *Fix*: Always inspect Context Recall before analyzing Faithfulness.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Faithfulness (Groundedness)**
      * *Definition*: The ratio of atomic claims in an LLM answer that can be strictly substantiated from the retrieved context without external hallucination.
    * **Flashcard 2: Context Precision**
      * *Definition*: A metric measuring whether the most informative chunks are placed at the highest rank positions in the prompt context.
    * **Flashcard 3: Context Recall**
      * *Definition*: The proportion of ground-truth reference facts successfully retrieved into the context window by the embedding search.
    * **Flashcard 4: The RAG Triad**
      * *Definition*: The three core evaluation relationships: Query-to-Context (Relevance), Context-to-Answer (Faithfulness), and Query-to-Answer (Relevance).
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of RAG Evaluation:

- [ ] I can articulate the three axes of The RAG Triad.
- [ ] I know how automated frameworks calculate Faithfulness via atomic claim extraction.
- [ ] I understand the difference between Context Relevance and Context Precision.
- [ ] I can diagnose whether an error is a Retrieval failure or a Generation hallucination.
- [ ] I understand why BLEU and ROUGE are inadequate for evaluating RAG pipelines.
""")
