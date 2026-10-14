import streamlit as st
import json
import time

st.set_page_config(
    page_title="21.3 Evaluation Metrics - Prompt Engineering",
    page_icon="📐",
    layout="wide"
)

# Header Section
st.title("21.3 Evaluation Metrics in Production Prompt Engineering")
st.caption("Chapter 21: Prompt Evaluation and Testing | Domain: 1. Prompt Engineering")
st.markdown("""
Evaluating generative AI systems requires balancing **classical classification metrics** (when answers have deterministic ground truth) 
with **semantic quality metrics** (relevance, faithfulness, hallucination) and **operational SLAs** (latency and token economics). 
A production prompt evaluation dashboard tracks all three dimensions concurrently.
""")

st.divider()

# Section 1: Classical Classification Metrics
st.header("1. Deterministic Metrics: Accuracy, Precision, Recall, and F1", divider="rainbow")
st.markdown("""
When prompts perform extraction, sentiment classification, entity recognition, or intent routing, outputs can be compared 
directly against annotated golden sets using classical statistical evaluation metrics.

* **Accuracy**: The fraction of total predictions that matched the golden target exactly. Vulnerable to class imbalance.
* **Precision**: When the prompt predicted a class (e.g. 'FRAUD' or 'URGENT'), how often was it genuinely correct?
* **Recall**: Out of all real instances of a class in the ground truth, what percentage did the prompt catch?
* **F1 Score**: The harmonic mean of Precision and Recall, providing a balanced single score especially in imbalanced distributions.
""")

# Classical Deterministic Metric Evaluation Function
def compute_classification_metrics(predictions, targets, positive_label="POSITIVE"):
    tp = sum(1 for p, t in zip(predictions, targets) if p == positive_label and t == positive_label)
    fp = sum(1 for p, t in zip(predictions, targets) if p == positive_label and t != positive_label)
    fn = sum(1 for p, t in zip(predictions, targets) if p != positive_label and t == positive_label)
    tn = sum(1 for p, t in zip(predictions, targets) if p != positive_label and t != positive_label)
    
    total = len(targets)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }

st.code('''# Classical Deterministic Metric Evaluation Function
def compute_classification_metrics(predictions, targets, positive_label="POSITIVE"):
    tp = sum(1 for p, t in zip(predictions, targets) if p == positive_label and t == positive_label)
    fp = sum(1 for p, t in zip(predictions, targets) if p == positive_label and t != positive_label)
    fn = sum(1 for p, t in zip(predictions, targets) if p != positive_label and t == positive_label)
    tn = sum(1 for p, t in zip(predictions, targets) if p != positive_label and t != positive_label)
    
    total = len(targets)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }''', language="python")

st.divider()

# Section 2: Semantic Quality Metrics
st.header("2. Semantic Metrics: Relevance, Faithfulness, and Hallucination Rate", divider="rainbow")
st.markdown("""
For open-ended generation, summarization, and RAG pipelines, exact string matching fails because multiple formulations are equally valid. 
Production teams evaluate semantic alignment across three foundational pillars:
""")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.subheader("Context Relevance")
    st.markdown("""
    Measures whether the retrieved context or provided prompt background contains **only pertinent information** without distracting noise.
    * High relevance: 90%+ sentences address user query.
    * Low relevance: Bloated prompt context causing needle-in-haystack degradation.
    """)

with col_m2:
    st.subheader("Answer Faithfulness")
    st.markdown("""
    Measures whether every single factual claim made in the LLM's response is **strictly grounded in the provided source context**.
    * High faithfulness: 100% of claims cite retrieved docs.
    * Low faithfulness: Model introduces unsupported assumptions.
    """)

with col_m3:
    st.subheader("Hallucination Rate")
    st.markdown("""
    The percentage of claims generated that are factually false, unverified, or contradict source material.
    * Goal in enterprise: < 1.0% hallucination rate.
    * Tracked via LLM-as-a-judge claim decomposition.
    """)

st.markdown("#### The Claim Decomposition Algorithm for Faithfulness & Hallucination")
st.code('''# Algorithmic Claim Verification Pipeline
import re

def evaluate_faithfulness_and_hallucination(response_text: str, context_docs: list[str]) -> dict:
    """
    Decomposes response into discrete factual claims and validates each against context.
    """
    # 1. Break response into atomic claims (simulated sentence parsing)
    sentences = [s.strip() for s in re.split(r'[.!?]', response_text) if len(s.strip()) > 5]
    if not sentences:
        return {"faithfulness": 1.0, "hallucination_rate": 0.0, "total_claims": 0}
        
    unified_context = " ".join(context_docs).lower()
    
    grounded_claims = 0
    hallucinated_claims = 0
    
    for sentence in sentences:
        words = [w.lower() for w in sentence.split() if len(w) > 3]
        # Check if core keywords reside in source context
        overlap = sum(1 for w in words if w in unified_context)
        overlap_ratio = overlap / len(words) if words else 1.0
        
        if overlap_ratio >= 0.6:
            grounded_claims += 1
        else:
            hallucinated_claims += 1
            
    faithfulness = grounded_claims / len(sentences)
    hallucination_rate = hallucinated_claims / len(sentences)
    
    return {
        "total_claims": len(sentences),
        "grounded_claims": grounded_claims,
        "hallucinated_claims": hallucinated_claims,
        "faithfulness_score": round(faithfulness, 2),
        "hallucination_rate": round(hallucination_rate, 2)
    }''', language="python")

st.divider()

# Section 3: Operational Metrics: Latency and Cost
st.header("3. Operational Metrics: Latency, Throughput, and Token Cost", divider="rainbow")
st.markdown("""
Prompt quality cannot be evaluated in a vacuum. A prompt that achieves 98% accuracy by requiring 6 round-trips and 12,000 reasoning tokens 
may be economically unfeasible or violate user SLA requirements.
""")

col_op1, col_op2 = st.columns(2)
with col_op1:
    st.markdown("""
    #### Latency Metrics
    * **Time to First Token (TTFT)**: Elapsed time between sending request and first character appearing (User perceived snappiness).
    * **Time Per Output Token (TPOT)**: Token generation velocity (e.g. 40 tokens/sec).
    * **Total Latency (P50, P90, P99)**: End-to-end elapsed request duration across distribution percentiles.
    """)
with col_op2:
    st.markdown("""
    #### Cost & Token Economics
    * **Input Token Cost**: Prompt template + retrieved context + conversation history.
    * **Output Token Cost**: Cost per generated completion token (typically 3x to 4x input cost).
    * **Cost per Successful Query (CPSQ)**: `(Total API Cost) / (Number of Queries meeting Acceptance Threshold)`.
    """)

st.divider()

# Section 4: Interactive Metrics Workbench
st.header("4. Interactive Metrics Evaluation Workbench", divider="rainbow")
st.markdown("Test deterministic and semantic metric evaluations live on synthetic test batches.")

tab1, tab2 = st.columns(2)

with tab1:
    st.subheader("Deterministic Batch Classifier")
    selected_scenario = st.selectbox(
        "Choose Evaluation Scenario:",
        ["Intent Routing (Balanced)", "Compliance Flagging (Imbalanced Rare Event)", "Low Precision Over-Triggering"]
    )
    
    if selected_scenario == "Intent Routing (Balanced)":
        targets = ["BILLING", "TECH_SUPPORT", "BILLING", "SALES", "TECH_SUPPORT", "BILLING", "SALES", "TECH_SUPPORT"]
        preds   = ["BILLING", "TECH_SUPPORT", "SALES",   "SALES", "TECH_SUPPORT", "BILLING", "SALES", "BILLING"]
        pos_target = "BILLING"
    elif selected_scenario == "Compliance Flagging (Imbalanced Rare Event)":
        targets = ["SAFE", "SAFE", "SAFE", "VIOLATION", "SAFE", "SAFE", "SAFE", "VIOLATION", "SAFE", "SAFE"]
        preds   = ["SAFE", "SAFE", "SAFE", "SAFE",      "SAFE", "SAFE", "SAFE", "VIOLATION", "SAFE", "SAFE"]
        pos_target = "VIOLATION"
    else:
        targets = ["SAFE", "SAFE", "SAFE", "SAFE", "VIOLATION", "SAFE", "SAFE", "SAFE", "SAFE", "SAFE"]
        preds   = ["VIOLATION", "VIOLATION", "SAFE", "SAFE", "VIOLATION", "SAFE", "VIOLATION", "SAFE", "SAFE", "SAFE"]
        pos_target = "VIOLATION"
        
    det_results = compute_classification_metrics(preds, targets, positive_label=pos_target)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Accuracy", f"{det_results['Accuracy'] * 100:.1f}%")
    m_col2.metric("Precision", f"{det_results['Precision'] * 100:.1f}%")
    m_col3.metric("Recall", f"{det_results['Recall'] * 100:.1f}%")
    m_col4.metric("F1 Score", f"{det_results['F1_Score'] * 100:.1f}%")
    
    st.caption(f"Raw Matrix (Class '{pos_target}'): TP={det_results['TP']}, FP={det_results['FP']}, FN={det_results['FN']}, TN={det_results['TN']}")

with tab2:
    st.subheader("Cost & Latency Budget Simulator")
    input_tokens = st.slider("Average Input Tokens per Request", 100, 8000, 1500, step=100)
    output_tokens = st.slider("Average Output Tokens per Request", 50, 2000, 300, step=50)
    daily_volume = st.number_input("Daily Production Request Volume", value=50000, step=10000)
    
    # Model pricing tiers per 1M tokens ($)
    input_rate = 0.50 / 1_000_000
    output_rate = 1.50 / 1_000_000
    
    daily_input_cost = daily_volume * input_tokens * input_rate
    daily_output_cost = daily_volume * output_tokens * output_rate
    total_daily = daily_input_cost + daily_output_cost
    monthly_cost = total_daily * 30
    
    st.metric("Estimated Daily Cost", f"${total_daily:,.2f}")
    st.metric("Estimated Monthly Cost (30 Days)", f"${monthly_cost:,.2f}")
    st.caption("Assumes $0.50/M input tokens and $1.50/M output tokens.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa_col1, qa_col2 = st.columns(2)

with qa_col1:
    with st.expander("Q: Why is Accuracy a dangerous metric for safety moderation prompts?"):
        st.write("""
        In safety moderation, violations occur rarely (e.g. 1% of queries). A trivial prompt that always outputs 'SAFE' 
        achieves 99% Accuracy while having a 0% Recall on actual violations. Precision, Recall, and PR-AUC are far more 
        reliable indicators than raw Accuracy for imbalanced classification tasks.
        """)

    with st.expander("Q: What is the mathematical and operational difference between Faithfulness and Answer Relevance?"):
        st.write("""
        * **Faithfulness** verifies that claims made in the answer are strictly supported by the reference context (prevents hallucination).
        * **Answer Relevance** verifies that the answer directly addresses the user's inquiry, regardless of whether it extracted it from context.
        An answer can be 100% faithful to the context (reciting facts about dogs) while being 0% relevant (user asked about quantum physics).
        """)

with qa_col2:
    with st.expander("Trap to Avoid: Evaluating Prompts on Single Averaged Scores"):
        st.write("""
        Never summarize a prompt rollout by a single aggregate number (e.g., 'our prompt got 89% quality'). 
        Always disaggregate by error categories:
        1. Format violations (JSON parsing errors)
        2. Context hallucinations
        3. P99 tail latency breaches
        4. Edge-case safety failures
        """)

    with st.expander("Flashcards: Core Metric Definitions"):
        card = st.selectbox(
            "Select Metric Flashcard:",
            ["Time to First Token (TTFT)", "Hallucination Rate", "Harmonic Mean (F1)", "CPSQ (Cost per Success)"]
        )
        if card == "Time to First Token (TTFT)":
            st.info("The latency from client dispatch until the very first stream chunk arrives at the user interface.")
        elif card == "Hallucination Rate":
            st.info("Percentage of atomic claims generated by the LLM that cannot be substantiated by verified reference context.")
        elif card == "Harmonic Mean (F1)":
            st.info("The balance between Precision and Recall: 2 * (P * R) / (P + R). Penalizes extreme imbalance between the two.")
        else:
            st.info("Total cost divided by queries passing quality guardrails; prevents celebrating cheap prompts that produce garbage.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
c1, c2 = st.columns(2)
with c1:
    st.checkbox("I know when to apply classical F1 vs semantic LLM-as-a-judge metrics.")
    st.checkbox("I can define and differentiate Faithfulness from Answer Relevance.")
with c2:
    st.checkbox("I understand why accuracy is deceptive on imbalanced prompt classification.")
    st.checkbox("I can calculate token cost economics and latency percentiles (P50 vs P99).")
