import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class RAGQualityTestCase:
    """Represents a curated evaluation test case with ground truth references."""
    def __init__(
        self,
        query: str,
        ground_truth: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        relevant_context_indices: List[int]
    ):
        self.query = query
        self.ground_truth = ground_truth
        self.retrieved_contexts = retrieved_contexts
        self.generated_answer = generated_answer
        self.relevant_context_indices = relevant_context_indices

class MockRAGQualityStudio:
    """
    Simulates the Ragas RAG Triad evaluation metrics:
    Context Relevance, Faithfulness, and Answer Relevance alongside IR Precision & Recall.
    """
    def __init__(self):
        self.test_cases = {
            "SLA Uptime Query": RAGQualityTestCase(
                query="What is the enterprise cloud failover timeout and uptime SLA?",
                ground_truth="The enterprise cloud infrastructure guarantees 99.995% uptime with automated 12-second failover on synchronous RDS clusters.",
                retrieved_contexts=[
                    "Enterprise cloud infrastructure guarantees 99.995% uptime for multi-region active-active deployments.",
                    "Automated failover triggers within 12 seconds of health-check timeout on synchronous RDS PostgreSQL clusters.",
                    "Employee parking permits are issued by the facilities team on the second Tuesday of each month."
                ],
                generated_answer="The cloud infrastructure provides 99.995% uptime with automated failover initiating in 12 seconds.",
                relevant_context_indices=[0, 1]
            ),
            "IAM Credential Query": RAGQualityTestCase(
                query="How often do IAM credentials expire and what MFA is required?",
                ground_truth="IAM credentials expire every 60 minutes and administrative API access requires FIDO2 hardware MFA.",
                retrieved_contexts=[
                    "IAM role credentials expire every 60 minutes.",
                    "Mandatory multi-factor authentication (MFA) with FIDO2 hardware keys is enforced for all administrative API access.",
                    "Annual performance appraisals must be completed by November 30."
                ],
                generated_answer="IAM credentials expire every 60 minutes, and FIDO2 hardware keys are enforced for administrative MFA.",
                relevant_context_indices=[0, 1]
            ),
            "Hallucination Sample": RAGQualityTestCase(
                query="What is the SLA for Redis cache cluster backups?",
                ground_truth="Redis cache clusters execute automated daily snapshots at 02:00 UTC with 7-day retention.",
                retrieved_contexts=[
                    "Redis clusters utilize sliding window counters for ingress rate limiting.",
                    "Database read replicas replicate asynchronously from the primary cluster."
                ],
                generated_answer="Redis cache backups are conducted every 4 hours and replicated across 3 AWS availability zones.",
                relevant_context_indices=[]
            )
        }

    def evaluate_test_case(self, case_name: str, chunk_budget: int) -> Dict[str, Any]:
        case = self.test_cases[case_name]
        active_contexts = case.retrieved_contexts[:chunk_budget]

        # 1. Context Precision: fraction of retrieved contexts that are relevant
        rel_in_budget = [idx for idx in case.relevant_context_indices if idx < chunk_budget]
        context_precision = round(len(rel_in_budget) / max(1, len(active_contexts)), 3)

        # 2. Context Recall: fraction of ground truth relevant contexts retrieved
        context_recall = round(len(rel_in_budget) / max(1, len(case.relevant_context_indices)), 3) if case.relevant_context_indices else 0.0

        # 3. Context Relevance: simulated density of query-relevant keywords in context
        q_tokens = set(case.query.lower().split())
        c_tokens = set(" ".join(active_contexts).lower().split())
        overlap = len(q_tokens.intersection(c_tokens))
        context_relevance = round(min(1.0, max(0.1, overlap / max(1, len(q_tokens)))), 3)

        # 4. Faithfulness: fraction of claims in generated answer supported by active context
        if case_name == "Hallucination Sample":
            faithfulness = 0.15 # Severe ungrounded hallucination
            hallucination_detected = True
        else:
            faithfulness = 0.95
            hallucination_detected = False

        # 5. Answer Relevance: alignment between question and generated answer
        ans_tokens = set(case.generated_answer.lower().split())
        ans_overlap = len(q_tokens.intersection(ans_tokens))
        answer_relevance = round(min(1.0, max(0.2, (ans_overlap * 0.25) + 0.3)), 3)

        return {
            "query": case.query,
            "ground_truth": case.ground_truth,
            "contexts_used": active_contexts,
            "generated_answer": case.generated_answer,
            "metrics": {
                "Context Precision": context_precision,
                "Context Recall": context_recall,
                "Context Relevance": context_relevance,
                "Faithfulness": faithfulness,
                "Answer Relevance": answer_relevance
            },
            "hallucination_detected": hallucination_detected
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="16.3 RAG Quality — LangChain Course",
    page_icon="🎯",
    layout="wide"
)

st.title("16.3 RAG Quality")
st.caption("Quantitative Evaluation of Retrieval Precision, Context Relevance, Faithfulness, Hallucination Suppression, and Embedding Benchmarking")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architecture Blueprint: The Enterprise RAG Quality & Evaluation Triad", divider="rainbow")
st.markdown("""
Building a proof-of-concept RAG system takes an afternoon; turning it into a **reliable production system requires 
continuous quantitative evaluation**. 

The **RAG Triad** framework decomposes system performance into three mathematically auditable axes:
1. **Context Relevance**: Does the retriever return signal without noise?
2. **Faithfulness / Groundedness**: Does the LLM answer strictly from the retrieved context?
3. **Answer Relevance**: Does the generated response directly answer the user's question?
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RAGQualityArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", arrowsize=0.8];

        subgraph cluster_triad {
            label = "The Ragas RAG Quality Triad";
            style = "rounded";
            color = "#0284C7";
            fontcolor = "#38BDF8";

            query_node [label="User Question: str", fillcolor="#1E293B", color="#0284C7"];
            context_node [label="Retrieved Context: List[Document]", fillcolor="#0F172A", color="#38BDF8"];
            answer_node [label="Generated Answer: str", fillcolor="#064E3B", color="#10B981"];
        }

        subgraph cluster_metrics {
            label = "Quantitative Quality Metrics";
            style = "rounded,dashed";
            color = "#334155";
            fontcolor = "#94A3B8";

            m_ctx_rel [label="Metric 1: Context Relevance\\nEvaluates retriever signal-to-noise ratio", fillcolor="#1E293B", color="#38BDF8"];
            m_faith [label="Metric 2: Faithfulness\\nDetects ungrounded claims & hallucinations", fillcolor="#1E293B", color="#EF4444"];
            m_ans_rel [label="Metric 3: Answer Relevance\\nMeasures user question fulfillment", fillcolor="#1E293B", color="#10B981"];
            m_rec [label="Metric 4: Context Recall\\nCompares retrieved chunks vs Ground Truth", fillcolor="#1E293B", color="#F59E0B"];
        }

        query_node -> m_ctx_rel [label="query text"];
        context_node -> m_ctx_rel [label="retrieved chunks"];

        context_node -> m_faith [label="grounding reference"];
        answer_node -> m_faith [label="claim verification"];

        query_node -> m_ans_rel [label="intent"];
        answer_node -> m_ans_rel [label="response"];

        context_node -> m_rec [label="retrieved facts"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: RETRIEVAL PRECISION ---
st.header("2. Retrieval Precision", divider="rainbow")
st.markdown("""
**Retrieval Precision** measures the proportion of retrieved chunks that contain genuine ground-truth facts 
relative to the total number of chunks returned.

`Retrieval Precision = (Number of Relevant Chunks Retrieved) / (Total Chunks Retrieved, k)`

If k = 5 chunks are returned, but only 1 chunk contains factual answers while the remaining 4 chunks describe unrelated 
policies, **Precision is 0.20 (20%)**. Low precision surfeits the LLM with distracting context, causing latency inflation 
and lost-in-the-middle degradation.
""")

st.divider()

# --- SECTION 3: RETRIEVAL RECALL ---
st.header("3. Retrieval Recall", divider="rainbow")
st.markdown("""
**Retrieval Recall** measures whether all the essential factual components required to answer the question were successfully fetched.

`Retrieval Recall = (Relevant Chunks Retrieved) / (Total Relevant Chunks in Ground Truth)`

In multi-hop queries (e.g. comparing Aurora PostgreSQL against DynamoDB SLAs), if the retriever only fetches the Aurora document 
and omits the DynamoDB document, **Recall is 0.50 (50%)**. High precision cannot compensate for low recall, as the LLM 
cannot generate facts that were never retrieved.
""")

st.divider()

# --- SECTION 4: CONTEXT RELEVANCE ---
st.header("4. Context Relevance", divider="rainbow")
st.markdown("""
**Context Relevance** quantifies the density of signal within the retrieved text. When using large chunk sizes (e.g. 1,500 tokens), 
a document might contain 1 relevant sentence buried in 50 sentences of unrelated legal boilerplate.

High context relevance is achieved via:
1. **Fine-grained semantic chunking**: Splitting around thematic section boundaries.
2. **Contextual Compression**: Using an extractive model to drop non-essential sentences before injecting context into the prompt.
""")

st.divider()

# --- SECTION 5: FAITHFULNESS ---
st.header("5. Faithfulness & Groundedness", divider="rainbow")
st.markdown("""
**Faithfulness** is the ultimate metric for detecting hallucinations. It measures whether every claim made in the generated 
answer can be logically inferred directly from the provided context chunks.

### Mathematical Evaluation Protocol:
1. Break down the generated answer into discrete atomic claims: S = {s_1, s_2, ..., s_n}.
2. For each claim s_i, prompt an LLM judge to verify if s_i is directly supported by the context.
3. `Faithfulness = (Number of Supported Claims) / (Total Claims in Answer)`.
""")

st.code("""
# Prompt for evaluating Faithfulness
judge_prompt = \"\"\"
Given the context and an answer, evaluate whether every claim in the answer is directly supported by the context.
Context: {context}
Answer: {answer}

List all atomic claims from the answer and mark each as [SUPPORTED] or [UNSUPPORTED].
\"\"\"
""", language="python")

st.divider()

# --- SECTION 6: HALLUCINATION REDUCTION ---
st.header("6. Hallucination Reduction Strategies", divider="rainbow")
st.markdown("""
Production systems deploy defensive engineering layers to drive hallucinations toward zero:
""")

col_h1, col_h2 = st.columns(2)
with col_h1:
    st.subheader("Architectural Safeguards")
    st.markdown("""
    - **Refusal Directives**: Prompt the model: *"If the provided context does not contain the answer, explicitly state 'I do not know'. Never speculate."*
    - **Relevance Thresholding**: If the top reranker score is below 0.55, return an empty context and trigger a deterministic refusal fallback.
    - **Chain-of-Thought Citations**: Require the model to quote exact sentence spans before formulating its response.
    """)
with col_h2:
    st.subheader("Post-Generation Guardrails")
    st.markdown("""
    - **Automated Faithfulness Check**: Run a lightweight LLM judge to verify claims before returning output to the client.
    - **Self-Consistency Consensus**: Sample 3 generations at temperature=0.2; reject responses where factual statements diverge.
    """)

st.divider()

# --- SECTION 7: CHUNK-SIZE OPTIMIZATION ---
st.header("7. Chunk-Size Optimization", divider="rainbow")
st.markdown("""
Chunk size is the single most critical hyperparameter in RAG ingestion. Selecting the optimal chunk size balances 
**semantic precision** against **narrative coherence**.
""")

chunk_df = pd.DataFrame({
    "Chunk Size Range": ["Small (100 - 250 tokens)", "Medium (400 - 600 tokens)", "Large (1,000 - 2,000 tokens)"],
    "Vector Similarity Precision": ["Very High: Tight, specific semantic vectors", "Balanced: Moderate specificity", "Low: Diluted, generalized embeddings"],
    "Context Window Coherence": ["Low: Loses surrounding narrative sentences", "Balanced: Preserves paragraph context", "Very High: Complete section context"],
    "Recommended Application": ["FAQ matching, exact entity lookups, definitions", "General technical docs, API guides, policy manuals", "Complex narrative contracts, case law analysis"]
})
st.dataframe(chunk_df, use_container_width=True)

st.divider()

# --- SECTION 8: EMBEDDING SELECTION ---
st.header("8. Embedding Selection & MTEB Benchmarks", divider="rainbow")
st.markdown("""
Embedding models should be selected based on the **Massive Text Embedding Benchmark (MTEB)** leaderboards. 
Different enterprise constraints dictate distinct choices:
""")

emb_df = pd.DataFrame({
    "Model Name": ["text-embedding-3-small", "text-embedding-3-large", "BAAI/bge-large-en-v1.5", "Cohere embed-v3"],
    "Dimensions": [1536, 3072, 1024, 1024],
    "Deployment Mode": ["OpenAI Cloud API", "OpenAI Cloud API", "Self-hosted HuggingFace / ONNX", "Cohere Cloud API"],
    "Cost / Latency": ["Lowest API Cost, < 20ms", "Higher API Cost, < 40ms", "Zero API Cost (GPU), < 10ms local", "Moderate API Cost, < 30ms"],
    "Best For": ["High-throughput SaaS RAG", "High-precision enterprise search", "Air-gapped, on-prem compliance", "Multi-lingual & compressed embeddings"]
})
st.dataframe(emb_df, use_container_width=True)

st.divider()

# --- SECTION 9: RETRIEVER EVALUATION ---
st.header("9. Quantitative Retriever Evaluation Frameworks", divider="rainbow")
st.markdown("""
Automated evaluation using frameworks like **Ragas** allows engineering teams to implement continuous integration (CI) tests 
for retrieval pipelines, preventing regressions when document chunks or embedding models change.
""")

st.code("""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

# Evaluation dataset structure
eval_dataset = Dataset.from_dict({
    "question": ["What is our uptime SLA?"],
    "contexts": [["Enterprise cloud guarantees 99.995% uptime."]],
    "answer": ["The uptime SLA is 99.995%."],
    "ground_truth": ["Our enterprise cloud uptime SLA is 99.995%."]
})

# Automated metric scoring
results = evaluate(
    eval_dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
)
print(results)
""", language="python")

st.divider()

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Automated RAG Quality Evaluation Harness", divider="rainbow")
st.markdown("""
Below is a self-contained, production-grade Python evaluation harness that calculates Context Precision, 
Context Recall, and Faithfulness across a golden evaluation test suite.
""")

st.code("""
from typing import List, Dict, Any
import numpy as np

class ProductionRAGEvaluationHarness:
    \"\"\"
    Lightweight, deterministic evaluation harness calculating core IR & RAG metrics.
    \"\"\"
    def __init__(self, golden_dataset: List[Dict[str, Any]]):
        self.dataset = golden_dataset

    def compute_context_precision(self, retrieved: List[str], ground_truth_relevant: List[str]) -> float:
        \"\"\"Fraction of retrieved chunks that are relevant.\"\"\"
        if not retrieved:
            return 0.0
        hits = sum(1 for c in retrieved if any(rel in c for rel in ground_truth_relevant))
        return round(hits / len(retrieved), 3)

    def compute_context_recall(self, retrieved: List[str], ground_truth_relevant: List[str]) -> float:
        \"\"\"Fraction of ground-truth relevant chunks that were retrieved.\"\"\"
        if not ground_truth_relevant:
            return 1.0
        hits = sum(1 for rel in ground_truth_relevant if any(rel in c for c in retrieved))
        return round(hits / len(ground_truth_relevant), 3)

    def run_benchmark(self, retriever_pipeline_fn) -> Dict[str, float]:
        precisions = []
        recalls = []

        for item in self.dataset:
            retrieved = retriever_pipeline_fn(item["question"])
            p = self.compute_context_precision(retrieved, item["relevant_spans"])
            r = self.compute_context_recall(retrieved, item["relevant_spans"])
            precisions.append(p)
            recalls.append(r)

        return {
            "Mean Context Precision": float(np.mean(precisions)),
            "Mean Context Recall": float(np.mean(recalls)),
            "Test Suite Count": len(self.dataset)
        }
""", language="python")

st.divider()

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: RAG Quality & Triad Evaluation Studio", divider="rainbow")
st.markdown("""
Inspect how **chunk budget**, **retrieval noise**, and **hallucination detection** impact the Ragas RAG Triad. 
Select test scenarios to see how precision, recall, and faithfulness fluctuate under different retrieval conditions.
""")

# Instantiate module-scope studio
quality_studio = MockRAGQualityStudio()

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    selected_scenario = st.selectbox(
        "Select Curated Evaluation Test Scenario:",
        ["SLA Uptime Query", "IAM Credential Query", "Hallucination Sample"]
    )

    chunk_budget_slider = st.slider(
        "Simulated Retrieval Chunk Budget (k):",
        min_value=1,
        max_value=3,
        value=3
    )

    run_eval_btn = st.button("📊 Evaluate RAG Quality Metrics", type="primary", use_container_width=True)

with wb_col2:
    st.subheader("Automated Quality Assessment")

    if run_eval_btn or selected_scenario:
        eval_res = quality_studio.evaluate_test_case(selected_scenario, chunk_budget_slider)

        st.markdown(f"**Query**: *'{eval_res['query']}'*")
        st.markdown(f"**Generated Answer**: `{eval_res['generated_answer']}`")

        if eval_res["hallucination_detected"]:
            st.error("🚨 CRITICAL ALERT: Hallucination Detected! Generated answer contains ungrounded factual claims not present in context.")
        else:
            st.success("✅ Factual Grounding Verified: Answer is faithful to retrieved context.")

        m_cols = st.columns(3)
        metrics = eval_res["metrics"]
        m_cols[0].metric("Context Precision", f"{metrics['Context Precision'] * 100:.1f}%")
        m_cols[1].metric("Context Recall", f"{metrics['Context Recall'] * 100:.1f}%")
        m_cols[2].metric("Faithfulness", f"{metrics['Faithfulness'] * 100:.1f}%")

        with st.expander("🔍 View Context Chunks Evaluated", expanded=True):
            for i, c in enumerate(eval_res["contexts_used"], 1):
                st.write(f"**Chunk #{i}**: `{c}`")

st.divider()

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What are the three pillars of the Ragas RAG Triad, and how are they measured?",
     "The RAG Triad consists of: (1) Context Relevance: evaluates whether retrieved chunks contain only information relevant to the question, calculated by measuring the ratio of useful sentences to total retrieved text. (2) Faithfulness: measures whether all claims made in the generated answer are grounded in and supported by the retrieved context chunks. (3) Answer Relevance: measures whether the generated answer directly addresses the original query without straying off-topic."),
    ("How does Context Precision differ from Context Recall in information retrieval?",
     "Context Precision measures the signal-to-noise ratio in retrieved context: the proportion of retrieved chunks that contain relevant facts. Context Recall measures coverage: whether all ground-truth facts required to answer the question were successfully retrieved. A system can have 100% precision with low recall (e.g. retrieving 1 perfect chunk but missing the second half of a multi-hop answer)."),
    ("Why does increasing chunk size often decrease retrieval precision while increasing narrative coherence?",
     "Large chunk sizes (e.g. 1500 tokens) preserve whole sections, ensuring complete narrative context for generation. However, because the embedding vector represents the average semantic meaning of all 1500 tokens, specific needle-in-a-haystack statements are diluted, causing vector search to retrieve noisy or tangentially related passages, which degrades retrieval precision."),
    ("How can automated evaluation detect hallucinations in production without human labels?",
     "Automated evaluation uses an LLM judge to extract atomic factual propositions from the generated answer and independently verify each proposition against the retrieved context chunks using natural language inference (NLI). If any proposition cannot be entailed directly from the context, it is flagged as an ungrounded hallucination."),
    ("What is Mean Reciprocal Rank (MRR), and why is it preferred over simple Hit Rate?",
     "Hit Rate @ k only measures whether a relevant document exists somewhere within the top-k results (binary 0 or 1). MRR calculates the reciprocal of the rank where the first relevant document appears: 1 / rank. If the relevant document is at rank 1, MRR is 1.0; if at rank 4, MRR is 0.25. MRR penalizes systems that bury relevant context deep in the prompt.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Evaluating RAG with High-Temperature LLM Judges. Using an LLM judge with temperature=0.7 introduces random scoring variance across evaluation runs. Always lock judge models to temperature=0.0 and seed parameters to ensure deterministic benchmark reproducibility.")
    st.error("**Trap 2 (Breaking Error)**: Relying Solely on Answer Similarity (BLEU/ROUGE). BLEU and ROUGE only evaluate n-gram lexical overlap between generated answers and reference text. An answer can achieve high BLEU overlap while introducing a single negation that completely reverses factual accuracy.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Running Heavy Evaluator Models on 100% of Production Queries. Invoking GPT-4 to evaluate faithfulness on every real-time production query doubles latency and API costs. Sample 1-5% of production traffic for continuous offline monitoring.")
    st.warning("**Trap 4 (Quality Warning)**: Optimizing for Retrieval Precision at the Expense of Recall. Pruning candidate pools too aggressively with high score thresholds (tau > 0.85) creates empty context sets that trigger unnecessary system refusals.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: RAG Triad"):
        st.markdown("**Context Relevance, Faithfulness (Groundedness), and Answer Relevance.**")
    with st.expander("🎴 Flashcard 2: Faithfulness"):
        st.markdown("**Fraction of claims in generated output that are directly entailed by context chunks.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Context Recall"):
        st.markdown("**Proportion of ground-truth factual references successfully fetched by the retriever.**")
    with st.expander("🎴 Flashcard 4: Atomic Claim Extraction"):
        st.markdown("**Decomposing an LLM response into independent propositions for automated NLI verification.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can articulate the three pillars of the Ragas RAG Quality Triad.", value=True)
chk2 = st.checkbox("I understand the difference between Context Precision and Context Recall.", value=True)
chk3 = st.checkbox("I know how to formulate an automated Faithfulness claim verification prompt.", value=True)
chk4 = st.checkbox("I understand the trade-offs of chunk size optimization on precision vs coherence.", value=True)
chk5 = st.checkbox("I can explain how MTEB benchmarks guide embedding model selection for enterprise RAG.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 RAG quality interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 16.3 RAG Quality!")
