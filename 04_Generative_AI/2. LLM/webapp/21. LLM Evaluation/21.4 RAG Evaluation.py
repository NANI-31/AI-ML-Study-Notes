import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="21.4 RAG Evaluation - LLM Engineering",
    page_icon="🎯",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class RAGTriadEvaluator:
    """
    Self-contained in-memory RAG Triad evaluation engine implementing:
    - Context Relevance: (Relevant sentences in context) / (Total sentences in context)
    - Groundedness / Faithfulness: (Supported claims in answer) / (Total claims in answer)
    - Answer Relevance: Semantic alignment between query and generated answer
    - Composite Ragas Triad Score
    """

    @staticmethod
    def evaluate_triad(query: str, context: str, answer: str) -> Dict[str, Any]:
        q_words = set(query.lower().split())
        c_sentences = [s.strip() for s in context.split(".") if len(s.strip()) > 5]
        a_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 5]
        
        # 1. Context Relevance
        # Ratio of context sentences containing query keyword overlap
        relevant_context = 0
        for sent in c_sentences:
            if any(w in sent.lower() for w in q_words if len(w) > 3):
                relevant_context += 1
        context_relevance = relevant_context / len(c_sentences) if c_sentences else 0.0

        # 2. Groundedness / Faithfulness
        # Ratio of answer sentences supported by context text
        c_full = context.lower()
        supported_claims = 0
        for sent in a_sentences:
            s_words = [w for w in sent.lower().split() if len(w) > 3]
            overlap = sum(1 for w in s_words if w in c_full)
            if len(s_words) > 0 and (overlap / len(s_words)) > 0.4:
                supported_claims += 1
        groundedness = supported_claims / len(a_sentences) if a_sentences else 0.0

        # 3. Answer Relevance
        # Overlap of answer with original query intent
        ans_words = set(answer.lower().split())
        q_overlap = len(q_words.intersection(ans_words))
        answer_relevance = min(1.0, (q_overlap / len(q_words)) * 1.5) if q_words else 0.0

        # Composite Harmonic Triad Score
        triad_harmonic = (3 * context_relevance * groundedness * answer_relevance) / (
            context_relevance * groundedness + groundedness * answer_relevance + context_relevance * answer_relevance + 1e-9
        )

        return {
            "context_relevance": round(context_relevance, 3),
            "groundedness": round(groundedness, 3),
            "answer_relevance": round(answer_relevance, 3),
            "composite_triad_score": round(max(0.0, min(1.0, triad_harmonic)), 3)
        }

# --- TOP-LEVEL HEADER ---
st.title("21.4 RAG Evaluation: The RAG Triad, Retrieval Metrics, & Groundedness")
st.caption("Chapter 21: LLM Evaluation | Master Architecture Specification")
st.markdown("""
Evaluating a Retrieval-Augmented Generation (RAG) system is fundamentally a two-stage challenge: 
one must evaluate the **Retriever** (did we fetch the right knowledge?) independently from the **Generator** 
(did the LLM produce a faithful answer from that knowledge?). 
The industry-standard framework for RAG evaluation is the **RAG Triad** (Ragas / TruLens).
""")

st.divider()

# --- SECTION 1: BLUEPRINT ARCHITECTURE ---
st.header("1. Blueprint Architecture: The RAG Triad & Retrieval-Generation Pipeline", divider="rainbow")

st.markdown("""
The architecture below illustrates the three interlocking evaluation vectors of the RAG Triad: 
**Context Relevance**, **Groundedness (Faithfulness)**, and **Answer Relevance**.
""")

dot_rag = """
digraph RAGTriadBlueprint {
    graph [bgcolor="#0B1120", rankdir=TB, pad="0.5", nodesep="0.5", ranksep="0.6"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
    edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

    subgraph cluster_elements {
        label = "Core RAG Interaction Elements";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#38BDF8";
        UserQuery [label="User Query (Prompt)", fillcolor="#1E3A8A", color="#3B82F6"];
        RetrievedContext [label="Retrieved Context (Chunks)", fillcolor="#1E3A8A", color="#3B82F6"];
        GeneratedAnswer [label="Generated LLM Answer", fillcolor="#1E3A8A", color="#3B82F6"];
    }

    subgraph cluster_triad {
        label = "The RAG Triad Evaluation Pillars";
        color = "#334155";
        style = "filled";
        fillcolor = "#1E293B";
        fontcolor = "#F59E0B";
        ContextRel [label="1. Context Relevance\\n[Query <-> Retrieved Context]\\nIs the context free of irrelevant noise?", fillcolor="#78350F", color="#D97706"];
        Groundedness [label="2. Groundedness / Faithfulness\\n[Retrieved Context <-> Generated Answer]\\nIs the answer 100% supported by context?", fillcolor="#78350F", color="#D97706"];
        AnswerRel [label="3. Answer Relevance\\n[User Query <-> Generated Answer]\\nDoes the answer satisfy the user's intent?", fillcolor="#78350F", color="#D97706"];
    }

    UserQuery -> ContextRel;
    RetrievedContext -> ContextRel;

    RetrievedContext -> Groundedness;
    GeneratedAnswer -> Groundedness;

    UserQuery -> AnswerRel;
    GeneratedAnswer -> AnswerRel;

    UserQuery -> RetrievedContext [label="Vector / Hybrid Retrieval", style=dotted, color="#94A3B8"];
    RetrievedContext -> GeneratedAnswer [label="LLM Synthesis", style=dotted, color="#94A3B8"];
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_rag, width="stretch")

st.divider()

# --- SECTION 2: TOPIC 1 - RETRIEVAL PRECISION ---
st.header("2. Retrieval Precision: Signal-to-Noise Ratio & Precision@K", divider="rainbow")

st.markdown("""
**Retrieval Precision** measures how cleanly the retriever extracts only relevant information, 
without flooding the prompt with distractor noise.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Precision@K Formulation")
    st.markdown("""
    `Precision@K = (Relevant Chunks in Top-K) / K`
    
    - If `K = 5` chunks are retrieved, and 4 contain information relevant to the question, `Precision@5 = 4 / 5 = 0.80`.
    - **Mean Average Precision (MAP)**: Averages precision across varying cutoff points, rewarding systems that place relevant chunks higher in the ranking.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: The Distractor Penalty
    Even if an LLM has a 128k context window, injecting irrelevant chunks with low retrieval precision 
    causes **Lost in the Middle** degradations: the model's attention fails to locate the gold needle amidst 
    surrounding distractor noise, increasing hallucination rates by up to 35%!
    """)

st.divider()

# --- SECTION 3: TOPIC 2 - RETRIEVAL RECALL ---
st.header("3. Retrieval Recall: Hit Rate, MRR, & Coverage Completeness", divider="rainbow")

st.markdown("""
**Retrieval Recall** measures whether the retriever succeeded in capturing *all* pieces of information 
necessary to synthesize a complete answer.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Key Retrieval Metrics")
    st.markdown("""
    - **Recall@K**: `(Relevant Chunks Retrieved in Top-K) / (Total Relevant Chunks in Knowledge Base)`.
    - **Hit Rate@K**: Binary flag indicating whether *at least one* relevant chunk was present in the top-K.
    - **Mean Reciprocal Rank (MRR)**: Evaluates the rank position `r` of the first relevant chunk:  
      `MRR = 1 / r`. (If the first relevant chunk is at rank 1, `MRR = 1.0`; if at rank 4, `MRR = 0.25`).
    """)

with col2:
    st.info("""
    **Multi-Hop Retrieval Recall**:
    For complex queries requiring multi-document synthesis (e.g. *"Compare Acme's Q1 revenue with Beta's Q1 revenue"*), 
    Recall@K must capture chunks for *both* entities; capturing only Acme results in a failed generation.
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - CONTEXT RELEVANCE ---
st.header("4. Context Relevance: Chunk-Query Alignment & Noise Ratio", divider="rainbow")

st.markdown("""
**Context Relevance** (Ragas Metric) evaluates whether the retrieved text chunks are focused exclusively on answering the user query.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Context Relevance Formula")
    st.markdown("""
    `Context_Relevance = (Number of Relevant Sentences in Context) / (Total Sentences in Context)`
    
    1. A Judge LLM extracts only the sentences from the retrieved chunks that directly contribute to answering the query.
    2. The score scales from `0.0` (entirely irrelevant noise) to `1.0` (pure concentrated signal).
    """)

with col2:
    st.warning("""
    **Remediation for Low Context Relevance**:
    If your pipeline exhibits low context relevance:
    - Reduce chunk sizes (e.g., from 1000 tokens to 300 tokens).
    - Deploy a **Cross-Encoder Reranker** (Cohere / BGE-Reranker) to prune low-scoring passages.
    - Implement Parent-Document retrieval (match small sentences, retrieve parent paragraphs).
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - CONTEXT FAITHFULNESS ---
st.header("5. Context Faithfulness: Hallucination Quantification & Claim Verification", divider="rainbow")

st.markdown("""
**Context Faithfulness** (or Groundedness) measures whether the claims in the generated response 
can be logically inferred strictly from the retrieved context.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Claim Verification Protocol")
    st.markdown("""
    `Faithfulness = (Number of Claims Supported by Context) / (Total Claims in Answer)`
    
    1. **Decomposition**: The Evaluator decomposes the generated answer into individual atomic factual claims.
    2. **Entailment Verification**: For each claim, a Natural Language Inference (NLI) model checks:  
       `Context |= Claim` (Does the context entail the claim?).
    3. If the answer introduces external training data facts not present in the context, faithfulness drops!
    """)

with col2:
    st.error("""
    **The Extrinsic Hallucination Failure**:
    Even if a claim is objectively true in the real world (e.g. *"Apple was founded in 1976"*), 
    if that fact is **not present in the retrieved context**, it is classified as an unfaithful extrinsic hallucination 
    in enterprise RAG, because the system cannot verify its provenance.
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - ANSWER RELEVANCE ---
st.header("6. Answer Relevance: Query Intent Fulfillment & Non-Committal Responses", divider="rainbow")

st.markdown("""
**Answer Relevance** evaluates whether the generated completion directly answers the user's question, 
regardless of whether the answer is factually correct.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Reverse-Query Embedding Formulation")
    st.markdown("""
    Ragas computes Answer Relevance by prompting an LLM to generate `N` hypothetical questions that 
    the candidate answer would answer:
    
    `Answer_Relevance = (1 / N) * sum(Cosine_Similarity(Embed(Generated_Question_i), Embed(Original_User_Query)))`
    
    - If the user asks *"What are the risks of drug X?"* and the model outputs *"Drug X was created in Germany in 2005"*, 
      the generated questions will not align with the original query, producing a low score.
    """)

with col2:
    st.info("""
    **The 'I don't know' Dilemma**:
    If the context contains no answer, an honest model outputting *"I cannot find this information in the provided context"* 
    exhibits high Groundedness (1.0), but receives a lower Answer Relevance score. Production evaluators exempt 
    graceful refusal statements from Answer Relevance penalties.
    """)

st.divider()

# --- SECTION 7: TOPIC 6 - GROUNDEDNESS ---
st.header("7. Groundedness & Source Attribution: Citations & Evidence Linking", divider="rainbow")

st.markdown("""
**Groundedness** verifies source attribution: does the generated answer provide precise inline citations 
linking claims directly back to specific document IDs or paragraph chunks?
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Citation Precision & Recall")
    st.markdown("""
    - **Citation Precision**: Of all citations emitted by the LLM (e.g. `[Doc 1]`), what percentage actually support the preceding sentence?
    - **Citation Recall**: What percentage of factual claims in the answer have an accompanying citation?
    - **Phantom Citations**: Hallucinated citation markers pointing to document IDs that were never retrieved in the context window.
    """)

with col2:
    st.code("""
# Citation Evaluation Assertion
for citation in answer.citations:
    assert citation.doc_id in retrieved_context.doc_ids, "Phantom Citation Error!"
    assert nli_entailment(citation.text, retrieved_context[citation.doc_id]), "Unsupported Citation!"
    """, language="python")

st.divider()

# --- SECTION 8: PRODUCTION CODE IMPLEMENTATION ---
st.header("8. Production Implementation: End-to-End RAG Triad Evaluator", divider="rainbow")

st.markdown("""
Below is an enterprise RAG Triad evaluation engine implementing Context Relevance, Faithfulness, 
and Answer Relevance with threshold gating.
""")

st.code('''
from typing import Dict, List, Any
import numpy as np

class ProductionRAGTriadEvaluator:
    """
    Automated RAG Triad evaluation engine for enterprise QA pipelines.
    """
    def __init__(self, min_context_rel: float = 0.5, min_faithfulness: float = 0.8, min_ans_rel: float = 0.7):
        self.min_context_rel = min_context_rel
        self.min_faithfulness = min_faithfulness
        self.min_ans_rel = min_ans_rel

    def evaluate_turn(self, query: str, context_chunks: List[str], answer: str) -> Dict[str, Any]:
        """Evaluates a RAG interaction across the Triad."""
        full_context = " ".join(context_chunks)
        
        # 1. Context Relevance
        q_tokens = set(query.lower().split())
        matched_chunks = sum(1 for c in context_chunks if any(w in c.lower() for w in q_tokens if len(w) > 3))
        context_rel = matched_chunks / len(context_chunks) if context_chunks else 0.0

        # 2. Groundedness / Faithfulness
        ans_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 5]
        supported = sum(1 for s in ans_sentences if any(w in full_context.lower() for w in s.lower().split() if len(w) > 4))
        faithfulness = supported / len(ans_sentences) if ans_sentences else 0.0

        # 3. Answer Relevance
        ans_tokens = set(answer.lower().split())
        overlap = len(q_tokens.intersection(ans_tokens))
        answer_rel = min(1.0, overlap / len(q_tokens)) if q_tokens else 0.0

        passed = (
            context_rel >= self.min_context_rel and
            faithfulness >= self.min_faithfulness and
            answer_rel >= self.min_ans_rel
        )

        return {
            "passed_sla": passed,
            "context_relevance": round(context_rel, 3),
            "faithfulness": round(faithfulness, 3),
            "answer_relevance": round(answer_rel, 3),
            "verdict": "PRODUCTION_READY" if passed else "REJECTED_BY_TRIAD_GATE"
        }
''', language="python")

st.divider()

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive RAG Triad Lab: Live Quality & Faithfulness Studio", divider="rainbow")

st.markdown("""
Experiment with the live RAG Triad Evaluator below. Modify the query, context chunks, and candidate answer 
to observe how **Context Relevance**, **Groundedness**, and **Answer Relevance** fluctuate.
""")

col1, col2 = st.columns([1, 2])

with col1:
    default_q = "What was Acme Corp's operating margin in Q1 2025?"
    default_ctx = "Acme Corp reported $4.2B revenue in Q1 2025. Operating margin reached 24%. The company also announced plans to expand its headquarters in Austin, Texas."
    default_ans = "Acme Corp achieved an operating margin of 24% in Q1 2025."
    
    t_query = st.text_input("User Query:", value=default_q)
    t_context = st.text_area("Retrieved Context Chunks:", value=default_ctx, height=100)
    t_answer = st.text_area("Generated Model Answer:", value=default_ans, height=80)
    
    eval_btn = st.button("🎯 Evaluate RAG Triad", type="primary", use_container_width=True)

with col2:
    evaluator = RAGTriadEvaluator()
    
    if eval_btn or True:
        scores = evaluator.evaluate_triad(t_query, t_context, t_answer)
        
        st.subheader("RAG Triad Scorecard")
        c1, c2, c3 = st.columns(3)
        c1.metric("Context Relevance", f"{scores['context_relevance'] * 100:.1f}%")
        c2.metric("Groundedness / Faithfulness", f"{scores['groundedness'] * 100:.1f}%")
        c3.metric("Answer Relevance", f"{scores['answer_relevance'] * 100:.1f}%")
        
        st.subheader("Composite Triad Harmonic Score")
        comp = scores["composite_triad_score"]
        st.progress(float(comp))
        st.markdown(f"**Composite Score**: `{comp} / 1.0`")
        
        if comp > 0.70:
            st.success("✅ RAG Interaction meets Production SLA Standards.")
        else:
            st.warning("⚠️ RAG Interaction flagged for potential hallucination or noise.")

st.divider()

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Staff-Level Engineering Q&A")
with st.expander("Q1: What are the three pillars of the RAG Triad, and how do their failure modes differ?"):
    st.markdown("""
    **Answer**:
    1. **Context Relevance**: Fails when the retriever fetches bloated, noisy chunks that do not relate to the query. (Symptom: High token costs, lost in the middle attention dilution).
    2. **Groundedness (Faithfulness)**: Fails when the LLM generates claims that cannot be derived from the retrieved context. (Symptom: Factual hallucination).
    3. **Answer Relevance**: Fails when the LLM generates a response that is grounded in context, but does not actually address the user's intent. (Symptom: Off-topic or non-committal answers).
    """)

with st.expander("Q2: Why is Extrinsic Hallucination particularly dangerous in enterprise RAG?"):
    st.markdown("""
    **Answer**:
    An extrinsic hallucination occurs when the model outputs a fact from its pre-training weights that is not present in the retrieved context documents. 
    Even if the claim is historically true, in enterprise use cases (legal discovery, internal compliance, financial audits), the business demands that answers be backed by verifiable internal sources. 
    Extrinsic claims break the verifiable audit trail.
    """)

with st.expander("Q3: How does Ragas calculate Answer Relevance without a ground truth reference answer?"):
    st.markdown("""
    **Answer**:
    Ragas uses **Reverse Question Generation**:
    1. It prompts an LLM to generate 3 hypothetical questions that the model's generated answer answers.
    2. It embeds each hypothetical question using a dense embedding model.
    3. It computes the cosine similarity between the embeddings of the hypothetical questions and the original user query. 
    If the answer is on-topic, the generated questions will closely align semantically with the user's prompt.
    """)

with st.expander("Q4: What is a Phantom Citation and how can it be detected automatically?"):
    st.markdown("""
    **Answer**:
    A Phantom Citation occurs when an LLM outputs inline citation markers (e.g. `[Doc 4]`) to give an impression of authority, 
    but Document 4 was never provided in the context window, or does not contain the cited claim. 
    Automated regex extraction verifies that every citation marker maps to a valid context chunk ID, followed by an NLI entailment check between the cited sentence and chunk text.
    """)

with st.expander("Q5: What metrics should be used to evaluate the Retriever independently of the LLM generator?"):
    st.markdown("""
    **Answer**:
    - **Hit Rate@K**: Fraction of queries where at least one ground-truth document was in the top-K.
    - **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant document.
    - **Normalized Discounted Cumulative Gain (NDCG@K)**: Measures ranking quality, giving higher credit when more relevant documents appear earlier in the top-K list.
    """)

st.subheader("Production Traps & Failure Modes")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error("""
    **TRAP 1 (Error): Evaluating RAG Only End-to-End**
    Grading only the final answer without isolating retrieval. A poor answer can be caused by either a bad retriever or a hallucinating generator; failing to isolate them makes root-cause debugging impossible.
    """)
    st.error("""
    **TRAP 2 (Error): Unchecked Phantom Citations**
    Displaying model-generated citations to users without programmatic validation. Users place blind trust in fake citations that do not support the claim.
    """)

with col_t2:
    st.warning("""
    **TRAP 3 (Warning): Dilution from Bloated Chunk Sizes**
    Using 2,000-token chunks to achieve high Recall@K. The large chunks tank Context Relevance, filling the prompt with 90% irrelevant noise.
    """)
    st.warning("""
    **TRAP 4 (Warning): Penalizing Valid 'I don't know' Refusals**
    Allowing Answer Relevance scorers to penalize models that correctly refuse to answer when the knowledge base lacks necessary data.
    """)

st.subheader("Architectural Flashcards")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    st.markdown("""
    **Context Relevance**  
    The proportion of retrieved context that is directly useful for answering the query.
    """)
with fc2:
    st.markdown("""
    **Groundedness**  
    The fraction of claims in the generated response that are strictly entailed by the context.
    """)
with fc3:
    st.markdown("""
    **Answer Relevance**  
    The degree to which the generated completion satisfies the user's original intent.
    """)
with fc4:
    st.markdown("""
    **NDCG@K**  
    Rank-aware metric that penalizes relevant documents appearing low in retrieval rankings.
    """)

st.divider()

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist: RAG Evaluation Mastery", divider="rainbow")

st.markdown("Verify your understanding of RAG evaluation systems by reviewing each criterion:")

chk1 = st.checkbox("I can differentiate Retriever evaluation (Hit Rate, MRR, NDCG) from Generator evaluation.", key="rag_chk1")
chk2 = st.checkbox("I understand the three pillars of the RAG Triad (Context Relevance, Groundedness, Answer Relevance).", key="rag_chk2")
chk3 = st.checkbox("I know how Ragas uses Reverse Question Generation to compute Answer Relevance without golden answers.", key="rag_chk3")
chk4 = st.checkbox("I can identify and programmatically detect Phantom Citations in generative answers.", key="rag_chk4")
chk5 = st.checkbox("I understand why Extrinsic Hallucinations violate enterprise compliance requirements.", key="rag_chk5")
chk6 = st.checkbox("I know how to calibrate evaluation gates to avoid penalizing faithful 'I don't know' refusals.", key="rag_chk6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Master Achievement Unlocked: You have demonstrated comprehensive mastery of RAG Evaluation and the RAG Triad Architecture!")
