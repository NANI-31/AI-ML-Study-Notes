import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="2.2 RAG Workflow - RAG Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title Block & 5-Pillar Overview
st.title("2.2 Retrieval-Augmented Generation Workflow: The 7-Stage End-to-End Execution Lifecycle")
st.caption("Chapter 2: RAG Fundamentals | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
A robust RAG deployment operates as a disciplined, multi-stage state machine that coordinates seven distinct operations 
for every incoming user interaction: **User Query Ingress**, **Query Transformation**, **Retrieval**, 
**Context Selection & Reranking**, **Prompt Construction**, **LLM Generation**, and **Response Validation**.

Each stage in this pipeline acts as an accuracy and latency filter. If query transformation fails to expand 
abbreviations, retrieval fails. If retrieval produces noise, context selection must filter it. If the generator 
overreaches, response validation must catch ungrounded claims before they reach the user.

In this lesson, you will master:
1. The deterministic 7-stage query execution lifecycle and its inter-stage contracts.
2. Advanced query transformation patterns: Query Rewriting, HyDE, and Sub-Query Decomposition.
3. Hybrid multi-modal retrieval fusion and cross-encoder relevance filtering.
4. Structuring bulletproof prompt templates with explicit citation constraints.
5. Production validation gates: Faithfulness scoring, Answer Relevance, and Hallucination Fallbacks.
""")

st.divider()

# Module-Scope Helper Classes for Interactive Workbench & Production Pipeline
class WorkflowStageResult:
    def __init__(self, stage_num: int, stage_name: str, status: str, details: Dict[str, Any]):
        self.stage_num = stage_num
        self.stage_name = stage_name
        self.status = status
        self.details = details

class SevenStageRAGPipeline:
    def __init__(self):
        self.documents = [
            {"id": "DOC-SEC-01", "text": "Production Kubernetes deployments must require mTLS authentication and deny unencrypted ingress traffic."},
            {"id": "DOC-SEC-02", "text": "All service account tokens must be rotated every 14 days using automated Vault sidecars."},
            {"id": "DOC-HR-01", "text": "Employees receive 25 days of paid annual leave plus company-wide holidays."},
            {"id": "DOC-FIN-01", "text": "Q3 2026 enterprise operating profit margin expanded to 28.4% across global operations."}
        ]

    def execute_workflow(self, raw_query: str) -> List[WorkflowStageResult]:
        steps = []
        
        # Stage 1: User Query
        steps.append(WorkflowStageResult(1, "User Query Ingress", "COMPLETED", {"raw_query": raw_query}))
        
        # Stage 2: Query Transformation
        transformed = raw_query.strip().lower()
        if "k8s" in transformed:
            transformed = transformed.replace("k8s", "kubernetes")
        expanded_query = f"{transformed} security policy requirement"
        steps.append(WorkflowStageResult(2, "Query Transformation", "COMPLETED", {
            "original": raw_query,
            "transformed_query": expanded_query,
            "transform_technique": "Acronym Expansion + Semantic Enrichment"
        }))
        
        # Stage 3: Retrieval
        query_words = set(expanded_query.split())
        candidates = []
        for doc in self.documents:
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words.intersection(doc_words))
            candidates.append({"id": doc["id"], "text": doc["text"], "overlap_score": overlap})
        candidates.sort(key=lambda x: x["overlap_score"], reverse=True)
        steps.append(WorkflowStageResult(3, "Retrieval", "COMPLETED", {
            "total_candidates": len(candidates),
            "top_candidates": candidates[:3]
        }))
        
        # Stage 4: Context Selection
        selected_chunks = [c for c in candidates[:2] if c["overlap_score"] > 0]
        steps.append(WorkflowStageResult(4, "Context Selection & Reranking", "COMPLETED", {
            "retained_chunks": selected_chunks,
            "dropped_count": len(candidates) - len(selected_chunks)
        }))
        
        # Stage 5: Prompt Construction
        context_str = "\n".join([f"[{c['id']}] {c['text']}" for c in selected_chunks])
        prompt_constructed = (
            f"SYSTEM: Answer using ONLY context. If unknown, state 'Unknown'.\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUERY: {raw_query}\nANSWER:"
        )
        steps.append(WorkflowStageResult(5, "Prompt Construction", "COMPLETED", {
            "assembled_prompt": prompt_constructed,
            "token_estimate": len(prompt_constructed.split()) * 1.3
        }))
        
        # Stage 6: LLM Generation
        if selected_chunks:
            generated_answer = f"According to internal records ({selected_chunks[0]['id']}), {selected_chunks[0]['text'][:60]}..."
        else:
            generated_answer = "Insufficient verified context to answer this query safely."
        steps.append(WorkflowStageResult(6, "LLM Generation", "COMPLETED", {
            "generated_text": generated_answer,
            "temperature": 0.1
        }))
        
        # Stage 7: Response Validation
        is_grounded = len(selected_chunks) > 0
        steps.append(WorkflowStageResult(7, "Response Validation", "PASSED" if is_grounded else "REJECTED", {
            "faithfulness_score": 0.98 if is_grounded else 0.1,
            "citation_verified": is_grounded,
            "final_status": "APPROVED FOR EGRESS" if is_grounded else "ABSTAINED"
        }))
        
        return steps

# ==============================================================================
# 1. Architecture Blueprint
# ==============================================================================
st.header("1. Architecture Blueprint: The 7-Stage End-to-End Query-to-Response Lifecycle", divider="rainbow")
st.markdown("""
The complete RAG query lifecycle is a deterministic assembly line. Every query passes sequentially through 
seven specialized computational stages in a top-to-bottom pipeline flow:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_workflow_7 = """
    digraph SevenStageWorkflow {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_phase1 {
            label="Tier 1: Query Preparation";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            s1 [label="1. User Query Ingress\\nRaw Natural Language User Input", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            s2 [label="2. Query Transformation\\nHyDE Generation, Rewriting, & Decomposition", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            s1 -> s2 [label="Normalize"];
        }

        subgraph cluster_phase2 {
            label="Tier 2: Candidate Discovery & Precision Filtering";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            fontsize=12;
            s3 [label="3. Parallel Retrieval\\nSparse BM25 Keyword + Dense Vector ANN", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
            s4 [label="4. Context Selection & Reranking\\nCross-Encoder All-to-All Attention Filter", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            s2 -> s3 [label="Execute Searches"];
            s3 -> s4 [label="Top-50 Candidates"];
        }

        subgraph cluster_phase3 {
            label="Tier 3: Synthesis & Verification";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            fontsize=12;
            s5 [label="5. Prompt Assembly Layer\\nContext Delimitation & Abstention Directives", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            s6 [label="6. Conditioned LLM Generation\\nSynthesizes Grounded Claims with Source Citations", fillcolor="#1E1B4B", fontcolor="#E0E7FF", color="#818CF8"];
            s7 [label="7. Egress Validation Gate\\nNLI Faithfulness & Citation Reality Check", fillcolor="#022C22", fontcolor="#A7F3D0", color="#10B981"];
            s4 -> s5 [label="Top-5 Chunks"];
            s5 -> s6 -> s7;
        }

        out [label="Audited Output Delivered to User", fillcolor="#047857", fontcolor="#ECFDF5", color="#10B981"];
        s7 -> out [label="Validation Passed"];
    }
    """
    st.graphviz_chart(dot_workflow_7, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Naive RAG stopped at Step 5, directly streaming whatever the LLM generated. Modern enterprise systems 
(Corrective RAG - CRAG, Self-RAG, and LangGraph Agentic Loops) enforce **Step 7 (Response Validation)** 
as a mandatory firewall: if the generated response introduces ungrounded external claims, the system 
autonomously triggers a fallback query or abstains, guaranteeing zero hallucination egress.
""")

st.divider()

# ==============================================================================
# 2. User Query
# ==============================================================================
st.header("2. User Query Ingress & Intent Profiling", divider="rainbow")
st.markdown("""
The workflow initiates when an authenticated user or upstream API submits a natural language question:
* **Session Metadata Binding**: Attach user role, department permissions, and conversation history.
* **Safety & Prompt Injection Screening**: Scan the query with lightweight classifiers to intercept jailbreak attempts before accessing the vector database.
* **Intent Categorization**: Determine whether the query requires factual document retrieval, conversational chit-chat, or transactional tool execution.
""")

st.divider()

# ==============================================================================
# 3. Query Transformation
# ==============================================================================
st.header("3. Query Transformation: Rewriting, Expansion, & Decomposition", divider="rainbow")
st.markdown("""
Raw user queries rarely match the vocabulary of indexed corporate documents. **Query Transformation** 
bridges this vocabulary gap using three proven techniques:
""")

col_qt1, col_qt2, col_qt3 = st.columns(3)
with col_qt1:
    st.subheader("Query Rewriting")
    st.markdown("""
    * Resolves ambiguous conversational pronouns from multi-turn chat history.
    * Expands internal acronyms (`K8s` -> `Kubernetes`, `mTLS` -> `mutual TLS`).
    """)
with col_qt2:
    st.subheader("HyDE (Hypothetical Docs)")
    st.markdown("""
    * An LLM hallucinates a hypothetical answer passage.
    * The hypothetical passage is embedded, matching real document passages in vector space with high semantic alignment.
    """)
with col_qt3:
    st.subheader("Query Decomposition")
    st.markdown("""
    * Splits complex multi-part questions into atomic sub-queries:
    * `"Compare Q2 vs Q3 margins"` -> Query A (`"Q2 margins"`) + Query B (`"Q3 margins"`).
    """)

with st.expander("🔍 View Query Transformation Strategies Flowchart", expanded=True):
    dot_transform = """
    digraph QueryTransformation {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        raw [label="Raw Ambiguous Input: 'What about its refund?'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        
        t_rewrite [label="Rewriting Strategy\\nResolves pronoun from session history:\\n'What is the refund policy for Enterprise Tier?'", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        t_hyde [label="HyDE Strategy\\nGenerates hypothetical answer passage\\nto match declarative document vectors", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        t_decomp [label="Decomposition Strategy\\nSplits multi-part comparison questions\\ninto independent atomic searches", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];

        raw -> t_rewrite;
        raw -> t_hyde;
        raw -> t_decomp;
    }
    """
    st.graphviz_chart(dot_transform, use_container_width=True)

st.divider()

# ==============================================================================
# 4. Retrieval
# ==============================================================================
st.header("4. Retrieval: Multi-Source Hybrid Search Execution", divider="rainbow")
st.markdown("""
The transformed query is executed concurrently across disparate retrieval engines:
* **Dense Vector ANN Search**: Embeds the query and scans HNSW indexes for conceptual similarity.
* **Sparse Lexical Search (BM25)**: Scans inverted indexes for exact keyword, part number, or code matches.
* **Reciprocal Rank Fusion (RRF)**: Merges the top-50 results from both branches into a unified candidate pool based on relative list rankings.
""")

st.divider()

# ==============================================================================
# 5. Context Selection
# ==============================================================================
st.header("5. Context Selection, Deduplication, & Precision Reranking", divider="rainbow")
st.markdown("""
Passing 50 retrieved chunks to an LLM saturates the prompt window and introduces noise. 
**Context Selection** filters and optimizes the candidate pool:
* **Deduplication**: Eliminates overlapping passages extracted from identical document chapters.
* **Cross-Encoder Reranking**: Evaluates query-document pairs through full transformer cross-attention, reordering candidates by exact relevance.
* **Threshold Truncation**: Drops any chunk whose relevance score falls below a calibrated quality threshold (e.g. score < 0.65).
""")

st.divider()

# ==============================================================================
# 6. Prompt Construction
# ==============================================================================
st.header("6. Prompt Construction & Grounded Context Assembly", divider="rainbow")
st.markdown("""
The top 3-5 verified chunks are compiled into a structured prompt contract:
""")

prompt_template_preview = """
<system>
You are an enterprise AI assistant. Your answers must be derived EXCLUSIVELY from the provided context.
If the context does not contain sufficient factual evidence to answer the question, state:
"The provided enterprise documentation does not contain sufficient evidence to answer this question."
Do NOT extrapolate, speculate, or introduce unverified facts.
Always append verifiable source tags: [Citation: Document_ID].
</system>

<context>
[Citation: DOC-SEC-01] Production Kubernetes deployments must require mTLS authentication...
[Citation: DOC-SEC-02] All service account tokens must be rotated every 14 days...
</context>

<user_query>
What are our production cluster authentication rules?
</user_query>
"""
st.code(prompt_template_preview, language="xml")

st.divider()

# ==============================================================================
# 7. LLM Generation
# ==============================================================================
st.header("7. LLM Generation: Constrained Context Synthesis", divider="rainbow")
st.markdown("""
The generative LLM synthesizes the final response conditioned strictly on the assembled prompt:
* **Decoding Parameters**: Set temperature to near-zero (0.0 or 0.1) to enforce deterministic, repeatable answers.
* **Streaming Egress**: Stream synthesized tokens to the client application while collecting the full response buffer for validation.
""")

st.divider()

# ==============================================================================
# 8. Response Validation
# ==============================================================================
st.header("8. Response Validation, Citation Verification, & Guardrail Gates", divider="rainbow")
st.markdown("""
Before returning the generated text to the end user, a production system runs **Egress Validation Gates**:
""")

df_gates = pd.DataFrame([
    {"Validation Gate": "Faithfulness Check", "Mechanism": "NLI entailment model verifies that every claim is entailed by retrieved chunks.", "Failure Action": "Reject answer & trigger fallback."},
    {"Validation Gate": "Citation Audit", "Mechanism": "Verifies that every [Citation: ID] corresponds to an actual chunk present in the prompt.", "Failure Action": "Strip invalid citations."},
    {"Validation Gate": "PII / Toxicity Screen", "Mechanism": "Scans output for leaked credentials, private identifiers, or unsafe text.", "Failure Action": "Redact sensitive tokens."}
])
st.dataframe(df_gates, use_container_width=True)

st.divider()

# ==============================================================================
# 9. Production Implementation
# ==============================================================================
st.header("9. Production Implementation: The 7-Stage Modular RAG Pipeline Engine", divider="rainbow")
st.markdown("""
Below is a runnable Python production implementation executing the 7 stages sequentially:
""")

prod_workflow_code = """
from typing import Dict, List, Any

class ModularSevenStageRAG:
    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus

    def stage_1_query_ingress(self, raw_input: str) -> str:
        return raw_input.strip()

    def stage_2_transform(self, query: str) -> str:
        # Acronym expansion & normalization
        normalized = query.replace("K8s", "Kubernetes").replace("mTLS", "mutual TLS")
        return normalized

    def stage_3_retrieval(self, query: str) -> List[Dict[str, str]]:
        q_tokens = set(query.lower().split())
        scored = []
        for doc in self.corpus:
            doc_tokens = set(doc["text"].lower().split())
            score = len(q_tokens.intersection(doc_tokens))
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored if score > 0]

    def stage_4_selection(self, candidates: List[Dict[str, str]], top_k: int = 2) -> List[Dict[str, str]]:
        return candidates[:top_k]

    def stage_5_prompt_assembly(self, query: str, context: List[Dict[str, str]]) -> str:
        context_block = "\\n".join([f"[{c['id']}] {c['text']}" for c in context])
        return f"CONTEXT:\\n{context_block}\\n\\nQUERY: {query}\\nANSWER:"

    def stage_6_generate(self, prompt: str, context: List[Dict[str, str]]) -> str:
        if not context:
            return "No verified documentation discovered."
        return f"According to {context[0]['id']}, {context[0]['text'][:50]}..."

    def stage_7_validate(self, answer: str, context: List[Dict[str, str]]) -> Dict[str, Any]:
        has_evidence = len(context) > 0
        return {
            "status": "APPROVED" if has_evidence else "REJECTED",
            "groundedness": 0.99 if has_evidence else 0.0,
            "final_answer": answer
        }

    def execute(self, user_query: str) -> Dict[str, Any]:
        q1 = self.stage_1_query_ingress(user_query)
        q2 = self.stage_2_transform(q1)
        candidates = self.stage_3_retrieval(q2)
        selected = self.stage_4_selection(candidates)
        prompt = self.stage_5_prompt_assembly(q1, selected)
        raw_answer = self.stage_6_generate(prompt, selected)
        validated = self.stage_7_validate(raw_answer, selected)
        return validated
"""
st.code(prod_workflow_code, language="python")

st.divider()

# ==============================================================================
# 10. Interactive Workbench
# ==============================================================================
st.header("10. Interactive Workbench: 7-Stage Workflow Step-Through Simulator", divider="rainbow")
st.markdown("""
Step through all 7 pipeline stages for a real query and inspect the intermediate data structures at each step:
""")

pipeline_7 = SevenStageRAGPipeline()

test_query = st.selectbox(
    "Select Query to Trace Across the 7-Stage Pipeline:",
    [
        "What are our K8s production cluster mTLS requirements?",
        "How often must service account tokens be rotated?",
        "What was our Q3 profit margin in 2026?",
        "What is our policy on remote work in Antarctica?"
    ]
)

if st.button("Execute & Trace 7-Stage Workflow", type="primary"):
    stages_executed = pipeline_7.execute_workflow(test_query)
    
    for stage in stages_executed:
        with st.expander(f"Stage {stage.stage_num}: {stage.stage_name} — Status: {stage.status}", expanded=(stage.stage_num in [2, 4, 7])):
            st.json(stage.details)

st.divider()

# ==============================================================================
# 11. Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: How does Query Transformation (HyDE) improve dense vector retrieval?"):
    st.markdown("""
    **Answer**: In dense vector spaces, questions and declarative answer passages occupy different geometric regions. 
    By asking an LLM to generate a hypothetical answer first, HyDE embeds a *declarative passage* that clusters much closer 
    to the target document chunks in vector space than the original question could.
    """)

with st.expander("Q2: Why is Response Validation (Stage 7) critical in enterprise RAG?"):
    st.markdown("""
    **Answer**: Without validation, prompt injection or subtle model hallucinations can slip past generation directly 
    to the user. Egress validation checks verify factual entailment, ensure every cited chunk actually exists in the prompt, 
    and block responses that fail safety thresholds.
    """)

with st.expander("Q3: What is the difference between Pre-Retrieval and Post-Retrieval optimization?"):
    st.markdown("""
    **Answer**: **Pre-Retrieval Optimization** operates on the query before database lookup (query expansion, rewriting, HyDE). 
    **Post-Retrieval Optimization** operates on retrieved chunks before generation (cross-encoder reranking, deduplication, context compression).
    """)

with st.expander("Q4: When should Sub-Query Decomposition be triggered?"):
    st.markdown("""
    **Answer**: When a query contains multi-hop reasoning or comparative intent (e.g., *"How do our 2025 revenue numbers compare to our 2026 targets?"*). 
    A single vector cannot match both distinct documents simultaneously. Decomposing into two sub-queries ensures independent, high-recall retrieval.
    """)

with st.expander("Q5: What metrics define a successful RAG evaluation trilemma?"):
    st.markdown("""
    **Answer**: The **RAG Triad**:
    1. **Context Relevance**: Did the retriever pull relevant chunks?
    2. **Groundedness / Faithfulness**: Is the generated answer derived strictly from the context?
    3. **Answer Relevance**: Does the generated answer address the original user question?
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Skipping Response Validation / Blind Egress)**: Assuming low LLM temperature guarantees zero hallucination. Without automated entailment checks, subtle hallucinations reach end users undetected.")
    st.warning("**Trap 2 (Unchecked Query Transformation Latency)**: Chaining 3 sequential LLM calls for query rewriting, HyDE, and decomposition adds 2+ seconds of latency before retrieval even begins.")
with col_trap2:
    st.error("**Trap 3 (Over-Stuffing Selected Context / Context Bleed)**: Injecting too many weakly scored chunks into the prompt, triggering the Lost-in-the-Middle failure mode and distracting the generator.")
    st.warning("**Trap 4 (Brittle Regex Citation Parsing)**: Relying on fragile regex patterns to extract source citations without validating that the referenced document ID actually exists in the vector store.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is HyDE (Hypothetical Document Embeddings)?")
        st.caption("Flip for answer:")
        st.info("**Back**: Generates a hypothetical answer first, then embeds it to match real document vectors in semantic space.")
    with st.container(border=True):
        st.markdown("**Front**: What are the 3 pillars of the RAG Triad?")
        st.caption("Flip for answer:")
        st.info("**Back**: 1. Context Relevance (Retriever precision), 2. Faithfulness / Groundedness (LLM truthfulness), 3. Answer Relevance (Query alignment).")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What are the 7 stages of the RAG execution lifecycle?")
        st.caption("Flip for answer:")
        st.info("**Back**: 1. Query -> 2. Transform -> 3. Retrieve -> 4. Select -> 5. Prompt -> 6. Generate -> 7. Validate.")
    with st.container(border=True):
        st.markdown("**Front**: What is Sub-Query Decomposition?")
        st.caption("Flip for answer:")
        st.info("**Back**: Breaks complex or comparative queries into atomic sub-questions for independent, high-recall vector lookups.")

st.divider()

# ==============================================================================
# 12. Self-Assessment Checklist
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I can articulate all 7 stages of the end-to-end RAG query execution lifecycle.", value=True)
c2 = st.checkbox("I know when to apply Query Rewriting, HyDE, and Sub-Query Decomposition.", value=True)
c3 = st.checkbox("I understand why Context Selection must prune and rerank the raw candidate pool.", value=True)
c4 = st.checkbox("I know how to design strict prompt contracts that instruct the model to abstain when context is missing.", value=True)
c5 = st.checkbox("I understand the function of Stage 7 (Response Validation) and the RAG Triad evaluation metrics.", value=True)
c6 = st.checkbox("I can trace the data flow contracts passed between each pipeline stage.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Module 2.2 Complete! You are ready to advance to Lesson 2.3: RAG Components.")
