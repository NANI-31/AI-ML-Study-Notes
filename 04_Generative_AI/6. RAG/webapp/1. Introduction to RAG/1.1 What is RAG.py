import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="1.1 What is RAG? - RAG Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title Block & 5-Pillar Overview
st.title("1.1 What is RAG? Foundations, Non-Parametric Memory, & Model Adaptation")
st.caption("Chapter 1: Introduction to RAG | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
A foundation Large Language Model (LLM) is an extraordinary reasoning engine, but as an enterprise knowledge store 
it has fatal vulnerabilities: its knowledge is frozen at training cutoff, it is completely blind to private corporate data, 
it cannot provide verifiable citations, and when faced with missing facts, it hallucinates plausible falsehoods.

**Retrieval-Augmented Generation (RAG)** solves these challenges by decoupling the reasoning engine from static data storage. 
Instead of forcing the neural network to memorize enterprise facts in its billions of static parameters, RAG interfaces 
the LLM with an external, dynamic non-parametric memory store (vector databases, keyword search indexes, and enterprise APIs).

In this lesson, you will master:
1. The architectural duality of Parametric Memory versus Non-Parametric External Stores.
2. The core failure modes of LLM-only applications (temporal cutoff, private data blindness, hallucination).
3. The strategic continuum contrasting Prompt Engineering, RAG, and Fine-Tuning.
4. Production-grade grounded retrieval workflows with auditable source attribution.
5. High-frequency enterprise interview questions, production antipatterns, and validation checkpoints.
""")

st.divider()

# Module-Scope Helper Classes for Interactive Workbench & Production Pipeline
class DocumentChunk:
    def __init__(self, doc_id: str, title: str, content: str, score: float, category: str):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.score = score
        self.category = category

class LightweightRAGPipeline:
    def __init__(self):
        self.knowledge_base = [
            DocumentChunk(
                "DOC-001",
                "Q3 2026 Enterprise Financials",
                "Operating profit margin expanded to 28.4% in Q3 2026, driven by enterprise adoption of generative AI services across APAC.",
                0.94,
                "Finance"
            ),
            DocumentChunk(
                "DOC-002",
                "Cloud Security Policy v4.2",
                "Zero-trust credential rotation must occur every 14 days for all production Kubernetes cluster service accounts handling PII.",
                0.91,
                "Security"
            ),
            DocumentChunk(
                "DOC-003",
                "Project Apollo Core Architecture",
                "Project Apollo implements a decoupled hybrid retriever combining sparse keyword indexing with dense vector semantic search.",
                0.88,
                "Engineering"
            ),
            DocumentChunk(
                "DOC-004",
                "Employee Benefits Handbook 2026",
                "Employees are eligible for a 5,000 USD annual continuous learning stipend applicable toward accredited AI and systems certifications.",
                0.82,
                "HR"
            )
        ]

    def retrieve(self, query: str, top_k: int = 2) -> List[DocumentChunk]:
        query_words = set(query.lower().split())
        scored_chunks = []
        for chunk in self.knowledge_base:
            chunk_words = set((chunk.title + " " + chunk.content).lower().split())
            overlap = len(query_words.intersection(chunk_words))
            boosted_score = chunk.score * (1.0 + 0.15 * overlap)
            scored_chunks.append((boosted_score, chunk))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_chunks[:top_k]]

    def synthesize(self, query: str, retrieved_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        if not retrieved_chunks:
            return {
                "answer": "No relevant evidence discovered in external non-parametric storage. Declining to answer to prevent hallucination.",
                "citations": []
            }
        citations = [f"[{c.doc_id}] {c.title}" for c in retrieved_chunks]
        context_block = "\n".join([f"Evidence from {c.title}: \"{c.content}\"" for c in retrieved_chunks])
        synthesized_text = (
            f"Based on verified external retrieval:\n\n"
            f"{context_block}\n\n"
            f"Synthesis: Addressing query '{query}' strictly grounded in enterprise non-parametric evidence."
        )
        return {
            "answer": synthesized_text,
            "citations": citations
        }

# ==============================================================================
# 1. Architecture Blueprint
# ==============================================================================
st.header("1. Architecture Blueprint: The Parametric vs. Non-Parametric RAG Paradigm", divider="rainbow")
st.markdown("""
Modern language models store knowledge in two distinct operational forms:
* **Parametric Memory**: Knowledge encoded implicitly across billions of static weight matrices during pretraining. It is fast but immutable without costly retraining, opaque, and incapable of dynamic source verification.
* **Non-Parametric Memory**: Knowledge stored externally in structured or semi-structured data stores (vector indices, inverted lexical indices, document repositories). It is dynamically modifiable in real-time and fully traceable.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph RAGParadigm {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Query Ingress Tier";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            q [label="User Query\\n'What is our Q3 2026 profit margin?'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_nonparam {
            label="2. External Non-Parametric Memory Tier";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            fontsize=12;
            vdb [label="Enterprise Vector & Lexical Store\\n(Real-Time Ingestion, Dynamic, Auditable)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            docs [label="Verified Context Chunks\\n[DOC-001: 28.4% profit margin in Q3 2026]", fillcolor="#022C22", fontcolor="#A7F3D0", color="#10B981"];
            vdb -> docs [label="Semantic Match & Filter"];
        }

        subgraph cluster_param {
            label="3. Frozen Foundation Model (Reasoning Engine)";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            fontsize=12;
            llm [label="LLM Synthesis Engine\\n(Frozen Parameters Theta: Provides Grammar & Reasoning)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        }

        subgraph cluster_synth {
            label="4. Grounded Output Tier";
            bgcolor="#1E293B";
            fontcolor="#EC4899";
            color="#334155";
            fontsize=12;
            resp [label="Grounded Response\\n'28.4% operating margin [Citation: DOC-001]'", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
        }

        q -> vdb [label="Query Ingress Search"];
        q -> llm [label="User Intent"];
        docs -> llm [label="Injected Evidence Chunks"];
        llm -> resp [label="Grounded Synthesis"];
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In early experiments, developers treated foundation LLMs as monolithic encyclopedias. Modern production AI systems 
(Perplexity, OpenAI Search, Google Gemini Grounding, and enterprise internal assistants) treat the foundation LLM 
strictly as an **orchestration and linguistic synthesis engine**. Factual claims are never trusted from raw weights; 
they are pulled dynamically from external indices at runtime, ensuring sub-second knowledge refreshes and auditable provenance.
""")

st.divider()

# ==============================================================================
# 2. Definition of Retrieval-Augmented Generation
# ==============================================================================
st.header("2. Definition of Retrieval-Augmented Generation (RAG)", divider="rainbow")
st.markdown("""
**Retrieval-Augmented Generation (RAG)** is an architectural pattern that combines an information retrieval subsystem 
with an autoregressive language model. 

In a production RAG system, processing divides into four distinct subsystems:
1. **The Ingestion Pipeline**: Connects to enterprise storage, parses unstructured formats, chunks long texts, generates vector representations, and populates index stores.
2. **The Retrieval Subsystem**: Evaluates the user query, searches lexical and vector indices, and extracts the top relevant candidate chunks.
3. **The In-Context Augmentation Layer**: Organizes retrieved passages into a structured prompt contract that bounds the model's generation space.
4. **The Grounded Synthesis Engine**: Directs the LLM to synthesize the final answer strictly from the supplied evidence while generating exact citations.
""")

df_rag_def = pd.DataFrame([
    {"Subsystem": "Document Ingestion", "Role": "Extracts, chunks, and indexes external documents into non-parametric storage."},
    {"Subsystem": "Retriever Subsystem", "Role": "Executes hybrid keyword and semantic vector search to find candidate passages."},
    {"Subsystem": "Context Augmentation", "Role": "Packages retrieved chunks into strict prompt templates with provenance metadata."},
    {"Subsystem": "Generation & Citations", "Role": "Synthesizes fluent output while attaching verifiable source tags."}
])
st.dataframe(df_rag_def, use_container_width=True)

st.divider()

# ==============================================================================
# 3. Why RAG is Important
# ==============================================================================
st.header("3. Why RAG is Important in Modern AI Systems", divider="rainbow")
st.markdown("""
Without retrieval augmentation, deploying foundation models in corporate and regulated environments introduces severe risks:
* **The Static Knowledge Barrier**: Pretraining ends on a specific date. A model trained in 2024 has zero knowledge of 2026 regulations or new product releases.
* **The Proprietary Data Gap**: Enterprise repositories (intranet wikis, customer contracts, private codebases) cannot be made public during open model pretraining.
* **The Hallucination Liability**: Autoregressive decoders are probabilistic word predictors, not databases. When uncertain, they generate convincing fictions.

RAG eliminates these blockers by converting enterprise knowledge access from an expensive **offline training problem** 
into a real-time **runtime query service**.
""")

col_imp1, col_imp2, col_imp3 = st.columns(3)
with col_imp1:
    st.info("**Zero Retraining Lag**\n\nNew documents added to the knowledge store are immediately retrievable within seconds without retraining models.")
with col_imp2:
    st.info("**Granular Access Control**\n\nSecurity permissions (RBAC) and tenant filters are enforced at the retriever stage before data reaches the model.")
with col_imp3:
    st.info("**Verifiable Source Trust**\n\nEvery factual assertion links back to an auditable document chunk ID, author, and page reference.")

st.divider()

# ==============================================================================
# 4. Problems with LLM-Only Applications
# ==============================================================================
st.header("4. Problems with LLM-Only Applications", divider="rainbow")
st.markdown("""
When organizations attempt to build production applications on standalone foundation models, they systematically 
encounter four fatal failure modes:
""")

col_prob1, col_prob2 = st.columns(2)
with col_prob1:
    st.error("""
    **1. Uncontrolled Hallucination**
    * Standalone LLMs generate plausible-sounding falsehoods when asked about specific enterprise facts.
    * The self-attention mechanism optimizes for fluency and coherence, not factual accuracy.
    """)
    st.error("""
    **2. Temporal Obsolescence**
    * Once pretraining ends, the model's knowledge snapshot is frozen.
    * It cannot answer queries about updated policies, recent security bulletins, or real-time metrics.
    """)
with col_prob2:
    st.error("""
    **3. Total Lack of Verifiable Citations**
    * Parametric memory cannot point to a specific page number, URL, or internal document ID.
    * Legal and clinical compliance frameworks strictly prohibit unverified black-box claims.
    """)
    st.error("""
    **4. Prohibitive Retraining Costs**
    * Continual retraining or fine-tuning to update facts requires massive GPU clusters and risks catastrophic forgetting of general reasoning skills.
    """)

st.divider()

# ==============================================================================
# 5. How RAG Solves Knowledge Limitations
# ==============================================================================
st.header("5. How RAG Solves Knowledge Limitations", divider="rainbow")
st.markdown("""
RAG solves knowledge limitations by transforming the model from a **closed-book memorization system** 
into an **open-book synthesis engine**. 

Instead of relying on weights learned months ago, the model receives fresh, verified source text directly in its 
immediate context window:
""")

with st.expander("🔍 View Contextual Grounding Flowchart", expanded=True):
    dot_sol = """
    digraph GroundingFlow {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_standalone {
            label="Standalone LLM (Parametric Failure)";
            bgcolor="#1E293B";
            fontcolor="#EF4444";
            color="#7F1D1D";
            fontsize=12;
            q_fail [label="User Query: 'What is our 2026 refund policy?'", fillcolor="#450A0A", fontcolor="#FCA5A5", color="#EF4444"];
            ans_fail [label="Hallucinated / Outdated Output\\n'Refunds are available for 14 days' (Incorrect Guess)", fillcolor="#450A0A", fontcolor="#FCA5A5", color="#EF4444"];
            q_fail -> ans_fail [label="Unbounded Guesswork"];
        }

        subgraph cluster_rag_sol {
            label="RAG Architecture (Grounded Factual Success)";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#064E3B";
            fontsize=12;
            q_ok [label="User Query: 'What is our 2026 refund policy?'", fillcolor="#022C22", fontcolor="#A7F3D0", color="#10B981"];
            fetch [label="Retriever fetches Policy Section 4.2\\n'Refunds granted within 30 days of purchase'", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            prompt_b [label="Prompt Contract: 'Answer strictly using evidence.\\nCite Document ID.'", fillcolor="#047857", fontcolor="#ECFDF5", color="#10B981"];
            ans_ok [label="Grounded Synthesis\\n'30 days from purchase [Citation: DOC-POL-42]'", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            q_ok -> fetch -> prompt_b -> ans_ok;
        }
    }
    """
    st.graphviz_chart(dot_sol, use_container_width=True)

st.divider()

# ==============================================================================
# 6. RAG vs Traditional LLM Applications
# ==============================================================================
st.header("6. RAG vs Traditional LLM Applications", divider="rainbow")
st.markdown("""
The table below contrasts standard unaugmented LLM applications against production RAG systems:
""")

df_compare_trad = pd.DataFrame([
    {
        "Evaluation Dimension": "Knowledge Storage Location",
        "Traditional LLM Application": "Baked into static weight matrices during pretraining.",
        "Retrieval-Augmented Generation (RAG)": "Decoupled into external vector, keyword, and graph databases."
    },
    {
        "Evaluation Dimension": "Knowledge Freshness",
        "Traditional LLM Application": "Fixed at pretraining cutoff timestamp.",
        "Retrieval-Augmented Generation (RAG)": "Real-time; updated immediately upon document ingestion."
    },
    {
        "Evaluation Dimension": "Source Attribution",
        "Traditional LLM Application": "None; cannot output verifiable document page citations.",
        "Retrieval-Augmented Generation (RAG)": "Exact and auditable; every claim references a chunk ID."
    },
    {
        "Evaluation Dimension": "Hallucination Control",
        "Traditional LLM Application": "High risk; relies solely on prompt begging.",
        "Retrieval-Augmented Generation (RAG)": "Lowest risk; bounded strictly by retrieved evidence."
    },
    {
        "Evaluation Dimension": "Enterprise Security (RBAC)",
        "Traditional LLM Application": "Impossible; all users query identical global weights.",
        "Retrieval-Augmented Generation (RAG)": "Native; security filters applied at the database retrieval stage."
    }
])
st.dataframe(df_compare_trad, use_container_width=True)

st.divider()

# ==============================================================================
# 7. RAG vs Fine-Tuning
# ==============================================================================
st.header("7. RAG vs Fine-Tuning: The Enterprise Adaptation Decision Matrix", divider="rainbow")
st.markdown("""
A common engineering misconception is viewing RAG and Fine-Tuning as competing techniques. 
In enterprise production, they solve completely different problems:
* **RAG is for knowledge injection** (facts, changing data, proprietary documents).
* **Fine-Tuning is for behavioral adaptation** (response tone, JSON formatting, specialized syntax, domain dialect).

The **Golden Rule of Enterprise LLMs**: *Never use fine-tuning to teach an LLM new facts! Use RAG for facts, and fine-tuning for form.*
""")

with st.expander("🔍 View Adaptation Decision Matrix Diagram", expanded=True):
    dot_decision = """
    digraph AdaptationDecision {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        need [label="Enterprise Problem: What needs updating?", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        facts [label="Need: New factual knowledge, dynamic data, or verifiable source citations", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        style [label="Need: Custom JSON syntax, specialized dialect, or specific tone", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        both [label="Need: Proprietary syntax AND proprietary dynamic facts", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];

        rag_choice [label="SOLUTION: Retrieval-Augmented Generation (RAG)\\nZero retraining; dynamic document ingestion", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        ft_choice [label="SOLUTION: Fine-Tuning (PEFT / LoRA)\\nTrain adapter weights on labeled formatting pairs", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        hybrid_choice [label="SOLUTION: Hybrid Stack\\nFine-tune small model for format + RAG for facts", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];

        need -> facts -> rag_choice;
        need -> style -> ft_choice;
        need -> both -> hybrid_choice;
    }
    """
    st.graphviz_chart(dot_decision, use_container_width=True)

df_rag_ft = pd.DataFrame([
    {"Criterion": "Primary Purpose", "RAG": "Injecting dynamic external facts and documents.", "Fine-Tuning (PEFT / LoRA)": "Adapting style, tone, syntax, and task behavior."},
    {"Criterion": "Knowledge Update Cycle", "RAG": "Immediate upon document upload.", "Fine-Tuning (PEFT / LoRA)": "Slow (requires dataset prep and retraining runs)."},
    {"Criterion": "Auditability & Citations", "RAG": "100% auditable with chunk references.", "Fine-Tuning (PEFT / LoRA)": "Zero (weights cannot cite source sentences)."},
    {"Criterion": "Hardware Requirements", "RAG": "Standard CPU/GPU for vector search.", "Fine-Tuning (PEFT / LoRA)": "Dedicated high-memory GPU training clusters."}
])
st.dataframe(df_rag_ft, use_container_width=True)

st.divider()

# ==============================================================================
# 8. RAG vs Prompt Engineering
# ==============================================================================
st.header("8. RAG vs Prompt Engineering: In-Context Learning vs. Grounded Retrieval", divider="rainbow")
st.markdown("""
**Prompt Engineering** relies solely on providing instructions, system personas, and few-shot examples within the 
immediate prompt window. While effective for simple formatting, it breaks down when applied to large document archives:
1. **Context Window Saturation**: Stuffing 50 documents into a 1M token prompt costs significantly more per API call and introduces latency.
2. **The Lost-in-the-Middle Problem**: LLM attention degrades over information located in the middle of long prompts.
3. **Dynamic Filtering**: Prompt engineering requires humans to manually select documents. RAG automates this by programmatically retrieving the top 3-5 relevant chunks.
""")

df_rag_prompt = pd.DataFrame([
    {"Strategy": "Prompt Engineering Only", "Best Used For": "Specifying output tone, formatting rules, few-shot examples.", "Scaling Limit": "Fails when documentation exceeds a handful of pages."},
    {"Strategy": "RAG Augmented In-Context", "Best Used For": "Searching across millions of enterprise files and injecting exact facts.", "Scaling Limit": "Scales to petabyte corpora with sub-second retrieval."}
])
st.dataframe(df_rag_prompt, use_container_width=True)

st.divider()

# ==============================================================================
# 9. Advantages of RAG
# ==============================================================================
st.header("9. Core Enterprise Advantages of RAG", divider="rainbow")
st.markdown("""
Enterprise architects select RAG because it provides five decisive operational advantages:
""")

col_adv1, col_adv2 = st.columns(2)
with col_adv1:
    st.success("""
    **1. Up-to-Date Information**
    * New documents added to disk or cloud buckets are chunked and searchable within seconds.
    * No scheduled retraining downtime or model rebuilding required.
    """)
    st.success("""
    **2. Private and Domain-Specific Knowledge**
    * Keep proprietary HR policies, financial records, and medical data secure within your VPC.
    * Data is not exposed to public foundation model pretraining pipelines.
    """)
    st.success("""
    **3. Reduced Hallucination**
    * Prompt contracts constrain the model: *"Answer strictly from provided evidence. If absent, state that you do not know."*
    """)
with col_adv2:
    st.success("""
    **4. Source Attribution & Auditability**
    * Every response includes exact citation references (`[DOC-ID, Section, Page]`), allowing human auditors to verify claims against original sources.
    """)
    st.success("""
    **5. No Model Retraining Required**
    * Easily switch foundation models (e.g. migrating from Llama 3 to Claude 3.5 or Gemini 1.5) without re-indexing your entire vector database.
    """)

st.divider()

# ==============================================================================
# 10. Production Implementation
# ==============================================================================
st.header("10. Production Implementation: End-to-End Grounded Retrieval & Citation Pipeline", divider="rainbow")
st.markdown("""
Below is a runnable Python production implementation demonstrating the complete 4-stage RAG execution pipeline: 
Ingestion, Retrieval, Grounded Prompt Assembly, and Synthesis with Citations.
""")

prod_code_snippet = """
from typing import List, Dict, Any

class DocumentChunk:
    def __init__(self, doc_id: str, title: str, content: str):
        self.doc_id = doc_id
        self.title = title
        self.content = content

class ProductionRAGSystem:
    def __init__(self, corpus: List[DocumentChunk]):
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 2) -> List[DocumentChunk]:
        \"\"\"Executes lexical overlap and relevance scoring.\"\"\"
        query_tokens = set(query.lower().split())
        scored = []
        for doc in self.corpus:
            doc_tokens = set(doc.content.lower().split())
            overlap = len(query_tokens.intersection(doc_tokens))
            score = overlap / (len(query_tokens) + 1e-6)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]

    def build_grounded_prompt(self, query: str, retrieved_docs: List[DocumentChunk]) -> str:
        \"\"\"Assembles a strict prompt contract with provenance bounding.\"\"\"
        context_str = "\\n\\n".join([
            f"[Source ID: {doc.doc_id} | Title: {doc.title}]\\n{doc.content}"
            for doc in retrieved_docs
        ])
        prompt = (
            f"SYSTEM: You are a factual enterprise AI assistant. "
            f"Answer the user query STRICTLY based on the provided evidence passages below. "
            f"If the answer cannot be deduced from the evidence, respond: 'Insufficient factual evidence.' "
            f"Always cite the Source ID for each factual claim.\\n\\n"
            f"--- VERIFIED CONTEXT ---\\n{context_str}\\n\\n"
            f"USER QUERY: {query}\\n"
            f"GROUNDED ANSWER:"
        )
        return prompt

    def generate_response(self, query: str) -> Dict[str, Any]:
        retrieved = self.retrieve(query)
        if not retrieved:
            return {
                "answer": "Insufficient factual evidence in non-parametric database.",
                "citations": []
            }
        prompt = self.build_grounded_prompt(query, retrieved)
        # In production: response = llm_client.generate(prompt)
        answer = f"According to verified enterprise records ({retrieved[0].doc_id}), " + retrieved[0].content[:80] + "..."
        return {
            "answer": answer,
            "citations": [doc.doc_id for doc in retrieved],
            "assembled_prompt": prompt
        }
"""
st.code(prod_code_snippet, language="python")

st.divider()

# ==============================================================================
# 11. Interactive Workbench
# ==============================================================================
st.header("11. Interactive Workbench: Grounded RAG vs. Parametric Model Studio", divider="rainbow")
st.markdown("""
Test real enterprise queries comparing a **Parametric-Only Model** (which hallucinates or relies on stale pretraining) 
against a **Grounded RAG Pipeline** (which retrieves non-parametric evidence passages in real time):
""")

pipeline_instance = LightweightRAGPipeline()

col_query_sel, col_topk = st.columns([3, 1])
with col_query_sel:
    selected_query = st.selectbox(
        "Select an Enterprise Query to Test:",
        [
            "What is our Q3 2026 profit margin?",
            "What is the service account credential rotation policy?",
            "How does Project Apollo handle hybrid retrieval?",
            "What is the annual education certification stipend?",
            "What were our corporate earnings in 2012?"
        ]
    )
with col_topk:
    top_k_val = st.slider("Retrieval Top-K Chunks:", min_value=1, max_value=4, value=2)

if st.button("Execute Pipeline Comparison", type="primary"):
    col_out_raw, col_out_rag = st.columns(2)
    with col_out_raw:
        st.subheader("❌ Parametric Model (Unaugmented)")
        if "2026" in selected_query or "Apollo" in selected_query or "rotation" in selected_query:
            st.error("""
            **Simulated LLM-Only Output:**
            'I do not possess access to internal enterprise documentation or real-time 2026 financials. 
            Historically, operating margins might range between 15% and 20%, but I cannot verify exact figures.'
            
            *Failure Mode*: Cutoff date blindness and missing private enterprise facts.
            """)
        else:
            st.warning("""
            **Simulated LLM-Only Output:**
            'Companies typically offer between $1,000 and $2,500 for tuition reimbursement.'
            
            *Failure Mode*: Generic parametric hallucination lacking exact policy grounding.
            """)

    with col_out_rag:
        st.subheader("✅ Grounded RAG Pipeline (Non-Parametric)")
        retrieved_results = pipeline_instance.retrieve(selected_query, top_k=top_k_val)
        synthesis_result = pipeline_instance.synthesize(selected_query, retrieved_results)
        st.success(synthesis_result["answer"])
        st.markdown("**Verifiable Citations Extracted:**")
        for cit in synthesis_result["citations"]:
            st.markdown(f"- 📄 `{cit}`")

st.divider()

# ==============================================================================
# 12. Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: Under what circumstances should you choose RAG over Fine-Tuning?"):
    st.markdown("""
    **Answer**: Choose RAG when the problem requires **dynamic factual updates, access to private/proprietary data, and source citation verifiability**. 
    Choose Fine-Tuning when you need to adapt the model's **linguistic style, structured JSON formatting, domain-specific terminology, or complex reasoning syntax**. 
    Fine-tuning is incapable of reliably memorizing rapid, changing factual assertions without severe hallucination risk.
    """)

with st.expander("Q2: Explain the difference between Parametric and Non-Parametric memory in generative AI."):
    st.markdown("""
    **Answer**: Parametric memory consists of the static continuous numerical weights learned during pretraining. It is fast but fixed at cutoff and susceptible to hallucination. 
    Non-parametric memory consists of external discrete data structures (vector indexes, document databases, knowledge graphs) that can be edited, augmented, deleted, or audited in real time without retraining the model.
    """)

with st.expander("Q3: How does the 'Lost-in-the-Middle' phenomenon degrade naive RAG performance?"):
    st.markdown("""
    **Answer**: Research by Liu et al. demonstrated that decoder-only Transformer self-attention is strongest at the very beginning and very end of the context window. 
    When a naive RAG retriever injects 20+ chunks into a large prompt, relevant evidence placed in the middle often fails to guide attention, leading to factual omissions. 
    Mitigations include precision reranking (top-3 chunks only) or sorting chunks to place highest-relevance evidence at the boundaries.
    """)

with st.expander("Q4: What role does Cross-Encoder Reranking play in a production RAG pipeline?"):
    st.markdown("""
    **Answer**: Vector search (bi-encoders) compresses chunks into independent static vectors, sacrificing deep inter-token interaction between query and document. 
    A cross-encoder reranker passes the query and candidate chunk jointly through full all-to-all self-attention layers, computing a high-precision relevance score. 
    This eliminates false-positive semantic matches before prompt injection.
    """)

with st.expander("Q5: Why is BM25 keyword retrieval frequently paired with dense vector retrieval in hybrid RAG?"):
    st.markdown("""
    **Answer**: Dense embeddings excel at semantic similarity (synonyms, conceptual themes) but fail on rare out-of-vocabulary tokens, specific part numbers, customer IDs, and acronyms. 
    BM25 provides exact keyword and token matching. Combining both via Reciprocal Rank Fusion (RRF) delivers the best of both worlds.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The Unfiltered Top-K Poisoning Trap)**: Retrieving irrelevant or noisy document chunks and injecting them uncritically into the context window distracts the LLM's attention mechanism and degrades response accuracy even when the correct chunk is present.")
    st.warning("**Trap 2 (Outdated Index Cache Drift)**: Failing to implement real-time change data capture (CDC) between primary operational databases and vector stores leads to the LLM serving stale or revoked enterprise policies.")
with col_trap2:
    st.error("**Trap 3 (Prompt Leaking Through Unsanitized Retrieved Chunks)**: Ingesting untrusted external HTML or third-party documents containing prompt injection strings directly into the context window without adversarial sanitization.")
    st.warning("**Trap 4 (Blindly Scaling Context Length Instead of Improving Retrieval)**: Relying on massive 1M+ token context windows to ingest raw unchunked files increases inference latency and API cost by orders of magnitude compared to targeted chunk retrieval.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the core architectural formula for Reciprocal Rank Fusion (RRF)?")
        st.caption("Flip for answer:")
        st.info("**Back**: Combines rank orders from disparate retrieval algorithms (BM25 and Dense Vector search) without requiring score calibration, prioritizing candidates that appear near the top of both lists.")
    with st.container(border=True):
        st.markdown("**Front**: When should you use RAG versus Fine-Tuning in enterprise AI?")
        st.caption("Flip for answer:")
        st.info("**Back**: Use RAG for facts, dynamic knowledge, and verifiable citations. Use Fine-Tuning for tone, style, structured JSON formatting, and specialized behavioral patterns.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the difference between Parametric and Non-Parametric memory?")
        st.caption("Flip for answer:")
        st.info("**Back**: Parametric memory lives in frozen model weights. Non-Parametric memory lives in external, dynamically updatable data stores (Vector DB, BM25, SQL).")
    with st.container(border=True):
        st.markdown("**Front**: What is the difference between a Bi-Encoder and a Cross-Encoder?")
        st.caption("Flip for answer:")
        st.info("**Back**: Bi-Encoders embed query and document separately for fast vector search. Cross-Encoders pass query and document jointly through full attention layers for high-precision reranking.")

st.divider()

# ==============================================================================
# 13. Self-Assessment Checklist
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I can articulate the architectural difference between parametric and non-parametric memory stores.", value=True)
c2 = st.checkbox("I understand why fine-tuning is inappropriate for injecting rapidly evolving factual data.", value=True)
c3 = st.checkbox("I can explain the three primary failure modes of LLM-only systems (hallucination, cutoff date, opacity).", value=True)
c4 = st.checkbox("I can diagram the complete 4-stage RAG lifecycle: Ingestion, Retrieval, Augmentation, and Generation.", value=True)
c5 = st.checkbox("I know how to construct strict prompt contracts that instruct the model to abstain when context is missing.", value=True)
c6 = st.checkbox("I understand the trade-offs of sparse BM25 keyword matching versus dense semantic vector retrieval.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Module 1.1 Complete! You are ready to advance to Lesson 1.2: RAG Architecture.")
