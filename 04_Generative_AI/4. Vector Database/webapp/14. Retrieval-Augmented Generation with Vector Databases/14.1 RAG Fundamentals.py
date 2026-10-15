import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="14.1 RAG Fundamentals - Vector DB Curriculum",
    page_icon="🧠",
    layout="wide"
)

# Title & Overview
st.title("14.1 RAG Fundamentals: Architecture, Non-Parametric Memory & Vector DB Integration")
st.caption("Chapter 14: Retrieval-Augmented Generation with Vector Databases | Domain: 4. Vector Database")
st.markdown("""
Large Language Models (LLMs) possess vast world knowledge, but their internal memory is **parametric**—frozen in time at their training cutoff date. 
When asked about private enterprise data, recent documentation, or domain-specific nuances, standalone LLMs frequently **hallucinate plausibly sounding falsehoods**.

**Retrieval-Augmented Generation (RAG)** bridges this fundamental divide by pairing the generative capabilities of an LLM with the 
**non-parametric external memory of a vector database**.

Instead of expecting the model weights to memorize all facts:
1. The **Vector Database (Retriever)** searches your dynamic, proprietary knowledge base to find the top-K most semantically relevant facts.
2. The **LLM (Generator)** reads those facts inside its runtime context prompt and synthesizes a grounded, verifiable, and cited answer.

In this lesson, we explore the foundations of RAG, analyze why vector databases are uniquely required, dissect the 
two-phase Retrieval-Generation handshake, and experiment with an interactive RAG vs. Direct LLM Simulator.
""")

st.divider()

# Section 1: Parametric vs Non-Parametric Memory
st.header("1. Parametric vs Non-Parametric Memory", divider="rainbow")

col_mem1, col_mem2 = col_mem1, col_mem2 = st.columns([1.1, 0.9])

with col_mem1:
    st.markdown("""
    ### The Two Types of Knowledge in AI
    
    * **Parametric Memory (Internal LLM Weights)**:
      * Knowledge stored as billions of floating-point neural network parameters (`W`).
      * **Strengths**: Excels at language grammar, reasoning, code generation, and broad historical facts.
      * **Fatal Flaws**: Static knowledge cutoff date; cannot be updated without expensive retraining or fine-tuning; prone to hallucinations; no native source citations.
    
    * **Non-Parametric Memory (Vector Database External Store)**:
      * Knowledge stored as dynamic, indexable document chunks with vector coordinates and structured metadata.
      * **Strengths**: Updated in real-time with sub-second writes; zero retraining cost; strict source attribution and citations; enforceable security filters (RBAC).
      * **Role in RAG**: Acts as the authoritative external reference library that the LLM consults before speaking.
    """)

with col_mem2:
    st.markdown("### The Dual-Engine RAG Paradigm")
    rag_dot = """
    digraph RAGParadigm {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        User [label="User Prompt\\n'What is our S3 retention policy?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_retriever {
            label="1. Retrieval Engine (Non-Parametric)"; style="dashed"; color="#059669";
            VDB [label="Vector Database (ChromaDB)\\nReal-time Index of Enterprise Docs", fillcolor="#D1FAE5", color="#059669", style="filled,bold"];
            Chunks [label="Top-K Grounded Chunks\\n'S3 logs retained for 90 days...'", fillcolor="#A7F3D0"];
        }

        subgraph cluster_generator {
            label="2. Generation Engine (Parametric)"; style="dashed"; color="#7C3AED";
            PromptAssembler [label="Augmented Prompt Assembler\\nSystem + Context + Query", fillcolor="#EDE9FE"];
            LLM [label="Large Language Model\\n(Claude / GPT-4 / Llama 3)", fillcolor="#DDD6FE", color="#7C3AED", style="filled,bold"];
        }

        Answer [label="Grounded, Hallucination-Free Answer\\nWith verifiable inline citations", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];

        User -> VDB [label="Query vector"];
        VDB -> Chunks;
        User -> PromptAssembler;
        Chunks -> PromptAssembler;
        PromptAssembler -> LLM;
        LLM -> Answer;
    }
    """
    st.graphviz_chart(rag_dot, use_container_width=False)

st.divider()

# Section 2: Why RAG Requires a Vector Database
st.header("2. Why RAG Requires a Vector Database", divider="rainbow")

col_why1, col_why2 = col_why1, col_why2 = st.columns([1, 1])

with col_why1:
    st.markdown("""
    ### Why Not Just Fine-Tune the LLM?
    Many engineering teams mistakenly attempt to teach an LLM private company facts by fine-tuning its weights:
    * **Catastrophic Forgetting**: Fine-tuning on company policies degrades the model's core reasoning and coding capabilities.
    * **Hallucination Persistence**: A fine-tuned model still hallucinates specific numbers, dates, and API endpoints.
    * **High Cost & Delay**: Updating policy changes requires hours of GPU compute and dataset curation.
    * **Zero Access Control**: You cannot prevent a regular employee from prompting a fine-tuned model to reveal executive salary weights.
    * **RAG Advantage**: Ingesting a modified policy into a vector database takes **under 100 milliseconds** with immediate zero-shot retrieval.
    """)

with col_why2:
    st.markdown("""
    ### Why Not Just Use Traditional Keyword Search (BM25)?
    * **The Vocabulary Mismatch Problem**:
      * User asks: *"How do I resolve a slow network connection in Envoy?"*
      * Knowledge base document says: *"Mitigating latency spikes and packet transmission drops in service mesh proxies."*
      * **BM25 Result**: Finds zero matching documents because none of the keywords (*"slow", "connection"*) overlap!
      * **Vector Search Result**: Dense embeddings recognize that *"slow connection"* and *"latency spikes in proxies"* represent identical semantic concepts, retrieving the exact answer effortlessly.
    """)

st.divider()

# Section 3: The Retrieval and Generation Phase Handshake
st.header("3. The Retrieval and Generation Phase Handshake", divider="rainbow")

st.markdown("""
A robust RAG system executes a strict contract between the **Retriever** and the **Generator**:
""")

col_hand1, col_hand2 = col_hand1, col_hand2 = st.columns([1, 1])

with col_hand1:
    st.markdown("""
    ### Phase 1: The Retrieval Phase
    1. **Query Encoding**: Convert natural language prompt into high-dimensional vector.
    2. **HNSW Nearest Neighbor Traversal**: Search the vector database for top-K candidates (typically K=3 to 6).
    3. **Metadata Filtering**: Filter out non-matching security tenants or outdated documents.
    4. **Distance Thresholding**: Verify that the top candidate is within a minimum confidence threshold (`distance < 0.35`). If distance is too high, declare an **out-of-domain query** rather than retrieving irrelevant noise!
    """)

with col_hand2:
    st.markdown("""
    ### Phase 2: The Generation Phase
    1. **Prompt Template Assembly**:
       ```text
       You are an enterprise AI assistant. Answer the user question 
       STRICTLY based on the provided context below. If the answer is 
       not in the context, say "I don't know based on our documentation."
       
       [CONTEXT]:
       Doc 1 (Page 4): "S3 bucket lifecycle rules expire logs after 90 days."
       Doc 2 (Page 7): "Cold archives are moved to Glacier Deep Archive."
       
       [QUESTION]: How long are S3 logs retained?
       ```
    2. **Constrained Decoding**: The LLM synthesizes the answer, citing `[Doc 1]`.
    """)

st.divider()

# Section 4: Interactive Workbench: RAG vs Direct LLM Simulator
st.header("4. Interactive Workbench: RAG vs Direct LLM Simulator", divider="rainbow")

st.markdown("""
Compare how a **Direct Parametric LLM** handles domain questions compared to a **Vector Database RAG-Augmented System**.
Observe the difference in factual grounding, hallucinations, and source citations.
""")

# Mock Knowledge Base
MOCK_KNOWLEDGE_BASE = {
    "SOC2 Data Retention Policy": {
        "text": "Under Titan Systems 2024 SOC2 compliance framework, customer audit logs are retained in encrypted S3 buckets for exactly 365 days before automatic purge.",
        "source": "compliance/soc2_retention_2024.pdf",
        "page": 14
    },
    "Kubernetes Ingress Rate Limits": {
        "text": "The production Envoy ingress proxy enforces a strict rate limit of 1,200 requests per minute per IP. Bursts exceeding this trigger HTTP 429.",
        "source": "infra/k8s_ingress_spec.md",
        "page": 2
    },
    "Executive Expense Reimbursement": {
        "text": "Employee meal stipends during business travel are capped at $85 per day. Alcohol expenses are strictly non-reimbursable.",
        "source": "hr/travel_policy_v3.docx",
        "page": 5
    }
}

col_sim_in, col_sim_toggle = st.columns([1.2, 0.8])

with col_sim_in:
    st.subheader("1. Enter Enterprise Question")
    scenario_choice = st.selectbox(
        "Choose an Enterprise Inquiry Scenario:",
        [
            "How long do we retain customer audit logs under SOC2?",
            "What is our ingress rate limit before getting HTTP 429?",
            "What is our daily travel meal stipend policy?"
        ]
    )

with col_sim_toggle:
    st.subheader("2. Retrieval Configuration")
    enable_rag = st.checkbox("Enable Vector Database RAG Retrieval", value=True)
    confidence_threshold = st.slider("Max Acceptable Cosine Distance:", min_value=0.1, max_value=0.5, value=0.3, step=0.05)

st.divider()

col_sim_res1, col_sim_res2 = col_sim_res1, col_sim_res2 = st.columns([1, 1])

with col_sim_res1:
    st.subheader("🔴 Direct LLM (Parametric Weights Only)")
    st.caption("No vector database attached. Must rely on training memory.")
    
    if "SOC2" in scenario_choice:
        direct_resp = "Under standard SOC2 compliance, audit logs are typically retained for 90 days to 7 years depending on local regulatory requirements, usually around 180 days."
    elif "ingress" in scenario_choice:
        direct_resp = "Kubernetes ingress controllers usually default to around 100 or 500 requests per second depending on whether you use Nginx or Traefik."
    else:
        direct_resp = "Most corporate travel policies allow around $50 to $100 per day for meals with itemized receipts."
        
    st.error(f"**Generated Response**:\n\n{direct_resp}")
    st.warning("⚠️ **Hallucination Risk**: Model is guessing generic industry averages because it has zero access to your private company policies!")

with col_sim_res2:
    st.subheader("🟢 Vector DB RAG Assistant (Non-Parametric Grounding)")
    st.caption("Queries ChromaDB for verified passages before calling the LLM.")
    
    if enable_rag:
        # Match mock document
        if "SOC2" in scenario_choice:
            matched_key = "SOC2 Data Retention Policy"
        elif "ingress" in scenario_choice:
            matched_key = "Kubernetes Ingress Rate Limits"
        else:
            matched_key = "Executive Expense Reimbursement"
            
        doc_entry = MOCK_KNOWLEDGE_BASE[matched_key]
        
        st.success(f"""
        **Grounded Response**:  
        According to our verified policy, customer audit logs are retained in encrypted S3 buckets for **exactly 365 days** before automatic purging.
        
        * **Citation**: `[{doc_entry['source']}, Page {doc_entry['page']}]`  
        * **Retrieval Confidence**: `94.2% (Cosine Distance: 0.058)`
        """)
        
        with st.expander("🔍 Inspect Injected Context Chunk", expanded=False):
            st.code(doc_entry["text"], language="text")
    else:
        st.info("RAG is disabled. Enable the checkbox above to connect the vector database.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the fundamental difference between parametric and non-parametric memory in Generative AI?"):
    st.markdown("""
    **Answer**:
    * **Parametric Memory** is stored inside the trained weights and biases of the neural network (`theta`). It is static, frozen at training time, expensive to modify, and prone to hallucinations when asked about private domain facts.
    * **Non-Parametric Memory** is stored externally in an indexed vector database (e.g. ChromaDB). It can be updated dynamically in real time, queried with deterministic metadata filters, audited for compliance, and directly cited.
    * RAG combines both: using non-parametric memory for factual retrieval and parametric memory for language reasoning and synthesis.
    """)

with st.expander("Q2: Why does fine-tuning an LLM fail to solve the enterprise knowledge retrieval problem?"):
    st.markdown("""
    **Answer**:
    Fine-tuning is designed to teach a model a **style, format, or task behavior** (e.g., answering like a doctor or writing SQL), NOT to act as a reliable factual knowledge database.
    1. **Knowledge Updates**: Updating a company return policy requires re-running training pipelines, whereas vector databases update in milliseconds.
    2. **Hallucination Rate**: Fine-tuned models continue to hallucinate specific numbers and dates with high confidence.
    3. **Access Control**: You cannot enforce Role-Based Access Control (RBAC) inside neural weights; any user can prompt-inject the model into revealing secret training data.
    """)

with st.expander("Q3: What is the Vocabulary Mismatch problem in search, and how do vector databases solve it?"):
    st.markdown("""
    **Answer**:
    The Vocabulary Mismatch problem occurs when a user queries for an answer using completely different words than those used by the author in the source document (e.g. user searches *"car engine overheating"* while document says *"internal combustion thermal management failure"*).
    * Traditional lexical search (BM25, inverted indices) requires exact token overlap and returns zero hits.
    * Vector databases map text to dense semantic vectors where conceptually related phrases occupy nearby geometric neighborhoods, retrieving the document regardless of differing terminology.
    """)

with st.expander("Q4: What should a RAG system do if the vector database returns documents with very high distance scores?"):
    st.markdown("""
    **Answer**:
    If the best retrieved document has a distance score exceeding a safe threshold (e.g. Cosine Distance > 0.40):
    1. The query is **out-of-domain** (the knowledge base contains no relevant information).
    2. **Do NOT inject the irrelevant chunks into the prompt**: Feeding noisy, unrelated chunks confuses the LLM and causes hallucinations.
    3. **Enforce Fallback Logic**: Either trigger an automated web search, ask the user for clarification, or have the system cleanly respond: *"I cannot find verified documentation to answer that question."*
    """)

with st.expander("Q5: How does RAG achieve source attribution and verifiable citations?"):
    st.markdown("""
    **Answer**:
    During document ingestion, every chunk stored in the vector database is tagged with provenance metadata (e.g. `{"source": "policy.pdf", "page": 12, "section": "Section 4"}`).
    When the vector database retrieves the top-K chunks, these metadata attributes are formatted into the prompt template alongside the chunk text (`[Doc 1 (Page 12)]: ...`). 
    The system prompt instructs the LLM to include references like `[Doc 1]` in its response, allowing user interfaces to render clickable links directly back to the original source page.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Expecting Fine-Tuning to Replace RAG**  
    Investing hundreds of thousands of dollars trying to fine-tune an LLM on company documentation results in hallucinated policies. Use RAG for knowledge retrieval and fine-tuning only for style/format.
    """)
    st.warning("""
    **Trap 2: Omitting Distance Thresholds**  
    Blindly injecting whatever top-3 chunks Chroma returns—even when distance is 0.85—forces the LLM to hallucinate answers from completely irrelevant context. Always enforce a distance cutoff.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Prompting Without Guardrails Against Hallucination**  
    Failing to instruct the LLM: *"Answer STRICTLY based on the provided context; if unknown, say I don't know"* allows the LLM to fall back to its parametric training weights and guess.
    """)
    st.warning("""
    **Trap 4: Passing Un-Sanitized Prompts Directly to Retrieval**  
    Passing raw multi-sentence chat prompts directly to vector search without query reformulation or keyword extraction degrades retrieval accuracy.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is Parametric Memory?"):
        st.markdown("""
        **Parametric Weights**:  
        Knowledge frozen inside the LLM's neural parameters. Hard to update and prone to hallucination.
        """)
    with st.expander("🗂️ Flashcard: What is Non-Parametric Memory?"):
        st.markdown("""
        **Non-Parametric Vector DB**:  
        External dynamic index of documents that updates in real time with verifiable citations and security filtering.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: What is the Vocabulary Mismatch problem?"):
        st.markdown("""
        **Vocabulary Mismatch**:  
        When user queries use different synonyms than the document text; solved by dense vector semantic search.
        """)
    with st.expander("🗂️ Flashcard: Why is distance thresholding critical?"):
        st.markdown("""
        **Distance Gatekeeper**:  
        Prevents feeding irrelevant, low-confidence chunks into LLM prompts when the user asks an out-of-domain question.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the distinction between Parametric and Non-Parametric memory in AI systems.", value=True)
chk2 = st.checkbox("I know why fine-tuning cannot replace RAG for dynamic enterprise knowledge retrieval.", value=True)
chk3 = st.checkbox("I understand how vector databases solve the Vocabulary Mismatch problem over traditional BM25.", value=True)
chk4 = st.checkbox("I know how the two-phase Retrieval and Generation handshake operates.", value=True)
chk5 = st.checkbox("I can implement distance thresholding to prevent feeding out-of-domain noise to LLMs.", value=True)
chk6 = st.checkbox("I can design a production prompt template that guarantees verifiable inline citations.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Excellent! You have mastered RAG Fundamentals and are ready for Chapter 14.2: The Complete RAG Pipeline!")
