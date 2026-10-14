import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="15.3 Advanced RAG Prompts - RAG Master Curriculum",
    page_icon="🧠",
    layout="wide"
)

# --- MODULE-LEVEL ENGINE IMPLEMENTATION ---
class AdvancedPromptSuiteEngine:
    """
    Production-grade Multi-Stage Prompt Suite Engine:
    1. Query rewriting prompts (HyDE, Sub-query decomposition, coreference resolution).
    2. Document relevance filtering prompts (Pointwise binary classification).
    3. Listwise and Pairwise LLM reranking prompts.
    4. Grounded factual summarization prompts.
    5. Multi-hop chain-of-thought scratchpad prompts.
    6. Citation verification and natural language inference (NLI) audit prompts.
    """
    def __init__(self):
        pass

    def build_query_rewrite_prompt(self, query: str, chat_history: str = "") -> str:
        """Constructs an advanced query rewriting and disambiguation prompt."""
        history_block = f"Conversation History:\n{chat_history}\n\n" if chat_history else ""
        return (
            "### ROLE: SEARCH QUERY REWRITER & OPTIMIZER\n"
            "Given the user utterance and prior conversation history, output an optimized standalone search query.\n"
            "Objectives:\n"
            "1. Resolve pronouns (it, that, they) using conversation context.\n"
            "2. Strip conversational pleasantries and filler.\n"
            "3. Append high-value domain synonyms to maximize BM25 and vector recall.\n\n"
            f"{history_block}"
            f"User Utterance: '{query}'\n\n"
            "Output Format: Output the rewritten search query ONLY. No preamble or explanation."
        )

    def build_document_relevance_prompt(self, query: str, document_text: str) -> str:
        """Constructs a pointwise zero-shot relevance evaluation prompt."""
        return (
            "### ROLE: DOCUMENT RELEVANCE JUDGE\n"
            "Evaluate whether the candidate document contains factual information relevant to answering the query.\n\n"
            f"Query: {query}\n\n"
            f"Candidate Document:\n{document_text}\n\n"
            "Evaluation Criteria:\n"
            "- RELEVANT: The document directly mentions concepts, data points, or rules needed to answer the query.\n"
            "- IRRELEVANT: The document discusses unrelated topics, tangential mentions, or contains zero answer-bearing facts.\n\n"
            "Output Format: Output EXACTLY either 'RELEVANT' or 'IRRELEVANT' followed by a 1-sentence justification."
        )

    def build_rerank_prompt(self, query: str, candidates: List[Dict[str, str]]) -> str:
        """Constructs a listwise reranking prompt."""
        items = "\n".join([f"[{c['id']}] {c['text'][:150]}..." for c in candidates])
        return (
            "### ROLE: LISTWISE RETRIEVAL RERANKER\n"
            "Rank the candidate passages in descending order of relevance to the query.\n\n"
            f"Query: {query}\n\n"
            f"Candidate Documents:\n{items}\n\n"
            "Instructions:\n"
            "Analyze which passages contain the most authoritative, direct, and complete answer to the query.\n"
            "Output Format: Output the ranked IDs separated by arrows, e.g., '[DOC-2] > [DOC-1] > [DOC-3]'."
        )

    def build_summarization_prompt(self, query: str, context_text: str) -> str:
        """Constructs a grounded extractive summarization prompt."""
        return (
            "### ROLE: GROUNDED CONTEXT SUMMARIZER\n"
            "Condense the following documentation into dense, fact-preserving bullet points tailored to the query.\n\n"
            f"Focus Query: {query}\n\n"
            f"Documentation:\n{context_text}\n\n"
            "Strict Rules:\n"
            "1. Retain all numerical values, percentages, system limits, and error codes.\n"
            "2. Discard narrative filler, corporate greetings, and marketing prose.\n"
            "3. Do not introduce any facts not present in the input text."
        )

    def build_multihop_prompt(self, query: str, context_text: str) -> str:
        """Constructs a multi-hop reasoning prompt with an explicit scratchpad."""
        return (
            "### ROLE: MULTI-HOP REASONING ENGINE\n"
            "Answer the query by connecting evidence across multiple documents. Follow this mandatory workflow:\n\n"
            "<scratchpad>\n"
            "Step 1: Identify the primary entity and look up its key attributes in Document A.\n"
            "Step 2: Cross-reference that attribute with the criteria described in Document B.\n"
            "Step 3: Synthesize the bridge inference resolving the multi-step dependency.\n"
            "</scratchpad>\n\n"
            "Final Answer:\n"
            "State the final conclusion concisely with bracketed source citations [DOC-ID].\n\n"
            f"<context>\n{context_text}\n</context>\n\n"
            f"<query>\n{query}\n</query>"
        )

    def build_citation_verification_prompt(self, claim: str, cited_text: str) -> str:
        """Constructs a Natural Language Inference (NLI) citation verification prompt."""
        return (
            "### ROLE: FACT-CHECKING AUDITOR & CITATION VERIFIER\n"
            "Determine whether the cited context logically entails the generated claim.\n\n"
            f"Claim to Verify: \"{claim}\"\n\n"
            f"Cited Document Snippet:\n\"{cited_text}\"\n\n"
            "Entailment Categories:\n"
            "- ENTAILED: The cited snippet directly proves and substantiates the claim.\n"
            "- NEUTRAL: The cited snippet is related, but does not provide sufficient evidence to confirm the claim.\n"
            "- CONTRADICTION: The cited snippet directly disproves or conflicts with the claim.\n\n"
            "Output Format: Output the category label ('ENTAILED', 'NEUTRAL', or 'CONTRADICTION') and a brief rationale."
        )


# Instantiate engine at module level
prompt_suite = AdvancedPromptSuiteEngine()

# --- HEADER SECTION ---
st.title("15.3 Advanced RAG Prompts: Rewriting, Relevance Scoring, Reranking, Multi-Hop Reasoning, & Verification")
st.caption("Chapter 15: Prompt Engineering for RAG | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
In naive RAG implementations, prompting is applied only at the very final generation step: formatting retrieved 
documents into an answer. In **Enterprise-Grade Agentic RAG**, prompting orchestrates every individual stage 
of the information lifecycle:
1. **Pre-Retrieval**: Query rewriting, hypothetical document synthesis (HyDE), and sub-query decomposition.
2. **Post-Retrieval**: Pointwise LLM relevance filtering, listwise attention-based reranking, and grounded summarization.
3. **Synthesis**: Multi-hop chain-of-thought scratchpads that systematically link fragmented evidence.
4. **Post-Generation**: Automated citation auditing and Natural Language Inference (NLI) fact verification.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Multi-Stage Agentic RAG Prompt Orchestrator", divider="rainbow")

st.markdown("""
The multi-stage prompt pipeline distributes specialized LLM prompts across each phase of retrieval and generation, 
guaranteeing high semantic precision and zero hallucination.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MultiStagePromptPipeline {
        graph [bgcolor="#0B1120", rankdir=TB, margin="0.2,0.1", compound=true];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [fontname="Helvetica", fontsize=9, color="#64748B", arrowsize=0.8, penwidth=1.2];

        // STAGE 1: PRE-RETRIEVAL
        subgraph cluster_pre {
            label="Stage 1: Pre-Retrieval Transformation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            UserRaw [label="Raw Conversational Query\\n('What about the other database?')", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
            RewritePrompt [label="Query Rewriter Prompt\\n(Coreference resolution & keyword expansion)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
            HydePrompt [label="HyDE Generator Prompt\\n(Synthesizes hypothetical ideal answer)", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];
        }

        // RETRIEVAL ENGINE
        VectorSearch [label="Hybrid Vector & Lexical Retriever\\n(Fetches Top-50 candidate chunks)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569", shape=cylinder];

        // STAGE 2: POST-RETRIEVAL FILTERING
        subgraph cluster_post {
            label="Stage 2: Post-Retrieval Validation";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            RelevancePrompt [label="Pointwise Relevance Filter\\n(Classifies chunks as RELEVANT / IRRELEVANT)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            RerankPrompt [label="Listwise LLM Reranker\\n(Orders chunks using self-attention)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
            SummarizePrompt [label="Grounded Summarization\\n(Compresses chunks to dense fact triples)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#D97706"];
        }

        // STAGE 3: SYNTHESIS
        subgraph cluster_synthesis {
            label="Stage 3: Multi-Hop Reasoning & Synthesis";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            MultiHopPrompt [label="Multi-Hop CoT Scratchpad\\n(Connects evidence bridges Doc A -> Doc B)", fillcolor="#0F172A", fontcolor="#10B981", color="#059669"];
            AnswerDraft [label="Draft Generation with Citations\\n('Failover completes in 120s [DOC-1]')", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569"];
        }

        // STAGE 4: POST-GENERATION VERIFICATION
        subgraph cluster_audit {
            label="Stage 4: Post-Generation Quality Assurance";
            color="#334155";
            style="dashed";
            fontcolor="#94A3B8";
            VerificationPrompt [label="Citation Verification & NLI Judge\\n(Verifies entailment: Claim -> Citation)", fillcolor="#0F172A", fontcolor="#A855F7", color="#7C3AED"];
        }

        // FINAL DELIVERY
        VerifiedAnswer [label="Audited Enterprise Response\\n(100% verified, grounded, and traceable)", fillcolor="#1E1B4B", fontcolor="#A855F7", color="#7C3AED", shape=doubleoctagon];

        // PIPELINE FLOW
        UserRaw -> RewritePrompt;
        RewritePrompt -> HydePrompt;
        HydePrompt -> VectorSearch [label="Optimized Vector Query"];
        VectorSearch -> RelevancePrompt [label="Raw Candidate Hits"];
        RelevancePrompt -> RerankPrompt [label="Filtered Chunks"];
        RerankPrompt -> SummarizePrompt [label="Top Ranked Chunks"];
        SummarizePrompt -> MultiHopPrompt [label="Dense Fact Triples"];
        MultiHopPrompt -> AnswerDraft;
        AnswerDraft -> VerificationPrompt [label="Draft & Citations"];
        VerificationPrompt -> VerifiedAnswer [label="Passes NLI Audit"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Advantages of Multi-Stage Prompt Orchestration:
* **Micro-Prompt Specialization**: Each prompt performs a narrow, low-temperature task, drastically reducing hallucinations.
* **Cost & Latency Tiering**: Fast, compact models (e.g., Llama-3-8B, GPT-4o-mini) handle rewriting and filtering, while large models are reserved for final multi-hop reasoning.
* **Deterministic Guardrails**: Fact-checking prompts catch 95%+ of hallucinations before answers reach end users.
""")

st.divider()

# ==============================================================================
# SECTION 2: QUERY REWRITING PROMPTS
# ==============================================================================
st.header("2. Query Rewriting Prompts (Hypothetical Document Embeddings, Sub-Queries & Disambiguation)", divider="rainbow")

st.markdown("""
User queries in conversational systems are often underspecified, ambiguous, or reference prior messages using pronouns 
(*"What about the secondary region?"*). Pre-retrieval query rewriting transforms these inputs into self-contained search vectors.
""")

qr_col1, qr_col2 = st.columns([1, 1])

with qr_col1:
    st.markdown("""
    ### Core Query Rewriting Patterns:
    1. **Coreference Resolution**: Replaces pronouns (*"it"*, *"they"*, *"that policy"*) with concrete entity names derived from chat history.
    2. **Hypothetical Document Embeddings (HyDE)**: Prompts an LLM to generate a speculative, ideal answer passage. The hypothetical answer is then embedded, bridging the vocabulary gap between short queries and long document passages.
    3. **Sub-Query Decomposition**: Splits complex composite questions (*"Compare the latency and pricing of Aurora vs DynamoDB"*) into independent sub-queries.
    """)

with qr_col2:
    st.markdown("### Production Query Rewriter Template")
    st.code(r'''
### INSTRUCTION: QUERY REWRITER & EXPANDER
Given the chat history and the latest user message, generate an 
expanded search query optimized for vector and keyword retrieval.

Rules:
1. Replace all pronouns with exact nouns from history.
2. Append 2-3 domain synonyms (e.g., SLA -> uptime, availability).
3. Do not answer the question; generate the search query only.

Output Format: A single search string.
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 3: DOCUMENT RELEVANCE PROMPTS
# ==============================================================================
st.header("3. Document Relevance Prompts (Binary Relevance Filtering, Noise Rejection & Pointwise Scoring)", divider="rainbow")

st.markdown("""
Standard vector search frequently retrieves chunks with high cosine similarity that are nevertheless completely off-topic 
(e.g., matching common lexical boilerplate). Pointwise document relevance prompts discard these false positives.
""")

rel_col1, rel_col2 = st.columns([1, 1])

with rel_col1:
    st.markdown("""
    ### Pointwise Zero-Shot Filtering:
    * **The Workflow**: For each retrieved candidate passage, a lightweight LLM evaluates whether the document contains direct factual evidence.
    * **Output Constraints**: Constrained to binary tokens (`RELEVANT` vs `IRRELEVANT`), enabling single-token decoding with minimal latency.
    * **Noise Elimination**: Discards introductory fluff, marketing boilerplate, and irrelevant legal disclosures, preventing them from consuming context window tokens.
    """)

with rel_col2:
    st.markdown("### Pointwise Binary Judge Template")
    st.code(r'''
### INSTRUCTION: POINTWISE RELEVANCE JUDGE
User Query: {query}
Candidate Passage: {passage}

Task: Determine if the candidate passage contains specific information 
that directly helps answer the user query.

Output: 
Decision: [RELEVANT / IRRELEVANT]
Confidence: [HIGH / MEDIUM / LOW]
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 4: RERANKING PROMPTS
# ==============================================================================
st.header("4. Reranking Prompts (Listwise & Pairwise LLM Attention Reranking with CoT)", divider="rainbow")

st.markdown("""
While cross-encoders (e.g., BGE-Reranker, Cohere) provide fast reranking, complex domain queries require 
deep multi-attribute reasoning. Listwise LLM reranking prompts pass multiple candidates to the model simultaneously.
""")

rr_col1, rr_col2 = st.columns([1, 1])

with rr_col1:
    st.markdown("""
    ### Listwise vs Pairwise LLM Reranking:
    * **Listwise Reranking**: The model receives candidate passages simultaneously (`[Passage 1]`, `[Passage 2]`, `[Passage 3]`) and outputs a ranked ordering: `[2] > [1] > [3]`. This allows the LLM's cross-attention heads to compare passages directly.
    * **Pairwise Tournament**: Compares passages in pairs: *Is Passage A more relevant than Passage B?* Highest precision, but incurs O(N log N) LLM calls.
    * **Chain-of-Thought Reranking**: Instructs the model to evaluate *why* Passage 1 is more relevant before producing the final ordering.
    """)

with rr_col2:
    st.markdown("### Listwise Attention Reranker Prompt")
    st.code(r'''
### INSTRUCTION: LISTWISE PASSAGE RERANKER
Query: {query}

Passages:
[ID: 1] {snippet_1}
[ID: 2] {snippet_2}
[ID: 3] {snippet_3}

Task: Rank the passages by descending relevance to the query.
Rank format: [ID] > [ID] > [ID]
Explanation: One sentence describing the ranking rationale.
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 5: SUMMARIZATION PROMPTS
# ==============================================================================
st.header("5. Summarization Prompts (Grounded Abstractive Distillation & Key Evidence Extraction)", divider="rainbow")

st.markdown("""
Long retrieved passages consume excessive token headroom and dilute attention. 
Grounded summarization prompts extract dense factual statements while discarding narrative padding.
""")

sum_col1, sum_col2 = st.columns([1, 1])

with sum_col1:
    st.markdown("""
    ### Grounded Information Extraction:
    * **Entity & Metric Preservation**: Mandates preserving all numerical values, percentages, system limits, and error codes without approximation.
    * **Syntactic Compression**: Eliminates conversational pleasantries, passive voice, and redundant introductory phrasing.
    * **Zero-Extrapolation Mandate**: Forbids adding external context or background explanations.
    """)

with sum_col2:
    st.markdown("### Grounded Extraction Prompt")
    st.code(r'''
### INSTRUCTION: FACTUAL DISTILLATION ENGINE
Query: {query}
Raw Context: {context}

Extract all facts, metrics, and procedures relevant to the query 
into concise bullet points.
Rules:
- Never alter numerical quantities or technical terms.
- Do not add commentary or background explanations.
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 6: MULTI-HOP REASONING PROMPTS
# ==============================================================================
st.header("6. Multi-Hop Reasoning Prompts (Chain-of-Thought Bridge Deductions across Fragmented Evidence)", divider="rainbow")

st.markdown("""
Multi-hop questions cannot be answered from a single document. They require bridging facts across disparate sources:
e.g., *Document A* states *“Service X relies on Database Y”*, while *Document B* states *“Database Y has an SLA of 99.95%”*.
""")

mh_col1, mh_col2 = st.columns([1, 1])

with mh_col1:
    st.markdown("""
    ### The Scratchpad Bridge Pattern:
    1. **Step 1 (Entity Disambiguation)**: Identify the primary subject in Document A.
    2. **Step 2 (Attribute Extraction)**: Extract the linking attribute (foreign key relationship).
    3. **Step 3 (Bridge Traversal)**: Locate the corresponding record in Document B.
    4. **Step 4 (Final Synthesis)**: Combine both assertions into a unified factual deduction.
    
    By enforcing a `<scratchpad>` or `<thinking>` block, the model's autoregressive attention attends to intermediate 
    logical deductions before generating the final answer.
    """)

with mh_col2:
    st.markdown("### Multi-Hop Chain-of-Thought Prompt")
    st.code(r'''
### INSTRUCTION: MULTI-HOP REASONING ENGINE
Query: {query}
Context: {context}

Resolve this multi-hop query using an explicit reasoning scratchpad:
<scratchpad>
Fact 1 from [Doc A]: ...
Fact 2 from [Doc B]: ...
Bridge Deduction: Connecting Fact 1 and Fact 2 proves that ...
</scratchpad>

Final Answer: Provide the definitive answer with citations [Doc A] and [Doc B].
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 7: CITATION VERIFICATION PROMPTS
# ==============================================================================
st.header("7. Citation Verification Prompts (Fact-Checking Auditor & Sentence-Level Attribution Validation)", divider="rainbow")

st.markdown("""
Even with defensive prompts, models occasionally generate citations for claims that are not supported by the cited text. 
Citation verification prompts run as an **automated post-generation judge** using Natural Language Inference (NLI).
""")

cv_col1, cv_col2 = st.columns([1, 1])

with cv_col1:
    st.markdown("""
    ### The Natural Language Inference (NLI) Protocol:
    The verification engine parses the generated answer into discrete sentences, identifies the cited document ID, 
    and prompts an LLM auditor to classify the relationship:
    * **ENTAILED**: The cited text fully proves the generated claim. (Accept)
    * **NEUTRAL**: The cited text discusses related ideas, but lacks sufficient proof. (Flag/Strip)
    * **CONTRADICTED**: The cited text conflicts with the generated claim. (Hard Reject & Alert)
    """)

with cv_col2:
    st.markdown("### NLI Citation Verifier Prompt")
    st.code(r'''
### INSTRUCTION: NLI CITATION AUDITOR
Premise (Cited Passage): "{passage}"
Hypothesis (Generated Claim): "{claim}"

Determine if the premise logically entails the hypothesis.
Choose exactly one:
- ENTAILED (Directly supported by facts in the passage)
- NEUTRAL (Insufficient evidence in the passage)
- CONTRADICTED (Passage contradicts the claim)

Justification: Explain the logical deduction in one sentence.
''', language="markdown")

st.divider()

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Multi-Stage Advanced RAG Prompt Pipeline", divider="rainbow")

st.markdown("""
Below is the production implementation of the `AdvancedPromptSuiteEngine`, managing all prompt generation 
and evaluation templates in module scope.
""")

st.code(r'''
class AdvancedPromptSuiteEngine:
    """
    Enterprise Multi-Stage RAG Prompt Engine:
    - Pre-Retrieval: Query rewriting, coreference resolution, and HyDE synthesis.
    - Post-Retrieval: Pointwise relevance classification, listwise reranking, and distillation.
    - Synthesis: Multi-hop reasoning with explicit chain-of-thought scratchpads.
    - Post-Generation: NLI citation verification and factual entailment auditing.
    """
    def build_query_rewrite_prompt(self, query: str, chat_history: str = "") -> str:
        history_block = f"Conversation History:\n{chat_history}\n\n" if chat_history else ""
        return (
            "### ROLE: SEARCH QUERY REWRITER\n"
            f"{history_block}User Utterance: '{query}'\n\n"
            "Generate an optimized standalone search query resolving pronouns and adding domain synonyms."
        )

    def build_document_relevance_prompt(self, query: str, document_text: str) -> str:
        return (
            "### ROLE: POINTWISE RELEVANCE JUDGE\n"
            f"Query: {query}\n\nDocument:\n{document_text}\n\n"
            "Output EXACTLY 'RELEVANT' or 'IRRELEVANT' with a 1-sentence justification."
        )

    def build_multihop_prompt(self, query: str, context_text: str) -> str:
        return (
            "### ROLE: MULTI-HOP REASONING ENGINE\n"
            "Use an explicit <scratchpad> to trace evidence bridges between documents:\n"
            "<scratchpad>\nStep 1 -> Step 2 -> Conclusion\n</scratchpad>\n\n"
            f"<context>\n{context_text}\n</context>\n\n<query>\n{query}\n</query>"
        )

    def build_citation_verification_prompt(self, claim: str, cited_text: str) -> str:
        return (
            "### ROLE: NLI CITATION AUDITOR\n"
            f"Claim: \"{claim}\"\nCited Passage: \"{cited_text}\"\n\n"
            "Classify logical entailment as 'ENTAILED', 'NEUTRAL', or 'CONTRADICTED'."
        )
''', language="python")

st.divider()

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Multi-Stage Prompt Pipeline Simulator & Verification Suite", divider="rainbow")

st.markdown("""
Experiment with all six advanced prompt patterns across the end-to-end RAG lifecycle. 
Inspect the dynamically generated prompt templates for each stage.
""")

sample_docs_db = {
    "DOC-ALPHA": "The Aurora global database cluster uses asynchronous storage-level replication across us-east-1 and eu-west-1 with an RPO of under 1 second.",
    "DOC-BETA": "Payment processing microservices require underlying databases with an RPO under 2 seconds to comply with PCI-DSS 4.0 regulations.",
    "DOC-GAMMA": "Corporate catering lunch is served weekdays from 12:00 PM to 2:00 PM in the 4th floor cafeteria."
}

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.markdown("### Pipeline Stage Selection")
    pipeline_stage = st.selectbox(
        "Choose Pipeline Stage to Inspect:",
        options=[
            "Stage 1: Pre-Retrieval Query Rewriting",
            "Stage 2: Post-Retrieval Relevance Filtering",
            "Stage 3: Listwise LLM Reranking",
            "Stage 4: Grounded Context Summarization",
            "Stage 5: Multi-Hop CoT Reasoning",
            "Stage 6: Citation Verification (NLI Audit)"
        ],
        index=0
    )

    if pipeline_stage == "Stage 1: Pre-Retrieval Query Rewriting":
        in_query = st.text_input("User Utterance:", value="What about its replication lag?")
        in_history = st.text_area("Chat History:", value="User: Tell me about the Aurora database cluster.\nAssistant: Aurora is an enterprise relational database.")
    elif pipeline_stage == "Stage 2: Post-Retrieval Relevance Filtering":
        in_query = st.text_input("User Query:", value="Does the database comply with PCI-DSS regulations?")
        in_doc_key = st.selectbox("Document to Judge:", options=list(sample_docs_db.keys()), index=0)
    elif pipeline_stage == "Stage 3: Listwise LLM Reranking":
        in_query = st.text_input("User Query:", value="Which database setup satisfies compliance requirements?")
    elif pipeline_stage == "Stage 4: Grounded Context Summarization":
        in_query = st.text_input("Focus Query:", value="What are the replication specs?")
        in_doc_key = "DOC-ALPHA"
    elif pipeline_stage == "Stage 5: Multi-Hop CoT Reasoning":
        in_query = st.text_input("Composite Multi-Hop Query:", value="Does our Aurora cluster meet the database requirements for payment processing?")
    else:
        in_claim = st.text_input("Generated Claim to Audit:", value="The Aurora cluster achieves replication lag of under 1 second.")
        in_doc_key = "DOC-ALPHA"

with wb_col2:
    st.markdown("### Compiled Multi-Stage Prompt Output")

    if pipeline_stage == "Stage 1: Pre-Retrieval Query Rewriting":
        compiled_res = prompt_suite.build_query_rewrite_prompt(in_query, in_history)
    elif pipeline_stage == "Stage 2: Post-Retrieval Relevance Filtering":
        compiled_res = prompt_suite.build_document_relevance_prompt(in_query, sample_docs_db[in_doc_key])
    elif pipeline_stage == "Stage 3: Listwise LLM Reranking":
        candidates = [{"id": k, "text": v} for k, v in sample_docs_db.items()]
        compiled_res = prompt_suite.build_rerank_prompt(in_query, candidates)
    elif pipeline_stage == "Stage 4: Grounded Context Summarization":
        compiled_res = prompt_suite.build_summarization_prompt(in_query, sample_docs_db[in_doc_key])
    elif pipeline_stage == "Stage 5: Multi-Hop CoT Reasoning":
        context_corpus = f"[DOC-ALPHA]: {sample_docs_db['DOC-ALPHA']}\n[DOC-BETA]: {sample_docs_db['DOC-BETA']}\n[DOC-GAMMA]: {sample_docs_db['DOC-GAMMA']}"
        compiled_res = prompt_suite.build_multihop_prompt(in_query, context_corpus)
    else:
        compiled_res = prompt_suite.build_citation_verification_prompt(in_claim, sample_docs_db[in_doc_key])

    st.text_area("Live Generated Stage Prompt Template:", value=compiled_res, height=320)

st.divider()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Core Systems Architecture Questions")

with st.expander("**Q1: What is Hypothetical Document Embeddings (HyDE), and when does it outperform standard query embedding?**"):
    st.markdown("""
    **Answer**:
    HyDE prompts an LLM to generate a hypothetical ideal document that *would* answer the query, without access to external context. 
    The generated hypothetical document is then passed to the embedding model to create the search vector.
    * **Why it works**: Short queries (e.g., *"TLS 1.3 requirements"*) lack the lexical and semantic depth of full document passages. Embedding a synthetic 150-word answer aligns much closer in vector space with the actual target documentation.
    * **When to use**: Highly effective for short, conceptual queries. Less effective for needle-in-a-haystack lookups of specific part numbers or rare named entities.
    """)

with st.expander("**Q2: How does a listwise LLM reranker differ from a cross-encoder model in production?**"):
    st.markdown("""
    **Answer**:
    * **Cross-Encoder**: Uses a small specialized BERT/RoBERTa encoder to output a scalar relevance score `score(query, doc)` for each document pair independently. Very fast (5-15ms), but cannot compare passages against one another.
    * **Listwise LLM Reranker**: Passes 5 to 10 candidate passages into an autoregressive LLM prompt simultaneously. The model's self-attention mechanism compares and contrasts all candidates in a single forward pass, identifying redundancies and ranking passages holistically.
    * **Tradeoff**: Listwise LLM reranking provides higher precision for nuanced queries, but costs more in token inference and latency (150-400ms).
    """)

with st.expander("**Q3: How does the Scratchpad / Chain-of-Thought pattern resolve multi-hop RAG queries?**"):
    st.markdown("""
    **Answer**:
    In multi-hop queries, the final answer depends on intermediate deductive steps across disparate documents. 
    If a model attempts to generate the answer directly, its autoregressive decoding is forced to commit to tokens before 
    computing the cross-document bridge.
    By enforcing a `<scratchpad>` or `<thinking>` block, the model generates intermediate tokens that explicitly record:
    1. Entity A connects to Attribute X via Document 1.
    2. Attribute X satisfies Condition Y in Document 2.
    These intermediate tokens then become available as keys and values in subsequent self-attention layers, making the final synthesis factually sound.
    """)

with st.expander("**Q4: What is Natural Language Inference (NLI) in citation verification, and how does it prevent hallucinated references?**"):
    st.markdown("""
    **Answer**:
    NLI evaluates whether a Premise (the cited context passage) logically entails a Hypothesis (the generated claim).
    Instead of relying on heuristic overlap, an NLI auditor classifies the claim into:
    * **Entailed**: 100% substantiated by the text.
    * **Neutral**: Possible, but unproven by the cited source.
    * **Contradicted**: Disproven by the cited source.
    Automated NLI judges catch hallucinations where the model correctly cites a document ID, but attributes a claim that the document does not actually make.
    """)

with st.expander("**Q5: Why is Pointwise Relevance Filtering critical when using Hybrid Search (BM25 + Dense Vectors)?**"):
    st.markdown("""
    **Answer**:
    Hybrid search increases recall by combining BM25 keyword matching with dense semantic embeddings. 
    However, BM25 frequently retrieves documents that match query keywords in completely irrelevant contexts 
    (e.g., matching *"database failover"* in an employee travel policy discussing flight cancellations).
    Pointwise relevance prompts act as an early gatekeeper, stripping these keyword false positives before they consume 
    valuable context window tokens or contaminate multi-hop reasoning.
    """)

st.markdown("### Production Traps & Antipatterns")

trap_col1, trap_col2 = st.columns(2)

with trap_col1:
    st.error("""
    **Production Trap 1: Monolithic Single-Prompt RAG Pipelines**
    * **Failure Mode**: Asking a single prompt to simultaneously rewrite the query, filter documents, synthesize facts, and verify citations leads to attention dilution and frequent hallucination.
    * **Solution**: Decompose RAG into specialized, modular micro-prompts executed across discrete stages.
    """)

    st.error("""
    **Production Trap 2: Unconstrained Free-Text Evaluators**
    * **Failure Mode**: Asking an evaluation prompt *"Is this document relevant?"* without output constraints produces conversational essays (*"Yes, because..."*) that crash automated parsers.
    * **Solution**: Constrain outputs to strict binary tokens: `RELEVANT` or `IRRELEVANT`.
    """)

with trap_col2:
    st.warning("""
    **Production Trap 3: Latency Stacking from Sequential Prompts**
    * **Failure Mode**: Running query rewriting, pointwise filtering on 20 chunks sequentially, and then generation creates 3+ seconds of user-perceived latency.
    * **Solution**: Run pointwise filtering in parallel (using `asyncio.gather`), or use fast SLMs (e.g., 8B parameters) for early pipeline stages.
    """)

    st.warning("""
    **Production Trap 4: Trusting Citation Tags Without Verification**
    * **Failure Mode**: Assuming that because an LLM generated `[DOC-1]`, the claim is factually accurate. LLMs frequently hallucinate citations that sound plausible.
    * **Solution**: Implement an automated post-generation NLI verifier to validate factual entailment.
    """)

st.markdown("### Systems Architecture Flashcards")

fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)

with fc_col1:
    with st.container(border=True):
        st.markdown("**Front**: Hypothetical Document Embeddings (HyDE)")
        st.caption("How does HyDE bridge query-passage vector divergence?")
        st.markdown("**Back**: Generates a speculative ideal answer and embeds that answer passage, aligning vector space with actual documents.")

with fc_col2:
    with st.container(border=True):
        st.markdown("**Front**: Multi-Hop Scratchpad Pattern")
        st.caption("Why require a scratchpad block before generating multi-document answers?")
        st.markdown("**Back**: Forces the model to compute intermediate bridge deductions as visible tokens before committing to the final answer.")

with fc_col3:
    with st.container(border=True):
        st.markdown("**Front**: NLI Citation Verification")
        st.caption("What three labels define Natural Language Inference fact auditing?")
        st.markdown("**Back**: 1. ENTAILED (substantiated), 2. NEUTRAL (insufficient proof), 3. CONTRADICTED (conflicted).")

with fc_col4:
    with st.container(border=True):
        st.markdown("**Front**: Listwise LLM Reranking")
        st.caption("How does listwise reranking differ from cross-encoders?")
        st.markdown("**Back**: Evaluates multiple candidates simultaneously using the LLM's cross-attention heads, ranking passages holistically.")

st.divider()

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your understanding of advanced multi-stage RAG prompt engineering before proceeding:")

c1 = st.checkbox("I understand how query rewriting, coreference resolution, and HyDE optimize pre-retrieval vector alignment.", value=True)
c2 = st.checkbox("I can implement pointwise zero-shot relevance filtering to eliminate retriever noise and lexical false positives.", value=True)
c3 = st.checkbox("I understand the mechanics of listwise LLM reranking using cross-document self-attention.", value=True)
c4 = st.checkbox("I know how to design multi-hop scratchpad prompts that establish logical bridges across fragmented documents.", value=True)
c5 = st.checkbox("I can configure automated post-generation citation verification using Natural Language Inference (NLI) entailment.", value=True)
c6 = st.checkbox("I can architect a modular multi-stage prompt pipeline balancing latency, cost, and factual rigor.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 **Advanced RAG Prompts Mastery Achieved!** You are ready to architect full agentic, multi-stage RAG pipelines.")
