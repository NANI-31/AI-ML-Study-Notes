import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="16.1 RAG Basics | GenAI Roadmap",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .rag-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("📚 16.1 RAG Basics — Retrieval-Augmented Generation Architecture")
st.markdown(
    """
Welcome to **Lesson 16.1: RAG Basics**.  
Large Language Models have static knowledge cutoffs and zero visibility into your organization's private files, internal wikis, or live customer records. Furthermore, relying purely on pre-trained parametric weights produces confident **hallucinations**.

**Retrieval-Augmented Generation (RAG)** bridges this fundamental gap: dynamically searching an external knowledge base at query time, injecting the most relevant document chunks into the prompt context, and constraining the model to generate factually grounded responses.
"""
)

st.divider()

# ==============================================================================
# Section 1: Why RAG is Essential — Parametric vs. Non-Parametric Memory
# ==============================================================================
st.header("1. Parametric Weights vs. Non-Parametric Retrieval", divider="rainbow")

col_rg1, col_rg2 = st.columns(2)

with col_rg1:
    with st.container(border=True):
        st.subheader("🧠 Parametric Memory (Model Weights)")
        st.markdown(
            """
- **Static Knowledge**: Frozen at the moment pre-training concluded; blind to recent developments.
- **Hallucination Prone**: Probabilistic token prediction creates convincing fabrications when facts are fuzzy.
- **Zero Access to Private Data**: Has never seen internal company policies, Jira tickets, or private customer databases.
- **Massive Update Cost**: Re-training or fine-tuning weights costs thousands of dollars and weeks of GPU compute.
"""
        )

with col_rg2:
    with st.container(border=True):
        st.subheader("📚 Non-Parametric Memory (RAG Index)")
        st.markdown(
            """
- **Dynamic & Live**: Real-time updates; adding a new PDF takes seconds without modifying model weights.
- **Factually Grounded**: Answers cite verbatim quotes and explicit document sources.
- **Private Data Access**: Indexes private enterprise repositories behind access control barriers.
- **Cost Effective**: Uses off-the-shelf foundation models with standard vector databases.
"""
        )

st.divider()

# ==============================================================================
# Section 2: The Core RAG Lifecycle Pipeline
# ==============================================================================
st.header("2. The 8 Stages of the RAG Lifecycle", divider="rainbow")
st.markdown("From raw unstructured documents to factually validated generation:")

rag_stages_df = pd.DataFrame({
    "Pipeline Stage": [
        "1. Document Ingestion",
        "2. Chunking & Splitting",
        "3. Dense Embedding",
        "4. Vector Indexing",
        "5. User Query Retrieval",
        "6. Reranking & Filtering",
        "7. Context Prompt Augmentation",
        "8. Grounded Generation"
    ],
    "Engineering Implementation": [
        "Parsing PDFs, Word docs, Markdown, Confluence pages, and SQL dumps into text",
        "Dividing text into semantic chunks (e.g. 500 tokens with 50-token overlap)",
        "Converting chunks into dense vectors using embedding models (e.g., text-embedding-3-small)",
        "Storing vectors and document metadata in vector databases (ChromaDB, Pinecone, Qdrant)",
        "Embedding user query and executing cosine similarity or Approximate Nearest Neighbor (ANN) search",
        "Passing top-20 candidates through a cross-encoder reranker to select top-3 most relevant chunks",
        "Packaging selected chunks into <context> tags inside the system prompt template",
        "Generating the final response constrained strictly to the provided context chunks"
    ],
    "Primary Failure Mode": [
        "OCR text corruption or layout loss",
        "Splitting mid-sentence or breaking tables",
        "Out-of-vocabulary terms or domain mismatch",
        "Slow indexing latency or memory leaks",
        "Retrieval of irrelevant distractors",
        "Excessive reranker latency spikes",
        "Context window overflow or token budget breach",
        "Hallucinating beyond provided source text"
    ]
})
st.dataframe(rag_stages_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The End-to-End RAG Pipeline", divider="rainbow")
st.markdown("Visualizing the ingestion and retrieval-generation runtime workflows:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        docs [label="Raw Documents\n(PDF / Wiki / SQL)", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        chunker [label="Chunking Engine\n(500 tokens / 50 overlap)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        vdb [label="Vector Database\n(Embeddings Index)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        user_q [label="User Query", fillcolor="#EAB308", fontcolor="#000000"]
        retriever [label="Similarity Search\n(Top-K Chunks)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        prompt [label="Augmented Prompt\n(<context> + Query)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        llm [label="LLM Generator", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        resp [label="Grounded Response\nwith Source Citations", fillcolor="#10B981", fontcolor="#FFFFFF"]

        docs -> chunker -> vdb
        user_q -> retriever
        vdb -> retriever [label=" Cosine Match"]
        retriever -> prompt
        user_q -> prompt
        prompt -> llm -> resp
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive RAG Step-by-Step Simulator
# ==============================================================================
st.header("4. Interactive Step-by-Step RAG Simulator", divider="rainbow")
st.markdown("Experience how a query triggers retrieval, prompt augmentation, and grounded synthesis:")

user_query = st.text_input("User Natural Language Query", value="What is the travel reimbursement limit for client dinners?")

# Simulated Vector DB Knowledge Base
MOCK_CHUNKS = [
    {
        "id": "chunk_01",
        "doc": "HR Employee Handbook (Rev 2026)",
        "text": "Standard travel meal reimbursement is capped at $75 per person for client dinners. Alcoholic beverages must be itemized separately and require VP pre-approval.",
        "sim_score": 0.89
    },
    {
        "id": "chunk_02",
        "doc": "Travel Policy FAQ",
        "text": "All expense claims exceeding $50 must include itemized tax invoices submitted via Expensify within 14 business days of travel completion.",
        "sim_score": 0.74
    },
    {
        "id": "chunk_03",
        "doc": "Engineering On-Call Policy",
        "text": "Engineers scheduled on primary rotation receive a $300 weekly standby stipend plus overtime for incident remediation over 30 minutes.",
        "sim_score": 0.22
    }
]

top_k = st.slider("Top-K Retrieval Cutoff", min_value=1, max_value=3, value=2)

col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    st.subheader("🔍 Retrieved Knowledge Chunks")
    selected_chunks = MOCK_CHUNKS[:top_k]
    for ch in selected_chunks:
        with st.container(border=True):
            st.markdown(f"**Source:** `{ch['doc']}` | **Similarity:** `{ch['sim_score']:.2f}`")
            st.write(ch['text'])

with col_sim2:
    st.subheader("📄 Dynamic Assembled Prompt & Output")
    
    context_str = "\n\n".join([f"<source id='{c['id']}' title='{c['doc']}'>\n{c['text']}\n</source>" for c in selected_chunks])
    
    assembled_rag_prompt = f"""<system>
Answer the user question strictly using the reference context provided below.
If the context does not contain the answer, reply: 'I cannot find that in the provided documents.'
Always cite your source titles.
</system>

<context>
{context_str}
</context>

<user_question>
{user_query}
</user_question>"""

    st.code(assembled_rag_prompt, language="xml")
    
    st.success("🟢 Grounded Generation Output:")
    st.markdown(
        """
```text
The travel reimbursement limit for client dinners is capped at $75 per person. 

Key Requirements:
- Alcoholic beverages must be itemized separately and require prior VP approval.
- Itemized tax invoices must be submitted via Expensify within 14 business days for all expenses over $50.

Sources:
[1] HR Employee Handbook (Rev 2026)
[2] Travel Policy FAQ
```
"""
    )

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: Minimal In-Memory RAG", divider="rainbow")
st.markdown("Lightweight Python script demonstrating document chunking, prompt augmentation, and grounded synthesis:")

st.code(
    '''from typing import List, Dict

# 1. Ingested Document Chunks
KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "chunk_id": "c1",
        "source": "Security Policy 2026",
        "content": "Mandatory password length is 16 characters. Hardware security keys (YubiKeys) are required for all production access."
    },
    {
        "chunk_id": "c2",
        "source": "Remote Work Guidelines",
        "content": "Employees may work remotely up to 3 days per week. Office attendance is mandatory on Tuesdays and Thursdays."
    }
]

def format_rag_prompt(query: str, retrieved_chunks: List[Dict[str, str]]) -> str:
    """
    Assembles a grounded RAG prompt with strict negative constraints
    and XML source encapsulation.
    """
    context_blocks = []
    for chunk in retrieved_chunks:
        block = f"<document id='{chunk['chunk_id']}' source='{chunk['source']}'>\\n{chunk['content']}\\n</document>"
        context_blocks.append(block)
        
    context_text = "\\n\\n".join(context_blocks)
    
    return f"""<system>
You are an enterprise knowledge assistant.
Answer the user question using ONLY the factual evidence inside <context>.
If the answer cannot be established from the text, reply: 'Information not available in context.'
Do not extrapolate, assume, or cite external training knowledge.
Include bracketed source citations (e.g. [Security Policy 2026]).
</system>

<context>
{context_text}
</context>

<user_query>
{query}
</user_query>
"""

# Example execution
prompt = format_rag_prompt("What are the mandatory in-office days?", [KNOWLEDGE_BASE[1]])
print(prompt)
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: What are the primary differences between Fine-Tuning an LLM and building a RAG pipeline?", expanded=True):
        st.markdown(
            """
- **Fine-Tuning**: Modifies model weights. It teaches the model **style, tone, specialized syntax, or vocabulary** (e.g. medical terminology or SQL dialects). However, it is poor for fact storage (weights still hallucinate), cannot easily unlearn outdated facts, and is expensive to re-train.
- **RAG (Retrieval-Augmented Generation)**: Freezes model weights and injects facts into the prompt context at inference time. It provides **100% auditable source citations**, instant knowledge updates without re-training, and strict access control per user.
- **Rule of Thumb**: *"Use Fine-Tuning to teach form; use RAG to teach facts."*
"""
        )
    with st.expander("Q2: What is the 'Chunk Size Dilemma' in RAG pipelines?"):
        st.markdown(
            """
- **Small Chunks (e.g. 100 tokens)**: High embedding precision (less noise), but they frequently fragment semantic context—splitting a sentence away from its subject or breaking a table row.
- **Large Chunks (e.g. 1,000 tokens)**: High context completeness, but the embedding vector becomes a generic blend of multiple topics, reducing retrieval precision. Additionally, large chunks consume excessive context window tokens.
- **Modern Solution**: **Hierarchical Chunking / Parent-Child Retrievers**: Index small 150-token child chunks for precision vector matching, but return the broader 800-token parent document chunk to the LLM context.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can RAG completely eliminate hallucinations?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, because the model only reads the provided document."*
- **Winning Response**: *"No. RAG dramatically reduces hallucinations, but models can still fabricate claims if: (1) the retrieved chunks contain conflicting or ambiguous facts, (2) the prompt lacks strict negative constraints forbidding extrapolation, or (3) the model's pre-trained parametric knowledge 'bleeds' through and overrides weak context. Eliminating hallucinations requires defensive prompt constraints, strict adherence scoring (e.g. Ragas faithfulness metrics), and automated citation checking."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**RAG (Retrieval-Augmented Generation)**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Augmenting LLM prompts with relevant external documents retrieved from a vector database to produce factually grounded responses.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Parametric vs Non-Parametric**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Parametric knowledge is frozen inside neural network weights; non-parametric knowledge lives in external searchable databases.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Parent-Child Retriever**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Searching dense small child chunks for retrieval precision while passing the surrounding parent chunk to the LLM for context completeness.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 16.1 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Facts via RAG**: Use RAG to inject dynamic, proprietary facts; avoid memorizing facts in weights.
2. **Lifecycle Steps**: Master chunking, dense embedding, vector search, and prompt augmentation.
3. **Defensive Prompting**: Mandate that models answer ONLY from provided context and state when info is missing.
4. **Source Citations**: Always enforce bracketed document attribution for verifiable answers.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand the architectural differences between Fine-Tuning and RAG.", value=True, key="chk_16_1_1")
    st.checkbox("I know the 8 core stages of the RAG lifecycle from ingestion to generation.", value=True, key="chk_16_1_2")
    st.checkbox("I can implement an in-memory RAG prompt formatter with XML source tagging.", value=True, key="chk_16_1_3")
    st.checkbox("I understand the chunk size trade-off and parent-child retrieval architectures.", value=True, key="chk_16_1_4")
