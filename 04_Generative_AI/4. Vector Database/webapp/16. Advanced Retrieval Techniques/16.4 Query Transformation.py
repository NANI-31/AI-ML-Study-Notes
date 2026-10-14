import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="16.4 Query Transformation - Vector DB Curriculum",
    page_icon="🔄",
    layout="wide"
)

# Title & Overview
st.title("16.4 Query Transformation: Rewriting, Expansion, Multi-Query, HyDE & Sub-Query Decomposition")
st.caption("Chapter 16: Advanced Retrieval Techniques | Domain: 4. Vector Database")
st.markdown("""
In real-world applications, user queries are notoriously terrible search vectors: 
they are often **ambiguous, colloquial, pronoun-heavy (*"How do I configure that?"*), or overly complex multi-hop inquiries**.

If you pass a raw, ambiguous query directly to a vector database, retrieval recall collapses. 
The database cannot match an index of authoritative reference manuals against conversational slang.

**Query Transformation** resolves this impedance mismatch by using an LLM to preprocess and reshape the query 
*before* vector embedding.

In this lesson, we explore the five core query transformation paradigms:
1. **Query Rewriting (Conversational Disambiguation)**: Replaces pronouns using conversational history.
2. **Query Expansion**: Enriches the inquiry with domain synonyms and technical vocabulary.
3. **Multi-Query Retrieval**: Generates multiple diverse phrasings and fuses candidate results with RRF.
4. **Hypothetical Document Embeddings (HyDE)**: Embeds a synthesized *answer* rather than the *question*.
5. **Sub-Query Decomposition**: Deconstructs multi-hop comparative questions into atomic sub-searches.
""")

st.divider()

# Section 1: The Five Query Transformation Paradigms Compared
st.header("1. The Five Query Transformation Paradigms Compared", divider="rainbow")

col_trans1, col_trans2 = col_trans1, col_trans2 = st.columns([1.1, 0.9])

with col_trans1:
    st.markdown("""
    ### Deep Dive into the 5 Transformation Strategies
    
    1. **Conversational Query Rewriting**:
       * *Problem*: User says: *"What about in production?"* (Meaningless without chat context).
       * *Transformation*: Rewrites to a standalone query: *"What are the deployment guidelines for Envoy mTLS in production environments?"*
    
    2. **Query Expansion**:
       * *Problem*: Query uses informal phrasing missing official terminology.
       * *Transformation*: Appends industry keywords: *"slow db"* -> *"slow database latency connection pool exhaustion PostgreSQL"*.
    
    3. **Multi-Query Retrieval**:
       * *Mechanism*: Generates 3 to 5 alternative formulations of the user's inquiry from different linguistic perspectives.
       * *Retrieval*: Queries the vector store for each variation and aggregates candidates via Reciprocal Rank Fusion.
    
    4. **Hypothetical Document Embeddings (HyDE - Gao et al., 2022)**:
       * *Breakthrough*: Questions and answers occupy different geometric sub-manifolds in vector space.
       * *Mechanism*: Instructs an LLM to generate a plausible hypothetical answer document. The system then embeds the *hypothetical document*, searching document-to-document space!
    
    5. **Sub-Query Decomposition**:
       * *Problem*: Multi-hop questions (*"Compare Kafka vs RabbitMQ message ordering guarantees"*).
       * *Transformation*: Splits into two independent queries: `Q1: Kafka message ordering` and `Q2: RabbitMQ message ordering`.
    """)

with col_trans2:
    st.markdown("### The HyDE Architectural Flow")
    hyde_dot = """
    digraph HyDEFlow {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="User Question\\n'How to configure Envoy mTLS?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        LLMGen [label="1. Fast Zero-Shot LLM\\nGenerates Hypothetical Answer\\n'Envoy mTLS requires certs...'", fillcolor="#FEF3C7", color="#D97706"];

        HypoDoc [label="Hypothetical Passage (HypoDoc)\\nContains authoritative domain syntax", fillcolor="#EDE9FE"];

        EmbModel [label="2. Embedding Model\\nEmbeds HypoDoc (Doc-to-Doc)", fillcolor="#D1FAE5", color="#059669"];

        VectorStore [label="3. ChromaDB Vector Store\\nSearches Real Ground-Truth Index", fillcolor="#C7D2FE", color="#1E40AF", style="filled,bold"];

        RealPassages [label="Verified Factual Passages\\n(Highest Semantic Overlap)", fillcolor="#A7F3D0", style="filled,bold"];

        UserQ -> LLMGen;
        LLMGen -> HypoDoc;
        HypoDoc -> EmbModel;
        EmbModel -> VectorStore;
        VectorStore -> RealPassages;
    }
    """
    st.graphviz_chart(hyde_dot, use_container_width=False)

st.divider()

# Section 2: Code Patterns for Production Query Transformers
st.header("2. Code Patterns for Production Query Transformers", divider="rainbow")

col_code_q1, col_code_q2 = col_code_q1, col_code_q2 = st.columns([1, 1])

with col_code_q1:
    st.markdown("### 1. Hypothetical Document Embeddings (HyDE)")
    st.code("""from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt LLM to draft a hypothetical answer
hyde_prompt = ChatPromptTemplate.from_template(\"\"\"
Please write a short, highly technical documentation passage answering this question.
Do not worry if the facts are entirely correct; focus on technical terminology and style.

Question: {question}
Passage:\"\"\")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
hyde_generator = hyde_prompt | llm | StrOutputParser()

# 2. Generate hypothetical document
user_question = "What is the Envoy ingress certificate TTL?"
hypothetical_doc = hyde_generator.invoke({"question": user_question})

# 3. Search vector store using the hypothetical text
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
results = vectorstore.similarity_search(hypothetical_doc, k=3)
""", language="python")

with col_code_q2:
    st.markdown("### 2. Multi-Query Generation & RRF Pipeline")
    st.code("""from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

multi_query_prompt = ChatPromptTemplate.from_template(\"\"\"
You are an AI language model assistant. Your task is to generate 3 different 
versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user 
overcome some of the limitations of the distance-based similarity search.

Provide these alternative questions separated by newlines.
Original question: {question}\"\"\")

multi_query_chain = multi_query_prompt | llm | StrOutputParser()

# Generates:
# 1. Envoy mTLS certificate configuration steps
# 2. Setting up mutual TLS on Envoy edge proxy listeners
# 3. Envoy ingress TLS client certificate verification
queries = multi_query_chain.invoke({"question": user_question}).splitlines()
""", language="python")

st.divider()

# Section 3: Sub-Query Decomposition for Multi-Hop Inquiries
st.header("3. Sub-Query Decomposition for Multi-Hop Inquiries", divider="rainbow")

st.markdown("""
When a user asks a complex multi-hop or comparative question, vector search fails because no single document 
compares both subjects simultaneously. 

**Sub-Query Decomposition** breaks the inquiry into autonomous sub-tasks:
""")

col_dec1, col_dec2 = col_dec1, col_dec2 = st.columns([1, 1])

with col_dec1:
    st.markdown("""
    ### The Multi-Hop Dilemma
    * **User Prompt**: *"Compare the data retention mandates of SOC2 vs GDPR for customer audit logs."*
    * **Single Search Failure**: The database has a 20-page SOC2 document and a 30-page GDPR document. No single chunk contains both terms in detail. A single vector search retrieves partial fragments from both, missing key comparative metrics.
    """)

with col_dec2:
    st.markdown("""
    ### The Decomposed Retrieval Plan
    1. **Decomposer LLM**:
       * Sub-Query 1: `"What is the data retention requirement for customer audit logs under SOC2 compliance?"`
       * Sub-Query 2: `"What are the data retention and erasure mandates for customer audit logs under GDPR Article 33?"`
    2. **Parallel Retrieval**: Executes two distinct vector searches in parallel.
    3. **Synthesizer LLM**: Takes both retrieved context sets and writes the comprehensive comparison!
    """)

st.divider()

# Section 4: Interactive Workbench: Live Query Transformation Simulator
st.header("4. Interactive Workbench: Live Query Transformation Simulator", divider="rainbow")

st.markdown("""
Select a raw, ambiguous user inquiry and test how different transformation strategies 
reshape it before vector database ingestion.
""")

RAW_QUERY_SAMPLES = {
    "Ambiguous / Conversational": "How do I configure that in production?",
    "Slang / Missing Technical Terms": "My database is super laggy and keeps dying under load.",
    "Complex Comparative Multi-Hop": "Compare the disaster recovery RTO requirements of PostgreSQL vs Redis in our architecture."
}

col_sim_q, col_sim_strat = st.columns([1.1, 0.9])

with col_sim_q:
    st.subheader("1. Raw User Input")
    q_choice = st.selectbox("Select Sample Input:", list(RAW_QUERY_SAMPLES.keys()))
    active_raw_q = st.text_input("Raw Input String:", value=RAW_QUERY_SAMPLES[q_choice])

with col_sim_strat:
    st.subheader("2. Transformation Paradigm")
    chosen_strategy = st.radio(
        "Choose Strategy:",
        [
            "Hypothetical Document Embeddings (HyDE)",
            "Multi-Query Generation (3 Perspectives)",
            "Conversational Query Rewriting",
            "Sub-Query Decomposition"
        ]
    )

st.divider()

col_trans_out, col_trans_impact = st.columns([1.2, 0.8])

with col_trans_out:
    st.subheader("3. Transformed Search Artifacts")
    
    if "HyDE" in chosen_strategy:
        st.info("🧠 **HyDE Synthesized Hypothetical Document** (Passage sent to vector embedding):")
        st.code("""To configure high-availability production ingress with Envoy proxy, deploy the proxy as a Kubernetes DaemonSet. 
Ensure client certificate verification is enforced by setting require_client_certificate: true in the downstream TLS listener filter chain. 
Certificates must be mounted into /etc/envoy/certs with automated rotation handled via SPIFFE/SPIRE agents communicating over local UNIX domain sockets.""", language="text")
        st.caption("Notice: Uses authoritative technical vocabulary (DaemonSet, TLS listener filter chain, SPIFFE/SPIRE).")
        
    elif "Multi-Query" in chosen_strategy:
        st.info("🔀 **Generated Alternative Perspectives** (Sent to parallel vector search):")
        st.markdown("""
        1. **Perspective A (Direct)**: *"Production deployment guidelines for Envoy proxy ingress gateways."*
        2. **Perspective B (Architectural)**: *"High-availability configuration and listener settings for enterprise Envoy proxies."*
        3. **Perspective C (Security-Focused)**: *"Envoy proxy production checklist: TLS termination, mutual TLS, and certificate management."*
        """)
        st.caption("Results from all 3 queries will be merged using Reciprocal Rank Fusion (RRF).")
        
    elif "Conversational" in chosen_strategy:
        st.info("📝 **Resolved Standalone Question** (Pronoun and history disambiguated):")
        st.code("How do I configure Envoy proxy mutual TLS (mTLS) client certificate verification in a Kubernetes production cluster?", language="text")
        st.caption("Replaced ambiguous 'that' with 'Envoy proxy mutual TLS' derived from conversation history.")
        
    else:  # Decomposition
        st.info("✂️ **Decomposed Independent Sub-Queries**:")
        st.markdown("""
        * **Sub-Task 1**: *"What is the disaster recovery Recovery Time Objective (RTO) for production PostgreSQL multi-region failover?"*
        * **Sub-Task 2**: *"What is the disaster recovery Recovery Time Objective (RTO) for production Redis cache clusters?"*
        """)
        st.caption("Executed as two independent parallel vector searches.")

with col_trans_impact:
    st.subheader("Retrieval Impact Analysis")
    st.metric("Vocabulary Density Gain", "+350%", delta="Dense Domain Terminology")
    st.metric("Expected Recall Improvement", "+28% Recall@5", delta="Sub-Manifold Alignment")
    st.success("✅ Impedance mismatch resolved: Search vector aligns with reference document embeddings.")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the Question-Answer vector distribution mismatch, and how does HyDE solve it?"):
    st.markdown("""
    **Answer**:
    In high-dimensional embedding spaces:
    * **User Questions** are short, interrogative, and express an information deficit (*"How does certificate rotation work?"*).
    * **Document Passages** are long, authoritative, and express factual statements (*"The SPIRE agent rotates X.509 certs every 60 minutes."*).
    * Because questions and answers occupy different geometric sub-manifolds, their cosine similarity is often lower than desired.
    * **HyDE (Hypothetical Document Embeddings)** uses an LLM to generate a synthetic *answer passage*. Even if the hypothetical facts are slightly wrong, the **document style, grammar, and vocabulary** match the real corpus. Embedding the hypothetical passage allows searching in **document-to-document space**, drastically boosting recall.
    """)

with st.expander("Q2: When can HyDE fail or hurt retrieval performance?"):
    st.markdown("""
    **Answer**:
    HyDE fails in **Ultra-Niche or Proprietary Zero-Knowledge Scenarios**:
    * If a user asks about an obscure internal company tool (e.g. *"How do I configure internal tool BobcatCLI v4?"*), an off-the-shelf LLM knows nothing about BobcatCLI.
    * The LLM will hallucinate facts about real bobcats (animals) or generic cloud CLI tools.
    * Embedding this hallucinated garbage will lead the vector database to search for feline biology or AWS CLI, completely poisoning retrieval!
    * **Rule**: Only use HyDE for broad technical, medical, or legal domains where the LLM's pre-trained parametric memory understands the domain vocabulary.
    """)

with st.expander("Q3: How does Multi-Query Retrieval differ from Query Expansion?"):
    st.markdown("""
    **Answer**:
    * **Query Expansion**: Modifies a single query by appending synonyms or related keywords (e.g. *"db latency pool slow"*). It results in a single, crowded search vector.
    * **Multi-Query Retrieval**: Generates **distinct complete sentences** representing different conceptual viewpoints (e.g. an architectural question, a debugging question, and a security question). It executes multiple independent vector searches and fuses the results using Reciprocal Rank Fusion (RRF), exploring multiple distinct neighborhoods in the vector index.
    """)

with st.expander("Q4: What is the latency and cost overhead of Query Transformation pipelines?"):
    st.markdown("""
    **Answer**:
    Query transformation introduces an extra LLM call *before* vector search:
    * **Latency**: Calling an LLM adds **250ms to 600ms** to total query latency.
    * **Cost**: Consumes 50-150 prompt tokens per user question.
    * **Production Optimization**: Use small, high-throughput models (e.g. Claude 3.5 Haiku or GPT-4o-mini) with `temperature=0` and low `max_tokens` (e.g. 100 tokens), or use an intent classifier to trigger HyDE/Multi-Query only on complex questions while passing simple queries directly to vector search.
    """)

with st.expander("Q5: How does Conversational Query Rewriting maintain multi-turn chat coherence?"):
    st.markdown("""
    **Answer**:
    In conversational AI:
    * User Turn 1: *"How do I configure PostgreSQL WAL archiving?"*
    * User Turn 2: *"What is the retention period for that?"*
    * User Turn 3: *"Can I restore it to an alternate region?"*
    A vector search for Turn 3 (*"Can I restore it to an alternate region?"*) fails because the vector database does not know what *"it"* is.
    * **The Query Rewriter**: Takes the full chat history and Turn 3, outputting: *"Can PostgreSQL continuous WAL archives be restored to an alternate cloud region?"* 
    This produces a standalone vector query that retrieves the exact runbook.
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Blindly Running HyDE on Internal Proprietary Tools**  
    Asking HyDE to hallucinate documentation for proprietary company codebases produces absurd hallucinations that ruin vector search. Limit HyDE to general tech stacks.
    """)
    st.warning("""
    **Trap 2: Using Slow Large LLMs for Query Rewriting**  
    Using a massive 70B or slow reasoning model to rewrite queries adds 2-3 seconds of latency before retrieval even starts. Use fast small models (GPT-4o-mini / Haiku).
    """)

with col_trap2:
    st.error("""
    **Trap 3: Omitting RRF When Using Multi-Query**  
    Taking the top 5 chunks from 3 queries and concatenating them without deduplication passes redundant duplicate chunks to the prompt. Always merge multi-query hits with RRF.
    """)
    st.warning("""
    **Trap 4: Over-Transforming Simple Direct Queries**  
    Rewriting a simple, exact query like *"Error code 504"* into an elaborate paragraph can dilute the specific keyword signal. Route simple queries directly to hybrid search.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is HyDE?"):
        st.markdown("""
        **Hypothetical Document Embeddings**:  
        Generates a synthetic answer with an LLM and embeds that passage, searching document-to-document vector space.
        """)
    with st.expander("🗂️ Flashcard: What is Conversational Query Rewriting?"):
        st.markdown("""
        **Chat Disambiguation**:  
        Converts multi-turn chat questions with pronouns (*"How do I setup that?"*) into standalone self-contained search vectors.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: How does Multi-Query work?"):
        st.markdown("""
        **Perspective Exploration**:  
        Generates 3-5 distinct question phrasings, queries the vector database for each, and fuses candidates with RRF.
        """)
    with st.expander("🗂️ Flashcard: When does HyDE fail?"):
        st.markdown("""
        **Proprietary Zero-Knowledge**:  
        Fails when questions involve internal company tools unknown to the LLM, leading to hallucinated noise.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the Question-Answer vector distribution mismatch and how HyDE solves it.", value=True)
chk2 = st.checkbox("I know when HyDE succeeds and when it fails on proprietary zero-knowledge domains.", value=True)
chk3 = st.checkbox("I can implement Multi-Query generation and merge candidate rankings via RRF in LangChain.", value=True)
chk4 = st.checkbox("I know how Conversational Query Rewriting resolves pronouns across multi-turn chats.", value=True)
chk5 = st.checkbox("I understand how Sub-Query Decomposition breaks comparative questions into atomic searches.", value=True)
chk6 = st.checkbox("I can design an adaptive query transformation router that balances latency, cost, and recall.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have completely mastered Chapter 16: Advanced Retrieval Techniques and are ready for Chapter 17: Reranking!")
