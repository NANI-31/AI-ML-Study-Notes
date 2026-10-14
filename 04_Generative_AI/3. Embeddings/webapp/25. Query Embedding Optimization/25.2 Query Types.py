import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="25.2 Query Types - Embeddings Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title & Overview
st.title("25.2 Query Types: Short, Long, Conversational, Ambiguous & Multi-Intent Queries")
st.caption("Chapter 25: Query Embedding Optimization | Domain: 3. Vector Embeddings")
st.markdown("""
In real-world retrieval applications, query text varies drastically in structure, length, intent, and clarity. 
A naive system that routes every query through the exact same embedding pipeline suffers catastrophic degradation:
* **Short queries** (1-3 words) lack contextual signal, triggering massive semantic drift across high-dimensional space.
* **Long queries** (50-300 words) dilute the core semantic intent, drowning out key search terms in conversational fluff.
* **Conversational queries** rely on previous chat turns with dangling pronouns (*"how do I install it?"*).
* **Ambiguous queries** span multiple unrelated semantic clusters (polysemy).
* **Multi-intent queries** contain multiple orthogonal tasks in a single sentence.

Understanding the five fundamental **Query Types** and deploying specialized pre-retrieval routing and optimization 
techniques is essential for production vector search and RAG architectures.
""")

st.divider()

# Section 1: Taxonomy of the 5 Query Types
st.header("1. Architectural Taxonomy of Query Types", divider="rainbow")

col_tax1, col_tax2 = col_tax1, col_tax2 = st.columns([1, 1])

with col_tax1:
    st.markdown("""
    ### The Five Core Query Types
    1. **Short Queries (1 - 3 Words)**:
       * *Characteristics*: High sparsity, missing syntactic structure, lexical intent.
       * *Example*: *"Kafka latency"*, *"reset password"*, *"403 forbidden"*.
       * *Risk*: The embedding vector is dominated by generic term vectors; misses specific problem context.
       * *Mitigation*: **Query Expansion** and **Entity Enriched Retrieval** (adding synonyms and domain tags).
    2. **Long Queries (Paragraphs & Error Logs)**:
       * *Characteristics*: Narrative-heavy, stack traces, multiple problem symptoms, conversational filler.
       * *Example*: *"I was trying to run our microservice yesterday on AWS ECS Fargate and suddenly during peak load the CPU spiked to 100% and then..."*
       * *Risk*: **Embedding Dilution**—the dense vector averages across 100+ tokens, smoothing out the actual error.
       * *Mitigation*: **Core Intent Extraction / Extractive Summarization** prior to embedding.
    3. **Conversational Queries (Chat Follow-ups)**:
       * *Characteristics*: Heavy reliance on pronouns (*"it"*, *"that"*, *"the second option"*), elliptical questions.
       * *Example*: *"How much does that one cost?"*, *"Can I deploy it in eu-west-1?"*
       * *Risk*: Zero semantic match because the noun is in previous conversation turns.
       * *Mitigation*: **Contextual Condensation / Standalone Query Rewriting**.
    4. **Ambiguous Queries (Polysemy & Underspecification)**:
       * *Characteristics*: Words with multiple distinct meanings depending on domain context.
       * *Example*: *"Apple returns"* (Finance stock vs. Retail refund), *"Transformer"* (Deep Learning vs. Electrical grid).
       * *Risk*: Retrieval lands in the wrong semantic cluster.
       * *Mitigation*: **Clarification Routing / Multi-Vector Diversification**.
    5. **Multi-Intent Queries (Compound Operations)**:
       * *Characteristics*: Query joins 2 or more distinct operations or subjects with conjunctions (*"and"*, *"compared to"*).
       * *Example*: *"Compare Redis vs MongoDB clustering and show how to backup both"*.
       * *Risk*: No single document chunk contains both answers.
       * *Mitigation*: **Query Decomposition & Parallel Branching**.
    """)

with col_tax2:
    st.markdown("### Adaptive Query Router Architecture")
    router_dot = """
    digraph QueryRouter {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        InQuery [label="Raw Inbound Query", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];
        
        Classifier [label="Query Type Classifier\\n(SLM / Token Count / Heuristics)", fillcolor="#FEF3C7", color="#F59E0B"];
        
        BranchShort [label="Short Query Branch\\nExpand & Add Entities", fillcolor="#DBEAFE"];
        BranchLong [label="Long Query Branch\\nExtract Core Intent", fillcolor="#E0E7FF"];
        BranchConv [label="Conversational Branch\\nResolve History & Coreference", fillcolor="#F3E8FF"];
        BranchAmbig [label="Ambiguous Branch\\nCluster Disambiguation", fillcolor="#CFFAFE"];
        BranchMulti [label="Multi-Intent Branch\\nDecompose to Sub-Queries", fillcolor="#DCFCE7"];

        VectorEngine [label="Optimized Vector Search\\nAccurate Manifold Alignment", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];

        InQuery -> Classifier;
        Classifier -> BranchShort [label="Tokens <= 4"];
        Classifier -> BranchLong [label="Tokens >= 40"];
        Classifier -> BranchConv [label="Has Pronouns"];
        Classifier -> BranchAmbig [label="High Entropy"];
        Classifier -> BranchMulti [label="Compound 'and'"];

        BranchShort -> VectorEngine;
        BranchLong -> VectorEngine;
        BranchConv -> VectorEngine;
        BranchAmbig -> VectorEngine;
        BranchMulti -> VectorEngine;
    }
    """
    st.graphviz_chart(router_dot, use_container_width=False)

st.markdown("""
| Query Type | Typical Length | Primary Failure Mode | Optimal Optimization Technique | Latency Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Short** | 1 - 4 words | Sparse vector, ambiguous target | Synonym expansion + hybrid BM25 boost | +15 - 30 ms |
| **Long** | 40 - 250+ words | Embedding vector dilution (washed out intent) | Extractive core intent summarization | +80 - 150 ms |
| **Conversational** | 3 - 10 words | Anaphoric reference failure (missing antecedent) | History condensation into standalone query | +100 - 180 ms |
| **Ambiguous** | 1 - 5 words | Multimodal distribution (wrong cluster retrieval) | Domain facet filtering or MMR diversification | +10 - 40 ms |
| **Multi-Intent** | 10 - 30 words | Partial retrieval (retrieves doc for intent A, misses B) | Parallel query decomposition + sub-query merge | +150 - 250 ms |
""")

st.divider()

# Section 2: Conversational Query Condensation & Coreference Resolution
st.header("2. Conversational Query Condensation & Coreference Resolution", divider="rainbow")

st.markdown("""
In conversational chat interfaces (e.g., enterprise support bots, Copilots), users almost never repeat the subject of their question.
Consider this typical conversation turn:

* **Turn 1 (User)**: *"What is the retention policy for S3 bucket access logs?"*
* **Turn 1 (Bot)**: *"S3 server access logs are retained according to the S3 Lifecycle configuration applied to the target bucket, typically defaulting to 90 days."*
* **Turn 2 (User)**: *"Can I replicate them to another region?"*
* **Turn 3 (User)**: *"How much does that cost per gigabyte?"*

If Turn 2 or Turn 3 is sent directly to an embedding model:
* Turn 2 embeds: *"Can I replicate them to another region?"* -> The embedding vector has no idea what *"them"* refers to!
* Turn 3 embeds: *"How much does that cost per gigabyte?"* -> The embedding matches generic cloud pricing tables rather than S3 log replication cross-region pricing!

### The Solution: Condensation into Standalone Search Queries
Before computing an embedding vector, we pass the **Chat History + Current Turn** into a fast instruction-tuned model 
to re-write the input into a **decontextualized standalone search query**.
""")

col_conv1, col_conv2 = col_conv1, col_conv2 = st.columns([1, 1])

with col_conv1:
    st.markdown("### Conversational Condensation Prompt Pattern")
    st.code("""
CONDENSE_QUERY_PROMPT = \"\"\"Given the chat history below and the user's latest follow-up question,
rephrase the follow-up question into an independent, standalone search query that contains all necessary
subjects, entity names, and specific context. 

DO NOT answer the question. ONLY return the reformulated standalone query.
If the query is already standalone, return it unchanged.

Chat History:
{chat_history}

User Follow-Up: {latest_query}

Standalone Query:\"\"\"
""", language="python")

with col_conv2:
    st.markdown("### Production Python Engine")
    st.code("""
from openai import OpenAI

client = OpenAI()

def condense_conversational_query(chat_history: list[dict], follow_up: str) -> str:
    formatted_history = "\\n".join([
        f"{m['role'].capitalize()}: {m['content']}" 
        for m in chat_history[-4:]  # Last 2 turns
    ])
    
    prompt = CONDENSE_QUERY_PROMPT.format(
        chat_history=formatted_history,
        latest_query=follow_up
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=60
    )
    return response.choices[0].message.content.strip()

# Turn 3 Input: "How much does that cost per gigabyte?"
# Result: "AWS S3 cross-region replication pricing per gigabyte for access logs"
""", language="python")

st.divider()

# Section 3: Handling Long Queries & Embedding Vector Dilution
st.header("3. Mitigating Embedding Dilution in Long Queries", divider="rainbow")

st.markdown("""
### The Problem of Embedding Dilution
Modern transformer embedding models (such as BERT, Sentence Transformers, or OpenAI `text-embedding-3`) use 
**mean pooling** or `[CLS]` token pooling over all input tokens.
When a user pastes a 200-word paragraph describing an issue:
* 80% of the tokens are narrative description (*"We noticed this during our sprint demo on Tuesday while running on our staging cluster..."*).
* Only 20% of the tokens contain the actionable technical problem (*"PostgreSQL deadlock on transactions touching table orders during foreign key cascade"*).

Because the dense vector is an average of the entire contextual sequence, **the 80% narrative noise dilutes the 20% critical signal**, 
pulling the query vector away from the technical troubleshooting documentation!
""")

col_dil1, col_dil2 = col_dil1, col_dil2 = st.columns([1, 1])

with col_dil1:
    st.markdown("### Embedding Vector Dilution Visualization")
    dilution_dot = """
    digraph Dilution {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_long {
            label="Long Raw Input (200 Words)"; style="dashed"; color="#EF4444";
            NoiseTokens [label="80% Narrative Noise:\\n'yesterday at 3pm, my team was testing...'", fillcolor="#FEE2E2"];
            CoreTokens [label="20% Technical Core:\\n'PostgreSQL foreign key cascade deadlock'", fillcolor="#DBEAFE", style="filled,bold"];
        }

        MeanPool [label="Transformer Mean Pooling\\nUniform vector average", fillcolor="#FEF3C7"];
        DilutedVec [label="Diluted Query Vector\\nFalls into generic narrative space\\n(Misses targeted tech docs)", fillcolor="#FCA5A5", color="#DC2626", style="filled,bold"];

        NoiseTokens -> MeanPool;
        CoreTokens -> MeanPool;
        MeanPool -> DilutedVec;
    }
    """
    st.graphviz_chart(dilution_dot, use_container_width=False)

with col_dil2:
    st.markdown("### Extractive Core Intent Distillation Pipeline")
    st.code("""
def extract_core_search_intent(long_query: str) -> str:
    \"\"\"Strips conversational noise and extracts the dense technical intent.\"\"\"
    prompt = f\"\"\"Extract the core technical problem, error signals, and system entities
from the text below. Eliminate all personal anecdotes, time references, and conversational filler.
Format as a concise, high-density search query (under 25 words).

User Text:
{long_query}

Extracted Core Intent:\"\"\"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=40
    )
    return response.choices[0].message.content.strip()
""", language="python")

st.divider()

# Section 4: Interactive Workbench — Query Type Analyzer & Optimization Sandbox
st.header("4. Interactive Workbench: Adaptive Query Router & Analyzer", divider="rainbow")

st.markdown("""
Test how an adaptive query router classifies real-world queries and applies specialized optimization logic. 
Select one of the benchmark test cases or type your own complex query.
""")

test_cases = {
    "Short Query (Sparse)": "redis evictions",
    "Long Query (Diluted)": "Hey team, we were observing intermittent 504 gateway timeout spikes on our Kubernetes ingress yesterday evening around 7:45 PM during our load testing run. The backend pods were running Java Spring Boot 3.2 on OpenJDK 17 and memory seemed stable at 65% utilization, but the Envoy proxy ingress logs reported connection reset by peer when routing traffic to the billing service cluster.",
    "Conversational Turn (Follow-up)": "What are the configuration flags to prevent it?",
    "Ambiguous Query (Polysemy)": "Apple security return policy",
    "Multi-Intent Query (Compound)": "Explain DynamoDB partition key best practices and how to configure global secondary indexes for multi-tenant isolation"
}

selected_case = st.selectbox(
    "Choose a Benchmark Query Case:",
    options=list(test_cases.keys()) + ["Custom Query"]
)

if selected_case == "Custom Query":
    active_query = st.text_area("Enter custom query:", value="Why did that service fail?")
else:
    active_query = st.text_area("Query to inspect:", value=test_cases[selected_case])

# Simulated Router Logic
def classify_query(q: str) -> Tuple[str, str, Dict[str, Any]]:
    tokens = q.strip().split()
    token_count = len(tokens)
    q_lower = q.lower()
    
    # Check conversational
    conv_pronouns = ["it", "that", "this", "them", "the previous", "those", "its"]
    has_pronoun = any(p in q_lower.split() for p in conv_pronouns)
    
    # Check multi-intent
    multi_indicators = [" and how to ", " and show ", " vs ", " compared to ", " along with "]
    is_multi = any(ind in q_lower for ind in multi_indicators)
    
    # Check ambiguous
    ambig_terms = ["apple", "transformer", "bank", "pool", "spark"]
    is_ambig = any(term in q_lower for term in ambig_terms) and token_count <= 6
    
    if token_count <= 3 and not has_pronoun:
        q_type = "Short Query (Sparse)"
        rec = "Inject domain synonyms, apply keyword expansion, and execute hybrid BM25 + dense vector search."
        optimized = f"{q} maxmemory-policy allkeys-lru volatile-lru eviction metrics diagnostic guide"
    elif has_pronoun:
        q_type = "Conversational Query"
        rec = "Condense chat history; resolve dangling pronouns to the historical antecedent."
        optimized = f"Kubernetes Envoy ingress 504 gateway timeout configuration flags to prevent connection reset by peer"
    elif is_multi:
        q_type = "Multi-Intent Query"
        rec = "Decompose into independent sub-queries, execute parallel retrieval, and merge with RRF."
        optimized = "Sub-Query 1: 'DynamoDB partition key design best practices'\nSub-Query 2: 'DynamoDB Global Secondary Index (GSI) multi-tenant isolation patterns'"
    elif is_ambig:
        q_type = "Ambiguous Query"
        rec = "Present disambiguation facets or apply MMR / multi-vector clustering diversification."
        optimized = "Branch A (Hardware Retail): 'Apple Store physical product return policy and refund timeframes'\nBranch B (Enterprise IT/MDM): 'Apple Device Enrollment Program (DEP) security return and compliance policy'"
    elif token_count > 25:
        q_type = "Long Query (Diluted)"
        rec = "Perform extractive intent summarization to strip narrative noise before computing embeddings."
        optimized = "Kubernetes Envoy ingress 504 gateway timeout connection reset by peer billing service"
    else:
        q_type = "Standard Technical Query"
        rec = "Direct dense vector retrieval with optional Bi-Encoder + Cross-Encoder reranking."
        optimized = q
        
    metrics = {
        "Token Count": token_count,
        "Char Count": len(q),
        "Pronoun Detected": has_pronoun,
        "Multi-Intent Triggered": is_multi
    }
    return q_type, rec, optimized, metrics

q_type, recommendation, optimized_output, metrics_data = classify_query(active_query)

col_out1, col_out2 = st.columns([1, 1])

with col_out1:
    st.subheader("Classification & Routing Decision")
    
    # Status badge
    if "Short" in q_type:
        st.warning(f"**Detected Type**: {q_type}")
    elif "Conversational" in q_type:
        st.info(f"**Detected Type**: {q_type}")
    elif "Long" in q_type:
        st.error(f"**Detected Type**: {q_type}")
    elif "Multi-Intent" in q_type:
        st.success(f"**Detected Type**: {q_type}")
    else:
        st.info(f"**Detected Type**: {q_type}")
        
    st.markdown(f"**Recommended Strategy**: {recommendation}")
    
    st.markdown("### Pre-Retrieval Optimization Result:")
    if "\n" in optimized_output:
        st.code(optimized_output, language="text")
    else:
        st.success(optimized_output)

with col_out2:
    st.subheader("Diagnostic Metrics & Routing Heuristics")
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Token Count", metrics_data["Token Count"])
    m_col2.metric("Character Count", metrics_data["Char Count"])
    
    st.markdown("""
    ### Routing Rules Triggered:
    * **Short Query Threshold**: Word count <= 3
    * **Long Query Threshold**: Word count >= 25 tokens
    * **Conversational Check**: Anaphoric pronoun regex detection (`it`, `that`, `this`, `them`)
    * **Multi-Intent Check**: Conjunction parsing (`and how to`, `compared to`, `vs`)
    """)
    
    st.markdown("### Impact Comparison:")
    st.markdown("""
    * **Without Pre-Retrieval Routing**: Query is embedded raw; accuracy drops by up to **42%** on conversational and long queries.
    * **With Pre-Retrieval Routing**: Query is normalized to target the exact document manifold; retrieval recall exceeds **94%**.
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: Why does mean pooling in dense embedding models degrade performance on long, paragraph-style queries?", expanded=True):
    st.markdown("""
    **Answer:**
    Dense embedding models like Sentence Transformers or BERT generate sequence representations by aggregating token vectors—most commonly via **mean pooling** across the sequence length. 
    In long, unstructured user queries (e.g., a customer ticket or forum post), the majority of tokens are conversational narrative (*"We tried running this yesterday...", "Our team noticed..."*) rather than specific technical concepts. 
    Because mean pooling assigns equal weight to every token vector, the overwhelming presence of non-discriminative narrative tokens pulls the resulting 1536-dimensional centroid away from the dense technical subspace, causing **Embedding Dilution**. 
    The resulting vector matches other generic narrative forum posts rather than the exact troubleshooting manual.
    """)

with st.expander("Q2: How do you handle conversational queries with anaphoric references in production RAG systems?"):
    st.markdown("""
    **Answer:**
    In production conversational RAG systems, handling anaphoric references (e.g., *"Can I replicate it?"*) involves a **Contextual Query Condensation** pre-retrieval stage:
    1. Maintain a sliding window of the last $N$ turns (typically the last 2 to 3 conversation turns).
    2. Pass the conversation history along with the latest user query to a small, low-latency instruction model (e.g., GPT-4o-mini or a fine-tuned 3B SLM).
    3. Prompt the model with a strict condensation instruction: *"Rewrite the user's latest follow-up question into an independent, standalone search query resolving all pronouns. DO NOT answer the question."*
    4. Only the rewritten standalone query is passed to the embedding model and vector database.
    5. The retrieved chunks and the original conversational history are then provided to the synthesis LLM for final generation.
    """)

with st.expander("Q3: What architectural pattern should be used for Multi-Intent queries like 'Compare Postgres and DynamoDB scaling and how to back up both'?"):
    st.markdown("""
    **Answer:**
    Multi-Intent queries cannot be satisfied by a single nearest neighbor vector lookup because no individual chunk in the vector database typically discusses all facets simultaneously. 
    The industry standard architectural pattern is **Query Decomposition & Parallel Branching**:
    1. **Decomposition**: An LLM decomposes the user prompt into $N$ atomic, single-intent sub-queries:
       * Sub-Query 1: *"PostgreSQL scaling architecture"*
       * Sub-Query 2: *"Amazon DynamoDB scaling and partitioning"*
       * Sub-Query 3: *"PostgreSQL backup and recovery procedures"*
       * Sub-Query 4: *"Amazon DynamoDB backup and point-in-time recovery (PITR)"*
    2. **Parallel Vector Retrieval**: Execute parallel vector queries against the vector database for each sub-query, retrieving top-5 chunks per branch.
    3. **Fusion & Deduplication**: Deduplicate chunks and rank them using Reciprocal Rank Fusion (RRF).
    4. **Joint Synthesis**: Inject the composite context into the LLM prompt to generate a cohesive comparative answer.
    """)

with st.expander("Q4: How do you handle ambiguous queries like 'Apple returns' in enterprise search?"):
    st.markdown("""
    **Answer:**
    Ambiguous queries exhibit a multimodal probability distribution in vector space. Two primary production techniques exist:
    * **Faceted Clarification Prompting**: If the confidence spread across top clusters is low, the search UI presents quick clarification chips: *"Did you mean: Apple Stock Financial Returns OR Apple Retail Store Returns?"*
    * **Maximal Marginal Relevance (MMR) Diversification**: Perform vector retrieval with MMR, which explicitly balances relevance with diversity. The top results will deliberately sample from both the financial cluster and the retail return policy cluster, ensuring the user sees their desired intent regardless of which cluster they meant.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Running LLM Condensation on Every Query Unconditionally**: Invoking an LLM to rewrite already standalone queries (e.g., *"Kubernetes crash loop backoff"*) wastes 150ms of latency and unnecessary API tokens. Use regex heuristic classifiers to bypass condensation for queries lacking pronouns.
* **Trap 2: Ignoring Domain Polysemy in Static Embeddings**: Assuming a general-purpose embedding model understands proprietary corporate acronyms (e.g., *"DEP"* = Device Enrollment Program vs. Data Execution Prevention). Without domain adaptation or expansion, retrieval precision plummets.
* **Trap 3: Truncating Long Queries Naively**: Slicing the first 50 tokens of a long user query often cuts off the actual error message located at the very end of the paragraph. Always use intent extraction or chunked matching rather than raw truncation.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Short Queries"):
        st.markdown("**Front**: Why do 1-2 word queries struggle in dense vector search?")
        st.divider()
        st.markdown("**Back**: High lexical sparsity and lack of syntactic context cause the embedding vector to wander toward generic word associations rather than specific technical targets.")

with card_cols[1]:
    with st.popover("Card 2: Long Queries"):
        st.markdown("**Front**: What is the primary failure mode of long paragraph queries?")
        st.divider()
        st.markdown("**Back**: **Embedding Vector Dilution**. Mean pooling averages narrative noise across hundreds of tokens, smoothing out the crucial technical signal.")

with card_cols[2]:
    with st.popover("Card 3: Conversational"):
        st.markdown("**Front**: What must be done before embedding a conversational chat follow-up?")
        st.divider()
        st.markdown("**Back**: **Coreference Resolution / Query Condensation** into a self-contained standalone query using recent chat turns.")

with card_cols[3]:
    with st.popover("Card 4: Multi-Intent"):
        st.markdown("**Front**: How should compound queries with multiple distinct operations be retrieved?")
        st.divider()
        st.markdown("**Back**: **Decompose** into atomic sub-queries, execute parallel vector searches, and fuse candidates using Reciprocal Rank Fusion (RRF).")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Query Types and Optimization strategies:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 5 core query types: Short, Long, Conversational, Ambiguous, Multi-Intent.", key="chk2_1")
    st.checkbox("I know how embedding dilution degrades long query representations during mean pooling.", key="chk2_2")
    st.checkbox("I can implement conversational query condensation with chat history.", key="chk2_3")
with c2:
    st.checkbox("I know when to apply parallel Query Decomposition vs. single-query expansion.", key="chk2_4")
    st.checkbox("I can design an adaptive query router that selects pre-retrieval optimization based on heuristics.", key="chk2_5")
    st.checkbox("I understand how MMR handles ambiguous polysemous search terms.", key="chk2_6")
