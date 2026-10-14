import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="25.1 Query Transformation - Embeddings Curriculum",
    page_icon="🔄",
    layout="wide"
)

# Title & Overview
st.title("25.1 Query Transformation: Rewriting, Expansion, Decomposition, Multi-Query & HyDE")
st.caption("Chapter 25: Query Embedding Optimization | Domain: 3. Vector Embeddings")
st.markdown("""
In real-world retrieval-augmented systems, user queries are notoriously imperfect. Users provide ambiguous phrases, 
elliptical phrasing, colloquial slang, or deeply nested compound questions. Passing a raw, untreated user query directly into a 
vector embedding model often leads to catastrophic retrieval failure due to **Query-Document Semantic Asymmetry**:
* A **user query** is typically short, interrogative, vague, and lacks technical terminology (e.g., *"How do I fix memory error in pods?"*).
* An **authoritative document** is long, declarative, formal, and loaded with specific technical terms (e.g., *"Kubernetes OOMKilled exit code 137 container termination and cgroup limit remediation"*).

**Query Transformation** is the disciplined practice of preprocessing, restructuring, and augmenting the user's input before 
computing embedding vectors. By transforming queries into rich, aligned, or multi-perspective representations, 
we dramatically increase retrieval recall and rank precision across high-dimensional vector spaces.
""")

st.divider()

# Section 1: The Core Query Transformation Strategies
st.header("1. The Five Core Query Transformation Paradigms", divider="rainbow")

col_trans1, col_trans2 = st.columns([1, 1])

with col_trans1:
    st.markdown("""
    ### Taxonomy of Query Transformations
    1. **Query Rewriting (Paraphrasing & Normalization)**:
       * Resolves pronoun ambiguities, normalizes typos and slang, and frames questions into self-contained declarative statements.
       * *Example*: *"what's the cost for that enterprise tier?"* -> *"What is the enterprise subscription pricing tier for Acme Cloud?"*
    2. **Query Expansion (Lexical & Conceptual Enrichment)**:
       * Injects synonyms, industry jargon, and related keywords into the query to activate dense embedding dimensions that the raw query missed.
       * *Example*: *"cardiac arrest"* -> *"cardiac arrest myocardial infarction heart attack arrhythmia resuscitation"*
    3. **Query Decomposition (Breakdown into Sub-Queries)**:
       * Takes complex, multi-hop, or multi-faceted queries and decomposes them into independent atomic questions that retrieve targeted subsets.
       * *Example*: *"Compare Redis vs Memcached latency and persistence"* -> Sub-query 1: *"Redis latency and disk persistence architecture"*, Sub-query 2: *"Memcached latency and in-memory persistence limitations"*.
    4. **Multi-Query Retrieval (Perspective Diversity)**:
       * Generates 3 to 5 diverse syntactic and conceptual variants of the original prompt, runs vector search for each, and fuses candidates with Reciprocal Rank Fusion (RRF).
    5. **Hypothetical Document Embeddings (HyDE)**:
       * Instructs an LLM to hallucinate a plausible answer document. Embeds the *hypothetical answer* rather than the *question*!
    """)

with col_trans2:
    st.markdown("### Query Transformation Pipeline Architecture")
    pipeline_dot = """
    digraph QueryTransformation {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawQuery [label="Raw User Query\\n'Why is my pod crashing with 137?'", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];
        
        Router [label="Query Transformation Router\\nDetects Complexity & Intent", fillcolor="#FEF3C7", color="#F59E0B"];
        
        Rewriter [label="1. Query Rewriter\\nClarify & Disambiguate", fillcolor="#DBEAFE"];
        Expander [label="2. Query Expander\\nSynonyms & Technical Jargon", fillcolor="#E0E7FF"];
        Decomposer [label="3. Query Decomposer\\nSub-queries for Multi-hop", fillcolor="#F3E8FF"];
        HyDE [label="4. HyDE Generator\\nPlausible Answer Document", fillcolor="#DCFCE7"];

        FusedEmbed [label="Transformed Embeddings Engine\\nGenerates Dense Query Vector(s)", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];
        VectorDB [label="Vector Database Search\\nHigh-Recall Retrieval", fillcolor="#CFFAFE", color="#06B6D4"];

        RawQuery -> Router;
        Router -> Rewriter;
        Router -> Expander;
        Router -> Decomposer;
        Router -> HyDE;

        Rewriter -> FusedEmbed;
        Expander -> FusedEmbed;
        Decomposer -> FusedEmbed;
        HyDE -> FusedEmbed;

        FusedEmbed -> VectorDB;
    }
    """
    st.graphviz_chart(pipeline_dot, use_container_width=False)

st.markdown("""
| Transformation Method | Primary Problem Solved | Computational Cost | Latency Overhead | Failure Mode / Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Query Rewriting** | Ambiguity, slang, pronouns | Low (Fast LLM / SLM) | 80 - 150 ms | LLM hallucinations shifting user intent |
| **Query Expansion** | Vocabulary mismatch, missing terms | Near Zero (Rule/BM25) to Low | 10 - 100 ms | Concept drift diluting specific precision |
| **Query Decomposition** | Multi-hop reasoning, compound questions | Medium (Structured JSON LLM) | 150 - 300 ms | Fragmented context losing inter-part relation |
| **Multi-Query (RRF)** | Synactic variance, fragile embedding boundaries | Medium (N queries + N vector searches) | 120 - 250 ms | Increased vector DB QPS load (N-fold) |
| **HyDE** | Query-Document structural asymmetry | Medium-High (LLM response generation) | 200 - 500 ms | False fact hallucination steering retrieval astray |
""")

st.divider()

# Section 2: Deep Dive into HyDE (Hypothetical Document Embeddings)
st.header("2. Deep Dive: Hypothetical Document Embeddings (HyDE)", divider="rainbow")

st.markdown("""
### Why HyDE Changes the Retrieval Game
In standard dense retrieval, we measure the cosine similarity between a **Question Vector** and a **Passage Vector**:
* Questions live in an interrogative linguistic subspace (*"Who", "What", "Why", "Can I..."*).
* Relevant passages live in an informative declarative subspace (*"Under Section 4.2...", "The system automatically...", "To configure..."*).
* Even with bi-encoder fine-tuning (e.g., contrastive learning with positive pairs), questions and documents rarely align perfectly in 1536-dimensional space.

**Luyu Gao et al. (2022)** introduced **HyDE (Hypothetical Document Embeddings)** to solve this:
1. Feed the user prompt into an ungrounded LLM with the instruction: *"Write a detailed, technical document that answers this question."*
2. Even if the LLM hallucinates specific facts (dates, numbers, names), **the semantic style, technical terminology, and document structure are identical to real documents in your corpus**!
3. Compute the embedding of the **hallucinated document**.
4. The hallucinated document vector sits directly inside the **informative document manifold**, producing vastly superior nearest neighbor retrieval.
""")

col_hyde1, col_hyde2 = col_hyde1, col_hyde2 = st.columns([1, 1])

with col_hyde1:
    st.markdown("### Asymmetric Manifold vs. HyDE Manifold")
    hyde_chart = """
    digraph HyDEManifold {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_query_space {
            label="Question Space (Sparse/Interrogative)"; style="dashed"; color="#EF4444";
            RawQ [label="Raw Query:\\n'What causes OOM 137?'", fillcolor="#FEE2E2"];
        }

        subgraph cluster_doc_space {
            label="Document Space (Dense/Declarative)"; style="dashed"; color="#10B981";
            HyDEDoc [label="HyDE Hypothetical Doc:\\n'Kubernetes exit code 137 indicates\\nOOMKilled by Linux kernel cgroup...'\\n(Generated via LLM)", fillcolor="#DCFCE7", color="#10B981", style="filled,bold"];
            RealDoc1 [label="Corpus Chunk A:\\n'Linux OOM killer triggers signal 9\\nexit code 128+9=137...'", fillcolor="#DBEAFE"];
            RealDoc2 [label="Corpus Chunk B:\\n'Kubernetes container memory limits\\nand OOM remediation steps...'", fillcolor="#DBEAFE"];
        }

        RawQ -> HyDEDoc [label="LLM Hallucination\\nStep (Zero Grounding)", color="#F59E0B", style="dashed"];
        HyDEDoc -> RealDoc1 [label="High Cosine Similarity\\n(Same Document Manifold!)", color="#10B981", style="bold"];
        HyDEDoc -> RealDoc2 [label="High Cosine Similarity", color="#10B981"];
        RawQ -> RealDoc1 [label="Weak Similarity\\n(Syntax Mismatch)", color="#EF4444", style="dotted"];
    }
    """
    st.graphviz_chart(hyde_chart, use_container_width=False)

with col_hyde2:
    st.markdown("### Python Implementation: Production HyDE Pipeline")
    st.code("""
from openai import OpenAI
import numpy as np

client = OpenAI()

def generate_hypothetical_document(query: str) -> str:
    \"\"\"Generates a plausible answer document using an LLM.\"\"\"
    prompt = f\"\"\"Please write a technical passage answering the question below.
Include technical terminology, explanations, and key concepts that would appear 
in an official engineering or product manual.

Question: {query}
Technical Passage:\"\"\"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=250
    )
    return response.choices[0].message.content

def get_embedding(text: str) -> np.ndarray:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    vec = np.array(res.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)

def hyde_retrieval(query: str, vector_store, top_k: int = 5):
    # 1. Hallucinate hypothetical document
    hypo_doc = generate_hypothetical_document(query)
    
    # 2. Embed the hypothetical document, NOT the raw query
    hypo_vec = get_embedding(hypo_doc)
    
    # 3. Perform vector search in document manifold
    results = vector_store.search(hypo_vec, k=top_k)
    return results, hypo_doc
    """, language="python")

st.divider()

# Section 3: Multi-Query Retrieval & Reciprocal Rank Fusion (RRF)
st.header("3. Multi-Query Retrieval & Reciprocal Rank Fusion (RRF)", divider="rainbow")

st.markdown("""
When users query a search engine, subtle word choices create artificial variance in vector embeddings.
For instance:
* Query A: *"How to reset postgres user password"*
* Query B: *"PostgreSQL alter user credentials command"*
* Query C: *"Recover lost admin authentication in Postgres"*

Even though all three represent the exact same semantic intent, their embedding vectors will cluster around slightly different points in high-dimensional space.
**Multi-Query Retrieval** circumvents this fragility:
1. An LLM generates 3 to 4 distinct query variants exploring different angles, synonyms, and formal structures.
2. The system executes parallel vector searches for each query variant.
3. The ranked result lists are merged using **Reciprocal Rank Fusion (RRF)**.
""")

col_rrf1, col_rrf2 = st.columns([1, 1])

with col_rrf1:
    st.markdown("""
    ### Reciprocal Rank Fusion (RRF) Formula
    Instead of averaging raw cosine scores (which can be calibrated differently across queries), RRF scores items strictly based on their rank positions in each list:
    
    ```text
    RRF_Score(doc) = Sum_{q in Queries} [ 1 / (k + Rank(doc, q)) ]
    ```
    
    Where:
    * `Rank(doc, q)` is the 1-based index of `doc` in the search results for query variant `q`.
    * `k` is a smoothing constant (standard industry default: `k = 60`).
    * If a document is absent from a query's result list, its reciprocal rank for that query is 0.
    
    **Why RRF is Robust**:
    * Documents appearing near the top across multiple queries receive an exponential boost.
    * A document ranked #1 in one query and #2 in another heavily outscores a document that ranked #1 in only one query and was omitted by all others.
    """)

with col_rrf2:
    st.markdown("### Multi-Query RRF Fusion Architecture")
    rrf_dot = """
    digraph MultiQueryRRF {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="Original Query\\n'Postgres authentication recovery'", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];
        
        LLMGen [label="LLM Multi-Query Generator\\nPrompt: Generate 3 diverse perspectives", fillcolor="#FEF3C7"];

        Q1 [label="Query 1:\\n'Reset postgres user password'", fillcolor="#DBEAFE"];
        Q2 [label="Query 2:\\n'PostgreSQL ALTER USER command'", fillcolor="#DBEAFE"];
        Q3 [label="Query 3:\\n'pg_hba.conf trust authentication'", fillcolor="#DBEAFE"];

        V1 [label="Vector Search 1\\nTop 10 chunks", fillcolor="#E0E7FF"];
        V2 [label="Vector Search 2\\nTop 10 chunks", fillcolor="#E0E7FF"];
        V3 [label="Vector Search 3\\nTop 10 chunks", fillcolor="#E0E7FF"];

        RRFNode [label="Reciprocal Rank Fusion (k=60)\\nCombines & Deduplicates", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];
        FinalList [label="Final Unified Top 5 Chunks\\nHigh Recall & High Stability", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        UserQ -> LLMGen;
        LLMGen -> Q1; LLMGen -> Q2; LLMGen -> Q3;
        Q1 -> V1; Q2 -> V2; Q3 -> V3;
        V1 -> RRFNode; V2 -> RRFNode; V3 -> RRFNode;
        RRFNode -> FinalList;
    }
    """
    st.graphviz_chart(rrf_dot, use_container_width=False)

st.divider()

# Section 4: Interactive Workbench — Query Transformation Simulator & HyDE Sandbox
st.header("4. Interactive Workbench: Transformation Engine & HyDE Simulator", divider="rainbow")

st.markdown("""
Test real query transformation pipelines live. Choose a typical imperfect user query or enter your own to see 
how Query Rewriting, Query Expansion, Query Decomposition, Multi-Query Generation, and HyDE reformulate the input.
""")

sample_queries = {
    "Vague Kubernetes Error": "Why is my pod crashing with 137?",
    "Compound Multi-Hop Question": "Compare Postgres and Snowflake for streaming ingestion and pricing",
    "Casual Non-Technical User": "How do I make my python script stop freezing when reading files?",
    "Implicit Anaphoric Query": "What is the cancellation policy for that tier and how do I downgrade?"
}

selected_sample = st.selectbox(
    "Choose a Preset Imperfect Query or Custom:",
    options=list(sample_queries.keys()) + ["Custom Query"]
)

if selected_sample == "Custom Query":
    user_input_query = st.text_input("Enter custom query:", value="Why does my vector search return irrelevant chunks?")
else:
    user_input_query = st.text_input("Enter custom query:", value=sample_queries[selected_sample])

col_bench1, col_bench2 = st.columns([1, 1])

# Simulation transformation functions
def simulate_rewriting(q: str) -> str:
    if "137" in q:
        return "What causes Kubernetes Pod exit code 137 (OOMKilled) and how to configure resource memory limits?"
    elif "Postgres and Snowflake" in q:
        return "Comprehensive architectural comparison between PostgreSQL and Snowflake regarding real-time streaming data ingestion throughput and monthly storage and compute pricing models."
    elif "freezing" in q:
        return "How to resolve Python GIL deadlocks and blocking I/O bottlenecks when reading large files asynchronously using asyncio or multiprocessing?"
    elif "cancellation" in q:
        return "What are the contractual cancellation policies and steps required to downgrade an active enterprise SaaS subscription?"
    else:
        return f"Explicit, self-contained formulation of technical query: '{q}', clarifying context, system constraints, and expected output."

def simulate_expansion(q: str) -> List[str]:
    if "137" in q:
        return [q, "Keywords: OOMKilled, Linux cgroup memory limit, cgroup v2, SIGKILL 9, exit code 137, requests.memory, limits.memory"]
    elif "Postgres and Snowflake" in q:
        return [q, "Keywords: PostgreSQL logical replication, Snowflake Snowpipe streaming, micro-batching, OLAP vs OLTP, compute credit costs"]
    elif "freezing" in q:
        return [q, "Keywords: Python blocking IO, GIL lock, threadpool executor, memory mapped files mmap, chunked streaming, buffering"]
    else:
        return [q, f"Keywords: domain taxonomy, canonical vocabulary, formal technical documentation phrases for '{q}'"]

def simulate_decomposition(q: str) -> List[str]:
    if "Postgres and Snowflake" in q:
        return [
            "Sub-Query 1: What is PostgreSQL performance and architecture for real-time streaming ingestion?",
            "Sub-Query 2: What is Snowflake Snowpipe performance and architecture for streaming data ingestion?",
            "Sub-Query 3: What are PostgreSQL hosting, storage, and compute pricing models?",
            "Sub-Query 4: What are Snowflake storage per TB and compute warehouse credit pricing structures?"
        ]
    elif "137" in q:
        return [
            "Sub-Query 1: What does Linux exit code 137 mean in container runtime environments?",
            "Sub-Query 2: How does Kubernetes kubelet enforce cgroup memory limits to trigger OOMKilled?",
            "Sub-Query 3: How to diagnose and monitor pod memory consumption with metrics-server?"
        ]
    else:
        return [
            f"Sub-Query 1: Foundational architectural concepts behind: {q}",
            f"Sub-Query 2: Implementation remediation steps and code patterns for: {q}",
            f"Sub-Query 3: Validation and monitoring considerations for: {q}"
        ]

def simulate_multi_query(q: str) -> List[str]:
    return [
        f"Perspective 1 (Engineering Root-Cause): 'Technical root causes, diagnostic logs, and architectural mechanisms for {q}'",
        f"Perspective 2 (Actionable Resolution): 'Step-by-step fix, configuration code snippet, and best practices for {q}'",
        f"Perspective 3 (Alternative Terminology): 'Industry standard nomenclature, error definitions, and official manual documentation for {q}'"
    ]

def simulate_hyde(q: str) -> str:
    if "137" in q:
        return """In Kubernetes, container exit code 137 indicates that the process was terminated by the Linux kernel Out-Of-Memory (OOM) killer via SIGKILL (signal 9, 128 + 9 = 137). When a container exceeds the memory limit defined under 'resources.limits.memory' in the Pod specification, the cgroup memory controller forcefully kills the processes within the cgroup. To remediate this issue:
1. Inspect the termination reason via 'kubectl describe pod <name>' checking for 'OOMKilled: true'.
2. Increase the memory limit in the deployment manifest or optimize memory allocation in the application runtime.
3. Configure Horizontal Pod Autoscaler (HPA) or Vertical Pod Autoscaler (VPA) to adaptively scale memory allocations."""
    elif "Postgres and Snowflake" in q:
        return """PostgreSQL is an open-source relational OLTP database optimized for low-latency row transactions, while Snowflake is a cloud-native analytical data warehouse designed for high-scale OLAP querying. For streaming ingestion, PostgreSQL utilizes WAL logical replication and Kafka Connect JDBC sinks with minimal latencies (<50ms). In contrast, Snowflake provides Snowpipe Streaming, which ingests rows via serverless ingestion channels directly into micro-partitions within 1-2 seconds. Pricing differs drastically: PostgreSQL is billed by persistent provisioned compute instances and EBS storage, whereas Snowflake bills storage separately ($23/TB/month) and computes dynamically based on active virtual warehouse credit runtime."""
    else:
        return f"""Official Technical Documentation for: {q}.
When diagnosing this behavior in modern distributed architectures, engineers must evaluate both runtime telemetry and configuration parameters. The standard remediation involves verifying resource boundaries, inspecting active thread locks, and applying idempotent configuration directives as documented in the engineering handbook."""

with col_bench1:
    st.subheader("Simulated Transformation Outputs")
    
    trans_tab = st.radio(
        "Select Transformation Mode to Inspect:",
        options=["Query Rewriter", "Query Expander", "Query Decomposer", "Multi-Query (3 Perspectives)", "HyDE Document Generator"],
        horizontal=True
    )
    
    if trans_tab == "Query Rewriter":
        st.markdown("**Transformed Single Declarative Query:**")
        rewritten = simulate_rewriting(user_input_query)
        st.info(rewritten)
        st.markdown("*Impact*: Eliminates conversational ambiguity, fixes slang, and ensures dense vector models capture exact domain tokens.")
        
    elif trans_tab == "Query Expander":
        st.markdown("**Enriched Query with Domain Terminology:**")
        expanded = simulate_expansion(user_input_query)
        st.code("\n".join(expanded), language="text")
        st.markdown("*Impact*: Fills vocabulary gaps, bridging lexical mismatches between everyday users and formal technical docs.")

    elif trans_tab == "Query Decomposer":
        st.markdown("**Decomposed Sub-Queries for Multi-Hop Retrieval:**")
        decomposed = simulate_decomposition(user_input_query)
        for i, sq in enumerate(decomposed, 1):
            st.markdown(f"**Step {i}**: `{sq}`")
        st.markdown("*Impact*: Solves compound queries that no single document chunk can answer completely.")

    elif trans_tab == "Multi-Query (3 Perspectives)":
        st.markdown("**Generated Multi-Query Ensemble:**")
        multi_q = simulate_multi_query(user_input_query)
        for i, mq in enumerate(multi_q, 1):
            st.markdown(f"**Variant {i}**: {mq}")
        st.markdown("*Impact*: Parallel vector searches cover varied semantic angles; results are merged via Reciprocal Rank Fusion (RRF).")

    elif trans_tab == "HyDE Document Generator":
        st.markdown("**Hallucinated Hypothetical Answer Document (Zero Grounding):**")
        hyde_doc = simulate_hyde(user_input_query)
        st.success(hyde_doc)
        st.markdown("*Impact*: Vector model embeds this 150-word declarative answer instead of the short 8-word question, achieving 100% syntactic parity with corpus chunks!")

with col_bench2:
    st.subheader("Reciprocal Rank Fusion (RRF) Live Simulator")
    st.markdown("""
    Suppose we run **Multi-Query Retrieval** with 3 query variants (Q1, Q2, Q3). 
    Below is a live simulated ranking across 6 documents in our vector database:
    """)
    
    # Simulate candidate ranks
    sim_data = {
        "Document ID": ["Doc_A (Official Manual)", "Doc_B (Troubleshooting Guide)", "Doc_C (Architecture Spec)", "Doc_D (Release Notes)", "Doc_E (Blog Post)", "Doc_F (FAQ Page)"],
        "Rank in Q1": [1, 2, 8, 15, 20, 30],
        "Rank in Q2": [3, 1, 4, 18, 25, 40],
        "Rank in Q3": [2, 5, 1, 22, 19, 35]
    }
    
    rrf_k = st.slider("RRF Smoothing Parameter k:", min_value=10, max_value=100, value=60, step=5)
    
    df_sim = pd.DataFrame(sim_data)
    
    # Calculate RRF
    def calc_rrf(row, k):
        score = 0.0
        for col in ["Rank in Q1", "Rank in Q2", "Rank in Q3"]:
            rank = row[col]
            score += 1.0 / (k + rank)
        return score

    df_sim["RRF Score"] = df_sim.apply(lambda r: calc_rrf(r, rrf_k), axis=1)
    df_sim["RRF Score Formatted"] = df_sim["RRF Score"].apply(lambda s: f"{s:.5f}")
    df_sim = df_sim.sort_values(by="RRF Score", ascending=False).reset_index(drop=True)
    df_sim["Final Fused Rank"] = df_sim.index + 1
    
    styled_df = df_sim[["Final Fused Rank", "Document ID", "Rank in Q1", "Rank in Q2", "Rank in Q3", "RRF Score Formatted"]]
    st.dataframe(styled_df, use_container_width=True)
    
    st.caption("""
    Notice that **Doc_A** and **Doc_B** dominate the fused ranking because they placed in the top 3 across all three query variants!
    Even though Doc_C placed #1 in Q3, its lower rank in Q1 (rank 8) kept it at rank 3 overall.
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: What is the semantic asymmetry problem between queries and documents, and how does HyDE resolve it?", expanded=True):
    st.markdown("""
    **Answer:**
    In information retrieval, user queries and target document passages occupy distinct linguistic and structural manifolds. 
    Queries are typically short (3 to 10 words), interrogative, open-ended, and informal. 
    Corpus documents are long (100 to 500 words), declarative, definitive, and dense with specialized jargon. 
    Because Bi-Encoder models measure cosine similarity between dense vector representations, this syntactic asymmetry causes 
    a distance penalty—the query vector often sits closer to other questions than to the authoritative explanatory text.
    
    **HyDE (Hypothetical Document Embeddings)** resolves this by introducing a generative intermediate step: 
    an instruction-tuned LLM is prompted to generate a hypothetical passage that would answer the query. 
    The embedding model then embeds this *hypothetical passage* rather than the original query. 
    Because the hypothetical passage is declarative, long, and rich in domain vocabulary, its embedding vector resides directly inside 
    the document manifold, aligning with true relevant corpus passages with significantly higher cosine similarity.
    """)

with st.expander("Q2: In what scenarios does HyDE severely degrade retrieval quality?"):
    st.markdown("""
    **Answer:**
    HyDE degrades retrieval performance in three primary scenarios:
    1. **Strict Entity & Fact Retrieval (Hallucination Poisoning)**: If a user asks about a proprietary internal error code (*"Why did Error CODE_9934 trigger?"*) or an obscure person/product, the LLM will hallucinate convincing but completely false facts, generating a hypothetical document about fictional mechanisms that guides vector search toward entirely irrelevant passages.
    2. **Low-Latency Production SLAs**: HyDE requires a full generative LLM call prior to vector search, introducing an extra 200ms to 600ms latency overhead. For real-time autocomplete or sub-100ms search applications, this is unacceptable.
    3. **Highly Specific Keyword Lookups**: When the user query already contains precise identifiers (e.g., UUIDs, part numbers, git commit hashes), traditional hybrid BM25 + dense search outperforms HyDE without any LLM cost.
    """)

with st.expander("Q3: How does Query Decomposition differ from Multi-Query Retrieval?"):
    st.markdown("""
    **Answer:**
    * **Multi-Query Retrieval** generates *synonymous perspectives of the SAME single-hop question*. Every generated query variant attempts to retrieve the exact same piece of information, using different words, styles, or phrasing to overcome dense vector fragility. Results are combined via union or RRF.
    * **Query Decomposition** addresses *compound or multi-hop questions* that cannot be answered by a single document chunk (e.g., *"Compare the latency of Redis and PostgreSQL"*). It breaks the query down into multiple *distinct sub-questions* (Sub-Q1: Redis latency; Sub-Q2: PostgreSQL latency). Each sub-query targets an entirely different subset of the vector database.
    """)

with st.expander("Q4: Why is Reciprocal Rank Fusion (RRF) preferred over raw Score Averaging when merging Multi-Query results?"):
    st.markdown("""
    **Answer:**
    Vector embedding cosine similarity scores are not universally calibrated across different queries. 
    One query variant might generate scores clustered between 0.82 and 0.88, while another variant might produce scores between 0.65 and 0.72 due to differing vector magnitudes and regional densities in embedding space. 
    Averaging raw scores introduces extreme bias toward whichever query generated higher uncalibrated scores.
    
    RRF is **rank-based and scale-invariant**. By utilizing `1 / (k + rank)`, it treats every query variant's top result with equal weight regardless of the absolute cosine distance, providing robust, outlier-resistant fusion across arbitrary retrieval models (dense, sparse BM25, and hybrid).
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Unconstrained Query Expansion Causing Topic Drift**: Using greedy synonym injection (e.g., WordNet or naive LLM expansion) can inadvertently expand *"Apple earnings"* to include *"fruit farming, agricultural commodities"*, poisoning the top-k results with botanical documents.
* **Trap 2: Double-Incurred LLM Cost**: Running both HyDE and Multi-Query generates up to 5 LLM inference calls before retrieval even begins, ballooning token costs and pushing latency beyond 1.5 seconds.
* **Trap 3: Overlooking Prompt Temperature in HyDE**: Setting high temperature (>0.7) during hypothetical document generation causes erratic hallucinations. In production, set temperature between 0.1 and 0.3 to ensure grounded, canonical technical style.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Query Rewriting"):
        st.markdown("**Front**: What is the core goal of Query Rewriting?")
        st.divider()
        st.markdown("**Back**: Disambiguate pronouns, resolve conversational context, and turn vague questions into self-contained declarative search queries.")

with card_cols[1]:
    with st.popover("Card 2: HyDE"):
        st.markdown("**Front**: What does HyDE embed into vector space?")
        st.divider()
        st.markdown("**Back**: The **hypothetical answer document** hallucinated by an LLM, NOT the raw user question!")

with card_cols[2]:
    with st.popover("Card 3: Decomposition"):
        st.markdown("**Front**: When should Query Decomposition be used?")
        st.divider()
        st.markdown("**Back**: For multi-hop or compound queries requiring facts from multiple distinct documents across the corpus.")

with card_cols[3]:
    with st.popover("Card 4: RRF Parameter k"):
        st.markdown("**Front**: What is the standard default for parameter k in RRF?")
        st.divider()
        st.markdown("**Back**: `k = 60`. It smooths out extreme rank differences and prevents rank-1 documents from excessively drowning out consistent rank-2/rank-3 hits.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Query Embedding Transformations:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the query-document semantic asymmetry problem.", key="chk1")
    st.checkbox("I know when to apply Query Rewriting vs. Query Expansion.", key="chk2")
    st.checkbox("I can implement a production HyDE pipeline using an LLM and embedding model.", key="chk3")
with c2:
    st.checkbox("I understand the mathematical formulation and benefits of Reciprocal Rank Fusion (RRF).", key="chk4")
    st.checkbox("I can identify the failure modes of HyDE (proprietary entities and latency overhead).", key="chk5")
    st.checkbox("I can architect a Query Decomposition pipeline for multi-hop questions.", key="chk6")
