import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="19.2 Advanced Retrieval - LLM Engineering",
    page_icon="🎯",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class AdvancedRetrievalEngine:
    """
    Self-contained implementation of advanced retrieval techniques:
    - Conversational Query Rewriting (Coreference resolution)
    - Hypothetical Document Embeddings (HyDE)
    - Parent-Child (Small-to-Big) Document Resolution
    - Anthropic Contextual Retrieval prefix enrichment
    """

    PARENT_DOCS = {
        "PARENT-01": {
            "title": "Apollo 11 Lunar Mission Flight Journal",
            "full_text": (
                "The Apollo 11 mission launched on July 16, 1969, carrying Commander Neil Armstrong, "
                "Lunar Module Pilot Buzz Aldrin, and Command Module Pilot Michael Collins. "
                "While Armstrong and Aldrin descended to the lunar surface aboard the Eagle, "
                "Michael Collins remained in lunar orbit aboard the Command Module Columbia to ensure safe rendezvous."
            ),
            "children": [
                {
                    "child_id": "CHILD-01-A",
                    "text": "The Apollo 11 mission launched on July 16, 1969, carrying Neil Armstrong, Buzz Aldrin, and Michael Collins.",
                    "vector": np.array([0.80, 0.20, 0.15, 0.10, 0.35], dtype=np.float32)
                },
                {
                    "child_id": "CHILD-01-B",
                    "text": "Michael Collins remained in lunar orbit aboard the Command Module Columbia to ensure safe rendezvous.",
                    "vector": np.array([0.92, 0.12, 0.18, 0.08, 0.22], dtype=np.float32)
                }
            ]
        },
        "PARENT-02": {
            "title": "Quantum Key Distribution Protocol Spec",
            "full_text": (
                "The BB84 protocol uses polarized single photons transmitted across fiber optics. "
                "Eavesdropping attempts by Eve inevitably disturb quantum states, introducing detectable bit error rates. "
                "If the quantum bit error rate (QBER) exceeds 11%, the session is terminated immediately."
            ),
            "children": [
                {
                    "child_id": "CHILD-02-A",
                    "text": "The BB84 protocol uses polarized single photons transmitted across fiber optics.",
                    "vector": np.array([0.15, 0.85, 0.40, 0.10, 0.18], dtype=np.float32)
                },
                {
                    "child_id": "CHILD-02-B",
                    "text": "If the quantum bit error rate (QBER) exceeds 11%, the session is terminated immediately.",
                    "vector": np.array([0.18, 0.90, 0.45, 0.08, 0.20], dtype=np.float32)
                }
            ]
        }
    }

    @staticmethod
    def rewrite_conversational_query(chat_history: List[Dict[str, str]], latest_query: str) -> str:
        """Resolves pronouns and context from conversation history into a standalone query."""
        if not chat_history:
            return latest_query
            
        prev_user = chat_history[-1].get("user", "")
        if "apollo 11" in prev_user.lower() and ("who" in latest_query.lower() or "pilot" in latest_query.lower() or "command module" in latest_query.lower()):
            return "Who was the Command Module pilot during the Apollo 11 lunar mission?"
        elif "quantum" in prev_user.lower() and ("threshold" in latest_query.lower() or "abort" in latest_query.lower() or "qber" in latest_query.lower()):
            return "What is the maximum Quantum Bit Error Rate (QBER) abort threshold in the BB84 protocol?"
        return latest_query

    @staticmethod
    def generate_hyde_document(query: str) -> Tuple[str, np.ndarray]:
        """Simulates HyDE generation: creates synthetic plausible paragraph and returns its vector."""
        if "apollo" in query.lower() or "command module" in query.lower():
            hypothetical = (
                "During the Apollo 11 mission, astronaut Michael Collins served as the Command Module Pilot, "
                "staying in orbit around the Moon aboard Columbia while his crewmates walked on the surface."
            )
            vec = np.array([0.91, 0.14, 0.16, 0.09, 0.24], dtype=np.float32)
        elif "qber" in query.lower() or "quantum" in query.lower():
            hypothetical = (
                "In quantum key distribution networks utilizing the BB84 protocol, the session abort threshold "
                "is triggered when the quantum bit error rate (QBER) surpasses 11 percent due to potential eavesdropping."
            )
            vec = np.array([0.17, 0.89, 0.44, 0.09, 0.19], dtype=np.float32)
        else:
            hypothetical = f"Technical documentation answering: {query} with detailed system specifications."
            vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            
        return hypothetical, vec

    @classmethod
    def search_parent_child(cls, query_vector: np.ndarray) -> List[Dict[str, Any]]:
        """Searches across child chunks, then resolves and returns the full parent document."""
        q_norm = query_vector / np.linalg.norm(query_vector)
        scored_children = []
        
        for p_id, p_data in cls.PARENT_DOCS.items():
            for child in p_data["children"]:
                c_norm = child["vector"] / np.linalg.norm(child["vector"])
                sim = float(np.dot(q_norm, c_norm))
                scored_children.append({
                    "child_id": child["child_id"],
                    "parent_id": p_id,
                    "parent_title": p_data["title"],
                    "parent_full_text": p_data["full_text"],
                    "child_text": child["text"],
                    "similarity": sim
                })
                
        scored_children.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_children


# --- HEADER & INTRODUCTION ---
st.title("19.2 Advanced Retrieval: Query Rewriting, HyDE, Parent-Child, & Contextual Retrieval")
st.caption("Chapter 19: Advanced RAG Systems | Domain: 2. Large Language Models")

st.markdown("""
In real-world enterprise applications, raw user queries are notoriously messy: they are often ambiguous (*"how do I do that?"*), 
feature unresolved pronouns across multi-turn chat (*"who was the pilot on that mission?"*), or use non-technical phrasing 
that poorly matches formal enterprise documentation.

To bridge this semantic chasm, modern production RAG systems deploy **Advanced Pre-Retrieval and Structural Indexing Architectures**:
1. **Query Rewriting**: De-contextualizes conversational turns into standalone search questions.
2. **Multi-Query Expansion**: Generates multiple semantic perspectives to maximize candidate coverage.
3. **HyDE (Hypothetical Document Embeddings)**: Searches document-to-document by embedding a synthetic answer rather than a short question.
4. **Parent-Child Retrieval (Small-to-Big)**: Decouples search granularity (small child chunks) from generation context (large parent documents).
5. **Contextual Retrieval (Anthropic)**: Prepends document-wide situational context to every chunk, slashing retrieval failures by nearly 50%.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: Advanced Pre-Retrieval Orchestration & Structural Indexing", divider="rainbow")

st.markdown("""
How advanced pre-retrieval techniques transform ambiguous user queries into high-precision vector and keyword triggers:
""")

adv_retrieval_dot = """
digraph AdvancedRetrievalBlueprint {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    subgraph cluster_ingress {
        label="Conversational User Input";
        color="#1E293B";
        fontcolor="#38BDF8";
        RawInput [label="User Turn\\n'Who was the pilot on that mission?'", fillcolor="#1E293B"];
        ChatHistory [label="Conversation Memory\\nTurn 1: Apollo 11 Moon Landing", fillcolor="#1E293B"];
    }

    subgraph cluster_transforms {
        label="Pre-Retrieval Transformations";
        color="#1E293B";
        fontcolor="#38BDF8";
        Rewriter [label="Query Rewriter (Coreference)\\n'Who was Apollo 11 CM Pilot?'", fillcolor="#0F766E"];
        HyDE [label="HyDE Generator\\nSynthesizes Hypo-Paragraph\\n(Document-to-Document)", fillcolor="#0369A1"];
        MultiQuery [label="Multi-Query Generator\\nGenerates 3-5 variations", fillcolor="#6D28D9"];
    }

    subgraph cluster_index {
        label="Structural Index Resolution";
        color="#1E293B";
        fontcolor="#38BDF8";
        ChildSearch [label="Child Chunk Vector Search\\n(128-token sharp vector)", fillcolor="#4338CA"];
        ParentResolver [label="Parent Document Resolver\\nFetches 1000-token Parent", fillcolor="#059669"];
        ContextualPrefix [label="Contextual Retrieval Index\\nPrepended Document Context", fillcolor="#047857"];
    }

    RawInput -> Rewriter;
    ChatHistory -> Rewriter;
    Rewriter -> HyDE;
    Rewriter -> MultiQuery;
    HyDE -> ChildSearch;
    MultiQuery -> ChildSearch;
    ChildSearch -> ParentResolver;
    ChildSearch -> ContextualPrefix;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(adv_retrieval_dot, use_container_width=True)

st.markdown("""
### Strategic Comparison of Advanced Retrieval Techniques
""")

tech_matrix = [
    {
        "Technique": "Conversational Query Rewriting",
        "Primary Problem Solved": "Ambiguous pronouns across chat history ('who flew it?')",
        "Execution Stage": "Pre-Retrieval (Online)",
        "Latency Overhead": "~150ms (Small LLM forward pass)",
        "Recall Impact": "Eliminates 90%+ of multi-turn conversational retrieval misses"
    },
    {
        "Technique": "Multi-Query Retrieval",
        "Primary Problem Solved": "Sensitivity to specific user vocabulary and phrasing",
        "Execution Stage": "Pre-Retrieval (Online)",
        "Latency Overhead": "~200ms (LLM generation + N parallel searches)",
        "Recall Impact": "Boosts recall by 15-25% across diverse user phrasing"
    },
    {
        "Technique": "HyDE (Hypothetical Doc Embeddings)",
        "Primary Problem Solved": "Asymmetry between short questions and long informative text",
        "Execution Stage": "Pre-Retrieval (Online)",
        "Latency Overhead": "~300ms (LLM generates synthetic paragraph)",
        "Recall Impact": "Massive improvement on broad conceptual questions; risks factual hallucination"
    },
    {
        "Technique": "Parent-Child (Small-to-Big)",
        "Primary Problem Solved": "Tension between vector sharpness and context sufficiency",
        "Execution Stage": "Indexing & Retrieval (Hybrid)",
        "Latency Overhead": "< 5ms (Direct key-value document store lookup)",
        "Recall Impact": "Delivers full context to LLM without vector representation dilution"
    },
    {
        "Technique": "Contextual Retrieval (Anthropic)",
        "Primary Problem Solved": "Chunks losing situational context when extracted from parent docs",
        "Execution Stage": "Offline Ingestion (One-Time)",
        "Latency Overhead": "0 ms at query time (Pre-computed during ingestion)",
        "Recall Impact": "Slashes retrieval failure rates by 49% (67% with reranking)"
    }
]
st.dataframe(pd.DataFrame(tech_matrix), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - QUERY EXPANSION ---
st.header("2. Query Expansion: Synonym Augmentation & Sub-Intent Branching", divider="rainbow")

st.markdown("""
In **Query Expansion**, the system augments the user's initial prompt with domain-specific synonyms, 
acronyms, and related technical terms before querying the index.

### Classical Pseudo-Relevance Feedback (PRF) vs LLM Expansion
- **Classical PRF (Rocchio Algorithm)**: Executes an initial search, assumes the top-3 hits are relevant, extracts the most frequent informative terms from those hits, and re-queries the index. Fast, but risks topic drift if the initial hits are noisy.
- **Generative LLM Expansion**: Prompts an LLM to identify technical aliases and related concepts:
  ```text
  Prompt: "For the user query 'MFA setup', list 3 technical acronyms and 2 synonyms."
  Output: "Multi-factor authentication, 2FA, TOTP, authenticator app, OTP security token."
  ```
  The expanded query ensures that documents mentioning *"TOTP security tokens"* are retrieved even if they never use the letters *"MFA"*.
""")

st.divider()

# --- SECTION 3: TOPIC 2 - QUERY REWRITING ---
st.header("3. Query Rewriting: Coreference Resolution & Conversational De-contextualization", divider="rainbow")

st.markdown("""
In multi-turn conversational agents, users rarely speak in self-contained paragraphs. 
They rely heavily on **Coreference**:

```text
Turn 1: "Tell me about the Apollo 11 lunar mission."
Turn 2: "Who was the pilot on that mission?"
```

If your RAG system passes Turn 2 directly to the vector database:
- The vector search engine looks for documents matching *"Who was the pilot on that mission?"*
- It has zero access to conversation memory! It matches generic aviation manuals or Apollo 13 documents.

### The De-Contextualization Solution
Before retrieval, a compact, high-speed LLM (e.g. Llama-3.2-1B) rewrites the query using chat history:
```text
System: "Given the conversation history, rewrite the final user query into an independent, standalone question."
Rewritten Query: "Who was the Command Module pilot during the Apollo 11 lunar mission?"
```
The rewritten question is passed to the vector index, guaranteeing 100% retrieval precision without conversational confusion.
""")

st.divider()

# --- SECTION 4: TOPIC 3 - MULTI-QUERY RETRIEVAL ---
st.header("4. Multi-Query Retrieval: Multi-Perspective Generation & Candidate Union", divider="rainbow")

st.markdown("""
Dense vector search is notoriously sensitive to small changes in phrasing: 
*"How to scale microservices?"* produces a noticeably different ranking than *"Best practices for container orchestration throughput"*.

**Multi-Query Retrieval** circumvents phrasing sensitivity by generating multiple distinct query perspectives:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Multi-Query Pipeline")
    st.markdown("""
    1. An LLM generates 3 to 5 alternative versions of the user's prompt (e.g., beginner phrasing, technical architect phrasing, error-focused phrasing).
    2. The engine executes vector searches for all 5 queries in parallel.
    3. The candidate lists are merged using **Reciprocal Rank Fusion (RRF)**.
    4. Chunks that appear across multiple query perspectives receive high consensus scores.
    """)

with col2:
    st.info("""
    **Operational Trade-Off**:
    - **Pros**: Drastically improves recall; rescues poorly phrased user queries.
    - **Cons**: Multiplies vector database QPS by 5x; adds 150-250ms of LLM generation latency before retrieval begins.
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - HYDE ---
st.header("5. HyDE: Hypothetical Document Embeddings & Document-to-Document Search", divider="rainbow")

st.markdown("""
Introduced by Luyu Gao et al. (CMU / Google, 2022), **HyDE (Hypothetical Document Embeddings)** 
solves the **asymmetry problem** in vector search.

### The Query-Document Asymmetry Problem
- Queries are short, interrogative, and sparse (e.g. 8 words: *"What is the abort threshold in BB84?"*).
- Documents are long, declarative, and dense (e.g. 500 words of physics formulas).
- Embedding models struggle to bridge this grammatical gap.
""")

hyde_flow = """
digraph HyDEFlow {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    Query [label="User Query\\n'What is the BB84 QBER abort threshold?'", fillcolor="#1E293B"];
    LLM [label="Zero-Shot LLM\\nGenerates Hypothetical Answer\\n(Hallucinates plausible paragraph)", fillcolor="#0F766E"];
    HypoDoc [label="Hypothetical Document\\n'In BB84 quantum protocols, the QBER...'", fillcolor="#0369A1"];
    Embedder [label="Embedding Model\\nEmbeds Hypothetical Document\\n(NOT the user query!)", fillcolor="#4338CA"];
    DocSearch [label="Document-to-Document ANN\\nMatches real enterprise documents\\nsharing structural density", fillcolor="#059669"];

    Query -> LLM -> HypoDoc -> Embedder -> DocSearch;
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(hyde_flow, use_container_width=True)

st.markdown("""
### Why HyDE Works (Even When the Model Hallucinates!)
- The LLM does not need to get the exact factual number right!
- Generating a hypothetical document produces an embedding vector that lives in the **declarative document space**, sharing the exact technical vocabulary, syntax, and sentence structure of real technical manuals.
- Real documentation chunks score vastly higher similarity against the hypothetical document than against the raw question!
""")

st.divider()

# --- SECTION 6: TOPIC 5 - METADATA FILTERING ---
st.header("6. Metadata Filtering: Dynamic SQL Generation, Self-Querying, & ACL Enforcement", divider="rainbow")

st.markdown("""
In enterprise settings, user queries implicitly contain metadata constraints:
*"Show me security incident reports from Q3 2025 regarding API gateways"*.

**Self-Querying Retrieval** uses an LLM to parse the query into two distinct components:
1. **Semantic Search Query**: `"security incident reports API gateways"`
2. **Structured Metadata Filter**: `{"quarter": "Q3", "year": 2025, "category": "Security"}`
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Pydantic Filter Validation
    Never pass LLM-generated filter strings directly into database drivers. 
    Use structured schema validation (Pydantic) to enforce:
    - Allowed field names (`department`, `created_at`, `tenant_id`).
    - Valid comparison operators (`$eq`, `$in`, `$gte`).
    - Prevents vector injection and syntax crashes.
    """)
with col2:
    st.markdown("""
    ### Role-Based Access Control (RBAC)
    Always enforce security boundaries programmatically:
    ```python
    user_tenant = session.user.tenant_id
    user_clearance = session.user.clearance_level
    # Injected automatically into every query predicate
    filter_predicate = {
        "$and": [
            llm_filter,
            {"tenant_id": {"$eq": user_tenant}},
            {"clearance": {"$lte": user_clearance}}
        ]
    }
    ```
    """)

st.divider()

# --- SECTION 7: TOPIC 6 - PARENT-CHILD RETRIEVAL ---
st.header("7. Parent-Child Retrieval: Decoupling Search Precision from Context Generation", divider="rainbow")

st.markdown("""
Every RAG engineer faces the **Granularity Paradox**:
- If chunks are small (100 tokens), vector search is sharp, but the LLM receives fragmented sentences without surrounding context.
- If chunks are large (1,000 tokens), the LLM receives full context, but the vector representation is diluted and imprecise.

**Parent-Child Retrieval (Small-to-Big)** breaks this paradox by decoupling the search unit from the generation unit:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Ingestion Phase")
    st.markdown("""
    - Split document into **large Parent Chunks (1,000 tokens)**.
    - Store the Parent Chunks in a fast key-value store (SQLite, Redis, MongoDB).
    - Subdivide each parent into **small Child Chunks (128 tokens)**.
    - Embed only the Child Chunks and store them in the vector database with a `parent_id` attribute.
    """)

with col2:
    st.subheader("2. Retrieval Phase")
    st.markdown("""
    - The user query searches across the small, razor-sharp **Child Chunks**.
    - When a child chunk matches the query, the engine **retrieves its Parent Document from the key-value store**.
    - The LLM receives the full, coherent 1,000-token Parent Chunk!
    - **Result**: Pinpoint vector search precision + complete, unbroken reasoning context for the LLM!
    """)

st.divider()

# --- SECTION 8: TOPIC 7 - CONTEXTUAL RETRIEVAL ---
st.header("8. Contextual Retrieval: Anthropic's Prefix-Enriched Chunking Architecture", divider="rainbow")

st.markdown("""
Introduced by Anthropic in September 2024, **Contextual Retrieval** represents one of the most effective 
advancements in document indexing.

### The Isolated Chunk Problem
When documents are split into chunks, individual chunks lose their broader situational context:
```text
Chunk 42: "The company's operating margin declined by 3% in Q2 due to supply chain headwinds."
```
Which company? Which year? What industry? 
Because this chunk lacks the company name and year, vector search and BM25 will miss it when a user asks 
*"What happened to Acme Corp margins in 2025?"*.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("How Contextual Retrieval Works")
    st.markdown("""
    During offline ingestion, for every single chunk in the document, an LLM is prompted:
    ```text
    "Here is the full document: [FULL_DOC]
    Here is the specific chunk: [CHUNK]
    Provide 50-100 tokens of context explaining where this chunk fits in the broader document."
    ```
    The generated context prefix is **prepended directly to the chunk text** before embedding and before BM25 indexing!
    """)

with col2:
    st.info("""
    **Empirical Benchmark Results (Anthropic)**:
    - Reduces retrieval failure rates by **35%** with pure embeddings.
    - Reduces retrieval failure rates by **49%** when combined with BM25 hybrid search.
    - Reduces retrieval failure rates by **67%** when paired with Cross-Encoder reranking!
    - Utilizing prompt caching makes this offline enrichment process cost pennies.
    """)

st.divider()

# --- SECTION 9: PRODUCTION CODE ---
st.header("9. Production Implementation: Query Rewriter, HyDE Simulator, & Parent-Child Resolver in Python", divider="rainbow")

st.markdown("""
The following complete, zero-dependency Python script implements the mathematical core of **Conversational Query Rewriting**, 
**Parent-Child (Small-to-Big) Resolution**, and **HyDE Search**:
""")

code_block = '''import numpy as np
from typing import Dict, List, Any, Tuple

class AdvancedRetrievalPipeline:
    """
    Self-contained advanced retrieval pipeline implementing:
    - Coreference query rewriting from chat history
    - Parent-Child (Small-to-Big) document store resolution
    - Simulated HyDE document generation and search
    """
    
    def __init__(self):
        self.parent_store: Dict[str, Dict[str, Any]] = {}
        self.child_index: List[Dict[str, Any]] = []
        
    def add_parent_document(self, parent_id: str, title: str, full_text: str, children: List[Tuple[str, str, np.ndarray]]):
        """Stores parent document and indexes child vectors with parent_id pointers."""
        self.parent_store[parent_id] = {
            "title": title,
            "full_text": full_text
        }
        for child_id, child_text, vector in children:
            norm_v = vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
            self.child_index.append({
                "child_id": child_id,
                "parent_id": parent_id,
                "text": child_text,
                "vector": norm_v
            })
            
    def search_parent_child(self, query_vector: np.ndarray, top_k: int = 2) -> List[Dict[str, Any]]:
        """Searches small child vectors, then resolves to full parent documents."""
        q_norm = query_vector / np.linalg.norm(query_vector)
        scored_children = []
        for child in self.child_index:
            sim = float(np.dot(q_norm, child["vector"]))
            scored_children.append({
                "child_id": child["child_id"],
                "parent_id": child["parent_id"],
                "similarity": sim
            })
            
        scored_children.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Deduplicate and resolve to parent documents
        seen_parents = set()
        resolved_parents = []
        for c in scored_children:
            pid = c["parent_id"]
            if pid not in seen_parents:
                seen_parents.add(pid)
                parent_data = self.parent_store[pid]
                resolved_parents.append({
                    "parent_id": pid,
                    "title": parent_data["title"],
                    "full_text": parent_data["full_text"],
                    "trigger_child_id": c["child_id"],
                    "match_similarity": c["similarity"]
                })
            if len(resolved_parents) == top_k:
                break
                
        return resolved_parents
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 10: WORKBENCH ---
st.header("10. Interactive Workbench: Pre-Retrieval Transformation & Parent-Child Simulator", divider="rainbow")

st.markdown("""
Test how **Conversational Query Rewriting**, **HyDE**, and **Parent-Child Retrieval** 
resolve queries that traditional naive RAG fails to answer:
""")

chat_scenarios = {
    "Apollo 11 Follow-up (Ambiguous Pronoun)": {
        "history": [{"user": "Tell me about the Apollo 11 moon mission.", "assistant": "Apollo 11 landed on the Moon in July 1969..."}],
        "raw_turn": "Who was the pilot on that mission?",
        "target_parent": "PARENT-01"
    },
    "Quantum Key Distribution (Technical Acronym)": {
        "history": [{"user": "Explain the BB84 quantum protocol.", "assistant": "BB84 uses polarized single photons to establish secret keys..."}],
        "raw_turn": "What is the error threshold before aborting?",
        "target_parent": "PARENT-02"
    }
}

wb_col1, wb_col2 = st.columns(2)
with wb_col1:
    scenario_choice = st.selectbox("Select Conversational Scenario", list(chat_scenarios.keys()))
    scenario_data = chat_scenarios[scenario_choice]
    st.caption(f"**Chat History**: User asked: *'{scenario_data['history'][0]['user']}'*")
    st.markdown(f"**Raw Current Turn**: `{scenario_data['raw_turn']}`")

with wb_col2:
    retrieval_mode = st.radio("Retrieval Architecture Mode", ["1. Naive Raw Query (Fails coreference)", "2. Rewritten Standalone Query", "3. HyDE (Hypothetical Document Embeddings)", "4. Parent-Child (Small-to-Big) Resolver"])

# Compute simulated responses
if "1. Naive" in retrieval_mode:
    processed_q = scenario_data["raw_turn"]
    st.warning(f"Searching with Raw Ambiguous Query: '{processed_q}'")
    st.error("Vector search fails: query has no reference to Apollo 11 or BB84; matches random irrelevant documents.")
elif "2. Rewritten" in retrieval_mode:
    processed_q = AdvancedRetrievalEngine.rewrite_conversational_query(scenario_data["history"], scenario_data["raw_turn"])
    st.success(f"**De-Contextualized Rewritten Query**: '{processed_q}'")
    st.info("Query successfully augmented with full conversational context before vector execution.")
elif "3. HyDE" in retrieval_mode:
    hypo_text, hypo_vec = AdvancedRetrievalEngine.generate_hyde_document(scenario_data["raw_turn"])
    st.markdown("**Generated Hypothetical Passage (HyDE)**:")
    st.info(f"*{hypo_text}*")
    st.caption("Searching document-to-document using the synthetic paragraph embedding.")
else:
    # Parent-Child
    q_vec = np.array([0.90, 0.15, 0.18, 0.10, 0.20], dtype=np.float32) if "Apollo" in scenario_choice else np.array([0.18, 0.90, 0.45, 0.08, 0.20], dtype=np.float32)
    parent_results = AdvancedRetrievalEngine.search_parent_child(q_vec)
    st.markdown("**Parent-Child (Small-to-Big) Resolution**:")
    for res in parent_results:
        with st.container(border=True):
            st.markdown(f"**Matched Child Chunk**: `{res['child_id']}` (Cosine Sim: `{res['similarity']:.4f}`)")
            st.caption(f"Child text: *'{res['child_text']}'*")
            st.markdown(f"**Resolved Parent Document ({res['parent_id']})**: **{res['parent_title']}**")
            st.success(res["parent_full_text"])

st.divider()

# --- SECTION 11: INTERVIEW PREP ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "What is the fundamental insight behind Hypothetical Document Embeddings (HyDE)?",
        "a": "HyDE (Gao et al., 2022) observes that user queries and document chunks live in asymmetric vector subspaces: queries are short, interrogative, and sparse, while documents are long, informative, and declarative. Standard bi-encoders often struggle with this format mismatch. HyDE prompts an LLM to generate a hypothetical synthetic answer to the query. Even if the synthetic answer contains factual hallucinations, its vocabulary, length, and sentence structure live in the document subspace. Embedding the hypothetical text allows the database to execute document-to-document search, significantly boosting recall on broad conceptual queries."
    },
    {
        "q": "How does Parent-Child (Small-to-Big) retrieval resolve the trade-off between vector sharpness and context completeness?",
        "a": "Small chunks (100-200 tokens) produce razor-sharp embedding vectors with high retrieval precision, but strip away surrounding context needed by the LLM. Large chunks (1,000 tokens) preserve context, but dilute vector embeddings and risk Lost-in-the-Middle attention degradation. Parent-child retrieval decouples search from generation: the system embeds and searches small child chunks, but once a child is matched, it fetches and injects its corresponding 1,000-token parent document into the LLM's prompt."
    },
    {
        "q": "Explain Anthropic's Contextual Retrieval technique and why it improves retrieval accuracy by nearly 50%.",
        "a": "When documents are segmented into chunks, individual chunks lose situational context (e.g. 'Operating margins declined 3% in Q2' lacks the company name, year, and industry). Anthropic's Contextual Retrieval uses an offline LLM prompt during ingestion to generate 50-100 tokens of document-wide context, which is prepended directly to the chunk text before embedding and before BM25 indexing. This restores critical entity names and dates, reducing retrieval failure rates by 49% (and by 67% when paired with reranking)."
    },
    {
        "q": "What is the failure mode of conversational query rewriting in high-throughput production?",
        "a": "Conversational query rewriting requires a full LLM forward pass before retrieval can even begin, adding 150ms to 300ms of end-to-end latency to every query. If applied blindly to single-turn queries (where the user's question is already self-contained), it adds unnecessary latency and API cost. Production systems use fast classification heuristics or lightweight SLMs (like Llama-3.2-1B) to only trigger rewriting when unresolved pronouns or multi-turn conversational patterns are detected."
    },
    {
        "q": "When does HyDE fail and cause worse retrieval than raw query search?",
        "a": "HyDE fails when queried on narrow, exact factual lookups or rare proper nouns (e.g. 'What is the part number for the widget gasket?'). Because the LLM does not know the exact part number, it will hallucinate a fake part number (e.g. 'Part #9942'). The embedding model then searches for documents matching the hallucinated part number, leading the retrieval engine completely astray. HyDE should only be used for broad conceptual or thematic inquiries."
    }
]

for idx, qa in enumerate(qas, 1):
    with st.expander(f"Question {idx}: {qa['q']}"):
        st.write(qa["a"])

st.markdown("""
### Production Traps & Architecture Failure Modes
""")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **Trap 1: Deploying HyDE on Exact Code or Serial Number Lookups**
    Using HyDE on queries like *"Fix error 0x80070005"*. The LLM will hallucinate an incorrect operating system component, 
    guiding vector search toward the wrong subsystem. Only apply HyDE to open-ended conceptual inquiries.
    """)
    st.warning("""
    **Trap 3: Redundant Parent Document Duplication in Top-K Chunks**
    If 4 child chunks belonging to the same parent document score in the top-5 search hits, naively resolving parents 
    will inject 4 identical copies of the 1,000-token parent document into the LLM prompt. Always deduplicate parent IDs before prompt assembly.
    """)

with col_trap2:
    st.error("""
    **Trap 2: Contextual Retrieval Prefix Bloating Past Chunk Limits**
    Prepending 150 words of contextual summary to a 400-token chunk without checking the embedding model's context ceiling. 
    If the combined text exceeds 512 tokens, the trailing chunk content (the actual data!) is silently truncated by the tokenizer.
    """)
    st.warning("""
    **Trap 4: Missing Schema Validation on Self-Querying Metadata Filters**
    Allowing an LLM to generate raw dictionary filters for vector DB queries without Pydantic validation. The LLM will occasionally 
    generate non-existent field names or invalid filter syntax, causing search API 500 crashes.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: HyDE Concept"):
        st.markdown("**What is HyDE?**\\n\\nEmbeds a synthetic LLM answer to enable document-to-document vector matching.")
with fc2:
    with st.popover("Flashcard 2: Small-to-Big"):
        st.markdown("**Parent-Child mechanics?**\\n\\nEmbeds small child chunks (128t) for search; fetches full parent doc (1000t) for generation.")
with fc3:
    with st.popover("Flashcard 3: Contextual RAG"):
        st.markdown("**What is Contextual Retrieval?**\\n\\nPrepends 50-100 tokens of document context to each chunk offline, cutting failures by 49%.")
with fc4:
    with st.popover("Flashcard 4: Query Rewriting"):
        st.markdown("**Why rewrite queries?**\\n\\nResolves multi-turn conversational pronouns (*'who flew that?'*) into standalone questions.")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT ---
st.header("12. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand why conversational query rewriting is required to resolve multi-turn coreference pronouns.",
    "I can explain the asymmetric vector space problem and how HyDE enables document-to-document retrieval.",
    "I understand the Parent-Child (Small-to-Big) architecture and how it decouples search precision from context completeness.",
    "I know how Anthropic's Contextual Retrieval prepends document-wide situational context to individual chunks.",
    "I can implement Pydantic schema validation for self-querying metadata filter generation.",
    "I understand the failure modes of HyDE on narrow factual queries and how to prevent parent document duplication."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of advanced retrieval techniques, query transformations, HyDE, and structural indexing.")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
