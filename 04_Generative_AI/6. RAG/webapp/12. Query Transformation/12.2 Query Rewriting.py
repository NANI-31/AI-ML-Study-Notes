import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class QueryRewritingEngine:
    """
    Simulates enterprise query rewriting techniques:
    1. Conversational Query Rewriting (Co-reference resolution)
    2. Query Expansion (Synonym & entity enrichment)
    3. Query Simplification (Conversational noise stripping)
    4. Query Decomposition (Multi-part to sub-queries)
    5. Query Normalization (Orthographic & grammatical cleaning)
    """
    SYNONYM_DICTIONARY = {
        "doctor": ["physician", "medical clinician", "MD"],
        "salary": ["compensation", "earnings", "pay", "remuneration"],
        "crash": ["outage", "failure", "exception", "core dump"],
        "fast": ["low latency", "high throughput", "performant"],
        "vacation": ["PTO", "paid time off", "annual leave", "holiday"]
    }

    PRESET_EXAMPLES = {
        "Co-reference Resolution (Chat History)": {
            "query": "How much does it cost and what is its warranty period?",
            "history": [
                {"role": "user", "content": "Tell me about the Sony WH-1000XM5 headphones."},
                {"role": "assistant", "content": "The Sony WH-1000XM5 is a premium wireless noise-canceling headphone."}
            ],
            "technique": "Conversational Rewriting"
        },
        "Synonym & Term Expansion": {
            "query": "Average doctor salary in metropolitan hospital networks",
            "history": [],
            "technique": "Query Expansion"
        },
        "Conversational Noise Simplification": {
            "query": "Hey can someone please tell me if there's any chance I can know how to rotate my AWS IAM secret key?",
            "history": [],
            "technique": "Query Simplification"
        },
        "Multi-Part Query Decomposition": {
            "query": "Compare Postgres versus Mongo for JSON documents, and summarize the storage compression differences.",
            "history": [],
            "technique": "Query Decomposition"
        },
        "Punctuation & Spelling Normalization": {
            "query": "  whaat is  postgress   connection-pooling???  ",
            "history": [],
            "technique": "Query Normalization"
        }
    }

    @classmethod
    def rewrite_conversational(cls, query: str, history: List[Dict[str, str]]) -> str:
        # Extract last subject mentioned in assistant or user turn
        subject = "the target product"
        for turn in reversed(history):
            text = turn["content"]
            m = re.search(r"\b(Sony WH-1000XM5|PostgreSQL|PgBouncer|Kafka|Dental Plan)\b", text, re.IGNORECASE)
            if m:
                subject = m.group(1)
                break

        # Substitute pronouns
        rewritten = re.sub(r"\b(it|its|that|this)\b", subject, query, flags=re.IGNORECASE)
        return rewritten

    @classmethod
    def expand_query(cls, query: str) -> Dict[str, Any]:
        words = re.findall(r"\b\w+\b", query.lower())
        added_synonyms = []
        for word in words:
            if word in cls.SYNONYM_DICTIONARY:
                added_synonyms.extend(cls.SYNONYM_DICTIONARY[word][:2])

        expanded_string = query
        if added_synonyms:
            expanded_string = f"{query} ({', '.join(added_synonyms)})"

        return {
            "original": query,
            "expanded": expanded_string,
            "synonyms_injected": added_synonyms
        }

    @classmethod
    def simplify_query(cls, query: str) -> str:
        # Remove conversational fluff
        fluff_patterns = [
            r"^(hey|hello|hi|please|can someone tell me if|could you please let me know)\b",
            r"\b(there's any chance i can know|i was wondering if|any chance you know)\b",
            r"\b(can you please|would you kindly)\b"
        ]
        cleaned = query
        for p in fluff_patterns:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[?!.,]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def decompose_query(cls, query: str) -> List[str]:
        # Split on conjunctions and comparison phrases
        parts = re.split(r"\b(and also|and summarize|and list|compare|versus|vs)\b", query, flags=re.IGNORECASE)
        sub_queries = []
        for part in parts:
            p_clean = part.strip()
            if len(p_clean) > 8 and not re.match(r"^(and|versus|vs|compare)$", p_clean, re.IGNORECASE):
                sub_queries.append(p_clean)

        if not sub_queries:
            sub_queries = [query]
        return sub_queries

    @classmethod
    def normalize_query(cls, query: str) -> str:
        # Fix whitespace, lower casing, punctuation stripping, basic typo correction
        norm = query.strip().lower()
        norm = re.sub(r"\s+", " ", norm)
        norm = re.sub(r"[?!.,;:]+", "", norm)
        # Typo correction simulation
        norm = norm.replace("whaat", "what").replace("postgress", "postgres")
        return norm

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="12.2 Query Rewriting - RAG Master Curriculum",
    page_icon="✏️",
    layout="wide"
)

st.title("12.2 Query Rewriting: Expansion, Simplification, Decomposition, & Normalization")
st.caption("Chapter 12: Query Transformation | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Human queries rarely arrive in the optimal linguistic structure for dense or lexical information retrieval. 
Users speak conversationally, omit essential entities, introduce ambiguous pronouns (*"it"*, *"that"*), 
use non-standard synonyms, and combine multiple complex questions into a single run-on sentence.

**Query Rewriting** transforms raw human inquiries into one or more mathematically optimized search strings 
before dispatching queries to vector databases or inverted indices. 
By applying **Co-reference Resolution**, **Synonym Expansion**, **Noise Simplification**, 
**Sub-Query Decomposition**, and **Orthographic Normalization**, modern RAG systems dramatically elevate retrieval recall and precision.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Query Rewriting and Transformation Pipeline", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    qr_bp_dot = """
    digraph QueryRewritingBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_input {
            label = "1. Raw Query & Context Ingress Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            raw_input [label="Raw User Prompt\\n'How much does it cost and what is its warranty?'", fillcolor="#0F172A", color="#38BDF8"];
            chat_memory [label="Conversation Session History\\nTurn 1: 'Tell me about Sony WH-1000XM5'", fillcolor="#0F172A", color="#8B5CF6"];
        }

        subgraph cluster_rewriter {
            label = "2. Cognitive Rewriting & Transformation Engine";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            coref_resolver [label="Co-reference Resolution Gate\\nSubstitutes Pronouns: 'it' -> 'Sony WH-1000XM5'", fillcolor="#334155", color="#10B981"];
            normalizer [label="Normalization & Noise Stripper\\nPrunes Conversational Filler & Fixes Typos", fillcolor="#334155", color="#F59E0B"];
            expansion_node [label="Synonym & Entity Expansion\\nAppends Domain Synonyms & Acronyms", fillcolor="#334155", color="#06B6D4"];
            decomposition_node [label="Query Decomposition Router\\nSplits Multi-Part Queries into Sub-Queries", fillcolor="#334155", color="#EC4899"];

            raw_input -> coref_resolver;
            chat_memory -> coref_resolver [label="Context Hydration"];
            coref_resolver -> normalizer [label="Resolved Pronouns"];
            normalizer -> expansion_node [label="Clean Text"];
            normalizer -> decomposition_node [label="Complex Logic"];
        }

        subgraph cluster_output {
            label = "3. Optimized Retrieval Dispatch Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            sub_q1 [label="Sub-Query A: 'Sony WH-1000XM5 retail price'", fillcolor="#0F172A", color="#10B981"];
            sub_q2 [label="Sub-Query B: 'Sony WH-1000XM5 warranty coverage period'", fillcolor="#0F172A", color="#10B981"];
            vdb_dispatch [label="Concurrent Vector Database Execution\\nParallel Proximity Search on Sub-Queries", fillcolor="#334155", color="#38BDF8"];

            decomposition_node -> sub_q1;
            decomposition_node -> sub_q2;
            sub_q1 -> vdb_dispatch;
            sub_q2 -> vdb_dispatch;
        }
    }
    """
    st.graphviz_chart(qr_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern RAG orchestration frameworks (e.g., **LangChain LCEL**, **LlamaIndex Query Pipelines**), 
query rewriting is implemented as a pre-retrieval pipeline stage (`rephrase_chain = condense_prompt | fast_llm | StrOutputParser()`). 
The rephrase chain takes conversation message history and the latest user turn, producing an autonomous, 
self-contained standalone search query that carries 100% of the required context without pronouns.
""")

st.divider()

# ==============================================================================
# SECTION 2: QUERY REWRITING
# ==============================================================================

st.header("2. Query Rewriting: Co-Reference Resolution and Standalone Query Formulation", divider="rainbow")

st.markdown("""
**Query Rewriting** addresses the conversational reality of multi-turn chat applications: humans naturally 
rely on pronouns, elisions, and implicit references to prior conversation turns.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Co-Reference Resolution Dilemma")
    st.markdown("""
    Consider a multi-turn conversation:
    * **Turn 1**: *User: "Explain the architecture of Apache Kafka consumer groups."* -> (Assistant responds).
    * **Turn 2**: *User: "What happens when it crashes?"*
    
    If the Turn 2 string (*"What happens when it crashes?"*) is embedded directly:
    * The vector database searches for the generic concept of *"crashing"* across your entire documentation!
    * Documents about PostgreSQL crashes, Kubernetes pod evictions, and browser crashes crowd out Kafka.
    
    **The Solution**:
    The Query Rewriter combines Turn 1 and Turn 2 to produce:
    `"What happens when an Apache Kafka consumer group coordinator crashes?"`
    This standalone query guarantees high-precision vector matches.
    """)

with col2:
    st.subheader("Prompt Pattern for Standalone Query Rewriting")
    st.code('''# Standard LCEL Query Rephrasing Prompt
CONDENSE_QUESTION_PROMPT = """Given the following conversation history 
and a follow up question, rephrase the follow up question to be 
a standalone question that contains all necessary entities and context.
Do NOT answer the question, only rewrite it.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone Question:"""
''', language="python")

st.divider()

# ==============================================================================
# SECTION 3: QUERY EXPANSION
# ==============================================================================

st.header("3. Query Expansion: Synonym Injection, Lexical Bridging, and PRF", divider="rainbow")

st.markdown("""
**Query Expansion** enriches the user's initial query with semantically related terms, domain synonyms, 
and pseudo-relevance feedback to bridge the vocabulary gap.
""")

col_exp1, col_exp2 = st.columns([1, 1])

with col_exp1:
    st.subheader("Three Major Expansion Methodologies")
    st.markdown("""
    1. **Dictionary / Ontology Expansion**:
       * Injects curated domain synonyms from knowledge graphs (e.g., medical MeSH ontologies or corporate glossaries).
       * Example: *"physician pay"* -> *"physician (doctor, MD) pay (compensation, salary)"*.
    2. **LLM-Generated Semantic Expansion**:
       * Uses a fast LLM to generate 3 alternative formulations or conceptual sub-topics related to the query.
       * Enables retrieval against documents written by authors with different writing styles.
    3. **Pseudo-Relevance Feedback (PRF)**:
       * Executes an initial coarse search, extracts the top informative terms from the top 3 retrieved documents, 
         appends those terms to the original query, and executes a refined second-stage search.
    """)

with col_exp2:
    st.subheader("Expansion Trade-Offs Matrix")
    exp_df = pd.DataFrame([
        {
            "Expansion Strategy": "Ontology / WordNet",
            "Latency Overhead": "< 1 ms",
            "Precision Risk": "Low (curated terms)",
            "Best For": "Specialized legal and medical domains"
        },
        {
            "Expansion Strategy": "LLM Generative Expansion",
            "Latency Overhead": "100 - 250 ms",
            "Precision Risk": "Moderate (potential topic drift)",
            "Best For": "Open-domain conversational assistants"
        },
        {
            "Expansion Strategy": "Pseudo-Relevance Feedback",
            "Latency Overhead": "25 - 50 ms (2 round-trips)",
            "Precision Risk": "High if first search is noisy",
            "Best For": "BM25 keyword search engines"
        }
    ])
    st.dataframe(exp_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 4: QUERY SIMPLIFICATION
# ==============================================================================

st.header("4. Query Simplification: Stripping Conversational Noise and Punctuation Bloat", divider="rainbow")

st.markdown("""
Users frequently surround their genuine core question with polite conversational pleasantries, 
indirect phrasing, and grammatical fluff. **Query Simplification** extracts the pure information need.
""")

col_simp1, col_simp2 = st.columns([1, 1])

with col_simp1:
    st.subheader("Conversational Fluff Dilution")
    st.markdown("""
    Consider this real-world user query:
    > *"Hey Antigravity! I was wondering if you could please do me a quick favor and check whether our company policy permits employees to rollover their unused vacation days into next year?"*

    **Why Fluff Degrades Retrieval**:
    * **In Dense Vectors**: The embedding model encodes the pleasantries (*"wondering"*, *"favor"*, *"please"*), 
      biasing the vector towards polite conversational memos rather than the authoritative policy document.
    * **In BM25 Lexical**: Words like *"wondering"*, *"favor"*, and *"quick"* consume token positions in inverted index matches.
    * **Simplified Result**:
      `"company policy vacation rollover next year"`
    """)

with col_simp2:
    st.subheader("Simplification Heuristics")
    st.markdown("""
    * **Regex Prefix Stripping**:
      Strip opening pleasantries (*"Can you please tell me"*, *"I would like to know if"*).
    * **Punctuation Pruning**:
      Remove run-on question marks, ellipses, and emojis that fragment tokenizers.
    * **Core Clause Extraction**:
      Identify the interrogative pronoun (*"how"*, *"what"*, *"when"*) and isolate the immediate noun phrase.
    """)

st.divider()

# ==============================================================================
# SECTION 5: QUERY DECOMPOSITION
# ==============================================================================

st.header("5. Query Decomposition: Breaking Multi-Part Questions into Sub-Queries", divider="rainbow")

st.markdown("""
When a user asks a complex multi-part question, executing a single search is guaranteed to fail. 
**Query Decomposition** splits the question into independent, self-contained sub-queries.
""")

col_decomp1, col_decomp2 = st.columns([1, 1])

with col_decomp1:
    st.subheader("The Multi-Part Query Problem")
    st.markdown("""
    **Complex Query**:
    *"Compare PostgreSQL connection pooling using PgBouncer against MySQL connection handling, and describe the memory overhead of both."*

    * **Single Search Failure**:
      A single query vector attempts to find a document that discusses PostgreSQL, PgBouncer, MySQL, AND memory overhead simultaneously. 
      Such a single comprehensive document rarely exists.
    * **Decomposed Sub-Queries**:
      1. *"PostgreSQL connection pooling architecture and PgBouncer configuration"*
      2. *"MySQL thread pool and connection memory overhead"*
      3. *"Memory comparison between PostgreSQL and MySQL connection handling"*
    * Each sub-query executes concurrently against the database, gathering all necessary facts before final synthesis.
    """)

with col_decomp2:
    st.subheader("Decomposition Workflow Diagram")
    decomp_dot = """
    digraph DecompWorkflow {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.35, ranksep=0.45];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin="0.15,0.08"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=8, fontcolor="#94A3B8"];

        complex_q [label="Multi-Part User Query", fillcolor="#0F172A", color="#38BDF8"];
        decomp_engine [label="Query Decomposition LLM\\nOutput: JSON List[str]", fillcolor="#334155", color="#EC4899"];

        sub1 [label="Sub-Query 1 (Postgres/PgBouncer)", fillcolor="#0F172A", color="#10B981"];
        sub2 [label="Sub-Query 2 (MySQL Thread Pool)", fillcolor="#0F172A", color="#10B981"];

        vdb [label="Vector Store / Hybrid Index", fillcolor="#334155", color="#38BDF8"];
        synthesizer [label="Joint LLM Answer Synthesizer", fillcolor="#0F172A", color="#F59E0B"];

        complex_q -> decomp_engine;
        decomp_engine -> sub1;
        decomp_engine -> sub2;
        sub1 -> vdb [label="async"];
        sub2 -> vdb [label="async"];
        vdb -> synthesizer [label="Union Candidates"];
    }
    """
    st.graphviz_chart(decomp_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 6: QUERY NORMALIZATION
# ==============================================================================

st.header("6. Query Normalization: Orthographic, Lexical, and Typographical Standardization", divider="rainbow")

st.markdown("""
**Query Normalization** ensures that superficial variations in spelling, casing, punctuation, 
and encoding do not distort search scoring.
""")

norm_table = pd.DataFrame([
    {
        "Normalization Step": "1. Case Folding",
        "Raw User Input": "PostgreSQL vs POSTGRESQL",
        "Normalized Result": "postgresql vs postgresql",
        "Engineering Rationale": "Guarantees parity across case-sensitive inverted indices and tokenizers."
    },
    {
        "Normalization Step": "2. Whitespace Collapse",
        "Raw User Input": "what    is    WAL   replication",
        "Normalized Result": "what is wal replication",
        "Engineering Rationale": "Removes irregular whitespace tabs and carriage returns that skew subword tokenizers."
    },
    {
        "Normalization Step": "3. Typo Correction",
        "Raw User Input": "kuberentes pod evicshun",
        "Normalized Result": "kubernetes pod eviction",
        "Engineering Rationale": "Applies Levenshtein distance or SymSpell dictionaries to fix common typos before search."
    },
    {
        "Normalization Step": "4. Punctuation Strip",
        "Raw User Input": "how to configure ssl???!!!",
        "Normalized Result": "how to configure ssl",
        "Engineering Rationale": "Eliminates repetitive terminal punctuation that introduces useless token IDs."
    }
])
st.dataframe(norm_table, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Enterprise Query Rewriter & Decomposition Engine", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python module implementing conversational query rewriting, 
noise simplification, synonym expansion, and multi-query decomposition.
""")

st.code(r'''
import re
from typing import List, Dict, Any, Optional

class EnterpriseQueryRewriter:
    """
    Production-grade query transformation pipeline supporting:
    - Co-reference pronoun resolution via chat history
    - Conversational noise simplification
    - Synonym enrichment
    - Multi-part question decomposition
    """
    PRONOUN_PATTERN = re.compile(r"\b(it|its|this|that|they|them|these|those)\b", re.IGNORECASE)
    FLUFF_PATTERN = re.compile(
        r"^(can you please|could you tell me|i want to know|please explain|hey)\b",
        re.IGNORECASE
    )

    @classmethod
    def resolve_coreference(cls, current_query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Substitutes ambiguous pronouns with the primary entity from recent conversation turns.
        """
        if not cls.PRONOUN_PATTERN.search(current_query) or not chat_history:
            return current_query

        # Inspect last assistant or user message for prominent entity
        last_turn = chat_history[-1]["content"]
        entity_match = re.search(r"\b([A-Z][a-zA-Z0-9_\-\.\s]{2,20})\b", last_turn)
        inferred_entity = entity_match.group(1).strip() if entity_match else "the referenced system"

        return cls.PRONOUN_PATTERN.sub(inferred_entity, current_query)

    @classmethod
    def simplify_query(cls, query: str) -> str:
        """
        Strips conversational pleasantries and trailing punctuation.
        """
        cleaned = cls.FLUFF_PATTERN.sub("", query.strip())
        cleaned = re.sub(r"[?!.,]", "", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def decompose_query(cls, query: str) -> List[str]:
        """
        Splits compound questions into independent sub-queries.
        """
        split_markers = re.compile(r"\b(and also|compare|versus|vs|as well as)\b", re.IGNORECASE)
        fragments = split_markers.split(query)
        sub_queries = [f.strip() for f in fragments if len(f.strip()) > 8 and not split_markers.match(f)]
        return sub_queries if sub_queries else [query]
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Query Rewriting and Expansion Simulator", divider="rainbow")

st.markdown("""
Experiment with all five query transformation techniques. 
Select an example preset below or enter custom text to observe real-time rewriting, expansion, and decomposition.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    preset_choice = st.selectbox(
        "Select Example Query Transformation Scenario:",
        list(QueryRewritingEngine.PRESET_EXAMPLES.keys())
    )
    preset_data = QueryRewritingEngine.PRESET_EXAMPLES[preset_choice]

    user_query_input = st.text_area(
        "Query Input:",
        value=preset_data["query"]
    )
    active_technique = preset_data["technique"]
    st.info(f"**Target Technique**: `{active_technique}`")

with wb_col2:
    st.markdown("### 🔄 Transformed Output")

    if active_technique == "Conversational Rewriting":
        st.markdown("**Conversation History Context**:")
        for turn in preset_data["history"]:
            st.caption(f"**{turn['role'].capitalize()}**: {turn['content']}")
        rewritten = QueryRewritingEngine.rewrite_conversational(user_query_input, preset_data["history"])
        st.success(f"**Resolved Standalone Query**:\n*\"{rewritten}\"*")

    elif active_technique == "Query Expansion":
        expanded_res = QueryRewritingEngine.expand_query(user_query_input)
        st.success(f"**Expanded Search String**:\n*\"{expanded_res['expanded']}\"*")
        st.write(f"**Injected Synonyms**: `{expanded_res['synonyms_injected']}`")

    elif active_technique == "Query Simplification":
        simplified = QueryRewritingEngine.simplify_query(user_query_input)
        st.success(f"**Stripped Core Question**:\n*\"{simplified}\"*")

    elif active_technique == "Query Decomposition":
        sub_q_list = QueryRewritingEngine.decompose_query(user_query_input)
        st.success(f"**Generated Sub-Queries ({len(sub_q_list)})**:")
        for idx, sq in enumerate(sub_q_list, start=1):
            st.markdown(f"{idx}. *\"{sq}\"*")

    else:  # Query Normalization
        norm = QueryRewritingEngine.normalize_query(user_query_input)
        st.success(f"**Standardized Normalized Query**:\n*\"{norm}\"*")

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: What is Co-reference Resolution in RAG and why is it essential for multi-turn conversational agents?"):
    st.markdown("""
    **Answer**: In multi-turn chat applications, users frequently refer to entities mentioned in previous turns using pronouns 
    (*"it"*, *"they"*, *"that"*). Because vector databases evaluate search queries independently without conversational context, 
    embedding a query like *"How do I configure it?"* results in searching for generic configuration text across unrelated documents. 
    Co-reference resolution uses conversation history to substitute ambiguous pronouns with the explicit entity name 
    (*"How do I configure PgBouncer connection pooling?"*), ensuring vector searches hit the correct semantic cluster.
    """)

with st.expander("Q2: What is Pseudo-Relevance Feedback (PRF) and how does it differ from LLM-based query expansion?"):
    st.markdown("""
    **Answer**:
    * **Pseudo-Relevance Feedback (PRF)** is a classical information retrieval algorithm. It executes an initial search, 
      assumes the top 3–5 retrieved documents are relevant ("pseudo-relevant"), extracts the most statistically distinctive terms 
      (high TF-IDF / BM25 terms) from those documents, appends them to the original query, and executes a second-stage search. 
      It requires zero generative LLM inference.
    * **LLM-Based Expansion** prompts a generative model to hypothesize synonyms or related sub-topics. 
      While LLM expansion is more semantically creative, it introduces higher latency (150–300ms) and cost, 
      whereas PRF executes entirely within the database engine in under 20ms.
    """)

with st.expander("Q3: When should a RAG system trigger Query Decomposition versus Multi-Query Expansion?"):
    st.markdown("""
    **Answer**:
    * **Multi-Query Expansion** is used for **single-intent** questions that might suffer from vocabulary mismatch. 
      It generates 3 variations of the *same* question (e.g., *"doctor pay"*, *"physician salary"*, *"MD compensation"*) 
      to maximize recall against differently phrased documents.
    * **Query Decomposition** is used for **multi-intent** or **comparative** questions where no single document contains 
      the entire answer (e.g., *"Compare product A with product B"*). It breaks the query into discrete, orthogonal sub-questions 
      (*"What are features of A?"* and *"What are features of B?"*), retrieving evidence for each before synthesizing the comparison.
    """)

with st.expander("Q4: How does query simplification prevent semantic drift in dense vector representations?"):
    st.markdown("""
    **Answer**: Transformer-based embedding models compute a sequence-level embedding by averaging or pooling self-attention token states. 
    When a user query is padded with polite fluff (*"I was wondering if you could please tell me..."*), the attention mechanism 
    allocates substantial attention weight to conversational markers. This pulls the resulting vector towards the conversational 
    subspace of the vector manifold, drifting away from dense technical documentation. 
    Simplifying the query strips non-informational tokens, ensuring the vector focuses 100% on domain-relevant entities.
    """)

with st.expander("Q5: What are the latency trade-offs of running query rewriting via LLM in production search?"):
    st.markdown("""
    **Answer**: Introducing an LLM rewrite step adds a serial network call and token generation overhead (typically 120–350ms). 
    To maintain user-perceived responsiveness:
    1. **Streaming First**: Stream rewrite tokens or speculative retrieval while generating.
    2. **Model Selection**: Use ultra-fast, small distilled models (e.g., GPT-4o-mini, Claude 3 Haiku, or fine-tuned 1B-parameter SLMs).
    3. **Conditional Execution**: Only invoke LLM rewriting if the query understanding stage detects pronouns or complexity; 
       route simple unambiguous queries directly to vector search without rewriting.
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Query Expansion Causing Fatal Topic Drift**
    * **Symptom**: Query expansion appends dozens of broad synonyms, causing the search engine to retrieve unrelated generic articles.
    * **Root Cause**: Expanding specific terms like "Python generator" with broad words like "software", "programming", "generator", and "power".
    * **Remedy**: Limit expansion to top 2 verified domain synonyms, or use rank fusion (RRF) where the original query maintains higher ranking weight.
    """)

    st.error("""
    **Trap 2: Recursive Pronoun Hallucination in Long Conversations**
    * **Symptom**: Standalone query rewriter misidentifies the antecedent entity after 10+ turns of conversation.
    * **Root Cause**: Passing an unpruned 20-turn chat history into the condensation prompt causes the rewriter to fixate on an entity mentioned 15 turns ago.
    * **Remedy**: Bound the chat history window passed to the rewriter to the last 3–5 turns, or maintain an explicit state tracking dictionary of the active entity.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Over-Decomposing Simple Queries with Multiple Adjectives**
    * **Symptom**: Query decomposition generates 4 sub-queries for a query like "fast reliable database", generating 4x database load unnecessarily.
    * **Root Cause**: Splitting on every conjunction ("and") without checking for distinct grammatical predicates or clauses.
    * **Remedy**: Only decompose when multiple distinct named entities or independent verb clauses are detected.
    """)

    st.warning("""
    **Trap 4: Discarding the Original User Query After Rewriting**
    * **Symptom**: Rewritten query fails to find the exact document because the rewriter altered a specific technical term.
    * **Root Cause**: Replacing the original query completely rather than executing both in parallel.
    * **Remedy**: Always include the original user query alongside the rewritten query in a multi-query ensemble.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is Co-reference Resolution in conversational RAG?")
        st.markdown("**Back**: The technique of replacing conversational pronouns ('it', 'that', 'she') with explicit entities from prior chat turns to create self-contained standalone search queries.")

    with st.container(border=True):
        st.markdown("**Front**: How does Query Simplification improve dense vector retrieval?")
        st.markdown("**Back**: By stripping polite filler ('could you please tell me'), preventing non-informational tokens from diluting the attention mechanism and causing vector semantic drift.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is Query Decomposition?")
        st.markdown("**Back**: The process of breaking a compound, multi-part, or comparative query into distinct independent sub-queries, retrieving evidence for each in parallel.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: Why should the original query never be discarded after rewriting?")
            st.markdown("**Back**: Rewriting can accidentally alter exact domain tokens or introduce assumptions. Executing both the original and rewritten queries via ensemble fusion guarantees high precision and recall.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Query Rewriting, Expansion, Simplification, and Decomposition:
""")

chk1 = st.checkbox("I understand why conversational pronouns must be resolved into standalone queries before vector search.", value=True)
chk2 = st.checkbox("I can articulate how query expansion bridges the vocabulary mismatch problem between users and authors.", value=True)
chk3 = st.checkbox("I know how conversational fluff dilutes dense vector embeddings and how simplification strips it.", value=True)
chk4 = st.checkbox("I understand the difference between Multi-Query Expansion (single intent) and Query Decomposition (multi-intent).", value=True)
chk5 = st.checkbox("I recognize the dangers of semantic topic drift when expanding queries with overly broad synonyms.", value=True)
chk6 = st.checkbox("I know why executing both the original query and the rewritten query in parallel provides optimal recall.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Masterful! You have conquered Query Rewriting, Expansion, and Decomposition pipelines.")
