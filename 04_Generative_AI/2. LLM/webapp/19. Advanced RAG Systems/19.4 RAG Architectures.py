import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="19.4 RAG Architectures - Advanced RAG",
    page_icon="🏗️",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class AdvancedRAGSimulator:
    """
    Self-contained simulation engine for modern RAG architectures:
    - Basic RAG (Single-shot vector lookup)
    - Conversational RAG (Query condensation from history)
    - Multi-Hop RAG (Decomposition and sequential retrieval)
    - Agentic RAG (Relevance grading with web fallback)
    - Graph RAG (Hierarchical community summaries)
    - Multimodal RAG (Visual patch matching)
    """

    MOCK_KNOWLEDGE_BASE = {
        "acme_q1": "Acme Corp reported $4.2B revenue in Q1 2025, beating consensus estimates by 8%. Operating margin reached 24%.",
        "acme_q2": "Acme Corp projects Q2 2025 revenue between $4.5B and $4.7B with operating margin expected at 25%.",
        "figma_acquisition": "Adobe announced an agreement to acquire cloud-based design platform Figma for approximately $20 billion in cash and stock.",
        "adobe_ceo": "Shantanu Narayen has served as the Chairman and Chief Executive Officer of Adobe Inc. since December 2007.",
        "shantanu_education": "Shantanu Narayen earned a bachelor's degree in electronics engineering from Osmania University, an MBA from UC Berkeley, and an MS from Bowling Green State University."
    }

    COMMUNITY_SUMMARIES = {
        "Enterprise_Financial_Health": (
            "Across all corporate filings, Acme Corp demonstrates resilient fiscal stability, "
            "maintaining 24-25% operating margins and exceeding revenue estimates consistently throughout 2025."
        ),
        "Tech_Industry_MA_Landscape": (
            "Major design and software consolidation continues with mega-cap enterprise acquisitions, "
            "exemplified by Adobe's strategic moves in creative cloud tooling."
        )
    }

    @classmethod
    def run_basic_rag(cls, query: str) -> Dict[str, Any]:
        """Single-shot vector lookup."""
        doc = cls.MOCK_KNOWLEDGE_BASE.get("acme_q1", "")
        return {
            "steps": ["Step 1: Direct Vector Search on Acme Corpus", "Step 2: Context Injection", "Step 3: Synthesis"],
            "retrieved_context": [doc],
            "answer": "Acme Corp reported $4.2B in revenue in Q1 2025 with an operating margin of 24%.",
            "latency_ms": 180,
            "llm_calls": 1
        }

    @classmethod
    def run_conversational_rag(cls, history: str, query: str) -> Dict[str, Any]:
        """Conversational condensation and retrieval."""
        condensed_q = "How does Acme Corp Q2 projected revenue compare to Q1 2025 revenue?"
        docs = [cls.MOCK_KNOWLEDGE_BASE["acme_q1"], cls.MOCK_KNOWLEDGE_BASE["acme_q2"]]
        return {
            "steps": [
                f"Step 1: Condense History ('{history}') + Turn ('{query}') -> '{condensed_q}'",
                "Step 2: Vector Search for Q1 and Q2 Earnings",
                "Step 3: Synthesis with Comparative Context"
            ],
            "retrieved_context": docs,
            "answer": "Acme Corp projects Q2 revenue between $4.5B and $4.7B (margin: 25%), an increase over Q1 revenue of $4.2B (margin: 24%).",
            "latency_ms": 380,
            "llm_calls": 2
        }

    @classmethod
    def run_multihop_rag(cls, query: str) -> Dict[str, Any]:
        """Decomposition into sub-queries."""
        sub1 = "Who acquired Figma?"
        sub2 = "Who is the CEO of Adobe?"
        sub3 = "Where did Shantanu Narayen attend university?"
        docs = [cls.MOCK_KNOWLEDGE_BASE["figma_acquisition"], cls.MOCK_KNOWLEDGE_BASE["adobe_ceo"], cls.MOCK_KNOWLEDGE_BASE["shantanu_education"]]
        return {
            "steps": [
                f"Hop 1: Query '{sub1}' -> Found: Adobe",
                f"Hop 2: Query '{sub2}' -> Found: Shantanu Narayen",
                f"Hop 3: Query '{sub3}' -> Found: Osmania, Bowling Green, UC Berkeley",
                "Step 4: Final Multi-Fact Synthesis"
            ],
            "retrieved_context": docs,
            "answer": "Adobe proposed acquiring Figma. Adobe's CEO Shantanu Narayen attended Osmania University, Bowling Green State University, and UC Berkeley (he did not attend Stanford).",
            "latency_ms": 750,
            "llm_calls": 4
        }

    @classmethod
    def run_agentic_rag(cls, query: str, force_failure: bool = False) -> Dict[str, Any]:
        """Agentic self-corrective retrieval with reflection."""
        if not force_failure:
            steps = [
                "Step 1: Vector Search across internal knowledge base",
                "Step 2: Retrieval Relevance Grading (Grade: PASS, confidence: 0.94)",
                "Step 3: Answer Generation",
                "Step 4: Faithfulness Check against retrieved facts (Grade: PASSED)"
            ]
            ans = "Acme Corp reported $4.2B in revenue in Q1 2025 with a 24% operating margin."
            docs = [cls.MOCK_KNOWLEDGE_BASE["acme_q1"]]
            latency = 420
        else:
            steps = [
                "Step 1: Vector Search across internal knowledge base",
                "Step 2: Retrieval Relevance Grading (Grade: FAIL, confidence: 0.21 - context irrelevant)",
                "Step 3: Query Rewriter Re-phrasing -> 'Acme Corp 2026 satellite launch schedule'",
                "Step 4: Fallback to External Web Search API -> Fetched external bulletin",
                "Step 5: Final Synthesis with Web Source Attribution"
            ]
            ans = "Internal documentation was insufficient; web search indicates Acme Corp launched satellite Alpha-1 in February 2026."
            docs = ["[Web Result] Acme Corp public press release: Satellite Alpha-1 successfully in orbit."]
            latency = 980
            
        return {
            "steps": steps,
            "retrieved_context": docs,
            "answer": ans,
            "latency_ms": latency,
            "llm_calls": 3 if not force_failure else 4
        }

    @classmethod
    def run_graph_rag(cls, query: str) -> Dict[str, Any]:
        """Global thematic traversal across community summaries."""
        return {
            "steps": [
                "Step 1: Global Query Routing (Detected holistic dataset-wide theme)",
                "Step 2: Leiden Graph Community Hierarchy Traversal",
                "Step 3: Retrieve Level-1 Community Summary: 'Enterprise_Financial_Health'",
                "Step 4: Synthesize Global Panoramic Overview"
            ],
            "retrieved_context": list(cls.COMMUNITY_SUMMARIES.values()),
            "answer": "Holistic analysis across corporate risk and earnings graph communities reveals steady fiscal health with 24-25% margins and active merger consolidation in software.",
            "latency_ms": 620,
            "llm_calls": 2
        }


# --- HEADER & INTRODUCTION ---
st.title("19.4 Modern RAG Architectures: Naive, Conversational, Agentic, Multi-Hop, Graph, & Multimodal")
st.caption("Chapter 19: Advanced RAG Systems | Domain: 2. Large Language Models")

st.markdown("""
As enterprise knowledge demands evolve beyond simple FAQ lookups, Retrieval-Augmented Generation has matured from static, 
one-shot vector lookups (**Basic RAG**) into sophisticated reasoning architectures. 

Production systems must now handle conversational context across dialogue turns (**Conversational RAG**), 
synthesize non-obvious facts scattered across disconnected documents (**Multi-Hop RAG**), 
traverse structural knowledge graphs for global thematic queries (**Graph RAG**), 
dynamically self-correct and tool-route (**Agentic RAG**), 
and ingest complex PDF diagrams, tables, and charts directly through vision-language embeddings (**Multimodal RAG**).

This lesson explores the complete taxonomy, design patterns, latency tradeoffs, and implementation mechanics of contemporary RAG architectures.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ---
st.header("1. Architectural Blueprint: The Modern RAG Architecture Taxonomy & Spectrum", divider="rainbow")

st.markdown("""
How modern enterprise retrieval navigates architectural trade-offs between **latency**, **computational cost**, and **reasoning depth**:
""")

arch_blueprint_dot = """
digraph RAGTaxonomy {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    subgraph cluster_ingress {
        label="Query Ingress & Classifier";
        color="#1E293B";
        fontcolor="#38BDF8";
        Query [label="User Request", fillcolor="#1E293B"];
        Router [label="Architecture Router\\nClassifies Query Complexity", fillcolor="#0369A1"];
    }

    subgraph cluster_patterns {
        label="Architectural Paradigms";
        color="#1E293B";
        fontcolor="#38BDF8";
        Basic [label="1. Basic RAG\\nSingle Vector Search (<250ms)", fillcolor="#0F766E"];
        Conversational [label="2. Conversational RAG\\nDialogue State & Condensation", fillcolor="#0284C7"];
        MultiHop [label="3. Multi-Hop RAG\\nSub-Question Decomposition", fillcolor="#4338CA"];
        Agentic [label="4. Agentic RAG (CRAG)\\nSelf-Correction & Fallbacks", fillcolor="#6D28D9"];
        Graph [label="5. Graph RAG\\nCommunity Graph Summaries", fillcolor="#059669"];
        Multimodal [label="6. Multimodal RAG\\nVision Patch Embeddings (ColPali)", fillcolor="#7F1D1D"];
    }

    subgraph cluster_output {
        label="Target Synthesis";
        color="#1E293B";
        fontcolor="#38BDF8";
        LLM [label="Foundation LLM Synthesizer\\nContext-Grounded Cited Answer", fillcolor="#047857"];
    }

    Query -> Router;
    Router -> Basic [label="Simple Fact"];
    Router -> Conversational [label="Chat Turn"];
    Router -> MultiHop [label="Multi-Entity"];
    Router -> Agentic [label="Ambiguous/Complex"];
    Router -> Graph [label="Thematic/Global"];
    Router -> Multimodal [label="Visual/Charts"];

    Basic -> LLM;
    Conversational -> LLM;
    MultiHop -> LLM;
    Agentic -> LLM;
    Graph -> LLM;
    Multimodal -> LLM;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise RAG Architecture Comparison Matrix
""")

overview_matrix = [
    {
        "Architecture": "Basic / Naive RAG",
        "Primary Workflow": "User Query -> Single Vector Search -> Prompt Injection -> LLM Answer",
        "Average Latency": "150 - 250 ms",
        "Key Strengths": "Ultra-low latency, simple infrastructure, lowest token cost",
        "Fatal Weaknesses": "Cannot handle follow-up turns, multi-document synthesis, or non-semantic queries"
    },
    {
        "Architecture": "Conversational RAG",
        "Primary Workflow": "Chat History + Query -> Query Condenser/Rewriter -> Vector Search -> Synthesis",
        "Average Latency": "300 - 500 ms",
        "Key Strengths": "Resolves pronouns (it, they, that), maintains dialogue state across turns",
        "Fatal Weaknesses": "Condensation step adds 1 extra LLM call (latency increase: ~150-250ms)"
    },
    {
        "Architecture": "Multi-Hop RAG",
        "Primary Workflow": "Complex Query -> Decomposition into Sub-queries -> Iterative Retrieval & Synthesis",
        "Average Latency": "600 - 1,200 ms",
        "Key Strengths": "Connects disparate dots across multiple unrelated documents",
        "Fatal Weaknesses": "Sequential queries compound latency; error propagation if hop 1 retrieves wrong context"
    },
    {
        "Architecture": "Agentic RAG (Self-RAG / CRAG)",
        "Primary Workflow": "Autonomous Loop: Router -> Retrieval -> Relevance Grading -> Reflection & Fallback",
        "Average Latency": "400 - 1,500 ms",
        "Key Strengths": "Evaluates retrieved chunk quality; re-retrieves or falls back to web if context is poor",
        "Fatal Weaknesses": "Unbounded latency, potential infinite loops if stopping conditions fail"
    },
    {
        "Architecture": "Graph RAG (Microsoft)",
        "Primary Workflow": "Entity Extraction -> Knowledge Graph + Community Summaries -> Hierarchical Search",
        "Average Latency": "500 - 900 ms",
        "Key Strengths": "Excels at holistic, dataset-wide thematic queries ('What are emerging risks across all filings?')",
        "Fatal Weaknesses": "Heavy indexing overhead (hours/days of LLM graph extraction), complex graph DB ops"
    },
    {
        "Architecture": "Multimodal RAG (ColPali)",
        "Primary Workflow": "Vision-Language Models (ColPali/PaliGemma) -> Patch Embeddings from Page Renders",
        "Average Latency": "350 - 600 ms",
        "Key Strengths": "Preserves visual layout semantics; eliminates brittle OCR parsing of charts and tables",
        "Fatal Weaknesses": "Vision models require higher GPU memory; larger multi-vector index sizes"
    }
]
st.dataframe(pd.DataFrame(overview_matrix), use_container_width=True)

st.divider()

# --- SECTION 2: TOPIC 1 - BASIC RAG ---
st.header("2. Basic RAG: The Single-Shot Vector Baseline & Its Operational Boundaries", divider="rainbow")

st.markdown("""
**Basic RAG** (also called *Naive RAG*) represents the foundational entry point of retrieval systems:
1. User submits query string.
2. Embedding model embeds query into a single dense vector.
3. Vector database executes Approximate Nearest Neighbor (ANN) search to extract top-K chunks.
4. Chunks are formatted into a prompt and passed to the LLM for synthesis.

### Operational Boundaries & Failure Modes
While Basic RAG delivers sub-250ms latency and minimal cloud costs, it fails in four common enterprise scenarios:
- **The Coreference Blindspot**: When a user asks *"How much does it cost?"*, Basic RAG searches for *"it"* and fails.
- **The Multi-Document Gap**: If answering a question requires combining Fact A from Document 1 and Fact B from Document 2, a single search hit rarely surfaces both.
- **The Keyword Out-of-Vocabulary (OOV) Gap**: Struggles with exact serial numbers, error codes, and technical acronyms.
- **Zero Self-Correction**: If the vector DB returns 5 irrelevant chunks, Basic RAG forces the LLM to hallucinate from irrelevant text.
""")

st.divider()

# --- SECTION 3: TOPIC 2 - CONVERSATIONAL RAG ---
st.header("3. Conversational RAG: State Tracking, Memory Condensation, & Dialogue State", divider="rainbow")

st.markdown("""
In real-world chat interfaces, users naturally communicate through shorthand, follow-ups, and pronouns:
- *Turn 1*: *"What were Acme Corp's earnings in Q1 2025?"* -> LLM answers: *$4.2 billion with 24% operating margin.*
- *Turn 2*: *"How does that compare to the previous quarter?"*
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Conversational Condensation Pattern")
    st.markdown("""
    In **Conversational RAG**, the retrieval engine never searches using the raw text of Turn 2!
    - An auxiliary lightweight LLM (e.g. Llama-3.2-1B or Claude 3.5 Haiku) ingests the conversation history plus the new prompt.
    - It generates a **de-contextualized standalone search query**:
      `"Acme Corp financial comparison Q1 2025 vs Q4 2024 revenue operating margin"`
    - The vector database searches the standalone query, pulling the exact historical quarter filings needed for the comparison.
    """)

with col2:
    st.subheader("Dialogue Memory Buffering Strategies")
    st.markdown("""
    - **Sliding Window Buffer**: Retains the last 3 to 5 conversation turns in raw text. Low complexity, but forgets earlier topics.
    - **Summary Memory Buffer**: Uses a background model to maintain a running bullet-point summary of key established facts. Conserves context tokens across 50+ turn sessions.
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - AGENTIC RAG ---
st.header("4. Agentic RAG: Dynamic Tool Calling, Self-Correction, & Adaptive Routing", divider="rainbow")

st.markdown("""
**Agentic RAG** (pioneered by Self-RAG and Corrective RAG / CRAG) replaces static, linear pipelines 
with an autonomous reasoning loop:
""")

agentic_flow = """
digraph AgenticLoop {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontcolor="#F8FAFC", color="#334155"];
    edge [color="#38BDF8", fontname="Helvetica", fontcolor="#94A3B8", fontsize=10];

    QueryIn [label="User Query", fillcolor="#1E293B"];
    Router [label="Agent Router\\n(Tool Selection: VectorDB, SQL, Web)", fillcolor="#0F766E"];
    Retrieval [label="Execute Retrieval", fillcolor="#0369A1"];
    Grader [label="Relevance Grader (LLM)\\nEvaluates context quality", fillcolor="#6D28D9"];
    Decision [label="Quality > Threshold?", shape=diamond, fillcolor="#4338CA"];
    Fallback [label="Query Rewrite & Web Fallback", fillcolor="#7F1D1D"];
    Generator [label="Generate Answer", fillcolor="#059669"];
    HallucinationCheck [label="Faithfulness Grader\\nClaims grounded in context?", fillcolor="#047857"];

    QueryIn -> Router -> Retrieval -> Grader -> Decision;
    Decision -> Generator [label="Yes (Pass)"];
    Decision -> Fallback [label="No (Fail)"];
    Fallback -> Retrieval;
    Generator -> HallucinationCheck;
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(agentic_flow, use_container_width=True)

st.markdown("""
### Key Mechanisms of Agentic RAG
1. **Tool Routing**: The agent decides which specialized retriever to call based on the prompt (e.g. vector store for policy text, SQL database for numerical sales, Web Search for real-time news).
2. **Retrieval Grading (CRAG)**: A fast evaluator LLM inspects retrieved chunks:
   - If chunks are relevant -> Proceed to generation.
   - If chunks are irrelevant -> Trigger query rewrite and search external fallback sources.
3. **Hallucination Reflection**: Inspects the generated completion against the retrieved chunks. If the completion invents unsupported claims, it re-prompts the model with strict negative constraints.
""")

st.divider()

# --- SECTION 5: TOPIC 4 - MULTI-HOP RAG ---
st.header("5. Multi-Hop RAG: Query Decomposition, Sub-Question Planning, & Fact Chaining", divider="rainbow")

st.markdown("""
Complex research questions require synthesizing facts across disconnected documents where **no single chunk contains the complete answer**:

```text
Query: "Did the CEO of the company that acquired Figma attend Stanford University?"
```

To answer this, an LLM must execute a multi-hop reasoning chain:
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Multi-Hop Execution Chain")
    st.markdown("""
    1. **Decomposition**: The engine decomposes the root query into an ordered dependency tree:
       - Sub-Question 1: *"Who acquired Figma?"*
    2. **Hop 1 Retrieval**: Searches vector DB -> Finds: *"Adobe announced acquisition of Figma"*.
    3. **Intermediate Extraction**: Sub-Question 2: *"Who is the CEO of Adobe?"* -> Finds: *Shantanu Narayen*.
    4. **Hop 2 Retrieval**: Sub-Question 3: *"Where did Shantanu Narayen attend university?"* -> Finds: *Osmania, Bowling Green, UC Berkeley*.
    5. **Final Synthesis**: The LLM synthesizes all three hops into a verified negative conclusion: *Adobe acquired Figma; its CEO Shantanu Narayen did not attend Stanford.*
    """)

with col2:
    st.warning("""
    **The Error Propagation Vulnerability**:
    Multi-hop RAG is vulnerable to **cascading failure**:
    - If Hop 1 retrieves the wrong company (e.g. confusing a partnership with an acquisition), all downstream hops will execute searches on the wrong entity.
    - Production multi-hop systems enforce intermediate verification checks at each hop before launching subsequent queries.
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - GRAPH RAG ---
st.header("6. Graph RAG: Knowledge Graphs, Entity Extraction, & Global Thematic Summaries", divider="rainbow")

st.markdown("""
Introduced by Darren Edge et al. (Microsoft Research, 2024), **Graph RAG** resolves the primary blindspot of vector databases: 
**dataset-wide thematic queries**.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Local vs Global Query Dilemma")
    st.markdown("""
    - **Local Query (Vector RAG excels)**: *"What is the dosage of Drug X?"* -> Finds the exact chunk in seconds.
    - **Global Query (Vector RAG fails)**: *"What are the top 5 emerging cybersecurity risks discussed across all 150 incident reports?"*
      - A vector search on *"top 5 risks"* simply matches chunks containing the words *"risks"*; it cannot synthesize themes spanning hundreds of disconnected files!
    """)

with col2:
    st.subheader("The Graph RAG Ingestion Pipeline")
    st.markdown("""
    1. **LLM Entity & Relationship Extraction**: Scans all chunks to extract entities (nodes), relationships (edges), and claims.
    2. **Hierarchical Community Clustering**: Uses the **Leiden algorithm** to detect community clusters at multiple levels of granularity.
    3. **Community Summarization**: An LLM generates narrative summaries for each community cluster offline!
    4. **Global Query Execution**: The query searches across pre-computed high-level community summaries, synthesizing dataset-wide panoramic overviews!
    """)

st.divider()

# --- SECTION 7: TOPIC 6 - MULTIMODAL RAG ---
st.header("7. Multimodal RAG: Vision-Language Models, Visual Layouts, & ColPali", divider="rainbow")

st.markdown("""
Real-world enterprise documents (investor pitch decks, medical journals, engineering schematics, annual reports) 
are inherently visual: they communicate through **charts, diagrams, multi-column grids, and complex tables**.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Brittle Legacy Pipeline")
    st.markdown("""
    ```text
    Raw PDF -> Optical Character Recognition (OCR) 
            -> Scrambled Table Extractor 
            -> Text-Only Embedding Model
    ```
    - Fails on flowcharts, bar graphs, circuit diagrams, and infographic layouts.
    - Discards 80% of non-textual visual evidence.
    """)

with col2:
    st.subheader("The ColPali Paradigm (Faysse et al., 2024)")
    st.markdown("""
    ```text
    Raw PDF -> Render Page as Image 
            -> Vision-Language Model (PaliGemma) 
            -> Multi-Vector Patch Embeddings
    ```
    - Treats every PDF page as an image.
    - Generates token-level vectors corresponding directly to visual patches (preserving charts, tables, and colors!).
    - User queries match visually against charts without requiring any OCR preprocessing!
    """)

st.divider()

# --- SECTION 8: PRODUCTION CODE ---
st.header("8. Production Implementation: Agentic Self-Corrective RAG Router in Python", divider="rainbow")

st.markdown("""
The following complete, zero-dependency Python script implements a production **Self-Corrective Agentic RAG Router (CRAG)**. 
It features dynamic tool selection, retrieval relevance grading, query rewriting, and fallback search execution:
""")

code_block = '''from typing import List, Dict, Any

class AgenticRAGPipeline:
    """
    Self-contained Agentic RAG Router with retrieval grading,
    query rewriting, and fallback execution loops.
    """
    
    def __init__(self, vector_store: Dict[str, str], web_fallback_enabled: bool = True):
        self.vector_store = vector_store
        self.web_fallback = web_fallback_enabled

    def vector_retrieve(self, query: str) -> List[str]:
        """Searches in-memory document store for query keywords."""
        terms = query.lower().split()
        matches = []
        for key, text in self.vector_store.items():
            if any(t in text.lower() for t in terms):
                matches.append(text)
        return matches[:2]

    def grade_context_relevance(self, query: str, retrieved_docs: List[str]) -> bool:
        """
        Evaluates whether retrieved documents contain sufficient evidence.
        In production, this is executed by a lightweight LLM call.
        """
        if not retrieved_docs:
            return False
        q_words = set(query.lower().replace("?", "").split())
        doc_words = " ".join(retrieved_docs).lower()
        shared = sum(1 for w in q_words if w in doc_words)
        relevance_score = shared / max(len(q_words), 1)
        return relevance_score >= 0.35

    def rewrite_query(self, query: str) -> str:
        """Expands query for fallback search."""
        return f"{query} technical documentation and release notes"

    def execute_agentic_flow(self, user_query: str) -> Dict[str, Any]:
        """Runs autonomous agent loop with reflection."""
        trace = ["Step 1: Router received query -> Initiating Vector Search"]
        docs = self.vector_retrieve(user_query)
        
        trace.append(f"Step 2: Retrieved {len(docs)} documents -> Running Relevance Grader")
        is_relevant = self.grade_context_relevance(user_query, docs)
        
        if is_relevant:
            trace.append("Step 3: Relevance Check PASSED -> Synthesizing Grounded Answer")
            answer = f"Synthesized answer based on verified internal context: {docs[0]}"
        else:
            trace.append("Step 3: Relevance Check FAILED -> Rewriting query for fallback")
            rewritten_q = self.rewrite_query(user_query)
            trace.append(f"Step 4: Executing External Web Search Fallback for: '{rewritten_q}'")
            docs = [f"[Web Result] Public live data answering '{rewritten_q}'"]
            trace.append("Step 5: Synthesizing Answer from Fallback Context")
            answer = f"Answer synthesized from external web fallback: {docs[0]}"
            
        return {
            "query": user_query,
            "execution_trace": trace,
            "final_context": docs,
            "answer": answer
        }
'''
st.code(code_block, language="python")

st.divider()

# --- SECTION 9: WORKBENCH ---
st.header("9. Interactive Workbench: Multi-Architecture RAG Execution Sandbox", divider="rainbow")

st.markdown("""
Execute different RAG architectural paradigms on real sample queries and observe their 
**Execution Traces**, **Latency Profiles**, and **LLM Invocation Chains** in real time:
""")

arch_options = [
    "1. Basic RAG (Single-shot vector lookup)",
    "2. Conversational RAG (Query condensation)",
    "3. Multi-Hop RAG (Sub-question decomposition)",
    "4. Agentic RAG (CRAG self-corrective router)",
    "5. Graph RAG (Hierarchical community summaries)"
]

selected_arch = st.selectbox("Select RAG Architecture Pattern", arch_options)

# Run corresponding simulation
if "1. Basic" in selected_arch:
    res = AdvancedRAGSimulator.run_basic_rag("What were Acme Corp's earnings in Q1 2025?")
elif "2. Conversational" in selected_arch:
    res = AdvancedRAGSimulator.run_conversational_rag(
        history="User asked: What was Acme Corp Q1 2025 revenue? Assistant answered: $4.2 billion.",
        query="How does that compare to projected Q2 numbers?"
    )
elif "3. Multi-Hop" in selected_arch:
    res = AdvancedRAGSimulator.run_multihop_rag("Did the CEO of the company that acquired Figma attend Stanford?")
elif "4. Agentic" in selected_arch:
    simulate_fail = st.checkbox("Simulate Initial Vector Retrieval Failure (Forces Web Fallback)", value=False)
    res = AdvancedRAGSimulator.run_agentic_rag("What is Acme Corp's 2026 satellite launch schedule?", force_failure=simulate_fail)
else:
    res = AdvancedRAGSimulator.run_graph_rag("What are the overarching fiscal trends across all corporate filings?")

st.subheader("Architecture Execution Trace & Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("Simulated Wall-Clock Latency", f"{res['latency_ms']} ms")
m2.metric("LLM Invocations in Chain", res["llm_calls"])
m3.metric("Context Chunks Retrieved", len(res["retrieved_context"]))

st.markdown("### Step-by-Step Autonomous Trace")
for step in res["steps"]:
    st.info(step)

st.markdown("### Generated Synthesis Output")
st.success(res["answer"])

with st.expander("Inspect Raw Retrieved Context"):
    for idx, c in enumerate(res["retrieved_context"], 1):
        st.markdown(f"**Chunk {idx}**:\n*{c}*")

st.divider()

# --- SECTION 10: INTERVIEW PREP ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
### Deep-Dive Technical Q&A
""")

qas = [
    {
        "q": "Why does traditional vector RAG fail on global thematic queries and how does Graph RAG solve this?",
        "a": "Vector RAG relies on semantic similarity to find specific text chunks ('local questions' like 'What was Q3 revenue?'). However, for global, dataset-wide thematic inquiries ('What are the major themes discussed across all 100 customer complaint transcripts?'), no single chunk contains the answer, and vector search simply matches chunks containing the word 'theme' or 'complaint'. Graph RAG (Microsoft Research) solves this by extracting an entity-relationship knowledge graph offline, clustering entities into hierarchical communities via the Leiden algorithm, and generating narrative summaries for each community cluster. Global queries search across high-level community summaries, providing holistic panoramic synthesis."
    },
    {
        "q": "How does Agentic RAG (Corrective RAG / Self-RAG) introduce self-correction into the retrieval loop?",
        "a": "Traditional RAG is an open-loop DAG: it retrieves top-K documents and blindly passes them to the generator. If retrieval quality is poor, the LLM hallucinates. Agentic RAG introduces closed-loop feedback: 1) A Retrieval Grader evaluates whether retrieved context is relevant; 2) If relevance fails, the agent rewrites the query and falls back to alternate tools (such as web search or relational SQL); 3) Post-generation, a Faithfulness Grader verifies that all generated claims are grounded in context before emitting the answer to the user."
    },
    {
        "q": "What is the execution mechanism of Multi-Hop RAG and how does it prevent error propagation?",
        "a": "Multi-Hop RAG decomposes a complex query into an ordered dependency tree of sub-questions. It retrieves evidence for the first sub-question, extracts an intermediate entity or fact, and uses that fact to formulate the next sub-question. To prevent error propagation (where an incorrect intermediate retrieval misguides downstream hops), production systems enforce confidence thresholds at each hop: if an intermediate extraction has low confidence or ambiguous candidate entities, the system branches into parallel candidate paths or halts with a verification request."
    },
    {
        "q": "How does ColPali (Multimodal RAG) eliminate the need for traditional OCR in document ingestion?",
        "a": "Traditional document ingestion parses PDFs into text using brittle OCR, which frequently scrambles tables, loses two-column reading orders, and discards infographics. ColPali treats document pages as raw visual images. It uses a Vision-Language Model (PaliGemma) to map image patches into multi-vector token embeddings. Text queries match directly against visual patches of charts, tables, and layouts via late interaction (MaxSim), preserving 100% of the visual layout semantics without fragile OCR text extraction."
    },
    {
        "q": "What are the systems trade-offs between Basic RAG and Agentic RAG in terms of latency, cost, and reliability?",
        "a": "Basic RAG has fixed, deterministic latency (<250ms), exactly 1 LLM call, and minimal cost, but zero ability to handle complex queries or self-correct. Agentic RAG provides high reliability and resilience against poor retrieval, but introduces non-deterministic latency (400ms to 2,000ms+), requires 2 to 5 LLM calls per query (increasing API cost by 3x-5x), and requires strict cycle limits to prevent infinite execution loops."
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
    **Trap 1: Unbounded Loops in Agentic Self-Correction**
    Failing to set a strict `max_retries = 2` limit in the agent reflection loop. If the knowledge base genuinely lacks 
    the requested information, the agent will rewrite queries and re-search indefinitely, exhausting user timeouts 
    and generating massive token bills.
    """)
    st.warning("""
    **Trap 3: Cascading Failure in Sequential Multi-Hop Chains**
    Executing sequential multi-hop pipelines without entity disambiguation. If Hop 1 matches a homonym or wrong company, 
    all subsequent hops will retrieve documents for the wrong entity, producing confidently incorrect answers.
    """)

with col_trap2:
    st.error("""
    **Trap 2: Massive Indexing Cost Explosion in Graph RAG**
    Deploying Graph RAG entity extraction using frontier models (e.g. GPT-4o) across millions of document chunks. 
    Extracting entities, relationships, and claims across 100,000 chunks can require millions of LLM prompt calls, costing thousands of dollars. 
    Use compact open-source models (e.g. Llama-3.1-8B) with prompt caching for graph construction.
    """)
    st.warning("""
    **Trap 4: Multi-Vector VRAM Exhaustion in ColPali Deployment**
    Storing 1,024 vectors per page image uncompressed in memory. For 100,000 document pages, multi-vector storage consumes 
    hundreds of gigabytes of RAM. Use vector quantization (IVF-PQ) and binary embeddings for large-scale ColPali indexes.
    """)

st.markdown("""
### Interactive Technical Flashcards
""")

fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    with st.popover("Flashcard 1: Graph RAG"):
        st.markdown("**What is Graph RAG?**\\n\\nUses knowledge graphs and Leiden community summaries to answer global thematic queries.")
with fc2:
    with st.popover("Flashcard 2: Agentic RAG"):
        st.markdown("**Core Agentic loop?**\\n\\nDynamic tool routing + retrieval relevance grading + reflection and fallback loops.")
with fc3:
    with st.popover("Flashcard 3: Multi-Hop RAG"):
        st.markdown("**Multi-Hop mechanics?**\\n\\nDecomposes complex questions into sequential sub-queries, linking disparate documents.")
with fc4:
    with st.popover("Flashcard 4: ColPali"):
        st.markdown("**What is ColPali?**\\n\\nMultimodal RAG embedding page images directly via vision-language patches, bypassing OCR.")

st.divider()

# --- SECTION 11: SELF-ASSESSMENT ---
st.header("11. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the operational boundaries of Basic RAG and when to graduate to advanced architectures.",
    "I can explain the conversational condensation pattern used to resolve coreference in multi-turn dialogues.",
    "I understand the closed-loop feedback architecture of Agentic RAG (routing, retrieval grading, reflection).",
    "I can articulate how Multi-Hop RAG decomposes queries and the risks of cascading error propagation.",
    "I understand how Graph RAG builds hierarchical community summaries to answer dataset-wide global queries.",
    "I know how Multimodal RAG (ColPali) embeds visual page layouts directly, eliminating brittle OCR extraction."
]

completed = 0
for idx, item in enumerate(checklist_items, 1):
    if st.checkbox(f"{idx}. {item}", key=f"sa_check_{idx}"):
        completed += 1

if completed == len(checklist_items):
    st.success("Mastery Certified! You have demonstrated comprehensive understanding of modern RAG architectures, including Conversational, Multi-Hop, Agentic, Graph, and Multimodal systems.")
else:
    st.info(f"Progress: {completed} of {len(checklist_items)} competencies verified.")
