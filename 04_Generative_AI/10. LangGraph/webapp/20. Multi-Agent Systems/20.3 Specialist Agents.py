import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="20.3 Specialist Agents - LangGraph Curriculum",
    page_icon="👷",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .badge-spec {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class SpecialistAgentsSimulator:
    """
    Simulates individual specialist agent behaviors:
    - Research, Coding, Data Analysis, Writing, Retrieval, Validation
    - Returns artifacts, tools utilized, and validation scores
    """
    def __init__(self):
        pass

    def execute_specialist(self, specialist_name: str, input_prompt: str) -> Dict[str, Any]:
        if specialist_name == "Research Agent":
            return {
                "specialist": specialist_name,
                "tools_used": ["GoogleSearch", "ArXivQuery", "WebScraper"],
                "deliverable": f"Dossier on '{input_prompt}': Identified 4 verified papers and 2 market analyses.",
                "quality_metric": "Source Verifiability: 96%"
            }
        elif specialist_name == "Coding Agent":
            return {
                "specialist": specialist_name,
                "tools_used": ["PythonREPL", "Linter", "UnitTestRunner"],
                "deliverable": f"def process_task():\\n    # Solves {input_prompt}\\n    return np.mean([10, 20, 30])",
                "quality_metric": "Unit Tests: 4/4 Passed"
            }
        elif specialist_name == "Data-Analysis Agent":
            return {
                "specialist": specialist_name,
                "tools_used": ["DuckDB", "PandasStats", "ChartRenderer"],
                "deliverable": f"Statistical Summary for '{input_prompt}': Mean=42.1, StdDev=3.4, P-Value=0.002.",
                "quality_metric": "Numerical Accuracy: 100%"
            }
        elif specialist_name == "Writing Agent":
            return {
                "specialist": specialist_name,
                "tools_used": ["GrammarChecker", "MarkdownFormatter"],
                "deliverable": f"# Executive Summary: {input_prompt}\\n\\nOur comprehensive findings indicate strong technical feasibility.",
                "quality_metric": "Readability Grade: 11.2 (Executive Standard)"
            }
        elif specialist_name == "Retrieval Agent":
            return {
                "specialist": specialist_name,
                "tools_used": ["ChromaVectorStore", "BM25KeywordIndex", "CohereReranker"],
                "deliverable": f"Retrieved top 5 passage chunks for '{input_prompt}' with average relevance score of 0.89.",
                "quality_metric": "Recall @ 5: 94%"
            }
        else: # Validation Agent
            return {
                "specialist": specialist_name,
                "tools_used": ["FactChecker", "HallucinationJudge", "CompliancePolicy"],
                "deliverable": f"Verification Report for '{input_prompt}': All assertions cross-referenced against ground truth.",
                "quality_metric": "Hallucination Risk: < 1.0%"
            }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Core Specialist Archetypes and Functional Boundaries", divider="rainbow")

st.markdown("""
In enterprise agent engineering, the division of labor across software systems mirrors high-performing human organizations.
Rather than configuring ad-hoc agents, production architectures standardize on **six foundational specialist archetypes**:
1. **Research Agent**: Traverses external knowledge sources to assemble empirical dossiers.
2. **Coding Agent**: Synthesizes and tests executable source code in a sandboxed environment.
3. **Data-Analysis Agent**: Computes statistical aggregates, queries analytical databases, and prepares metrics.
4. **Writing Agent**: Transforms raw technical findings into executive deliverables aligned with tone and formatting guidelines.
5. **Retrieval Agent**: Executes multi-stage semantic and keyword information retrieval over internal enterprise indexes.
6. **Validation Agent**: Operates as an independent quality gate to audit assertions and eliminate hallucinations.

The blueprint below displays how these six archetypes collaborate within a production LangGraph topology.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph SpecialistArchetypes {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_gathering {
            label="1. Information Gathering Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            spec_retrieval [label="Retrieval Agent\\nInternal Vector DB & Hybrid Search", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
            spec_research [label="Research Agent\\nExternal Web & ArXiv Scraper", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        }

        subgraph cluster_processing {
            label="2. Execution & Synthesis Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            spec_coding [label="Coding Agent\\nSandboxed Python REPL & Unit Tests", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            spec_analysis [label="Data-Analysis Agent\\nStatistical Modeling & Pandas Stats", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            spec_writing [label="Writing Agent\\nTone Calibration & Executive Formatter", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
        }

        subgraph cluster_gate {
            label="3. Verification & Quality Assurance Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            spec_validation [label="Validation Agent\\nFact-Checking Gate & Ground Truth Audit", fillcolor="#701A75", fontcolor="#F5D0FE", color="#E879F9"];
            client_delivery [label="Validated Production Response\\nZero Hallucination Guaranteed", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        spec_retrieval -> spec_analysis [label="Passes Documents"];
        spec_research -> spec_coding [label="Passes Specifications"];
        
        spec_coding -> spec_writing [label="Code Deliverable"];
        spec_analysis -> spec_writing [label="Metric Tables"];
        
        spec_writing -> spec_validation [label="Draft Submitted for Audit"];
        spec_validation -> spec_writing [label="Fails Assertion (Feedback Loop)", color="#EF4444", style=dashed];
        spec_validation -> client_delivery [label="Passes All Gates", color="#34D399"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: RESEARCH AGENT
# ==============================================================================
st.header("2. Research Agent: Deep Web Exploration, Source Verification, and Synthesis", divider="rainbow")

st.markdown("""
The **Research Agent** is tasked with answering open-ended queries requiring real-time external data.
- **Tools**: Google Search, Bing API, ArXiv, Wikipedia, Trafilatura HTML scrapers.
- **Prompt Guardrails**: Instructed to cite explicit URLs, extract author credentials, and reject unverified claims.
- **Output Contract**: Emits a structured `ResearchReport` containing findings, source links, and confidence metrics.
""")

# ==============================================================================
# SECTION 3: CODING AGENT
# ==============================================================================
st.header("3. Coding Agent: Program Synthesis, Test-Driven Execution, and Sandboxed Runtime", divider="rainbow")

st.markdown("""
The **Coding Agent** writes, inspects, and executes software programs.
- **Tools**: Sandboxed Python REPL (e.g. Docker, E2B, or Pyodide), abstract syntax tree linters, unit test runners.
- **Test-Driven Loop**: The coding agent should not just generate code; it executes the code against test assertions. If execution fails with a traceback, the agent inspects the error and self-corrects before returning.
- **Security Sandboxing**: Code execution must always be isolated from the host server.
""")

# ==============================================================================
# SECTION 4: DATA-ANALYSIS AGENT
# ==============================================================================
st.header("4. Data-Analysis Agent: Numeric Computation, DataFrame Manipulation, and Statistical Insight", divider="rainbow")

st.markdown("""
The **Data-Analysis Agent** specializes in structured numerical processing.
- **Tools**: DuckDB, SQLite, Pandas, NumPy, Scipy.
- **Role**: Rather than having an LLM estimate mathematical calculations (which causes severe hallucination), the Data-Analysis Agent writes deterministic queries or scripts to compute exact metrics, standard deviations, and correlations.
""")

# ==============================================================================
# SECTION 5: WRITING AGENT
# ==============================================================================
st.header("5. Writing Agent: Style Adherence, Tone Calibration, and Structured Formatting", divider="rainbow")

st.markdown("""
The **Writing Agent** translates dense technical and numerical data into clear, audience-appropriate language.
- **Tools**: Document templates, Markdown formatters, readability checkers.
- **Persona**: Framed as an editorial director who ensures consistent executive voice, eliminates jargon, and formats tables.
""")

# ==============================================================================
# SECTION 6: RETRIEVAL AGENT
# ==============================================================================
st.header("6. Retrieval Agent: Semantic Vector Search, Keyword BM25, and Document Reranking", divider="rainbow")

st.markdown("""
The **Retrieval Agent** acts as an expert information librarian for internal knowledge bases.
- **Tools**: Dense vector similarity search (Chroma, Pinecone), BM25 sparse keyword search, Cross-Encoder rerankers.
- **Multi-Query Strategy**: Breaks ambiguous user questions into multiple targeted search queries and merges retrieved chunks via Reciprocal Rank Fusion (RRF).
""")

# ==============================================================================
# SECTION 7: VALIDATION AGENT
# ==============================================================================
st.header("7. Validation Agent: Automated Fact-Checking, Hallucination Auditing, and Guardrails", divider="rainbow")

st.markdown("""
The **Validation Agent** serves as the quality gate of the multi-agent system.
- **Independence Principle**: To prevent bias, the Validation Agent must never be the same model instance that authored the text.
- **Verification Strategy**: Checks every assertion in the final draft against the original source documents and research notes.
- **Feedback Mechanism**: If an assertion is ungrounded or contradicts sources, the Validation Agent generates specific critique notes and routes back for revision.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Multi-Specialist Agent Graph with Quality Gates", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a multi-specialist pipeline featuring a Researcher, Writer, and Validation Gate.
""")

st.code('''from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class SpecialistPipelineState(TypedDict):
    topic: str
    research_notes: Optional[str]
    draft: Optional[str]
    is_valid: bool
    audit_notes: Optional[str]

def research_specialist_node(state: SpecialistPipelineState) -> dict:
    # Simulates research retrieval
    return {
        "research_notes": f"Empirical findings on {state['topic']}: Verified facts 1, 2, and 3."
    }

def writer_specialist_node(state: SpecialistPipelineState) -> dict:
    notes = state["research_notes"]
    return {
        "draft": f"EXECUTIVE BRIEF on {state['topic']}: Based on verified evidence ({notes}), strategy is sound."
    }

def validation_specialist_node(state: SpecialistPipelineState) -> dict:
    draft = state.get("draft", "")
    # In production, uses an LLM judge to verify groundedness
    valid = "verified evidence" in draft
    return {
        "is_valid": valid,
        "audit_notes": "Passed all factual grounding assertions." if valid else "Failed: Ungrounded claims."
    }

def route_validation_gate(state: SpecialistPipelineState) -> str:
    if state["is_valid"]:
        return END
    return "writer" # Route back for revision if validation fails

builder = StateGraph(SpecialistPipelineState)
builder.add_node("researcher", research_specialist_node)
builder.add_node("writer", writer_specialist_node)
builder.add_node("validator", validation_specialist_node)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "validator")
builder.add_conditional_edges("validator", route_validation_gate, {END: END, "writer": "writer"})

app = builder.compile(checkpointer=MemorySaver())
print("Multi-Specialist Agent Pipeline with Validation Gate Compiled.")''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Specialist Agent Playground and Execution Audit", divider="rainbow")

st.markdown("""
Execute individual specialist archetypes against custom prompts and inspect their tools, deliverables, and quality metrics.
""")

sim = SpecialistAgentsSimulator()

col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    st.subheader("⚙️ Specialist Selector")
    specialist_choice = st.selectbox(
        "Choose Specialist Archetype",
        [
            "Research Agent",
            "Coding Agent",
            "Data-Analysis Agent",
            "Writing Agent",
            "Retrieval Agent",
            "Validation Agent"
        ]
    )
    user_prompt = st.text_input("Specialist Task Input", value="Analyze Q3 Cloud Infrastructure Costs")
    dispatch_btn = st.button("🚀 Invoke Specialist Agent")

with col_sim2:
    st.subheader(f"📋 Specialist Deliverable Output")
    if dispatch_btn:
        spec_res = sim.execute_specialist(specialist_choice, user_prompt)
        st.success(f"Specialist: **{spec_res['specialist']}**")
        st.markdown(f"**Tools Utilized**: `{', '.join(spec_res['tools_used'])}`")
        st.markdown(f"**Quality Benchmark**: `{spec_res['quality_metric']}`")
        st.markdown("#### Deliverable Artifact:")
        st.code(spec_res["deliverable"], language="markdown" if "Coding" not in specialist_choice else "python")

st.subheader("📊 Specialist Archetype Reference Directory")
df_specs = pd.DataFrame({
    "Specialist Archetype": ["Research Agent", "Coding Agent", "Data-Analysis Agent", "Writing Agent", "Retrieval Agent", "Validation Agent"],
    "Primary Mandate": ["External exploration & fact discovery", "Program synthesis & test execution", "Numerical computing & DB queries", "Editorial formatting & tone calibration", "Internal vector & hybrid search", "Quality audit & fact-checking gate"],
    "Typical Tools": ["Search, Scraper, ArXiv", "Python REPL, Docker, Pytest", "DuckDB, Pandas, SQLite", "Markdown, ToneGuide, Linter", "ChromaDB, BM25, Reranker", "FactJudge, PydanticSchema"],
    "Key Risk Mitigated": ["Information obsolescence", "Code bugs & syntax errors", "Math calculation hallucinations", "Unprofessional or inconsistent tone", "Internal knowledge silos", "Untruthful / ungrounded claims"]
})
st.dataframe(df_specs, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: Why should an architecture separate the Data-Analysis Agent from the LLM Writer Agent?", expanded=False):
    st.markdown("""
    **Answer**:
    LLMs are probabilistic token predictors and frequently hallucinate arithmetic calculations.
    A dedicated Data-Analysis Agent executes deterministic code (e.g. Pandas, SQL) to compute exact figures, and then passes those verified numbers to the Writer Agent for narrative synthesis.
    """)

with st.expander("Q2: What is the 'Independence Principle' in Validation Agent design?", expanded=False):
    st.markdown("""
    **Answer**:
    The Validation Agent must use an independent prompt (and ideally a different model family) from the generating agent.
    If the generating agent reviews its own output, confirmation bias causes it to overlook its own hallucinations.
    """)

with st.expander("Q3: How does a Coding Agent protect host infrastructure from malicious or runaway code?", expanded=False):
    st.markdown("""
    **Answer**:
    By running code inside isolated sandboxes (e.g. ephemeral Docker containers, E2B virtual environments, or restricted WebAssembly runners) with hard memory limits, execution timeouts, and disabled network access.
    """)

with st.expander("Q4: What tools does a modern Retrieval Agent employ beyond simple cosine similarity?", expanded=False):
    st.markdown("""
    **Answer**:
    Modern retrieval agents combine dense vector search with sparse keyword search (BM25) via Reciprocal Rank Fusion (RRF), followed by Cross-Encoder reranking models to score deep passage relevance.
    """)

with st.expander("Q5: What should happen when a Validation Agent rejects a draft?", expanded=False):
    st.markdown("""
    **Answer**:
    The Validation Agent should populate an `audit_notes` or `critique` channel with actionable, structured feedback and route execution back to the generating node via a conditional edge.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Running Untrusted Code in Host Python Environment")
    st.markdown("""
    *Mechanism*: Using Python `exec()` directly on LLM-generated code in the worker host.  
    *Consequence*: A malicious or buggy prompt can overwrite host files, crash the server, or exfiltrate environment variables.  
    *Fix*: Always use an isolated sandbox runner with restricted privileges.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Infinite Validation Rejection Loops")
    st.markdown("""
    *Mechanism*: A Validation Agent with overly strict assertions rejecting a Writer's output indefinitely.  
    *Consequence*: The graph cycles infinitely between Writer and Validator, consuming budget and timing out.  
    *Fix*: Enforce a `revision_count <= 2` limit; if still failing, escalate to a human or return the best effort with a disclaimer.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Giving the Research Agent Direct Writing Authority")
    st.markdown("""
    *Mechanism*: Allowing the research agent to format the final customer email.  
    *Consequence*: The response becomes an unformatted wall of research links without executive polish.  
    *Fix*: Separate research gathering from narrative drafting.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Cost Warning): Unnecessary Cross-Encoder Reranking on Trivial Queries")
    st.markdown("""
    *Mechanism*: Running expensive 50-chunk cross-encoder rerankers for simple FAQ lookups.  
    *Consequence*: Adds 800ms of latency and increases GPU compute costs.  
    *Fix*: Gate reranking behind query complexity classifiers.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Research Agent**\\nSpecialist that gathers external web data and empirical evidence with source citations.")
with col_fc2:
    st.info("**Coding Agent**\\nSpecialist executing sandboxed code and verifying solutions with unit tests.")
with col_fc3:
    st.info("**Data-Analysis Agent**\\nSpecialist computing deterministic numerical calculations via SQL and Pandas.")
with col_fc4:
    st.info("**Validation Agent**\\nAn independent quality gate auditing assertions against source documents.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know the mandates and boundaries of all 6 core specialist agent archetypes.", value=True)
c2 = st.checkbox("I understand why mathematical analysis must be delegated to deterministic code runners.", value=True)
c3 = st.checkbox("I understand the Independence Principle in Validation Agent architecture.", value=True)
c4 = st.checkbox("I know how to implement sandboxed code execution for Coding Agents.", value=True)
c5 = st.checkbox("I can build a multi-specialist pipeline with an automated quality gate in LangGraph.", value=True)
c6 = st.checkbox("I know how to prevent infinite revision cycles between Writers and Validators.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 20.3 'Specialist Agents' with full architectural and practical comprehension!")
