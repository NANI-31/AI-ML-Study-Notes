import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="29.3 Research Agents - Autonomous Deep Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for research agent telemetry
st.markdown("""
<style>
    .research-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-cite {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class AutonomousResearchPlannerSimulator:
    """
    Simulates multi-stage autonomous deep research:
    generating a multi-step research plan, dispatching search queries across sources,
    cross-verifying facts, generating citations, and producing long-form reports.
    """
    def __init__(self):
        self.preset_topics = {
            "State of Enterprise LLM Checkpointing 2026": {
                "plan": [
                    "1. Investigate LangGraph StateGraph checkpointer implementations (Postgres vs Sqlite).",
                    "2. Compare fault recovery latency between thread persistence architectures.",
                    "3. Evaluate human-in-the-loop interrupt mechanisms in high-concurrency environments."
                ],
                "sources_verified": [
                    {"title": "LangGraph Persistence Architecture", "source": "https://langchain.dev/docs/langgraph/persistence", "authority": 0.96},
                    {"title": "Distributed State Machines in GenAI", "source": "https://arxiv.org/abs/2602.9912", "authority": 0.91}
                ],
                "report_summary": "Enterprise agent systems have shifted decisively toward external database checkpointers (e.g. PostgresSaver) to achieve persistent thread memory and support human-in-the-loop review [1]. Empirical benchmarks demonstrate that thread checkpoints enable sub-50ms fault recovery across distributed worker nodes [2].",
                "citations": [
                    "[1] LangChain Documentation: LangGraph Persistence Architecture (2026)",
                    "[2] ArXiv: Distributed State Machines in Generative AI (2026)"
                ]
            },
            "Vector Database Indexing Tradeoffs: HNSW vs DiskANN": {
                "plan": [
                    "1. Benchmark RAM footprint and recall tradeoffs between HNSW and DiskANN.",
                    "2. Evaluate billion-scale index build times and NVMe SSD retrieval speeds."
                ],
                "sources_verified": [
                    {"title": "Microsoft DiskANN Research", "source": "https://github.com/microsoft/diskann", "authority": 0.95},
                    {"title": "Vector DB Performance Index 2026", "source": "https://vector-benchmarks.com", "authority": 0.89}
                ],
                "report_summary": "While HNSW delivers ultra-low latency ($< 3ms$) for in-memory vector collections, DiskANN enables billion-scale search with 80% lower RAM utilization by compressing compressed vectors on SSDs [1]. For enterprise datasets exceeding 100M items, hybrid SSD-RAM architectures offer superior cost efficiency [2].",
                "citations": [
                    "[1] Microsoft Research: DiskANN Fast Vector Indexing (2026)",
                    "[2] Vector DB Performance Benchmarks (2026)"
                ]
            }
        }

    def generate_plan_and_report(self, topic: str) -> Dict[str, Any]:
        return self.preset_topics.get(topic, self.preset_topics["State of Enterprise LLM Checkpointing 2026"])


# --- LESSON HEADER ---
st.title("29.3 Research Agents: Search, Corroborate & Synthesize")
st.markdown("""
Autonomous research agents represent one of the most advanced applications of LLM agents.
Instead of answering questions from memory, research agents act as autonomous investigative analysts:
they formulate structured research plans, decompose complex problems into sub-queries, query multiple heterogeneous sources,
cross-verify facts to eliminate hallucinations, and synthesize publication-grade research reports with verified inline citations.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Autonomous Deep Research Multi-Stage Architecture", divider="rainbow")
st.markdown("""
Deep research architectures move beyond single-turn RAG by orchestrating a hierarchical multi-stage workflow:
a **Planner** drafts a research outline; a **Researcher** searches and gathers evidence across diverse sources;
a **Critic/Verifier** cross-checks claims for consistency; and a **Writer** compiles the findings into a cohesive, cited report.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ResearchAgents_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_plan_stage {
            label = "Stage 1: Planning & Query Decomposition";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ResearchTopic [label="Complex Research Inquirer\\n('Analyze enterprise vector search latency vs cost')", fillcolor="#0F172A", color="#3B82F6"];
            PlanGenerator [label="Research Planner Node\\n(Generates structured outline & targeted sub-queries)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_gather_stage {
            label = "Stage 2: Multi-Source Gathering";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            WebSearcher [label="Web Search Worker\\n(Tavily / Google API)", fillcolor="#0369A1", color="#38BDF8"];
            AcademicSearcher [label="Academic Paper Worker\\n(ArXiv / Semantic Scholar API)", fillcolor="#0369A1", color="#38BDF8"];
            EnterpriseKB [label="Internal Document Worker\\n(VectorDB / SQL Database)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_verification_stage {
            label = "Stage 3: Fact Verification & Cross-Checking";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            EvidenceStore [label="Unified Evidence Collector\\n(Tagged Passages + Metadata + URL)", fillcolor="#1E293B", color="#60A5FA"];
            VerifierCritic [label="Fact Verifier & Contradiction Critic\\n(Cross-checks claims across >= 2 independent sources)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_synthesis_stage {
            label = "Stage 4: Synthesis & Publication";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ReportWriter [label="Long-Form Report Writer\\n(Synthesizes sections with inline [1] citations)", fillcolor="#065F46", color="#10B981"];
            FinalDeliverable [label="Executive Briefing / Whitepaper\\n(Executive Summary, Methodology, Footnotes)", fillcolor="#065F46", color="#10B981"];
        }

        ResearchTopic -> PlanGenerator [label="1. Decompose topic"];
        PlanGenerator -> WebSearcher [label="Sub-query A"];
        PlanGenerator -> AcademicSearcher [label="Sub-query B"];
        PlanGenerator -> EnterpriseKB [label="Sub-query C"];

        WebSearcher -> EvidenceStore [label="Web snippets"];
        AcademicSearcher -> EvidenceStore [label="Paper abstracts"];
        EnterpriseKB -> EvidenceStore [label="Internal docs"];

        EvidenceStore -> VerifierCritic [label="Verify evidence"];
        VerifierCritic -> ReportWriter [label="Corroborated facts"];
        ReportWriter -> FinalDeliverable [label="Generated whitepaper"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: SEARCH TO RETRIEVE TO SUMMARIZE ---
st.header("2. The Core Research Loop: Search, Retrieve, and Summarize Workflows", divider="rainbow")
st.markdown("""
The atomic unit of an autonomous research agent is the **Search-Retrieve-Summarize** triangle:
1. **Search**: Issue targeted keyword queries to discover candidate sources.
2. **Retrieve**: Fetch full text passages and isolate relevant sections.
3. **Summarize**: Distill passages into atomic factual notes tagged with source metadata.
""")

col_lp1, col_lp2 = st.columns(2)
with col_lp1:
    st.markdown("### The Atomic Research Cycle")
    st.markdown("""
    - **Query Specialization**: Transform broad questions ('tell me about AI agents') into specific technical queries ('LangGraph checkpointer persistence SQLite benchmarks').
    - **Atomic Notes**: Avoid passing 10,000 raw words into the final report prompt; extract atomic 50-word bullet points capturing core metrics.
    - **Source Tagging**: Ensure every atomic fact is paired with its originating URL.
    """)
with col_lp2:
    st.code("""
# Atomic Note Extraction Pattern
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

note_prompt = PromptTemplate.from_template(\"\"\"
Given the following source text, extract 2-3 key factual findings relevant to: {topic}.
Each bullet point must cite the specific source URL.

Source URL: {url}
Text: {text}
\"\"\")

summarizer = note_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    """, language="python")

# --- SECTION 3: MULTI-SOURCE RESEARCH ---
st.header("3. Multi-Source Research: Aggregating Across Web, Papers, and Enterprise Datastores", divider="rainbow")
st.markdown("""
Relying on a single search engine introduces algorithmic bias and misses critical private context.
A robust research agent aggregates information across multiple source tiers:
- **Public Web**: Breaking news, documentation, and product releases (Tavily).
- **Academic Repositories**: Peer-reviewed scientific literature and preprints (ArXiv).
- **Enterprise Warehouses**: Private customer databases, internal wikis, and historical reports (pgvector/SQL).
""")

st.code("""
import asyncio
from langchain_core.runnables import RunnableParallel

# Multi-Source Concurrent Gathering Pipeline
async def gather_all_sources(sub_query: str):
    results = await asyncio.gather(
        search_web_tool.ainvoke(sub_query),
        search_arxiv_tool.ainvoke(sub_query),
        search_internal_kb.ainvoke(sub_query)
    )
    return {
        "web": results[0],
        "academic": results[1],
        "enterprise": results[2]
    }
""", language="python")

# --- SECTION 4: SOURCE VERIFICATION ---
st.header("4. Source Verification & Fact-Checking: Corroboration & Contradiction Detection", divider="rainbow")
st.markdown("""
LLMs can easily be deceived by unverified blog posts or outdated forum replies.
A research agent's verification step cross-checks claims against independent sources.
If Source A claims 'Feature X was released in May' and Source B claims 'Feature X was released in November',
the agent flags the contradiction and seeks an authoritative primary source (such as the official release notes).
""")

st.code("""
from pydantic import BaseModel, Field

class FactVerification(BaseModel):
    claim: str
    verified: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    corroborating_sources: list[str]
    contradictions_detected: list[str]
    reasoning: str

verifier_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(FactVerification)
""", language="python")

# --- SECTION 5: CITATION GENERATION ---
st.header("5. Citation Generation & Grounded Footnotes: Inline Markers and Grounding Links", divider="rainbow")
st.markdown("""
Uncited reports lack credibility in enterprise settings.
The agent must generate precise inline numerical citations (e.g. `[1]`, `[2]`) linked to a structured bibliography.
Every statement must directly trace back to a specific document chunk.
""")

st.code("""
# Citation Insertion Prompt Directive
CITATION_INSTRUCTION = \"\"\"You are synthesizing an enterprise research report.
RULES FOR CITATIONS:
1. Every factual statement or metric MUST be followed by an inline citation marker, e.g. [1], [2].
2. Markers must correspond to the indexed sources provided in the context.
3. At the end of the report, include a 'References' section listing each marker with its full title and URL:
   [1] Author, 'Document Title', https://domain.com/url
4. Never make ungrounded assertions without a citation.\"\"\"
""", language="python")

# --- SECTION 6: RESEARCH PLANNING ---
st.header("6. Research Planning & Task Decomposition: Generating Multi-Hop Queries and Outlines", divider="rainbow")
st.markdown("""
A complex research inquiry cannot be solved with a single search query.
The research planner decomposes the topic into a structured outline and generates specific sub-queries
that investigate distinct dimensions (architecture, benchmarks, costs, security).
""")

col_pl1, col_pl2 = st.columns(2)
with col_pl1:
    st.markdown("### Planning Methodology")
    st.markdown("""
    - **Step 1: Outline Drafting**: Formulate sections (Introduction, Technical Architecture, Empirical Benchmarks, Enterprise Recommendations).
    - **Step 2: Sub-Query Mapping**: Assign 1-2 focused search queries per section.
    - **Step 3: Dependency Resolution**: Execute foundational queries before comparative queries.
    """)
with col_pl2:
    st.code("""
class ResearchPlan(BaseModel):
    outline_sections: list[str]
    sub_queries: list[str] = Field(..., min_items=3, max_items=8)

planner = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(ResearchPlan)

plan = planner.invoke("Evaluate vector search latency tradeoffs across HNSW, IVF, and ScaNN.")
print(plan.sub_queries)
    """, language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Full Deep Research Autonomous Agent System", divider="rainbow")
st.markdown("""
Below is an end-to-end autonomous research agent pipeline implementing planning, multi-source gathering,
and cited long-form synthesis.
""")

st.code("""
# deep_research_system.py
from typing import Dict, Any, List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

class ResearchAgent:
    def __init__(self):
        self.search = TavilySearchResults(max_results=3)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def run_deep_research(self, topic: str) -> str:
        # 1. Planning phase
        planner_prompt = ChatPromptTemplate.from_template(
            "Break the topic into 3 specific investigative search queries: {topic}"
        )
        sub_queries = (planner_prompt | self.llm).content.splitlines()
        
        # 2. Gather phase
        all_evidence = []
        for q in sub_queries[:3]:
            if q.strip():
                results = self.search.invoke({"query": q})
                for r in results:
                    all_evidence.append(f"Source: {r.get('url', 'N/A')}\\n{r.get('content', '')}")
                    
        # 3. Synthesis phase with citations
        writer_prompt = ChatPromptTemplate.from_template(\"\"\"
You are an executive research analyst. Using the provided evidence, write a detailed briefing.
Include inline citations [1], [2] and append a complete Bibliography section.

Topic: {topic}
Evidence:
{evidence}
\"\"\")
        report = (writer_prompt | self.llm).invoke({
            "topic": topic,
            "evidence": "\\n\\n".join(all_evidence[:6])
        })
        return report.content
""", language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Multi-Source Research Planner & Report Generator", divider="rainbow")
st.markdown("""
Simulate autonomous research missions: inspect decomposed planning steps,
reviewed source authorities, and generated reports with verified inline citations.
""")

research_sim = AutonomousResearchPlannerSimulator()

col_t, col_rep = st.columns([1, 1])

with col_t:
    st.subheader("🔬 Research Topic Selector")
    chosen_topic = st.selectbox(
        "Choose Research Subject:",
        list(research_sim.preset_topics.keys())
    )
    execute_res_btn = st.button("🚀 Execute Autonomous Research Mission", type="primary")

with col_rep:
    st.subheader("📊 Research Telemetry & Deliverable")
    if execute_res_btn:
        res = research_sim.generate_plan_and_report(chosen_topic)
        
        st.caption("Stage 1: Decomposed Research Plan:")
        for step in res["plan"]:
            st.markdown(f"- <code>{step}</code>", unsafe_allow_html=True)
            
        st.caption("Stage 2: Verified Source Authorities:")
        for s in res["sources_verified"]:
            st.markdown(f"- <span class='badge-cite'>Score: {s['authority']}</span> <b>{s['title']}</b> (<a href='{s['source']}'>{s['source']}</a>)", unsafe_allow_html=True)
            
        with st.container(border=True):
            st.markdown("### Generated Executive Briefing")
            st.write(res["report_summary"])
            st.markdown("#### Citations & Footnotes")
            for c in res["citations"]:
                st.caption(c)
    else:
        st.info("Select a research topic and execute to observe multi-stage planning and cited report generation.")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: What architectural advantages do hierarchical research agents have over basic single-turn RAG?"):
    st.markdown("""
    **Answer**: Basic single-turn RAG performs a single vector lookup and immediately writes a response. If the query is complex or multifaceted,
    retrieval misses sub-topics. Hierarchical research agents use multi-stage workflows: a planner breaks the topic into sub-queries,
    workers query multiple sources, critics cross-verify facts, and a writer synthesizes a cohesive report, achieving significantly higher
    depth, accuracy, and coverage.
    """)

with st.expander("Q2: How do research agents detect and resolve conflicting information across sources?"):
    st.markdown("""
    **Answer**: Agents use verification critic prompts. The critic extracts atomic claims from each source and compares them for logical consistency.
    When a discrepancy is detected (e.g. differing benchmark numbers), the agent prompts the model to assign higher weight to primary authoritative
    sources (official documentation or peer-reviewed papers) over secondary blogs, or explicitly highlights the discrepancy in the report.
    """)

with st.expander("Q3: How do you prevent citation hallucination in generated reports?"):
    st.markdown("""
    **Answer**: Models can hallucinate plausible-looking URLs. To prevent this, assign explicit integer identifiers (`[Source 1]`, `[Source 2]`)
    to each retrieved document chunk before passing them to the writer prompt. Instruct the model to cite only the provided numbers.
    A post-processing validator verifies that every citation marker corresponds to an actual chunk in the context.
    """)

with st.expander("Q4: Why should research agents extract atomic factual notes before final synthesis?"):
    st.markdown("""
    **Answer**: Passing 20 full web articles into the synthesis prompt introduces context bloat (over 50,000 tokens), increasing latency
    and triggering 'lost in the middle' degradation. Extracting atomic notes (50-word summaries per passage) compresses evidence by 90%,
    allowing the final synthesis prompt to focus exclusively on verified facts.
    """)

with st.expander("Q5: What is the role of multi-hop query decomposition in research planning?"):
    st.markdown("""
    **Answer**: Multi-hop queries are questions where answering the final question requires knowing the answer to an intermediate question
    (e.g. 'Compare the memory efficiency of the algorithm invented by Malkov in 2018 against modern vector databases'). The planner breaks
    this into Hop 1: 'What algorithm did Malkov invent in 2018?' (HNSW), followed by Hop 2: 'Compare HNSW memory efficiency against modern vector databases'.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Fabricated URLs in Citations)**: Allowing the LLM to write URLs from memory. Always generate citations by referencing pre-indexed source IDs from actual search results.")
    st.warning("**Trap 2 (Single-Source Over-Reliance)**: Accepting a single unverified blog post as ground truth on controversial technical claims without corroboration.")
with col_trap2:
    st.error("**Trap 3 (Unbounded Research Loops)**: Allowing the agent to endlessly spawn new sub-queries without a depth or search budget limit, resulting in astronomical API bills.")
    st.warning("**Trap 4 (Omitting Publication Dates)**: Citing technical documentation without extracting publication dates, leading to reports recommending deprecated libraries.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is the core three-step atomic loop of a research agent?")
        st.caption("Flip for answer:")
        st.info("**Back**: Search -> Retrieve -> Summarize.")
    with st.container(border=True):
        st.markdown("**Front**: How are citation hallucinations prevented in report generation?")
        st.caption("Flip for answer:")
        st.info("**Back**: By indexing evidence with explicit source IDs (`[1]`, `[2]`) and validating citations against actual retrieved chunks.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is multi-hop query decomposition?")
        st.caption("Flip for answer:")
        st.info("**Back**: Breaking complex questions into sequential sub-queries where later queries depend on earlier findings.")
    with st.container(border=True):
        st.markdown("**Front**: Why should evidence be compressed into atomic notes before synthesis?")
        st.caption("Flip for answer:")
        st.info("**Back**: To prevent prompt context window bloat and eliminate irrelevant noisy text.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand the multi-stage architecture of autonomous deep research agents.", value=True)
c2 = st.checkbox("I can implement atomic Search-Retrieve-Summarize loops for evidence gathering.", value=True)
c3 = st.checkbox("I know how to aggregate evidence across public web, academic repositories, and internal KBs.", value=True)
c4 = st.checkbox("I understand how to detect and resolve conflicting claims across sources.", value=True)
c5 = st.checkbox("I can implement verified citation generation without URL hallucination.", value=True)
c6 = st.checkbox("I know how to decompose multifaceted research topics into multi-hop sub-queries.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You have completed Chapter 29: Web and API Agents.")
