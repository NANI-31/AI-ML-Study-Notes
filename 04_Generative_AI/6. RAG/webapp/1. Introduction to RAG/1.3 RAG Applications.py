import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="1.3 RAG Applications - RAG Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Title Block & 5-Pillar Overview
st.title("1.3 RAG Applications: Enterprise Knowledge, Legal, Healthcare, Code, & Support")
st.caption("Chapter 1: Introduction to RAG | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Retrieval-Augmented Generation has emerged as the universal standard architecture across every enterprise vertical 
requiring grounded factual synthesis. While generic chat models struggle with domain boundaries, domain-specialized 
RAG systems tailor chunking granularity, retrieval scoring, metadata filtering, and safety guardrails to the 
unique regulatory and latency requirements of each industry.

From real-time customer support bots requiring sub-second response times to high-stakes legal discovery platforms 
requiring 100% precision and verifiable page citations, RAG serves as the cognitive bridge between unstructured data 
and generative action.

In this lesson, you will master:
1. The nine dominant enterprise application archetypes of RAG systems.
2. Domain-specific architectural adaptations across Legal, Healthcare, Code, and Support.
3. Multi-tenant access controls, regulatory compliance frameworks, and strict latency SLAs.
4. Building a modular, domain-aware enterprise RAG router in production Python.
5. High-frequency enterprise architecture interview questions, production antipatterns, and validation checkpoints.
""")

st.divider()

# Module-Scope Helper Classes for Interactive Workbench & Production Pipeline
class DomainKnowledgeStore:
    def __init__(self):
        self.domains: Dict[str, List[Dict[str, str]]] = {
            "Legal": [
                {
                    "id": "LEG-101",
                    "title": "Delaware General Corporation Law Section 141",
                    "text": "The business and affairs of every corporation organized under this chapter shall be managed by or under the direction of a board of directors.",
                    "citation": "8 Del. C. § 141(a)"
                },
                {
                    "id": "LEG-102",
                    "title": "Standard NDA Non-Solicitation Clause",
                    "text": "The Recipient agrees not to solicit or recruit any employee of the Disclosing Party for a period of twelve (12) months following contract termination.",
                    "citation": "Commercial NDA Template Section 4.2"
                }
            ],
            "Medical": [
                {
                    "id": "MED-201",
                    "title": "AHA Hypertension Management Guidelines 2026",
                    "text": "First-line pharmacotherapy for Stage 1 essential hypertension includes thiazide diuretics, calcium channel blockers, and ACE inhibitors or ARBs.",
                    "citation": "Am. Heart Assoc. Clinical Review 2026, p. 142"
                },
                {
                    "id": "MED-202",
                    "title": "Type 2 Diabetes Metformin Protocol",
                    "text": "Metformin remains the standard initial therapy. Renal function must be verified via eGFR prior to initiation, with caution advised if eGFR is below 45 mL/min.",
                    "citation": "ADA Standards of Care 2026, Sec. 9"
                }
            ],
            "Engineering & Code": [
                {
                    "id": "ENG-301",
                    "title": "FastAPI Async Middleware Best Practices",
                    "text": "Avoid blocking I/O calls inside async route handlers. Use run_in_threadpool or delegate heavy compute to Celery/Redis background workers.",
                    "citation": "Internal Python Architecture RFC #88"
                },
                {
                    "id": "ENG-302",
                    "title": "PostgreSQL Connection Pooling Guide",
                    "text": "Enforce PgBouncer transaction-mode pooling to avoid allocating dedicated server processes for ephemeral serverless lambdas.",
                    "citation": "Database Reliability Handbook v3"
                }
            ],
            "Customer Support": [
                {
                    "id": "SUP-401",
                    "title": "Global Refund & Chargeback Policy",
                    "text": "Customers are eligible for a 100% full refund within 30 calendar days of invoice date, provided usage has not exceeded 10,000 API tokens.",
                    "citation": "SaaS Customer Terms of Service Sec. 8"
                }
            ]
        }

    def route_and_search(self, domain: str, query: str) -> List[Dict[str, str]]:
        docs = self.domains.get(domain, [])
        query_words = set(query.lower().split())
        scored = []
        for doc in docs:
            doc_words = set((doc["title"] + " " + doc["text"]).lower().split())
            overlap = len(query_words.intersection(doc_words))
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]

# ==============================================================================
# 1. Architecture Blueprint
# ==============================================================================
st.header("1. Architecture Blueprint: Enterprise Multi-Domain RAG Application Topology", divider="rainbow")
st.markdown("""
In an enterprise deployment, a single monolithic vector index cannot handle diverse departmental workloads. 
Modern systems employ an **Intelligent Intent & Domain Router** that directs incoming queries to specialized, 
access-controlled sub-indices:
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_domain_mesh = """
    digraph DomainMesh {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="Tier 1: Enterprise Ingress Gateway";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            q [label="User / Microservice API Query", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            router [label="Semantic Domain Intent Router\\nClassifies Query & Enforces Tenant Security", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            q -> router;
        }

        subgraph cluster_domains {
            label="Tier 2: Specialized Enterprise RAG Fabrics";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            fontsize=12;
            leg [label="Legal Document RAG\\n(Clause-level chunking, strict precedent verification)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            med [label="Clinical & Healthcare RAG\\n(Ontology mapping, HIPAA guardrails, dosage verification)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            code [label="Codebase Intelligence RAG\\n(AST syntax-aware parsing, repository dependency graph)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
            cust [label="Customer Support RAG\\n(Semantic FAQ cache, sub-second latency SLA)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        }

        subgraph cluster_synthesis {
            label="Tier 3: Grounded Synthesis & Egress Audit";
            bgcolor="#1E293B";
            fontcolor="#EC4899";
            color="#334155";
            fontsize=12;
            llm [label="Domain-Conditioned LLM Generator\\nSynthesizes answer with strict attribution contracts", fillcolor="#831843", fontcolor="#FCE7F3", color="#EC4899"];
            out [label="Auditable Enterprise Response Delivered to User", fillcolor="#022C22", fontcolor="#A7F3D0", color="#10B981"];
            llm -> out;
        }

        router -> leg [label="Legal Intent"];
        router -> med [label="Clinical Intent"];
        router -> code [label="Code / Engineering Intent"];
        router -> cust [label="Customer Support Intent"];

        leg -> llm;
        med -> llm;
        code -> llm;
        cust -> llm;
    }
    """
    st.graphviz_chart(dot_domain_mesh, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern enterprise architectures have moved beyond single monolithic vector collections. Organizations deploy 
**Agentic Multi-RAG Mesh Architectures**, where a central orchestrator directs queries to distinct, specialized 
vector fabrics equipped with purpose-built chunking strategies (e.g. tree-sitter AST parsers for code repositories, 
table-aware vision parsers for financial SEC filings, and biomedical ontology graphs for clinical trials).
""")

st.divider()

# ==============================================================================
# 2. Document Question Answering
# ==============================================================================
st.header("2. Document Question Answering Systems", divider="rainbow")
st.markdown("""
**Document Q&A** is the foundational RAG application pattern: users upload complex PDFs, contracts, manuals, 
or specifications and query them conversational style.
* **Core Challenge**: Preserving cross-page reasoning, multi-column reading flow, and visual elements (tables, charts).
* **Key Innovation**: Vision-Language Model (VLM) chunking (e.g., ColPali) that embeds entire page images directly, eliminating brittle OCR extraction steps.
""")

st.divider()

# ==============================================================================
# 3. Enterprise Knowledge Assistants
# ==============================================================================
st.header("3. Enterprise Knowledge Assistants", divider="rainbow")
st.markdown("""
Internal enterprise assistants synthesize knowledge across heterogeneous corporate silos: Confluence, Google Docs, 
Slack, Notion, and Jira.
* **Core Challenge**: Access Control Lists (ACLs). An employee in Marketing must never retrieve executive payroll spreadsheets or unreleased HR investigations.
* **Architecture Solution**: Metadata-filtered vector search where user permission tokens are injected into the vector query filter at database execution time.
""")

with st.expander("🔍 View Pre-Retrieval Security Filtering Flowchart", expanded=True):
    dot_security = """
    digraph SecurityFilter {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        q [label="User Query + User Identity Token\\n(User: Alice, Role: Marketing)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        filter [label="Vector DB Pre-Filter Layer\\nEnforces: tenant_id == 'T1' AND access_role IN ['Public', 'Marketing']", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
        search [label="HNSW Graph Search\\nScans strictly permitted document vectors", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        safe_out [label="Zero Data Leakage Result\\nExecutive payroll vectors excluded before distance scoring", fillcolor="#047857", fontcolor="#ECFDF5", color="#10B981"];

        q -> filter -> search -> safe_out;
    }
    """
    st.graphviz_chart(dot_security, use_container_width=True)

st.divider()

# ==============================================================================
# 4. Customer Support
# ==============================================================================
st.header("4. Customer Support Automated Resolution Engines", divider="rainbow")
st.markdown("""
Customer-facing support bots resolve ticket queues autonomously by pulling troubleshooting manuals, return policies, 
and live order statuses.
* **Core SLA**: Ultra-low latency (< 1.5 seconds end-to-end) and zero brand hallucination.
* **Architecture Solution**: High-speed small embedding models, semantic caching of top-100 FAQs, and deterministic fallback to human agents when retrieval confidence falls below threshold.
""")

st.divider()

# ==============================================================================
# 5. Research Assistants
# ==============================================================================
st.header("5. Scientific & Academic Research Assistants", divider="rainbow")
st.markdown("""
Research assistants ingest thousands of scientific preprints (arXiv, PubMed) to assist researchers in literature 
reviews, methodology comparisons, and hypothesis generation.
* **Core Challenge**: Dense scientific terminology, mathematical notation, and dense multi-hop citation chains.
* **Architecture Solution**: GraphRAG—combining vector similarity with Knowledge Graphs to trace relationships across cited papers and authors.
""")

st.divider()

# ==============================================================================
# 6. Code Assistants
# ==============================================================================
st.header("6. Code Assistants & Repository Intelligence", divider="rainbow")
st.markdown("""
Modern developer assistants index entire codebases to assist in feature implementation, refactoring, and bug triage.
* **Core Challenge**: Natural language chunking breaks code logic.
* **Architecture Solution**: **AST-Aware Chunking** (using Tree-sitter) that divides files along class, function, and module boundaries, indexing symbol definitions and import graphs.
""")

st.divider()

# ==============================================================================
# 7. Legal Document Search
# ==============================================================================
st.header("7. Legal Document Search & Precedent Discovery", divider="rainbow")
st.markdown("""
Legal platforms process litigation filings, statutes, and contracts.
* **Core Challenge**: Extremely high cost of factual errors. A hallucinated precedent can result in court sanctions.
* **Architecture Solution**: Hybrid BM25 + dense search with exact quotation matching, clause-level chunking, and strict provenance displaying page and paragraph coordinates.
""")

st.divider()

# ==============================================================================
# 8. Medical Knowledge Systems
# ==============================================================================
st.header("8. Medical Knowledge Systems & Clinical Evidence", divider="rainbow")
st.markdown("""
Clinical decision support engines query peer-reviewed medical journals, drug interaction guides, and hospital protocols.
* **Core Challenge**: Zero tolerance for misinformation; patient safety and HIPAA compliance.
* **Architecture Solution**: Biomedical embeddings, strict terminology ontologies (UMLS, SNOMED CT), and mandatory human-in-the-loop review.
""")

st.divider()

# ==============================================================================
# 9. Educational Assistants
# ==============================================================================
st.header("9. Educational Assistants & Adaptive Tutoring", divider="rainbow")
st.markdown("""
AI tutors ingest textbook curricula and problem sets to deliver personalized Socratic guidance.
* **Core Challenge**: Providing hints and conceptual scaffolding rather than simply outputting direct homework answers.
* **Architecture Solution**: Pedagogical prompt contracts that constrain retrieval to curriculum standards and instruct the LLM to guide students through step-by-step reasoning.
""")

st.divider()

# ==============================================================================
# 10. Internal Company Search
# ==============================================================================
st.header("10. Internal Company Enterprise Search", divider="rainbow")
st.markdown("""
Enterprise search engines modernize traditional intranet search by providing direct synthesized answers with 
links rather than a generic list of ten blue links:
""")

df_app_matrix = pd.DataFrame([
    {"Application Domain": "Document Q&A", "Latency SLA": "< 3 sec", "Top Challenge": "Complex PDF tables & multi-column layouts", "Standard Retrieval": "Hierarchical Chunking + Cross-Encoder"},
    {"Application Domain": "Customer Support", "Latency SLA": "< 1 sec", "Top Challenge": "Stale policies & hallucinations", "Standard Retrieval": "Semantic FAQ Cache + Dense ANN"},
    {"Application Domain": "Code Assistants", "Latency SLA": "< 800 ms", "Top Challenge": "Non-linear dependency graphs", "Standard Retrieval": "AST Chunking + Symbol Graph Index"},
    {"Application Domain": "Legal Discovery", "Latency SLA": "< 5 sec", "Top Challenge": "100% precision & exact quotation", "Standard Retrieval": "Hybrid BM25 + Strict Verification"},
    {"Application Domain": "Clinical Decision", "Latency SLA": "< 2 sec", "Top Challenge": "Patient safety & HIPAA audit", "Standard Retrieval": "Ontology Graph + Biomedical Embeddings"}
])
st.dataframe(df_app_matrix, use_container_width=True)

st.divider()

# ==============================================================================
# 11. Production Implementation
# ==============================================================================
st.header("11. Production Implementation: Domain-Aware Enterprise RAG Router", divider="rainbow")
st.markdown("""
Below is a runnable Python production implementation of an **Intent Classifier & Domain Router** that 
analyzes incoming user queries, identifies the target domain, and executes domain-filtered retrieval:
""")

prod_router_code = """
from typing import Dict, List, Any

class EnterpriseRAGRouter:
    def __init__(self):
        # Keyword & intent signatures for domain routing
        self.domain_signatures = {
            "Legal": ["contract", "clause", "statute", "delaware", "nda", "liability", "breach"],
            "Medical": ["hypertension", "diabetes", "dosage", "clinical", "patient", "eGFR", "therapy"],
            "Engineering": ["fastapi", "kubernetes", "postgres", "async", "redis", "docker", "python"]
        }

    def route_query(self, query: str) -> str:
        \"\"\"Determines the appropriate domain sub-index.\"\"\"
        q_tokens = set(query.lower().split())
        best_domain = "General"
        max_overlap = 0
        for domain, keywords in self.domain_signatures.items():
            overlap = len(q_tokens.intersection(set(keywords)))
            if overlap > max_overlap:
                max_overlap = overlap
                best_domain = domain
        return best_domain

    def execute_routed_query(self, query: str, domain_store: Dict[str, List[str]]) -> Dict[str, Any]:
        target_domain = self.route_query(query)
        candidates = domain_store.get(target_domain, ["General enterprise knowledge base."])
        
        system_role = f"You are a specialized enterprise AI assistant for the {target_domain} domain."
        return {
            "query": query,
            "detected_domain": target_domain,
            "system_role": system_role,
            "retrieved_evidence": candidates[:2]
        }
"""
st.code(prod_router_code, language="python")

st.divider()

# ==============================================================================
# 12. Interactive Workbench
# ==============================================================================
st.header("12. Interactive Workbench: Multi-Domain Enterprise RAG Simulator", divider="rainbow")
st.markdown("""
Select an enterprise domain and test how domain-specialized retrieval extracts evidence 
with verified citations and domain-specific system contracts:
""")

store = DomainKnowledgeStore()

col_dom, col_ex = st.columns([1, 2])
with col_dom:
    active_domain = st.selectbox("Select Enterprise Domain:", ["Legal", "Medical", "Engineering & Code", "Customer Support"])
with col_ex:
    preset_queries = {
        "Legal": "What are board requirements under Delaware General Corporation Law?",
        "Medical": "What is the recommended first-line therapy for hypertension?",
        "Engineering & Code": "How should blocking I/O be handled in FastAPI async routes?",
        "Customer Support": "What is our customer refund policy window?"
    }
    query_input = st.text_input("Enterprise Query:", value=preset_queries[active_domain])

if st.button("Route & Retrieve Domain Knowledge", type="primary"):
    matched_docs = store.route_and_search(active_domain, query_input)
    st.markdown(f"### Target Domain: `{active_domain}`")
    if matched_docs:
        for d in matched_docs:
            st.success(f"**[{d['id']}] {d['title']}**\n\n\"{d['text']}\"\n\n*Citation Provenance:* `{d['citation']}`")
    else:
        st.warning("No domain evidence found matching the query tokens.")

st.divider()

# ==============================================================================
# 13. Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("13. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: How do you implement multi-tenant security in an enterprise RAG assistant?"):
    st.markdown("""
    **Answer**: Implement **pre-retrieval metadata filtering**. During ingestion, attach tenant IDs, user group permissions, 
    and document classification levels (Confidential, Public) as payload metadata to every vector. At query time, inject 
    the authenticated user's permission token into the vector database filter so the index only searches permitted vectors.
    """)

with st.expander("Q2: Why is AST-based chunking superior to fixed-size chunking for code search?"):
    st.markdown("""
    **Answer**: Code is inherently hierarchical, not linear. An Abstract Syntax Tree (AST) parser (via Tree-sitter) understands 
    class, function, and decorator boundaries. Fixed-token chunking splits functions in half, severing variable declarations 
    from return statements and destroying semantic cohesion.
    """)

with st.expander("Q3: In customer support RAG, how do you handle sub-second latency constraints?"):
    st.markdown("""
    **Answer**: Use a **tiered caching architecture**:
    1. Exact query cache for frequent questions (< 5ms).
    2. Semantic vector cache (cosine threshold > 0.96) for paraphrased FAQs (< 30ms).
    3. Low-latency hybrid retrieval using quantized HNSW vectors + lightweight embedding models.
    4. Fast streaming LLM output with speculative decoding.
    """)

with st.expander("Q4: What is GraphRAG and why is it essential for research assistants?"):
    st.markdown("""
    **Answer**: Standard vector RAG struggles with global thematic queries (e.g. *"What are the main research trends across all 500 papers?"*). 
    GraphRAG extracts entity-relation graphs and generates hierarchical community summaries, allowing the system to answer 
    both local pinpoint questions and global corpus-wide syntheses.
    """)

with st.expander("Q5: How do you guarantee zero-hallucination compliance in medical or legal RAG?"):
    st.markdown("""
    **Answer**: Enforce a multi-layered guardrail protocol:
    1. Strict negative prompt instructions (*"If not explicitly stated in context, reply 'Insufficient clinical evidence'"*).
    2. Fact-checking cross-encoders to verify generated sentences against retrieved chunks.
    3. Mandatory provenance highlighting with verbatim quote comparison before presenting the answer to end users.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (The Unfiltered Tenant Cross-Talk Trap)**: Failing to enforce hard tenant ID isolation at the vector database query layer, allowing User A from Tenant 1 to retrieve confidential financial plans from Tenant 2 via semantic similarity.")
    st.warning("**Trap 2 (Silent Token Expiration in Ingestion Connectors)**: OAuth refresh tokens expiring on Confluence/SharePoint connectors, causing the vector database to silently cease indexing new documents while the system continues serving outdated knowledge.")
with col_trap2:
    st.error("**Trap 3 (Code RAG Context Fragmentation)**: Chunking code files by character length instead of AST boundaries, causing syntax errors in retrieved code snippets and misleading developers with partial function signatures.")
    st.warning("**Trap 4 (The Single Global Index Bottleneck)**: Lumping marketing blogs, legal contracts, and engineering repositories into one single vector collection, leading to severe retrieval noise and poor relevance scores across all departments.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What is Pre-Retrieval Metadata Filtering?")
        st.caption("Flip for answer:")
        st.info("**Back**: The vector DB applies tenant and ACL constraints *before* calculating similarity distance, guaranteeing strict data isolation.")
    with st.container(border=True):
        st.markdown("**Front**: Why is AST-aware chunking necessary for code?")
        st.caption("Flip for answer:")
        st.info("**Back**: Tree-sitter parsers split code along function and class syntax boundaries rather than arbitrary character cuts, preserving complete logic.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How does Semantic FAQ Caching reduce latency?")
        st.caption("Flip for answer:")
        st.info("**Back**: Returns cached answers for queries with semantic similarity > 0.95, slashing latency and API cost from seconds to sub-30ms.")
    with st.container(border=True):
        st.markdown("**Front**: What is the primary advantage of GraphRAG?")
        st.caption("Flip for answer:")
        st.info("**Back**: Combines vector retrieval with entity-relation graphs for global thematic corpus reasoning and cross-document synthesis.")

st.divider()

# ==============================================================================
# 14. Self-Assessment Checklist
# ==============================================================================
st.header("14. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I can identify the unique chunking and latency requirements of Legal, Medical, Code, and Support RAG.", value=True)
c2 = st.checkbox("I know how to implement tenant-isolated metadata filtering to enforce enterprise security ACLs.", value=True)
c3 = st.checkbox("I understand why AST-based parsing is essential for high-fidelity code retrieval.", value=True)
c4 = st.checkbox("I know how to design a domain router that routes queries to specialized sub-indices.", value=True)
c5 = st.checkbox("I understand how semantic caching provides sub-second customer support response times.", value=True)
c6 = st.checkbox("I can design strict provenance and citation schemas for regulatory compliance.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 Chapter 1 Complete! You are ready to advance to Chapter 2: RAG Fundamentals.")
