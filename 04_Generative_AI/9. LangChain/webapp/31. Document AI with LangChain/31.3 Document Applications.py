import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="31.3 Document Applications - LangChain Architecture",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for domain application telemetry
st.markdown("""
<style>
    .app-doc-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-app-domain {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class DocumentApplicationEngine:
    """
    Simulation engine showcasing specialized extraction and analysis
    across 6 primary enterprise Document AI domains.
    """
    APP_DOMAINS = {
        "PDF Chatbot": {
            "use_case": "Interactive Multi-Turn Document QA with Page Jumping",
            "extracted_schema": "Conversational Response with [Page X] Markdown Anchors",
            "sample_output": "According to Section 3.2 on Page 14, customer data must be encrypted at rest using AES-256."
        },
        "Legal Document Assistant": {
            "use_case": "Jurisdiction Cross-Referencing & Liability Clause Auditing",
            "extracted_schema": "Structured Risk Assessment (Low / Medium / Critical)",
            "sample_output": "Clause 14.1 (Indemnity) imposes uncapped liability on the vendor without mutual exclusion."
        },
        "Research Assistant": {
            "use_case": "Multi-Paper Literature Synthesis & Empirical Benchmark Extraction",
            "extracted_schema": "Literature Matrix with DOI Validation",
            "sample_output": "Vaswani et al. (2017) demonstrated 28.4 BLEU on WMT 2014, surpassing RNN baselines by 2.0 BLEU."
        },
        "Resume Analyzer": {
            "use_case": "Candidate Skill Extraction & Job Description Fit Scoring",
            "extracted_schema": "Typed Pydantic Profile: YearsExp, TechStack, FitScore",
            "sample_output": "Candidate Profile: 6 yrs Python/ML, 3 yrs LangChain/VectorDBs. Match Score: 92/100."
        },
        "Invoice Analyzer": {
            "use_case": "Line-Item Financial Extraction & Automated ERP Reconciliation",
            "extracted_schema": "JSON Line-Items, Subtotal, Tax, Vendor VAT, Net Amount",
            "sample_output": "Invoice #9941: 3x GPU Cluster ($4,500), Tax 8.25% ($371.25), Total Due: $4,871.25."
        },
        "Contract Analyzer": {
            "use_case": "Obligation Tracking, Non-Compete Auditing, Redlining Diffs",
            "extracted_schema": "Termination Notice Period, Non-Compete Scope, Exclusivity",
            "sample_output": "Termination Clause: Requires 60 days written notice. Auto-renews annually unless canceled."
        }
    }

    def process_application(self, domain_key: str) -> Dict[str, Any]:
        info = self.APP_DOMAINS.get(domain_key, self.APP_DOMAINS["PDF Chatbot"])
        return {
            "domain": domain_key,
            "use_case": info["use_case"],
            "schema": info["extracted_schema"],
            "result": info["sample_output"]
        }


# --- LESSON HEADER ---
st.title("31.3 Enterprise Document AI Applications & Domain Workflows")
st.caption("Chapter 31: Document AI with LangChain | Domain: 9. LangChain Orchestration & LCEL")
st.markdown("""
Document AI moves beyond generic question-answering into high-value vertical automation.
From automated contract redlining in corporate legal departments to zero-touch invoice reconciliation
in enterprise ERP systems, specialized document architectures power critical business operations.

In this lesson, you will master:
1. **PDF Chatbots**: Multi-turn conversational retrieval with page citations and interactive jumping.
2. **Legal Document Assistants**: Jurisdiction-aware clause auditing and indemnification risk analysis.
3. **Research Assistants**: Automated multi-paper synthesis, methodology cross-checking, and citation extraction.
4. **Resume Analyzers**: Unbiased skill extraction, candidate profiling, and job-fit ranking schemas.
5. **Invoice Analyzers**: Vision/OCR line-item extraction, mathematical subtotal reconciliation, and accounting sync.
6. **Contract Analyzers**: Obligation tracking, non-compete detection, and automated redlining suggestions.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. End-to-End Architectural Blueprint: Enterprise Document AI Applications", divider="rainbow")
st.markdown("""
A production Document AI application combines document layout loaders, schema-enforced Pydantic extractors,
cross-referencing vector databases, and enterprise API sinks (ERP, CRM, ATS, Git).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    app_doc_dot = """
    digraph DocAppBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
        edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

        subgraph cluster_domain_docs {
            label="1. Ingress Document Streams";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            legal_doc [label="Legal Contract (NDA, MSA)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            finance_doc [label="Invoice / Financial Receipt", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            hr_doc [label="Candidate Resumes / CVs", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_specialized_extractors {
            label="2. Domain-Tuned LangChain Pipelines";
            bgcolor="#1E293B";
            fontcolor="#F59E0B";
            color="#334155";
            contract_engine [label="Contract Analyzer Pipeline\\nExtracts liabilities, indemnities, terms", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            invoice_engine [label="Invoice Vision/OCR Pipeline\\nExtracts table line items & taxes", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
            resume_engine [label="Resume Profiler Pipeline\\nExtracts skills, experience, projects", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
        }

        subgraph cluster_enterprise_sinks {
            label="3. Validated Structured Deliverables";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            sap_sink [label="ERP System (SAP / QuickBooks)\\nAutomated invoice bill booking", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            legal_report [label="Legal Risk Memo & Redline Diff\\nHighlights high-risk clauses", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
            ats_sink [label="ATS Platform (Workday / Greenhouse)\\nRanked candidate leaderboard", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];
        }

        legal_doc -> contract_engine [label="PDF / Word"];
        finance_doc -> invoice_engine [label="PDF / Image"];
        hr_doc -> resume_engine [label="PDF / Word"];
        contract_engine -> legal_report [label="Emits Risk Audit"];
        invoice_engine -> sap_sink [label="Validated JSON"];
        resume_engine -> ats_sink [label="Candidate Profile"];
    }
    """
    st.graphviz_chart(app_doc_dot, use_container_width=True)

# --- SECTION 2: PDF CHATBOT ---
st.header("2. PDF Chatbot", divider="rainbow")
st.markdown("""
A **PDF Chatbot** provides multi-turn conversational interaction over one or more uploaded PDF manuals,
research papers, or financial disclosures.
""")

col_pc1, col_pc2 = st.columns(2)
with col_pc1:
    st.markdown("""
    #### Production Architecture Requirements:
    - **Session-Scoped Memory**: Maintaining conversational history while isolating individual users (`ChatMessageHistory`).
    - **Clickable Page Citations**: Returning page numbers formatted as anchors so the user interface can scroll directly to the exact bounding box in the PDF viewer.
    - **Streaming Token Delivery**: Real-time token delivery using `chain.stream()` to ensure conversational responsiveness.
    - **Cross-Document Citations**: When searching across 10 PDFs, clearly attributing which specific document and page supplied each claim.
    """)

with col_pc2:
    with st.container(border=True):
        st.markdown("💬 **Citation-Preserving Prompt Pattern**")
        st.code("""
SYSTEM_PROMPT = \"\"\"
You are an expert PDF assistant.
Answer questions strictly based on the provided page context.
Always cite the source document and page number in brackets: [DocName, p. X].

<context>
{context}
</context>
\"\"\"
        """, language="python")

# --- SECTION 3: LEGAL DOCUMENT ASSISTANT ---
st.header("3. Legal Document Assistant", divider="rainbow")
st.markdown("""
**Legal Document Assistants** audit complex Master Service Agreements (MSAs), Non-Disclosure Agreements (NDAs),
and commercial contracts. In legal AI, precision is paramount; missing an uncapped liability clause or a non-compete
restriction can cost an organization millions of dollars.
""")

st.subheader("Key Legal Extraction Capabilities")
leg_c1, leg_c2, leg_c3 = st.columns(3)
with leg_c1:
    with st.container(border=True):
        st.markdown("⚖️ **Liability & Indemnity Audit**")
        st.markdown("Identifies whether liability caps are mutual, capped at contract value (12 months fees), or dangerously unlimited.")

with leg_c2:
    with st.container(border=True):
        st.markdown("📅 **Termination & Auto-Renewal**")
        st.markdown("Extracts required written notice periods (30 vs 60 vs 90 days) and flags evergreen auto-renewal traps.")

with leg_c3:
    with st.container(border=True):
        st.markdown("📍 **Governing Law & Jurisdiction**")
        st.markdown("Identifies governing law state (e.g. Delaware, New York, UK) and flags unfavorable dispute resolution venues.")

# --- SECTION 4: RESEARCH ASSISTANT ---
st.header("4. Research Assistant", divider="rainbow")
st.markdown("""
A **Research Assistant** processes multiple scientific papers simultaneously, synthesizing literature reviews,
comparing methodological baselines, and identifying empirical state-of-the-art results.
""")

col_ra1, col_ra2 = st.columns(2)
with col_ra1:
    st.markdown("""
    #### Core Architectural Capabilities:
    - **Cross-Paper Synthesis**: Compares Paper A and Paper B side-by-side, summarizing points of agreement and conflicting experimental results.
    - **Tabular Benchmark Extraction**: Extracts empirical results (e.g., accuracy, latency, FLOPs) across different model architectures into unified markdown tables.
    - **Automated Bibliography Generation**: Outputs formal BibTeX and APA citations with verified DOIs.
    """)

with col_ra2:
    with st.container(border=True):
        st.markdown("🔬 **Multi-Paper Synthesis Matrix**")
        st.markdown("""
        ```
        | Paper Title | Architecture | Parameter Count | Benchmark Result |
        | :--- | :--- | :---: | :---: |
        | Attention is All You Need | Transformer | 65M | 28.4 BLEU |
        | BERT: Pre-training | Transformer Enc | 340M | 80.5 GLUE |
        | LLaMA: Open Models | Autoregressive | 7B - 65B | Competitive with GPT-3 |
        ```
        """)

# --- SECTION 5: RESUME ANALYZER ---
st.header("5. Resume Analyzer", divider="rainbow")
st.markdown("""
**Resume Analyzers** parse unstructured candidate resumes in PDF and Word formats, extracting structured candidate profiles
and computing objective alignment scores against open Job Descriptions.
""")

col_res1, col_res2 = st.columns(2)
with col_res1:
    st.markdown("""
    #### Unbiased Structured Extraction Architecture:
    - **PII Scrubbing for Bias Mitigation**: Automatically masks candidate names, genders, graduation years, and physical addresses to enforce blind candidate evaluation.
    - **Skill Taxonomy Normalization**: Standardizes non-uniform terms (e.g., maps 'JS', 'ReactJS', and 'Node.js' to the unified 'JavaScript Ecosystem' taxonomy).
    - **Strict Pydantic Output**: Emits structured profiles conforming to corporate ATS API schemas.
    """)

with col_res2:
    with st.container(border=True):
        st.markdown("📋 **Candidate Profile Pydantic Schema**")
        st.code("""
from pydantic import BaseModel, Field
from typing import List

class CandidateProfile(BaseModel):
    total_experience_years: float
    primary_skills: List[str]
    secondary_skills: List[str]
    education_highest_degree: str
    jd_match_score: int = Field(ge=0, le=100)
    match_rationale: str
        """, language="python")

# --- SECTION 6: INVOICE ANALYZER ---
st.header("6. Invoice Analyzer", divider="rainbow")
st.markdown("""
**Invoice Analyzers** automate Accounts Payable operations by extracting vendor details, invoice numbers,
tax breakdowns, and individual line items from scanned and digital invoices.
""")

col_inv1, col_inv2 = st.columns(2)
with col_inv1:
    st.markdown("""
    #### Mathematical Reconciliation Guardrails:
    An invoice extraction model must not merely extract numbers; the application must enforce **deterministic mathematical reconciliation**:
    `Total Due = sum(Line Item Quantity * Unit Price) + Tax + Shipping`
    If the extracted line items do not sum to the extracted Total Due, the system raises an immediate flag for human accounting review.
    """)

with col_inv2:
    with st.container(border=True):
        st.markdown("🧾 **Structured Invoice Schema**")
        st.code("""
class InvoiceLineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    line_total: float

class InvoicePayload(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: str
    subtotal: float
    tax_amount: float
    total_amount_due: float
    line_items: List[InvoiceLineItem]
        """, language="python")

# --- SECTION 7: CONTRACT ANALYZER ---
st.header("7. Contract Analyzer", divider="rainbow")
st.markdown("""
**Contract Analyzers** perform deep semantic audits on commercial agreements, identifying deviations from corporate
fallback positions, missing mandatory privacy clauses, and proposing surgical redline edits.
""")

st.subheader("Automated Redlining Pipeline")
contract_dot = """
digraph ContractRedlining {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    third_party_contract [label="Third-Party Supplier MSA\\nContains 30-day payment term", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    playbook_retriever [label="Corporate Legal Playbook\\nRequires Net-60 payment term", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
    auditor_llm [label="Contract Auditor Node\\nCompares clause to standard playbook", fillcolor="#78350F", fontcolor="#FDE68A", color="#F59E0B"];
    redline_output [label="Redline Diff Deliverable\\n- Net-30 payment\\n+ Net-60 payment", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981"];

    third_party_contract -> auditor_llm [label="Target Clause"];
    playbook_retriever -> auditor_llm [label="Standard Policy"];
    auditor_llm -> redline_output [label="Generates Track Changes"];
}
"""
st.graphviz_chart(contract_dot, use_container_width=True)

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Structured Document Analyzer in LCEL", divider="rainbow")
st.markdown("""
Below is a runnable LangChain pipeline using `with_structured_output` to perform deterministic,
schema-validated entity extraction over business documents.
""")

prod_app_code = '''from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_core.prompts import ChatPromptTemplate

# 1. Define Strict Pydantic Extraction Schema
class ContractAuditReport(BaseModel):
    contract_title: str = Field(description="Name or title of the agreement.")
    governing_jurisdiction: str = Field(description="State or country governing the agreement.")
    risk_rating: Literal["LOW", "MEDIUM", "CRITICAL"] = Field(description="Overall legal risk rating.")
    indemnity_capped: bool = Field(description="True if indemnification is capped; False if unlimited.")
    flagged_clauses: List[str] = Field(description="Specific high-risk clauses requiring legal review.")

# 2. Build Extraction Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert corporate legal auditor. Analyze the contract text and extract a structured audit report."),
    ("human", "{contract_text}")
])

# 3. Production LCEL Structured Extraction Chain
# In production, bind with real ChatOpenAI / ChatAnthropic:
# extraction_chain = prompt | model.with_structured_output(ContractAuditReport)

def simulate_contract_audit(contract_text: str) -> ContractAuditReport:
    """Mock execution illustrating output schema adherence."""
    return ContractAuditReport(
        contract_title="Master Software Services Agreement",
        governing_jurisdiction="Delaware, USA",
        risk_rating="CRITICAL",
        indemnity_capped=False,
        flagged_clauses=[
            "Section 8.2: Uncapped indemnification on third-party IP claims.",
            "Section 11.4: Auto-renews annually without written notice requirement."
        ]
    )
'''
st.code(prod_app_code, language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Multi-Domain Document Application Simulator", divider="rainbow")
st.markdown("""
Simulate Document AI extraction and audit workflows across 6 primary enterprise domains.
Inspect the specialized schemas, domain extraction rules, and generated audit artifacts in real time.
""")

app_engine = DocumentApplicationEngine()

col_aw1, col_aw2 = st.columns([1, 1])

with col_aw1:
    st.subheader("💼 Document Application Domain")
    chosen_domain = st.selectbox(
        "Choose an Enterprise Document Application:",
        list(app_engine.APP_DOMAINS.keys())
    )
    run_app_btn = st.button("🚀 Execute Domain Document Analysis", type="primary", use_container_width=True)

with col_aw2:
    st.subheader("📊 Domain Extraction Telemetry")
    if run_app_btn:
        res = app_engine.process_application(chosen_domain)
        
        st.markdown(
            f"<div class='app-doc-card'><span class='badge-app-domain'>{chosen_domain.upper()}</span>"
            f"<br><br><b>Workflow Objective:</b> {res['use_case']}"
            f"<br><b>Output Schema:</b> <code>{res['schema']}</code></div>",
            unsafe_allow_html=True
        )

        st.markdown("##### 📄 Extracted Structured Finding")
        st.info(f"**Domain Analysis Result:** {res['result']}")
        st.success("✅ Audit Passed: Schema validated with 100% type conformance.")
    else:
        st.info("Select an application domain and click 'Execute Domain Document Analysis'.")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Architectural Interview Questions")
with st.expander("Q1: How do you prevent hallucination when building a customer-facing PDF Chatbot?"):
    st.markdown("""
    **Answer**: (1) **Prompt Anchoring**: Strict instructions commanding the model to refuse to answer if the information is missing from the context;
    (2) **Citation Enforcement**: Demanding explicit page and paragraph references for every claim;
    (3) **Confidence Thresholding**: If the top retrieved chunks have low semantic similarity scores (e.g. cosine distance > 0.4), abort generation and declare no match;
    and (4) **Self-RAG Grounding**: Running a post-generation checker that verifies every sentence against the retrieved chunks.
    """)

with st.expander("Q2: Why must Invoice Analyzers combine LLMs with deterministic mathematical validation?"):
    st.markdown("""
    **Answer**: Large Language Models are probabilistic text generators, not arithmetic calculators. While an LLM excels at extracting
    numbers from confusing visual layouts, it can easily make minor arithmetic hallucinations (e.g., claiming 10 + 5 = 14).
    In production, the LLM extracts the numbers into a Pydantic schema, and deterministic Python code recalculates the sums,
    asserting that `quantity * price == line_total` and `subtotal + tax == total_due`.
    """)

with st.expander("Q3: How do Resume Analyzers mitigate unconscious algorithmic bias?"):
    st.markdown("""
    **Answer**: By enforcing a two-stage blinded pipeline. Stage 1 scrubs all personally identifiable attributes: candidate name,
    email, physical location, gender markers, graduation years (to mitigate age bias), and school names (to mitigate institutional bias).
    Stage 2 evaluates candidate fit purely on demonstrated skills, architectural achievements, and project complexities against
    the job description criteria.
    """)

with st.expander("Q4: What is the Redline Diff pattern in legal contract assistants?"):
    st.markdown("""
    **Answer**: Instead of rewriting an entire contract, a Redline Assistant compares third-party contract clauses against an internal
    corporate legal playbook. When a clause violates policy (e.g., net-30 payment when company policy demands net-60), the assistant
    outputs a surgical unified diff showing the deletion of unfavorable terms and insertion of approved fallback language.
    """)

with st.expander("Q5: How does a Research Assistant evaluate conflicting claims across multiple papers?"):
    st.markdown("""
    **Answer**: By structuring extraction around empirical conditions. If Paper A claims an algorithm achieves 85% accuracy and
    Paper B claims only 72%, the assistant extracts the experimental metadata (dataset used, hardware platform, hyperparameter settings).
    It then outputs a comparative analysis highlighting that Paper A evaluated on clean synthetic data while Paper B evaluated on
    noisy real-world data, synthesizing the nuanced divergence rather than presenting a contradictory answer.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Blind Financial Trust)**: Booking invoices directly into an ERP without deterministic code asserting that line items sum up to total due. One hallucinated digit can result in massive accounting discrepancies.")
    st.warning("**Trap 2 (Omission of PDF Page Jump Anchors)**: Presenting users with an answer without citing the exact page number, forcing users to manually read 100 pages to verify whether the AI's answer was accurate.")
with col_trap2:
    st.error("**Trap 3 (Parsing Legal Contracts as Unstructured Chunks)**: Slicing legal contracts with arbitrary 500-token character limits. Slicing an indemnification clause in half strips its exceptions, reversing its legal meaning.")
    st.warning("**Trap 4 (Unconstrained Free-Text Resumes)**: Generating qualitative free-text resume reviews rather than strict Pydantic schemas, making it impossible to sort or filter candidates programmatically.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: Why is mathematical reconciliation mandatory in Invoice Analyzers?")
        st.caption("Flip for answer:")
        st.info("**Back**: Because LLMs are probabilistic generators and can hallucinate arithmetic; deterministic code must verify that itemized totals match the invoice balance.")
    with st.container(border=True):
        st.markdown("**Front**: How do Legal Document Assistants audit risk?")
        st.caption("Flip for answer:")
        st.info("**Back**: By cross-referencing extracted contract clauses (liability caps, termination terms) against a corporate policy playbook.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the Blind Evaluation pattern in Resume Analysis?")
        st.caption("Flip for answer:")
        st.info("**Back**: Stripping names, gender indicators, graduation dates, and personal data before scoring to prevent algorithmic demographic bias.")
    with st.container(border=True):
        st.markdown("**Front**: What is the Redline Diff pattern in contract automation?")
        st.caption("Flip for answer:")
        st.info("**Back**: Producing surgical strike-through and addition edits to align third-party contracts with internal corporate legal standards.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand how to build conversational PDF chatbots with clickable page citations.", value=True)
c2 = st.checkbox("I know how legal assistants audit indemnification and liability clause risk.", value=True)
c3 = st.checkbox("I understand multi-paper research synthesis and benchmark extraction.", value=True)
c4 = st.checkbox("I know how to implement blinded resume scoring using structured Pydantic schemas.", value=True)
c5 = st.checkbox("I understand why deterministic mathematical reconciliation is mandatory for invoice AI.", value=True)
c6 = st.checkbox("I recognize how automated redlining diffs assist corporate legal teams.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You have completed Chapter 31: Document AI with LangChain.")
