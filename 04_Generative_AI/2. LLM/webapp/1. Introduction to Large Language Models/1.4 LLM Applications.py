import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="1.4 LLM Applications - LLM Curriculum",
    page_icon="💼",
    layout="wide"
)

# ==============================================================================
# Title Block & Overview
# ==============================================================================
st.title("1.4 Enterprise LLM Application Topologies: Architecture Archetypes & Deployment Patterns")
st.caption("Chapter 1: Introduction to Large Language Models | Domain: 2. Large Language Models")

st.markdown("""
Large Language Models have transitioned from isolated research curiosities into the core operational fabric 
of modern software architecture. Deploying LLMs at enterprise scale requires engineering **specialized application topologies**: 
from conversational customer chatbots and IDE-integrated coding copilots to hybrid neural search engines, 
autonomous multi-tool agents, and high-throughput document extraction pipelines.

Building production-grade applications requires mastering the interplay between **Foundation Models, Retrieval-Augmented 
Generation (RAG), Role-Based Access Control (RBAC), Tool Execution Sandboxes, and Deterministic Financial Guardrails**.

In this lesson, you will master:
1. The 8 enterprise application archetypes that define production AI deployments.
2. How modern foundation models power chatbots, coding copilots, neural search, and autonomous agents.
3. Strict SLA latency budgets, token consumption profiles, and infrastructure dependencies.
4. Production role-based access control (RBAC) pre-filtering and deterministic financial guardrails in Python.
5. Critical deployment traps including prompt injection, data poisoning, and tool runaway loops.
""")

st.divider()

# ==============================================================================
# Section 1: Enterprise LLM Application Topologies & Architectural Taxonomy
# ==============================================================================
st.header("1. Enterprise LLM Application Topologies & Architectural Taxonomy", divider="rainbow")

st.markdown("""
Every production LLM workload maps to one of eight core architectural archetypes. 
Each archetype imposes distinct latency budgets, safety thresholds, and infrastructure dependencies:
""")

col_top1, col_top2 = st.columns([1.3, 0.7])

with col_top1:
    archetypes_dot = """
    digraph ApplicationArchetypes {
        bgcolor="#0B1120";
        rankdir=TB;
        compound=true;
        nodesep=0.35;
        ranksep=0.65;

        node [
            fontname="Helvetica",
            fontsize=10,
            fontcolor="#F8FAFC",
            shape=box,
            style="filled,rounded",
            color="#334155",
            penwidth=1.5
        ];

        edge [
            fontname="Helvetica",
            fontsize=8,
            fontcolor="#94A3B8",
            color="#38BDF8",
            penwidth=1.6,
            arrowsize=0.7
        ];

        subgraph cluster_enterprise {
            label="Enterprise Core";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            FoundationCore [
                label="Foundation LLM Core\\n(Pretrained Weights + Context Memory)",
                fillcolor="#0F172A",
                color="#818CF8",
                shape=hexagon,
                penwidth=2.2,
                fontsize=11
            ];
        }

        subgraph cluster_interactive {
            label="Interactive & Conversational Tier";
            style=filled;
            color="#1E293B";
            fontcolor="#34D399";
            A1 [label="1. Conversational Chatbots\\n(Stateful dialog, persona grounding)", fillcolor="#064E3B", color="#34D399"];
            A2 [label="2. Context-Aware Coding Copilots\\n(AST parsing, FIM, test repair)", fillcolor="#064E3B", color="#34D399"];
            A6 [label="6. Customer Support Concierges\\n(Cognitive triage, omnichannel)", fillcolor="#064E3B", color="#34D399"];
        }

        subgraph cluster_knowledge {
            label="Knowledge & Retrieval Tier";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            A3 [label="3. Hybrid Neural Search\\n(Dense vector + BM25 + Rerank)", fillcolor="#0C4A6E", color="#38BDF8"];
            A4 [label="4. Recommendation Engines\\n(Latent embeddings, cold-start)", fillcolor="#0C4A6E", color="#38BDF8"];
            A5 [label="5. Structured Document Extraction\\n(Multimodal OCR, Pydantic schemas)", fillcolor="#0C4A6E", color="#38BDF8"];
        }

        subgraph cluster_autonomous {
            label="Autonomous & Governance Tier";
            style=filled;
            color="#1E293B";
            fontcolor="#F59E0B";
            A7 [label="7. Autonomous AI Agents\\n(ReAct loops, sandboxed tools)", fillcolor="#78350F", color="#F59E0B"];
            A8 [label="8. Enterprise Governance Fabric\\n(RBAC firewalls, audit logging)", fillcolor="#78350F", color="#F59E0B"];
        }

        FoundationCore -> A1;
        FoundationCore -> A2;
        FoundationCore -> A3;
        FoundationCore -> A4;
        FoundationCore -> A5;
        FoundationCore -> A6;
        FoundationCore -> A7;
        FoundationCore -> A8;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(archetypes_dot, use_container_width=True)

with col_top2:
    st.markdown("### The 8 Enterprise Archetypes")
    st.markdown("""
    * **1. Chatbots**: Multi-turn dialogue with memory buffers and system steerability.
    * **2. Coding Assistants**: In-IDE code generation, Fill-In-The-Middle (FIM), and AST-aware context.
    * **3. Search Systems**: Hybrid BM25 + dense vector retrieval with Cross-Encoder reranking.
    * **4. Recommendation Systems**: Latent semantic preference modeling and cold-start content synthesis.
    * **5. Document Analysis**: High-throughput multi-page document parsing into strict JSON schemas.
    * **6. Customer Support**: Autonomous tier-1 triage and human escalation workflows.
    * **7. AI Agents**: Multi-step tool-calling, ReAct loops, and sandboxed code execution.
    * **8. Enterprise AI Systems**: Multi-tenant RBAC security, rate limits, and audit firewalls.
    """)

st.divider()

# ==============================================================================
# Section 2: 1. Chatbots
# ==============================================================================
st.header("2. Chatbots: Conversational State, Memory Buffers & Persona Grounding", divider="rainbow")

st.markdown("""
Chatbots are stateful interactive systems designed for multi-turn human-computer conversation. 
Unlike stateless API calls where each request is independent, a chatbot must maintain a rolling history 
of preceding exchanges while strictly observing corporate persona guidelines, safety guardrails, and domain boundaries.

### Conversational Architecture & Memory Sizing
At each turn `t`, the input to the foundation model is an ordered message array:

`Messages = [System_Message, User_Message_1, Assistant_Message_1, ..., User_Message_t]`

As conversation length grows, the total token count approaches the context limit:
1. **Sliding Window Buffers**: Retains strictly the last `K` turns (e.g. `K = 10`), discarding older turns.
2. **Rolling Summary Compaction**: When tokens exceed 80% of budget, a background LLM call summarizes turns 1 through `t-5` into a dense narrative block injected into the system prompt.
3. **Vector-Augmented Session Memory**: Older turns are embedded into a session vector index; relevant past conversational context is retrieved dynamically based on the current user query.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Pre-Trained Chat Templates**: Modern foundation models (Llama 3, Mistral, Qwen) are instruction-tuned with dedicated chat template tokens (e.g., `<|start_header_id|>system<|end_header_id|>`, `<|eot_id|>`). Runtimes apply Jinja2 chat templates automatically, preventing role spoofing and prompt injection.
* **Streaming Token Generation**: Production chatbots stream tokens via Server-Sent Events (SSE) or WebSockets. This achieves a Time-To-First-Token (TTFT) under 300ms, providing an immediate responsive experience while remaining tokens stream at 40-80 tokens/second.
* **Multi-Turn Context Caching**: Modern API providers (OpenAI, Anthropic, Gemini) cache static system prompts and conversation history across turns, reducing per-turn token costs and prefill latency by 50% to 80%.
""")

df_a1 = pd.DataFrame({
    "Dimension": [
        "State Management",
        "Context Retention Strategy",
        "Role Separation & Injection",
        "Latency Profile (TTFT)",
        "Safety Enforcement"
    ],
    "Legacy Chatbots (Rasa / Dialogflow)": [
        "Predefined Finite State Machines (FSM) and rigid slot variables",
        "Fixed slot memory; cleared on unhandled or off-topic user input",
        "Rule-based string matching; easily tricked by adversarial prompts",
        "Fast (under 100ms on CPU) but strictly canned scripted responses",
        "Regex profanity filters; zero semantic awareness of toxic nuance"
    ],
    "Modern Foundation Chatbots (Llama 3 / Claude 3.5)": [
        "Autoregressive decoder over structured message array with Jinja templates",
        "Sliding windows + dynamic rolling summarization + prompt caching",
        "Cryptographically bounded special tokens (<|start_header_id|>) preventing spoofing",
        "Sub-300ms TTFT via streaming SSE tokens with KV-cache persistence",
        "Real-time semantic safety classifiers (Llama Guard, NeMo Guardrails)"
    ]
})
st.dataframe(df_a1, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: 2. Coding Assistants
# ==============================================================================
st.header("3. Coding Assistants: AST Graphs, Fill-In-The-Middle & Repository Context", divider="rainbow")

st.markdown("""
Coding assistants (such as GitHub Copilot, Cursor, and Continue) embed directly inside developer IDEs 
to provide real-time code completion, multi-file refactoring, docstring authoring, and unit test generation. 
Code differs from natural language: it requires strict syntax validity, long-range variable bindings across modular imports, 
and sub-200ms completion latencies.

### Fill-In-The-Middle (FIM) & Repository Graphs
Standard LLMs generate code only from left-to-right. However, a developer typically codes in the middle of an existing file:
* **Prefix**: The code written above the cursor.
* **Suffix**: The code written below the cursor (closing brackets, downstream callers).
* **FIM Ingestion**: The model is trained on transformed sequence triplets: `<|fim_prefix|> Prefix <|fim_suffix|> Suffix <|fim_middle|> Generated_Code`.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Tree-sitter AST Context Injection**: Modern coding tools parse the file's Abstract Syntax Tree (AST) in real time. If the cursor is inside a function that references a class `UserAuth`, the IDE retrieves the class declaration from `models/auth.py` and injects it into the prompt prefix.
* **Speculative In-IDE Decoding**: In-editor tab completions run ultra-compact 1B to 3B parameter models (e.g., DeepSeek-Coder-1.3B, Qwen-Coder-1.5B) locally or on low-latency inference endpoints, achieving completions in under 80 milliseconds.
* **Closed-Loop Sandbox Repair**: Production coding agents execute generated code inside ephemeral Docker or WebAssembly sandboxes, capture unit test failures or Python stack traces, and feed them back to the model to autonomously repair errors before showing the solution to the developer.
""")

df_a2 = pd.DataFrame({
    "Dimension": [
        "Completion Paradigm",
        "Context Awareness",
        "Insertion Capability",
        "Latency SLA (Editor Tab)",
        "Refactoring Scope"
    ],
    "Legacy IDE Auto-Complete (IntelliSense)": [
        "Static type reflection and symbol lookup tables",
        "Limited to loaded symbols and explicit namespace imports",
        "Append-only at cursor position; static snippet templates",
        "Extremely fast (under 10ms on local CPU)",
        "Single-symbol renaming or basic manual method extraction"
    ],
    "Modern Foundation Code Copilots": [
        "Probabilistic next-token code synthesis across 80+ programming languages",
        "Cross-file repository context via AST parsing and semantic embeddings",
        "Native Fill-In-The-Middle (FIM) predicting code conditioned on prefix and suffix",
        "Sub-100ms via compact distilled draft models and speculative execution",
        "Autonomous multi-file refactoring, test suite generation, and legacy migration"
    ]
})
st.dataframe(df_a2, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: 3. Search Systems
# ==============================================================================
st.header("4. Search Systems: Neural Dense Retrieval, BM25 Hybrid Fusion & Cross-Encoder Reranking", divider="rainbow")

st.markdown("""
Enterprise search systems connect knowledge workers to millions of unstructured internal documents. 
Modern neural search transcends keyword matching (ElasticSearch / BM25) by encoding documents and queries 
into high-dimensional dense vector spaces where conceptual semantic proximity is measured by cosine similarity.

### The Two-Stage Hybrid Search Pipeline
1. **Stage 1: Candidate Generation (Hybrid Retrieval)**:
   * **Dense Retrieval**: Bi-Encoder embeddings (`text-embedding-3`, BGE-M3) capture semantic intent, synonyms, and multilingual queries.
   * **Sparse Retrieval**: BM25 inverted index captures exact keywords, part numbers, acronyms, and error codes.
   * **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse ranked lists into a unified candidate pool of 50 to 100 documents:
     `RRF_Score(d) = 1.0 / (k + Rank_Dense(d)) + 1.0 / (k + Rank_Sparse(d))`
2. **Stage 2: Precision Reranking**:
   * A Cross-Encoder model (e.g. Cohere Rerank, BGE-Reranker) processes query and document pairs jointly through full cross-attention layers, computing fine-grained relevance scores and passing the top 5 to 10 passages to the generative LLM.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Eliminating Lexical Blind Spots**: BM25 alone fails when a user searches for `"cardiac arrest protocols"` if the document uses `"myocardial infarction emergency guide"`. Dense embeddings resolve this synonymy natively. Conversely, dense search fails on specific part numbers like `"Part-XZ-9812-B"`; hybrid BM25 ensures exact code matches are never dropped.
* **Vector Database Scale**: Modern architectures deploy specialized vector databases (Milvus, Pinecone, Qdrant, ChromaDB) utilizing HNSW (Hierarchical Navigable Small World) graph indexing to query millions of vectors in under 15 milliseconds.
* **Synthesized Grounded Answers**: Rather than returning a list of blue links, the foundation LLM reads the top reranked chunks and synthesizes an authoritative executive summary with inline source citations.
""")

df_a3 = pd.DataFrame({
    "Dimension": [
        "Retrieval Primitive",
        "Semantic Generalization",
        "Exact Match Handling",
        "Reranking Architecture",
        "User Experience"
    ],
    "Legacy Keyword Search (ElasticSearch / BM25)": [
        "Inverted index term frequency (TF-IDF / BM25)",
        "Zero; fails on synonyms, paraphrasing, and cross-lingual queries",
        "Excellent; captures exact serial numbers, error codes, and names",
        "Heuristic BM25 scoring or shallow learning-to-rank trees",
        "List of 10 blue document links with highlighted keyword snippets"
    ],
    "Modern Hybrid Neural Search (RAG Engine)": [
        "Hybrid Dense Vector + Sparse Inverted Index + Reciprocal Rank Fusion",
        "High; maps conceptual intent across 100+ languages and diverse phrasing",
        "Preserved 100% via BM25 hybrid channel integration",
        "Cross-Encoder neural reranker computing deep token-level cross-attention",
        "Synthesized natural language answer with verified inline document citations"
    ]
})
st.dataframe(df_a3, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: 4. Recommendation Systems
# ==============================================================================
st.header("5. Recommendation Systems: Latent Preference Alignment, Cold-Start Synthesis & Two-Tower Retrieval", divider="rainbow")

st.markdown("""
Recommendation systems predict user preferences over vast catalogs of items (e-commerce products, streaming media, job listings). 
Foundation models revolutionize recommendation architectures by bridging two historical failure modes: 
**the cold-start problem** (recommending brand-new items with zero user interaction history) and **unstructured catalog comprehension**.

### Generative Recommendation Mechanisms
1. **Semantic Two-Tower Projections**: User interaction histories and rich product text descriptions are mapped into shared latent spaces using foundation embedding encoders.
2. **Cold-Start Content Synthesis**: When a new product is added, the LLM reads its raw unstructured supplier specifications, generates dense semantic embeddings, extracts structured tags, and places it into the recommendation graph immediately—requiring zero historical clicks.
3. **Conversational Re-Ranking**: Instead of passive feeds, modern systems allow users to interactively steer their recommendations (`"Show me these running shoes, but in waterproof trail versions under $120"`).
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Zero-Shot Catalog Alignment**: Traditional matrix factorization (Collaborative Filtering) required dense user-item interaction matrices. Foundation models perform zero-shot item matching by understanding domain relationships (e.g., knowing that a customer buying a DSLR camera also needs an SD card and lens cleaner).
* **Explainable Recommendations**: In production, LLMs generate personalized justification narratives (`"Recommended because you frequently listen to ambient electronic music with low tempo"`), dramatically increasing user engagement and click-through rates (CTR).
""")

df_a4 = pd.DataFrame({
    "Dimension": [
        "Core Modeling Engine",
        "Cold-Start Item Handling",
        "Feature Representation",
        "Steerability",
        "Explainability"
    ],
    "Legacy Collaborative Filtering (Matrix Factorization)": [
        "Matrix Factorization (SVD / ALS) over user-item click matrices",
        "Severe failure; cannot recommend items with zero past interaction data",
        "Sparse one-hot IDs; blind to product descriptions, images, or reviews",
        "Passive; user cannot converse or redirect recommendations in real time",
        "Black-box dot product score; zero natural language explanation"
    ],
    "Modern Foundation Generative Recommenders": [
        "Hybrid Two-Tower Dense Encoders + LLM Conversational Re-rankers",
        "Instant zero-shot indexing based on rich unstructured text and metadata",
        "Deep semantic representations capturing nuances, style, and intent",
        "Interactive multi-turn steering via conversational user prompts",
        "Personalized, transparent natural language explanations for every item"
    ]
})
st.dataframe(df_a4, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: 5. Document Analysis
# ==============================================================================
st.header("6. Document Analysis: Long-Context Compaction, Multimodal OCR & Zero-Shot Schema Extraction", divider="rainbow")

st.markdown("""
Document analysis pipelines process unstructured corporate documents—invoices, regulatory filings (10-K), legal contracts, 
and medical encounter charts—transforming messy PDFs and scans into validated relational database records.

### The Modern Document Extraction Pipeline
1. **Multimodal Visual Ingestion**: Modern vision-language foundation models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5) ingest raw document page images directly, bypassing fragile intermediate OCR text extraction.
2. **Layout & Table Comprehension**: The model reads tabular borders, multi-column layouts, handwritten signatures, and footnotes natively.
3. **Grammar-Guided Pydantic Extraction**: Logit masks enforce strict JSON Schema compliance, ensuring extracted data types (dates, currencies, float amounts) are 100% syntactically valid before database insertion.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Elimination of Template OCR Rules**: Legacy OCR systems (Kofax, ABBYY) required manual template authoring for every vendor layout. Foundation models generalize across thousands of invoice layouts zero-shot.
* **Million-Token Document Scans**: With context windows spanning hundreds of thousands of tokens, entire 200-page loan dossiers or discovery folders can be ingested in a single call, allowing cross-page reconciliation (e.g. verifying that a figure on page 12 matches the summary balance sheet on page 184).
""")

df_a5 = pd.DataFrame({
    "Dimension": [
        "Extraction Mechanism",
        "Layout Flexibility",
        "Multi-Page Reconciliation",
        "Schema Definition",
        "Handwriting & Visual Understanding"
    ],
    "Legacy Template OCR (ABBYY / Tesseract)": [
        "Pixel coordinate bounding boxes and brittle rule-based regex",
        "Fails whenever vendors shift field positions or change column counts",
        "Processes pages in isolation; cannot reconcile data across 100 pages",
        "Hardcoded post-processing regex; prone to unhandled null pointers",
        "Severely degrades on cursive, rotated scans, stamps, or watermarks"
    ],
    "Modern Multimodal Foundation Models": [
        "Direct visual-language token processing with grammar-constrained decoding",
        "Zero-shot layout invariance; understands semantic relationships anywhere on page",
        "Native multi-page context; reconciles claims across 200+ pages in 1 call",
        "Pydantic Python classes compiled to JSON Schema grammar masks",
        "High accuracy on handwritten notes, multi-column tables, and complex charts"
    ]
})
st.dataframe(df_a5, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: 6. Customer Support
# ==============================================================================
st.header("7. Customer Support: Cognitive Triage, Escalation Routing & Omnichannel Coherence", divider="rainbow")

st.markdown("""
Customer support platforms deploy foundation models as tier-1 automated concierges and tier-2 copilot assistants. 
Beyond simple Q&A, customer support systems must perform **cognitive triage**: evaluating sentiment, detecting churn risk, 
verifying user authentication, executing self-service resolutions (refunds, cancellations), and escalating to human agents 
when edge cases occur.

### Multi-Tier Escalation Architecture
* **Tier 1 (Autonomous Resolution)**: Solves high-frequency, low-risk requests (order status, policy queries, basic troubleshooting) using strictly bounded tool calls.
* **Tier 2 (Human Copilot Drafting)**: When complex inquiries occur, the LLM reads the complete customer history, summarizes the issue for the human agent, and drafts a proposed response for human review.
* **Tier 3 (Deterministic Guardrails)**: Sensitive financial actions (refunds > $50, account deletions) are blocked from autonomous execution and routed to supervisor approval queues.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Empathy Modulation & Tone Steerability**: System prompts dynamically adjust the conversational tone based on customer sentiment score: adopting a deferential, apologetic tone for frustrated VIP customers and an efficient, direct tone for technical inquiries.
* **Automated Ticket Summarization**: At the conclusion of a conversation, the model writes a structured ticket resolution note (`Category: Billing_Dispute, Root_Cause: Duplicate_Charge, Resolution: Refund_Issued`), updating the CRM without requiring manual agent documentation.
""")

df_a6 = pd.DataFrame({
    "Dimension": [
        "Triage Capability",
        "Resolution Scope",
        "Escalation Mechanism",
        "Tone Adaptability",
        "CRM Integration"
    ],
    "Legacy Support Bots (IVR / Keyword Rules)": [
        "Keyword matching into fixed problem categories (e.g., 'Press 1 for Billing')",
        "Links to static FAQ articles; zero ability to execute actions",
        "Arbitrary escalation when user repeatedly types 'Agent' or 'Human'",
        "Monotone robotic canned messages regardless of user frustration",
        "Manual agent notes required after every call; zero automated synthesis"
    ],
    "Modern Foundation Support Concierges": [
        "Multi-dimensional cognitive triage (sentiment, intent, churn risk, urgency)",
        "Executes self-service transactions via secure API tool-calling",
        "Intelligent escalation with synthesized executive context briefs for agents",
        "Dynamic real-time empathy modulation based on sentiment trajectory",
        "Automated post-call CRM ticket drafting and structured field extraction"
    ]
})
st.dataframe(df_a6, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: 7. AI Agents
# ==============================================================================
st.header("8. AI Agents: Autonomous Tool Calling, ReAct Loops, Plan-and-Solve & Sandboxed Execution", divider="rainbow")

st.markdown("""
An **AI Agent** transcends passive text prediction by pairing a foundation model with **reasoning loops, memory stores, 
and external tool execution environments**. Agents operate autonomously over multi-step tasks by generating thought plans, 
invoking APIs or database queries, observing results, and iterating until the high-level objective is accomplished.

### The ReAct (Reason + Act) Execution Loop
1. **Thought**: The LLM reasons about the current state and decomposes the next required sub-action.
2. **Action**: The LLM emits a structured function call payload (e.g., `execute_sql_query(query="SELECT ...")`).
3. **Observation**: The execution environment runs the function and injects the raw observation back into the context window as a `ToolMessage`.
4. **Loop / Termination**: The LLM evaluates the observation. If more steps are needed, it repeats the loop; if complete, it outputs the final answer.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Native Tool-Calling Finetuning**: Modern models (GPT-4o, Claude 3.5, Llama-3-Tool) are trained with native function-calling formats. Runtimes pass tool schemas directly in the API request, and models emit syntactically valid JSON tool arguments zero-shot.
* **Ephemeral Sandboxed Execution**: In production, code and terminal tools are never executed on host servers. They run in network-isolated, ephemeral WebAssembly (Wasm) or Docker sandboxes with strict CPU, memory, and timeout bounds.
* **Loop runaway Protection**: Enterprise agent orchestrators enforce hard ceilings: maximum iteration caps (e.g. 10 loops), tool budget limits, and human-in-the-loop (HITL) signoffs for high-impact actions.
""")

df_a7 = pd.DataFrame({
    "Dimension": [
        "Execution Architecture",
        "Tool Selection Mechanism",
        "Error Self-Correction",
        "Sandboxing Requirements",
        "Autonomous Decision Depth"
    ],
    "Legacy Automation (RPA / Python Scripts)": [
        "Rigid, hardcoded DAG workflows (Airflow, UiPath)",
        "Deterministic conditional if-else statements",
        "Crashes immediately on unhandled exceptions or altered API schemas",
        "Runs with full host OS privileges; vulnerable to script errors",
        "Zero reasoning; strictly follows predefined static paths"
    ],
    "Modern Foundation AI Agents": [
        "Autonomous ReAct loops with dynamic multi-step planning and backtracking",
        "Zero-shot tool routing based on natural language function docstrings",
        "Inspects runtime error traces, reformulates queries, and self-repairs",
        "Strict ephemeral containerization with network isolation and timeout bounds",
        "Multi-step symbolic problem solving with human-in-the-loop signoffs"
    ]
})
st.dataframe(df_a7, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: 8. Enterprise AI Systems
# ==============================================================================
st.header("9. Enterprise AI Systems: RBAC Authorization, Multi-Tenant Isolation & Observability Firewalls", divider="rainbow")

st.markdown("""
Enterprise AI systems encompass the complete end-to-end operational architecture required to deploy foundation models safely, 
compliantly, and reliably across Fortune 500 organizations. 
An enterprise system is not just an LLM API call; it is an integrated ecosystem comprising **Role-Based Access Control (RBAC), 
multi-tenant data isolation, rate limiting, cryptographic audit logging, and dual security firewalls**.

### The Enterprise Defense-in-Depth Architecture
1. **Pre-Execution Ingress Firewall**: Screens incoming user prompts for prompt injection, jailbreaks, PII leakage, and toxic content before tokens reach the LLM.
2. **RBAC Pre-Filtering**: Restricts document retrieval strictly to documents the user's role is authorized to view, preventing unauthorized knowledge leakage.
3. **Post-Execution Egress Firewall**: Scans model responses for hallucinations, unredacted secrets (API keys, social security numbers), and competitor mentions before returning output to the client.
4. **Audit Observability**: Logs every prompt, completion, latency metric, and token cost to centralized telemetry stores (LangSmith, OpenTelemetry) with cryptographic user attribution.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Tenant Data Boundary Guarantees**: Enterprise deployments enforce strict zero-data-retention agreements: foundation model providers are contractually and architecturally barred from training on customer prompt tokens.
* **On-Premise & Private VPC Deployments**: Highly regulated industries (finance, defense, healthcare) deploy open-weights models (Llama 3, Mistral) inside air-gapped VPCs (AWS GovCloud, Azure Private Enclaves) using TensorRT-LLM and vLLM.
* **Semantic Cache Layers**: Enterprise gateways deploy Redis semantic caches: identical or semantically similar queries are served from cache in under 5ms at $0.00 cost, saving 20% to 40% on monthly API spend.
""")

df_a8 = pd.DataFrame({
    "Dimension": [
        "Security Boundary",
        "Data Privacy & Retention",
        "Access Authorization",
        "Cost & Rate Governance",
        "Observability & Auditing"
    ],
    "Consumer LLM APIs (ChatGPT / Claude Free)": [
        "Shared public multi-tenant infrastructure",
        "User prompts may be retained and used for future model training",
        "Single individual user account with zero corporate role filtering",
        "Fixed personal rate limits; zero enterprise spend controls",
        "Local session history only; zero centralized enterprise compliance logging"
    ],
    "Enterprise AI Production Platform": [
        "Private VPC / Air-gapped on-premise deployment with egress firewalls",
        "Zero data retention guarantees; model weights isolated from customer data",
        "Strict RBAC pre-filtering mapped to Active Directory / Okta SAML roles",
        "Granular department-level token budgets, semantic caching, and rate limiting",
        "Centralized OpenTelemetry audit trails with cryptographic attribution"
    ]
})
st.dataframe(df_a8, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 10: Production Implementation: Enterprise Guardrailed RAG Pipeline
# ==============================================================================
st.header("10. Production Implementation: Enterprise Guardrailed RAG Pipeline in Python", divider="rainbow")

st.markdown("""
In enterprise deployments, LLMs must **never** be given direct access to raw unpartitioned databases or ungoverned financial tools. 
The following runnable script demonstrates production patterns for **RBAC Pre-Filtering** and **Deterministic Financial Guardrails**:
""")

st.code(r'''from typing import List, Dict, Any, Optional

# ----------------------------------------------------------------------
# 1. RBAC Knowledge Retriever: Pre-Filtering Authorization
# ----------------------------------------------------------------------
class EnterpriseKnowledgeRetriever:
    """
    Demonstrates RBAC Pre-Filtering: Document authorization metadata 
    is filtered before semantic similarity scoring takes place.
    Prevents unauthorized knowledge from leaking into LLM context.
    """
    def __init__(self, document_corpus: List[Dict[str, Any]]):
        self.corpus = document_corpus

    def search_with_rbac(self, query: str, user_roles: List[str]) -> List[Dict[str, Any]]:
        # STEP 1: Pre-Filter strictly by user role permissions
        authorized_docs = [
            doc for doc in self.corpus 
            if any(role in doc["authorized_roles"] for role in user_roles)
        ]
        
        # STEP 2: Compute relevance score strictly over authorized subspace
        query_terms = set(query.lower().split())
        results = []
        for doc in authorized_docs:
            doc_terms = set(doc["content"].lower().split())
            score = len(query_terms.intersection(doc_terms))
            if score > 0:
                results.append({
                    "doc_id": doc["id"], 
                    "title": doc["title"],
                    "content": doc["content"], 
                    "score": score
                })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)

# ----------------------------------------------------------------------
# 2. Deterministic Financial Execution Guardrail
# ----------------------------------------------------------------------
class FinancialActionGuardrail:
    """
    Deterministic Financial Execution Guardrail: Prevents an LLM agent 
    from executing unconstrained financial actions without human signoff.
    """
    MAX_AUTONOMOUS_REFUND = 50.00  # Cap single autonomous refund at $50

    @classmethod
    def validate_refund_action(cls, amount: float, customer_id: str) -> Dict[str, Any]:
        if amount <= 0:
            return {"status": "REJECTED", "reason": "Refund amount must be positive."}
        
        if amount > cls.MAX_AUTONOMOUS_REFUND:
            return {
                "status": "ESCALATE_TO_HUMAN",
                "reason": f"Amount ${amount:.2f} exceeds autonomous threshold of ${cls.MAX_AUTONOMOUS_REFUND:.2f}.",
                "requires_approval_by": "Finance_Supervisor",
                "action_dispatched": False
            }
        
        return {
            "status": "APPROVED",
            "reason": f"Amount ${amount:.2f} within autonomous limits.",
            "action_dispatched": True
        }

# ----------------------------------------------------------------------
# 3. Demonstration Execution
# ----------------------------------------------------------------------
sample_corpus = [
    {
        "id": "DOC-101",
        "title": "Public Holiday Calendar 2026",
        "authorized_roles": ["All_Employees", "Contractor", "Executive"],
        "content": "All company employees receive 12 paid public holidays including New Year and Thanksgiving."
    },
    {
        "id": "DOC-202",
        "title": "Engineering Kubernetes Runbook",
        "authorized_roles": ["Engineering", "Executive"],
        "content": "To scale the production Kubernetes cluster, update deployment replicas in the Helm values file."
    },
    {
        "id": "DOC-303",
        "title": "Executive Equity Grants & Bonuses",
        "authorized_roles": ["Executive"],
        "content": "Q3 Executive equity grants are allocated at 15,000 shares per VP subject to 4-year vesting."
    }
]

retriever = EnterpriseKnowledgeRetriever(sample_corpus)

# Scenario A: Contractor searches for Executive Compensation
contractor_results = retriever.search_with_rbac(query="Executive equity grants", user_roles=["Contractor"])
print("Contractor Search Results Count:", len(contractor_results))  # Strictly 0!

# Scenario B: Executive searches for Executive Compensation
exec_results = retriever.search_with_rbac(query="Executive equity grants", user_roles=["Executive"])
print("Executive Search Results:", [doc["title"] for doc in exec_results])  # Returns DOC-303!

# Scenario C: Agent attempts autonomous refund of $120.00
guardrail_result = FinancialActionGuardrail.validate_refund_action(amount=120.00, customer_id="CUST-491")
print("Financial Guardrail Decision:", guardrail_result["status"], "| Reason:", guardrail_result["reason"])
''', language="python")

st.divider()

# ==============================================================================
# Section 11: Interactive Workbench: Application Archetype & Latency Simulator
# ==============================================================================
st.header("11. Interactive Workbench: Application Archetype & Latency Simulator", divider="rainbow")

st.markdown("""
Every application archetype operates under distinct latency constraints, token consumption profiles, 
and hardware architectures. Use this workbench to evaluate operational trade-offs and latency SLAs across archetypes.
""")

col_wb_in, col_wb_out = st.columns([1, 1.2])

with col_wb_in:
    selected_archetype = st.selectbox(
        "Select Enterprise Application Archetype:",
        [
            "Interactive Coding Assistant (In-IDE)",
            "Enterprise Hybrid Search & RAG",
            "High-Throughput Invoice Extraction",
            "Autonomous Multi-Step AI Agent"
        ]
    )

    if selected_archetype == "Interactive Coding Assistant (In-IDE)":
        default_ttft = 100
        default_throughput = 90
        default_input_tok = 1500
        default_output_tok = 80
        sla_threshold = 200
    elif selected_archetype == "Enterprise Hybrid Search & RAG":
        default_ttft = 450
        default_throughput = 60
        default_input_tok = 4000
        default_output_tok = 400
        sla_threshold = 2000
    elif selected_archetype == "High-Throughput Invoice Extraction":
        default_ttft = 800
        default_throughput = 50
        default_input_tok = 3000
        default_output_tok = 500
        sla_threshold = 5000
    else:
        default_ttft = 600
        default_throughput = 45
        default_input_tok = 6000
        default_output_tok = 1200
        sla_threshold = 15000

    sim_ttft = st.slider("Time-To-First-Token (TTFT in ms)", min_value=50, max_value=2000, value=default_ttft, step=50)
    sim_throughput = st.slider("Inference Generation Speed (Tokens/sec)", min_value=10, max_value=150, value=default_throughput, step=5)
    sim_input_tokens = st.number_input("Input Prompt Tokens", min_value=100, max_value=32000, value=default_input_tok, step=500)
    sim_output_tokens = st.number_input("Output Generated Tokens", min_value=10, max_value=4000, value=default_output_tok, step=50)

# Total Latency Calculation
generation_time_ms = (sim_output_tokens / sim_throughput) * 1000.0
total_latency_ms = sim_ttft + generation_time_ms
total_tokens = sim_input_tokens + sim_output_tokens
sla_met = total_latency_ms <= sla_threshold

with col_wb_out:
    st.subheader("Operational SLA & Performance Telemetry")

    sla_m1, sla_m2 = st.columns(2)
    sla_m1.metric("Total End-to-End Latency", f"{total_latency_ms:,.0f} ms", delta=f"{total_latency_ms / 1000.0:.2f} seconds")
    sla_m2.metric("Target SLA Ceiling", f"{sla_threshold:,} ms", delta="MET" if sla_met else "BREACHED", delta_color="normal" if sla_met else "inverse")

    tok_m1, tok_m2 = st.columns(2)
    tok_m1.metric("Time-To-First-Token (TTFT)", f"{sim_ttft} ms")
    tok_m2.metric("Generation Time", f"{generation_time_ms:,.0f} ms")

    st.markdown("##### Latency Breakdown")
    df_lat = pd.DataFrame({
        "Phase": [
            "1. Network Transmission & Prompt Ingress",
            "2. Prompt Prefill Computation (TTFT)",
            "3. Autoregressive Token Generation",
            "Total Turnaround Time"
        ],
        "Duration (ms)": [
            50.0,
            float(sim_ttft),
            float(generation_time_ms),
            float(total_latency_ms)
        ],
        "Percentage of Latency": [
            f"{(50.0 / total_latency_ms) * 100.0:.1f}%",
            f"{(sim_ttft / total_latency_ms) * 100.0:.1f}%",
            f"{(generation_time_ms / total_latency_ms) * 100.0:.1f}%",
            "100.0%"
        ]
    })
    st.dataframe(df_lat, use_container_width=True, hide_index=True)

    # Latency Stacked Bar Chart
    df_lat_chart = pd.DataFrame({
        "Component": ["Prefill TTFT", "Generation Streaming"],
        "Milliseconds": [sim_ttft, generation_time_ms]
    })
    st.bar_chart(df_lat_chart.set_index("Component"))

st.divider()

# ==============================================================================
# Section 12: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Enterprise Interview Questions")

qa_col1, qa_col2 = st.columns(2)

with qa_col1:
    with st.expander("Q1: How do you prevent Role-Based Access Control (RBAC) data leakage in enterprise RAG systems?"):
        st.markdown("""
        **Core Architectural Answer:**  
        In enterprise RAG, **Post-Filtering is an antipattern**: retrieving top-K documents purely by vector similarity and then discarding unauthorized documents frequently results in an empty context window if all top-K chunks belong to restricted folders.
        
        **Production Solution: Metadata Pre-Filtering**:
        1. Every document chunk in the vector database stores authorization metadata tags (e.g. `allowed_roles: ["Engineering", "Executive"]`, `department_id: "D-401"`).
        2. When a user queries the system, the application extracts the user's validated SAML/JWT claims.
        3. The vector database query applies a boolean filter **before** cosine distance scoring:
           `VectorSearch(query_vector, filter={"allowed_roles": {"$in": user_roles}})`
        This mathematically guarantees that the model only sees and synthesizes facts the user is authorized to inspect.
        """)

    with st.expander("Q2: What is the difference between Ingress and Egress Guardrails in enterprise AI?"):
        st.markdown("""
        **Answer:**  
        * **Ingress Guardrails (Pre-LLM)**: Inspect user inputs before tokens reach the foundation model. Functions include:
          * *Prompt Injection / Jailbreak Detection*: Identifying attempts to override system prompts.
          * *PII Redaction*: Masking social security numbers, credit card tokens, and passwords.
          * *Query Intent Classification*: Blocking inquiries outside company service scope.
        * **Egress Guardrails (Post-LLM)**: Inspect model completions before returning text to the user. Functions include:
          * *Hallucination Verification*: Verifying that claims strictly map to retrieved RAG context.
          * *Secrets Leakage Detection*: Blocking leaked API keys or credentials.
          * *Toxicity & Regulatory Compliance*: Ensuring legal disclaimers are present and toxic content is blocked.
        """)

    with st.expander("Q3: When should you use Prompt Engineering vs RAG vs Model Fine-Tuning?"):
        st.markdown("""
        **Answer:**  
        The enterprise decision matrix depends on two axes: **External Knowledge Requirements** vs **Behavioral / Style Adaptation**:
        1. **Prompt Engineering**: Use for rapid prototyping, general cognitive tasks, and when the model already possesses the requisite domain knowledge.
        2. **Retrieval-Augmented Generation (RAG)**: Mandatory when the model requires access to dynamic, proprietary, or frequently changing enterprise facts. RAG is auditable, supports RBAC, and requires zero retraining compute.
        3. **Fine-Tuning (LoRA / Full Weights)**: Use strictly to alter the model's **behavior, style, syntax, or tone** (e.g., teaching an 8B model to emit strict domain DSL syntax, or compacting a complex reasoning prompt into weights to slash inference latency). Fine-tuning is NOT an efficient way to teach the model factual memory.
        """)

with qa_col2:
    with st.expander("Q4: How do you prevent autonomous AI agents from entering infinite runaway execution loops?"):
        st.markdown("""
        **Answer:**  
        Autonomous ReAct agents can enter infinite loops when a tool returns an unexpected error or cyclical observations. Production systems implement five defense mechanisms:
        1. **Hard Max-Iteration Caps**: Capping the maximum ReAct execution steps (typically 8 to 12 turns max).
        2. **Deduplication Checkers**: Detecting when the model emits identical function calls or queries more than twice consecutively.
        3. **Financial & Tool Budgets**: Allocating strict token and API cost budgets per agent task.
        4. **Ephemeral Network Sandboxing**: Executing code in containers with hard CPU, memory, and 10-second timeout limits.
        5. **Human-in-the-Loop Escalation**: Triggering supervisor approval when risk thresholds or iteration ceilings are reached.
        """)

    with st.expander("Q5: Why is semantic caching critical for enterprise LLM application cost and latency?"):
        st.markdown("""
        **Answer:**  
        In enterprise customer support and internal search, 20% to 40% of incoming user queries are semantically equivalent (e.g., `"How do I reset my password?"` vs `"Forgot password reset steps"`).
        
        A standard exact-string cache (Redis key-value) misses these variations. A **Semantic Cache** (e.g., GPTCache, Redis Vector Search):
        1. Embeds incoming queries into vector embeddings.
        2. Computes cosine similarity against previously cached queries.
        3. If similarity exceeds a strict threshold (e.g., `0.96`), the cached response is returned immediately.
        This delivers responses in under **5 milliseconds at $0.00 token cost**, saving tens of thousands of dollars monthly.
        """)

st.divider()

st.subheader("Production Traps & Deployment Gotchas")

t_col1, t_col2 = st.columns(2)

with t_col1:
    st.error("""
    **Production Trap 1 (Critical): Post-Filtering RBAC Documents in Vector Search**  
    * **Failure Mode**: Querying the vector database for top-5 chunks by cosine similarity, then discarding chunks the user lacks permissions to view.
    * **Impact**: If all top-5 chunks belong to restricted executive folders, the context window is empty, causing the model to hallucinate or say it cannot find information that exists in public documents.
    * **Remediation**: Always enforce RBAC metadata **pre-filtering** directly in the vector database query before similarity scoring.
    """)

    st.error("""
    **Production Trap 2 (Critical): Direct Unsandboxed SQL or Python Tool Execution**  
    * **Failure Mode**: Allowing an LLM agent to execute generated SQL queries or Python scripts directly against production databases.
    * **Impact**: Prompt injection attacks or accidental model hallucinations executing `DROP TABLE`, exfiltrating customer tables, or causing Denial of Service.
    * **Remediation**: Execute read-only queries with restricted database users, sanitize inputs with query parameterization, and run code in ephemeral Docker/Wasm sandboxes.
    """)

with t_col2:
    st.warning("""
    **Production Trap 3 (Caution): Uncapped Autoregressive Generation Budgets**  
    * **Failure Mode**: Setting `max_tokens=4096` on high-throughput customer chatbots without monitoring average generation lengths.
    * **Impact**: If a user prompts the model to write a novel or loop indefinitely, API token bills surge unexpectedly and latency spikes for other users.
    * **Remediation**: Enforce strict archetype-specific token caps (e.g., `max_tokens=250` for support chatbots) and monitor Inter-Token-Latency (ITL).
    """)

    st.warning("""
    **Production Trap 4 (Caution): Blindly Trusting Fine-Tuning for Knowledge Updates**  
    * **Failure Mode**: Attempting to update product catalog knowledge or company policies by fine-tuning model weights weekly.
    * **Impact**: Severe catastrophic forgetting of general reasoning, hallucinations on stale facts, and exorbitant GPU training bills.
    * **Remediation**: Use RAG for all dynamic knowledge updates; reserve fine-tuning strictly for behavioral alignment and syntax formatting.
    """)

st.divider()

st.subheader("Core Application Flashcards")

app_flash = st.selectbox(
    "Select Application Flashcard to Review:",
    [
        "RBAC Pre-Filtering",
        "Semantic Caching",
        "ReAct Agent Loop",
        "Fill-In-The-Middle (FIM)"
    ]
)

if app_flash == "RBAC Pre-Filtering":
    st.info("""
    **RBAC Pre-Filtering**: Restricting document vector retrieval strictly to authorized subsets of documents based on user role metadata before similarity scoring, preventing unauthorized data leakage in enterprise RAG systems.
    """)
elif app_flash == "Semantic Caching":
    st.info("""
    **Semantic Caching**: Serving cached responses for semantically equivalent queries using vector similarity thresholds (e.g. 0.96 cosine similarity), reducing latency to under 5ms and cutting API token expenditure by 20% to 40%.
    """)
elif app_flash == "ReAct Agent Loop":
    st.info("""
    **ReAct Agent Loop**: A synergy of Reasoning and Acting where an autonomous LLM interleaves natural language thoughts, structured tool actions (API/SQL calls), and environment observations in an iterative closed loop.
    """)
else:
    st.info("""
    **Fill-In-The-Middle (FIM)**: An architectural training technique enabling coding LLMs to predict code conditioned simultaneously on preceding prefix code and subsequent suffix code, powering in-IDE tab completions.
    """)

st.divider()

# ==============================================================================
# Section 13: Self-Assessment Checklist
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of enterprise foundation model application topologies:")

cb_col1, cb_col2 = st.columns(2)

with cb_col1:
    cb1 = st.checkbox("I can enumerate and describe all 8 enterprise application archetypes.", value=True)
    cb2 = st.checkbox("I understand why RBAC metadata pre-filtering is mandatory to prevent RAG authorization leakage.", value=True)
    cb3 = st.checkbox("I know how Fill-In-The-Middle (FIM) powers in-IDE coding assistants conditioned on prefix and suffix.", value=True)

with cb_col2:
    cb4 = st.checkbox("I can architect a two-stage hybrid search pipeline combining BM25, dense vectors, and neural rerankers.", value=True)
    cb5 = st.checkbox("I understand the ReAct execution loop and why ephemeral sandboxing is required for AI agents.", value=True)
    cb6 = st.checkbox("I know when to use Prompt Engineering vs RAG vs Model Fine-Tuning based on knowledge and style axes.", value=True)

if all([cb1, cb2, cb3, cb4, cb5, cb6]):
    st.success("🎯 **Lesson 1.4 Complete! You have demonstrated comprehensive mastery of enterprise LLM application topologies, governance, and production deployment patterns.**")
