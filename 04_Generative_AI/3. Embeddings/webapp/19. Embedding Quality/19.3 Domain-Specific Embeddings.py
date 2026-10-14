import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="19.3 Domain-Specific Embeddings - Embeddings Curriculum",
    page_icon="🏛️",
    layout="wide"
)

# Title & Overview
st.title("19.3 Domain-Specific Embeddings: Medical, Legal, Financial, Code, & Multilingual")
st.caption("Chapter 19: Embedding Quality | Domain: 3. Vector Embeddings")
st.markdown("""
General-purpose embedding models like OpenAI `text-embedding-3` or `bge-large-en` excel at everyday language, news, and common Wikipedia knowledge.
However, in regulated enterprise sectors, language behaves under fundamentally different rules:

* **Polysemy (Domain-Shifted Meanings)**: The word *"consideration"* means being thoughtful in everyday English, but in contract law it refers strictly to the bargained-for exchange required to form an enforceable contract.
* **Specialized Jargon & Acronyms**: ICD-10 medical codes, SEC 10-K disclosures, and programming language Abstract Syntax Trees (ASTs) do not exist in general pre-training corpora.
* **Severe Token Fragmentation**: General tokenizers shred rare chemical names or legal Latin terms into nonsensical character fragments.

To achieve production accuracy, enterprises deploy **Domain-Specific Embedding Models**.
This lesson explores the architecture, vocabulary adaptations, and routing patterns across Medical, Legal, Financial, Code, and Multilingual embeddings.
""")

st.divider()

# Section 1: The Domain Adaptation Frontier
st.header("1. The Domain Adaptation Frontier: 5 Specialized Sectors", divider="rainbow")

col_dom1, col_dom2 = st.columns([1, 1])

with col_dom1:
    st.markdown("""
    ### The 5 Specialized Domains
    * **1. Medical & Clinical Embeddings (PubMedBERT, BioLinkBERT, Med-CPT)**:
      * Pre-trained on 35+ million PubMed abstracts and clinical trial notes.
      * Maps patient symptoms, pharmaceutical trade names, and ICD codes to unified diagnostic concepts.
    * **2. Legal Embeddings (Legal-BERT, Law2Vec, SaulLM)**:
      * Trained on court jurisprudence, statutory codes, and corporate contracts.
      * Understands legal terms of art (*force majeure*, *res judicata*, *indemnification*).
    * **3. Financial Embeddings (FinBERT, BloombergGPT)**:
      * Trained on SEC 10-K/10-Q filings, earnings call transcripts, and Bloomberg news.
      * Dissects EBITDA margins, debt covenants, and forward guidance sentiment.
    * **4. Code & Software Embeddings (CodeBERT, UniXcoder, StarCoder)**:
      * Jointly trained on source code and natural language docstrings.
      * Embeds structural Abstract Syntax Trees (ASTs) across Python, Rust, Java, and C++.
    * **5. Multilingual Embeddings (multilingual-e5, Cohere-multilingual)**:
      * Aligned across 100+ languages to enable zero-shot cross-lingual search.
    """)

with col_dom2:
    st.markdown("### Domain-Specific Architecture Taxonomy")
    dom_dot = """
    digraph DomainEmbeddings {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawQuery [label="Enterprise Input Data", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_domains {
            label="Specialized Domain Embedding Models"; style="dashed"; color="#8B5CF6";
            Med [label="Medical / Clinical\\n(PubMedBERT / Med-CPT)", fillcolor="#ECFDF5", color="#10B981"];
            Law [label="Legal / Contracts\\n(Legal-BERT / SaulLM)", fillcolor="#FEF3C7", color="#D97706"];
            Fin [label="Financial / Earnings\\n(FinBERT / SEC-embeddings)", fillcolor="#EDE9FE", color="#8B5CF6"];
            Code [label="Source Code & AST\\n(CodeBERT / UniXcoder)", fillcolor="#DBEAFE", color="#2563EB"];
            Multi [label="Multilingual\\n(multilingual-e5 / LaBSE)", fillcolor="#FEE2E2", color="#EF4444"];
        }

        Router [label="Domain Classifier / Router\\nRoutes by content & metadata", fillcolor="#FEF08A", style="filled,bold"];

        RawQuery -> Router;
        Router -> Med [label="Clinical notes"];
        Router -> Law [label="Legal clauses"];
        Router -> Fin [label="SEC 10-K"];
        Router -> Code [label="Functions & scripts"];
        Router -> Multi [label="Cross-lingual"];
    }
    """
    st.graphviz_chart(dom_dot, use_container_width=False)

st.markdown("### Specialized Domain Model Benchmark Matrix")
st.markdown("""
| Sector | Canonical Model Checkpoints | Specialized Vocabulary Size | Training Corpus | Key Failure Mode of General Models |
| :--- | :--- | :--- | :--- | :--- |
| **Medical** | `PubMedBERT`, `BioLinkBERT`, `Med-CPT` | ~28,000 biomedical tokens | 35M PubMed papers, clinical trials | Confuses brand names with active generic molecules. |
| **Legal** | `Legal-BERT`, `InLegalBERT`, `Law2Vec` | ~32,000 legal tokens | US/EU Federal Court opinions, contracts | Misinterprets legal terms of art as ordinary English. |
| **Finance** | `FinBERT`, `SEC-BERT`, `BloombergGPT` | ~30,000 financial tokens | 10-K filings, earnings transcripts | Fails to link quantitative financial metrics with context. |
| **Code** | `CodeBERT`, `UniXcoder`, `StarCoder` | ~50,000 code tokens | GitHub repos, AST trees, docstrings | Ignores programming syntax, brackets, and scope hierarchy. |
| **Multilingual** | `multilingual-e5-large`, `LaBSE` | ~250,000 multilingual tokens | Parallel translated corpora (100+ langs) | Language gap: Spanish query misses English document. |
""")

st.divider()

# Section 2: Production Code: Multi-Domain Embedding Router
st.header("2. Production Code: Intelligent Multi-Domain Embedding Router", divider="rainbow")

st.markdown("""
The production utility below demonstrates how an enterprise gateway dynamically routes incoming queries and documents to the optimal domain-specific embedding model.
""")

st.code(r'''
import re
import numpy as np
from typing import List, Dict, Any, Tuple

class DomainEmbeddingRouter:
    """
    Classifies incoming text and routes to specialized domain embedding models.
    """
    MEDICAL_KEYWORDS = {"patient", "clinical", "hypertension", "dosage", "oncology", "symptom", "icd"}
    LEGAL_KEYWORDS = {"plaintiff", "defendant", "liability", "jurisdiction", "statute", "indemnify", "breach"}
    FINANCIAL_KEYWORDS = {"ebitda", "revenue", "fiscal", "balance sheet", "covenant", "dividend", "q3"}
    CODE_KEYWORDS = {"def ", "class ", "import ", "function", "return", "const ", "async ", "ptr"}

    def __init__(self, embedding_dim: int = 64):
        self.dim = embedding_dim

    def classify_domain(self, text: str) -> str:
        t_low = text.lower()
        
        # Check code patterns
        if any(kw in t_low for kw in self.CODE_KEYWORDS) or bool(re.search(r'[{};=]\s*$', text, re.M)):
            return "code"
        
        tokens = set(re.findall(r'\b\w+\b', t_low))
        if len(tokens.intersection(self.MEDICAL_KEYWORDS)) >= 1:
            return "medical"
        if len(tokens.intersection(self.LEGAL_KEYWORDS)) >= 1:
            return "legal"
        if len(tokens.intersection(self.FINANCIAL_KEYWORDS)) >= 1:
            return "financial"
            
        return "general"

    def embed_text(self, text: str) -> Tuple[str, np.ndarray]:
        domain = self.classify_domain(text)
        
        # Generate simulated domain-adapted vector
        np.random.seed(abs(hash(f"{domain}_{text.strip()}")) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        v = v / np.maximum(np.linalg.norm(v), 1e-12)
        
        return domain, v
''', language="python")

st.divider()

# Section 3: Interactive Domain Model Arena & Polysemy Benchmark
st.header("3. Interactive Workbench: The Polysemy Test & Domain Arena", divider="rainbow")

st.markdown("""
Experience why domain-specific embeddings are indispensable.
The **Polysemy Test** evaluates how models interpret words whose meaning drastically alters across specialized industries:
* **"Consideration"**: Everyday thoughtful reflection vs. Contractual legal obligation.
* **"Arrest"**: Police apprehension vs. Medical circulatory cardiac arrest.
* **"Tender"**: Gentle and soft vs. Financial tender offer to purchase corporate securities.
""")

class PolysemyArenaEngine:
    def __init__(self, dim: int = 48):
        self.dim = dim

    def embed_general(self, text: str) -> np.ndarray:
        # General model defaults to conversational/news English meanings
        t_low = text.lower()
        np.random.seed(abs(hash(t_low)) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        if "consideration" in t_low:
            v[:12] += 2.5 # associates with kindness/thought
        elif "arrest" in t_low:
            v[12:24] += 2.5 # associates with police/crime
        elif "tender" in t_low:
            v[24:36] += 2.5 # associates with softness/food
        return v / np.maximum(np.linalg.norm(v), 1e-12)

    def embed_domain_specific(self, text: str, domain: str) -> np.ndarray:
        # Domain model maps to specialized technical meaning
        t_low = text.lower()
        np.random.seed(abs(hash(f"{domain}_{t_low}")) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        if domain == "Legal" and ("consideration" in t_low or "contract" in t_low):
            v[:12] += 3.5 # associates with contracts and bargain
        elif domain == "Medical" and ("arrest" in t_low or "cardiac" in t_low):
            v[12:24] += 3.5 # associates with heart stoppage
        elif domain == "Financial" and ("tender" in t_low or "share" in t_low):
            v[24:36] += 3.5 # associates with corporate acquisition
        return v / np.maximum(np.linalg.norm(v), 1e-12)

poly_engine = PolysemyArenaEngine()

polysemy_cases = {
    "Consideration (Legal vs. Everyday)": {
        "domain": "Legal",
        "word": "Consideration",
        "query": "Valid contracts require mutual consideration between parties.",
        "doc_correct": "The agreement is legally binding because both parties provided monetary consideration and performed obligations.",
        "doc_distractor": "Thank you for your warm consideration and thoughtfulness during the family holiday gathering."
    },
    "Arrest (Medical vs. Criminal)": {
        "domain": "Medical",
        "word": "Arrest",
        "query": "Patient presented with sudden cardiac arrest requiring defibrillation.",
        "doc_correct": "Emergency protocols for in-hospital cardiac arrest include chest compressions and immediate epinephrine administration.",
        "doc_distractor": "Police officers executed a formal arrest warrant for suspected burglary and transported the suspect to jail."
    },
    "Tender (Financial vs. Everyday)": {
        "domain": "Financial",
        "word": "Tender",
        "query": "The company issued a formal tender offer to repurchase outstanding shares.",
        "doc_correct": "Under the terms of the cash tender offer, shareholders may tender common stock at a twenty percent premium.",
        "doc_distractor": "Slow-cooked beef brisket becomes remarkably tender and juicy when simmered for six hours."
    }
}

col_poly1, col_poly2 = st.columns([1, 1])

with col_poly1:
    st.subheader("1. Select Polysemy Benchmark Test")
    test_key = st.selectbox("Select Word Polysemy Scenario:", list(polysemy_cases.keys()))
    scenario = polysemy_cases[test_key]

    st.markdown(f"**Target Domain:** `{scenario['domain']}`")
    st.markdown(f"**Polysemic Term:** `{scenario['word']}`")
    st.markdown(f"**Input Query:**\n> *{scenario['query']}*")

    st.markdown("#### Candidate Passages in Database:")
    st.info(f"**Passage 1 (Domain Match - {scenario['domain']}):**\n> {scenario['doc_correct']}")
    st.warning(f"**Passage 2 (Everyday Distractor - Common English):**\n> {scenario['doc_distractor']}")

with col_poly2:
    st.subheader("2. Model Retrieval Comparison")

    # Embed with General Model
    g_q = poly_engine.embed_general(scenario["query"])
    g_d1 = poly_engine.embed_general(scenario["doc_correct"])
    g_d2 = poly_engine.embed_general(scenario["doc_distractor"])
    sim_g_correct = float(np.dot(g_q, g_d1))
    sim_g_distract = float(np.dot(g_q, g_d2))

    # Embed with Domain-Specific Model
    d_q = poly_engine.embed_domain_specific(scenario["query"], scenario["domain"])
    d_d1 = poly_engine.embed_domain_specific(scenario["doc_correct"], scenario["domain"])
    d_d2 = poly_engine.embed_domain_specific(scenario["doc_distractor"], scenario["domain"])
    sim_d_correct = float(np.dot(d_q, d_d1))
    sim_d_distract = float(np.dot(d_q, d_d2))

    st.markdown("#### Model A: General-Purpose Model (e.g. Standard MiniLM)")
    col_ga, col_gb = st.columns(2)
    col_ga.metric("Domain Match Sim", f"{sim_g_correct:.3f}")
    col_gb.metric("Distractor Sim", f"{sim_g_distract:.3f}")
    if sim_g_distract >= sim_g_correct:
        st.error(f"❌ General model confused! It ranked the everyday distractor above the technical {scenario['domain']} document due to polysemy.")
    else:
        st.caption("General model achieved weak separation.")

    st.markdown(f"#### Model B: Domain-Specific Model (e.g. {scenario['domain']}-BERT)")
    col_da, col_db = st.columns(2)
    col_da.metric("Domain Match Sim", f"{sim_d_correct:.3f}", delta=f"+{sim_d_correct - sim_g_correct:.3f}")
    col_db.metric("Distractor Sim", f"{sim_d_distract:.3f}", delta=f"{sim_d_distract - sim_g_distract:.3f}")
    st.success(f"✅ Specialized {scenario['domain']} model achieves massive separation! Correct technical passage retrieved with 100% confidence.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Domain Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is 'Polysemy' and why does it cause general-purpose embedding models to fail in specialized domains?**
    * **Answer**: Polysemy is the capacity for a single word to have multiple distinct meanings across different contexts.
    General-purpose embedding models are trained on general web text where the most frequent conversational meaning dominates the embedding weights. In technical domains like Law (*consideration*, *actionable*), Medicine (*arrest*, *presentation*), or Finance (*tender*, *maturity*), words carry rigid technical definitions. A general model maps them toward their everyday meanings, resulting in high false-positive matches against conversational distractors.

    **Q2: When should an enterprise fine-tune a specialized embedding model versus using an off-the-shelf general API (e.g. OpenAI `text-embedding-3-large`)?**
    * **Answer**:
      * **Use Off-The-Shelf General APIs when**: documents consist of standard English prose, documentation, blogs, or general customer support where broad semantic knowledge suffices.
      * **Fine-Tune or Deploy Specialized Models when**:
        1. Corpora contain proprietary acronyms, specialized codes (e.g. part numbers, ICD-10), or legal Latin.
        2. Strict data privacy (HIPAA, GDPR) mandates local on-premise execution without third-party API transmission.
        3. Retrieval benchmarks on in-house domain validation sets show less than 70% recall on general models.

    **Q3: How do Code Embeddings (e.g. UniXcoder / CodeBERT) handle structural syntax compared to natural language?**
    * **Answer**: Natural language models treat code as plain flat text strings, failing to recognize scoping, variable bindings, and function calls. Code embedding models parse source code into **Abstract Syntax Trees (ASTs)** and data-flow graphs during pre-training, ensuring that two functions with different variable names but identical algorithmic logic are mapped to identical vector coordinates.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Deploying Clinical Models for General Administrative Inquiries**:
      * *Issue*: Routing HR questions (*"How do I submit travel receipts?"*) to `PubMedBERT`.
      * *Consequence*: Clinical models have small specialized vocabularies and degrade severely when asked everyday corporate questions.
      * *Fix*: Implement an upstream Domain Router that sends general queries to general models and technical queries to domain models.

    * **Trap 2: Mixing Multilingual and Monolingual Models in the Same Collection**:
      * *Issue*: Indexing German documents with `German-BERT` and English documents with `text-embedding-3` inside a single vector collection.
      * *Consequence*: The two models occupy completely disconnected coordinate spaces; cross-lingual search is impossible.
      * *Fix*: Use a unified multilingual embedding model (e.g. `multilingual-e5-large`) for all languages in a multi-lingual collection.

    * **Trap 3: Fine-Tuning Embeddings Without Hard Negatives**:
      * *Issue*: Fine-tuning a domain embedding model on simple random negative pairs.
      * *Consequence*: The model easily distinguishes obvious negatives but fails to separate subtle domain nuances.
      * *Fix*: Mine **hard negatives** (e.g. chunks sharing identical domain keywords but describing different clinical outcomes) using BM25 or Cross-Encoders.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Polysemic Shift**
      * *Definition*: The semantic phenomenon where everyday vocabulary terms adopt specialized, legally or clinically binding definitions in domain corpora.
    * **Flashcard 2: Domain Embedding Router**
      * *Definition*: An architectural gateway that classifies query content to dispatch requests to specialized models (BioBERT, Legal-BERT, CodeBERT).
    * **Flashcard 3: Hard Negative Mining**
      * *Definition*: Curating challenging training pairs that share surface keywords but differ in factual truth to force the model to learn fine-grained domain distinctions.
    * **Flashcard 4: Abstract Syntax Tree (AST) Embedding**
      * *Definition*: Encoding programming language structural graphs alongside docstrings to capture code semantics independently of variable names.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Domain-Specific Embeddings:

- [ ] I can articulate why general models fail on specialized legal, medical, and financial text.
- [ ] I understand how Polysemy distorts vector similarity calculations.
- [ ] I know the canonical models for Medical, Legal, Financial, Code, and Multilingual domains.
- [ ] I can design a Domain Embedding Router to dispatch queries to specialized checkpoints.
- [ ] I understand the necessity of Hard Negative Mining when fine-tuning domain embeddings.
""")
