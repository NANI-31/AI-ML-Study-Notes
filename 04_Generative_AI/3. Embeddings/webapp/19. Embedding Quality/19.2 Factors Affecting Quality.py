import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="19.2 Factors Affecting Quality - Embeddings Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# Title & Overview
st.title("19.2 Factors Affecting Quality: Architectures, Preprocessing, Tokenization, & Query Formulation")
st.caption("Chapter 19: Embedding Quality | Domain: 3. Vector Embeddings")
st.markdown("""
A common pitfall when debugging a poorly performing RAG pipeline is blaming the vector database or LLM for flawed answers.
In reality, embedding quality is governed by a **cumulative pipeline of upstream variables**:

A failure at any of these upstream stages degrades vector accuracy before search even begins:
1. **Model Architecture**: Bi-Encoder design, parameter count (33M vs 7B), and pre-training objectives.
2. **Text Preprocessing**: Retaining vs destroying syntactic cues, HTML tags, and whitespace.
3. **Chunk Size**: The balance between vector specificity and context completeness.
4. **Chunk Overlap**: Preventing fact severance at sliding-window seams.
5. **Query Quality**: Short 2-word keyword queries vs. well-specified natural language questions.
6. **Domain Terminology**: Out-of-vocabulary subword fragmentation on rare acronyms and codes.
7. **Language & Multilingual Alignment**: Cross-lingual semantic transfer and tokenization efficiency.

This lesson analyzes how each factor shapes the final embedding representation.
""")

st.divider()

# Section 1: The 7 Critical Quality Factors
st.header("1. The Cumulative Quality Cascade: 7 Upstream Factors", divider="rainbow")

col_fac1, col_fac2 = col_fac1, col_fac2 = st.columns([1, 1])

with col_fac1:
    st.markdown("""
    ### The 7 Quality Determinants
    * **1. Model Architecture & Pre-training**:
      * Transformer scale matters: 33M-parameter MiniLM vs 335M-parameter BGE-large vs 7B-parameter NV-Embed. Larger models have richer latent manifolds and broader knowledge priors.
    * **2. Text Preprocessing Hygiene**:
      * Aggressive legacy NLP cleaning (removing stop words, stemming) **severely damages** transformer embeddings because transformers rely on functional words (`not`, `between`, `before`) for contextual attention!
    * **3. Chunk Size Calibration**:
      * Oversized chunks dilute fine-grained factual needles; undersized chunks lack pronouns and background.
    * **4. Chunk Overlap**:
      * Bridges split-clause boundaries without introducing duplicate context.
    * **5. Query Formulation & Asymmetry**:
      * A 2-word query (`"aws s3"`) lacks intent; an expanded query (`"How do I configure lifecycle transition to Glacier in S3?"`) maps cleanly to answer passages.
    * **6. Domain Subword Tokenization**:
      * In general tokenizers, specialized medical or legal terms get shredded into 6-8 nonsensical subword fragments, diluting semantic coordinates.
    * **7. Multilingual Alignment**:
      * Cross-lingual models must project translations into identical geometric coordinates.
    """)

with col_fac2:
    st.markdown("### The Quality Cascade Architecture")
    casc_dot = """
    digraph QualityCascade {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        RawCorp [label="Raw Text Corpus", fillcolor="#EFF6FF"];
        PreProc [label="Factor 2: Text Preprocessing\\n(Clean tags, preserve syntax)", fillcolor="#DBEAFE"];
        Chunking [label="Factor 3 & 4: Chunk Size & Overlap\\n(256 tokens, 32 overlap)", fillcolor="#BFDBFE"];
        Tokenize [label="Factor 6: Domain Tokenization\\n(Subword fragmentation rate)", fillcolor="#93C5FD"];
        ModelArch [label="Factor 1: Model Architecture\\n(BGE / OpenAI / NV-Embed)", fillcolor="#60A5FA", color="#1D4ED8", style="filled,bold"];
        
        UserQ [label="Factor 5: Query Quality\\n(Intent-rich prompt + HyDE)", fillcolor="#FEF3C7", color="#D97706", style="filled,bold"];
        VDB [label="Final Retrieval Quality\\nHigh Precision & Recall", fillcolor="#10B981", color="#059669", style="filled,bold"];

        RawCorp -> PreProc -> Chunking -> Tokenize -> ModelArch;
        ModelArch -> VDB [label="Corpus Vectors"];
        UserQ -> VDB [label="Query Vector"];
    }
    """
    st.graphviz_chart(casc_dot, use_container_width=False)

st.markdown("### Cause-and-Effect Engineering Matrix")
st.markdown("""
| Pipeline Factor | Poor Practice | Optimal Best Practice | Downstream Failure Avoided |
| :--- | :--- | :--- | :--- |
| **Preprocessing** | Removing stop words (`not`, `never`). | Preserving all syntax, punctuation, and casing. | Polarity reversal (negations treated as affirmations). |
| **Chunk Size** | 2000-token whole-page chunks. | 250 - 500 token coherent paragraphs. | Vector averaging dilution. |
| **Chunk Overlap** | 0% overlap. | 10% - 20% sliding window overlap. | Boundary severance of cross-clause facts. |
| **Domain Terms** | Generic BERT tokenizer on clinical records. | Domain-adapted tokenizer (BioBERT / CodeBERT). | Severe subword fragmentation into character gibberish. |
| **Query Quality** | Single keyword (`"refund"`). | Natural question or HyDE expansion. | Low cosine similarity against explanatory passages. |
""")

st.divider()

# Section 2: Production Code: Pipeline Quality Audit Suite
st.header("2. Production Code: Pipeline Quality & Fragmentation Auditor", divider="rainbow")

st.markdown("""
The production utility below audits text corpus hygiene, calculating tokenization fragmentation ratios on technical jargon and evaluating query intent richness.
""")

st.code(r'''
import re
from typing import List, Dict, Any

class PipelineQualityAuditor:
    """
    Evaluates factors affecting embedding quality:
    - Text hygiene and syntactic preservation
    - Technical vocabulary subword fragmentation ratio
    - Query intent richness
    """
    @staticmethod
    def audit_text_hygiene(raw_text: str) -> Dict[str, Any]:
        has_html = bool(re.search(r'<[^>]+>', raw_text))
        has_broken_newlines = bool(re.search(r'\w+-\n\w+', raw_text))
        stop_words_present = any(w in raw_text.lower().split() for w in ["not", "never", "without", "between"])
        
        return {
            "has_html_artifacts": has_html,
            "has_broken_hyphenated_newlines": has_broken_newlines,
            "preserves_syntactic_negations": stop_words_present,
            "word_count": len(raw_text.split()),
            "clean_score": 100 - (30 if has_html else 0) - (20 if has_broken_newlines else 0)
        }

    @staticmethod
    def estimate_subword_fragmentation(terms: List[str]) -> Dict[str, Any]:
        """
        Estimates how many subword tokens a term is fragmented into.
        Simulates standard BPE tokenizers on rare acronyms.
        """
        results = {}
        for term in terms:
            # Simple heuristic: capital letters, underscores, and length
            fragments = max(1, len(term) // 3)
            results[term] = {
                "estimated_tokens": fragments,
                "is_severely_fragmented": fragments > 3
            }
        return results

    @staticmethod
    def evaluate_query_quality(query: str) -> Dict[str, Any]:
        words = query.strip().split()
        is_keyword_only = len(words) <= 2
        has_interrogative = any(w.lower() in ["how", "what", "why", "when", "where", "can", "is"] for w in words)
        
        quality = "Poor (Short Keyword)" if is_keyword_only else ("Excellent (Intent-Rich)" if has_interrogative else "Moderate")
        return {
            "word_count": len(words),
            "quality_rating": quality,
            "recommended_action": "Use HyDE or query expansion" if is_keyword_only else "Direct embedding ready"
        }
''', language="python")

st.divider()

# Section 3: Interactive Quality Factor Sensitivity Workbench
st.header("3. Interactive Workbench: Quality Factor Sensitivity Studio", divider="rainbow")

st.markdown("""
Tweak the 4 primary pipeline factors and observe how the **Estimated Retrieval Quality Index** reacts in real time.
""")

col_sens1, col_sens2 = st.columns([1, 1])

with col_sens1:
    st.subheader("1. Configure Upstream Pipeline Factors")

    # Factor 1: Preprocessing Hygiene
    prep_mode = st.radio(
        "Factor 1: Text Preprocessing Hygiene",
        ["Modern Clean (Preserve Syntax & Negations)", "Legacy Bad (Remove Stop Words & Strip Punctuation)"],
        index=0
    )

    # Factor 2: Chunk Sizing
    chunk_choice = st.select_slider(
        "Factor 2: Chunk Size & Boundary Window",
        options=["50 Tokens (Fragmented)", "300 Tokens (Optimal Goldilocks)", "2000 Tokens (Diluted Whole-Page)"],
        value="300 Tokens (Optimal Goldilocks)"
    )

    # Factor 3: Model Architecture
    model_choice = st.selectbox(
        "Factor 3: Embedding Model Scale",
        ["Modern State-of-the-Art (BGE-Large / text-embedding-3 / 384+ dim)", "Legacy Lightweight (BERT-base / 768 dim)", "Tiny Custom Hash (32 dim)"],
        index=0
    )

    # Factor 4: Query Formulation
    query_choice = st.radio(
        "Factor 4: Query Formulation Quality",
        ["Intent-Rich Natural Question (e.g. 'How do I rotate IAM keys?')", "Vague Keyword Fragment (e.g. 'iam keys')"],
        index=0
    )

with col_sens2:
    st.subheader("2. Real-Time Retrieval Quality Impact")

    # Calculate synthetic quality score
    score = 100
    breakdowns = []

    if "Legacy Bad" in prep_mode:
        score -= 25
        breakdowns.append(("Preprocessing", -25, "Removed critical stop words and syntactic modifiers (polarity loss)."))
    else:
        breakdowns.append(("Preprocessing", 0, "Preserved syntax, negations, and grammatical structure."))

    if "50 Tokens" in chunk_choice:
        score -= 20
        breakdowns.append(("Chunk Size", -20, "Chunks fragmented; antecedent pronouns lack context."))
    elif "2000 Tokens" in chunk_choice:
        score -= 30
        breakdowns.append(("Chunk Size", -30, "Severe vector averaging dilution; specific facts washed out."))
    else:
        breakdowns.append(("Chunk Size", 0, "Optimal 300-token Goldilocks window."))

    if "Legacy Lightweight" in model_choice:
        score -= 15
        breakdowns.append(("Model Scale", -15, "Older architecture with lower MTEB semantic retrieval benchmark."))
    elif "Tiny Custom" in model_choice:
        score -= 40
        breakdowns.append(("Model Scale", -40, "Low dimensionality severely compresses latent geometry."))
    else:
        breakdowns.append(("Model Scale", 0, "State-of-the-art embedding architecture."))

    if "Vague Keyword" in query_choice:
        score -= 20
        breakdowns.append(("Query Formulation", -20, "Asymmetric gap: 2-word query maps poorly to explanatory answers."))
    else:
        breakdowns.append(("Query Formulation", 0, "Rich natural language question aligns with passage semantics."))

    score = max(5, score)

    st.metric("Estimated Retrieval Quality Index", f"{score}%", delta=f"{score - 100}%" if score < 100 else "Optimal 100%")
    st.progress(score / 100.0)

    st.markdown("#### Factor Breakdown Diagnostic")
    df_bd = pd.DataFrame(breakdowns, columns=["Pipeline Factor", "Impact", "Diagnostic Explanation"])
    st.dataframe(df_bd, use_container_width=True)

    if score >= 90:
        st.success("🌟 **Production Grade**: Your configuration achieves peak precision and recall across enterprise RAG benchmarks.")
    elif score >= 60:
        st.warning("⚠️ **Suboptimal**: Retrieval accuracy will suffer from occasional hallucination and missed facts.")
    else:
        st.error("🚨 **Critical Defects Detected**: Pipeline will suffer catastrophic retrieval failures.")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Factor Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why does removing stop-words (e.g. 'not', 'never', 'with', 'above') severely degrade modern transformer embeddings?**
    * **Answer**: In classical TF-IDF or bag-of-words models, stop words appeared everywhere and provided zero lexical signal.
    However, **Transformer attention mechanisms** rely heavily on functional stop words to determine relational meaning and polarity.
    * Removing *"not"* turns *"The medication is **not** safe for pregnant patients"* into *"The medication safe pregnant patients"*, completely inverting the clinical meaning and causing fatal retrieval errors!

    **Q2: How does subword token fragmentation impact embeddings on technical domain corpora?**
    * **Answer**: BPE and WordPiece tokenizers are trained on general web text. Rare technical acronyms, chemical formulas, or code identifiers (e.g. `CVE-2024-3094`, `hydroxychloroquine`, `torch.nn.functional`) do not exist in the vocabulary and are shattered into 5-8 subwords (e.g. `['hyd', 'ro', 'xy', 'ch', 'lor', 'o', 'qui', 'ne']`).
    This forces the model to synthesize a vector across multiple generic syllables, diluting semantic precision unless the model is fine-tuned on domain corpora.

    **Q3: Why do 2-word keyword queries perform worse in dense vector search than full natural questions?**
    * **Answer**: Dense embedding models are trained on continuous language manifolds. A 2-word fragment like `"s3 glacier"` has very little grammatical context or syntactic direction.
    A full question like `"How long does it take to retrieve archived files from S3 Glacier Flexible Archive?"` provides rich semantic coordinates that point directly toward explanatory answer passages in the vector space.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Applying Classical NLP Stemming / Lemmatization**:
      * *Issue*: Passing text through PorterStemmer before embedding (e.g. converting *"running"* to *"run"*).
      * *Consequence*: Destroys grammatical tense, part-of-speech context, and embedding accuracy.
      * *Fix*: Feed raw, unstemmed UTF-8 text directly to modern transformer embedding models.

    * **Trap 2: Ignoring Embedding Model Max Sequence Length Limits**:
      * *Issue*: Chunking at 1,000 words on an embedding model with a 512-token limit.
      * *Consequence*: The model silently discards the second half of every chunk without throwing an error!
      * *Fix*: Always verify `max_seq_length` and measure chunks strictly in tokenizer tokens.

    * **Trap 3: Cross-Language Semantic Misalignment**:
      * *Issue*: Querying in Spanish against an English corpus using a monolingual English embedding model.
      * *Consequence*: Near-zero similarity because English and Spanish words occupy completely different subspaces.
      * *Fix*: Deploy true multilingual models (e.g. `multilingual-e5-large` or `text-embedding-3-small`).
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: The Stop-Word Polarity Trap**
      * *Definition*: The severe degradation in transformer embeddings caused by stripping functional words like 'not' or 'never', inverting sentence meaning.
    * **Flashcard 2: Subword Fragmentation Rate**
      * *Definition*: The number of subword tokens a specialized domain term is split into by a general-purpose BPE tokenizer.
    * **Flashcard 3: Asymmetric Query Gap**
      * *Definition*: The structural mismatch between short keyword queries and long informational document passages.
    * **Flashcard 4: Cross-Lingual Alignment**
      * *Definition*: The ability of multilingual embedding models to map equivalent sentences in different languages to identical coordinates in vector space.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Factors Affecting Embedding Quality:

- [ ] I can trace the 7 upstream pipeline factors influencing vector search accuracy.
- [ ] I understand why stop-word removal and stemming harm transformer embeddings.
- [ ] I know how subword token fragmentation degrades specialized technical terms.
- [ ] I can articulate the asymmetric gap between keyword queries and explanatory passages.
- [ ] I understand how cross-lingual models align representations across languages.
""")
