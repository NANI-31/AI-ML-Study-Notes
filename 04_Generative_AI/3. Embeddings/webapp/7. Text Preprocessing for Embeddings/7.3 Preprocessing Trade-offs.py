import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import re
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="7.3 Preprocessing Trade-offs - Embeddings Curriculum",
    page_icon="⚖️",
    layout="wide"
)

# Title & Overview
st.title("7.3 Preprocessing Trade-offs: Over-Cleaning Pitfalls, Context Preservation, & Structural Integrity")
st.caption("Chapter 7: Text Preprocessing for Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
A dangerous habit among machine learning practitioners transitioning from classical NLP (TF-IDF, BM25, Bag-of-Words) to Modern Dense Transformers is **over-cleaning**.
In classical search, stripping punctuation, removing stopwords, lowercasing everything, and stemming words down to their morphological roots (*"operating"* -> *"oper"*) was necessary to collapse sparse vocabulary dimensions.

In modern **Dense Neural Embeddings**, however:
* Transformers are pre-trained on billions of tokens of rich, grammatical natural language.
* They thrive on nuance: punctuation encodes syntactic clauses, stopwords encode logical conditions (*"if"*, *"then"*, *"not"*), and capitalization encodes proper nouns.
* **Over-cleaning destroys this rich semantic fabric**, crippling retrieval performance!

This lesson evaluates the engineering trade-offs between sanitizing genuine formatting noise and preserving critical semantic signals.
""")

st.divider()

# Section 1: The Spectrum: Under-Cleaning vs. Optimal Cleaning vs. Over-Cleaning
st.header("1. The Preprocessing Spectrum: Under-Cleaning vs. Over-Cleaning", divider="rainbow")

col_spec1, col_spec2 = st.columns([1, 1])

with col_spec1:
    st.markdown("""
    ### The 3 Zones of Preprocessing:
    * **1. Under-Cleaning (Noise Contamination)**:
      * Leaving raw HTML scripts, PDF running footers, and non-printable control characters.
      * *Consequence*: Wastes token budgets; pulls diverse documents toward shared boilerplate clusters.
    * **2. Optimal Modern Cleaning (Semantic Preservation)**:
      * Strip HTML tags, fix broken line hyphens, and normalize Unicode (NFKC).
      * **Preserve**: Casing, punctuation, stopwords, numbers, version tags, and markdown heading hierarchy (`##`).
      * *Outcome*: Cleanest possible signal with 100% semantic fidelity.
    * **3. Over-Cleaning (Semantic Amputation)**:
      * Applying legacy NLP techniques: aggressive lowercasing, stopword stripping, stemming, and removing all punctuation.
      * *Disaster Example*:
        * Original: *"Do NOT administer penicillin if the patient has a history of anaphylaxis."*
        * Over-cleaned: *"administer penicillin patient history anaphylaxis"*
        * The critical negation *"NOT"* and condition *"if"* are wiped out, inverting medical meaning!
    """)

with col_spec2:
    st.markdown("### The Preprocessing Sweet Spot")
    spec_dot = """
    digraph PreprocessSpectrum {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        UnderClean [label="Zone 1: Under-Cleaning\\nRaw HTML, PDF footers, null bytes\\nHigh Token Waste | False Clusters", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];
        
        OptimalClean [label="Zone 2: Optimal Clean (Sweet Spot)\\nFormat sanitized, Unicode normalized\\nPunctuation, casing, & negation preserved", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];
        
        OverClean [label="Zone 3: Over-Cleaning (Legacy Trap)\\nLowercased, stemmed, stopwords removed\\nSemantics Inverted | Context Lost", fillcolor="#FEE2E2", color="#EF4444", style="filled,bold"];

        UnderClean -> OptimalClean [label="Format sanitization"];
        OptimalClean -> OverClean [label="Aggressive legacy NLP\\n(Destroys nuance)"];
    }
    """
    st.graphviz_chart(spec_dot, use_container_width=False)

st.markdown("### Legacy NLP vs. Modern Transformer Preprocessing Comparison")
st.markdown("""
| Preprocessing Step | Classical NLP (TF-IDF / BM25) | Modern Dense Embeddings (Transformers) | Rationale for Dense Embeddings |
| :--- | :--- | :--- | :--- |
| **Stopword Removal** | Mandatory (reduces sparsity) | **STRICTLY FORBIDDEN** | Negation (*"not"*), prepositions (*"under"*), and conjunctions define sentence logic |
| **Stemming / Lemmatization** | Common (collapses forms) | **STRICTLY FORBIDDEN** | Destroys tense, tone, and grammatical role; subword tokenizers handle inflection natively |
| **Lowercasing** | Mandatory (merges counts) | **DISCOURAGED** | Cased tokenizers distinguish proper nouns (*"Apple"*), acronyms (*"US"*), and entities |
| **Punctuation Removal** | Common | **STRICTLY FORBIDDEN** | Question marks, commas, and periods define self-attention clause boundaries |
| **HTML / Footer Stripping** | Optional | **MANDATORY** | Eliminates boilerplate token waste and false semantic clustering |
""")

st.divider()

# Section 2: Production Code: Balanced Preprocessing Pipeline
st.header("2. Production Implementation: Context-Preserving Sanitizer", divider="rainbow")

st.markdown("""
The production utility below demonstrates the **Golden Mean** of preprocessing: aggressively removing format noise and non-printable bytes while strictly preserving punctuation, grammatical stopwords, and proper noun casing.
""")

st.code(r'''
import unicodedata
import re
from typing import str

class ContextPreservingSanitizer:
    """
    Production-grade text preprocessor optimized for dense transformer embeddings.
    Sanitizes structural noise while strictly protecting semantic tokens.
    """
    def __init__(self):
        # Match HTML tags and script/style content
        self.html_tag_re = re.compile(r'<[^>]+>')
        # Match non-printable control characters
        self.control_chars_re = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufeff\u200b-\u200f]')
        # Match PDF hyphenated line breaks
        self.hyphen_re = re.compile(r'(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)')

    def sanitize(self, raw_text: str) -> str:
        """Sanitizes formatting noise without destroying linguistic nuance."""
        if not raw_text:
            return ""

        # 1. Strip non-printable and invisible control bytes
        text = self.control_chars_re.sub('', raw_text)

        # 2. Unicode compatibility normalization (NFKC)
        text = unicodedata.normalize('NFKC', text)

        # 3. Stitch PDF split hyphens
        text = self.hyphen_re.sub(r'\1\2', text)

        # 4. Strip HTML tags (preserving inner text)
        text = self.html_tag_re.sub(' ', text)

        # 5. Normalize whitespace runs while keeping paragraph breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        # CRITICAL: Do NOT lowercase, do NOT strip punctuation, do NOT remove stopwords!
        return text.strip()
''', language="python")

# Actual Python implementation for the interactive workbench
class CleaningTradeoffSimulator:
    def __init__(self):
        # Demonstrative sentences with sensitive nuance:
        self.benchmarks = {
            "Medical Negation": {
                "original": "Do NOT administer penicillin if the patient has a history of anaphylaxis.",
                "query": "Is it safe to give penicillin to allergic patients?",
                # Coordinate axes: [Safety, Danger/Prohibition, Medical Treatment, Allergy, History]
                "vec_optimal": np.array([0.05, 0.94, 0.70, 0.88, 0.30]),
                # Over-cleaned loses 'NOT' and 'if', vector flips toward 'Safety'
                "vec_overcleaned": np.array([0.85, 0.15, 0.72, 0.85, 0.32]),
                "vec_query": np.array([0.10, 0.92, 0.65, 0.89, 0.25])
            },
            "Acronym vs Common Noun": {
                "original": "The US Department of State met with WHO representatives in Geneva.",
                "query": "Diplomatic meetings of the United States government",
                # Coordinate axes: [Diplomacy, Nation State, Health Agency, Geography, Generic Pronoun]
                "vec_optimal": np.array([0.92, 0.95, 0.40, 0.35, 0.02]),
                # Overcleaned 'US' -> 'us' (pronoun), losing state identity
                "vec_overcleaned": np.array([0.70, 0.25, 0.35, 0.30, 0.85]),
                "vec_query": np.array([0.90, 0.94, 0.20, 0.30, 0.05])
            },
            "Technical Version Specificity": {
                "original": "Upgrade to Python 3.12 for per-interpreter GIL and improved exception groups.",
                "query": "Python 3.12 GIL improvements and multi-threading",
                # Coordinate axes: [Python Language, Version Specifics, Threading/GIL, Upgrade Action, General Code]
                "vec_optimal": np.array([0.90, 0.95, 0.92, 0.40, 0.25]),
                # Over-cleaned strips numbers and punctuation '3.12' -> empty
                "vec_overcleaned": np.array([0.88, 0.10, 0.65, 0.35, 0.80]),
                "vec_query": np.array([0.89, 0.96, 0.94, 0.35, 0.20])
            }
        }

    def evaluate_retrieval(self, test_key: str) -> Dict[str, Any]:
        item = self.benchmarks[test_key]
        q = item["vec_query"] / np.linalg.norm(item["vec_query"])
        
        opt = item["vec_optimal"] / np.linalg.norm(item["vec_optimal"])
        over = item["vec_overcleaned"] / np.linalg.norm(item["vec_overcleaned"])
        
        score_opt = float(np.dot(q, opt))
        score_over = float(np.dot(q, over))
        
        return {
            "score_optimal": score_opt,
            "score_overcleaned": score_over,
            "delta_pct": ((score_opt - score_over) / score_opt) * 100.0
        }

st.divider()

# Section 3: Interactive Workbench: Over-Cleaning Destruction Simulator
st.header("3. Interactive Workbench: Over-Cleaning Semantic Degradation Simulator", divider="rainbow")

st.markdown("""
Observe in real time how **over-cleaning** (lowercasing proper nouns, stripping numbers/punctuation, and removing stopwords like *NOT*) destroys embedding fidelity and causes severe retrieval rank drops.
""")

tradeoff_sim = CleaningTradeoffSimulator()

col_bench_l, col_bench_r = st.columns([1, 1])

with col_bench_l:
    st.subheader("1. Select Nuanced Case Study")
    case_choice = st.selectbox(
        "Select Sensitive Nuance Scenario:",
        list(tradeoff_sim.benchmarks.keys())
    )

    case_data = tradeoff_sim.benchmarks[case_choice]
    
    st.markdown("#### Input Text Comparison:")
    st.markdown(f"**Original Text**: `{case_data['original']}`")
    
    # Overcleaned representation
    if case_choice == "Medical Negation":
        over_text = "administer penicillin patient history anaphylaxis"
    elif case_choice == "Acronym vs Common Noun":
        over_text = "us department state met representatives geneva"
    else:
        over_text = "upgrade python per interpreter gil improved exception groups"
        
    st.markdown(f"**Over-Cleaned Legacy Text**: `{over_text}`")
    st.markdown(f"**User Search Query**: *\"{case_data['query']}\"*")

with col_bench_r:
    st.subheader("2. Retrieval Cosine Similarity Comparison")
    res = tradeoff_sim.evaluate_retrieval(case_choice)

    m1, m2 = st.columns(2)
    m1.metric("Optimal Preserved Preprocessing", f"{res['score_optimal']:.4f}")
    m2.metric("Over-Cleaned Legacy Preprocessing", f"{res['score_overcleaned']:.4f}", delta=f"-{res['delta_pct']:.1f}%", delta_color="inverse")

    st.markdown("#### Performance Impact Summary")
    df_eval = pd.DataFrame({
        "Pipeline Strategy": [
            "Optimal Semantic Preservation (NFKC, Casing & Punctuation Retained)",
            "Legacy Over-Cleaning (Lowercased, Numbers Stripped, Stopwords Removed)"
        ],
        "Cosine Similarity with Query": [
            res["score_optimal"],
            res["score_overcleaned"]
        ],
        "Quality Status": [
            "✅ High Fidelity Match",
            "❌ Severe Semantic Degradation"
        ]
    })
    st.dataframe(
        df_eval.style.format({"Cosine Similarity with Query": "{:.4f}"}),
        use_container_width=True
    )

    st.error(f"⚠️ **Retrieval Degradation**: Over-cleaning resulted in a **{res['delta_pct']:.1f}% drop** in cosine alignment because vital syntactic signals (negation, acronyms, or numbers) were discarded!")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Preprocessing Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why is stemming (e.g. Porter Stemmer) counter-productive when preparing text for modern Transformer embeddings?**
    * **Answer**:
      * In classical NLP (TF-IDF), stemming chopped off suffixes to conflate words (`organize`, `organizes`, `organizing` -> `organ`).
      * Modern Transformer models use **Subword Tokenization** (WordPiece, BPE) where the vocabulary already includes common prefixes, roots, and suffixes.
      * Furthermore, the transformer's self-attention layers depend on grammatical inflection: the difference between past tense (*"agreed"*), future tense (*"will agree"*), and continuous (*"agreeing"*) alters contextual semantics.
      * Stemming words destroys grammatical syntax and forces subword tokenizers to handle artificial non-words, severely corrupting embeddings.

    **Q2: What happens when an engineer strips punctuation from documents before feeding Sentence Transformers?**
    * **Answer**:
      * Punctuation marks (periods, commas, question marks, colons, hyphens) act as **syntactic boundary anchors** in self-attention.
      * Attention heads learn to distribute weights within clauses separated by commas, and terminate sentence scopes at periods.
      * Stripping punctuation concatenates adjacent sentences into an unpunctuated stream, causing cross-clause attention bleeding where the predicate of Sentence 1 falsely attaches to the subject of Sentence 2.

    **Q3: How do you decide which metadata to clean from text vs which to preserve in payload fields?**
    * **Answer**:
      * **Rule of Thumb**:
        * If the metadata is **syntactic structure or boilerplate** (e.g. page numbers, author email, copyright notice), strip it from the embedding text and store it only in the structured metadata payload.
        * If the metadata is **topical or categorical context** (e.g. document title, chapter header, breadcrumb path), prepend it to the chunk text (`"Title: {title} | Section: {section} | Content: {chunk}"`) so the vector space reflects the global topic.
    """)

with traps_tab:
    st.markdown(r"""
    * **Trap 1: Applying NLTK Default Stopword Lists**:
      * *Issue*: Importing `nltk.corpus.stopwords.words('english')` and filtering documents.
      * *Consequence*: Wipes out vital words like `not`, `no`, `never`, `under`, `before`, and `after`. Sentences containing negative restrictions flip into positive assertions.
      * *Fix*: Never apply stopword filtering to inputs for dense embedding models.

    * **Trap 2: Stripping Numbers and Version Identifiers**:
      * *Issue*: Running regex `re.sub(r'\d+', '', text)` to remove numbers.
      * *Consequence*: Destroys product model numbers (iPhone 15), software versions (Python 3.12), financial quantities ($50M), and dates (2024). Queries searching for specific versions will fail.
      * *Fix*: Preserve numbers and decimal points.

    * **Trap 3: Preserving Base64 Image Blobs inside Markdown**:
      * *Issue*: Ingesting markdown documents containing inline base64 image strings (`data:image/png;base64,iVBORw...`).
      * *Consequence*: The base64 string consumes 5,000+ tokens of meaningless characters, triggering context truncation and producing garbage embeddings.
      * *Fix*: Strip inline base64 strings with regex before chunking.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Semantic Over-Cleaning**
      * *Definition*: The damaging practice of applying legacy NLP techniques (stemming, stopword removal, punctuation stripping) to dense transformer inputs.
    * **Flashcard 2: Subword Tokenizer Protection**
      * *Definition*: The capability of modern BPE/WordPiece tokenizers to handle inflections natively without requiring artificial stemming.
    * **Flashcard 3: Syntactic Attention Anchors**
      * *Definition*: Punctuation marks (commas, periods, colons) that guide self-attention heads in scoping grammatical clauses.
    * **Flashcard 4: Breadcrumb Context Injection**
      * *Definition*: Prepending structural document headers to chunk text to prevent semantic isolation without polluting payload fields.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Preprocessing Trade-offs:

- [ ] I can articulate why legacy NLP techniques like stemming and stopword removal destroy transformer embeddings.
- [ ] I understand how punctuation acts as a syntactic anchor for transformer self-attention mechanisms.
- [ ] I know how over-cleaning inverts medical, legal, and financial negation semantics.
- [ ] I know why numbers, version tags, and acronyms must be strictly preserved during sanitization.
- [ ] I can implement a context-preserving text sanitizer in Python that strikes the optimal balance between noise removal and semantic preservation.
""")
