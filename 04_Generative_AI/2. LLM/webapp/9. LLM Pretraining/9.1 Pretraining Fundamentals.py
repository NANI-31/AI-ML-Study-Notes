"""
9.1 Pretraining Fundamentals
============================
Interactive Master Lesson exploring Self-Supervised Pretraining, Foundation Models,
Multi-Trillion Token Text Corpora, Web Scraping, Quality Filtering, and MinHash Deduplication.
"""

from typing import Dict, List, Tuple
import graphviz
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="9.1 Pretraining Fundamentals - LLM Architecture",
    page_icon="🏗️",
    layout="wide",
)

# Custom CSS for aesthetic styling & dark theme typography
st.markdown(
    """
<style>
    .metric-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .callout-box {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 12px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# TOP NAVIGATION / HERO
# -----------------------------------------------------------------------------
st.title("🏗️ 9.1 Pretraining Fundamentals")
st.caption(
    "Chapter 9: LLM Pretraining | Module 1 of 4"
)

st.markdown(
    """
Pretraining is the capital-intensive core of modern generative artificial intelligence.
By exposing massive transformer neural networks to trillions of tokens of diverse text, code,
and scientific literature, models absorb general world knowledge, grammar, reasoning heuristics,
and algorithmic synthesis capabilities in an entirely **self-supervised** manner.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & THE PRETRAINING FOUNDATION
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Pretraining Foundation", divider="rainbow")

st.markdown(
    """
### The Pretraining Ingestion Funnel
The pipeline transforms raw, noisy multi-petabyte internet dumps into structured, high-signal token streams
that power multi-month cluster pretraining runs.
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "pretraining_funnel",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("raw_web", "Raw Web Dumps\\nCommon Crawl (Petabytes)", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("clean", "Heuristic / PII Clean\\nLanguage ID, FastText", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("dedup", "Deduplication\\nMinHash LSH, Suffix Array", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("mixture", "Domain Mixture\\nCode, Books, STEM, Web", fillcolor="#047857", fontcolor="#F8FAFC")
arch_dot.node("cluster", "GPU Supercluster\\n16k H100s, 3D Parallelism", fillcolor="#7C3AED", fontcolor="#F8FAFC")
arch_dot.node("base_model", "Base Foundation Model\\n(Llama 3, Mistral)", fillcolor="#DC2626", fontcolor="#F8FAFC")

arch_dot.edge("raw_web", "clean")
arch_dot.edge("clean", "dedup")
arch_dot.edge("dedup", "mixture")
arch_dot.edge("mixture", "cluster", label="15 Trillion Tokens")
arch_dot.edge("cluster", "base_model")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Token Scale Explosion**: While GPT-3 (2020) trained on 300 billion tokens, Llama 3 (2024) trained on **15.6 trillion tokens** (over 50x more data).
* **Synthetic Data Ingestion**: As public internet text approaches exhaustion, frontier labs augment pretraining mixtures with synthetically generated textbooks, verified reasoning chains, and distilled code executions.
* **Aggressive Pruning**: Common Crawl contains >80% SEO spam, machine translations, and duplicate template text. Frontier labs filter out up to 85% of raw scrapes, retaining only top-tier content.
"""
)

# Comparison Table
pretrain_comp_df = pd.DataFrame(
    {
        "Model": ["GPT-2 (2019)", "GPT-3 (2020)", "Llama 1 (2023)", "Llama 2 (2023)", "Llama 3 (2024)"],
        "Training Tokens": ["40 Billion", "300 Billion", "1.4 Trillion", "2.0 Trillion", "15.6 Trillion"],
        "Primary Corpus": ["WebText (Reddit links)", "Common Crawl, Books, Wiki", "Common Crawl, GitHub, ArXiv", "C4, Fine-tuned mix", "Heavily Filtered Multi-Lingual Web"],
        "Deduplication": ["Exact string match", "Fuzzy Spark deduplication", "CCNet + MinHash", "Advanced MinHash LSH", "Semantic Clustering + MinHash"],
        "Compute (GPU Hours)": ["~1,000 V100", "~3,000 V100", "~82,000 A100", "~180,000 A100", "~7,700,000 H100 hours"],
    }
)
st.dataframe(pretrain_comp_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: WHAT IS PRETRAINING?
# -----------------------------------------------------------------------------
st.header("2. What is Pretraining? (Self-Supervised Knowledge Ingestion)", divider="rainbow")

st.markdown(
    """
**Pretraining** is the initial, computationally dominant phase where a model learns general representations
from raw text without human-annotated labels.
Instead of relying on supervised pairs `(x, y)`, the task is **self-supervised**:
the text itself provides the supervision by treating preceding words as inputs and upcoming words as targets.

### The Compression Hypothesis
By minimizing next-token cross-entropy loss over trillions of tokens, the network is forced to learn
the most compressed generative model of the world. To predict the next word in:
```text
The boiling point of water at sea level is 100 degrees ...
```
The model must implicitly store physical constants, grammatical syntax, and semantic ontology.
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: FOUNDATION MODEL TRAINING DYNAMICS
# -----------------------------------------------------------------------------
st.header("3. Foundation Model Training Dynamics & Extreme Scale", divider="rainbow")

st.markdown(
    """
Foundation model pretraining operates at unprecedented computational scales:
* **Cluster Hardware**: Runs across thousands of interconnected GPUs (e.g. 16,384 NVIDIA H100 SXM5 GPUs on 3.2 Tbps InfiniBand).
* **Fault Tolerance**: Hardware failures (GPU memory ECC errors, silent data corruption, optical transceiver drops) occur daily. Pretraining orchestrators implement asynchronous checkpointing and automatic node health remediation.
* **Energy Consumption**: A frontier 15-trillion token pretraining run consumes megawatt-hours of power, requiring gigawatt datacenter planning.
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: PRETRAINING DATASETS & CURATED MIXTURES
# -----------------------------------------------------------------------------
st.header("4. Pretraining Datasets & Curated Multi-Domain Mixtures", divider="rainbow")

st.markdown(
    """
A foundation model's personality, reasoning ceiling, and factual accuracy are determined by its **data mixture**.
Leading public and open-weight pretraining datasets include:

### Major Public Pretraining Corpora
1. **Common Crawl**: Open repository of web crawl data spanning billions of web pages. High noise, high volume.
2. **C4 (Colossal Clean Crawled Corpus)**: 750GB filtered Common Crawl dataset introduced for T5.
3. **RedPajama / Dolma / FineWeb**: Open reproducible 15T-token datasets replicating state-of-the-art pretraining mixtures with transparent filtering recipes.
4. **The Stack / StarCoder**: Multi-terabyte permissively licensed code repositories spanning 300+ programming languages.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: TEXT CORPORA TAXONOMY
# -----------------------------------------------------------------------------
st.header("5. Text Corpora Taxonomy (Web, Code, Books, Academic Papers)", divider="rainbow")

st.markdown(
    """
Balancing different data modalities is critical for model capability:
"""
)

col_corp1, col_corp2 = st.columns(2)
with col_corp1:
    st.subheader("High-Density Reasoning Domains")
    st.markdown(
        """
        * **Source Code (GitHub, StackOverflow)**: 10% to 20% of modern mixtures. Teaches rigorous logical structures, variable dependency tracing, and algorithmic execution.
        * **Scientific & Mathematical Literature (arXiv, PubMed)**: 5% to 10%. Instills quantitative reasoning, formal notation, and precision.
        * **Books & Curated Literature (Books3, Project Gutenberg)**: Provides long-form narrative coherence, discourse structure, and vocabulary richness.
        """
    )
with col_corp2:
    st.subheader("Breadth & World Knowledge Domains")
    st.markdown(
        """
        * **Curated Web Pages (Wikipedia, High-Quality News)**: High factual density and encyclopedic cross-references.
        * **Filtered General Web (Common Crawl)**: 50% to 70% of mixture. Provides broad world knowledge, conversational diversity, and multilingual breadth.
        """
    )

# -----------------------------------------------------------------------------
# SECTION 6: DATA COLLECTION PIPELINES
# -----------------------------------------------------------------------------
st.header("6. Data Collection Pipelines & Web Scraping at Terabyte Scale", divider="rainbow")

st.markdown(
    """
Modern data engineering pipelines process petabytes of raw WARC (Web ARChive) files:
1. **WARC Extraction**: Parsing HTML structure, extracting text bodies using libraries like `trafilatura` or `resiliparse`, and stripping boilerplate (navbars, footers, ads).
2. **Encoding Normalization**: Converting all documents into UTF-8, handling mojibake, and standardizing whitespace.
3. **Metadata Enrichment**: Extracting URL domain reputation, publication dates, and content length statistics.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: DATA QUALITY STANDARDS
# -----------------------------------------------------------------------------
st.header("7. Data Quality Standards & The High-Quality Token Scarcity", divider="rainbow")

st.markdown(
    """
The AI industry faces the **Token Wall**: high-quality human-written text on the internet is finite (estimated at 20-30 trillion tokens).
Consequently, quality filtering is paramount:
* **Garbage-In, Garbage-Out**: Training on low-quality web text increases perplexity and breeds hallucinations.
* **Data Pruning**: Quality filtering removes machine-generated spam, repetitive keyword farms, low-effort link lists, and auto-translated junk.
* **Perplexity Filtering**: Using a small, high-quality reference language model (e.g., trained on Wikipedia and high-quality books) to score candidate web documents. Documents with unusually high perplexity (gibberish) or unnaturally low perplexity (repetitive text) are discarded.
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: DATA FILTERING MECHANICS
# -----------------------------------------------------------------------------
st.header("8. Data Filtering Mechanics (Classifier, Gopher, FastText)", divider="rainbow")

st.markdown(
    """
Modern preprocessing applies a cascade of specialized filters:

### 1. Heuristic Rule-Based Filters (Gopher Recipe)
* **Word Length**: Remove documents where mean word length is outside 3 to 10 characters.
* **Symbol-to-Word Ratio**: Discard pages where symbol characters (`#`, `@`, `%`, `...`) exceed 10% of total tokens.
* **Ellipsis & Bullet Filtering**: Strip documents ending in excessive ellipses or repetitive boilerplate.
* **Stop Word Thresholds**: Verify natural language distribution (e.g., English text must contain common stop words like "the", "and", "of").

### 2. FastText Language Identification
FastText classifier filters non-target languages and flags mixed-language corrupted pages with >99% precision.

### 3. Toxic & PII Scrubbing
* Regex filters mask Social Security Numbers, credit cards, emails, and phone numbers.
* Fast linear classifiers flag hate speech, CSAM, and extreme toxicity.
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: DATA DEDUPLICATION ALGORITHMS
# -----------------------------------------------------------------------------
st.header("9. Data Deduplication Algorithms (MinHash, LSH, Suffix Arrays)", divider="rainbow")

st.markdown(
    """
The internet contains massive redundancy: identical news wire articles syndicated across thousands of local sites, open-source license headers repeated across millions of GitHub files, and SEO spam.
Training on duplicated data causes:
* **Memorization & Privacy Leakage**: The model memorizes exact verbatim sequences instead of learning concepts.
* **Degenerated Repetition Loops**: Models develop high probabilities for repetitive continuation loops during inference.

### MinHash & Locality Sensitive Hashing (LSH)
1. **Shingling**: Break document into character or word n-grams (e.g. 5-grams).
2. **Hash Signatures**: Apply `K` independent hash functions to compute minimum hash values:
   `MinHash(D) = [min_h1(D), min_h2(D), ..., min_hK(D)]`.
3. **LSH Bucketing**: Partition signatures into bands. Documents that share hash signatures in any band are candidate duplicates.
4. **Jaccard Similarity Comparison**: Pairs with Jaccard similarity `J(A, B) > 0.8` are clustered, and only one copy is retained.
"""
)

# -----------------------------------------------------------------------------
# SECTION 10: PRODUCTION-GRADE PRETRAINING DATA CLEANING PIPELINE
# -----------------------------------------------------------------------------
st.header("10. Production-Grade Pretraining Data Cleaning & MinHash Pipeline", divider="rainbow")

st.markdown(
    """
A complete, vectorized Python data filtering and MinHash deduplication module:
"""
)

code_snippet = """import re
import hashlib
from typing import List, Set, Dict

class DocumentFilter:
    # Heuristic and quality filter based on Gopher and FineWeb standards
    @staticmethod
    def is_valid_document(text: str) -> Tuple[bool, str]:
        words = text.split()
        if len(words) < 50:
            return False, "Too short (< 50 words)"
        if len(words) > 100000:
            return False, "Too long / auto-generated log dump"

        # Check mean word length
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3.0 or avg_word_len > 12.0:
            return False, f"Abnormal mean word length: {avg_word_len:.2f}"

        # Symbol-to-word ratio
        symbols = sum(1 for char in text if char in "!@#$%^&*()_+=[]{}|;:,.<>?")
        if symbols / len(text) > 0.20:
            return False, "Excessive symbol density (> 20%)"

        # Stop word check (English)
        stop_words = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i"}
        lower_words = set(w.lower() for w in words[:100])
        if len(stop_words.intersection(lower_words)) < 2:
            return False, "Missing common natural language stop words"

        return True, "Valid"

class MinHashDeduplicator:
    # 5-gram MinHash LSH deduplicator
    def __init__(self, num_hashes: int = 16, shingle_size: int = 5):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size

    def _get_shingles(self, text: str) -> Set[str]:
        words = text.lower().split()
        if len(words) < self.shingle_size:
            return set(words)
        return {" ".join(words[i : i + self.shingle_size]) for i in range(len(words) - self.shingle_size + 1)}

    def compute_signature(self, text: str) -> List[int]:
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_hashes

        signature = []
        for seed in range(self.num_hashes):
            min_val = float("inf")
            for shingle in shingles:
                h = int(hashlib.md5(f"{seed}_{shingle}".encode()).hexdigest(), 16)
                if h < min_val:
                    min_val = h
            signature.append(min_val)
        return signature

    @staticmethod
    def estimated_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 11: INTERACTIVE PRETRAINING DATA MIXTURE WORKBENCH
# -----------------------------------------------------------------------------
st.header("11. Interactive Pretraining Data Mixture & Quality Filtering Workbench", divider="rainbow")

st.markdown(
    """
Adjust pretraining data mixture weights and observe the simulated impact on model capabilities.
"""
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    web_pct = st.slider("Web Crawl Ratio (%)", min_value=10, max_value=80, value=50, step=5)
    code_pct = st.slider("Code Repositories Ratio (%)", min_value=5, max_value=40, value=20, step=5)
with col_ctrl2:
    stem_pct = st.slider("STEM & Math Literature (%)", min_value=5, max_value=30, value=15, step=5)
    books_pct = st.slider("Curated Books & High-Quality Articles (%)", min_value=5, max_value=30, value=15, step=5)

total_pct = web_pct + code_pct + stem_pct + books_pct
if total_pct != 100:
    st.warning(f"Total mixture sums to {total_pct}%. Normalizing to 100%...")
    norm_factor = 100.0 / total_pct
    web_pct = round(web_pct * norm_factor)
    code_pct = round(code_pct * norm_factor)
    stem_pct = round(stem_pct * norm_factor)
    books_pct = 100 - (web_pct + code_pct + stem_pct)

# Simulated downstream benchmark capabilities
code_score = min(88, int(20 + 1.8 * code_pct + 0.3 * stem_pct))
math_score = min(85, int(15 + 2.0 * stem_pct + 0.5 * code_pct))
general_knowledge = min(92, int(35 + 0.6 * web_pct + 0.8 * books_pct))
long_context = min(90, int(25 + 2.2 * books_pct + 0.3 * code_pct))

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("HumanEval Code", f"{code_score}%")
with col_m2:
    st.metric("GSM8K Math", f"{math_score}%")
with col_m3:
    st.metric("MMLU Knowledge", f"{general_knowledge}%")
with col_m4:
    st.metric("Long Context Coherence", f"{long_context}%")

mix_df = pd.DataFrame(
    {
        "Corpus Domain": ["Web Crawl (Common Crawl)", "Source Code (GitHub)", "STEM & Math (arXiv)", "Curated Books / Wiki"],
        "Mixture Share (%)": [web_pct, code_pct, stem_pct, books_pct],
    }
)
st.subheader("Configured Pretraining Data Mixture")
st.dataframe(mix_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why is MinHash preferred over exact MD5 hash matching for web deduplication?"):
    st.markdown(
        """
        **Answer:**
        Web documents frequently contain minor variations (different timestamps, page view counters, syndicated ads, slight formatting edits).
        An exact MD5 or SHA-256 hash treats documents with a single differing character as 100% distinct, missing nearly all near-duplicate syndicated web pages.
        MinHash computes Jaccard similarity between n-gram shingles, enabling locality-sensitive clustering of documents that share >80% overlapping content.
        """
    )

with st.expander("Q2: What is the impact of training on duplicated data according to research (Lee et al., 2021)?"):
    st.markdown(
        """
        **Answer:**
        1. **Memorization**: Models are 10x to 100x more likely to verbatim memorize sequences that appear more than 10 times in the corpus, posing severe privacy risks.
        2. **Repetition Degeneration**: The model allocates excessive probability to repetitive loops during inference.
        3. **Inefficient Compute**: Deduplicating Common Crawl removes up to 30% of tokens without hurting downstream perplexity, saving hundreds of thousands of GPU hours.
        """
    )

with st.expander("Q3: Why is code included in general-purpose foundation LLM pretraining mixtures?"):
    st.markdown(
        """
        **Answer:**
        Code forces the model to learn rigorous syntactic parsing, state tracking, logic flow, and execution traces.
        Empirically, adding 15-20% code to pretraining mixtures dramatically improves performance on non-code tasks, including commonsense reasoning, multi-step math problems (GSM8K), and structured JSON generation.
        """
    )

with st.expander("Q4: How does Perplexity Filtering work in pretraining data preparation?"):
    st.markdown(
        """
        **Answer:**
        A small, high-quality reference language model is trained on a clean corpus (Wikipedia and vetted books).
        Candidate web pages are evaluated through this reference model:
        * High perplexity indicates non-natural text, garbled OCR, machine translation spam, or corrupted encodings.
        * Extremely low perplexity indicates repetitive boilerplate (e.g., "terms and conditions" repeated 50 times).
        Filtering out documents in both tails dramatically elevates dataset quality.
        """
    )

with st.expander("Q5: What is the 'Token Wall' and how are frontier labs addressing it?"):
    st.markdown(
        """
        **Answer:**
        The 'Token Wall' refers to the projected exhaustion of high-quality, human-written public internet text by 2026.
        Labs address it via:
        1. Multi-epoch training on synthetic data generated and verified by reasoning models.
        2. Ingesting multimodal data (video, audio transcriptions).
        3. Partnering with proprietary publishers for paywalled scientific literature and private enterprise archives.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Aggressive Heuristic Filtering Discarding Valid Code & Math**
        *Root Cause*: Applying standard natural language rules (e.g. discarding text with >15% symbols) to Python code or LaTeX formulas.
        *Production Fix*: Segment files by MIME type / language and apply domain-specific filtering pipelines (code requires specialized AST parsers, not natural language filters).
        """
    )
    st.warning(
        """
        **Trap 2: Ignoring Benchmark Contamination in Pretraining Crawls**
        *Root Cause*: Training on web pages containing MMLU or HumanEval problem sets.
        *Production Fix*: Run strict 13-gram substring decontamination against all evaluation benchmark test sets before starting pretraining.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Memory Exhaustion During MinHash Shingling**
        *Root Cause*: Storing billions of n-gram shingle sets in RAM during distributed Spark deduplication.
        *Production Fix*: Stream shingles directly into integer hash buckets and utilize 64-bit hashing with disk-backed storage (RocksDB/Arrow).
        """
    )
    st.warning(
        """
        **Trap 4: Mojibake Character Corruption**
        *Root Cause*: Inconsistent handling of ISO-8859-1 vs. UTF-8 encodings in raw Common Crawl HTML.
        *Production Fix*: Run `ftfy` (fixes text for you) on all extracted documents to normalize unicode and repair broken character encoding chains.
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Llama 3 Token Count"):
        st.write("How many training tokens were used to pretrain Llama 3?")
        st.info("15.6 Trillion tokens.")

    with st.expander("⚡ Flashcard 2: MinHash Goal"):
        st.write("What mathematical quantity does MinHash estimate efficiently?")
        st.info("Jaccard similarity between document n-gram sets.")

with col_fc2:
    with st.expander("⚡ Flashcard 3: Self-Supervised Paradigm"):
        st.write("Where do training labels come from in self-supervised pretraining?")
        st.info("From the text itself (predicting the next token given preceding context).")

    with st.expander("⚡ Flashcard 4: Common Crawl Raw Volume"):
        st.write("What percentage of raw Common Crawl text is typically retained after quality filtering?")
        st.info("Only about 15% to 25% (up to 85% is discarded as noise/boilerplate).")

# -----------------------------------------------------------------------------
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("13. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the self-supervised pretraining paradigm and the compression hypothesis.",
    "I know the typical domain mixture breakdown of modern 15T-token foundation models.",
    "I can explain why code is critical for general reasoning and mathematical abilities.",
    "I understand why MinHash and Locality Sensitive Hashing (LSH) are used for fuzzy deduplication.",
    "I know the heuristic filtering criteria used in the Gopher and FineWeb data cleaning pipelines.",
    "I understand the 'Token Wall' and how synthetic data helps overcome high-quality text scarcity.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Pretraining Fundamentals, Data Mixtures, and Web Deduplication Engineering."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
