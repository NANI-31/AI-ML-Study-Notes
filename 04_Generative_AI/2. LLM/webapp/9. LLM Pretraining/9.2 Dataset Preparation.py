"""
9.2 Dataset Preparation
=======================
Interactive Master Lesson exploring Text Cleaning, Duplicate Removal, Corrupted Data Filtering,
Language Identification, Safety Scrubbing, PII Masking, Document Packing, and Tokenization.
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
    page_title="9.2 Dataset Preparation - LLM Architecture",
    page_icon="🧹",
    layout="wide",
)

# Custom CSS for aesthetic typography & dark theme styling
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
st.title("🧹 9.2 Dataset Preparation")
st.caption(
    "Chapter 9: LLM Pretraining | Module 2 of 4"
)

st.markdown(
    """
Turning raw internet dumps into a pristine, pretraining-ready dataset is one of the most
complex distributed data engineering feats in computer science. The dataset preparation
pipeline systematically removes corrupted text, strips exact and near-duplicates, isolates target languages,
purges toxic and private personal information (PII), and segments and packs billions of documents
into optimal contiguous token sequences.
"""
)

# -----------------------------------------------------------------------------
# SECTION 1: ARCHITECTURAL OVERVIEW & THE DATA PREPARATION FACTORY
# -----------------------------------------------------------------------------
st.header("1. Architectural Overview & The Data Preparation Factory", divider="rainbow")

st.markdown(
    """
### The Multi-Stage Distributed Data Cleaning Pipeline
Raw internet archives (WARC files) pass through successive stages of distributed MapReduce / Ray / Spark jobs.
Each stage filters out noise and refines text quality before tokenization.
"""
)

# Graphviz Architectural Diagram
arch_dot = graphviz.Digraph(
    "dataset_prep_pipeline",
    node_attr={"shape": "box", "style": "filled", "fontname": "Helvetica"},
    edge_attr={"color": "#38BDF8", "fontname": "Helvetica"},
)
arch_dot.attr(bgcolor="#0B1120", rankdir="TB")

arch_dot.node("raw", "Raw WARC\\nHTML Scrapes", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("clean", "Text Cleaning\\nHTML & Mojibake", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("lang", "Language Filter\\nFastText > 0.65", fillcolor="#1E293B", fontcolor="#F8FAFC")
arch_dot.node("pii", "PII & Safety\\nRegex / Masking", fillcolor="#DC2626", fontcolor="#F8FAFC")
arch_dot.node("dedup", "Deduplication\\nMinHash / Suffix", fillcolor="#0369A1", fontcolor="#F8FAFC")
arch_dot.node("pack", "Packing & Tokens\\n8k Packed Chunks", fillcolor="#047857", fontcolor="#F8FAFC")

arch_dot.edge("raw", "clean")
arch_dot.edge("clean", "lang")
arch_dot.edge("lang", "pii")
arch_dot.edge("pii", "dedup")
arch_dot.edge("dedup", "pack", label="Tokenized Arrow / Bin")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(arch_dot, use_container_width=True)

st.markdown(
    """
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Zero Padding via Sequence Packing**: Modern pretraining (Llama 3, Mistral) never pads short documents with `<pad>` tokens. Instead, multiple documents are concatenated together with `<|endoftext|>` delimiters into exact 8,192-token buffers. Attention matrices utilize 2D block masks so documents inside the same buffer do not attend to each other.
* **PII Anonymization at Scale**: Strict regulatory frameworks (GDPR, CCPA) demand aggressive scrubbing of private identifiable information (emails, phone numbers, IP addresses, credit card numbers) before training to prevent models from regurgitating user data during inference.
"""
)

# Data Cleaning Stages Table
stages_df = pd.DataFrame(
    {
        "Pipeline Stage": ["Text Normalization", "Corrupted Data Pruning", "Language Identification", "PII / Toxic Scrubbing", "Deduplication", "Document Packing"],
        "Tools Used": ["Trafilatura, ftfy, regex", "Perplexity filter, Gopher heuristics", "FastText, CLD3", "Microsoft Presidio, Detoxify", "MinHash LSH, Suffix Arrays", "Hugging Face Datasets, Arrow"],
        "Throughput": ["~50 GB/s on CPU clusters", "~100 MB/s per core", "~500k words/sec per worker", "~50k words/sec per GPU", "Distributed Spark cluster", "Multi-threaded Rust tokenizers"],
        "Typical Drop Rate": ["20% - 30% of raw bytes", "10% - 15% of pages", "5% - 10% non-target", "2% - 5% policy violations", "20% - 35% near-duplicates", "0% (lossless packing)"],
    }
)
st.dataframe(stages_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: CLEANING TEXT
# -----------------------------------------------------------------------------
st.header("2. Cleaning Text (HTML Normalization, Unicode & Whitespace)", divider="rainbow")

st.markdown(
    """
Raw scraped text from Common Crawl is full of HTML artifacts, navigation menus, ads, and broken whitespace.
Text cleaning standardizes text representations:

### 1. HTML Boilerplate Extraction
Raw HTML must be parsed to extract the primary text article while stripping:
* Navigation bars, header links, and footer disclaimers.
* Advertisements, cookie consent banners, and tracking scripts (`<script>`, `<style>`).
Tools like `trafilatura` and `resiliparse` use DOM tree density heuristics to isolate main article content.

### 2. Unicode Normalization & Whitespace Standardization
* **NFC Normalization**: Converts disparate Unicode representations of accented characters (e.g., `e` + accent vs. single accented character `é`) into standard canonical form.
* **Whitespace Cleaning**: Collapses repeated spaces, strips trailing spaces, and standardizes carriage returns (`\\r\\n` -> `\\n`).
"""
)

# -----------------------------------------------------------------------------
# SECTION 3: REMOVING DUPLICATES
# -----------------------------------------------------------------------------
st.header("3. Removing Duplicates (Exact URL, Substring, and MinHash Stages)", divider="rainbow")

st.markdown(
    """
Duplicate removal operates at multiple granularity levels:
1. **URL-Level Deduplication**: Deduplicating pages with identical canonical URLs or identical domain redirects.
2. **Exact Document Deduplication**: Computing 64-bit cryptographic hashes (e.g. xxHash, SHA-256) of document contents and retaining only the first occurrence.
3. **Fuzzy MinHash Deduplication**: Grouping near-identical pages (e.g. the same article with different advertisements) using Locality Sensitive Hashing.
4. **Exact Substring Deduplication**: Using suffix arrays to identify and strip common template boilerplate paragraphs repeated across millions of pages (e.g. "Subscribe to our newsletter!").
"""
)

# -----------------------------------------------------------------------------
# SECTION 4: REMOVING CORRUPTED DATA
# -----------------------------------------------------------------------------
st.header("4. Removing Corrupted Data (Mojibake, Incomplete Files, Low-Quality OCR)", divider="rainbow")

st.markdown(
    """
Corrupted text introduces high-loss noise into transformer attention matrices:
* **Mojibake**: Garbled character strings resulting from decoding text using the incorrect character set (e.g., viewing UTF-8 text as Windows-1252 produces `"Ã©"` instead of `"é"`). The `ftfy` library automatically identifies and corrects known encoding reversal tables.
* **Low-Quality OCR**: Scanned PDFs frequently produce hallucinated characters (`"th1s pr0blem"`), broken words, and irregular hyphenation.
* **Heuristic Drop Flags**: Pages with abnormally low punctuation ratios, excessive uppercase characters (>30%), or high frequencies of repetitive strings are purged.
"""
)

# -----------------------------------------------------------------------------
# SECTION 5: LANGUAGE FILTERING & FASTTEXT CLASSIFICATION
# -----------------------------------------------------------------------------
st.header("5. Language Filtering & FastText Classification", divider="rainbow")

st.markdown(
    """
Foundation models require intentional linguistic mixtures (e.g. 80% English, 10% European languages, 10% Asian languages).
* **FastText Language ID**: A lightweight, highly optimized n-gram linear classifier (`lid.176.bin`) that classifies text into 176 languages in microseconds.
* **Confidence Thresholding**: Documents are typically required to have `P(target_language) >= 0.65`.
* **Corrupted Multi-Language Spans**: Pages with rapid sentence-by-sentence language alternation (frequent in machine-translated spam blogs) are identified and discarded.
"""
)

# -----------------------------------------------------------------------------
# SECTION 6: TOXIC-CONTENT & SAFETY FILTERING
# -----------------------------------------------------------------------------
st.header("6. Toxic-Content & Safety Filtering (Hate Speech, CSAM, Adult Content)", divider="rainbow")

st.markdown(
    """
Safety filtering prevents foundation models from learning harmful, illegal, or abusive representations:
* **Blocklist Filtering**: Discarding pages originating from domains associated with malware, known hate groups, or adult content.
* **Classifier-Based Content Filtering**: Fast linear or lightweight Transformer models (e.g. Detoxify) score documents on toxicity, sexual violence, hate speech, and harassment.
* **CSAM Scrubbing**: Rigorous multi-stage hash-matching against international missing-children registries and domain blocklists.
"""
)

# -----------------------------------------------------------------------------
# SECTION 7: PII FILTERING
# -----------------------------------------------------------------------------
st.header("7. PII Filtering (Regex Masking, Presidio, Privacy Anonymization)", divider="rainbow")

st.markdown(
    """
Personally Identifiable Information (PII) must be scrubbed to protect privacy:
* **Regex Patterns**: Detect standard formats:
  * Email addresses (`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}`)
  * Phone numbers, Social Security Numbers (SSNs), Credit Card Numbers, IP addresses.
* **NER-Based Anonymization**: Named Entity Recognition models flag real human names, residential addresses, and private contact profiles, replacing them with generic placeholders:
  ```text
  "Contact John Doe at jdoe@example.com" -> "Contact <NAME> at <EMAIL>"
  ```
"""
)

# -----------------------------------------------------------------------------
# SECTION 8: DOCUMENT SEGMENTATION & PACKING STRATEGIES
# -----------------------------------------------------------------------------
st.header("8. Document Segmentation & Packing Strategies", divider="rainbow")

st.markdown(
    """
In standard NLP, inputs are padded to the maximum batch length with `<pad>` tokens.
In pretraining at 15-trillion token scale, **padding is strictly forbidden**:
* If documents average 1,500 tokens, padding to 8,192 wastes **81% of GPU compute** on pad tokens!

### The Sequence Packing Solution
Documents are packed contiguously into continuous 8,192-token buffers:

```text
[Doc 1 (2,100 tok)] <|endoftext|> [Doc 2 (4,300 tok)] <|endoftext|> [Doc 3 (1,792 tok)]
Total = Exactly 8,192 Tokens (100% Compute Utilization!)
```

### Attention Masking in Packed Sequences
To prevent Doc 2 from attending to Doc 1:
* The model supplies a 2D cumulative block attention mask, or uses FlashAttention's variable-length (`varlen`) API which takes document cumulative length boundaries (`cu_seqlens = [0, 2100, 6400, 8192]`).
"""
)

# -----------------------------------------------------------------------------
# SECTION 9: TOKENIZATION AT PRETRAINING SCALE
# -----------------------------------------------------------------------------
st.header("9. Tokenization at Pretraining Scale (Byte-Pair Encoding, Vocabulary Caching)", divider="rainbow")

st.markdown(
    """
Tokenizing 15 trillion tokens requires processing hundreds of terabytes of text:
* **High-Throughput Multi-Threaded Tokenizers**: Implemented in Rust (e.g. Hugging Face `tokenizers`, OpenAI `tiktoken`) utilizing SIMD and rayon multi-threading.
* **Direct Binary Serialization**: Rather than saving text or JSON, tokenized integer IDs are memory-mapped directly into flat binary files (`uint16` or `uint32` arrays) using NumPy `memmap` or Apache Arrow.
* **Zero Host-to-Device Latency**: During training, GPUs stream token IDs directly from NVMe SSDs into GPU VRAM using GPUDirect Storage (GDS).
"""
)

# -----------------------------------------------------------------------------
# SECTION 10: PRODUCTION-GRADE PRETRAINING DATA PROCESSING PIPELINE
# -----------------------------------------------------------------------------
st.header("10. Production-Grade Pretraining Data Processing & Packing Pipeline", divider="rainbow")

st.markdown(
    """
A complete, production-ready implementation of PII scrubbing and zero-pad sequence packing:
"""
)

code_snippet = """import re
from typing import List, Dict, Generator

class TextCleanerPII:
    # Production regex-based PII anonymizer and cleaner
    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+")
    PHONE_REGEX = re.compile(r"\\b\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b")
    IPV4_REGEX = re.compile(r"\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b")

    @classmethod
    def clean_and_mask_pii(cls, text: str) -> str:
        # Standardize whitespace
        text = re.sub(r"\\r\\n", "\\n", text)
        text = re.sub(r"[ \\t]+", " ", text)
        
        # Mask PII
        text = cls.EMAIL_REGEX.sub("<EMAIL>", text)
        text = cls.PHONE_REGEX.sub("<PHONE>", text)
        text = cls.IPV4_REGEX.sub("<IP_ADDRESS>", text)
        return text.strip()

def pack_sequences(
    tokenized_docs: List[List[int]],
    max_seq_len: int = 8192,
    eos_token_id: int = 128001,
) -> Generator[Dict[str, List[int]], None, None]:
    # Packs variable-length tokenized documents into fixed max_seq_len buffers
    current_buffer = []
    current_doc_indices = []

    for doc_idx, doc in enumerate(tokenized_docs):
        doc_with_eos = doc + [eos_token_id]
        
        # If document is longer than max_seq_len, chunk it
        while len(doc_with_eos) > 0:
            remaining_space = max_seq_len - len(current_buffer)
            chunk = doc_with_eos[:remaining_space]
            doc_with_eos = doc_with_eos[remaining_space:]
            
            current_buffer.extend(chunk)
            current_doc_indices.extend([doc_idx] * len(chunk))
            
            if len(current_buffer) == max_seq_len:
                yield {
                    "input_ids": current_buffer,
                    "doc_ids": current_doc_indices,
                }
                current_buffer = []
                current_doc_indices = []

    # Yield remaining buffer if not empty (padded or dropped in production)
    if current_buffer:
        yield {
            "input_ids": current_buffer,
            "doc_ids": current_doc_indices,
        }
"""
st.code(code_snippet, language="python")

# -----------------------------------------------------------------------------
# SECTION 11: INTERACTIVE DATASET CLEANING & PACKING WORKBENCH
# -----------------------------------------------------------------------------
st.header("11. Interactive Dataset Cleaning, PII Masking & Tokenization Workbench", divider="rainbow")

st.markdown(
    """
Test raw noisy text input against real-time **PII scrubbing**, **Whitespace normalization**,
and **Sequence Packing** visualization.
"""
)

sample_raw_input = st.text_area(
    "Raw Uncleaned Text Input",
    value="""Welcome to our platform! Please contact support at alice.smith@enterprise.org or call 415-555-0199 for assistance.
Our primary server is located at 192.168.1.104. 
Click here to read more... Subscribe to our newsletter!""",
    height=120,
)

# Cleaning logic using standard string operations
words = sample_raw_input.split()
cleaned_words = []
emails = []
phones = []
ips = []

for w in words:
    clean_w = w.strip(".,;:!?()[]\"'")
    if "@" in clean_w and "." in clean_w:
        emails.append(clean_w)
        cleaned_words.append("<EMAIL>")
    elif sum(c == "." for c in clean_w) == 3 and all(p.isdigit() for p in clean_w.split(".") if p):
        ips.append(clean_w)
        cleaned_words.append("<IP_ADDRESS>")
    elif sum(c.isdigit() for c in clean_w) >= 7 and ("-" in clean_w or "(" in clean_w):
        phones.append(clean_w)
        cleaned_words.append("<PHONE>")
    else:
        cleaned_words.append(w)

cleaned_text = " ".join(cleaned_words)

col_w1, col_w2 = st.columns(2)
with col_w1:
    st.subheader("Sanitized Text Output")
    st.code(cleaned_text, language="text")
with col_w2:
    st.subheader("Detected & Masked PII Entities")
    st.write(f"📧 **Emails Found**: {emails if emails else 'None'}")
    st.write(f"📞 **Phones Found**: {phones if phones else 'None'}")
    st.write(f"🌐 **IP Addresses Found**: {ips if ips else 'None'}")

# -----------------------------------------------------------------------------
# SECTION 12: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# -----------------------------------------------------------------------------
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions & Staff-Level Answers")

with st.expander("Q1: Why is sequence packing preferred over batch padding during LLM pretraining?"):
    st.markdown(
        """
        **Answer:**
        Natural text documents vary wildly in length (from 100 to 5,000 words).
        If documents are padded to the model's 8,192 context window using `<pad>` tokens, over 60-80% of all tokens in a batch would be non-informative padding. This would waste millions of dollars of GPU compute.
        Sequence packing concatenates multiple documents into fixed-length 8,192 buffers, ensuring 100% arithmetic throughput.
        """
    )

with st.expander("Q2: How do you prevent cross-document attention contamination when packing sequences?"):
    st.markdown(
        """
        **Answer:**
        When Document A and Document B share the same packed sequence buffer, Document B should not attend to Document A.
        This is prevented via:
        1. **2D Block Attention Masking**: Setting attention logits to `-inf` whenever queries and keys belong to different document indices.
        2. **FlashAttention Cumulative Sequence Lengths (`cu_seqlens`)**: Supplying document boundaries directly to the FlashAttention kernel, which limits attention computation to within each individual document block.
        """
    )

with st.expander("Q3: What is Mojibake and why does it corrupt tokenizer representations?"):
    st.markdown(
        """
        **Answer:**
        Mojibake occurs when text encoded in one format (e.g. UTF-8) is decoded as another (e.g. Latin-1 / Windows-1252), turning multi-byte characters like `’` into garbled strings like `â€™`.
        In Byte-Pair Encoding, this splits a single intended punctuation token into 3-4 bizarre low-frequency subword tokens, degrading model vocabulary efficiency.
        """
    )

with st.expander("Q4: Explain the trade-off between strict PII masking and preserving factual information."):
    st.markdown(
        """
        **Answer:**
        Overly aggressive regexes or NER models can accidentally mask public historical figures ("Barack Obama", "Albert Einstein") or legitimate public contact info (e.g. official support emails for public services).
        Production pipelines maintain public entity allowlists (e.g. from Wikipedia and Wikidata) so public knowledge is preserved while private personal records are anonymized.
        """
    )

with st.expander("Q5: Why are pretraining tokens stored as binary memory-mapped files rather than JSON/CSV?"):
    st.markdown(
        """
        **Answer:**
        Parsing JSON or CSV files during training consumes heavy CPU cycles and incurs Python garbage collection overhead.
        Storing tokens as contiguous binary `uint16` or `uint32` files on disk allows PyTorch workers to use `np.memmap()`, reading arrays directly from the OS page cache without memory copying.
        """
    )

st.markdown("### Production Traps & Failure Modes")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error(
        """
        **Trap 1: Cross-Document Attention Leakage in Packed Sequences**
        *Root Cause*: Packing documents without providing document boundaries to FlashAttention or masking attention matrices.
        *Production Fix*: Ensure attention masks or `cu_seqlens` are strictly enforced so documents inside a packed buffer are mutually invisible.
        """
    )
    st.warning(
        """
        **Trap 2: PII Anonymization Placeholders Corrupting Code**
        *Root Cause*: Regex masking turning Python variable names like `email_address = ...` or regex strings inside code into `<EMAIL>`.
        *Production Fix*: Exempt programming language source files from natural language PII scrubbing rules.
        """
    )
with col_t2:
    st.error(
        """
        **Trap 3: Tokenizer Out-of-Bounds on Unpadded Endings**
        *Root Cause*: Packing sequences right up to `max_seq_len` without appending `<|endoftext|>` delimiters between documents.
        *Production Fix*: Always inject explicit end-of-text separator tokens between joined documents.
        """
    )
    st.warning(
        """
        **Trap 4: Discarding Short Documents and Biasing Data Diversity**
        *Root Cause*: Blindly filtering out all documents under 100 words, thereby discarding valuable short poems, mathematical formulas, and terminal commands.
        *Production Fix*: Relax length filters for specialized domains (code, definitions, equations).
        """
    )

st.markdown("### Technical Flashcards: Rapid Recall")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("⚡ Flashcard 1: Sequence Packing Compute Benefit"):
        st.write("Why does sequence packing achieve near 100% compute efficiency?")
        st.info("It eliminates non-informative <pad> tokens by filling buffers with contiguous document segments.")

    with st.expander("⚡ Flashcard 2: Memory-Mapped Tokens"):
        st.write("What Python feature enables zero-copy streaming of token IDs from disk?")
        st.info("np.memmap() / Arrow memory mapping directly to OS page cache.")

with col_fc2:
    with st.expander("⚡ Flashcard 3: FastText Language ID Speed"):
        st.write("Why is FastText preferred over BERT for language identification?")
        st.info("It evaluates hundreds of thousands of words per second with minimal CPU footprint.")

    with st.expander("⚡ Flashcard 4: Mojibake Fixer"):
        st.write("What popular Python library automatically resolves broken Unicode encoding tables?")
        st.info("ftfy (fixes text for you).")

# -----------------------------------------------------------------------------
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# -----------------------------------------------------------------------------
st.header("13. Self-Assessment Checklist", divider="rainbow")

checklist_items = [
    "I understand the multi-stage architecture of distributed pretraining dataset preparation.",
    "I can explain why sequence packing is essential for 100% GPU compute utilization.",
    "I know how to prevent attention leakage between packed documents using cu_seqlens or 2D block masks.",
    "I understand how regex and NER models scrub Personally Identifiable Information (PII).",
    "I know how FastText language identification filters non-target corpora.",
    "I understand why pretraining tokens are stored as binary memory-mapped arrays rather than text.",
]

checked_count = 0
for item in checklist_items:
    if st.checkbox(item, key=f"chk_{hash(item)}"):
        checked_count += 1

if checked_count == len(checklist_items):
    st.success(
        "🎉 Mastery Demonstrated! You have successfully mastered Dataset Preparation, Text Cleaning, and Pretraining Token Packing."
    )
else:
    st.info(f"Progress: {checked_count} of {len(checklist_items)} concepts checked.")
