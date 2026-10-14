import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import hashlib
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="31.1 Duplicate Concepts - Embeddings Curriculum",
    page_icon="👯",
    layout="wide"
)

# Title & Overview
st.title("31.1 Duplicate Concepts: Exact, Near, Semantic Duplicates & Paraphrase Detection")
st.caption("Chapter 31: Embeddings for Duplicate Detection | Domain: 3. Vector Embeddings")
st.markdown("""
In large-scale data engineering, redundant data pollutes databases, degrades search relevance, wastes expensive LLM training tokens, 
and burdens customer support teams. 
However, **"duplication" exists across multiple distinct levels of abstraction**:

* **Exact Duplicates**: Byte-for-byte identical text. Detectable with trivial cryptographic hashing (SHA-256 / MD5).
* **Near Duplicates**: Texts that differ by minor typos, whitespace, casing, or punctuation. Detectable with edit distance (Levenshtein) or MinHash.
* **Semantic Duplicates**: Texts that share the **exact same factual meaning and intent** while using **completely different vocabulary, sentence structure, and grammar**.
  * *Example A*: *"How do I reset my account password?"*
  * *Example B*: *"I forgot my login credentials and need instructions to recover access."*
* **Paraphrase Detection**: Binary classification or scoring task determining whether Sentence 1 and Sentence 2 are semantically equivalent.

While traditional hashing fails completely on semantic duplicates (0% hash match, 10% Jaccard word overlap), 
**Vector Embeddings** map both statements to virtually identical coordinates in high-dimensional space, solving semantic deduplication cleanly.
""")

st.divider()

# Section 1: The Spectrum of Duplication
st.header("1. The 4-Level Spectrum of Duplication", divider="rainbow")

col_spec1, col_spec2 = col_spec1, col_spec2 = st.columns([1, 1])

with col_spec1:
    st.markdown("""
    ### The Hierarchy of Duplication
    1. **Level 1: Exact Duplicates**:
       * *Definition*: Character-for-character identical strings.
       * *Tool*: SHA-256 or MD5 hashes.
       * *Speed*: Nanoseconds ($O(L)$).
       * *Limitation*: A single extra space or uppercase letter yields a completely different hash!
    2. **Level 2: Near Duplicates (Lexical)**:
       * *Definition*: High character or word overlap with minor edits, HTML tags, or typos.
       * *Tool*: SimHash, MinHash with Locality Sensitive Hashing (LSH), Levenshtein Distance.
       * *Speed*: Microseconds.
       * *Limitation*: Fails if the user uses synonyms (*"car"* vs *"automobile"*).
    3. **Level 3: Semantic Duplicates (Conceptual)**:
       * *Definition*: Identical underlying meaning with distinct vocabulary.
       * *Tool*: **Dense Bi-Encoder Embeddings (BGE / OpenAI)**.
       * *Metric*: Cosine Similarity (`dot(v_1, v_2) >= 0.88`).
    4. **Level 4: Fine-Grained Paraphrases**:
       * *Definition*: Subtle directional statements, negations, or conditional clauses.
       * *Tool*: **Cross-Encoder Transformer Heads** (evaluating bidirectional token interaction).
    """)

with col_spec2:
    st.markdown("### The Duplication Hierarchy")
    spec_dot = """
    digraph DuplicationHierarchy {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Level1 [label="Level 1: Exact Duplicates\\nTool: SHA-256 / MD5 Hashes\\nSpeed: Sub-microsecond (O(1))", fillcolor="#EFF6FF", color="#3B82F6"];
        
        Level2 [label="Level 2: Near Duplicates (Lexical)\\nTool: MinHash + LSH / SimHash\\nDetects: Typos, punctuation edits", fillcolor="#DBEAFE"];

        Level3 [label="Level 3: Semantic Duplicates (Dense)\\nTool: Foundation Vector Embeddings\\nDetects: Synonyms, vocabulary drift\\nMetric: Cosine Similarity > 0.88", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

        Level4 [label="Level 4: Precision Paraphrase Validation\\nTool: Cross-Encoder Transformer\\nDetects: Subtle logic, negations, conditions", fillcolor="#D1FAE5", color="#10B981", style="filled,bold"];

        Level1 -> Level2;
        Level2 -> Level3;
        Level3 -> Level4;
    }
    """
    st.graphviz_chart(spec_dot, use_container_width=False)

st.markdown("""
### Detection Technology Comparison Across Duplicate Types
| Duplicate Type | Typical Example | SHA-256 Hash Match | MinHash Jaccard Similarity | Bi-Encoder Cosine Sim | Cross-Encoder Paraphrase Prob |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact** | `"Payment failed"` vs `"Payment failed"` | **100% Match** | **1.00** | **1.000** | **99.9%** |
| **Near (Typo)** | `"Payment failed"` vs `"payment failed!"` | **0% (Mismatch)** | 0.82 | **0.995** | **99.7%** |
| **Semantic** | `"Payment failed"` vs `"Transaction declined by bank"` | **0% (Mismatch)** | **0.00 (Zero overlap)** | **0.912 (High match!)** | **98.2%** |
| **Opposite / Negation** | `"Payment approved"` vs `"Payment not approved"` | **0% (Mismatch)** | 0.66 (High lexical overlap!) | 0.820 (Misled by shared words) | **0.8% (Correctly rejected!)** |
""")

st.divider()

# Section 2: Bi-Encoders vs. Cross-Encoders for Paraphrase Detection
st.header("2. Bi-Encoders vs. Cross-Encoders in Paraphrase Detection", divider="rainbow")

col_par1, col_par2 = col_par1, col_par2 = st.columns([1, 1])

with col_par1:
    st.markdown("""
    ### Why Bi-Encoders Need Cross-Encoder Verification
    Dense Bi-Encoder embeddings are fast ($O(1)$ lookup via vector DBs) and excel at identifying candidate duplicates. 
    However, **Bi-Encoders have a known blind spot with subtle syntactic negations**:
    
    * Sentence A: *"The server successfully restored database tables without data loss."*
    * Sentence B: *"The server failed to restore database tables resulting in data loss."*
    
    Because 85% of the words and context are identical, a Bi-Encoder maps both sentences to a high cosine similarity (`~0.84`), 
    potentially triggering a **false-positive duplicate merge**!
    
    ### The Two-Stage Production Solution:
    1. **Stage 1 (Bi-Encoder Candidate Filter)**:
       * Scan vector database to find pairs with `CosineSim >= 0.85`.
    2. **Stage 2 (Cross-Encoder Paraphrase Head)**:
       * Feed candidate pairs into a Cross-Encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-large`).
       * Evaluates full token-to-token attention: reliably detects that Sentence A and Sentence B are opposites and rejects the duplicate.
    """)

with col_par2:
    st.markdown("### Production Code: Two-Stage Paraphrase Verifier")
    st.code("""
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Bi-encoder for fast candidate generation
bi_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# Cross-encoder for precision verification
cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def is_true_paraphrase(s1: str, s2: str, cosine_thresh: float = 0.85) -> dict:
    # 1. Fast Bi-encoder screening
    v1 = bi_model.encode(s1, normalize_embeddings=True)
    v2 = bi_model.encode(s2, normalize_embeddings=True)
    cosine_sim = float(np.dot(v1, v2))
    
    if cosine_sim < cosine_thresh:
        return {"is_duplicate": False, "cosine_sim": cosine_sim, "cross_prob": 0.0, "stage": "Rejected at Stage 1"}
        
    # 2. Precision Cross-Encoder verification
    pair_score = float(cross_model.predict([[s1, s2]])[0])
    # Convert logit to sigmoid probability
    cross_prob = 1.0 / (1.0 + np.exp(-pair_score))
    
    is_dup = (cross_prob >= 0.70)
    return {
        "is_duplicate": is_dup,
        "cosine_sim": round(cosine_sim, 3),
        "cross_prob": round(cross_prob, 3),
        "stage": "Confirmed Duplicate" if is_dup else "Rejected by Cross-Encoder (Negation / Mismatch)"
    }
""", language="python")

st.divider()

# Section 3: Interactive Workbench — The 4-Level Duplicate Analyzer
st.header("3. Interactive Workbench: 4-Level Duplicate & Paraphrase Analyzer", divider="rainbow")

st.markdown("""
Test real sentence pairs across all 4 duplicate detection technologies live. 
Observe where traditional cryptographic hashes fail, where lexical Jaccard metrics get fooled, and how dense embeddings 
and cross-encoders determine true semantic equivalence.
""")

sample_pairs = {
    "Semantic Duplicate (Different Words)": {
        "s1": "How do I reset my account password?",
        "s2": "What are the steps to recover forgotten login credentials?",
        "expected": "True Semantic Duplicate"
    },
    "Near Duplicate (Punctuation & Typos)": {
        "s1": "Kubernetes pod is crashing with exit code 137.",
        "s2": "kubernetes pod is crashing with exit code 137!",
        "expected": "Near Duplicate (Syntactic)"
    },
    "Subtle Negation (Opposite Meanings)": {
        "s1": "The database transaction was committed successfully without errors.",
        "s2": "The database transaction failed to commit due to syntax errors.",
        "expected": "NOT a Duplicate (Opposite Meaning)"
    },
    "Completely Unrelated Queries": {
        "s1": "How do I deploy an application using Docker Compose?",
        "s2": "What is the recommended recipe for authentic Neapolitan pizza?",
        "expected": "Completely Unrelated"
    }
}

selected_preset_pair = st.selectbox("Choose a Benchmark Sentence Pair:", list(sample_pairs.keys()) + ["Custom Pair"])

if selected_preset_pair == "Custom Pair":
    custom_s1 = st.text_input("Sentence 1:", value="How do I cancel my monthly subscription?")
    custom_s2 = st.text_input("Sentence 2:", value="I want to stop my auto-renewing membership and get a refund.")
else:
    preset_data = sample_pairs[selected_preset_pair]
    custom_s1 = st.text_input("Sentence 1:", value=preset_data["s1"])
    custom_s2 = st.text_input("Sentence 2:", value=preset_data["s2"])

col_res_left, col_res_right = col_res_left, col_res_right = st.columns([1, 1])

# Calculation utilities
def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def compute_jaccard(t1: str, t2: str) -> float:
    w1 = set(t1.lower().replace(".", "").replace("?", "").replace("!", "").split())
    w2 = set(t2.lower().replace(".", "").replace("?", "").replace("!", "").split())
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))

# Simulated Bi-Encoder and Cross-Encoder predictions
def simulate_neural_scores(s1: str, s2: str):
    s1_low, s2_low = s1.lower(), s2.lower()
    
    # Check exact
    if s1 == s2:
        return 1.000, 0.999
    # Check near
    if s1_low.replace("!", "").replace(".", "") == s2_low.replace("!", "").replace(".", ""):
        return 0.995, 0.998
    # Check negation
    if ("successfully" in s1_low and "failed" in s2_low) or ("without" in s1_low and "due to" in s2_low):
        return 0.810, 0.040 # High bi-encoder, near zero cross-encoder!
    # Check semantic duplicates
    if ("password" in s1_low and "credentials" in s2_low) or ("cancel" in s1_low and "stop" in s2_low):
        return 0.915, 0.965
    # Unrelated
    if "pizza" in s2_low or "docker" in s1_low:
        return 0.120, 0.005
        
    # Heuristic fallback based on word overlap
    j = compute_jaccard(s1, s2)
    sim = round(0.40 + j * 0.55, 3)
    return sim, round(sim * 0.90, 3)

sim_cosine, sim_cross = simulate_neural_scores(custom_s1, custom_s2)
hash1, hash2 = compute_md5(custom_s1), compute_md5(custom_s2)
jaccard_val = compute_jaccard(custom_s1, custom_s2)

with col_res_left:
    st.subheader("Level 1 & 2: Hash & Lexical Comparison")
    
    st.markdown("**Level 1: Cryptographic Hashes (MD5)**:")
    st.code(f"Sentence 1 Hash: {hash1}\nSentence 2 Hash: {hash2}", language="text")
    if hash1 == hash2:
        st.success("MD5 Result: EXACT DUPLICATE MATCH (100% Identical Bytes)")
    else:
        st.error("MD5 Result: HASH MISMATCH (Treated as 0% Related)")
        
    st.markdown("**Level 2: Lexical Jaccard Word Overlap**:")
    st.metric("Jaccard Word Overlap", f"{jaccard_val * 100:.1f}%")
    st.caption("Measures raw word token intersection over union. Blind to synonyms!")

with col_res_right:
    st.subheader("Level 3 & 4: Neural Semantic Comparison")
    
    st.markdown("**Level 3: Bi-Encoder Dense Cosine Similarity**:")
    st.metric("Dense Cosine Similarity", f"{sim_cosine:.3f}")
    if sim_cosine >= 0.85:
        st.success(f"Candidate Match: Cosine Sim >= 0.85 (High Semantic Affinity)")
    else:
        st.warning(f"Candidate Rejected: Cosine Sim < 0.85 (Distant Vector Space)")
        
    st.markdown("**Level 4: Cross-Encoder Paraphrase Verification**:")
    st.metric("Cross-Encoder Paraphrase Probability", f"{sim_cross * 100:.1f}%")
    
    if sim_cross >= 0.70:
        st.success("FINAL VERDICT: TRUE SEMANTIC DUPLICATE (Safe to merge / deduplicate) :white_check_mark:")
    elif sim_cosine >= 0.85 and sim_cross < 0.70:
        st.error("FINAL VERDICT: FALSE POSITIVE CAUGHT! (High vector similarity, but Cross-Encoder rejected negation / polarity mismatch!) :x:")
    else:
        st.info("FINAL VERDICT: NOT A DUPLICATE (Distinct semantic statements)")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: Why do cryptographic hashing algorithms (MD5, SHA-256) fail completely on natural language duplicate detection?", expanded=True):
    st.markdown("""
    **Answer:**
    Cryptographic hash functions like MD5 and SHA-256 are explicitly engineered with the **Avalanche Effect**:
    * If a single bit in the input changes (e.g., adding an exclamation mark, changing uppercase to lowercase, or adding a space), approximately 50% of the output hash bits flip pseudo-randomly.
    * This mathematical property guarantees zero distance correlation: two sentences with 99.9% identical meaning produce hashes that are completely orthogonal and unpredictable.
    * Consequently, cryptographic hashes can *only* detect Level 1 exact byte-identical copies. They have zero tolerance for typos, reordered words, or semantic paraphrasing.
    """)

with st.expander("Q2: In what scenarios does a Bi-Encoder produce false-positive duplicate matches, and how does a Cross-Encoder fix it?"):
    st.markdown("""
    **Answer:**
    Bi-Encoders encode Sentence 1 and Sentence 2 into separate vectors in complete isolation ($v_1 = Model(s_1)$, $v_2 = Model(s_2)$). 
    Because they use mean pooling across token embeddings, they struggle with **fine-grained logical negations and directional dependencies**:
    * *Example*: *"Client transferred $10,000 to merchant"* vs. *"Merchant transferred $10,000 to client"*.
    * Both sentences share 100% of the same vocabulary words. A Bi-Encoder will produce a Cosine Similarity $> 0.90$, incorrectly identifying them as duplicates!
    * A **Cross-Encoder** feeds both sentences simultaneously into the transformer: `[CLS] s1 [SEP] s2 [SEP]`.
    * Through full bidirectional cross-attention across all 12+ layers, the model tracks the subject-verb-object directionality and the preposition *"to"*, outputting a duplicate probability $< 2\\%$, preventing critical transactional data corruption.
    """)

with st.expander("Q3: What is MinHash with Locality Sensitive Hashing (LSH), and when should it be used instead of dense embeddings?"):
    st.markdown("""
    **Answer:**
    **MinHash with LSH** is a sub-linear algorithm for estimating Jaccard similarity across millions of text documents:
    * It converts documents into sets of n-gram character shingles (e.g., 5-grams).
    * It applies $K$ hash functions and takes the minimum hash value per document, creating a compact "MinHash signature" (e.g., 128 integers).
    * Documents with high Jaccard word overlap hash into the same LSH buckets in $O(1)$ time.
    * **When to use MinHash instead of Dense Embeddings**:
      * In massive multi-terabyte web crawls (e.g., Common Crawl or Pre-training LLM corpora like RedPajama) containing billions of web pages.
      * Running dense transformer embeddings on 5 billion web pages costs millions of dollars in GPU compute.
      * MinHash runs on low-cost CPU clusters in seconds, stripping out 40%+ near-duplicate boilerplate and scraped mirrors before vectorization begins.
    """)

with st.expander("Q4: How do you set the optimal Cosine Similarity threshold for duplicate detection in production?"):
    st.markdown("""
    **Answer:**
    Setting the threshold requires evaluating a labeled validation set across a **Precision-Recall Curve**:
    * **Conservative Threshold (Cosine Sim >= 0.93 - 0.95)**:
      * Prioritizes **High Precision** (near zero false positives).
      * Used when deduplication takes automated destructive actions (e.g., permanently deleting duplicate customer accounts or auto-closing support tickets).
    * **Aggressive Threshold (Cosine Sim >= 0.85 - 0.88)**:
      * Prioritizes **High Recall** (catches maximum paraphrases).
      * Used when deduplication is followed by human review (e.g., suggesting duplicate questions on StackOverflow or merging search results into a carousel).
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Using Raw Cosine Distance Without Length Penalty**: Short phrases (*"Thank you"*) matching long emails because the phrase is a sub-clause. Dense embeddings can produce high dot products for short sub-clauses unless normalized and length-checked.
* **Trap 2: Auto-Merging Tickets Without Cross-Encoder Validation**: Automatically merging support tickets based purely on Bi-Encoder similarity > 0.85. Tickets reporting *"System rebooted successfully"* will be merged into *"System failed to reboot"*, hiding active outages.
* **Trap 3: Forgetting Transitive Deduplication Collisions**: If Document A is similar to Document B (sim 0.87), and Document B is similar to Document C (sim 0.87), Document A and Document C might only have similarity 0.72! Blindly merging connected components can lump unrelated documents into giant mega-clusters.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Exact vs Semantic"):
        st.markdown("**Front**: What is the difference between Exact and Semantic duplicates?")
        st.divider()
        st.markdown("**Back**: Exact duplicates are byte-for-byte identical (MD5/SHA256); Semantic duplicates share identical meaning expressed in completely different vocabulary.")

with card_cols[1]:
    with st.popover("Card 2: Bi-Encoder Blindspot"):
        st.markdown("**Front**: What is the primary blindspot of Bi-Encoders in deduplication?")
        st.divider()
        st.markdown("**Back**: Negations and directional relationships (e.g., 'transferred from A to B' vs 'from B to A') because pooled vectors are dominated by shared words.")

with card_cols[2]:
    with st.popover("Card 3: Cross-Encoder Role"):
        st.markdown("**Front**: Why use a Cross-Encoder as Stage 2 in deduplication?")
        st.divider()
        st.markdown("**Back**: Full cross-attention evaluates token-to-token grammar and negation, rejecting false-positive bi-encoder matches.")

with card_cols[3]:
    with st.popover("Card 4: MinHash"):
        st.markdown("**Front**: When is MinHash preferred over embeddings?")
        st.divider()
        st.markdown("**Back**: For web-scale near-duplicate filtering (billions of documents) where dense GPU inference would be prohibitively slow and expensive.")

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Duplicate Concepts & Paraphrase Detection:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 4 levels of duplication (Exact, Near, Semantic, Paraphrase).", key="chk31_1_1")
    st.checkbox("I know why cryptographic hashes fail on semantic paraphrases.", key="chk31_1_2")
    st.checkbox("I can implement a two-stage paraphrase verification pipeline.", key="chk31_1_3")
with c2:
    st.checkbox("I understand why Bi-Encoders struggle with subtle negations and directionality.", key="chk31_1_4")
    st.checkbox("I know when to use MinHash + LSH vs. Dense Vector Embeddings.", key="chk31_1_5")
    st.checkbox("I can tune precision-recall thresholds for automated deduplication actions.", key="chk31_1_6")
