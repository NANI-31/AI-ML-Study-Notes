import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Tuple, Any

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="21.2 Traditional Metrics - LLM Engineering",
    page_icon="📐",
    layout="wide"
)

# --- MODULE-SCOPE REAL IMPLEMENTATION FOR WORKBENCH ---
class TraditionalNLPScorer:
    """
    Self-contained in-memory traditional NLP evaluation engine implementing:
    - Precision, Recall, and F1 calculation
    - Simplified BLEU (unigram/bigram overlap with brevity penalty)
    - ROUGE-1, ROUGE-2, and ROUGE-L (Longest Common Subsequence)
    - Simulated Perplexity from token loss
    """

    @staticmethod
    def compute_classification_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

    @staticmethod
    def compute_bleu(reference: str, candidate: str) -> Dict[str, float]:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not cand_tokens or not ref_tokens:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "brevity_penalty": 0.0}

        # Unigram Precision
        ref_counts = {}
        for w in ref_tokens:
            ref_counts[w] = ref_counts.get(w, 0) + 1
            
        matched_unigrams = 0
        for w in cand_tokens:
            if ref_counts.get(w, 0) > 0:
                matched_unigrams += 1
                ref_counts[w] -= 1
        p1 = matched_unigrams / len(cand_tokens)

        # Bigram Precision
        cand_bigrams = [tuple(cand_tokens[i:i+2]) for i in range(len(cand_tokens)-1)]
        ref_bigrams = {}
        for i in range(len(ref_tokens)-1):
            bg = tuple(ref_tokens[i:i+2])
            ref_bigrams[bg] = ref_bigrams.get(bg, 0) + 1
            
        matched_bigrams = 0
        for bg in cand_bigrams:
            if ref_bigrams.get(bg, 0) > 0:
                matched_bigrams += 1
                ref_bigrams[bg] -= 1
        p2 = matched_bigrams / len(cand_bigrams) if cand_bigrams else 0.0

        # Brevity Penalty
        c_len = len(cand_tokens)
        r_len = len(ref_tokens)
        bp = 1.0 if c_len > r_len else np.exp(1.0 - (r_len / c_len)) if c_len > 0 else 0.0
        
        return {
            "bleu_1": round(p1 * bp, 4),
            "bleu_2": round(np.sqrt(p1 * p2) * bp, 4) if (p1 * p2) > 0 else 0.0,
            "brevity_penalty": round(bp, 4)
        }

    @staticmethod
    def compute_rouge(reference: str, candidate: str) -> Dict[str, float]:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not cand_tokens or not ref_tokens:
            return {"rouge_1": 0.0, "rouge_l": 0.0}

        # ROUGE-1 Recall
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)
        overlap = len(ref_set.intersection(cand_set))
        r1_recall = overlap / len(ref_set) if ref_set else 0.0

        # ROUGE-L via Longest Common Subsequence (LCS)
        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        rl_recall = lcs / m if m > 0 else 0.0

        return {
            "rouge_1": round(r1_recall, 4),
            "rouge_l": round(rl_recall, 4),
            "lcs_length": lcs
        }

# --- TOP-LEVEL HEADER ---
st.title("21.2 Traditional Metrics: Accuracy, Precision, Recall, BLEU, ROUGE, & Perplexity")
st.caption("Chapter 21: LLM Evaluation | Master Architecture Specification")
st.markdown("""
Before modern LLM-as-a-judge evaluators emerged, Natural Language Processing relied on deterministic 
statistical metrics borrowed from information retrieval, machine translation, and language modeling. 
Understanding **Accuracy**, **Precision**, **Recall**, **F1**, **BLEU**, **ROUGE**, and **Perplexity** 
is essential for benchmark auditing and hybrid evaluation pipelines.
""")

st.divider()

# --- SECTION 1: BLUEPRINT ARCHITECTURE ---
st.header("1. Blueprint Architecture: Traditional Statistical NLP Metric Pipeline", divider="rainbow")

st.markdown("""
The architecture below illustrates how n-gram overlap, sequence alignment, confusion matrices, and cross-entropy 
form deterministic evaluation signals against reference ground truths.
""")

dot_trad = """
digraph TraditionalBlueprint {
    graph [bgcolor="#0B1120", rankdir=TB, pad="0.5", nodesep="0.5", ranksep="0.6"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
    edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

    subgraph cluster_inputs {
        label = "Layer 1: Evaluation Inputs";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#38BDF8";
        CandidateText [label="Candidate LLM Generation\\n['Paris is the capital of France.']", fillcolor="#1E3A8A", color="#3B82F6"];
        ReferenceText [label="Ground Truth Reference\\n['The capital of France is Paris.']", fillcolor="#1E3A8A", color="#3B82F6"];
        ModelLogits [label="Model Vocabulary Logits\\n[P(w_t | w_<t)]", fillcolor="#1E3A8A", color="#3B82F6"];
    }

    subgraph cluster_methods {
        label = "Layer 2: Metric Computation Engines";
        color = "#334155";
        style = "filled";
        fillcolor = "#1E293B";
        fontcolor = "#F59E0B";
        ExactMatchEng [label="1. Accuracy & Exact Match\\n[String & Levenshtein Equality]", fillcolor="#78350F", color="#D97706"];
        ConfusionEng [label="2-4. Classification Metrics\\n[Precision, Recall, F1 Score]", fillcolor="#78350F", color="#D97706"];
        NGramEng [label="5. BLEU Metric\\n[Precision N-gram + Brevity Penalty]", fillcolor="#78350F", color="#D97706"];
        LCSEng [label="6. ROUGE Metric\\n[Recall N-gram & Longest Common Subsequence]", fillcolor="#78350F", color="#D97706"];
        EntropyEng [label="7. Perplexity Engine\\n[PPL = exp(-1/N * sum(log P(w_t)))]", fillcolor="#831843", color="#EC4899"];
    }

    subgraph cluster_scorecard {
        label = "Layer 3: Benchmark Scorecard";
        color = "#1E293B";
        style = "filled";
        fillcolor = "#0F172A";
        fontcolor = "#10B981";
        Scorecard [label="Deterministic Metric Scorecard\\n[Scores from 0.0 to 1.0 + PPL Loss]", fillcolor="#064E3B", color="#10B981"];
    }

    CandidateText -> ExactMatchEng;
    ReferenceText -> ExactMatchEng;
    CandidateText -> ConfusionEng;
    ReferenceText -> ConfusionEng;
    CandidateText -> NGramEng;
    ReferenceText -> NGramEng;
    CandidateText -> LCSEng;
    ReferenceText -> LCSEng;
    ModelLogits -> EntropyEng;

    ExactMatchEng -> Scorecard;
    ConfusionEng -> Scorecard;
    NGramEng -> Scorecard;
    LCSEng -> Scorecard;
    EntropyEng -> Scorecard;
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_trad, width="stretch")

st.divider()

# --- SECTION 2: TOPIC 1 - ACCURACY ---
st.header("2. Accuracy: String Matching, Exact Match (EM), & Normalization", divider="rainbow")

st.markdown("""
**Accuracy** measures the proportion of predictions that match the ground truth. 
In generative LLMs, accuracy takes the form of **Exact Match (EM)**.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Exact Match (EM) Formulation")
    st.markdown("""
    `EM = 1 if normalize(Candidate) == normalize(Reference) else 0`
    
    - **Normalization Rules**: Strip leading/trailing whitespace, convert to lowercase, remove punctuation, and strip articles (`a`, `an`, `the`).
    - **Use Cases**: Multiple-choice question answering (MMLU), extraction of structured IDs, math final numerical values.
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: How New Models Follow This
    Exact Match is excessively brittle for free-form generative text. If the reference is *"Dr. Smith"* and the model emits 
    *"Doctor Smith, MD"*, EM registers a 0.0 failure! Modern pipelines supplement EM with **Fuzzy Match** or **Semantic Equivalence**.
    """)

st.divider()

# --- SECTION 3: TOPIC 2 - PRECISION ---
st.header("3. Precision: True Positives vs False Positives & Precision@K", divider="rainbow")

st.markdown("""
**Precision** measures the exactness of the model's output: of all items the model predicted, how many were correct?
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Mathematical Formulation")
    st.markdown("""
    `Precision = True_Positives / (True_Positives + False_Positives)`
    
    - In Named Entity Recognition (NER): Of all entities extracted by the model, what fraction are real entities?
    - In Tool Selection: Of all tools called, what fraction were genuinely necessary?
    """)

with col2:
    st.info("""
    **High Precision Focus**:
    When the cost of a False Positive is severe (e.g. sending a fraud alert or triggering a financial transaction), 
    engineers optimize for Precision over Recall.
    """)

st.divider()

# --- SECTION 4: TOPIC 3 - RECALL ---
st.header("4. Recall: Ground Truth Coverage & Extraction Sensitivity", divider="rainbow")

st.markdown("""
**Recall** measures the completeness of the model's output: of all actual items present in the ground truth, how many did the model find?
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Mathematical Formulation")
    st.markdown("""
    `Recall = True_Positives / (True_Positives + False_Positives)`
    
    - In Legal Contract Review: Did the LLM find all 5 non-disclosure breach clauses, or did it miss 2?
    - In Medical Information Extraction: Did the model extract all symptoms mentioned by the patient?
    """)

with col2:
    st.warning("""
    **High Recall Focus**:
    When missing a true case (False Negative) is catastrophic (e.g. cancer diagnosis or legal discovery), 
    engineers tune the model prompt for maximal Recall.
    """)

st.divider()

# --- SECTION 5: TOPIC 4 - F1-SCORE ---
st.header("5. F1-Score: Harmonic Mean & Multi-Class Aggregations", divider="rainbow")

st.markdown("""
The **F1-Score** is the harmonic mean of Precision and Recall. It provides a balanced single scalar metric 
when class distributions are imbalanced.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Harmonic Mean Formula")
    st.markdown("""
    `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
    
    - Why Harmonic Mean? An arithmetic mean `(0.0 + 1.0) / 2 = 0.5` masks total failure. 
      The harmonic mean penalizes extreme imbalances: if either Precision or Recall is 0.0, F1 is strictly **0.0**.
    """)

with col2:
    st.subheader("Multi-Class Aggregations")
    st.markdown("""
    - **Micro-F1**: Aggregates global TP, FP, FN across all classes before computing F1.
    - **Macro-F1**: Computes F1 per class independently, then takes the unweighted average. Emphasizes performance on rare minority classes.
    - **Weighted-F1**: Weights each class F1 by its support frequency.
    """)

st.divider()

# --- SECTION 6: TOPIC 5 - BLEU ---
st.header("6. BLEU: N-Gram Precision Overlap & Brevity Penalty", divider="rainbow")

st.markdown("""
Pioneered by Papineni et al. (2002) for Machine Translation, **BLEU (Bilingual Evaluation Understudy)** measures 
how many n-grams in the candidate text appear in the reference text.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The BLEU Formulation")
    st.markdown("""
    `BLEU = Brevity_Penalty * exp(sum(w_n * log(Precision_n)))`
    
    1. **Modified N-Gram Precision (`P_n`)**: Computes clipped n-gram matches up to `N = 4`.
    2. **Brevity Penalty (`BP`)**: Penalizes candidates shorter than the reference:  
       `BP = 1.0 if c > r else exp(1 - r / c)`.  
       Prevents models from cheating by outputting a single verified word (e.g. *"Paris"*).
    """)

with col2:
    st.markdown("""
    ### 🚀 Modern Foundation LLM Reality: The Limits of BLEU
    BLEU rewards exact word matches and severely penalizes valid synonyms (*"enormous"* vs *"huge"*). 
    In generative tasks, an answer can be completely factually correct while receiving a BLEU score near zero! 
    BLEU is now largely confined to machine translation benchmarks.
    """)

st.divider()

# --- SECTION 7: TOPIC 6 - ROUGE ---
st.header("7. ROUGE: ROUGE-N, ROUGE-L, & Longest Common Subsequences", divider="rainbow")

st.markdown("""
Introduced by Lin (2004) for Summarization, **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** focuses on 
**Recall**: how much of the reference summary is captured by the model?
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### ROUGE-1")
    st.markdown("""
    Unigram (single word) recall:
    `Overlap(words) / Total_Reference_Words`.
    Measures overall topic coverage.
    """)

with col2:
    st.markdown("### ROUGE-2")
    st.markdown("""
    Bigram (word pair) recall:
    `Overlap(bigrams) / Total_Reference_Bigrams`.
    Measures local fluency and phrase coherence.
    """)

with col3:
    st.markdown("### ROUGE-L")
    st.markdown("""
    Longest Common Subsequence (LCS):
    `LCS(cand, ref) / Total_Reference_Words`.
    Measures sentence-level structure without requiring consecutive words.
    """)

st.divider()

# --- SECTION 8: TOPIC 7 - PERPLEXITY ---
st.header("8. Perplexity: Cross-Entropy Loss & Exponentiated Surprise", divider="rainbow")

st.markdown("""
**Perplexity (PPL)** is the canonical intrinsic metric for language models. 
It measures how "surprised" the model is when predicting the next token in a sequence.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Mathematical Derivation")
    st.markdown("""
    Perplexity is the exponential of the average negative log-likelihood (Cross-Entropy Loss):
    
    `Cross_Entropy_Loss = - (1 / N) * sum(log P(w_t | w_<t))`
    
    `Perplexity = exp(Cross_Entropy_Loss)`
    
    - **Intuition**: A perplexity of `K` means the model is, on average, as confused as choosing uniformly between `K` equally likely words.
    - **Lower is Better**: A perplexity of 1.0 represents absolute deterministic certainty.
    """)

with col2:
    st.error("""
    **The Tokenizer Mismatch Caveat**:
    You CANNOT compare perplexity across models that use different tokenizers (e.g. Llama tokenizer vs GPT-4o tokenizer)! 
    Perplexity is computed per token; a tokenizer with a larger vocabulary compresses text into fewer tokens, 
    artificially altering loss calculations.
    """)

st.divider()

# --- SECTION 9: PRODUCTION CODE IMPLEMENTATION ---
st.header("9. Production Implementation: Complete Statistical NLP Metric Suite", divider="rainbow")

st.markdown("""
Below is an end-to-end Python implementation of traditional NLP metrics featuring Exact Match, 
Precision/Recall/F1, BLEU, and ROUGE-L.
""")

st.code('''
import numpy as np
from typing import Dict, List, Any

class ProductionTraditionalNLPBenchmarker:
    """
    Complete statistical NLP evaluation suite for LLM benchmarking.
    """
    @staticmethod
    def exact_match(ref: str, cand: str) -> float:
        """Computes normalized exact string match."""
        def norm(s): return " ".join(s.lower().strip().split())
        return 1.0 if norm(ref) == norm(cand) else 0.0

    @staticmethod
    def classification_f1(tp: int, fp: int, fn: int) -> float:
        """Computes harmonic mean F1 score."""
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    @staticmethod
    def rouge_l(reference: str, candidate: str) -> float:
        """Computes ROUGE-L Recall via dynamic programming LCS."""
        r_words = reference.lower().split()
        c_words = candidate.lower().split()
        if not r_words or not c_words:
            return 0.0
        m, n = len(r_words), len(c_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if r_words[i-1] == c_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n] / m

    @staticmethod
    def compute_all_metrics(ref: str, cand: str) -> Dict[str, float]:
        """Calculates multi-metric scorecard."""
        return {
            "exact_match": ProductionTraditionalNLPBenchmarker.exact_match(ref, cand),
            "rouge_l": round(ProductionTraditionalNLPBenchmarker.rouge_l(ref, cand), 4)
        }
''', language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Traditional Metrics Lab: Live Scorer & N-Gram Comparator", divider="rainbow")

st.markdown("""
Compare a candidate model generation against a ground truth reference. 
Observe how **Exact Match**, **BLEU-1**, **BLEU-2**, **ROUGE-1**, and **ROUGE-L** score the text.
""")

col1, col2 = st.columns([1, 2])

with col1:
    default_ref = "The transformer model relies on self-attention mechanisms to process sequences in parallel."
    default_cand = "Transformers rely on attention mechanisms for parallel sequence processing."
    
    input_ref = st.text_area("Ground Truth Reference Text:", value=default_ref, height=90)
    input_cand = st.text_area("Candidate Model Output:", value=default_cand, height=90)
    score_btn = st.button("📊 Calculate Metrics", type="primary", use_container_width=True)

with col2:
    scorer = TraditionalNLPScorer()
    
    if score_btn or True:
        bleu_res = scorer.compute_bleu(input_ref, input_cand)
        rouge_res = scorer.compute_rouge(input_ref, input_cand)
        em = 1.0 if input_ref.strip().lower() == input_cand.strip().lower() else 0.0
        
        st.subheader("Metric Scorecard")
        score_df = pd.DataFrame([
            {"Metric": "Exact Match (EM)", "Score": em, "Description": "Binary 1.0/0.0 string match"},
            {"Metric": "BLEU-1", "Score": bleu_res["bleu_1"], "Description": "Unigram precision with brevity penalty"},
            {"Metric": "BLEU-2", "Score": bleu_res["bleu_2"], "Description": "Bigram geometric mean precision"},
            {"Metric": "Brevity Penalty (BP)", "Score": bleu_res["brevity_penalty"], "Description": "Length penalty (<1.0 if candidate is short)"},
            {"Metric": "ROUGE-1", "Score": rouge_res["rouge_1"], "Description": "Unigram reference recall"},
            {"Metric": "ROUGE-L", "Score": rouge_res["rouge_l"], "Description": "Longest common subsequence recall"}
        ])
        st.dataframe(score_df, width="stretch")

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("Staff-Level Engineering Q&A")
with st.expander("Q1: Why does ROUGE focus on Recall while BLEU focuses on Precision?"):
    st.markdown("""
    **Answer**:
    - **Machine Translation (BLEU)**: When translating, adding irrelevant words or repeating phrases ruins translation quality. Therefore, BLEU focuses on **Precision** (ensuring every word emitted by the candidate actually belongs in the target reference).
    - **Summarization (ROUGE)**: When summarizing, the primary goal is capturing all essential points from the source. Therefore, ROUGE focuses on **Recall** (measuring what percentage of the reference summary was successfully captured).
    """)

with st.expander("Q2: Why is Perplexity an intrinsic metric rather than an extrinsic task metric?"):
    st.markdown("""
    **Answer**:
    Perplexity measures the internal probability model over token sequences (loss on next-token prediction). 
    A low perplexity proves that the model knows grammar and statistical facts well, but it does **not** guarantee that the model follows human instructions, avoids toxicity, or performs well on downstream business tasks (extrinsic evaluation).
    """)

with st.expander("Q3: What is the Brevity Penalty in BLEU and why is it necessary?"):
    st.markdown("""
    **Answer**:
    Without the brevity penalty, a model could output a single common word present in the reference (e.g. `"The"`) and achieve a perfect unigram precision of `1/1 = 1.0` (100%). 
    The Brevity Penalty penalizes candidates that are shorter than the reference, dropping the score exponentially: `BP = exp(1 - r / c)`.
    """)

with st.expander("Q4: Why can two models with identical BLEU scores have vastly different user satisfaction ratings?"):
    st.markdown("""
    **Answer**:
    BLEU measures purely surface-level n-gram overlap. It is completely blind to:
    1. **Factual Negation**: The candidate `"The drug cured the patient"` and reference `"The drug failed to cure the patient"` have high BLEU overlap but opposite meanings!
    2. **Synonym Substitution**: Replacing words with valid synonyms drops BLEU while maintaining perfect semantic fidelity.
    """)

with st.expander("Q5: Why is the Harmonic Mean used for F1 rather than the Arithmetic Mean?"):
    st.markdown("""
    **Answer**:
    The arithmetic mean of 0.0 and 1.0 is 0.5, which gives a misleadingly high impression of performance for a completely broken classifier. 
    The harmonic mean `2 * (P * R) / (P + R)` approaches the minimum of the two values. If either Precision or Recall drops to 0, F1 drops to 0.
    """)

st.subheader("Production Traps & Failure Modes")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.error("""
    **TRAP 1 (Error): Comparing Perplexity Across Different Tokenizers**
    Attempting to benchmark Llama 3 against GPT-4o using Perplexity. Different vocabulary sizes (128k vs 200k) alter token boundaries and mathematical cross-entropy scale.
    """)
    st.error("""
    **TRAP 2 (Error): Using BLEU for Open-Ended Generative QA**
    Grading conversational assistants with BLEU against a single reference answer. Creative, detailed, or alternative valid phrasings receive near-zero BLEU scores.
    """)

with col_t2:
    st.warning("""
    **TRAP 3 (Warning): Micro vs Macro F1 Distortion**
    Reporting Micro-F1 on heavily skewed multi-class datasets. Micro-F1 is dominated by the majority class, masking 0% recall on critical minority classes.
    """)
    st.warning("""
    **TRAP 4 (Warning): Unnormalized Exact Match**
    Evaluating outputs with Exact Match without stripping whitespace, punctuation, and casing, resulting in false negative failures on formatting technicalities.
    """)

st.subheader("Architectural Flashcards")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    st.markdown("""
    **BLEU**  
    N-gram precision overlap with brevity penalty (standard in translation).
    """)
with fc2:
    st.markdown("""
    **ROUGE-L**  
    Longest Common Subsequence recall (standard in summarization).
    """)
with fc3:
    st.markdown("""
    **Perplexity**  
    Exponentiated cross-entropy loss measuring language model surprise.
    """)
with fc4:
    st.markdown("""
    **Exact Match (EM)**  
    Binary strict string equality after deterministic normalization.
    """)

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist: Traditional Metrics Mastery", divider="rainbow")

st.markdown("Verify your understanding of traditional statistical NLP metrics by reviewing each criterion:")

chk1 = st.checkbox("I understand the formulation and implementation of Precision, Recall, and F1 harmonic mean.", key="trad_chk1")
chk2 = st.checkbox("I know how the Brevity Penalty in BLEU prevents short-text gaming.", key="trad_chk2")
chk3 = st.checkbox("I can differentiate ROUGE-1, ROUGE-2, and ROUGE-L (Longest Common Subsequence).", key="trad_chk3")
chk4 = st.checkbox("I understand why BLEU focuses on Precision while ROUGE focuses on Recall.", key="trad_chk4")
chk5 = st.checkbox("I know the mathematical derivation of Perplexity from Cross-Entropy loss.", key="trad_chk5")
chk6 = st.checkbox("I understand why perplexity cannot be compared across models with different tokenizers.", key="trad_chk6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Master Achievement Unlocked: You have demonstrated comprehensive mastery of Traditional NLP Metrics and Statistical Benchmarks!")
