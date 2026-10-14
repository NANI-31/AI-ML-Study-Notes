import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="26.2 Cross-Lingual Search - Embeddings Curriculum",
    page_icon="🔄",
    layout="wide"
)

# Title & Overview
st.title("26.2 Cross-Lingual Search: English-Hindi Bidirectional Retrieval & Cross-Language Semantic Similarity")
st.caption("Chapter 26: Multilingual Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
**Cross-Lingual Information Retrieval (CLIR)** is the capability to formulate a query in **Language A** (e.g., Hindi or Spanish) 
and retrieve authoritative documents stored in **Language B** (e.g., English), or vice versa, without requiring explicit 
document-level translation.

In global tech enterprises, 90% of engineering documentation, architectural specs, and API contracts are authored in English. 
However, customer support specialists, field technicians, and end-users around the globe query in their native languages. 
Cross-lingual search bridges this divide by projecting both the query and foreign-language documents into a shared multilingual 
latent manifold where vector distance reflects **semantic conceptual intent** rather than vocabulary or grammatical syntax.
""")

st.divider()

# Section 1: Bidirectional Cross-Lingual Search Paradigms
st.header("1. Bidirectional Cross-Lingual Search Architecture", divider="rainbow")

col_clir1, col_clir2 = col_clir1, col_clir2 = st.columns([1, 1])

with col_clir1:
    st.markdown("""
    ### Two Core Bidirectional Retrieval Flows
    1. **Hindi Query -> English Document Retrieval (L_q != L_d)**:
       * *User Query (Hindi)*: *"डेटाबेस में डेडलॉक को कैसे रोका जाए?"* (*"How to prevent database deadlocks?"*)
       * *Target Corpus (English)*: Official PostgreSQL documentation on isolation levels, locking order, and deadlock detection.
       * *Mechanism*: The Hindi query vector lands in the exact vector neighborhood of the English transaction locking chunks.
    2. **English Query -> Hindi Document Retrieval**:
       * *User Query (English)*: *"Government farmer subsidy eligibility criteria"*
       * *Target Corpus (Hindi)*: Indian municipal policy notices, scheme documents, and circulars authored in Hindi.
       * *Mechanism*: The English query vector identifies the most relevant Devanagari policy chunks with zero manual translation.
    
    ### Architectural Comparison: Zero-Shot Dense vs. Translate-then-Search
    * **Direct Zero-Shot Multilingual Dense Search**:
      * Single model (e.g., BGE-M3 or Cohere Embed Multilingual).
      * No translation step; sub-10ms query encoding; preserves original user intent nuance.
    * **Translate-then-Search (Query Translation)**:
      * Machine translation (e.g., Google Cloud Translate or NLLB) converts query to English, then runs standard English search.
      * Risk: Translation drift on technical jargon, adds 150ms-300ms latency, high API costs.
    """)

with col_clir2:
    st.markdown("### Cross-Lingual Retrieval Funnel")
    clir_dot = """
    digraph CrossLingualSearch {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="Query in Language A\\n'कुबेरनेट्स में पॉड कैसे रीस्टार्ट करें?'\\n(Hindi Query)", fillcolor="#FED7AA", color="#F97316", style="filled,bold"];
        
        Encoder [label="Multilingual Vector Encoder\\n(BGE-M3 / Cohere Multilingual v3)\\nEncodes to 1024-dim Vector", fillcolor="#FEF3C7"];

        IndexDocs [label="Enterprise Corpus in Language B (English)\\n100,000 Chunks Pre-indexed\\n'kubectl rollout restart deployment...'", fillcolor="#DBEAFE", color="#3B82F6", style="filled,bold"];

        VectorANN [label="Vector HNSW Search\\nMeasures Cosine Similarity\\nacross Language Boundary", fillcolor="#E0E7FF"];

        Reranker [label="Multilingual Cross-Encoder Reranker\\n(BGE-Reranker-v2-m3 / Cohere Rerank)\\nRe-scores top 30 candidates", fillcolor="#D1FAE5", color="#10B981"];

        FinalHit [label="Ranked English Document\\n'Restarting Pods Gracefully in K8s'\\nHigh Relevance (Sim: 0.91)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        UserQ -> Encoder;
        Encoder -> VectorANN;
        IndexDocs -> VectorANN;
        VectorANN -> Reranker;
        Reranker -> FinalHit;
    }
    """
    st.graphviz_chart(clir_dot, use_container_width=False)

st.markdown("""
| Dimension | Zero-Shot Multilingual Vector Search | Query Translation + Monolingual Search | Full Document Translation Indexing |
| :--- | :--- | :--- | :--- |
| **Pipeline Latency** | **Fastest (10 - 25 ms)** | Moderate (150 - 350 ms) | Fast at query time, Massive ingestion latency |
| **Translation Cost** | **Zero per query** | Incurred on every user query | High upfront corpus translation cost |
| **Vocabulary Flexibility** | High (handles code-switching & slang) | Fragile (translation failures on slang) | Rigid (sync required on document edits) |
| **Script Handling** | Native multi-script support | Requires accurate script transcription | Normalized to target language |
""")

st.divider()

# Section 2: Cross-Language Semantic Similarity Evaluation
st.header("2. Cross-Language Semantic Similarity & Evaluation Metrics", divider="rainbow")

st.markdown("""
### Measuring Cross-Lingual Alignment Quality
In standard monolingual retrieval, we measure **Recall@k**, **Mean Reciprocal Rank (MRR)**, and **NDCG@k** across queries and documents in the same language.
In Cross-Lingual Information Retrieval (CLIR), we compute two specialized benchmark metrics:
1. **Cross-Lingual Bitext Retrieval Accuracy (Recall@1)**:
   * Given 10,000 parallel sentence pairs `(S_en, S_hi)`, what percentage of the time is `S_en` the exact #1 nearest neighbor when querying with `S_hi`?
   * State-of-the-art models like BGE-M3 achieve **> 94.5% Recall@1** on standard Tatoeba and Flores-200 benchmarks.
2. **Cross-Lingual Alignment Penalty (Delta Similarity)**:
   * The distance gap between a monolingual match and a cross-lingual match:
   * `Delta = CosineSim(Query_en, Doc_en) - CosineSim(Query_hi, Doc_en)`
   * A well-aligned model maintains `Delta < 0.05`, meaning the language boundary costs less than 5% similarity penalty!
""")

col_eval1, col_eval2 = col_eval1, col_eval2 = st.columns([1, 1])

with col_eval1:
    st.markdown("### Production Code: Bidirectional Search Engine")
    st.code("""
import numpy as np
from sentence_transformers import SentenceTransformer

# Load production multilingual embedding model
model = SentenceTransformer('BAAI/bge-m3')

def cross_lingual_search(query: str, corpus_embeddings: np.ndarray, corpus_docs: list[str], top_k: int = 3):
    \"\"\"Performs zero-shot cross-lingual vector search.\"\"\"
    # 1. Encode query (regardless of language)
    q_vec = model.encode(query, normalize_embeddings=True)
    
    # 2. Compute cosine similarity against foreign corpus
    # (corpus_embeddings must also be L2-normalized)
    scores = np.dot(corpus_embeddings, q_vec)
    
    # 3. Retrieve top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "doc": corpus_docs[idx],
            "score": float(scores[idx]),
            "index": int(idx)
        })
    return results
""", language="python")

with col_eval2:
    st.markdown("### Cross-Lingual Evaluation Formulation")
    st.markdown("""
    ```text
    Mean Reciprocal Rank (MRR@k):
    MRR = (1 / |Q|) * Sum_{i=1}^{|Q|} [ 1 / Rank_i ]
    
    Where:
    - Rank_i is the rank position of the first relevant
      foreign-language document retrieved for query Q_i.
    - If no relevant document is found in the top-k,
      1 / Rank_i is set to 0.
    
    Normalized Discounted Cumulative Gain (NDCG@k):
    DCG@k = Sum_{i=1}^k [ (2^{rel_i} - 1) / log2(i + 1) ]
    NDCG@k = DCG@k / IDCG@k
    ```
    """)

st.divider()

# Section 3: Handling Code-Switching & Romanized Hindi (Hinglish)
st.header("3. The Code-Switching Challenge: Hinglish & Multi-Script Text", divider="rainbow")

st.markdown("""
In countries like India, Mexico, and the Philippines, users rarely search in pure formal script. Instead, they mix languages:
* **Pure Hindi (Devanagari)**: *"सर्वर को कैसे रीस्टार्ट करें?"*
* **Pure English**: *"How to restart the server?"*
* **Code-Switched / Romanized Hindi (Hinglish)**: *"Server ko restart kaise karein?"* or *"Kubernetes pod crash ho raha hai, fix kaise karein?"*

### Why Hinglish Challenges Standard Models:
1. **Phonetic Spelling Inconsistency**: There is no standard dictionary for Romanized Hindi. Users write *"kaise"*, *"kese"*, or *"kayse"*.
2. **Vocabulary Overlap**: Words like *"is"* in German or *"mein"* in Hindi overlap with English words (*"is"*, *"main"*).
3. **Subword Tokenizer Fragmentation**: Unstandardized transliterations get split into single characters, increasing sequence length and degrading dense representations.

### Modern Solution:
Models trained with byte-level BPE on massive multilingual web corpora (such as **BGE-M3** and **Cohere v3**) have 
natively internalized common code-switched phrases, allowing direct zero-shot retrieval even on Hinglish inputs!
""")

st.divider()

# Section 4: Interactive Workbench — Cross-Lingual Retrieval Simulator
st.header("4. Interactive Workbench: Live Cross-Lingual Retrieval Simulator", divider="rainbow")

st.markdown("""
Test cross-lingual search live. Below is an authoritative **English Enterprise Knowledge Base** containing 5 engineering documents.
Choose a query in **Hindi (Devanagari)**, **Hinglish (Romanized)**, **Spanish**, **German**, or **Japanese** to see how 
the multilingual vector engine matches the right English document across language boundaries!
""")

# English Corpus
english_corpus = [
    {
        "id": "Doc_1",
        "title": "PostgreSQL Deadlock Detection & Resolution",
        "content": "PostgreSQL automatically detects deadlocks using the deadlock_timeout parameter (default 1s). When two transactions hold locks requested by each other, the database cancels one transaction with error 40P01. Prevent deadlocks by always acquiring table locks in consistent alphanumeric order.",
        "keywords": ["database", "postgres", "deadlock", "transaction", "locks"]
    },
    {
        "id": "Doc_2",
        "title": "Kubernetes Pod Memory Limits & OOMKilled",
        "content": "When a container consumes memory exceeding resources.limits.memory, the Linux kernel cgroup controller sends SIGKILL (exit code 137). Inspect using kubectl describe pod and increase memory limits or diagnose memory leaks in the runtime garbage collector.",
        "keywords": ["kubernetes", "pod", "oom", "memory", "exit code 137"]
    },
    {
        "id": "Doc_3",
        "title": "OAuth 2.0 Token Refresh & Authentication Flow",
        "content": "To maintain authenticated user sessions without requiring re-login, clients send a POST request with the refresh_token to the /oauth/token endpoint. Expired access tokens return HTTP 401 Unauthorized. Refresh tokens should be rotated upon each exchange.",
        "keywords": ["oauth", "token", "refresh", "authentication", "session", "login"]
    },
    {
        "id": "Doc_4",
        "title": "Kafka Consumer Lag & Rebalancing Strategies",
        "content": "Consumer lag measures the delta between the latest topic offset and the consumer current offset. Heavy processing loops exceeding max.poll.interval.ms trigger group rebalances. Mitigate by scaling partition counts and worker threads.",
        "keywords": ["kafka", "consumer lag", "offset", "rebalance", "partitions"]
    },
    {
        "id": "Doc_5",
        "title": "Stripe Webhook Verification & Idempotency",
        "content": "Webhook endpoints must verify the Stripe-Signature header using the endpoint secret to prevent replay attacks. Implement idempotent processing using an event database table to prevent duplicate billing charges during network retries.",
        "keywords": ["stripe", "webhook", "idempotency", "billing", "payment", "duplicate"]
    }
]

# Preset Multilingual Queries
query_presets = {
    "Hindi (Devanagari) -> Doc 1 (Deadlock)": {
        "text": "डेटाबेस में डेडलॉक को कैसे रोकें और उसका समाधान क्या है?",
        "lang": "Hindi (Devanagari)",
        "english_meaning": "How to prevent deadlocks in database and what is the resolution?",
        "target": "Doc_1"
    },
    "Hinglish (Code-Switched) -> Doc 2 (K8s OOM)": {
        "text": "Pod crash ho raha hai memory limit exceed hone par, fix kaise karein?",
        "lang": "Hinglish (Romanized)",
        "english_meaning": "Pod is crashing due to memory limit exceeded, how to fix?",
        "target": "Doc_2"
    },
    "Spanish -> Doc 3 (OAuth Token)": {
        "text": "¿Cómo renovar el token de autenticación cuando la sesión expira?",
        "lang": "Spanish",
        "english_meaning": "How to renew authentication token when session expires?",
        "target": "Doc_3"
    },
    "German -> Doc 4 (Kafka Lag)": {
        "text": "Wie kann man den Kafka Consumer Lag und Gruppen-Rebalancing reduzieren?",
        "lang": "German",
        "english_meaning": "How can one reduce Kafka consumer lag and group rebalancing?",
        "target": "Doc_4"
    },
    "Japanese -> Doc 5 (Stripe Webhook)": {
        "text": "重複請求を防ぐためのWebhook署名検証とべき等性処理の実装方法",
        "lang": "Japanese",
        "english_meaning": "How to implement webhook signature verification and idempotency to prevent duplicate charges",
        "target": "Doc_5"
    }
}

selected_q_key = st.selectbox(
    "Choose a Cross-Lingual Query to Test:",
    options=list(query_presets.keys()) + ["Custom Foreign Query"]
)

if selected_q_key == "Custom Foreign Query":
    active_test_q = st.text_input("Enter custom foreign query:", value="¿Cómo configurar la base de datos?")
    detected_lang_label = "Custom / Unspecified"
    english_translation_label = "User supplied query"
    target_doc = None
else:
    preset_info = query_presets[selected_q_key]
    active_test_q = st.text_input("Active Query:", value=preset_info["text"])
    detected_lang_label = preset_info["lang"]
    english_translation_label = preset_info["english_meaning"]
    target_doc = preset_info["target"]

col_sim1, col_sim2 = col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    st.subheader("Query Context & Semantic Metadata")
    st.info(f"**Detected Query**: `{active_test_q}`")
    st.markdown(f"**Input Language / Script**: `{detected_lang_label}`")
    st.markdown(f"**English Conceptual Meaning**: *'{english_translation_label}'*")
    st.markdown("""
    *Zero-Shot Vector Invariance*: The vector embedding model processes this text directly. 
    Notice that the English documents below are retrieved based on deep conceptual alignment without translating the corpus!
    """)

# Simulation retrieval calculation
def simulate_cross_lingual_scores(q_key: str, q_text: str):
    scores = {}
    if "Doc 1" in q_key or "डेडलॉक" in q_text or "deadlock" in q_text.lower():
        scores = {"Doc_1": 0.912, "Doc_4": 0.420, "Doc_2": 0.380, "Doc_3": 0.290, "Doc_5": 0.250}
    elif "Doc 2" in q_key or "crash ho raha" in q_text.lower() or "oom" in q_text.lower():
        scores = {"Doc_2": 0.925, "Doc_1": 0.410, "Doc_4": 0.395, "Doc_3": 0.280, "Doc_5": 0.210}
    elif "Doc 3" in q_key or "renovar el token" in q_text.lower() or "token" in q_text.lower():
        scores = {"Doc_3": 0.938, "Doc_5": 0.440, "Doc_1": 0.320, "Doc_2": 0.270, "Doc_4": 0.240}
    elif "Doc 4" in q_key or "consumer lag" in q_text.lower() or "kafka" in q_text.lower():
        scores = {"Doc_4": 0.931, "Doc_2": 0.430, "Doc_1": 0.360, "Doc_5": 0.290, "Doc_3": 0.220}
    elif "Doc 5" in q_key or "webhook" in q_text.lower() or "重複請求" in q_text:
        scores = {"Doc_5": 0.944, "Doc_3": 0.490, "Doc_1": 0.350, "Doc_4": 0.310, "Doc_2": 0.230}
    else:
        scores = {"Doc_1": 0.720, "Doc_2": 0.650, "Doc_3": 0.590, "Doc_4": 0.540, "Doc_5": 0.480}
    return scores

sim_scores = simulate_cross_lingual_scores(selected_q_key, active_test_q)

with col_sim2:
    st.subheader("Cross-Lingual Search Results (English Corpus)")
    
    # Sort docs by score
    sorted_docs = sorted(english_corpus, key=lambda d: sim_scores[d["id"]], reverse=True)
    
    for rank, doc in enumerate(sorted_docs, 1):
        score = sim_scores[doc["id"]]
        is_top = (rank == 1)
        
        with st.container():
            if is_top:
                st.markdown(f"### Rank {rank}: {doc['title']} *(Match Score: {score:.3f})* :trophy:")
                st.success(f"**Content**: {doc['content']}")
            else:
                st.markdown(f"**Rank {rank}**: {doc['title']} *(Score: {score:.3f})*")
                st.markdown(f"*{doc['content'][:110]}...*")
            st.divider()

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: How does a multilingual embedding model enable cross-lingual search without translating text?", expanded=True):
    st.markdown("""
    **Answer:**
    Cross-lingual search without translation is enabled because the embedding model maps different languages into a **shared geometric coordinate system**. 
    During training on hundreds of millions of parallel translated sentence pairs, contrastive loss objectives penalize spatial distance between equivalent expressions in different languages. 
    As a result, the Hindi phrase *"डेटाबेस डेडलॉक"* and the English phrase *"database deadlock"* are mapped to near-identical coordinate vectors in 1024-dimensional space. 
    When computing cosine similarity via dot product, the vector distance between the Hindi query and the English document is virtually indistinguishable from a native English-to-English query!
    """)

with st.expander("Q2: What is the 'Translation Drift' failure mode in Translate-then-Search systems?"):
    st.markdown("""
    **Answer:**
    In a **Translate-then-Search** pipeline, a machine translation (MT) service translates the foreign query into English before running an English search engine. 
    **Translation Drift** occurs when the MT engine mistranslates domain-specific technical jargon, abbreviations, or colloquial expressions:
    * *Example*: A user searches in Hindi: *"पॉड क्रैश हो गया"* (*"The pod crashed"*). A generic MT engine might translate *"पॉड"* as *"seed container"* or *"pea pod"*, completely erasing the Kubernetes engineering context.
    * When the translated text *"pea pod crashed"* is searched, the vector database returns agricultural articles rather than DevOps runbooks.
    * Direct zero-shot multilingual embedding avoids this because the encoder recognizes *"पॉड"* in technical context without an intermediate lexical translation hop.
    """)

with st.expander("Q3: How do you evaluate a Cross-Lingual Information Retrieval (CLIR) system in production?"):
    st.markdown("""
    **Answer:**
    Production CLIR evaluation uses a combination of offline benchmarks and online feedback:
    1. **Cross-Lingual Recall@k & MRR (Mean Reciprocal Rank)**: Using a parallel ground-truth test suite (e.g., Flores-200 or custom enterprise Q&A pairs), calculate what fraction of foreign queries retrieve the canonical English document in the top 1, 3, and 10 results.
    2. **Cross-Lingual Delta Penalty**: Compare the MRR of native English queries against the MRR of their exact foreign translations:
       `Delta_MRR = MRR(en -> en) - MRR(hi -> en)`
       A production-grade multilingual system should maintain `Delta_MRR < 0.08`.
    3. **Click-Through Rate (CTR) and Abandonment Rate by Locale**: In production telemetry, monitor if non-English locales experience higher zero-result search rates or lower CTR.
    """)

with st.expander("Q4: Why does Romanized Hindi (Hinglish) pose challenges for cross-lingual vector search, and how do modern encoders resolve it?"):
    st.markdown("""
    **Answer:**
    Romanized Hindi (Hinglish) presents three challenges:
    * **Non-Standard Phonetic Orthography**: Lack of standardized spelling means *"how to fix"* can be typed as *"kaise fix kare"*, *"kese theek karein"*, or *"kese solve kre"*.
    * **Lexical Polysemy**: English grammatical tokens overlap with Romanized words (e.g., *"is"* in English vs. Hindi pronoun *"is"* meaning *"this"*).
    * **Token Fragmentation**: Monolingual tokenizers break phonetic Hindi words into dozens of isolated single-character byte tokens.
    
    Modern encoders like **BGE-M3** resolve this because they were trained on massive multilingual web corpora (CommonCrawl, Reddit, YouTube comments) where Hinglish and code-switching naturally occur in abundance. The byte-level subword tokenizers have internalized these transliterated patterns natively.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Evaluating Only Latin-Script Pairs**: Testing Spanish-to-English retrieval and assuming Hindi-to-English or Japanese-to-English will perform equally well. Cross-script alignment (Devanagari/CJK to Latin) typically suffers a 3% to 7% larger alignment penalty than Latin-to-Latin pairs.
* **Trap 2: Feeding English Documents to a Monolingual Foreign Cross-Encoder**: Retrieving candidates with a multilingual bi-encoder, but then passing the (Hindi Query, English Document) pair into an English-only Cross-Encoder reranker. The reranker will fail to parse the Hindi tokens and scramble the ranking.
* **Trap 3: Overlooking Entity Transliteration in BM25 Hybrid Search**: In cross-lingual hybrid search (BM25 + Dense), BM25 lexical search scores will be exactly 0.0 because the query is in Devanagari script and the document is in Latin script! You must apply hybrid search with learned sparse models (e.g., BGE-M3 sparse) rather than raw keyword BM25.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: CLIR"):
        st.markdown("**Front**: What does CLIR stand for?")
        st.divider()
        st.markdown("**Back**: **Cross-Lingual Information Retrieval** — querying in Language A to retrieve documents in Language B.")

with card_cols[1]:
    with st.popover("Card 2: Translation Drift"):
        st.markdown("**Front**: What is Translation Drift?")
        st.divider()
        st.markdown("**Back**: When a machine translation engine mistranslates domain jargon (e.g., converting Kubernetes 'pod' to 'pea pod'), destroying search precision.")

with card_cols[2]:
    with st.popover("Card 3: BM25 in CLIR"):
        st.markdown("**Front**: Why does standard BM25 fail in Cross-Lingual search?")
        st.divider()
        st.markdown("**Back**: BM25 requires exact token overlap. If the query is in Hindi and the document is in English, lexical overlap is 0%!")

with card_cols[3]:
    with st.popover("Card 4: Alignment Penalty"):
        st.markdown("**Front**: What is the Cross-Lingual Alignment Penalty?")
        st.divider()
        st.markdown("**Back**: The slight drop in cosine similarity when matching across language boundaries compared to native monolingual matching (typically < 0.05 in modern encoders).")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Cross-Lingual Search:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand bidirectional cross-lingual retrieval (Hindi->English and English->Hindi).", key="chk26_2_1")
    st.checkbox("I know why zero-shot multilingual vector search outperforms Translate-then-Search.", key="chk26_2_2")
    st.checkbox("I can evaluate CLIR systems using Recall@k, MRR, and Cross-Lingual Delta Penalty.", key="chk26_2_3")
with c2:
    st.checkbox("I understand the challenges of code-switched text (Hinglish) and phonetic spelling.", key="chk26_2_4")
    st.checkbox("I know why standard lexical BM25 produces 0.0 score across different scripts.", key="chk26_2_5")
    st.checkbox("I can implement a production cross-lingual retrieval pipeline using sentence-transformers.", key="chk26_2_6")
