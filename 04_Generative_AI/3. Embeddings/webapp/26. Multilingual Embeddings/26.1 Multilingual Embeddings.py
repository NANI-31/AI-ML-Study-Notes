import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="26.1 Multilingual Embeddings - Embeddings Curriculum",
    page_icon="🌐",
    layout="wide"
)

# Title & Overview
st.title("26.1 Multilingual Embeddings: Shared Representation Spaces, Alignment & Language Detection")
st.caption("Chapter 26: Multilingual Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In global enterprise environments, information is authored in dozens of languages across continents. 
Historically, building a search engine for multiple languages required maintaining separate language-specific tokenizers, 
stemmers, and isolated index catalogs for every single supported locale (e.g., an English Elasticsearch cluster, a Hindi index, a Japanese index).

**Multilingual Embedding Models** fundamentally revolutionized this paradigm by mapping texts from over 100+ distinct languages 
into a **single, unified high-dimensional vector space**. 
In this shared geometric space, semantic meaning—not spelling, grammar, or alphabet—determines vector position:
* *"The server crashed due to an out of memory error"* (English)
* *"मेमोरी की कमी के कारण सर्वर क्रैश हो गया"* (Hindi)
* *"El servidor se cayó debido a un error de falta de memoria"* (Spanish)
* *"メモリ不足のためサーバーがクラッシュしました"* (Japanese)

All four sentences cluster tightly in the exact same neighborhood of 1024-dimensional space, enabling seamless 
cross-lingual semantic matching, unified retrieval, and global RAG architectures without translation overhead.
""")

st.divider()

# Section 1: The Shared Multilingual Vector Space
st.header("1. Architecture of Shared Multilingual Vector Spaces", divider="rainbow")

col_space1, col_space2 = col_space1, col_space2 = st.columns([1, 1])

with col_space1:
    st.markdown("""
    ### How Cross-Language Alignment Works
    Multilingual models (such as **BGE-M3**, **Cohere embed-multilingual-v3.0**, **LaBSE**, and **XLM-RoBERTa**) achieve joint alignment through specialized training objectives:
    
    1. **Parallel Corpus Contrastive Learning**:
       * The model is fed parallel translation pairs `(sentence_en, sentence_hi)`.
       * The objective maximizes the cosine similarity between translations while minimizing similarity against unrelated sentences in any language:
         `Sim(Embedding(English), Embedding(Hindi)) -> 1.0`
    2. **Shared Subword / Byte-Level Vocabularies**:
       * Uses tokenizers like SentencePiece or Byte-Pair Encoding (BPE) with 250,000+ shared subword tokens covering Latin, Devanagari, Cyrillic, Arabic, CJK, and Tamil scripts.
    3. **Language-Agnostic Semantic Anchors**:
       * Mathematical entities, technical terms (*"Kubernetes"*, *"HTTP 500"*, *"SQL"*), and foundational human concepts anchor the manifold, pulling syntax across language boundaries together.
    """)

with col_space2:
    st.markdown("### The Shared Multilingual Embedding Manifold")
    manifold_dot = """
    digraph MultilingualManifold {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        subgraph cluster_languages {
            label="Input Texts in Diverse Scripts"; style="dashed"; color="#3B82F6";
            TxtEN [label="English (Latin):\\n'Payment failed'", fillcolor="#DBEAFE"];
            TxtHI [label="Hindi (Devanagari):\\n'भुगतान विफल रहा'", fillcolor="#FED7AA"];
            TxtES [label="Spanish (Latin):\\n'El pago falló'", fillcolor="#D1FAE5"];
            TxtJA [label="Japanese (Kanji):\\n'支払いに失敗しました'", fillcolor="#FCE7F3"];
        }

        MultiModel [label="Multilingual Encoder\\n(BGE-M3 / Cohere v3 / LaBSE)\\nShared 1024-dim Projection", fillcolor="#FEF3C7", color="#F59E0B", style="filled,bold"];

        subgraph cluster_manifold {
            label="Single Unified Vector Space (R^1024)"; style="dashed"; color="#10B981";
            ClusterFinance [label="Semantic Cluster: 'Payment Failure'\\nv_EN ≈ v_HI ≈ v_ES ≈ v_JA\\nCosine Sim > 0.92", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
            ClusterOther [label="Semantic Cluster: 'Server Deployment'\\nDistinct orthogonal region", fillcolor="#E5E7EB"];
        }

        TxtEN -> MultiModel;
        TxtHI -> MultiModel;
        TxtES -> MultiModel;
        TxtJA -> MultiModel;

        MultiModel -> ClusterFinance [label="Aligned into same cluster", color="#10B981", style="bold"];
    }
    """
    st.graphviz_chart(manifold_dot, use_container_width=False)

st.markdown("""
### Multilingual Model Landscape Comparison
| Model Name | Creator / License | Embedding Dimensions | Supported Languages | Maximum Context Window | Primary Strength |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BGE-M3** | BAAI (Open Source) | 1024 | 100+ | 8,192 tokens | Multi-linguality, Multi-granularity, Multi-functionality (Dense + Sparse + ColBERT) |
| **embed-multilingual-v3.0** | Cohere (Commercial) | 1024 | 100+ | 512 tokens | Superior cross-lingual semantic alignment, search and compression modes |
| **text-embedding-3-large** | OpenAI (Commercial) | 3072 (or 1536) | 100+ | 8,191 tokens | High multilingual capacity, dynamic MRL dimension reduction |
| **LaBSE** | Google (Open Source) | 768 | 109 | 512 tokens | Benchmark pioneer for bi-text sentence translation alignment |
| **paraphrase-multilingual-mpnet-base-v2** | HuggingFace / SBERT | 768 | 50+ | 128 tokens | Lightweight, low-latency CPU deployment |
""")

st.divider()

# Section 2: Language Detection & Metadata Tagging
st.header("2. Language Detection & Pre-Retrieval Metadata Tagging", divider="rainbow")

st.markdown("""
Even when using a unified multilingual vector model, **Language Detection** remains a critical component of enterprise architecture:
1. **Metadata Filtering**: In many enterprise use-cases, users only want documents legally binding in their specific jurisdiction or language. Pre-tagging vectors with `{ "lang": "es" }` allows filtered ANN search.
2. **Dynamic Prompt & Reranker Routing**: Knowing the user's language allows routing the query to language-optimized rerankers (e.g., Japanese-specific vs. English-specific cross-encoders) and instructing the generation LLM to reply in the detected language.
3. **Low-Resource Language Warning**: Detecting languages with sparse representation in the embedding model allows falling back to machine translation pipelines.
""")

col_det1, col_det2 = col_det1, col_det2 = st.columns([1, 1])

with col_det1:
    st.markdown("### Language Detection Tools Comparison")
    st.markdown("""
    * **FastText (`fasttext-langdetect`)**:
      * Sub-1ms inference time on CPU; supports 176 languages.
      * Uses n-gram embeddings; highly accurate on short phrases.
    * **CLD3 (Compact Language Detector v3)**:
      * Google's neural network based detector implemented in C++/Python.
      * Highly robust on mixed-script and web content.
    * **Langdetect**:
      * Port of Google's language-detection library using naive Bayes.
      * Fast, but can struggle on 1-2 word queries.
    """)

with col_det2:
    st.markdown("### Production Code: Ingestion Language Tagging")
    st.code("""
import fasttext
import chromadb

# Load lightweight FastText language identification model
# model = fasttext.load_model('lid.176.ftz')

def detect_language(text: str) -> str:
    \"\"\"Simulated fast language detection.\"\"\"
    # In production:
    # predictions = model.predict(text.replace('\\n', ' '))
    # return predictions[0][0].replace('__label__', '')
    return "en"

def index_multilingual_document(collection, doc_id: str, text: str, embedding: list[float]):
    lang_code = detect_language(text)
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "language": lang_code,
            "char_count": len(text),
            "script": "Latin" if lang_code in ["en", "es", "fr"] else "Devanagari"
        }]
    )
""", language="python")

st.divider()

# Section 3: The Hubness Problem & Cross-Lingual Vector Geometry
st.header("3. The Hubness Problem & Cross-Lingual Geometry", divider="rainbow")

st.markdown("""
### The "Hubness" Anomaly in High-Dimensional Multilingual Spaces
In high-dimensional multilingual spaces (typically 768 to 1536 dimensions), researchers discovered an intriguing mathematical pathology known as **Hubness**:
* A small subset of points—called **Hubs**—appear as the nearest neighbors to a disproportionately large number of query vectors across languages, even when semantically unrelated!
* Hubness is caused by the concentration of distances in high-dimensional manifolds combined with linguistic asymmetry.
* High-resource languages (English) often occupy dense, central regions of the manifold, while low-resource languages (e.g., Swahili, Marathi) can be pushed toward peripheral sub-spaces.

### Production Mitigations:
1. **L2 Normalization**: Always project all embeddings onto the unit hypersphere (`||v|| = 1.0`). Cosine distance mitigates norm distortion.
2. **Cross-Lingual Cross-Encoders**: Reranking the top 50 candidates through a multilingual cross-encoder (e.g., `BAAI/bge-reranker-v2-m3` or `Cohere Rerank 3 Multilingual`) effectively purges false positive hubs.
3. **Centroid Centering**: In extreme cases, subtracting the language-specific centroid vector (`v - mean_lang`) centers each language manifold before cosine distance computation.
""")

st.divider()

# Section 4: Interactive Workbench — Multilingual Alignment & Similarity Matrix
st.header("4. Interactive Workbench: Cross-Language Semantic Alignment Simulator", divider="rainbow")

st.markdown("""
Explore how identical semantic concepts across English, Hindi, Spanish, French, German, and Japanese 
align in a shared multilingual vector space.
""")

concept_presets = {
    "Payment Failure Error": {
        "en": "The transaction failed because the credit card was declined by the issuing bank.",
        "hi": "लेन-देन विफल रहा क्योंकि जारीकर्ता बैंक द्वारा क्रेडिट कार्ड अस्वीकार कर दिया गया था।",
        "es": "La transacción falló porque la tarjeta de crédito fue rechazada por el banco emisor.",
        "fr": "La transaction a échoué car la carte de crédit a été refusée par la banque émettrice.",
        "de": "Die Transaktion ist fehlgeschlagen, da die Kreditkarte von der ausstellenden Bank abgelehnt wurde.",
        "ja": "発行銀行によってクレジットカードが拒否されたため、取引は失敗しました。"
    },
    "Kubernetes Resource Exhaustion": {
        "en": "The container was terminated by the OOM killer due to excessive memory usage.",
        "hi": "अत्यधिक मेमोरी उपयोग के कारण OOM किलर द्वारा कंटेनर को समाप्त कर दिया गया था।",
        "es": "El contenedor fue terminado por el OOM killer debido al uso excesivo de memoria.",
        "fr": "Le conteneur a été arrêté par l'OOM killer en raison d'une utilisation excessive de la mémoire.",
        "de": "Der Container wurde vom OOM-Killer aufgrund übermäßiger Speichernutzung beendet.",
        "ja": "メモリの過剰使用により、コンテナはOOMキラーによって終了されました。"
    },
    "User Password Reset": {
        "en": "Please click the security link below to reset your enterprise account password.",
        "hi": "अपने एंटरप्राइज़ खाते का पासवर्ड रीसेट करने के लिए कृपया नीचे दिए गए सुरक्षा लिंक पर क्लिक करें।",
        "es": "Haga clic en el enlace de seguridad a continuación para restablecer la contraseña de su cuenta empresarial.",
        "fr": "Veuillez cliquer sur le lien de sécurité ci-dessous pour réinitialiser le mot de passe de votre compte d'entreprise.",
        "de": "Bitte klicken Sie auf den unten stehenden Sicherheitslink, um Ihr Passwort für das Unternehmenskonto zurückzusetzen.",
        "ja": "エンタープライズアカウントのパスワードをリセットするには、下のセキュリティリンクをクリックしてください。"
    }
}

selected_concept = st.selectbox(
    "Choose a Real-World Concept to Inspect Cross-Language Alignment:",
    options=list(concept_presets.keys())
)

concept_data = concept_presets[selected_concept]

col_wb1, col_wb2 = col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("Parallel Semantic Text Instances")
    lang_display = [
        {"Language": "English (en)", "Script": "Latin", "Sentence": concept_data["en"]},
        {"Language": "Hindi (hi)", "Script": "Devanagari", "Sentence": concept_data["hi"]},
        {"Language": "Spanish (es)", "Script": "Latin", "Sentence": concept_data["es"]},
        {"Language": "French (fr)", "Script": "Latin", "Sentence": concept_data["fr"]},
        {"Language": "German (de)", "Script": "Latin", "Sentence": concept_data["de"]},
        {"Language": "Japanese (ja)", "Script": "Kanji / Kana", "Sentence": concept_data["ja"]},
    ]
    st.dataframe(pd.DataFrame(lang_display), use_container_width=True, hide_index=True)

with col_wb2:
    st.subheader("Cross-Lingual Cosine Similarity Heatmap")
    
    # Generate realistic calibrated multilingual cosine similarity matrix
    langs = ["English", "Hindi", "Spanish", "French", "German", "Japanese"]
    n_langs = len(langs)
    
    # High baseline similarity between aligned translations (~0.88 to 0.96)
    np.random.seed(42 if "Payment" in selected_concept else (101 if "Kubernetes" in selected_concept else 202))
    sim_mat = np.ones((n_langs, n_langs), dtype=float)
    
    for i in range(n_langs):
        for j in range(i + 1, n_langs):
            # Slightly lower similarity across distant scripts (e.g. English vs Hindi / Japanese: 0.88 - 0.92, Latin vs Latin: 0.92 - 0.96)
            is_latin_pair = (i in [0, 2, 3, 4] and j in [0, 2, 3, 4])
            base_score = 0.93 if is_latin_pair else 0.89
            jitter = np.random.uniform(0.01, 0.04)
            score = round(base_score + jitter, 3)
            sim_mat[i, j] = score
            sim_mat[j, i] = score

    df_heatmap = pd.DataFrame(sim_mat, index=langs, columns=langs)
    
    styled_heatmap = df_heatmap.style.format("{:.3f}").background_gradient(
        cmap="YlGn", vmin=0.85, vmax=1.0
    )
    st.dataframe(styled_heatmap, use_container_width=True)
    
    st.caption("""
    *Key Insight*: In a monolingual model, English vs. Hindi would yield near 0.0 similarity due to disjoint token vocabularies. 
    In **BGE-M3** and **Cohere Multilingual**, cosine similarity remains consistently **> 0.89**, validating seamless cross-lingual retrieval.
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: How do multilingual embedding models map texts in completely different alphabets into the same vector space?", expanded=True):
    st.markdown("""
    **Answer:**
    Multilingual models accomplish cross-lingual alignment primarily through **multilingual contrastive pre-training on parallel and translation corpora** (e.g., bitext pairs from Wikipedia, UN proceedings, OpenSubtitles). 
    During training, the model's loss function minimizes the distance between paired translation sentences: 
    the vector for *"The server is down"* is explicitly pulled toward the vector for *"सर्वर डाउन है"* while being pushed away from negative non-translation sentences.
    
    Additionally, these models employ massive shared subword vocabularies (e.g., SentencePiece tokenizers with 250,000+ tokens) spanning all unicode scripts. 
    Through cross-lingual masked language modeling (MLM) and translation language modeling (TLM), the transformer layers learn that distinct script representations share invariant semantic properties.
    """)

with st.expander("Q2: What is the Hubness Problem in multilingual vector retrieval, and how is it addressed?"):
    st.markdown("""
    **Answer:**
    The **Hubness Problem** is an intrinsic geometric phenomenon in high-dimensional vector spaces where a small number of points (hubs) frequently appear among the $k$-nearest neighbors of almost all query vectors, even when semantically unrelated. 
    In multilingual spaces, hubness is exacerbated because high-resource languages (English) tend to cluster near the geometric center of the representation space, causing English vectors to act as universal nearest-neighbor attractors for queries in low-resource languages.
    
    **Production Mitigations**:
    1. **Strict L2 Normalization**: Forces all vectors onto the unit sphere, preventing high-magnitude central hubs from dominating Euclidean distance.
    2. **Multilingual Cross-Encoder Reranking**: Filtering the top 50 candidates through a cross-encoder evaluates bidirectional token attention, rejecting false positive hubs.
    3. **Inverted Softmax / Local Scaling (CSLS)**: Cross-Domain Similarity Local Scaling normalizes cosine similarity by the average distance of a vector to its nearest neighbors in both languages, punishing universal hubs.
    """)

with st.expander("Q3: When should you use a unified multilingual model versus a 'Translate-then-Search' pipeline?"):
    st.markdown("""
    **Answer:**
    * **Use Unified Multilingual Embeddings (BGE-M3, Cohere)** when:
      * Low latency is critical (< 50ms total SLA): Machine translation adds 100ms - 300ms of extra overhead per request.
      * The corpus is naturally mixed-language: Documents contain code snippets, multilingual customer feedback, or multilingual clauses.
      * Cost efficiency is paramount: Avoiding an external MT translation API saves substantial inference expenses.
    * **Use 'Translate-then-Search'** when:
      * Extremely low-resource languages are involved (e.g., dialects where multilingual embeddings have poor alignment < 0.60).
      * The underlying index is strictly lexical BM25 or fine-tuned on specialized English technical jargon where English models significantly outperform multilingual models.
    """)

with st.expander("Q4: Why does BGE-M3 feature 'multi-functionality' (Dense + Sparse + Multi-Vector ColBERT) for multilingual retrieval?"):
    st.markdown("""
    **Answer:**
    Different languages exhibit vastly different linguistic properties: 
    * Dense vectors excel at conceptual and semantic alignment across languages (matching concepts regardless of phrasing).
    * However, dense vectors struggle with exact technical keywords, product codes, or proper nouns transliterated across scripts.
    * BGE-M3 solves this by outputting three representations from a single forward pass:
      1. **Dense Vector (1024-d)**: For deep semantic alignment.
      2. **Sparse Lexical Vector (Learned BM25 weights)**: For exact keyword and entity matching in the target language.
      3. **Multi-Vector (ColBERT token-level vectors)**: For fine-grained token-to-token late interaction across languages.
    Combining all three delivers industry-leading Cross-Lingual Information Retrieval (CLIR) accuracy.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Searching Across Languages with Monolingual Models**: Using an English-only model (e.g., `all-MiniLM-L6-v2`) to encode a Hindi query against an English corpus. The Hindi tokens map to random out-of-vocabulary hash buckets, producing complete garbage retrieval.
* **Trap 2: Neglecting Language-Specific Tokenizer Expansion**: Using a multilingual model that compresses Asian languages (Chinese, Japanese, Korean) into multiple byte tokens per character, causing context windows to overflow prematurely on short paragraphs.
* **Trap 3: Omitting Language Metadata in Multi-Jurisdiction Legal Search**: Failing to tag documents with their canonical language, causing a Spanish search to return an English policy document when the user legally requires the Colombian-jurisdiction Spanish text.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Unified Manifold"):
        st.markdown("**Front**: How are multilingual texts positioned in embedding space?")
        st.divider()
        st.markdown("**Back**: Texts expressing the same semantic meaning cluster together regardless of language, alphabet, or grammatical syntax.")

with card_cols[1]:
    with st.popover("Card 2: Hubness"):
        st.markdown("**Front**: What is the Hubness Problem in high dimensions?")
        st.divider()
        st.markdown("**Back**: An anomaly where a few central vectors become false-positive nearest neighbors to unrelated queries across languages.")

with card_cols[2]:
    with st.popover("Card 3: BGE-M3"):
        st.markdown("**Front**: What are the 3 outputs of BGE-M3?")
        st.divider()
        st.markdown("**Back**: 1. Dense vector (1024-d), 2. Sparse lexical weights, 3. Multi-vector ColBERT late interaction embeddings.")

with card_cols[3]:
    with st.popover("Card 4: FastText Detection"):
        st.markdown("**Front**: Why is language detection used if the embedding model supports 100+ languages?")
        st.divider()
        st.markdown("**Back**: For metadata filtering, dynamic prompt routing, language-specific rerankers, and jurisdictional compliance.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Multilingual Embeddings:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand how parallel contrastive learning aligns languages in a shared space.", key="chk26_1")
    st.checkbox("I know the leading multilingual embedding models (BGE-M3, Cohere v3, OpenAI large).", key="chk26_2")
    st.checkbox("I can implement pre-retrieval language detection and metadata tagging.", key="chk26_3")
with c2:
    st.checkbox("I understand the Hubness problem and its mitigation strategies (L2 norm, reranking).", key="chk26_4")
    st.checkbox("I know the architectural trade-offs between Unified Multilingual vs Translate-then-Search.", key="chk26_5")
    st.checkbox("I can explain BGE-M3's multi-functional dense + sparse + multi-vector paradigm.", key="chk26_6")
