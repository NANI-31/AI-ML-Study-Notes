import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="26.3 Multilingual RAG - Embeddings Curriculum",
    page_icon="🌍",
    layout="wide"
)

# Title & Overview
st.title("26.3 Multilingual RAG: Ingestion, Multilingual Retrieval & Cross-Lingual Generation")
st.caption("Chapter 26: Multilingual Embeddings | Domain: 3. Vector Embeddings")
st.markdown("""
In global enterprise deployments, **Multilingual Retrieval-Augmented Generation (Multilingual RAG)** is the gold standard for customer support, 
clinical intelligence, and international engineering operations.
A true Multilingual RAG system achieves three remarkable capabilities simultaneously:
1. **Multilingual Corpus Ingestion**: Ingests unstructured PDFs, manuals, and support tickets authored in English, Japanese, German, Hindi, and Spanish into a single vector database without upfront translation.
2. **Cross-Lingual Retrieval**: Allows an end-user to query in their native tongue (e.g., Hindi or Portuguese) and retrieves the most relevant technical evidence even if that evidence exists only in English.
3. **Target-Language Generation**: Injects the foreign-language retrieved evidence into a frontier LLM prompt, instructing the model to synthesize a grounded, fluent answer **in the user's original language** with source citations.

This lesson explores the end-to-end architecture, script-aware chunking, cross-lingual reranking, and generation patterns of Multilingual RAG.
""")

st.divider()

# Section 1: End-to-End Multilingual RAG Architecture
st.header("1. End-to-End Multilingual RAG Pipeline Architecture", divider="rainbow")

col_pipe1, col_pipe2 = col_pipe1, col_pipe2 = st.columns([1, 1])

with col_pipe1:
    st.markdown("""
    ### The 4 Stages of Multilingual RAG
    * **Stage 1: Script-Aware Ingestion & Chunking**:
      * Ingests documents across various languages and alphabets.
      * Applies language-specific sentence splitters (e.g., splitting Japanese on `。`, Hindi on `।`, and Latin text on `.`).
      * Computes 1024-dimensional embeddings using a unified multilingual encoder (e.g., **BGE-M3** or **Cohere embed-multilingual-v3.0**).
      * Stores chunks in a single vector store tagged with metadata `{ "language": "en", "doc_type": "manual" }`.
    * **Stage 2: Cross-Lingual Candidate Retrieval**:
      * User submits a query in Language A (e.g., Hindi).
      * Multilingual encoder projects the query into the shared vector space.
      * Fast HNSW search retrieves the top 30 candidate chunks (which may be in English, German, or Japanese).
    * **Stage 3: Multilingual Cross-Encoder Reranking**:
      * Evaluates `(Query_langA, Chunk_langB)` pairs through a cross-lingual cross-encoder (e.g., **BAAI/bge-reranker-v2-m3**).
      * Distills candidates down to the top 3 to 5 highest-precision chunks.
    * **Stage 4: Grounded Multilingual Synthesis**:
      * LLM is instructed: *"Answer the user's question in Hindi using ONLY the provided English context chunks. Do not hallucinate."*
      * Produces a fluent, localized response with accurate citations.
    """)

with col_pipe2:
    st.markdown("### The Complete Pipeline Flow")
    rag_dot = """
    digraph MultilingualRAGPipeline {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="1. Inbound User Query (Hindi)\\n'डेटाबेस डेडलॉक कैसे हल करें?'", fillcolor="#FED7AA", color="#F97316", style="filled,bold"];
        
        LangDetect [label="Language Detector\\nIdentifies: 'hi' (Hindi)", fillcolor="#FEF3C7"];
        
        Encoder [label="2. Multilingual Encoder\\nBGE-M3 / Cohere v3 (1024-d)", fillcolor="#DBEAFE"];

        VectorStore [label="Unified Vector Store\\nEnglish Technical Manuals\\n100,000 Chunks", fillcolor="#E0E7FF", color="#3B82F6", style="filled,bold"];

        TopCandidates [label="Top 30 Candidates\\nEnglish & Mixed Language Chunks", fillcolor="#CFFAFE"];

        CrossRerank [label="3. Multilingual Reranker\\nBGE-Reranker-v2-m3\\nTop 4 Chunks", fillcolor="#D1FAE5", color="#10B981"];

        LLMPrompt [label="4. Cross-Lingual LLM Prompt\\nSystem: Answer in Hindi\\nContext: Top English Chunks", fillcolor="#F3E8FF"];

        FinalAnswer [label="Fluent Hindi Grounded Answer\\n'डेडलॉक रोकने के लिए तालिकाओं को...'\\nWith English Doc Citations", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        UserQ -> LangDetect;
        LangDetect -> Encoder;
        Encoder -> VectorStore [label="HNSW Cosine Search"];
        VectorStore -> TopCandidates;
        TopCandidates -> CrossRerank;
        CrossRerank -> LLMPrompt;
        UserQ -> LLMPrompt [label="User Query Language Anchor"];
        LLMPrompt -> FinalAnswer;
    }
    """
    st.graphviz_chart(rag_dot, use_container_width=False)

st.divider()

# Section 2: Script-Aware Chunking Challenges
st.header("2. Script-Aware Chunking: CJK, Devanagari & German Compounds", divider="rainbow")

st.markdown("""
A major pitfall in multilingual RAG is applying standard English-centric chunking algorithms (`RecursiveCharacterTextSplitter` splitting on spaces ` ` and periods `.`) to non-Latin languages:
""")

col_chunk1, col_chunk2 = col_chunk1, col_chunk2 = st.columns([1, 1])

with col_chunk1:
    st.markdown("""
    ### 1. CJK Languages (Chinese, Japanese, Korean)
    * **No Whitespace**: Japanese and Chinese do not use spaces between words.
    * **Sentence Terminators**: Full stops are denoted by ideographic full stops `。` (U+3002) rather than ASCII periods `.`.
    * **English Splitter Failure**: An English splitter looking for ` ` will treat an entire 2,000-character Japanese paragraph as a single unbroken word!
    * **Remediation**: Use language-aware splitters (e.g., `Konoha` or `MeCab` for Japanese, `jieba` for Chinese) or configure separators: `["\n\n", "\n", "。", "！", "？", " "]`.

    ### 2. Devanagari Scripts (Hindi, Marathi, Sanskrit)
    * **Sentence Terminators**: The traditional sentence delimiter is the Purna Virama `।` (U+0964) or double virama `॥`.
    * **Modern Mixed Text**: Modern web documents mix `।` and English `.` interchangeably.
    * **Remediation**: Invert separator hierarchy to prioritize `["\n\n", "।", ".", " "]`.
    """)

with col_chunk2:
    st.markdown("""
    ### 3. German & Agglutinative Languages
    * **Compound Nouns**: German routinely creates 30+ character words by joining nouns (e.g., *"Kraftfahrzeug-Haftpflichtversicherung"* = motor vehicle liability insurance).
    * **Token Bloat**: Standard subword tokenizers split long compounds into 8 to 12 fragmented byte tokens, consuming large portions of the model's context window.
    * **Remediation**: Adjust chunk character overlap thresholds upward (15% to 20% overlap) to prevent splitting critical compound words across boundary cuts.
    """)
    st.code("""
# Production Multilingual Separator Hierarchy
from langchain_text_splitters import RecursiveCharacterTextSplitter

multilingual_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    separators=[
        "\\n\\n",
        "\\n",
        "。",   # Chinese/Japanese full stop
        "！",   # Fullwidth exclamation
        "？",   # Fullwidth question mark
        "।",   # Hindi Purna Virama
        ".",    # Latin period
        "!",
        "?",
        " ",
        ""
    ]
)
""", language="python")

st.divider()

# Section 3: Grounded Multilingual Synthesis & Prompting
st.header("3. Grounded Multilingual Synthesis & Cross-Lingual Prompts", divider="rainbow")

st.markdown("""
When prompting the generative LLM (e.g., GPT-4o, Claude 3.5 Sonnet, or Gemini 1.5 Pro) with English context chunks to answer a Hindi, Spanish, or Japanese query, 
strict instruction constraints are mandatory to prevent **Cross-Lingual Hallucination**:
""")

col_syn1, col_syn2 = col_syn1, col_syn2 = st.columns([1, 1])

with col_syn1:
    st.markdown("### Production Cross-Lingual Prompt Template")
    st.code("""
MULTILINGUAL_RAG_PROMPT = \"\"\"You are an expert enterprise technical assistant.
You will be provided with:
1. The user's question, written in their native language ({user_lang}).
2. Authoritative context passages, which are written in English.

STRICT INSTRUCTIONS:
- You MUST generate your entire answer in the user's language: {user_lang}.
- Use ONLY facts directly mentioned in the English context passages.
- Do NOT extrapolate, speculate, or introduce external knowledge.
- If the English passages do not contain enough information, respond in {user_lang}:
  "I cannot answer this question based on the available documentation."
- Maintain accurate technical names in their canonical form (e.g., "Kubernetes", "PostgreSQL", "OAuth 2.0") alongside native explanations.
- Cite the source Document IDs [Doc_X] for every claim.

Context Passages:
{context_passages}

User Question ({user_lang}):
{user_query}

Grounded Response in {user_lang}:\"\"\"
""", language="python")

with col_syn2:
    st.markdown("### Why LLMs Excel at Cross-Lingual Synthesis")
    st.markdown("""
    Frontier LLMs (such as GPT-4o and Claude 3.5) have massive multilingual representation layers in their feed-forward networks:
    * **Latent Cross-Lingual Reasoning**: The model internalizes the factual logic of the English text and naturally expresses it in the target syntax.
    * **Preservation of Technical Entities**: High-parameter models intuitively avoid translating standard code commands (`kubectl`, `SELECT`, `SIGKILL`) while fluently translating descriptive prose.
    * **Hallucination Containment**: Forcing citations back to `[Doc_X]` forces the model to bind every translated phrase to a specific sentence in the English context.
    """)

st.divider()

# Section 4: Interactive Workbench — End-to-End Multilingual RAG Simulator
st.header("4. Interactive Workbench: Live Multilingual RAG Pipeline Stepper", divider="rainbow")

st.markdown("""
Experience all 4 stages of a production Multilingual RAG pipeline live! 
Choose an inbound query in **Hindi**, **Spanish**, or **Japanese**. Watch how the system identifies the language, 
retrieves the right English technical passages, filters them through a multilingual reranker, and synthesizes 
a fully grounded answer in the user's native language.
""")

rag_scenarios = {
    "Scenario 1: Hindi Query (DevOps Pod OOM)": {
        "query": "Kubernetes में pod exit code 137 से crash क्यों होता है और इसे कैसे ठीक करें?",
        "lang_code": "hi",
        "lang_name": "Hindi",
        "retrieved_docs": [
            {
                "id": "Doc_K8S_01",
                "title": "Kubernetes Pod Memory Limits & OOM Killer",
                "snippet": "Exit code 137 indicates that the Linux kernel OOM killer sent SIGKILL (signal 9) to the container because it exceeded the memory limit defined in 'resources.limits.memory'. To remediate, increase limits or tune JVM/runtime heap flags.",
                "initial_rank": 1,
                "rerank_score": 0.942
            },
            {
                "id": "Doc_K8S_02",
                "title": "Configuring Pod Resource Requests vs Limits",
                "snippet": "Kubernetes schedules pods based on requests. If limits are set too low, the kubelet or kernel will terminate pods during bursts. Set limits equal to or 1.5x of peak expected load.",
                "initial_rank": 2,
                "rerank_score": 0.885
            }
        ],
        "generated_answer": """Kubernetes में पॉड का **exit code 137** से क्रैश होना यह दर्शाता है कि कंटेनर को लिनक्स कर्नेल के **OOM (Out of Memory) किलर** द्वारा **SIGKILL (सिग्नल 9)** भेजकर समाप्त कर दिया गया था [Doc_K8S_01]। यह तब होता है जब कंटेनर पॉड विनिर्देश में परिभाषित `resources.limits.memory` की सीमा को पार कर जाता है।

**इसे ठीक करने के उपाय:**
1. परिनियोजन (Deployment) घोषणापत्र में `limits.memory` के मान को बढ़ाएं [Doc_K8S_01]।
2. सुनिश्चित करें कि संसाधन सीमाएं अपेक्षित पीक लोड के अनुसार 1.5 गुना पर कॉन्फ़िगर की गई हैं [Doc_K8S_02]।
3. रनटाइम मेमोरी हीप या कचरा संग्रहकर्ता (Garbage Collector) को ट्यून करें ताकि अत्यधिक मेमोरी खपत न हो [Doc_K8S_01] ।"""
    },
    "Scenario 2: Spanish Query (PostgreSQL Deadlocks)": {
        "query": "¿Cómo puedo prevenir los bloqueos mutuos (deadlocks) en PostgreSQL y cuál es la causa?",
        "lang_code": "es",
        "lang_name": "Spanish",
        "retrieved_docs": [
            {
                "id": "Doc_PG_01",
                "title": "PostgreSQL Lock Hierarchy & Deadlock Resolution",
                "snippet": "Deadlocks occur when two or more transactions hold exclusive locks that the other transactions need to acquire. PostgreSQL cancels the transaction after deadlock_timeout. Prevent deadlocks by always acquiring locks on tables in a consistent order.",
                "initial_rank": 1,
                "rerank_score": 0.938
            },
            {
                "id": "Doc_PG_02",
                "title": "Transaction Isolation Levels in Postgres",
                "snippet": "Using Read Committed or Repeatable Read isolation levels reduces lock duration. Keep transactions short and avoid long-running updates.",
                "initial_rank": 3,
                "rerank_score": 0.864
            }
        ],
        "generated_answer": """En PostgreSQL, los **bloqueos mutuos (deadlocks)** ocurren cuando dos o más transacciones mantienen bloqueos exclusivos que las otras transacciones necesitan adquirir simultáneamente [Doc_PG_01]. Tras alcanzar el umbral de `deadlock_timeout`, PostgreSQL cancela automáticamente una de las transacciones con el error 40P01.

**Medidas de prevención:**
1. **Adquirir bloqueos en orden consistente**: Asegúrese de que todas las consultas de la aplicación accedan y bloqueen las tablas en el mismo orden alfanumérico [Doc_PG_01].
2. **Mantener transacciones cortas**: Reduzca el tiempo de retención de bloqueos ejecutando transacciones breves [Doc_PG_02].
3. **Optimizar niveles de aislamiento**: Utilizar `Read Committed` o `Repeatable Read` según convenga para minimizar la duración del bloqueo [Doc_PG_02]."""
    },
    "Scenario 3: Japanese Query (OAuth Token Refresh)": {
        "query": "セッションが切れた際に、ユーザーをログアウトさせずにアクセストークンを更新するにはどうすればよいですか？",
        "lang_code": "ja",
        "lang_name": "Japanese",
        "retrieved_docs": [
            {
                "id": "Doc_AUTH_01",
                "title": "OAuth 2.0 Refresh Token Grant Specification",
                "snippet": "Clients obtain a new access token by presenting the refresh_token to the /oauth/token endpoint with grant_type=refresh_token. This keeps the session alive without requiring the user to re-authenticate with credentials.",
                "initial_rank": 1,
                "rerank_score": 0.951
            },
            {
                "id": "Doc_AUTH_02",
                "title": "Refresh Token Rotation & Security Guidelines",
                "snippet": "For maximum security in Single Page Apps, issue a new refresh token with every refresh exchange and immediately invalidate the old token.",
                "initial_rank": 2,
                "rerank_score": 0.890
            }
        ],
        "generated_answer": """セッションが期限切れになった際にユーザーをログアウトさせずにアクセストークンを更新するには、**OAuth 2.0 リフレッシュトークン（Refresh Token）** を使用します [Doc_AUTH_01]。

**実装手順:**
1. **トークン更新リクエストの送信**: クライアントは、認証エンドポイント（`/oauth/token`）に対して `grant_type=refresh_token` と保存されている `refresh_token` を含むPOSTリクエストを送信します [Doc_AUTH_01]。
2. **新しいアクセストークンの受信**: 認可サーバーがリフレッシュトークンを検証し、新しいアクセストークンを発行するため、ユーザーはパスワードを再入力することなくセッションを維持できます [Doc_AUTH_01]。
3. **トークンローテーションの実施**: セキュリティ向上のため、トークン更新ごとに新しいリフレッシュトークンを発行し、古いトークンを無効化する「リフレッシュトークンローテーション」を実装することが推奨されます [Doc_AUTH_02]。"""
    }
}

selected_scenario_name = st.selectbox(
    "Select Multilingual RAG Scenario to Step Through:",
    options=list(rag_scenarios.keys())
)

active_scenario = rag_scenarios[selected_scenario_name]

col_step1, col_step2 = col_step1, col_step2 = st.columns([1, 1])

with col_step1:
    st.subheader("Stage 1 & 2: Inbound Query & Cross-Lingual Retrieval")
    st.markdown(f"**Inbound User Query**: `{active_scenario['query']}`")
    st.info(f"**Detected Language**: `{active_scenario['lang_name']}` (`{active_scenario['lang_code']}`)")
    
    st.markdown("### Retrieved English Corpus Chunks (Pre-Rerank):")
    for doc in active_scenario["retrieved_docs"]:
        st.markdown(f"**{doc['id']} - {doc['title']}** (Initial Rank: #{doc['initial_rank']})")
        st.markdown(f"*{doc['snippet']}*")
        st.divider()

with col_step2:
    st.subheader("Stage 3 & 4: Cross-Encoder Rerank & Native Generation")
    
    st.markdown("### Multilingual Cross-Encoder Reranking Scores:")
    score_data = [
        {"Doc ID": d["id"], "Title": d["title"], "Cross-Lingual Score": f"{d['rerank_score']:.3f}"}
        for d in active_scenario["retrieved_docs"]
    ]
    st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)
    
    st.markdown(f"### Final LLM Synthesis (Generated in {active_scenario['lang_name']}):")
    st.success(active_scenario["generated_answer"])
    st.caption("""
    *Zero Hallucination Guarantee*: The response accurately explains complex English DevOps/Auth concepts in fluent native syntax while retaining canonical entity names (`resources.limits.memory`, `grant_type=refresh_token`) and strict source attribution tags!
    """)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

with st.expander("Q1: What are the primary failure modes of naive sentence splitters on non-Latin languages in RAG ingestion?", expanded=True):
    st.markdown("""
    **Answer:**
    The primary failure modes include:
    1. **Lack of Whitespace in CJK Languages**: In Chinese and Japanese, words are not separated by spaces. An English character splitter that defaults to splitting on `" "` will treat hundreds of characters as a single unbreakable token, causing chunks to overflow model token limits or split in the middle of ideographic characters.
    2. **Unrecognized Sentence Delimiters**: Non-Latin scripts use distinct full stops:
       * Chinese/Japanese ideographic period: `。` (U+3002)
       * Hindi/Sanskrit Purna Virama: `।` (U+0964)
       * Arabic full stop: `۔` (U+06D4)
       If the splitter only looks for ASCII `.`, sentences are never terminated, resulting in fragmented half-sentence chunks.
    3. **Subword Over-Tokenization in Agglutinative Languages**: German, Finnish, and Turkish combine morphemes into massive compound words. Standard English tokenizers fragment these into dozen of sub-byte tokens, inflating context window consumption.
    """)

with st.expander("Q2: How do you prevent cross-lingual hallucinations when generating answers in Language A from English context?"):
    st.markdown("""
    **Answer:**
    Cross-lingual hallucination occurs when an LLM, forced to translate while synthesizing, introduces foreign facts or misinterprets technical terms.
    To prevent this in production:
    * **Explicit Language Conditioning**: State the target language explicitly in the prompt: `Answer strictly in {detected_lang}`.
    * **Strict Context Bounding**: Instruct the LLM: *"Use ONLY the facts in the provided English passages. If the answer is not explicitly stated in the English context, state 'I do not have enough documentation to answer' in {detected_lang}."*
    * **Preserve Technical Proper Nouns**: Instruct the model: *"Keep all software package names, CLI commands, HTTP status codes, and configuration parameters in their original canonical Latin form (e.g., do not translate 'Kubernetes' or 'POST')."*
    * **Mandatory Citation Anchoring**: Require bracketed source citations `[Doc_ID]` after every factual assertion. Models forced to ground translated sentences with citation IDs exhibit 80%+ lower hallucination rates.
    """)

with st.expander("Q3: Why is a multilingual Cross-Encoder reranker critical in Multilingual RAG pipelines?"):
    st.markdown("""
    **Answer:**
    While modern multilingual Bi-Encoders (like BGE-M3 or Cohere) provide remarkable recall across languages, their vector representations inevitably suffer from minor **cross-lingual alignment noise** (e.g., cultural nuance, translation ambiguity, or script asymmetry). 
    A Bi-Encoder encodes the Hindi query and English document in complete isolation.
    
    A **Multilingual Cross-Encoder** (such as `BAAI/bge-reranker-v2-m3` or `Cohere Rerank 3 Multilingual`) concatenates the Hindi query and English document into a single transformer input sequence:
    `[CLS] Hindi Query [SEP] English Document [SEP]`
    The model computes full bidirectional self-attention between Hindi tokens and English tokens across all attention layers. 
    This allows the model to verify whether the English technical document actually answers the specific nuances of the Hindi query, effectively eliminating false-positive bi-encoder candidates before context injection.
    """)

with st.expander("Q4: Should documents in a multilingual enterprise corpus be translated upfront or indexed as-is?"):
    st.markdown("""
    **Answer:**
    In 95% of modern architectures, **documents should be indexed as-is in their original authored language** using a unified multilingual embedding model:
    * **Translation Loss & Inaccuracies**: Translating millions of documents upfront permanently bakes machine translation errors into your vector database.
    * **Maintenance Synchronization**: If a German contract is edited, you must remember to re-translate and re-index the English mirror. Storing the original German source guarantees a single source of truth.
    * **Cost**: Upfront translation of millions of enterprise documents incurs massive LLM/MT API expenses.
    * With zero-shot multilingual encoders and frontier LLMs, the system dynamically bridges language gaps at query time with zero upfront translation overhead.
    """)

st.subheader("Production Failure Traps")
st.error("""
* **Trap 1: Translating Code Snippets and Commands**: Prompting an LLM to generate an answer in Spanish and having it translate `kubectl get pods` into `kubectl obtener vainas`. Always add explicit prompt guardrails protecting code blocks and technical identifiers.
* **Trap 2: Ignoring Low-Resource Language Degradation**: Assuming that a multilingual model performs as well on low-resource languages (e.g., Swahili, Basque, Bengali) as it does on French or Spanish. Always benchmark MRR per locale; for low-resource languages, a Translate-then-RAG fallback pipeline is often necessary.
* **Trap 3: Using Monolingual Rerankers in Cross-Lingual Pipelines**: Feeding a Hindi query and an English document into `ms-marco-MiniLM-L-6-v2`. The English-only reranker cannot parse Devanagari tokens and returns nonsensical reranking scores.
""")

st.subheader("Interactive Knowledge Flashcards")
card_cols = st.columns(4)

with card_cols[0]:
    with st.popover("Card 1: Script-Aware Splitters"):
        st.markdown("**Front**: What separators must be used for CJK and Devanagari texts?")
        st.divider()
        st.markdown("**Back**: Include ideographic full stop `。` for Chinese/Japanese, and Purna Virama `।` for Hindi, before fallback space characters.")

with card_cols[1]:
    with st.popover("Card 2: Grounded Synthesis"):
        st.markdown("**Front**: How does Multilingual RAG answer a Hindi query from English docs?")
        st.divider()
        st.markdown("**Back**: The English context chunks are injected into the prompt, and the frontier LLM synthesizes a fluent Hindi response with grounded citations.")

with card_cols[2]:
    with st.popover("Card 3: Cross-Lingual Reranker"):
        st.markdown("**Front**: Why is a multilingual cross-encoder essential?")
        st.divider()
        st.markdown("**Back**: It allows bidirectional attention between query tokens in Language A and document tokens in Language B to filter out bi-encoder alignment noise.")

with card_cols[3]:
    with st.popover("Card 4: Upfront vs Live Translation"):
        st.markdown("**Front**: Should enterprise docs be translated into English before indexing?")
        st.divider()
        st.markdown("**Back**: No! Index documents as-is in their native languages to prevent translation error baking and avoid massive upfront translation costs.")

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("Check off each concept as you master Multilingual RAG systems:")

c1, c2 = st.columns(2)
with c1:
    st.checkbox("I understand the 4 stages of Multilingual RAG (Ingest, Retrieve, Rerank, Synthesize).", key="chk26_3_1")
    st.checkbox("I know how to configure script-aware text splitters for CJK and Devanagari.", key="chk26_3_2")
    st.checkbox("I can structure cross-lingual LLM prompts that prevent translation hallucination.", key="chk26_3_3")
with c2:
    st.checkbox("I understand why multilingual cross-encoders outperform bi-encoders on precision.", key="chk26_3_4")
    st.checkbox("I know why indexing original documents is superior to upfront translation.", key="chk26_3_5")
    st.checkbox("I can guardrail technical terms and CLI commands from accidental translation.", key="chk26_3_6")
