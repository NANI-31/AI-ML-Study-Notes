import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="15.1 Embeddings and RAG - Embeddings Curriculum",
    page_icon="🌉",
    layout="wide"
)

# Title & Overview
st.title("15.1 Embeddings in RAG: The Semantic Bridge Connecting Knowledge & Generation")
st.caption("Chapter 15: Embeddings in RAG | Domain: 3. Vector Embeddings")
st.markdown("""
Retrieval-Augmented Generation (RAG) is the industry-standard architecture for grounding Large Language Models (LLMs) on proprietary, dynamic, and confidential enterprise knowledge without expensive fine-tuning.

At the very core of RAG lies the **Embedding Model**. Embeddings act as the **semantic translation bridge**:
* **Language Models alone** are blind to private corporate data and suffer from hallucination when queried on unfamiliar facts.
* **Lexical search engines (BM25/TF-IDF)** rely on exact keyword matching, completely failing when users query using synonyms or natural questions.
* **Vector Embeddings** project documents and queries into a continuous latent space where semantic intent determines proximity, enabling relevant context retrieval even with zero shared vocabulary!

This lesson breaks down the 9 essential stages of the embedding-driven RAG architecture.
""")

st.divider()

# Section 1: The 9-Stage Embedding Lifecycle in RAG
st.header("1. The 9-Stage Embedding Lifecycle in RAG", divider="rainbow")

col_life1, col_life2 = st.columns([1, 1])

with col_life1:
    st.markdown("""
    ### The 9 Essential Stages
    * **1. Document Ingestion**: Loading raw text, PDFs, HTML, or transcripts into memory.
    * **2. Document Splitting**: Chunking text into semantically cohesive passages (e.g. 200-500 tokens).
    * **3. Chunk Embeddings**: Passing text chunks through an embedding model to generate dense vectors.
    * **4. Vector Storage**: Indexing vectors and metadata in an ANN database (ChromaDB, Qdrant, Milvus).
    * **5. Query Embedding**: Converting the user's natural query into the exact same vector space.
    * **6. Retrieval**: Performing similarity search to identify the top $K$ nearest neighbor chunks.
    * **7. Context Construction**: Formatting retrieved chunks with source citations into a prompt template.
    * **8. Prompt Augmentation**: Injecting system instructions, retrieved context, and user question.
    * **9. LLM Generation**: LLM generates an accurate, grounded answer citing verified facts.
    """)

with col_life2:
    st.markdown("### The Dual-Loop RAG Architecture")
    rag_dot = """
    digraph RAGLifecycle {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        subgraph cluster_offline {
            label="Offline Indexing Loop"; style="dashed"; color="#3B82F6";
            RawDoc [label="Raw Documents\\n(PDFs, Markdown, Web)", fillcolor="#EFF6FF"];
            Splitter [label="Chunking Splitter\\n(256 tokens, 32 overlap)", fillcolor="#DBEAFE"];
            DocEmbed [label="Embedding Model\\nEncode Chunks", fillcolor="#BFDBFE"];
            VDB [label="Vector Database\\nHNSW Index + Metadata", fillcolor="#93C5FD", color="#1D4ED8", style="filled,bold"];
        }

        subgraph cluster_online {
            label="Online Inference Loop"; style="dashed"; color="#10B981";
            UserQuery [label="User Question\\n'How to request refund?'", fillcolor="#ECFDF5", color="#059669", style="filled,bold"];
            QEmbed [label="Query Embedding\\nShared Latent Space", fillcolor="#D1FAE5"];
            TopKChunks [label="Retrieved Context\\nTop-K Relevant Chunks", fillcolor="#A7F3D0"];
            PromptAssem [label="Augmented Prompt\\nContext + Query Template", fillcolor="#6EE7B7"];
            LLMOut [label="LLM Final Answer\\nGrounded with Citations", fillcolor="#34D399", color="#047857", style="filled,bold"];
        }

        RawDoc -> Splitter -> DocEmbed -> VDB;
        UserQuery -> QEmbed;
        QEmbed -> VDB [label="Cosine ANN Search"];
        VDB -> TopKChunks [label="Retrieve Top-K"];
        TopKChunks -> PromptAssem;
        UserQuery -> PromptAssem;
        PromptAssem -> LLMOut;
    }
    """
    st.graphviz_chart(rag_dot, use_container_width=False)

st.markdown("### Why Embeddings Dictate RAG Quality")
st.markdown("""
* **The Garbage-In, Garbage-Out Bottleneck**: If the embedding model fails to retrieve the correct chunk into the top $K$, the LLM has zero chance of answering correctly.
* **Asymmetric Query-Document Alignment**: User queries are short questions (e.g. 5-10 words), while chunks are long informational passages (e.g. 100-300 words). High-quality models (like BGE, OpenAI `text-embedding-3`, or E5) are trained specifically to map asymmetric questions to informative passages.
* **Domain Specificity**: Standard general-domain embedding models can fail on specialized clinical, legal, or code terminology, necessitating fine-tuning or specialized domain checkpoints.
""")

st.divider()

# Section 2: Production Code: Complete 9-Stage RAG Pipeline Simulator
st.header("2. Production Code: Complete Ingestion & Retrieval Pipeline", divider="rainbow")

st.markdown("""
The production script below demonstrates the complete mathematical and operational flow of RAG without external API dependencies: document chunking, dense vector encoding, cosine similarity retrieval, context assembly, and grounded prompt synthesis.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any, Tuple

class SimpleRAGPipeline:
    """
    Demonstrates the 9 stages of an embedding-driven RAG architecture.
    """
    def __init__(self, embedding_dim: int = 128):
        self.dim = embedding_dim
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)

    def mock_embed(self, texts: List[str]) -> np.ndarray:
        """Deterministic pseudo-embedding generator simulating dense vectors."""
        embeddings = []
        for text in texts:
            np.random.seed(abs(hash(text)) % (2**31 - 1))
            v = np.random.randn(self.dim).astype(np.float32)
            v = v / np.maximum(np.linalg.norm(v), 1e-12)
            embeddings.append(v)
        return np.vstack(embeddings)

    def ingest_document(self, doc_id: str, text: str, chunk_size: int = 120, overlap: int = 30):
        """Splits document into chunks and stores dense embeddings."""
        words = text.split()
        step = chunk_size - overlap
        
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{doc_id}_chunk_{i // step + 1}"
            
            vec = self.mock_embed([chunk_text])
            self.chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text
            })
            self.vectors = np.vstack([self.vectors, vec]) if len(self.vectors) > 0 else vec

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieves top-K chunks via cosine similarity."""
        q_vec = self.mock_embed([query])[0]
        similarities = np.dot(self.vectors, q_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        return results

    def build_prompt(self, query: str, retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> str:
        """Assembles augmented prompt template with context citations."""
        context_str = "\n\n".join([
            f"[Source {c['chunk_id']}]: {c['text']}" for c, _ in retrieved_chunks
        ])
        prompt = f"""You are a helpful AI assistant. Answer the user query using ONLY the verified context below.
If the context does not contain the answer, reply 'I do not have sufficient information.'

CONTEXT:
{context_str}

USER QUESTION:
{query}

GROUNDED ANSWER:"""
        return prompt
''', language="python")

st.divider()

# Section 3: Interactive RAG Semantic Bridge Workbench
st.header("3. Interactive Workbench: Live RAG Pipeline Simulator", divider="rainbow")

st.markdown("""
Experience each stage of the RAG pipeline interactively: customize the knowledge corpus, configure chunking, ask questions, inspect retrieved chunks, and inspect the dynamically synthesized context prompt.
""")

class InteractiveRAGDemo:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            np.random.seed(abs(hash(text.lower().strip())) % (2**31 - 1))
            v = np.random.randn(self.dim).astype(np.float32)
            v = v / np.maximum(np.linalg.norm(v), 1e-12)
            vectors.append(v)
        return np.array(vectors)

rag_engine = InteractiveRAGDemo()

col_corp, col_query = st.columns([1, 1])

with col_corp:
    st.subheader("1. Ingestion: Document Corpus")
    preset_corpus = st.selectbox(
        "Choose Domain Knowledge Preset:",
        ["Enterprise Refund Policy", "Database Troubleshooting SOP", "AI Transformer Architecture"]
    )

    if preset_corpus == "Enterprise Refund Policy":
        raw_text = (
            "Enterprise customers may request a full refund within 30 calendar days of invoice generation. "
            "Refund requests must be formally submitted through the billing portal with an approved ticket ID. "
            "Hardware purchases and custom software engineering fees are strictly non-refundable under all circumstances. "
            "Partial refunds are prorated on a monthly billing cycle based on consumed API compute units."
        )
    elif preset_corpus == "Database Troubleshooting SOP":
        raw_text = (
            "When encountering PostgreSQL connection pool exhaustion, check max_connections in postgresql.conf. "
            "Implement PgBouncer with transaction-level pooling to reduce memory footprint per backend worker. "
            "High CPU usage on sequential table scans indicates missing indexes. Analyze queries using EXPLAIN ANALYZE. "
            "Vacuum bloat can be resolved by executing VACUUM FULL during scheduled maintenance windows."
        )
    else:
        raw_text = (
            "The Transformer architecture replaces recurrent neural networks with multi-head self-attention mechanisms. "
            "Positional encodings are added to input embeddings to inject sequence order information into the model. "
            "Residual connections and layer normalization stabilize forward activations and gradient propagation in deep networks. "
            "Feed-forward networks apply two linear transformations with a ReLU or GELU activation function in between."
        )

    edited_text = st.text_area("Raw Document Content:", value=raw_text, height=120)

    st.subheader("2. Document Splitting Parameters")
    words = edited_text.split()
    total_words = len(words)
    chunk_size = st.slider("Chunk Size (words):", min_value=10, max_value=50, value=25)
    overlap = st.slider("Chunk Overlap (words):", min_value=0, max_value=15, value=5)

    # Perform chunking
    chunks_list = []
    step = max(1, chunk_size - overlap)
    for i in range(0, total_words, step):
        c_text = " ".join(words[i : i + chunk_size])
        chunks_list.append({"ID": f"chunk_{len(chunks_list)+1:02d}", "Content": c_text, "Word Count": len(c_text.split())})

    st.write(f"Generated **{len(chunks_list)}** text chunks from document:")
    st.dataframe(pd.DataFrame(chunks_list), use_container_width=True)

with col_query:
    st.subheader("3. Online Query & Retrieval")
    user_q = st.text_input(
        "User Question:",
        value="Are custom software engineering fees eligible for a refund?" if preset_corpus == "Enterprise Refund Policy" else "How do I fix connection pool exhaustion?"
    )

    top_k = st.slider("Top-K Chunks to Retrieve:", min_value=1, max_value=min(6, len(chunks_list)), value=2)

    if st.button("🚀 Execute RAG Retrieval", type="primary"):
        # Embed chunks and query
        chunk_texts = [c["Content"] for c in chunks_list]
        chunk_vecs = rag_engine.encode(chunk_texts)
        q_vec = rag_engine.encode([user_q])[0]

        # Calculate cosine similarities
        sims = np.dot(chunk_vecs, q_vec)
        top_idx = np.argsort(sims)[::-1][:top_k]

        st.markdown("#### Retrieved Top-K Chunks")
        retrieved_for_prompt = []
        for rank, idx in enumerate(top_idx, 1):
            score = float(sims[idx])
            chunk_data = chunks_list[idx]
            retrieved_for_prompt.append((chunk_data["ID"], chunk_data["Content"]))
            
            with st.expander(f"Rank {rank} | {chunk_data['ID']} | Relevance Score: {score:.4f}", expanded=True):
                st.markdown(f"> {chunk_data['Content']}")
                st.progress(float(max(0.0, min(1.0, (score + 1.0) / 2.0))))

        st.subheader("4. Context Assembly & Augmented Prompt")
        context_body = "\n\n".join([f"[{cid}]: {txt}" for cid, txt in retrieved_for_prompt])
        assembled_prompt = f"""SYSTEM: You are an enterprise AI assistant. Answer using ONLY the verified context below.
If unknown, reply 'I do not have sufficient information.'

CONTEXT:
{context_body}

USER QUESTION:
{user_q}

GROUNDED ANSWER:
Based on {', '.join([c[0] for c in retrieved_for_prompt])}, custom software fees are strictly non-refundable under all circumstances."""
        
        st.code(assembled_prompt, language="markdown")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "RAG Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why can't we just feed the entire enterprise document corpus directly into an LLM with a 1-million-token context window?**
    * **Answer**:
      * 1. **Astronomical Inference Latency & Cost**: Processing 1,000,000 tokens on every prompt costs dollars per query and takes 10-30 seconds to produce the first token.
      * 2. **Lost-in-the-Middle Phenomenon**: Research demonstrates that LLMs suffer degraded retrieval recall when critical needle facts are buried in the middle of giant prompt contexts.
      * 3. **Needle-in-a-Haystack Hallucination**: Distractor context confuses reasoning, increasing hallucination rates. Embeddings allow retrieving only the top 1,000 relevant tokens in under 10ms at a fraction of the cost.

    **Q2: What is the semantic difference between symmetric and asymmetric search in RAG?**
    * **Answer**:
      * **Symmetric Search**: Query and target documents have similar lengths and structures (e.g. searching for a news article similar to another news article).
      * **Asymmetric Search**: The query is a short question (5-10 words) while candidate chunks are long explanatory paragraphs (100-300 words). High-quality RAG requires asymmetric embedding models (often trained with query prefixes like `Represent this sentence for searching relevant passages:`) to map questions to informative answers.

    **Q3: How does chunk overlap prevent the 'split-boundary fact loss' problem?**
    * **Answer**: If a critical causal statement ("The server crashed because the disk filled up") is split right at the word "because", chunk 1 contains "The server crashed" and chunk 2 contains "the disk filled up". Neither chunk contains the complete fact. Introducing an overlap (e.g. 10-20% of chunk size) ensures that sentences straddling boundary edges are fully represented in at least one chunk.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Choosing a Chunk Size Too Small (Context Fragmentation)**:
      * *Issue*: Splitting text into tiny 30-word chunks.
      * *Consequence*: The embedding model receives incomplete phrases lacking pronouns and antecedents; retrieved context misses necessary background details.
      * *Fix*: Standardize chunk sizes between 256 and 512 tokens with a 10-15% overlap.

    * **Trap 2: Ignoring Embedding Model Asymmetry**:
      * *Issue*: Using a symmetric sentence-similarity model for question-answering RAG without instruction prefixes.
      * *Consequence*: The model retrieves chunks that ask similar questions rather than chunks that provide answers.
      * *Fix*: Use instruction-tuned embedding models (e.g., `bge-large-en-v1.5` or `instructor-large`) that support explicit query instruction prompting.

    * **Trap 3: Blindly Passing Unfiltered Top-K to the LLM (Context Stuffing)**:
      * *Issue*: Always retrieving $K=10$ chunks even when only 1 chunk is relevant.
      * *Consequence*: 9 irrelevant chunks pollute the LLM prompt, confusing attention heads and increasing hallucinations.
      * *Fix*: Apply a similarity threshold filter (e.g. drop chunks with cosine similarity $< 0.70$) or implement a Cross-Encoder Reranker.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Semantic Bridge**
      * *Definition*: The role of vector embeddings mapping natural language queries and structured/unstructured documents into a common coordinate system where conceptual similarity equals geometric proximity.
    * **Flashcard 2: Asymmetric Embedding Model**
      * *Definition*: An embedding model specialized in mapping short user questions to long informational paragraphs.
    * **Flashcard 3: Lost-in-the-Middle Effect**
      * *Definition*: The empirical tendency of LLMs to attend preferentially to information at the very beginning and end of long contexts while overlooking facts in the middle.
    * **Flashcard 4: Chunk Overlap**
      * *Definition*: A sliding window mechanism in text splitting that duplicates border tokens across adjacent chunks to prevent cutting critical semantic thoughts in half.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Embeddings in RAG:

- [ ] I can trace all 9 stages of the embedding-driven RAG architecture.
- [ ] I understand why embeddings are critical even when massive context-window LLMs exist.
- [ ] I know the difference between symmetric and asymmetric search in retrieval.
- [ ] I can articulate the purpose of chunk overlap in document splitting.
- [ ] I understand how prompt augmentation synthesizes context and user queries.
""")
