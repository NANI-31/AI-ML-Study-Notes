import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="19.2 Advanced RAG Architecture",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class AdvancedRAGEngine:
    """Enterprise Multi-Stage Advanced RAG Engine: Rewriting -> Hybrid -> Rerank -> Compress -> Grounded Gen."""

    def __init__(self, corpus: Optional[List[Dict[str, str]]] = None):
        if corpus is None:
            self.corpus = [
                {
                    "id": "doc_101",
                    "title": "PCI-DSS 4.0 Key Management Requirements",
                    "content": "Requirement 3.5.1 states that symmetric cryptographic keys protecting stored cardholder data must use AES-256 or ChaCha20-Poly1305. Cryptographic keys must be rotated at least once every 12 months, or immediately if compromise is suspected. Split knowledge and dual control must be enforced for manual cleartext key operations."
                },
                {
                    "id": "doc_102",
                    "title": "Hardware Security Module (HSM) Deployment Guide",
                    "content": "FIPS 140-3 Level 3 certified HSMs must be deployed in multi-region clusters with synchronous state replication across dual physical availability zones. Network interfaces must terminate in isolated management VPCs with mTLS authentication. Manual key rotation is strictly prohibited in production clusters."
                },
                {
                    "id": "doc_103",
                    "title": "Incident Response & Key Revocation Protocols",
                    "content": "Upon confirmation of root key compromise, emergency key revocation procedures must execute within 15 minutes. All access tokens signed by the compromised key are invalidated globally via an in-memory Redis blacklist. The backup disaster recovery key vault is promoted synchronously."
                },
                {
                    "id": "doc_104",
                    "title": "Audit Logging & SIEM Event Streaming",
                    "content": "Cryptographic operations including key generation, encryption, decryption, and export attempts must generate immutable audit events streamed to AWS CloudTrail and Splunk. Logs must record key ID, calling IAM role, and timestamp, but never key material."
                },
                {
                    "id": "doc_105",
                    "title": "Employee Cloud Access Onboarding",
                    "content": "All engineering employees must complete annual security awareness training. Multi-factor authentication with hardware security keys (FIDO2/WebAuthn) is mandatory for corporate SSO access. Remote developers must route through wireguard VPN gateways."
                }
            ]
        else:
            self.corpus = corpus

    def rewrite_query(self, raw_query: str, mode: str = "HyDE") -> Dict[str, Any]:
        """Perform query transformation: HyDE synthetic expansion or sub-query decomposition."""
        if mode == "HyDE":
            # Hypothetical Document Embeddings expansion
            hypothetical = (
                f"In enterprise infrastructure compliance, {raw_query}. "
                "Standards specify cryptographic algorithms, rotation cadences, dual-control procedures, "
                "and HSM certified hardware boundaries."
            )
            expanded_query = f"{raw_query} {hypothetical}"
            description = "Synthesized hypothetical technical document passage to bridge semantic vocabulary gap."
        elif mode == "Sub-Query":
            sub_queries = [
                f"{raw_query} regulatory algorithms and requirements",
                f"{raw_query} rotation timeframes and emergency protocols",
                f"{raw_query} HSM hardware and enforcement constraints"
            ]
            expanded_query = " ".join(sub_queries)
            description = f"Decomposed into 3 targeted sub-queries: {sub_queries}"
        else:
            expanded_query = raw_query
            description = "Passed raw query without transformation."

        return {
            "raw_query": raw_query,
            "expanded_query": expanded_query,
            "mode": mode,
            "description": description
        }

    def _dense_score(self, query: str, text: str) -> float:
        """Mock dense semantic similarity using character trigram overlap."""
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        inter = len(q_words.intersection(t_words))
        return float(inter / (len(q_words) + 1e-5))

    def _sparse_score(self, query: str, text: str) -> float:
        """Mock BM25 lexical keyword matching score."""
        q_tokens = query.lower().split()
        t_text = text.lower()
        score = 0.0
        for token in q_tokens:
            if len(token) > 3 and token in t_text:
                score += 1.5
        return float(score)

    def hybrid_retrieve(self, query_obj: Dict[str, Any], top_k: int = 4) -> List[Dict[str, Any]]:
        """Dense + Sparse Retrieval fused using Reciprocal Rank Fusion (RRF)."""
        q_text = query_obj["expanded_query"]
        
        # Dense retrieval
        dense_candidates = []
        for doc in self.corpus:
            d_score = self._dense_score(q_text, doc["title"] + " " + doc["content"])
            dense_candidates.append((doc, d_score))
        dense_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Sparse retrieval
        sparse_candidates = []
        for doc in self.corpus:
            s_score = self._sparse_score(q_text, doc["title"] + " " + doc["content"])
            sparse_candidates.append((doc, s_score))
        sparse_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Reciprocal Rank Fusion (RRF constant k=60)
        rrf_scores: Dict[str, float] = {}
        doc_map = {doc["id"]: doc for doc in self.corpus}
        
        for rank, (doc, _) in enumerate(dense_candidates, start=1):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (1.0 / (60.0 + rank))
            
        for rank, (doc, _) in enumerate(sparse_candidates, start=1):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (1.0 / (60.0 + rank))
            
        fused = []
        for doc_id, score in rrf_scores.items():
            doc = doc_map[doc_id]
            fused.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "rrf_score": round(score, 5)
            })
            
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused[:top_k]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], relevance_cutoff: float = 0.018) -> List[Dict[str, Any]]:
        """Simulate cross-encoder deep contextual reranking and relevance thresholding."""
        reranked = []
        for cand in candidates:
            # Cross-encoder evaluates query and document joint attention
            joint_text = (query + " " + cand["title"] + " " + cand["content"]).lower()
            key_matches = sum(1 for kw in ["key", "rotation", "pci-dss", "hsm", "revoke", "aes", "months"] if kw in joint_text)
            cross_score = cand["rrf_score"] * 10.0 + (key_matches * 0.15)
            
            if cross_score >= relevance_cutoff:
                cand_copy = dict(cand)
                cand_copy["cross_encoder_score"] = round(cross_score, 4)
                reranked.append(cand_copy)
                
        reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return reranked

    def compress_context(self, reranked_docs: List[Dict[str, Any]], max_sentences: int = 2) -> List[Dict[str, Any]]:
        """Selective extractive context compression: prunes tangential sentences to protect token budget."""
        compressed = []
        for doc in reranked_docs:
            sentences = [s.strip() for s in doc["content"].split(".") if len(s.strip()) > 10]
            # Keep only the top relevant sentences
            selected = sentences[:max_sentences]
            compressed_content = ". ".join(selected) + "."
            compressed.append({
                "id": doc["id"],
                "title": doc["title"],
                "original_tokens": len(doc["content"].split()),
                "compressed_tokens": len(compressed_content.split()),
                "compressed_content": compressed_content,
                "score": doc.get("cross_encoder_score", doc.get("rrf_score", 0.0))
            })
        return compressed

    def grounded_generate(self, raw_query: str, compressed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Closed-book epistemic contract: strict citation anchoring and guaranteed abstention."""
        if not compressed_docs:
            return {
                "answer": "I cannot answer this question. The verified knowledge base contains no documents meeting relevance thresholds.",
                "citations": [],
                "grounded": True,
                "token_savings_pct": 0.0
            }

        citations = []
        answer_parts = []
        for i, doc in enumerate(compressed_docs, start=1):
            citations.append(f"[{doc['title']} ({doc['id']})]")
            answer_parts.append(f"Under {doc['title']}, {doc['compressed_content']} [{doc['id']}]")

        total_orig = sum(d["original_tokens"] for d in compressed_docs)
        total_comp = sum(d["compressed_tokens"] for d in compressed_docs)
        savings = round(((total_orig - total_comp) / total_orig) * 100.0, 1) if total_orig > 0 else 0.0

        final_answer = (
            f"Based strictly on verified documentation:\n"
            + "\n".join(answer_parts) +
            f"\n\nAll cryptographic requirements comply with documented governance."
        )

        return {
            "answer": final_answer,
            "citations": citations,
            "grounded": True,
            "token_savings_pct": savings,
            "total_tokens_used": total_comp
        }


# --- Streamlit Layout ---

st.title("19.2 Advanced RAG Architecture")
st.caption("The Five Pillars: Query Rewriting, Hybrid RRF Retrieval, Cross-Encoder Reranking, Context Compression & Grounded Synthesis")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Enterprise Multi-Stage Advanced RAG Architecture", divider="rainbow")

st.markdown(
    """
    **Advanced RAG** resolves the brittle failure modes of Basic RAG by transforming the linear pipeline into a
    **multi-stage, orchestrated pipeline**. Rather than directly injecting raw queries and static nearest neighbors into an LLM,
    Advanced RAG adds pre-retrieval optimization (Query Rewriting), multi-representation search (Hybrid Lexical + Dense),
    post-retrieval scoring (Cross-Encoder Reranking), context budget optimization (Extractive Compression), and epistemic grounding.
    
    The diagram below models the complete **enterprise multi-stage flow** and demonstrates how each pillar filters and refines information.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Advanced_RAG_Architecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_pre_retrieval {
            label = "STAGE 1: PRE-RETRIEVAL (QUERY REWRITING & EXPANSION)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            raw_query [label="Raw User Query\\n'Key rotation rules?'", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            query_rewriter [label="Query Transformation Engine\\n- HyDE (Hypothetical Embeddings)\\n- Sub-Query Decomposition\\n- Step-Back Prompting", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            rewritten_query [label="Expanded Multi-Facet Query", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            raw_query -> query_rewriter -> rewritten_query;
        }

        subgraph cluster_retrieval {
            label = "STAGE 2: RETRIEVAL (HYBRID DENSE + SPARSE FUSION)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            dense_search [label="Dense Vector Search\\n(HNSW Cosine Similarity / BGE / OpenAI)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            sparse_search [label="Sparse Lexical Search\\n(BM25 / SPLADE Exact Token Matching)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            rrf_fusion [label="Reciprocal Rank Fusion (RRF)\\nRRF(d) = Sum(1 / (60 + Rank_i))", fillcolor="#312E81", fontcolor="#A5B4FC"];
            top_candidates [label="Top-20 Fused Candidates Pool", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            
            rewritten_query -> dense_search;
            rewritten_query -> sparse_search;
            dense_search -> rrf_fusion [label="Dense Ranks"];
            sparse_search -> rrf_fusion [label="BM25 Ranks"];
            rrf_fusion -> top_candidates;
        }

        subgraph cluster_post_retrieval {
            label = "STAGE 3: POST-RETRIEVAL (RERANKING & CONTEXT COMPRESSION)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            cross_encoder [label="Cross-Encoder Reranker\\n(Cohere Rerank / BGE-Reranker-Large)\\nFull Token-to-Token Cross-Attention", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            threshold_filter [label="Relevance Threshold Gate\\n(Discard scores < Cutoff)", fillcolor="#78350F", fontcolor="#FDE68A"];
            context_compressor [label="Context Compressor (LLMLingua)\\n- Extractive Sentence Pruning\\n- Header & Boilerplate Elimination", fillcolor="#0F172A", fontcolor="#10B981"];
            lean_context [label="Compressed High-Density Context\\n(Top 3 Clean Chunks, 60% Token Savings)", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            top_candidates -> cross_encoder;
            cross_encoder -> threshold_filter;
            threshold_filter -> context_compressor [label="Pass Valid Chunks"];
            context_compressor -> lean_context;
        }

        subgraph cluster_generation {
            label = "STAGE 4: GROUNDED GENERATION (EPISTEMIC CONTRACT & CITATION)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            grounded_prompt [label="Strict Epistemic Prompt\\n- In-line Citation Mandate\\n- Explicit Abstention Protocol", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            frontier_llm [label="Frontier LLM Synthesis\\n(GPT-4o / Claude 3.5 Sonnet)", fillcolor="#0F172A", fontcolor="#10B981"];
            verified_response [label="Verified Factual Response\\nWith Exact Chunk Attribution [Doc 101]", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            lean_context -> grounded_prompt;
            raw_query -> grounded_prompt [label="Inject Original Query"];
            grounded_prompt -> frontier_llm -> verified_response;
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: QUERY REWRITING ---
st.header("2. Query Rewriting (HyDE, Sub-Query Decomposition, Step-Back Prompting & Disambiguation)", divider="rainbow")

st.markdown(
    """
    User queries in enterprise systems are notoriously ambiguous, underspecified, or mismatched with the formal vocabulary of reference documents.
    **Query Rewriting** bridges this vocabulary and conceptual divide before the retriever is ever queried.
    
    ### The Core Pre-Retrieval Rewriting Strategies
    """
)

col_qr1, col_qr2, col_qr3 = st.columns(3)
with col_qr1:
    with st.container(border=True):
        st.markdown("#### 1. HyDE (Hypothetical Document Embeddings)")
        st.markdown(
            """
            - **Method**: Prompts an LLM to hallucinate a synthetic, ideal answer passage to the user's question without external data.
            - **Advantage**: The embedding of the hypothetical answer shares higher cosine proximity with actual document chunks than a brief 5-word question.
            - **Application**: Complex conceptual questions and domain-specific knowledge bases.
            """
        )
with col_qr2:
    with st.container(border=True):
        st.markdown("#### 2. Sub-Query Decomposition")
        st.markdown(
            """
            - **Method**: Breaks multi-part comparative queries (*"Compare AWS KMS and HashiCorp Vault key rotation"*) into distinct sub-queries.
            - **Advantage**: Executes targeted parallel retrievals across different document sets, preventing single-vector averaging.
            - **Application**: Comparative analysis and multi-hop fact retrieval.
            """
        )
with col_qr3:
    with st.container(border=True):
        st.markdown("#### 3. Step-Back Prompting")
        st.markdown(
            """
            - **Method**: Abstracts the specific query into a broader, fundamental conceptual question (*"What are the foundational principles of encryption key lifecycle management?"*).
            - **Advantage**: Retrieves overarching principles and context that explain specific implementation rules.
            - **Application**: Complex engineering debugging and legal interpretations.
            """
        )

# --- SECTION 3: HYBRID RETRIEVAL ---
st.header("3. Hybrid Retrieval (Dense Vector Embedding + Sparse BM25 Lexical Fusion via RRF)", divider="rainbow")

st.markdown(
    """
    Neither dense vector embeddings nor sparse keyword algorithms are sufficient on their own:
    - **Dense Embeddings** excel at conceptual semantics and synonyms (*"key renewal"* matches *"cryptographic rotation"*), but fail on exact part numbers, error codes, and unique identifiers (*"CVE-2026-1182"*, *"PCI-DSS 3.5.1"*).
    - **Sparse BM25 Search** excels at exact token and keyword matches, but fails when users use synonyms or colloquial phrasing.
    
    **Hybrid Retrieval** executes both algorithms concurrently and merges their candidate rankings using **Reciprocal Rank Fusion (RRF)**:
    """
)

st.code(
    """RRF_Score(doc) = Sum( 1.0 / (k + rank_i(doc)) )  for each retriever i, where k is typically 60""",
    language="text"
)

st.markdown(
    """
    RRF is rank-based rather than score-based, completely eliminating the need to calibrate or normalize disparate score distributions between vector cosine similarities and BM25 BM-scores.
    """
)

# --- SECTION 4: RERANKING ---
st.header("4. Reranking (Cross-Encoder Neural Scoring, Listwise Attention & Relevance Thresholding)", divider="rainbow")

st.markdown(
    """
    While bi-encoder vector retrieval can rapidly scan millions of documents in milliseconds, it computes similarity from static vectors without considering the query-document token interactions.
    **Reranking** passes the Top-20 candidates from hybrid retrieval into a heavy **Cross-Encoder Transformer** (e.g., `Cohere Rerank v3`, `BGE-Reranker-Large`).
    
    ### Bi-Encoder vs Cross-Encoder Architecture
    """
)

col_rr1, col_rr2 = st.columns(2)
with col_rr1:
    st.markdown(
        """
        - **Bi-Encoder**:  
          `Score = e(Query) . e(Document)`  
          Fast (`O(1)` vector lookup), but token-level cross-attention is impossible.
        - **Cross-Encoder**:  
          `Score = Transformer([CLS] + Query + [SEP] + Document)`  
          Computes full all-to-all attention between every token in the query and every token in the document.
        - **Relevance Gating**: Chunks whose cross-encoder score falls below a strict threshold (e.g., `0.30`) are eliminated, preventing noisy context from reaching the LLM.
        """
    )
with col_rr2:
    rr_matrix = pd.DataFrame({
        "Attribute": ["Computation Latency", "Context Window Attention", "Candidate Pool Size", "Fine-Grained Precision"],
        "Bi-Encoder Retriever": ["5ms - 25ms", "Independent (No joint attention)", "Millions of chunks", "Moderate (High Recall)"],
        "Cross-Encoder Reranker": ["80ms - 250ms", "All-to-all cross-attention", "Top 10 - Top 50 chunks", "Exceptional (High Precision)"]
    })
    st.dataframe(rr_matrix, use_container_width=True)

# --- SECTION 5: CONTEXT COMPRESSION ---
st.header("5. Context Compression (Selective Extractive Pruning, LLM-Lingua Token Trimming & Budget Enforcement)", divider="rainbow")

st.markdown(
    """
    Even after reranking prunes the candidate pool to the top 3-4 chunks, documents still contain substantial **boilerplate tokens**:
    introductory sentences, legal disclaimers, repeated headers, and tangential clauses.
    
    ### Engineering Benefits of Compression:
    1. **Elimination of Attention Distraction**: By removing non-relevant sentences, the LLM focuses its entire self-attention mechanism on the exact factual clauses.
    2. **Mitigation of Lost-in-the-Middle**: Shorter, compressed prompts keep all crucial evidence within the primacy/recency attention sweet spots.
    3. **Token Economics & TTFT Reduction**: Pruning context by 40% - 60% directly slashes Time To First Token (TTFT) and cuts prompt token expenditures by half.
    """
)

# --- SECTION 6: GROUNDED GENERATION ---
st.header("6. Grounded Generation (Closed-Book Epistemic Contracts, Strict Citation Anchoring & Abstention Guarantees)", divider="rainbow")

st.markdown(
    """
    The final stage of Advanced RAG enforces a **Closed-Book Epistemic Contract** on the generator.
    Rather than allowing the LLM to freely synthesize parametric knowledge, the system prompt strictly constrains the model:
    
    - **Citation Anchoring**: Every factual claim must include an immediate in-line citation pointing to a validated chunk identifier (`[Doc 101]`).
    - **Zero-Speculation Protocol**: If the compressed context lacks sufficient evidence to answer the question, the LLM is instructed to explicitly state its inability to answer rather than extrapolating.
    - **Negative Constraint Guardrails**: System prompts explicitly forbid the model from using external pre-trained assumptions when analyzing proprietary internal policies.
    """
)

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Enterprise Advanced RAG Orchestrator (`AdvancedRAGEngine`)", divider="rainbow")

st.markdown(
    """
    Below is the complete, self-contained reference implementation of `AdvancedRAGEngine`.
    It orchestrates all five advanced pillars: HyDE/sub-query rewriting, hybrid dense/sparse retrieval with RRF,
    cross-encoder scoring, extractive sentence compression, and citation-anchored grounded generation.
    """
)

st.code(
    '''
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class AdvancedRAGEngine:
    """Enterprise Multi-Stage Advanced RAG Engine: Rewriting -> Hybrid -> Rerank -> Compress -> Grounded Gen."""

    def __init__(self, corpus: Optional[List[Dict[str, str]]] = None):
        if corpus is None:
            self.corpus = [
                {
                    "id": "doc_101",
                    "title": "PCI-DSS 4.0 Key Management Requirements",
                    "content": "Requirement 3.5.1 states that symmetric cryptographic keys protecting stored cardholder data must use AES-256 or ChaCha20-Poly1305. Cryptographic keys must be rotated at least once every 12 months, or immediately if compromise is suspected. Split knowledge and dual control must be enforced for manual cleartext key operations."
                },
                {
                    "id": "doc_102",
                    "title": "Hardware Security Module (HSM) Deployment Guide",
                    "content": "FIPS 140-3 Level 3 certified HSMs must be deployed in multi-region clusters with synchronous state replication across dual physical availability zones. Network interfaces must terminate in isolated management VPCs with mTLS authentication. Manual key rotation is strictly prohibited in production clusters."
                },
                {
                    "id": "doc_103",
                    "title": "Incident Response & Key Revocation Protocols",
                    "content": "Upon confirmation of root key compromise, emergency key revocation procedures must execute within 15 minutes. All access tokens signed by the compromised key are invalidated globally via an in-memory Redis blacklist. The backup disaster recovery key vault is promoted synchronously."
                },
                {
                    "id": "doc_104",
                    "title": "Audit Logging & SIEM Event Streaming",
                    "content": "Cryptographic operations including key generation, encryption, decryption, and export attempts must generate immutable audit events streamed to AWS CloudTrail and Splunk. Logs must record key ID, calling IAM role, and timestamp, but never key material."
                },
                {
                    "id": "doc_105",
                    "title": "Employee Cloud Access Onboarding",
                    "content": "All engineering employees must complete annual security awareness training. Multi-factor authentication with hardware security keys (FIDO2/WebAuthn) is mandatory for corporate SSO access. Remote developers must route through wireguard VPN gateways."
                }
            ]
        else:
            self.corpus = corpus

    def rewrite_query(self, raw_query: str, mode: str = "HyDE") -> Dict[str, Any]:
        """Perform query transformation: HyDE synthetic expansion or sub-query decomposition."""
        if mode == "HyDE":
            hypothetical = (
                f"In enterprise infrastructure compliance, {raw_query}. "
                "Standards specify cryptographic algorithms, rotation cadences, dual-control procedures, "
                "and HSM certified hardware boundaries."
            )
            expanded_query = f"{raw_query} {hypothetical}"
            description = "Synthesized hypothetical technical document passage to bridge semantic vocabulary gap."
        elif mode == "Sub-Query":
            sub_queries = [
                f"{raw_query} regulatory algorithms and requirements",
                f"{raw_query} rotation timeframes and emergency protocols",
                f"{raw_query} HSM hardware and enforcement constraints"
            ]
            expanded_query = " ".join(sub_queries)
            description = f"Decomposed into 3 targeted sub-queries: {sub_queries}"
        else:
            expanded_query = raw_query
            description = "Passed raw query without transformation."

        return {
            "raw_query": raw_query,
            "expanded_query": expanded_query,
            "mode": mode,
            "description": description
        }

    def _dense_score(self, query: str, text: str) -> float:
        """Mock dense semantic similarity using word overlap."""
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        inter = len(q_words.intersection(t_words))
        return float(inter / (len(q_words) + 1e-5))

    def _sparse_score(self, query: str, text: str) -> float:
        """Mock BM25 lexical keyword matching score."""
        q_tokens = query.lower().split()
        t_text = text.lower()
        score = 0.0
        for token in q_tokens:
            if len(token) > 3 and token in t_text:
                score += 1.5
        return float(score)

    def hybrid_retrieve(self, query_obj: Dict[str, Any], top_k: int = 4) -> List[Dict[str, Any]]:
        """Dense + Sparse Retrieval fused using Reciprocal Rank Fusion (RRF)."""
        q_text = query_obj["expanded_query"]
        
        dense_candidates = []
        for doc in self.corpus:
            d_score = self._dense_score(q_text, doc["title"] + " " + doc["content"])
            dense_candidates.append((doc, d_score))
        dense_candidates.sort(key=lambda x: x[1], reverse=True)
        
        sparse_candidates = []
        for doc in self.corpus:
            s_score = self._sparse_score(q_text, doc["title"] + " " + doc["content"])
            sparse_candidates.append((doc, s_score))
        sparse_candidates.sort(key=lambda x: x[1], reverse=True)
        
        rrf_scores: Dict[str, float] = {}
        doc_map = {doc["id"]: doc for doc in self.corpus}
        
        for rank, (doc, _) in enumerate(dense_candidates, start=1):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (1.0 / (60.0 + rank))
            
        for rank, (doc, _) in enumerate(sparse_candidates, start=1):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + (1.0 / (60.0 + rank))
            
        fused = []
        for doc_id, score in rrf_scores.items():
            doc = doc_map[doc_id]
            fused.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "rrf_score": round(score, 5)
            })
            
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused[:top_k]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], relevance_cutoff: float = 0.018) -> List[Dict[str, Any]]:
        """Simulate cross-encoder deep contextual reranking and relevance thresholding."""
        reranked = []
        for cand in candidates:
            joint_text = (query + " " + cand["title"] + " " + cand["content"]).lower()
            key_matches = sum(1 for kw in ["key", "rotation", "pci-dss", "hsm", "revoke", "aes", "months"] if kw in joint_text)
            cross_score = cand["rrf_score"] * 10.0 + (key_matches * 0.15)
            
            if cross_score >= relevance_cutoff:
                cand_copy = dict(cand)
                cand_copy["cross_encoder_score"] = round(cross_score, 4)
                reranked.append(cand_copy)
                
        reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return reranked

    def compress_context(self, reranked_docs: List[Dict[str, Any]], max_sentences: int = 2) -> List[Dict[str, Any]]:
        """Selective extractive context compression: prunes tangential sentences."""
        compressed = []
        for doc in reranked_docs:
            sentences = [s.strip() for s in doc["content"].split(".") if len(s.strip()) > 10]
            selected = sentences[:max_sentences]
            compressed_content = ". ".join(selected) + "."
            compressed.append({
                "id": doc["id"],
                "title": doc["title"],
                "original_tokens": len(doc["content"].split()),
                "compressed_tokens": len(compressed_content.split()),
                "compressed_content": compressed_content,
                "score": doc.get("cross_encoder_score", doc.get("rrf_score", 0.0))
            })
        return compressed

    def grounded_generate(self, raw_query: str, compressed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Closed-book epistemic contract: strict citation anchoring and guaranteed abstention."""
        if not compressed_docs:
            return {
                "answer": "I cannot answer this question. The verified knowledge base contains no documents meeting relevance thresholds.",
                "citations": [],
                "grounded": True,
                "token_savings_pct": 0.0
            }

        citations = []
        answer_parts = []
        for i, doc in enumerate(compressed_docs, start=1):
            citations.append(f"[{doc['title']} ({doc['id']})]")
            answer_parts.append(f"Under {doc['title']}, {doc['compressed_content']} [{doc['id']}]")

        total_orig = sum(d["original_tokens"] for d in compressed_docs)
        total_comp = sum(d["compressed_tokens"] for d in compressed_docs)
        savings = round(((total_orig - total_comp) / total_orig) * 100.0, 1) if total_orig > 0 else 0.0

        final_answer = (
            f"Based strictly on verified documentation:\\n"
            + "\\n".join(answer_parts) +
            f"\\n\\nAll cryptographic requirements comply with documented governance."
        )

        return {
            "answer": final_answer,
            "citations": citations,
            "grounded": True,
            "token_savings_pct": savings,
            "total_tokens_used": total_comp
        }
''',
    language="python"
)

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Multi-Stage Advanced Pipeline Simulator", divider="rainbow")

st.markdown(
    """
    Step through the complete multi-stage Advanced RAG pipeline.
    Toggle query rewriting modes, adjust cross-encoder cutoffs, and observe how extractive context compression
    reduces prompt token volume while preserving grounded accuracy.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🎛️ Pipeline Configuration")
    
    active_query = st.text_input(
        "User Query:",
        "What are the cryptographic key rotation rules and HSM deployment requirements?"
    )
    
    rewrite_mode = st.radio(
        "1. Query Rewriting Mode:",
        ["HyDE", "Sub-Query", "None"],
        horizontal=True
    )
    
    top_k_retrieve = st.slider("2. Hybrid Candidates Retrieved (Top-K):", min_value=2, max_value=5, value=4, step=1)
    
    rerank_cutoff = st.slider("3. Reranker Cutoff Threshold:", min_value=0.010, max_value=0.060, value=0.018, step=0.002)
    
    max_sentences = st.slider("4. Context Compression (Max Sentences per Chunk):", min_value=1, max_value=3, value=2, step=1)

with wb_col2:
    st.subheader("⚡ Multi-Stage Execution Telemetry")
    
    engine = AdvancedRAGEngine()
    
    # 1. Rewriting
    query_obj = engine.rewrite_query(active_query, mode=rewrite_mode)
    # 2. Hybrid Retrieve
    hybrid_res = engine.hybrid_retrieve(query_obj, top_k=top_k_retrieve)
    # 3. Rerank
    reranked_res = engine.rerank(active_query, hybrid_res, relevance_cutoff=rerank_cutoff)
    # 4. Compress
    compressed_res = engine.compress_context(reranked_res, max_sentences=max_sentences)
    # 5. Generate
    gen_result = engine.grounded_generate(active_query, compressed_res)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Retrieved Candidates", len(hybrid_res))
    m2.metric("Passed Reranker", len(reranked_res))
    m3.metric("Token Savings", f"{gen_result['token_savings_pct']}%")
    
    st.markdown("#### Grounded Answer Output")
    st.info(gen_result["answer"])
    
    st.markdown("#### Document Citations")
    st.write(gen_result["citations"])

st.markdown("### Stage-by-Stage Artifact Inspection")

tab_labels = ["1. Rewritten Query", "2. Hybrid Candidates (RRF)", "3. Reranked & Compressed Chunks"]
s_tab1, s_tab2, s_tab3 = st.tabs(tab_labels)

with s_tab1:
    st.markdown(f"**Strategy**: `{query_obj['mode']}`")
    st.markdown(f"**Description**: {query_obj['description']}")
    st.code(query_obj["expanded_query"], language="text")

with s_tab2:
    rrf_df = pd.DataFrame([
        {"Doc ID": d["id"], "Title": d["title"], "RRF Fusion Score": d["rrf_score"]}
        for d in hybrid_res
    ])
    st.dataframe(rrf_df, use_container_width=True)

with s_tab3:
    comp_df = pd.DataFrame([
        {
            "Doc ID": d["id"],
            "Title": d["title"],
            "Original Tokens": d["original_tokens"],
            "Compressed Tokens": d["compressed_tokens"],
            "Compressed Content": d["compressed_content"]
        }
        for d in compressed_res
    ])
    st.dataframe(comp_df, use_container_width=True)

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: What is Reciprocal Rank Fusion (RRF) and why is it preferred over weighted score averaging in Hybrid Search?**"):
    st.markdown(
        """
        In hybrid retrieval, dense vector models emit cosine similarities bounded in `[-1, 1]` or `[0, 1]`, whereas sparse BM25 produces unbounded lexical scores (`[0, inf)`) dependent on document length and inverse document frequency.
        
        Directly combining these scores requires complex, brittle normalization (MinMax or Z-score), which breaks under domain shifts.
        **Reciprocal Rank Fusion (RRF)** bypasses raw scores entirely by operating strictly on ordinal rankings:
        
        `RRF(d) = Sum(1 / (k + rank_m(d)))`
        
        where `rank_m(d)` is document `d`'s rank in retriever `m`, and `k` is a smoothing constant (typically `60`). RRF is completely robust to disparate score distributions and guarantees smooth fusion.
        """
    )

with st.expander("**Q2: How does a Cross-Encoder Reranker differ mathematically from a Bi-Encoder Dense Retriever?**"):
    st.markdown(
        """
        - **Bi-Encoder**: The query `Q` and document `D` are embedded independently into vectors `u = E(Q)` and `v = E(D)`. The score is their dot product `u . v`. The tokens in `Q` never interact with tokens in `D` during embedding calculation.
        - **Cross-Encoder**: `Q` and `D` are concatenated into a single sequence: [CLS] + Q + [SEP] + D + [SEP]. The full transformer applies cross-attention between every token in `Q` and every token in `D` across all transformer layers.
        
        While computationally expensive (`O((|Q| + |D|)^2)` vs `O(1)` lookup), cross-encoders capture subtle linguistic nuances, negation, and strict constraints that bi-encoders miss.
        """
    )

with st.expander("**Q3: What are the primary failure modes of Hypothetical Document Embeddings (HyDE)?**"):
    st.markdown(
        """
        HyDE relies on an LLM to generate a hypothetical passage before retrieval. It fails when:
        1. **Domain Hallucinations**: For cutting-edge or private corporate topics, the LLM hallucinates false terminology, producing a hypothetical embedding that pulls completely irrelevant documents.
        2. **Latency Penalty**: Adding an LLM inference call in the pre-retrieval critical path adds 300ms - 800ms of TTFT latency before the vector search even starts.
        3. **Adversarial / Negative Queries**: For queries asking what *not* to do, the hypothetical answer frequently generates affirmative examples, confusing vector search.
        """
    )

with st.expander("**Q4: How does Extractive Context Compression prevent 'Lost in the Middle' degradation?**"):
    st.markdown(
        """
        Large retrieved contexts contain massive amounts of irrelevant prose (disclaimers, transitional paragraphs).
        As prompt length scales past 2,000 tokens, transformer attention heads exhibit a U-shaped curve, heavily attending to tokens at the beginning and end of the prompt while dropping attention on the middle chunks.
        
        Extractive compression scores individual sentences or paragraphs using lightweight semantic similarity, keeping only the top 2-3 essential sentences per document.
        This shrinks total prompt length by 50% - 70%, ensuring that all evidence fits inside the LLM's optimal attention zone.
        """
    )

with st.expander("**Q5: How do you enforce a Closed-Book Epistemic Contract in the system prompt?**"):
    st.markdown(
        """
        An enterprise closed-book epistemic contract combines three prompt engineering techniques:
        1. **Explicit Role Boundary**: *"You are an objective legal clerk. You possess zero pre-trained factual knowledge. You only know what is explicitly written in the provided context."*
        2. **Mandatory Citation Anchor**: *"Every claim must be immediately followed by the exact bracketed identifier of the supporting chunk (e.g., [doc_101])."*
        3. **Abstention Fallback**: *"If the provided context does not contain the answer, you must state exactly: 'The provided documents do not contain sufficient information to answer this question.' Any speculation will result in termination."*
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Applying Cross-Encoder Reranking to Entire Corpora**  
        **Antipattern**: Passing 500+ documents from vector search directly into a cross-encoder model.  
        **Consequence**: Cross-encoders require `O(N)` transformer forward passes; reranking 500 documents adds 3,000ms+ of latency, crashing API throughput.  
        **Production Fix**: Prune the candidate pool to Top-20 or Top-30 using fast hybrid retrieval, and only rerank the filtered candidate set.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Normalizing BM25 and Vector Scores Linearly**  
        **Antipattern**: Calculating `Score = alpha * Cosine + (1 - alpha) * BM25` without rank fusion.  
        **Consequence**: BM25 scores can range from 0 to 45 while cosine is bounded in [0, 1]. A single high BM25 outlier completely drowns out semantic vector scores.  
        **Production Fix**: Use Reciprocal Rank Fusion (RRF) unconditionally for hybrid search fusion.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Unconditional HyDE Execution**  
        **Antipattern**: Running HyDE query rewriting on every single incoming query, including simple factoid searches like *"What is the office address?"*.  
        **Consequence**: Wastes LLM tokens and adds 500ms of needless latency to simple lookups that standard BM25 could resolve in 2ms.  
        **Production Fix**: Implement a lightweight query classifier to route only ambiguous or conceptual queries to HyDE.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Compressing Without Respecting Sentence Boundaries**  
        **Antipattern**: Compressing context using character token truncators that slice sentences in half.  
        **Consequence**: Slices off critical negation words (*"not"*, *"never"*) or condition clauses (*"except in emergencies"*), causing catastrophic factual inversion.  
        **Production Fix**: Ensure extractive compressors operate on complete grammatical sentences or semantic clauses.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What are the 5 pillars of Advanced RAG?")
        st.markdown("**Back**: Query Rewriting, Hybrid Dense+Sparse Retrieval, Cross-Encoder Reranking, Context Compression, and Grounded Generation.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: Why use Reciprocal Rank Fusion (RRF)?")
        st.markdown("**Back**: Combines dense vector and sparse BM25 search based on ordinal rankings rather than unbounded scores, eliminating score normalization issues.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is Cross-Encoder Reranking?")
        st.markdown("**Back**: A heavy transformer model that calculates token-to-token cross-attention between query and candidate documents, delivering high-precision relevance scoring.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is Context Compression?")
        st.markdown("**Back**: Pruning non-essential sentences and boilerplate from retrieved chunks to protect token budgets, reduce latency, and eliminate attention distraction.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of the enterprise Advanced RAG multi-stage architecture:")

chk1 = st.checkbox("I understand the end-to-end flow of the five pillars: Rewriting -> Hybrid -> Rerank -> Compress -> Grounded Gen.", value=True)
chk2 = st.checkbox("I can explain how HyDE and Sub-Query decomposition resolve vocabulary mismatch and multi-part queries.", value=True)
chk3 = st.checkbox("I know why Reciprocal Rank Fusion (RRF) is superior to linear score weighting in hybrid search.", value=True)
chk4 = st.checkbox("I understand the structural and attention differences between Bi-Encoder retrievers and Cross-Encoder rerankers.", value=True)
chk5 = st.checkbox("I can implement extractive context compression to prune boilerplate and eliminate lost-in-the-middle degradation.", value=True)
chk6 = st.checkbox("I know how to construct strict epistemic contracts that enforce citation anchors and guaranteed abstention.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Outstanding! You have mastered the complete Multi-Stage Advanced RAG Architecture.")
