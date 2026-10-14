import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="2.3 NLP Tasks - LLM Curriculum",
    page_icon="🎯",
    layout="wide"
)

# ==============================================================================
# Title Block and Executive Overview
# ==============================================================================
st.title("2.3 Foundational NLP Tasks and the Text-to-Text Paradigm Shift")
st.caption("Chapter 2: NLP Foundations for LLMs | Domain: 2. Large Language Models")

st.markdown("""
Historically, Natural Language Processing was fractured into dozens of specialized sub-disciplines. 
Machine learning engineering teams designed, trained, and deployed bespoke neural models with custom task-specific 
output heads for **Text Classification, Sentiment Analysis, Named Entity Recognition (NER), Extractive Question Answering, 
Semantic Text Similarity, Semantic Search, Text Generation, and Machine Translation**.

The advent of modern foundation models sparked an architectural revolution: **the Unified Text-to-Text Paradigm** 
(pioneered by T5 and generalized by GPT, Llama, and Claude). Every natural language task—regardless of whether it 
classifies sentiment, extracts entity attributes, scores semantic distance, or translates across polyglot languages—is 
now formulated as an autoregressive text-to-text sequence transformation driven by causal cross-entropy optimization.

In this lesson, you will master:
1. The 8 foundational NLP task domains and how each was formulated in classical machine learning vs. modern LLMs.
2. The architectural collapse from bespoke classification heads and CRF decoders into unified autoregressive token prediction.
3. Why foundation models replace extractive span pointers with abstractive grounded reasoning in RAG pipelines.
4. How logit evaluation and constrained grammar decoding enable zero-shot classification and structured JSON schema extraction.
5. A runnable pure Python implementation of a unified multi-task engine with deterministic execution and validation.
""")

st.divider()

# ==============================================================================
# Section 1: Architectural Blueprint: The Unified Text-to-Text Paradigm Shift
# ==============================================================================
st.header("1. Architectural Blueprint: The Unified Text-to-Text Paradigm Shift", divider="rainbow")

st.markdown("""
Prior to foundation models, deploying enterprise NLP required managing an expensive matrix of heterogeneous models, 
each with distinct tokenizers, custom loss functions, and task-specific classification heads. Modern foundation models 
collapse this fragmentation into a single autoregressive generative backbone conditioned by natural language instructions:
""")

col_diag1, col_diag2 = st.columns([1.3, 0.7])

with col_diag1:
    unification_dot = """
    digraph UnifiedTextToText {
        bgcolor="#0B1120";
        rankdir=TB;
        compound=true;
        nodesep=0.35;
        ranksep=0.65;

        node [
            fontname="Helvetica",
            fontsize=10,
            fontcolor="#F8FAFC",
            shape=box,
            style="filled,rounded",
            color="#334155",
            penwidth=1.5
        ];

        edge [
            fontname="Helvetica",
            fontsize=8,
            fontcolor="#94A3B8",
            color="#38BDF8",
            penwidth=1.6,
            arrowsize=0.7
        ];

        subgraph cluster_legacy {
            label="Legacy NLP: Fragmented Bespoke Task Heads";
            style=filled;
            color="#1E293B";
            fontcolor="#EF4444";
            
            L_Input [label="Input Document x", fillcolor="#334155"];
            Head_Cls [label="Linear Head [D, C]\\nSoftmax Loss\\n(Classification)", fillcolor="#450A0A", color="#EF4444"];
            Head_NER [label="BiLSTM + CRF\\nViterbi Decoding\\n(Entity Tagging)", fillcolor="#450A0A", color="#EF4444"];
            Head_QA [label="Span Pointers\\nStart and End Logits\\n(Extractive QA)", fillcolor="#450A0A", color="#EF4444"];
            Head_Trans [label="Encoder-Decoder\\nPairwise MarianMT\\n(Translation)", fillcolor="#450A0A", color="#EF4444"];
            
            L_Input -> Head_Cls;
            L_Input -> Head_NER;
            L_Input -> Head_QA;
            L_Input -> Head_Trans;
        }

        subgraph cluster_modern {
            label="Modern LLM: Unified Causal Autoregressive Backbone";
            style=filled;
            color="#1E293B";
            fontcolor="#34D399";
            
            M_Prompt [label="Prompt Conditioning\\nDirective + Input Context", fillcolor="#064E3B", color="#34D399"];
            Transformer_Core [label="Causal Decoder Transformer\\nUniversal Weight Tensor W\\nShared Vocab Embeddings", fillcolor="#047857", color="#34D399", shape=hexagon];
            M_Output [label="Unified String Stream\\n(Labels, JSON Schemas, Answers, Translations)", fillcolor="#065F46", color="#34D399"];
            
            M_Prompt -> Transformer_Core -> M_Output;
        }
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(unification_dot, use_container_width=True)

with col_diag2:
    st.markdown("#### Architectural Principles of Unification")
    st.markdown("""
    * **Single Parameter Tensor**: Instead of maintaining 10 distinct fine-tuned model artifacts, one foundation model handles all task domains simultaneously.
    * **In-Context Conditioning**: Task instructions are provided directly inside the sequence context window rather than hardcoded in network topology.
    * **Universal Loss Formulation**: Replaces task-specific objectives (CRF loss, span cross-entropy, triplet loss) with a single universal loss: **Next-Token Causal Cross-Entropy**.
    * **Generative Flexibility**: Tasks that previously required strict structural assumptions (such as span bounds) can now synthesize novel formulations, explanations, and structured JSON schemas.
    """)

# Data Structure Diagram
st.markdown("#### Data Structure Architecture: Legacy Task Heads vs. Universal Text Generation")

task_table_dot = """
digraph TaskFormatTable {
    bgcolor="#0B1120";
    rankdir=TB;

    node [fontname="Helvetica", fontsize=10, fontcolor="#F8FAFC", shape=plaintext];

    TableNode [label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="8" bgcolor="#1E293B">
            <tr>
                <td bgcolor="#0F172A" colspan="4"><font color="#38BDF8"><b>Task Formulation Matrix: Legacy Specialized Heads vs Foundation LLM</b></font></td>
            </tr>
            <tr bgcolor="#334155">
                <td><b>Task Domain</b></td>
                <td><b>Legacy Architecture (Pre-2020)</b></td>
                <td><b>Foundation LLM Formulation (Modern)</b></td>
                <td><b>Output Interface</b></td>
            </tr>
            <tr>
                <td bgcolor="#0C4A6E"><b>Text Classification</b></td>
                <td>BERT + Linear Layer [Hidden_Dim, C] + Softmax</td>
                <td>In-context prompt conditioning + First-token logit scoring</td>
                <td>Scalar Class or String Label</td>
            </tr>
            <tr>
                <td bgcolor="#1E1B4B"><b>Sentiment Analysis</b></td>
                <td>VADER lexicon or DistilBERT 3-class classifier</td>
                <td>Chain-of-Thought aspect reasoning + Polar scalar rating</td>
                <td>Structured Polarity Record</td>
            </tr>
            <tr>
                <td bgcolor="#78350F"><b>Named Entity Recognition</b></td>
                <td>BiLSTM + Conditional Random Field (CRF)</td>
                <td>Constrained decoding to strict Pydantic JSON Schema</td>
                <td>Typed Nested JSON Object</td>
            </tr>
            <tr>
                <td bgcolor="#064E3B"><b>Question Answering</b></td>
                <td>Span Head: Predict start_index and end_index</td>
                <td>RAG context retrieval + Abstractive synthesis</td>
                <td>Grounded Natural Language Paragraph</td>
            </tr>
            <tr>
                <td bgcolor="#451A03"><b>Text Similarity</b></td>
                <td>Siamese Bi-Encoder (Sentence-BERT cosine score)</td>
                <td>Dual dense embeddings and LLM-as-a-Judge arbitration</td>
                <td>Continuous Metric and Comparative Critique</td>
            </tr>
            <tr>
                <td bgcolor="#312E81"><b>Semantic Search</b></td>
                <td>Sparse BM25 inverted index / Elasticsearch</td>
                <td>MIPS Dense Index (HNSW) + Cross-Encoder re-ranking</td>
                <td>Ranked Passage Identifiers</td>
            </tr>
            <tr>
                <td bgcolor="#042F2E"><b>Text Generation</b></td>
                <td>N-gram models or LSTM with beam search</td>
                <td>Causal Transformer with KV cache and Nucleus sampling</td>
                <td>Autoregressive Token Stream</td>
            </tr>
            <tr>
                <td bgcolor="#1F2937"><b>Machine Translation</b></td>
                <td>Bespoke pairwise MarianMT Seq2Seq models</td>
                <td>Unified polyglot pretraining across 100+ languages</td>
                <td>Native Multi-lingual String</td>
            </tr>
        </table>
    >];
}
"""
with st.expander("🔍 View Architecture & Process Flow Diagram (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(task_table_dot, use_container_width=True)

st.divider()

# ==============================================================================
# Section 2: Text Classification
# ==============================================================================
st.header("2. Text Classification: From Linear Probing to Autoregressive Scoring", divider="rainbow")

st.markdown("""
Text classification is the fundamental task of mapping an arbitrary text passage to one or more predefined 
discrete categorical classes:

`f: Document -> Label in {C_1, C_2, ..., C_k}`

Common enterprise applications include spam detection, support ticket topic routing, legal contract triage, and intent detection.
""")

col_cls_a, col_cls_b = st.columns(2)

with col_cls_a:
    st.markdown("#### Classical Formulation: Supervised Linear Heads")
    st.markdown("""
    * **Feature Extraction**: Documents were initially transformed into sparse TF-IDF vectors or passed through an encoder (like BERT) to extract a pooled embedding tensor:
      `h_doc = Pooling(Encoder(x))   # Dimension: [Batch_Size, Hidden_Dimension]`
    * **Classification Head**: A specialized linear weight matrix was projected over the hidden states:
      `logits = W_cls * h_doc + b_cls   # Dimension: [Batch_Size, Num_Classes]`
    * **Training and Inference**: Trained with cross-entropy loss against one-hot target distributions. 
    * **Inherent Limitations**:
      * Rigid label taxonomy: adding a new class requires changing `W_cls` dimensions and retraining from scratch.
      * Zero knowledge transfer: cannot classify categories never seen in the supervised fine-tuning dataset.
      * High operational overhead: deploying 50 distinct classification tasks required 50 separate fine-tuned neural models.
    """)

with col_cls_b:
    st.markdown("#### Modern Foundation Model Reality: In-Context and Logit Scoring")
    st.markdown("""
    Modern LLMs eliminate the dedicated linear classification head entirely. Text classification is achieved via two primary mechanisms:
    
    1. **Generative In-Context Classification**:
       * The model is conditioned with system instructions and class definitions:
         `"Classify the following customer ticket as BILLING, TECHNICAL, or ACCOUNT: [Text]. Category: "`
       * The model autoregressively generates the text label token directly.
    
    2. **Constrained First-Token Logit Scoring (Zero-Shot)**:
       * Rather than allowing the LLM to generate arbitrary open-ended text, production pipelines evaluate the unnormalized logit values at the very first generated token position:
         `P(Label_i) = Softmax(Logits[Token_ID(Label_i)])`
       * This guarantees 100% deterministic label assignment, eliminates hallucinated category labels, and requires zero labeled fine-tuning data.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Foundation models such as Llama 3 and GPT-4 treat classification as an in-context inference task. When extreme low latency 
and cost efficiency are paramount, modern architectures use **Logit Bias / Logit Masking**: constraining the vocabulary 
sampling distribution exclusively to the candidate class token IDs (`[' SPAM', ' HAM']`), extracting classification 
probabilities with mathematically guaranteed single-token inference.
""")

cls_comp_df = pd.DataFrame({
    "Dimension": [
        "Architecture",
        "Output Mechanism",
        "Adding New Class",
        "Few-Shot Adaptability",
        "Cold-Start Capability",
        "Inference Compute Cost"
    ],
    "Classical NLP (BERT / RoBERTa)": [
        "Encoder + Linear Layer [768, C]",
        "Argmax over C linear logit scores",
        "Requires architecture modification and retraining",
        "Poor; requires dozens of labeled examples per class",
        "Zero; fails completely on unseen classes",
        "Low (single encoder forward pass)"
    ],
    "Modern Foundation LLM (Llama 3 / GPT-4)": [
        "Unified Autoregressive Decoder Transformer",
        "In-context prompt generation or first-token logit extraction",
        "Instantaneous via prompt update (zero retraining)",
        "Exceptional; adapts from 1-3 in-context exemplars",
        "Superhuman zero-shot generalization via semantic understanding",
        "Higher; requires decoder generative forward pass"
    ]
})
st.dataframe(cls_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Sentiment Analysis
# ==============================================================================
st.header("3. Sentiment Analysis: Polarity, Aspect-Based Mining, and Nuance", divider="rainbow")

st.markdown("""
Sentiment analysis evaluates the emotional tone, polarity, and subjective attitude expressed in text. 
While early methods treated sentiment as a flat scalar rating (Negative vs. Neutral vs. Positive), enterprise language 
often contains multi-faceted, nuanced evaluations.
""")

col_sent_a, col_sent_b = st.columns(2)

with col_sent_a:
    st.markdown("#### Classical Sentiment: Lexicons and Flat Classifiers")
    st.markdown("""
    * **Lexicon-Based Scoring (VADER, SentiWordNet)**: Computed polarity by summing pre-compiled dictionary sentiment valence scores with heuristic rules for negation ("not bad") and intensifiers ("extremely good").
    * **Machine Learning Classifiers**: Fine-tuned BERT on datasets like SST-2 or IMDb to predict a 2-way or 3-way classification.
    * **Failure Modes**:
      * Completely blind to sarcasm, irony, and double entendres.
      * **Aspect Blending**: In the review *"The camera hardware is breathtaking, but the battery life is an absolute disaster"*, flat classifiers output a neutralized score (0.5), completely losing the distinct feedback on individual components.
    """)

with col_sent_b:
    st.markdown("#### Aspect-Based Sentiment Analysis (ABSA) with LLMs")
    st.markdown("""
    * **Fine-Grained Decomposition**: Modern LLMs break reviews down into explicit tuples: `(Aspect, Polarity, Evidence_Quote, Severity)`.
    * **Chain-of-Thought Emotional Reasoning**: By prompting the model to explain its rationale before declaring a polarity label, models correctly resolve complex sarcasm:
      * *Review*: "Oh wonderful, another mandatory software update that took 4 hours and erased my local cache."
      * *LLM Reasoning*: "The speaker uses sarcastic enthusiasm ('Oh wonderful') followed by negative business impacts (4 hours lost, erased cache). The true intent is intensely negative."
      * *Output*: `NEGATIVE` with confidence 0.98.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern enterprise pipelines use LLMs to perform Aspect-Based Sentiment Extraction directly into structured schemas. 
Instead of relying on fragile rule-based sentiment engines, LLMs parse user reviews into JSON payloads containing 
disaggregated sentiment scores across product dimensions (e.g. UX, Reliability, Price, Support), along with direct textual citations.
""")

sent_comp_df = pd.DataFrame({
    "Criterion": [
        "Granularity",
        "Sarcasm Handling",
        "Mixed Sentiments",
        "Aspect Extraction",
        "Explainability"
    ],
    "Legacy Sentiment (VADER / BERT)": [
        "Document-level or sentence-level flat scalar",
        "Fails frequently on ironic/sarcastic phrasing",
        "Averages conflicting sentiments into misleading 'neutral'",
        "Requires complex secondary sequence tagging models",
        "Black-box probability distribution"
    ],
    "Modern Foundation LLM": [
        "Entity-level and aspect-level hierarchical breakdown",
        "Decodes conversational nuance and contextual sarcasm",
        "Disentangles competing sentiments across entities",
        "Extracts targets, aspects, and polarity in a single pass",
        "Generates explicit chain-of-thought rationale"
    ]
})
st.dataframe(sent_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: Named Entity Recognition (NER)
# ==============================================================================
st.header("4. Named Entity Recognition (NER): From BIO Sequence Tagging to Schema Extraction", divider="rainbow")

st.markdown("""
Named Entity Recognition (NER) identifies contiguous spans of text and categorizes them into predefined semantic 
classes such as **Person, Organization, Location, Monetary Value, Date, or Gene Sequence**.
""")

col_ner_a, col_ner_b = st.columns(2)

with col_ner_a:
    st.markdown("#### Classical NER: Token-Level BIO Tagging with CRFs")
    st.markdown("""
    * **BIO Tagging Scheme**: Every token in the input sequence was tagged with a prefix:
      * `B-PER`: Beginning of a Person entity.
      * `I-PER`: Inside/continuation of a Person entity.
      * `O`: Outside of any entity.
    * **Architecture**: A Bidirectional LSTM or BERT model generated token emission scores, which were fed into a **Conditional Random Field (CRF)** layer:
      * The CRF layer maintained a learned transition matrix `A[y_t, y_{t-1}]` enforcing syntactic constraints (e.g., forbidding `I-PER` immediately after `O`).
    * **Critical Bottlenecks**:
      * **Nested Entities**: Unable to represent overlapping entities (e.g., in *"Bank of America"*, the entire span is an `ORG`, but *"America"* is simultaneously a `GPE/LOC`).
      * **Discontinuous Entities**: Cannot link non-adjacent tokens belonging to the same entity.
      * **Token Alignment Fragility**: Any mismatch between tokenizer boundaries and entity spans corrupts extraction.
    """)

with col_ner_b:
    st.markdown("#### Modern LLM: Schema-Constrained Extraction")
    st.markdown("""
    * **Structured Object Generation**: Modern LLMs bypass token-level labeling entirely. The model receives raw text and outputs structured JSON objects conforming to strict schemas:
      ```json
      {
        "entities": [
          {"name": "Satya Nadella", "category": "PERSON", "role": "CEO"},
          {"name": "Microsoft", "category": "ORGANIZATION", "ticker": "MSFT"}
        ]
      }
      ```
    * **Constrained Grammar Decoding**: Tools like Guidance, Outlines, and native JSON schema modes enforce grammar rules during autoregressive generation. At each generation step, the vocabulary logits are masked so that only tokens satisfying valid JSON schema syntax can be sampled.
    * **Zero-Shot Entity Discovery**: By prompting with natural language entity descriptions, LLMs extract domain-specific entities (e.g., medical symptoms, legal obligations) without training on annotator BIO tags.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern LLMs have replaced token BIO taggers with schema-constrained decoding. Systems like instructor, outlines, 
and OpenAI Function Calling force the causal language model to emit valid Pydantic models. This natively solves 
nested entities, standardizes attribute naming, and extracts relationship links between entities in a single generative call.
""")

ner_comp_df = pd.DataFrame({
    "Feature": [
        "Formulation",
        "Nested / Overlapping Entities",
        "Discontinuous Spans",
        "Attribute Normalization",
        "Schema Enforcement",
        "New Entity Types"
    ],
    "Legacy NER (BiLSTM-CRF / SpaCy)": [
        "Token-level BIO classification",
        "Impossible without complex multi-label extensions",
        "Cannot group disjoint tokens",
        "Requires separate downstream entity resolution models",
        "Constrained by CRF transition matrix",
        "Requires thousands of manual BIO token annotations"
    ],
    "Modern Foundation LLM": [
        "Autoregressive structured JSON synthesis",
        "Natively handles arbitrary nested hierarchies",
        "Effortlessly links disjoint references",
        "Normalizes dates, currencies, and names inline",
        "Guaranteed by logit grammar masking (JSON mode)",
        "Zero-shot extraction defined via natural language prompt"
    ]
})
st.dataframe(ner_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: Question Answering
# ==============================================================================
st.header("5. Question Answering: Extractive Span Prediction vs. Abstractive Grounded Synthesis", divider="rainbow")

st.markdown("""
Question Answering (QA) systems allow users to query document knowledge bases in natural language and receive 
precise, factual responses.
""")

col_qa_a, col_qa_b = st.columns(2)

with col_qa_a:
    st.markdown("#### Extractive QA (SQuAD Era: 2016-2020)")
    st.markdown("""
    * **Problem Formulation**: Given a context passage and question, the answer is assumed to be a verbatim substring within the passage.
    * **Model Mechanism**: An encoder (BERT, RoBERTa) processed the concatenated sequence `[CLS] Q [SEP] C [SEP]`.
    * Two linear heads projected the hidden states into two probability distributions over the context tokens:
      `P_start(i) = Softmax(W_start * h_i)   # Index where answer starts`
      `P_end(j) = Softmax(W_end * h_j)       # Index where answer ends`
    * The answer span was extracted as the slice minimizing the combined loss: `Span = Context[start_idx : end_idx + 1]`.
    * **Severe Limitations**:
      * If the answer was distributed across multiple sentences, the model failed.
      * Could not perform arithmetic, synthesis, comparison, or summarization.
      * If the question required reasoning or paraphrasing, extractive models returned garbled or irrelevant snippets.
    """)

with col_qa_b:
    st.markdown("#### Abstractive Grounded Synthesis (RAG Era: 2023+)")
    st.markdown("""
    * **Problem Formulation**: The system generates an original, fluent, and comprehensive response synthesized from multiple reference documents.
    * **Retrieval-Augmented Generation (RAG)**:
      1. **Retrieval**: A dense vector index retrieves the top-k relevant chunks across a corpus of millions of documents.
      2. **Context Injection**: Retrieved passages are injected into the prompt as verified grounding evidence.
      3. **Generative Synthesis**: The causal LLM reads the context, resolves contradictions, performs numerical calculations, and synthesizes a structured answer complete with inline citations.
    * **Key Capabilities**:
      * Cross-document multi-hop reasoning.
      * Synthesizes answers that do not appear verbatim anywhere in the source text.
      * Explicitly admits when the retrieved context lacks sufficient evidence ("I do not have enough information to answer").
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern QA architectures abandon span-pointer heads in favor of RAG pipelines backed by foundation LLMs. 
Systems like Perplexity, Google Search Generative Experience, and enterprise RAG engines perform multi-document 
retrieval followed by abstractive synthesis, citing source documents and checking factual attribution.
""")

qa_comp_df = pd.DataFrame({
    "Aspect": [
        "Answer Nature",
        "Source Dependency",
        "Multi-Hop Reasoning",
        "Synthesis & Summarization",
        "Missing Information Handling",
        "Hallucination Risk"
    ],
    "Extractive QA (BERT Span Head)": [
        "Verbatim contiguous substring slice",
        "Single short passage (< 512 tokens)",
        "Zero capability across disparate passages",
        "Cannot summarize or rephrase text",
        "Selects arbitrary high-confidence false span",
        "Zero hallucination (only extracts existing text)"
    ],
    "Abstractive Grounded QA (RAG + LLM)": [
        "Grammatically novel, synthesized natural language",
        "Multi-document corpus across massive vector databases",
        "Synthesizes facts across multiple paragraphs and files",
        "Exceptional; condenses, explains, and tabulates data",
        "Recognizes absence of evidence and refrains from guessing",
        "Present if grounding context is weak (mitigated by strict prompt rules)"
    ]
})
st.dataframe(qa_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: Text Similarity
# ==============================================================================
st.header("6. Text Similarity: Lexical Overlap, Siamese Bi-Encoders, and Cross-Encoders", divider="rainbow")

st.markdown("""
Semantic Textual Similarity (STS) quantifies the degree of semantic equivalence between two sentences or documents, 
typically producing a continuous score from 0.0 (completely unrelated) to 1.0 (identical meaning).
""")

col_sim_a, col_sim_b = st.columns(2)

with col_sim_a:
    st.markdown("#### The Bi-Encoder vs. Cross-Encoder Trade-Off")
    st.markdown("""
    * **Lexical Similarity (Jaccard, Levenshtein, BM25)**:
      * Measured character or token overlap.
      * Failed completely on synonyms: *"The physician examined the patient"* and *"The doctor checked the sick person"* share zero content words, producing a Jaccard score of 0.0.
    * **Siamese Bi-Encoders (Sentence-BERT)**:
      * Passages A and B are encoded independently into dense vectors u and v:
        `sim(A, B) = Cosine(u, v) = (u . v) / (||u|| * ||v||)`
      * Highly scalable (O(N) encoding, pre-computable offline index), but suffers from the **information compression bottleneck**: an entire paragraph is compressed into 768 floating point numbers.
    * **Cross-Encoders**:
      * Concatenate both passages together: `Encoder([CLS] A [SEP] B)`. Full self-attention operates across all token pairs of both texts simultaneously.
      * Highly accurate, but scales quadratically: computing pairwise similarity across 10,000 documents requires 100,000,000 cross-encoder evaluations, which is computationally intractable for search.
    """)

with col_sim_b:
    st.markdown("#### LLM-as-a-Judge and Hierarchical Similarity")
    st.markdown("""
    * **Multi-Stage Ranking Architecture**:
      1. **Stage 1 (Bi-Encoder Retrieval)**: Fast Approximate Nearest Neighbor (ANN) search scans millions of vectors in under 10 milliseconds, narrowing candidate documents to top-100.
      2. **Stage 2 (Cross-Encoder Re-Ranking)**: A cross-encoder re-ranks the top-100 candidates to select the top-10.
      3. **Stage 3 (LLM Semantic Arbitration)**: A foundation model acts as a semantic judge for subtle edge cases (e.g., regulatory compliance, contract variance analysis):
         * Evaluates whether two clauses are legally equivalent, contradictory, or conditionally aligned.
         * Generates an itemized breakdown of discrepancies and semantic divergence.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern enterprise search and deduplication systems adopt a two-tier paradigm: dense embeddings (OpenAI `text-embedding-3`, 
BGE-M3) handle high-throughput approximate retrieval, while foundation LLMs perform deep semantic similarity arbitration 
on complex passages where cosine distance is too coarse to detect subtle factual contradictions.
""")

sim_comp_df = pd.DataFrame({
    "Method": [
        "Lexical (BM25 / Jaccard)",
        "Bi-Encoder (Sentence-BERT)",
        "Cross-Encoder",
        "LLM-as-a-Judge"
    ],
    "Representational Power": [
        "Surface token overlap only",
        "Coarse semantic vector alignment",
        "Deep token-to-token cross-attention",
        "Full logical and factual reasoning"
    ],
    "Query-Time Complexity": [
        "O(1) inverted index lookup",
        "O(1) vector dot product (MIPS)",
        "O(N) expensive Transformer forward passes",
        "High latency; reserved for top-k arbitration"
    ],
    "Handles Paraphrasing": [
        "No (fails on synonyms)",
        "Yes (good semantic mapping)",
        "Exceptional (precise nuance)",
        "Superhuman (understands logical implication)"
    ],
    "Scalability to 10M Documents": [
        "Trivial (standard search engines)",
        "Trivial via HNSW / ScaNN vector indexing",
        "Completely impossible at query time",
        "Impossible for retrieval; used on final top-5"
    ]
})
st.dataframe(sim_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: Semantic Search
# ==============================================================================
st.header("7. Semantic Search: Dense Vector Retrieval, MIPS, and RAG Pipelines", divider="rainbow")

st.markdown("""
Semantic search retrieves documents based on the **conceptual intent and underlying meaning** of a query rather 
than exact keyword strings. It represents the retrieval backbone powering all enterprise RAG architectures.
""")

col_srch_a, col_srch_b = st.columns(2)

with col_srch_a:
    st.markdown("#### The Sparse vs. Dense Search Revolution")
    st.markdown("""
    * **Sparse Keyword Search (BM25 / Lucene)**:
      * Relies on term frequency and inverse document frequency:
        `Score(D, Q) = Sum_q IDF(q) * (TF(q, D) * (k1 + 1)) / (TF(q, D) + k1 * (1 - b + b * (|D| / avgdl)))`
      * Suffers from the **Vocabulary Mismatch Problem**: searching for *"reduce server latency"* fails to match a document titled *"optimizing database query response times"*.
    * **Dense Vector Search**:
      * The query and documents are projected into continuous semantic manifolds:
        `q_vec = Embed(Query);   d_vec = Embed(Document)`
      * Retrieval is executed via **Maximum Inner Product Search (MIPS)** or Cosine Similarity:
        `Relevant_Docs = argmax_d (q_vec . d_vec)`
    * Graph-based indexing algorithms such as **Hierarchical Navigable Small World (HNSW)** enable millisecond-range approximate nearest neighbor retrieval across billions of vector embeddings.
    """)

with col_srch_b:
    st.markdown("#### Hybrid Search: Combining Sparse and Dense")
    st.markdown("""
    * **Why Dense Search Alone Fails**:
      * Dense models frequently stumble on rare acronyms, product serial numbers, exact SKU codes, and telephone numbers (which lack dense semantic training signals).
    * **Reciprocal Rank Fusion (RRF)**:
      * Production search engines run BM25 and Dense Vector search in parallel, fusing their ranked lists:
        `RRF_Score(d) = Sum_m (1 / (k + Rank_m(d)))`
      * Where k is a smoothing constant (typically 60) and Rank_m(d) is the position of document d in system m.
      * This delivers the best of both worlds: exact lexical precision for unique identifiers and dense semantic generalization for broad conceptual queries.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
Modern semantic search systems integrate dense vector retrieval directly into generative LLM loops. 
Dense embeddings provide candidate context passages, while foundation models perform query rewriting, 
multi-query expansion (generating 5 alternative queries for a single user question), and final answer synthesis.
""")

srch_comp_df = pd.DataFrame({
    "Component": [
        "BM25 Lexical Search",
        "Dense Vector Search",
        "Hybrid Search (RRF)",
        "Agentic RAG Search"
    ],
    "Query Processing": [
        "Exact word tokenization and stemming",
        "Dense embedding projection [1536 dims]",
        "Dual sparse + dense pipeline execution",
        "LLM multi-query expansion and HyDE"
    ],
    "Matching Mechanism": [
        "Inverted index term frequency",
        "Vector dot product via HNSW graph",
        "Reciprocal rank normalization",
        "Iterative retrieval with corrective critique"
    ],
    "Exact SKU / Keyword Precision": [
        "Extremely High",
        "Low to Moderate",
        "Extremely High",
        "Extremely High"
    ],
    "Conceptual / Paraphrase Recall": [
        "Poor",
        "Extremely High",
        "Extremely High",
        "Superhuman"
    ]
})
st.dataframe(srch_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: Text Generation
# ==============================================================================
st.header("8. Text Generation: Autoregressive Language Modeling and Controlled Decoding", divider="rainbow")

st.markdown("""
Text generation is the core computational capability from which all other modern LLM capabilities emerge. 
Given a sequence of input context tokens, the model generates continuation text by iteratively predicting the probability 
distribution over the vocabulary for the next token:

`P(w_1, w_2, ..., w_T) = Product_{t=1}^T P(w_t | w_1, ..., w_{t-1})`
""")

col_gen_a, col_gen_b = st.columns(2)

with col_gen_a:
    st.markdown("#### Historical Text Generation and Its Failures")
    st.markdown("""
    * **Statistical N-Gram Models**: Predicted the next word based exclusively on the prior $N-1$ words:
      `P(w_t | w_{t-1}, w_{t-2})`
      Suffered from combinatorial explosion and zero long-range memory.
    * **Recurrent Neural Networks (RNNs and LSTMs)**: Maintained a recurrent hidden vector updated at each step:
      `h_t = tanh(W_h * h_{t-1} + W_x * x_t)`
      Suffered from vanishing gradients, context amnesia after ~100 tokens, and severe repetition loops during open-ended generation.
    * **Greedy Search Degradation**: Always selecting the argmax token rapidly locks generation into pathological degeneration (e.g., repeating the same sentence indefinitely).
    """)

with col_gen_b:
    st.markdown("#### Modern Controlled Decoding Strategies")
    st.markdown("""
    Modern Foundation LLMs use Transformer self-attention paired with sophisticated sampling strategies:
    
    * **Temperature (T)**: Rescales raw logit scores before the Softmax function:
      `P(w_i) = exp(z_i / T) / Sum_j exp(z_j / T)`
      * Low Temperature (T = 0.1): Sharpens the distribution; deterministic, ideal for coding and classification.
      * High Temperature (T = 0.8): Flattens the distribution; creative and diverse prose.
    * **Top-k Sampling**: Truncates the sampling pool strictly to the top k most probable tokens, zeroing out the remaining vocabulary.
    * **Top-p (Nucleus) Sampling**: Dynamically truncates the candidate pool to the smallest set of tokens whose cumulative probability exceeds threshold p (e.g., p = 0.9).
    * **Min-p Sampling**: Filters out any token whose probability is less than a fraction p of the highest candidate token's probability.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
All modern generative foundation models (GPT-4, Claude 3.5, Llama 3) operate on autoregressive causal self-attention. 
During inference, high-performance engines (such as vLLM and TensorRT-LLM) employ **PagedAttention** to store and manage 
Key-Value (KV) cache tensors in GPU memory, avoiding redundant recomputation of historical context tokens.
""")

gen_comp_df = pd.DataFrame({
    "Decoding Strategy": [
        "Greedy Search",
        "Beam Search",
        "Temperature Sampling",
        "Top-p (Nucleus) Sampling",
        "Min-p Sampling"
    ],
    "Core Mechanism": [
        "Selects argmax token at every step",
        "Tracks top B candidate hypotheses across steps",
        "Divides logits by temperature scalar T before Softmax",
        "Samples from smallest token set with sum(P) >= p",
        "Prunes tokens with P < min_p * max(P)"
    ],
    "Diversity vs. Coherence": [
        "High coherence, zero diversity, prone to repetition loops",
        "High quality for translation, but unnatural for dialogue",
        "Tunable; low T is deterministic, high T is creative",
        "Dynamic adaptation; scales pool based on model confidence",
        "Superior balance; preserves long tail without low-quality tokens"
    ],
    "Production Best Use Case": [
        "Structured JSON extraction, unit test generation",
        "Machine translation, short summary extraction",
        "General-purpose chatbot dialogue",
        "Creative writing, brainstorming, synthesis",
        "Modern frontier model default sampling"
    ]
})
st.dataframe(gen_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: Machine Translation
# ==============================================================================
st.header("9. Machine Translation: From Statistical Phrase Tables to Polyglot Foundation LLMs", divider="rainbow")

st.markdown("""
Machine Translation (MT) automates the translation of text from a source human language to a target human language 
while preserving semantic intent, tone, cultural idiom, and technical domain terminology.
""")

col_mt_a, col_mt_b = st.columns(2)

with col_mt_a:
    st.markdown("#### The Evolution of Machine Translation")
    st.markdown("""
    * **Statistical Machine Translation (SMT / Moses)**:
      * Decomposed translation into two probability distributions:
        `Translation = argmax_E P(E | F) = argmax_E P(F | E) * P(E)`
        Where $P(F | E)$ was a phrase translation table and $P(E)$ was an N-gram language model.
      * Produced fragmented, literal word-for-word translations lacking grammatical fluency.
    * **Neural Machine Translation (NMT / MarianMT, Seq2Seq)**:
      * Encoder-Decoder architectures (e.g. Google NMT, BART, MarianMT) mapped source sentences into continuous vectors and decoded target sentences.
      * **The Pairwise Explosion Problem**: Supporting L languages required training and maintaining L * (L - 1) separate translation models, or routing all pairs through an intermediate English pivot (e.g., German -> English -> Japanese), causing severe compounding translation errors.
    """)

with col_mt_b:
    st.markdown("#### Polyglot Foundation LLMs")
    st.markdown("""
    * **Unified Multilingual Latent Space**:
      * Foundation models are pre-trained on trillions of multilingual tokens simultaneously.
      * Languages share a single high-dimensional geometric manifold in the Transformer's residual stream. Concepts (like "freedom" or "gravity") exist as shared geometric coordinates across English, Spanish, Mandarin, and Hindi.
    * **Zero-Shot Direct Translation**:
      * LLMs translate directly between arbitrary non-English language pairs without requiring an English pivot step.
      * Conditioned via natural language prompting:
        `"Translate the following legal clause from German to Japanese, preserving formal business honorifics: [Text]"`
    * **Cultural and Stylistic Adaptation**:
      * Translates idiomatic expressions naturally rather than producing rigid literal interpretations.
    """)

st.markdown("### 🚀 Modern Foundation LLM Reality: How New Models Follow This")
st.markdown("""
State-of-the-art multilingual LLMs (such as Llama 3 and Mistral Large) exhibit emergent cross-lingual reasoning. 
Because mathematical proofs, programming code, and factual knowledge are shared across languages in pretraining, 
an LLM can reason through a complex logic puzzle presented in Japanese and formulate the solution directly in French.
""")

mt_comp_df = pd.DataFrame({
    "Generation": [
        "Statistical MT (SMT: 1990-2015)",
        "Neural MT (NMT: 2016-2022)",
        "Polyglot Foundation LLM (2023+)"
    ],
    "Underlying Architecture": [
        "Phrase alignment tables + N-gram language models",
        "Dedicated Encoder-Decoder RNN/Transformer (MarianMT)",
        "Massive Autoregressive Decoder Transformer (Llama 3)"
    ],
    "Language Pair Scaling": [
        "Requires separate phrase tables per pair",
        "Requires separate model or English pivot",
        "Single unified model handles all pairs directly"
    ],
    "Cultural Idiom & Tone": [
        "Completely broken; translates verbatim words",
        "Moderate; fluent grammar, but loses subtle idioms",
        "Superhuman; adapts tone, formal honorifics, and dialect"
    ],
    "Context Window": [
        "Single sentence only",
        "1-3 sentences maximum",
        "Full books and documents (32k to 2M tokens)"
    ]
})
st.dataframe(mt_comp_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 10: Production Implementation: Unified Multi-Task Zero-Shot Engine in Python
# ==============================================================================
st.header("10. Production Implementation: Unified Multi-Task Zero-Shot Engine in Python", divider="rainbow")

st.markdown("""
The complete, self-contained Python script below demonstrates how a single foundation text-to-text architecture 
replaces dozens of bespoke models. It implements prompt synthesis, simulated constrained logit extraction, 
and structured schema validation for multiple foundational NLP tasks:
""")

st.code('''from typing import Dict, Any, List, Optional
import json

class UnifiedFoundationTaskEngine:
    """
    Demonstrates the unified Text-to-Text foundation model paradigm.
    Replaces separate legacy classification heads, CRF sequence taggers, 
    and extractive span heads with a single autoregressive prompt interface.
    """
    
    def __init__(self, model_name: str = "foundation-llm-70b"):
        self.model_name = model_name
        self.supported_tasks = [
            "CLASSIFICATION", "SENTIMENT", "NER", "QA", 
            "SIMILARITY", "SEARCH", "GENERATION", "TRANSLATION"
        ]

    def build_task_prompt(self, task_type: str, text: str, context: str = "", target_lang: str = "Spanish") -> str:
        """
        Formats instructions and inputs into an autoregressive prompt conditioning stream.
        """
        task_type = task_type.upper()
        if task_type == "CLASSIFICATION":
            return (
                f"System: You are an enterprise classifier. Output exactly one category from [URGENT, NORMAL, LOW].\\n"
                f"Input: {text}\\n"
                f"Category:"
            )
        elif task_type == "SENTIMENT":
            return (
                f"System: Perform aspect-based sentiment analysis. Output JSON with fields 'polarity' and 'aspects'.\\n"
                f"Review: {text}\\n"
                f"JSON Output:"
            )
        elif task_type == "NER":
            return (
                f"System: Extract all named entities conforming to schema: [{{\\"entity\\": str, \\"type\\": str}}].\\n"
                f"Text: {text}\\n"
                f"Entities JSON:"
            )
        elif task_type == "QA":
            return (
                f"System: Answer the question using ONLY the provided grounding context.\\n"
                f"Context: {context}\\n"
                f"Question: {text}\\n"
                f"Answer:"
            )
        elif task_type == "SIMILARITY":
            return (
                f"System: Evaluate semantic equivalence between Passage A and Passage B from 0.0 to 1.0.\\n"
                f"Passage A: {text}\\n"
                f"Passage B: {context}\\n"
                f"Score and Justification JSON:"
            )
        elif task_type == "SEARCH":
            return (
                f"System: Act as an expert search re-ranker. Given a query and candidate document, score relevance 0-100.\\n"
                f"Query: {text}\\n"
                f"Candidate: {context}\\n"
                f"Relevance Score:"
            )
        elif task_type == "GENERATION":
            return (
                f"System: Write a concise, professional executive brief continuing from the prompt.\\n"
                f"Prompt: {text}\\n"
                f"Executive Brief:"
            )
        elif task_type == "TRANSLATION":
            return (
                f"System: Translate the following text into {target_lang}, preserving professional tone.\\n"
                f"Source: {text}\\n"
                f"Translation:"
            )
        else:
            raise ValueError(f"Unsupported task: {task_type}")

    def execute_task(self, task_type: str, text: str, context: str = "", target_lang: str = "Spanish") -> Dict[str, Any]:
        """
        Simulates foundation model inference across distinct NLP task domains.
        In production, this dispatches to an LLM inference runtime (vLLM / TensorRT-LLM).
        """
        prompt = self.build_task_prompt(task_type, text, context, target_lang)
        
        # Deterministic simulation of unified foundation outputs
        raw_output: str = ""
        structured_payload: Optional[Dict[str, Any]] = None
        
        task_type = task_type.upper()
        if task_type == "CLASSIFICATION":
            raw_output = "URGENT" if any(k in text.lower() for k in ["asap", "down", "outage", "emergency"]) else "NORMAL"
            structured_payload = {"predicted_class": raw_output, "confidence": 0.94}
            
        elif task_type == "SENTIMENT":
            is_neg = any(w in text.lower() for w in ["terrible", "slow", "broken", "bug", "crash"])
            polarity = "NEGATIVE" if is_neg else "POSITIVE"
            raw_output = json.dumps({"polarity": polarity, "aspects": [{"aspect": "General Experience", "sentiment": polarity}]})
            structured_payload = json.loads(raw_output)
            
        elif task_type == "NER":
            raw_output = json.dumps({
                "entities": [
                    {"entity": "Apple Inc.", "type": "ORGANIZATION"},
                    {"entity": "Tim Cook", "type": "PERSON"},
                    {"entity": "$2.5 Billion", "type": "MONETARY_VALUE"}
                ]
            })
            structured_payload = json.loads(raw_output)
            
        elif task_type == "QA":
            raw_output = f"Grounded response: The evidence confirmed that {text.lower().rstrip('?')} is supported by the context."
            structured_payload = {"answer": raw_output, "grounding_verified": True}
            
        elif task_type == "SIMILARITY":
            raw_output = json.dumps({"similarity_score": 0.88, "rationale": "High conceptual alignment with minor lexical variance."})
            structured_payload = json.loads(raw_output)
            
        elif task_type == "SEARCH":
            raw_output = "94"
            structured_payload = {"relevance_score": 94, "verdict": "STRONG_MATCH"}
            
        elif task_type == "GENERATION":
            raw_output = f"Executive Summary: Following up on '{text}', operational readiness has been achieved across all regions."
            structured_payload = {"text": raw_output}
            
        elif task_type == "TRANSLATION":
            translations = {
                "Spanish": "El rendimiento operativo superó todas las expectativas proyectadas.",
                "German": "Die Betriebsleistung übertraf alle prognostizierten Erwartungen.",
                "French": "La performance opérationnelle a dépassé toutes les prévisions attendues."
            }
            raw_output = translations.get(target_lang, "Operative Leistung hervorragend.")
            structured_payload = {"translated_text": raw_output, "target_language": target_lang}
            
        return {
            "model": self.model_name,
            "task_type": task_type,
            "prompt_length_chars": len(prompt),
            "generated_tokens_approx": len(raw_output.split()),
            "synthesized_prompt": prompt,
            "raw_output": raw_output,
            "structured_payload": structured_payload
        }

# Execution verification
if __name__ == "__main__":
    engine = UnifiedFoundationTaskEngine()
    result = engine.execute_task(
        task_type="CLASSIFICATION",
        text="Production database outage detected in us-east-1 region ASAP!"
    )
    print("Task Result:", json.dumps(result, indent=2))
''', language="python")

st.divider()

# ==============================================================================
# Section 11: Interactive Workbench: NLP Task Simulator & Generalization Playground
# ==============================================================================
st.header("11. Interactive Workbench: Multi-Task Inference & Generalization Simulator", divider="rainbow")

st.markdown("""
Test how a single input sentence is processed across the 8 foundational NLP task domains using the unified 
text-to-text paradigm. Adjust the parameters below to observe prompt synthesis, simulated logit scoring, 
and structured output generation:
""")

col_wb_ctrl1, col_wb_ctrl2 = st.columns([1.2, 0.8])

with col_wb_ctrl1:
    user_input_text = st.text_area(
        "Enter Primary Input Text:",
        value="Tesla announced Q3 automotive revenues of $25.18 billion, outperforming Wall Street consensus estimates by 4.2%!",
        height=100
    )

with col_wb_ctrl2:
    selected_task_domain = st.selectbox(
        "Select Foundational NLP Task Domain:",
        [
            "Text Classification",
            "Sentiment Analysis",
            "Named Entity Recognition (NER)",
            "Question Answering",
            "Text Similarity",
            "Semantic Search",
            "Text Generation",
            "Machine Translation"
        ]
    )
    
    decoding_temp = st.slider("Sampling Temperature (T)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

# Optional context for QA and Similarity
context_text = ""
target_language = "Spanish"

if selected_task_domain in ["Question Answering", "Text Similarity", "Semantic Search"]:
    context_text = st.text_input(
        "Grounding Context / Comparison Passage:",
        value="Tesla Motors reported quarterly financial results exceeding analyst forecasts across both automotive gross margins and total net income."
    )
elif selected_task_domain == "Machine Translation":
    target_language = st.selectbox("Target Translation Language:", ["Spanish", "German", "French", "Japanese"])

# Execution simulation
task_mapping = {
    "Text Classification": "CLASSIFICATION",
    "Sentiment Analysis": "SENTIMENT",
    "Named Entity Recognition (NER)": "NER",
    "Question Answering": "QA",
    "Text Similarity": "SIMILARITY",
    "Semantic Search": "SEARCH",
    "Text Generation": "GENERATION",
    "Machine Translation": "TRANSLATION"
}

mapped_task = task_mapping[selected_task_domain]

# Simulation logic
def simulate_workbench(task_key: str, text: str, ctx: str, lang: str) -> Dict[str, Any]:
    if task_key == "CLASSIFICATION":
        prompt = f"System: Classify text into [FINANCIAL_REPORT, TECH_NEWS, GENERAL].\\nInput: {text}\\nLabel:"
        output = "FINANCIAL_REPORT"
        metrics = {"Legacy Architecture": "RoBERTa-base + Linear Head", "Legacy Latency": "18 ms", "LLM Latency": "45 ms (1 token)"}
        payload = {"assigned_label": output, "softmax_logit_prob": 0.982}
    elif task_key == "SENTIMENT":
        prompt = f"System: Analyze sentiment and aspects.\\nReview: {text}\\nJSON:"
        payload = {"polarity": "POSITIVE", "score": 0.91, "aspects": [{"aspect": "Revenue", "tone": "Exceptional"}]}
        output = '{"polarity": "POSITIVE", "score": 0.91, "aspects": [{"aspect": "Revenue", "tone": "Exceptional"}]}'
        metrics = {"Legacy Architecture": "DistilBERT 3-class classifier", "Legacy Latency": "15 ms", "LLM Latency": "110 ms"}
    elif task_key == "NER":
        prompt = f"System: Extract entities as JSON.\\nText: {text}\\nJSON:"
        payload = {"entities": [{"text": "Tesla", "type": "ORG"}, {"text": "$25.18 billion", "type": "MONEY"}, {"text": "Q3", "type": "DATE"}, {"text": "Wall Street", "type": "ORG"}]}
        output = '{"entities": [{"text": "Tesla", "type": "ORG"}, {"text": "$25.18 billion", "type": "MONEY"}, {"text": "Q3", "type": "DATE"}, {"text": "Wall Street", "type": "ORG"}]}'
        metrics = {"Legacy Architecture": "BiLSTM + CRF layer", "Legacy Latency": "22 ms", "LLM Latency": "145 ms"}
    elif task_key == "QA":
        prompt = f"System: Answer question grounded in context.\\nContext: {ctx}\\nQuestion: {text}\\nAnswer:"
        output = "According to the financial release, Tesla automotive revenues reached $25.18 billion, beating analyst estimates by 4.2%."
        metrics = {"Legacy Architecture": "BERT Span Head (SQuAD)", "Legacy Latency": "30 ms", "LLM Latency": "180 ms"}
        payload = {"grounded_answer": output, "citation_span": "0:85"}
    elif task_key == "SIMILARITY":
        prompt = f"System: Score semantic similarity 0.0-1.0.\\nText A: {text}\\nText B: {ctx}\\nJSON:"
        payload = {"similarity_score": 0.92, "semantic_alignment": "High - both passages discuss Tesla exceeding financial expectations"}
        output = '{"similarity_score": 0.92, "semantic_alignment": "High - both passages discuss Tesla exceeding financial expectations"}'
        metrics = {"Legacy Architecture": "Sentence-BERT Bi-Encoder", "Legacy Latency": "12 ms", "LLM Latency": "125 ms"}
    elif task_key == "SEARCH":
        prompt = f"System: Re-rank candidate passage for query.\\nQuery: {text}\\nCandidate: {ctx}\\nRelevance:"
        output = "Relevance Score: 96 / 100 (Direct topical hit for automotive financial metrics)"
        metrics = {"Legacy Architecture": "BM25 + MonoBERT Re-ranker", "Legacy Latency": "40 ms", "LLM Latency": "95 ms"}
        payload = {"relevance_score": 96, "action": "INCLUDE_IN_TOP_3"}
    elif task_key == "GENERATION":
        prompt = f"System: Complete the executive brief continuation.\\nInput: {text}\\nContinuation:"
        output = "This impressive financial performance reflects continuous manufacturing efficiencies and strong delivery momentum heading into the fourth quarter."
        metrics = {"Legacy Architecture": "GPT-2 Medium with Beam Search", "Legacy Latency": "250 ms", "LLM Latency": "190 ms"}
        payload = {"generated_text": output}
    else:
        prompt = f"System: Translate to {lang}.\\nSource: {text}\\nTranslation:"
        trans_dict = {
            "Spanish": "¡Tesla anunció ingresos automotrices en el tercer trimestre de $25.18 mil millones de dólares, superando las estimaciones de Wall Street en un 4.2%!",
            "German": "Tesla meldete für das 3. Quartal einen Automobilumsatz von 25,18 Milliarden US-Dollar und übertraf damit die Schätzungen der Wall Street um 4,2 %!",
            "French": "Tesla a annoncé des revenus automobiles de 25,18 milliards de dollars pour le troisième trimestre, dépassant les prévisions de Wall Street de 4,2 % !",
            "Japanese": "テスラは第3四半期の自動車売上高が251億8,000万ドルに達し、ウォール街のコンセンサス予想を4.2%上回ったと発表しました。"
        }
        output = trans_dict.get(lang, "Translated text.")
        metrics = {"Legacy Architecture": "MarianMT Opus-MT (Pairwise)", "Legacy Latency": "45 ms", "LLM Latency": "160 ms"}
        payload = {"translated": output}
        
    return {"prompt": prompt, "output": output, "metrics": metrics, "payload": payload}

wb_res = simulate_workbench(mapped_task, user_input_text, context_text, target_language)

st.markdown("#### Execution Pipeline Diagnostic Output")

col_res1, col_res2 = st.columns([1.1, 0.9])

with col_res1:
    st.markdown("**Synthesized Foundation Prompt Stream:**")
    st.code(wb_res["prompt"], language="text")
    st.markdown("**Raw Autoregressive Generative Output:**")
    st.code(wb_res["output"], language="json" if "{" in wb_res["output"] else "text")

with col_res2:
    st.markdown("**Parsed Structured Payload:**")
    st.json(wb_res["payload"])
    st.markdown("**Comparative Architecture Performance Profile:**")
    perf_df = pd.DataFrame([wb_res["metrics"]]).T.reset_index()
    perf_df.columns = ["Metric Dimension", "Profile Value"]
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 12: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("""
Mastering the unified NLP tasks paradigm requires understanding both theoretical model formulation 
and real-world production traps when migrating from legacy pipelines to foundation LLMs.
""")

st.subheader("Principal / Staff-Level Interview Questions")

with st.expander("Q1: How do you perform deterministic classification with an autoregressive LLM without risking string hallucination?"):
    st.markdown("""
    **Direct Answer:**
    Rather than letting the LLM generate unconstrained open-ended text strings, production pipelines use **Constrained Logit Evaluation (Logit Bias / Logit Masking)**:
    
    1. Identify the exact token IDs representing the candidate class labels (e.g., Token ID for `' SPAM'` and Token ID for `' HAM'`).
    2. Intercept the unnormalized logits at the very first generated token position: `z = Logits[0]`.
    3. Compute the Softmax exclusively across the subset of valid candidate tokens:
       `P(SPAM) = exp(z_SPAM) / (exp(z_SPAM) + exp(z_HAM))`
    4. Apply an argmax or decision threshold on this probability.
    
    **Architectural Impact**: This guarantees 100% deterministic label assignment, runs in a single forward pass step (O(1) decoding tokens), completely prevents hallucinated labels, and provides well-calibrated confidence scores.
    """)

with st.expander("Q2: Why did foundation models replace extractive span-pointer heads with abstractive generation in Question Answering?"):
    st.markdown("""
    **Direct Answer:**
    Extractive span heads (like BERT on SQuAD) are fundamentally constrained by the assumption that the exact answer exists as a single contiguous verbatim substring in the passage:
    
    1. **Multi-Hop Synthesis**: In real-world enterprise documents, the answer to a question like *"Did gross margins improve faster than operating expenses?"* requires comparing numbers across two distinct tables in different chapters. An extractive model cannot answer this because the comparison sentence does not exist verbatim.
    2. **Paraphrasing and Normalization**: Real users need clear, conversational summaries rather than disjoint sentence fragments.
    3. **Graceful Failure**: When information is absent, extractive heads often return high-scoring irrelevant spans. Generative LLMs can be prompted to reason over the absence of evidence and explicitly state when information is missing.
    """)

with st.expander("Q3: When should you use a Siamese Bi-Encoder vs. a Cross-Encoder vs. an LLM for semantic textual similarity?"):
    st.markdown("""
    **Direct Answer:**
    The selection depends strictly on the Pareto frontier between **computational complexity** and **interaction capacity**:
    
    * **Siamese Bi-Encoder (Sentence-BERT)**: Embeds passages independently. Scales to billions of documents via Approximate Nearest Neighbor (ANN / HNSW) indexing. Used for the initial **retrieval stage** (Top-1000 from 10,000,000 documents) in sub-10ms.
    * **Cross-Encoder**: Feeds `(Query, Passage)` simultaneously into full cross-attention. Captures intricate token-to-token interactions but scales as O(N) forward passes. Used for **re-ranking** the top-100 down to top-10.
    * **Foundation LLM (LLM-as-a-Judge)**: Capable of multi-step logical deduction, detecting contradictions, and evaluating nuanced legal/contractual semantics. Too expensive for retrieval; used for final **semantic arbitration** or generating executive comparisons on the final 1-3 candidates.
    """)

with st.expander("Q4: Why does BIO sequence tagging with CRFs fail on nested entities, and how do LLMs resolve this?"):
    st.markdown("""
    **Direct Answer:**
    In a standard BIO tagging sequence, each token position is assigned a single discrete tag from the label set (such as B-PER, I-PER, or O). 
    Because a scalar token index cannot belong to two tags simultaneously, a string like *"Bank of America"* cannot be tagged as an `ORGANIZATION` while simultaneously recognizing *"America"* as a `GPE/LOCATION`.
    
    Foundation models bypass sequential token labeling completely by casting entity extraction as a **structured JSON schema generation problem**. By generating a JSON payload or conforming to a Pydantic schema, nested objects, parent-child hierarchies, and discontinuous entity mentions are natively supported as standard object trees without topological restrictions.
    """)

with st.expander("Q5: What is the primary operational trade-off of replacing 20 fine-tuned BERT models with a single unified 70B Foundation LLM?"):
    st.markdown("""
    **Direct Answer:**
    The trade-off balances **development velocity and task flexibility** against **inference latency, operational compute cost, and serving throughput**:
    
    * **Advantages of Unified LLM**: Zero labeled training data required (zero-shot/few-shot), single serving infrastructure to maintain, instantaneous deployment of new tasks via prompt changes, and superior generalization on long-tail inputs.
    * **Disadvantages of Unified LLM**: A 70B model requires multiple enterprise GPUs (e.g. 2x NVIDIA A100/H100) and incurs 50ms-200ms of latency per query, compared to a fine-tuned DistilBERT model that runs on a cheap single CPU core in 5ms with predictable microsecond tail latencies.
    * **Modern Best Practice**: Use the Foundation LLM to annotate and distill data, then train small specialized student models for high-QPS, latency-critical point tasks.
    """)

st.subheader("Production Engineering Traps")

st.error("""
🚨 **Trap 1 (Error): Stochastic Label Drift & Hallucination in Classification Prompts**
When using open-ended text generation for classification, temperature settings > 0.0 or ambiguous system prompts cause the LLM to output unexpected text variations (e.g., generating `"The label is: SPAM!"` or `"Likely spam"` instead of the required `"SPAM"`). 
*Remediation*: Always evaluate first-token logits using logit bias / logit masking, or enforce strict grammar-based constrained sampling (e.g., via Outlines, Guidance, or OpenAI JSON Schema mode).
""")

st.error("""
🚨 **Trap 2 (Error): Span Index Desynchronization in Extractive-to-Abstractive Migration**
Migrating downstream document annotation systems from extractive QA (which emits character start and end offsets `[124:158]`) to abstractive LLM generation without updating downstream databases will corrupt UI highlighting. Abstractive LLMs generate novel token sequences that do not exist in the source document.
*Remediation*: If exact document highlighting is required, prompt the LLM to return the exact verbatim quote alongside its abstractive answer, then programmatically fuzzy-match the quote back into the source text.
""")

st.warning("""
⚠️ **Trap 3 (Warning): The Bi-Encoder Information Compression Bottleneck**
Relying solely on cosine similarity over dense bi-encoder embeddings (e.g., Sentence-BERT) for fine-grained similarity tasks often fails when sentences differ by subtle negations or single numerical quantities (e.g., *"Revenue increased by 15%"* vs. *"Revenue increased by 50%"*). Dense embeddings project both sentences to nearly identical vectors because 95% of the lexical context is shared.
*Remediation*: Always augment dense retrieval with a cross-encoder re-ranker or an LLM semantic verification pass for numerical and negation validation.
""")

st.warning("""
⚠️ **Trap 4 (Warning): The Compounding Cost & Latency of Using Frontier LLMs for High-QPS Point Tasks**
Replacing simple regex or lightweight classification models (like spam filtering or language detection at 10,000 queries per second) with frontier 70B LLM API calls introduces massive operational costs ($10,000s/month) and unacceptable 200ms latency bottlenecks.
*Remediation*: Employ the **Two-Tier Architecture**: route 95% of trivial high-volume traffic through optimized lightweight models (FastText, DistilBERT), routing only the ambiguous 5% edge cases to the Foundation LLM.
""")

st.subheader("Concept Flashcards")

col_fc1, col_fc2 = st.columns(2)

with col_fc1:
    with st.container():
        st.info("**Text-to-Text Paradigm**")
        st.caption("The unified architectural formulation where every NLP problem (classification, extraction, translation, QA) is framed as consuming an input string and autoregressively generating an output string.")

    with st.container():
        st.info("**Logit Masking / Logit Bias**")
        st.caption("A production technique that constrains the LLM sampling distribution to a predetermined subset of token IDs, enabling deterministic classification and schema enforcement.")

with col_fc2:
    with st.container():
        st.info("**Extractive vs. Abstractive QA**")
        st.caption("Extractive QA predicts start and end index pointers to slice verbatim context spans; Abstractive QA synthesizes novel natural language paragraphs grounded in retrieved evidence.")

    with st.container():
        st.info("**Bi-Encoder vs. Cross-Encoder**")
        st.caption("Bi-encoders map texts independently to dense vectors for sub-10ms MIPS retrieval; Cross-encoders process texts jointly with full cross-attention for high-accuracy re-ranking.")

st.divider()

# ==============================================================================
# Section 13: Self-Assessment Checklist
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of foundational NLP tasks and the modern text-to-text paradigm shift. 
Complete all 6 checklist items:
""")

cb1 = st.checkbox("1. I understand how the Text-to-Text paradigm unifies disparate NLP tasks into an autoregressive causal generation interface.", key="chk_23_1")
cb2 = st.checkbox("2. I can explain how first-token logit scoring enables deterministic text classification without risk of string hallucination.", key="chk_23_2")
cb3 = st.checkbox("3. I know why BIO sequence tagging with CRFs fails on nested entities and how schema-constrained decoding solves this.", key="chk_23_3")
cb4 = st.checkbox("4. I understand the architectural limitations of extractive span-pointer QA and why modern RAG uses abstractive synthesis.", key="chk_23_4")
cb5 = st.checkbox("5. I can formulate the multi-stage search pipeline combining Bi-Encoder retrieval, Cross-Encoder re-ranking, and LLM arbitration.", key="chk_23_5")
cb6 = st.checkbox("6. I understand controlled decoding strategies (Temperature, Top-p, Min-p) and how PagedAttention accelerates text generation.", key="chk_23_6")

if cb1 and cb2 and cb3 and cb4 and cb5 and cb6:
    st.success("🎉 Outstanding! You have mastered all 8 foundational NLP tasks and the modern Foundation LLM text-to-text architectural paradigm!")
