import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="1.2 Why LLMs are Important - LLM Curriculum",
    page_icon="🌍",
    layout="wide"
)

# ==============================================================================
# Title Block & Overview
# ==============================================================================
st.title("1.2 Why Large Language Models are Important")
st.caption("Chapter 1: Introduction to Large Language Models | Domain: 2. Large Language Models")

st.markdown("""
Large Language Models represent a generational paradigm shift in computer science: **they serve as universal 
software engines programmed in natural language**. Prior to foundation models, extracting intelligence from text 
required assembling fragile multi-stage machine learning pipelines—combining tokenizers, POS taggers, dependency parsers, 
named entity recognizers, and task-specific classification heads.

Foundation models collapse these disparate, brittle systems into a single end-to-end differentiable neural architecture. 
By scaling parameter count, training compute, and token volume, LLMs display emergent generalisation across text generation, 
deep question answering, context summarization, polyglot translation, structured information extraction, code synthesis, 
symbolic multi-step reasoning, and stateful conversational AI.

In this lesson, you will master:
1. The 8 foundational capability pillars that power modern enterprise LLM deployments.
2. The architectural transition from brittle multi-stage pipelines to unified foundation models.
3. How modern foundation models (Llama 3, GPT-4, Mistral) follow each capability pillar in production.
4. Enterprise unit economics, labor substitution frontiers, and automated ROI calculation.
5. Production failure modes including sandbox escapes, unconstrained decoding traps, and context window explosion.
""")

st.divider()

# ==============================================================================
# Section 1: End-to-End Capabilities Blueprint & Operational Taxonomy
# ==============================================================================
st.header("1. End-to-End Capabilities Blueprint & Operational Taxonomy", divider="rainbow")

st.markdown("""
Every real-world enterprise LLM workload maps to one or more of eight fundamental capability pillars. 
Modern foundation models do not merely match surface-level text patterns; they project unstructured human thought 
into structured, actionable computational artifacts across diverse modalities and symbolic domains.
""")

col_arch1, col_arch2 = st.columns([1.3, 0.7])

with col_arch1:
    dot_taxonomy = """
    digraph CapabilityTaxonomy {
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

        subgraph cluster_inputs {
            label="Input Modalities and Context";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";
            In_Text [label="Unstructured Text\\n(Memos, PDFs, Transcripts)", fillcolor="#1E293B"];
            In_Code [label="Source Code & Schema\\n(SQL, Python, JSON)", fillcolor="#1E293B"];
            In_Query [label="User Dialog & Query\\n(Multi-Turn Prompts)", fillcolor="#1E293B"];
        }

        subgraph cluster_core {
            label="Foundation LLM Core Engine";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            Core_LLM [
                label="Decoder-Only Transformer\\n(RoPE, SwiGLU, RMSNorm)\\n32k to 2M Token Context\\nWeights + KV-Cache",
                fillcolor="#0F172A",
                color="#818CF8",
                shape=hexagon,
                penwidth=2.2,
                fontsize=11
            ];
        }

        subgraph cluster_synthesis {
            label="Synthesis Tier";
            style=filled;
            color="#1E293B";
            fontcolor="#34D399";
            P1 [label="1. Text Generation\\n(Autoregressive Sampling)", fillcolor="#064E3B", color="#34D399"];
            P3 [label="3. Summarization\\n(Abstractive Compaction)", fillcolor="#064E3B", color="#34D399"];
            P8 [label="8. Conversational AI\\n(Multi-Turn Persona)", fillcolor="#064E3B", color="#34D399"];
        }

        subgraph cluster_grounding {
            label="Grounding and Transfer Tier";
            style=filled;
            color="#1E293B";
            fontcolor="#38BDF8";
            P2 [label="2. Question Answering\\n(Grounded RAG / Tools)", fillcolor="#0C4A6E", color="#38BDF8"];
            P4 [label="4. Translation\\n(Cross-Lingual Latent Space)", fillcolor="#0C4A6E", color="#38BDF8"];
        }

        subgraph cluster_symbolic {
            label="Symbolic and Execution Tier";
            style=filled;
            color="#1E293B";
            fontcolor="#F59E0B";
            P5 [label="5. Information Extraction\\n(Grammar-Guided JSON)", fillcolor="#78350F", color="#F59E0B"];
            P6 [label="6. Code Generation\\n(FIM, AST Syntax)", fillcolor="#78350F", color="#F59E0B"];
            P7 [label="7. Reasoning & Planning\\n(CoT / Test-Time Search)", fillcolor="#78350F", color="#F59E0B"];
        }

        In_Text -> Core_LLM;
        In_Code -> Core_LLM;
        In_Query -> Core_LLM;

        Core_LLM -> P1;
        Core_LLM -> P2;
        Core_LLM -> P3;
        Core_LLM -> P4;
        Core_LLM -> P5;
        Core_LLM -> P6;
        Core_LLM -> P7;
        Core_LLM -> P8;
    }
    """
    with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
        st.graphviz_chart(dot_taxonomy, use_container_width=True)

with col_arch2:
    st.markdown("### The 8 Capability Pillars at a Glance")
    st.markdown("""
    * **1. Text Generation**: Autoregressive next-token sampling with KV-cache and speculative decoding.
    * **2. Question Answering**: Closed-book parametric knowledge vs open-book vector-grounded RAG.
    * **3. Summarization**: Abstractive synthesis overcoming context bottlenecks via map-reduce patterns.
    * **4. Translation**: Unified cross-lingual representation across 100+ native languages without pivot.
    * **5. Information Extraction**: Zero-shot structured outputs guaranteed via grammar-constrained decoding.
    * **6. Code Generation**: Fill-In-The-Middle (FIM) and automated unit-test debug correction loops.
    * **7. Reasoning**: Test-time compute scaling, Chain-of-Thought (CoT), and search trees.
    * **8. Conversational AI**: Dynamic steerability, conversational state tracking, and safety firewalls.
    """)

st.divider()

# ==============================================================================
# Section 2: 1. Text Generation
# ==============================================================================
st.header("2. Text Generation", divider="rainbow")

st.markdown("""
Text generation is the core operational primitive of every decoder-only Large Language Model. 
Given a prefix sequence of prompt tokens, the model computes an unnormalized vector of logits across the entire vocabulary 
`[Vocabulary_Size]`, normalizes them into a categorical probability distribution using the Softmax function, 
and samples the next token. This process repeats autoregressively until an End-Of-Sequence (`<|eos|>`) token is emitted.

### Mathematical Formulation of Autoregressive Sampling
The joint probability of generating a text sequence of length T decomposes into the product of conditional probabilities:

`P(w_1, w_2, ..., w_T) = Product(t=1..T) P(w_t | w_1, ..., w_{t-1})`

At each generation step `t`:
1. **Logit Projection**: `z_t = W_unembed * h_t`, where `h_t` has dimension `[Batch_Size, Hidden_Dimension]` and `z_t` has dimension `[Batch_Size, Vocabulary_Size]`.
2. **Temperature Scaling**: `z_t_scaled = z_t / Temperature`. As `Temperature -> 0`, the distribution approaches greedy argmax; as `Temperature -> 1.0+`, entropy increases.
3. **Softmax Normalization**: `P(w_t = v) = exp(z_{t,v} / T) / Sum_{j} exp(z_{t,j} / T)`.
4. **Nucleus (Top-p) Filtering**: Restricts sampling to the smallest subset of vocabulary tokens whose cumulative probability satisfies `Sum P(w_t) >= p` (typically `p = 0.9`).
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **KV-Caching**: Older autoregressive scripts recomputed the entire forward pass for all previous tokens at each step (`O(N^2)` compute). Modern foundation models (Llama 3, Mistral, GPT-4) cache Key and Value tensors in GPU HBM (`[Num_Layers, Num_KV_Heads, Sequence_Length, Head_Dim]`), ensuring each new token requires strictly `O(1)` attention compute per layer.
* **Speculative Decoding**: Instead of running a massive 70B parameter model token-by-token, modern inference runtimes pair a small draft model (e.g., Llama-3-8B) that speculates 4 to 8 tokens rapidly, with the 70B target model verifying all tokens simultaneously in a single forward pass—achieving 2x to 3x throughput speedups with mathematically identical output distributions.
* **Chunked Prefill & FlashAttention-3**: Modern serving frameworks (vLLM, SGLang, TensorRT-LLM) split large prompt prefill into chunks interleaved with decoding steps to preserve strict Time-To-First-Token (TTFT) and Inter-Token-Latency (ITL) SLAs.
""")

df_p1 = pd.DataFrame({
    "Dimension": [
        "Generation Paradigm",
        "Context Memory",
        "Sampling Technique",
        "Inference Acceleration",
        "Output Consistency"
    ],
    "Legacy NLP (Markov / RNN / LSTM)": [
        "N-gram frequency count or hidden recurrence state h_t",
        "Limited to 3-5 preceding words (N-grams) or prone to vanishing gradients",
        "Greedy argmax or naive random beam search",
        "None; recurrent steps executed sequentially on CPU/early GPUs",
        "Severe semantic drift after 20 words; repetitive looping"
    ],
    "Modern Foundation LLMs (Llama 3 / GPT-4)": [
        "Self-attention residual stream over massive causal transformer",
        "32k to 2,000,000 token context window with RoPE positional encoding",
        "Temperature-scaled Top-p nucleus sampling and Min-P filtering",
        "KV-caching, FlashAttention-3, PagedAttention, and Speculative Decoding",
        "Coherent long-form multi-chapter narrative generation with consistent tone"
    ]
})
st.dataframe(df_p1, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: 2. Question Answering
# ==============================================================================
st.header("3. Question Answering", divider="rainbow")

st.markdown("""
Question Answering (QA) represents the primary interface through which humans and enterprise systems extract knowledge from LLMs. 
Architecturally, Question Answering is categorized into two paradigms:
1. **Closed-Book Parametric QA**: The model relies exclusively on facts compressed into its static neural weights during pretraining.
2. **Open-Book Retrieval-Augmented QA (RAG)**: The model is provided dynamic external reference documents inside its inference context window, synthesizing responses strictly grounded in retrieved evidence.

### Parametric Limitations vs Context Grounding
In closed-book QA, knowledge is stored implicitly across feed-forward layer weights (`W_gate`, `W_up`, `W_down`). 
Because weights are frozen at training cut-off, closed-book models suffer from **hallucination** when queried about private corporate data, 
recent events, or fine-grained numerical facts. Open-book systems transform QA from a memory retrieval task into an **in-context reasoning task**.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Hybrid Dense-Sparse RAG**: Modern production architectures pair dense vector embeddings (e.g., `text-embedding-3`, BGE-M3) with sparse lexical search (BM25) and reciprocal rank fusion (RRF), passed through a Cross-Encoder Re-ranker (e.g., Cohere Rerank) before context injection.
* **Inline Verifiable Citations**: Rather than generating unstructured prose, foundation models are instructed via system prompts to emit verifiable bracketed citations (`[Source: Q3_Report.pdf, Page 14]`), where the application runtime verifies that cited claims match retrieved text spans.
* **Self-Correction & Agentic Retrieval**: Advanced systems employ agentic loops where the model evaluates its own retrieval sufficiency: if retrieved documents do not contain the answer, the LLM emits a search tool call with a refined query rather than hallucinating an answer.
""")

df_p2 = pd.DataFrame({
    "Dimension": [
        "Knowledge Source",
        "Update Mechanism",
        "Hallucination Risk",
        "Private Data Access",
        "Auditability"
    ],
    "Legacy QA Systems (BiDAF / BERT SQuAD)": [
        "Fixed reading comprehension over single short paragraph (512 tokens max)",
        "Requires full model fine-tuning and pipeline retraining",
        "High extractive span error; crashes when answer is outside the paragraph",
        "Requires fine-tuning on internal corporate datasets",
        "Provides span indices [start, end] but zero reasoning or cross-document synthesis"
    ],
    "Modern Foundation LLMs + RAG": [
        "Hybrid Vector + Lexical Index over millions of enterprise documents",
        "Instant index update without touching model weights",
        "Near-zero when temperature is 0.0 and prompt mandates strict grounding",
        "Zero-shot access via enterprise RBAC-filtered retrieval pipelines",
        "Explicit sentence-level citations with cryptographic document hash tracing"
    ]
})
st.dataframe(df_p2, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 4: 3. Summarization
# ==============================================================================
st.header("4. Summarization", divider="rainbow")

st.markdown("""
Summarization condenses expansive text documents into concise, salient representations while preserving essential factual integrity. 
There are two core methodologies:
* **Extractive Summarization**: Graph-based or statistical algorithms score sentences in the source text and extract the top-K highest-ranking sentences verbatim.
* **Abstractive Summarization**: The model parses the source text into high-dimensional hidden representations, synthesizes core themes, and generates novel natural language prose that paraphrases, merges, and distills the source material.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Solving the Long-Context Bottleneck**: While early transformers were capped at 512 or 2,048 tokens, modern LLMs handle 128k (Llama 3), 200k (Claude 3.5), and 2,000,000 tokens (Gemini 1.5). For massive multi-volume corpora exceeding 2M tokens, enterprise pipelines deploy **Map-Reduce** and **Refine** chunking patterns.
* **Mitigating 'Lost-in-the-Middle'**: Research proves that LLMs retrieve information at the very beginning and very end of long contexts with much higher fidelity than information in the middle. Modern foundation models mitigate this through RoPE base frequency scaling (`base = 500,000` in Llama 3) and synthetic needle-in-a-haystack pretraining.
* **Constraint-Guided Synthesis**: Modern models accept strict output constraints—such as executive bullet points, 150-word limits, key risk matrices, and action item tables—without losing critical liabilities or quantitative metrics.
""")

df_p3 = pd.DataFrame({
    "Dimension": [
        "Summarization Method",
        "Fluency & Synthesis",
        "Numerical Aggregation",
        "Pronoun & Coreference Ambiguity",
        "Input Length Ceiling"
    ],
    "Legacy Extractive NLP (TextRank / LexRank)": [
        "Extractive sentence graph centrality and TF-IDF overlap",
        "Disjointed; stitches isolated sentences without narrative transitions",
        "Zero capability; cannot sum quarterly metrics or compute growth percentages",
        "Severe; extracts 'He stated that...' without including who 'he' refers to",
        "Constrained by graph compute; typically under 10 pages"
    ],
    "Modern Foundation LLMs": [
        "Abstractive semantic synthesis and cross-paragraph distillation",
        "Human-level executive prose with tailored corporate tone and structure",
        "Capable of reading raw tables, synthesizing trends, and computing deltas",
        "Resolves all coreference chains across 50,000+ words seamlessly",
        "Up to 2,000,000 tokens natively in a single prompt context"
    ]
})
st.dataframe(df_p3, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 5: 4. Translation
# ==============================================================================
st.header("5. Translation", divider="rainbow")

st.markdown("""
Machine translation transforms text from a source language into a target language while preserving semantic meaning, 
syntactic naturalness, tone, and domain-specific terminology. 
Modern Large Language Models achieve cross-lingual capabilities not through language-specific phrase tables, 
but through a **shared high-dimensional semantic manifold**. Because pretraining datasets incorporate trillions of tokens 
spanning dozens of languages, concept representations (e.g., the concept of 'hospital' or 'quantum physics') map to 
overlapping geometric clusters in the transformer residual stream.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Direct Zero-Shot Transfer**: Classical Neural Machine Translation (NMT) required bilingual parallel corpora for every specific language pair (e.g., German to Japanese). If parallel data did not exist, systems translated through an intermediate English pivot, compounding errors. Modern LLMs translate directly between arbitrary language pairs via their internal shared representation.
* **Byte-Level Tokenizer Efficiency**: Earlier tokenizers (WordPiece, early BPE) suffered from severe 'token fertility' penalties on non-Latin scripts (e.g., Hindi, Arabic, Korean), where a single word could require 6 to 10 tokens. Modern foundation tokenizers (such as Llama 3's 128k Tiktoken vocabulary) feature balanced multilingual compression ratios.
* **In-Context Glossary Injection**: Production enterprise translation uses prompt injection to enforce strict corporate terminology (`"Always translate 'Direct Debit' as 'Lastschriftverfahren' in German"`), ensuring zero drift across technical documentation and legal filings.
""")

df_p4 = pd.DataFrame({
    "Dimension": [
        "Architecture",
        "Language Pair Coverage",
        "Context Window Scope",
        "Terminology Steering",
        "Tone and Register Adaptation"
    ],
    "Legacy NMT (MarianMT / Google NMT 2016)": [
        "Dedicated encoder-decoder RNN / Transformer trained per language pair",
        "Requires paired parallel corpora; relies on English pivot for rare pairs",
        "Sentence-by-sentence (50-100 tokens max); no cross-paragraph memory",
        "Requires hardcoded post-processing regex or custom dictionary heads",
        "Rigid output style; cannot switch between casual, legal, or diplomatic tone"
    ],
    "Modern Foundation LLMs": [
        "Unified decoder-only transformer with shared multilingual latent space",
        "Direct any-to-any translation across 100+ languages without intermediate pivot",
        "Full document scope (32k+ tokens); preserves clause definitions across 50 pages",
        "Dynamic prompt injection of terminology glossaries and regulatory rules",
        "Instantly adjusts register (e.g., formal German 'Sie' vs informal 'Du') via prompt"
    ]
})
st.dataframe(df_p4, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 6: 5. Information Extraction
# ==============================================================================
st.header("6. Information Extraction", divider="rainbow")

st.markdown("""
Information Extraction (IE) converts unstructured, free-form text (emails, clinical encounter notes, invoices, contracts) 
into structured relational databases and typed JSON objects. 
Prior to foundation models, information extraction was notoriously brittle, requiring multi-stage pipelines: 
Tokenization -> Part-of-Speech Tagging -> Named Entity Recognition (NER) -> Relation Extraction (RE) -> Regular Expressions. 
A single typographical error or unexpected layout shift caused the entire downstream pipeline to fail.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Grammar-Constrained Decoding**: In production, prompting an LLM with `"Please return JSON"` is an antipattern because the model may output conversational preambles or markdown backticks (` ```json `). Modern inference engines (SGLang, Outlines, llama.cpp, OpenAI Structured Outputs) compile JSON Schemas into Context-Free Grammars (CFGs). At every generation step, the engine masks out all vocabulary tokens that would violate the schema, guaranteeing 100% syntactically valid JSON.
* **Pydantic Model Schema Alignment**: Modern software frameworks (Instructor, LangChain, Pydantic) pass Python class definitions directly to the model as JSON Schema specifications. The model extracts nested arrays, enums, optional fields, and type-validated primitives in a single zero-shot forward pass.
* **Resilience to Document Layout Drift**: Unlike OCR template matchers that fail when a vendor moves an invoice total from the bottom-right to the bottom-left, foundation models parse semantics rather than pixel coordinates.
""")

df_p5 = pd.DataFrame({
    "Dimension": [
        "Pipeline Architecture",
        "Training Data Requirements",
        "Schema Modification Agility",
        "Handling Layout Drift",
        "Syntactic Guarantee"
    ],
    "Legacy Extraction (Spacy NER + Regex)": [
        "Multi-stage pipeline: Tokenizer -> POS -> Dependency -> NER -> Regex",
        "Requires 5,000 to 20,000 manually labeled BIO entity spans",
        "Weeks to months; adding a new field requires re-labeling and re-training",
        "Extremely brittle; breaks on formatting changes, rotated scans, or typos",
        "Regex matching yields frequent null pointers or unhandled format exceptions"
    ],
    "Modern Foundation LLMs + Guided Decoding": [
        "Single unified foundation model with grammar-constrained decoding",
        "Zero-shot or 1 to 3 few-shot exemplars in prompt context",
        "Instant; updating a field takes 10 seconds by editing a Pydantic class",
        "Extremely robust; semantic comprehension ignores cosmetic formatting shifts",
        "100% mathematically guaranteed valid JSON adhering to target schema"
    ]
})
st.dataframe(df_p5, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 7: 6. Code Generation
# ==============================================================================
st.header("7. Code Generation", divider="rainbow")

st.markdown("""
Code generation encompasses synthesizing executable programming code, SQL queries, infrastructure-as-code manifests, 
and unit tests from natural language specifications or code context. 
Programming languages are formal languages characterized by strict syntax, deterministic execution semantics, 
and long-range variable dependencies across file boundaries. LLMs treat code generation as an extension of language modeling, 
trained on hundreds of millions of public and private software repositories.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Fill-In-The-Middle (FIM) Architecture**: Standard autoregressive models generate strictly left-to-right. However, real-world coding assistants (GitHub Copilot, Cursor) require inserting code inside an existing file where both prefix (code above cursor) and suffix (code below cursor) exist. Modern code LLMs (DeepSeek-Coder, StarCoder, Llama 3) are trained with FIM transformations: `<|fim_prefix|> Prefix <|fim_suffix|> Suffix <|fim_middle|> GeneratedCode`.
* **Repository-Level AST Context**: Modern coding agents do not examine isolated functions; they parse Abstract Syntax Trees (ASTs) using Tree-sitter to inject relevant class definitions, type interfaces, and imported module signatures into the prompt context.
* **Automated Test-Driven Repair**: Production code agents execute generated code in isolated sandboxes, capture unit test failures or runtime stack traces, feed the error trace back into the prompt, and autonomously repair the code in a closed loop until all assertions pass.
""")

df_p6 = pd.DataFrame({
    "Dimension": [
        "Synthesis Paradigm",
        "Context Awareness",
        "Insertion Capability",
        "Verification Loop",
        "Polyglot Migration"
    ],
    "Legacy Code Assistants (IntelliSense / Snippets)": [
        "Static type reflection, AST symbol lookups, and regex snippet expansion",
        "Limited to current file imports and static namespace declarations",
        "Append-only or manual snippet template placeholder filling",
        "None; developer must manually compile, run, and debug",
        "Zero cross-language capability; separate tooling per language ecosystem"
    ],
    "Modern Foundation Code LLMs (Claude 3.5 Sonnet / DeepSeek)": [
        "Generative probabilistic code synthesis across 80+ programming languages",
        "Repository-wide context graphs with semantic code search and dependency mapping",
        "Native Fill-In-The-Middle (FIM) predicting middle tokens from prefix and suffix",
        "Closed-loop sandbox execution with automated test execution and stack trace repair",
        "Instant polyglot refactoring (e.g., migrating COBOL or legacy Java to modern Rust)"
    ]
})
st.dataframe(df_p6, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 8: 7. Reasoning and Problem Solving
# ==============================================================================
st.header("8. Reasoning and Problem Solving", divider="rainbow")

st.markdown("""
Reasoning is the capacity to decompose complex, ambiguous objectives into ordered deductive steps, evaluate hypotheses, 
and arrive at mathematically or logically sound conclusions. 
Standard autoregressive language modeling is fundamentally **System 1 thinking** (fast, associative, reflexive next-token prediction). 
Complex mathematical proofs, algorithmic design, and multi-step enterprise workflows require **System 2 thinking** (deliberate, 
exploratory, iterative planning).
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Chain-of-Thought (CoT) Prompting**: Instructing a model to `"Think step by step"` allows it to allocate intermediate tokens in its residual stream before generating the final answer. Each intermediate token acts as external scratchpad memory, avoiding error accumulation.
* **Tree-of-Thoughts (ToT) & Self-Consistency**: Generating multiple reasoning paths and applying majority voting (Self-Consistency) or heuristic search (Tree-of-Thoughts) dramatically improves accuracy on complex logic puzzles and strategic planning.
* **Test-Time Compute Scaling**: Reasoning foundation models (such as OpenAI o1, o3, and DeepSeek-R1) introduce a new scaling frontier: scaling inference-time tokens. These models generate thousands of internal, hidden reasoning tokens—backtracking, verifying intermediate calculations, and exploring alternative hypotheses—before emitting a single visible user token.
""")

df_p7 = pd.DataFrame({
    "Dimension": [
        "Reasoning Mode",
        "Scratchpad Computation",
        "Error Correction",
        "Scaling Frontier",
        "Enterprise Applicability"
    ],
    "Standard Autoregressive LLM (System 1)": [
        "Immediate reflexive answer generation in a single forward pass",
        "Zero internal scratchpad; must emit answer tokens directly",
        "Prone to early deductive mistakes; cannot backtrack or revise",
        "Scales primarily with pretraining compute and parameter size",
        "Excellent for creative drafting, summary, and factual recall; weak on formal logic"
    ],
    "Test-Time Reasoning Models (System 2: o1 / DeepSeek-R1)": [
        "Deliberate multi-step planning, exploration, and verification",
        "Allocates thousands of hidden internal thinking tokens for computation",
        "Autonomously detects mistakes, backtracks, and tries alternative paths",
        "Scales with test-time compute: spending more inference tokens yields higher accuracy",
        "Excels at competitive programming, complex legal analysis, and mathematical proofs"
    ]
})
st.dataframe(df_p7, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 9: 8. Conversational AI
# ==============================================================================
st.header("9. Conversational AI", divider="rainbow")

st.markdown("""
Conversational AI represents the interactive synthesis of all foundation capabilities into a coherent, stateful dialogue agent. 
Conversational agents maintain multi-turn context memory, adapt dynamically to user persona requirements, 
modulate empathy, invoke external tools, and navigate ambiguous human requests while adhering to safety and regulatory boundaries.
""")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: How New Models Follow This
* **Context State Management**: Unlike stateless single-turn APIs, modern conversational systems maintain structured message arrays `[SystemMessage, UserMessage, AssistantMessage, ToolMessage]`. When conversation history approaches the context limit, production systems use sliding context windows, rolling semantic summaries, or dynamic vector retrieval over past dialogue turns.
* **Dynamic System Steerability**: Modern models exhibit near-perfect steerability via system prompts, allowing developers to define strict roles (`"You are an empathetic pediatric clinical assistant. Never prescribe medication; always instruct the user to visit an emergency room for symptoms X, Y, Z"`).
* **Multi-Tier Safety Guardrails**: Enterprise conversational bots incorporate pre-execution and post-execution guardrail firewalls (such as Llama Guard, NeMo Guardrails) that screen inputs and outputs for prompt injection, jailbreaks, PII leakage, and toxic content in real time.
""")

df_p8 = pd.DataFrame({
    "Dimension": [
        "Dialogue Architecture",
        "Intent Recognition",
        "Context Memory Retention",
        "Edge Case Handling",
        "Persona Flexibility"
    ],
    "Legacy Dialogue Systems (Rasa / Dialogflow / Rule Trees)": [
        "Predefined Finite State Machines (FSM) with brittle intent matching",
        "Intent classification (SVM/BERT) mapping user utterance to fixed intent bucket",
        "Fixed slot-filling slots; easily cleared or corrupted by tangential remarks",
        "Fails catastrophically on unexpected inputs ('I am sorry, I did not understand that')",
        "Rigid, scripted canned responses; zero linguistic nuance or tone modulation"
    ],
    "Modern Conversational Foundation LLMs": [
        "Fluid autoregressive dialogue conditioned on full multi-turn conversational history",
        "Zero-shot intent comprehension with multi-intent decomposition and tool calling",
        "Retains nuanced context across 30+ conversation turns without slot corruption",
        "Gracefully addresses unexpected questions, jokes, typos, and complex edge cases",
        "Fully steerable persona with dynamic tone modulation and safety alignment"
    ]
})
st.dataframe(df_p8, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 10: Production Implementation: Multi-Capability Pipeline in Python
# ==============================================================================
st.header("10. Production Implementation: Multi-Capability Pipeline in Python", divider="rainbow")

st.markdown("""
The following complete, runnable Python script demonstrates an enterprise-grade multi-capability pipeline. 
It incorporates **structured information extraction via Pydantic**, **abstractive summarization**, 
and **error-resilient schema validation** using modern Hugging Face / standard library tooling.
""")

st.code(r'''import json
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

# ----------------------------------------------------------------------
# 1. Define Strict Pydantic Data Contracts
# ----------------------------------------------------------------------
class InvoiceItem(BaseModel):
    description: str = Field(description="Line item description")
    quantity: int = Field(ge=1, description="Quantity purchased")
    unit_price: float = Field(ge=0.0, description="Unit price per item")
    total_price: float = Field(ge=0.0, description="Computed total for line item")

class InvoiceExtractionSchema(BaseModel):
    invoice_id: str = Field(description="Unique invoice identifier")
    vendor_name: str = Field(description="Name of supplying vendor")
    date: str = Field(description="Date of invoice issue (YYYY-MM-DD)")
    items: List[InvoiceItem] = Field(description="List of purchased line items")
    subtotal: float = Field(ge=0.0, description="Subtotal before taxes")
    tax_amount: float = Field(ge=0.0, description="Tax charged")
    grand_total: float = Field(ge=0.0, description="Final payable amount")
    currency: str = Field(default="USD", description="Currency ISO code")

# ----------------------------------------------------------------------
# 2. Production Simulation: Raw Text to Validated Structured Object
# ----------------------------------------------------------------------
raw_invoice_ocr_text = """
INVOICE #INV-2026-9812
Vendor: CloudScale Compute Corp, 500 Silicon Way, San Jose, CA
Date: 2026-10-12
Customer: Apex Data Systems

Items:
1. 8x NVIDIA H100 80GB GPU Node (Monthly Lease) - Qty: 4, Rate: $12,500.00, Total: $50,000.00
2. InfiniBand 400Gbps Dedicated Interconnect - Qty: 2, Rate: $1,250.00, Total: $2,500.00

Subtotal: $52,500.00
Sales Tax (8.25%): $4,331.25
Total Balance Due: $56,831.25 USD
"""

def extract_and_validate_invoice(raw_text: str) -> InvoiceExtractionSchema:
    """
    Simulates a production constrained extraction call.
    In real production, this integrates with OpenAI client.beta.chat.completions.parse
    or Outlines/SGLang with regex/CFG grammar masks.
    """
    # Simulated model response adhering strictly to schema
    simulated_llm_json_output = {
        "invoice_id": "INV-2026-9812",
        "vendor_name": "CloudScale Compute Corp",
        "date": "2026-10-12",
        "items": [
            {
                "description": "8x NVIDIA H100 80GB GPU Node (Monthly Lease)",
                "quantity": 4,
                "unit_price": 12500.00,
                "total_price": 50000.00
            },
            {
                "description": "InfiniBand 400Gbps Dedicated Interconnect",
                "quantity": 2,
                "unit_price": 1250.00,
                "total_price": 2500.00
            }
        ],
        "subtotal": 52500.00,
        "tax_amount": 4331.25,
        "grand_total": 56831.25,
        "currency": "USD"
    }

    try:
        # Validate through Pydantic contract
        validated_invoice = InvoiceExtractionSchema(**simulated_llm_json_output)
        
        # Verify internal mathematical consistency
        computed_item_sum = sum(item.total_price for item in validated_invoice.items)
        assert abs(computed_item_sum - validated_invoice.subtotal) < 0.01, "Subtotal mismatch!"
        computed_grand_total = validated_invoice.subtotal + validated_invoice.tax_amount
        assert abs(computed_grand_total - validated_invoice.grand_total) < 0.01, "Grand total mismatch!"
        
        return validated_invoice
    except (ValidationError, AssertionError) as e:
        raise RuntimeError(f"Extraction payload failed schema validation: {e}")

# Execute Pipeline
parsed_result = extract_and_validate_invoice(raw_invoice_ocr_text)
print(f"Successfully Extracted Invoice: {parsed_result.invoice_id}")
print(f"Vendor: {parsed_result.vendor_name} | Total: ${parsed_result.grand_total:,.2f} {parsed_result.currency}")
''', language="python")

st.divider()

# ==============================================================================
# Section 11: Interactive Capability Workbench & Automation ROI Simulator
# ==============================================================================
st.header("11. Interactive Capability Workbench & Automation ROI Simulator", divider="rainbow")

st.markdown("""
Deploying LLMs in enterprise production is fundamentally an economic optimization: trading high-cost, high-latency 
human labor for low-cost, low-latency foundation model tokens, automated validation, and exception-handling infrastructure. 
Use this live interactive simulator to model financial ROI, labor substitution ratios, and unit cost reductions.
""")

col_sim_in, col_sim_out = st.columns([1, 1.2])

with col_sim_in:
    st.subheader("Operational Baseline Parameters")
    doc_volume = st.number_input("Monthly Task / Document Volume", min_value=1000, max_value=1000000, value=30000, step=5000)
    handling_time_min = st.slider("Average Human Handling Time (Minutes)", min_value=1.0, max_value=60.0, value=7.5, step=0.5)
    hourly_labor_rate = st.slider("Fully-Loaded Human Labor Rate ($/hour)", min_value=15.0, max_value=150.0, value=40.0, step=5.0)
    automation_rate = st.slider("Target Automation Success Rate (%)", min_value=10, max_value=95, value=75, step=5)

    st.markdown("##### Foundation LLM Unit Pricing")
    input_tokens_per_task = st.number_input("Prompt + Context Tokens per Task", min_value=100, max_value=50000, value=2000, step=500)
    output_tokens_per_task = st.number_input("Output Generated Tokens per Task", min_value=50, max_value=4000, value=350, step=50)
    input_price_per_m = st.number_input("Input Token Cost per 1M Tokens ($)", min_value=0.01, max_value=20.0, value=0.15, step=0.05)
    output_price_per_m = st.number_input("Output Token Cost per 1M Tokens ($)", min_value=0.05, max_value=80.0, value=0.60, step=0.10)
    cache_read_rate = st.slider("Prompt Cache Hit Rate (%)", min_value=0, max_value=90, value=50, step=10)

# Economic Calculations
human_unit_cost = (handling_time_min / 60.0) * hourly_labor_rate
total_human_spend = doc_volume * human_unit_cost

auto_docs = doc_volume * (automation_rate / 100.0)
human_exception_docs = doc_volume - auto_docs

# Effective token cost with prompt caching (50% discount on cached reads)
effective_input_price = input_price_per_m * (1.0 - 0.5 * (cache_read_rate / 100.0))
llm_token_unit_cost = (input_tokens_per_task / 1_000_000.0 * effective_input_price) + (output_tokens_per_task / 1_000_000.0 * output_price_per_m)
total_llm_token_spend = doc_volume * llm_token_unit_cost

fixed_infra_monthly = 2000.0  # Hosting, telemetry, vector DB, human-in-the-loop tooling
total_modern_spend = (human_exception_docs * human_unit_cost) + total_llm_token_spend + fixed_infra_monthly

monthly_savings = total_human_spend - total_modern_spend
annual_savings = monthly_savings * 12.0
savings_percentage = (monthly_savings / total_human_spend) * 100.0

with col_sim_out:
    st.subheader("Financial ROI & Efficiency Metrics")

    met_c1, met_c2 = st.columns(2)
    met_c1.metric("Status Quo Human Spend", f"${total_human_spend:,.2f}/mo")
    met_c2.metric("Modernized Hybrid Spend", f"${total_modern_spend:,.2f}/mo")

    met_s1, met_s2 = st.columns(2)
    met_s1.metric("Net Monthly Savings", f"${monthly_savings:,.2f}/mo", delta=f"{savings_percentage:.1f}% reduction")
    met_s2.metric("Projected Annual Savings", f"${annual_savings:,.2f}/yr")

    st.markdown("##### Detailed Expenditure Allocation")
    df_economic_breakdown = pd.DataFrame({
        "Cost Component": [
            "Baseline Human Labor",
            "Residual Human Review (Exceptions)",
            "LLM API Token Inference",
            "Validation & Infrastructure Overhead",
            "Total Modernized Monthly Run Rate"
        ],
        "Monthly Spend ($)": [
            f"${total_human_spend:,.2f}",
            f"${(human_exception_docs * human_unit_cost):,.2f}",
            f"${total_llm_token_spend:,.2f}",
            f"${fixed_infra_monthly:,.2f}",
            f"${total_modern_spend:,.2f}"
        ],
        "Unit Cost ($/Task)": [
            f"${human_unit_cost:.2f}",
            f"${(human_exception_docs * human_unit_cost) / doc_volume:.2f}",
            f"${total_llm_token_spend / doc_volume:.4f}",
            f"${fixed_infra_monthly / doc_volume:.4f}",
            f"${total_modern_spend / doc_volume:.2f}"
        ]
    })
    st.dataframe(df_economic_breakdown, use_container_width=True, hide_index=True)

    # Unit Cost Bar Chart
    df_chart = pd.DataFrame({
        "Operating Model": ["Status Quo All-Human", "Modernized Hybrid LLM"],
        "Unit Cost Per Task ($)": [human_unit_cost, total_modern_spend / doc_volume]
    })
    st.bar_chart(df_chart.set_index("Operating Model"))

st.divider()

# ==============================================================================
# Section 12: Technical Interview Preparation & Production Traps
# ==============================================================================
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Technical Interview Questions")

q_col1, q_col2 = st.columns(2)

with q_col1:
    with st.expander("Q1: Why did unified foundation models obsolete specialized NLP pipelines?"):
        st.markdown("""
        **Core Architectural Answer:**  
        Traditional NLP partitioned language understanding across sequential, isolated sub-models (Tokenizer -> POS Tagger -> Dependency Parser -> NER Tagger -> Task Classifier). 
        This created two fundamental bottlenecks:
        1. **Cascading Error Propagation**: A 5% error in POS tagging or tokenization multiplied downstream, causing the final entity extractor or classifier to fail catastrophically.
        2. **Representation Dissociation**: Syntactic representations were separated from semantic knowledge; the parser had no access to world knowledge.
        
        Modern foundation models solve both issues by training a **single unified high-dimensional transformer** end-to-end on autoregressive next-token prediction. 
        Syntactic grammar, semantic ontology, world facts, and reasoning emerge natively inside the self-attention residual stream, allowing zero-shot transfer without multi-model maintenance.
        """)

    with st.expander("Q2: What is the architectural difference between Extractive and Abstractive Summarization?"):
        st.markdown("""
        **Answer:**  
        * **Extractive Summarization** (e.g., LexRank, TextRank): Scores sentences in the source text using graph centrality or TF-IDF weights and extracts the top-k sentences verbatim. 
          * *Pros*: 0% hallucination risk; computationally lightweight.
          * *Cons*: Fragmented flow, severe pronoun ambiguity, zero synthesis or numerical aggregation across sentences.
        * **Abstractive Summarization** (Foundation LLMs): Reads the source text, compresses semantic intent into internal hidden states, and generates novel, paraphrased sentences from scratch.
          * *Pros*: High fluency, executive synthesis, cross-paragraph aggregation, and tabular numerical reconciliation.
          * *Cons*: Hallucination risk if not grounded via strict RAG context and verification guardrails.
        """)

    with st.expander("Q3: How does In-Context Learning function without backpropagation?"):
        st.markdown("""
        **Answer:**  
        In-context learning operates **entirely at inference time with zero parameter updates** (all weights remain frozen). 
        When demonstration examples (input-output pairs) are placed in the prompt:
        1. **Induction Heads**: Transformer attention heads specifically identify repeating patterns across token positions and form associative bindings.
        2. **Activation Steering**: Demonstration tokens shift the residual stream's trajectory into a localized latent subspace corresponding to the target task.
        3. **Meta-Optimization Analogy**: Research demonstrates that transformer attention layers implicitly simulate gradient descent steps inside forward pass activation space.
        """)

with q_col2:
    with st.expander("Q4: What is the 'Jagged Technological Frontier' in enterprise LLM capabilities?"):
        st.markdown("""
        **Answer:**  
        Coined by Dell'Acqua et al. (Harvard / BCG study, 2023), the **Jagged Technological Frontier** refers to the empirical reality that an LLM can perform at superhuman levels on tasks humans perceive as intellectually demanding (passing the Bar exam, synthesizing complex legal contracts, writing polyglot code) while simultaneously failing at tasks humans find trivial (counting letters in a word, performing multi-digit multiplication without code execution, or identifying if 9.11 is greater than 9.9).
        
        **Production Implication:** Software engineers cannot assume an LLM capable of complex synthesis can handle basic arithmetic or string formatting without explicit verification tools.
        """)

    with st.expander("Q5: When should an enterprise NOT use an LLM and stick to traditional rule-based or shallow ML?"):
        st.markdown("""
        **Answer:**  
        An enterprise should reject LLMs in favor of deterministic or traditional ML (XGBoost, RegEx, ElasticSearch) under four conditions:
        1. **Strict Microsecond Latency SLAs**: Real-time high-frequency trading or edge sensor loops requiring responses under 5 milliseconds.
        2. **100% Deterministic Compliance**: Scenarios where zero stochastic variation is legally mandated (e.g., calculating tax liability brackets).
        3. **Tabular Dense Numerical Data**: Structured relational tables where tree-based models (XGBoost, LightGBM) consistently outperform transformers at 1/1000th the compute cost.
        4. **Zero-Compute Edge Devices**: Extreme low-power microcontrollers with under 10MB of RAM.
        """)

st.divider()

st.subheader("Production Traps & Operational Gotchas")

t_col1, t_col2 = st.columns(2)

with t_col1:
    st.error("""
    **Production Trap 1 (Critical): Deploying Code Generation Without Sandboxing**  
    * **Failure Mode**: Executing LLM-generated Python or SQL scripts directly on host application servers or production databases.
    * **Impact**: Data destruction, SQL table drops, infinite memory loops, or remote code execution (RCE) via prompt injection.
    * **Remediation**: Always execute generated code inside ephemeral, network-isolated sandboxes (Docker containers, gVisor, or WebAssembly/Pyodide) with strict CPU, memory, and timeout bounds.
    """)

    st.error("""
    **Production Trap 2 (Critical): Relying on Unconstrained String Output for Critical Schemas**  
    * **Failure Mode**: Prompting an LLM with "Return only JSON" and parsing with `json.loads()` without grammar constraints.
    * **Impact**: Production pipeline crashes when the model inserts markdown backticks (` ```json...``` `), trailing commas, or conversational preambles.
    * **Remediation**: Enforce schema adherence at the decoding level using structured outputs (OpenAI JSON mode, Outlines, or SGLang regex/CFG logit masking).
    """)

with t_col2:
    st.warning("""
    **Production Trap 3 (Caution): Overestimating Symbolic Reasoning Without Tool-Calling**  
    * **Failure Mode**: Asking an LLM to calculate compounding interest tables or complex balance sheets purely in natural language autoregression.
    * **Impact**: Subtle, confident numerical hallucinations that slip past human spot-checks.
    * **Remediation**: Augment the model with deterministic tool-calling (e.g., Python REPL or calculator functions) so the LLM plans the equation while an interpreter computes the exact arithmetic.
    """)

    st.warning("""
    **Production Trap 4 (Caution): Context Window Token Explosion During Bulk Processing**  
    * **Failure Mode**: Stuffing entire 200-page document histories into every prompt turn without chunking, caching, or semantic filtering.
    * **Impact**: Exponentially escalating API bills and severe latency spikes (Time-To-First-Token scaling quadratically or linearly with prompt length).
    * **Remediation**: Implement prompt caching (Anthropic/OpenAI prompt cache), hierarchical map-reduce summarization, and vector pre-filtering before context injection.
    """)

st.divider()

st.subheader("Core Concept Flashcards")

flash_choice = st.selectbox(
    "Select Concept Flashcard to Review:",
    [
        "In-Context Learning",
        "Abstractive Synthesis",
        "Jagged Technological Frontier",
        "Constrained Decoding (Grammar Masking)"
    ]
)

if flash_choice == "In-Context Learning":
    st.info("""
    **In-Context Learning**: The remarkable capability of foundation models to adapt to completely new tasks at inference time given only a few task demonstrations in the prompt, requiring zero backpropagation or neural network weight modifications.
    """)
elif flash_choice == "Abstractive Synthesis":
    st.info("""
    **Abstractive Synthesis**: Compressing and paraphrasing unstructured source information into novel, coherent natural language sentences capturing underlying semantic meaning, rather than copying verbatim text snippets.
    """)
elif flash_choice == "Jagged Technological Frontier":
    st.info("""
    **Jagged Technological Frontier**: The empirical reality that LLMs perform at superhuman levels on select complex cognitive tasks (legal analysis, code synthesis) while simultaneously failing on basic intuitive tasks (letter counting, simple deterministic logic).
    """)
else:
    st.info("""
    **Constrained Decoding (Grammar Masking)**: A decoding technique that masks out unauthorized token logits at every generation step based on a Context-Free Grammar (CFG) or JSON schema, mathematically guaranteeing that generated output strictly obeys valid syntax.
    """)

st.divider()

# ==============================================================================
# Section 13: Self-Assessment Checklist
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your technical mastery of foundation model importance and enterprise architecture:")

cb_col1, cb_col2 = st.columns(2)

with cb_col1:
    cb1 = st.checkbox("I can enumerate and describe all 8 foundational capability pillars of modern LLMs.", value=True)
    cb2 = st.checkbox("I understand why unified foundation models eliminate cascading error propagation in legacy NLP.", value=True)
    cb3 = st.checkbox("I can model enterprise unit economics and calculate automation breakeven timelines.", value=True)

with cb_col2:
    cb4 = st.checkbox("I understand the architectural and operational differences between extractive and abstractive summarization.", value=True)
    cb5 = st.checkbox("I know why tool-calling execution environments are required to mitigate symbolic reasoning limits.", value=True)
    cb6 = st.checkbox("I understand how constrained decoding and grammar masking guarantee valid structured outputs.", value=True)

if all([cb1, cb2, cb3, cb4, cb5, cb6]):
    st.success("🎯 **Lesson 1.2 Complete! You have demonstrated comprehensive mastery of why LLMs are important, the 8 foundational capability pillars, and their enterprise economic impact.**")
