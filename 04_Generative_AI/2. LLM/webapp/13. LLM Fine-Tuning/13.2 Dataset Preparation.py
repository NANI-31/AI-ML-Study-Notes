import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="13.2 Dataset Preparation - LLM Curriculum",
    page_icon="🎓",
    layout="wide"
)

# ==============================================================================
# PRODUCTION MODULE SCOPE CLASSES (For Real Runtime Execution in Workbench)
# ==============================================================================
class ProductionDatasetPacker:
    """
    Production-grade Dataset Formatter and Sequence Packer.
    Demonstrates ChatML serialization, response-only label masking,
    and bin-packing multiple short conversations into a fixed context window.
    """
    def __init__(self, max_seq_len: int = 512, pad_token_id: int = 0, ignore_index: int = -100):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def format_chatml(self, messages: List[Dict[str, str]]) -> str:
        """Renders messages into standardized ChatML format."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted

    def format_llama3(self, messages: List[Dict[str, str]]) -> str:
        """Renders messages into Llama-3 format."""
        formatted = "<|begin_of_text|>"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        return formatted

    def format_alpaca(self, instruction: str, input_text: str, output_text: str) -> str:
        """Renders Alpaca legacy instruction format."""
        if input_text.strip():
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"

    def simulate_tokenize_and_mask(self, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        """
        Simulates tokenization and response-only label masking.
        User/System tokens receive ignore_index (-100).
        Assistant response tokens receive real token IDs.
        """
        input_ids: List[int] = []
        labels: List[int] = []
        
        token_counter = 100
        for msg in messages:
            role = msg.get("role", "user")
            words = msg.get("content", "").split()
            # Assign dummy token IDs
            msg_tokens = [token_counter + i for i in range(len(words))]
            token_counter += len(words) + 5
            
            input_ids.extend(msg_tokens)
            if role == "assistant":
                labels.extend(msg_tokens)
            else:
                labels.extend([self.ignore_index] * len(msg_tokens))
                
        return input_ids, labels

    def pack_sequences(self, dataset: List[List[int]], max_len: int) -> List[Dict[str, Any]]:
        """
        Packs multiple short token sequences into fixed-length windows
        to eliminate padding waste (Bin Packing Algorithm).
        """
        packed_batches = []
        current_batch: List[int] = []
        current_boundaries: List[int] = []

        for seq in dataset:
            if len(current_batch) + len(seq) <= max_len:
                current_batch.extend(seq)
                current_boundaries.append(len(current_batch))
            else:
                if current_batch:
                    # Pad remaining
                    pad_len = max_len - len(current_batch)
                    padded = current_batch + [self.pad_token_id] * pad_len
                    packed_batches.append({
                        "tokens": padded,
                        "valid_length": len(current_batch),
                        "num_conversations": len(current_boundaries),
                        "boundaries": current_boundaries
                    })
                current_batch = list(seq)
                current_boundaries = [len(seq)]

        if current_batch:
            pad_len = max_len - len(current_batch)
            padded = current_batch + [self.pad_token_id] * pad_len
            packed_batches.append({
                "tokens": padded,
                "valid_length": len(current_batch),
                "num_conversations": len(current_boundaries),
                "boundaries": current_boundaries
            })

        return packed_batches

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
st.title("13.2 Dataset Preparation")
st.caption("Chapter 13: LLM Fine-Tuning | Domain: 2. Large Language Models")

st.markdown("""
In post-training LLM engineering, the adage **"Garbage In, Garbage Out"** is an understatement. 
A poorly formatted instruction dataset will irrevocably ruin the generation syntax of an 8-billion parameter model in under 200 optimization steps.
Conversely, as proven by Meta and the Stanford Alpaca/LIMA teams, a few thousand pristine, diverse, and rigorously templated dialogue pairs 
can turn an unruly base model into a world-class conversational assistant.

In this lesson, you will master:
1. The structural anatomy of Instruction, Prompt-Response, and Multi-Turn Conversation datasets.
2. Industry-standard schemas: Alpaca vs ShareGPT vs OpenAI Messages.
3. Chat Templates, Jinja2 rendering engines, and special token delimitations (ChatML, Llama-3, Mistral).
4. Storage formats: JSONL, Apache Arrow, and Hugging Face Datasets.
5. Contamination-free Train/Validation splitting strategies.
6. Data quality filtering: heuristic deduplication, length balance, and synthetic LLM-as-a-judge curation.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Instruction Data Engineering Pipeline", divider="rainbow")

st.markdown("""
Dataset preparation transforms raw domain records or multi-turn dialogues into tensor-ready batches 
optimized for maximum compute throughput and zero loss leakage.
""")

data_pipe_dot = """
digraph DataPipeline {
    bgcolor="#0B1120";
    rankdir=TB;
    node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontname="sans-serif", fontcolor="#F8FAFC", color="#38BDF8", penwidth=1.5];
    edge [color="#94A3B8", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];

    subgraph cluster_curation {
        label = "Phase 1: Ingestion and Curation";
        color = "#334155";
        fontcolor = "#94A3B8";
        RawData [label="Raw Text / Records\\n(Docs, APIs, Logs)", fillcolor="#0F172A"];
        Dedupe [label="Quality Filtering\\nMinHash / Decontamination", fillcolor="#1E293B"];
        CuratedPairs [label="Canonical Message Schema\\n(System, User, Assistant)", fillcolor="#0369A1"];
        RawData -> Dedupe -> CuratedPairs;
    }

    subgraph cluster_formatting {
        label = "Phase 2: Serialization and Tokenization";
        color = "#334155";
        fontcolor = "#94A3B8";
        JinjaTemplate [label="Jinja2 Chat Template\\n(<|im_start|> / <|eot_id|>)", fillcolor="#047857"];
        Tokenized [label="Token IDs and Labels\\n(ignore_index=-100 on Prompts)", fillcolor="#047857"];
        CuratedPairs -> JinjaTemplate -> Tokenized;
    }

    subgraph cluster_packing {
        label = "Phase 3: High-Throughput Batching";
        color = "#334155";
        fontcolor = "#94A3B8";
        SequencePacking [label="Bin-Packing Optimizer\\n(Zero Pad Waste)", fillcolor="#B45309"];
        GPUReady [label="Packed Tensor Batches\\n[Batch_Size, Max_Seq_Len]", fillcolor="#7C3AED"];
        Tokenized -> SequencePacking -> GPUReady;
    }
}
"""
with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(data_pipe_dot, use_container_width=True)

st.markdown("""
### The Three Fundamental Stages of Dataset Preparation
1. **Canonical Schema Normalization**: Ingesting heterogeneous sources (ShareGPT, Alpaca, raw Q&A) and transforming them into normalized `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` dictionaries.
2. **Template Serialization**: Applying the tokenizer's exact Jinja2 template to interleave special boundary markers (`<|im_start|>`, `<|start_header_id|>`).
3. **Loss Masking and Packing**: Replacing user prompt target labels with `-100` and packing multiple short conversations into a single 4096-token tensor to avoid wasted computation on pad tokens.
""")

st.divider()

# ==============================================================================
# SECTION 2: TOPIC 1 - INSTRUCTION DATASETS
# ==============================================================================
st.header("2. Instruction Datasets", divider="rainbow")

st.markdown("""
An **Instruction Dataset** consists of single-turn tasks where an explicit directive is provided to the model, 
optionally accompanied by context or input data, and paired with a desired demonstration answer.
""")

col_inst1, col_inst2 = st.columns(2)
with col_inst1:
    st.subheader("The Classical Alpaca Format (Stanford)")
    st.markdown("""
    Popularized by Taori et al. (Stanford, 2023), the Alpaca dataset used a 3-field JSON object structure:
    ```json
    {
      "instruction": "Summarize the primary risk factors in this contract.",
      "input": "Clause 4: The licensor is not liable for indirect damages...",
      "output": "The licensor disclaims liability for indirect damages."
    }
    ```
    * **`instruction`**: The primary task verb and command.
    * **`input`**: Optional contextual text or payload (can be empty string).
    * **`output`**: The ground-truth demonstration.
    """)

with col_inst2:
    st.subheader("Why Legacy Alpaca Formatting is Outdated")
    st.markdown("""
    * **Single-Turn Limitation**: Cannot represent multi-turn conversations or back-and-forth debugging sessions.
    * **Inflexible Roles**: Does not support system prompts or tool-call execution messages.
    * **Inconsistent Header Parsing**: If the model output inadvertently contains `### Response:`, downstream parsers fail.
    * **Modern Migration**: All major modern libraries (TRL, Axolotl, Unsloth) now automatically convert Alpaca records into the standard **Messages Schema**.
    """)

st.divider()

# ==============================================================================
# SECTION 3: TOPIC 2 - PROMPT-RESPONSE DATASETS
# ==============================================================================
st.header("3. Prompt-Response Datasets", divider="rainbow")

st.markdown("""
A **Prompt-Response Dataset** is the general mathematical formulation of instruction pairs: 
a conditioning prefix `x` paired with a target completion `y`.

In modern enterprise applications, prompt-response data generally falls into three functional categories:
""")

col_pr1, col_pr2, col_pr3 = st.columns(3)
with col_pr1:
    st.markdown("#### 1. Task-Conditioned")
    st.markdown("""
    * **Objective**: Transform inputs directly into target formats (e.g. SQL translation, JSON entity extraction).
    * **Prefix**: Contains schema definitions and input queries.
    * **Target**: Exact syntax-valid code or structured JSON.
    * **Diversity**: High input diversity, strictly rigid output schema.
    """)

with col_pr2:
    st.markdown("#### 2. System-Conditioned")
    st.markdown("""
    * **Objective**: Enforce persona, tone, guardrails, and compliance.
    * **Prefix**: Explicit `<|system|>` instructions defining rules (e.g. *"You are a pediatric medical assistant. Always advise consulting an MD."*).
    * **Target**: Compassionate, compliant, refusal-safe answers.
    """)

with col_pr3:
    st.markdown("#### 3. Reasoning Chains (CoT)")
    st.markdown("""
    * **Objective**: Teach intermediate step-by-step thinking.
    * **Prefix**: Complex multi-step math or logic question.
    * **Target**: `<think> step 1... step 2... </think> Final Answer: 42`.
    * **Impact**: Dramatically improves accuracy on GSM8K and coding benchmarks.
    """)

st.divider()

# ==============================================================================
# SECTION 4: TOPIC 3 - CONVERSATION DATASETS
# ==============================================================================
st.header("4. Conversation Datasets", divider="rainbow")

st.markdown("""
Real-world human interactions with LLMs are inherently multi-turn. 
A user asks a question, the assistant clarifies, the user refines, and the assistant provides updated code.
To teach an LLM conversational context retention, training data must be formatted as **Multi-Turn Conversation Trees**.
""")

st.markdown("""
#### The Canonical OpenAI / Hugging Face "Messages" Schema
This structure has become the universal lingua franca across LLM training frameworks:
```json
[
  {"role": "system", "content": "You are a concise Unix system administrator."},
  {"role": "user", "content": "How do I check open ports in Ubuntu?"},
  {"role": "assistant", "content": "Use `sudo ss -tuln` or `sudo netstat -tuln`."},
  {"role": "user", "content": "Which one displays the process name?"},
  {"role": "assistant", "content": "Add the `-p` flag: `sudo ss -tulnp`."}
]
```
""")

st.markdown("""
#### The ShareGPT Schema (Historical Standard)
Frequently encountered in open-source datasets (e.g. Vicuna, UltraChat):
```json
{
  "id": "conv_9823",
  "conversations": [
    {"from": "human", "value": "How do I check open ports?"},
    {"from": "gpt", "value": "Use `sudo ss -tuln`."}
  ]
}
```
*TRL and Hugging Face Datasets provide automatic converters: `human -> user`, `gpt -> assistant`.*
""")

st.divider()

# ==============================================================================
# SECTION 5: TOPIC 4 - CHAT TEMPLATES
# ==============================================================================
st.header("5. Chat Templates", divider="rainbow")

st.markdown("""
Base models know nothing about roles like "user" or "assistant". They only see a flat sequence of token IDs.
A **Chat Template** is a deterministic text serialization rule that injects **Special Control Tokens** 
to clearly delimit who is speaking.
""")

st.markdown("""
### Comparison of Industry-Standard Chat Templates
""")

tmpl_col1, tmpl_col2, tmpl_col3 = st.columns(3)
with tmpl_col1:
    st.markdown("#### ChatML (OpenAI / Qwen / Yi)")
    st.code("""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>""", language="text")
    st.caption("Explicit start and end markers for every turn.")

with tmpl_col2:
    st.markdown("#### Llama-3 (Meta)")
    st.code("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

4<|eot_id|>""", language="text")
    st.caption("Header delimiters and End-Of-Turn (`<|eot_id|>`) tokens.")

with tmpl_col3:
    st.markdown("#### Mistral / Zephyr")
    st.code("""<s>[INST] What is 2+2? [/INST] 4</s>""", language="text")
    st.caption("Uses bracket tags `[INST]` and `[/INST]` with `<s>` and `</s>`.")

st.markdown("""
### 🚀 Modern Foundation LLM Reality: Jinja2 in Tokenizer Configs
Modern Hugging Face tokenizers include an embedded `chat_template.json` written in the **Jinja2 templating language**.
Engineers must **never hardcode raw string concatenation**; always invoke:
```python
formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
```
Setting `add_generation_prompt=True` automatically appends the trailing `<|start_header_id|>assistant<|end_header_id|>\n\n`, 
signaling to the model that it is time to generate.
""")

st.divider()

# ==============================================================================
# SECTION 6: TOPIC 5 - DATASET FORMATTING
# ==============================================================================
st.header("6. Dataset Formatting", divider="rainbow")

st.markdown("""
When scaling to hundreds of thousands of instruction samples, how data is serialized on disk and batched in memory 
directly impacts GPU throughput and training stability.
""")

col_fmt1, col_fmt2 = st.columns(2)
with col_fmt1:
    st.subheader("On-Disk Storage Formats")
    st.markdown("""
    * **JSON Lines (`.jsonl`)**: The universal standard for raw curation. Each line is an independent valid JSON dictionary. Enables streaming large files without loading entire datasets into RAM.
    * **Apache Parquet (`.parquet`)**: Columnar compressed format. 3x to 5x smaller on disk, fast deserialization, standard format for Hugging Face Hub.
    * **Apache Arrow Memory Mapping (`datasets`)**: Hugging Face Datasets uses Arrow memory-mapped files. Zero-copy reads allow processing 100GB datasets with only 2GB RAM.
    """)

with col_fmt2:
    st.subheader("Padding vs. Sequence Packing")
    st.markdown("""
    * **Naive Padding**: Pad every sequence to `max_seq_len = 4096` with `<pad>` tokens. If average conversation length is 512 tokens, **87.5% of GPU compute is wasted computing self-attention over useless pad tokens!**
    * **Sequence Packing (Multipack / Bin Packing)**: Concatenates multiple independent conversations together up to 4096 tokens, separated by `<|eos|>`.
    * **Cumulative Attention (FlashAttention 2)**: Uses an array of cumulative sequence lengths (`cu_seqlens`) to prevent attention cross-contamination between packed conversations.
    """)

st.divider()

# ==============================================================================
# SECTION 7: TOPIC 6 - TRAIN/VALIDATION SPLIT
# ==============================================================================
st.header("7. Train/Validation Split", divider="rainbow")

st.markdown("""
In traditional machine learning, a random 80/20 train/test split is standard. 
In LLM fine-tuning, a naive random split frequently leads to **disastrous validation leakage and benchmark contamination**.
""")

st.markdown("""
### The Contamination Vectors in LLM Post-Training
1. **System Prompt / Template Leakage**: If 90% of your dataset shares identical instructions with slightly different inputs, a random split tests the model on instructions it has already overfit on.
2. **Benchmark Contamination**: If the training dataset contains synthetic samples derived from GSM8K, HumanEval, or MMLU, fine-tuning produces falsely inflated evaluation metrics.
3. **Multi-Turn Splitting**: Splitting individual turns of the same conversation between train and val leaks conversational context. Entire conversation trees must stay in the same split.
""")

st.info("""
**Best Practice Split Strategy**:
* **Group Splitting by Source / Topic**: Group conversations by document ID or topic cluster before splitting.
* **Stratified Task Sampling**: Ensure rare tasks (e.g. tool calling, complex math) are proportionally represented in both splits.
* **13-Gram Decontamination**: Run 13-gram overlap checks between the training corpus and public evaluation benchmarks (e.g. MMLU, GSM8K, MT-Bench).
""")

st.divider()

# ==============================================================================
# SECTION 8: TOPIC 7 - DATA QUALITY
# ==============================================================================
st.header("8. Data Quality", divider="rainbow")

st.markdown("""
The **LIMA (Less Is More for Alignment)** principle demonstrated that 1,000 meticulously curated samples beat 50,000 noisy samples. 
In modern SFT, data curation follows a rigorous multi-stage filtering funnel.
""")

quality_df = pd.DataFrame({
    "Filter Stage": [
        "1. Length & Ratio Filtering",
        "2. Exact & Semantic Deduplication",
        "3. Toxicity & Refusal Sanitization",
        "4. Repetition & Gibberish Removal",
        "5. LLM-as-a-Judge Scoring"
    ],
    "Criterion": [
        "Prompt > 10 tokens, Response > 20 tokens, Response/Prompt token ratio between 0.2 and 15.0",
        "Exact hash match removal; MinHash LSH (Jaccard similarity > 0.8) cluster pruning",
        "Filter out generic refusal templates ('As an AI model...') and hazardous content",
        "Drop samples with n-gram repetition cycles > 4 or character compression ratio < 0.3",
        "Frontier LLM (e.g. GPT-4o) rates Helpful, Factuality, and Tone from 1 to 5; drop < 4.0"
    ],
    "Typical Rejection Rate": [
        "10% - 15%",
        "20% - 30%",
        "5% - 8%",
        "2% - 5%",
        "25% - 40%"
    ]
})
st.dataframe(quality_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# SECTION 9: PRODUCTION PYTHON IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: End-to-End ChatML Formatter and Token Sequence Packer in Python", divider="rainbow")

st.markdown("""
The following self-contained module demonstrates the complete pipeline of modern dataset preparation:
1. Canonical multi-turn message formatting.
2. Template rendering into ChatML and Llama-3 standards.
3. Token sequence packing with bin-packing optimization.
""")

st.code("""
from typing import List, Dict, Any, Tuple

class ProductionDatasetPacker:
    \"\"\"
    Production-grade Dataset Formatter and Sequence Packer.
    \"\"\"
    def __init__(self, max_seq_len: int = 512, pad_token_id: int = 0, ignore_index: int = -100):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def format_chatml(self, messages: List[Dict[str, str]]) -> str:
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\\n{content}<|im_end|>\\n"
        return formatted

    def pack_sequences(self, dataset: List[List[int]], max_len: int) -> List[Dict[str, Any]]:
        packed_batches = []
        current_batch: List[int] = []
        current_boundaries: List[int] = []

        for seq in dataset:
            if len(current_batch) + len(seq) <= max_len:
                current_batch.extend(seq)
                current_boundaries.append(len(current_batch))
            else:
                if current_batch:
                    pad_len = max_len - len(current_batch)
                    padded = current_batch + [self.pad_token_id] * pad_len
                    packed_batches.append({
                        "tokens": padded,
                        "valid_length": len(current_batch),
                        "num_conversations": len(current_boundaries),
                        "boundaries": current_boundaries
                    })
                current_batch = list(seq)
                current_boundaries = [len(seq)]
        return packed_batches
""", language="python")

st.divider()

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Chat Template Renderer, Packer & Quality Auditor", divider="rainbow")

packer = ProductionDatasetPacker(max_seq_len=256)

wb_tab1, wb_tab2 = st.columns(2)

with wb_tab1:
    st.subheader("1. Interactive Chat Template Renderer")
    sys_prompt = st.text_input("System Prompt", value="You are a senior Python architect.")
    user_q = st.text_area("User Turn", value="How do I serialize dataclasses to JSON efficiently?")
    asst_ans = st.text_area("Assistant Turn", value="Use the standard `asdict` method or `msgspec` for 10x higher throughput.")

    template_type = st.selectbox("Target Template Engine", ["ChatML", "Llama-3", "Alpaca"])

    sample_msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_q},
        {"role": "assistant", "content": asst_ans}
    ]

    if template_type == "ChatML":
        rendered = packer.format_chatml(sample_msgs)
    elif template_type == "Llama-3":
        rendered = packer.format_llama3(sample_msgs)
    else:
        rendered = packer.format_alpaca(user_q, "", asst_ans)

    st.markdown("##### Rendered Text Representation:")
    st.code(rendered, language="markdown")

with wb_tab2:
    st.subheader("2. Sequence Packing Efficiency Calculator")
    st.markdown("""
    Compare GPU compute waste between naive padding and sequence packing.
    """)
    num_samples = st.slider("Number of Conversations in Batch", min_value=5, max_value=30, value=12)
    target_window = st.selectbox("Context Window (Tokens)", [256, 512, 1024, 2048], index=1)

    np.random.seed(42)
    sample_lengths = np.random.randint(40, 150, size=num_samples)
    synthetic_seqs = [[int(x) for x in np.random.randint(100, 500, size=l)] for l in sample_lengths]

    packed = packer.pack_sequences(synthetic_seqs, max_len=target_window)
    
    total_valid_tokens = sum(sample_lengths)
    naive_padded_tokens = num_samples * target_window
    packed_padded_tokens = len(packed) * target_window

    naive_efficiency = (total_valid_tokens / naive_padded_tokens) * 100
    packed_efficiency = (total_valid_tokens / packed_padded_tokens) * 100

    st.metric("Total Valid Content Tokens", f"{total_valid_tokens} tokens")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.metric("Naive Padding Efficiency", f"{naive_efficiency:.1f}%", f"{num_samples} batches")
    with col_e2:
        st.metric("Packed Sequence Efficiency", f"{packed_efficiency:.1f}%", f"{len(packed)} batches")

    efficiency_gain = packed_efficiency - naive_efficiency
    st.success(f"⚡ Sequence Packing boosts GPU compute utilization by +{efficiency_gain:.1f}% and reduces batch steps from {num_samples} to {len(packed)}!")

st.divider()

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("#### High-Frequency Technical Interview Questions")

q1, q2 = st.columns(2)
with q1:
    with st.expander("Q1: What is the risk of using naive padding when training on multi-turn conversations?"):
        st.write("""
        Naive padding pads every sequence to the maximum context length (e.g. 4096).
        Because conversational turns often average only 200 to 500 tokens, 80% to 90% of the tensor is filled with pad tokens.
        This wastes massive GPU memory and compute doing attention over pad tokens. 
        Sequence packing concatenates multiple conversations together, slashing training time by 2x to 5x.
        """)
    with st.expander("Q2: Why must chat templates be rendered at tokenization time rather than hardcoded in the dataset?"):
        st.write("""
        Hardcoding template strings locks the dataset to a single model architecture (e.g. hardcoding ChatML means the dataset cannot be used on Llama-3 without lossy regex cleaning).
        Storing clean canonical JSON messages allows switching tokenizers dynamically and applying native Jinja2 templates via `tokenizer.apply_chat_template()`.
        """)
    with st.expander("Q3: How do you prevent cross-contamination between conversations during sequence packing?"):
        st.write("""
        When packing multiple conversations into one sequence, naive self-attention allows tokens in Conversation B to attend to tokens in Conversation A.
        In modern frameworks, this is prevented using:
        1. **FlashAttention cumulative sequence lengths (`cu_seqlens`)**: Sets block attention masks so attention never crosses conversation boundaries.
        2. Position IDs reset to 0 at the start of each new conversation.
        """)

with q2:
    with st.expander("Q4: What is MinHash LSH and why is it used in dataset curation?"):
        st.write("""
        MinHash Locality-Sensitive Hashing is a probabilistic algorithm for detecting near-duplicate documents at scale in sub-quadratic time.
        In web scrapes and synthetic instruction sets, exact deduplication misses paraphrased or reordered prompts.
        MinHash clusters documents by Jaccard similarity of n-gram shingle sets, allowing pruning of samples with >80% semantic overlap.
        """)
    with st.expander("Q5: What is benchmark decontamination and how is it executed?"):
        st.write("""
        Benchmark decontamination ensures that none of the questions or answers from public evaluation benchmarks (e.g. MMLU, GSM8K, HumanEval) exist in the training set.
        Standard protocol checks for 13-gram overlap between training samples and evaluation test splits. 
        Any training sample sharing 13 consecutive words with a test benchmark is discarded.
        """)

st.markdown("#### Production Traps & Antipatterns")

col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("""
    **🚨 Production Trap 1: Attention Leakage in Packed Sequences**
    * **Failure**: Packing multiple conversations into one 4096-token sequence without configuring `cu_seqlens` or block masking.
    * **Impact**: The model learns that Conversation 2 can answer using facts found in the prompt of Conversation 1, causing incoherent generation during inference.
    * **Remedy**: Always verify that the attention backend (e.g. FlashAttention-2 or xFormers) receives an explicit document boundary mask.
    """)
    st.error("""
    **🚨 Production Trap 2: Missing System Prompt Consistency**
    * **Failure**: Training 50% of conversations with system prompts and 50% without, while using a chat template that generates empty system headers.
    * **Impact**: Creates token shifts where the model generates spurious empty headers or hallucinates system role overrides.
    * **Remedy**: Enforce a standardized default system prompt across all samples if the template expects one.
    """)

with col_trap2:
    st.warning("""
    **⚠️ Production Trap 3: Hardcoded EOS Tokens in Assistant Outputs**
    * **Failure**: Appending `<|end_of_text|>` manually in dataset strings while the chat template also appends `<|eot_id|>`.
    * **Impact**: Double end-of-turn tokens confuse generation stopping criteria, leading to models that stop prematurely after 1 token.
    * **Remedy**: Rely solely on the tokenizer's chat template to insert boundary and stop tokens.
    """)
    st.warning("""
    **⚠️ Production Trap 4: Imbalanced Response Length Distribution**
    * **Failure**: Training dataset where 95% of assistant responses are under 50 tokens (e.g. brief answers).
    * **Impact**: Model suffers from 'brevity bias', refusing to write detailed explanations or complete code implementations.
    * **Remedy**: Balance dataset with structured length stratification, ensuring diverse short, medium, and long response distributions.
    """)

st.markdown("#### Flashcards for Rapid Review")

fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("⚡ Flashcard 1: Universal Message Schema"):
        st.write("What are the 3 mandatory roles in the OpenAI / HF Messages schema?")
        st.info("`system`, `user`, and `assistant` (plus `tool` for function calling).")
    with st.expander("⚡ Flashcard 2: Sequence Packing Advantage"):
        st.write("What primary resource waste is eliminated by sequence bin packing?")
        st.info("Wasted GPU memory and compute spent processing `<pad>` tokens.")

with fc_col2:
    with st.expander("⚡ Flashcard 3: End-Of-Turn vs End-Of-Text"):
        st.write("What is the difference between `<|eot_id|>` and `<|end_of_text|>` in Llama-3?")
        st.info("`<|eot_id|>` terminates a single conversational turn; `<|end_of_text|>` terminates the entire document sequence.")
    with st.expander("⚡ Flashcard 4: Decontamination Metric"):
        st.write("What is the standard n-gram threshold for detecting benchmark leakage?")
        st.info("13-gram overlap between training data and public test sets.")

st.divider()

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your technical mastery of LLM Dataset Preparation before advancing to the Fine-Tuning Process:
""")

chk1 = st.checkbox("I understand the difference between legacy Alpaca format and the canonical Messages schema.", value=True)
chk2 = st.checkbox("I know how Jinja2 chat templates inject special tokens for ChatML and Llama-3.", value=True)
chk3 = st.checkbox("I understand why add_generation_prompt=True is required for inference completions.", value=True)
chk4 = st.checkbox("I can explain how sequence bin-packing eliminates GPU compute waste over naive padding.", value=True)
chk5 = st.checkbox("I know how to prevent attention cross-contamination in packed sequences using cu_seqlens.", value=True)
chk6 = st.checkbox("I understand the 13-gram benchmark decontamination protocol.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Comprehensive Mastery Achieved! You are ready to proceed to Lesson 13.3: Fine-Tuning Process.")
