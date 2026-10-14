import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="2.3 Basic Model Interaction - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("2.3 Basic Model Interaction: Unified Chat Model Invocation & Configuration")
st.caption("Chapter 2: LangChain Fundamentals | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
At the core of every Generative AI application is the interaction with foundation Chat Models. 
Historically, developers had to learn and maintain disparate SDKs—each with unique client configurations, 
differing authentication headers, idiosyncratic message formats, and proprietary response schemas.

**LangChain (v0.3+)** normalizes all foundation models into a unified `BaseChatModel` interface. 
Whether interacting with OpenAI, Anthropic, Google Gemini, AWS Bedrock, or local Ollama instances, 
developers enjoy an identical invocation lifecycle, standardized parameter control, and seamless 
fallback mechanisms.

In this lesson, you will master:
- **Chat Model Instantiation**: Direct provider packages versus the unified `init_chat_model` factory.
- **The Four Invocation Protocols**: Synchronous (`invoke`), asynchronous (`ainvoke`), concurrent (`batch`), and progressive (`stream`).
- **Input Signatures**: Supplying prompts as message tuples, `BaseMessage` arrays, or `PromptValue` instances.
- **Hyperparameter Mechanics**: Controlling stochasticity with `temperature`, bounding costs with `max_tokens`, and halting with `stop` sequences.
- **Provider-Specific Customization**: Passing vendor parameters via `model_kwargs` without breaking cross-vendor portability.
- **Production Resilience**: Multi-model routing and automated zero-downtime failover with `.with_fallbacks()`.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockModelProviderSimulator:
    """Simulates multi-provider ChatModel behaviors, hyperparameters, and response generation in module scope."""
    def __init__(self, provider: str, model_name: str, temperature: float, max_tokens: int, stop_sequences: List[str]):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences

    def calculate_sampling_probabilities(self, base_logits: Dict[str, float]) -> Dict[str, float]:
        """Calculates temperature-scaled softmax probabilities without LaTeX math."""
        words = list(base_logits.keys())
        logits = np.array([base_logits[w] for w in words])
        
        # When temperature is near zero, use argmax
        if self.temperature <= 0.05:
            probs = np.zeros_like(logits)
            probs[np.argmax(logits)] = 1.0
        else:
            scaled = logits / self.temperature
            # Softmax stabilization
            exp_scaled = np.exp(scaled - np.max(scaled))
            probs = exp_scaled / np.sum(exp_scaled)
            
        return {word: float(round(p, 4)) for word, p in zip(words, probs)}

    def execute_mock_invocation(self, prompt_text: str, protocol: str = "invoke") -> Dict[str, Any]:
        """Simulates model invocation across sync, stream, and batch protocols."""
        base_candidates = {
            "LangChain": 3.8,
            "LCEL": 3.2,
            "orchestration": 2.6,
            "state machine": 2.1,
            "framework": 1.5,
            "unstructured": 0.4
        }
        word_probs = self.calculate_sampling_probabilities(base_candidates)
        
        # Determine chosen sequence based on temperature
        sorted_candidates = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)
        top_word = sorted_candidates[0][0]
        
        # Build generated content
        raw_completion = (
            f"Synthesizing response for: '{prompt_text}'. "
            f"The primary architectural pillar is {top_word}. "
            f"Execution verified under {self.provider} ({self.model_name}). "
            f"Temperature is set to {self.temperature:.2f}."
        )
        
        # Apply stop sequence check
        truncated_by_stop = False
        for stop_seq in self.stop_sequences:
            if stop_seq and stop_seq in raw_completion:
                raw_completion = raw_completion.split(stop_seq)[0]
                truncated_by_stop = True
                break
                
        # Apply max token truncation
        est_tokens = len(raw_completion.split()) * 2
        truncated_by_tokens = False
        if est_tokens > self.max_tokens:
            words = raw_completion.split()[:max(1, self.max_tokens // 2)]
            raw_completion = " ".join(words) + " [TRUNCATED_MAX_TOKENS]"
            truncated_by_tokens = True
            
        finish_reason = "stop"
        if truncated_by_stop:
            finish_reason = "stop_sequence"
        elif truncated_by_tokens:
            finish_reason = "length"

        # Telemetry
        latency_map = {
            "OpenAI": 310,
            "Anthropic": 360,
            "Google Gemini": 280,
            "Ollama (Local)": 140
        }
        base_lat = latency_map.get(self.provider, 300)
        
        return {
            "protocol_used": protocol,
            "provider": self.provider,
            "model": self.model_name,
            "content": raw_completion,
            "finish_reason": finish_reason,
            "latency_ms": int(base_lat + (self.max_tokens * 0.8)),
            "usage": {
                "input_tokens": len(prompt_text.split()) * 2,
                "output_tokens": len(raw_completion.split()) * 2,
                "total_tokens": (len(prompt_text.split()) * 2) + (len(raw_completion.split()) * 2)
            },
            "sampling_distribution": word_probs
        }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Chat Model Lifecycle & Invocation Protocol", divider="rainbow")

st.markdown("""
The diagram below illustrates the end-to-end lifecycle of a Chat Model in LangChain v0.3+: 
from unified client initialization and runtime parameter compilation, through the four core invocation protocols, 
to standardized `AIMessage` telemetry and automated fallback routing.
""")

dot_blueprint = """
digraph ModelLifecycle {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_init {
        label="1. Model Instantiation & Configuration Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_init [label="init_chat_model('gpt-4o', model_provider='openai')\nOr Direct: ChatAnthropic(model='claude-3-5-sonnet')", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_params [label="Hyperparameters Configuration\ntemperature, max_tokens, timeout, stop, model_kwargs", fillcolor="#1E293B", fontcolor="#F8FAFC"];

        node_init -> node_params [label="configures", color="#38BDF8"];
    }

    subgraph cluster_input {
        label="2. Input Normalization Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_input_types [label="Supported Inputs\n- String prompt\n- List[BaseMessage] (System, Human, AI, Tool)\n- PromptValue (from ChatPromptTemplate)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    subgraph cluster_protocols {
        label="3. Runnable Invocation Protocols Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_sync [label=".invoke(input)\nSynchronous blocking request", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_async [label=".ainvoke(input)\nAsyncio event loop non-blocking", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_batch [label=".batch([input1, input2])\nConcurrent threadpool execution", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_stream [label=".stream(input)\nProgressive token chunk iterator", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    subgraph cluster_resilience {
        label="4. Resilience & Fallback Routing Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_primary [label="Primary Provider Endpoint\n(e.g., OpenAI gpt-4o)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_fallback [label="Fallback Model (.with_fallbacks)\n(e.g., Anthropic Claude / Gemini)", fillcolor="#1E293B", fontcolor="#F59E0B"];
        
        node_primary -> node_fallback [label="on 429/500/timeout", style="dashed", color="#EF4444"];
    }

    subgraph cluster_output {
        label="5. Standardized Generation Contract Tier";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_aimsg [label="Standardized AIMessage\n- .content (generated text)\n- .tool_calls (parsed function calls)\n- .usage_metadata (input/output tokens)\n- .response_metadata (finish_reason, model)", fillcolor="#1E293B", fontcolor="#34D399"];
    }

    // Inter-cluster connections
    node_params -> node_input_types [style="invis"];
    node_input_types -> node_sync;
    node_input_types -> node_async;
    node_input_types -> node_batch;
    node_input_types -> node_stream;

    node_sync -> node_primary;
    node_async -> node_primary;
    node_batch -> node_primary;
    node_stream -> node_primary;

    node_primary -> node_aimsg [label="returns", color="#34D399"];
    node_fallback -> node_aimsg [label="returns on recovery", color="#34D399"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: All Chat Models in LangChain subclass `BaseChatModel`, which in turn inherits 
> from `Runnable`. This guarantees that every model implements the exact same execution methods (`invoke`, `ainvoke`, 
> `stream`, `batch`) and accepts identical hyperparameter configurations regardless of whether you are calling a cloud API 
> or an on-premise open-weight cluster.
""")

# --- SECTION 2: CREATING A CHAT MODEL ---
st.header("2. Creating a Chat Model: init_chat_model, Provider Packages & Unified Factories", divider="rainbow")

st.markdown("""
In LangChain v0.3+, you can instantiate a chat model using two complementary approaches:
1. **Direct Provider Packages**: Explicit instantiation using dedicated integration packages (e.g., `langchain-openai`, `langchain-anthropic`).
2. **The Unified Factory (`init_chat_model`)**: Dynamically configuring models from environment variables or runtime configurations without hardcoding provider classes.
""")

col_cm1, col_cm2 = st.columns(2)
with col_cm1:
    st.markdown("""
    #### Direct Provider Instantiation
    ```python
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic

    # Explicit provider binding
    model_openai = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key="sk-..."
    )

    model_claude = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.2
    )
    ```
    """)
with col_cm2:
    st.markdown("""
    #### The Modern Unified Factory (`init_chat_model`)
    ```python
    from langchain.chat_models import init_chat_model

    # Dynamically select provider via string
    llm = init_chat_model(
        "gpt-4o", 
        model_provider="openai", 
        temperature=0.2
    )

    # Seamless switch to Anthropic:
    # llm = init_chat_model("claude-3-5-sonnet-20241022", model_provider="anthropic")
    ```
    """)

# --- SECTION 3: INVOKING A MODEL ---
st.header("3. Invoking a Model: Synchronous, Asynchronous, Batching & Streaming Protocols", divider="rainbow")

st.markdown("""
Because Chat Models are first-class **Runnables**, they natively support four distinct execution patterns. 
Selecting the correct protocol for your workload prevents system bottlenecks and UI freezes.
""")

dot_protocols = """
digraph InvocationProtocols {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_sync {
        label="Synchronous (.invoke)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        node_s1 [label="Input Prompt", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_s2 [label="Network Block\n(300ms - 2000ms)", fillcolor="#1E293B", fontcolor="#F59E0B"];
        node_s3 [label="Complete AIMessage", fillcolor="#1E293B", fontcolor="#34D399"];
        node_s1 -> node_s2 -> node_s3;
    }

    subgraph cluster_stream {
        label="Streaming (.stream)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        node_st1 [label="Input Prompt", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_st2 [label="Generator Yields\nAIMessageChunk 1...N", fillcolor="#1E293B", fontcolor="#38BDF8"];
        node_st3 [label="Progressive Token UI", fillcolor="#1E293B", fontcolor="#34D399"];
        node_st1 -> node_st2 -> node_st3;
    }

    subgraph cluster_batch {
        label="Concurrent Batching (.batch)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        node_b1 [label="List of 100 Prompts", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_b2 [label="ThreadPool Executor\n(Concurrency=max_concurrency)", fillcolor="#1E293B", fontcolor="#F59E0B"];
        node_b3 [label="List of 100 AIMessages", fillcolor="#1E293B", fontcolor="#34D399"];
        node_b1 -> node_b2 -> node_b3;
    }
}
"""

with st.expander("🔍 Contextual Diagram: Comparative Model Invocation Protocols", expanded=True):
    st.graphviz_chart(dot_protocols, use_container_width=True)

st.code('''from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-4o", temperature=0.1)

# 1. Synchronous Invocation
response = model.invoke("Briefly define zero-shot learning.")
print(response.content)

# 2. Progressive Streaming
for chunk in model.stream("List three advantages of LCEL:"):
    print(chunk.content, end="", flush=True)

# 3. Concurrent Batching
prompts = ["Define Docker in 5 words", "Define Kubernetes in 5 words", "Define Redis in 5 words"]
batch_responses = model.batch(prompts, config={"max_concurrency": 5})

# 4. Asynchronous Invocation (FastAPI / Asyncio Web Servers)
async def async_call():
    async_res = await model.ainvoke("Analyze asynchronous I/O advantages.")
    return async_res.content
''', language="python")

# --- SECTION 4: PASSING MESSAGES ---
st.header("4. Passing Messages: Tuples, Lists of BaseMessages & PromptValues", divider="rainbow")

st.markdown("""
LangChain provides remarkable flexibility in how inputs can be formatted when calling a model. 
A model accepts:
1. **A raw string**: Automatically converted into a single `HumanMessage`.
2. **A list of role tuples**: Syntactic sugar for `("role", "content")` strings.
3. **A list of `BaseMessage` objects**: The canonical, type-safe representation.
4. **A `PromptValue`**: Output generated directly by a `ChatPromptTemplate`.
""")

st.code('''from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# Format 1: Role Tuples (Convenient & Compact)
res1 = model.invoke([
    ("system", "You are an infrastructure auditor."),
    ("human", "Check Redis cluster replication status.")
])

# Format 2: Explicit BaseMessage Objects (Type-Safe Production Standard)
res2 = model.invoke([
    SystemMessage(content="You are an infrastructure auditor."),
    HumanMessage(content="Check Redis cluster replication status.")
])

# Format 3: Via LCEL ChatPromptTemplate (Composed Pipeline)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {technology}."),
    ("human", "{question}")
])
chain = prompt | model
res3 = chain.invoke({"technology": "PostgreSQL", "question": "Explain WAL archiving."})
''', language="python")

# --- SECTION 5: MODEL RESPONSES ---
st.header("5. Model Responses: AIMessage Anatomy, usage_metadata & Generation Chunks", divider="rainbow")

st.markdown("""
When a Chat Model returns a response, the output is an **`AIMessage`** (or an **`AIMessageChunk`** during streaming). 
Understanding the internal anatomy of this response object allows you to extract precise telemetry and handle generation lifecycle events.
""")

col_mr1, col_mr2 = st.columns(2)
with col_mr1:
    st.markdown("""
    #### Anatomy of an `AIMessage`
    - **`content`**: String representation of generated response text.
    - **`tool_calls`**: List of parsed structured tool call requests.
    - **`response_metadata`**: Vendor-specific generation details:
      - `finish_reason`: `"stop"`, `"length"`, `"tool_calls"`, or `"content_filter"`.
      - `model_name`: Exact provider checkpoint name.
      - `system_fingerprint`: Backend hardware/cluster revision identifier.
    - **`usage_metadata`**: Standardized token telemetry dictionary.
    """)
with col_mr2:
    st.markdown("""
    #### Streaming with `AIMessageChunk`
    During streaming, every yield produces an `AIMessageChunk`. 
    Chunks support **additive arithmetic**:
    ```python
    chunk1 = AIMessageChunk(content="Hello ")
    chunk2 = AIMessageChunk(content="World!")
    full_message = chunk1 + chunk2
    print(full_message.content) # "Hello World!"
    ```
    LangChain handles accumulating tokens and partial tool calls automatically when concatenating chunks!
    """)

# --- SECTION 6: MODEL PARAMETERS ---
st.header("6. Model Parameters: Initialization Attributes vs Runtime Dynamic Invocations", divider="rainbow")

st.markdown("""
Hyperparameters can be defined at two distinct moments in an application's lifecycle:
1. **Static Initialization**: Set when constructing the model class.
2. **Runtime Binding (`.bind()`)**: Overridden per request or dynamically within an LCEL pipeline without modifying the shared base model.
""")

st.code('''from langchain_openai import ChatOpenAI

# 1. Static Configuration
base_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=500,
    timeout=30.0,
    max_retries=3
)

# 2. Dynamic Runtime Binding (Zero Mutation of base_model)
# Creates a new Runnable with overridden parameters:
deterministic_model = base_model.bind(temperature=0.0, max_tokens=100)
creative_model = base_model.bind(temperature=0.9, max_tokens=1500)

# 3. Dynamic Configuration via RunnableConfig
# Override parameters during the actual .invoke() call:
res = base_model.invoke(
    "Generate code snippet",
    config={"configurable": {"temperature": 0.1}}
)
''', language="python")

# --- SECTION 7: TEMPERATURE ---
st.header("7. Temperature: Softmax Sampling Mechanics, Determinism & Creative Latitudes", divider="rainbow")

st.markdown("""
Large Language Models do not generate words deterministically by default. They generate **logits** 
(unnormalized log-odds) for every token in their vocabulary, which are converted to probabilities 
via a temperature-scaled Softmax transformation:

```text
Probability P(token_i) = exp(logit_i / Temperature) / Sum(exp(logit_j / Temperature))
```

### Operational Regimes for Temperature

| Setting | Temperature Range | Primary Behavior | Production Use Cases |
| :--- | :--- | :--- | :--- |
| **Deterministic** | `0.0` to `0.2` | Argmax selection; highly repeatable, minimal hallucination | Code generation, SQL extraction, JSON structured outputs, math |
| **Balanced** | `0.3` to `0.7` | Moderate diversity with factual grounding | Technical documentation, summarization, conversational agents |
| **Creative** | `0.8` to `1.2` | High token diversity, exploration of low-probability tails | Brainstorming, creative writing, synthetic data generation |
| **Destructive** | `> 1.5` | Probability distribution approaches uniform noise | Gibberish tokens, syntax collapse (Avoid in production) |
""")

# --- SECTION 8: MAXIMUM TOKENS ---
st.header("8. Maximum Tokens: max_tokens, Truncation Detection & Cost Guardrails", divider="rainbow")

st.markdown("""
The `max_tokens` (or `max_completion_tokens`) parameter places a hard ceiling on the number of generated tokens. 
It is the most critical financial and operational guardrail in production:

1. **Denial-of-Wallet Defense**: Prevents infinite loops or run-away generation from exhausting API billing quotas.
2. **Latency Ceiling**: Because token generation is sequential, bounding `max_tokens` guarantees an upper bound on network latency.
3. **Truncation Detection**: When output hits `max_tokens`, `response_metadata["finish_reason"]` is set to `"length"`. 
   Production systems must monitor this flag to detect truncated JSON schemas or incomplete answers.
""")

st.code('''from langchain_openai import ChatOpenAI

# Setting conservative token limits
guarded_model = ChatOpenAI(model="gpt-4o", max_tokens=50)

response = guarded_model.invoke("Provide a 500-word comprehensive history of computing.")

# Check for truncation in production:
finish_reason = response.response_metadata.get("finish_reason")
if finish_reason == "length":
    print("WARNING: Model response was prematurely truncated by max_tokens limit!")
''', language="python")

# --- SECTION 9: STOP SEQUENCES ---
st.header("9. Stop Sequences: Generation Halting Strings, Grammar Constraints & Output Boundaries", divider="rainbow")

st.markdown("""
A **Stop Sequence** is a string (or list of strings) that causes the model to halt token generation 
immediately upon outputting that sequence. The stop sequence itself is excluded from the returned text.

### Production Use Cases for Stop Sequences
- **Preventing Multi-Turn Hallucinations**: In few-shot prompting, passing `stop=["Human:", "User:"]` prevents the model from generating fake conversational turns.
- **Syntax Boundaries**: In code generation, passing `stop=["```", "\nclass ", "\ndef "]` ensures the model generates only the target function body without extraneous markdown.
- **Delimiter Containment**: Halting extraction when encountering a section delimiter like `---` or `=== END ===`.
""")

st.code('''from langchain_openai import ChatOpenAI

# Halting generation at markdown delimiters or conversational turns
model_with_stops = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2
).bind(stop=["\\n\\nHuman:", "\\n\\nAssistant:", "```"])

# Generation will instantly halt if any stop token appears
response = model_with_stops.invoke("Write Python function to calculate SHA256.\\n```python\\n")
''', language="python")

# --- SECTION 10: MODEL SELECTION ---
st.header("10. Model Selection: Capability-Based Routing, Fallback Chains & Cost Matrices", divider="rainbow")

st.markdown("""
In enterprise systems, relying on a single foundation model provider introduces single-point-of-failure (SPOF) risks, 
rate limit throttling, and unnecessary cost overhead. 

LangChain provides **`.with_fallbacks()`** to construct resilient multi-model failover chains. 
If the primary provider throws a 429 (Rate Limit), 503 (Service Unavailable), or timeout exception, 
LangChain automatically retries the request against secondary providers with zero application downtime.
""")

st.code('''from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Primary: High-intelligence frontier model
primary_model = ChatOpenAI(model="gpt-4o", timeout=10.0, max_retries=1)

# Secondary: High-availability enterprise alternative
fallback_one = ChatAnthropic(model="claude-3-5-sonnet-20241022", timeout=10.0)

# Tertiary: Cost-effective emergency fallback
fallback_two = ChatGoogleGenerativeAI(model="gemini-1.5-flash", timeout=8.0)

# Resilient composite chain
resilient_model = primary_model.with_fallbacks([fallback_one, fallback_two])

# Invocation will seamlessly transition to Claude or Gemini if OpenAI times out:
result = resilient_model.invoke("Summarize high-concurrency database locking strategies.")
''', language="python")

# --- SECTION 11: PROVIDER-SPECIFIC PARAMETERS ---
st.header("11. Provider-Specific Parameters: model_kwargs, Extra Headers & Backend Flags", divider="rainbow")

st.markdown("""
While LangChain standardizes common parameters like `temperature` and `max_tokens`, frontier providers 
frequently introduce proprietary flags (e.g., Anthropic's prompt caching headers, OpenAI's `seed` or `logprobs`, 
or reasoning effort in OpenAI o1/o3-mini).

LangChain supports provider-specific configuration through **`model_kwargs`** and runtime headers without breaking the unified API contract.
""")

st.code('''from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI deterministic seed & logprobs configuration
openai_advanced = ChatOpenAI(
    model="gpt-4o",
    model_kwargs={
        "seed": 42,            # Deterministic RNG seed
        "logprobs": True,      # Return token log-probabilities
        "top_logprobs": 3      # Top 3 alternative token options
    }
)

# Anthropic prompt caching headers configuration
anthropic_cached = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    model_kwargs={
        "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}
    }
)
''', language="python")

# --- SECTION 12: PRODUCTION IMPLEMENTATION ---
st.header("12. Production Implementation: Enterprise Multi-Provider Model Client with Fallbacks", divider="rainbow")

st.markdown("""
Below is an enterprise-grade model gateway implementing unified model initialization, dynamic temperature overrides, 
failover routing via `.with_fallbacks()`, and detailed telemetry extraction.
""")

st.code('''from typing import Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class EnterpriseModelGateway:
    """Production gateway providing multi-provider routing, timeouts, and automatic failovers."""

    def __init__(self, openai_key: str, anthropic_key: str):
        # 1. Primary Model: OpenAI GPT-4o
        self.primary_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=2048,
            timeout=12.0,
            max_retries=1,
            api_key=openai_key
        )

        # 2. Secondary Model: Anthropic Claude 3.5 Sonnet
        self.secondary_model = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.2,
            max_tokens=2048,
            timeout=12.0,
            max_retries=2,
            api_key=anthropic_key
        )

        # 3. Assemble Resilient Fallback Chain
        self.resilient_client: RunnableWithFallbacks = self.primary_model.with_fallbacks(
            [self.secondary_model]
        )

    def execute_query(self, system_prompt: str, user_prompt: str, temperature_override: Optional[float] = None) -> Dict[str, Any]:
        """Executes query with optional runtime temperature override and extracts standardized telemetry."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        # Apply runtime hyperparameter override if specified
        active_runnable = self.resilient_client
        if temperature_override is not None:
            active_runnable = self.resilient_client.bind(temperature=temperature_override)

        # Execute invocation across failover boundary
        ai_msg = active_runnable.invoke(messages)

        # Standardized telemetry extraction
        usage = ai_msg.usage_metadata or {}
        resp_meta = ai_msg.response_metadata or {}

        return {
            "status": "SUCCESS",
            "content": str(ai_msg.content),
            "model_invoked": resp_meta.get("model_name", "unknown_model"),
            "finish_reason": resp_meta.get("finish_reason", "completed"),
            "token_metrics": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
        }
''', language="python")

# --- SECTION 13: INTERACTIVE WORKBENCH ---
st.header("13. Interactive Workbench: Live Model Interaction & Hyperparameter Playground", divider="rainbow")

st.markdown("""
Interact with the live model simulator below. Adjust **Temperature**, **Max Tokens**, and **Stop Sequences**, 
toggle invocation protocols (Synchronous, Streaming, Batching), and observe real-time Softmax sampling distributions 
and response telemetry.
""")

col_wb1, col_wb2 = st.columns(2)
with col_wb1:
    selected_provider = st.selectbox(
        "Provider & Model Profile",
        ["OpenAI (gpt-4o)", "Anthropic (claude-3-5-sonnet)", "Google Gemini (gemini-1.5-pro)", "Ollama (Local llama-3.1)"],
        index=0
    )
    prov_name = selected_provider.split(" (")[0]
    model_name_extracted = selected_provider.split(" (")[1].replace(")", "")
    
    temp_input = st.slider("Temperature (Stochasticity)", min_value=0.0, max_value=1.5, value=0.2, step=0.05)
    max_token_input = st.slider("Max Completion Tokens", min_value=10, max_value=200, value=60, step=5)
    stop_seq_input = st.text_input("Stop Sequence (e.g., 'pillar' or 'Execution')", value="")

with col_wb2:
    protocol_choice = st.radio(
        "Invocation Protocol",
        ["Synchronous (.invoke)", "Streaming Simulation (.stream)", "Batch Simulation (.batch)"],
        index=0
    )
    user_prompt_wb = st.text_area(
        "Prompt Input",
        value="Explain why unified chat models simplify multi-provider architecture."
    )

if st.button("🚀 Execute Model Interaction Protocol", type="primary"):
    stop_list = [stop_seq_input.strip()] if stop_seq_input.strip() else []
    simulator = MockModelProviderSimulator(
        provider=prov_name,
        model_name=model_name_extracted,
        temperature=temp_input,
        max_tokens=max_token_input,
        stop_sequences=stop_list
    )
    
    exec_result = simulator.execute_mock_invocation(user_prompt_wb, protocol=protocol_choice)
    
    st.subheader("📊 Execution Telemetry & Response Inspection")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Provider", exec_result["provider"])
    m2.metric("Latency", f"{exec_result['latency_ms']} ms")
    m3.metric("Finish Reason", exec_result["finish_reason"])
    m4.metric("Total Tokens", exec_result["usage"]["total_tokens"])

    st.write("### Model Response Payload (`AIMessage.content`)")
    if protocol_choice == "Streaming Simulation (.stream)":
        st.info("Streaming progressive chunks:")
        chunk_placeholder = st.empty()
        accumulated = ""
        words = exec_result["content"].split(" ")
        for w in words:
            accumulated += w + " "
            chunk_placeholder.code(accumulated, language="markdown")
    else:
        st.code(exec_result["content"], language="markdown")

    st.write("### Sampling Probability Distribution (Softmax Logits Transformation)")
    probs_df = pd.DataFrame([
        {"Candidate Token": k, "Probability": v, "Percentage": f"{round(v * 100, 2)}%"}
        for k, v in exec_result["sampling_distribution"].items()
    ])
    st.dataframe(probs_df, use_container_width=True)

# --- SECTION 14: TECHNICAL INTERVIEW PREPARATION ---
st.header("14. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does init_chat_model simplify multi-cloud deployments compared to direct class instantiation?",
     "init_chat_model acts as a dynamic factory that accepts provider names as strings (e.g., 'openai', 'anthropic', 'google_vertexai'). This allows applications to switch model providers via environment variables or runtime configurations without code refactoring or modifying imports, promoting true vendor neutrality."),
    ("Why is .ainvoke() mandatory when integrating LangChain with FastAPI or Tornado?",
     "Calling .invoke() runs a synchronous blocking network request. Inside an asynchronous Python event loop (such as FastAPI), a synchronous blocking call freezes the entire single-threaded event loop, preventing all concurrent requests from being handled. Calling .ainvoke() uses non-blocking asyncio HTTP clients."),
    ("What is the difference between static model initialization and runtime binding with .bind()?",
     "Initialization (e.g., ChatOpenAI(temperature=0.7)) sets default parameters on the base instance. Calling .bind(temperature=0.0) creates a new RunnableBinding without mutating the original object. This allows sharing a single configured client across multiple threads and LCEL chains while applying specialized parameters per route."),
    ("How does .with_fallbacks() handle rate limit (HTTP 429) errors in high-throughput applications?",
     "When the primary model raises an exception (such as RateLimitError, APIConnectionError, or Timeout), .with_fallbacks() intercepts the exception before it bubbles to user space, transparently catches it, and immediately forwards the identical input payload to the next fallback model in the list."),
    ("How do stop sequences differ from max_tokens in bounding model generation?",
     "max_tokens halts generation when a predetermined token count is reached, regardless of sentence completeness. Stop sequences halt generation as soon as a specific semantic delimiter string (e.g., '\nHuman:' or '```') is output by the model, allowing clean linguistic termination without slicing off text mid-word.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Blocking Async Servers with .invoke(). Using `.invoke()` inside `async def` routes completely blocks ASGI web workers. Always use `await model.ainvoke()` or `model.astream()` in async web backends.")
    st.error("**Trap 2 (Breaking Error)**: Ignored 'length' Finish Reason. Treating truncated outputs as complete answers leads to malformed JSON or half-written code. Always check `response_metadata['finish_reason'] == 'length'`.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: High Temperature for Structured Outputs. Setting `temperature > 0.5` when extracting JSON or Pydantic schemas introduces stochastic variation, increasing syntax failure and validation retry rates. Always use `temperature=0.0` for structured extraction.")
    st.warning("**Trap 4 (Latency Warning)**: Cascading Timeouts in Fallback Chains. If primary and fallback models have 60-second timeouts, a cascading failure forces clients to wait 120+ seconds before seeing an error. Keep timeout budgets aggressive (e.g., 8–12 seconds) for each tier.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Chat Model Invocation Methods"):
        st.markdown("**.invoke() (Sync), .ainvoke() (Async), .stream() (Streaming chunks), .batch() (Concurrent batch).**")
    with st.expander("🎴 Flashcard 2: Dynamic Parameter Binding"):
        st.markdown("**model.bind(temperature=0.0, stop=['End']) creates a parameter-overridden Runnable.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Zero-Downtime Resilience"):
        st.markdown("**primary_model.with_fallbacks([secondary_model]) provides seamless automated failover.**")
    with st.expander("🎴 Flashcard 4: Unified Chat Model Factory"):
        st.markdown("**init_chat_model('model_name', model_provider='provider') unifies model instantiation.**")

st.divider()

# --- SECTION 15: SELF-ASSESSMENT CHECKLIST ---
st.header("15. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can initialize chat models using direct integration packages and init_chat_model.", value=True)
chk2 = st.checkbox("I understand the 4 invocation protocols: invoke, ainvoke, stream, and batch.", value=True)
chk3 = st.checkbox("I know how to pass inputs as strings, message tuples, and BaseMessage arrays.", value=True)
chk4 = st.checkbox("I understand the mechanics of temperature, max_tokens, and stop sequences.", value=True)
chk5 = st.checkbox("I can implement production failover routing using .with_fallbacks().", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 2.3 Basic Model Interaction. You have completed Chapter 2: LangChain Fundamentals!")
