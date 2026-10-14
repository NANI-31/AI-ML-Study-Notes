import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="3.3 Model Capabilities - LangChain Curriculum",
    page_icon="🦜️🔗",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("3.3 Model Capabilities: Tool Calling, Multimodality, Reasoning & Massive Contexts")
st.caption("Chapter 3: Working with LLMs and Chat Models | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
Modern Chat Models have evolved far beyond simple text-in/text-out transformers into multifaceted cognitive engines. 
Today's frontier models natively invoke external tools, enforce strongly-typed Pydantic schemas, stream 
progressive tokens, parse high-resolution images, comprehend audio streams, utilize dynamic test-time reasoning 
tokens, and ingest context windows spanning millions of tokens.

In LangChain v0.3+, these heterogeneous capabilities are unified into standard Python interfaces, 
shielding developers from vendor-specific payload variations while maximizing model performance.

In this lesson, you will master:
- **Autonomous Tool Calling**: Declarative tool binding via `bind_tools()` and structured payload dispatching.
- **Strict Structured Outputs**: Extracting type-safe domain objects via `.with_structured_output()`.
- **Multimodal Comprehension**: Formatting images, charts, and audio blocks in standard `HumanMessage` payloads.
- **Reasoning Architectures**: Managing test-time compute in models like OpenAI o1/o3-mini and DeepSeek R1.
- **Massive Context Management**: Navigating 128k to 2M token horizons while mitigating 'Lost in the Middle' degradation.
""")

st.divider()

# --- MODULE-SCOPE HELPER CLASSES FOR WORKBENCH ---
class MockCapabilityFeatureRecord:
    """Represents a capability feature in module scope."""
    def __init__(self, capability: str, support_tier: str, key_api_method: str, enterprise_utility: str):
        self.capability = capability
        self.support_tier = support_tier
        self.key_api_method = key_api_method
        self.enterprise_utility = enterprise_utility

class MockCapabilitySimulator:
    """Simulates multimodal execution, reasoning token budgets, and tool extraction in module scope."""
    def __init__(self):
        self.features = [
            MockCapabilityFeatureRecord("Tool Calling", "Universal (OpenAI, Claude, Gemini, Mistral)", "model.bind_tools(tools)", "API execution, SQL querying, external computation"),
            MockCapabilityFeatureRecord("Structured Output", "Universal via Pydantic / JSON Mode", "model.with_structured_output(Schema)", "Database ingestion, microservice DTO contracts"),
            MockCapabilityFeatureRecord("Streaming", "Universal (Sync & Async)", "model.stream() / astream()", "Real-time UI responsiveness, sub-second TTFT"),
            MockCapabilityFeatureRecord("Multimodal Vision", "Frontier (GPT-4o, Claude 3.5, Gemini 1.5)", "HumanMessage(content=[text, image_url])", "Diagram OCR, UI auditing, document understanding"),
            MockCapabilityFeatureRecord("Audio Modalities", "Specialized (Gemini 1.5, GPT-4o-Audio)", "HumanMessage(content=[text, audio_b64])", "Customer call analytics, voice assistants"),
            MockCapabilityFeatureRecord("Reasoning Models", "Frontier (o1, o3-mini, DeepSeek R1)", "model_kwargs={'reasoning_effort': 'high'}", "Formal logic, mathematical proof, complex refactoring"),
            MockCapabilityFeatureRecord("Massive Context", "Frontier (128k to 2M Tokens)", "invoke([HumanMessage(large_context)])", "Whole-codebase analysis, multi-year report RAG")
        ]

    def get_feature_matrix(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "Capability": f.capability,
                "Support Tier": f.support_tier,
                "Core LangChain Method": f.key_api_method,
                "Production Utility": f.enterprise_utility
            }
            for f in self.features
        ])

    def simulate_capability_execution(self, capability: str, user_input: str, reasoning_tokens: int) -> Dict[str, Any]:
        """Simulates response generation across diverse capabilities."""
        if capability == "Tool Calling":
            tool_payload = {
                "name": "lookup_stock_ticker",
                "arguments": {"ticker": "AAPL", "market": "NASDAQ"},
                "id": "call_9823x"
            }
            return {
                "capability": capability,
                "tool_calls": [tool_payload],
                "reasoning_tokens": 0,
                "content": "",
                "status": "TOOL_INVOCATION_REQUESTED"
            }
        elif capability == "Reasoning Models":
            visible_tokens = 85
            return {
                "capability": capability,
                "tool_calls": [],
                "reasoning_tokens": reasoning_tokens,
                "visible_tokens": visible_tokens,
                "total_tokens": reasoning_tokens + visible_tokens,
                "content": (
                    f"Chain-of-Thought verified ({reasoning_tokens} hidden reasoning tokens consumed). "
                    f"Result: The optimal consensus protocol for high-throughput cross-datacenter state machines "
                    f"under non-Byzantine failure models is Raft with multi-group leases."
                ),
                "status": "REASONING_COMPLETED"
            }
        elif capability == "Multimodal Vision":
            return {
                "capability": capability,
                "tool_calls": [],
                "reasoning_tokens": 0,
                "content": (
                    f"Vision Analysis for input: Detected 3 microservice clusters (Auth, Payment, OrderDB). "
                    f"Single point of failure identified at payment gateway reverse proxy. Resolution: add redundant ingress."
                ),
                "status": "IMAGE_PARSED"
            }
        else:
            return {
                "capability": capability,
                "tool_calls": [],
                "reasoning_tokens": 0,
                "content": f"Processed capability '{capability}' successfully for prompt: '{user_input[:30]}...'",
                "status": "SUCCESS"
            }

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Chat Model Capability Matrix & Feature Orchestration", divider="rainbow")

st.markdown("""
The diagram below illustrates how LangChain abstracts diverse foundation model capabilities 
into a unified execution topology.
""")

dot_blueprint = """
digraph CapabilitiesTopology {
    rankdir=TB;
    bgcolor="#0B1120";
    fontname="Helvetica";
    compound=true;

    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    subgraph cluster_inputs {
        label="1. Ingestion & Multimodal Payloads";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_text [label="Text Prompts & Directives", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_images [label="Multimodal Vision Blocks\n(Image URLs & Base64 PNG/JPEG)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_audio [label="Audio Snippets & Waveforms\n(Audio Base64 Payloads)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_context [label="Massive Contexts (Up to 2M Tokens)\n(Whole Codebases & Multi-Doc RAG)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    subgraph cluster_core {
        label="2. Unified Cognitive Layer (BaseChatModel)";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_model [label="Chat Model (GPT-4o, Claude 3.5, Gemini 1.5, o3-mini)\nBaseChatModel Implementation", fillcolor="#1E293B", fontcolor="#34D399"];
        node_reasoning [label="Test-Time Compute Engine\n(Hidden CoT Reasoning Tokens)", fillcolor="#1E293B", fontcolor="#F59E0B"];

        node_model -> node_reasoning [label="allocates compute", style="dashed", color="#F59E0B"];
    }

    subgraph cluster_outputs {
        label="3. Specialized Output Protocols";
        style="filled,rounded";
        color="#1E293B";
        fillcolor="#0F172A";
        fontcolor="#38BDF8";
        fontsize=12;

        node_tools [label="Tool Calling Payload\nAIMessage.tool_calls -> [Call ID, Args]", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_struct [label="Structured Output DTO\nValidated Pydantic v2 Object", fillcolor="#1E293B", fontcolor="#F8FAFC"];
        node_stream [label="Progressive Token Stream\nAIMessageChunk Generator", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    }

    // Connections
    node_text -> node_model;
    node_images -> node_model;
    node_audio -> node_model;
    node_context -> node_model;

    node_model -> node_tools [label="bind_tools()", color="#38BDF8"];
    node_model -> node_struct [label="with_structured_output()", color="#34D399"];
    node_model -> node_stream [label=".stream()", color="#38BDF8"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

st.markdown("""
> **Key Architecture Insight**: Rather than treating vision, tools, and structured outputs as separate plugins, 
> LangChain v0.3+ integrates them directly into the core `Runnable` lifecycle. The same prompt-to-model-to-parser 
> pipeline handles a 2-sentence text summary or a 50-page PDF visual architectural audit.
""")

# --- SECTION 2: TOOL CALLING ---
st.header("2. Tool Calling: Native Function Calling, bind_tools, Tool Schemas & Dispatch Lifecycle", divider="rainbow")

st.markdown("""
Tool calling allows foundation models to act as autonomous controllers by returning structured JSON arguments 
destined for external code execution. 

LangChain's **`bind_tools`** takes Python functions, Pydantic classes, or raw JSON schemas, converts them 
into the provider's native format, and binds them to the model.
""")

st.code('''from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def execute_sql_query(query: str, database_alias: str) -> str:
    """Execute read-only SQL query against corporate data warehouse."""
    return f"Simulated result for {query} on {database_alias}"

model = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Bind tool to model
model_with_tools = model.bind_tools([execute_sql_query])

# Invocation produces AIMessage with parsed tool_calls
response = model_with_tools.invoke("How many active clusters are running in database 'dw_prod'?")
print(response.tool_calls)
# [{'name': 'execute_sql_query', 'args': {'query': 'SELECT COUNT(*) FROM clusters WHERE status = \\'active\\'', 'database_alias': 'dw_prod'}, 'id': 'call_123'}]
''', language="python")

# --- SECTION 3: STRUCTURED OUTPUT ---
st.header("3. Structured Output: with_structured_output, Pydantic Schema Validation & Repair Protocols", divider="rainbow")

st.markdown("""
While tool calling can be used for general function invocation, `.with_structured_output()` is explicitly designed 
to force the model to respond in a guaranteed **Pydantic schema**.
""")

col_so1, col_so2 = st.columns(2)
with col_so1:
    st.markdown("""
    #### Architectural Guarantees
    - **Pydantic v2 Parsing**: Return type is directly an instance of your Pydantic model.
    - **Validation Enforcement**: Validates required fields, typing constraints, and custom validators.
    - **Provider Adaptation**: Automatically uses OpenAI function calling, Anthropic tool use, or Gemini JSON mode depending on the active provider.
    """)
with col_so2:
    st.code('''from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI

class SecurityAuditDTO(BaseModel):
    threat_level: str = Field(description="LOW, MEDIUM, HIGH, CRITICAL")
    vulnerabilities_detected: List[str] = Field(description="Identified CVEs or risks")
    compliance_passed: bool = Field(description="Whether the system passes SOC2 controls")

model = ChatOpenAI(model="gpt-4o", temperature=0.0)
structured_model = model.with_structured_output(SecurityAuditDTO)

# Invocation returns verified SecurityAuditDTO object
# report = structured_model.invoke("Audit report: SSH root login enabled, no MFA...")
# print(report.threat_level)  # 'CRITICAL'
''', language="python")

# --- SECTION 4: STREAMING ---
st.header("4. Streaming: Progressive Token Emitting, astream_events, Intermediate State & Chunk Merging", divider="rainbow")

st.markdown("""
Production applications must maintain responsive user interfaces. LangChain provides two streaming levels:
1. **`.stream()` / `.astream()`**: Yields token chunks (`AIMessageChunk`) directly from the model.
2. **`.astream_events()`**: Emits fine-grained lifecycle events across the entire LCEL DAG (prompt formatted, retriever matched, tool started, model generated).
""")

st.code('''from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-4o")

# Full LCEL streaming with chunk concatenation
async def stream_chat():
    full_response = ""
    async for chunk in model.astream("Explain consensus algorithms in 20 words:"):
        full_response += chunk.content
        print(chunk.content, end="", flush=True)

# asyncio.run(stream_chat())
''', language="python")

# --- SECTION 5: MULTIMODAL INPUT ---
st.header("5. Multimodal Input: Content Blocks, Mixed Media Payloads & Base64 Transport Standards", divider="rainbow")

st.markdown("""
Frontier models do not treat images, diagrams, or documents as external attachments. 
Instead, they accept **multimodal content blocks** directly inside `HumanMessage.content`.

A multimodal payload is structured as a list of dictionaries containing `type: "text"` and `type: "image_url"` blocks.
""")

st.code('''from langchain_core.messages import HumanMessage
import base64

def encode_image_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

# Constructing multimodal HumanMessage
multimodal_message = HumanMessage(
    content=[
        {"type": "text", "text": "Inspect this database topology diagram for failure modes:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://storage.company.internal/diagrams/cluster_topo.png",
                "detail": "high"  # Options: 'low', 'high', 'auto'
            }
        }
    ]
)
''', language="python")

# --- SECTION 6: IMAGE UNDERSTANDING ---
st.header("6. Image Understanding: Visual OCR, Diagram Parsing, High/Low Detail Regimes & Coordinate Detection", divider="rainbow")

st.markdown("""
When processing images, modern vision models calculate visual tokens based on image resolution and detail modes:

### Low Detail vs High Detail Regimes

| Mode | Token Cost (OpenAI / Claude) | Resolution Processing | Recommended Use Cases |
| :--- | :--- | :--- | :--- |
| **`low`** | Fixed ~85 tokens | Image downscaled to 512x512 thumbnail | General scene classification, dominant color detection |
| **`high`** | 85 base + 170 per 512x512 tile | Image split into tiles up to 2048x2048 | Architecture diagrams, code screenshots, dense tables, visual OCR |
| **`auto`** | Dynamic | Provider selects based on input size | Default production setting |
""")

# --- SECTION 7: AUDIO CAPABILITIES ---
st.header("7. Audio Capabilities: Direct Audio Ingestion, Audio Output Generation & Speech Modalities", divider="rainbow")

st.markdown("""
Leading multimodal models (e.g., Google Gemini 1.5 and OpenAI GPT-4o-Audio) support **direct native audio ingestion**, 
bypassing separate Speech-To-Text (Whisper) models. This preserves emotion, vocal tone, inflection, and acoustic cues.
""")

st.code('''from langchain_core.messages import HumanMessage

# Passing audio payload directly into native audio-enabled chat models
audio_message = HumanMessage(
    content=[
        {"type": "text", "text": "Transcribe this customer support call and detect sentiment:"},
        {
            "type": "input_audio",
            "input_audio": {
                "data": "<BASE64_ENCODED_WAV_OR_MP3_DATA>",
                "format": "wav"
            }
        }
    ]
)
''', language="python")

# --- SECTION 8: REASONING MODELS ---
st.header("8. Reasoning Models: Frontier Test-Time Compute & Chain-of-Thought Budgets", divider="rainbow")

st.markdown("""
A new generation of foundation models—including **OpenAI o1 / o3-mini** and **DeepSeek R1**—introduces 
**dynamic test-time compute**. 

Instead of generating text immediately, these models allocate thousands of hidden **reasoning tokens** 
to explore candidate hypotheses, self-correct errors, and verify logical deductions before producing a visible answer.
""")

dot_reasoning_flow = """
digraph ReasoningCompute {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=rect, style="filled,rounded", fontname="Helvetica", fontsize=11, color="#38BDF8", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#E2E8F0"];

    node_in [label="Complex Engineering Prompt\n(Formal logic, algorithm design, security audit)", fillcolor="#1E293B", fontcolor="#F8FAFC"];
    node_cot [label="Hidden Test-Time Compute (Reasoning Tokens)\n- Explores solution trees\n- Detects edge-case failure modes\n- Self-corrects incorrect deductions", fillcolor="#1E293B", fontcolor="#F59E0B"];
    node_out [label="Visible Generated Output\nSuccinct, mathematically verified solution", fillcolor="#1E293B", fontcolor="#34D399"];

    node_in -> node_cot [label="allocates reasoning budget", color="#F59E0B"];
    node_cot -> node_out [label="synthesizes answer", color="#34D399"];
}
"""

with st.expander("🔍 Contextual Diagram: Test-Time Compute & Hidden Reasoning Dynamics", expanded=True):
    st.graphviz_chart(dot_reasoning_flow, use_container_width=True)

st.code('''from langchain_openai import ChatOpenAI

# Configuring OpenAI o3-mini with medium reasoning effort
reasoning_llm = ChatOpenAI(
    model="o3-mini",
    model_kwargs={"reasoning_effort": "medium"},  # 'low', 'medium', 'high'
    max_completion_tokens=4000  # Must account for hidden reasoning tokens + visible output
)
''', language="python")

# --- SECTION 9: CONTEXT WINDOWS ---
st.header("9. Context Windows: 128k to 2M Token Frontiers, 'Lost in the Middle' & Context Management", divider="rainbow")

st.markdown("""
Modern models support gargantuan context windows (e.g., GPT-4o at 128k, Claude 3.5 Sonnet at 200k, 
and Gemini 1.5 Pro at 2,000,000 tokens). 

However, naively stuffing 500k tokens into a prompt introduces two major hazards:
1. **Financial & Latency Cost**: Ingestion latency increases linearly with prompt length.
2. **The 'Lost in the Middle' Phenomenon**: Research proves that LLMs retrieve facts placed at the extreme beginning 
   or end of a massive context far more reliably than facts placed in the middle.
""")

st.code('''# Best Practice: Structuring Massive Contexts
# 1. Place critical system instructions and guidelines at the TOP.
# 2. Place background knowledge / retrieved documents in the MIDDLE.
# 3. Restate the exact user question and output format rules at the VERY BOTTOM.
''', language="python")

# --- SECTION 10: PRODUCTION IMPLEMENTATION ---
st.header("10. Production Implementation: Multi-Capability Agentic Pipeline with Vision & Tools", divider="rainbow")

st.markdown("""
Below is an enterprise-grade multimodal diagnostic pipeline that accepts a system architecture screenshot, 
performs visual OCR and component detection, invokes external latency verification tools, and emits 
a strongly-typed Pydantic audit report.
""")

st.code('''from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. Structured Output Schema
class InfrastructureAuditReport(BaseModel):
    detected_components: List[str] = Field(description="Services or databases detected in diagram")
    single_point_of_failure: bool = Field(description="Whether a SPOF was detected")
    risk_assessment: str = Field(description="Concise technical risk analysis")
    recommended_mitigation: str = Field(description="Concrete architectural fix")

# 2. Production Tool Definition
@tool
def query_component_metrics(component_name: str) -> Dict[str, Any]:
    """Retrieve live availability and p99 latency for an infrastructure component."""
    return {"component": component_name, "p99_latency_ms": 28.4, "availability_sla": 99.95}

class MultimodalAuditor:
    """Enterprise auditor combining vision understanding, tool calling, and structured schema outputs."""

    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=api_key)
        self.structured_auditor = self.model.with_structured_output(InfrastructureAuditReport)

    def audit_architecture_diagram(self, diagram_url: str, prompt_directive: str) -> InfrastructureAuditReport:
        # Construct multimodal input message
        messages = [
            SystemMessage(content="You are a Principal Cloud Security and Infrastructure Architect."),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt_directive},
                    {
                        "type": "image_url",
                        "image_url": {"url": diagram_url, "detail": "high"}
                    }
                ]
            )
        ]

        # Execute structured multimodal inference
        audit_result: InfrastructureAuditReport = self.structured_auditor.invoke(messages)
        return audit_result
''', language="python")

# --- SECTION 11: INTERACTIVE WORKBENCH ---
st.header("11. Interactive Workbench: Interactive Capability Lab", divider="rainbow")

st.markdown("""
Experiment with different model capabilities below. Toggle between **Tool Calling**, 
**Multimodal Vision**, and **Reasoning Models**, adjust reasoning token budgets, and inspect 
standardized execution telemetry.
""")

cap_sim = MockCapabilitySimulator()

st.subheader("📋 Model Capability Reference Matrix")
st.dataframe(cap_sim.get_feature_matrix(), use_container_width=True)

st.divider()

st.subheader("⚡ Live Capability Execution Simulator")
col_c1, col_c2 = st.columns(2)
with col_c1:
    selected_cap = st.selectbox(
        "Select Capability to Test",
        ["Tool Calling", "Reasoning Models", "Multimodal Vision"],
        index=0
    )
    user_prompt_wb = st.text_input(
        "Prompt / Task",
        value="Verify distributed database replica synchronization SLA."
    )

with col_c2:
    if selected_cap == "Reasoning Models":
        cot_budget = st.slider("Reasoning Token Budget (Hidden CoT)", min_value=100, max_value=2000, value=650, step=50)
    else:
        cot_budget = 0
        st.info("Direct capability execution without test-time reasoning tokens.")

if st.button("🚀 Execute Capability Simulation", type="primary"):
    res = cap_sim.simulate_capability_execution(selected_cap, user_prompt_wb, cot_budget)
    
    st.subheader("📊 Execution Telemetry")
    t1, t2, t3 = st.columns(3)
    t1.metric("Capability", res["capability"])
    t2.metric("Status", res["status"])
    t3.metric("Hidden Reasoning Tokens", res.get("reasoning_tokens", 0))

    if res.get("tool_calls"):
        st.write("### Tool Invocations Dispatched (`AIMessage.tool_calls`)")
        st.json(res["tool_calls"])
    
    if res.get("content"):
        st.write("### Model Output Payload (`AIMessage.content`)")
        st.code(res["content"], language="markdown")

# --- SECTION 12: TECHNICAL INTERVIEW PREPARATION ---
st.header("12. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("How does LangChain's bind_tools differ from with_structured_output?",
     "bind_tools attaches tool schemas to a model without forcing tool execution, returning an AIMessage that may contain tool_calls or standard text. with_structured_output forces the model to conform strictly to a specific Pydantic schema or JSON Schema, automatically parsing and validating the response into an instantiated object."),
    ("What are the key trade-offs between 'low' and 'high' detail modes in multimodal vision models?",
     "'low' mode downsamples images to a 512x512 thumbnail and consumes a fixed ~85 tokens, making it fast and cheap for simple scene recognition. 'high' mode tiles images into 512x512 blocks, consuming 170 tokens per tile plus base overhead, providing high resolution for OCR, diagrams, and small text but significantly increasing cost and latency."),
    ("How do reasoning models (like OpenAI o1/o3-mini) handle system prompts and temperature?",
     "Many reasoning models disallow non-zero temperatures (temperature must be 1.0 or is fixed internally) because the model requires its default sampling distribution to explore diverse internal thoughts. Furthermore, early reasoning models replace SystemMessage with DeveloperMessage or fold directives directly into the user turn."),
    ("What is the 'Lost in the Middle' problem in massive context models (e.g., 1M+ tokens)?",
     "While frontier models possess large context horizons, empirical evaluations show that models retrieve information at the beginning (primacy effect) and end (recency effect) of the context window with significantly higher accuracy than information buried in the middle 60% of the prompt."),
    ("How does astream_events improve observability compared to standard astream?",
     "astream only streams token chunks from the final model. astream_events streams fine-grained execution events (e.g., 'on_chat_model_start', 'on_tool_end', 'on_retriever_chunk') across the entire LCEL DAG, enabling detailed tracing, real-time progress bars, and intermediate state debugging.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Insufficient max_completion_tokens on Reasoning Models. Reasoning models like o1 and o3-mini consume tokens for internal hidden thoughts. Setting `max_completion_tokens=200` causes the reasoning process to exhaust the budget before any visible answer can be written.")
    st.error("**Trap 2 (Breaking Error)**: Sending Uncompressed 4K Images. Uploading raw multi-megabyte image payloads directly into API calls causes client-side timeout crashes and inflated billing. Resize images locally to max 2048x2048 before transmission.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Unfiltered Full-Document Context Stuffing. Ingesting 200,000 tokens into Gemini or Claude for every query instead of using selective RAG multiplies token consumption costs by orders of magnitude. Use semantic chunk retrieval.")
    st.warning("**Trap 4 (Quality Warning)**: Passing Unclear Docstrings to Tools in bind_tools. Models rely entirely on function docstrings and argument descriptions to decide when and how to call tools. Vague docstrings result in incorrect tool selection or hallucinated parameters.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Tool Binding Method"):
        st.markdown("**model.bind_tools([func_or_schema]) binds executable tools to the model.**")
    with st.expander("🎴 Flashcard 2: Structured Output Binding"):
        st.markdown("**model.with_structured_output(PydanticModel) guarantees validated typed output.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: Vision Detail Control"):
        st.markdown("**image_url dictionary supports 'detail': 'low', 'high', or 'auto'.**")
    with st.expander("🎴 Flashcard 4: Full DAG Event Streaming"):
        st.markdown("**chain.astream_events(version='v2') streams granular lifecycle events from LCEL.**")

st.divider()

# --- SECTION 13: SELF-ASSESSMENT CHECKLIST ---
st.header("13. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I can bind tools using model.bind_tools() and handle resulting tool_calls.", value=True)
chk2 = st.checkbox("I know how to enforce Pydantic schemas using .with_structured_output().", value=True)
chk3 = st.checkbox("I understand how to construct multimodal HumanMessage payloads with images.", value=True)
chk4 = st.checkbox("I understand the mechanics of reasoning models and hidden reasoning token budgets.", value=True)
chk5 = st.checkbox("I know how to manage massive contexts while avoiding the 'Lost in the Middle' trap.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 3.3 Model Capabilities. You have completed Chapter 3: Working with LLMs and Chat Models!")
