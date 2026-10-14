import streamlit as st
import pandas as pd
import json

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="17.1 Tool Calling Basics | GenAI Roadmap",
    page_icon="🛠️",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .tool-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("🛠️ 17.1 Tool Calling Basics — JSON Schemas, Function Signatures & Runtime")
st.markdown(
    """
Welcome to **Lesson 17.1: Tool Calling Basics**.  
Large Language Models cannot execute code or query APIs on their own—they run inside isolated compute containers that predict text tokens. However, modern models are fine-tuned to act as **intelligent orchestration engines**.

Through **Tool Calling** (formerly Function Calling), an LLM can inspect a catalog of available functions, decide when an external action is needed, and generate a **strictly typed JSON payload** specifying the function name and exact arguments for the client application to execute.
"""
)

st.divider()

# ==============================================================================
# Section 1: The Evolution from Chatbot to Action Engine
# ==============================================================================
st.header("1. From Conversational Chatbot to Action Engine", divider="rainbow")

col_tc1, col_tc2 = st.columns(2)

with col_tc1:
    with st.container(border=True):
        st.subheader("❌ Naive Natural Language Attempt")
        st.markdown(
            """
**User Prompt:**
```text
Please check our production database and tell me how many active users logged in today.
```

**LLM Response (Pure Prose):**
```text
I do not have access to your database, but typically you could run 
a query like 'SELECT COUNT(*) FROM users WHERE last_login = TODAY()' 
in your terminal!
```
*The model is trapped in text space. It can describe actions, but cannot interact with the real world.*
"""
        )

with col_tc2:
    with st.container(border=True):
        st.subheader("✅ Tool Calling Architecture")
        st.markdown(
            """
**Available Tool Defined in API Request:**
```json
{"name": "query_database", "description": "Executes read-only SQL queries"}
```

**LLM Emits Structured Tool Call:**
```json
{
  "name": "query_database",
  "arguments": {
    "sql": "SELECT COUNT(*) FROM users WHERE DATE(last_login) = CURRENT_DATE;"
  }
}
```
*The host application intercepts this JSON, runs the query safely on the database, and feeds the numerical result back to the model!*
"""
        )

st.divider()

# ==============================================================================
# Section 2: Anatomy of a Production Tool Schema
# ==============================================================================
st.header("2. Anatomy of an OpenAI / Anthropic Tool Schema", divider="rainbow")
st.markdown("Tools are registered using JSON Schema specifications passed in the API payload:")

schema_parts_df = pd.DataFrame({
    "Schema Component": [
        "1. type",
        "2. name",
        "3. description",
        "4. parameters.properties",
        "5. parameters.required",
        "6. parameters.additionalProperties"
    ],
    "Specification Requirement": [
        "Must be set to 'function'",
        "Snake_case identifier (e.g. `lookup_customer_by_email`); max 64 chars",
        "Detailed docstring explaining exactly WHAT the tool does and WHEN the model should choose it",
        "Dictionary of typed parameters (`string`, `number`, `boolean`, `array`, `object`)",
        "Array of parameter names that MUST be supplied for the tool to function",
        "Set to false in strict schema mode to prevent the model from hallucinating unexpected arguments"
    ],
    "Model Conditioning Impact": [
        "Specifies function execution protocol",
        "Acts as primary keyword anchor during tool selection",
        "Most critical field: the model's self-attention reads this to determine semantic relevance",
        "Constrains token generation to valid data types",
        "Forces the model to populate mandatory parameters before calling",
        "Guarantees 100% adherence to schema boundaries"
    ]
})
st.dataframe(schema_parts_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Client-LLM Tool Execution Loop", divider="rainbow")
st.markdown("How tool calling bridges LLM reasoning with real-world client execution environments:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        user [label="User Query\n'Send $50 to Alice'", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        llm [label="LLM Ingestion\n(Prompt + Tool Schemas)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        tool_call [label="LLM Output:\ntool_call(send_payment)", fillcolor="#EAB308", fontcolor="#000000"]
        client_runner [label="Host App Execution\n(Runs Python / SQL / API)", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        tool_msg [label="Tool Message\n{'status': 'success'}", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        final_gen [label="LLM Final Synthesis\n'Sent $50 to Alice!'", fillcolor="#10B981", fontcolor="#FFFFFF"]

        user -> llm -> tool_call -> client_runner -> tool_msg -> llm -> final_gen
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Tool Calling Simulation Workbench
# ==============================================================================
st.header("4. Interactive Tool Calling Simulation Workbench", divider="rainbow")
st.markdown("Submit user requests to see how an LLM decides whether to respond with prose or emit a structured tool call:")

available_tools = [
    {
        "name": "get_current_weather",
        "description": "Get current weather conditions and temperature for a given city.",
        "params": {"city": "str (required)", "units": "'celsius' | 'fahrenheit' (optional)"}
    },
    {
        "name": "search_customer_account",
        "description": "Look up customer account balance, plan tier, and subscription status by email.",
        "params": {"email": "str (required)"}
    },
    {
        "name": "refund_transaction",
        "description": "Issues a payment refund for a given transaction ID.",
        "params": {"tx_id": "str (required)", "reason": "str (required)"}
    }
]

st.subheader("🛠️ Active Tool Catalog")
catalog_cols = st.columns(3)
for idx, tool in enumerate(available_tools):
    with catalog_cols[idx]:
        with st.container(border=True):
            st.markdown(f"**`{tool['name']}`**")
            st.caption(tool['description'])
            st.code(json.dumps(tool['params'], indent=2), language="json")

st.divider()

user_test_msg = st.selectbox(
    "Select User Prompt to Test Tool Routing",
    [
        "What's the weather like in Tokyo right now in Celsius?",
        "Can you explain the difference between supervised and unsupervised learning?",
        "Look up account details for user sarah@example.com",
        "Refund transaction TX-88219 because the goods were damaged upon arrival."
    ]
)

col_out1, col_out2 = st.columns([1, 1])

with col_out1:
    st.subheader("📥 Processed User Input")
    st.write(f"**Query:** `{user_test_msg}`")

with col_out2:
    st.subheader("📤 Decision & Emitted Payload")
    if "Tokyo" in user_test_msg:
        st.success("🎯 Action: TOOL CALL EMITTED")
        call_payload = {
            "name": "get_current_weather",
            "arguments": {
                "city": "Tokyo",
                "units": "celsius"
            }
        }
        st.json(call_payload)
        st.caption("The model recognizes an external real-time data dependency and emits the tool call.")

    elif "supervised" in user_test_msg:
        st.info("💬 Action: DIRECT TEXT GENERATION (No Tool Required)")
        st.markdown(
            """
```text
Supervised learning trains models on labeled input-output pairs (e.g., classification, regression), 
whereas unsupervised learning finds hidden patterns, clusters, or representations in unlabeled data 
(e.g., k-means clustering, PCA).
```
*The query can be answered purely using parametric pre-trained knowledge; no tool call is triggered.*
"""
        )

    elif "sarah@example.com" in user_test_msg:
        st.success("🎯 Action: TOOL CALL EMITTED")
        call_payload = {
            "name": "search_customer_account",
            "arguments": {
                "email": "sarah@example.com"
            }
        }
        st.json(call_payload)

    else:
        st.success("🎯 Action: TOOL CALL EMITTED")
        call_payload = {
            "name": "refund_transaction",
            "arguments": {
                "tx_id": "TX-88219",
                "reason": "goods damaged upon arrival"
            }
        }
        st.json(call_payload)

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: OpenAI Function Calling", divider="rainbow")
st.markdown("Python implementation demonstrating Pydantic schema generation and tool handling:")

st.code(
    '''from pydantic import BaseModel, Field
from typing import Literal
import json

# 1. Define Tool Arguments using Pydantic
class WeatherToolArgs(BaseModel):
    city: str = Field(description="The city name, e.g. San Francisco or Tokyo")
    unit: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="Temperature measurement unit")

# 2. Convert Pydantic Model to OpenAI Tool Definition
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get current weather conditions and temperature for a given city.",
        "parameters": WeatherToolArgs.model_json_schema()
    }
}

# 3. Simulate Client Execution Loop
def execute_tool_call(tool_name: str, arguments_json: str) -> str:
    args_dict = json.loads(arguments_json)
    if tool_name == "get_current_weather":
        city = args_dict.get("city")
        # In production: call actual weather API
        return json.dumps({"city": city, "temperature": 22, "condition": "Sunny", "units": args_dict.get("unit")})
    return json.dumps({"error": f"Unknown tool: {tool_name}"})

print("Tool Definition Schema Generated:")
print(json.dumps(weather_tool, indent=2))
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: Does the LLM itself execute the Python or SQL code when it calls a tool?", expanded=True):
        st.markdown(
            """
**No.** This is the most common misconception. 
1. The LLM only generates **text tokens representing a JSON schema**.
2. The LLM's inference engine pauses. The client application (your backend server) receives the JSON payload.
3. Your server validates the arguments and runs the actual Python function, database query, or REST API call.
4. Your server formats the execution output into a `"role": "tool"` message and sends it back to the LLM to resume generation.
"""
        )
    with st.expander("Q2: What is `tool_choice` in the OpenAI API and when should you use `'required'` or a specific tool name?"):
        st.markdown(
            """
- `tool_choice="auto"` (Default): The model autonomously decides whether to answer in natural language or call a tool.
- `tool_choice="none"`: Forbids the model from calling tools; forces text output.
- `tool_choice="required"`: Forces the model to call AT LEAST ONE tool, but lets it choose which one.
- `tool_choice={"type": "function", "function": {"name": "X"}}`: Forces the model to call specifically tool `X`, bypassing all natural language chat decisions. Used in strict deterministic pipeline stages.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can we give the model 50 different tools in the API payload?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, more tools give the model more superpowers."*
- **Winning Response**: *"No. Supplying 50 tools severely degrades tool selection accuracy (tool hallucination / calling the wrong function) and bloats the prompt token prefill phase ($0.05+ per call before the user even types a word). In enterprise architectures with dozens of tools, use a **Tool Retrieval Layer**: embed the tool descriptions in a vector index, retrieve only the top 3-5 relevant tools for that specific user turn, and pass only that minimal subset to the LLM."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Tool Calling**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** The capability of an LLM to emit structured JSON function names and arguments for client-side execution.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**tool_choice='required'**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** An API parameter forcing the LLM to invoke a tool rather than responding in conversational text.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Tool Retrieval Layer**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Dynamically searching a vector index to inject only the 3-5 most relevant tool schemas into prompt context to prevent tool confusion.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 17.1 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Model as Orchestrator**: The LLM emits JSON arguments; your application executes the actual code.
2. **Detailed Descriptions**: Tool descriptions are critical for self-attention routing.
3. **Control Tool Choice**: Use `tool_choice="required"` to guarantee function execution in deterministic flows.
4. **Tool Pruning**: Never pass dozens of tools at once; dynamically retrieve the top-3 to 5 tools.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand the client-LLM tool execution loop.", value=True, key="chk_17_1_1")
    st.checkbox("I know how to define OpenAI/Anthropic tool schemas using Pydantic.", value=True, key="chk_17_1_2")
    st.checkbox("I understand the different options for tool_choice ('auto', 'required', specific function).", value=True, key="chk_17_1_3")
    st.checkbox("I know why large tool catalogs require dynamic tool retrieval.", value=True, key="chk_17_1_4")
