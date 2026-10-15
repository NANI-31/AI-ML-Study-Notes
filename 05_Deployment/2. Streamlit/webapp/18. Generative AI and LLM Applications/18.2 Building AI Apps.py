import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="18.2 Building AI Apps",
    page_icon="🤖",
    layout="wide"
)

st.title("18.2 Building Generative AI & Chat Apps in Streamlit")
st.caption("Native Streamlit GenAI primitives: st.chat_message, st.chat_input, conversational session state, st.write_stream token streaming, token telemetry, and reset workflows.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The Streamlit Conversational AI Architecture

Streamlit provides first-class native primitives for chat applications:
- **`st.chat_input`**: A pinned, auto-clearing bottom text entry bar.
- **`st.chat_message`**: Visual conversation containers for `"user"`, `"assistant"`, and custom personas.
- **`st.session_state["messages"]`**: The persistent multi-turn message history dictionary list.
- **`st.write_stream`**: Real-time token streaming consuming Python generators directly.

""")

st.graphviz_chart(
    r"""
    digraph ChatApp_Architecture {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        subgraph cluster_ui {
            label="Streamlit Reactive Viewport";
            fontname="Helvetica"; fontsize=11; color="#94A3B8"; style="rounded,dashed";
            user_msg [label="st.chat_message('user')\nDisplays submitted user prompt", fillcolor="#EFF6FF", color="#3B82F6"];
            ai_msg [label="st.chat_message('assistant')\nst.write_stream(token_generator)", fillcolor="#F0FDF4", color="#16A34A"];
            chat_in [label="prompt = st.chat_input('Ask a question...')\nCaptures submission & triggers rerun", fillcolor="#FEF3C7", color="#D97706"];
        }

        state [label="st.session_state['messages']\n• Persistent Multi-Turn ChatML History\n• Token Telemetry Tracker", fillcolor="#EDE9FE", color="#7C3AED"];

        chat_in -> state [label="Appends user prompt"];
        state -> user_msg [label="Restores history on rerun"];
        state -> ai_msg [label="Streams response tokens"];
    }
    """
)


tabs = st.tabs([
    "1. Text Generation",
    "2. st.chat_message",
    "3. st.chat_input",
    "4. Chat History State",
    "5. Streaming Responses",
    "6. Loading & Errors",
    "7. Token Telemetry",
    "8. Conversation Reset"
])

# =============================================================================
# 1. TEXT GENERATION
# =============================================================================
with tabs[0]:
    st.header("1. Single-Turn Text Generation Interfaces")
    st.markdown(r"""
    Single-turn AI tools (summarizers, code converters, grammar polishers) operate on discrete forms 
    without requiring persistent multi-turn conversational history.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

with st.form("summarizer_form"):
    st.subheader("Executive Document Summarizer")
    source_text = st.text_area("Input Text to Synthesize", height=120)
    target_format = st.selectbox("Summary Format", ["3 Bullet Points", "Executive TL;DR", "Technical Abstract"])
    
    submitted = st.form_submit_button("Generate Summary", type="primary")

if submitted and source_text:
    # Simulated model response
    st.success("Summary Generated:")
    st.markdown("""
    - **Key Finding 1**: Inference latency reduced by 42% through INT8 quantization.
    - **Key Finding 2**: Model accuracy maintained within 0.3% of FP16 baseline.
    - **Key Finding 3**: Hardware compute cost reduced from $1.20/hr to $0.45/hr.
    """)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            with st.form("demo_summarizer_form"):
                st.caption("Executive Document Summarizer:")
                s_text = st.text_area("Source Text", "Modern inference serving requires quantization and streaming.", height=80, key="sum_in_ta")
                t_fmt = st.selectbox("Format", ["3 Bullet Points", "Executive TL;DR"], key="sum_fmt_sel")
                sub_btn = st.form_submit_button("Generate Summary", type="primary")

            if sub_btn:
                st.success("Summary Generated:")
                st.markdown("""
                - **Key Finding 1**: Inference latency reduced by 42% through INT8 quantization.
                - **Key Finding 2**: Memory footprint decreased from 16GB to 4.2GB.
                """)

# =============================================================================
# 2. ST.CHAT_MESSAGE
# =============================================================================
with tabs[1]:
    st.header("2. Chat Message Containers (`st.chat_message`)")
    st.markdown(r"""
    `st.chat_message(name, avatar=None)` creates dedicated conversational containers.
    - Standard roles: `"user"` and `"assistant"`.
    - Custom avatars: Emojis (`"🤖"`, `"👤"`), image file paths, or remote URLs.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# User turn
with st.chat_message("user", avatar="👤"):
    st.write("How do vector databases perform approximate nearest neighbor search?")

# Assistant turn
with st.chat_message("assistant", avatar="🤖"):
    st.write("""
    Vector databases use indexing structures such as:
    1. **HNSW (Hierarchical Navigable Small World)**: Graph-based navigation.
    2. **IVF-PQ (Inverted File with Product Quantization)**: Voronoi cell partitioning.
    """)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            with st.chat_message("user", avatar="👤"):
                st.write("How do vector databases perform approximate nearest neighbor search?")

            with st.chat_message("assistant", avatar="🤖"):
                st.write(r"""
                Vector databases use indexing algorithms:
                1. **HNSW**: Graph-based multi-layer search with $O(\log N)$ complexity.
                2. **IVF-PQ**: Voronoi clustering combined with vector compression.
                """)

# =============================================================================
# 3. ST.CHAT_INPUT
# =============================================================================
with tabs[2]:
    st.header("3. Bottom-Pinned Chat Input (`st.chat_input`)")
    st.markdown(r"""
    `st.chat_input(placeholder=...)` renders an input bar that automatically clears its text upon enter/submission 
    and sticks to the bottom of the viewport.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Returns string on submission, None otherwise
user_prompt = st.chat_input("Ask anything about your ML models...")

if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)
        
    with st.chat_message("assistant"):
        st.write(f"Received prompt: '{user_prompt}'")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### `st.chat_input` Key Properties

            - **Auto-Clearing**: Text is submitted and cleared in a single user action.
            - **Disabled State**: Can be locked during generation via `st.chat_input(..., disabled=True)`.
            - **Single per Page**: Only one `st.chat_input` may be active on a given page.
            """)
            st.info("💡 See Tab 4 for interactive multi-turn session state integration.")

# =============================================================================
# 4. CHAT HISTORY STATE
# =============================================================================
with tabs[3]:
    st.header("4. Managing Multi-Turn Chat History in Session State")
    st.markdown(r"""
    Because Streamlit reruns the script on every submission, past turns must be replayed from `st.session_state["messages"]`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Complete Chat Loop Pattern**")
        st.code(r'''
import streamlit as st

# 1. Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Render historical messages from state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Accept new user input
if prompt := st.chat_input("Enter message..."):
    # Append user turn to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and append assistant turn
    response = f"Echo: {prompt}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            if "demo_chat_history" not in st.session_state:
                st.session_state.demo_chat_history = [
                    {"role": "user", "content": "What is INT8 quantization?"},
                    {"role": "assistant", "content": "INT8 converts 32-bit floating point weights into 8-bit integers, reducing memory by 75%."}
                ]

            for m in st.session_state.demo_chat_history:
                with st.chat_message(m["role"]):
                    st.write(m["content"])

            st.caption("Historical messages replay sequentially from `st.session_state`.")

# =============================================================================
# 5. STREAMING RESPONSES
# =============================================================================
with tabs[4]:
    st.header("5. Streaming Model Responses (`st.write_stream`)")
    st.markdown(r"""
    `st.write_stream` accepts an iterable/generator of text chunks, renders them with a typing effect, 
    and **returns the complete concatenated response string** so you can append it directly to session state.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import time

def fake_llm_stream():
    chunks = [
        "Reinforcement ", "Learning ", "from ", "Human ", "Feedback ", 
        "(RLHF) ", "aligns ", "large ", "models ", "with ", "human ", "intent."
    ]
    for chunk in chunks:
        yield chunk
        time.sleep(0.05)

if st.button("Simulate Streaming"):
    with st.chat_message("assistant"):
        # st.write_stream yields typing effect and returns full string!
        full_text = st.write_stream(fake_llm_stream())
        
    st.caption(f"Saved full response: '{full_text}'")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            def demo_stream_gen():
                chunks = [
                    "Reinforcement ", "Learning ", "from ", "Human ", "Feedback ", 
                    "(RLHF) ", "aligns ", "large ", "models ", "with ", "human ", "intent."
                ]
                for chunk in chunks:
                    yield chunk
                    time.sleep(0.04)

            if st.button("Simulate Streaming", key="chat_stream_demo_btn"):
                with st.chat_message("assistant"):
                    full_res = st.write_stream(demo_stream_gen())
                st.caption(f"Captured output: `{full_res}`")
            else:
                st.info("Click the button above to observe `st.write_stream` in action.")

# =============================================================================
# 6. LOADING & ERRORS
# =============================================================================
with tabs[5]:
    st.header("6. Handling Loading & Upstream Error States")
    st.markdown(r"""
    Network timeouts, rate limits (HTTP 429), and content filter blocks must be trapped gracefully 
    without displaying raw tracebacks.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

with st.chat_message("assistant"):
    with st.spinner("Model is reasoning..."):
        try:
            # Simulated upstream call
            raise ConnectionError("Gateway timeout after 15s")
        except ConnectionError:
            st.error("🚨 Upstream Gateway Timeout: The LLM provider took longer than 15s to respond.")
            st.caption("Action: Retry query or switch to local Ollama fallback engine.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            with st.chat_message("assistant"):
                st.error("🚨 Upstream Gateway Timeout: The LLM provider took longer than 15s to respond.")
                st.caption("Action: Retry query or switch to local Ollama fallback engine.")

# =============================================================================
# 7. TOKEN TELEMETRY
# =============================================================================
with tabs[6]:
    st.header("7. Real-Time Token Usage & Cost Telemetry")
    st.markdown(r"""
    Track cumulative session consumption in a sidebar ribbon to provide cost visibility to users.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

if "prompt_tokens" not in st.session_state:
    st.session_state.prompt_tokens = 1420
if "completion_tokens" not in st.session_state:
    st.session_state.completion_tokens = 3850

# Pricing: $2.50 / 1M prompt, $10.00 / 1M completion
total_cost = (
    (st.session_state.prompt_tokens / 1e6) * 2.50 +
    (st.session_state.completion_tokens / 1e6) * 10.00
)

st.sidebar.subheader("Session Token Telemetry")
st.sidebar.metric("Prompt Tokens", f"{st.session_state.prompt_tokens:,}")
st.sidebar.metric("Completion Tokens", f"{st.session_state.completion_tokens:,}")
st.sidebar.metric("Estimated Cost", f"${total_cost:.4f}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.caption("Active Session Telemetry HUD:")
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Prompt Tokens", "1,420")
            tc2.metric("Completion Tokens", "3,850")
            tc3.metric("Estimated Cost", "$0.0421")
            st.info("💡 Real-time token tracking enables quota enforcement and prevents surprise cloud bills.")

# =============================================================================
# 8. CONVERSATION RESET
# =============================================================================
with tabs[7]:
    st.header("8. Conversation Reset & State Cleanup")
    st.markdown(r"""
    A conversation reset clears the message history list and triggers an immediate script rerun via `st.rerun()`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

if st.button("🗑️ Clear Conversation History"):
    st.session_state.messages = []
    st.session_state.prompt_tokens = 0
    st.session_state.completion_tokens = 0
    st.rerun()
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            if st.button("🗑️ Clear Conversation History", key="clear_chat_demo_btn"):
                st.session_state.demo_chat_history = []
                st.rerun()

            st.caption("Clicking reset purges `st.session_state['messages']` and returns to initial greeting state.")

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Building AI Apps Best Practices

| Concern | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Input Widgets** | Using `st.text_input` for chat conversations | Use native `st.chat_input` for bottom-pinned auto-clearing ergonomics. |
| **Response Latency** | Blocking until the complete 800-word answer is finished | Use `st.write_stream` with an SSE generator. |
| **History Replay** | Forgetting to loop over `st.session_state.messages` on rerun | Re-render history on every script execution before accepting new input. |
| **Error Handling** | Letting raw OpenAI `APIConnectionError` break the app | Wrap in try/except and render user-friendly diagnostic guidance. |
""")
