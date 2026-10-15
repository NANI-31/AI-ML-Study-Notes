import streamlit as st
import pandas as pd
import time

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="1.4 Streamlit Architecture | Complete Guide",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ 1.4 Streamlit Architecture — Lifecycle, WebSockets & Internal Mechanics")
st.markdown(
    """
Welcome to **Lesson 1.4: Streamlit Architecture**.
To master Streamlit and pass senior technical interviews, you must understand what happens
beneath the surface when you run a script: the client-server boundary, the reactive rerun engine,
Tornado web server mechanics, Google Protobuf serialization, and WebSocket communication loops.
"""
)

st.divider()

# ==============================================================================
# 1. Python Script Execution Model
# ==============================================================================
st.header("1. The Top-to-Bottom Execution Model", divider="rainbow")

col_exec1, col_exec2 = st.columns(2)

with col_exec1:
    st.markdown(
        """
### The Procedural Paradigm
Unlike traditional web frameworks (Django, Flask, React) that rely on event-driven callback trees,
Streamlit executes code in a **linear, procedural top-to-bottom sequence**:

1. **No Event Listeners**: You never write `button.onClick(callback)` or `addEventListener()`.
2. **Widgets Return Values**: A widget function returns its current state immediately:
   ```python
   # The slider function halts nothing; it returns the current int value:
   threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)
   ```
3. **Execution Freshness**: On every user interaction, the script executes from line 1 to the final line.
4. **Declarative Rendering**: Python calls dynamically generate a virtual UI tree streamed directly to the browser.
"""
    )

with col_exec2:
    st.markdown("#### 🧪 Interactive Rerun Execution Demonstration")
    with st.container(border=True):
        st.write("Execution timestamp:", time.strftime("%H:%M:%S"))
        interactive_toggle = st.toggle("Toggle Model Pipeline", value=False, key="arch_demo_toggle")
        if interactive_toggle:
            st.success("✅ Model Pipeline is ACTIVE: Running inference branch.")
            st.code("def run_inference():\n    return 'Predictions generated!'", language="python")
        else:
            st.info("ℹ️ Pipeline is INACTIVE: Running data ingestion branch.")
        st.caption("Notice how changing the toggle above causes the entire script to rerun from line 1 and execute the corresponding branch.")

st.divider()

# ==============================================================================
# 2. Frontend vs Backend Concepts
# ==============================================================================
st.header("2. Frontend and Backend Architecture", divider="rainbow")

st.markdown(
    """
Streamlit is physically decoupled into two distinct runtime environments communicating across a network bridge:
"""
)

col_fe, col_be = st.columns(2)

with col_fe:
    st.markdown(
        """
### 🖥️ The Frontend (Browser / Client)
- **Technology Stack**: React, TypeScript, BaseWeb UI, Emotion (CSS-in-JS).
- **Runtime**: Runs completely inside the user's web browser (Chrome, Edge, Safari).
- **Responsibilities**:
  - Renders the virtual DOM component tree.
  - Listens to physical user events (clicks, mouse drags, keystrokes).
  - Maintains zero business logic—acts purely as a visual renderer.
  - Forwards user interaction payloads to the backend over WebSockets.
"""
    )

with col_be:
    st.markdown(
        """
### 🐍 The Backend (Python / Server)
- **Technology Stack**: Python, Tornado Async Web Server, PyArrow.
- **Runtime**: Runs on the server host machine or cloud container.
- **Responsibilities**:
  - Executes your Python application script (`app.py`).
  - Performs heavy computations, database queries, and ML inferences.
  - Maintains private in-memory session states (`st.session_state`).
  - Serializes UI commands into binary Protobuf deltas and streams them to the browser.
"""
    )

st.divider()

# ==============================================================================
# 3. Browser-Server Communication (The Protobuf Protocol)
# ==============================================================================
st.header("3. Browser-Server Communication & Serialization", divider="rainbow")

col_proto1, col_proto2 = st.columns(2)

with col_proto1:
    st.markdown(
        """
### Why Google Protocol Buffers (Protobuf)?
Traditional web apps pass JSON over HTTP. However, transferring heavy data science DataFrames and
high-frequency charts over JSON introduces massive serialization latency.

Streamlit solves this using **Google Protocol Buffers (Protobuf)**:
- **Binary Format**: Serializes structured data into compact, typed binary payloads.
- **Ultra-Low Overhead**: 3–10x smaller and significantly faster to decode than JSON.
- **Strict Schema Enforcement**: Ensures frontend and backend contract compliance.
"""
    )

with col_proto2:
    st.markdown(
        """
### Message Types: `ForwardMsg` vs `BackMsg`
1. **`ForwardMsg` (Server ➔ Browser)**:
   - Generated by Python commands (`st.title`, `st.dataframe`).
   - Carries UI **Deltas** instructions telling React what element to add, update, or remove.
2. **`BackMsg` (Browser ➔ Server)**:
   - Generated by user interaction (e.g., slider moved to `25`).
   - Carries the updated widget state and triggers a new script execution run.
"""
    )

st.divider()

# ==============================================================================
# 4. Script Rerun Lifecycle
# ==============================================================================
st.header("4. The Script Rerun Lifecycle Step-by-Step", divider="rainbow")

st.markdown(
    """
Here is the exact sequential timeline that occurs whenever a user interacts with a Streamlit app:
"""
)

lifecycle_steps = [
    {
        "Step": "1. User Interaction",
        "Actor": "Browser (React)",
        "Action": "User adjusts a slider, clicks a button, or uploads a file.",
    },
    {
        "Step": "2. State Packaging",
        "Actor": "Browser (React)",
        "Action": "Frontend packages widget ID and new value into a binary BackMsg payload.",
    },
    {
        "Step": "3. WebSocket Transmission",
        "Actor": "Network",
        "Action": "Transmits BackMsg over the persistent WebSocket connection to the Tornado server.",
    },
    {
        "Step": "4. ScriptRunner Spawn",
        "Actor": "Tornado Server",
        "Action": "Tornado assigns or restarts an isolated worker ScriptRunner thread for this session.",
    },
    {
        "Step": "5. Top-to-Bottom Execution",
        "Actor": "Python Runtime",
        "Action": "The Python script executes from line 1. Widget functions return the new values.",
    },
    {
        "Step": "6. Delta Streaming",
        "Actor": "Python ➔ Tornado",
        "Action": "As each line executes, DeltaGenerator emits ForwardMsg deltas to the frontend.",
    },
    {
        "Step": "7. DOM Reconciliation",
        "Actor": "Browser (React)",
        "Action": "React updates only the modified virtual DOM nodes. The page never reloads.",
    },
]

st.dataframe(pd.DataFrame(lifecycle_steps), width="stretch", hide_index=True)

st.divider()

# ==============================================================================
# 5. Widget State Management
# ==============================================================================
st.header("5. Widget State Management", divider="rainbow")

col_w1, col_w2 = st.columns(2)

with col_w1:
    st.markdown(
        """
### How Widgets Retain Values Across Reruns
If the entire script reruns from top to bottom on every click, why don't widget values reset to their initial defaults?

- **Widget Identity**: Every widget has a unique internal key computed from its label, parameters, and explicit `key` argument.
- **The Widget Manager**: Before the script reruns, Streamlit's backend caches the latest user-provided value.
- When execution reaches the widget call again:
  ```python
  # On rerun, Streamlit checks the widget manager:
  # Finds value 85 from user input -> returns 85 instead of default 10!
  score = st.slider("Exam Score", 0, 100, 10, key="score_slider")
  ```
"""
    )

with col_w2:
    st.markdown(
        """
### ⚠️ The Conditional Widget Pruning Gotcha
If a widget is inside an `if` block that evaluates to `False`:
```python
if show_advanced:
    learning_rate = st.slider("LR", 0.01, 1.0, key="lr")
```
- During a run where `show_advanced` is `False`, Streamlit notices the `lr` widget was never instantiated.
- **Automatic Pruning**: Streamlit automatically deletes its key from session memory to prevent memory leaks!
"""
    )

st.divider()

# ==============================================================================
# 6. Session Management (st.session_state)
# ==============================================================================
st.header("6. Session Management & Multi-User Isolation", divider="rainbow")

col_sess1, col_sess2 = st.columns(2)

with col_sess1:
    st.markdown(
        """
### 🔒 Complete Multi-User Isolation
- Every browser tab or window that connects to Streamlit receives a unique **`SessionId` (UUID)**.
- **Independent States**: User A selecting "Option 1" will **never** alter what User B sees in their session.
- `st.session_state` behaves like a private Python dictionary tied exclusively to that single user session.

#### ⚠️ Global Variables Danger:
```python
# DANGEROUS: Shared across ALL users!
SHARED_CACHE = []

# SAFE & ISOLATED: Private to the current user tab
if "user_data" not in st.session_state:
    st.session_state.user_data = []
```
"""
    )

with col_sess2:
    st.markdown("#### 🧪 Live Session State Statefulness Demo")
    with st.container(border=True):
        if "session_items" not in st.session_state:
            st.session_state.session_items = ["Item A", "Item B"]

        new_item = st.text_input("Add an item to session state:", value="Item C", key="arch_new_item_input")
        if st.button("Append to Session List"):
            st.session_state.session_items.append(new_item)

        st.write("Current Session Items:", st.session_state.session_items)
        st.caption("Values appended to `st.session_state` persist across all top-to-bottom reruns.")

st.divider()

# ==============================================================================
# 7. Streamlit Server & Tornado Backend
# ==============================================================================
st.header("7. The Streamlit Server (Tornado Architecture)", divider="rainbow")

col_tor1, col_tor2 = st.columns(2)

with col_tor1:
    st.markdown(
        """
### Why Tornado?
Streamlit is powered by **Tornado**, a high-performance Python web framework and asynchronous networking library:

- **Asynchronous Event Loop**: Uses non-blocking I/O to handle thousands of open WebSocket connections simultaneously.
- **Low Memory Footprint**: Idle connections consume minimal RAM.
- **Thread Separation**:
  - The **Tornado Event Loop** runs on the main thread, handling network events.
  - User scripts run inside dedicated worker **`ScriptRunner` threads**.
  - Long computations in your script will **not** block network heartbeats or freeze other users' connections!
"""
    )

with col_tor2:
    st.markdown("### 🖥️ High-Resolution Vector Architecture Diagram")
    arch_dot = """
    digraph G {
        rankdir=TB;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#1E293B", fontcolor="#F8FAFC", fontname="Helvetica", penwidth=1.5, color="#475569"];
        edge [color="#38BDF8", fontcolor="#94A3B8", fontname="Helvetica", fontsize=10];

        subgraph cluster_clients {
            label="Client Tier (Web Browsers)";
            fontcolor="#94A3B8";
            style="dashed";
            color="#334155";
            B1 [label="Browser 1\\n(React SPA)", fillcolor="#0F766E"];
            B2 [label="Browser 2\\n(React SPA)", fillcolor="#0F766E"];
        }

        subgraph cluster_server {
            label="Streamlit Host Server Runtime (Port 8501)";
            fontcolor="#94A3B8";
            style="rounded";
            color="#475569";

            Tornado [label="Tornado Web Server\\n(Async Event Loop)", fillcolor="#1E3A8A", shape=component];
            
            SR1 [label="ScriptRunner Worker 1\\n(User 1 Session)\\nst.session_state", fillcolor="#4338CA"];
            SR2 [label="ScriptRunner Worker 2\\n(User 2 Session)\\nst.session_state", fillcolor="#4338CA"];
        }

        B1 -> Tornado [label="WebSocket /ws", dir="both"];
        B2 -> Tornado [label="WebSocket /ws", dir="both"];
        Tornado -> SR1 [label="Spawns Worker"];
        Tornado -> SR2 [label="Spawns Worker"];
    }
    """
    st.graphviz_chart(arch_dot)

st.divider()

# ==============================================================================
# 8. WebSocket Communication Concepts
# ==============================================================================
st.header("8. WebSocket Communication Concepts", divider="rainbow")

col_ws1, col_ws2 = st.columns(2)

with col_ws1:
    st.markdown(
        """
### HTTP Polling vs. WebSockets
- **Standard HTTP (Request / Response)**:
  - The client must repeatedly poll the server (`GET /data`) to check for updates.
  - High latency, significant HTTP header overhead, and inefficient.
- **WebSockets (`ws://localhost:8501/ws`)**:
  - **Full-Duplex**: Both client and server can send messages at any instant without waiting for requests.
  - **Single Handshake**: Starts as standard HTTP, then upgrades (`101 Switching Protocols`) to a persistent TCP socket.
  - **Real-Time Streaming**: Allows commands like `st.write_stream()` and progress bars (`st.progress()`) to stream tokens dynamically!
"""
    )

with col_ws2:
    st.markdown(
        """
### WebSocket Lifecycle in Streamlit
1. **Connection Handshake**: Browser requests upgrade to `GET /_stcore/stream` or `/ws`.
2. **Heartbeat Pings**: Client and server exchange regular ping/pong packets every few seconds to verify connection liveness.
3. **Reconnection Resilience**: If network drops temporarily (e.g. laptop sleeps), the frontend automatically attempts reconnection and state recovery.
4. **Disconnection Pruning**: When a user closes the browser tab, the server tears down the `ScriptRunner` thread and frees session memory.
"""
    )

st.divider()

# ==============================================================================
# Summary: Architecture Interview Master Checklist
# ==============================================================================
st.header("🎯 Summary: Key Takeaways for Technical Interviews", divider="rainbow")

st.markdown(
    """
1. **Procedural Execution**: Streamlit has no callback trees; user interactions trigger a clean top-to-bottom re-execution of the Python script.
2. **Separation of Concerns**: Python backend executes data logic; React frontend renders the virtual DOM.
3. **Protobuf Optimization**: UI deltas and interaction payloads are transmitted as compact binary Google Protocol Buffers over WebSockets.
4. **Session Isolation**: Each browser tab possesses an isolated `SessionId` and private `st.session_state`. Never store user-specific mutable data in global module variables.
5. **Tornado Concurrency**: The Tornado async event loop separates network transport from Python `ScriptRunner` worker threads, enabling high multi-user scalability.
"""
)
