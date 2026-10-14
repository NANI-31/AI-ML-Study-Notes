import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Resource Caching with @st.cache_resource",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Resource Caching with `@st.cache_resource` — Complete Guide")
st.markdown(
    """
While `@st.cache_data` is built for serializable data values (DataFrames, JSON, lists) and 
returns a fresh copy on every call, **`@st.cache_resource` is designed for global singletons**. 
It caches **non-serializable, stateful objects** — such as PyTorch/Hugging Face model weights, 
database connection pools, OpenAI/Anthropic client sessions, and thread locks — and returns 
**the exact same object reference** across all user sessions and reruns.
"""
)

st.divider()

# ==============================================================================
# 1. Understanding @st.cache_resource vs. @st.cache_data
# ==============================================================================
st.header("1. Core Architectural Difference: Data Cache vs. Resource Cache", divider="rainbow")
st.markdown(
    r"""
Understanding when to use `@st.cache_data` versus `@st.cache_resource` is critical:

| Architectural Property | `@st.cache_data` | `@st.cache_resource` |
| :--- | :--- | :--- |
| **Primary Target** | Data, computations, queries | Stateful singletons, connections, models |
| **Object Type** | Serializable (Picklable) | Any Python object (non-serializable ok) |
| **Return Mechanism** | **Returns a COPY** (Mutation safe) | **Returns EXACT SAME REFERENCE** |
| **Scope** | Session / User data | Shared globally across ALL connected users |
| **Memory Behavior** | Can write to disk (`persist="disk"`) | Lives strictly in process memory (RAM) |
| **Use Cases** | `df = pd.read_csv()`, API responses | `model = load_model()`, `conn = db.connect()` |
"""
)

# ==============================================================================
# 2. Caching Machine Learning Models
# ==============================================================================
st.header("2. Caching Machine Learning & Neural Network Models", divider="rainbow")
st.markdown(
    """
Loading a modern Transformer or neural network (e.g. BERT, Llama, Whisper) involves 
reading gigabytes of tensor weights from disk and loading them onto GPU VRAM. 
Doing this on every rerun causes catastrophic slowdowns and immediate Out-Of-Memory (OOM) crashes.
With `@st.cache_resource`, the model is loaded **only once** for the entire server lifespan!
"""
)

col_ml1, col_ml2 = st.columns(2)

with col_ml1:
    st.markdown("**💻 Python Source Code (Cached Model Singleton)**")
    code_model = '''# Simulated Neural Classifier
class SentimentClassifier:
    def __init__(self, model_name: str):
        time.sleep(2.0)  # Simulates loading 2GB weights
        self.model_name = model_name
    
    def predict(self, text: str) -> dict:
        return {"sentiment": "Positive", "confidence": 0.94}

# ✅ Load once; shared across all users and reruns:
@st.cache_resource(show_spinner="Loading PyTorch model weights...")
def get_classifier(model_name: str = "distilbert-base"):
    return SentimentClassifier(model_name)

# In your UI:
model = get_classifier()  # <1ms on all subsequent calls!
result = model.predict("Streamlit makes caching easy!")'''
    st.code(code_model, language="python")

with col_ml2:
    st.markdown("**🖥️ Live Interactive Model Loading Simulation**")
    class MockTransformerModel:
        def __init__(self, name: str):
            time.sleep(1.5)
            self.name = name
            self.loaded_at = datetime.now().strftime("%H:%M:%S")

        def infer(self, text: str):
            return {"label": "Bullish / Positive", "confidence": 0.962, "tokens": len(text.split())}

    @st.cache_resource(show_spinner="Mounting neural network into VRAM...")
    def load_cached_transformer(name: str):
        return MockTransformerModel(name)

    sel_model_type = st.selectbox("Select Model Architecture", ["DistilBERT-Finance", "RoBERTa-Sentiment"], key="sel_model_res")
    
    t_mod0 = time.perf_counter()
    active_model = load_cached_transformer(sel_model_type)
    t_mod_ms = (time.perf_counter() - t_mod0) * 1000

    sample_prompt = st.text_input("Enter text for inference:", "AI infrastructure spending hit record highs this quarter.", key="txt_inf_prompt")
    inference_out = active_model.infer(sample_prompt)

    with st.container(border=True):
        if t_mod_ms > 600:
            st.warning(f"🐢 **Model Loaded From Disk**: Took **{t_mod_ms/1000:.2f}s** (Loaded at: `{active_model.loaded_at}`)")
        else:
            st.success(f"⚡ **Model Singleton Reused**: Instant **{t_mod_ms:.2f} ms** (Mounted at: `{active_model.loaded_at}`)")
        st.write(f"Inference Result: `{inference_out}`")

st.divider()

# ==============================================================================
# 3. Caching Database Connections & Client Pools
# ==============================================================================
st.header("3. Caching Database Connections & Client Pools", divider="rainbow")
st.markdown(
    """
Database connections (PostgreSQL, Snowflake, Pinecone, Redis) maintain stateful TCP sockets 
and internal connection pools. If you open a new connection on every script rerun, 
you will rapidly exhaust database connection limits.
"""
)

col_db1, col_db2 = st.columns(2)

with col_db1:
    st.markdown("**💻 Python Source Code (Cached Database Singleton)**")
    code_db = '''class DatabasePool:
    def __init__(self, dsn: str):
        time.sleep(1.0)  # Establish TCP handshake
        self.dsn = dsn
        self.active_queries = 0

    def query(self, sql: str):
        self.active_queries += 1
        return pd.DataFrame({"id": [1, 2], "status": ["OK", "OK"]})

# Cache connection pool globally:
@st.cache_resource
def get_db_connection(dsn: str):
    return DatabasePool(dsn)

conn = get_db_connection("postgresql://user:pass@host/db")
df = conn.query("SELECT * FROM metrics")'''
    st.code(code_db, language="python")

with col_db2:
    st.markdown("**🖥️ Live Interactive DB Pool Demo**")
    class MockDatabasePool:
        def __init__(self, uri: str):
            time.sleep(1.0)
            self.uri = uri
            self.session_id = f"TCP-CONN-{np.random.randint(1000, 9999)}"
            self.queries_served = 0

        def execute_query(self, sql: str):
            self.queries_served += 1
            return self.queries_served

    @st.cache_resource
    def get_global_db_pool(uri: str):
        return MockDatabasePool(uri)

    t_db0 = time.perf_counter()
    pool = get_global_db_pool("postgres://ai-analytics-db")
    t_db_ms = (time.perf_counter() - t_db0) * 1000

    q_count = pool.execute_query("SELECT 1")

    with st.container(border=True):
        st.markdown(f"**Connection Session ID**: `{pool.session_id}` *(Shared Socket)*")
        st.markdown(f"**Total Queries Executed on This Socket**: `{q_count}`")
        if t_db_ms > 400:
            st.warning(f"🐢 Initial Handshake Established in **{t_db_ms/1000:.2f}s**")
        else:
            st.success(f"⚡ Socket Reused Instantly in **{t_db_ms:.2f} ms**")

st.divider()

# ==============================================================================
# 4. Caching Shared Resources & 5. Resource Lifecycle
# ==============================================================================
st.header("4. Shared Resources & 5. Resource Lifecycle Management", divider="rainbow")
st.markdown(
    r"""
### Lifecycle of an `@st.cache_resource` Object
1. **Instantiation**: Function runs on first invocation; object is created and placed in memory.
2. **Reuse**: Every subsequent call across every user tab receives the exact same pointer in RAM.
3. **Destruction / Invalidation**: 
   - Server process terminates or restarts.
   - You call `get_resource.clear()`.
   - File modifying the function body is edited on disk.
"""
)

col_lc1, col_lc2 = st.columns(2)

with col_lc1:
    st.markdown("**💻 Python Source Code (Object ID Verification)**")
    code_obj = '''@st.cache_resource
def get_shared_service():
    return {"created_at": time.time()}

# Call A and Call B return the exact same Python object pointer in memory:
obj1 = get_shared_service()
obj2 = get_shared_service()

assert id(obj1) == id(obj2)  # Identical memory address!'''
    st.code(code_obj, language="python")

with col_lc2:
    st.markdown("**🖥️ Live Interactive Memory Address Inspector**")
    @st.cache_resource
    def get_singleton_resource():
        return {"id": "RESOURCE-SINGLETON-ALPHA"}

    res_ptr = get_singleton_resource()
    memory_addr = hex(id(res_ptr))

    with st.container(border=True):
        st.markdown("##### 🔍 Global Memory Address Inspector")
        st.write(f"• **Object Pointer (RAM Address)**: `{memory_addr}`")
        st.write(f"• **Object Identifier**: `{res_ptr['id']}`")
        st.caption("Interact with widgets on this page: the hexadecimal RAM address remains constant, proving that zero copies are made.")

st.divider()

# ==============================================================================
# 6. Thread-Safety Considerations & 7. Resource Reuse
# ==============================================================================
st.header("6. Thread-Safety Considerations & 7. Safe Resource Reuse", divider="rainbow")
st.markdown(
    r"""
### The Multi-Threading Warning
Streamlit runs user sessions concurrently on multiple background threads! 
Because `@st.cache_resource` returns the **exact same object pointer** to all concurrent users:
- **Read-Only Objects (Safe)**: ML models in eval mode (`model.eval()`), read-only configuration trees.
- **Stateful / Mutating Objects (Unsafe without Locks)**: In-memory queues, shared dictionaries where users append data simultaneously.

### The Fix: Thread Locks (`threading.Lock`)
```python
import threading

class ThreadSafeBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = []

    def append_safe(self, item):
        with self.lock:  # Prevents race conditions!
            self.data.append(item)
```
"""
)

col_ts1, col_ts2 = st.columns(2)

with col_ts1:
    st.markdown("**💻 Python Source Code (Thread-Safe Shared Cache)**")
    code_lock = '''import threading

class SafeTelemetryBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._logs = []

    def log(self, msg: str):
        with self._lock:  # Thread-safe write
            self._logs.append(msg)

    def get_logs(self):
        with self._lock:
            return list(self._logs)

@st.cache_resource
def get_global_logger():
    return SafeTelemetryBuffer()'''
    st.code(code_lock, language="python")

with col_ts2:
    st.markdown("**🖥️ Live Interactive Thread-Safe Queue Demo**")
    import threading

    class SafeQueue:
        def __init__(self):
            self.lock = threading.Lock()
            self.items = ["Server boot initialized"]

        def add(self, item):
            with self.lock:
                self.items.append(item)

        def count(self):
            with self.lock:
                return len(self.items)

    @st.cache_resource
    def get_global_queue():
        return SafeQueue()

    q = get_global_queue()

    if st.button("Add Event to Global Thread-Safe Queue", key="btn_add_q"):
        q.add(f"Event logged at {datetime.now().strftime('%H:%M:%S')}")
        st.rerun()

    with st.container(border=True):
        st.write(f"• Total events queued: **{q.count()}**")
        st.json(q.items[-4:])

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 `@st.cache_resource` vs. `@st.cache_data` Decision Guide", divider="rainbow")
st.markdown(
    r"""
| Decision Question | `@st.cache_data` | `@st.cache_resource` |
| :--- | :--- | :--- |
| **Can the output be pickled?** | Yes (Pandas, JSON, dicts, arrays) | Irrelevant (Any object supported) |
| **Is it safe to share the same pointer?** | No $\to$ Returns deep copy | Yes $\to$ Returns exact reference |
| **Should it survive server restarts?** | Yes (`persist="disk"`) | No (Lives only in RAM) |
| **Is it a database connection pool?** | ❌ Never | ✅ Always |
| **Is it an ML model (PyTorch, TensorFlow)?** | ❌ Never (Copies waste GBs) | ✅ Always (Shared VRAM pointer) |
| **Is it a DataFrame query result?** | ✅ Always | ❌ Avoid |
| **Thread-Safety requirement** | Automatic (each user gets copy) | Manual lock needed if mutating |
"""
)
