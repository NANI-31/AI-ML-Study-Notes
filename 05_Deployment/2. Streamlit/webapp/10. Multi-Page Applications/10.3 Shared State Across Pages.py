import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Shared State Across Pages",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 Shared State Across Pages — Complete Guide")
st.markdown(
    """
In a multi-page Streamlit application, `st.session_state` is **globally shared across all pages 
within the user's browser session**. When a user navigates between pages, variables stored in 
`st.session_state` persist. However, without disciplined state patterns, multi-page applications 
frequently suffer from **widget key collisions**, **accidental state resets**, and **leaky abstractions**.
"""
)

st.divider()

# ==============================================================================
# 1. Sharing Session State Across Pages
# ==============================================================================
st.header("1. How Session State Operates Across Pages", divider="rainbow")
st.markdown(
    r"""
### The Single-Session Mental Model
There is **only ONE `st.session_state` per user session**. 
Navigating to a different page does **NOT** destroy or clear session state:
"""
)

st.graphviz_chart(
    r"""
    digraph SharedStateMentalModel {
        graph [bgcolor="transparent", rankdir=LR];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        session [label="User Browser Session\n(Isolated per Client Tab)", fillcolor="#EFF6FF", color="#3B82F6"];
        state [label="st.session_state (Global State Dictionary)\nShared across all page routes", fillcolor="#DBEAFE", color="#1D4ED8"];
        
        k1 [label="global_auth_user\n'sarah@ai.org'", fillcolor="#FEF3C7", color="#D97706"];
        k2 [label="global_selected_model\n'Claude-3.7'", fillcolor="#FEF3C7", color="#D97706"];
        k3 [label="page_telemetry_timeframe\n'7D'", fillcolor="#D1FAE5", color="#059669"];
        k4 [label="page_eval_min_score\n0.85", fillcolor="#D1FAE5", color="#059669"];

        session -> state;
        state -> k1;
        state -> k2;
        state -> k3;
        state -> k4;
    }
    """
)

col_ss1, col_ss2 = st.columns(2)

with col_ss1:
    st.markdown("**💻 Python Source Code (Central Initialization Pattern)**")
    code_init = '''# In entrypoint (app.py) or shared state initializer:
def initialize_global_state():
    defaults = {
        "user_email": "engineer@ai-corp.com",
        "active_env": "Production",
        "selected_model": "Claude 3.7 Sonnet",
        "dark_mode": True,
        "shopping_cart": []
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# Call once at the top of your app:
initialize_global_state()'''
    st.code(code_init, language="python")

with col_ss2:
    st.markdown("**🖥️ Live Interactive State Persistence Inspector**")
    if "demo_global_user" not in st.session_state:
        st.session_state["demo_global_user"] = "sarah.chen@anthropic.ai"
    if "demo_global_model" not in st.session_state:
        st.session_state["demo_global_model"] = "Claude 3.7 Sonnet"

    with st.container(border=True):
        st.markdown("##### 🔍 Current Active Session State Snapshot")
        st.write(f"• **Global User**: `{st.session_state['demo_global_user']}`")
        st.write(f"• **Active Model**: `{st.session_state['demo_global_model']}`")
        st.caption("These variables remain intact when navigating between pages in this app.")

st.divider()

# ==============================================================================
# 2. Shared Filters Across Multiple Pages
# ==============================================================================
st.header("2. Shared Filters (Global Dashboard Controls)", divider="rainbow")
st.markdown(
    """
A common design pattern is a **Global Filter Panel** (e.g. Date Range, Department, or Model Family). 
When selected on one page, the choice must seamlessly filter all other views without forcing 
the user to re-select it.
"""
)

col_flt1, col_flt2 = st.columns(2)

with col_flt1:
    st.markdown("**💻 Python Source Code (Two-Way Synced Filter)**")
    code_filter = '''# Render filter bound directly to global state key:
MODELS = ["Claude 3.7", "GPT-4.5", "DeepSeek R1", "Llama 3.3"]

if "shared_model" not in st.session_state:
    st.session_state["shared_model"] = "Claude 3.7"

selected = st.selectbox(
    "Global Target Model",
    options=MODELS,
    key="shared_model"  # Direct binding ensures cross-page persistence!
)

# On any other page:
st.write(f"Filtered to: {st.session_state['shared_model']}")'''
    st.code(code_filter, language="python")

with col_flt2:
    st.markdown("**🖥️ Live Interactive Shared Filter Demo**")
    filter_models = ["Claude 3.7 Sonnet", "GPT-4.5 Omni", "DeepSeek R1", "Llama 3.3 70B"]
    if "shared_demo_model" not in st.session_state:
        st.session_state["shared_demo_model"] = "Claude 3.7 Sonnet"

    curr_choice = st.selectbox(
        "Select Active Global Model",
        filter_models,
        index=filter_models.index(st.session_state["shared_demo_model"]),
        key="sel_shared_demo_widget"
    )
    st.session_state["shared_demo_model"] = curr_choice

    with st.container(border=True):
        st.success(f"🎯 Global Model Filter set to: **{st.session_state['shared_demo_model']}**")
        st.caption("Any other page reading `st.session_state['shared_demo_model']` automatically consumes this value.")

st.divider()

# ==============================================================================
# 3. Shared Configuration & 4. Shared Functions
# ==============================================================================
st.header("3. Shared Configuration & 4. Shared Utility Functions", divider="rainbow")
st.markdown(
    """
Never redefine utility functions, formatting rules, or API clients inside individual page files. 
Create a shared **`config/`** and **`utils/`** package to establish a single source of truth.
"""
)

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    st.markdown("**💻 Python Directory Pattern (`utils/` & `config/`)**")
    code_cfg = '''# config/settings.py
APP_TITLE = "Nexus AI Enterprise"
SUPPORTED_MODELS = ["Claude 3.7", "GPT-4.5", "DeepSeek R1"]
DEFAULT_PAGE_SIZE = 25

# utils/formatting.py
def format_currency(val: float) -> str:
    return f"${val:,.2f}"

def format_tokens(n: int) -> str:
    return f"{n / 1e6:.2f}M tokens"

# Usage in ANY page:
from config.settings import SUPPORTED_MODELS
from utils.formatting import format_currency'''
    st.code(code_cfg, language="python")

with col_cfg2:
    st.markdown("**🖥️ Rendered Output (Simulated Utility Calls)**")
    with st.container(border=True):
        st.markdown("##### 🛠️ Utility Function Invocations")
        sample_revenue = 142850.75
        sample_tokens = 48500000
        st.write(f"• Raw Value: `{sample_revenue}` $\\to$ Formatted: **${sample_revenue:,.2f}**")
        st.write(f"• Raw Value: `{sample_tokens}` $\\to$ Formatted: **{sample_tokens / 1e6:.2f}M tokens**")

st.divider()

# ==============================================================================
# 5. Common Sidebar Components Across All Pages
# ==============================================================================
st.header("5. Common Sidebar Components (Entrypoint Sidebar Wrapping)", divider="rainbow")
st.markdown(
    r"""
In modern `st.navigation()` apps, any UI declared in the entrypoint script **outside** of `pg.run()` 
persists across every single page. This is the ideal place to render:
- Organization branding / Logo
- Current logged-in user badge
- Global environment switcher (Staging vs. Production)
"""
)

col_sb1, col_sb2 = st.columns(2)

with col_sb1:
    st.markdown("**💻 Python Source Code (Persistent Sidebar Shell)**")
    code_sidebar_shell = '''# app.py (Main Router Shell)
import streamlit as st

# Render persistent sidebar header across ALL pages:
with st.sidebar:
    st.image("assets/logo.png", width=120)
    st.markdown("### 🏢 Nexus AI Suite")
    st.selectbox("Environment", ["Production", "Staging", "Dev"], key="global_env")
    st.divider()

# st.navigation builds the page menu directly below the header!
pg = st.navigation(pages)
pg.run()

# Render persistent sidebar footer across ALL pages:
with st.sidebar:
    st.divider()
    if st.button("Log Out", icon="🚪"):
        st.session_state.clear()
        st.rerun()'''
    st.code(code_sidebar_shell, language="python")

with col_sb2:
    st.markdown("**🖥️ Live Interactive Sidebar Simulation**")
    with st.container(border=True):
        st.caption("Persistent Shell Simulation:")
        st.markdown("🏢 **Nexus AI Suite** *(Persistent Logo)*")
        env_sel = st.selectbox("Active Environment", ["Production (US-East)", "Staging (EU-West)", "Development (Local)"], key="sim_env_sel")
        st.info(f"Environment `{env_sel}` is globally active.")
        st.button("🚪 Log Out (Clears Session State)", key="sim_logout_btn")

st.divider()

# ==============================================================================
# 6. Shared Data Loading with `@st.cache_data` & `@st.cache_resource`
# ==============================================================================
st.header("6. Shared Data Loading (Memory-Efficient Singletons)", divider="rainbow")
st.markdown(
    r"""
When multiple pages query the same database or use the same ML embeddings model:
- **`@st.cache_data`**: Shares serializable query results (DataFrames, JSON) across all pages and users.
- **`@st.cache_resource`**: Shares non-serializable objects (database connection pools, PyTorch models, vector store clients).

```python
# utils/db.py
@st.cache_resource
def get_vector_store_client():
    # Only connects ONCE across the entire app lifespan!
    return PineconeClient(api_key=st.secrets["PINECONE_API_KEY"])

@st.cache_data(ttl=600)
def load_model_benchmarks():
    # Cached for 10 minutes across all pages!
    return pd.read_sql("SELECT * FROM model_evals", conn)
```
"""
)

# ==============================================================================
# 7. Page-Specific State vs. 8. Avoiding Naming Conflicts
# ==============================================================================
st.header("7. Page-Specific State & 8. Avoiding Naming Conflicts", divider="rainbow")
st.markdown(
    r"""
### The Silent Collision Bug
Because session state is global, if **Page 1** and **Page 2** both use `st.slider("Threshold", key="threshold")`, 
**they will overwrite each other's values!**

### Solution: The Namespaced State Pattern
Always prefix page-scoped keys with a unique page namespace:

| Scope | Key Naming Anti-Pattern | Recommended Namespaced Key |
| :--- | :--- | :--- |
| **Global User** | `st.session_state["user"]` | `st.session_state["global_user"]` |
| **Page 1 (Analytics)** | `st.session_state["date_range"]` | `st.session_state["page_analytics_date_range"]` |
| **Page 2 (Settings)** | `st.session_state["date_range"]` | `st.session_state["page_settings_date_range"]` |
| **Page 1 Widget** | `st.slider(..., key="slider")` | `st.slider(..., key="analytics_slider_threshold")` |
"""
)

col_ns1, col_ns2 = st.columns(2)

with col_ns1:
    st.markdown("**💻 Python Source Code (Namespace Helper Pattern)**")
    code_namespace = '''# utils/state.py
def get_page_state(page_name: str, key: str, default=None):
    """Retrieve a namespaced session state key safely."""
    namespaced_key = f"{page_name}_{key}"
    if namespaced_key not in st.session_state:
        st.session_state[namespaced_key] = default
    return st.session_state[namespaced_key]

# Usage in Page 1 (Analytics):
thresh = get_page_state("analytics", "confidence_cutoff", default=0.85)

# Usage in Page 2 (Playground):
thresh = get_page_state("playground", "confidence_cutoff", default=0.50)
# Zero conflict! Both co-exist in session state cleanly.'''
    st.code(code_namespace, language="python")

with col_ns2:
    st.markdown("**🖥️ Live Interactive Namespaced Session Registry**")
    # Demonstrate two independent thresholds residing in the same session state cleanly
    if "p1_thresh" not in st.session_state:
        st.session_state["p1_thresh"] = 0.85
    if "p2_thresh" not in st.session_state:
        st.session_state["p2_thresh"] = 0.50

    p1_val = st.slider("Page 1 (Analytics) Threshold", 0.0, 1.0, st.session_state["p1_thresh"], 0.05, key="widget_p1_thresh")
    p2_val = st.slider("Page 2 (Playground) Threshold", 0.0, 1.0, st.session_state["p2_thresh"], 0.05, key="widget_p2_thresh")
    st.session_state["p1_thresh"] = p1_val
    st.session_state["p2_thresh"] = p2_val

    with st.container(border=True):
        st.markdown("##### 🗄️ Raw `st.session_state` Namespaced Map:")
        st.json({
            "page_analytics_threshold": st.session_state["p1_thresh"],
            "page_playground_threshold": st.session_state["p2_thresh"],
            "collision_status": "NONE (Isolated)"
        })

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Multi-Page State Management Commandments", divider="rainbow")
st.markdown(
    r"""
| State Tier | Recommended Scope | Storage Mechanism | Best Practice |
| :--- | :--- | :--- | :--- |
| **User Identity & Auth** | Global | `st.session_state["global_auth"]` | Initialize in `app.py` before `pg.run()` |
| **Cross-Page Filters** | Global | `st.session_state["global_filter_*"]` | Bind widget keys directly to global key |
| **Heavy Singletons** | Global | `@st.cache_resource` | Database connections, vector store clients |
| **Cached Queries** | Global | `@st.cache_data` | DataFrames, API fetch results with TTL |
| **Page-Specific Sliders**| Local | `st.session_state["<page>_<key>"]` | Always namespace keys to avoid collision |
| **Cleanup on Logout** | Global | `st.session_state.clear()` | Follow with `st.rerun()` to reset views |
"""
)
