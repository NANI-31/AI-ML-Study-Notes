import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Navigation with st.navigation()",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Modern Navigation with `st.navigation()` & `st.Page()`")
st.markdown(
    """
Streamlit revolutionized multi-page architecture with **programmatic navigation** 
(`st.Page` and `st.navigation`). Instead of relying on rigid file-system conventions 
in the `pages/` directory, developers can now define pages dynamically in Python code, 
group them into hierarchical sections, assign custom URL paths and icons, 
and implement **role-based conditional routing** (e.g., hiding Admin pages from non-admin users).
"""
)

st.divider()

# ==============================================================================
# 1. Understanding the Modern Navigation API vs. pages/ Directory
# ==============================================================================
st.header("1. Modern Navigation API vs. Classic `pages/` Directory", divider="rainbow")
st.markdown(
    r"""
Streamlit's modern routing architecture replaces file-system magic with explicit Python objects:

| Feature | Classic `pages/` Directory | Modern `st.navigation([st.Page(...)])` |
| :--- | :--- | :--- |
| **Declaration** | File system convention (`pages/01_X.py`) | Programmatic in code (`st.Page("path.py")`) |
| **Hierarchical Grouping** | Flat list only (no nested sections) | Supported via dictionary sections |
| **Dynamic / Role-Based Pages** | ❌ Cannot conditionally hide pages | ✅ Filter list of pages based on user roles |
| **Custom URL Paths** | Tied directly to filename on disk | Custom slugs: `url_path="telemetry-v2"` |
| **In-Memory Functions as Pages** | ❌ Must be a physical file on disk | ✅ Directly pass Python callables: `st.Page(my_func)` |
| **Default Landing Page** | First sorted file | `default=True` flag on any `st.Page` |
"""
)

# ==============================================================================
# 2. st.Page() API & Parameters
# ==============================================================================
st.header("2. Anatomy of `st.Page()`", divider="rainbow")
st.markdown(
    r"""
The `st.Page` constructor defines metadata and execution targets for individual views:

```python
st.Page(
    page: str | Callable,
    *,
    title: str | None = None,
    icon: str | None = None,
    url_path: str | None = None,
    default: bool = False
)
```

- **`page`**: File path (`"views/analytics.py"`) OR a Python callable function (`def render_analytics(): ...`).
- **`title`**: Label displayed in the sidebar navigation.
- **`icon`**: Emoji (`"📈"`) or Material icon (`":material/analytics:"`).
- **`url_path`**: Custom browser URL slug (`https://app.com/custom-slug`).
- **`default`**: If `True`, marks this page as the initial landing page.
"""
)

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown("**💻 Python Source Code (Defining Pages)**")
    code_pages = '''# Approach A: File-path based page
p1 = st.Page(
    "views/telemetry.py",
    title="GPU Telemetry",
    icon="⚡",
    url_path="gpu-telemetry",
    default=True
)

# Approach B: Callable function-based page (No file on disk needed!)
def render_settings():
    st.header("User Settings")
    st.toggle("Dark Mode")

p2 = st.Page(
    render_settings,
    title="Preferences",
    icon=":material/settings:",
    url_path="preferences"
)'''
    st.code(code_pages, language="python")

with col_p2:
    st.markdown("**🖥️ Rendered Architecture Visualization**")
    with st.container(border=True):
        st.markdown("##### 📄 `st.Page` Object Representation")
        st.json({
            "title": "GPU Telemetry",
            "icon": "⚡",
            "url_path": "gpu-telemetry",
            "default": True,
            "target_type": "file_path ('views/telemetry.py')"
        })
        st.json({
            "title": "Preferences",
            "icon": ":material/settings:",
            "url_path": "preferences",
            "default": False,
            "target_type": "callable_function (render_settings)"
        })

st.divider()

# ==============================================================================
# 3. st.navigation() Architecture
# ==============================================================================
st.header("3. The Router: `st.navigation()` & `.run()` Pattern", divider="rainbow")
st.markdown(
    r"""
`st.navigation()` takes your defined `st.Page` objects, builds the sidebar UI, 
handles URL routing, and returns the selected page. You then call **`pg.run()`** to execute it!

```python
# In your main app entrypoint (app.py):
import streamlit as st

home_page = st.Page("views/home.py", title="Home", icon="🏠", default=True)
analytics_page = st.Page("views/analytics.py", title="Analytics", icon="📊")

# Initialize router:
pg = st.navigation([home_page, analytics_page])

# Execute selected page:
pg.run()
```
"""
)

# ==============================================================================
# 4. Page Titles, Icons & 5. Hierarchical Page Groups
# ==============================================================================
st.header("4. Titles, Icons & 5. Hierarchical Section Groups", divider="rainbow")
st.markdown(
    r"""
To organize a large enterprise app into collapsible sidebar sections, 
pass a **dictionary of lists** into `st.navigation()`:

```python
pg = st.navigation({
    "Core Workspaces": [home_page, data_explorer],
    "Model Observability": [metrics_page, drift_monitor],
    "Administration": [users_page, billing_page]
})
pg.run()
```
"""
)

col_grp1, col_grp2 = st.columns(2)

with col_grp1:
    st.markdown("**💻 Python Source Code (Section Groups Dictionary)**")
    code_groups = '''pages_dict = {
    "Intelligence": [
        st.Page("views/chat.py", title="AI Chatbot", icon="💬"),
        st.Page("views/rag.py", title="Knowledge Base", icon="📚")
    ],
    "Analytics": [
        st.Page("views/tokens.py", title="Token Usage", icon="🪙"),
        st.Page("views/latency.py", title="P99 Latency", icon="⏱️")
    ],
    "Settings": [
        st.Page("views/keys.py", title="API Keys", icon="🔑")
    ]
}

router = st.navigation(pages_dict)
router.run()'''
    st.code(code_groups, language="python")

with col_grp2:
    st.markdown("**🖥️ Live Interactive Navigation Tree Preview**")
    with st.container(border=True):
        st.caption("Sidebar Hierarchy Simulation:")
        with st.expander("📂 **Intelligence**", expanded=True):
            st.markdown("• 💬 AI Chatbot (`/chat`)")
            st.markdown("• 📚 Knowledge Base (`/rag`)")
        with st.expander("📂 **Analytics**", expanded=True):
            st.markdown("• 🪙 Token Usage (`/tokens`)")
            st.markdown("• ⏱️ P99 Latency (`/latency`)")
        with st.expander("📂 **Settings**", expanded=False):
            st.markdown("• 🔑 API Keys (`/keys`)")

st.divider()

# ==============================================================================
# 6. Conditional Navigation (Role-Based Access Control - RBAC)
# ==============================================================================
st.header("6. Conditional Navigation & Dynamic Role-Based Routing (RBAC)", divider="rainbow")
st.markdown(
    r"""
Because `st.navigation()` accepts dynamic Python lists, **you can conditionally include or exclude pages** 
based on authentication tokens, tenant subscription tiers, or user permissions!
"""
)

col_rbac1, col_rbac2 = st.columns(2)

with col_rbac1:
    st.markdown("**💻 Python Source Code (Dynamic RBAC Router)**")
    code_rbac = '''user_role = st.session_state.get("role", "viewer")

# Public pages visible to everyone:
available_pages = [
    st.Page("views/dashboard.py", title="Public Telemetry", icon="📊")
]

# Developer & Admin features:
if user_role in ["developer", "admin"]:
    available_pages.append(
        st.Page("views/playground.py", title="Model Playground", icon="🧪")
    )

# Super-admin exclusive:
if user_role == "admin":
    available_pages.append(
        st.Page("views/billing.py", title="Billing & Cloud Quotas", icon="💳")
    )

pg = st.navigation(available_pages)
pg.run()'''
    st.code(code_rbac, language="python")

with col_rbac2:
    st.markdown("**🖥️ Live Interactive RBAC Navigation Simulator**")
    simulated_role = st.selectbox(
        "Simulate Active User Role:",
        ["Viewer (Public)", "Developer (Engineering)", "Administrator (Super-User)"],
        key="sel_sim_role"
    )

    with st.container(border=True):
        st.markdown(f"##### Active Navigation Menu for `{simulated_role}`:")
        st.markdown("✅ 📊 **Public Telemetry** *(Visible to all)*")
        if "Developer" in simulated_role or "Administrator" in simulated_role:
            st.markdown("✅ 🧪 **Model Playground** *(Unlocked for Dev & Admin)*")
        else:
            st.markdown("🔒 ~~Model Playground~~ *(Hidden)*")

        if "Administrator" in simulated_role:
            st.markdown("✅ 💳 **Billing & Cloud Quotas** *(Super-Admin Only)*")
        else:
            st.markdown("🔒 ~~Billing & Cloud Quotas~~ *(Hidden)*")

st.divider()

# ==============================================================================
# 7. Entry-Point Scripts & Shared Shell Layout
# ==============================================================================
st.header("7. The Entry-Point Script as the Application Shell", divider="rainbow")
st.markdown(
    r"""
In modern `st.navigation()` architecture, your entry-point (`app.py`) controls 
everything that wraps the active page:
1. **Pre-Navigation Hooks**: User authentication checks, theme initialization.
2. **Global Layout Wrappers**: Persistent navigation sidebars, breadcrumbs, logos.
3. **`pg.run()`**: Executes the selected page in the main canvas.
4. **Post-Navigation Hooks**: Global footers, feedback dialogs.
"""
)

col_ep1, col_ep2 = st.columns(2)

with col_ep1:
    st.markdown("**💻 Python Source Code (Shell Architecture)**")
    code_shell = '''# app.py (Entrypoint Shell)
import streamlit as st

st.set_page_config(page_title="Enterprise AI", layout="wide")

# 1. Global Branding
st.logo("assets/logo.png", link="https://company.ai")
st.sidebar.caption("v3.2 Production")

# 2. Dynamic Routing
pg = st.navigation(build_nav_tree())

# 3. Render Active Page
pg.run()

# 4. Global Persistent Footer
st.divider()
st.caption("© 2026 Enterprise AI Inc. All rights reserved.")'''
    st.code(code_shell, language="python")

with col_ep2:
    st.markdown("**🖥️ Execution Lifecycle Diagram**")
    dag_nav = """
    digraph {
        rankdir=TB;
        node [shape=box, style="filled,rounded", fontname="Helvetica", fillcolor="#f8fafc", color="#94a3b8"];
        edge [color="#64748b", fontname="Helvetica", fontsize=9];

        Init [label="1. app.py starts\\n(set_page_config, logo)", fillcolor="#e0f2fe", color="#0284c7"];
        Auth [label="2. Check st.session_state\\n(Auth & User Role)", fillcolor="#fef3c7", color="#d97706"];
        Tree [label="3. Build Page List\\n[st.Page(...)]", fillcolor="#ecfdf5", color="#059669"];
        Nav [label="4. pg = st.navigation(pages)\\n(Renders Sidebar & URL Sync)", fillcolor="#f5f3ff", color="#7c3aed"];
        Run [label="5. pg.run()\\n(Executes Active Page Code)", fillcolor="#fae8ff", color="#c026d3"];
        Footer [label="6. Render Global Footer", fillcolor="#dcfce7", color="#16a34a"];

        Init -> Auth -> Tree -> Nav -> Run -> Footer;
    }
    """
    st.graphviz_chart(dag_nav, width="stretch")

st.divider()

# ==============================================================================
# 8. Page Routing & URL Query Synchronization
# ==============================================================================
st.header("8. Page Routing, Custom Slugs & URL Query Parameters", divider="rainbow")
st.markdown(
    r"""
Modern navigation automatically syncs with the browser's address bar:
- **Custom Slugs**: By specifying `url_path="custom-slug"`, the browser route becomes `https://app.com/custom-slug`.
- **Query Parameters**: Inspect and update parameters with **`st.query_params`** to enable shareable deep links!

```python
# Deep link handling:
current_model = st.query_params.get("model", "Claude-3.7")
st.selectbox("Model", ["Claude-3.7", "GPT-4.5"], index=0 if current_model=="Claude-3.7" else 1)
```
"""
)

col_url1, col_url2 = st.columns(2)

with col_url1:
    st.markdown("**💻 Python Source Code (URL Query Sync)**")
    code_query = '''# Read deep link parameter from URL:
model_param = st.query_params.get("model", "Claude-3.7")

selected_model = st.selectbox(
    "Active Model",
    ["Claude-3.7", "GPT-4.5", "DeepSeek-R1"],
    index=["Claude-3.7", "GPT-4.5", "DeepSeek-R1"].index(model_param)
)

# Keep URL in sync:
st.query_params["model"] = selected_model'''
    st.code(code_query, language="python")

with col_url2:
    st.markdown("**🖥️ Live Interactive Query Sync Demo**")
    with st.container(border=True):
        active_tab_demo = st.selectbox("Select Model for Deep Link:", ["Claude-3.7", "GPT-4.5", "DeepSeek-R1"], key="demo_deep_link_sel")
        simulated_url = f"https://enterprise-ai.internal/analytics?model={active_tab_demo.lower().replace('.', '')}"
        st.markdown(f"**Shareable Browser URL**: `{simulated_url}`")
        st.caption("Deep links allow users to bookmark or share exact dashboard views with team members.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Modern Navigation API Best Practices", divider="rainbow")
st.markdown(
    r"""
| Capability | Method | Key Parameters |
| :--- | :--- | :--- |
| **Page Definition** | `st.Page()` | `page` (file path or callable), `title`, `icon`, `url_path`, `default` |
| **Router Initialization**| `st.navigation()` | `pages` (list of `st.Page` or dict of sections) |
| **Page Execution** | `pg.run()` | Called on the returned router object in the entrypoint |
| **Programmatic Link** | `st.page_link()` | `page` (target page), `label`, `icon`, `disabled` |
| **In-Page Calling** | Callable Page | `st.Page(render_function)` — no disk file required |
| **RBAC / Auth Filtering**| Dynamic Python list | Filter pages list before passing to `st.navigation()` |
"""
)
