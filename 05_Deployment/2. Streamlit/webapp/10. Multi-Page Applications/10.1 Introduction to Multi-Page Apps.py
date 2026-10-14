import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Introduction to Multi-Page Apps",
    page_icon="📑",
    layout="wide",
)

st.title("📑 Introduction to Multi-Page Applications — Complete Guide")
st.markdown(
    """
As Streamlit applications evolve from single-script prototypes into enterprise SaaS products,
internal analytics suites, or GenAI platforms, placing all UI and logic into a single monolithic script 
causes severe maintainability, state conflict, and performance bottlenecks. 

Streamlit supports two distinct multi-page paradigms:
1. **The Classic File-System Pattern (`pages/` directory)**: Automatic convention-over-configuration routing.
2. **The Modern Programmatic Router (`st.Page` & `st.navigation`)**: Dynamic, role-based, custom URL routing.
"""
)

st.divider()

# ==============================================================================
# 1. Why Multi-Page Applications are Useful
# ==============================================================================
st.header("1. Why Multi-Page Applications are Essential", divider="rainbow")
st.markdown(
    r"""
Monolithic single-page Streamlit apps suffer from key structural issues as they scale:

| Challenge in Monolithic Apps | Multi-Page Solution |
| :--- | :--- |
| **Script Bloat** (thousands of lines in one file) | **Modular Separation**: Each page is a self-contained, isolated script. |
| **Unnecessary Code Execution on Rerun** | **Selective Execution**: Only the active page runs on widget interactions. |
| **Cluttered User Interface** | **Contextual Navigation**: Clear domain separation (e.g. Dashboard vs Settings vs Data Explorer). |
| **Shared State Collisions** | **Scoped Namespaces**: Clear separation of page-specific state from global state. |
| **Deep Linking & Bookmarking** | **Clean URL Routing**: Dedicated URLs (`/analytics`, `/settings`) for direct browser access. |
"""
)

col_why1, col_why2 = st.columns(2)

with col_why1:
    st.markdown("**💻 Architectural Comparison: Monolith vs. Multi-Page**")
    code_comp = '''# ❌ Monolithic Anti-Pattern (Single script with huge radio switcher)
page = st.sidebar.radio("Navigate", ["Home", "Analytics", "Settings"])

if page == "Home":
    render_home()  # All code, imports, and variables loaded simultaneously!
elif page == "Analytics":
    render_analytics()
elif page == "Settings":
    render_settings()

# ✅ Modern Multi-Page Pattern (Independent Page Objects)
# Each page file is loaded and executed only when navigated to!'''
    st.code(code_comp, language="python")

with col_why2:
    st.markdown("**🖥️ Live Interactive Mode Switcher Simulation**")
    with st.container(border=True):
        st.subheader("Simulated App Architecture Overhead")
        app_type = st.radio("Select Architecture", ["Monolithic (Single File)", "Multi-Page App (Modular)"], horizontal=True, key="rad_arch_type")
        if "Monolithic" in app_type:
            st.error("⚠️ **Monolithic**: Every interaction reruns the entire 2,500-line script, parsing all visualization imports.")
            st.metric("Memory Footprint", "~180 MB", "+120% vs baseline")
            st.metric("Initial Load Latency", "2.8s", "+1.9s penalty")
        else:
            st.success("✅ **Multi-Page**: Only the active page script executes. Imports and widgets are lazy-loaded on demand.")
            st.metric("Memory Footprint", "~45 MB", "-75% efficiency gain")
            st.metric("Initial Load Latency", "0.4s", "Instant rendering")

st.divider()

# ==============================================================================
# 2. Page-Based Application Architecture
# ==============================================================================
st.header("2. Page-Based Application Architecture", divider="rainbow")
st.markdown(
    """
In a multi-page Streamlit application, the **Root Entrypoint** acts as the application conductor, 
while individual pages serve as modular views:

""")

st.graphviz_chart(
    r'''
    digraph MultiPageNavigation {
        graph [bgcolor="transparent", rankdir=LR];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        entry [label="app.py (Master Router)\nst.navigation([...]).run()", fillcolor="#DBEAFE", color="#1D4ED8"];
        p1 [label="Dashboard Page\nst.Page('pages/dashboard.py')", fillcolor="#FEF3C7", color="#D97706"];
        p2 [label="Models Page\nst.Page('pages/models.py')", fillcolor="#FEF3C7", color="#D97706"];
        p3 [label="Settings Page\nst.Page('pages/settings.py')", fillcolor="#D1FAE5", color="#059669"];

        entry -> p1;
        entry -> p2;
        entry -> p3;
    }
    '''
)


col_arch1, col_arch2 = st.columns(2)

with col_arch1:
    st.markdown("**💻 Python Source Code (Shared Layout Architecture)**")
    code_arch = '''# main.py (Shared Entrypoint Shell)
import streamlit as st

st.set_page_config(page_title="AI Enterprise Hub", layout="wide")

# Global Application Banner (Rendered on all pages):
st.sidebar.title("🚀 Enterprise AI Suite")
st.sidebar.caption(f"Active User: admin@enterprise.ai")

# Individual page scripts run within this shell!'''
    st.code(code_arch, language="python")

with col_arch2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.container(border=True):
        st.caption("Shared Header / Top Ribbon Simulation:")
        top_c1, top_c2, top_c3 = st.columns([3, 1, 1])
        top_c1.markdown("### 🏢 Nexus AI Enterprise Hub")
        top_c2.markdown("🟢 **API Status**: Operational")
        top_c3.markdown("👤 **Role**: Administrator")
        st.info("💡 Top bars and global sidebar elements initialized in the entrypoint remain visible across all pages.")

st.divider()

# ==============================================================================
# 3. The `pages/` Directory Approach
# ==============================================================================
st.header("3. The Classic `pages/` Directory Convention", divider="rainbow")
st.markdown(
    r"""
Streamlit introduced native multi-page apps using the **convention-based `pages/` subfolder**:
- Create a directory named exactly **`pages/`** alongside your main entry script.
- Any `.py` file placed inside `pages/` is automatically discovered and listed in the sidebar navigation.
- No routing configuration or extra code is required!
"""
)

col_pg1, col_pg2 = st.columns(2)

with col_pg1:
    st.markdown("**💻 Python Directory Layout (`pages/` Pattern)**")
    code_pages_dir = '''# File System Layout:
app.py                          # Main landing page
pages/
├── 1_📈_Sales_Dashboard.py     # Discovered as "1 📈 Sales Dashboard"
├── 2_🔍_Data_Explorer.py       # Discovered as "2 🔍 Data Explorer"
└── 3_⚙️_Admin_Console.py       # Discovered as "3 ⚙️ Admin Console"

# Running the app:
# streamlit run app.py'''
    st.code(code_pages_dir, language="bash")

with col_pg1:
    st.markdown("**How Streamlit Formats File Names:**")
    st.markdown(
        r"""
- Numbers and underscores (`1_`) are used for ordering and stripped from the title display.
- Underscores (`_`) are converted to whitespace.
- Emojis in filenames (`📈`) are parsed and rendered as sidebar icons automatically!
"""
    )

with col_pg2:
    st.markdown("**🖥️ Live Interactive File-to-Title Parser Demo**")
    sample_file = st.text_input(
        "Enter Raw Filename to Test Parser:",
        value="01_📊_Model_Evaluation_Metrics.py",
        key="txt_filename_parser"
    )

    import re
    cleaned_stem = sample_file.replace(".py", "")
    # Remove leading number prefix (e.g. '01_')
    title_no_num = re.sub(r"^\d+_", "", cleaned_stem)
    # Extract emoji if present
    emoji_match = re.search(r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]", title_no_num)
    found_icon = emoji_match.group(0) if emoji_match else "📄"
    title_clean = re.sub(r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]", "", title_no_num).replace("_", " ").strip()

    with st.container(border=True):
        st.markdown(f"**Sidebar Display Title**: `{title_clean}`")
        st.markdown(f"**Sidebar Display Icon**: `{found_icon}`")
        st.markdown(f"**URL Path Route**: `/{title_clean.lower().replace(' ', '-')}`")

st.divider()

# ==============================================================================
# 4. Page Naming and Ordering Rules
# ==============================================================================
st.header("4. Page Naming, Sorting & URL Routing Rules", divider="rainbow")
st.markdown(
    r"""
Streamlit uses deterministic natural sort order to arrange pages in the sidebar.

| Filename on Disk | Display Title in Sidebar | URL Route |
| :--- | :--- | :--- |
| `app.py` | Home (or main filename) | `/` |
| `pages/01_Overview.py` | Overview | `/Overview` |
| `pages/02_📊_Telemetry.py` | Telemetry (with 📊 icon) | `/Telemetry` |
| `pages/10_Settings.py` | Settings (sorted after 02) | `/Settings` |

> **Pro Tip: Always use zero-padded numbers (`01_`, `02_`, `10_`)**! 
> Without zero-padding, standard alphabetical sort will place `10_Settings.py` *before* `2_Telemetry.py`!
"""
)

st.divider()

# ==============================================================================
# 5. Shared Application Components Across Pages
# ==============================================================================
st.header("5. Structuring Shared Components Across Pages", divider="rainbow")
st.markdown(
    """
In multi-page architectures, common UI elements (headers, footers, authentication banners, 
and database connections) must not be duplicated across every page file.
Create a **`components/`** or **`utils/`** package and import them cleanly.
"""
)

col_sh1, col_sh2 = st.columns(2)

with col_sh1:
    st.markdown("**💻 Python Source Code (`components/shared.py`)**")
    code_shared = '''# components/shared.py
import streamlit as st

def render_global_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()

def render_global_footer():
    st.divider()
    st.caption("🔒 Enterprise AI Platform • v2.4.0 • Built with Streamlit")

# In any page file (e.g. pages/01_Analytics.py):
from components.shared import render_global_header, render_global_footer

render_global_header("📊 Analytics Engine", "Real-time telemetry")
# ... page specific code ...
render_global_footer()'''
    st.code(code_shared, language="python")

with col_sh2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    with st.container(border=True):
        st.markdown("##### 🚀 Model Evaluation Page (Using Shared Component)")
        st.caption("Real-time telemetry across cluster workers")
        st.divider()
        st.success("Page specific interactive visualizations and metrics go here...")
        st.divider()
        st.caption("🔒 Enterprise AI Platform • v2.4.0 • Built with Streamlit")

st.divider()

# ==============================================================================
# 6. Navigation Between Pages (`st.page_link`)
# ==============================================================================
st.header("6. Programmatic In-App Navigation (`st.page_link`)", divider="rainbow")
st.markdown(
    r"""
Streamlit provides **`st.page_link()`** to navigate programmatically between pages from 
within page content, buttons, hero banners, or call-to-actions!

```python
st.page_link(
    page: str | Page,
    label: str | None = None,
    icon: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    use_container_width: bool = False
)
```

- **`page`**: Relative path to target page script (e.g. `"pages/02_Analytics.py"` or modern `st.Page` reference).
- **`label`**: Text displayed on the link button.
- **`icon`**: Emoji or Material icon string.
"""
)

col_nav1, col_nav2 = st.columns(2)

with col_nav1:
    st.markdown("**💻 Python Source Code (In-Page Navigation Grid)**")
    code_nav = '''# Create quick navigation action cards on the home page:
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Model Telemetry")
        st.write("Inspect live loss curves and GPU cluster VRAM.")
        st.page_link("pages/01_Telemetry.py", label="Open Telemetry", icon="📈")

with col2:
    with st.container(border=True):
        st.subheader("Platform Settings")
        st.write("Configure API keys, model endpoints, and quotas.")
        st.page_link("pages/02_Settings.py", label="Open Settings", icon="⚙️")'''
    st.code(code_nav, language="python")

with col_nav2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    nav_c1, nav_c2 = st.columns(2)
    with nav_c1:
        with st.container(border=True):
            st.subheader("Model Telemetry")
            st.caption("Inspect live loss curves and GPU cluster VRAM.")
            try:
                st.page_link("app.py", label="Explore Telemetry", icon="📈")
            except Exception:
                st.button("📈 Explore Telemetry (Simulated Link)", key="demo_nav_link_1")

    with nav_c2:
        with st.container(border=True):
            st.subheader("Platform Settings")
            st.caption("Configure API keys, model endpoints, and quotas.")
            try:
                st.page_link("app.py", label="Configure Platform", icon="⚙️")
            except Exception:
                st.button("⚙️ Configure Platform (Simulated Link)", key="demo_nav_link_2")

st.divider()

# ==============================================================================
# 7. Organizing Page-Specific Code (Clean Architecture Pattern)
# ==============================================================================
st.header("7. Best Practices for Organizing Page-Specific Code", divider="rainbow")
st.markdown(
    r"""
### The 3 Golden Rules of Multi-Page Production Apps

1. **Keep Page Files Lean**: 
   A page script should only coordinate UI presentation and user workflows. Place heavy data processing, database connections, and ML inference inside a dedicated `services/` or `src/` folder.
2. **Isolate Page-Specific Session State Keys**: 
   Prefix widget keys and state variables with the page name to prevent accidental cross-contamination:
   ```python
   # ✅ Page-scoped session state keys:
   st.session_state["telemetry_selected_model"] = "Claude 3.7"
   st.session_state["settings_api_key"] = "sk-..."
   ```
3. **Use Cached Resource Singletons**: 
   Database connections, vector store clients, and large ML model checkpoints should be loaded using **`@st.cache_resource`** so they remain shared across all pages without reconnecting on every page switch!
"""
)

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Multi-Page Architecture Checklist", divider="rainbow")
st.markdown(
    r"""
| Architecture Concept | Best Practice | Example |
| :--- | :--- | :--- |
| **Directory Name** | Name folder exactly `pages/` | `app.py` + `pages/` |
| **Numeric Sorting** | Zero-pad page prefixes | `01_Home.py`, `02_Analytics.py`, `10_Admin.py` |
| **Sidebar Display** | Use underscores for spaces | `01_Model_Evaluation.py` $\to$ "Model Evaluation" |
| **Page Linking** | Use `st.page_link()` | `st.page_link("pages/01_Home.py", label="Go Home")` |
| **Shared State** | Prefix session state keys by page | `st.session_state["analytics_filter_date"]` |
| **Heavy Logic** | Extract into service modules | Import from `services.data_loader` |
"""
)
