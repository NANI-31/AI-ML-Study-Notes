import streamlit as st
import pandas as pd

# st.set_page_config MUST be the very first Streamlit command invoked in the script
st.set_page_config(
    page_title="15.1 Page Configuration",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.streamlit.io",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "# Enterprise AI Telemetry Console\nBuilt with Streamlit 1.65.0."
    }
)

st.title("15.1 Page Configuration in Streamlit")
st.caption("Controlling application viewport, browser metadata, favicons, responsive layouts, sidebar behavior, and system menu branding.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW & FORMAL SIGNATURE
# -----------------------------------------------------------------------------
st.markdown(r"""
### The Role of `st.set_page_config()`

`st.set_page_config` configures browser metadata, viewport dimensions, and top-level shell chrome before any UI widgets render.

#### Formal API Signature:
```python
st.set_page_config(
    page_title: str | None = None,
    page_icon: str | Image | None = None,
    layout: "centered" | "wide" = "centered",
    initial_sidebar_state: "auto" | "expanded" | "collapsed" = "auto",
    menu_items: dict[str, str | None] | None = None
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `page_title` | `str \| None` | `None` (Filename) | The text displayed in the browser's tab title bar. |
| `page_icon` | `str \| Image \| None` | `None` (Streamlit logo) | Favicon in browser tab: emoji (`"🚀"`), image path, URL, or PIL Image. |
| `layout` | `"centered" \| "wide"` | `"centered"` | `"centered"` bounds content to ~736px; `"wide"` utilizes full screen width. |
| `initial_sidebar_state` | `"auto" \| "expanded" \| "collapsed"` | `"auto"` | Controls whether the sidebar starts open or closed. |
| `menu_items` | `dict[str, str] \| None` | `None` | Customizes top-right hamburger menu (`"Get Help"`, `"Report a bug"`, `"About"`). |

> [!IMPORTANT]
> **Cardinal Execution Rule**: `st.set_page_config()` **must be the very first Streamlit command** executed in your script. 
> Calling it after any `st.write()`, `st.sidebar`, or widget will raise `StreamlitSetPageConfigMustBeFirstCommandError`. 
> Furthermore, it may only be executed once per script run.
""")

tabs = st.tabs([
    "1. Page Title & Icons",
    "2. Layout: Wide vs Centered",
    "3. Initial Sidebar State",
    "4. Custom Menu Items",
    "5. Page Metadata & SEO",
    "6. Multi-Page Configuration",
    "7. Interactive Demo Sandbox"
])

# =============================================================================
# 1. PAGE TITLE & ICONS
# =============================================================================
with tabs[0]:
    st.header("1. Page Title & Favicon Configuration")
    st.markdown(r"""
    The `page_title` and `page_icon` establish the application's identity in the browser tab and bookmarks.
    The icon supports UTF-8 emojis (`"🤖"`), image filepaths, or remote URLs.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# 1. Emoji Favicon
st.set_page_config(
    page_title="LLM Analytics Dashboard",
    page_icon="🧠"
)

# 2. Custom Image / Brand Favicon
st.set_page_config(
    page_title="Enterprise ML Portal",
    page_icon="./assets/company_favicon.png"
)

# 3. Remote URL Favicon
st.set_page_config(
    page_title="Cloud Monitoring Console",
    page_icon="https://static.example.com/icon.png"
)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            **Browser Tab Representation:**
            
            ```
            ┌────────────────────────────────────────────────────────┐
            │  🧠 LLM Analytics Dashboard  ✕  │  +                   │
            ├─────────────────────────────────┴──────────────────────┤
            │  🔒 https://analytics.enterprise.ai/models             │
            └────────────────────────────────────────────────────────┘
            ```
            """)
            st.success("Configuring `page_icon` instantly personalizes browser tab tabs, bookmark entries, and mobile home-screen bookmarks.")

# =============================================================================
# 2. LAYOUT: WIDE VS CENTERED
# =============================================================================
with tabs[1]:
    st.header("2. Viewport Layout: Wide vs Centered")
    st.markdown(r"""
    Streamlit supports two core viewport bounding strategies:
    - `layout="centered"` (Default): Restricts content container to a fixed width (~736px). Best for articles, surveys, single-column forms, and mobile-first apps.
    - `layout="wide"`: Expands content across 100% of the viewport width. Essential for analytical dashboards, multi-column KPI grids, and wide DataFrames.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# Centered Layout (Ideal for sequential forms)
st.set_page_config(layout="centered")

# Wide Layout (Ideal for multi-column dashboards)
st.set_page_config(layout="wide")

# Example wide-screen multi-column arrangement
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active Models", "48")
kpi2.metric("P99 Latency", "142 ms")
kpi3.metric("Daily Tokens", "14.2 M")
kpi4.metric("Error Rate", "0.02%")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.caption("Demonstrating 4-Column Wide Grid Layout:")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Active Models", "48", "+3")
            k2.metric("P99 Latency", "142 ms", "-12 ms")
            k3.metric("Daily Tokens", "14.2 M", "+8.5%")
            k4.metric("Error Rate", "0.02%", "-0.01%")

            st.info("💡 `layout='wide'` provides room for dense analytical visualizations without horizontal clipping.")

# =============================================================================
# 3. INITIAL SIDEBAR STATE
# =============================================================================
with tabs[2]:
    st.header("3. Initial Sidebar State (`auto`, `expanded`, `collapsed`)")
    st.markdown(r"""
    Controls whether the sidebar is visible or minimized on initial load:
    - `"auto"` (Default): Intelligently collapses on mobile devices (<992px) and expands on desktop monitors.
    - `"expanded"`: Forces the sidebar to open immediately, prioritizing navigation menus and global filter controls.
    - `"collapsed"`: Hides the sidebar behind the top-left chevron/hamburger toggle, maximizing horizontal workspace.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Force sidebar collapsed for immersive full-screen dashboard
st.set_page_config(
    page_title="Executive Overview",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Sidebar contents remain accessible via chevron toggle
with st.sidebar:
    st.header("Administrative Filters")
    st.selectbox("Tenant", ["Production-US", "Staging-EU"])
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Sidebar Mode Matrix

            | Setting | Desktop ($\ge$ 992px) | Mobile (< 992px) | Recommended Use Case |
            | :--- | :--- | :--- | :--- |
            | `"auto"` | Expanded | Collapsed | Standard multi-device applications. |
            | `"expanded"` | Expanded | Expanded | Navigation-heavy apps requiring immediate user orientation. |
            | `"collapsed"` | Collapsed | Collapsed | Visual dashboards, presentation kiosks, and mobile-first portals. |
            """)

# =============================================================================
# 4. CUSTOM MENU ITEMS
# =============================================================================
with tabs[3]:
    st.header("4. Custom Top-Right Menu Configuration (`menu_items`)")
    st.markdown(r"""
    The top-right three-dots menu can be customized to replace or extend default options.
    Supported keys:
    - `"Get Help"`: External URL for technical documentation or support tickets.
    - `"Report a bug"`: URL for GitHub issues or bug tracking portals.
    - `"About"`: Custom Markdown text displayed in a modal dialog box when clicked.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

st.set_page_config(
    page_title="Production AI Platform",
    menu_items={
        "Get Help": "https://wiki.enterprise.ai/help",
        "Report a bug": "https://github.com/enterprise/ml-platform/issues",
        "About": """
        ### Enterprise AI Governance Platform v2.4
        
        - **Maintainer**: MLOps Core Engineering
        - **Compliance**: SOC-2 Type II Certified
        - **Support SLA**: 24/7 Severity-1 Response
        """
    }
)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            **Top-Right Hamburger Menu (`⋮`):**
            
            ```
            ┌──────────────────────────────────────────────┐
            │  ℹ️  About                                    │
            │  📖 Get Help          → wiki.enterprise.ai   │
            │  🐛 Report a bug      → github.com/...       │
            ├──────────────────────────────────────────────┤
            │  ⚙️  Settings                                 │
            │  🖨️  Print                                    │
            │  ⚡ Rerun                                     │
            └──────────────────────────────────────────────┘
            ```
            """)
            st.success("Configuring `menu_items` embeds white-labeled organizational compliance and help links into the native Streamlit shell.")

# =============================================================================
# 5. PAGE METADATA & SEO
# =============================================================================
with tabs[4]:
    st.header("5. Page Metadata & Head Elements")
    st.markdown(r"""
    `st.set_page_config()` injects HTML `<head>` tags into the rendered webpage.
    This impacts Search Engine Optimization (SEO), browser history tracking, and Web Application Manifest behavior.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# How Streamlit maps Python configs to HTML <head> elements:

# Python:
st.set_page_config(
    page_title="AI Model Leaderboard",
    page_icon="🏆"
)

# Resulting HTML DOM:
"""
<head>
  <title>AI Model Leaderboard</title>
  <link rel="icon" href="data:image/svg+xml,...">
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
"""
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Injected HTML Head Attributes

            | Python Property | Generated HTML Tag | Purpose |
            | :--- | :--- | :--- |
            | `page_title="X"` | `<title>X</title>` | Tab name, search engine SERP snippet, bookmark title. |
            | `page_icon="🚀"` | `<link rel="shortcut icon" href="...">` | Browser tab icon, mobile home-screen icon. |
            | `layout="wide"` | `<div class="stMainBlockContainer block-container" style="max-width: 100%;">` | Dynamic viewport container styling. |
            """)

# =============================================================================
# 6. MULTI-PAGE CONFIGURATION
# =============================================================================
with tabs[5]:
    st.header("6. Page Configuration in Multi-Page Applications (`st.Page`)")
    st.markdown(r"""
    In modern Multi-Page applications (Streamlit 1.35.0+ / 1.65.0), page titles and icons 
    are declared in the central router via `st.Page()` rather than calling `st.set_page_config()` in every child page.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# In the central router (app.py):
home_page = st.Page(
    "views/home.py", 
    title="Executive Overview", 
    icon="📊", 
    default=True
)
models_page = st.Page(
    "views/models.py", 
    title="Model Catalog", 
    icon="🤖"
)
settings_page = st.Page(
    "views/settings.py", 
    title="System Configuration", 
    icon="⚙️"
)

# Centralized page navigation
pg = st.navigation([home_page, models_page, settings_page])

# Centralized set_page_config on main entrypoint only!
st.set_page_config(page_title="MLOps Suite", layout="wide")
pg.run()
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Modern Multi-Page Best Practice

            1. **Single Call**: Call `st.set_page_config()` **only once** in the root entrypoint script (`app.py`).
            2. **Do Not Call in Page Modules**: Individual view files (e.g. `views/home.py`) should **not** invoke `st.set_page_config()`.
            3. **Declare in `st.Page`**: Provide `title` and `icon` directly to `st.Page(page, title="...", icon="...")` for dynamic title updates on page transitions.
            """)

# =============================================================================
# 7. INTERACTIVE DEMO SANDBOX
# =============================================================================
with tabs[6]:
    st.header("7. Interactive Configuration Code Generator")
    st.markdown(r"""
    Use this interactive tool to design your `st.set_page_config()` snippet and copy it directly into your application.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⚙️ Configuration Builder**")
        demo_title = st.text_input("Page Title", "Enterprise AI Studio", key="demo_cfg_title")
        demo_icon = st.selectbox("Page Icon (Emoji)", ["🤖", "🚀", "📈", "🛡️", "🔬", "⚡"], key="demo_cfg_icon")
        demo_layout = st.radio("Layout", ["wide", "centered"], horizontal=True, key="demo_cfg_layout")
        demo_sidebar = st.radio("Initial Sidebar State", ["expanded", "collapsed", "auto"], horizontal=True, key="demo_cfg_sb")

    with col2:
        st.markdown("**📋 Generated Python Code Block**")
        generated_code = f'''import streamlit as st

st.set_page_config(
    page_title="{demo_title}",
    page_icon="{demo_icon}",
    layout="{demo_layout}",
    initial_sidebar_state="{demo_sidebar}",
    menu_items={{
        "Get Help": "https://docs.streamlit.io",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "# {demo_title}\\nBuilt with Streamlit."
    }}
)
'''
        st.code(generated_code, language="python")
        st.info("Paste this at the very top of your entrypoint script (e.g., `app.py`).")

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Page Configuration Best Practices

| Concern | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Execution Order** | Calling `st.set_page_config()` after `st.write()` or `st.sidebar` | Place it as line 1 of executable Python code right after imports. |
| **Multi-Page Apps** | Calling `st.set_page_config()` inside multiple pages | Call once in the root router script; configure subpages via `st.Page(title=..., icon=...)`. |
| **Viewport Fitting** | Defaulting to `centered` for complex dashboards | Always specify `layout="wide"` for multi-column grids and large tables. |
| **Mobile UX** | Forcing `initial_sidebar_state="expanded"` on mobile | Use `"auto"` so mobile screens collapse the drawer cleanly. |
""")
