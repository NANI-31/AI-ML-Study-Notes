import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="15.2 Themes in Streamlit",
    page_icon="🎨",
    layout="wide"
)

st.title("15.2 Themes and Design Systems in Streamlit")
st.caption("Customizing application appearance via .streamlit/config.toml, brand palettes, primary/secondary color tokens, typography, and chart consistency.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The Streamlit Theming Engine

Streamlit provides a centralized configuration engine driven by `.streamlit/config.toml`. 
Themes define root design tokens that cascade into all native widgets, layout containers, and visualization adapters.

""")

st.graphviz_chart(
    r'''
    digraph ThemeFlow {
        graph [bgcolor="transparent", rankdir=LR];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        config [label=".streamlit/config.toml\n[theme]\nprimaryColor, backgroundColor\ntextColor, font", fillcolor="#DBEAFE", color="#1D4ED8"];
        engine [label="Streamlit Theming Engine\nGlobal CSS Variables", fillcolor="#FEF3C7", color="#D97706"];
        ui [label="Active Streamlit App\nst.button, st.slider, st.metric\nCharts & Typography", fillcolor="#D1FAE5", color="#059669"];

        config -> engine [label="Loaded on startup"];
        engine -> ui [label="Applies theme tokens"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Config.toml Specs",
    "2. Light Themes",
    "3. Dark Themes",
    "4. Color Tokens",
    "5. Font Configurations",
    "6. Chart Theme Consistency",
    "7. Interactive Theme Builder"
])

# =============================================================================
# 1. CONFIG.TOML SPECS
# =============================================================================
with tabs[0]:
    st.header("1. Streamlit Theme Configuration (`config.toml`)")
    st.markdown(r"""
    Theming is controlled globally via `.streamlit/config.toml` in your project root or user home directory.
    Streamlit reloads theme modifications automatically without requiring server restarts.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# .streamlit/config.toml
# Location: <project_root>/.streamlit/config.toml

[theme]
# Base archetype ("light" or "dark")
base = "dark"

# Brand accent color for buttons, active tabs, and sliders
primaryColor = "#6366F1"

# Main canvas background
backgroundColor = "#0B0F19"

# Sidebar, input widgets, and card backgrounds
secondaryBackgroundColor = "#1E2640"

# Main body text and header color
textColor = "#E2E8F0"

# Font family ("sans serif", "serif", or "monospace")
font = "sans serif"
''', language="toml")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Theme Token Properties Reference

            | Token | Key in `config.toml` | Description |
            | :--- | :--- | :--- |
            | **Base Mode** | `base` | `"light"` or `"dark"`. Derives contrast defaults. |
            | **Primary Accent** | `primaryColor` | Hex code for CTA buttons, active sliders, tab underlines. |
            | **Canvas Background** | `backgroundColor` | Main page canvas background color. |
            | **Container Background** | `secondaryBackgroundColor` | Sidebar, expander headers, inputs, metric cards. |
            | **Typography Color** | `textColor` | Body text, labels, headers, and icon fills. |
            | **Typography Family** | `font` | `"sans serif"`, `"serif"`, or `"monospace"`. |
            """)

# =============================================================================
# 2. LIGHT THEMES
# =============================================================================
with tabs[1]:
    st.header("2. Professional Light Themes")
    st.markdown(r"""
    Light themes are ideal for analytical reports, printing, and bright office environments.
    High-contrast text (`#1E293B`) paired with soft slate backgrounds (`#F8FAFC`) ensures WCAG AA compliance.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# Enterprise Light Theme Preset (.streamlit/config.toml)
[theme]
base = "light"
primaryColor = "#2563EB"              # Corporate Royal Blue
backgroundColor = "#FFFFFF"           # Pure White Canvas
secondaryBackgroundColor = "#F1F5F9"  # Cool Slate Gray
textColor = "#0F172A"                 # Deep Navy Ink
font = "sans serif"
''', language="toml")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.caption("Simulated Light Mode Palette Preview:")
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown("<div style='background-color:#FFFFFF; border:1px solid #CBD5E1; padding:15px; border-radius:6px; color:#0F172A; text-align:center;'><b>Canvas</b><br>#FFFFFF</div>", unsafe_allow_html=True)
            c2.markdown("<div style='background-color:#F1F5F9; border:1px solid #CBD5E1; padding:15px; border-radius:6px; color:#0F172A; text-align:center;'><b>Secondary</b><br>#F1F5F9</div>", unsafe_allow_html=True)
            c3.markdown("<div style='background-color:#2563EB; padding:15px; border-radius:6px; color:#FFFFFF; text-align:center;'><b>Primary</b><br>#2563EB</div>", unsafe_allow_html=True)
            c4.markdown("<div style='background-color:#0F172A; padding:15px; border-radius:6px; color:#FFFFFF; text-align:center;'><b>Text Ink</b><br>#0F172A</div>", unsafe_allow_html=True)

            st.write("")
            st.success("✅ Clean contrast with high readability for dense financial and spreadsheet tables.")

# =============================================================================
# 3. DARK THEMES
# =============================================================================
with tabs[2]:
    st.header("3. High-Performance Dark Themes")
    st.markdown(r"""
    Dark themes minimize eye fatigue in continuous-monitoring consoles and MLOps telemetry dashboards.
    Avoid pure pitch black (`#000000`) in favor of rich deep slates (`#0B0F17` or `#0F172A`) to avoid harsh contrast glare.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# Midnight Cyber Dark Theme (.streamlit/config.toml)
[theme]
base = "dark"
primaryColor = "#10B981"              # Neon Emerald Green
backgroundColor = "#0F172A"           # Deep Midnight Slate
secondaryBackgroundColor = "#1E293B"  # Muted Blue-Gray
textColor = "#F8FAFC"                 # Off-White High Contrast
font = "sans serif"
''', language="toml")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.caption("Simulated Dark Mode Palette Preview:")
            d1, d2, d3, d4 = st.columns(4)
            d1.markdown("<div style='background-color:#0F172A; border:1px solid #334155; padding:15px; border-radius:6px; color:#F8FAFC; text-align:center;'><b>Canvas</b><br>#0F172A</div>", unsafe_allow_html=True)
            d2.markdown("<div style='background-color:#1E293B; border:1px solid #334155; padding:15px; border-radius:6px; color:#F8FAFC; text-align:center;'><b>Secondary</b><br>#1E293B</div>", unsafe_allow_html=True)
            d3.markdown("<div style='background-color:#10B981; padding:15px; border-radius:6px; color:#0F172A; text-align:center;'><b>Primary</b><br>#10B981</div>", unsafe_allow_html=True)
            d4.markdown("<div style='background-color:#F8FAFC; padding:15px; border-radius:6px; color:#0F172A; text-align:center;'><b>Text</b><br>#F8FAFC</div>", unsafe_allow_html=True)

            st.write("")
            st.info("💡 High visual impact for real-time observability telemetry and AI chat consoles.")

# =============================================================================
# 4. COLOR TOKENS
# =============================================================================
with tabs[3]:
    st.header("4. Primary, Background, and Accent Color Tokens")
    st.markdown(r"""
    How primary and secondary colors affect individual Streamlit interactive widgets.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Demonstration of widget styling inheriting theme tokens:
st.button("Primary Action Button", type="primary")
st.button("Secondary Neutral Button", type="secondary")

# Sliders inherit primaryColor on active track:
st.slider("Model Temperature", 0.0, 1.0, 0.7)

# Checkboxes and radio buttons inherit primaryColor when active:
st.checkbox("Enable Streaming Output", value=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            bc1, bc2 = st.columns(2)
            with bc1:
                st.button("Primary Action Button", type="primary", key="btn_pri_demo")
            with bc2:
                st.button("Secondary Neutral Button", type="secondary", key="btn_sec_demo")

            st.slider("Model Temperature", 0.0, 1.0, 0.7, key="sl_temp_demo")
            st.checkbox("Enable Streaming Output", value=True, key="chk_stream_demo")

# =============================================================================
# 5. FONT CONFIGURATIONS
# =============================================================================
with tabs[4]:
    st.header("5. Typography and Font Family Configuration")
    st.markdown(r"""
    Streamlit supports three built-in font families configured via `font` in `config.toml`:
    - `"sans serif"`: Clean, modern interface typography (Inter / Helvetica Neue / Segoe UI).
    - `"serif"`: Editorial, formal publication typography (Georgia / Times New Roman).
    - `"monospace"`: Code-focused, developer-centric typography (JetBrains Mono / Source Code Pro / Courier).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# Monospace Typography Configuration (.streamlit/config.toml)
[theme]
base = "dark"
font = "monospace"

# Serif Typography Configuration (.streamlit/config.toml)
[theme]
base = "light"
font = "serif"
''', language="toml")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Typography Styles Comparison

            <div style="font-family: sans-serif; padding: 10px; border-left: 3px solid #6366F1; margin-bottom: 8px;">
                <b>Sans-Serif (Default):</b> The quick brown fox jumps over the lazy dog. Modern, readable UI standard.
            </div>
            
            <div style="font-family: serif; padding: 10px; border-left: 3px solid #10B981; margin-bottom: 8px;">
                <b>Serif:</b> The quick brown fox jumps over the lazy dog. Academic, journalistic, and literary tone.
            </div>

            <div style="font-family: monospace; padding: 10px; border-left: 3px solid #F59E0B; margin-bottom: 8px;">
                <b>Monospace:</b> The quick brown fox jumps over the lazy dog. Matrix telemetry, tensor shapes, and code.
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# 6. CHART THEME CONSISTENCY
# =============================================================================
with tabs[5]:
    st.header("6. Visual Consistency Across Native & Plotly Charts")
    st.markdown(r"""
    When plotting data via Plotly or Altair, always ensure the figure adapts to the active theme.
    By default, `st.plotly_chart(fig, theme="streamlit")` automatically extracts background and accent colors 
    from `config.toml` to style axes, tooltips, and grid lines.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np

# Streamlit native charts automatically match active theme
chart_data = pd.DataFrame(
    np.random.randn(20, 3), 
    columns=["P90 Latency", "P95 Latency", "P99 Latency"]
)

# Built-in charts inherit primaryColor and palette automatically
st.line_chart(chart_data, height=220)

# In Plotly, keep theme="streamlit" (default) for unified styling:
# st.plotly_chart(fig, theme="streamlit", width="stretch")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            np.random.seed(42)
            c_df = pd.DataFrame(
                np.cumsum(np.random.randn(20, 3), axis=0), 
                columns=["P90 Latency", "P95 Latency", "P99 Latency"]
            )
            st.caption("Native Chart Inheriting Theme Palette:")
            st.line_chart(c_df, height=200)
            st.success("Grid lines, background alpha, and series colors harmonize seamlessly with the active theme.")

# =============================================================================
# 7. INTERACTIVE THEME BUILDER
# =============================================================================
with tabs[6]:
    st.header("7. Interactive Theme Preset Generator")
    st.markdown(r"""
    Customize color tokens interactively to preview design harmony and copy the generated `.streamlit/config.toml` file.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎨 Palette Token Controls**")
        sel_base = st.selectbox("Base Archetype", ["dark", "light"], key="theme_bld_base")
        sel_font = st.selectbox("Font Family", ["sans serif", "serif", "monospace"], key="theme_bld_font")
        
        c_pri, c_bg = st.columns(2)
        with c_pri:
            pri_color = st.color_picker("Primary Accent", "#6366F1" if sel_base == "dark" else "#2563EB", key="t_cp_pri")
        with c_bg:
            bg_color = st.color_picker("Canvas Background", "#0F172A" if sel_base == "dark" else "#FFFFFF", key="t_cp_bg")
            
        c_sec, c_txt = st.columns(2)
        with c_sec:
            sec_color = st.color_picker("Secondary Background", "#1E293B" if sel_base == "dark" else "#F1F5F9", key="t_cp_sec")
        with c_txt:
            txt_color = st.color_picker("Text Color", "#F8FAFC" if sel_base == "dark" else "#0F172A", key="t_cp_txt")

    with col2:
        st.markdown("**📋 Generated `.streamlit/config.toml`**")
        config_toml_output = f'''# .streamlit/config.toml
[theme]
base = "{sel_base}"
primaryColor = "{pri_color}"
backgroundColor = "{bg_color}"
secondaryBackgroundColor = "{sec_color}"
textColor = "{txt_color}"
font = "{sel_font}"
'''
        st.code(config_toml_output, language="toml")
        
        st.markdown("**Simulated Card Preview:**")
        preview_html = f"""
        <div style="background-color:{bg_color}; padding:16px; border-radius:8px; border:1px solid {sec_color};">
            <h4 style="color:{txt_color}; margin:0 0 8px 0; font-family:{sel_font};">Theme Card Preview</h4>
            <p style="color:{txt_color}; opacity:0.85; margin:0 0 12px 0; font-family:{sel_font};">
                Active theme demonstration card with embedded primary CTA button.
            </p>
            <button style="background-color:{pri_color}; color:#FFFFFF; border:none; padding:8px 16px; border-radius:4px; font-weight:600; cursor:pointer;">
                Simulated Primary Action
            </button>
        </div>
        """
        st.markdown(preview_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Streamlit Theming Best Practices

| Concern | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Hardcoded Colors** | Hardcoding `#000000` text in custom HTML cards | Use CSS variables or inherit Streamlit tokens so cards adapt across light & dark modes. |
| **Plotly Styling** | Setting hardcoded black/white backgrounds in Plotly templates | Use `st.plotly_chart(fig, theme="streamlit")` for auto-theming. |
| **File Location** | Placing `config.toml` in random subdirectories | Keep it strictly in `<project_root>/.streamlit/config.toml`. |
| **Contrast Ratio** | Choosing faint accent colors with low contrast | Verify a minimum 4.5:1 contrast ratio against the secondary background (WCAG AA). |
""")
