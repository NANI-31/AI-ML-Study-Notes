import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="15.3 Custom CSS",
    page_icon="🎨",
    layout="wide"
)

st.title("15.3 Custom CSS & Styling in Streamlit")
st.caption("Mastering custom HTML/CSS injection via st.html(), glassmorphism cards, responsive flexbox layouts, selector stability, and accessibility.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### Modern CSS Injection Architecture in Streamlit

In Streamlit 1.35.0+, `st.html()` was introduced as the first-class method for injecting raw HTML and `<style>` blocks into the DOM.
While `st.markdown(..., unsafe_allow_html=True)` is widely used, `st.html()` avoids markdown parser overhead and escapes markdown sanitization boundaries.

""")

st.graphviz_chart(
    r'''
    digraph CSSInjection {
        graph [bgcolor="transparent", rankdir=LR];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        style [label="st.html('<style>...')\nCustom CSS Rules", fillcolor="#DBEAFE", color="#1D4ED8"];
        dom [label="Streamlit DOM\ndiv[data-testid='stMetricValue']", fillcolor="#FEF3C7", color="#D97706"];
        render [label="Styled Component\nEnterprise Brand Appearance", fillcolor="#D1FAE5", color="#059669"];

        style -> dom [label="Matches Selector"];
        dom -> render [label="Renders UI"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. CSS Basics & Box Model",
    "2. Modern st.html()",
    "3. Text & Typography",
    "4. Container Styling",
    "5. Custom Backgrounds",
    "6. Spacing & Flexbox",
    "7. Glassmorphism Cards",
    "8. Responsive Media Queries",
    "9. Selector Fragility",
    "10. Accessibility"
])

# =============================================================================
# 1. CSS BASICS & BOX MODEL
# =============================================================================
with tabs[0]:
    st.header("1. Understanding CSS Basics & The Box Model")
    st.markdown(r"""
    Every rendered element in the browser conforms to the **CSS Box Model**, consisting of:
    - **Content**: The text, icon, or image itself.
    - **Padding**: Transparent space between content and the border.
    - **Border**: A border wrapping the padding.
    - **Margin**: Transparent clearance outside the border separating adjacent elements.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

st.html("""
<style>
.box-demo {
    background-color: #1E293B;
    color: #F8FAFC;
    padding: 24px;
    border: 2px solid #6366F1;
    border-radius: 8px;
    margin: 12px 0;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    font-family: sans-serif;
}
</style>
<div class="box-demo">
    <b>Box Model Demonstration:</b><br>
    Content area padded by 24px with a 2px indigo border and 12px vertical margin.
</div>
""")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.html("""
            <style>
            .box-demo {
                background-color: #1E293B;
                color: #F8FAFC;
                padding: 20px;
                border: 2px solid #6366F1;
                border-radius: 8px;
                margin: 8px 0;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
                font-family: sans-serif;
            }
            </style>
            <div class="box-demo">
                <b>Box Model Demonstration:</b><br>
                Content area padded by 20px with a 2px indigo border and smooth box shadow.
            </div>
            """)

# =============================================================================
# 2. USING ST.HTML()
# =============================================================================
with tabs[1]:
    st.header("2. Using `st.html()` vs `st.markdown(unsafe_allow_html=True)`")
    st.markdown(r"""
    `st.html()` (Streamlit 1.35.0+) provides a clean interface for embedding custom HTML elements and global CSS rules.
    It does not wrap content in `<p>` tags or trigger markdown formatting glitches.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Preferred Modern Approach (Streamlit 1.35.0+):
st.html("""
<style>
.badge-pill {
    background-color: #10B981;
    color: #FFFFFF;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}
</style>
<span class="badge-pill">Production Operational</span>
""")

# Legacy Equivalent (Compatible with older versions):
# st.markdown('<span class="badge-pill">Operational</span>', unsafe_allow_html=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.html("""
            <style>
            .badge-pill {
                background-color: #10B981;
                color: #FFFFFF;
                padding: 4px 12px;
                border-radius: 9999px;
                font-size: 13px;
                font-weight: 600;
                display: inline-block;
            }
            </style>
            <span class="badge-pill">Production Operational</span>
            """)
            st.write("")
            st.caption("✅ Directly injects valid HTML tags without unexpected `<p>` tag wrapping.")

# =============================================================================
# 3. STYLING TEXT
# =============================================================================
with tabs[2]:
    st.header("3. Advanced Text Styling: Gradients & Glowing Accents")
    st.markdown(r"""
    Modern web typography utilizes linear background clipping to achieve gradient text effects.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

st.html("""
<style>
.gradient-headline {
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #6366F1 0%, #EC4899 50%, #F59E0B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
    margin-bottom: 8px;
}
.neon-glow {
    font-family: monospace;
    font-size: 16px;
    color: #38BDF8;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.6);
}
</style>
<div>
    <div class="gradient-headline">Generative AI Telemetry Gateway</div><br>
    <span class="neon-glow">> Latency: 42ms | Active Context: 128k</span>
</div>
""")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.html("""
            <style>
            .gradient-headline {
                font-size: 26px;
                font-weight: 800;
                background: linear-gradient(135deg, #6366F1 0%, #EC4899 50%, #F59E0B 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: inline-block;
                margin-bottom: 6px;
            }
            .neon-glow {
                font-family: monospace;
                font-size: 15px;
                color: #38BDF8;
                text-shadow: 0 0 8px rgba(56, 189, 248, 0.6);
            }
            </style>
            <div>
                <div class="gradient-headline">Generative AI Telemetry Gateway</div><br>
                <span class="neon-glow">> Latency: 42ms | Active Context: 128k</span>
            </div>
            """)

# =============================================================================
# 4. CONTAINER STYLING
# =============================================================================
with tabs[3]:
    st.header("4. Styling Custom Containers")
    st.markdown(r"""
    Containers group related controls and content. Custom CSS adds layered depth, borders, and transitions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

st.html("""
<style>
.enterprise-panel {
    background: #0F172A;
    border: 1px solid #334155;
    border-left: 4px solid #6366F1;
    border-radius: 8px;
    padding: 18px 24px;
    margin: 10px 0;
}
.panel-title {
    color: #F8FAFC;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 4px;
}
.panel-desc {
    color: #94A3B8;
    font-size: 14px;
}
</style>
<div class="enterprise-panel">
    <div class="panel-title">🛡️ SOC-2 Compliance Enforcer</div>
    <div class="panel-desc">All prompt evaluations and external API requests are encrypted at rest with AES-256.</div>
</div>
""")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.html("""
            <style>
            .enterprise-panel {
                background: #0F172A;
                border: 1px solid #334155;
                border-left: 4px solid #6366F1;
                border-radius: 8px;
                padding: 16px 20px;
                margin: 8px 0;
            }
            .panel-title {
                color: #F8FAFC;
                font-size: 16px;
                font-weight: 700;
                margin-bottom: 4px;
            }
            .panel-desc {
                color: #94A3B8;
                font-size: 14px;
            }
            </style>
            <div class="enterprise-panel">
                <div class="panel-title">🛡️ SOC-2 Compliance Enforcer</div>
                <div class="panel-desc">All prompt evaluations and external API requests are encrypted at rest with AES-256.</div>
            </div>
            """)

# =============================================================================
# 5. CUSTOM BACKGROUNDS
# =============================================================================
with tabs[4]:
    st.header("5. Gradient and Subtle Mesh Backgrounds")
    st.markdown(r"""
    Using radial and linear gradients for hero banners and section dividers.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

st.html("""
<style>
.mesh-banner {
    background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.25) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.2) 0%, transparent 40%),
                #0F172A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
}
.banner-head {
    color: #F8FAFC;
    font-size: 22px;
    font-weight: 700;
}
</style>
<div class="mesh-banner">
    <div class="banner-head">Next-Gen Large Language Model Orchestrator</div>
    <p style="color: #94A3B8; margin-top: 8px;">Unified inference routing across Anthropic, OpenAI, and local Ollama clusters.</p>
</div>
""")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.html("""
            <style>
            .mesh-banner {
                background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.25) 0%, transparent 40%),
                            radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.2) 0%, transparent 40%),
                            #0F172A;
                border: 1px solid #1E293B;
                border-radius: 12px;
                padding: 24px;
                text-align: center;
            }
            .banner-head {
                color: #F8FAFC;
                font-size: 20px;
                font-weight: 700;
            }
            </style>
            <div class="mesh-banner">
                <div class="banner-head">Next-Gen Large Language Model Orchestrator</div>
                <p style="color: #94A3B8; margin-top: 6px; font-size:14px;">Unified inference routing across Anthropic, OpenAI, and local Ollama clusters.</p>
            </div>
            """)

# =============================================================================
# 6. SPACING & FLEXBOX
# =============================================================================
with tabs[5]:
    st.header("6. Spacing, Alignment & CSS Flexbox")
    st.markdown(r"""
    CSS Flexbox (`display: flex`) provides pixel-perfect alignment for cards, badges, and horizontal KPI ribbons.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

st.html("""
<style>
.flex-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #1E293B;
    padding: 12px 20px;
    border-radius: 8px;
    border: 1px solid #334155;
}
.flex-item-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.status-dot {
    width: 10px;
    height: 10px;
    background: #10B981;
    border-radius: 50%;
}
</style>
<div class="flex-container">
    <div class="flex-item-left">
        <div class="status-dot"></div>
        <span style="color:#F8FAFC; font-weight:600;">Vector Database Cluster (Milvus-01)</span>
    </div>
    <span style="color:#94A3B8; font-size:13px;">Uptime: 99.99% • QPS: 1,420</span>
</div>
""")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.html("""
            <style>
            .flex-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #1E293B;
                padding: 12px 18px;
                border-radius: 8px;
                border: 1px solid #334155;
            }
            .flex-item-left {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .status-dot {
                width: 10px;
                height: 10px;
                background: #10B981;
                border-radius: 50%;
            }
            </style>
            <div class="flex-container">
                <div class="flex-item-left">
                    <div class="status-dot"></div>
                    <span style="color:#F8FAFC; font-weight:600; font-size:14px;">Vector Database Cluster (Milvus-01)</span>
                </div>
                <span style="color:#94A3B8; font-size:13px;">Uptime: 99.99% • QPS: 1,420</span>
            </div>
            """)

# =============================================================================
# 7. GLASSMORPHISM CARDS
# =============================================================================
with tabs[6]:
    st.header("7. Modern Glassmorphism Card Components")
    st.markdown(r"""
    Glassmorphism combines translucent backgrounds, backdrop blur (`backdrop-filter: blur(12px)`), 
    subtle white borders, and soft shadows to create a modern frosted glass aesthetic.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

st.html("""
<style>
.glass-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.glass-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.5);
}
.glass-stat {
    font-size: 26px;
    font-weight: 700;
    color: #F8FAFC;
}
.glass-label {
    font-size: 13px;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
<div class="glass-grid">
    <div class="glass-card">
        <div class="glass-label">Token Throughput</div>
        <div class="glass-stat">18.4k tok/s</div>
    </div>
    <div class="glass-card">
        <div class="glass-label">Inference Cost</div>
        <div class="glass-stat">$0.0012 / req</div>
    </div>
</div>
""")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.html("""
            <style>
            .glass-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
            }
            .glass-card {
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            }
            .glass-stat {
                font-size: 24px;
                font-weight: 700;
                color: #F8FAFC;
                margin-top: 4px;
            }
            .glass-label {
                font-size: 12px;
                color: #94A3B8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            </style>
            <div class="glass-grid">
                <div class="glass-card">
                    <div class="glass-label">Token Throughput</div>
                    <div class="glass-stat">18.4k tok/s</div>
                </div>
                <div class="glass-card">
                    <div class="glass-label">Inference Cost</div>
                    <div class="glass-stat">$0.0012 / req</div>
                </div>
            </div>
            """)

# =============================================================================
# 8. RESPONSIVE MEDIA QUERIES
# =============================================================================
with tabs[7]:
    st.header("8. Responsive Layouts with CSS Media Queries")
    st.markdown(r"""
    Media queries (`@media`) allow styles to adapt seamlessly across mobile screens (<768px), tablets, and desktop displays.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

st.html("""
<style>
.responsive-bar {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    background: #0F172A;
    padding: 16px;
    border-radius: 8px;
}

/* Stack vertically on mobile screens */
@media (max-width: 768px) {
    .responsive-bar {
        flex-direction: column;
        gap: 12px;
    }
}
</style>
<div class="responsive-bar">
    <div style="color:#F8FAFC;">Dashboard Header Component</div>
    <div style="color:#38BDF8;">Adaptive Alignment</div>
</div>
""")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.html("""
            <style>
            .responsive-bar {
                display: flex;
                flex-direction: row;
                justify-content: space-between;
                background: #0F172A;
                padding: 14px;
                border-radius: 8px;
                border: 1px solid #334155;
            }
            @media (max-width: 768px) {
                .responsive-bar {
                    flex-direction: column;
                    gap: 10px;
                }
            }
            </style>
            <div class="responsive-bar">
                <div style="color:#F8FAFC; font-weight:600;">Dashboard Header Component</div>
                <div style="color:#38BDF8; font-size:14px;">Adaptive Alignment (Row on Desktop, Column on Mobile)</div>
            </div>
            """)

# =============================================================================
# 9. SELECTOR FRAGILITY
# =============================================================================
with tabs[8]:
    st.header("9. CSS Selector Fragility & Stable Targeting Rules")
    st.markdown(r"""
    Streamlit dynamically generates internal emotion-cache class names (e.g., `.st-emotion-cache-1wmy0hl`). 
    **Targeting emotion hashes is strictly discouraged** because hashes change across Streamlit minor releases and break styles.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⚠️ Fragile vs Stable Selectors**")
        st.code(r'''
/* ❌ FRAGILE: Breaks on Streamlit library updates */
div.st-emotion-cache-1wmy0hl {
    background-color: #1E293B !important;
}

/* ✅ STABLE 1: Target Official Data Test IDs */
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    color: #6366F1 !important;
}

/* ✅ STABLE 2: Scope to Custom Container Class */
.custom-wrapper [data-testid="stButton"] button {
    border-radius: 9999px !important;
}
''', language="css")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Selector Reliability Hierarchy

            | Rank | Selector Strategy | Example | Reliability |
            | :-: | :--- | :--- | :--- |
            | **1** | Custom Classes inside `st.html()` | `.my-custom-card {...}` | 🟢 **100% Stable** (Independent of Streamlit internals). |
            | **2** | `data-testid` Attributes | `[data-testid="stMetric"]` | 🟡 **High** (Streamlit maintains testids for testing suites). |
            | **3** | Base HTML Element Types | `div[data-testid="stApp"] button` | 🟠 **Moderate** (Relies on button tag existence). |
            | **4** | Dynamic Emotion Hashes | `.st-emotion-cache-r421...` | 🔴 **Forbidden** (Breaks on minor version bumps). |
            """)

# =============================================================================
# 10. ACCESSIBILITY
# =============================================================================
with tabs[9]:
    st.header("10. Accessibility (a11y) Considerations")
    st.markdown(r"""
    When injecting custom styles, maintain **WCAG 2.1 AA** accessibility standards:
    - Minimum **4.5:1** contrast ratio for normal text and **3:1** for large text against backgrounds.
    - Support `prefers-reduced-motion` for users sensitive to animations.
    - Maintain keyboard focus outlines (`:focus-visible`).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
st.html("""
<style>
/* Respect user preference for reduced motion */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* Accessible focus ring for keyboard navigation */
.accessible-btn:focus-visible {
    outline: 3px solid #6366F1;
    outline-offset: 2px;
}
</style>
""")
''', language="css")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Accessibility Checklist

            - ✅ **Text Contrast**: Text `#F8FAFC` on `#0F172A` yields an exceptional **15.8:1** contrast ratio (Exceeds WCAG AAA).
            - ✅ **Focus Visibility**: Custom buttons must provide clear outline indicators for keyboard tab navigation.
            - ✅ **Reduced Motion**: Disabling hover transforms when the OS motion-reduction flag is enabled.
            - ✅ **Semantic Structure**: Use `<h1>` to `<h6>` tags hierarchy rather than plain styled `<div>` blocks.
            """)

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Custom CSS Best Practices

| Objective | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **HTML Injection** | `st.markdown(..., unsafe_allow_html=True)` | Use native `st.html(...)` (Streamlit 1.35.0+). |
| **DOM Targeting** | Targeting `.st-emotion-cache-*` classes | Target `[data-testid="..."]` or encapsulate in custom HTML wrappers. |
| **Theme Conflicts** | Hardcoding absolute colors on native widgets | Use relative CSS or Streamlit CSS variables (`var(--primary-color)`). |
| **Animations** | Unbounded heavy keyframe animations | Keep transitions subtle ($\le 250\text{ms}$) and honor `prefers-reduced-motion`. |
""")
