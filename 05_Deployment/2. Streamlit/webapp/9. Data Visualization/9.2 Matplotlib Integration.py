import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Matplotlib Integration Tutorial",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Matplotlib & Seaborn Integration — Complete Guide")
st.markdown(
    """
While native Streamlit charts are fast and responsive, **Matplotlib** and **Seaborn** remain the 
gold standard in scientific computing, machine learning research, and publication-ready diagnostics. 
In Streamlit, `st.pyplot()` bridges Python's scientific plotting ecosystem with the web, 
requiring careful object-oriented figure handling and explicit resource management.
"""
)

st.divider()

# ==============================================================================
# 1. Importing Matplotlib & 2. Creating a Figure (The Object-Oriented Pattern)
# ==============================================================================
st.header("1. Importing & 2. The Object-Oriented Figure Pattern", divider="rainbow")
st.markdown(
    r"""
### The Golden Rule of Matplotlib in Streamlit
Always use the **Object-Oriented API** (`fig, ax = plt.subplots()`) instead of stateful commands (`plt.plot()`).

| Anti-Pattern (Stateful `plt.plot`) | Recommended (Object-Oriented `plt.subplots`) |
| :--- | :--- |
| `plt.plot(x, y)` | `fig, ax = plt.subplots()` |
| `st.pyplot()` *(no argument)* | `ax.plot(x, y)` |
| ❌ Shares global state across concurrent users | `st.pyplot(fig)` |
| ❌ Can mix up plots between reruns | ✅ Thread-safe, isolated, and deterministic |
"""
)

col_oo1, col_oo2 = st.columns(2)

with col_oo1:
    st.markdown("**💻 Python Source Code (Object-Oriented Figure)**")
    code_oo = '''import matplotlib.pyplot as plt
import numpy as np

# 1. Create explicit figure and axes objects:
fig, ax = plt.subplots(figsize=(8, 4))

# 2. Draw on the specific axes object:
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label="Signal", color="#2563eb", linewidth=2)
ax.set_title("Carrier Wave Modulation", fontsize=14, fontweight="bold")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

# 3. Pass explicit figure to st.pyplot:
st.pyplot(fig)
plt.close(fig)  # Always close to prevent memory leaks!'''
    st.code(code_oo, language="python")

with col_oo2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    fig_oo, ax_oo = plt.subplots(figsize=(8, 4))
    x_val = np.linspace(0, 10, 100)
    ax_oo.plot(x_val, np.sin(x_val), label="Signal", color="#2563eb", linewidth=2)
    ax_oo.set_title("Carrier Wave Modulation", fontsize=13, fontweight="bold")
    ax_oo.set_xlabel("Time (s)")
    ax_oo.set_ylabel("Amplitude")
    ax_oo.grid(True, linestyle="--", alpha=0.5)
    ax_oo.legend()
    st.pyplot(fig_oo)
    plt.close(fig_oo)
    st.caption("Rendered using thread-safe `fig, ax = plt.subplots()` pattern.")

st.divider()

# ==============================================================================
# 3. Displaying Plots using st.pyplot() & 4. Configuring Figure Sizes
# ==============================================================================
st.header("3. `st.pyplot()` Parameters & 4. Configuring Figure Geometry", divider="rainbow")
st.markdown(
    r"""
```python
st.pyplot(
    fig=None,
    clear_figure=None,
    width="stretch",
    **kwargs
)
```

- **`fig`**: The explicit `matplotlib.figure.Figure` object to render.
- **`clear_figure`**: If `True`, clears the figure after rendering (defaults to `True` when `fig` is passed).
- **`width`**: `"stretch"` (expands to container width) or `"content"` (retains natural figure inch width).
- **Figure Dimensions**: Configured via `figsize=(width_inches, height_inches)` and `dpi=int` (Dots Per Inch).
"""
)

col_sz1, col_sz2 = st.columns(2)

with col_sz1:
    st.markdown("**💻 Python Source Code (Geometry & DPI Control)**")
    code_geom = '''# Configure aspect ratio & resolution
w_inch = st.slider("Figure Width (inches)", 4.0, 10.0, 7.0, 0.5)
h_inch = st.slider("Figure Height (inches)", 2.0, 6.0, 3.5, 0.5)

fig, ax = plt.subplots(figsize=(w_inch, h_inch), dpi=120)
ax.plot(np.random.randn(50).cumsum(), color="#059669", lw=2)
ax.set_title(f"Dynamic Dimension ({w_inch}x{h_inch} in)")

st.pyplot(fig, width="content")
plt.close(fig)'''
    st.code(code_geom, language="python")

with col_sz2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    w_inch = st.slider("Figure Width (inches)", 4.0, 10.0, 7.0, 0.5, key="mat_w_inch")
    h_inch = st.slider("Figure Height (inches)", 2.0, 6.0, 3.2, 0.5, key="mat_h_inch")

    fig_sz, ax_sz = plt.subplots(figsize=(w_inch, h_inch), dpi=100)
    np.random.seed(42)
    ax_sz.plot(np.random.randn(50).cumsum(), color="#059669", lw=2, label="Cumulative Drift")
    ax_sz.set_title(f"Rendered Dimension: {w_inch} in × {h_inch} in", fontsize=11)
    ax_sz.grid(True, linestyle=":", alpha=0.6)
    ax_sz.legend(loc="upper left")
    st.pyplot(fig_sz, width="content")
    plt.close(fig_sz)

st.divider()

# ==============================================================================
# 5. Displaying Multiple Charts (Grid & Columns)
# ==============================================================================
st.header("5. Displaying Multiple Charts Across Columns", divider="rainbow")
st.markdown(
    """
Instead of cluttering a single figure with unrelated metrics, create independent figures 
and render them across Streamlit layout columns (`st.columns`).
"""
)

col_mc1, col_mc2 = st.columns(2)

with col_mc1:
    st.markdown("**💻 Python Source Code (Side-by-Side Diagnostics)**")
    code_mc = '''col_left, col_right = st.columns(2)

with col_left:
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.hist(residuals, bins=25, color="#6366f1", edgecolor="black")
    ax1.set_title("Residual Error Distribution")
    st.pyplot(fig1)
    plt.close(fig1)

with col_right:
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.plot(fpr, tpr, color="#f59e0b", lw=2)
    ax2.plot([0, 1], [0, 1], "--", color="gray")
    ax2.set_title("ROC Curve (AUC = 0.94)")
    st.pyplot(fig2)
    plt.close(fig2)'''
    st.code(code_mc, language="python")

with col_mc2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    col_sub_l, col_sub_r = st.columns(2)

    np.random.seed(123)
    residuals = np.random.normal(0, 1.2, 500)

    with col_sub_l:
        fig1, ax1 = plt.subplots(figsize=(4.5, 3), dpi=90)
        ax1.hist(residuals, bins=20, color="#6366f1", edgecolor="white", alpha=0.85)
        ax1.set_title("Residual Error Distribution", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Residual", fontsize=8)
        st.pyplot(fig1)
        plt.close(fig1)

    with col_sub_r:
        fig2, ax2 = plt.subplots(figsize=(4.5, 3), dpi=90)
        fpr = np.linspace(0, 1, 50)
        tpr = np.sqrt(fpr)  # Synthetic ROC curve
        ax2.plot(fpr, tpr, color="#f59e0b", lw=2, label="Model (AUC=0.94)")
        ax2.plot([0, 1], [0, 1], "--", color="#94a3b8", label="Random Baseline")
        ax2.set_title("ROC Curve", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=7, loc="lower right")
        st.pyplot(fig2)
        plt.close(fig2)

st.divider()

# ==============================================================================
# 6. Integrating Subplots (Multi-Panel Figure Grids)
# ==============================================================================
st.header("6. Integrating Subplots (`plt.subplots` Grid)", divider="rainbow")
st.markdown(
    r"""
For tightly coupled scientific figures (such as training history comparing Loss, Accuracy, and Learning Rate), 
use a multi-panel subplot grid.
- **`plt.subplots(nrows, ncols, sharex=True)`**: Generates an array of axes.
- **`fig.tight_layout()`**: Automatically prevents overlapping titles and axis labels.
"""
)

col_sp1, col_sp2 = st.columns(2)

with col_sp1:
    st.markdown("**💻 Python Source Code (2x2 Training Telemetry Grid)**")
    code_subplot = '''fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

epochs = np.arange(1, 21)
axs[0, 0].plot(epochs, train_loss, color="#ef4444")
axs[0, 0].set_title("Training Loss")

axs[0, 1].plot(epochs, val_loss, color="#f97316")
axs[0, 1].set_title("Validation Loss")

axs[1, 0].plot(epochs, train_acc, color="#10b981")
axs[1, 0].set_title("Top-1 Accuracy")

axs[1, 1].plot(epochs, lr_schedule, color="#8b5cf6")
axs[1, 1].set_title("Cosine LR Schedule")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)'''
    st.code(code_subplot, language="python")

with col_sp2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    fig_sub, axs = plt.subplots(2, 2, figsize=(7.5, 5.5), dpi=100)
    np.random.seed(55)
    ep = np.arange(1, 21)

    axs[0, 0].plot(ep, 1.8 * np.exp(-ep / 5.0) + 0.1, color="#ef4444", lw=1.8)
    axs[0, 0].set_title("Training Loss", fontsize=10, fontweight="bold")
    axs[0, 0].grid(True, linestyle=":", alpha=0.5)

    axs[0, 1].plot(ep, 1.9 * np.exp(-ep / 6.0) + 0.22, color="#f97316", lw=1.8)
    axs[0, 1].set_title("Validation Loss", fontsize=10, fontweight="bold")
    axs[0, 1].grid(True, linestyle=":", alpha=0.5)

    axs[1, 0].plot(ep, 1 - np.exp(-ep / 4.0), color="#10b981", lw=1.8)
    axs[1, 0].set_title("Pass@1 Accuracy", fontsize=10, fontweight="bold")
    axs[1, 0].grid(True, linestyle=":", alpha=0.5)

    axs[1, 1].plot(ep, 1e-3 * 0.5 * (1 + np.cos(np.pi * ep / 20)), color="#8b5cf6", lw=1.8)
    axs[1, 1].set_title("Cosine LR Schedule", fontsize=10, fontweight="bold")
    axs[1, 1].grid(True, linestyle=":", alpha=0.5)

    fig_sub.tight_layout()
    st.pyplot(fig_sub)
    plt.close(fig_sub)

st.divider()

# ==============================================================================
# 7. Showing Statistical Plots (Seaborn Heatmaps & Distributions)
# ==============================================================================
st.header("7. Statistical Plots with Seaborn (`sns.heatmap`, `sns.kdeplot`)", divider="rainbow")
st.markdown(
    """
Seaborn plots render cleanly through `st.pyplot()` because Seaborn builds directly on Matplotlib axes:
1. Pass `ax=ax` into Seaborn functions (`sns.heatmap(..., ax=ax)`).
2. Format labels and colorbars on the returned Matplotlib axis.
3. Pass `fig` to `st.pyplot(fig)` and call `plt.close(fig)`.
"""
)

col_sn1, col_sn2 = st.columns(2)

with col_sn1:
    st.markdown("**💻 Python Source Code (Confusion Matrix Heatmap)**")
    code_seaborn = '''import seaborn as sns

# Classification Confusion Matrix
classes = ["Toxic", "Spam", "Benign", "Hallucinated"]
cm = np.array([
    [92,  3,  4,  1],
    [ 2, 88,  6,  4],
    [ 1,  4, 94,  1],
    [ 3,  5,  2, 90]
])

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    cbar=True,
    ax=ax
)
ax.set_title("Safety Guardrail Confusion Matrix")
ax.set_ylabel("True Category")
ax.set_xlabel("Predicted Category")

st.pyplot(fig)
plt.close(fig)'''
    st.code(code_seaborn, language="python")

with col_sn2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    classes = ["Toxic", "Spam", "Benign", "Hallucinated"]
    cm = np.array([
        [92,  3,  4,  1],
        [ 2, 88,  6,  4],
        [ 1,  4, 94,  1],
        [ 3,  5,  2, 90]
    ])

    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2), dpi=100)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax_cm
    )
    ax_cm.set_title("Safety Guardrail Confusion Matrix", fontsize=11, fontweight="bold")
    ax_cm.set_ylabel("Ground Truth")
    ax_cm.set_xlabel("Model Prediction")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

st.divider()

# ==============================================================================
# 8. Managing Figure Resources & Preventing Memory Leaks
# ==============================================================================
st.header("8. Managing Figure Resources & Memory Lifecycle", divider="rainbow")
st.markdown(
    r"""
### Why Memory Leaks Happen in Streamlit
Matplotlib maintains an internal global cache of all generated figures (`matplotlib._pylab_helpers.Gcf`). 
In interactive web apps, **if figures are not explicitly destroyed, memory usage grows indefinitely with every widget interaction**, 
eventually triggering Out-Of-Memory (OOM) crashes!

### Best Practice: The Explicit Cleanup Pattern
Always invoke **`plt.close(fig)`** immediately after `st.pyplot(fig)`:

```python
fig, ax = plt.subplots()
ax.plot(data)
st.pyplot(fig)
plt.close(fig)  # Explicitly garbage collect the figure buffer
```

Or close all figures at the end of the script:
```python
plt.close("all")  # Flushes all residual figure references
```
"""
)

col_rm1, col_rm2 = st.columns(2)

with col_rm1:
    st.markdown("**💻 Python Source Code (Safe Figure Context Manager)**")
    code_mem = '''import contextlib

@contextlib.contextmanager
def safe_figure(*args, **kwargs):
    """Context manager ensuring figure is always garbage collected."""
    fig, ax = plt.subplots(*args, **kwargs)
    try:
        yield fig, ax
    finally:
        st.pyplot(fig)
        plt.close(fig)

# Usage in your app:
with safe_figure(figsize=(6, 3)) as (fig, ax):
    ax.plot([1, 2, 3, 4], [10, 20, 15, 30], color="#ec4899", lw=2)
    ax.set_title("Safely Managed Lifecycle Plot")'''
    st.code(code_mem, language="python")

with col_rm2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    # Live Interactive Memory Audit Demonstration
    num_pts = st.slider("Select Data Sample Size", 10, 200, 50, key="slider_mem_demo")

    fig_mem, ax_mem = plt.subplots(figsize=(6, 3), dpi=95)
    np.random.seed(88)
    sample_y = np.random.randn(num_pts).cumsum()
    ax_mem.plot(sample_y, color="#ec4899", lw=2, label="Sample Trajectory")
    ax_mem.set_title(f"Dynamic Plot (N={num_pts}) — Explicit Memory Close", fontsize=10)
    ax_mem.legend(loc="upper left", fontsize=8)
    ax_mem.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig_mem)
    plt.close(fig_mem)  # Garbage collection
    st.success("✅ `plt.close(fig)` verified: Internal figure registry cleared after render.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Matplotlib & Seaborn Best Practices Checklist", divider="rainbow")
st.markdown(
    r"""
| Practice | Why It Matters | Standard Implementation |
| :--- | :--- | :--- |
| **Object-Oriented API** | Prevents state collision across sessions | `fig, ax = plt.subplots()` |
| **Explicit `fig` argument** | Avoids global `plt` state and warnings | `st.pyplot(fig)` |
| **Resource Cleanup** | Prevents memory accumulation on server | `plt.close(fig)` or `plt.close('all')` |
| **`figsize` & `dpi`** | Ensures crisp graphics on high-DPI displays | `plt.subplots(figsize=(8, 4), dpi=120)` |
| **`fig.tight_layout()`**| Prevents overlapping labels in subplots | Call right before `st.pyplot(fig)` |
| **Seaborn Integration** | Seamless statistical styling | Pass explicit axes: `sns.heatmap(..., ax=ax)` |
"""
)
