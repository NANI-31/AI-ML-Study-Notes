import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="14.1 NumPy Integration",
    page_icon="🔢",
    layout="wide"
)

st.title("14.1 NumPy Integration in Streamlit")
st.caption("High-performance numerical computing, tensor transformations, vector math, matrix linear algebra, and interactive scientific tools in Streamlit.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### NumPy in Modern Streamlit Architecture

NumPy (`ndarray`) forms the foundational computational layer for data science, machine learning, and numerical engineering in Python.
Integrating NumPy with Streamlit requires understanding how in-memory memory buffers interface with Streamlit's Apache Arrow serialization and browser presentation layers.

""")

st.graphviz_chart(
    r'''
    digraph NumPyInterop {
        graph [bgcolor="transparent", rankdir=LR];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        numpy [label="Vectorized NumPy Core\n• Matrices & Tensors (N, D)\n• Linear algebra & SIMD ops", fillcolor="#DBEAFE", color="#1D4ED8"];
        interop [label="Zero-Copy Serialization\nDirect array ingestion", fillcolor="#FEF3C7", color="#D97706"];
        ui [label="Streamlit Visualizers\n• st.line_chart()\n• st.image()\n• st.metric(np.mean(x))", fillcolor="#D1FAE5", color="#059669"];

        numpy -> interop -> ui;
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Displaying Arrays",
    "2. Numerical Inputs",
    "3. Calculations & SIMD",
    "4. Mathematical Functions",
    "5. Random Distributions",
    "6. Matrix Operations",
    "7. Statistical Analysis",
    "8. Scientific Signal Tool"
])

# =============================================================================
# 1. DISPLAYING NUMPY ARRAYS
# =============================================================================
with tabs[0]:
    st.header("1. Displaying NumPy Arrays")
    st.markdown(r"""
    NumPy arrays can be rendered via `st.write()` (raw inspect), converted to tabular format via `st.dataframe()` 
    (powered by Apache Arrow), or displayed as mathematical matrices using `st.latex()`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np
import pandas as pd

# 1D Vector (Embedding layer output)
embedding_1d = np.array([0.142, -0.891, 0.450, 1.205, -0.312])

# 2D Weight Matrix (Dense layer weights)
weights_2d = np.array([
    [1.0, 0.5, -0.2],
    [-0.5, 2.0, 0.8],
    [0.3, -0.1, 1.5]
])

st.caption("1. Tabular Representation via st.dataframe:")
df_weights = pd.DataFrame(weights_2d, columns=["Col 0", "Col 1", "Col 2"])
st.dataframe(df_weights.style.format("{:.2f}").background_gradient(cmap="Blues"), width="stretch")

st.caption("2. Mathematical LaTeX Matrix Notation:")
latex_matrix = r"\begin{bmatrix} " + r" \\ ".join([" & ".join(f"{x:.1f}" for x in row) for row in weights_2d]) + r" \end{bmatrix}"
st.latex(r"W = " + latex_matrix)

st.caption("3. Raw ndarray Metadata:")
st.write(f"Shape: `{weights_2d.shape}` | Dtype: `{weights_2d.dtype}` | Size: `{weights_2d.nbytes}` bytes")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            weights_2d = np.array([
                [1.0, 0.5, -0.2],
                [-0.5, 2.0, 0.8],
                [0.3, -0.1, 1.5]
            ])
            st.caption("1. Tabular Representation via st.dataframe:")
            df_weights = pd.DataFrame(weights_2d, columns=["Col 0", "Col 1", "Col 2"])
            st.dataframe(df_weights.style.format("{:.2f}").background_gradient(cmap="Blues"), width="stretch")

            st.caption("2. Mathematical LaTeX Matrix Notation:")
            latex_matrix = r"\begin{bmatrix} " + r" \\ ".join([" & ".join(f"{x:.1f}" for x in row) for row in weights_2d]) + r" \end{bmatrix}"
            st.latex(r"W = " + latex_matrix)

            st.caption("3. Raw ndarray Metadata:")
            st.write(f"Shape: `{weights_2d.shape}` | Dtype: `{weights_2d.dtype}` | Size: `{weights_2d.nbytes}` bytes")

# =============================================================================
# 2. ACCEPTING NUMERICAL INPUTS
# =============================================================================
with tabs[1]:
    st.header("2. Accepting Numerical Inputs")
    st.markdown(r"""
    Streamlit widgets allow capturing numerical scalars and vectors dynamically.
    The primary input mechanisms include `st.number_input`, `st.slider`, and vector input grids.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np

# Scalar inputs
learning_rate = st.number_input("Learning Rate (float)", min_value=1e-5, max_value=1.0, value=0.001, step=1e-4, format="%.4f")
batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128, 256], value=32)
regularization = st.slider("L2 Regularization (lambda)", min_value=0.0, max_value=0.1, value=0.01, step=0.005)

# Dynamic 1D Vector Input from Columns
st.markdown("**Vector Weight Input Grid:**")
v_cols = st.columns(3)
w0 = v_cols[0].number_input("Weight w0", value=1.0, step=0.1)
w1 = v_cols[1].number_input("Weight w1", value=-0.5, step=0.1)
w2 = v_cols[2].number_input("Weight w2", value=2.2, step=0.1)

weight_vec = np.array([w0, w1, w2])
st.info(f"Synthesized Weight Vector: `w = {weight_vec.tolist()}` | L2 Norm: `{np.linalg.norm(weight_vec):.3f}`")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            lr = st.number_input("Learning Rate (float)", min_value=1e-5, max_value=1.0, value=0.001, step=1e-4, format="%.4f", key="lr_input")
            b_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128, 256], value=32, key="bs_slider")
            reg = st.slider("L2 Regularization (lambda)", min_value=0.0, max_value=0.1, value=0.01, step=0.005, key="reg_slider")

            st.caption("Vector Weight Input Grid:")
            v_cols = st.columns(3)
            w0 = v_cols[0].number_input("Weight w0", value=1.0, step=0.1, key="w0_in")
            w1 = v_cols[1].number_input("Weight w1", value=-0.5, step=0.1, key="w1_in")
            w2 = v_cols[2].number_input("Weight w2", value=2.2, step=0.1, key="w2_in")

            weight_vec = np.array([w0, w1, w2])
            st.info(f"Synthesized Weight Vector: `w = {weight_vec.tolist()}` | L2 Norm: `{np.linalg.norm(weight_vec):.3f}`")

# =============================================================================
# 3. CALCULATIONS & SIMD
# =============================================================================
with tabs[2]:
    st.header("3. Performing Vectorized Calculations vs Python Loops")
    st.markdown(r"""
    NumPy operations execute inside pre-compiled C loops using SIMD (Single Instruction, Multiple Data) CPU instructions.
    Writing vectorized code avoids Python GIL overhead and yields **10x to 100x speedups**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np
import time

n_elements = st.select_slider("Vector Size", [10000, 100000, 500000, 1000000], value=100000)

# Generate test arrays
x = np.random.uniform(0.1, 10.0, n_elements)
y = np.random.uniform(0.1, 10.0, n_elements)

# 1. Pure Python Loop
t0 = time.perf_counter()
py_res = [xi ** 2 + 2 * yi for xi, yi in zip(x, y)]
py_time_ms = (time.perf_counter() - t0) * 1000

# 2. Vectorized NumPy SIMD
t1 = time.perf_counter()
np_res = x ** 2 + 2 * y
np_time_ms = (time.perf_counter() - t1) * 1000

speedup = py_time_ms / max(np_time_ms, 0.001)

m1, m2, m3 = st.columns(3)
m1.metric("Python Loop Time", f"{py_time_ms:.2f} ms")
m2.metric("NumPy SIMD Time", f"{np_time_ms:.2f} ms")
m3.metric("SIMD Acceleration", f"{speedup:.1f}x Faster", delta=f"{speedup:.1f}x")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            n_elements = st.select_slider("Vector Size", [10000, 50000, 100000], value=50000, key="n_elem_slider")
            x = np.random.uniform(0.1, 10.0, n_elements)
            y = np.random.uniform(0.1, 10.0, n_elements)

            t0 = time.perf_counter()
            _ = [xi ** 2 + 2 * yi for xi, yi in zip(x, y)]
            py_time_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            _ = x ** 2 + 2 * y
            np_time_ms = (time.perf_counter() - t1) * 1000

            speedup = py_time_ms / max(np_time_ms, 0.001)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Python Loop Time", f"{py_time_ms:.1f} ms")
            mc2.metric("NumPy SIMD Time", f"{np_time_ms:.1f} ms")
            mc3.metric("SIMD Speedup", f"{speedup:.1f}x Faster", delta=f"{speedup:.1f}x")

# =============================================================================
# 4. MATHEMATICAL FUNCTIONS
# =============================================================================
with tabs[3]:
    st.header("4. Visualizing Mathematical Functions")
    st.markdown(r"""
    Plotting activation functions (Sigmoid, ReLU, GELU, LeakyReLU) and harmonic waveforms 
    using vectorized NumPy evaluations and `st.line_chart()`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np
import pandas as pd

fn_choice = st.selectbox(
    "Activation Function", 
    ["Sigmoid", "ReLU", "GELU (Gaussian Error)", "Leaky ReLU", "Sine Wave"]
)

x_vals = np.linspace(-5.0, 5.0, 200)

if fn_choice == "Sigmoid":
    y_vals = 1.0 / (1.0 + np.exp(-x_vals))
elif fn_choice == "ReLU":
    y_vals = np.maximum(0, x_vals)
elif fn_choice == "GELU (Gaussian Error)":
    y_vals = 0.5 * x_vals * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_vals + 0.044715 * x_vals ** 3)))
elif fn_choice == "Leaky ReLU":
    alpha = st.slider("Alpha slope", 0.01, 0.3, 0.1)
    y_vals = np.where(x_vals > 0, x_vals, alpha * x_vals)
elif fn_choice == "Sine Wave":
    freq = st.slider("Frequency (Hz)", 0.5, 5.0, 1.0)
    y_vals = np.sin(freq * x_vals)

df_plot = pd.DataFrame({"x": x_vals, "f(x)": y_vals}).set_index("x")
st.line_chart(df_plot, height=260)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            fn_choice = st.selectbox(
                "Activation Function", 
                ["Sigmoid", "ReLU", "GELU (Gaussian Error)", "Leaky ReLU", "Sine Wave"],
                key="act_fn_select"
            )
            x_vals = np.linspace(-5.0, 5.0, 200)

            if fn_choice == "Sigmoid":
                y_vals = 1.0 / (1.0 + np.exp(-x_vals))
            elif fn_choice == "ReLU":
                y_vals = np.maximum(0, x_vals)
            elif fn_choice == "GELU (Gaussian Error)":
                y_vals = 0.5 * x_vals * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_vals + 0.044715 * x_vals ** 3)))
            elif fn_choice == "Leaky ReLU":
                alpha = st.slider("Alpha slope", 0.01, 0.3, 0.1, key="lrelu_alpha")
                y_vals = np.where(x_vals > 0, x_vals, alpha * x_vals)
            elif fn_choice == "Sine Wave":
                freq = st.slider("Frequency (Hz)", 0.5, 5.0, 1.0, key="sine_freq")
                y_vals = np.sin(freq * x_vals)

            df_act = pd.DataFrame({"x": x_vals, "f(x)": y_vals}).set_index("x")
            st.line_chart(df_act, height=220)

# =============================================================================
# 5. RANDOM DISTRIBUTIONS
# =============================================================================
with tabs[4]:
    st.header("5. Random Data Generation & Probability Distributions")
    st.markdown(r"""
    Modern NumPy standardizes on `np.random.default_rng(seed)` for reproducible, high-quality pseudo-random numbers.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np
import pandas as pd

seed = st.number_input("Random Seed", value=42, step=1)
dist_type = st.selectbox("Distribution", ["Gaussian (Normal)", "Uniform", "Exponential", "Poisson"])
sample_size = st.slider("Sample Count", 500, 10000, 2500, step=500)

rng = np.random.default_rng(int(seed))

if dist_type == "Gaussian (Normal)":
    data = rng.normal(loc=0.0, scale=1.0, size=sample_size)
elif dist_type == "Uniform":
    data = rng.uniform(low=-3.0, high=3.0, size=sample_size)
elif dist_type == "Exponential":
    data = rng.exponential(scale=1.5, size=sample_size)
elif dist_type == "Poisson":
    data = rng.poisson(lam=4.0, size=sample_size)

counts, bin_edges = np.histogram(data, bins=35)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
hist_df = pd.DataFrame({"Bin": bin_centers, "Count": counts}).set_index("Bin")

st.bar_chart(hist_df, height=240)
st.caption(f"Mean: `{np.mean(data):.3f}` | Std Dev: `{np.std(data):.3f}` | Skew: `{((data - np.mean(data))**3).mean() / (np.std(data)**3):.3f}`")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            r_seed = st.number_input("Random Seed", value=42, step=1, key="rand_seed")
            d_type = st.selectbox("Distribution", ["Gaussian (Normal)", "Uniform", "Exponential", "Poisson"], key="dist_select")
            s_count = st.slider("Sample Count", 500, 5000, 1500, step=500, key="sample_slider")

            rng = np.random.default_rng(int(r_seed))
            if d_type == "Gaussian (Normal)":
                raw_d = rng.normal(loc=0.0, scale=1.0, size=s_count)
            elif d_type == "Uniform":
                raw_d = rng.uniform(low=-3.0, high=3.0, size=s_count)
            elif d_type == "Exponential":
                raw_d = rng.exponential(scale=1.5, size=s_count)
            else:
                raw_d = rng.poisson(lam=4.0, size=s_count)

            cnts, b_edges = np.histogram(raw_d, bins=30)
            b_cents = 0.5 * (b_edges[1:] + b_edges[:-1])
            h_df = pd.DataFrame({"Bin": np.round(b_cents, 2), "Count": cnts}).set_index("Bin")

            st.bar_chart(h_df, height=200)
            st.caption(f"Mean: `{np.mean(raw_d):.3f}` | Std Dev: `{np.std(raw_d):.3f}` | Median: `{np.median(raw_d):.3f}`")

# =============================================================================
# 6. MATRIX OPERATIONS
# =============================================================================
with tabs[5]:
    st.header("6. Matrix Operations & Linear Algebra (`np.linalg`)")
    st.markdown(r"""
    Demonstrating matrix multiplication (`@`), inversion (`np.linalg.inv`), determinants (`np.linalg.det`), 
    and eigenvalue decomposition (`np.linalg.eig`).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np

# 2x2 Interactive Matrix Inputs
st.markdown("**Matrix A elements:**")
c1, c2 = st.columns(2)
a00 = c1.number_input("A[0,0]", value=4.0, step=0.5)
a01 = c2.number_input("A[0,1]", value=2.0, step=0.5)
a10 = c1.number_input("A[1,0]", value=1.0, step=0.5)
a11 = c2.number_input("A[1,1]", value=3.0, step=0.5)

A = np.array([[a00, a01], [a10, a11]])
det_A = np.linalg.det(A)

m1, m2 = st.columns(2)
m1.metric("Determinant |A|", f"{det_A:.3f}")
is_invertible = not np.isclose(det_A, 0.0)
m2.metric("Invertible", "Yes" if is_invertible else "Singular")

if is_invertible:
    inv_A = np.linalg.inv(A)
    st.markdown("**Inverse Matrix $A^{-1}$:**")
    st.latex(r"A^{-1} = \begin{bmatrix} " + f"{inv_A[0,0]:.3f} & {inv_A[0,1]:.3f} \\\\ {inv_A[1,0]:.3f} & {inv_A[1,1]:.3f}" + r" \end{bmatrix}")

eigenvalues, _ = np.linalg.eig(A)
st.write(f"Eigenvalues: `lambda_1 = {eigenvalues[0]:.3f}`, `lambda_2 = {eigenvalues[1]:.3f}`")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.caption("Matrix A Elements:")
            mc1, mc2 = st.columns(2)
            a00 = mc1.number_input("A[0,0]", value=4.0, step=0.5, key="a00_in")
            a01 = mc2.number_input("A[0,1]", value=2.0, step=0.5, key="a01_in")
            a10 = mc1.number_input("A[1,0]", value=1.0, step=0.5, key="a10_in")
            a11 = mc2.number_input("A[1,1]", value=3.0, step=0.5, key="a11_in")

            A = np.array([[a00, a01], [a10, a11]])
            det_A = np.linalg.det(A)

            lm1, lm2 = st.columns(2)
            lm1.metric("Determinant |A|", f"{det_A:.3f}")
            is_inv = not np.isclose(det_A, 0.0)
            lm2.metric("Invertible", "Yes" if is_inv else "Singular")

            if is_inv:
                inv_A = np.linalg.inv(A)
                st.latex(r"A^{-1} = \begin{bmatrix} " + f"{inv_A[0,0]:.3f} & {inv_A[0,1]:.3f} \\\\ {inv_A[1,0]:.3f} & {inv_A[1,1]:.3f}" + r" \end{bmatrix}")

            e_vals, _ = np.linalg.eig(A)
            st.write(f"Eigenvalues: `λ₁ = {e_vals[0]:.3f}`, `λ₂ = {e_vals[1]:.3f}`")

# =============================================================================
# 7. STATISTICAL ANALYSIS
# =============================================================================
with tabs[6]:
    st.header("7. Statistical Calculations & Correlation Matrices")
    st.markdown(r"""
    Computing multivariate statistics, covariance, and Pearson correlation coefficients 
    across synthetic feature datasets.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np
import pandas as pd

# Generate 3 correlated synthetic feature series
n_samples = 200
rng = np.random.default_rng(101)
f1 = rng.normal(50, 10, n_samples)
f2 = 0.8 * f1 + rng.normal(0, 5, n_samples)
f3 = -0.5 * f1 + rng.normal(0, 8, n_samples)

data_matrix = np.column_stack([f1, f2, f3])
corr_matrix = np.corrcoef(data_matrix, rowvar=False)

df_corr = pd.DataFrame(
    corr_matrix, 
    columns=["Feature_1", "Feature_2", "Feature_3"],
    index=["Feature_1", "Feature_2", "Feature_3"]
)

st.dataframe(
    df_corr.style.format("{:.3f}").background_gradient(cmap="coolwarm", vmin=-1.0, vmax=1.0),
    width="stretch"
)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            n_samples = 200
            rng_stat = np.random.default_rng(101)
            f1 = rng_stat.normal(50, 10, n_samples)
            f2 = 0.8 * f1 + rng_stat.normal(0, 5, n_samples)
            f3 = -0.5 * f1 + rng_stat.normal(0, 8, n_samples)

            data_matrix = np.column_stack([f1, f2, f3])
            corr_matrix = np.corrcoef(data_matrix, rowvar=False)

            df_corr = pd.DataFrame(
                corr_matrix, 
                columns=["Feature_1", "Feature_2", "Feature_3"],
                index=["Feature_1", "Feature_2", "Feature_3"]
            )
            st.caption("Pearson Correlation Matrix ($[-1.0, +1.0]$):")
            st.dataframe(
                df_corr.style.format("{:.3f}").background_gradient(cmap="coolwarm", vmin=-1.0, vmax=1.0),
                width="stretch"
            )

# =============================================================================
# 8. SCIENTIFIC SIGNAL TOOL
# =============================================================================
with tabs[7]:
    st.header("8. Interactive Scientific Signal Processing Simulator")
    st.markdown(r"""
    A real-time digital signal synthesizer demonstrating **harmonic Fourier summation**, 
    **Gaussian noise corruption**, and **moving-average denoising filter**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np
import pandas as pd

# Synthesizer Controls
f1_freq = st.slider("Harmonic 1 Freq (Hz)", 1, 10, 2)
f2_freq = st.slider("Harmonic 2 Freq (Hz)", 1, 20, 5)
noise_level = st.slider("Gaussian Noise (sigma)", 0.0, 1.5, 0.4, step=0.1)
filter_window = st.slider("Moving Average Kernel Window", 3, 25, 9, step=2)

t = np.linspace(0, 1.0, 300)
clean_signal = np.sin(2 * np.pi * f1_freq * t) + 0.5 * np.cos(2 * np.pi * f2_freq * t)
noisy_signal = clean_signal + np.random.normal(0, noise_level, len(t))

# 1D Convolution for moving average filter
kernel = np.ones(filter_window) / filter_window
filtered_signal = np.convolve(noisy_signal, kernel, mode="same")

df_signals = pd.DataFrame({
    "Time": t,
    "Noisy Raw Signal": noisy_signal,
    "Filtered Signal": filtered_signal,
    "Ground Truth": clean_signal
}).set_index("Time")

st.line_chart(df_signals[["Noisy Raw Signal", "Filtered Signal", "Ground Truth"]], height=280)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            f1_f = st.slider("Harmonic 1 Freq (Hz)", 1, 10, 2, key="sig_f1")
            f2_f = st.slider("Harmonic 2 Freq (Hz)", 1, 20, 5, key="sig_f2")
            n_lvl = st.slider("Gaussian Noise (σ)", 0.0, 1.5, 0.4, step=0.1, key="sig_noise")
            filt_w = st.slider("Moving Avg Window", 3, 25, 9, step=2, key="sig_window")

            t_vec = np.linspace(0, 1.0, 250)
            clean_sig = np.sin(2 * np.pi * f1_f * t_vec) + 0.5 * np.cos(2 * np.pi * f2_f * t_vec)
            np.random.seed(42)
            noisy_sig = clean_sig + np.random.normal(0, n_lvl, len(t_vec))

            kernel = np.ones(filt_w) / filt_w
            filt_sig = np.convolve(noisy_sig, kernel, mode="same")

            df_sig = pd.DataFrame({
                "Time": np.round(t_vec, 3),
                "Noisy Raw": noisy_sig,
                "Filtered": filt_sig,
                "Ground Truth": clean_sig
            }).set_index("Time")

            st.line_chart(df_sig[["Noisy Raw", "Filtered", "Ground Truth"]], height=220)
            st.caption("Demonstrating real-time convolution filtering on synthetic noisy harmonics.")

# -----------------------------------------------------------------------------
# NUMPY + STREAMLIT CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### NumPy & Streamlit Integration Best Practices

| Task | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Array Visualization** | Printing raw multi-line strings `st.text(str(arr))` | Wrap in `pd.DataFrame(arr)` for interactive Arrow rendering or `st.latex()` for math. |
| **Random Numbers** | Legacy `np.random.seed()` + `np.random.rand()` | Modern Generator API: `rng = np.random.default_rng(seed)`. |
| **Loop Computations** | Pure Python `for` loops across millions of items | Vectorized SIMD broadcasts (`arr * 2 + 5`) for 50x+ acceleration. |
| **Linear Algebra** | Manual nested matrix loops | Built-in `np.linalg.inv`, `np.linalg.det`, `np.linalg.eig`, and `@` operator. |
| **Memory Footprint** | Storing `float64` for all operations | Downcast to `float32` or `int32` when rendering large arrays in browser UI. |
""")
