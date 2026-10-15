import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(
    page_title="17.2 Scikit-learn Integration",
    page_icon="🔬",
    layout="wide"
)

st.title("17.2 Scikit-Learn Integration in Streamlit")
st.caption("Deploying Scikit-Learn workflows: joblib serialization, safe deserialization, classification, regression, clustering, feature scaling, and end-to-end Pipelines.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### Scikit-Learn Pipelines in Streamlit

In production, raw models should rarely be deployed in isolation. 
Instead, scikit-learn's `Pipeline` bundles feature scaling, imputation, categorical encoding, and the final estimator into a **single immutable artifact**.

""")

st.graphviz_chart(
    r'''
    digraph SklearnPipeline {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        input [label="1. Streamlit User Input Vector\nRaw Features: [Age=34, Salary=85k, Tier='Pro']", fillcolor="#DBEAFE", color="#1D4ED8"];
        pipeline [label="2. Cached Scikit-learn Pipeline (joblib)\n• ColumnTransformer (OneHot + Scaler)\n• Estimator (RandomForest / GradientBoosting)", fillcolor="#FEF3C7", color="#D97706"];
        output [label="3. Streamlit Reactive Presentation\n• st.metric() Class Probability\n• st.progress() Confidence Bar\n• Decision Badges", fillcolor="#D1FAE5", color="#059669"];

        input -> pipeline [label="Raw Features"];
        pipeline -> output [label="predict_proba()"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Loading with Joblib",
    "2. Joblib vs Pickle",
    "3. Safe Deserialization",
    "4. Classification Apps",
    "5. Regression Apps",
    "6. Clustering Apps",
    "7. Feature Scalers",
    "8. Pipeline Integration"
])

# =============================================================================
# 1. LOADING WITH JOBLIB
# =============================================================================
with tabs[0]:
    st.header("1. Loading a Saved Model with Joblib")
    st.markdown(r"""
    `joblib` is optimized for NumPy-heavy objects, efficiently serializing large array buffers.
    In Streamlit, load the model once into RAM using `@st.cache_resource`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import joblib

@st.cache_resource
def load_production_model(model_path: str):
    """Loads model pipeline once into memory, surviving reruns."""
    # joblib uses memory-mapping for zero-copy read of large arrays
    return joblib.load(model_path)

# In Streamlit session:
pipeline = load_production_model("models/credit_risk_v2.joblib")
st.success("Model loaded and cached in RAM.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Joblib Serialization Best Practices

            - **Use `@st.cache_resource`**: Never load models at script root without caching; disk I/O degrades response time.
            - **Keep artifacts immutable**: Model instances in `@st.cache_resource` are shared across all concurrent user threads. Never mutate model attributes in-place.
            - **Store version metadata**: Accompany serialized models with metadata JSON files documenting training timestamps, feature schemas, and test metrics.
            """)

# =============================================================================
# 2. JOBLIB VS PICKLE
# =============================================================================
with tabs[1]:
    st.header("2. Joblib vs Standard Pickle: Performance Comparison")
    st.markdown(r"""
    Why `joblib` is preferred over standard Python `pickle` for scikit-learn models:
    `joblib` compresses large internal NumPy arrays with compression levels (`compress=3`) and supports memory-mapping (`mmap_mode='r'`).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import joblib

# Saving with compression (Zstandard / Gzip / Lz4):
# compress=3 offers the best balance of speed and storage ratio
joblib.dump(fitted_estimator, "model_artifact.joblib", compress=3)

# Loading with memory-mapping for multi-gigabyte models:
large_model = joblib.load("model_artifact.joblib", mmap_mode="r")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Performance Benchmarks

            | Metric | Python `pickle` | `joblib` (compress=3) |
            | :--- | :--- | :--- |
            | **Large NumPy Serialization**| Slow (copies bytes) | Fast (optimized for arrays) |
            | **Disk Footprint** | Uncompressed (Large) | ~40-60% smaller file size |
            | **Memory-Mapped Loading**| Not supported natively | Supported (`mmap_mode='r'`) |
            | **Recommended Usage** | Simple Python objects | SciPy / NumPy / Scikit-Learn |
            """)

# =============================================================================
# 3. SAFE DESERIALIZATION
# =============================================================================
with tabs[2]:
    st.header("3. Using Serialized Models Safely")
    st.markdown(r"""
    **Critical Security Warning**: `pickle` and `joblib` are not secure against malicious data. 
    Loading an untrusted model file can execute arbitrary code on the host server.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Checksum Verification Pattern**")
        st.code(r'''
import hashlib
import streamlit as st

def verify_and_load_model(filepath: str, expected_sha256: str):
    """Verifies cryptographic hash before deserializing."""
    with open(filepath, "rb") as f:
        file_bytes = f.read()
        calculated_hash = hashlib.sha256(file_bytes).hexdigest()
        
    if calculated_hash != expected_sha256:
        raise ValueError("Security Alert: Model artifact hash mismatch! Potential tampering.")
        
    # Safe to load only after signature/checksum validation
    return joblib.load(filepath)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Model Security Safeguards

            1. **Never load user-uploaded pickle files**: Do not let arbitrary web users upload `.pkl` or `.joblib` files to run in `st.file_uploader`.
            2. **Checksum Verification**: Store SHA-256 hashes in version control and verify artifacts before calling `joblib.load()`.
            3. **Modern Alternatives**:
               - **ONNX (Open Neural Network Exchange)**: Safe, language-agnostic graph format.
               - **Skops**: Scikit-learn's secure serialization alternative that rejects untrusted code execution.
            """)

# =============================================================================
# 4. CLASSIFICATION APPS
# =============================================================================
with tabs[3]:
    st.header("4. Interactive Classification Applications")
    st.markdown(r"""
    Classification models predict discrete categories and calibrated class probabilities (`predict_proba`).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np

# Simulated Logistic Regression weights
weights = np.array([0.05, -0.02, 0.40])
bias = -1.2

with st.form("loan_approval_form"):
    st.subheader("Credit Risk Assessment")
    credit_score = st.slider("Credit Score", 300, 850, 710)
    debt_ratio = st.slider("Debt-to-Income Ratio (%)", 5, 65, 28)
    prior_defaults = st.selectbox("Prior Defaults", [0, 1, 2, 3])
    
    submitted = st.form_submit_button("Evaluate Credit Risk")

if submitted:
    x = np.array([credit_score, debt_ratio, prior_defaults])
    logit = np.dot(x, weights) + bias
    approval_prob = 1.0 / (1.0 + np.exp(-logit))
    
    if approval_prob >= 0.50:
        st.success(f"Approved (Confidence: {approval_prob:.1%})")
    else:
        st.error(f"Rejected (Risk Level: {1.0 - approval_prob:.1%})")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            weights = np.array([0.008, -0.04, -0.80])
            bias = -1.5

            with st.form("demo_loan_form"):
                st.caption("Credit Risk Assessment Form:")
                cs = st.slider("Credit Score", 300, 850, 710, key="clf_cs")
                dti = st.slider("Debt-to-Income (%)", 5, 65, 28, key="clf_dti")
                pdef = st.selectbox("Prior Defaults", [0, 1, 2], key="clf_pdef")
                clf_sub = st.form_submit_button("Evaluate Credit Risk", type="primary")

            if clf_sub:
                x_vec = np.array([cs, dti, pdef])
                logit = np.dot(x_vec, weights) + bias
                app_prob = float(1.0 / (1.0 + np.exp(-logit)))

                m1, m2 = st.columns(2)
                m1.metric("Approval Probability", f"{app_prob:.1%}")
                m2.metric("Decision", "Approved" if app_prob >= 0.5 else "Rejected")
                st.progress(app_prob)

# =============================================================================
# 5. REGRESSION APPS
# =============================================================================
with tabs[4]:
    st.header("5. Interactive Regression Applications")
    st.markdown(r"""
    Regression models predict continuous quantities (real estate prices, inventory demand, compute latency).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np

# Simulated Ridge Regression Model for Cloud GPU Costs
base_cost = 1.25  # $/hr
gpu_ram_coef = 0.08
bandwidth_coef = 0.015

st.subheader("Compute Instance Cost Predictor")
vram_gb = st.slider("GPU VRAM (GB)", 8, 128, 48, step=8)
network_gbps = st.slider("Network Bandwidth (Gbps)", 10, 400, 100, step=10)

predicted_cost = base_cost + (vram_gb * gpu_ram_coef) + (network_gbps * bandwidth_coef)
lower_bound = predicted_cost * 0.95
upper_bound = predicted_cost * 1.05

st.metric("Predicted Hourly Cost", f"${predicted_cost:.2f} / hr")
st.caption(f"95% Confidence Interval: [${lower_bound:.2f}, ${upper_bound:.2f}]")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            base_cost = 1.25
            gpu_ram_coef = 0.08
            bw_coef = 0.015

            st.caption("Compute Instance Cost Predictor:")
            vram = st.slider("GPU VRAM (GB)", 8, 128, 48, step=8, key="reg_vram")
            bw = st.slider("Network Bandwidth (Gbps)", 10, 400, 100, step=10, key="reg_bw")

            pred_c = base_cost + (vram * gpu_ram_coef) + (bw * bw_coef)
            lb = pred_c * 0.95
            ub = pred_c * 1.05

            st.metric("Estimated Hourly Cost", f"${pred_c:.2f} / hr", delta=f"+${pred_c * 0.08:.2f} vs baseline")
            st.caption(f"Estimated 95% Confidence Band: **[${lb:.2f}, ${ub:.2f}]**")

# =============================================================================
# 6. CLUSTERING APPS
# =============================================================================
with tabs[5]:
    st.header("6. Unsupervised Clustering Applications (K-Means)")
    st.markdown(r"""
    Unsupervised models group data points into distinct behavioral segments based on spatial proximity.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np

# 3 Pre-computed Cluster Centroids (Annual Spend vs Engagement Days)
centroids = np.array([
    [1200.0, 15.0],   # Cluster 0: Casual Low-Spend
    [6500.0, 85.0],   # Cluster 1: Engaged Mid-Tier
    [24000.0, 240.0]  # Cluster 2: Enterprise Champions
])

user_spend = st.number_input("Annual Spend ($)", value=7200.0, step=500.0)
user_days = st.number_input("Active Days / Year", value=95, step=5)

user_point = np.array([user_spend, user_days])
# Assign nearest centroid via Euclidean distance
distances = np.linalg.norm(centroids - user_point, axis=1)
assigned_cluster = int(np.argmin(distances))

labels = ["Casual Low-Spend", "Engaged Mid-Tier", "Enterprise Champions"]
st.success(f"Assigned Cluster: **{labels[assigned_cluster]}**")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            centroids = np.array([
                [1200.0, 15.0],
                [6500.0, 85.0],
                [24000.0, 240.0]
            ])
            u_spend = st.number_input("Annual Spend ($)", value=7200.0, step=500.0, key="km_spend")
            u_days = st.number_input("Active Days / Year", value=95, step=5, key="km_days")

            u_pt = np.array([u_spend, u_days])
            dists = np.linalg.norm(centroids - u_pt, axis=1)
            cluster_id = int(np.argmin(dists))

            seg_names = ["Casual Low-Spend", "Engaged Mid-Tier", "Enterprise Champion"]
            st.metric("Assigned Customer Segment", seg_names[cluster_id])
            st.caption(f"Euclidean distance to closest cluster centroid: `{dists[cluster_id]:.1f}`")

# =============================================================================
# 7. FEATURE SCALERS
# =============================================================================
with tabs[6]:
    st.header("7. Feature Scaling & Preventing Data Leakage")
    st.markdown(r"""
    Feature scalers must be **fit on training data only** and saved alongside the model.
    Never call `.fit_transform()` on runtime user inference data; always call `.transform()`.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# ❌ DANGEROUS LEAKAGE ANTI-PATTERN:
# Fitting scaler on new user input in Streamlit:
# scaler = StandardScaler()
# scaled = scaler.fit_transform(user_input)  # Output will always be 0.0!

# ✅ CORRECT PRODUCTION PATTERN:
# Load pre-fitted scaler from disk and call transform() only:
scaler = joblib.load("models/fitted_scaler.joblib")
scaled_features = scaler.transform(user_input)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Common Scaler Trade-offs

            | Scaler | Formula | Behavior on Outliers |
            | :--- | :--- | :--- |
            | **`StandardScaler`** | $z = \frac{x - \mu}{\sigma}$ | Sensitive to severe outliers; centers data at mean 0 with variance 1. |
            | **`MinMaxScaler`** | $z = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | Bounds features strictly to $[0, 1]$; compresses normal values if outliers exist. |
            | **`RobustScaler`** | $z = \frac{x - \text{median}}{\text{IQR}}$ | Robust to outliers; scales using median and interquartile range ($Q_3 - Q_1$). |
            """)

# =============================================================================
# 8. PIPELINE INTEGRATION
# =============================================================================
with tabs[7]:
    st.header("8. End-to-End Scikit-Learn Pipeline Integration")
    st.markdown(r"""
    The ultimate best practice is packaging the entire transformation and estimation graph 
    into a single `sklearn.pipeline.Pipeline` instance.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Complete Pipeline Pattern**")
        st.code(r'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Training Script (Offline):
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "models/champion_pipeline.joblib")

# Streamlit App (Online Inference):
@st.cache_resource
def get_pipeline():
    return joblib.load("models/champion_pipeline.joblib")

# Single call handles scaling + prediction automatically:
prediction = get_pipeline().predict(raw_user_dataframe)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Pipeline Advantages in Streamlit

            1. **Zero Discrepancy**: Preprocessing in the UI is identical to preprocessing during model training.
            2. **Single File Artifact**: One `.joblib` file contains both transformers and the classifier.
            3. **Clean Streamlit Code**: Eliminates boilerplate scaling code from the Streamlit UI script.
            4. **DataFrame Compatible**: Accepts raw Pandas DataFrames directly into `pipeline.predict(df)`.
            """)

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Scikit-Learn Integration Best Practices

| Concern | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Model Loading** | Calling `joblib.load()` on every script rerun | Wrap inside `@st.cache_resource` for singleton in-memory caching. |
| **Security** | Deserializing untrusted `.pkl` uploads | Use hash verification, SafeTensors, or Skops. |
| **Data Leakage** | Calling `scaler.fit_transform()` on inference data | Call `scaler.transform()` using the pre-fitted scaler from training. |
| **Pipeline Drift** | Manually scaling features in Streamlit | Export a single `Pipeline([("scaler", ...), ("model", ...)])` artifact. |
""")
