import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="17.1 ML Application Fundamentals",
    page_icon="🤖",
    layout="wide"
)

st.title("17.1 Machine Learning Application Fundamentals in Streamlit")
st.caption("Deploying trained models: offline training vs online inference, @st.cache_resource model singletons, input collection, feature pipelines, inference latency, and validation.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The ML Model Lifecycle in Streamlit

Streamlit serves as the presentation and interactive inference gateway for machine learning pipelines.
Rather than training models inside the UI, production applications deploy **pre-trained model artifacts** cached as singleton objects in RAM.

""")

st.graphviz_chart(
    r'''
    digraph MLDeployment {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        train [label="Offline Model Training\n• Feature engineering & tuning\n• Artifact serialization (joblib/ONNX)", fillcolor="#DBEAFE", color="#1D4ED8"];
        cache [label="1. Load Model\n@st.cache_resource (In-Memory)", fillcolor="#FEF3C7", color="#D97706"];
        input [label="2. Collect Features\nst.form() & interactive widgets", fillcolor="#FEF3C7", color="#D97706"];
        predict [label="3. Transform & Infer\npipeline.predict_proba(df)", fillcolor="#D1FAE5", color="#059669"];
        ui [label="4. Present Results\nst.metric(), Gauges & Confidence", fillcolor="#D1FAE5", color="#059669"];

        train -> cache [label="Serialized Model"];
        cache -> predict;
        input -> predict [label="Input Features"];
        predict -> ui [label="Predictions & Probabilities"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Intro to Deployment",
    "2. Training vs Inference",
    "3. Loading Models",
    "4. Collecting Inputs",
    "5. Feature Pipelines",
    "6. Generating Predictions",
    "7. Displaying Results",
    "8. Input Validation"
])

# =============================================================================
# 1. INTRO TO DEPLOYMENT
# =============================================================================
with tabs[0]:
    st.header("1. Introduction to Machine Learning Deployment")
    st.markdown(r"""
    Model deployment transitions a static model artifact into an operational service.
    Streamlit enables rapid model exploration, internal stakeholder testing, and interactive what-if scenario simulations.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Architectural Overview**")
        st.markdown(r"""
        ### Deployment Paradigms

        | Paradigm | Latency Target | Serving Mechanism | Use Case in ML |
        | :--- | :--- | :--- | :--- |
        | **Interactive UI** | $50\text{ms} - 500\text{ms}$ | Streamlit (`@st.cache_resource`) | Executive dashboards, clinical diagnostic tools, loan underwriting. |
        | **Microservice REST**| $10\text{ms} - 100\text{ms}$ | FastAPI / Triton Inference Server | High-throughput mobile apps, search autocompletion. |
        | **Batch Offline** | Minutes - Hours | Apache Spark / Ray Train | Overnight customer churn scoring, weekly fraud reports. |
        """)

    with col2:
        st.markdown("**🖥️ Model Status Indicator**")
        with st.container(border=True):
            st.markdown("### 🚀 Production Model Endpoint")
            c1, c2 = st.columns(2)
            c1.metric("Active Architecture", "GradientBoostingClassifier")
            c2.metric("Target Metric", "AUC-ROC: 0.924")
            
            st.success("🟢 Inference engine initialized in memory and ready for queries.")

# =============================================================================
# 2. TRAINING VS INFERENCE
# =============================================================================
with tabs[1]:
    st.header("2. Training versus Inference Architecture")
    st.markdown(r"""
    A critical anti-pattern in Streamlit is retraining models on every script rerun.
    Training is computationally heavy ($O(N \cdot M)$) and asynchronous; 
    inference is lightweight ($O(1)$) and synchronous.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
# ❌ ANTI-PATTERN: Retraining on every user rerun
# model = RandomForestClassifier().fit(X_train, y_train)  # Sinks CPU on every slider!

# ✅ PRODUCTION PATTERN: Separate Training from Inference
# Step 1: Train offline and save artifact
# joblib.dump(fitted_pipeline, "model_pipeline.joblib")

# Step 2: In Streamlit, load once into RAM as a singleton
import streamlit as st
import joblib

@st.cache_resource
def get_model_pipeline():
    # Only runs once per server lifecycle
    return joblib.load("model_pipeline.joblib")

pipeline = get_model_pipeline()
# Fast sub-second inference:
prediction = pipeline.predict(user_features)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            st.markdown(r"""
            ### Training vs Inference Comparison

            | Attribute | Training Phase | Online Inference Phase |
            | :--- | :--- | :--- |
            | **Execution** | Offline (GPU / Cloud batch) | Online inside Streamlit |
            | **Computation** | Gradient updates, loss optimization | Matrix forward pass ($\mathbf{x} \mathbf{W}^T + \mathbf{b}$) |
            | **Frequency** | Weekly / Monthly / Scheduled | On-demand on user input ($<100\text{ms}$) |
            | **Memory** | Backprop gradients, entire dataset | Frozen weights in `@st.cache_resource` |
            """)

# =============================================================================
# 3. LOADING TRAINED MODELS
# =============================================================================
with tabs[2]:
    st.header("3. Loading Trained Models with `@st.cache_resource`")
    st.markdown(r"""
    `@st.cache_resource` caches the model instance by memory reference across all reruns and user sessions.
    This prevents reloading weights on every interaction, cutting latency from seconds to milliseconds.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np

class MockChurnModel:
    """Simulated logistic regression model."""
    def __init__(self):
        self.weights = np.array([0.45, -0.65, 0.85])
        self.bias = -0.20
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.weights) + self.bias
        prob = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - prob, prob])

@st.cache_resource
def load_churn_model():
    # Simulated model load with startup delay
    return MockChurnModel()

model = load_churn_model()
st.info(f"Model singleton loaded in memory at ID: `{hex(id(model))}`")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            class MockChurnModel:
                def __init__(self):
                    self.weights = np.array([0.45, -0.65, 0.85])
                    self.bias = -0.20
                def predict_proba(self, X: np.ndarray) -> np.ndarray:
                    logits = np.dot(X, self.weights) + self.bias
                    prob = 1.0 / (1.0 + np.exp(-logits))
                    return np.column_stack([1.0 - prob, prob])

            @st.cache_resource
            def load_churn_model():
                return MockChurnModel()

            model = load_churn_model()
            st.info(f"Model singleton loaded in memory at ID: `{hex(id(model))}`")
            st.caption("Subsequent reruns reuse this exact in-memory instance with zero disk I/O overhead.")

# =============================================================================
# 4. COLLECTING INPUTS
# =============================================================================
with tabs[3]:
    st.header("4. Collecting Prediction Inputs via `st.form`")
    st.markdown(r"""
    Prediction inputs must be debounced using `st.form`. 
    Without a form, adjusting three sliders triggers three consecutive full script reruns.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

with st.form("customer_input_form"):
    st.subheader("Customer Feature Attributes")
    
    col_a, col_b = st.columns(2)
    with col_a:
        tenure_months = st.slider("Account Tenure (Months)", 1, 72, 24)
        contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    with col_b:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=200.0, value=75.0)
        tech_support = st.checkbox("Has Dedicated Tech Support", value=True)
        
    submitted = st.form_submit_button("Run Churn Prediction", type="primary")

if submitted:
    st.success(f"Collected features: Tenure={tenure_months}mo, Charges=${monthly_charges}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            with st.form("demo_customer_input_form"):
                st.caption("Customer Feature Attributes Form:")
                col_a, col_b = st.columns(2)
                with col_a:
                    tenure_val = st.slider("Account Tenure (Months)", 1, 72, 24, key="in_tenure")
                    contract_val = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"], key="in_contract")
                with col_b:
                    charges_val = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=200.0, value=75.0, key="in_charges")
                    support_val = st.checkbox("Dedicated Tech Support", value=True, key="in_support")

                submitted = st.form_submit_button("Run Churn Prediction", type="primary")

            if submitted:
                st.success(f"Features ready for inference: Tenure={tenure_val}mo, Charges=${charges_val}")

# =============================================================================
# 5. FEATURE PIPELINES
# =============================================================================
with tabs[4]:
    st.header("5. Feature Preprocessing & Scaling Pipelines")
    st.markdown(r"""
    Raw UI inputs cannot be fed directly into models without transformation.
    Numerical inputs must be standardized ($z = \frac{x - \mu}{\sigma}$) and categoricals one-hot encoded.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import numpy as np
import pandas as pd

def preprocess_features(tenure: int, charges: float, has_support: bool) -> np.ndarray:
    # Feature 1: Standardized tenure (mu=32, sigma=24)
    norm_tenure = (tenure - 32.0) / 24.0
    
    # Feature 2: Standardized charges (mu=65, sigma=30)
    norm_charges = (charges - 65.0) / 30.0
    
    # Feature 3: Binary indicator
    support_flag = 1.0 if has_support else 0.0
    
    # Return 2D array of shape (1, n_features)
    return np.array([[norm_tenure, norm_charges, support_flag]])

vec = preprocess_features(24, 75.0, True)
st.write(f"Preprocessed Feature Vector: `{vec.tolist()}`")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            def preprocess_features(tenure: int, charges: float, has_support: bool) -> np.ndarray:
                norm_t = (tenure - 32.0) / 24.0
                norm_c = (charges - 65.0) / 30.0
                sup_flag = 1.0 if has_support else 0.0
                return np.array([[norm_t, norm_c, sup_flag]])

            p_vec = preprocess_features(24, 75.0, True)
            df_vec = pd.DataFrame(p_vec, columns=["Norm_Tenure", "Norm_Charges", "Support_Flag"])
            st.caption("Transformed Feature Tensor:")
            st.dataframe(df_vec.style.format("{:.3f}"), width="stretch", hide_index=True)
            st.info("Input vectors must preserve exact feature ordering and data types matching the training schema.")

# =============================================================================
# 6. GENERATING PREDICTIONS
# =============================================================================
with tabs[5]:
    st.header("6. Generating Real-Time Predictions & Tracking Latency")
    st.markdown(r"""
    Execute inference via `predict_proba()` to obtain both discrete class decisions 
    and continuous calibrated risk probabilities.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import time
import numpy as np

# Feature vector from preprocessing step
feature_vector = np.array([[-0.333, 0.333, 1.0]])

# Track inference latency with high-resolution timer
t0 = time.perf_counter()
probabilities = model.predict_proba(feature_vector)[0]
latency_ms = (time.perf_counter() - t0) * 1000

churn_probability = float(probabilities[1])
predicted_class = "At Risk" if churn_probability >= 0.50 else "Retained"

st.write(f"Inference Latency: **{latency_ms:.2f} ms**")
st.write(f"Predicted Class: **{predicted_class}** (Prob: {churn_probability:.1%})")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            t0 = time.perf_counter()
            probs = model.predict_proba(np.array([[-0.333, 0.333, 1.0]]))[0]
            lat_ms = (time.perf_counter() - t0) * 1000

            churn_prob = float(probs[1])
            pred_label = "At Risk" if churn_prob >= 0.50 else "Retained"

            m1, m2 = st.columns(2)
            m1.metric("Predicted Status", pred_label)
            m2.metric("Inference Latency", f"{lat_ms:.2f} ms", delta="Fast (<5ms)")

            st.caption(f"Risk Probability: {churn_prob:.1%}")
            st.progress(churn_prob)

# =============================================================================
# 7. DISPLAYING RESULTS
# =============================================================================
with tabs[6]:
    st.header("7. Displaying Prediction Results & Decision Visualizations")
    st.markdown(r"""
    Communicate predictions with rich visual indicators: confidence progress bars, 
    risk tier badges, and prescriptive recommendations.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

prob = 0.72  # High Risk example

st.subheader("Model Diagnostic Result")
if prob >= 0.70:
    st.error(f"🚨 High Churn Risk ({prob:.1%})")
    st.markdown("**Prescriptive Remediation**:")
    st.markdown("- Dispatch dedicated customer success outreach within 24h.")
    st.markdown("- Offer 15% discount on annual contract renewal.")
elif prob >= 0.40:
    st.warning(f"⚠️ Moderate Risk ({prob:.1%})")
else:
    st.success(f"✅ Healthy Account ({prob:.1%})")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            sim_prob = st.slider("Simulate Output Probability", 0.0, 1.0, 0.72, key="sim_res_prob")

            if sim_prob >= 0.70:
                st.error(f"🚨 High Churn Risk ({sim_prob:.1%})")
                st.caption("Action Plan: Dispatch retention offer and schedule proactive check-in.")
            elif sim_prob >= 0.40:
                st.warning(f"⚠️ Moderate Risk ({sim_prob:.1%})")
                st.caption("Action Plan: Monitor usage telemetry for dropping query volume.")
            else:
                st.success(f"✅ Healthy Account ({sim_prob:.1%})")
                st.caption("Action Plan: Standard lifecycle engagement.")

# =============================================================================
# 8. INPUT VALIDATION
# =============================================================================
with tabs[7]:
    st.header("8. Handling Invalid Inputs & Out-of-Distribution Data")
    st.markdown(r"""
    Defensive ML applications detect out-of-distribution (OOD) inputs and boundary violations 
    before passing malformed tensors to the model.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

def validate_inference_inputs(tenure: int, monthly_bill: float) -> list[str]:
    violations = []
    if tenure <= 0 or tenure > 120:
        violations.append(f"Tenure ({tenure} mo) outside plausible range [1, 120].")
    if monthly_bill < 0:
        violations.append("Monthly charges cannot be negative.")
    if monthly_bill > 1000:
        violations.append(f"Charges (${monthly_bill}) exceed maximum threshold ($1,000).")
    return violations

# Validate before calling model.predict():
errors = validate_inference_inputs(0, -45.0)
if errors:
    for err in errors:
        st.warning(f"⚠️ Input Validation Warning: {err}")
    st.stop()
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            val_t = st.number_input("Test Tenure Input", value=-5, key="val_t_in")
            val_b = st.number_input("Test Monthly Bill Input", value=1250.0, key="val_b_in")

            violations = []
            if val_t <= 0 or val_t > 120:
                violations.append(f"Tenure ({val_t} mo) is outside allowable range [1, 120].")
            if val_b < 0 or val_b > 1000:
                violations.append(f"Monthly bill (${val_b}) is outside allowable range [$0, $1,000].")

            if violations:
                for v in violations:
                    st.warning(f"⚠️ {v}")
                st.error("Inference halted: Please correct invalid features above.")
            else:
                st.success("✅ All feature attributes passed schema validation bounds.")

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### ML Application Fundamentals Best Practices

| Concern | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Model Reloading** | Calling `joblib.load()` at the script root | Wrap inside `@st.cache_resource` to load weights once into RAM. |
| **Model Retraining** | Fitting models on user slider updates | Train offline; keep Streamlit strictly focused on low-latency inference. |
| **Form Debouncing** | Predicting on raw unbundled slider changes | Wrap all feature inputs inside `st.form`. |
| **Feature Mismatch** | Passing dictionaries with mismatched keys | Use a frozen scikit-learn `Pipeline` or `ColumnTransformer` object. |
| **Prediction Output** | Displaying raw float numbers (`0.74281923`) | Format as percentages (`74.3%`) and assign visual severity badges. |
""")
