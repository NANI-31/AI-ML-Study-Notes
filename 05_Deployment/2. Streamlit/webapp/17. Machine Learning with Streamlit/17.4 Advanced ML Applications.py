import streamlit as st
import pandas as pd
import numpy as np
import io
import time

st.set_page_config(
    page_title="17.4 Advanced ML Applications",
    page_icon="🚀",
    layout="wide"
)

st.title("17.4 Advanced Machine Learning Applications in Streamlit")
st.caption("Production MLOps systems: bulk CSV batch scoring, model registry versioning, hyperparameter playgrounds, SHAP explainability, and drift monitoring.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### Advanced MLOps System Topology

Enterprise ML applications in Streamlit evolve beyond single-prediction prototypes into complete **Model Operations Consoles**:

""")

st.graphviz_chart(
    r'''
    digraph AdvancedML {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        registry [label="Enterprise Model Registry\nv1.2.0 (Legacy) | v2.0.0 (Champion) | v2.1.0-rc", fillcolor="#DBEAFE", color="#1D4ED8"];
        engine [label="Streamlit Inference Engine\n• Single-Item Prediction Form\n• Batch CSV Inference (10k+ rows)\n• Real-Time Confidence Gauges", fillcolor="#FEF3C7", color="#D97706"];
        xai [label="Explainability & Governance\n• SHAP TreeExplainer Attributions\n• Waterfall / Summary Plots\n• Drift & Telemetry Audit", fillcolor="#D1FAE5", color="#059669"];

        registry -> engine [label="Active Champion Pipeline"];
        engine -> xai [label="Feature Contributions & Telemetry"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Batch Prediction",
    "2. CSV Inference Pipeline",
    "3. Model Selection UI",
    "4. Hyperparameter Explorer",
    "5. Confidence Displays",
    "6. Explainability (SHAP)",
    "7. Drift Monitoring",
    "8. Model Version Registry"
])

# =============================================================================
# 1. BATCH PREDICTION
# =============================================================================
with tabs[0]:
    st.header("1. Vectorized Batch Prediction Engine")
    st.markdown(r"""
    Batch prediction evaluates thousands of rows concurrently using vectorized NumPy/Pandas pipelines 
    rather than looping row-by-row, achieving $100\times$ throughput speedups.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np

def score_batch_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    # Vectorized scoring logic
    features = df_input[["Tenure", "MonthlyCharges"]].values
    weights = np.array([-0.04, 0.02])
    logits = np.dot(features, weights) - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    df_output = df_input.copy()
    df_output["Churn_Probability"] = np.round(probs, 3)
    df_output["Predicted_Class"] = np.where(probs >= 0.5, "Churn", "Retained")
    return df_output

# Simulated batch of 500 records
batch_data = pd.DataFrame({
    "CustomerID": [f"C-{1000 + i}" for i in range(500)],
    "Tenure": np.random.randint(1, 72, 500),
    "MonthlyCharges": np.random.uniform(20, 120, 500).round(2)
})

scored_df = score_batch_dataframe(batch_data)
st.metric("Total Rows Scored", len(scored_df), delta="Vectorized SIMD")
st.dataframe(scored_df.head(6), width="stretch", hide_index=True)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            np.random.seed(42)
            b_data = pd.DataFrame({
                "CustomerID": [f"C-{1000 + i}" for i in range(250)],
                "Tenure": np.random.randint(1, 72, 250),
                "MonthlyCharges": np.random.uniform(20, 120, 250).round(2)
            })

            features = b_data[["Tenure", "MonthlyCharges"]].values
            logits = np.dot(features, np.array([-0.04, 0.02])) - 0.5
            probs = 1.0 / (1.0 + np.exp(-logits))

            res_df = b_data.copy()
            res_df["Churn_Probability"] = np.round(probs, 3)
            res_df["Predicted_Class"] = np.where(probs >= 0.5, "Churn", "Retained")

            st.metric("Batch Records Processed", len(res_df), delta="Sub-millisecond Vectorization")
            st.dataframe(res_df.head(6), width="stretch", hide_index=True)

# =============================================================================
# 2. CSV INFERENCE PIPELINE
# =============================================================================
with tabs[1]:
    st.header("2. End-to-End CSV File Inference & Export")
    st.markdown(r"""
    A self-service CSV scoring portal: users upload an unscored CSV, the pipeline validates required columns, 
    generates predictions, and yields an enriched CSV for download.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import io

uploaded_csv = st.file_uploader("Upload Unscored Dataset", type=["csv"])

if uploaded_csv:
    input_df = pd.read_csv(uploaded_csv)
    REQUIRED = {"Tenure", "MonthlyCharges"}
    
    if not REQUIRED.issubset(input_df.columns):
        st.error(f"Missing required columns: {REQUIRED - set(input_df.columns)}")
    else:
        # Score dataset
        scored_df = score_batch_dataframe(input_df)
        csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
        
        st.success(f"Successfully scored {len(scored_df)} customer rows.")
        st.download_button(
            "📥 Download Scored CSV",
            data=csv_bytes,
            file_name="scored_predictions.csv",
            mime="text/csv"
        )
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            # Pre-loaded mock file buffer for demonstration
            mock_csv = "CustomerID,Tenure,MonthlyCharges\nC-01,12,65.5\nC-02,48,110.0\nC-03,4,30.0\nC-04,36,88.2\n"
            input_df = pd.read_csv(io.StringIO(mock_csv))
            
            st.caption("Input Preview (Uploaded Dataset):")
            st.dataframe(input_df, width="stretch", hide_index=True)
            
            # Run inference
            f_vals = input_df[["Tenure", "MonthlyCharges"]].values
            p_vals = 1.0 / (1.0 + np.exp(-(np.dot(f_vals, np.array([-0.04, 0.02])) - 0.5)))
            out_df = input_df.copy()
            out_df["Predicted_Risk"] = np.round(p_vals, 3)
            out_df["Status"] = np.where(p_vals >= 0.5, "High Risk", "Low Risk")

            st.caption("Scored Enriched Dataset Ready for Download:")
            st.dataframe(out_df, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download Scored CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_predictions.csv",
                mime="text/csv",
                key="csv_dl_demo_btn"
            )

# =============================================================================
# 3. MODEL SELECTION UI
# =============================================================================
with tabs[2]:
    st.header("3. Multi-Model Architecture Selection Interface")
    st.markdown(r"""
    Compare multiple candidate model architectures side-by-side on identical input features.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

selected_model = st.selectbox(
    "Select Inference Engine",
    ["XGBoost (Champion)", "Random Forest (Ensemble)", "Logistic Regression (Linear)"]
)

# Feature inputs
t = st.slider("Account Tenure", 1, 72, 18)
c = st.slider("Monthly Charges ($)", 20.0, 150.0, 85.0)

# Multi-model prediction simulation
if selected_model.startswith("XGBoost"):
    prob = 0.68
elif selected_model.startswith("Random Forest"):
    prob = 0.61
else:
    prob = 0.54

st.metric(f"{selected_model} Risk Estimate", f"{prob:.1%}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            m_choice = st.selectbox(
                "Select Inference Engine",
                ["XGBoost (Champion)", "Random Forest (Ensemble)", "Logistic Regression (Linear)"],
                key="multi_mod_sel"
            )

            t_val = st.slider("Account Tenure", 1, 72, 18, key="mm_t_sl")
            c_val = st.slider("Monthly Charges ($)", 20.0, 150.0, 85.0, key="mm_c_sl")

            if "XGBoost" in m_choice:
                p_est = 0.68
                lat = "3.2 ms"
            elif "Random Forest" in m_choice:
                p_est = 0.61
                lat = "12.4 ms"
            else:
                p_est = 0.54
                lat = "0.7 ms"

            c1, c2 = st.columns(2)
            c1.metric("Predicted Churn Risk", f"{p_est:.1%}")
            c2.metric("Inference Latency", lat)
            st.progress(p_est)

# =============================================================================
# 4. HYPERPARAMETER EXPLORER
# =============================================================================
with tabs[3]:
    st.header("4. Interactive Hyperparameter Exploration Playground")
    st.markdown(r"""
    Explore the effects of tree depth, regularization, and estimator counts on bias-variance tradeoffs in real-time.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np

c1, c2 = st.columns(2)
with c1:
    max_depth = st.slider("Max Tree Depth", 1, 15, 6)
with c2:
    n_trees = st.slider("Number of Estimators", 10, 200, 50, step=10)

# Simulated train vs validation performance curve
train_auc = min(0.99, 0.70 + (max_depth * 0.02) + (n_trees * 0.0005))
val_auc = min(0.92, 0.72 + (max_depth * 0.02) - max(0, (max_depth - 8) * 0.03))

m1, m2, m3 = st.columns(3)
m1.metric("Train AUC-ROC", f"{train_auc:.3f}")
m2.metric("Val AUC-ROC", f"{val_auc:.3f}")
overfit_gap = train_auc - val_auc
m3.metric("Overfitting Gap", f"{overfit_gap:+.3f}", delta=f"{-overfit_gap:.3f}", delta_color="inverse")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            hc1, hc2 = st.columns(2)
            with hc1:
                depth_in = st.slider("Max Tree Depth", 1, 15, 6, key="hp_depth_sl")
            with hc2:
                trees_in = st.slider("Number of Estimators", 10, 200, 50, step=10, key="hp_trees_sl")

            tr_auc = min(0.99, 0.70 + (depth_in * 0.02) + (trees_in * 0.0005))
            v_auc = min(0.92, 0.72 + (depth_in * 0.02) - max(0, (depth_in - 8) * 0.03))
            gap = tr_auc - v_auc

            hm1, hm2, hm3 = st.columns(3)
            hm1.metric("Train AUC", f"{tr_auc:.3f}")
            hm2.metric("Val AUC", f"{v_auc:.3f}")
            hm3.metric("Generalization Gap", f"{gap:+.3f}", delta=f"{-gap:.3f}", delta_color="inverse")

            if depth_in > 8:
                st.warning("⚠️ High tree depth is introducing variance overfitting.")

# =============================================================================
# 5. CONFIDENCE DISPLAYS
# =============================================================================
with tabs[4]:
    st.header("5. Calibrated Prediction Confidence & Entropy Displays")
    st.markdown(r"""
    Highlighting classification certainty using **Shannon Entropy**:
    $$H(p) = - \sum_{i} p_i \log_2(p_i)$$
    When entropy is high (probabilities close to 50/50), models are uncertain and require human-in-the-loop review.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np

p_pos = st.slider("Model Output Probability (Positive)", 0.01, 0.99, 0.52)
p_neg = 1.0 - p_pos

# Shannon Entropy (Certainty: 0 = completely certain, 1 = complete uncertainty)
entropy = - (p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))

c1, c2 = st.columns(2)
c1.metric("Prediction Confidence", f"{max(p_pos, p_neg):.1%}")
c2.metric("Decision Entropy", f"{entropy:.3f} bits")

if entropy > 0.95:
    st.warning("⚠️ High Ambiguity: Model is uncertain. Route to human underwriter.")
else:
    st.success("✅ High Confidence: Auto-approval policy applies.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            p_pos = st.slider("Positive Class Probability", 0.01, 0.99, 0.52, key="conf_p_sl")
            p_neg = 1.0 - p_pos
            entropy = - (p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))

            cc1, cc2 = st.columns(2)
            cc1.metric("Confidence", f"{max(p_pos, p_neg):.1%}")
            cc2.metric("Entropy", f"{entropy:.3f} bits")

            if entropy > 0.95:
                st.warning("⚠️ High Ambiguity: Model is uncertain. Route to human underwriter.")
            else:
                st.success("✅ High Confidence: Auto-decision policy applies.")

# =============================================================================
# 6. EXPLAINABILITY (SHAP)
# =============================================================================
with tabs[5]:
    st.header("6. Local Feature Attribution (SHAP Waterfall Concept)")
    st.markdown(r"""
    Explain individual predictions by decomposing the final score into baseline expectation plus additive feature contributions:
    $$f(x) = \phi_0 + \sum_{i=1}^M \phi_i$$
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# SHAP additive contributions for Customer #108
base_val = 0.35  # Population baseline churn rate
shap_contributions = pd.DataFrame({
    "Feature": ["Tenure (4 mo)", "MonthlyCharges ($115)", "Contract (Month-to-Month)", "TechSupport (No)"],
    "SHAP_Value": [+0.18, +0.12, +0.10, +0.05]
}).sort_values(by="SHAP_Value", ascending=True)

final_prediction = base_val + shap_contributions["SHAP_Value"].sum()

st.metric("Final Predicted Churn", f"{final_prediction:.1%}", delta=f"+{final_prediction - base_val:.1%} vs baseline")
st.caption("Feature Attribution (Pushes towards Churn):")
st.bar_chart(shap_contributions.set_index("Feature")["SHAP_Value"], horizontal=True, height=220)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            base_val = 0.35
            shap_df = pd.DataFrame({
                "Feature": ["Tenure (4 mo)", "MonthlyCharges ($115)", "Contract (Month-to-Month)", "TechSupport (No)"],
                "SHAP_Impact": [+0.18, +0.12, +0.10, +0.05]
            }).sort_values(by="SHAP_Impact", ascending=True)

            final_p = base_val + shap_df["SHAP_Impact"].sum()

            st.metric("Final Model Prediction", f"{final_p:.1%}", delta=f"+{final_p - base_val:.1%} vs Base (35%)")
            st.caption("SHAP Feature Attributions:")
            st.bar_chart(shap_df.set_index("Feature")["SHAP_Impact"], horizontal=True, height=180)

# =============================================================================
# 7. DRIFT MONITORING
# =============================================================================
with tabs[6]:
    st.header("7. Production Model & Data Drift Monitoring")
    st.markdown(r"""
    Detecting feature distribution shifts between baseline training data and live production inference streams 
    using the **Population Stability Index (PSI)**:
    $$\text{PSI} = \sum \left( \% \text{Actual} - \% \text{Expected} \right) \times \ln\left(\frac{\% \text{Actual}}{\% \text{Expected}}\right)$$
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

drift_telemetry = pd.DataFrame({
    "Feature Name": ["Monthly_Charges", "Account_Tenure", "Support_Tickets", "Contract_Type"],
    "PSI_Score": [0.042, 0.118, 0.285, 0.015],
    "Drift_Status": ["Stable", "Moderate Shift", "Severe Drift 🚨", "Stable"],
    "Action_Required": ["None", "Monitor Closely", "Retrain Model", "None"]
})

st.dataframe(
    drift_telemetry.style.apply(
        lambda row: ['background-color: #551111' if 'Severe' in row['Drift_Status'] else '' for _ in row],
        axis=1
    ),
    width="stretch",
    hide_index=True
)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            drift_telemetry = pd.DataFrame({
                "Feature Name": ["Monthly_Charges", "Account_Tenure", "Support_Tickets", "Contract_Type"],
                "PSI_Score": [0.042, 0.118, 0.285, 0.015],
                "Drift_Status": ["Stable", "Moderate Shift", "Severe Drift 🚨", "Stable"],
                "Action_Required": ["None", "Monitor Closely", "Retrain Model", "None"]
            })
            st.caption("Feature Population Stability Index (PSI):")
            st.dataframe(drift_telemetry, width="stretch", hide_index=True)
            st.error("🚨 Support_Tickets PSI (0.285) > 0.25 indicates significant data drift. Model retraining recommended.")

# =============================================================================
# 8. MODEL VERSION REGISTRY
# =============================================================================
with tabs[7]:
    st.header("8. Enterprise Model Version Selection & Registry")
    st.markdown(r"""
    Switch dynamically between production Champion, Candidate Shadow, and Legacy models.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

registry = {
    "v2.1.0 (Champion)": {"algo": "XGBoost", "auc": 0.935, "deployed": "2024-09-15", "traffic": "90%"},
    "v2.2.0-rc1 (Canary)": {"algo": "LightGBM", "auc": 0.941, "deployed": "2024-10-10", "traffic": "10%"},
    "v1.8.0 (Legacy)": {"algo": "RandomForest", "auc": 0.895, "deployed": "2024-03-01", "traffic": "0%"}
}

chosen_ver = st.selectbox("Active Registry Checkpoint", list(registry.keys()))
meta = registry[chosen_ver]

c1, c2, c3 = st.columns(3)
c1.metric("Architecture", meta["algo"])
c2.metric("Validation AUC", str(meta["auc"]))
c3.metric("Live Traffic Split", meta["traffic"])
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            registry = {
                "v2.1.0 (Champion)": {"algo": "XGBoost", "auc": 0.935, "deployed": "2024-09-15", "traffic": "90%"},
                "v2.2.0-rc1 (Canary)": {"algo": "LightGBM", "auc": 0.941, "deployed": "2024-10-10", "traffic": "10%"},
                "v1.8.0 (Legacy)": {"algo": "RandomForest", "auc": 0.895, "deployed": "2024-03-01", "traffic": "0%"}
            }

            c_ver = st.selectbox("Active Registry Checkpoint", list(registry.keys()), key="reg_ver_sel")
            m_data = registry[c_ver]

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Architecture", m_data["algo"])
            rc2.metric("Validation AUC", str(m_data["auc"]))
            rc3.metric("Traffic Routing", m_data["traffic"])

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Advanced ML Applications Best Practices

| Capability | Anti-Pattern | Recommended Solution |
| :--- | :--- | :--- |
| **Batch Scoring** | Iterating with `df.iterrows()` | Vectorized batch scoring via NumPy matrix operations. |
| **CSV Inference** | Assuming uploaded CSV columns match | Validate `set(required_cols).issubset(df.columns)` before scoring. |
| **Explainability** | Presenting black-box numbers | Decompose into additive SHAP feature impacts. |
| **Model Drift** | Deploying once and never tracking distributions | Monitor PSI and Kolmogorov-Smirnov distance on production streams. |
| **Model Updates** | Overwriting a single `model.joblib` file | Version model artifacts semantically (`v1.0.0`, `v1.1.0`) in a registry. |
""")
