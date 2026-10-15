import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="17.3 Model Evaluation Dashboards",
    page_icon="📈",
    layout="wide"
)

st.title("17.3 Model Evaluation Dashboards in Streamlit")
st.caption("Comprehensive MLOps diagnostic consoles: Accuracy, Precision-Recall tradeoffs, F1 metrics, Confusion Matrices, ROC curves, Feature Importance, and Model Comparison Leaderboards.")

# -----------------------------------------------------------------------------
# ARCHITECTURAL OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(r"""
### The MLOps Model Evaluation Architecture

Evaluation dashboards provide the critical verification stage in the machine learning lifecycle.
They translate test set predictions into actionable diagnostic views for data scientists and executive stakeholders:

""")

st.graphviz_chart(
    r'''
    digraph EvalDashboard {
        graph [bgcolor="transparent", rankdir=TB];
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.5];
        edge [color="#64748B", fontname="Helvetica", fontsize=9, penwidth=1.2];

        holdout [label="Holdout Validation / Test Dataset\nGround Truth (y) + Model Predictions (y_hat)", fillcolor="#DBEAFE", color="#1D4ED8"];
        hub [label="Streamlit Evaluation Analytics Hub\n• Scalar KPIs (Accuracy, Precision, Recall, F1)\n• Diagnostic Curves (ROC-AUC, PR Curves)\n• Confusion Matrix Heatmap\n• Residual & Feature Importance Plots", fillcolor="#FEF3C7", color="#D97706"];
        decision [label="Production Decision\nChampion vs Challenger Leaderboard Matrix", fillcolor="#D1FAE5", color="#059669"];

        holdout -> hub [label="Evaluates Predictions"];
        hub -> decision [label="Selects Optimal Champion"];
    }
    '''
)

st.markdown("""
""")

tabs = st.tabs([
    "1. Accuracy Metrics",
    "2. Precision & Recall",
    "3. F1 Harmonic Score",
    "4. Confusion Matrix",
    "5. ROC & AUC Curves",
    "6. Feature Importance",
    "7. Actual vs Predicted",
    "8. Model Comparison Leaderboard"
])

# =============================================================================
# 1. ACCURACY METRICS
# =============================================================================
with tabs[0]:
    st.header("1. Accuracy Metrics & Class Imbalance Caveats")
    st.markdown(r"""
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
    **Caution**: In imbalanced regimes (e.g., 99% benign transactions, 1% fraud), a naive model predicting all zeros 
    achieves 99% accuracy while detecting zero fraud instances. Always supplement accuracy with balanced metrics.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

# Test set evaluation: 1,000 samples
tp, tn, fp, fn = 85, 870, 30, 15
total = tp + tn + fp + fn

raw_accuracy = (tp + tn) / total
balanced_acc = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))

c1, c2 = st.columns(2)
c1.metric("Overall Accuracy", f"{raw_accuracy:.1%}")
c2.metric("Balanced Accuracy", f"{balanced_acc:.1%}", delta=f"{balanced_acc - raw_accuracy:+.1%}")

if raw_accuracy - balanced_acc > 0.05:
    st.warning("⚠️ High accuracy is masking class imbalance performance degradation.")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            tp, tn, fp, fn = 85, 870, 30, 15
            total = tp + tn + fp + fn
            raw_acc = (tp + tn) / total
            bal_acc = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))

            c1, c2 = st.columns(2)
            c1.metric("Overall Accuracy", f"{raw_acc:.1%}")
            c2.metric("Balanced Accuracy", f"{bal_acc:.1%}", delta=f"{bal_acc - raw_acc:+.1%}")

            st.warning("⚠️ High accuracy is masking class imbalance performance degradation.")

# =============================================================================
# 2. PRECISION & RECALL
# =============================================================================
with tabs[1]:
    st.header("2. Precision, Recall & Dynamic Threshold Tuning")
    st.markdown(r"""
    - **Precision**: $\frac{TP}{TP + FP}$ — Out of all predicted positives, how many were correct? (Cost of False Alarm).
    - **Recall**: $\frac{TP}{TP + FN}$ — Out of all actual positives, how many did the model find? (Cost of Missed Positive).
    - Adjusting the decision threshold ($\tau$) directly balances precision against recall.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import numpy as np

# Decision threshold slider
threshold = st.slider("Classification Decision Threshold (tau)", 0.10, 0.90, 0.50, step=0.05)

# Simulated probabilities on 200 validation samples
# High threshold -> High Precision, Low Recall
# Low threshold -> Low Precision, High Recall
precision = min(0.98, 0.40 + (threshold * 0.65))
recall = max(0.35, 1.05 - (threshold * 0.75))

p_col, r_col = st.columns(2)
p_col.metric("Precision (Reliability)", f"{precision:.1%}")
r_col.metric("Recall (Sensitivity)", f"{recall:.1%}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            thresh = st.slider("Classification Threshold (τ)", 0.10, 0.90, 0.50, step=0.05, key="eval_thresh_sl")
            prec_val = min(0.98, 0.40 + (thresh * 0.65))
            rec_val = max(0.35, 1.05 - (thresh * 0.75))

            pc, rc = st.columns(2)
            pc.metric("Precision (Reliability)", f"{prec_val:.1%}")
            rc.metric("Recall (Coverage)", f"{rec_val:.1%}")

            st.caption("Trade-off Behavior:")
            st.progress(prec_val, text=f"Precision: {prec_val:.1%}")
            st.progress(rec_val, text=f"Recall: {rec_val:.1%}")

# =============================================================================
# 3. F1 HARMONIC SCORE
# =============================================================================
with tabs[2]:
    st.header("3. F1 Score (Harmonic Mean)")
    st.markdown(r"""
    $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
    The harmonic mean severely penalizes extreme imbalances between precision and recall, 
    making it a superior single-number benchmark for imbalanced datasets.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st

p = 0.88
r = 0.74

# Harmonic mean calculation
f1 = 2 * (p * r) / (p + r)
arithmetic_mean = (p + r) / 2

m1, m2 = st.columns(2)
m1.metric("F1 Score (Harmonic Mean)", f"{f1:.3f}")
m2.metric("Arithmetic Mean", f"{arithmetic_mean:.3f}", delta=f"{f1 - arithmetic_mean:.3f}")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            p = 0.88
            r = 0.74
            f1 = 2 * (p * r) / (p + r)
            am = (p + r) / 2

            m1, m2 = st.columns(2)
            m1.metric("F1 Score", f"{f1:.3f}")
            m2.metric("Arithmetic Mean", f"{am:.3f}", delta=f"{f1 - am:.3f}")
            st.info("💡 Notice that F1 is lower than the arithmetic mean, reflecting sensitivity to the lower recall score.")

# =============================================================================
# 4. CONFUSION MATRIX
# =============================================================================
with tabs[3]:
    st.header("4. Interactive Confusion Matrix Visualization")
    st.markdown(r"""
    Tabular breakdown of prediction outcomes categorized into True Positives, 
    False Positives (Type I Error), True Negatives, and False Negatives (Type II Error).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

# 2x2 Matrix DataFrame
cm_data = pd.DataFrame(
    [[870, 30],   # Actual Negative: [TN, FP]
     [15,  85]],   # Actual Positive: [FN, TP]
    index=["Actual: Negative (0)", "Actual: Positive (1)"],
    columns=["Pred: Negative (0)", "Pred: Positive (1)"]
)

st.dataframe(
    cm_data.style.background_gradient(cmap="Blues", axis=None),
    width="stretch"
)

st.caption("TN = 870 | FP (Type I) = 30 | FN (Type II) = 15 | TP = 85")
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            cm_data = pd.DataFrame(
                [[870, 30],
                 [15,  85]],
                index=["Actual: Negative (0)", "Actual: Positive (1)"],
                columns=["Pred: Negative (0)", "Pred: Positive (1)"]
            )
            st.caption("Confusion Matrix with Heatmap Gradient:")
            st.dataframe(
                cm_data.style.background_gradient(cmap="Blues", axis=None),
                width="stretch"
            )
            st.caption("TN: **870** | FP: **30** | FN: **15** | TP: **85**")

# =============================================================================
# 5. ROC & AUC CURVES
# =============================================================================
with tabs[4]:
    st.header("5. Receiver Operating Characteristic (ROC) & AUC")
    st.markdown(r"""
    The ROC curve charts the **True Positive Rate (Recall)** against the **False Positive Rate ($1 - \text{Specificity}$)** 
    across all classification thresholds. The **Area Under the Curve (AUC)** measures discriminative power ($0.5 = \text{random}$, $1.0 = \text{perfect}$).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np

# Synthesizing ROC curve coordinates
fpr = np.linspace(0.0, 1.0, 50)
# Typical convex ROC trajectory
tpr = 1.0 - (1.0 - fpr) ** 2.8

df_roc = pd.DataFrame({
    "False Positive Rate": np.round(fpr, 3),
    "Model Classifier": np.round(tpr, 3),
    "Random Chance (0.5 AUC)": np.round(fpr, 3)
}).set_index("False Positive Rate")

st.metric("ROC-AUC Score", "0.914", delta="Exceptional Discriminator")
st.line_chart(df_roc, height=240)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            fpr = np.linspace(0.0, 1.0, 40)
            tpr = 1.0 - (1.0 - fpr) ** 2.8
            df_roc = pd.DataFrame({
                "False Positive Rate": np.round(fpr, 3),
                "Model Classifier": np.round(tpr, 3),
                "Random Guess (AUC=0.5)": np.round(fpr, 3)
            }).set_index("False Positive Rate")

            st.metric("ROC-AUC Score", "0.914", delta="+0.414 vs random")
            st.line_chart(df_roc, height=200)

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
with tabs[5]:
    st.header("6. Feature Importance & Model Explainability")
    st.markdown(r"""
    Ranking feature predictors by Mean Decrease in Impurity (MDI) or Permutation Importance.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

importance_data = pd.DataFrame({
    "Feature": ["Account_Tenure", "Monthly_Charges", "Tech_Support_Flag", "Contract_TwoYear", "Paperless_Billing"],
    "Importance": [0.385, 0.292, 0.165, 0.110, 0.048]
}).sort_values(by="Importance", ascending=True)

st.caption("Relative Feature Importance:")
st.bar_chart(importance_data.set_index("Feature")["Importance"], horizontal=True, height=240)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            imp_df = pd.DataFrame({
                "Feature": ["Account_Tenure", "Monthly_Charges", "Tech_Support_Flag", "Contract_TwoYear", "Paperless_Billing"],
                "Importance": [0.385, 0.292, 0.165, 0.110, 0.048]
            }).sort_values(by="Importance", ascending=True)

            st.caption("Relative Feature Importance:")
            st.bar_chart(imp_df.set_index("Feature")["Importance"], horizontal=True, height=200)

# =============================================================================
# 7. ACTUAL VS PREDICTED
# =============================================================================
with tabs[6]:
    st.header("7. Regression Diagnostics: Actual versus Predicted Plots")
    st.markdown(r"""
    Evaluating regression performance by comparing actual target values ($y$) against predictions ($\hat{y}$) 
    relative to the ideal 45-degree parity line.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd
import numpy as np

np.random.seed(42)
y_actual = np.linspace(10, 100, 30)
# Predicted values with normal noise
y_predicted = y_actual + np.random.normal(0, 4.5, 30)

residuals = y_actual - y_predicted
rmse = np.sqrt(np.mean(residuals ** 2))
r2 = 1.0 - (np.sum(residuals ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2))

c1, c2 = st.columns(2)
c1.metric("RMSE", f"${rmse:.2f}")
c2.metric("R² Score", f"{r2:.3f}")

reg_df = pd.DataFrame({"Actual": y_actual, "Predicted": y_predicted})
st.scatter_chart(reg_df, x="Actual", y="Predicted", height=240)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            np.random.seed(42)
            y_act = np.linspace(10, 100, 30)
            y_pred = y_act + np.random.normal(0, 4.5, 30)
            res = y_act - y_pred
            rmse_val = np.sqrt(np.mean(res ** 2))
            r2_val = 1.0 - (np.sum(res ** 2) / np.sum((y_act - np.mean(y_act)) ** 2))

            rc1, rc2 = st.columns(2)
            rc1.metric("RMSE", f"${rmse_val:.2f}")
            rc2.metric("R² Score", f"{r2_val:.3f}")

            reg_plot_df = pd.DataFrame({"Actual": y_act, "Predicted": y_pred})
            st.scatter_chart(reg_plot_df, x="Actual", y="Predicted", height=180)

# =============================================================================
# 8. MODEL COMPARISON LEADERBOARD
# =============================================================================
with tabs[7]:
    st.header("8. Model Comparison Leaderboard")
    st.markdown(r"""
    A comparative MLOps tournament table comparing candidate architectures 
    across accuracy, precision, recall, F1, and inference latency.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💻 Python Source Code**")
        st.code(r'''
import streamlit as st
import pandas as pd

leaderboard = pd.DataFrame({
    "Model Architecture": ["XGBoost Classifier", "LightGBM", "Random Forest", "Logistic Regression"],
    "Accuracy": [0.892, 0.887, 0.865, 0.812],
    "F1 Score": [0.884, 0.879, 0.851, 0.789],
    "ROC-AUC": [0.935, 0.928, 0.912, 0.860],
    "P95 Latency (ms)": [4.2, 2.1, 14.8, 0.8],
    "Status": ["Champion 🏆", "Candidate", "Baseline", "Legacy"]
})

st.dataframe(
    leaderboard.style.format({
        "Accuracy": "{:.1%}",
        "F1 Score": "{:.3f}",
        "ROC-AUC": "{:.3f}",
        "P95 Latency (ms)": "{:.1f} ms"
    }).background_gradient(subset=["ROC-AUC", "F1 Score"], cmap="Blues"),
    width="stretch",
    hide_index=True
)
''', language="python")

    with col2:
        st.markdown("**🖥️ Rendered Streamlit Output**")
        with st.container(border=True):
            leaderboard = pd.DataFrame({
                "Model Architecture": ["XGBoost Classifier", "LightGBM", "Random Forest", "Logistic Regression"],
                "Accuracy": [0.892, 0.887, 0.865, 0.812],
                "F1 Score": [0.884, 0.879, 0.851, 0.789],
                "ROC-AUC": [0.935, 0.928, 0.912, 0.860],
                "P95 Latency (ms)": [4.2, 2.1, 14.8, 0.8],
                "Status": ["Champion 🏆", "Candidate", "Baseline", "Legacy"]
            })

            st.caption("Model Evaluation Leaderboard:")
            st.dataframe(
                leaderboard.style.format({
                    "Accuracy": "{:.1%}",
                    "F1 Score": "{:.3f}",
                    "ROC-AUC": "{:.3f}",
                    "P95 Latency (ms)": "{:.1f} ms"
                }).background_gradient(subset=["ROC-AUC", "F1 Score"], cmap="Blues"),
                width="stretch",
                hide_index=True
            )

# -----------------------------------------------------------------------------
# SUMMARY CHEATSHEET
# -----------------------------------------------------------------------------
st.markdown(r"""
---
### Model Evaluation Best Practices

| Metric / Tool | Primary Use Case | Caution / Watch-Out |
| :--- | :--- | :--- |
| **Accuracy** | Balanced multi-class datasets | Highly deceptive under severe class imbalance ($>90\%:10\%$). |
| **Precision** | Cost of False Alarm is high (Spam filter, Fraud alert) | Can be artificially boosted by predicting positive very rarely. |
| **Recall** | Cost of Missed Positive is high (Cancer detection, Defects) | Can be trivially maximized by predicting positive for everything. |
| **F1 Score** | Balanced tradeoff on skewed targets | Does not weight true negatives. |
| **ROC-AUC** | Global discriminative ranking power | Insensitive to class skew; use PR-AUC for extreme rarity. |
| **Residual Plots**| Validating regression linearity | Patterns/curves in residuals indicate non-linear structure uncaptured by model. |
""")
