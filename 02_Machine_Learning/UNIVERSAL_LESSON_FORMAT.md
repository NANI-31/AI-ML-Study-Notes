# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for All `.py` Lesson Files in `02_Machine_Learning`
### Reference Implementation Standard: Continuous Scroll · Graphviz Architecture · Zero LaTeX Math
### Target Domains: 1. Machine Learning · 2. Scikit-Learn

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `02_Machine_Learning` represents an interactive, production-grade learning module covering statistical learning algorithms, inductive biases, feature engineering, and model evaluation lifecycles.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden**.
2. **Zero LaTeX Math Notation**: Never use `$$...$$` or `$x$`. Express all loss functions and algorithmic updates in plain English and clean code formulations (e.g., `Loss = (1/2N) * Sum (y_pred - y_true)^2`, `Gini = 1 - Sum (p_i^2)`, `O(N * d)`).
3. **Heavy Architectural & Modeling Pipeline Diagrams**: Every lesson must feature Graphviz diagrams with dark theme styling (`bgcolor="#0B1120"`):
   - **Pipelines**: ColumnTransformer -> Preprocessor -> Estimator Fit -> Predict Flow.
   - **Tree Architectures**: Binary split nodes, threshold rules, leaf impurity scores.
   - **Validation Structures**: K-Fold cross-validation partitioning, train/val/test splits, data leakage barriers.
   - **Loss Landscapes / Boundaries**: Optimization paths, hyperplanes, margin boundaries.
4. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
5. **Dual Theory & Practical Delivery**: Combine pipeline diagrams, algorithmic derivations in plain text, standard `sklearn` estimator code, and interactive hyperparameter simulators (e.g., observing decision boundaries or regularization strength in real time).
6. **Uniform Pedagogical Endings**:
   - **Section N+1: Technical Interview Preparation & Production Traps** (Exactly 5 Q&As + 4 Traps [2 error, 2 warning] + 4 Flashcards).
   - **Section N+2: Self-Assessment Checklist** (Exactly 6 Checkboxes + `st.balloons()` completion banner).

---

## 2. Complete File Anatomy (Top-to-Bottom Structure)

```
┌────────────────────────────────────────────────────────────────────────┐
│ [A] Standard Imports Block (Exactly 5 Standard Imports)                │
├────────────────────────────────────────────────────────────────────────┤
│ [B] st.set_page_config(page_title=..., page_icon=..., layout="wide")   │
├────────────────────────────────────────────────────────────────────────┤
│ [C] Title Block: st.title + st.caption + Narrative with 5-Point List   │
├────────────────────────────────────────────────────────────────────────┤
│ [D] First st.divider()                                                 │
├────────────────────────────────────────────────────────────────────────┤
│ [E] Content Sections 1 through N (Separated by st.divider())           │
│     ├── Type 1: Estimator Lifecycle / Pipeline Diagram (Graphviz Dark) │
│     ├── Type 2: Model Trade-Off Spectrum (Bias vs Variance)            │
│     ├── Type 3: Production Runnable Scikit-Learn Pipeline Code        │
│     ├── Type 4: Interactive Hyperparameter & Metric Simulator          │
│     └── Type 6: Benchmark Metrics & Cross-Validation DataFrames        │
├────────────────────────────────────────────────────────────────────────┤
│ [F] Section N+1: Technical Interview Preparation (Mandatory 2nd-Last)  │
│     ├── Exactly 5 High-Frequency Q&A Expanders                         │
│     ├── Exactly 4 Production Traps (2x st.error + 2x st.warning)       │
│     └── Exactly 4 Interactive Flashcards in 2 Columns                  │
├────────────────────────────────────────────────────────────────────────┤
│ [G] st.divider()                                                       │
├────────────────────────────────────────────────────────────────────────┤
│ [H] Section N+2: Self-Assessment Checklist (Mandatory Last Section)    │
│     ├── Exactly 6 Checkboxes (All Initialized with value=True)         │
│     └── if chk1..chk6: -> st.balloons() + st.success(Next Lesson)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Structural Component Specifications

### 3.1 Standard Imports Block
```python
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional
```
Strictly prohibit `import os` and `from pathlib import Path`.

### 3.2 Page Configuration (`st.set_page_config`)
```python
st.set_page_config(
    page_title="<X.Y> <Lesson Title> - Machine Learning",
    page_icon="🤖",
    layout="wide"
)
```

### 3.3 Title Block & Opening Narrative
```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: 2. Machine Learning")
st.markdown("""
<2–4 sentence high-level narrative detailing the algorithmic foundation, inductive bias, 
and practical value of this technique in production predictive systems.>

In this lesson, you will master:
1. <Algorithmic formulation and mathematical objective 1>
2. <Feature preprocessing and assumption validation 2>
3. <Computational complexity and optimization dynamics 3>
4. <Scikit-Learn pipeline implementation 4>
5. <Overfitting diagnostics and hyperparameter tuning 5>
""")
st.divider()
```

---

## 4. Heavy Multi-Tier & Pipeline Diagrams Standard

### 4.1 Scikit-Learn Pipeline Architecture Template
```python
dot_ml_pipeline = """
digraph MLPipeline {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    subgraph cluster_data {
        label="Feature Ingestion";
        bgcolor="#1E293B";
        fontcolor="#38BDF8";
        color="#334155";
        raw_x [label="Feature Matrix X\\n[N samples, D features]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_prep {
        label="Feature Transformations";
        bgcolor="#1E293B";
        fontcolor="#F59E0B";
        color="#334155";
        scaler [label="StandardScaler\\n(Zero mean, unit variance)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        encoder [label="OneHotEncoder\\n(Categorical expansion)", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
    }

    subgraph cluster_model {
        label="Supervised Estimator";
        bgcolor="#1E293B";
        fontcolor="#10B981";
        color="#334155";
        clf [label="Trained Classifier / Regressor\\nFit via Gradient / Loss Minimization", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
    }

    raw_x -> scaler [label="Numerical Cols"];
    raw_x -> encoder [label="Categorical Cols"];
    scaler -> clf [label="Normalized Tensors"];
    encoder -> clf [label="Encoded Sparsity"];
}
"""
st.graphviz_chart(dot_ml_pipeline, use_container_width=True)
```

---

## 5. Mandatory Pedagogical Endings

### 5.1 Technical Interview Preparation (Section N+1)
Must contain:
1. **Exactly 5 Expanders**: High-frequency ML systems, bias-variance, and optimization interview questions.
2. **Exactly 4 Traps**:
   - `Trap 1` (`st.error`): Data leakage (fitting scalers on raw un-split dataset before train/test split).
   - `Trap 2` (`st.warning`): Class imbalance metric trap (evaluating accuracy instead of PR-AUC / F1).
   - `Trap 3` (`st.error`): Multicollinearity distortion in unregularized linear models.
   - `Trap 4` (`st.warning`): Target leakage in temporal or grouping splits.
3. **Exactly 4 Flashcards**: Rendered in a 2-column layout (`col_fc1`, `col_fc2`).

### 5.2 Self-Assessment Checklist (Section N+2)
Must contain:
- Exactly 6 checkboxes initialized with `value=True`.
- `if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:` triggering `st.balloons()` and `st.success()`.
