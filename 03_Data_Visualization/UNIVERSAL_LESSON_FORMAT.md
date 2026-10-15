# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for All `.py` Lesson Files in `03_Data_Visualization`
### Reference Implementation Standard: Continuous Scroll · Graphviz Architecture · Zero LaTeX Math
### Target Domains: 1. Data Visualization · 2. Matplotlib · 3. Seaborn · 4. Plotly

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `03_Data_Visualization` represents an interactive, production-grade learning module covering visual grammar, graphical perception, object-oriented rendering hierarchies, and interactive charts.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden**.
2. **Zero LaTeX Math Notation**: Never use `$$...$$` or `$x$`. Express all statistical transformations and visual scales in plain English and clean code formulations (e.g., `Log10(x)`, `Kernel Density Estimate with Gaussian Kernel`, `Min-Max Normalization: (x - min) / (max - min)`).
3. **Heavy Architectural & Object-Hierarchy Diagrams**: Every lesson must feature Graphviz diagrams with dark theme styling (`bgcolor="#0B1120"`):
   - **Rendering Trees**: Figure -> Axes -> Axis (X, Y) -> Ticks -> Artist/Spines hierarchy.
   - **Grammar of Graphics**: Data -> Aesthetics (Aesthetic Mapping) -> Geometric Objects (Geoms) -> Facets -> Coordinates.
   - **Chart Decision Flowcharts**: Distribution vs Relationship vs Composition vs Comparison decision trees.
4. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
5. **Dual Theory & Practical Delivery**: Combine object-oriented architecture diagrams, comparative visual aesthetics, production code with `matplotlib`/`seaborn`/`plotly`, and an interactive visual workbench where users adjust chart parameters dynamically.
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
│     ├── Type 1: Object Hierarchy / Grammar Tree (Graphviz Dark)        │
│     ├── Type 2: Chart Type Taxonomy & Trade-Off Matrix                 │
│     ├── Type 3: Production Runnable Visualization Code                 │
│     ├── Type 4: Interactive Parameter Workbench & Chart Studio         │
│     └── Type 6: Aesthetic Guidelines & Color Theory DataFrames         │
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
    page_title="<X.Y> <Lesson Title> - Data Visualization",
    page_icon="📊",
    layout="wide"
)
```

### 3.3 Title Block & Opening Narrative
```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: 3. Data Visualization")
st.markdown("""
<2–4 sentence high-level narrative detailing graphical perception, visual encoding principles, 
and how this visualization structure communicates insights without distortion.>

In this lesson, you will master:
1. <Visual encoding mechanics and data mapping 1>
2. <Object-oriented figure and axes architecture 2>
3. <Chart taxonomy selection and cognitive ergonomics 3>
4. <Production styling and multi-subplot layout 4>
5. <Color vision deficiency accessibility and export traps 5>
""")
st.divider()
```

---

## 4. Heavy Multi-Tier & Object Hierarchy Diagrams Standard

### 4.1 Matplotlib Object Hierarchy Template
```python
dot_viz_hierarchy = """
digraph VizHierarchy {
    rankdir=TB;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9, shape=box, style="filled,rounded"];
    edge [fontname="Helvetica", fontsize=8, color="#38BDF8", fontcolor="#94A3B8"];

    subgraph cluster_fig {
        label="Figure Canvas (Root Object)";
        bgcolor="#1E293B";
        fontcolor="#38BDF8";
        color="#334155";
        fig [label="Figure Object\\n(Canvas, DPI, Background, Subplot Grid)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
    }

    subgraph cluster_ax {
        label="Axes Coordinates (Plot Area)";
        bgcolor="#1E293B";
        fontcolor="#F59E0B";
        color="#334155";
        ax [label="Axes Object (Subplot)\\nContains Coordinate System and Geometry", fillcolor="#78350F", fontcolor="#FEF3C7", color="#F59E0B"];
        xaxis [label="X-Axis & Ticks", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
        yaxis [label="Y-Axis & Ticks", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
        artists [label="Artists (Lines, Bars, Patches, Legends)", fillcolor="#064E3B", fontcolor="#D1FAE5", color="#10B981"];
    }

    fig -> ax [label="Creates Subplots"];
    ax -> xaxis [label="Manages"];
    ax -> yaxis [label="Manages"];
    ax -> artists [label="Renders Visuals"];
}
"""
st.graphviz_chart(dot_viz_hierarchy, use_container_width=True)
```

---

## 5. Mandatory Pedagogical Endings

### 5.1 Technical Interview Preparation (Section N+1)
Must contain:
1. **Exactly 5 Expanders**: High-frequency visual encoding, perception, and plotting library questions.
2. **Exactly 4 Traps**:
   - `Trap 1` (`st.error`): Truncated y-axis distortion (exaggerating relative variance dishonestly).
   - `Trap 2` (`st.warning`): Color blindness inaccessibility (red-green rainbow palettes).
   - `Trap 3` (`st.error`): Global pyplot state mutation race condition in multi-threaded webapps.
   - `Trap 4` (`st.warning`): Overplotting in large N scatter plots (lacking alpha transparency or hexbinning).
3. **Exactly 4 Flashcards**: Rendered in a 2-column layout (`col_fc1`, `col_fc2`).

### 5.2 Self-Assessment Checklist (Section N+2)
Must contain:
- Exactly 6 checkboxes initialized with `value=True`.
- `if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:` triggering `st.balloons()` and `st.success()`.
