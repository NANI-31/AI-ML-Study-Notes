# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for All `.py` Lesson Files in `01_Data_Science_Foundations`
### Reference Implementation Standard: Continuous Scroll · Graphviz Architecture · Zero LaTeX Math
### Target Domains: 1. Numpy · 2. Pandas · 3. Spreadsheet · 4. Regex · 5. SQLite

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `01_Data_Science_Foundations` represents an interactive, production-grade learning module covering computational data manipulation, vectorization, tabular schemas, and data structures.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden**.
2. **Zero LaTeX Math Notation**: Never use `$$...$$` or `$x$`. Express all mathematical and computational operations in plain English, code expressions, or explicit tensor/array dimension shapes (e.g., `Shape: [Rows, Columns]`, `O(N)`, `Sum_i (x_i - mean)^2 / N`).
3. **Heavy Architectural & Data Structure Diagrams**: Every lesson must feature clean Graphviz diagrams with dark theme styling (`bgcolor="#0B1120"`):
   - **Numpy**: Multi-dimensional memory layouts, strided memory buffers, broadcasting alignments.
   - **Pandas**: Index alignments, split-apply-combine trees, merge/join Venn hierarchies.
   - **Regex**: Deterministic and Non-deterministic Finite Automata (DFA/NFA) state machines.
   - **SQLite**: B-Tree index structures, relational entity schemas, query execution plans.
4. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
5. **Dual Theory & Practical Delivery**: Combine data structure diagrams, tabular audits, vectorized code benchmarks against naive Python loops, and a live interactive workbench simulator.
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
│     ├── Type 1: Data Structure / Memory Layout Diagram (Graphviz Dark) │
│     ├── Type 2: Vectorized vs Iterative Comparative Matrix             │
│     ├── Type 3: Production Runnable Python Code Block                 │
│     ├── Type 4: Interactive Workbench / Parameter Simulator            │
│     └── Type 6: Structured DataFrames & Benchmark Telemetry Tables     │
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
    page_title="<X.Y> <Lesson Title> - Data Science Foundations",
    page_icon="📐",
    layout="wide"
)
```

### 3.3 Title Block & Opening Narrative
```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: 1. Data Science Foundations")
st.markdown("""
<2–4 sentence high-level narrative detailing the computational mechanism, memory mechanics, 
and why vectorization or structured queries matter in data systems.>

In this lesson, you will master:
1. <Core data structure or memory layout 1>
2. <Transformation mechanic or vectorization principle 2>
3. <Computational complexity or algorithm intuition 3>
4. <Production code pattern and common pitfalls 4>
5. <Edge-case handling and performance profiling 5>
""")
st.divider()
```

---

## 4. Heavy Multi-Tier & Data Structure Diagrams Standard

All Graphviz diagrams use the unified dark theme:
* `bgcolor="#0B1120"`
* Outer boundary / cluster: `#1E293B` with border `#334155`
* Edges: `#38BDF8` with font `#94A3B8`
* Storage / Buffers: `shape=cylinder` with `#1E293B`

### 4.1 Memory Layout & Broadcasting Diagram Template (Numpy)
```python
dot_memory_layout = """
digraph MemoryLayout {
    rankdir=LR;
    bgcolor="#0B1120";
    node [fontname="Helvetica", fontsize=9];

    buffer_table [shape=plaintext, label=<
        <table border="0" cellborder="1" cellspacing="0" cellpadding="6" bgcolor="#1E293B">
            <tr><td colspan="4" bgcolor="#334155"><font color="#FFFFFF"><b>Contiguous C-Order Strided Memory Buffer</b></font></td></tr>
            <tr bgcolor="#0F172A">
                <td><font color="#38BDF8">Offset +0: 10</font></td>
                <td><font color="#38BDF8">Offset +8: 20</font></td>
                <td><font color="#38BDF8">Offset +16: 30</font></td>
                <td><font color="#38BDF8">Offset +24: 40</font></td>
            </tr>
        </table>
    >];
}
"""
st.graphviz_chart(dot_memory_layout, use_container_width=True)
```

---

## 5. Mandatory Pedagogical Endings

### 5.1 Technical Interview Preparation (Section N+1)
Must contain:
1. **Exactly 5 Expanders**: High-frequency data science architectural & indexing questions.
2. **Exactly 4 Traps**:
   - `Trap 1` (`st.error`): Memory copy / slicing trap (e.g. `SettingWithCopyWarning`).
   - `Trap 2` (`st.warning`): Type casting / downcasting precision loss.
   - `Trap 3` (`st.error`): Unvectorized loop explosion in large corpora.
   - `Trap 4` (`st.warning`): NaN propagation / implicit null coercion.
3. **Exactly 4 Flashcards**: Rendered in a 2-column layout (`col_fc1`, `col_fc2`).

### 5.2 Self-Assessment Checklist (Section N+2)
Must contain:
- Exactly 6 checkboxes initialized with `value=True`.
- `if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:` triggering `st.balloons()` and `st.success()`.
