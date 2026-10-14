# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for Domain: `1. Numpy`
### Target Curriculum: `01_Data_Science_Foundations/1. Numpy`
### Reference Implementation Standard: Continuous Vertical Scroll · Graphviz Memory Architecture · Zero LaTeX Math · Multi-Expander Technical Interview Hub

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `1. Numpy` represents an interactive, production-grade learning module covering multi-dimensional array architecture, strided memory buffers, vectorization semantics, broadcasting rules, cache locality, and numerical computation pipelines.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden** because they hide educational content behind clicks.
2. **Mandatory 1:1 Section Density (Every Roadmap Bullet Gets a Dedicated Section)**: Every single numbered bullet point under a lesson in `numpy.txt` **MUST receive its own dedicated `st.header("X. <Topic Name>", divider="rainbow")` section**. Never compress, merge, or collapse syllabus topics into a single generic overview.
   - Total Sections per lesson file is locked by formula:
     `Total Sections = 1 (Architecture & Memory Layout Blueprint) + K (Roadmap Syllabus Topics) + 1 (Production Implementation) + 1 (Interactive Workbench) + 1 (Comprehensive Technical Interview Hub) + 1 (Self-Assessment Checklist) = K + 5`.
3. **Clean Single Sequential Section Numbering (Strictly Zero Compound Numbering)**: Section headers MUST ALWAYS use clean, single sequential integer numbering from 1 to N (`st.header("1. <Blueprint>", divider="rainbow")`, `st.header("2. <Topic 1 Title>", divider="rainbow")`, ..., `st.header(f"{K+2}. Production Implementation: ...")`). Strictly prohibit compound, dotted, or nested section numbering (NEVER use `st.header("2. 1. ...")`, `st.header("2.1 ...")`, `st.header("3. 2. ...")`).
4. **ABSOLUTELY ZERO LATEX MATHEMATICAL EQUATIONS**:
   - This curriculum is strictly focused on Array Systems Engineering, Computational Memory Models, and Vectorized Data Pipelines.
   - Strictly prohibit ALL LaTeX notation (`$$...$$`, `$x$`).
   - Express all mathematical formulas and array shapes in plain English, code expressions, or explicit dimension tuples:
     - Instead of matrix LaTeX equations: Explain shapes as `Shape: (Rows, Columns)`, `ndim = 2`, and `size = Rows * Columns`.
     - Instead of algebraic summations: Explain reductions along axes as `axis=0 (collapsing across rows, producing shape (Cols,))` or `axis=1 (collapsing across columns, producing shape (Rows,))`.
     - Instead of dot product sigma formulas: Explain `np.dot` as sum of element-wise products along matching inner dimensions `(M, K) @ (K, N) -> (M, N)`.
5. **Modern Engineering & Memory Perspective Contrast**:
   - Every section covering core array manipulations must feature a prominent callout block:
     `### ⚡ Memory & Performance Reality: C-Order Buffer vs Python List Pointers`
   - Explicitly contrast Python Object Overhead vs NumPy C-Contiguous Memory:
     - Python Lists: Array of 64-bit pointers to heap-allocated, boxed `PyObject` integers/floats (huge memory footprint, cache misses, pointer dereferencing).
     - NumPy `ndarray`: Single contiguous byte buffer in RAM, typed elements without boxing, homogeneous byte strides, and SIMD CPU vectorization.
     - View vs Copy: Basic slicing (`arr[1:5]`) creates a lightweight `ndarray` view sharing the same data pointer with adjusted strides and offsets (O(1) time and memory), whereas advanced/fancy indexing (`arr[[0, 2]]`) always allocates a new buffer (O(N) copy).
     - Broadcasting: Virtual expansion of trailing singleton dimensions by setting strides to 0 bytes, executing element-wise operations without duplicating memory.
6. **Heavy Multi-Tier Architectural & Memory Layout Diagrams (Mandatory Top-to-Bottom Flow)**: Every lesson must feature clean Graphviz diagrams with dark theme styling (`bgcolor="#0B1120"`, cluster containers `#1E293B`, edges `#38BDF8`).
   - **Mandatory Top-to-Bottom (`rankdir=TB;`) Orientation**: Strictly prohibit long horizontal `rankdir=LR;` layouts (more than 3–4 sequential nodes). Always use **`rankdir=TB;` (Top to Bottom)** so memory buffers, stride calculations, and array dimensions expand downwards with large, readable fonts (`fontsize=11`) and generous padding.
   - **Mandatory `st.expander` Container for Blueprints**: Wrap architectural diagrams in `with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):` calling `st.graphviz_chart(..., use_container_width=True)`.
   - **Contextual In-Section Diagrams**: Whenever any roadmap topic section covers a multi-step operation, memory layout, indexing pattern, or broadcasting alignment, it **MUST include its own dedicated contextual Graphviz diagram directly inside that section**.
7. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;` instead of bare `&`).
8. **Module Scope for Interactive Workbench Classes & Functions**: Any classes, custom array simulators, or benchmark helper functions needed by the Interactive Workbench must be defined in real Python module scope within the script. Never define executable functions solely inside string literals passed to `st.code(...)`.
9. **Standard Imports**: Exactly 5 standard imports at the top of the file:
   ```python
   import streamlit as st
   import pandas as pd
   import numpy as np
   import graphviz
   from typing import List, Dict, Any, Tuple, Optional, Union
   ```
   Strictly prohibit `import os` and `from pathlib import Path` inside lesson modules.
10. **Dedicated Multi-Expander Technical Interview Hub (Section K+4)**:
    - Organize the rich question taxonomy from `numpy_prompt.md` into **dedicated, collapsible `st.expander` containers** so the page remains uncluttered, interactive, and structured:
      - Expander 1: `📚 Category A: Fundamental & Fresher Core Questions (In-Depth Multi-Sentence Answers)`
      - Expander 2: `🧠 Category B: Intermediate Mechanics, Strides & Memory Layout (Views, Copies & Cache)`
      - Expander 3: `💻 Category C: Code Evaluation & Output Prediction (Shapes, Dtypes & Masking)`
      - Expander 4: `🐛 Category D: Insidious NumPy Bugs & Debugging Challenges (Code Fixes & Explanations)`
      - Expander 5: `⚠️ Category E: Production Traps & Antipatterns (Trap Question, Naive vs Winning Answers)`
      - Expander 6: `🏢 Category F: Production Scenarios & Real-World ML Preprocessing Pipelines`
      - Expander 7: `⚡ Category G: Rapid-Fire Questions & Concept Flashcards`
11. **Self-Assessment Checklist (Section K+5)**:
    - Exactly 6 checkboxes initialized with `value=True`.
    - If all 6 checked: Render `st.success(...)` confirmation banner. **Strictly prohibit `st.balloons()`**.

---

## 2. Complete File Anatomy (Top-to-Bottom Structure)

```
┌────────────────────────────────────────────────────────────────────────┐
│ [A] Standard Imports Block (Exactly 5 Standard Imports)                │
├────────────────────────────────────────────────────────────────────────┤
│ [B] st.set_page_config(page_title=..., page_icon="🔢", layout="wide") │
├────────────────────────────────────────────────────────────────────────┤
│ [C] Title Block: st.title + st.caption + Narrative with 5-Point List   │
├────────────────────────────────────────────────────────────────────────┤
│ [D] First st.divider()                                                 │
├────────────────────────────────────────────────────────────────────────┤
│ [E] Section 1: Memory Layout & Array Architecture Blueprint (Graphviz) │
│     └── Wrapped in st.expander(..., expanded=True)                    │
├────────────────────────────────────────────────────────────────────────┤
│ [F] Sections 2 through K+1: Mandatory 1:1 Syllabus Roadmap Topics      │
│     ├── Deep conceptual explanation (Memory Layout, Strides, Buffers)  │
│     ├── 📊 Contextual Diagram (Memory layout table, broadcasting align)│
│     ├── ⚡ Memory & Performance Reality: Vectorized C vs Python Loops  │
│     ├── Side-by-Side Comparison Matrix (Pandas/List vs NumPy Array)    │
│     └── Interactive Mini-Demo / Code Demonstration                     │
├────────────────────────────────────────────────────────────────────────┤
│ [G] Section K+2: Production Implementation: End-to-End Pipeline        │
│     ├── Real-world runnable Python script (Vectorized calculation)     │
│     └── Explicit memory footprint and execution benchmark comparison   │
├────────────────────────────────────────────────────────────────────────┤
│ [H] Section K+3: Interactive Workbench Simulator                       │
│     ├── Real-time input manipulation widgets (Shape sliders, Dtypes)   │
│     └── Live array buffer inspection, strides calculation, memory size │
├────────────────────────────────────────────────────────────────────────┤
│ [I] Section K+4: Technical Interview Preparation Hub (Multi-Expander)  │
│     ├── Expander A: Fundamental & Fresher Core Questions (In-Depth)    │
│     ├── Expander B: Intermediate Mechanics, Strides & Memory Layout    │
│     ├── Expander C: Code Evaluation & Output Prediction                │
│     ├── Expander D: Insidious NumPy Bugs & Debugging Challenges        │
│     ├── Expander E: Interview Traps (Naive Answer vs Winning Response) │
│     ├── Expander F: Production Scenarios & Real-World ML Data Pipelines│
│     └── Expander G: Rapid-Fire Precision Q&As & Concept Flashcards     │
├────────────────────────────────────────────────────────────────────────┤
│ [J] Section K+5: Self-Assessment Checklist (Mandatory Last Section)    │
│     ├── Exactly 6 Checkboxes (All Initialized with value=True)         │
│     └── if chk1..chk6: -> st.success(...) [Strictly NO balloons]       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Structural Component Code Templates

### 3.1 Standard Imports Block
```python
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
```

### 3.2 Page Configuration (`st.set_page_config`)
```python
st.set_page_config(
    page_title="<X.Y> <Lesson Title> - NumPy Masterclass",
    page_icon="🔢",
    layout="wide"
)
```

### 3.3 Title Block & 5-Pillar Overview
```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: 1. NumPy Foundations")
st.markdown("""
<2–4 sentence high-level narrative detailing the computational mechanism, memory buffer layout, 
and why vectorization and strided indexing are foundational to numerical engineering and machine learning pipelines.>

In this lesson, you will master:
1. <Memory layout, buffer allocation, or stride calculation pillar 1>
2. <Canonical NumPy creation API and type coercion pillar 2>
3. <Computational efficiency: Vectorized SIMD vs naive loop overhead pillar 3>
4. <View vs Copy memory implications during transformation pillar 4>
5. <Production debugging: Dtype overflow, broadcast mismatches, and precision loss 5>
""")
st.divider()
```

### 3.4 Dark-Theme Graphviz Template for Memory Layout Blueprint (Top-to-Bottom Flow)
```python
# Section 1: Architectural Blueprint
st.header("1. Memory Layout and Array Architecture Blueprint", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_memory_layout = """
    digraph NumPyMemoryLayout {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_header {
            label="1. Python ndarray Wrapper Object (Metadata in RAM)";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;
            nd_obj [label="ndarray Object Metadata\\n- Data Pointer: 0x7ffd980\\n- Shape: (3, 4)\\n- Dtype: int64 (8 bytes)\\n- Strides: (32, 8) bytes\\n- Order: C-Contiguous (Row-Major)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_buffer {
            label="2. Contiguous Physical Memory Buffer (Flat C-Array in RAM)";
            bgcolor="#1E293B";
            fontcolor="#10B981";
            color="#334155";
            fontsize=12;
            raw_buffer [shape=plaintext, label=<
                <table border="0" cellborder="1" cellspacing="0" cellpadding="6" bgcolor="#0F172A">
                    <tr><td colspan="4" bgcolor="#064E3B"><font color="#D1FAE5"><b>Row 0 (Stride +0 bytes)</b></font></td>
                        <td colspan="4" bgcolor="#1E3A8A"><font color="#BFDBFE"><b>Row 1 (Stride +32 bytes)</b></font></td>
                        <td colspan="4" bgcolor="#78350F"><font color="#FEF3C7"><b>Row 2 (Stride +64 bytes)</b></font></td></tr>
                    <tr>
                        <td><font color="#38BDF8">Offset 0: [0,0]</font></td>
                        <td><font color="#38BDF8">Offset 8: [0,1]</font></td>
                        <td><font color="#38BDF8">Offset 16: [0,2]</font></td>
                        <td><font color="#38BDF8">Offset 24: [0,3]</font></td>
                        <td><font color="#60A5FA">Offset 32: [1,0]</font></td>
                        <td><font color="#60A5FA">Offset 40: [1,1]</font></td>
                        <td><font color="#60A5FA">Offset 48: [1,2]</font></td>
                        <td><font color="#60A5FA">Offset 56: [1,3]</font></td>
                        <td><font color="#FBBF24">Offset 64: [2,0]</font></td>
                        <td><font color="#FBBF24">Offset 72: [2,1]</font></td>
                        <td><font color="#FBBF24">Offset 80: [2,2]</font></td>
                        <td><font color="#FBBF24">Offset 88: [2,3]</font></td>
                    </tr>
                </table>
            >];
        }

        nd_obj -> raw_buffer [label="Points to Raw Buffer (0x7ffd980)"];
    }
    """
    st.graphviz_chart(dot_memory_layout, use_container_width=True)
```

### 3.5 Dedicated Technical Interview Hub (Section K+4 Multi-Expander Pattern)
```python
# Section K+4: Technical Interview Preparation Hub
st.header(f"{K+4}. Technical Interview Preparation Hub", divider="rainbow")

st.markdown("""
This interview preparation hub organizes high-frequency questions, memory architecture deep-dives, 
code evaluations, and real-world failure modes. Click each expander below to review each category.
""")

with st.expander("📚 Category A: Fundamental & Fresher Core Questions (In-Depth Answers)", expanded=False):
    st.markdown("""
    #### Q1: What is the fundamental difference between a Python list and a NumPy ndarray in memory?
    - **Direct Answer**: A Python list is an array of pointers referencing disjoint heap-allocated Python objects, whereas a NumPy `ndarray` is a single contiguous block of raw memory typed by a uniform `dtype`.
    - **Technical Reasoning**: Because Python list elements are pointers, accessing elements requires double dereferencing and creates cache misses across RAM. In contrast, an `ndarray` stores homogeneous binary numbers sequentially, enabling CPU cache-line prefetching and SIMD (Single Instruction Multiple Data) register vectorization.
    - **Practical Syntax**: `arr = np.array([1, 2, 3], dtype=np.int32)` consumes 12 bytes of payload data, whereas `[1, 2, 3]` in Python consumes over 160 bytes due to `PyObject` headers.
    - **Nuances & Pitfalls**: Mixing types in `np.array([1, "a", 3.0])` forces silent upcasting to string (`<U32`), stripping numerical capabilities.
    """)
    st.divider()

with st.expander("🧠 Category B: Intermediate Mechanics, Strides & Memory Layout (Deep-Dives)", expanded=False):
    st.markdown("""
    #### Q2: What are array strides, and how do they allow zero-copy transpositions and slicing?
    - **Direct Answer**: Strides are a tuple of integers indicating the number of bytes to step in physical memory to advance one element along each dimension.
    - **Technical Reasoning**: For a 2D array of shape `(R, C)` and 8-byte integers in C-order, strides are `(C * 8, 8)`. When performing a transpose `arr.T`, NumPy does not rearrange bytes in RAM; it simply swaps the shape to `(C, R)` and swaps the strides to `(8, C * 8)`. This makes transposition an instantaneous O(1) operation.
    - **Nuances & Pitfalls**: Transposing or non-contiguous slicing changes memory layout from C-contiguous to non-contiguous. Calling operations requiring contiguous memory or passing to C-libraries will force an implicit copy.
    """)
    st.divider()

with st.expander("💻 Category C: Code Evaluation & Output Prediction (5 Code Challenges)", expanded=False):
    st.markdown("""
    #### Challenge 1: Predict Shape and Memory View
    ```python
    x = np.arange(12).reshape(3, 4)
    y = x[:, [1, 2]]
    y[0, 0] = 999
    print(x[0, 1])
    ```
    - **Output**: `1` (The original array `x` remains unchanged!)
    - **Explanation**: Slicing with a list or integer array `[:, [1, 2]]` triggers **Advanced / Fancy Indexing**, which always allocates a brand-new copy in memory. Modifying `y` has zero effect on `x`. Had we used basic slicing `x[:, 1:3]`, it would have returned a view and modified `x`.
    """)
    st.divider()

with st.expander("🐛 Category D: Insidious NumPy Bugs & Debugging Challenges (5 Bug Fixes)", expanded=False):
    st.markdown("""
    #### Bug 1: Silent Integer Overflow
    - **Problem**: In-place summation of large counts unexpectedly yields negative numbers without throwing an exception.
    - **Faulty Code**:
    ```python
    counts = np.array([2000000000, 1000000000], dtype=np.int32)
    total = np.sum(counts)  # Silently overflows to -1294967296
    ```
    - **Corrected Code**:
    ```python
    total = np.sum(counts, dtype=np.int64)  # Explicit 64-bit accumulator
    ```
    - **Root Cause**: NumPy does not raise `OverflowError` during integer arithmetic; it follows C-style bit wrapping. Always specify `dtype=np.int64` on accumulators.
    """)
    st.divider()

with st.expander("⚠️ Category E: Production Traps & Antipatterns (Naive vs Winning Responses)", expanded=False):
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.error("""
        **Trap 1: The Appending Anti-Pattern (`np.append` in a loop)**
        - **Naive Answer**: "I use `np.append(arr, item)` to collect streaming records dynamically."
        - **Trap Unveiled**: `np.append` does not append in-place; it allocates a brand-new array and copies every single element on every iteration, leading to catastrophic O(N^2) time complexity.
        - **Winning Response**: Pre-allocate the entire array buffer with `np.empty((N, D))` if size is known, or collect records in a standard Python list and call `np.array(records)` once at the boundary.
        """)
    with col_t2:
        st.warning("""
        **Trap 2: Assuming NaN Equality**
        - **Naive Answer**: "I check for missing values using `arr == np.nan`."
        - **Trap Unveiled**: By IEEE-754 floating point standard, `NaN != NaN` is always True. Any element compared with `np.nan` evaluates to `False`.
        - **Winning Response**: Always use `np.isnan(arr)` or `np.nan_to_num(arr)` for deterministic missing-value filtering.
        """)

with st.expander("🏢 Category F: Production Scenarios & Real-World ML Preprocessing Pipelines", expanded=False):
    st.markdown("""
    #### Scenario: Memory-Constrained Batch Normalization
    - **Context**: You must normalize a 10-million row by 50-column dataset `(10000000, 50)` with mean 0 and std 1 on a machine with only 8 GB of RAM.
    - **Engineering Strategy**:
      1. An un-vectorized loop is too slow; creating temporary intermediate arrays `(arr - mean) / std` allocates an extra 4 GB copy.
      2. Use in-place universal functions with the `out` parameter:
      ```python
      mean = np.mean(arr, axis=0, keepdims=True)
      std = np.std(arr, axis=0, keepdims=True)
      np.subtract(arr, mean, out=arr)
      np.divide(arr, std, out=arr)
      ```
      3. This executes in-place at full SIMD speed with 0 bytes of extra memory allocation.
    """)

with st.expander("⚡ Category G: Rapid-Fire Questions & Concept Flashcards", expanded=False):
    col_fc1, col_fc2 = st.columns(2)
    with col_fc1:
        with st.container(border=True):
            st.markdown("**Front**: What does `keepdims=True` do in reduction operations (`sum`, `mean`)?")
            st.caption("Flip for answer:")
            st.info("**Back**: It preserves the reduced axes with size 1 (e.g. reducing shape (3, 4) with axis=1 yields (3, 1) instead of (3,)), enabling seamless immediate broadcasting against the original array without manual reshaping.")
    with col_fc2:
        with st.container(border=True):
            st.markdown("**Front**: What is the computational difference between `flatten()` and `ravel()`?")
            st.caption("Flip for answer:")
            st.info("**Back**: `flatten()` always returns a newly allocated 1D copy in memory. `ravel()` returns a contiguous 1D view whenever possible, only allocating a copy if the source array has non-contiguous strides.")
```

### 3.6 Self-Assessment Checklist (Section K+5 Template)
```python
# Section K+5: Self-Assessment Checklist
st.header(f"{K+5}. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the contiguous C-order vs Fortran-order memory layout and byte strides.", value=True)
c2 = st.checkbox("I can differentiate between basic slicing (views) and fancy indexing (copies).", value=True)
c3 = st.checkbox("I know how NumPy implements broadcasting without physical memory allocation.", value=True)
c4 = st.checkbox("I can identify and prevent integer overflow and silent upcasting in numeric pipelines.", value=True)
c5 = st.checkbox("I can implement memory-efficient in-place computations using ufunc out parameters.", value=True)
c6 = st.checkbox("I can articulate winning technical interview answers to memory and performance traps.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to the next NumPy module.")
```

---

## 4. Master Multi-Page Navigation Structure (`app.py`)

All generated `.py` lesson files reside under `01_Data_Science_Foundations/1. Numpy/webapp/` and are automatically discovered by the top-level `app.py`:

```python
# File: 01_Data_Science_Foundations/1. Numpy/webapp/app.py
import streamlit as st
import re
from pathlib import Path

st.set_page_config(
    page_title="NumPy Master Technical Interview Curriculum",
    page_icon="🔢",
    layout="wide"
)

def natural_key(string_):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', str(string_))]

def scan_numpy_lessons():
    """Scans all chapter directories and .py lesson files in the webapp directory."""
    current_dir = Path(__file__).parent
    chapters = [d for d in current_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))]
    chapters.sort(key=lambda p: natural_key(p.name))
    
    nav_dict = {}
    for chap in chapters:
        py_files = [f for f in chap.glob("*.py") if not f.name.startswith("__") and f.stat().st_size > 0]
        py_files.sort(key=lambda f: natural_key(f.name))
        if py_files:
            pages = []
            for pf in py_files:
                pages.append(st.Page(str(pf.relative_to(current_dir)), title=pf.stem))
            nav_dict[chap.name] = pages
            
    return nav_dict

nav_structure = scan_numpy_lessons()

if nav_structure:
    pg = st.navigation(nav_structure)
    pg.run()
else:
    st.info("NumPy Technical Curriculum ready. Add lesson files in webapp/<Chapter>/<Lesson>.py")
```

---

## 5. Standard Prompt to Generate Any NumPy Lesson `.py` File

When generating any lesson file (e.g. from Section 2 onwards in `numpy.txt`), use this exact prompt:

```text
You are an expert NumPy Systems Engineer and Technical Interview Coach.

Generate a complete, self-contained, and runnable Streamlit Python webapp file (.py) following the master standard in:
`01_Data_Science_Foundations/1. Numpy/UNIVERSAL_LESSON_FORMAT.md`.

INPUT:
CHAPTER: [e.g. 2. Creating Arrays]
LESSON: [e.g. 2.1 Array Creation Methods]
SYLLABUS TOPICS:
1. Creating arrays from Python lists
2. Using np.array()
3. Creating 1D arrays
4. Creating 2D arrays
5. Creating 3D arrays

RULES:
1. Section count formula: K + 5 sections (where K is the number of syllabus topics).
   Section 1: Architecture Blueprint (Graphviz Top-to-Bottom in expander)
   Sections 2 to K+1: Mandatory 1:1 Syllabus Topics (Every topic gets a dedicated st.header(..., divider="rainbow"))
   Section K+2: Production Implementation (Vectorized benchmark vs Python loop)
   Section K+3: Interactive Workbench Simulator (Live array shape/strides widget)
   Section K+4: Technical Interview Preparation Hub (Multi-Expander with Categories A to G)
   Section K+5: Self-Assessment Checklist (6 checkboxes + st.success)
2. Single sequential integer numbering (1, 2, ..., K+5). Strictly zero compound numbers.
3. Zero LaTeX math syntax ($ or $$). Plain English and shape tuples only.
4. Exactly 5 standard imports (streamlit, pandas, numpy, graphviz, typing). Zero os/pathlib.
5. All interview categories in Section K+4 must be placed in separate, dedicated st.expander containers.
```
