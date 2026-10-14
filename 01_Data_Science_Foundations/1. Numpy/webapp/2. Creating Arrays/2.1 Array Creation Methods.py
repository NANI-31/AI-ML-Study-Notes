import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# Page Configuration
st.set_page_config(
    page_title="2.1 Array Creation Methods - NumPy Technical Interview Masterclass",
    page_icon="🎯",
    layout="wide"
)

# Title Block & Interview Focus Header
st.title("2.1 Array Creation Methods: Technical Interview Masterclass")
st.caption("Chapter 2: Creating Arrays | Domain: 1. NumPy Systems & Interview Engineering")
st.markdown("""
This masterclass module prepares candidates for senior-level technical interviews on **Array Creation Methods**.
Rather than an elementary syntax reference or a raw text dump, this application translates the comprehensive 
study syllabus into an interactive, systems-level masterclass. It analyzes the physical memory transformation from 
Python list object graphs to C-contiguous buffers, automatic type inference hierarchies, multi-axis stride mechanics, 
output predictions, live debugging drills, and production zero-copy techniques.

### 🎯 Key Interview Competencies Evaluated:
1. **Memory Conversion Mechanics**: Unpacking heap-allocated `PyObject` pointer graphs into flat C-contiguous buffers.
2. **`np.array()` Parameter Control**: Mastering `dtype`, `copy`, `order`, and `ndmin` to avoid silent performance bottlenecks.
3. **Multi-Dimensional Geometry**: Formulating shapes and byte strides for 1D, 2D, and 3D arrays.
4. **Type Promotion & Safety**: Preventing silent string upcasting and handling jagged/ragged sequence errors.
5. **Interview Traps**: Articulating staff-level trade-offs between `np.array()` and zero-copy `np.asarray()`.
""")
st.divider()

# Section 1: Memory Layout and Systems Architecture Blueprint
st.header("1. Memory Layout and Systems Architecture Blueprint", divider="rainbow")

with st.expander("🔍 View Complete Memory Architecture Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_memory = """
    digraph ArrayCreationMemoryModel {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_python_list {
            label="1. Python Nested List in RAM (Dispersed Pointer Graph)";
            bgcolor="#1E293B";
            fontcolor="#F87171";
            color="#334155";
            fontsize=12;

            list_outer [label="Outer list Object (size=2)\\n[ Ptr to Row 0 | Ptr to Row 1 ]\\n(Header: 56 bytes + 16 bytes ptrs)", fillcolor="#0F172A", fontcolor="#FCA5A5", color="#EF4444"];
            
            row0 [label="Inner list 0 (size=3)\\n[ Ptr to 10 | Ptr to 20 | Ptr to 30 ]", fillcolor="#0F172A", fontcolor="#FCA5A5", color="#EF4444"];
            row1 [label="Inner list 1 (size=3)\\n[ Ptr to 40 | Ptr to 50 | Ptr to 60 ]", fillcolor="#0F172A", fontcolor="#FCA5A5", color="#EF4444"];

            obj10 [label="PyLongObject (10)\\n28 bytes heap", fillcolor="#182234", fontcolor="#CBD5E1", color="#64748B"];
            obj20 [label="PyLongObject (20)\\n28 bytes heap", fillcolor="#182234", fontcolor="#CBD5E1", color="#64748B"];
            obj30 [label="PyLongObject (30)\\n28 bytes heap", fillcolor="#182234", fontcolor="#CBD5E1", color="#64748B"];
            obj40 [label="PyLongObject (40)\\n28 bytes heap", fillcolor="#182234", fontcolor="#CBD5E1", color="#64748B"];
            obj50 [label="PyLongObject (50)\\n28 bytes heap", fillcolor="#182234", fontcolor="#CBD5E1", color="#64748B"];
            obj60 [label="PyLongObject (60)\\n28 bytes heap", fillcolor="#182234", fontcolor="#CBD5E1", color="#64748B"];

            list_outer -> row0 [label="deref index 0"];
            list_outer -> row1 [label="deref index 1"];
            row0 -> obj10; row0 -> obj20; row0 -> obj30;
            row1 -> obj40; row1 -> obj50; row1 -> obj60;
        }

        conversion [label="np.array(nested_list, dtype=np.int64)\\n1. Traverse dimensions to determine shape=(2, 3)\\n2. Allocate single contiguous C-buffer (6 elements * 8 bytes = 48 bytes)\\n3. Unpack primitive C-longs from PyLongObjects and copy sequentially into buffer", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#10B981", shape=record];

        subgraph cluster_numpy_array {
            label="2. Resulting NumPy ndarray (Compact Flat C-Buffer)";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            color="#334155";
            fontsize=12;

            meta [label="PyArrayObject Metadata (Header in RAM)\\n- data_pointer: 0x7ffd980\\n- ndim: 2\\n- shape: (2, 3)\\n- dtype: int64 (8 bytes)\\n- strides: (24, 8) bytes\\n- flags: C_CONTIGUOUS=True, OWNDATA=True", fillcolor="#0F172A", fontcolor="#38BDF8", color="#0284C7"];

            buf [shape=plaintext, label=<
                <table border="0" cellborder="1" cellspacing="0" cellpadding="6" bgcolor="#0F172A">
                    <tr><td colspan="3" bgcolor="#064E3B"><font color="#D1FAE5"><b>Row 0 (Base Offset +0 bytes)</b></font></td>
                        <td colspan="3" bgcolor="#1E3A8A"><font color="#BFDBFE"><b>Row 1 (Base Offset +24 bytes)</b></font></td></tr>
                    <tr>
                        <td><font color="#38BDF8">Offset +0B: 10</font></td>
                        <td><font color="#38BDF8">Offset +8B: 20</font></td>
                        <td><font color="#38BDF8">Offset +16B: 30</font></td>
                        <td><font color="#60A5FA">Offset +24B: 40</font></td>
                        <td><font color="#60A5FA">Offset +32B: 50</font></td>
                        <td><font color="#60A5FA">Offset +40B: 60</font></td>
                    </tr>
                </table>
            >];
        }

        list_outer -> conversion [label="input Python list graph"];
        conversion -> meta [label="initializes metadata"];
        meta -> buf [label="data pointer references raw C-buffer"];
    }
    """
    st.graphviz_chart(dot_memory, use_container_width=True)

    st.markdown("""
    **Architectural Breakdown**:
    - **Python List Overhead**: Every element in a Python list requires an independent heap-allocated `PyLongObject` (28 bytes) plus an 8-byte pointer inside the list's pointer array. A list of 1,000,000 integers consumes ~36 MB of fragmented memory.
    - **NumPy Contiguous Buffer**: Converting via `np.array()` packs raw C primitives (`int64` = 8 bytes) sequentially into a flat memory block of exactly 8 MB.
    - **Cache Locality & SIMD**: Because numbers are placed sequentially in RAM, a CPU cache line (64 bytes) fetches 8 consecutive 64-bit numbers in a single memory access cycle, enabling vectorized SIMD (Single Instruction Multiple Data) register execution.
    """)

# Section 2: High-Frequency Core Interview Questions
st.header("2. High-Frequency Core Interview Questions", divider="rainbow")

with st.expander("📚 Category A: Python Lists to NumPy Arrays & Memory Conversion", expanded=True):
    st.markdown("""
    #### Q1: What exact low-level operations occur when converting a nested Python list to a NumPy array?
    - **Direct Interview Answer**: NumPy traverses the nested list structure to validate dimension uniformity, infers a homogeneous `dtype`, allocates a single contiguous C-memory buffer sized to `size * itemsize`, dereferences each Python object pointer to extract its raw primitive C value, copies the values into the buffer, and attaches an `ndarray` metadata header with calculated shapes and strides.
    - **Under-the-Hood Technical Reasoning**: Python lists store 64-bit pointers to arbitrary heap objects. An integer in Python 3 is a `PyLongObject` requiring 28 bytes of overhead (reference count, type pointer, digit size, digit array). Converting a list of 1,000,000 integers (`~36 MB` with pointers and objects) to an `int64` NumPy array condenses the data into a single 8 MB contiguous buffer (`1_000_000 * 8 bytes`). This eliminates pointer indirection and enables hardware SIMD registers to process multiple values per cycle.
    - **Concrete Code Demonstration**:
    ```python
    import sys
    py_list = [10, 20, 30]
    np_arr = np.array(py_list, dtype=np.int64)

    list_ptr_size = sys.getsizeof(py_list)  # ~88 bytes (pointers only)
    array_payload = np_arr.nbytes          # Exactly 24 bytes (3 * 8 bytes)
    ```
    - **Nuance & Pitfalls**: If the nested list contains ragged sequences (e.g. `[[1, 2], [3]]`), modern NumPy (v1.24+) immediately raises `ValueError: setting an array element with a sequence`. Older versions silently fell back to an array of `dtype=object`, which preserves pointers and entirely negates vectorization.
    """)
    st.divider()

    st.markdown("""
    #### Q2: Why is `np.array()` on a massive Python list slower than native NumPy allocators like `np.empty()` or `np.zeros()`?
    - **Direct Interview Answer**: `np.array(py_list)` must iterate through every element in the Python list, unwrap each `PyObject` scalar wrapper via C-level type dispatching, and copy the bytes into memory. In contrast, `np.empty()` and `np.zeros()` perform a single `malloc()` or `calloc()` system call directly in C without touching Python objects.
    - **Under-the-Hood Technical Reasoning**: Unpacking 10,000,000 Python objects incurs 10,000,000 pointer dereferences across fragmented heap memory, causing frequent L1/L2 cache misses. Native NumPy allocators bypass the Python interpreter and Python memory allocator (`PyMalloc`) entirely, asking the OS kernel directly for a chunk of contiguous memory.
    - **Concrete Code Demonstration**:
    ```python
    # Slow: creates 1M Python objects, then unwraps them
    arr_from_list = np.array([i for i in range(1_000_000)])  # ~80-100 ms

    # Fast: direct C allocation and sequence generation
    arr_direct = np.arange(1_000_000)                        # ~1.5 ms (60x faster!)
    ```
    - **Nuance & Pitfalls**: Freshers often write `np.array([0] * 1_000_000)` instead of `np.zeros(1_000_000)`. Always advise pre-allocating with `np.zeros` or `np.empty` in performance-critical code.
    """)

with st.expander("📚 Category B: `np.array()` Parameter Mechanics (dtype, copy, order, ndmin)", expanded=False):
    st.markdown("""
    #### Q3: What are the key parameters of `np.array()` and how do they impact performance and memory?
    - **Direct Interview Answer**: The primary parameters are `object` (input sequence), `dtype` (target primitive type), `copy` (whether to force a buffer copy), `order` (`'C'` for row-major, `'F'` for column-major), and `ndmin` (minimum dimensions required).
    - **Under-the-Hood Technical Reasoning**:
      - `dtype`: If omitted, NumPy scans all elements to infer the lowest common denominator type using the Type Promotion Hierarchy. Explicitly passing `dtype=np.float32` skips scanning ambiguity and guarantees exact precision.
      - `copy`: Defaults to `True`. When passing an existing `ndarray`, `np.array(existing, copy=False)` avoids allocating a new buffer *only if* the dtype and order already match. If a dtype conversion is required, a copy is made regardless.
      - `order`: Controls whether contiguous rows (`'C'`) or contiguous columns (`'F'`) are laid out sequentially in RAM, directly impacting cache prefetching during strided traversal.
      - `ndmin`: Forces lower-dimensional data to have prepended singleton dimensions (e.g., `ndmin=2` converts a 1D vector of shape `(5,)` into a 2D row matrix `(1, 5)`).
    - **Concrete Code Demonstration**:
    ```python
    x = np.array([1, 2, 3], dtype=np.int32)
    # copy=False reuses memory if compatible
    y = np.array(x, copy=False)
    assert y.base is x or y is x

    # ndmin ensures matrix rank for linear algebra pipelines
    z = np.array([1, 2, 3], ndmin=2)
    assert z.shape == (1, 3)
    ```
    - **Nuance & Pitfalls**: In NumPy 2.0+, `copy=False` raises a `ValueError` if a copy cannot be avoided. To safely create a view or borrow memory without raising errors, interview candidates must recommend `np.asarray()`.
    """)
    st.divider()

    st.markdown("""
    #### Q4: How does `order='C'` versus `order='F'` affect memory layout and downstream calculation speed?
    - **Direct Interview Answer**: `order='C'` stores elements in row-major order (last index varies fastest in memory), matching C and Python conventions. `order='F'` stores elements in column-major order (first index varies fastest in memory), matching Fortran and MATLAB conventions.
    - **Under-the-Hood Technical Reasoning**: For a 2D array of shape `(M, N)`:
      - In C-order: elements `arr[i, j]` and `arr[i, j+1]` are adjacent in RAM (`stride = itemsize`). Iterating across rows utilizes CPU cache lines with 100% efficiency.
      - In Fortran-order: elements `arr[i, j]` and `arr[i+1, j]` are adjacent in RAM. Iterating down columns is cache-friendly, whereas row-wise iteration triggers continuous cache-line misses.
    - **Concrete Code Demonstration**:
    ```python
    a_c = np.array([[1, 2], [3, 4]], order='C')  # strides: (16, 8) bytes (float64)
    a_f = np.array([[1, 2], [3, 4]], order='F')  # strides: (8, 16) bytes (float64)

    assert a_c.flags.c_contiguous is True
    assert a_f.flags.f_contiguous is True
    ```
    - **Nuance & Pitfalls**: Passing a Fortran-ordered array to C-based external libraries (e.g., OpenCV, PyTorch CUDA kernels) often triggers an implicit, silent copy to restore C-contiguity, causing hidden performance bottlenecks.
    """)

with st.expander("📚 Category C: Multi-Dimensional Geometry & Strides (1D vs 2D vs 3D)", expanded=False):
    st.markdown("""
    #### Q5: What is the architectural difference between a 1D Rank-1 array, a 2D Row vector, and a 2D Column vector?
    - **Direct Interview Answer**: A 1D array has `shape=(N,)` and 1 stride entry `(itemsize,)`. A 2D row vector has `shape=(1, N)` and strides `(N * itemsize, itemsize)`. A 2D column vector has `shape=(N, 1)` and strides `(itemsize, itemsize)`.
    - **Under-the-Hood Technical Reasoning**: While all three occupy identical contiguous byte counts in memory (`N * itemsize`), their broadcasting behavior during matrix operations differs radically:
      - Transposing a 1D array `arr.T` is a no-op that returns the identical 1D array because reversing a 1-tuple `(N,)` yields `(N,)`.
      - Transposing a row vector `(1, N)` swaps shape and strides to produce a column vector `(N, 1)`.
      - Outer products between `(N, 1)` and `(1, N)` broadcast into an `(N, N)` matrix, whereas operating between two 1D arrays `(N,)` executes an element-wise reduction or vector operation.
    - **Concrete Code Demonstration**:
    ```python
    v1 = np.array([1, 2, 3])       # shape: (3,),    strides: (8,)
    v_row = v1.reshape(1, 3)       # shape: (1, 3), strides: (24, 8)
    v_col = v1.reshape(3, 1)       # shape: (3, 1), strides: (8, 8)

    assert v1.T.shape == (3,)      # 1D transpose is a NO-OP!
    assert v_row.T.shape == (3, 1) # 2D row becomes column
    ```
    - **Nuance & Pitfalls**: Candidates often mistakenly write `v.T` on a 1D array expecting a column vector for matrix multiplication. In senior interviews, explain that `v[:, np.newaxis]` or `v.reshape(-1, 1)` is mandatory to create a column vector.
    """)
    st.divider()

    st.markdown("""
    #### Q6: How are 3D arrays structured in memory and how do you interpret their axes in ML pipelines?
    - **Direct Interview Answer**: A 3D array has `shape=(dim0, dim1, dim2)`. In C-contiguous order, memory is flattened such that `dim2` varies fastest (consecutive bytes), followed by `dim1`, then `dim0`. In ML, this typically represents `(batch_size, sequence_length, feature_dim)` in NLP, or `(batch_size, height, width)` in computer vision grayscale batches.
    - **Under-the-Hood Technical Reasoning**: Strides for a 3D array of shape `(D0, D1, D2)` with itemsize `S` are:
      - `stride[0] = D1 * D2 * S` (bytes to jump to the next 2D slice/channel).
      - `stride[1] = D2 * S` (bytes to jump to the next row within a slice).
      - `stride[2] = S` (bytes to jump to the next element within a row).
      Memory access along axis 2 hits consecutive cache lines with 100% L1 cache efficiency, whereas jumping across axis 0 incurs large pointer jumps.
    - **Concrete Code Demonstration**:
    ```python
    tensor_3d = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]]
    ], dtype=np.int32)

    # Shape: (3, 2, 2), itemsize = 4 bytes
    # stride[0] = 2 * 2 * 4 = 16 bytes
    # stride[1] = 2 * 4 = 8 bytes
    # stride[2] = 4 bytes
    assert tensor_3d.strides == (16, 8, 4)
    ```
    - **Nuance & Pitfalls**: In image processing, PyTorch uses Channel-First `(N, C, H, W)` while TensorFlow uses Channel-Last `(N, H, W, C)`. Transposing axes via `arr.transpose(0, 2, 3, 1)` changes the shape and strides without copying data, creating a non-contiguous array.
    """)

with st.expander("📚 Category D: Edge Cases, Dtype Casting & Ragged Sequence Prevention", expanded=False):
    st.markdown("""
    #### Q7: What occurs when elements in a Python list exceed the numeric bounds of an explicitly supplied integer dtype?
    - **Direct Interview Answer**: In modern NumPy, passing values exceeding the bounds of a fixed-width integer dtype (e.g. 300 into `np.uint8`, which maxes out at 255) triggers C-style integer overflow wrap-around modulo $2^{\\text{bits}}$, producing silent data corruption unless overflow checking is explicitly triggered.
    - **Under-the-Hood Technical Reasoning**: Fixed-width C types truncate the higher-order bits. For `uint8`, the formula is `value % 256`. Thus `300 % 256 = 44`. An image pixel intended to be bright white (255) silently wraps around to dark gray (44).
    - **Concrete Code Demonstration**:
    ```python
    overflow_arr = np.array([250, 255, 256, 300], dtype=np.uint8)
    # Output: array([250, 255,   0,  44], dtype=uint8)
    assert overflow_arr[2] == 0
    assert overflow_arr[3] == 44
    ```
    - **Nuance & Pitfalls**: In Python, standard integers have arbitrary precision and never overflow. Freshers often assume NumPy numbers inherit Python's overflow protection. Always highlight that NumPy operates at bare-metal C fixed precision.
    """)
    st.divider()

    st.markdown("""
    #### Q8: What happens when an empty Python list `[]` is passed to `np.array()`?
    - **Direct Interview Answer**: It creates an empty 1D array of shape `(0,)` and default `dtype=float64` (consuming 0 bytes of buffer payload, with a standard ~112-byte `ndarray` metadata header).
    - **Under-the-Hood Technical Reasoning**: Because there are zero elements from which to infer a data type, NumPy defaults to its standard numeric float type (`np.float64`).
    - **Concrete Code Demonstration**:
    ```python
    empty_arr = np.array([])
    assert empty_arr.shape == (0,)
    assert empty_arr.dtype == np.float64
    assert empty_arr.size == 0
    assert empty_arr.nbytes == 0
    ```
    - **Nuance & Pitfalls**: If you subsequently attempt to concatenate integer arrays with `np.array([])`, the entire concatenated result will be silently upcast to `float64`! Always initialize empty accumulators with explicit dtypes: `np.array([], dtype=np.int64)`.
    """)

# Section 3: Memory Mechanics, Strides and Advanced Internals
st.header("3. Memory Mechanics, Strides and Advanced Internals", divider="rainbow")

with st.expander("🧠 Expand to Review Strides, Memory Layout & View Mechanics", expanded=False):
    st.markdown("""
    ### 🔬 Deep Dive: Memory Pointer Arithmetic Across Dimensions

    Every element in an N-dimensional array is resolved via direct C pointer arithmetic without any hash lookups or pointer chasing:
    """)
    
    st.code("""
Memory Address Formula:
Address(arr[i, j, k]) = Data_Pointer + (i * stride[0]) + (j * stride[1]) + (k * stride[2])
    """, language="text")

    st.markdown("""
    #### 1. The Stride Transformation Matrix
    Consider an array created with `np.array` with shape `(2, 3, 4)` and `dtype=float64` (itemsize = 8 bytes):
    - **Total Elements**: `2 * 3 * 4 = 24 elements`.
    - **Total Memory Allocated**: `24 * 8 = 192 contiguous bytes`.
    - **Row-Major (C-Order) Strides**:
      - `stride[2] = 8 bytes` (1 element step along width).
      - `stride[1] = 4 * 8 = 32 bytes` (1 row step along height).
      - `stride[0] = 3 * 4 * 8 = 96 bytes` (1 slice step along depth).
      - Strides tuple: `(96, 32, 8)`.

    #### 2. The Comprehensive API Comparison Matrix
    This matrix settles one of the most common senior technical interview points of confusion:
    """)

    df_api = pd.DataFrame({
        "Function": ["np.array()", "np.asarray()", "np.ascontiguousarray()", "np.ndarray()"],
        "Input: Python List": ["Allocates new C-buffer & copies", "Allocates new C-buffer & copies", "Allocates new C-buffer & copies", "Requires tuple shape; allocates uninitialized RAM"],
        "Input: ndarray (matching dtype & order)": ["Forces memory duplicate (copy=True)", "Returns original array directly (Zero Copy)", "Returns original array directly (Zero Copy)", "N/A - Direct low-level class constructor"],
        "Input: Non-contiguous ndarray": ["Creates contiguous copy", "Returns non-contiguous view", "Forces memory copy to restore C-order", "N/A"],
        "Primary Production Use Case": ["Guaranteed defensive isolation", "Zero-copy function parameter ingestion", "Sanitizing arrays before C/CUDA kernel calls", "Internal NumPy C-API binding development"]
    })
    st.dataframe(df_api, use_container_width=True)

    st.markdown("""
    #### 3. Automatic Type Promotion Lattice
    When passing mixed types to `np.array()`, NumPy determines the resulting array dtype using the strict promotion lattice:
    `bool -> int8/16/32/64 -> float16/32/64 -> complex64/128 -> string (Unicode)`
    
    If even a single string literal exists in a list of 1,000,000 floats:
    `np.array([1.5, 2.8, "3.0"])` -> `dtype='<U32'` (all elements silently converted to 32-character Unicode strings).
    """)

# Section 4: Code Evaluation and Output Prediction Challenges
st.header("4. Code Evaluation and Output Prediction Challenges", divider="rainbow")

with st.expander("💻 Expand to Test Your Code Prediction & Dry-Run Skills", expanded=False):
    st.markdown("""
    #### Challenge 1: The Transpose on 1D Array Mutation Trap
    Examine the code snippet below. What does it print?
    ```python
    x = np.array([10, 20, 30])
    y = x.T
    y[0] = 999

    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    print("x[0]:", x[0])
    ```
    """)
    
    with st.expander("🔍 Reveal Output & Step-by-Step Dry Run"):
        st.code("""
Output:
x.shape: (3,)
y.shape: (3,)
x[0]: 999
        """, language="text")
        st.markdown("""
        **Dry Run Explanation**:
        1. `x = np.array([10, 20, 30])` creates a 1D array of shape `(3,)`.
        2. `x.T` attempts to transpose `x`. For a 1D array with shape tuple `(3,)`, reversing the shape tuple produces `(3,)`. Transposing a 1D array is a zero-op that returns `x` itself (a view of the same buffer).
        3. `y[0] = 999` mutates the underlying buffer. Because `y` references the exact same buffer as `x`, `x[0]` is mutated to `999`.
        4. To obtain a true 2D column vector with shape `(3, 1)`, the candidate must use `x.reshape(-1, 1)` or `x[:, np.newaxis]`.
        """)

    st.divider()

    st.markdown("""
    #### Challenge 2: Subtle Type Promotion and Memory Upcasting
    Predict the output, dtype, and itemsize of the array created below:
    ```python
    mixed = np.array([True, 2, 3.5])
    print("dtype:", mixed.dtype)
    print("elements:", mixed)
    print("itemsize:", mixed.itemsize)
    ```
    """)

    with st.expander("🔍 Reveal Output & Step-by-Step Dry Run"):
        st.code("""
Output:
dtype: float64
elements: [1.  2.  3.5]
itemsize: 8
        """, language="text")
        st.markdown("""
        **Dry Run Explanation**:
        1. The list contains a `bool` (`True`), an `int` (`2`), and a `float` (`3.5`).
        2. Following NumPy's type promotion hierarchy, `bool` is promoted to `int`, and `int` is promoted to `float`.
        3. By default on 64-bit architectures, floating-point promotion defaults to `np.float64`.
        4. `True` becomes `1.0`, `2` becomes `2.0`, and each element occupies 8 bytes (`itemsize = 8`).
        """)

    st.divider()

    st.markdown("""
    #### Challenge 3: Inhomogeneous / Ragged Sequences in Modern NumPy
    Predict what happens when running this code in NumPy 1.24+:
    ```python
    jagged_list = [[1, 2, 3], [4, 5]]
    arr = np.array(jagged_list)
    ```
    """)

    with st.expander("🔍 Reveal Output & Step-by-Step Dry Run"):
        st.code("""
Output:
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.
        """, language="text")
        st.markdown("""
        **Dry Run Explanation**:
        1. Legacy NumPy (pre-1.20) silently created an array of shape `(2,)` with `dtype=object` storing Python list pointers.
        2. Modern NumPy strictly disallows automatic fallback to ragged arrays to prevent silent vectorization failures and subtle downstream production bugs.
        3. If ragged data is genuinely required, candidate must pass `dtype=object` explicitly: `np.array(jagged_list, dtype=object)`.
        """)

    st.divider()

    st.markdown("""
    #### Challenge 4: Memory Downcasting with Integer Overflow Wrap-around
    Predict the values inside the resulting array:
    ```python
    raw_ints = [250, 255, 256, 300]
    u8_arr = np.array(raw_ints, dtype=np.uint8)
    print(u8_arr)
    ```
    """)

    with st.expander("🔍 Reveal Output & Step-by-Step Dry Run"):
        st.code("""
Output:
[250 255   0  44]
        """, language="text")
        st.markdown("""
        **Dry Run Explanation**:
        1. `uint8` allocates 1 byte (8 bits) per element, supporting unsigned values in the range `[0, 255]`.
        2. Values `250` and `255` fit within range and remain unchanged.
        3. `256` in binary is `1 0000 0000`. Truncating to the lower 8 bits yields `0000 0000` (`0`).
        4. `300` in binary is `1 0010 1100`. Truncating to the lower 8 bits yields `0010 1100` (`44`).
        """)

# Section 5: Insidious Bugs and Live Debugging Drills
st.header("5. Insidious Bugs and Live Debugging Drills", divider="rainbow")

with st.expander("🐛 Expand to Inspect Faulty Code Snippets and Root-Cause Bug Fixes", expanded=False):
    st.markdown("""
    #### Bug 1: The Ragged Nested List Crash in Modern NumPy
    - **Symptom**: Code that used to run on legacy NumPy (v1.18) crashes in production after upgrading to NumPy 1.24+ with `ValueError: setting an array element with a sequence`.
    - **Faulty Code**:
    ```python
    data = [[1, 2, 3], [4, 5]]
    arr = np.array(data)  # Crashes in modern NumPy!
    ```
    - **Root Cause**: Earlier NumPy versions silently created an array of `dtype=object` containing pointers to the inner Python lists. Modern NumPy forbids this implicit fallback to prevent silent vectorization failures.
    - **Corrected Code**:
    ```python
    # Option A: If rectangular padding is required (Standard ML Pipeline)
    max_len = max(len(row) for row in data)
    padded = [row + [0] * (max_len - len(row)) for row in data]
    arr_padded = np.array(padded, dtype=np.int64)

    # Option B: If explicitly intending an array of object pointers
    arr_obj = np.array(data, dtype=object)
    ```
    """)
    st.divider()

    st.markdown("""
    #### Bug 2: Assuming `np.array()` from a List Creates a Memory View
    - **Symptom**: A developer modifies elements in the original Python list expecting the NumPy array to reflect the changes, but the array remains unchanged.
    - **Faulty Code**:
    ```python
    raw_coords = [10.0, 20.0, 30.0]
    coord_arr = np.array(raw_coords)

    # Developer updates raw coordinates
    raw_coords[0] = 999.0

    # Silent Bug: coord_arr[0] is still 10.0!
    assert coord_arr[0] == 999.0  # AssertionError!
    ```
    - **Root Cause**: Converting a Python list to a NumPy array ALWAYS copies values into a newly allocated contiguous memory buffer. Python lists and NumPy arrays cannot share memory buffers because Python lists store pointers to heap objects, whereas NumPy arrays store flat primitive bytes.
    - **Corrected Code**:
    ```python
    # Work directly within the NumPy buffer from the start
    coord_arr = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    coord_arr[0] = 999.0  # Directly modifies the C-buffer
    ```
    """)
    st.divider()

    st.markdown("""
    #### Bug 3: Accidental String Pollution During Dataset Ingestion
    - **Symptom**: Performing arithmetic on an ingested numeric dataset raises `TypeError: ufunc 'add' did not contain a loop with signature matching types`.
    - **Faulty Code**:
    ```python
    raw_samples = [1.2, 4.5, "NA", 8.9]
    arr = np.array(raw_samples)
    result = arr * 2.0  # TypeError!
    ```
    - **Root Cause**: Because `"NA"` is a string, NumPy's type promotion converts the entire array to Unicode string `dtype='<U32'`. Universal functions (`ufuncs`) like multiplication cannot execute on string arrays.
    - **Corrected Code**:
    ```python
    # Clean or cast missing values to np.nan before conversion
    cleaned_samples = [np.nan if x == "NA" else float(x) for x in raw_samples]
    arr = np.array(cleaned_samples, dtype=np.float64)
    result = arr * 2.0  # Valid vectorized operation!
    ```
    """)
    st.divider()

    st.markdown("""
    #### Bug 4: Memory Downcasting Overflow Silent Data Corruption
    - **Symptom**: Normalized image batches scaled between 0 and 255 become distorted and black when serialized to 8-bit unsigned integer arrays.
    - **Faulty Code**:
    ```python
    pixel_vals = [254.0, 255.0, 256.0, 310.0]
    img_arr = np.array(pixel_vals, dtype=np.uint8)  # pixel 256 wraps to 0!
    ```
    - **Root Cause**: Downcasting floating-point values exceeding 255 to `uint8` triggers modulo wrap-around without raising any warning.
    - **Corrected Code**:
    ```python
    clipped_vals = np.clip(pixel_vals, 0, 255)
    img_arr = np.array(clipped_vals, dtype=np.uint8)
    ```
    """)

# Section 6: Interview Traps and Antipatterns
st.header("6. Interview Traps and Antipatterns", divider="rainbow")

with st.expander("⚠️ Expand to Review Common Interview Traps (Naive vs Winning Responses)", expanded=False):
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.error("""
        **Trap 1: Repeated `np.array()` or `np.append()` Inside Loops**
        - **Interviewer Question**: "How would you build a 2D matrix of embeddings as items arrive from a streaming generator?"
        - **Naive Candidate Answer**: "I initialize an empty array `arr = np.array([])` and call `arr = np.append(arr, new_embedding, axis=0)` for every incoming item."
        - **Trap Unveiled**: `np.append` is NOT `O(1)` like Python's `list.append()`. It allocates a brand-new contiguous array and copies all existing elements on every call, degrading runtime to catastrophic `O(N^2)` time and causing heap fragmentation.
        - **Staff-Level Winning Response**: "If the total count is known, pre-allocate the memory block with `np.empty((total_count, embedding_dim), dtype=np.float32)` and assign slices by index `arr[i] = emb`. If the count is unknown, append embeddings to a standard Python list (which amortizes `O(1)` pointer append), and perform a single `np.array(list_of_embeddings)` conversion upon completion."
        """)
    with col_t2:
        st.warning("""
        **Trap 2: Redundant Defensive Copying with `np.array()`**
        - **Interviewer Question**: "In a high-throughput inference service, your function takes an input that might be a list or an existing `ndarray`. How do you ensure it is an array?"
        - **Naive Candidate Answer**: "I always call `X = np.array(X, dtype=np.float32)` at the beginning of the function to ensure it's a NumPy array."
        - **Trap Unveiled**: Calling `np.array()` on an input that is *already* an `ndarray` forces a complete duplicate allocation and memory copy of millions of bytes, needlessly doubling memory footprint and wasting CPU memory bus bandwidth.
        - **Staff-Level Winning Response**: "Use `np.asarray(X, dtype=np.float32)`. If `X` is already an `ndarray` with matching dtype and contiguous layout, `np.asarray` is a zero-copy no-op that returns the original reference with zero heap allocation. It only allocates if `X` is a list or requires dtype casting."
        """)

    col_t3, col_t4 = st.columns(2)
    with col_t3:
        st.error("""
        **Trap 3: Testing Array Truth Value via `if arr:`**
        - **Interviewer Question**: "How do you check whether an input array contains any non-zero data?"
        - **Naive Candidate Answer**: "I write `if arr: print('Has data')`."
        - **Trap Unveiled**: In Python, testing truth value of an `ndarray` with more than 1 element raises `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`."
        - **Staff-Level Winning Response**: "Always use `if arr.size > 0:` to check for non-emptiness, or `if arr.any():` to check for non-zero values, avoiding Python truth ambiguity."
        """)
    with col_t4:
        st.warning("""
        **Trap 4: Using `np.arange` with Floating-Point Steps**
        - **Interviewer Question**: "How do you generate 10 evenly spaced values between 0.0 and 1.0?"
        - **Naive Candidate Answer**: "I use `np.arange(0.0, 1.0, 0.1)`."
        - **Trap Unveiled**: Due to IEEE 754 floating-point rounding errors, the number of elements produced by `np.arange` with float step can vary inconsistently (sometimes including or excluding the endpoint).
        - **Staff-Level Winning Response**: "Always use `np.linspace(0.0, 1.0, 10)` for floating-point intervals. `linspace` directly computes step sizes from the requested count, guaranteeing exact element counts and deterministic boundary behavior."
        """)

# Section 7: Interactive Interview Workbench and Vectorized Sandbox
st.header("7. Interactive Interview Workbench and Vectorized Sandbox", divider="rainbow")

st.markdown("""
Experiment with real-time array memory buffers, inspect byte strides across 1D, 2D, and 3D dimensions, 
explore practical coding tasks from the syllabus, and benchmark the memory overhead difference between 
Python list graphs and NumPy C-contiguous buffers.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("🛠️ Multi-Dimensional Array Memory Inspector")
    
    selected_dim = st.selectbox("Select Dimensionality:", ["1D Array", "2D Array", "3D Array"])
    selected_dtype = st.selectbox("Select Primitive Dtype:", ["int32", "int64", "float32", "float64"])
    
    dt = getattr(np, selected_dtype)
    item_bytes = np.dtype(dt).itemsize

    if selected_dim == "1D Array":
        d1 = st.slider("Dimension Length (N):", min_value=2, max_value=16, value=6)
        demo_arr = np.arange(d1, dtype=dt)
        shape_repr = f"({d1},)"
        expected_strides = f"({item_bytes},)"
    elif selected_dim == "2D Array":
        d1 = st.slider("Rows (Dim 0):", min_value=2, max_value=8, value=3)
        d2 = st.slider("Columns (Dim 1):", min_value=2, max_value=8, value=4)
        demo_arr = np.arange(d1 * d2, dtype=dt).reshape(d1, d2)
        shape_repr = f"({d1}, {d2})"
        expected_strides = f"({d2 * item_bytes}, {item_bytes})"
    else:
        d1 = st.slider("Batches / Depth (Dim 0):", min_value=2, max_value=4, value=2)
        d2 = st.slider("Rows (Dim 1):", min_value=2, max_value=6, value=3)
        d3 = st.slider("Columns (Dim 2):", min_value=2, max_value=6, value=4)
        demo_arr = np.arange(d1 * d2 * d3, dtype=dt).reshape(d1, d2, d3)
        shape_repr = f"({d1}, {d2}, {d3})"
        expected_strides = f"({d2 * d3 * item_bytes}, {d3 * item_bytes}, {item_bytes})"

    st.write("**Generated Array Representation:**")
    st.code(str(demo_arr), language="text")

    st.markdown(f"""
    - **Shape**: `{demo_arr.shape}`
    - **Rank (`ndim`)**: `{demo_arr.ndim}`
    - **Total Elements (`size`)**: `{demo_arr.size}`
    - **Element Size (`itemsize`)**: `{demo_arr.itemsize} bytes`
    - **Total Buffer Memory (`nbytes`)**: `{demo_arr.nbytes} bytes`
    - **Byte Strides Tuple**: `{demo_arr.strides}`
    - **Memory Flags**: `C_CONTIGUOUS: {demo_arr.flags.c_contiguous}`, `OWNDATA: {demo_arr.flags.owndata}`
    """)

with col_wb2:
    st.subheader("💻 Syllabus Coding Task Explorer")
    task_choice = st.selectbox(
        "Select Practical Coding Task to Inspect:",
        [
            "Task 1: 1D Vector to 2D Column Vector",
            "Task 2: Robust Rectangular Padding for Ragged Sequences",
            "Task 3: 3D Multi-Channel Volumetric Slice Extractor",
            "Task 4: Zero-Copy Interoperability Verification Probe"
        ]
    )

    if task_choice == "Task 1: 1D Vector to 2D Column Vector":
        st.markdown("**Problem**: Convert a 1D Python list into a column vector of shape `(N, 1)` without `.reshape()`.")
        st.code("""
# Solution using np.newaxis / ndmin:
py_list = [10, 20, 30, 40]
col_vec = np.array(py_list)[:, np.newaxis]
print("Shape:", col_vec.shape)  # (4, 1)
print("Strides:", col_vec.strides)  # (8, 8)
        """, language="python")
        ex_arr = np.array([10, 20, 30, 40])[:, np.newaxis]
        st.write("Live Execution Output Shape:", ex_arr.shape)

    elif task_choice == "Task 2: Robust Rectangular Padding for Ragged Sequences":
        st.markdown("**Problem**: Ingest a ragged list of sequences and pad with zeros to form a rectangular matrix.")
        st.code("""
sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
max_len = max(len(seq) for seq in sequences)
padded_matrix = np.zeros((len(sequences), max_len), dtype=np.int32)
for i, seq in enumerate(sequences):
    padded_matrix[i, :len(seq)] = seq
print(padded_matrix)
        """, language="python")
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        max_l = max(len(s) for s in seqs)
        pad_mat = np.zeros((len(seqs), max_l), dtype=np.int32)
        for i, s in enumerate(seqs):
            pad_mat[i, :len(s)] = s
        st.write("Live Output Matrix:")
        st.dataframe(pd.DataFrame(pad_mat), use_container_width=True)

    elif task_choice == "Task 3: 3D Multi-Channel Volumetric Slice Extractor":
        st.markdown("**Problem**: Extract a single channel slice from a 3D batch tensor of shape `(Batch, Height, Width)`.")
        st.code("""
tensor_3d = np.arange(24).reshape(2, 3, 4)
# Extract batch 0, all rows, columns 1 to 3
slice_view = tensor_3d[0, :, 1:3]
print("Slice Shape:", slice_view.shape)
print("Is View:", slice_view.base is tensor_3d)
        """, language="python")
        t3d = np.arange(24).reshape(2, 3, 4)
        slc = t3d[0, :, 1:3]
        st.write(f"Extracted View Shape: {slc.shape}, Shared Buffer: {slc.base is not None}")

    elif task_choice == "Task 4: Zero-Copy Interoperability Verification Probe":
        st.markdown("**Problem**: Verify whether `np.asarray()` avoids duplicating memory buffers.")
        st.code("""
original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
borrowed = np.asarray(original, dtype=np.float32)
duplicated = np.array(original, dtype=np.float32, copy=True)

assert borrowed is original          # True: Zero-copy pointer reuse!
assert duplicated is not original      # True: Memory allocated and copied!
        """, language="python")
        orig = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        borrowed = np.asarray(orig, dtype=np.float32)
        st.success("Verification Passed: `np.asarray()` achieved zero-copy reference reuse.")

st.divider()

st.subheader("⚡ Memory Footprint & Conversion Benchmark")
num_elements = st.select_slider(
    "Select Benchmark Array Size:",
    options=[10_000, 50_000, 100_000, 500_000],
    value=100_000
)

# Theoretical memory calculations
py_list_bytes = (28 + 8) * num_elements + 56
np_int64_bytes = num_elements * 8 + 128
np_float32_bytes = num_elements * 4 + 128

df_mem = pd.DataFrame({
    "Data Structure": [
        "Python List of Integers",
        "NumPy ndarray (int64)",
        "NumPy ndarray (float32)"
    ],
    "Bytes per Element": ["~36 bytes (object + pointer)", "8 bytes (raw C-long)", "4 bytes (raw C-float)"],
    "Total RAM Footprint (KB)": [
        f"{py_list_bytes / 1024:.1f} KB",
        f"{np_int64_bytes / 1024:.1f} KB",
        f"{np_float32_bytes / 1024:.1f} KB"
    ],
    "Relative Memory Ratio": [
        "1.0x (Baseline)",
        f"{np_int64_bytes / py_list_bytes:.2f}x ({(1 - np_int64_bytes / py_list_bytes) * 100:.1f}% savings)",
        f"{np_float32_bytes / py_list_bytes:.2f}x ({(1 - np_float32_bytes / py_list_bytes) * 100:.1f}% savings)"
    ]
})
st.dataframe(df_mem, use_container_width=True)

if st.button("🚀 Run Live List vs NumPy Creation Benchmark"):
    import time

    t0 = time.perf_counter()
    raw_list = [i for i in range(num_elements)]
    t_list_gen = time.perf_counter() - t0

    t1 = time.perf_counter()
    arr_from_list = np.array(raw_list, dtype=np.int64)
    t_conv = time.perf_counter() - t1

    t2 = time.perf_counter()
    arr_direct = np.arange(num_elements, dtype=np.int64)
    t_direct = time.perf_counter() - t2

    speedup_direct = t_conv / max(t_direct, 1e-9)

    st.success(f"**Direct `np.arange()` is {speedup_direct:.1f}x Faster than converting a Python List!**")
    df_timing = pd.DataFrame({
        "Operation": [
            "1. Python List Comprehension",
            "2. `np.array(python_list)` (Pointer Unpacking)",
            "3. `np.arange()` (Direct C Allocation)"
        ],
        "Elapsed Time (ms)": [t_list_gen * 1000, t_conv * 1000, t_direct * 1000],
        "Execution Mode": ["Interpreted bytecode loop", "C-loop unpacking PyObjects", "Single direct malloc + memset"]
    })
    st.dataframe(df_timing, use_container_width=True)

# Section 8: Rapid-Fire Drill, Flashcards and Self-Assessment Checklist
st.header("8. Rapid-Fire Drill, Flashcards and Self-Assessment", divider="rainbow")

with st.expander("⚡ Expand for Rapid-Fire Precision Q&As", expanded=False):
    st.markdown("""
    1. **Q: Does `np.array()` reuse the memory of an existing Python list?**  
       *A: Never. Python lists store object pointers; NumPy requires a contiguous primitive buffer. It always copies values.*
    2. **Q: What is the return type and shape of `np.array([1, 2, 3]).T`?**  
       *A: A 1D array of shape `(3,)`. Transposing a 1D array is a no-op.*
    3. **Q: How does `np.asarray()` differ from `np.array()` when passed an existing ndarray?**  
       *A: `np.asarray()` returns the input array directly with zero memory allocation. `np.array()` duplicates the buffer.*
    4. **Q: What happens if a list contains floats and strings: `np.array([1.0, 'a'])`?**  
       *A: All elements are silently upcast to Unicode string `dtype='<U32'`, breaking numerical operations.*
    5. **Q: What is the difference between `np.array()` and direct `np.ndarray()` instantiation?**  
       *A: `np.array()` is a safe factory function that copies or infers data. `np.ndarray()` is a low-level constructor that allocates uninitialized memory containing random RAM garbage.*
    6. **Q: How do strides differ between C-order and Fortran-order for shape `(3, 4)` with `int64`?**  
       *A: C-order strides are `(32, 8)` bytes. Fortran-order strides are `(8, 24)` bytes.*
    """)

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: Why is `np.array(existing_arr, copy=False)` deprecated/altered in modern NumPy versions?")
        st.caption("Flip for answer:")
        st.info("**Back**: `copy=False` had ambiguous semantics because if a dtype conversion was required, NumPy was forced to copy anyway. In modern NumPy, use `np.asarray()` for safe view sharing, or `copy=False` which now raises `ValueError` if a copy cannot be avoided.")

    with st.container(border=True):
        st.markdown("**Front**: What does `arr.base` return when called on an array?")
        st.caption("Flip for answer:")
        st.info("**Back**: If `arr` is a view over another array, `arr.base` returns the original owning array. If `arr` owns its own memory buffer (as when created via `np.array()` from a list), `arr.base` is `None`.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How are strides calculated for a 3D array of shape `(D, H, W)` with itemsize `S`?")
        st.caption("Flip for answer:")
        st.info("**Back**: In row-major (C-order): `stride[2] = S`, `stride[1] = W * S`, and `stride[0] = H * W * S`. Jumping across slices incurs the largest stride offset.")

    with st.container(border=True):
        st.markdown("**Front**: Why does `arr[[0, 1]]` (fancy indexing) allocate a copy while `arr[0:2]` (slicing) creates a view?")
        st.caption("Flip for answer:")
        st.info("**Back**: Basic slicing elements are always separated by regular, constant stride intervals, so an `ndarray` view can address them by adjusting offset and strides without copying. Fancy indexing selects arbitrary, non-regular indices that cannot be addressed with a single stride step, forcing NumPy to allocate a brand-new contiguous buffer.")

st.subheader("✅ Technical Interview Self-Assessment Checklist")
c1 = st.checkbox("I can explain why Python list conversion requires unpacking PyObject pointers into a contiguous C-buffer.", value=True)
c2 = st.checkbox("I understand the exact functional differences between `np.array()` and zero-copy `np.asarray()`.", value=True)
c3 = st.checkbox("I can calculate shape tuples and byte strides for 1D, 2D, and 3D arrays given arbitrary dtypes.", value=True)
c4 = st.checkbox("I understand why `arr.T` on a 1D array is a no-op and how to create true 2D row and column vectors.", value=True)
c5 = st.checkbox("I know how to prevent silent string upcasting and resolve jagged/ragged sequence errors.", value=True)
c6 = st.checkbox("I can articulate the O(N^2) memory and performance trap of calling `np.append()` inside loops.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All interview competencies verified! You are ready to advance to the next module.")
