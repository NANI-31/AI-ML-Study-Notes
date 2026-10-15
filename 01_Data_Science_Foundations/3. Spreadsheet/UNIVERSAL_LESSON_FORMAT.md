# Universal Streamlit Lesson File Standard & Architecture Specification
### Single Source of Truth & Master Format for Domain: `Spreadsheet` (Grid Architecture, Dynamic Arrays, Power Query ETL, Power Pivot & DAX)
### Target Curriculum: `01_Data_Science_Foundations/3. Spreadsheet/spreadsheet.txt` (Chapters 1–28)
### Core Reference Standard: Continuous Vertical Scroll · Dark Graphviz · Top-to-Bottom Flow · Zero LaTeX Math · Power Query & DAX Core

---

## 1. Architectural Philosophy & Master Design Rules

Every `.py` lesson file in `3. Spreadsheet` represents an interactive, production-grade learning module covering spreadsheet calculation engines, dynamic array spill mechanics, business financial logic, structured data cleaning, automated Power Query (M Language) ETL pipelines, Power Pivot relational data modeling (Star Schemas), DAX filter context evaluations, interactive executive dashboards, and enterprise automation via Python and AI.

To ensure consistency, pedagogical depth, and zero UI breakage across all lessons, **every single generated `.py` file must strictly follow this single unified structure**.

### 1.1 Core Principles
1. **Continuous Vertical Scroll Flow**: All content resides on a single scrollable page (`st.set_page_config(layout="wide")`). Top-level `st.tabs` are **strictly forbidden** because they hide educational content behind clicks. (Sub-tabs are only permitted inside self-contained widgets or workbench result panels).
2. **Mandatory 1:1 Section Density (Every Roadmap Bullet Gets a Dedicated Section)**: Every single numbered bullet point under a lesson in `spreadsheet.txt` **MUST receive its own dedicated `st.header("X. <Topic Name>", divider="rainbow")` section**. Never compress or merge topics.
   - Total Sections per lesson file is locked by formula:
     `Total Sections = 1 (Architecture Blueprint) + K (Roadmap Syllabus Topics) + 1 (Production Implementation) + 1 (Interactive Workbench) + 1 (Technical Interview Prep) + 1 (Self-Assessment Checklist) = K + 5`.
3. **Clean Single Sequential Section Numbering (Strictly Zero Compound Numbering)**: Section headers MUST ALWAYS use clean, single sequential integer numbering from 1 to N (`st.header("1. <Blueprint>", divider="rainbow")`, `st.header("2. <Topic 1 Title>", divider="rainbow")`, ..., `st.header(f"{K+2}. Production Implementation: ...")`). Strictly prohibit compound, dotted, or nested section numbering (e.g. NEVER use `st.header("2. 1. ...")`, `st.header("2.1 ...")`, `st.header("3. 2. ...")`).
4. **ABSOLUTELY ZERO MATHEMATICAL EQUATIONS**:
   - This curriculum is strictly focused on **Spreadsheet Engineering, Relational Data Modeling, and Practical Business Operations**.
   - Strictly prohibit ALL LaTeX notation (`$$...$$`, `$x$`), calculus, linear algebra matrices, and abstract mathematical sigma summation notation.
   - Frame all calculations strictly in practical spreadsheet syntax (`=SUMIFS(...)`, `=XLOOKUP(...)`, `=CALCULATE(..., FILTER(...))`, `=FILTER(...)`), business financial metrics (gross margin, variance, CAGR, EBITDA, budget variance), and relational table data structures.
5. **Modern Spreadsheet Production Reality Contrast**:
   - Every section covering legacy or suboptimal spreadsheet techniques must feature a prominent callout block:
     `### 🚀 Modern Production Reality: How Current Industry Stacks Implement This`
   - Explicitly contrast legacy vs modern implementations:
     - Fragile nested `VLOOKUP` (which breaks whenever a column is inserted) vs Robust `XLOOKUP` and dynamic array `INDEX+MATCH`.
     - Monolithic legacy array formulas (`Ctrl+Shift+Enter`) vs Modern native Dynamic Arrays with automatic Spill Ranges (`#`).
     - Manual copy-pasting and volatile formulas (`OFFSET`/`INDIRECT`) vs Automated Power Query (M Language) repeatable ETL data pipelines.
     - Bloated single-sheet flattened tables vs Relational Power Pivot Star Schemas (Fact vs Dimension tables) with lightning-fast VertiPaq compression.
     - Manual cell calculations vs DAX measures with explicit Filter Context, Row Context, and Time Intelligence (`YTD`, `SAMEPERIODLASTYEAR`).
6. **Mandatory Top-to-Bottom Diagram Flow (`rankdir=TB;`) & `st.expander` Encapsulation**:
   - **Top-to-Bottom Flow (`rankdir=TB;`) Mandatory**: Long horizontal pipelines (`rankdir=LR;` with more than 3–4 sequential nodes) are **strictly forbidden** because Streamlit scales SVGs to container width, crushing wide diagrams into flat, illegible horizontal strips. All ETL pipelines, data models, and dependency trees must use **`rankdir=TB;` (Top to Bottom)** so they expand vertically with readable fonts (`fontsize=11`) and generous margins (`margin="0.2,0.1"`).
   - **Mandatory `st.expander` Container for Blueprints**: Wrap macro architectural blueprints inside an expander:
     ```python
     with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
         st.graphviz_chart(dot_blueprint, use_container_width=True)
     ```
   - **Contextual In-Section Diagrams (Mandatory Whenever Required)**: Diagrams must NOT be restricted to Section 1 alone. Whenever any roadmap topic section covers a multi-step workflow, data structure, or relational model (e.g. Power Query ETL transformation steps, Star Schema entity relationships, Dynamic Array spill ranges, DAX filter context transition paths), it **MUST include its own dedicated contextual Graphviz diagram inside that specific section**.
7. **Graphviz XML Rule**: Text inside HTML-like labels (`< ... >`) is parsed by a strict XML parser. Never use raw `&`, `<`, or `>` in headers or cells (always use `and` or `&amp;`).
8. **Module Scope for Interactive Workbench Classes**: Any classes, formula evaluation engines, mock ETL pipelines, or DAX simulators needed by the Interactive Workbench must be defined in real Python module scope within the script. Never define executable classes solely inside string literals passed to `st.code(...)`.
9. **Standard Imports**: Exactly 5 standard imports at the top of the file:
   ```python
   import streamlit as st
   import pandas as pd
   import numpy as np
   import graphviz
   from typing import List, Dict, Any, Tuple, Optional
   ```
   Strictly prohibit `import os` and `from pathlib import Path`.
10. **Uniform Pedagogical Endings**: Every single lesson concludes with:
    - **Section K+4: Technical Interview Preparation & Production Traps** (Exactly 5 Q&As + 4 Traps [2 error, 2 warning] + 4 Flashcards).
    - **Section K+5: Self-Assessment Checklist** (Exactly 6 Checkboxes + clean `st.success(...)` completion banner. Strictly prohibit `st.balloons()`).

---

## 2. Complete File Anatomy (Top-to-Bottom Structure)

```
┌────────────────────────────────────────────────────────────────────────┐
│ [A] Standard Imports Block (Exactly 5 Standard Imports)                │
├────────────────────────────────────────────────────────────────────────┤
│ [B] st.set_page_config(page_title=..., page_icon=..., layout="wide")   │
├────────────────────────────────────────────────────────────────────────┤
│ [C] Title Block: st.title + st.caption + Narrative with 4-5 Pillars    │
├────────────────────────────────────────────────────────────────────────┤
│ [D] First st.divider()                                                 │
├────────────────────────────────────────────────────────────────────────┤
│ [E] Section 1: End-to-End Architectural Blueprint (Graphviz Dark TB)   │
│     └── Wrapped in st.expander with rankdir=TB;                        │
├────────────────────────────────────────────────────────────────────────┤
│ [F] Sections 2 through K+1: Mandatory 1:1 Syllabus Roadmap Topics      │
│     ├── Deep conceptual explanation (Formulas / Data Modeling)         │
│     ├── 📊 Contextual Diagram (Mandatory whenever workflows appear)    │
│     ├── 🚀 Modern Production Reality: Industry Standard Callout        │
│     ├── Side-by-Side Comparison Matrix (Legacy vs Modern)              │
│     └── Interactive Mini-Demo / Code Demonstration                     │
├────────────────────────────────────────────────────────────────────────┤
│ [G] Section K+2: Production Implementation: End-to-End Pipeline        │
│     ├── Real-world runnable Python / Pandas / Excel model pipeline     │
│     └── Clean data handling, formula verification & robust outputs     │
├────────────────────────────────────────────────────────────────────────┤
│ [H] Section K+3: Interactive Workbench Simulator                       │
│     ├── Real-time input manipulation widgets (Sliders, Text Areas)     │
│     └── Live Formula Evaluator / Power Query Step Inspector            │
├────────────────────────────────────────────────────────────────────────┤
│ [I] Section K+4: Technical Interview Preparation & Production Traps    │
│     ├── Exactly 5 High-Frequency Interview Q&A Expanders               │
│     ├── Exactly 4 Production Traps (2x st.error, 2x st.warning)        │
│     └── Exactly 4 Flashcards in a 2-Column Responsive Grid             │
├────────────────────────────────────────────────────────────────────────┤
│ [J] st.divider()                                                       │
├────────────────────────────────────────────────────────────────────────┤
│ [K] Section K+5: Self-Assessment Checklist (Mandatory Last Section)    │
│     ├── Exactly 6 Checkboxes (All Initialized with value=True)         │
│     └── if chk1..chk6: -> st.success(...) [Strictly NO balloons]       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Structural Component Code Templates

### 3.1 Title Block & 5-Pillar Overview
```python
st.title("<X.Y> <Full Descriptive Lesson Title>")
st.caption("Chapter <N>: <Chapter Name> | Domain: Spreadsheet Engineering & Modern Analytics")
st.markdown("""
<2-4 sentence high-level overview explaining how this spreadsheet mechanism operates,
why modern business analytics and data operations depend on this pattern, and how it prevents errors.>

In this lesson, you will master:
- **Core Grid & Calculation Engine**: Cell referencing, evaluation trees, and formula semantics.
- **Modern Analytical Functions**: Dynamic array operations, criteria lookups, and aggregations.
- **Automated Data Transformation**: Power Query ETL pipelines and data normalization.
- **Relational Data Modeling**: Star schemas, relationships, and DAX filter context.
- **Technical Interview Readiness**: Business scenario questions and critical spreadsheet traps.
""")
st.divider()
```

### 3.2 Dark-Theme Graphviz Template for Spreadsheet Architecture (Top-to-Bottom Flow)
```python
dot_blueprint = """
digraph SpreadsheetArchitecture {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_raw {
        label="Data Ingestion & Extraction Tier";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        raw_sources [label="Raw Data Sources\\n(CSV, SQL DB, Web API, ERP)", fillcolor="#0F172A"];
        power_query [label="Power Query (M Engine)\\n(Clean, Unpivot, Type Cast, Filter)", fillcolor="#1E293B"];
    }

    subgraph cluster_model {
        label="Relational Data Model & Engine (Power Pivot)";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        dim_tables [label="Dimension Tables\\n(Dim_Customer, Dim_Date, Dim_Product)", fillcolor="#1E293B"];
        fact_table [label="Fact Table\\n(Fact_Sales / Transactions)", fillcolor="#1E293B"];
        dax_engine [label="DAX Calculation Engine\\n(Filter Context & Dynamic Measures)", fillcolor="#1E293B"];
    }

    subgraph cluster_presentation {
        label="Presentation & Reporting Tier";
        fontcolor="#10B981";
        style=dashed;
        color="#334155";
        dynamic_arrays [label="Dynamic Array Spill Grids\\n(=FILTER, =SORT, =XLOOKUP)", fillcolor="#064E3B"];
        executive_dash [label="Executive Dashboard & Slicers\\n(Interactive KPI Cards, PivotCharts)", fillcolor="#064E3B"];
    }

    raw_sources -> power_query [label="Import"];
    power_query -> dim_tables [label="Load Dimensions"];
    power_query -> fact_table [label="Load Facts"];
    dim_tables -> fact_table [label="1-to-Many Relationship", dir=both];
    fact_table -> dax_engine [label="Aggregate"];
    dax_engine -> dynamic_arrays [label="Evaluate Measures"];
    dax_engine -> executive_dash [label="Render KPIs"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)
```

---

## 4. Master Specification Reference Implementation

Below is the complete reference implementation demonstrating all required sections:

```python
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="1.1 What is a Spreadsheet? - Spreadsheet Curriculum",
    page_icon="📊",
    layout="wide"
)

# --- TITLE BLOCK ---
st.title("1.1 What is a Spreadsheet?")
st.caption("Chapter 1: Introduction to Spreadsheets | Domain: Spreadsheet Engineering & Modern Analytics")
st.markdown("""
A spreadsheet is an interactive software application that organizes data into a two-dimensional grid of cells
arranged in rows and columns. Beyond simple tabular data storage, modern enterprise spreadsheets serve as powerful
analytical engines featuring dynamic array formula computation, automated Power Query (M Language) data pipelines,
relational Power Pivot data modeling, and DAX-driven business intelligence reporting.

In this lesson, you will master:
- **Grid Architecture & Calculation Engine**: Cell addressing, formula evaluation trees, and data types.
- **Spreadsheet Ecosystem**: Excel vs Google Sheets, enterprise data warehouses, and programmatic automation.
- **Modern Analytical Paradigms**: Moving from fragile nested lookups to dynamic arrays and automated ETL.
- **Business Operations Value**: Financial modeling, KPI monitoring, and audit-proof data transformation.
- **Production Traps**: Preventing circular references, broken formulas, and hardcoded data anti-patterns.
""")
st.divider()

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Modern Enterprise Spreadsheet Pipeline", divider="rainbow")

dot_blueprint = """
digraph SpreadsheetPipeline {
    rankdir=TB;
    bgcolor="#0B1120";
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", color="#334155", fillcolor="#1E293B", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

    subgraph cluster_ingestion {
        label="1. Automated Data Ingestion Tier";
        fontcolor="#38BDF8";
        style=dashed;
        color="#334155";
        raw_csv [label="External CSV / ERP / SQL\\n(Raw Transaction Records)"];
        pq_etl [label="Power Query (M Engine)\\n(Clean, Cast Types, Unpivot)"];
    }

    subgraph cluster_modeling {
        label="2. Semantic Data Modeling Tier";
        fontcolor="#F59E0B";
        style=dashed;
        color="#334155";
        star_schema [label="Star Schema (Power Pivot)\\n(Fact_Sales + Dim_Calendar + Dim_Customer)"];
        dax_calc [label="DAX Calculation Engine\\n(CALCULATE, Filter Context, Measures)"];
    }

    subgraph cluster_presentation {
        label="3. Dynamic Reporting Tier";
        fontcolor="#10B981";
        style=dashed;
        color="#334155";
        spill_grid [label="Dynamic Array Spill Grid\\n(=FILTER, =SORT, =XLOOKUP)"];
        dashboard [label="Interactive Executive Dashboard\\n(KPI Cards, PivotCharts, Slicers)"];
    }

    raw_csv -> pq_etl [label="Extract"];
    pq_etl -> star_schema [label="Transform & Load"];
    star_schema -> dax_calc [label="Model Relationships"];
    dax_calc -> spill_grid [label="Spill Dynamic Results"];
    dax_calc -> dashboard [label="Feed Visuals"];
}
"""

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    st.graphviz_chart(dot_blueprint, use_container_width=True)

# --- SECTIONS 2 THROUGH K+1: ROADMAP TOPICS ---
# (Example for Topic 1: Definition of Spreadsheets)
st.header("2. Definition of Spreadsheets", divider="rainbow")
st.markdown("""
A **spreadsheet** is an electronic ledger composed of cells defined by coordinate intersections of columns (letters)
and rows (numbers). Each cell holds either a raw literal value (Text, Number, Date, Boolean) or a dynamic formula that
evaluates to a computed result. Modern spreadsheet calculation engines build directed dependency graphs to automatically
recalculate dependent cells whenever precedent values change.
""")

with st.expander("🔍 View Flow Diagram: Calculation Engine Dependency Tree", expanded=False):
    dot_sub = """
    digraph DependencyTree {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", fillcolor="#1E293B", color="#334155"];
        edge [color="#38BDF8"];
        cell_a1 [label="Cell A1: Unit Price ($50)"];
        cell_b1 [label="Cell B1: Quantity Sold (12)"];
        cell_c1 [label="Cell C1: Subtotal (=A1*B1)"];
        cell_d1 [label="Cell D1: Tax (=C1*0.08)"];
        cell_e1 [label="Cell E1: Total (=C1+D1)"];
        cell_a1 -> cell_c1;
        cell_b1 -> cell_c1;
        cell_c1 -> cell_d1;
        cell_c1 -> cell_e1;
        cell_d1 -> cell_e1;
    }
    """
    st.graphviz_chart(dot_sub, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current Industry Stacks Implement This
In modern enterprise operations, manual data entry and nested `VLOOKUP` chains have been completely superseded. Professional
analysts construct automated ETL pipelines via Power Query to clean and reshape source feeds automatically. Data is loaded into
Power Pivot's internal columnar database (VertiPaq) to build normalized Star Schemas, and dynamic array functions (`FILTER`, `UNIQUE`, `XLOOKUP`)
drive dynamic dashboards with zero manual maintenance.
""")

df_compare = pd.DataFrame({
    "Dimension": ["Legacy Spreadsheet Practice", "Modern Spreadsheet Engineering"],
    "Lookup Mechanism": ["VLOOKUP (breaks on column inserts, slow)", "XLOOKUP / INDEX+MATCH (robust, two-way, exact by default)"],
    "Data Reshaping": ["Manual copy-paste & text-to-columns", "Power Query (M Language) automated, repeatable ETL"],
    "Data Modeling": ["Flat monolithic tables with duplicate data", "Relational Star Schema with Fact and Dimension tables"],
    "Array Computation": ["Complex CSE formulas (Ctrl+Shift+Enter)", "Dynamic Arrays with native Spill Ranges (=FILTER, =UNIQUE)"]
})
st.table(df_compare)

# --- SECTION K+2: PRODUCTION IMPLEMENTATION ---
st.header("3. Production Implementation: Dynamic Array & Financial Pipeline", divider="rainbow")
st.code("""
import pandas as pd
import numpy as np

# Enterprise Financial Modeling Pipeline in Python (Paralleling Modern Excel)
def build_sales_model(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = raw_data.copy()
    
    # 1. Clean & Cast (Equiv. to Power Query Step)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["SalesAmount"] = df["Quantity"] * df["UnitPrice"]
    df["COGS"] = df["Quantity"] * df["UnitCost"]
    
    # 2. Add Business Metrics
    df["GrossMargin"] = df["SalesAmount"] - df["COGS"]
    df["MarginPct"] = np.where(df["SalesAmount"] > 0, df["GrossMargin"] / df["SalesAmount"], 0.0)
    
    # 3. Dynamic Array Style Aggregation (Equiv. to =FILTER and =SORT)
    high_margin_records = df[df["MarginPct"] >= 0.35].sort_values(by="SalesAmount", ascending=False)
    return high_margin_records
""", language="python")

# --- SECTION K+3: INTERACTIVE WORKBENCH ---
st.header("4. Interactive Workbench: Dynamic Array & Margin Calculator", divider="rainbow")

class SpreadsheetSimulator:
    def calculate(self, base_price: float, qty: int, discount: float, tax_rate: float) -> Dict[str, Any]:
        gross_sales = base_price * qty
        discount_amount = gross_sales * (discount / 100.0)
        net_sales = gross_sales - discount_amount
        tax = net_sales * (tax_rate / 100.0)
        final_total = net_sales + tax
        return {
            "Gross Sales ($)": round(gross_sales, 2),
            "Discount ($)": round(discount_amount, 2),
            "Net Sales ($)": round(net_sales, 2),
            "Tax ($)": round(tax, 2),
            "Final Total ($)": round(final_total, 2)
        }

calc = SpreadsheetSimulator()
st.subheader("Configure Financial Formula Inputs")
col1, col2 = st.columns(2)
with col1:
    unit_p = st.number_input("Unit Price ($)", min_value=1.0, value=75.0, step=5.0)
    quantity = st.slider("Quantity Sold", 1, 100, 15)
with col2:
    disc_pct = st.slider("Discount (%)", 0, 50, 10)
    tax_pct = st.slider("Tax Rate (%)", 0, 25, 8)

if st.button("🚀 Evaluate Spreadsheet Formulas", type="primary"):
    res = calc.calculate(unit_p, quantity, disc_pct, tax_pct)
    st.write("### Calculated Cell Outputs")
    st.json(res)

# --- SECTION K+4: TECHNICAL INTERVIEW PREPARATION ---
st.header("5. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qa = [
    ("Why is XLOOKUP superior to VLOOKUP for enterprise data models?",
     "XLOOKUP searches to the left or right without column index numbers, preventing broken formulas when columns are inserted or deleted. It defaults to exact match, eliminates costly sorting prerequisites, supports reverse searches, and handles missing values natively without IFERROR wrappers."),
    ("What is a Dynamic Array and how does the Spill Range (#) operator work?",
     "Dynamic arrays allow a single formula in one cell to return multiple values that automatically populate neighboring cells (spilling). The spill range operator (#) references the entire generated array dynamically (e.g. A2#), automatically expanding or contracting when source data changes."),
    ("Explain the difference between Power Query and Power Pivot.",
     "Power Query is an ETL tool (Extract, Transform, Load) powered by the M Language that connects to external data, cleanses, shapes, and imports it. Power Pivot is an in-memory data modeling tool that defines relationships between tables (Star Schema) and computes business metrics via DAX."),
    ("What is the difference between Row Context and Filter Context in DAX?",
     "Row Context exists during iteration across each individual row of a table (e.g. in calculated columns or iterator functions like SUMX). Filter Context consists of all filters applied by PivotTable rows, columns, slicers, and CALCULATE statements that restrict data before aggregation."),
    ("Why should volatile functions like OFFSET and INDIRECT be minimized in large workbooks?",
     "Volatile functions recalculate every time any cell in the workbook changes, even if precedent cells were untouched. In workbooks with tens of thousands of rows, hundreds of volatile formulas force continuous full recalculations, causing massive performance lag.")
]

for i, (q, a) in enumerate(interview_qa, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Breaking Error)**: Circular Reference Error. When a formula refers to its own cell directly or indirectly, the calculation engine cannot resolve the dependency graph, resulting in frozen iterations or incorrect results.")
    st.error("**Trap 2 (Breaking Error)**: Hardcoding Formula Inputs. Embedding literal exchange rates or tax constants directly inside formula strings (=A1*0.0825) creates maintenance nightmares. Always isolate assumptions in dedicated input cells.")
with col_trap2:
    st.warning("**Trap 3 (Performance Warning)**: Volatile Function Overuse. Relying heavily on OFFSET, INDIRECT, and NOW across thousands of rows causes sluggish workbooks because they recalculate on every single sheet edit.")
    st.warning("**Trap 4 (Integrity Warning)**: Accidental Text-Formatted Numbers. Importing numbers stored as text causes SUM and AVERAGE to silently ignore those cells, leading to corrupt financial reporting without displaying an explicit error.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_col1, fc_col2 = st.columns(2)
with fc_col1:
    with st.expander("🎴 Flashcard 1: Absolute Cell Referencing"):
        st.markdown("**$A$1 locks row and column; $A1 locks column only; A$1 locks row only during copy.**")
    with st.expander("🎴 Flashcard 2: Spill Operator (#)"):
        st.markdown("**Appending # to a top-left cell reference (e.g. A2#) dynamically grabs the entire spill array.**")
with fc_col2:
    with st.expander("🎴 Flashcard 3: Power Query M Language"):
        st.markdown("**A functional, case-sensitive ETL language recording sequential step transformations.**")
    with st.expander("🎴 Flashcard 4: DAX CALCULATE"):
        st.markdown("**The supreme DAX function that overrides and manipulates the active Filter Context.**")

st.divider()

# --- SECTION K+5: SELF-ASSESSMENT CHECKLIST ---
st.header("6. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand cell coordinate systems and relative vs absolute referencing.", value=True)
chk2 = st.checkbox("I can replace fragile VLOOKUP formulas with robust XLOOKUP and dynamic arrays.", value=True)
chk3 = st.checkbox("I understand how Power Query ETL automates repetitive data cleaning workflows.", value=True)
chk4 = st.checkbox("I can explain the difference between Row Context and Filter Context in DAX.", value=True)
chk5 = st.checkbox("I can identify and eliminate volatile formulas that degrade calculation performance.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 technical interview questions without notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on this lesson. Ready for the next module!")
```
