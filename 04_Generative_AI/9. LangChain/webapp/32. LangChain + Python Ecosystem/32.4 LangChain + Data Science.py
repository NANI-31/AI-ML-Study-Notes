import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="32.4 LangChain + Data Science - Analytics & Statistical Copilots",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for data science telemetry
st.markdown("""
<style>
    .stat-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .code-badge {
        background-color: #0F766E;
        color: #CCFBF1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class DataScienceAgentSimulator:
    """
    Simulates autonomous data science workflows including DataFrame manipulation,
    NumPy statistical calculations, text-to-SQL conversion, and chart artifact generation.
    """
    def __init__(self):
        # Sample business dataset
        self.sample_df = pd.DataFrame({
            "region": ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"],
            "q3_revenue_m": [45.2, 38.6, 29.4, 12.1, 8.5],
            "operating_cost_m": [28.4, 25.1, 18.2, 8.9, 5.2],
            "customer_satisfaction": [4.6, 4.4, 4.7, 4.1, 4.3]
        })

    def execute_eda_query(self, query_type: str) -> Dict[str, Any]:
        if query_type == "Correlation & Profit Margin":
            df = self.sample_df.copy()
            df["profit_margin_pct"] = ((df["q3_revenue_m"] - df["operating_cost_m"]) / df["q3_revenue_m"]) * 100
            corr = float(np.corrcoef(df["profit_margin_pct"], df["customer_satisfaction"])[0, 1])
            return {
                "code": "df['profit_margin'] = (df['q3_revenue_m'] - df['operating_cost_m']) / df['q3_revenue_m'] * 100\\ncorr = np.corrcoef(df['profit_margin'], df['customer_satisfaction'])[0, 1]",
                "result": f"Calculated Pearson Correlation: {corr:.3f}",
                "summary": "Mild positive correlation detected between profit margins and regional customer satisfaction scores.",
                "data": df[["region", "profit_margin_pct", "customer_satisfaction"]]
            }
        elif query_type == "SQL Aggregation Simulation":
            sql = "SELECT region, SUM(q3_revenue_m) as total_rev FROM enterprise_sales GROUP BY region ORDER BY total_rev DESC;"
            return {
                "code": f"-- Text-to-SQL Generated Query\\n{sql}",
                "result": "Aggregated 5 geographical regions with North America leading at $45.2M.",
                "summary": "Top revenue driver: North America ($45.2M), followed by Europe ($38.6M).",
                "data": self.sample_df[["region", "q3_revenue_m"]]
            }
        else: # Outlier Detection
            q3_rev = self.sample_df["q3_revenue_m"].to_numpy()
            mean = float(np.mean(q3_rev))
            std = float(np.std(q3_rev))
            z_scores = (q3_rev - mean) / std
            return {
                "code": "z_scores = (revenue - np.mean(revenue)) / np.std(revenue)",
                "result": f"Dataset Mean: ${mean:.1f}M, StdDev: ${std:.1f}M. Max Z-Score: {float(np.max(z_scores)):.2f}",
                "summary": "No extreme outliers detected beyond 2 standard deviations across all five active regions.",
                "data": pd.DataFrame({"region": self.sample_df["region"], "z_score": np.round(z_scores, 2)})
            }


# --- LESSON HEADER ---
st.title("32.4 LangChain + Data Science: Analytics & Statistical Copilots")
st.markdown("""
Integrating LangChain with the modern Python data science ecosystem (NumPy, Pandas, Matplotlib, SQLAlchemy)
unlocks autonomous analytical copilots capable of exploring datasets, generating SQL queries from natural language,
executing verified numerical computations, and rendering publication-grade visualization artifacts.
This module covers tool design, sandboxed code execution safety, and robust multi-step data analysis agents.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Autonomous Data Science & Statistical Copilots", divider="rainbow")
st.markdown("""
An autonomous data science agent operates within a disciplined Observe-Plan-Code-Execute loop.
User natural language prompts are translated into validated Python AST or SQL syntax, dispatched into
an isolated REPL execution sandbox, and inspected for runtime errors before generating visual and numerical artifacts.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph DataScience_LangChain_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_user_interface {
            label = "Analytical User Prompt";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            PromptIn [label="User Query\\n('Find revenue outliers and plot margin correlation')", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_orchestrator {
            label = "LangChain Data Science Agent (ReAct / Plan-and-Solve)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            AgentPlanner [label="Reasoning & Tool Selection\\n(Schema Inspection, Code Generation)", fillcolor="#1E293B", color="#60A5FA"];
            ASTValidator [label="AST & Security Guardrails\\n(Block os, subprocess, sys, network sockets)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_execution_engine {
            label = "Sandboxed Execution Runtime (Python AST REPL / SQL Engine)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            NumPyEngine [label="NumPy Vectorized Ops\\n(Matrix algebra, z-score, std dev)", fillcolor="#0369A1", color="#38BDF8"];
            PandasEngine [label="Pandas DataFrame Tool\\n(Filtering, grouping, window aggregates)", fillcolor="#0369A1", color="#38BDF8"];
            SQLEngine [label="SQLAlchemy Engine\\n(Read-only dialect, EXPLAIN cost, query execution)", fillcolor="#0369A1", color="#38BDF8"];
            MatplotlibEngine [label="Matplotlib Headless Renderer\\n(plt.savefig -> base64 PNG capture)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_artifacts {
            label = "Synthesized Data Deliverables";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            VisualArtifacts [label="Visual & Statistical Output\\n(Markdown tables, rendered charts, business summary)", fillcolor="#065F46", color="#10B981"];
        }

        PromptIn -> AgentPlanner [label="User Intent"];
        AgentPlanner -> ASTValidator [label="Generated Code / SQL"];
        ASTValidator -> NumPyEngine [label="Validated Numerical Ops"];
        ASTValidator -> PandasEngine [label="Validated Tabular Ops"];
        ASTValidator -> SQLEngine [label="Validated Read-only SQL"];
        ASTValidator -> MatplotlibEngine [label="Validated Plot Code"];
        NumPyEngine -> AgentPlanner [label="Vectorized Results"];
        PandasEngine -> AgentPlanner [label="Aggregated DataFrames"];
        SQLEngine -> AgentPlanner [label="Database Result Sets"];
        MatplotlibEngine -> VisualArtifacts [label="Rendered Chart Images"];
        AgentPlanner -> VisualArtifacts [label="Final Business Narrative"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: NUMPY ---
st.header("2. NumPy Integration & Vectorized Numerical Operations", divider="rainbow")
st.markdown("""
LLMs struggle with reliable raw arithmetic (e.g., matrix inversion, standard deviations, confidence intervals).
Equipping LangChain agents with NumPy tool wrappers offloads numerical heavy lifting to high-performance C-extensions.
This prevents hallucinations and provides microsecond execution speeds for mathematical formulas.
""")

col_np1, col_np2 = st.columns(2)
with col_np1:
    st.markdown("### Why Vectorization Matters for LLMs")
    st.markdown("""
    - **Zero Hallucination**: Mathematical operations are computed deterministically rather than guessed by token probabilities.
    - **Token Efficiency**: Instead of outputting 50 intermediate arithmetic calculation steps, the model outputs 2 lines of NumPy code.
    - **High-Dimensional Operations**: Allows agents to compute cosine distances, covariance matrices, and PCA directly.
    """)
with col_np2:
    st.code("""
import numpy as np
from langchain_core.tools import tool

@tool
def calculate_z_scores(data: list[float]) -> list[float]:
    \"\"\"Compute statistical z-scores for a list of numerical floats.\"\"\"
    arr = np.array(data)
    std = np.std(arr)
    if std == 0:
        return [0.0] * len(data)
    mean = np.mean(arr)
    z_scores = (arr - mean) / std
    return z_scores.round(3).tolist()
    """, language="python")

# --- SECTION 3: PANDAS ---
st.header("3. Pandas DataFrame Agents & Safe Python AST Execution", divider="rainbow")
st.markdown("""
LangChain provides `create_pandas_dataframe_agent` to query tabular data using natural language.
Under the hood, the agent writes Python Pandas expressions and executes them against in-memory DataFrames.
In production, running raw `exec()` or `eval()` is catastrophic; AST (Abstract Syntax Tree) sanitization is mandatory
to block imports of dangerous modules like `os`, `sys`, and `subprocess`.
""")

st.code("""
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

# Load enterprise dataset
df = pd.read_csv("enterprise_sales_q3.csv")

# Create DataFrame Agent with restricted permissions
agent = create_pandas_dataframe_agent(
    llm=ChatOpenAI(model="gpt-4o", temperature=0.0),
    df=df,
    verbose=True,
    agent_type="openai-tools",
    allow_dangerous_code=False  # Requires explicit opt-in or safe sandboxing
)

# Example natural language invocation
response = agent.invoke({
    "input": "Calculate the average profit margin grouped by region and identify any negative values."
})
""", language="python")

# --- SECTION 4: MATPLOTLIB ---
st.header("4. Matplotlib Chart Generation & Headless Visual Artifact Extraction", divider="rainbow")
st.markdown("""
Agents can dynamically generate Matplotlib and Seaborn visualization code. In backend microservices,
Matplotlib must run in non-GUI headless mode (`matplotlib.use('Agg')`). Instead of displaying windows with `plt.show()`,
the script captures the chart as an in-memory buffer (`io.BytesIO`) and returns base64-encoded images to the client.
""")

st.code("""
import io
import base64
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from langchain_core.tools import tool

@tool
def render_bar_chart(categories: list[str], values: list[float], title: str) -> str:
    \"\"\"Generate a bar chart and return it as a base64 encoded PNG string.\"\"\"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(categories, values, color="#38BDF8")
    ax.set_title(title)
    ax.set_ylabel("Value")
    
    # Save chart to memory buffer
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"
""", language="python")

# --- SECTION 5: SQL ---
st.header("5. Enterprise SQL Integration: SQLAlchemy, Schema Discovery, and Text-to-SQL", divider="rainbow")
st.markdown("""
Connecting LangChain to relational databases involves SQLAlchemy. The agent queries database catalog metadata,
inspects table schemas, generates dialect-correct SQL (PostgreSQL, Snowflake, BigQuery), validates query safety,
and returns structured query results.
""")

st.code("""
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

# Connect using SQLAlchemy URI (Read-only user connection)
db = SQLDatabase.from_uri(
    "postgresql://readonly_agent:secure_pass@db.prod.internal:5432/analytics",
    include_tables=["orders", "customers", "line_items"],
    sample_rows_in_table_info=2
)

sql_agent = create_sql_agent(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    db=db,
    agent_type="openai-tools",
    verbose=True
)

# Natural language to verified SQL execution
result = sql_agent.invoke({
    "input": "What were the top 5 largest customer orders in July 2026?"
})
""", language="python")

# --- SECTION 6: DATA ANALYSIS AGENTS ---
st.header("6. Autonomous Data Analysis Agents: Multi-Step Exploratory Data Analysis & Hypothesis Testing", divider="rainbow")
st.markdown("""
Advanced data science agents do not stop at single queries. They formulate hypotheses, inspect missingness,
calculate correlations, test statistical significance (e.g. p-values via SciPy), and generate narrative reports.
By structuring agents with LangGraph, workflows alternate deterministically between statistical analysis and narrative synthesis.
""")

col_ds1, col_ds2 = st.columns(2)
with col_ds1:
    st.markdown("### Multi-Step EDA Workflow")
    st.markdown("""
    1. **Schema & Missingness Audit**: Inspect null ratios, column types, and duplicate rows.
    2. **Distribution Analysis**: Compute skewness, kurtosis, and quartile spreads.
    3. **Hypothesis Formulation**: Propose relationships between key business indicators.
    4. **Significance Testing**: Run ANOVA, Chi-square, or t-tests to evaluate statistical validity.
    5. **Executive Synthesis**: Compile actionable business takeaways.
    """)
with col_ds2:
    st.code("""
# Statistical Hypothesis Testing Tool
import scipy.stats as stats
from langchain_core.tools import tool

@tool
def evaluate_t_test(sample_a: list[float], sample_b: list[float]) -> dict:
    \"\"\"Conduct two-sample independent t-test for statistical significance.\"\"\"
    t_stat, p_val = stats.ttest_ind(sample_a, sample_b)
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 5),
        "significant_at_05": bool(p_val < 0.05)
    }
    """, language="python")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: End-to-End Enterprise Data Science Agentic Pipeline", divider="rainbow")
st.markdown("""
A robust production implementation pairs AST security filters with isolated execution environments,
returning structured analytical findings, generated SQL, and visualization artifacts.
""")

st.code("""
import ast
import pandas as pd
from typing import Dict, Any

class SafeDataAnalysisSandbox:
    ALLOWED_MODULES = {"math", "numpy", "pandas", "scipy"}
    
    def validate_ast(self, code_str: str) -> bool:
        \"\"\"Inspect Python code AST to verify no forbidden system modules are imported.\"\"\"
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in self.ALLOWED_MODULES:
                        raise SecurityError(f"Unauthorized module import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.ALLOWED_MODULES:
                    raise SecurityError(f"Unauthorized module import: {node.module}")
        return True

    def run_analysis(self, df: pd.DataFrame, code_snippet: str) -> Dict[str, Any]:
        self.validate_ast(code_snippet)
        local_scope = {"df": df, "pd": pd, "np": np}
        exec(code_snippet, {}, local_scope)
        return {k: v for k, v in local_scope.items() if k not in ("df", "pd", "np")}
""", language="python")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Data Science Agent Sandbox & Code Execution Simulator", divider="rainbow")
st.markdown("""
Execute exploratory queries against an enterprise dataset, observe generated code snippets, and review statistical outputs.
""")

ds_sim = DataScienceAgentSimulator()

col_query, col_results = st.columns([1, 1])

with col_query:
    st.subheader("🔍 Analytical Task Selector")
    chosen_task = st.selectbox(
        "Select Data Science Workflow:",
        ["Correlation & Profit Margin", "SQL Aggregation Simulation", "Outlier Detection (Z-Score)"]
    )
    run_btn = st.button("🚀 Execute Data Science Pipeline", type="primary")

with col_results:
    st.subheader("📊 Output & Telemetry")
    if run_btn:
        res = ds_sim.execute_eda_query(chosen_task)
        st.markdown(f"**Execution Summary**: {res['summary']}")
        st.code(res["code"], language="python")
        st.info(res["result"])
        st.dataframe(res["data"], use_container_width=True)
    else:
        st.info("Select a workflow and execute to inspect code generation and analytical results.")

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why is `allow_dangerous_code=True` required for Pandas DataFrame agents, and how is it secured in production?"):
    st.markdown("""
    **Answer**: The Pandas DataFrame agent generates and executes arbitrary Python code strings using `eval()` or `exec()`.
    In LangChain, this flag was introduced to prevent accidental Remote Code Execution (RCE) vulnerabilities. In production,
    this must NEVER run on un-sandboxed host processes; instead, code execution must be delegated to containerized microVMs (e.g. AWS Firecracker,
    E2B, Modal) or Docker containers stripped of network interfaces and host filesystem access.
    """)

with st.expander("Q2: How do you prevent text-to-SQL agents from dropping or modifying database tables?"):
    st.markdown("""
    **Answer**: Relying purely on prompt instructions ('Please only write SELECT queries') is fundamentally unsafe. Security must be enforced
    at the database layer: connect using a database user granted strictly read-only permissions (`GRANT SELECT ON ALL TABLES`). Furthermore,
    use AST SQL parsers (such as `sqlglot`) to reject any AST containing `DROP`, `DELETE`, `UPDATE`, `ALTER`, or `INSERT` statements prior to execution.
    """)

with st.expander("Q3: Why must Matplotlib use `matplotlib.use('Agg')` in backend LangChain microservices?"):
    st.markdown("""
    **Answer**: By default, Matplotlib attempts to attach to an interactive graphical display backend (like Tkinter, Qt, or Cocoa).
    In server environments (Docker, headless Linux, FastAPI ASGI servers), there is no display attached ($DISPLAY is unset), causing
    immediate server crashes with `RuntimeError: Invalid DISPLAY variable`. `Agg` provides a headless, off-screen raster backend.
    """)

with st.expander("Q4: How does schema size affect SQL agent performance, and how do you mitigate schema bloat?"):
    st.markdown("""
    **Answer**: Enterprise databases often contain hundreds of tables with dozens of columns each. Dumping the entire database catalog into the
    LLM prompt exhausts context limits and causes tool hallucinations. Mitigation strategies include providing table description metadata,
    using semantic vector search over table schemas to inject only top-k relevant tables, and whitelisting specific tables in `SQLDatabase(include_tables=...)`.
    """)

with st.expander("Q5: What is the benefit of combining NumPy vectorization with LLM code generation?"):
    st.markdown("""
    **Answer**: Vectorized NumPy computations execute across contiguous C-memory blocks at native hardware speed, orders of magnitude faster
    than Python loops or iterative LLM reasoning. Delegating numerical tasks to NumPy prevents arithmetic calculation hallucinations,
    reduces prompt token overhead, and scales seamlessly to arrays containing millions of elements.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Host Process Code Execution)**: Running `create_pandas_dataframe_agent` directly in your application server process. A user prompt can inject `__import__('os').system('rm -rf /')` and wipe the server.")
    st.warning("**Trap 2 (Text-to-SQL Write Access Leaks)**: Connecting SQL agents using read-write administrative credentials. A hallucinated or injected query can truncate production tables.")
with col_trap2:
    st.error("**Trap 3 (Memory Leaks via Matplotlib Figures)**: Generating hundreds of charts using `plt.figure()` without calling `plt.close(fig)`. Unclosed figures accumulate indefinitely in server RAM, triggering out-of-memory errors.")
    st.warning("**Trap 4 (Unbounded SQL Query Result Sets)**: Executing queries without an enforced `LIMIT` clause. An agent querying `SELECT * FROM event_logs` can pull 10 million rows into application memory, crashing the service.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: Why is AST analysis necessary before running LLM-generated Python code?")
        st.caption("Flip for answer:")
        st.info("**Back**: It validates the code syntax tree before execution, blocking forbidden imports (`os`, `sys`) and unauthorized function calls.")
    with st.container(border=True):
        st.markdown("**Front**: What Matplotlib backend is required for headless server chart generation?")
        st.caption("Flip for answer:")
        st.info("**Back**: `matplotlib.use('Agg')` (non-interactive off-screen raster graphics).")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How should SQL agent write permissions be restricted in production?")
        st.caption("Flip for answer:")
        st.info("**Back**: At the database role level (read-only SELECT user) and with query AST validation tools like `sqlglot`.")
    with st.container(border=True):
        st.markdown("**Front**: How does vectorization prevent LLM mathematical errors?")
        st.caption("Flip for answer:")
        st.info("**Back**: Computations are executed deterministically by compiled C/NumPy binaries instead of predicted probabilistically by token generation.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand why LLMs must delegate numerical and statistical math to NumPy tools.", value=True)
c2 = st.checkbox("I know how to restrict and sandbox Pandas DataFrame agents to prevent Remote Code Execution.", value=True)
c3 = st.checkbox("I can generate headless Matplotlib visual artifacts and export base64 PNG images.", value=True)
c4 = st.checkbox("I understand how to configure read-only SQLAlchemy connections for enterprise SQL agents.", value=True)
c5 = st.checkbox("I know how to prevent context bloat when working with large database schemas.", value=True)
c6 = st.checkbox("I understand why unclosed Matplotlib figures cause server memory leaks.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Chapter 33: Testing LangChain Applications.")
