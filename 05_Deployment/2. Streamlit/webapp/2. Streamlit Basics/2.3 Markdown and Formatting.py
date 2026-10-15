import streamlit as st

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Markdown and Formatting Guide",
    page_icon="✍️",
    layout="wide",
)

st.title("✍️ Markdown and Text Formatting — Complete Streamlit Guide")
st.markdown(
    """
Streamlit natively parses **GitHub-Flavored Markdown (GFM)** across almost all text-rendering primitives. 
This master guide covers standard typographic markup, mathematical LaTeX formatting, fenced code syntax, 
and modern custom HTML/CSS styling with the dedicated `st.html()` engine.
"""
)

st.divider()

# ==============================================================================
# 1. Markdown Headings (H1 to H6)
# ==============================================================================
st.header("1. Markdown Headings (`#` through `######`)", divider="rainbow")
st.markdown(
    """
Headings establish hierarchical document structure. In Streamlit Markdown:
- `#` through `######` correspond to HTML `<h1>` through `<h6>`.
- Streamlit automatically generates URL slug anchors for headers, enabling deep links.
- **Heading Hierarchy vs Native Commands**: `#` $\\approx$ `st.title()`, `##` $\\approx$ `st.header()`, `###` $\\approx$ `st.subheader()`.
"""
)

st.markdown("#### 🧪 Live Code vs. Rendered Output Comparison")
col_h1, col_h2 = st.columns(2)

with col_h1:
    st.markdown("**💻 Python Source Code**")
    code_headings = '''markdown_headings = """
# Heading 1 (Primary Title)
## Heading 2 (Major Section Header)
### Heading 3 (Subsection Subheader)
#### Heading 4 (Component Level)
##### Heading 5 (Minor Category)
###### Heading 6 (Fine Caption Header)
"""
st.markdown(markdown_headings)'''
    st.code(code_headings, language="python")

with col_h2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown(
        """
# Heading 1 (Primary Title)
## Heading 2 (Major Section Header)
### Heading 3 (Subsection Subheader)
#### Heading 4 (Component Level)
##### Heading 5 (Minor Category)
###### Heading 6 (Fine Caption Header)
"""
    )

st.divider()

# ==============================================================================
# 2. Bold, Italic, Strikethrough & Colored Text
# ==============================================================================
st.header("2. Emphasis, Text Formatting & Semantic Colors", divider="rainbow")
st.markdown(
    """
Markdown supports standard typography (`**bold**`, `*italic*`, `~~strike~~`), while Streamlit 
extends standard Markdown with **semantic colored spans** (`:color[text]`) and **background highlights** (`:color-background[text]`).
"""
)

col_fmt1, col_fmt2 = st.columns(2)

with col_fmt1:
    st.markdown("**💻 Python Source Code**")
    code_fmt = '''# Standard Typography
st.markdown("**Bold text** using double asterisks or __double underscores__")
st.markdown("*Italic text* using single asterisks or _single underscores_")
st.markdown("***Bold and italic text*** combined with triple asterisks")
st.markdown("~~Strikethrough text~~ using double tildes")

# Streamlit-Exclusive Semantic Colors:
st.markdown(":blue[Blue text], :green[Green text], :red[Red text], :orange[Orange text], :violet[Violet text]")
st.markdown(":rainbow[Vibrant rainbow gradient animated text span!]")

# Streamlit Background Color Highlights:
st.markdown(":blue-background[Blue Highlight] | :green-background[Green Highlight] | :red-background[Red Highlight]")'''
    st.code(code_fmt, language="python")

with col_fmt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown("**Bold text** using double asterisks or __double underscores__")
    st.markdown("*Italic text* using single asterisks or _single underscores_")
    st.markdown("***Bold and italic text*** combined with triple asterisks")
    st.markdown("~~Strikethrough text~~ using double tildes")
    st.markdown(":blue[Blue text], :green[Green text], :red[Red text], :orange[Orange text], :violet[Violet text]")
    st.markdown(":rainbow[Vibrant rainbow gradient animated text span!]")
    st.markdown(":blue-background[Blue Highlight] | :green-background[Green Highlight] | :red-background[Red Highlight]")

st.divider()

# ==============================================================================
# 3. Ordered Lists
# ==============================================================================
st.header("3. Ordered Lists (Numbered Sequences)", divider="rainbow")
st.markdown(
    """
Ordered lists represent chronological pipelines or prioritized workflows:
- Numbers followed by a period (`1.`, `2.`, `3.`) initialize an ordered list.
- Markdown automatically computes sequential ordering regardless of the literal digits typed (e.g. typing `1.` on every row).
- Nested ordered lists require **4 spaces or a tab** before the child number.
"""
)

col_ol1, col_ol2 = st.columns(2)

with col_ol1:
    st.markdown("**💻 Python Source Code**")
    code_ol = '''ml_pipeline_steps = """
1. **Data Ingestion**: Pull raw parquet shards from AWS S3
2. **Feature Engineering**:
    1. Fill missing numeric values with median
    2. Quantile transformation & one-hot encoding
    3. Generate 128-dim dense embedding vectors
3. **Model Training**:
    1. Distributed gradient boosting with LightGBM
    2. Hyperparameter optimization via Optuna
4. **Validation & Promotion**: Verify test ROC-AUC exceeds 0.92
5. **Deployment**: Serve ONNX model via FastAPI endpoint
"""
st.markdown(ml_pipeline_steps)'''
    st.code(code_ol, language="python")

with col_ol2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown(
        """
1. **Data Ingestion**: Pull raw parquet shards from AWS S3
2. **Feature Engineering**:
    1. Fill missing numeric values with median
    2. Quantile transformation & one-hot encoding
    3. Generate 128-dim dense embedding vectors
3. **Model Training**:
    1. Distributed gradient boosting with LightGBM
    2. Hyperparameter optimization via Optuna
4. **Validation & Promotion**: Verify test ROC-AUC exceeds 0.92
5. **Deployment**: Serve ONNX model via FastAPI endpoint
"""
    )

st.divider()

# ==============================================================================
# 4. Unordered Lists & Task Checklists
# ==============================================================================
st.header("4. Unordered Lists & Task Checklists", divider="rainbow")
st.markdown(
    """
Unordered lists display non-sequential collections using `-`, `*`, or `+`. 
GitHub-Flavored Markdown also supports **interactive task lists** (`- [ ]` and `- [x]`).
"""
)

col_ul1, col_ul2 = st.columns(2)

with col_ul1:
    st.markdown("**💻 Python Source Code**")
    code_ul = '''tech_stack = """
- **Deep Learning Frameworks**:
  - PyTorch 2.3 (CUDA 12.1)
  - Hugging Face Transformers
  - vLLM High-Throughput Serving
- **Vector Databases**:
  - Qdrant (HNSW Indexing)
  - Milvus / Pinecone

**Deployment Readiness Checklist:**
- [x] API Authentication & Rate Limiting Configured
- [x] Prometheus Metrics & Grafana Dashboards Connected
- [ ] Multi-region Disaster Recovery Failover Tested
"""
st.markdown(tech_stack)'''
    st.code(code_ul, language="python")

with col_ul2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown(
        """
- **Deep Learning Frameworks**:
  - PyTorch 2.3 (CUDA 12.1)
  - Hugging Face Transformers
  - vLLM High-Throughput Serving
- **Vector Databases**:
  - Qdrant (HNSW Indexing)
  - Milvus / Pinecone

**Deployment Readiness Checklist:**
- [x] API Authentication & Rate Limiting Configured
- [x] Prometheus Metrics & Grafana Dashboards Connected
- [ ] Multi-region Disaster Recovery Failover Tested
"""
    )

st.divider()

# ==============================================================================
# 5. Hyperlinks (Inline, Tooltips & Anchors)
# ==============================================================================
st.header("5. Hyperlinks & Navigation", divider="rainbow")
st.markdown(
    """
Hyperlinks navigate to external resources, documentation, or internal section anchors:
- **Inline Link**: `[Anchor Text](https://url)`
- **Link with Hover Tooltip**: `[Anchor Text](https://url "Tooltip Text")`
- **Internal Anchor Jump**: `[Jump to Headings](#1-markdown-headings-through)`
"""
)

col_link1, col_link2 = st.columns(2)

with col_link1:
    st.markdown("**💻 Python Source Code**")
    code_links = '''link_markdown = """
- [Streamlit Official Documentation](https://docs.streamlit.io)
- [Streamlit App Gallery](https://streamlit.io/gallery "Explore Community Applications")
- [PyTorch Foundation](https://pytorch.org)
- [Hugging Face Models Hub](https://huggingface.co/models)
- ⚓ [Back to Section 1 (Headings)](#1-markdown-headings-through)
"""
st.markdown(link_markdown)'''
    st.code(code_links, language="python")

with col_link2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown(
        """
- [Streamlit Official Documentation](https://docs.streamlit.io)
- [Streamlit App Gallery](https://streamlit.io/gallery "Explore Community Applications")
- [PyTorch Foundation](https://pytorch.org)
- [Hugging Face Models Hub](https://huggingface.co/models)
- ⚓ [Back to Section 1 (Headings)](#1-markdown-headings-through)
"""
    )

st.divider()

# ==============================================================================
# 6. Blockquotes & Callout Alerts
# ==============================================================================
st.header("6. Blockquotes & Callout Alerts (`>`)", divider="rainbow")
st.markdown(
    """
Blockquotes represent quotes, caveats, tips, and highlighted contextual notes:
- Prefix lines with `>` to generate an indented quotation block with a left accent border.
- Nest quotes with `>>` for layered discussions.
- Use GitHub callout patterns (`> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`) for standardized emphasis.
"""
)

col_bq1, col_bq2 = st.columns(2)

with col_bq1:
    st.markdown("**💻 Python Source Code**")
    code_bq = '''quote_markdown = """
> **Architectural Law**: "Premature optimization is the root of all evil in software engineering."  
> — *Sir Tony Hoare / Donald Knuth*

> 💡 **Production Pro-Tip**:
> In Streamlit, prefer caching resource-intensive model initializations with `@st.cache_resource` 
> and deterministic data transformations with `@st.cache_data`.
>
>> **Sub-clause**: Never mutate cached objects directly in place, as modifications will persist 
>> across all concurrent user sessions!
"""
st.markdown(quote_markdown)'''
    st.code(code_bq, language="python")

with col_bq2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown(
        """
> **Architectural Law**: "Premature optimization is the root of all evil in software engineering."  
> — *Sir Tony Hoare / Donald Knuth*

> 💡 **Production Pro-Tip**:
> In Streamlit, prefer caching resource-intensive model initializations with `@st.cache_resource` 
> and deterministic data transformations with `@st.cache_data`.
>
>> **Sub-clause**: Never mutate cached objects directly in place, as modifications will persist 
>> across all concurrent user sessions!
"""
    )

st.divider()

# ==============================================================================
# 7. Inline Code
# ==============================================================================
st.header("7. Inline Code (`` `code` ``)", divider="rainbow")
st.markdown(
    """
Inline code highlights variable names, function signatures, CLI arguments, and file paths inside sentences:
- Enclose text in single backticks: `` `variable_name` ``.
- To display a backtick *inside* inline code, enclose with double backticks: ```` `` ` `` ````.
"""
)

col_inc1, col_inc2 = st.columns(2)

with col_inc1:
    st.markdown("**💻 Python Source Code**")
    code_inc = '''inline_markdown = """
To start the application server on port `8501`, execute `streamlit run app.py` 
inside your activated virtual environment (`venv`).

Key session variables to monitor include:
- `st.session_state["user_id"]` (User GUID)
- `st.session_state["chat_history"]` (List of message dicts)
- `st.query_params.get("session_token")` (OAuth State)
"""
st.markdown(inline_markdown)'''
    st.code(code_inc, language="python")

with col_inc2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown(
        """
To start the application server on port `8501`, execute `streamlit run app.py` 
inside your activated virtual environment (`venv`).

Key session variables to monitor include:
- `st.session_state["user_id"]` (User GUID)
- `st.session_state["chat_history"]` (List of message dicts)
- `st.query_params.get("session_token")` (OAuth State)
"""
    )

st.divider()

# ==============================================================================
# 8. Fenced Code Blocks & st.code()
# ==============================================================================
st.header("8. Fenced Code Blocks & `st.code()`", divider="rainbow")
st.markdown(
    """
Multi-line source code can be rendered either via **Markdown triple backticks** (```` ```python ... ``` ````) 
or through Streamlit's dedicated **`st.code()`** element.
`st.code()` provides syntax highlighting, an integrated one-click copy button, and optional line numbers.
"""
)

st.markdown("#### 📐 Formal Signature: `st.code()`")
st.code(
    """st.code(
    body: str,
    language: str | None = "python",
    *,
    line_numbers: bool = False,
    wrap_lines: bool = False,
    width: "stretch" | "content" = "stretch"
) -> DeltaGenerator""",
    language="python",
)

col_cb1, col_cb2 = st.columns(2)

with col_cb1:
    st.markdown("**💻 Python Source Code**")
    code_snippet_py = '''import torch
import torch.nn as nn

class TransformerFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))'''
    
    st.code(f'''python_code = """{code_snippet_py}"""
st.code(python_code, language="python", line_numbers=True)''', language="python")

with col_cb2:
    st.markdown("**🖥️ Rendered `st.code()` Block (with Line Numbers)**")
    st.code(code_snippet_py, language="python", line_numbers=True)

st.divider()

# ==============================================================================
# 9. Mathematical Expressions (LaTeX & st.latex)
# ==============================================================================
st.header("9. Mathematical Expressions (LaTeX Math & `st.latex`)", divider="rainbow")
st.markdown(
    """
Mathematical notations and equations are parsed via KaTeX:
1. **Inline Math**: Enclose expression in single dollar signs: `$f(x) = ax + b$`.
2. **Display Block Math**: Enclose expression in double dollar signs: `$$ \\sum_{i=1}^{n} x_i $$`.
3. **Dedicated `st.latex()`**: Renders standalone centered mathematical formulations.
"""
)

st.markdown("#### 📐 Formal Signature: `st.latex()`")
st.code(
    """st.latex(body: str | object, *, help: str | None = None, width: "stretch" | "content" = "stretch") -> DeltaGenerator""",
    language="python"
)

col_mat1, col_mat2 = st.columns(2)

with col_mat1:
    st.markdown("**💻 Python Source Code**")
    code_latex = r'''# 1. Scaled Dot-Product Attention (Transformer Architecture)
st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")

# 2. Multi-Class Cross-Entropy Loss Function
st.latex(r"\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})")

# 3. Adam Optimizer Weight Update Equation
st.latex(r"\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t")'''
    st.code(code_latex, language="python")

with col_mat2:
    st.markdown("**🖥️ Rendered LaTeX Formulations**")
    st.markdown("**1. Scaled Dot-Product Attention:**")
    st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
    
    st.markdown("**2. Multi-Class Cross-Entropy Loss:**")
    st.latex(r"\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})")

    st.markdown("**3. Adam Optimizer Parameter Update:**")
    st.latex(r"\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t")

with st.expander("🧪 Interactive LaTeX Formula Sandbox"):
    user_formula = st.text_input(
        "Enter raw LaTeX math expression:",
        value=r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2}",
        help="Type standard LaTeX notation without surrounding dollar signs"
    )
    st.latex(user_formula)

st.divider()

# ==============================================================================
# 10. Custom HTML & CSS Concepts Using st.html()
# ==============================================================================
st.header("10. Custom HTML & CSS Concepts with `st.html()`", divider="rainbow")
st.markdown(
    """
Streamlit introduced **`st.html()`** as the primary first-class API for injecting custom HTML and CSS styling.
Unlike older workarounds (`st.markdown(..., unsafe_allow_html=True)`), `st.html()` provides:
- Clean semantic HTML structure.
- Full custom `<style>` injection that styles elements natively in the DOM.
- Superior performance without escaping quirks.
"""
)

st.markdown("#### 📐 Formal Signature: `st.html()`")
st.code(
    """st.html(
    body: str | Path | SupportsStr | SupportsReprHtml,
    *,
    width: "stretch" | "content" = "stretch",
    unsafe_allow_javascript: bool = False
) -> DeltaGenerator""",
    language="python"
)

st.markdown("#### 📋 Parameter Specification: `st.html()`")
st.markdown(
    """
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`body`** | `str`, `Path`, `object` | *Required* | Raw HTML string or path to an HTML file to render directly in the DOM. |
| **`width`** | `str` | `"stretch"` | Controls container boundary (`"stretch"` or `"content"`). |
| **`unsafe_allow_javascript`** | `bool` | `False` | Whether `<script>` tags can execute JavaScript. Default `False` prevents XSS vulnerabilities. |
"""
)

col_html1, col_html2 = st.columns(2)

custom_html_code = """<style>
.metric-card {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    margin-bottom: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #38BDF8;
}
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #94A3B8;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.card-value {
    color: #F8FAFC;
    font-size: 28px;
    font-weight: 700;
    margin: 8px 0;
}
.card-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.2);
    color: #34D399;
}
.pulse-dot {
    height: 8px;
    width: 8px;
    background-color: #10B981;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 8px #10B981;
}
</style>

<div class="metric-card">
    <div class="card-header">
        <span>GPU Cluster Node 01</span>
        <span><span class="pulse-dot"></span>LIVE</span>
    </div>
    <div class="card-value">99.82%</div>
    <div class="card-badge">▲ 0.14% vs 24h baseline</div>
</div>
"""

with col_html1:
    st.markdown("**💻 Python Source Code (`st.html`)**")
    code_display_html = f'''import streamlit as st

html_component = """{custom_html_code}"""

# Render custom HTML and scoped CSS
st.html(html_component)'''
    st.code(code_display_html, language="python")

with col_html2:
    st.markdown("**🖥️ Rendered Custom HTML Output**")
    st.html(custom_html_code)

st.divider()

# ==============================================================================
# 11. Interactive Markdown Live Playground
# ==============================================================================
st.header("🛝 Interactive Markdown Live Playground", divider="rainbow")
st.markdown("Experiment with real-time Markdown syntax, emojis, LaTeX, and colored spans below:")

sample_sandbox_text = """### 🚀 Model Card: Llama-3-70B-Instruct

> **Status**: :green[Production Ready] | **License**: :blue[Llama Community License]

The model demonstrates strong performance across standard benchmarks:
- **MMLU**: `82.0%`
- **GSM8K**: `93.0%`
- **HumanEval**: `81.7%`

#### 🔬 Attention Formulation:
$$ \\text{GQA}(Q, K, V) = \\text{Attention}(Q, \\text{repeat}(K), \\text{repeat}(V)) $$

Visit the official documentation at [Meta AI](https://ai.meta.com).
"""

col_sb1, col_sb2 = st.columns(2)

with col_sb1:
    st.markdown("**📝 Raw Markdown Input Editor**")
    user_md = st.text_area(
        "Type or edit Markdown here:",
        value=sample_sandbox_text,
        height=320,
        key="live_markdown_playground"
    )

with col_sb2:
    st.markdown("**🖥️ Live Dynamic Render Output**")
    st.markdown(user_md)
