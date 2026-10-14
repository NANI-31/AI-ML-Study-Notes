import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO, BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit File Downloading Tutorial",
    page_icon="📥",
    layout="wide",
)

st.title("📥 File Downloading & Export Pipelines — Complete Guide")
st.markdown(
    """
Exporting processed artifacts, fine-tuning datasets, machine learning models, and executive evaluation reports 
is a critical capability for production AI applications. In Streamlit, `st.download_button()` provides 
a seamless mechanism to download text, binary buffers, tables, and images directly in the user's browser.
"""
)

st.divider()

# ==============================================================================
# 1. st.download_button() Mechanics & Anatomy
# ==============================================================================
st.header("1. `st.download_button()` Mechanics & Anatomy", divider="rainbow")
st.markdown(
    r"""
The `st.download_button()` widget creates a button that triggers a browser download when clicked.

```python
st.download_button(
    label: str,
    data: str | bytes | io.BytesIO | io.StringIO,
    file_name: str | None = None,
    mime: str | None = None,
    key: str | int | None = None,
    help: str | None = None,
    on_click: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    disabled: bool = False,
    use_container_width: bool = False,
    type: "secondary" | "primary" | "tertiary" = "secondary",
    icon: str | None = None
)
```

> **Crucial Execution Model Gotcha**: Unlike a regular `st.button()` where code runs *after* clicking, 
> with `st.download_button()`, **the data payload must be fully generated in memory beforehand**. 
> When the user clicks the button, the browser downloads the already-buffered data without triggering a script rerun!
"""
)

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown("**💻 Python Source Code (Button Anatomy)**")
    code_anatomy = '''# Basic In-Memory String Download:
sample_content = "Antigravity AI Evaluation Log\\nTimestamp: 2026-10-12"

st.download_button(
    label="📥 Download Log File",
    data=sample_content,
    file_name="execution_log.txt",
    mime="text/plain",
    type="primary",
    icon="📄",
    help="Click to download plain text log"
)'''
    st.code(code_anatomy, language="python")

with col_d2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    sample_content = f"Antigravity AI Evaluation Log\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nStatus: System Operational\nTokens Processed: 1,420,950"
    st.download_button(
        label="📥 Download Log File",
        data=sample_content,
        file_name="execution_log.txt",
        mime="text/plain",
        type="primary",
        icon="📄",
        help="Click to download plain text log",
        key="anatomy_dl_btn"
    )
    st.caption("Clicking this button directly serves the pre-buffered string buffer as `execution_log.txt`.")

st.divider()

# ==============================================================================
# 2. Downloading Plain Text & Markdown Files
# ==============================================================================
st.header("2. Downloading Text & Markdown Files", divider="rainbow")
st.markdown(
    """
Plain text (`text/plain`) and Markdown (`text/markdown`) are standard formats for exporting 
system prompts, agent reasoning traces, model card summaries, and configuration files.
"""
)

col_txt1, col_txt2 = st.columns(2)

with col_txt1:
    st.markdown("**💻 Python Source Code (System Prompt Exporter)**")
    code_txt = '''system_prompt = st.text_area(
    "Agent System Prompt",
    "You are a helpful coding assistant specialized in PyTorch and Streamlit."
)

st.download_button(
    label="💾 Export Prompt (.md)",
    data=f"# System Prompt\\n\\n{system_prompt}",
    file_name="agent_prompt.md",
    mime="text/markdown",
    icon="📝"
)'''
    st.code(code_txt, language="python")

with col_txt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    user_prompt = st.text_area(
        "Agent System Prompt Configuration",
        value="You are an autonomous ML evaluation agent. Your role is to compute BLEU, ROUGE, and BERTScore on LLM outputs.",
        height=100,
        key="txt_export_prompt"
    )
    md_content = f"# System Prompt Specification\n\n*Generated on: {datetime.now().strftime('%Y-%m-%d')}*\n\n```text\n{user_prompt}\n```\n"
    st.download_button(
        label="💾 Export Prompt as Markdown",
        data=md_content,
        file_name="system_prompt.md",
        mime="text/markdown",
        icon="📝",
        key="txt_download_btn"
    )

st.divider()

# ==============================================================================
# 3. Downloading CSV Files with Caching Optimization
# ==============================================================================
st.header("3. Downloading CSV Datasets (`@st.cache_data` Pattern)", divider="rainbow")
st.markdown(
    r"""
Converting large DataFrames to CSV repeatedly on every script rerun causes severe performance lag.
Always wrap the serialization function in **`@st.cache_data`** so Pandas only serializes when the data actually changes!

```python
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    # IMPORTANT: Cache the conversion to prevent re-serializing on every rerun
    return df.to_csv(index=False).encode("utf-8")
```
"""
)

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

col_csv1, col_csv2 = st.columns(2)

with col_csv1:
    st.markdown("**💻 Python Source Code (Cached CSV Exporter)**")
    code_csv = '''@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

csv_bytes = convert_df_to_csv(filtered_df)

st.download_button(
    label="📊 Download Filtered CSV",
    data=csv_bytes,
    file_name="model_benchmark.csv",
    mime="text/csv",
    icon="📈"
)'''
    st.code(code_csv, language="python")

with col_csv2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    bench_data = pd.DataFrame({
        "Model": ["Claude 3.7 Sonnet", "GPT-4.5 Omni", "Gemini 2.5 Pro", "DeepSeek R1", "Llama 3.3 70B"],
        "MMLU (%)": [89.4, 88.7, 88.2, 90.8, 86.4],
        "HumanEval (%)": [92.0, 90.2, 89.5, 93.6, 82.1],
        "Latency (ms)": [380, 410, 320, 950, 240]
    })
    
    min_score = st.slider("Filter by Min HumanEval (%)", 80.0, 95.0, 85.0, 0.5, key="csv_min_score")
    filtered_bench = bench_data[bench_data["HumanEval (%)"] >= min_score]
    st.dataframe(filtered_bench, width="stretch", hide_index=True)
    
    csv_payload = convert_df_to_csv(filtered_bench)
    st.download_button(
        label=f"📊 Download Filtered CSV ({len(filtered_bench)} rows)",
        data=csv_payload,
        file_name="filtered_model_benchmarks.csv",
        mime="text/csv",
        icon="📈",
        key="csv_download_btn"
    )

st.divider()

# ==============================================================================
# 4. Downloading Multi-Sheet Excel Workbooks
# ==============================================================================
st.header("4. Downloading Multi-Sheet Excel Workbooks (`io.BytesIO`)", divider="rainbow")
st.markdown(
    """
Excel `.xlsx` workbooks are binary files. To generate them in memory without saving temporary files to disk:
1. Initialize an in-memory byte buffer: `output = BytesIO()`.
2. Open a Pandas `ExcelWriter` using engine `'openpyxl'`.
3. Write multiple DataFrames to distinct worksheet names.
4. Call `writer.close()` and retrieve bytes with `output.getvalue()`.
5. Specify MIME: `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`.
"""
)

@st.cache_data
def generate_excel_workbook(df_metrics: pd.DataFrame, df_params: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, index=False, sheet_name="Model_Evaluation")
        df_params.to_excel(writer, index=False, sheet_name="Hyperparameters")
    return output.getvalue()

col_xl1, col_xl2 = st.columns(2)

with col_xl1:
    st.markdown("**💻 Python Source Code (Multi-Sheet Excel)**")
    code_excel = '''from io import BytesIO

@st.cache_data
def make_excel(df_eval, df_config):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_eval.to_excel(writer, sheet_name="Evaluations", index=False)
        df_config.to_excel(writer, sheet_name="Configurations", index=False)
    return output.getvalue()

excel_bytes = make_excel(df_eval, df_config)

st.download_button(
    label="📑 Download Multi-Sheet Excel",
    data=excel_bytes,
    file_name="evaluation_workbook.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)'''
    st.code(code_excel, language="python")

with col_xl2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    df_params = pd.DataFrame({
        "Parameter": ["Temperature", "Top_P", "Max_Tokens", "Context_Window", "Quantization"],
        "Value": ["0.7", "0.95", "4096", "128,000", "FP8"]
    })
    st.caption("Workbook contains 2 Worksheets: `Model_Evaluation` and `Hyperparameters`")
    
    excel_payload = generate_excel_workbook(filtered_bench, df_params)
    st.download_button(
        label="📑 Download Evaluation Workbook (.xlsx)",
        data=excel_payload,
        file_name="llm_evaluation_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon="📑",
        key="excel_download_btn"
    )

st.divider()

# ==============================================================================
# 5. Downloading JSON and JSON Lines (JSONL for Fine-Tuning)
# ==============================================================================
st.header("5. Downloading JSON & JSON Lines (JSONL)", divider="rainbow")
st.markdown(
    """
JSON is the standard format for web APIs, model configuration checkpoints, and instruction-tuning datasets.
- **Hierarchical JSON**: Serialized with `json.dumps(obj, indent=2).encode("utf-8")`.
- **JSON Lines (`.jsonl`)**: Each row is an independent valid JSON record, standard for OpenAI / Hugging Face fine-tuning pipelines.
"""
)

col_js1, col_js2 = st.columns(2)

with col_js1:
    st.markdown("**💻 Python Source Code (JSON & JSONL Exporter)**")
    code_json = '''# 1. Export Standard Config JSON:
config_dict = {"model": "Qwen-2.5-Coder", "lora_rank": 32, "lr": 2e-4}
json_bytes = json.dumps(config_dict, indent=2).encode("utf-8")

# 2. Export JSONL Fine-Tuning Format:
jsonl_str = "\\n".join([json.dumps(row) for row in training_samples])

st.download_button(
    label="📥 Download JSONL Dataset",
    data=jsonl_str,
    file_name="tuning_data.jsonl",
    mime="application/jsonlines"
)'''
    st.code(code_json, language="python")

with col_js2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    lora_config = {
        "base_model": "deepseek-ai/DeepSeek-V3",
        "lora_r": 64,
        "lora_alpha": 128,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": 0.0002,
        "epochs": 3,
        "optimizer": "adamw_bnb_8bit"
    }
    
    col_jbtn1, col_jbtn2 = st.columns(2)
    with col_jbtn1:
        st.download_button(
            label="⚙️ Download LoRA Config (.json)",
            data=json.dumps(lora_config, indent=2).encode("utf-8"),
            file_name="adapter_config.json",
            mime="application/json",
            key="json_config_btn",
            width="stretch"
        )
    with col_jbtn2:
        samples = [
            {"messages": [{"role": "user", "content": "Write quicksort in Python"}, {"role": "assistant", "content": "def quicksort(arr):..."}]},
            {"messages": [{"role": "user", "content": "Explain attention mechanism"}, {"role": "assistant", "content": "Attention maps query..."}]}
        ]
        jsonl_payload = "\n".join(json.dumps(s) for s in samples)
        st.download_button(
            label="📚 Download SFT Dataset (.jsonl)",
            data=jsonl_payload,
            file_name="chat_finetune.jsonl",
            mime="text/plain",
            key="jsonl_data_btn",
            width="stretch"
        )

st.divider()

# ==============================================================================
# 6. Downloading Generated Reports (Dynamic HTML & Markdown)
# ==============================================================================
st.header("6. Downloading Generated Reports (Executive Summaries)", divider="rainbow")
st.markdown(
    """
Generate rich executive summary reports on the fly, with automated timestamping in the filename 
(`model_eval_YYYYMMDD_HHMMSS.html`).
"""
)

col_rep1, col_rep2 = st.columns(2)

with col_rep1:
    st.markdown("**💻 Python Source Code (Dynamic HTML Report Generator)**")
    code_report = '''def generate_html_report(model_name, accuracy, latency):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
    <html>
    <head><style>body {{ font-family: sans-serif; padding: 20px; }}</style></head>
    <body>
        <h1>Model Evaluation Certificate</h1>
        <p><strong>Generated:</strong> {now_str}</p>
        <hr>
        <p>Model: {model_name}</p>
        <p>Accuracy: {accuracy}% | Latency: {latency} ms</p>
    </body>
    </html>"""
    return html.encode("utf-8")'''
    st.code(code_report, language="python")

with col_rep2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    rep_model = st.selectbox("Select Model for Report", ["Claude 3.7 Sonnet", "GPT-4.5 Omni", "Gemini 2.5 Pro"], key="rep_model_sel")
    rep_format = st.radio("Report Format", ["Markdown (.md)", "HTML (.html)"], horizontal=True, key="rep_fmt_rad")
    
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "Markdown" in rep_format:
        report_text = f"""# Executive Model Evaluation Report
**Model Evaluated:** {rep_model}  
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Status:** PASSED (Production Ready)  

---

### Benchmark Highlights
- **HumanEval Pass@1:** 92.4%
- **MMLU 5-Shot:** 89.1%
- **Inference P99 Latency:** 420 ms

*Report automatically generated by Streamlit AI Evaluation Suite.*
"""
        st.download_button(
            label=f"📄 Download {rep_model} Report (.md)",
            data=report_text,
            file_name=f"eval_report_{rep_model.lower().replace(' ', '_')}_{time_stamp}.md",
            mime="text/markdown",
            key="rep_md_dl",
            type="primary"
        )
    else:
        html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; color: #1e293b; }}
        h1 {{ color: #0284c7; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
        .badge {{ background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #cbd5e1; padding: 12px; text-align: left; }}
        th {{ background-color: #f8fafc; }}
    </style>
</head>
<body>
    <h1>🚀 LLM Benchmark Certification</h1>
    <p><strong>Target Model:</strong> <span class="badge">{rep_model}</span></p>
    <p><strong>Generated At:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <table>
        <tr><th>Metric</th><th>Benchmark Score</th><th>Target Threshold</th><th>Result</th></tr>
        <tr><td>HumanEval (Python)</td><td>92.4%</td><td>85.0%</td><td>✅ PASSED</td></tr>
        <tr><td>MMLU (Reasoning)</td><td>89.1%</td><td>80.0%</td><td>✅ PASSED</td></tr>
        <tr><td>P99 Latency</td><td>420 ms</td><td>500 ms</td><td>✅ PASSED</td></tr>
    </table>
</body>
</html>"""
        st.download_button(
            label=f"🌐 Download {rep_model} Report (.html)",
            data=html_report.encode("utf-8"),
            file_name=f"eval_report_{rep_model.lower().replace(' ', '_')}_{time_stamp}.html",
            mime="text/html",
            key="rep_html_dl",
            type="primary"
        )

st.divider()

# ==============================================================================
# 7. Downloading Images & Visual Artifacts (Pillow to BytesIO)
# ==============================================================================
st.header("7. Downloading Generated Images & Plots", divider="rainbow")
st.markdown(
    """
When generating synthetic visual assets, confusion matrices, or annotated inference bounding boxes:
1. Render or manipulate the image in Pillow (`PIL.Image`).
2. Write image bytes to an in-memory `io.BytesIO()` buffer using `img.save(buf, format="PNG")`.
3. Provide `buf.getvalue()` and `mime="image/png"` to `st.download_button()`.
"""
)

def create_synthetic_metric_badge(model_name: str, score: float) -> bytes:
    img = Image.new("RGB", (400, 160), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    # Draw simple gradient box
    draw.rectangle([(10, 10), (390, 150)], outline=(56, 189, 248), width=3)
    draw.text((30, 30), f"MODEL: {model_name}", fill=(248, 250, 252))
    draw.text((30, 65), f"Pass@1 Score: {score:.1f}%", fill=(56, 189, 248))
    draw.text((30, 105), f"Verified: {datetime.now().strftime('%Y-%m-%d')}", fill=(148, 163, 184))
    
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

col_img1, col_img2 = st.columns(2)

with col_img1:
    st.markdown("**💻 Python Source Code (Pillow to BytesIO Buffer)**")
    code_img = '''from io import BytesIO
from PIL import Image

def generate_image_badge(title, score):
    img = Image.new("RGB", (400, 160), color=(15, 23, 42))
    # ... draw annotations ...
    
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

image_bytes = generate_image_badge("Claude 3.7", 92.4)

st.download_button(
    label="🖼️ Download Metric Badge (.png)",
    data=image_bytes,
    file_name="evaluation_badge.png",
    mime="image/png"
)'''
    st.code(code_img, language="python")

with col_img2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    badge_score = st.slider("Select Badge Benchmark Score (%)", 70.0, 100.0, 92.4, 0.1, key="img_badge_score")
    badge_png_bytes = create_synthetic_metric_badge("Sonnet 3.7", badge_score)
    
    st.image(badge_png_bytes, caption="Live Rendered Metric Certification Badge", width=300)
    st.download_button(
        label="🖼️ Download Badge (.png)",
        data=badge_png_bytes,
        file_name="model_badge.png",
        mime="image/png",
        icon="🖼️",
        key="img_download_btn"
    )

st.divider()

# ==============================================================================
# 8. Downloading Processed Datasets (End-to-End Pipeline)
# ==============================================================================
st.header("8. End-to-End Processed Dataset Export Pipeline", divider="rainbow")
st.markdown(
    """
A complete, real-world data processing workflow:
1. Generate or upload raw records.
2. Select feature subsets, filter thresholds, and clean nulls.
3. Dynamically export to the user's preferred format (**CSV**, **Excel**, or **JSONL**).
"""
)

# Synthetic ML Raw Dataset
np.random.seed(42)
n_rows = 120
raw_dataset = pd.DataFrame({
    "Prompt_ID": [f"PRM-{1000 + i}" for i in range(n_rows)],
    "Category": np.random.choice(["Coding", "Reasoning", "Mathematics", "Safety"], size=n_rows),
    "Tokens_Prompt": np.random.randint(20, 500, size=n_rows),
    "Tokens_Completion": np.random.randint(50, 1200, size=n_rows),
    "Quality_Score": np.round(np.random.uniform(0.65, 0.99, size=n_rows), 3),
    "Flagged": np.random.choice([False, False, False, True], size=n_rows)
})

fcol1, fcol2, fcol3 = st.columns(3)

with fcol1:
    cat_filter = st.multiselect("Filter Category", options=raw_dataset["Category"].unique(), default=["Coding", "Reasoning"], key="pipe_cat")
with fcol2:
    min_quality = st.slider("Minimum Quality Score", 0.65, 0.95, 0.80, 0.01, key="pipe_qual")
with fcol3:
    drop_flagged = st.checkbox("Drop Flagged Items", value=True, key="pipe_flag")

# Apply Transformation Pipeline
processed_df = raw_dataset.copy()
if cat_filter:
    processed_df = processed_df[processed_df["Category"].isin(cat_filter)]
processed_df = processed_df[processed_df["Quality_Score"] >= min_quality]
if drop_flagged:
    processed_df = processed_df[~processed_df["Flagged"]]

st.markdown(f"##### Transformed Dataset Preview: `{len(processed_df)}` rows retained (out of {n_rows})")
st.dataframe(processed_df.head(4), width="stretch", hide_index=True)

# Export controls
exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    export_csv = convert_df_to_csv(processed_df)
    st.download_button(
        label="📥 Export as CSV",
        data=export_csv,
        file_name=f"cleaned_dataset_{len(processed_df)}_rows.csv",
        mime="text/csv",
        width="stretch",
        key="dl_pipe_csv"
    )

with exp_col2:
    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as w:
        processed_df.to_excel(w, index=False, sheet_name="Cleaned_Data")
    st.download_button(
        label="📥 Export as Excel (.xlsx)",
        data=excel_buf.getvalue(),
        file_name=f"cleaned_dataset_{len(processed_df)}_rows.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        key="dl_pipe_excel"
    )

with exp_col3:
    # Convert to JSONL records
    jsonl_records = processed_df.to_json(orient="records", lines=True)
    st.download_button(
        label="📥 Export as JSONL",
        data=jsonl_records,
        file_name=f"cleaned_dataset_{len(processed_df)}_rows.jsonl",
        mime="text/plain",
        width="stretch",
        key="dl_pipe_jsonl"
    )

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 File Downloading MIME Types & Buffers Reference", divider="rainbow")
st.markdown(
    r"""
| Data Format | Buffer Type | MIME Type String | Serialization Method |
| :--- | :--- | :--- | :--- |
| **Plain Text (`.txt`)** | `str` or `bytes` | `text/plain` | Direct string or `str.encode('utf-8')` |
| **Markdown (`.md`)** | `str` or `bytes` | `text/markdown` | Direct markdown formatted string |
| **CSV (`.csv`)** | `bytes` | `text/csv` | `df.to_csv(index=False).encode('utf-8')` |
| **Excel (`.xlsx`)** | `io.BytesIO` | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | `pd.ExcelWriter(buf, engine='openpyxl')` |
| **JSON (`.json`)** | `bytes` | `application/json` | `json.dumps(obj, indent=2).encode('utf-8')` |
| **JSON Lines (`.jsonl`)** | `str` or `bytes` | `text/plain` or `application/jsonlines` | `df.to_json(orient='records', lines=True)` |
| **HTML Report (`.html`)**| `str` or `bytes` | `text/html` | HTML formatted string template |
| **PNG Image (`.png`)** | `io.BytesIO` | `image/png` | `pil_img.save(buf, format='PNG'); buf.getvalue()` |
| **JPEG Image (`.jpg`)** | `io.BytesIO` | `image/jpeg` | `pil_img.save(buf, format='JPEG'); buf.getvalue()` |
"""
)
