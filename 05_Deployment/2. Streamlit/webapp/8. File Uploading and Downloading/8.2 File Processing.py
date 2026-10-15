import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO, BytesIO
from PIL import Image

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit File Processing Tutorial",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ File Processing & Pipeline Validation — Complete Guide")
st.markdown(
    """
Uploading a file is only the first step. Robust production applications must safely process byte streams, 
handle character encodings, rewind stream pointers (`file.seek(0)`), enforce size limits, 
convert images to NumPy inference tensors, parse JSON Lines (JSONL), and validate schema invariants.
"""
)

st.divider()

# ==============================================================================
# 1. Reading Uploaded Files & The seek(0) Stream Pointer Pattern
# ==============================================================================
st.header("1. Stream Pointer Mechanics & The `file.seek(0)` Pattern", divider="rainbow")
st.markdown(
    """
An `UploadedFile` object is an in-memory byte buffer (`io.BytesIO`). 
- When any library (Pandas, Pillow, JSON) reads the file, the **internal read pointer advances to the end of the file (EOF)**.
- If you attempt to read the file a second time without rewinding, you will get an `EmptyDataError`!
- **The Solution**: Always call **`file.seek(0)`** before subsequent read passes.
"""
)

col_sk1, col_sk2 = st.columns(2)

with col_sk1:
    st.markdown("**💻 Python Source Code (The seek(0) Pattern)**")
    code_seek = '''uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    # Pass 1: Read raw text to check line count:
    raw_lines = uploaded.getvalue().decode("utf-8").splitlines()
    st.write(f"Total raw lines: {len(raw_lines)}")
    
    # ⚠️ Pointer is now at EOF! Must rewind:
    uploaded.seek(0)
    
    # Pass 2: Load into Pandas cleanly:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(2))'''
    st.code(code_seek, language="python")

with col_sk2:
    st.markdown("**🖥️ Live Interactive Stream Pointer Demo**")
    demo_file = st.file_uploader("Upload CSV to Test Multi-Pass Reading", type=["csv"], key="demo_seek_uploader")
    
    if demo_file is not None:
        try:
            # Pass 1
            raw_text = demo_file.getvalue().decode("utf-8", errors="ignore")
            lines_count = len(raw_text.splitlines())
            st.caption(f"Pass 1 (Byte Stream Inspection): **{lines_count}** lines read.")
            
            # Rewind
            demo_file.seek(0)
            
            # Pass 2
            df_seek = pd.read_csv(demo_file)
            st.success(f"Pass 2 (Pandas Parsing after `seek(0)`): Successfully loaded **{len(df_seek)}** rows!")
            st.dataframe(df_seek.head(2), width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Processing error: {e}")
    else:
        st.caption("Upload any `.csv` file above to observe dual-pass stream reading.")

st.divider()

# ==============================================================================
# 2. Advanced CSV Processing with pandas.read_csv()
# ==============================================================================
st.header("2. Robust CSV Processing (`pandas.read_csv`)", divider="rainbow")
st.markdown(
    """
Real-world CSVs often contain formatting quirks. Production configurations:
- **`sep=None, engine='python'`**: Automatically detects separators (comma, tab, semicolon).
- **`encoding='utf-8'` with `'latin-1'` fallback**: Gracefully handles legacy character sets.
- **`on_bad_lines='skip'`**: Prevents crashing on corrupted rows with mismatched column counts.
"""
)

col_csv1, col_csv2 = st.columns(2)

with col_csv1:
    st.markdown("**💻 Python Source Code (Robust CSV Engine)**")
    code_csv_adv = '''def load_csv_safely(file):
    # Try standard UTF-8 first, fallback to Latin-1
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            file.seek(0)
            return pd.read_csv(
                file,
                encoding=enc,
                sep=None,          # Auto-sniff delimiter
                engine="python",
                on_bad_lines="skip"
            )
        except UnicodeDecodeError:
            continue
    raise ValueError("Unrecognized file encoding")'''
    st.code(code_csv_adv, language="python")

with col_csv2:
    st.markdown("**🖥️ Live Interactive Robust CSV Loader**")
    adv_csv_file = st.file_uploader("Upload Delimited Dataset (TSV, CSV, Semicolon)", type=["csv", "tsv", "txt"], key="adv_csv_inp")
    
    if adv_csv_file is not None:
        try:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    adv_csv_file.seek(0)
                    df_res = pd.read_csv(adv_csv_file, encoding=enc, sep=None, engine="python", on_bad_lines="skip")
                    st.success(f"✅ Loaded with encoding `{enc}`: **{len(df_res):,} rows** × **{df_res.shape[1]} columns**")
                    st.dataframe(df_res.head(3), width="stretch", hide_index=True)
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            st.error(f"Failed to parse delimited file: {e}")
    else:
        st.caption("Upload any CSV or TSV file to test auto-delimiter detection.")

st.divider()

# ==============================================================================
# 3. Excel Processing with pandas.read_excel()
# ==============================================================================
st.header("3. Multi-Sheet Excel Processing (`pandas.read_excel`)", divider="rainbow")
st.markdown(
    """
Excel workbooks frequently contain multiple sheets, metadata banners, and header offsets:
- Use `pd.ExcelFile` to inspect available worksheets before parsing.
- Use `header=int` to skip introductory title rows.
"""
)

col_xl1, col_xl2 = st.columns(2)

with col_xl1:
    st.markdown("**💻 Python Source Code**")
    code_xl = '''excel_file = st.file_uploader("Upload Excel", type=["xlsx"])

if excel_file is not None:
    xl = pd.ExcelFile(excel_file)
    target_sheet = st.selectbox("Worksheet", xl.sheet_names)
    header_offset = st.number_input("Header Row Index", 0, 10, 0)
    
    excel_file.seek(0)
    df_xl = pd.read_excel(excel_file, sheet_name=target_sheet, header=header_offset)
    st.dataframe(df_xl.head(), width="stretch")'''
    st.code(code_xl, language="python")

with col_xl2:
    st.markdown("**🖥️ Live Multi-Sheet Excel Inspector**")
    xl_in = st.file_uploader("Upload Multi-Sheet Excel Workbook", type=["xlsx"], key="xl_multi_inp")
    if xl_in is not None:
        try:
            xl_obj = pd.ExcelFile(xl_in)
            chosen_tab = st.selectbox("Select Sheet Tab", xl_obj.sheet_names, key="sel_xl_tab")
            xl_in.seek(0)
            df_parsed = pd.read_excel(xl_in, sheet_name=chosen_tab)
            st.success(f"Worksheet `{chosen_tab}` parsed: {len(df_parsed)} records.")
            st.dataframe(df_parsed.head(3), width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Excel error: {e}")
    else:
        st.caption("Upload `.xlsx` workbook to inspect sheets.")

st.divider()

# ==============================================================================
# 4. Reading JSON and JSON Lines (JSONL for LLM Datasets)
# ==============================================================================
st.header("4. Reading JSON & JSON Lines (JSONL for LLM Tuning)", divider="rainbow")
st.markdown(
    """
Most modern instruction-tuning datasets (e.g. OpenAI, Anthropic, Hugging Face) use **JSON Lines (`.jsonl`)**:
- Each line is an independent, valid JSON record (`{"prompt": "...", "response": "..."}`).
- Reading with `pd.read_json(file, lines=True)` automatically converts JSONL streams into tabular DataFrames!
"""
)

col_jsl1, col_jsl2 = st.columns(2)

with col_jsl1:
    st.markdown("**💻 Python Source Code (JSON vs. JSONL Parsing)**")
    code_jsonl = '''# Detect and parse standard JSON vs JSON Lines:
if file.name.endswith(".jsonl"):
    df_jsonl = pd.read_json(file, lines=True)
    st.dataframe(df_jsonl.head())
else:
    data = json.load(file)
    st.json(data)'''
    st.code(code_jsonl, language="python")

with col_jsl2:
    st.markdown("**🖥️ Live Interactive JSON / JSONL Demo**")
    json_in = st.file_uploader("Upload JSON or JSONL File", type=["json", "jsonl", "txt"], key="json_demo_inp")
    if json_in is not None:
        try:
            raw_content = json_in.getvalue().decode("utf-8")
            if json_in.name.endswith(".jsonl") or "\n{" in raw_content[:200]:
                json_in.seek(0)
                df_jsonl = pd.read_json(json_in, lines=True)
                st.success(f"✅ Ingested JSONL Dataset: **{len(df_jsonl)}** instruction pairs.")
                st.dataframe(df_jsonl.head(3), width="stretch", hide_index=True)
            else:
                parsed_json = json.loads(raw_content)
                st.success(f"✅ Ingested Standard JSON Structure.")
                st.json(parsed_json)
        except Exception as e:
            st.error(f"JSON parsing error: {e}")
    else:
        st.caption("Upload `.json` or `.jsonl` fine-tuning dataset.")

st.divider()

# ==============================================================================
# 5. Processing Uploaded Images for Vision AI
# ==============================================================================
st.header("5. Processing Images: Pillow to NumPy Tensors", divider="rainbow")
st.markdown(
    """
Computer vision models (YOLO, ResNet, ViT) require images formatted as **NumPy multidimensional arrays** or **PyTorch tensors**:
1. Open byte stream with `Image.open(file)`.
2. Ensure RGB channel format (`.convert("RGB")`).
3. Convert to array: `img_array = np.array(img)` with shape `(Height, Width, Channels)`.
4. Normalize pixel values to $[0.0, 1.0]$.
"""
)

col_vis1, col_vis2 = st.columns(2)

with col_vis1:
    st.markdown("**💻 Python Source Code (Vision Pipeline Preprocessing)**")
    code_vision = '''from PIL import Image
import numpy as np

uploaded_img = st.file_uploader("Upload Image", type=["png", "jpg"])

if uploaded_img is not None:
    # 1. Pillow open and RGB guarantee:
    pil_img = Image.open(uploaded_img).convert("RGB")
    
    # 2. Resize to model input dimensions (e.g. 224x224):
    resized = pil_img.resize((224, 224))
    
    # 3. Convert to normalized NumPy tensor:
    tensor = np.array(resized) / 255.0
    st.write(f"Tensor Shape: {tensor.shape} | Range: [{tensor.min():.2f}, {tensor.max():.2f}]")'''
    st.code(code_vision, language="python")

with col_vis2:
    st.markdown("**🖥️ Live Interactive Vision Preprocessing Demo**")
    v_img_in = st.file_uploader("Upload Image for Tensor Conversion", type=["png", "jpg", "jpeg", "webp"], key="vision_img_inp")
    if v_img_in is not None:
        try:
            p_img = Image.open(v_img_in).convert("RGB")
            resized_p = p_img.resize((224, 224))
            tensor_arr = np.array(resized_p, dtype=np.float32) / 255.0
            
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.image(p_img, caption="Original Image", width=180)
            with t_col2:
                st.markdown("##### 📐 Preprocessed Tensor Spec")
                st.write(f"• Shape: `{tensor_arr.shape}` (H, W, C)")
                st.write(f"• Dtype: `{tensor_arr.dtype}`")
                st.write(f"• Pixel Min: `{tensor_arr.min():.3f}`")
                st.write(f"• Pixel Max: `{tensor_arr.max():.3f}`")
        except Exception as e:
            st.error(f"Image tensor conversion error: {e}")
    else:
        st.caption("Upload an image above to observe live tensor transformation.")

st.divider()

# ==============================================================================
# 6. Handling File Size Limits & 7. Invalid Formats
# ==============================================================================
st.header("6. Enforcing File Size Limits & 7. Guarding Invalid Formats", divider="rainbow")
st.markdown(
    """
Never rely solely on client-side extensions. Implement robust defensive checks:
- **Widget Limit**: `max_upload_size=<MB>` parameter in `st.file_uploader()`.
- **Programmatic Byte Cap**: Reject files exceeding maximum payload memory limits before parsing.
- **Magic Byte Validation**: Validate binary signatures (e.g. PNG files always start with `\x89PNG\r\n\x1a\n`).
"""
)

col_lim1, col_lim2 = st.columns(2)

with col_lim1:
    st.markdown("**💻 Python Source Code (Defensive Size & Format Guards)**")
    code_limits = '''MAX_BYTES = 5 * 1024 * 1024  # 5 MB threshold

file = st.file_uploader("Upload Secure File", type=["csv", "png"])

if file is not None:
    # 1. Guard against oversized payloads:
    if file.size > MAX_BYTES:
        st.error(f"File size ({file.size / 1e6:.1f} MB) exceeds 5 MB ceiling!")
        st.stop()
        
    # 2. Guard against zero-byte empty files:
    if file.size == 0:
        st.error("Uploaded file is completely empty (0 bytes).")
        st.stop()'''
    st.code(code_limits, language="python")

with col_lim2:
    st.markdown("**🖥️ Live Interactive Size Guard Demo**")
    guard_file = st.file_uploader("Upload File (Strict 2 MB Limit Enforced)", key="guard_demo_uploader")
    
    if guard_file is not None:
        TWO_MB = 2 * 1024 * 1024
        if guard_file.size > TWO_MB:
            st.error(f"❌ File rejected: `{guard_file.name}` ({guard_file.size / 1024:.1f} KB) exceeds strict 2 MB limit!")
        elif guard_file.size == 0:
            st.error("❌ File rejected: Zero-byte empty file detected.")
        else:
            st.success(f"✅ File accepted: `{guard_file.name}` ({guard_file.size / 1024:.1f} KB) within safety bounds.")
    else:
        st.caption("Upload any file to test strict size threshold validation.")

st.divider()

# ==============================================================================
# 8. Validating Uploaded Content (Schema Compliance)
# ==============================================================================
st.header("8. Content & Schema Compliance Validation", divider="rainbow")
st.markdown(
    """
Before feeding uploaded data into ML models, verify that mandatory feature columns exist:
- **Mandatory Columns Check**: `set(REQUIRED_COLS).issubset(df.columns)`
- **Null Value Tolerance**: Ensure critical target columns do not contain missing values.
"""
)

REQUIRED_LLM_COLS = ["prompt", "response"]

col_sch1, col_sch2 = st.columns(2)

with col_sch1:
    st.markdown("**💻 Python Source Code (Schema Invariant Enforcement)**")
    code_schema = '''REQUIRED_COLUMNS = ["prompt", "response"]

uploaded_dataset = st.file_uploader("Upload RLHF Dataset", type=["csv"])

if uploaded_dataset is not None:
    df = pd.read_csv(uploaded_dataset)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    
    if missing:
        st.error(f"Schema Violation! Missing required columns: {missing}")
    elif df["prompt"].isnull().any():
        st.warning("Warning: Found null values in 'prompt' column!")
    else:
        st.success("Schema verified! Dataset ready for model training.")'''
    st.code(code_schema, language="python")

with col_sch2:
    st.markdown("**🖥️ Live Interactive Schema Validator Demo**")
    schema_upload = st.file_uploader("Upload RLHF Dataset (Requires: `prompt`, `response`)", type=["csv"], key="schema_val_uploader")
    
    if schema_upload is not None:
        try:
            df_sch = pd.read_csv(schema_upload)
            missing_cols = [c for c in REQUIRED_LLM_COLS if c not in df_sch.columns]
            
            if missing_cols:
                st.error(f"❌ Schema Violation: Missing required columns: **{missing_cols}**")
                st.caption(f"Detected columns in file: `{list(df_sch.columns)}`")
            else:
                null_prompts = df_sch["prompt"].isnull().sum()
                if null_prompts > 0:
                    st.warning(f"⚠️ Data Quality Warning: Found **{null_prompts}** missing rows in `'prompt'`.")
                st.success(f"✅ Schema Verified: Found all required columns (`prompt`, `response`) across **{len(df_sch)}** rows.")
                st.dataframe(df_sch.head(3), width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Error validating CSV schema: {e}")
    else:
        st.caption("Upload a CSV file containing `prompt` and `response` columns.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 File Processing Pipeline Checklist", divider="rainbow")
st.markdown(
    r"""
| Pipeline Stage | Critical Best Practice | Code Example |
| :--- | :--- | :--- |
| **Stream Pointer** | Rewind pointer after every read | `file.seek(0)` |
| **Encoding Fallback** | Try UTF-8 first, fallback to Latin-1 | `for enc in ['utf-8', 'latin-1']:` |
| **Delimiters** | Let Python engine sniff delimiter | `pd.read_csv(f, sep=None, engine='python')` |
| **Bad Lines** | Skip corrupt rows rather than crashing | `pd.read_csv(f, on_bad_lines='skip')` |
| **JSON Lines (JSONL)** | Use `lines=True` for streaming datasets | `pd.read_json(f, lines=True)` |
| **Vision AI** | Convert Pillow images to normalized arrays | `np.array(img.resize((224, 224))) / 255.0` |
| **Size Limit** | Reject files exceeding memory ceiling | `if file.size > MAX_BYTES: st.stop()` |
| **Schema Check** | Assert required columns exist | `if not set(COLS).issubset(df.columns):` |
"""
)
