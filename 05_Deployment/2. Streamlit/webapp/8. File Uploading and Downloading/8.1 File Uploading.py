import streamlit as st
import pandas as pd
import json
from io import StringIO, BytesIO
from PIL import Image
import pypdf

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit File Uploading Tutorial",
    page_icon="📤",
    layout="wide",
)

st.title("📤 File Uploading (`st.file_uploader`) — Complete Master Guide")
st.markdown(
    """
Ingesting external data is the starting point for analytical dashboards, computer vision pipelines, 
and Retrieval-Augmented Generation (RAG) applications. 
Streamlit's **`st.file_uploader()`** accepts files directly from the user's local filesystem into memory as an 
in-memory byte stream (`io.BytesIO`). 
This guide covers CSV, Excel, JSON, text, image, and PDF ingestion, along with multi-file batch processing.
"""
)

st.divider()

# ==============================================================================
# 1. st.file_uploader() Core Architecture & UploadedFile Interface
# ==============================================================================
st.header("1. Core Architecture & The `UploadedFile` Interface", divider="rainbow")
st.markdown(
    """
When a user uploads a file, Streamlit returns an **`UploadedFile`** object:
- **`io.BytesIO` Subclass**: Behaves like an open Python file handle in binary mode.
- **Attributes**:
  - `file.name`: String filename (e.g., `"train_dataset.csv"`).
  - `file.size`: File size in bytes (e.g., `1048576`).
  - `file.type`: MIME type (e.g., `"text/csv"`, `"image/png"`, `"application/pdf"`).
- **Key Methods**:
  - `file.read()`: Reads bytes from current pointer.
  - `file.getvalue()`: Returns all raw bytes without advancing the pointer.
  - `file.seek(0)`: Resets pointer to the beginning (critical after reading!).
"""
)

st.markdown("#### 📐 Formal Signature: `st.file_uploader()`")
st.code(
    """st.file_uploader(
    label: str,
    type: str | Sequence[str] | None = None,  # Allowed extensions without leading dot
    accept_multiple_files: bool = False,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    disabled: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible",
    max_upload_size: int | None = None  # Size in MB (default 200MB)
) -> UploadedFile | list[UploadedFile] | None""",
    language="python",
)

st.divider()

# ==============================================================================
# 2. Uploading CSV Files
# ==============================================================================
st.header("2. Uploading CSV Datasets (`type=['csv']`)", divider="rainbow")
st.markdown(
    """
Pass the `UploadedFile` object directly into **`pd.read_csv()`**:
- Pandas accepts `io.BytesIO` streams natively without needing to write temporary files to disk.
- Wrap in a `try/except` block to catch encoding or parsing exceptions.
"""
)

col_csv1, col_csv2 = st.columns(2)

with col_csv1:
    st.markdown("**💻 Python Source Code**")
    code_csv = '''import pandas as pd

uploaded_csv = st.file_uploader(
    label="Upload Evaluation Dataset (CSV)",
    type=["csv"],
    help="Upload your custom test dataset for model evaluation"
)

if uploaded_csv is not None:
    # Read stream directly into Pandas DataFrame:
    df = pd.read_csv(uploaded_csv)
    st.success(f"Loaded {uploaded_csv.name} ({len(df):,} rows)")
    st.dataframe(df.head(), width="stretch")'''
    st.code(code_csv, language="python")

with col_csv2:
    st.markdown("**🖥️ Live Interactive CSV Uploader Demo**")
    uploaded_csv = st.file_uploader(
        label="Upload Evaluation Dataset (CSV)",
        type=["csv"],
        key="uploader_demo_csv",
        help="Upload your custom test dataset for model evaluation"
    )

    if uploaded_csv is not None:
        try:
            df_csv = pd.read_csv(uploaded_csv)
            st.success(f"✅ Ingested **{uploaded_csv.name}**: {len(df_csv):,} rows × {df_csv.shape[1]} columns")
            st.dataframe(df_csv.head(3), width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
    else:
        st.caption("No CSV uploaded yet. Drag and drop any `.csv` file above.")

st.divider()

# ==============================================================================
# 3. Uploading Excel Files (.xlsx, .xls)
# ==============================================================================
st.header("3. Uploading Excel Spreadsheets (`type=['xlsx', 'xls']`)", divider="rainbow")
st.markdown(
    """
Use **`pd.read_excel()`** with the `openpyxl` engine:
- You can inspect sheet names using `pd.ExcelFile(file).sheet_names` before loading!
- Allows users to select specific worksheet tabs from a dropdown.
"""
)

col_xls1, col_xls2 = st.columns(2)

with col_xls1:
    st.markdown("**💻 Python Source Code (Sheet Inspector & Loader)**")
    code_excel = '''uploaded_excel = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

if uploaded_excel is not None:
    # 1. Inspect sheet names:
    excel_file = pd.ExcelFile(uploaded_excel)
    selected_sheet = st.selectbox("Select Sheet", excel_file.sheet_names)
    
    # 2. Read selected worksheet:
    df_sheet = pd.read_excel(uploaded_excel, sheet_name=selected_sheet)
    st.dataframe(df_sheet.head(), width="stretch")'''
    st.code(code_excel, language="python")

with col_xls2:
    st.markdown("**🖥️ Live Interactive Excel Uploader Demo**")
    uploaded_excel = st.file_uploader(
        "Upload Financial Budget or Data (Excel)",
        type=["xlsx", "xls"],
        key="uploader_demo_excel"
    )

    if uploaded_excel is not None:
        try:
            xl = pd.ExcelFile(uploaded_excel)
            sheet = st.selectbox("Select Worksheet Tab", xl.sheet_names, key="sel_excel_sheet")
            df_xl = pd.read_excel(uploaded_excel, sheet_name=sheet)
            st.success(f"✅ Sheet `{sheet}` loaded: {len(df_xl)} rows")
            st.dataframe(df_xl.head(3), width="stretch")
        except Exception as e:
            st.error(f"Error reading Excel spreadsheet: {e}")
    else:
        st.caption("No Excel spreadsheet uploaded yet.")

st.divider()

# ==============================================================================
# 4. Uploading JSON Files
# ==============================================================================
st.header("4. Uploading JSON Configurations & Payloads (`type=['json']`)", divider="rainbow")
st.markdown(
    """
JSON files can be parsed using Python's native `json.load()` or `pd.read_json()`:
- Display structured hierarchical JSON trees using **`st.json()`**.
- Ideal for LLM prompt schemas, fine-tuning instruction sets, and API configurations.
"""
)

col_js1, col_js2 = st.columns(2)

with col_js1:
    st.markdown("**💻 Python Source Code**")
    code_json = '''import json

uploaded_json = st.file_uploader("Upload JSON File", type=["json"])

if uploaded_json is not None:
    # Parse JSON directly from byte stream:
    data = json.load(uploaded_json)
    st.success(f"Parsed JSON payload: {uploaded_json.name}")
    st.json(data)'''
    st.code(code_json, language="python")

with col_js2:
    st.markdown("**🖥️ Live Interactive JSON Uploader Demo**")
    uploaded_json = st.file_uploader(
        "Upload JSON Hyperparameter Spec",
        type=["json"],
        key="uploader_demo_json"
    )

    if uploaded_json is not None:
        try:
            json_payload = json.load(uploaded_json)
            st.success(f"✅ Valid JSON parsed from `{uploaded_json.name}`")
            st.json(json_payload)
        except Exception as e:
            st.error(f"Invalid JSON syntax: {e}")
    else:
        st.caption("No JSON configuration uploaded yet.")

st.divider()

# ==============================================================================
# 5. Uploading Text Files (.txt, .md, .py)
# ==============================================================================
st.header("5. Uploading Text Documents (`type=['txt', 'md']`)", divider="rainbow")
st.markdown(
    """
Extract raw text strings by decoding the bytes via `uploaded_file.getvalue().decode("utf-8")`:
- Display document contents in `st.text_area()`, `st.markdown()`, or `st.code()`.
- Standard entry point for feeding text chunks into RAG vectorization embedding models.
"""
)

col_txt1, col_txt2 = st.columns(2)

with col_txt1:
    st.markdown("**💻 Python Source Code**")
    code_text = '''uploaded_txt = st.file_uploader("Upload Text Document", type=["txt", "md", "py"])

if uploaded_txt is not None:
    # Decode binary byte stream to UTF-8 string:
    text_content = uploaded_txt.getvalue().decode("utf-8")
    st.text_area("Document Preview", text_content, height=180)
    st.caption(f"Word Count: {len(text_content.split()):,} words")'''
    st.code(code_text, language="python")

with col_txt2:
    st.markdown("**🖥️ Live Interactive Text Uploader Demo**")
    uploaded_txt = st.file_uploader(
        "Upload Text Document for RAG Ingestion",
        type=["txt", "md", "py"],
        key="uploader_demo_txt"
    )

    if uploaded_txt is not None:
        try:
            txt_body = uploaded_txt.getvalue().decode("utf-8")
            st.success(f"✅ Ingested: **{uploaded_txt.name}** ({len(txt_body.split()):,} words)")
            st.text_area("Extracted Text Preview", txt_body, height=140, key="txt_preview_area")
        except Exception as e:
            st.error(f"Error decoding text file: {e}")
    else:
        st.caption("No text file uploaded yet.")

st.divider()

# ==============================================================================
# 6. Uploading Images (PNG, JPG, WebP)
# ==============================================================================
st.header("6. Uploading Images for Vision AI (`type=['png', 'jpg', 'jpeg', 'webp']`)", divider="rainbow")
st.markdown(
    """
Load image byte streams into Python using **`PIL.Image.open(uploaded_file)`**:
- Inspect image metadata: `.size` (width, height), `.format` (PNG/JPEG), and `.mode` (RGB/RGBA).
- Render image on screen with **`st.image(image, caption=...)`**.
"""
)

col_img1, col_img2 = st.columns(2)

with col_img1:
    st.markdown("**💻 Python Source Code (Pillow Image Ingestion)**")
    code_image = '''from PIL import Image

uploaded_img = st.file_uploader(
    "Upload Image for Vision Inference", 
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_img is not None:
    # Open image stream via Pillow:
    img = Image.open(uploaded_img)
    st.image(img, caption=f"{uploaded_img.name} ({img.width}x{img.height})", width=350)
    st.write(f"Dimensions: {img.width} x {img.height} | Color Mode: {img.mode}")'''
    st.code(code_image, language="python")

with col_img2:
    st.markdown("**🖥️ Live Interactive Image Uploader Demo**")
    uploaded_img = st.file_uploader(
        "Upload Image for Vision Inference",
        type=["png", "jpg", "jpeg", "webp"],
        key="uploader_demo_img"
    )

    if uploaded_img is not None:
        try:
            pil_image = Image.open(uploaded_img)
            st.image(pil_image, caption=f"{uploaded_img.name} ({pil_image.width} × {pil_image.height} px)", width=320)
            st.caption(f"Format: **{pil_image.format}** | Color Mode: **{pil_image.mode}** | Size: **{uploaded_img.size:,}** bytes")
        except Exception as e:
            st.error(f"Error opening image: {e}")
    else:
        st.caption("No image uploaded yet.")

st.divider()

# ==============================================================================
# 7. Uploading PDF Documents
# ==============================================================================
st.header("7. Uploading PDF Documents & Extracting Text (`type=['pdf']`)", divider="rainbow")
st.markdown(
    """
Process PDF byte streams using libraries like **`pypdf`**:
- Read total page counts and extract text page-by-page.
- The standard foundation for document QA and PDF RAG architectures.
"""
)

col_pdf1, col_pdf2 = st.columns(2)

with col_pdf1:
    st.markdown("**💻 Python Source Code (pypdf Text Extraction)**")
    code_pdf = '''import pypdf

uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_pdf is not None:
    # Read PDF stream via pypdf.PdfReader:
    reader = pypdf.PdfReader(uploaded_pdf)
    st.write(f"Total Pages: {len(reader.pages)}")
    
    # Extract text from page 1:
    first_page_text = reader.pages[0].extract_text()
    st.text_area("Page 1 Text", first_page_text, height=150)'''
    st.code(code_pdf, language="python")

with col_pdf2:
    st.markdown("**🖥️ Live Interactive PDF Ingestion Demo**")
    uploaded_pdf = st.file_uploader(
        "Upload PDF Research Paper or Manual",
        type=["pdf"],
        key="uploader_demo_pdf"
    )

    if uploaded_pdf is not None:
        try:
            pdf_reader = pypdf.PdfReader(uploaded_pdf)
            total_pages = len(pdf_reader.pages)
            st.success(f"✅ Ingested PDF: **{uploaded_pdf.name}** ({total_pages} pages)")
            if total_pages > 0:
                p1_text = pdf_reader.pages[0].extract_text()
                st.text_area("Extracted Page 1 Text", p1_text[:500] if p1_text else "(No extractable text)", height=120)
        except Exception as e:
            st.error(f"Error parsing PDF file: {e}")
    else:
        st.caption("No PDF file uploaded yet.")

st.divider()

# ==============================================================================
# 8. Uploading Multiple Files in Batch
# ==============================================================================
st.header("8. Batch Uploading Multiple Files (`accept_multiple_files=True`)", divider="rainbow")
st.markdown(
    """
Setting **`accept_multiple_files=True`** transforms the return value into a **`list[UploadedFile]`**:
- Users can drag and drop dozens of files at once.
- Iterate over the list in a loop to process batches.
- Display aggregate metrics such as total files and total uploaded payload size.
"""
)

col_mul1, col_mul2 = st.columns(2)

with col_mul1:
    st.markdown("**💻 Python Source Code (Batch Ingestion)**")
    code_multi = '''uploaded_files = st.file_uploader(
    "Upload Batch Files",
    type=["csv", "json", "txt", "png"],
    accept_multiple_files=True  # Returns list[UploadedFile]
)

if uploaded_files:
    total_bytes = sum(f.size for f in uploaded_files)
    st.metric("Total Batch Files", len(uploaded_files))
    st.metric("Batch Payload", f"{total_bytes / 1024:.1f} KB")
    
    for f in uploaded_files:
        st.write(f"• {f.name} ({f.size} bytes)")'''
    st.code(code_multi, language="python")

with col_mul2:
    st.markdown("**🖥️ Live Batch Uploader Demo**")
    batch_files = st.file_uploader(
        "Upload Batch Training Chunks",
        type=["csv", "json", "txt", "png", "pdf"],
        accept_multiple_files=True,
        key="uploader_demo_multi"
    )

    if batch_files:
        total_size = sum(f.size for f in batch_files)
        m1, m2 = st.columns(2)
        m1.metric("Uploaded Files", len(batch_files))
        m2.metric("Total Batch Size", f"{total_size / 1024:.1f} KB")

        with st.container(height=140, border=True):
            st.markdown("##### File Inventory:")
            for f in batch_files:
                st.write(f"📄 **{f.name}** — `{f.type}` ({f.size:,} bytes)")
    else:
        st.caption("No batch files uploaded yet. Drag multiple files above.")

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 File Ingestion Cheat Sheet", divider="rainbow")
st.markdown(
    r"""
| File Type | Extension Param | Recommended Ingestion Library | Syntax |
| :--- | :--- | :--- | :--- |
| **CSV** | `type=["csv"]` | `pandas` | `df = pd.read_csv(file)` |
| **Excel** | `type=["xlsx", "xls"]` | `pandas` + `openpyxl` | `df = pd.read_excel(file, sheet_name=...)` |
| **JSON** | `type=["json"]` | `json` / `pandas` | `data = json.load(file)` |
| **Plain Text** | `type=["txt", "md"]` | Built-in | `text = file.getvalue().decode("utf-8")` |
| **Images** | `type=["png", "jpg"]`| `PIL` (Pillow) | `img = Image.open(file)` |
| **PDF** | `type=["pdf"]` | `pypdf` | `reader = pypdf.PdfReader(file)` |
| **Batch** | `accept_multiple_files=True` | Iterable list | `for file in uploaded_files:` |
"""
)
