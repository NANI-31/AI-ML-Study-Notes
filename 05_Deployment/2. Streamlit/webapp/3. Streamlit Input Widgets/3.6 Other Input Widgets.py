import streamlit as st
import pandas as pd
from datetime import date, time, timedelta
from io import StringIO

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Streamlit Other Input Widgets Tutorial",
    page_icon="🧰",
    layout="wide",
)

st.title("🧰 Other Input Widgets — Complete Master Guide")
st.markdown(
    """
Streamlit provides a rich suite of specialized input controls beyond basic text and numeric inputs. 
This master guide covers boolean switches, temporal calendar and clock pickers, visual color pickers, 
multimodal sensors (**files, camera, microphone**), human feedback controls for RLHF, 
and session state key management.
"""
)

st.divider()

# ==============================================================================
# 1. st.checkbox() — Boolean Toggles & Conditional UI
# ==============================================================================
st.header("1. Boolean Checkbox (`st.checkbox`)", divider="rainbow")
st.markdown(
    """
`st.checkbox()` displays a standard checkbox that returns a boolean (`True` or `False`).
- Commonly used for accepting terms, enabling opt-in features, and conditionally unveiling advanced settings.
- When toggled, Streamlit immediately reruns the application with the updated boolean state.
"""
)

st.markdown("#### 📐 Formal Signature: `st.checkbox()`")
st.code(
    """st.checkbox(
    label: str,
    value: bool = False,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    disabled: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible"
) -> bool""",
    language="python",
)

col_cb1, col_cb2 = st.columns(2)

with col_cb1:
    st.markdown("**💻 Python Source Code**")
    code_checkbox = '''# 1. Standard boolean checkbox
agree = st.checkbox(
    label="I accept the Terms of Service & Model Usage Policy",
    value=False,
    help="You must agree before generating AI inferences"
)

# 2. Conditional UI rendering based on checkbox
show_advanced = st.checkbox("Show Advanced LLM Hyperparameters", value=False)

if show_advanced:
    st.info("Advanced Hyperparameters Unlocked: Top-K, Repetition Penalty, Seed")
    
if agree:
    st.success("Access Granted! Inferences Enabled.")'''
    st.code(code_checkbox, language="python")

with col_cb2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    agree_val = st.checkbox(
        label="I accept the Terms of Service & Model Usage Policy",
        value=False,
        help="You must agree before generating AI inferences",
        key="demo_cb_agree"
    )
    show_advanced_val = st.checkbox(
        "Show Advanced LLM Hyperparameters", 
        value=False, 
        key="demo_cb_advanced"
    )

    if show_advanced_val:
        st.info("Advanced Hyperparameters Unlocked: Top-K = 40, Repetition Penalty = 1.15, Seed = 42")
    if agree_val:
        st.success("Access Granted! Inferences Enabled.")
    else:
        st.caption("Status: Terms not yet accepted.")

st.divider()

# ==============================================================================
# 2. st.toggle() — Modern Switch Widgets
# ==============================================================================
st.header("2. Modern Toggle Switch (`st.toggle`)", divider="rainbow")
st.markdown(
    """
`st.toggle()` provides a sleek modern switch widget with identical boolean return semantics (`True` / `False`).
- **Semantic Distinction**: Checkboxes typically denote selections within a list or form confirmation. 
  Toggles signify **immediate binary states** (e.g., turning hardware acceleration on/off, streaming mode, or debug telemetry).
"""
)

col_tog1, col_tog2 = st.columns(2)

with col_tog1:
    st.markdown("**💻 Python Source Code**")
    code_toggle = '''# Toggle switches for live operational modes:
stream_mode = st.toggle("Enable Token Streaming", value=True)
gpu_accel = st.toggle("GPU CUDA Hardware Acceleration", value=False)
debug_mode = st.toggle("Verbose Debug Telemetry Logs", value=False)

st.write({
    "streaming": stream_mode,
    "cuda_active": gpu_accel,
    "debug_level": "DEBUG" if debug_mode else "INFO"
})'''
    st.code(code_toggle, language="python")

with col_tog2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    stream_val = st.toggle("Enable Token Streaming", value=True, key="demo_tog_stream")
    gpu_val = st.toggle("GPU CUDA Hardware Acceleration", value=False, key="demo_tog_gpu")
    debug_val = st.toggle("Verbose Debug Telemetry Logs", value=False, key="demo_tog_debug")

    status_pill = "🟢 Active" if gpu_val else "⚪ Disabled"
    st.caption(f"CUDA Status: **{status_pill}** | Streaming: **{'ON' if stream_val else 'OFF'}** | Log Level: **{'DEBUG' if debug_val else 'INFO'}**")

st.divider()

# ==============================================================================
# 3. st.date_input() — Date Pickers & Range Intervals
# ==============================================================================
st.header("3. Date Input Widget (`st.date_input`)", divider="rainbow")
st.markdown(
    """
`st.date_input()` renders an interactive popup calendar:
- **Single Date**: Pass a single `datetime.date` or `"today"` as `value` (returns `datetime.date`).
- **Date Range**: Pass a **2-tuple of dates** `(start_date, end_date)` to allow range selection (returns `tuple[date, date]`).
- Supports `min_value`, `max_value`, and custom formatting tokens (e.g. `format="YYYY/MM/DD"`).
"""
)

st.markdown("#### 📐 Formal Signature: `st.date_input()`")
st.code(
    """st.date_input(
    label: str,
    value: date | tuple[date, date] | Sequence[date] | "today" = "today",
    min_value: date | None = None,
    max_value: date | None = None,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    format: str = "YYYY/MM/DD",
    disabled: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible"
) -> date | tuple[date, date] | Sequence[date]""",
    language="python",
)

col_dt1, col_dt2 = st.columns(2)

with col_dt1:
    st.markdown("**💻 Python Source Code**")
    code_date = '''from datetime import date, timedelta

# 1. Single Date Picker:
selected_date = st.date_input(
    label="Model Checkpoint Deployment Date",
    value=date.today(),
    min_value=date(2025, 1, 1),
    max_value=date(2026, 12, 31),
    format="YYYY/MM/DD"
)

# 2. Date Range Picker (Interval):
date_range = st.date_input(
    label="Training Dataset Ingestion Window",
    value=(date(2026, 1, 1), date(2026, 6, 30)),
    min_value=date(2025, 1, 1),
    max_value=date(2026, 12, 31)
)'''
    st.code(code_date, language="python")

with col_dt2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    sel_date = st.date_input(
        label="Model Checkpoint Deployment Date",
        value=date.today(),
        min_value=date(2025, 1, 1),
        max_value=date(2026, 12, 31),
        format="YYYY/MM/DD",
        key="demo_date_single"
    )
    sel_range = st.date_input(
        label="Training Dataset Ingestion Window",
        value=(date(2026, 1, 1), date(2026, 6, 30)),
        min_value=date(2025, 1, 1),
        max_value=date(2026, 12, 31),
        key="demo_date_range"
    )
    st.caption(f"Selected Checkpoint: **{sel_date}**")
    if isinstance(sel_range, tuple) and len(sel_range) == 2:
        st.caption(f"Ingestion Range: **{sel_range[0]}** to **{sel_range[1]}** (Span: **{(sel_range[1] - sel_range[0]).days}** days)")

st.divider()

# ==============================================================================
# 4. st.time_input() — Precise Time Pickers
# ==============================================================================
st.header("4. Time Input Widget (`st.time_input`)", divider="rainbow")
st.markdown(
    """
`st.time_input()` renders an interactive clock selector:
- Returns a `datetime.time` object.
- **`step`**: Granularity interval specified as `datetime.timedelta` (default is 15 minutes = `timedelta(minutes=15)`).
- **`format`**: Options include `"24h"` (e.g. `14:30`) or `"12h"` (e.g. `02:30 PM`).
"""
)

col_tm1, col_tm2 = st.columns(2)

with col_tm1:
    st.markdown("**💻 Python Source Code**")
    code_time = '''from datetime import time, timedelta

# Scheduled time picker with 30-minute step increments:
cron_time = st.time_input(
    label="Daily Model Retraining Scheduled Trigger",
    value=time(2, 30),  # 02:30 AM UTC
    step=timedelta(minutes=30),
    help="Time in UTC when daily retraining pipeline initiates"
)

st.write(f"Trigger set for: **{cron_time.strftime('%H:%M UTC')}**")'''
    st.code(code_time, language="python")

with col_tm2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    cron_val = st.time_input(
        label="Daily Model Retraining Scheduled Trigger",
        value=time(2, 30),
        step=timedelta(minutes=30),
        help="Time in UTC when daily retraining pipeline initiates",
        key="demo_time_cron"
    )
    st.caption(f"Configured Trigger Time: **{cron_val.strftime('%H:%M UTC')}** ({cron_val.strftime('%I:%M %p')})")

st.divider()

# ==============================================================================
# 5. st.color_picker() — Visual Color Selection
# ==============================================================================
st.header("5. Visual Color Picker (`st.color_picker`)", divider="rainbow")
st.markdown(
    """
`st.color_picker()` displays a native color wheel and palette modal:
- Returns a standard lowercase hex color code string (e.g., `"#ff4b4b"`).
- Ideal for customizing visualization chart themes, UI branding, and computer vision bounding box colors.
"""
)

col_cp1, col_cp2 = st.columns(2)

with col_cp1:
    st.markdown("**💻 Python Source Code**")
    code_color = '''# Select UI accent color:
chart_color = st.color_picker(
    label="Chart Accent & Bounding Box Color",
    value="#00A67E",  # OpenAI Mint Green default
    help="Hex code applied to plots and segmentation overlays"
)

# Dynamic HTML preview banner:
st.html(
    f"""
    <div style="background-color: {chart_color}; padding: 12px; border-radius: 8px; color: white; font-weight: bold; text-align: center;">
        Active Accent Theme: {chart_color}
    </div>
    """
)'''
    st.code(code_color, language="python")

with col_cp2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    picked_color = st.color_picker(
        label="Chart Accent & Bounding Box Color",
        value="#00A67E",
        help="Hex code applied to plots and segmentation overlays",
        key="demo_color_picker"
    )
    st.html(
        f"""
        <div style="background-color: {picked_color}; padding: 12px; border-radius: 8px; color: white; font-weight: bold; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            Active Accent Theme: {picked_color}
        </div>
        """
    )
    st.caption(f"Hex Return Value: `{picked_color}`")

st.divider()

# ==============================================================================
# 6. st.file_uploader() — Dataset & Document Ingestion
# ==============================================================================
st.header("6. File Uploader (`st.file_uploader`)", divider="rainbow")
st.markdown(
    """
`st.file_uploader()` allows users to upload local files directly into the Streamlit session:
- **`type`**: Constrain accepted file extensions (e.g., `["csv", "json", "txt", "png"]`).
- **`accept_multiple_files`**: When `True`, returns a list of `UploadedFile` objects.
- **UploadedFile Interface**: Inherits from `io.BytesIO`. Provides `.name`, `.size`, `.type`, `.read()`, and `.getvalue()`.
"""
)

st.markdown("#### 📐 Formal Signature: `st.file_uploader()`")
st.code(
    """st.file_uploader(
    label: str,
    type: str | Sequence[str] | None = None,
    accept_multiple_files: bool = False,
    key: str | int | None = None,
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *,
    disabled: bool = False,
    label_visibility: "visible" | "hidden" | "collapsed" = "visible"
) -> UploadedFile | list[UploadedFile] | None""",
    language="python",
)

col_fu1, col_fu2 = st.columns(2)

with col_fu1:
    st.markdown("**💻 Python Source Code**")
    code_uploader = '''import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader(
    label="Upload Dataset (CSV or JSON)",
    type=["csv", "json"],
    help="Upload your custom test dataset for batch evaluation"
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), width="stretch")'''
    st.code(code_uploader, language="python")

with col_fu2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    uploaded_file = st.file_uploader(
        label="Upload Dataset (CSV or JSON)",
        type=["csv", "json", "txt"],
        help="Upload your custom test dataset for batch evaluation",
        key="demo_file_uploader"
    )

    if uploaded_file is not None:
        st.success(f"Received: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
        if uploaded_file.name.endswith(".csv"):
            try:
                sample_df = pd.read_csv(uploaded_file)
                st.dataframe(sample_df.head(3), width="stretch")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        elif uploaded_file.name.endswith(".txt"):
            content = uploaded_file.getvalue().decode("utf-8")
            st.text_area("File Contents Preview", content[:300], height=100)
    else:
        st.caption("No file uploaded yet. Drag and drop any `.csv` or `.txt` file above.")

st.divider()

# ==============================================================================
# 7. st.camera_input() — Webcam Capture for Computer Vision
# ==============================================================================
st.header("7. Camera Input (`st.camera_input`)", divider="rainbow")
st.markdown(
    """
`st.camera_input()` enables direct webcam snapshot capture right from the user's browser:
- Prompts browser permissions to access the webcam.
- User clicks "Take Photo" to capture a high-resolution snapshot.
- Returns an `UploadedFile` object containing standard JPEG/PNG image bytes.
- Essential for live facial recognition, OCR invoice scanning, and real-time object classification.
"""
)

col_cam1, col_cam2 = st.columns(2)

with col_cam1:
    st.markdown("**💻 Python Source Code**")
    code_camera = '''# Capture live photo from webcam:
picture = st.camera_input(
    label="Take Snapshot for Vision Model Inference",
    help="Ensure good lighting for optimal classification"
)

if picture:
    st.success("Snapshot Captured!")
    # Can be read into PIL Image or OpenCV:
    # img = Image.open(picture)
    st.image(picture, caption="Captured Image Preview", width=300)'''
    st.code(code_camera, language="python")

with col_cam2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    cam_picture = st.camera_input(
        label="Take Snapshot for Vision Model Inference",
        help="Ensure good lighting for optimal classification",
        key="demo_cam_input"
    )

    if cam_picture:
        st.success("Snapshot captured successfully!")
        st.image(cam_picture, caption=f"Snapshot ({cam_picture.size} bytes)", width=300)
    else:
        st.caption("Camera inactive. Click 'Take Photo' above to capture a snapshot.")

st.divider()

# ==============================================================================
# 8. st.audio_input() — Microphone Voice Recording
# ==============================================================================
st.header("8. Microphone Audio Input (`st.audio_input`)", divider="rainbow")
st.markdown(
    """
`st.audio_input()` captures live voice audio recordings directly from the user's microphone:
- Prompts browser permissions to record microphone audio.
- User clicks record, speaks their query, and stops recording.
- Returns an `UploadedFile` object containing raw WAV audio bytes.
- Ideal for **Whisper speech-to-text transcription**, voice AI bots, and audio sentiment analysis.
"""
)

col_aud1, col_aud2 = st.columns(2)

with col_aud1:
    st.markdown("**💻 Python Source Code**")
    code_audio = '''# Record audio prompt from microphone:
audio_recording = st.audio_input(
    label="Record Voice Prompt for Speech-to-Text Model",
    help="Click record to capture audio from your microphone"
)

if audio_recording:
    st.success("Audio Recorded!")
    st.audio(audio_recording)
    # audio_bytes = audio_recording.read()
    # transcript = whisper_model.transcribe(audio_bytes)'''
    st.code(code_audio, language="python")

with col_aud2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    audio_val = st.audio_input(
        label="Record Voice Prompt for Speech-to-Text Model",
        help="Click record to capture audio from your microphone",
        key="demo_audio_input"
    )

    if audio_val:
        st.success(f"Audio captured! ({audio_val.size:,} bytes)")
        st.audio(audio_val)
    else:
        st.caption("Microphone inactive. Click record above to capture voice audio.")

st.divider()

# ==============================================================================
# 9. st.feedback() — RLHF & Human-in-the-Loop Ratings
# ==============================================================================
st.header("9. Human Feedback Widget (`st.feedback`)", divider="rainbow")
st.markdown(
    """
`st.feedback()` displays an icon-based rating control designed for **Reinforcement Learning from Human Feedback (RLHF)**:
- **`options="thumbs"`**: 2 icons (`0` = 👎 Thumbs Down, `1` = 👍 Thumbs Up).
- **`options="faces"`**: 5 sentiment faces (indices `0` to `4` from 😞 to 😄).
- **`options="stars"`**: 5 rating stars (indices `0` to `4` corresponding to 1 to 5 stars).
- Returns `int | None` (or `None` if the user hasn't rated yet).
"""
)

col_fb1, col_fb2 = st.columns(2)

with col_fb1:
    st.markdown("**💻 Python Source Code**")
    code_feedback = '''# 1. Thumbs feedback for LLM response:
thumbs_rating = st.feedback("thumbs", key="fb_thumbs")

# 2. 5-star rating for overall generation quality:
star_rating = st.feedback("stars", key="fb_stars")

# 3. Sentiment faces for user satisfaction:
faces_rating = st.feedback("faces", key="fb_faces")

if thumbs_rating is not None:
    label = "Positive 👍" if thumbs_rating == 1 else "Negative 👎"
    st.write(f"Logged RLHF signal: **{label}**")'''
    st.code(code_feedback, language="python")

with col_fb2:
    st.markdown("**🖥️ Rendered Streamlit Output**")
    st.markdown("##### AI Assistant Response Quality:")
    st.caption("Did the model provide an accurate answer?")
    fb_thumbs = st.feedback("thumbs", key="demo_fb_thumbs")

    st.markdown("##### 5-Star Generation Rating:")
    fb_stars = st.feedback("stars", key="demo_fb_stars")

    st.markdown("##### User Satisfaction Level:")
    fb_faces = st.feedback("faces", key="demo_fb_faces")

    feedback_summary = []
    if fb_thumbs is not None:
        feedback_summary.append(f"Thumbs: **{'👍 (+1)' if fb_thumbs == 1 else '👎 (-1)'}**")
    if fb_stars is not None:
        feedback_summary.append(f"Stars: **{fb_stars + 1}/5 ⭐**")
    if fb_faces is not None:
        feedback_summary.append(f"Face Index: **{fb_faces}/4**")

    if feedback_summary:
        st.info(" | ".join(feedback_summary))
    else:
        st.caption("No feedback submitted yet. Click any icon above to log RLHF data.")

st.divider()

# ==============================================================================
# 10. Widget Keys and Default Values — State Management Deep-Dive
# ==============================================================================
st.header("10. Widget Keys & Default Values (`st.session_state`)", divider="rainbow")
st.markdown(
    """
Every input widget in Streamlit can take an optional `key` parameter. Understanding keys is essential for mastering state:
1. **Implicit vs Explicit Keys**: If omitted, Streamlit hashes the widget type and parameters to generate an internal key.
2. **Session State Registration**: Providing `key="my_key"` automatically exposes the widget's current value at `st.session_state["my_key"]`.
3. **State Pre-Initialization**: Setting `st.session_state["my_key"] = default_value` *before* the widget renders initializes its default value without relying on widget parameter defaults.
4. **Duplicate Key Constraint**: Keys must be globally unique across all active widgets on the page; sharing keys raises `DuplicateWidgetID`.
5. **Programmatic Reset Pattern**: Modify `st.session_state[key]` inside a button callback to reset or override inputs programmatically.
"""
)

# Demo of session state pre-initialization and programmatic reset
if "active_seed" not in st.session_state:
    st.session_state["active_seed"] = 42

def reset_settings_callback():
    st.session_state["active_seed"] = 1337
    st.session_state["user_profile_name"] = "Default_Researcher"

col_wk1, col_wk2 = st.columns(2)

with col_wk1:
    st.markdown("**💻 Python Source Code (Key & State Management)**")
    code_keys = '''# 1. State pre-initialization:
if "active_seed" not in st.session_state:
    st.session_state["active_seed"] = 42

# 2. Callback function modifying session state:
def reset_to_defaults():
    st.session_state["active_seed"] = 1337
    st.session_state["user_profile_name"] = "Default_Researcher"

# 3. Widgets tied directly to keys:
st.number_input("Random Generator Seed", key="active_seed")
st.text_input("Experimenter Name", key="user_profile_name", value="Dr. Ada")

# 4. Trigger reset via button:
st.button("Reset Settings to Standard Defaults", on_click=reset_to_defaults)'''
    st.code(code_keys, language="python")

with col_wk2:
    st.markdown("**🖥️ Live Interactive State Management Demo**")
    
    st.number_input(
        "Random Generator Seed", 
        min_value=0, 
        max_value=99999, 
        step=1, 
        key="active_seed"
    )
    st.text_input(
        "Experimenter Name", 
        value="Dr. Ada Lovelace", 
        key="user_profile_name"
    )
    
    st.button("Reset Settings to Standard Defaults", on_click=reset_settings_callback, key="btn_reset_state")
    
    st.markdown("##### 🔍 Current `st.session_state` Snapshot:")
    st.json({
        "active_seed": st.session_state.get("active_seed"),
        "user_profile_name": st.session_state.get("user_profile_name")
    })

st.divider()

# ==============================================================================
# Summary Reference Table
# ==============================================================================
st.header("📋 Master Input Widgets Reference", divider="rainbow")
st.markdown(
    """
| Widget | Return Type | Primary AI / Data Science Application |
| :--- | :--- | :--- |
| **`st.checkbox`** | `bool` | Terms confirmation, conditional display of advanced hyperparameter panels |
| **`st.toggle`** | `bool` | Real-time switch for CUDA acceleration, live token streaming, or verbose logs |
| **`st.date_input`** | `date` or `(date, date)` | Dataset temporal ingestion windows, model evaluation cutoff dates |
| **`st.time_input`** | `time` | Cron job trigger time, automated batch pipeline inference scheduling |
| **`st.color_picker`** | `str` (Hex) | Customizing chart accent colors, confusion matrix heatmaps, bounding box overlays |
| **`st.file_uploader`** | `UploadedFile` / `list` | Uploading CSV/Parquet datasets, PDF documents for RAG indexing |
| **`st.camera_input`** | `UploadedFile` (Image) | Real-time computer vision snapshots (face detection, OCR document scan) |
| **`st.audio_input`** | `UploadedFile` (Audio) | Microphone speech recordings for Whisper speech-to-text dictation |
| **`st.feedback`** | `int` or `None` | RLHF reward signaling (`thumbs`, `faces`, `stars`) on LLM outputs |
| **`Widget Keys`** | `st.session_state[k]` | Global state persistence, cross-rerun memory, and programmatic resets |
"""
)
