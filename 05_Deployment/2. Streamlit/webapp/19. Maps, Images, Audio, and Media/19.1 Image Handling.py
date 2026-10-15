import streamlit as st
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="19.1 Image Handling",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.html(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    </style>
    """
)

st.markdown('<div class="main-title">19.1 Image Handling in Streamlit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Comprehensive architectural guide to st.image(), local and remote assets, file upload pipelines, container sizing, gallery layouts, Pillow (PIL) transformations, and computer vision previews.</div>',
    unsafe_allow_html=True
)

# Helper function to generate synthetic test images in-memory
def create_sample_cv_image(label="CV Input", color=(30, 58, 138), size=(400, 300)):
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    # Draw geometric patterns resembling vision bounding boxes
    draw.rectangle([30, 30, size[0] - 30, size[1] - 30], outline=(245, 158, 11), width=3)
    draw.rectangle([60, 60, 220, 180], outline=(16, 185, 129), width=2)
    draw.text((70, 70), f"Obj: {label} (0.94)", fill=(255, 255, 255))
    draw.ellipse([260, 100, 340, 180], outline=(239, 68, 68), width=2)
    draw.text((270, 110), "Target", fill=(255, 255, 255))
    return img

# -----------------------------------------------------------------------------
# TOPIC 1: ST.IMAGE() FUNDAMENTALS & PARAMETER SIGNATURE
# -----------------------------------------------------------------------------
st.header("1. st.image() Fundamentals & Parameter Signature", divider="blue")

st.markdown(
    r"""
`st.image()` is Streamlit's primary media primitive for rendering raster and vector imagery.
It accepts numpy arrays, raw bytes, PIL Images, file paths, and web URLs.

```python
st.image(
    image,
    caption=None,
    width=None,
    use_container_width=None,  # Deprecated in Streamlit 1.60+ in favor of width='stretch'
    clamp=False,
    channels="RGB",
    output_format="auto"
)
```
"""
)

st.markdown(
    r"""
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `Image`, `np.ndarray`, `bytes`, `str`, `list` | *Required* | Image source (PIL Image, array, URL, filepath, or list of images). |
| `caption` | `str` or `list[str]` | `None` | Text or markdown caption placed directly beneath the image. |
| `width` | `int` or `str` | `None` | Display width in pixels (`int`) or `"stretch"` / `"auto"`. |
| `clamp` | `bool` | `False` | Clamps pixel values to `[0, 255]` for integer arrays or `[0.0, 1.0]` for float arrays. |
| `channels` | `str` | `"RGB"` | Channel order: `"RGB"` or `"BGR"` (essential for OpenCV arrays). |
| `output_format` | `str` | `"auto"` | Serialization format: `"PNG"`, `"JPEG"`, or `"auto"`. |
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
import numpy as np

# 1. Direct NumPy array rendering (Synthetic feature map)
feature_map = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

st.image(
    feature_map,
    caption="Synthetic Latent Embedding Map (200x200 RGB)",
    channels="RGB",
    output_format="PNG"
)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("NumPy Feature Map Pipeline")
        np.random.seed(42)
        demo_arr = np.random.randint(0, 256, (180, 280, 3), dtype=np.uint8)
        st.image(
            demo_arr,
            caption="Synthetic Latent Embedding Map (180x280 RGB)",
            channels="RGB",
            output_format="PNG"
        )
        st.info("💡 `channels='BGR'` can be used for OpenCV matrices without `cv2.cvtColor()`.")

# -----------------------------------------------------------------------------
# TOPIC 2: DISPLAYING LOCAL IMAGES
# -----------------------------------------------------------------------------
st.header("2. Displaying Local Images", divider="blue")

st.markdown(
    r"""
When referencing local assets, Streamlit supports relative filesystem paths, absolute filesystem paths, 
and reading file handles via binary streaming. Using relative paths directly prevents environment portability errors.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
from pathlib import Path
from PIL import Image

# Method A: Path string directly
st.image("assets/logo.png", caption="Static Workspace Asset")

# Method B: In-memory byte streaming
with open("assets/diagram.png", "rb") as f:
    img_bytes = f.read()
st.image(img_bytes, caption="Buffered Byte Stream")

# Method C: PIL Image instance
pil_img = Image.open("assets/model_architecture.jpg")
st.image(pil_img, caption="PIL Object Asset")
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Local Asset Loader Simulation")
        
        sim_asset = st.selectbox(
            "Select Local Architecture Diagram",
            ["ConvNeXt Backbone (Local Path)", "Vision Transformer Patch Grid (PIL Handle)"],
            key="local_asset_sel"
        )
        
        if "ConvNeXt" in sim_asset:
            sample_img = create_sample_cv_image("ConvNeXt-Large", color=(15, 23, 42))
            st.image(sample_img, caption="Local Asset: convnext_weights_v2.png", width="stretch")
        else:
            sample_img = create_sample_cv_image("ViT-B/16 Patches", color=(30, 41, 59))
            st.image(sample_img, caption="Local Asset: vit_b16_attention_heads.png", width="stretch")

# -----------------------------------------------------------------------------
# TOPIC 3: DISPLAYING UPLOADED IMAGES
# -----------------------------------------------------------------------------
st.header("3. Displaying Uploaded Images", divider="blue")

st.markdown(
    r"""
Integrating `st.file_uploader()` with `st.image()` enables interactive user workflows for computer vision inference.
`st.file_uploader` returns an `UploadedFile` buffer inheriting from `io.BytesIO`, which can be passed directly to 
`st.image()` or opened via `Image.open()`.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
from PIL import Image

uploaded_file = st.file_uploader(
    "Choose an image for inspection",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    # 1. Direct pass-through to st.image
    st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}")
    
    # 2. Extract dimensions and channels via PIL
    image = Image.open(uploaded_file)
    st.write(f"Dimensions: {image.width}x{image.height} | Format: {image.format}")
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Interactive Upload Pipeline Simulator")
        
        upload_choice = st.radio(
            "Evaluation Input Stream",
            ["Use Synthetic Upload Sample", "Manual File Upload"],
            horizontal=True,
            key="upload_stream_choice"
        )
        
        if upload_choice == "Manual File Upload":
            user_up = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="manual_up_cv")
            if user_up:
                pil_up = Image.open(user_up)
                st.image(pil_up, caption=f"File: {user_up.name} ({pil_up.width}x{pil_up.height})", width="stretch")
            else:
                st.info("Awaiting user file upload...")
        else:
            synth_pil = create_sample_cv_image("Medical CT Slice", color=(17, 24, 39), size=(360, 240))
            st.image(synth_pil, caption="Synthetic Upload Buffer (medical_scan_04.png)", width="stretch")
            c1, c2, c3 = st.columns(3)
            c1.metric("Width", f"{synth_pil.width} px")
            c2.metric("Height", f"{synth_pil.height} px")
            c3.metric("Channels", f"{len(synth_pil.getbands())} ({''.join(synth_pil.getbands())})")

# -----------------------------------------------------------------------------
# TOPIC 4: DISPLAYING IMAGE URLS
# -----------------------------------------------------------------------------
st.header("4. Displaying Image URLs", divider="blue")

st.markdown(
    r"""
Streamlit seamlessly fetches and streams remote images when passed a standard HTTP/HTTPS URL.
This avoids storing large media files locally in repository checkouts.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st

# Direct HTTPS image URL
remote_url = "https://images.unsplash.com/photo-1579783902614-a3fb3927b675?w=600"

st.image(
    remote_url,
    caption="Remote Asset fetched from Unsplash CDN",
    width="stretch"
)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("CDN Remote Fetcher")
        
        url_input = st.text_input(
            "Remote Image Endpoint URL",
            value="https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=500&auto=format&fit=crop&q=60",
            key="remote_url_inp"
        )
        
        if st.button("Fetch and Render URL", key="fetch_url_btn", width="stretch"):
            with st.spinner("Fetching image headers from remote host..."):
                st.image(
                    url_input,
                    caption="Fetched from Remote CDN (500px width)",
                    width="stretch"
                )
                st.success("HTTP 200 OK — Remote media stream successfully decoded.")

# -----------------------------------------------------------------------------
# TOPIC 5: IMAGE SIZING & RESPONSIVE CONTAINERS
# -----------------------------------------------------------------------------
st.header("5. Image Sizing & Responsive Containers", divider="blue")

st.markdown(
    r"""
Streamlit 1.60+ introduces modern container sizing tokens:
- **`width="stretch"`**: Expands the image to occupy 100% of the parent column or container width (replaces legacy `use_container_width=True`).
- **`width="auto"`**: Retains native image dimensions without scaling up.
- **`width=int`**: Explicitly fixes pixel width (e.g., `width=300`), maintaining aspect ratio automatically.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st

# Mode 1: Exact Pixel Width
st.image("sample.png", width=250, caption="Fixed 250px")

# Mode 2: Full Responsive Container Stretch (Modern 1.60+ Standard)
st.image("sample.png", width="stretch", caption="Stretch to Container")

# Mode 3: Native Aspect Ratio Auto Sizing
st.image("sample.png", width="auto", caption="Auto Width")
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Dynamic Sizing Engine")
        
        sizing_mode = st.radio(
            "Width Sizing Strategy",
            ["Fixed Width (Pixels)", "Responsive Width ('stretch')", "Native Auto ('auto')"],
            horizontal=True,
            key="sizing_mode_radio"
        )
        
        synth_sizing = create_sample_cv_image("Sizing Test", color=(76, 29, 149), size=(400, 200))
        
        if sizing_mode == "Fixed Width (Pixels)":
            px_val = st.slider("Select Pixel Width", 100, 500, 260, step=20, key="px_width_slider")
            st.image(synth_sizing, width=px_val, caption=f"Fixed Width: {px_val}px")
        elif sizing_mode == "Responsive Width ('stretch')":
            st.image(synth_sizing, width="stretch", caption="Responsive Stretch (100% of container)")
        else:
            st.image(synth_sizing, width="auto", caption="Auto Sizing (Intrinsic 400px)")

# -----------------------------------------------------------------------------
# TOPIC 6: MULTIPLE IMAGE LAYOUTS & GALLERIES
# -----------------------------------------------------------------------------
st.header("6. Multiple Image Layouts & Galleries", divider="blue")

st.markdown(
    r"""
For model comparison dashboards, computer vision dataset inspection, and ablation comparisons,
images can be arranged in multi-column grids or passed as a list to `st.image([img1, img2, ...])`.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st

# Method A: Multi-column side-by-side grid
cols = st.columns(3)
for idx, col in enumerate(cols):
    with col:
        st.image(images[idx], caption=f"Model Checkpoint #{idx+1}", width="stretch")

# Method B: Native list batch rendering
st.image(
    [img1, img2, img3],
    caption=["Original Input", "Ground Truth Mask", "Model Prediction"],
    width=200
)
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Computer Vision Dataset Grid (3 Columns)")
        
        gallery_cols = st.columns(3)
        colors = [(15, 23, 42), (88, 28, 135), (19, 78, 74)]
        labels = ["Class: Vehicle", "Class: Pedestrian", "Class: Traffic Light"]
        
        for idx, col in enumerate(gallery_cols):
            with col:
                g_img = create_sample_cv_image(labels[idx].split(": ")[1], color=colors[idx], size=(240, 180))
                st.image(g_img, caption=labels[idx], width="stretch")
                st.caption(f"Score: {0.91 + idx*0.03:.2f}")

# -----------------------------------------------------------------------------
# TOPIC 7: IMAGE PROCESSING WITH PILLOW
# -----------------------------------------------------------------------------
st.header("7. Image Processing with Pillow", divider="blue")

st.markdown(
    r"""
Streamlit integrates seamlessly with Python's Pillow (`PIL`) library for in-memory data augmentation,
color channel transformations, edge detection, filtering, and normalization prior to model inference.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''from PIL import Image, ImageOps, ImageFilter, ImageEnhance

# 1. Grayscale conversion
gray_img = ImageOps.grayscale(image)

# 2. Gaussian blur / Edge detection filter
edge_img = image.filter(ImageFilter.FIND_EDGES)

# 3. Contrast adjustment
enhancer = ImageEnhance.Contrast(image)
contrast_img = enhancer.enhance(factor=1.8)

# 4. Dimension resizing for CNN input
tensor_ready = image.resize((224, 224), Image.Resampling.LANCZOS)
st.image(tensor_ready, caption="Tensor Ready (224x224)")
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Interactive PIL Vision Pipeline")
        
        filter_type = st.selectbox(
            "Select PIL Transformation",
            ["Original", "Grayscale (ImageOps.grayscale)", "Find Edges (ImageFilter.FIND_EDGES)", "High Contrast (ImageEnhance)", "Invert Colors (ImageOps.invert)"],
            key="pil_filter_sel"
        )
        
        base_img = create_sample_cv_image("PIL Demo", color=(30, 58, 138), size=(360, 220))
        
        if filter_type == "Original":
            out_img = base_img
        elif filter_type == "Grayscale (ImageOps.grayscale)":
            out_img = ImageOps.grayscale(base_img)
        elif filter_type == "Find Edges (ImageFilter.FIND_EDGES)":
            out_img = base_img.filter(ImageFilter.FIND_EDGES)
        elif filter_type == "High Contrast (ImageEnhance)":
            enhancer = ImageEnhance.Contrast(base_img)
            out_img = enhancer.enhance(2.5)
        else:
            out_img = ImageOps.invert(base_img)
            
        c_orig, c_proc = st.columns(2)
        with c_orig:
            st.image(base_img, caption="Input Tensor (Base)", width="stretch")
        with c_proc:
            st.image(out_img, caption=f"Processed: {filter_type.split()[0]}", width="stretch")

# -----------------------------------------------------------------------------
# TOPIC 8: IMAGE PREVIEWS & THUMBNAILS
# -----------------------------------------------------------------------------
st.header("8. Image Previews & Thumbnails", divider="blue")

st.markdown(
    r"""
High-resolution images consume substantial bandwidth and memory. For rapid UI previews,
generate low-resolution thumbnails using `image.thumbnail((width, height))` or cache pre-rendered 
downsampled previews with `@st.cache_data`.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 Python Source Code")
    st.code(
        r'''import streamlit as st
from PIL import Image

@st.cache_data(show_spinner=False)
def generate_thumbnail(image_bytes: bytes, max_size=(120, 120)) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

# Display thumbnail in compact view, full image in modal / expander
thumb = generate_thumbnail(raw_bytes)
st.image(thumb, width=120, caption="Compact Preview")

with st.expander("🔍 View Full High-Resolution Image"):
    st.image(raw_bytes, width="stretch", caption="Full 4K Image")
''',
        language="python"
    )

with col2:
    st.subheader("🖥️ Rendered Streamlit Output")
    with st.container(border=True):
        st.caption("Thumbnail & Full-Resolution Inspection Hub")
        
        thumb_size = st.select_slider(
            "Thumbnail Max Bounding Box",
            options=[64, 96, 128, 160, 200],
            value=128,
            key="thumb_slider_size"
        )
        
        full_res_img = create_sample_cv_image("High-Res Asset", color=(2, 44, 34), size=(800, 600))
        
        # In-memory thumbnail calculation
        thumb_copy = full_res_img.copy()
        thumb_copy.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        
        col_th, col_meta = st.columns([1, 2])
        with col_th:
            st.image(thumb_copy, caption=f"Thumbnail ({thumb_copy.width}x{thumb_copy.height})")
        with col_meta:
            st.write(f"**Original Size:** {full_res_img.width}x{full_res_img.height} px")
            st.write(f"**Thumbnail Size:** {thumb_copy.width}x{thumb_copy.height} px")
            st.write(f"**Compression Factor:** ~{(full_res_img.width*full_res_img.height)/(thumb_copy.width*thumb_copy.height):.1f}x memory reduction")
            
        with st.expander("🔍 Open Full High-Resolution 800x600 Inspection View"):
            st.image(full_res_img, caption="Uncompressed Full Resolution View", width="stretch")

# -----------------------------------------------------------------------------
# SUMMARY & BEST PRACTICES
# -----------------------------------------------------------------------------
st.header("9. Image Handling Architecture Best Practices", divider="blue")

st.markdown(
    r"""
| Best Practice | Rationale | Recommended Streamlit Implementation |
| :--- | :--- | :--- |
| **Use `width="stretch"`** | Replaces deprecated `use_container_width=True` to guarantee responsive behavior across mobile & desktop. | `st.image(img, width="stretch")` |
| **Pre-downsample previews** | High-resolution assets cause slow page re-runs and browser memory bloat. | `img.thumbnail((200, 200))` cached via `@st.cache_data` |
| **Verify Channel Order** | OpenCV arrays default to `BGR`, resulting in inverted colors unless corrected. | `st.image(cv2_mat, channels="BGR")` |
| **Use BytesIO for Pipeline Chaining** | Avoids unnecessary disk writes when performing multi-stage PIL transformations. | `buf = io.BytesIO(); img.save(buf, format="PNG")` |
| **Streamlit UploadedFile is BytesIO** | No temporary files are needed on disk to feed PIL or torchvision models. | `Image.open(uploaded_file)` |
"""
)
