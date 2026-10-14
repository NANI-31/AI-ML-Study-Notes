# Matplotlib Imshow & Heatmaps — Test 01
## imshow Fundamentals & Image Data Shapes

---

### Question 1

A developer loads a grayscale image using PIL and displays it with `imshow`:

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("scan.png").convert("L")
arr = np.array(img)
print(arr.shape, arr.dtype)

fig, ax = plt.subplots()
ax.imshow(arr)
ax.set_title("Medical Scan")
plt.show()
```

The array shape is `(512, 512)` and dtype is `uint8`. What colormap does Matplotlib apply by default, and how can the developer switch to a clinical grayscale rendering?

- [ ] A) Default is `viridis`; switch with `ax.imshow(arr, cmap='gray')`
- [ ] B) Default is `gray`; no change needed — grayscale is already applied for 2D arrays
- [ ] C) Default is `jet`; switch with `ax.imshow(arr, cmap='Greys')`
- [ ] D) Default is `gray`; switch with `ax.imshow(arr, colormap='gray')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Default is `viridis`; switch with `ax.imshow(arr, cmap='gray')`

**Explanation:**

- **A is correct:** For a 2D scalar array `(M, N)`, Matplotlib applies the default colormap — which is `viridis` (since Matplotlib 2.0). The `viridis` colormap maps scalar values to blue-green-yellow, so the grayscale scan appears in false color. To render a clinical grayscale image, pass `cmap='gray'` explicitly: `ax.imshow(arr, cmap='gray')`.
- **B is incorrect:** Matplotlib does NOT automatically apply `gray` for 2D arrays. A 2D array is always rendered through the active colormap (default: `viridis`). The `gray` colormap is not the default; it must be specified explicitly.
- **C is incorrect:** The default is `viridis`, not `jet`. While `cmap='Greys'` is a valid grayscale colormap (reversed from `gray`), the correct standard clinical grayscale is `cmap='gray'`. `Greys` and `gray` differ in direction — `gray` maps 0→black and 255→white; `Greys` does the same.
- **D is incorrect:** The parameter name is `cmap`, not `colormap`. Passing `colormap=` raises a `TypeError` because `ax.imshow()` does not accept a `colormap` keyword argument.

</details>

---

### Question 2

A developer passes a 3D NumPy array to `ax.imshow()`:

```python
rgb = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
ax.imshow(rgb)
```

What does Matplotlib do with the three channels, and what is the colormap applied?

- [ ] A) Matplotlib treats the 3 channels as R, G, B and renders them as a true-color image — no colormap is applied
- [ ] B) Matplotlib averages the 3 channels into grayscale and renders with `viridis`
- [ ] C) Matplotlib uses only channel 0 (red) and discards G and B
- [ ] D) Matplotlib raises `ValueError` because `imshow` only accepts 2D arrays

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Matplotlib treats the 3 channels as R, G, B and renders them as a true-color image — no colormap is applied

**Explanation:**

- **A is correct:** When `ax.imshow()` receives an array of shape `(M, N, 3)`, it interprets the three channels as Red, Green, and Blue respectively. The image is rendered as a **true-color RGB image** — no colormap is needed or applied. Each pixel's color is determined directly by its RGB values. For `uint8`, valid range is [0, 255]; for float, valid range is [0.0, 1.0].
- **B is incorrect:** Matplotlib does NOT average channels. RGB arrays are rendered directly without channel mixing. Grayscale conversion must be done manually (e.g., PIL's `.convert("L")` or `np.mean(rgb, axis=2)`).
- **C is incorrect:** Matplotlib uses all three channels. Only discarding to a single channel would require explicit slicing: `arr[:, :, 0]`.
- **D is incorrect:** `imshow` explicitly supports `(M, N)`, `(M, N, 3)`, and `(M, N, 4)` shaped arrays. A `(100, 100, 3)` array is a standard RGB input — no error is raised.

</details>

---

### Question 3

A developer has a floating-point image array with pixel values in the range [0.0, 1.0]:

```python
img_float = np.random.rand(64, 64).astype(np.float32)
ax.imshow(img_float)
```

A colleague then creates the same-content array but in `uint8` format:

```python
img_uint8 = (img_float * 255).astype(np.uint8)
ax.imshow(img_uint8)
```

Without specifying `vmin`/`vmax`, how does Matplotlib interpret these two arrays differently?

- [ ] A) Both are treated identically — Matplotlib normalizes all arrays to [0, 1] internally before rendering
- [ ] B) `float32` in [0.0, 1.0] is already in the valid clip range; `uint8` is auto-scaled using [0, 255] as the valid range — both render identical-looking images
- [ ] C) `float32` clips values above 1.0 to white; `uint8` auto-scales to its actual min/max per image
- [ ] D) `uint8` is always rendered without any normalization; `float32` values outside [0,1] are silently clipped to the extremes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `float32` in [0.0, 1.0] is already in the valid clip range; `uint8` is auto-scaled using [0, 255] as the valid range — both render identical-looking images

**Explanation:**

- **A is incorrect:** Matplotlib does NOT normalize all arrays uniformly. The interpretation depends on dtype. For float arrays, values must be in [0.0, 1.0]; for integer arrays, the full integer range is used.
- **B is correct:** For **float arrays**, Matplotlib expects values in [0.0, 1.0] and clips anything outside. For **uint8 arrays**, Matplotlib uses [0, 255] as the full range. Since `img_uint8 = img_float * 255`, the scaling is equivalent — both produce the same visual result (same colormap mapping).
- **C is incorrect:** The `uint8` array does NOT auto-scale per-image min/max when using a colormap on 2D arrays. For `uint8`, the full [0, 255] range is the fixed scale. Auto-scaling (to data min/max) happens only when `vmin`/`vmax` are not explicitly set AND the data is float — but for RGB uint8 arrays specifically it is direct.
- **D is incorrect:** `float32` values outside [0,1] ARE clipped, which is stated correctly. But `uint8` does use a fixed [0,255] range — it is not "no normalization." The statement about uint8 is misleadingly incomplete.

</details>

---

### Question 4

A developer creates a 3D RGBA array for transparent image rendering:

```python
rgba = np.zeros((80, 80, 4), dtype=np.float32)
rgba[:40, :, 0] = 1.0   # Top half: red channel = 1
rgba[:40, :, 3] = 1.0   # Top half: fully opaque
rgba[40:, :, 2] = 1.0   # Bottom half: blue channel = 1
rgba[40:, :, 3] = 0.5   # Bottom half: 50% transparent
ax.imshow(rgba)
```

What does the fourth channel (index 3) control in an RGBA array?

- [ ] A) The brightness/luminance of each pixel
- [ ] B) The alpha (transparency) value — 0.0 = fully transparent, 1.0 = fully opaque
- [ ] C) A secondary colormap blend ratio between two colormaps
- [ ] D) The saturation of the RGB color at each pixel

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The alpha (transparency) value — 0.0 = fully transparent, 1.0 = fully opaque

**Explanation:**

- **A is incorrect:** Brightness/luminance is a property of the RGB channels combined. The A channel in RGBA does not affect luminance — it controls opacity independently of the RGB color values.
- **B is correct:** In an RGBA array `(M, N, 4)`, the fourth channel (index 3) is the **alpha channel** controlling pixel transparency. `alpha=0.0` makes the pixel fully transparent (background shows through); `alpha=1.0` makes it fully opaque. In the code: the top half (red) is fully opaque, while the bottom half (blue) renders at 50% transparency — the background color bleeds through where `alpha=0.5`.
- **C is incorrect:** There is no built-in secondary colormap blending mechanism in RGBA arrays. Colormaps apply only to 2D scalar arrays; RGBA arrays bypass colormaps entirely.
- **D is incorrect:** Saturation is a concept from HSL/HSV color spaces. In the RGBA model, saturation has no direct channel representation — it is derived from the relationship between R, G, and B values, not from the A channel.

</details>

---

### Question 5

A developer runs:

```python
import numpy as np
data = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)
print(data.shape)  # (1, 5)
```

And then calls `ax.imshow(data)`. What is rendered?

- [ ] A) A 1×5 pixel image showing 5 shades from black to white using the `gray` colormap
- [ ] B) A 1×5 pixel image showing 5 colors from the `viridis` colormap ranging from dark blue to yellow
- [ ] C) An error — `imshow` requires at least a 2×2 array
- [ ] D) A 5×1 pixel image (transposed) because Matplotlib swaps shape dimensions

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A 1×5 pixel image showing 5 colors from the `viridis` colormap ranging from dark blue to yellow

**Explanation:**

- **A is incorrect:** The default colormap is `viridis`, not `gray`. A `uint8` 2D array is rendered using the default colormap — which maps 0 to dark blue/purple and 255 to yellow/green in `viridis`. To get black-to-white, `cmap='gray'` must be specified.
- **B is correct:** `data.shape = (1, 5)` is a valid 2D array with 1 row and 5 columns. `ax.imshow()` renders it as a 1×5 pixel strip using the default `viridis` colormap. Values [0, 64, 128, 192, 255] map to colors spanning the full viridis range from dark blue (0) to yellow (255).
- **C is incorrect:** `imshow` has no minimum array dimension requirement beyond being 2D (or 3D for RGB/RGBA). A 1×5 array is perfectly valid.
- **D is incorrect:** Matplotlib does NOT transpose array dimensions. Array shape `(M, N)` is rendered as M rows and N columns — matching the standard matrix convention. `(1, 5)` renders as 1 row, 5 columns — NOT transposed to 5 rows, 1 column.

</details>

---

### Question 6

A developer writes a PIL-to-NumPy pipeline:

```python
from PIL import Image
import numpy as np

img_rgb   = Image.open("photo.jpg")          # Mode: RGB
img_gray  = img_rgb.convert("L")             # Convert to grayscale
arr_rgb   = np.array(img_rgb)
arr_gray  = np.array(img_gray)

print(arr_rgb.shape)   # ?
print(arr_gray.shape)  # ?
```

The image is 640×480 pixels (width × height). What are the shapes?

- [ ] A) `arr_rgb.shape = (640, 480, 3)` and `arr_gray.shape = (640, 480)`
- [ ] B) `arr_rgb.shape = (480, 640, 3)` and `arr_gray.shape = (480, 640)`
- [ ] C) `arr_rgb.shape = (480, 640, 3)` and `arr_gray.shape = (480, 640, 1)`
- [ ] D) `arr_rgb.shape = (3, 480, 640)` and `arr_gray.shape = (480, 640)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `arr_rgb.shape = (480, 640, 3)` and `arr_gray.shape = (480, 640)`

**Explanation:**

- **A is incorrect:** NumPy/PIL convention uses **(height, width)** ordering for array shapes — NOT (width, height). A 640×480 image (width=640, height=480) becomes `(480, 640)` in NumPy — rows (height) first, then columns (width).
- **B is correct:** PIL's `.size` property returns `(width, height) = (640, 480)`. When converted to NumPy via `np.array()`, dimensions are transposed to `(height, width)` = **(480, 640)**. The RGB image adds a channel dimension: `(480, 640, 3)`. The grayscale image has no channel dimension: `(480, 640)`. This is the fundamental PIL↔NumPy shape convention.
- **C is incorrect:** `arr_gray.shape = (480, 640, 1)` would only occur if the image were saved with an explicit channel dimension (e.g., via `np.expand_dims`). PIL's `.convert("L")` followed by `np.array()` always produces a 2D array without a trailing channel axis.
- **D is incorrect:** NumPy arrays from PIL use `(H, W, C)` channel-last format, not `(C, H, W)` (PyTorch's channel-first format). `(3, 480, 640)` is the PyTorch convention — not what `np.array(PIL_image)` produces.

</details>

---

### Question 7

A developer loads a `float64` image where pixel values range from 0.0 to 255.0 (not 0.0 to 1.0) due to a preprocessing bug:

```python
img_buggy = np.random.uniform(0, 255, (100, 100)).astype(np.float64)
ax.imshow(img_buggy, cmap='gray')
```

What does the rendered image look like?

- [ ] A) A correct grayscale image with values properly mapped from black (0) to white (255)
- [ ] B) An almost entirely white image — most pixels are clipped to 1.0 (maximum white) because float arrays clip values above 1.0 to the maximum
- [ ] C) An error is raised because float64 values above 1.0 are invalid for `imshow`
- [ ] D) A correct grayscale image scaled automatically because Matplotlib auto-scales float arrays to their data range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) An almost entirely white image — most pixels are clipped to 1.0 (maximum white) because float arrays clip values above 1.0 to the maximum

**Explanation:**

- **A is incorrect:** For float arrays, Matplotlib does NOT use [0, 255] as the range. It strictly expects [0.0, 1.0]. Values in [0, 255] will not be correctly mapped — they will mostly clip to white.
- **B is correct:** This is a **classic floating-point image bug**. Matplotlib clips float array values to [0.0, 1.0]. All values above 1.0 are treated as 1.0 (white in `gray` colormap). Since `img_buggy` has values uniformly distributed in [0, 255], virtually all pixels (≈99.6%) have values > 1.0 and are clipped to white. The resulting image appears almost entirely white. The fix is: `img_fixed = img_buggy / 255.0`.
- **C is incorrect:** No error is raised for out-of-range float values. Matplotlib silently clips values and issues a warning in some versions — it does not raise an exception.
- **D is incorrect:** Matplotlib does NOT auto-scale float arrays to their data range. Auto-scaling (using actual data min/max) only occurs when `vmin`/`vmax` are explicitly specified or when using `norm` objects. Without these, float arrays are assumed to be in [0, 1].

</details>

---

### Question 8

A developer creates a composite image with both a 2D array and a 3D RGB array on the same figure:

```python
fig, (ax1, ax2) = plt.subplots(1, 2)

data_2d = np.random.rand(50, 50)             # Shape: (50, 50)
data_rgb = np.random.rand(50, 50, 3)         # Shape: (50, 50, 3)

im1 = ax1.imshow(data_2d, cmap='plasma')
im2 = ax2.imshow(data_rgb)

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)   # ?
```

What happens when `fig.colorbar(im2, ax=ax2)` is called for the RGB image?

- [ ] A) A colorbar is created showing the R, G, B channel intensities as three separate color bands
- [ ] B) A `TypeError` or `AttributeError` is raised — RGB images do not have a scalar colormap mapping, so no colorbar can be created
- [ ] C) A colorbar is created showing luminance values computed by averaging the RGB channels
- [ ] D) A colorbar is created using the `viridis` colormap by default for RGB arrays

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A `TypeError` or `AttributeError` is raised — RGB images do not have a scalar colormap mapping, so no colorbar can be created

**Explanation:**

- **A is incorrect:** Colorbars only encode a single scalar dimension (one colormap). RGB images have three independent channels — there is no single scalar-to-color mapping to represent in a colorbar.
- **B is correct:** A colorbar requires an `AxesImage` object with a defined `norm` and `cmap` — which only exists for scalar (2D) arrays rendered through a colormap. For an RGB image (`(M, N, 3)`), no colormap or norm is applied — Matplotlib uses the raw RGB values directly. Calling `fig.colorbar()` on an RGB `AxesImage` raises an error because the required colormap attribute is absent.
- **C is incorrect:** Matplotlib does not automatically compute or display luminance for RGB images. Luminance computation must be done manually before passing to `imshow`.
- **D is incorrect:** RGB arrays bypass the colormap system entirely. No colormap (including viridis) is applied to `(M, N, 3)` or `(M, N, 4)` arrays.

</details>

---

### Question 9

A developer inspects array dtypes before calling `imshow`:

```python
a = np.array([[100, 200], [50, 150]], dtype=np.int32)
b = np.array([[100, 200], [50, 150]], dtype=np.float32)
c = np.array([[100, 200], [50, 150]], dtype=np.uint8)
```

How does Matplotlib treat `int32` dtype differently from `uint8` and `float32` when passed to `ax.imshow()`?

- [ ] A) `int32` is treated identically to `uint8` — both use [0, 255] as the full range
- [ ] B) `int32` is treated like `float32` — Matplotlib clips to [0, 1] and most values appear white
- [ ] C) `int32` arrays are auto-scaled to the data's actual min/max (similar to float with explicit `vmin`/`vmax`) — the color range spans the actual minimum to maximum values in the array
- [ ] D) `int32` raises `TypeError` because only `uint8` and `float32` are supported

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `int32` arrays are auto-scaled to the data's actual min/max (similar to float with explicit `vmin`/`vmax`) — the color range spans the actual minimum to maximum values in the array

**Explanation:**

- **A is incorrect:** `int32` is not treated as `uint8`. `uint8` uses the fixed [0, 255] range for RGB images. For scalar arrays, Matplotlib auto-normalizes based on data range, not dtype range.
- **B is incorrect:** `int32` is not clipped to [0, 1]. The [0, 1] clipping only applies to float arrays (float32, float64). Integer arrays like `int32` are handled differently.
- **C is correct:** For non-uint8 integer arrays (`int16`, `int32`, `int64`) passed to `imshow` as 2D scalar arrays, Matplotlib auto-scales the colormap to the actual data minimum and maximum. For array `a`: min=50, max=200, so the colormap spans [50, 200]. This is equivalent to calling `imshow(a, vmin=50, vmax=200)`. This contrasts with float arrays (clipped to [0,1]) and uint8 (full [0,255] range for RGB; auto-scaled for scalar 2D).
- **D is incorrect:** Matplotlib supports many integer dtypes including `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, etc. No `TypeError` is raised.

</details>

---

### Question 10

A developer creates an animation pipeline and generates frames as 2D NumPy arrays. They notice that the first frame has values in [0, 100] and the second has values in [0, 10]. Without setting `vmin`/`vmax`, what problem occurs?

```python
fig, ax = plt.subplots()
for frame in [frame1, frame2]:  # frame1: [0,100], frame2: [0,10]
    ax.cla()
    ax.imshow(frame, cmap='hot')
    plt.pause(0.1)
```

- [ ] A) No problem — Matplotlib consistently uses [0, 255] for all arrays
- [ ] B) No problem — the `hot` colormap automatically normalizes both frames to the same visual scale
- [ ] C) The two frames appear visually identical in color despite having different data magnitudes — the auto-scaling maps both to the full colormap range [dark→bright], hiding the real intensity difference
- [ ] D) Frame 2 appears all black because its values (0–10) are too small to render

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The two frames appear visually identical in color despite having different data magnitudes — the auto-scaling maps both to the full colormap range [dark→bright], hiding the real intensity difference

**Explanation:**

- **A is incorrect:** Matplotlib does NOT use a fixed [0, 255] range for float arrays. Auto-scaling uses the data's own min/max.
- **B is incorrect:** No colormap has awareness of inter-frame consistency. Each call to `ax.imshow()` creates an independent normalization mapping min→darkest color and max→brightest color.
- **C is correct:** Without `vmin`/`vmax`, `ax.imshow()` auto-scales each call independently: frame1 maps [0, 100] → full colormap, frame2 maps [0, 10] → full colormap. Both frames appear to span the full colormap range — a value of 10 in frame2 looks the same as 100 in frame1 (both are max). The real intensity differences are completely hidden. Fix: `ax.imshow(frame, cmap='hot', vmin=0, vmax=100)` locks the scale across frames.
- **D is incorrect:** Frame2 values [0, 10] are not "too small" — they are valid numbers that auto-scale to the full colormap range. The image would look normally colored (dark to bright), not all black.

</details>
