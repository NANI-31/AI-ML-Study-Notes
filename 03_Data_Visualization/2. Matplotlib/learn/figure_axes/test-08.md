# Matplotlib Figure & Axes Objective Test — Set 08
# Topic: High-Resolution Figure Export (Questions 71–80)

---

### Question 71

A developer exports a training loss figure for an academic journal that requires 300 DPI minimum:

```python
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs, train_loss)
ax.plot(epochs, val_loss)
ax.set_title("Training & Validation Loss")
fig.savefig("loss_curve.png", dpi=300)
```

What are the pixel dimensions of the saved PNG file?

- [ ] A) 700 × 500 pixels (figsize in pixels)
- [ ] B) 2100 × 1500 pixels (7×300, 5×300)
- [ ] C) 300 × 300 pixels (dpi × dpi)
- [ ] D) 2100 × 1500 points at 72 points-per-inch

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 2100 × 1500 pixels (7×300, 5×300)

**Explanation:**

- **A is incorrect:** `figsize=(7, 5)` specifies dimensions in INCHES, not pixels. 700 × 500 would be the result at `dpi=100`, not `dpi=300`.
- **B is correct:** Pixels = inches × DPI. Width = 7 × 300 = **2100 pixels**. Height = 5 × 300 = **1500 pixels**. The `dpi=300` argument in `savefig()` overrides the figure's screen DPI for the saved file only, producing a high-resolution 2100 × 1500 pixel image suitable for academic publication.
- **C is incorrect:** 300 × 300 would result from a 1×1 inch figure at 300 DPI. The DPI is not the output pixel count.
- **D is incorrect:** PNG is a raster format measured in pixels, not points. "Points" is a typographic unit (1/72 inch) used in PDF/PostScript — not applicable to PNG raster dimensions.

</details>

---

### Question 72

A developer saves a figure and finds the **legend is cut off** at the right edge. The legend extends outside the axes bounding box. Which `savefig` argument fixes this?

```python
fig.savefig("report.png", dpi=150)   # Legend is clipped!
fig.savefig("report.png", dpi=150, ____________)  # Fixed
```

- [ ] A) `pad_inches=1`
- [ ] B) `bbox_inches="tight"`
- [ ] C) `clip_on=False`
- [ ] D) `legend_fit=True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `bbox_inches="tight"`

**Explanation:**

- **A is incorrect:** `pad_inches=1` adds 1 inch of padding around the figure but only takes effect when `bbox_inches="tight"` is also set. Alone it does not fix legend clipping because the figure boundary is still used as the crop boundary.
- **B is correct:** `bbox_inches="tight"` instructs Matplotlib to compute the **tight bounding box** of all rendered artists — including legends, tick labels, and any content that extends beyond the declared `figsize` boundary. The saved file is resized to include all content. This is the standard fix for clipped legends, axis labels, and suptitles.
- **C is incorrect:** `clip_on=False` is an artist-level property (e.g., `ax.plot(..., clip_on=False)`) that prevents individual artists from being clipped at the axes boundary. It is not a `savefig()` parameter and does not affect the saved file crop.
- **D is incorrect:** `legend_fit=True` is not a valid `savefig()` parameter. This would raise a `TypeError`.

</details>

---

### Question 73

A developer compares saving a figure in **vector** vs **raster** format:

```python
fig.savefig("report.pdf")   # Vector
fig.savefig("report.png", dpi=300)  # Raster
```

A colleague asks: "Why would you use PDF over PNG for a publication figure?" Which answer is most accurate?

- [ ] A) PDF files are always smaller than PNG files for complex figures
- [ ] B) PDF is a vector format — lines, curves, and text are stored as mathematical descriptions rather than pixels, so the figure can be scaled to any print size (e.g., 2 columns wide in a journal) without pixelation or blurring
- [ ] C) PDF supports more colors than PNG
- [ ] D) PNG files cannot be embedded in academic papers — only PDF is accepted

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) PDF is a vector format — lines, curves, and text are stored as mathematical descriptions rather than pixels, so the figure can be scaled to any print size without pixelation or blurring

**Explanation:**

- **A is incorrect:** For complex figures with many elements, PDF vector files can actually be LARGER than high-DPI PNG rasters (because every curve is stored as a mathematical path). For simpler figures, PDF is often smaller. File size is not the primary reason to choose PDF.
- **B is correct:** PDF (like SVG and EPS) is a **vector format** — all plot elements (lines, axes, text, markers) are stored as resolution-independent mathematical objects (Bezier curves, paths, fonts). This means a PDF figure can be printed at any scale — from a small web image to a large poster — without any pixelation. Academic publishers require vector figures precisely because they need to scale figures for different layouts and print resolutions.
- **C is incorrect:** Both PDF and PNG support millions of colors (24-bit RGB). PDF additionally supports CMYK color spaces used in professional printing, but the core reason to prefer PDF is resolution independence, not color depth.
- **D is incorrect:** Many journals accept PNG at sufficient DPI (300+ for halftone images, 600+ for line art). The preference for PDF/EPS/SVG is resolution independence, not a blanket rejection of PNG.

</details>

---

### Question 74

A developer saves a figure with `transparent=True` for use in a dark-themed presentation:

```python
fig, ax = plt.subplots(figsize=(6, 4))
fig.set_facecolor("#1A1A2E")
ax.set_facecolor("#16213E")
ax.plot([1, 2, 3], [4, 5, 6], color="cyan")
fig.savefig("slide_plot.png", transparent=True, dpi=150)
```

What happens to `fig.set_facecolor("#1A1A2E")` and `ax.set_facecolor("#16213E")` when `transparent=True` is used?

- [ ] A) Only the figure facecolor becomes transparent; the axes background retains `#16213E`
- [ ] B) Both the figure facecolor and the axes facecolor are rendered with alpha=0 (fully transparent) in the saved file, regardless of their set colors
- [ ] C) The `transparent=True` setting has no effect because explicit facecolors override it
- [ ] D) The line `color="cyan"` also becomes transparent

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Both the figure facecolor and the axes facecolor are rendered with alpha=0 (fully transparent) in the saved file, regardless of their set colors

**Explanation:**

- **A is incorrect:** `transparent=True` affects BOTH the figure background and all axes backgrounds — not just the figure facecolor.
- **B is correct:** `savefig(transparent=True)` renders both `fig.patch` (figure background) and all `ax.patch` objects (axes backgrounds) with **alpha=0** in the saved file. The colors set by `set_facecolor()` are overridden by the transparency setting during save. This is intentional: the developer wants the plot elements (lines, labels, ticks) to be visible while the background is transparent, allowing the dark slide background to show through. If you want the dark backgrounds AND transparency of just the figure margins, you'd instead save with `facecolor=fig.get_facecolor()` without `transparent=True`.
- **C is incorrect:** `transparent=True` DOES override the facecolor settings. It specifically sets alpha=0 on background patches regardless of their color.
- **D is incorrect:** `transparent=True` only affects background patches (`fig.patch` and `ax.patch`). Data artists like `Line2D` (the cyan line), text, and tick marks retain their full opacity.

</details>

---

### Question 75

A developer uses `pad_inches` in `savefig()`:

```python
fig.savefig("output.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
```

What does `pad_inches=0.2` do?

- [ ] A) Adds 0.2 inches of white padding around the tight bounding box in the saved file
- [ ] B) Sets the internal padding between subplot axes to 0.2 inches
- [ ] C) Reduces the figure size by 0.2 inches on each side
- [ ] D) Sets the resolution to 0.2 dots-per-inch (overriding `dpi=200`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Adds 0.2 inches of white padding around the tight bounding box in the saved file

**Explanation:**

- **A is correct:** When `bbox_inches="tight"` is used, Matplotlib computes the tight bounding box of all visible content. `pad_inches=0.2` then **adds 0.2 inches of additional padding** around this tight box before saving — creating a small white border that prevents content from touching the image edge. This is useful for figures that will be displayed or printed where a small margin improves aesthetics. The default is `pad_inches=0.1` when `bbox_inches="tight"` is active.
- **B is incorrect:** Internal subplot spacing is controlled by `tight_layout(pad=...)` or `subplots_adjust(hspace=..., wspace=...)`, not `pad_inches` in `savefig()`.
- **C is incorrect:** `pad_inches` ADDS padding — it does not subtract from the figure size. The effective saved image is larger than the tight bounding box by `2 × pad_inches` in each dimension.
- **D is incorrect:** `pad_inches` has nothing to do with DPI (dots per inch). DPI is controlled by the `dpi=` parameter.

</details>

---

### Question 76

A developer creates a figure and saves in **EPS format** for a LaTeX publication:

```python
fig.savefig("figure_1.eps", dpi=300, bbox_inches="tight")
```

A colleague asks: "Is the `dpi=300` meaningful for EPS?" What is the correct answer?

- [ ] A) Yes — EPS renders at exactly 300 DPI, producing high-quality output
- [ ] B) For vector content (lines, text, axes), DPI has no effect — EPS is resolution-independent; however, if the figure contains rasterized elements (e.g., `imshow` heatmaps, scatter plots with many points), the DPI controls the resolution of those rasterized portions
- [ ] C) EPS does not support DPI settings — the `dpi` parameter is ignored entirely
- [ ] D) `dpi=300` is required to enable font embedding in EPS

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) For vector content (lines, text, axes), DPI has no effect — EPS is resolution-independent; however, if the figure contains rasterized elements, DPI controls their resolution

**Explanation:**

- **A is incorrect:** EPS is a vector format — it stores mathematical descriptions of shapes, not pixel grids. "300 DPI" doesn't apply to vector elements in the way it applies to PNG rasters.
- **B is correct:** EPS (Encapsulated PostScript) is a vector format — `Line2D` plots, `Text` artists, axes spines, and tick marks are all stored as vector paths, completely resolution-independent. However, certain Matplotlib elements are RASTERIZED even in vector output: `imshow()` heatmaps, very dense scatter plots using `rasterized=True`, and custom raster artists. For these elements, `dpi=300` controls the resolution of the embedded raster image within the EPS file. So `dpi=300` is good practice even for EPS/PDF/SVG when the figure contains any rasterized components.
- **C is incorrect:** The `dpi` parameter IS processed for EPS export — it controls rasterized element resolution within the vector container.
- **D is incorrect:** Font embedding in EPS is controlled by the `rcParams["ps.fonttype"]` setting (Type 3 or Type 42 fonts), not by DPI.

</details>

---

### Question 77

A developer compares JPEG and PNG for saving ML evaluation figures. A senior engineer says "never use JPEG for Matplotlib figures with sharp lines." Why?

- [ ] A) JPEG files are always larger than PNG for the same figure
- [ ] B) JPEG uses lossy compression that introduces visible artifacts (ringing, blocking) around sharp edges, text, and lines — PNG uses lossless compression that preserves all pixel values exactly
- [ ] C) Matplotlib's JPEG backend is slower than PNG
- [ ] D) JPEG does not support the RGB color space

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) JPEG uses lossy compression that introduces visible artifacts around sharp edges, text, and lines — PNG uses lossless compression that preserves all pixel values exactly

**Explanation:**

- **A is incorrect:** For photographic content, JPEG is typically much SMALLER than PNG. The file size comparison depends on content type. JPEG excels at compressing photographic images; PNG excels at compressing graphics with solid colors and sharp edges.
- **B is correct:** JPEG uses **discrete cosine transform (DCT)-based lossy compression** that fundamentally destroys information during compression. Sharp edges (axes lines, tick marks, text characters) create high-frequency signals that JPEG approximates — producing visible "ringing" artifacts, color bleeding, and blocking. PNG uses **lossless compression (DEFLATE)** — every pixel is stored exactly as rendered. For Matplotlib figures containing axes, text, and lines, PNG is strongly preferred over JPEG.
- **C is incorrect:** Backend rendering speed is not a meaningful concern here — both JPEG and PNG saving speeds are fast compared to figure computation time.
- **D is incorrect:** JPEG fully supports RGB color space. In fact, JPEG was designed primarily for photographic RGB images. It doesn't support transparency (RGBA), but this is separate from RGB support.

</details>

---

### Question 78

A developer wants to save a figure with a specific background color in the saved file that is **different from** the figure's display facecolor:

```python
fig, ax = plt.subplots(figsize=(7, 5))
fig.set_facecolor("#FAFAFA")   # Light gray for display
ax.plot(x, y)
fig.savefig("white_bg.png", dpi=200, facecolor="white")
```

What does `facecolor="white"` in `savefig()` do?

- [ ] A) It raises an error — `facecolor` is not a valid `savefig()` parameter
- [ ] B) It overrides the figure's display facecolor (`#FAFAFA`) for the saved file only, rendering the saved PNG with a pure white background without changing `fig.get_facecolor()`
- [ ] C) It changes `fig.set_facecolor()` permanently to white
- [ ] D) It applies a white color to all axes backgrounds in the saved file

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It overrides the figure's display facecolor for the saved file only, rendering the saved PNG with a pure white background without changing `fig.get_facecolor()`

**Explanation:**

- **A is incorrect:** `facecolor` IS a valid `savefig()` parameter. It is documented and functional.
- **B is correct:** `fig.savefig("white_bg.png", facecolor="white")` sets the **saved file's** figure background to white, overriding the figure object's own facecolor (`#FAFAFA`) only for this save operation. After `savefig()` returns, `fig.get_facecolor()` still returns `#FAFAFA` — the figure object is unchanged. This allows saving the same figure with different background colors for different purposes (e.g., dark facecolor for slides, white for journal submission) without modifying the figure object.
- **C is incorrect:** `facecolor` in `savefig()` does not permanently change the figure's stored facecolor. It is a save-time override only.
- **D is incorrect:** `facecolor` in `savefig()` controls only the **figure** background patch, not individual axes backgrounds. Axes backgrounds (`ax.patch`) are not affected by this parameter.

</details>

---

### Question 79

A developer generates a figure with `rasterized=True` on a scatter plot for a PDF export:

```python
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, s=2, alpha=0.3, rasterized=True)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
fig.savefig("scatter.pdf", dpi=300)
```

Why would a developer use `rasterized=True` on a scatter plot being saved to PDF (a vector format)?

- [ ] A) To improve color accuracy of individual data points
- [ ] B) To prevent an enormous PDF file — without rasterization, 100,000 scatter points become 100,000 individual vector paths, creating a massive file; with rasterization, the scatter layer is saved as a single bitmap embedded in the PDF
- [ ] C) To enable transparency (`alpha=0.3`) in PDF format
- [ ] D) To allow the scatter points to be edited in Adobe Illustrator

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) To prevent an enormous PDF file — without rasterization, 100,000 scatter points become 100,000 individual vector paths, creating a massive file

**Explanation:**

- **A is incorrect:** Color accuracy is not improved by rasterization — rasterization converts the plot to a bitmap, which may slightly reduce color precision compared to exact vector paths.
- **B is correct:** When saving a scatter plot with thousands of points to a vector PDF, each point becomes a separate vector path. A scatter plot with 100,000 points produces a PDF with 100,000 circle/marker path objects — resulting in file sizes of tens to hundreds of megabytes, slow rendering in PDF viewers, and slow loading in Illustrator. `rasterized=True` tells Matplotlib to rasterize the scatter artist into a single embedded bitmap at the specified DPI before inserting it into the PDF. The axes, text, and grid remain vector objects. The PDF file size is dramatically reduced while maintaining sharp axes and labels.
- **C is incorrect:** PDF supports transparency (alpha) natively in vector format. `rasterized=True` is not required to enable transparency in PDF.
- **D is incorrect:** `rasterized=True` actually REDUCES editability in vector editors — the scatter points become a flat bitmap layer instead of individual editable objects. Keeping `rasterized=False` preserves full editability.

</details>

---

### Question 80

A developer saves a figure using multiple format calls and compares file characteristics:

```python
fig.savefig("plot.svg")
fig.savefig("plot.pdf")
fig.savefig("plot.png", dpi=300)
fig.savefig("plot.tiff", dpi=600)
```

A colleague asks: "Which formats support true **lossless** storage of all figure information?" 

- [ ] A) Only PNG — SVG, PDF, and TIFF are all lossy
- [ ] B) SVG, PDF, and PNG are lossless (and TIFF in lossless mode) — they faithfully preserve all figure content without information loss
- [ ] C) Only PDF — it is the only format Matplotlib can export with full fidelity
- [ ] D) TIFF is lossless; PNG, SVG, and PDF all compress data with lossy algorithms

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) SVG, PDF, and PNG are lossless (and TIFF in lossless mode)

**Explanation:**

- **A is incorrect:** SVG, PDF, and TIFF are all capable of lossless representation. Only JPEG (not listed here) is a lossy format among common figure export formats.
- **B is correct:** All four formats listed can achieve lossless storage: **SVG** (Scalable Vector Graphics) stores all elements as XML vector descriptions — perfectly lossless. **PDF** stores elements as PostScript paths and fonts — lossless. **PNG** uses DEFLATE lossless compression — every pixel is preserved exactly. **TIFF** can use lossless compression (LZW, PackBits, or no compression) — though it also supports lossy JPEG compression as an option. None of these four formats inherently apply lossy compression in their default Matplotlib export modes.
- **C is incorrect:** PDF is not the only lossless format. PNG, SVG, TIFF, and EPS are all capable of lossless figure representation.
- **D is incorrect:** TIFF IS lossless in its standard modes, but PNG and SVG are also lossless — not lossy. PDF is a vector format that is inherently lossless for vector content.

</details>
