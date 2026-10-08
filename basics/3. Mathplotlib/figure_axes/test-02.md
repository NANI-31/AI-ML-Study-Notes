# Matplotlib Figure & Axes Objective Test — Set 02
# Topic: Figure Dimensions & DPI Mathematics (Questions 11–20)

---

### Question 11

A developer creates a figure for a scientific publication with the following code:

```python
fig = plt.figure(figsize=(7, 5), dpi=150)
```

What are the **pixel dimensions** of this figure?

- [ ] A) 700 × 500 pixels
- [ ] B) 1050 × 750 pixels
- [ ] C) 350 × 250 pixels
- [ ] D) 7000 × 5000 pixels

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 1050 × 750 pixels

**Explanation:**

- **A is incorrect:** 700 × 500 would result from `figsize=(7, 5)` at `dpi=100` (7×100=700, 5×100=500). The dpi here is 150, not 100.
- **B is correct:** The pixel dimension formula is: pixels = inches × DPI. Width: 7 × 150 = **1050 pixels**. Height: 5 × 150 = **750 pixels**. So the figure is 1050 × 750 pixels. This formula is fundamental: `figsize` always specifies dimensions in **inches**, and `dpi` is dots-per-inch. The rendered pixel count is their product.
- **C is incorrect:** 350 × 250 would result from dividing instead of multiplying (7/2=3.5×100, 5/2=2.5×100) — an incorrect application of the formula.
- **D is incorrect:** 7000 × 5000 would require `dpi=1000`, not 150. This would be an exceptionally high-resolution export, not a typical figure.

</details>

---

### Question 12

A developer creates a figure for web display and needs it to be exactly **960 × 540 pixels** at `dpi=120`. What `figsize` tuple should be passed to `plt.figure()`?

- [ ] A) `figsize=(960, 540)`
- [ ] B) `figsize=(8, 4.5)`
- [ ] C) `figsize=(80, 45)`
- [ ] D) `figsize=(9.6, 5.4)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `figsize=(8, 4.5)`

**Explanation:**

- **A is incorrect:** `figsize` takes values in **inches**, not pixels. Passing `figsize=(960, 540)` would create an absurdly large figure of 960 inches × 540 inches (that's 80 feet wide), not a 960×540 pixel figure.
- **B is correct:** To find inches from pixels: inches = pixels / DPI. Width: 960 / 120 = **8 inches**. Height: 540 / 120 = **4.5 inches**. So `figsize=(8, 4.5)` at `dpi=120` produces exactly 960 × 540 pixels.
- **C is incorrect:** `figsize=(80, 45)` at `dpi=120` would produce 9600 × 5400 pixels — 10× too large.
- **D is incorrect:** `figsize=(9.6, 5.4)` at `dpi=120` = 1152 × 648 pixels — not 960 × 540.

</details>

---

### Question 13

A data scientist creates a publication figure at `dpi=300` for a journal requiring minimum **2400 × 1800 pixels**. What is the minimum `figsize` (in inches) they must use?

- [ ] A) `figsize=(8, 6)` — gives exactly 2400 × 1800 px at dpi=300
- [ ] B) `figsize=(24, 18)` — giving 24 × 18 at dpi=300
- [ ] C) `figsize=(2400, 1800)` — in pixel units
- [ ] D) `figsize=(7.2, 5.4)` — giving 2160 × 1620 px at dpi=300

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `figsize=(8, 6)` — gives exactly 2400 × 1800 px at dpi=300

**Explanation:**

- **A is correct:** Width: 8 × 300 = **2400 pixels**. Height: 6 × 300 = **1800 pixels**. `figsize=(8, 6)` at `dpi=300` exactly meets the 2400×1800 minimum requirement. This is a common academic publication figure size (8 × 6 inches at 300 DPI).
- **B is incorrect:** `figsize=(24, 18)` at `dpi=300` = 7200 × 5400 pixels — far larger than necessary and would create impractically large files.
- **C is incorrect:** `figsize=(2400, 1800)` passes pixel values directly to `figsize`, which expects inches. At `dpi=300`, this creates a figure 720,000 × 540,000 pixels large — not what is intended.
- **D is incorrect:** `figsize=(7.2, 5.4)` at `dpi=300` = 2160 × 1620 pixels — below the 2400 × 1800 minimum requirement.

</details>

---

### Question 14

A developer checks the DPI of an existing figure:

```python
fig = plt.figure(figsize=(6, 4))
print(fig.get_dpi())
```

What does this print if no `dpi` argument was explicitly passed to `plt.figure()`?

- [ ] A) `72` — the screen DPI default
- [ ] B) `100` — Matplotlib's default figure DPI
- [ ] C) `96` — Windows screen DPI standard
- [ ] D) `None` — DPI is not set until `savefig()` is called

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `100` — Matplotlib's default figure DPI

**Explanation:**

- **A is incorrect:** 72 DPI is a historical Mac/PostScript screen standard, but it is NOT Matplotlib's default. Some documentation historically mentioned 72 DPI for vector formats, but Matplotlib's default figure DPI is 100.
- **B is correct:** By default, `plt.figure()` uses `dpi=100` when no `dpi` argument is provided. This is set by `matplotlib.rcParams["figure.dpi"]` whose default value is 100. So `fig.get_dpi()` returns `100.0`. This means a default `figsize=(6.4, 4.8)` figure (Matplotlib's default size) produces 640 × 480 pixels at screen display.
- **C is incorrect:** 96 DPI is the Windows screen DPI standard, but Matplotlib's default is set by its own `rcParams`, not the OS screen DPI. The default is 100.
- **D is incorrect:** DPI is set at figure creation time (defaulting to 100 if not specified). It is not `None` before `savefig()`. You can always check it with `fig.get_dpi()`, and you can change it at any time with `fig.set_dpi(300)` before saving.

</details>

---

### Question 15

A developer dynamically changes the DPI of an existing figure before saving:

```python
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot([1, 2, 3], [4, 5, 6])
fig.set_dpi(200)
print(fig.get_dpi())
```

After `fig.set_dpi(200)`, what are the pixel dimensions when the figure is rendered?

- [ ] A) Still 500 × 300 — `figsize` locks in the pixel dimensions at creation
- [ ] B) 1000 × 600 pixels — figsize (5×3 inches) multiplied by new dpi (200)
- [ ] C) 200 × 200 — `set_dpi` changes both width and height to the DPI value
- [ ] D) 500 × 300 pixels on screen; 1000 × 600 pixels only in saved files

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 1000 × 600 pixels — figsize (5×3 inches) multiplied by new dpi (200)

**Explanation:**

- **A is incorrect:** `figsize` specifies dimensions in inches, which are fixed. DPI can be changed at any time via `fig.set_dpi()`, and the rendered pixel dimensions will update accordingly. The pixel count is NOT locked at creation.
- **B is correct:** After `fig.set_dpi(200)`, the figure's DPI is updated to 200. Pixel dimensions = inches × DPI: Width = 5 × 200 = **1000 pixels**; Height = 3 × 200 = **600 pixels**. `fig.get_dpi()` confirms 200.0.
- **C is incorrect:** `set_dpi(200)` does NOT change the `figsize` or make both dimensions equal to 200. The DPI is a resolution setting; the physical size (in inches) remains `(5, 3)`.
- **D is incorrect:** The distinction between screen and saved file DPI is not about `set_dpi()`. The `dpi` parameter in `savefig()` overrides the figure's DPI for the saved file, but `fig.set_dpi()` changes the figure's own DPI attribute uniformly.

</details>

---

### Question 16

A developer creates a figure for a 16:9 widescreen monitor display at `dpi=96`. They want the figure to be exactly `1920 × 1080` pixels. Which `figsize` is correct?

- [ ] A) `figsize=(20, 11.25)`
- [ ] B) `figsize=(19.2, 10.8)`
- [ ] C) `figsize=(16, 9)`
- [ ] D) `figsize=(15, 8.44)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `figsize=(20, 11.25)`

**Explanation:**

- **A is correct:** inches = pixels / DPI. Width: 1920 / 96 = **20 inches**. Height: 1080 / 96 = **11.25 inches**. `figsize=(20, 11.25)` at `dpi=96` → 1920 × 1080 pixels exactly.
- **B is incorrect:** `figsize=(19.2, 10.8)` at `dpi=96` = 1843.2 × 1036.8 ≈ 1843 × 1037 pixels (not 1920 × 1080). These values would be correct at `dpi=100`.
- **C is incorrect:** `figsize=(16, 9)` at `dpi=96` = 1536 × 864 pixels — a 16:9 ratio but not 1920 × 1080.
- **D is incorrect:** `figsize=(15, 8.44)` at `dpi=96` = 1440 × 810.24 ≈ 1440 × 810 pixels — not the target resolution.

</details>

---

### Question 17

A developer creates a high-resolution figure for print and examines the aspect ratio:

```python
fig = plt.figure(figsize=(10, 4), dpi=300)
```

What is the **aspect ratio** (width:height) of this figure, and what are the pixel dimensions?

- [ ] A) Aspect ratio 2.5:1 — pixel dimensions 3000 × 1200
- [ ] B) Aspect ratio 2.5:1 — pixel dimensions 1000 × 400
- [ ] C) Aspect ratio 4:1 — pixel dimensions 4000 × 1000
- [ ] D) Aspect ratio 2:1 — pixel dimensions 2000 × 1000

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Aspect ratio 2.5:1 — pixel dimensions 3000 × 1200

**Explanation:**

- **A is correct:** Aspect ratio = width / height = 10 / 4 = **2.5:1** (a wide panoramic figure). Pixel dimensions: Width = 10 × 300 = **3000 px**; Height = 4 × 300 = **1200 px**. This is a common shape for training loss or latency timeline charts in ML papers.
- **B is incorrect:** 1000 × 400 would result from `dpi=100` (10×100=1000, 4×100=400), not `dpi=300`. The aspect ratio (2.5:1) is correct but the pixel dimensions are wrong.
- **C is incorrect:** A 4:1 aspect ratio would require `figsize=(16, 4)` or similar. Here the ratio is 10:4 = 2.5:1.
- **D is incorrect:** A 2:1 ratio would require `figsize=(8, 4)` or `(10, 5)`. Here 10/4 = 2.5, not 2.0.

</details>

---

### Question 18

A machine learning engineer generates 500 evaluation plots in a loop, each with `dpi=300` and `figsize=(8, 6)`. Each figure is saved to disk. What is the **approximate memory** held by each figure while it exists in RAM (before garbage collection)?

- [ ] A) About 0.5 MB — figures are memory-efficient objects
- [ ] B) Approximately 17.3 MB — 2400 × 1800 pixels × 4 bytes (RGBA) per pixel ≈ 17.3 MB for an uncompressed RGBA raster buffer
- [ ] C) Exactly 300 KB — DPI directly equals kilobytes
- [ ] D) No RAM — figures are stored directly on disk

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Approximately 17.3 MB — 2400 × 1800 pixels × 4 bytes (RGBA) per pixel ≈ 17.3 MB for an uncompressed RGBA raster buffer

**Explanation:**

- **A is incorrect:** High-DPI figures are NOT lightweight. A 300 DPI, 8×6 inch figure creates a 2400×1800 raster buffer that requires significant RAM.
- **B is correct:** Pixel dimensions: 8 × 300 = 2400 px wide; 6 × 300 = 1800 px tall. Total pixels = 2400 × 1800 = 4,320,000 pixels. With 4 bytes per pixel (RGBA channels): 4,320,000 × 4 = **17,280,000 bytes ≈ 17.3 MB** per figure in the raster backend buffer. With 500 such figures without `plt.close()`, that's ~8.65 GB of RAM — causing Out-Of-Memory (OOM) crashes. This is why production loops MUST call `plt.close(fig)` after each save.
- **C is incorrect:** There is no formula relating DPI directly to kilobytes. Memory depends on pixel area (inches × DPI)² × bytes per pixel.
- **D is incorrect:** Saving to disk does not free the in-memory figure object. The figure remains in RAM until `plt.close(fig)` is called and the garbage collector runs.

</details>

---

### Question 19

A developer uses `fig.get_size_inches()` and `fig.get_dpi()` to report figure properties:

```python
fig = plt.figure(figsize=(9, 6), dpi=150)
w, h = fig.get_size_inches()
d = fig.get_dpi()
print(f"Width: {w*d:.0f}px, Height: {h*d:.0f}px")
```

What does this print?

- [ ] A) `Width: 9px, Height: 6px`
- [ ] B) `Width: 1350px, Height: 900px`
- [ ] C) `Width: 900px, Height: 600px`
- [ ] D) `Width: 150px, Height: 150px`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `Width: 1350px, Height: 900px`

**Explanation:**

- **A is incorrect:** `w*d` = 9 × 150 = 1350, not 9. The code multiplies inches by DPI to compute pixels.
- **B is correct:** `fig.get_size_inches()` returns `(9.0, 6.0)`. `fig.get_dpi()` returns `150.0`. Width in pixels: 9.0 × 150.0 = **1350**. Height in pixels: 6.0 × 150.0 = **900**. The f-string `:.0f` formats as zero decimal places, printing: `Width: 1350px, Height: 900px`.
- **C is incorrect:** 900 × 600 would result from `dpi=100` (9×100=900, 6×100=600). The DPI here is 150.
- **D is incorrect:** 150 × 150 would only occur if both figsize dimensions equaled 1 inch (1×150=150). Here figsize is (9, 6).

</details>

---

### Question 20

A developer wants to know the **default `figsize`** used by `plt.figure()` when no `figsize` argument is provided. What is Matplotlib's default figure size?

- [ ] A) `(8, 6)` inches — the classic 4:3 ratio
- [ ] B) `(6.4, 4.8)` inches — the Matplotlib default (set by `rcParams["figure.figsize"]`)
- [ ] C) `(10, 7)` inches — the "golden ratio" approximation
- [ ] D) `(5, 4)` inches — a compact default for Jupyter notebooks

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `(6.4, 4.8)` inches — the Matplotlib default (set by `rcParams["figure.figsize"]`)

**Explanation:**

- **A is incorrect:** `(8, 6)` is a popular choice for manual figures but is NOT Matplotlib's built-in default. It produces a 800 × 600 px display at 100 DPI.
- **B is correct:** Matplotlib's `rcParams["figure.figsize"]` defaults to **`[6.4, 4.8]`** — an aspect ratio of 4:3 (6.4/4.8 = 1.333...). At the default `dpi=100`, this produces 640 × 480 pixels. Developers can override this globally with `plt.rcParams["figure.figsize"] = [10, 6]` or per-figure with `figsize=(...)`.
- **C is incorrect:** `(10, 7)` is not Matplotlib's default. It does approximate a golden ratio (10/7 ≈ 1.43 vs. golden ratio ≈ 1.618) but is not the factory default.
- **D is incorrect:** `(5, 4)` is not Matplotlib's factory default. Jupyter notebooks might display figures with different scaling, but the Matplotlib default remains `(6.4, 4.8)`.

</details>
