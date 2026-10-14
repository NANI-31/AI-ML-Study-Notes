# Matplotlib Imshow & Heatmaps — Test 10
## Machine Learning & Computer Vision Applications

---

### Question 91

A developer creates a **multi-class confusion matrix** for a 4-class classifier and annotates each cell with both raw count and recall percentage:

```python
cm = np.array([[85, 5, 7, 3],
               [4, 90, 3, 3],
               [6, 2, 88, 4],
               [2, 3, 5, 90]])

# Normalize by row (true class) to get recall
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1,
               interpolation='nearest', aspect='equal')

for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i,j]}\n{cm_norm[i,j]:.1%}",
                ha='center', va='center',
                color='white' if cm_norm[i,j] > 0.6 else 'black',
                fontsize=9, fontweight='bold')
```

`cm.sum(axis=1, keepdims=True)` — what does `keepdims=True` ensure?

- [ ] A) The sum includes NaN values in the computation
- [ ] B) The result has shape `(4, 1)` instead of `(4,)` — preserving the 2D structure for correct broadcasting division across rows
- [ ] C) The sum is computed column-wise instead of row-wise
- [ ] D) Each row sum is divided by the total matrix sum for global normalization

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The result has shape `(4, 1)` instead of `(4,)` — preserving the 2D structure for correct broadcasting division across rows

**Explanation:**

- **A is incorrect:** `keepdims` has no relationship to NaN handling. NaN propagation is controlled by `np.nansum()` vs `np.sum()`.
- **B is correct:** `cm.sum(axis=1)` without `keepdims` produces shape `(4,)` — a 1D array. When dividing a `(4, 4)` matrix by a `(4,)` array, NumPy broadcasts along the LAST axis (columns) — dividing each column by the row sums, which is wrong (it would normalize by column, not row). With `keepdims=True`, the result has shape `(4, 1)` — a column vector. Dividing `(4, 4) / (4, 1)` broadcasts correctly: each **row** of the matrix is divided by its corresponding row sum — producing per-class recall (row normalization). This is the standard pattern for confusion matrix normalization.
- **C is incorrect:** `axis=1` always sums across columns (row-wise sum), regardless of `keepdims`. `keepdims` only affects the output shape, not which axis is reduced.
- **D is incorrect:** Global normalization would require dividing by `cm.sum()` (a scalar, the total count). `cm.sum(axis=1, keepdims=True)` computes the sum of each individual row — producing per-row normalization, not global.

</details>

---

### Question 92

A developer displays **CNN feature maps** (activations) from an intermediate layer. The layer outputs a tensor of shape `(1, 32, 56, 56)` — batch=1, 32 channels, 56×56 spatial. They want to display all 32 channels in a 4×8 grid:

```python
activations = model(input_tensor)  # (1, 32, 56, 56)
acts = activations[0].detach().numpy()  # (32, 56, 56)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(acts[idx], cmap='viridis', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f"Ch {idx}", fontsize=7)

plt.tight_layout()
plt.savefig("feature_maps.png", dpi=150, bbox_inches='tight')
plt.close()
```

Without setting `vmin`/`vmax`, different channels auto-scale independently. What diagnostic problem does this create?

- [ ] A) Some channels appear blurry due to missing `vmin`/`vmax`
- [ ] B) A channel with small activations (range [0, 0.01]) and one with large activations (range [0, 10]) appear visually identical in color intensity — making cross-channel comparison of activation magnitude impossible
- [ ] C) `plt.tight_layout()` fails when `vmin`/`vmax` are not set
- [ ] D) The `viridis` colormap requires `vmin=0` to function correctly for activation maps

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A channel with small activations (range [0, 0.01]) and one with large activations (range [0, 10]) appear visually identical in color intensity — making cross-channel comparison of activation magnitude impossible

**Explanation:**

- **A is incorrect:** Blurriness is caused by interpolation mode, not `vmin`/`vmax`. The two are independent parameters.
- **B is correct:** Without `vmin`/`vmax`, each channel auto-scales to its own [min, max] range. Channel 5 with activations [0, 0.01] maps its small range to the full viridis spectrum; Channel 12 with activations [0, 10] also maps to the full spectrum. Both appear visually similar in color intensity — yet their actual activation magnitudes differ by 1000×. This is a critical diagnostic blind spot: channels that are nearly dead (near-zero activations) appear identical to highly active channels. For cross-channel magnitude comparison, a shared `vmin=0, vmax=acts.max()` (global maximum) is necessary.
- **C is incorrect:** `plt.tight_layout()` is entirely unrelated to `vmin`/`vmax`. It adjusts subplot spacing — it always works regardless of color scaling parameters.
- **D is incorrect:** `viridis` works correctly for any data range with or without explicit `vmin`/`vmax`. The colormap itself has no requirement for a specific minimum value.

</details>

---

### Question 93

A developer creates a **normalized confusion matrix recall heatmap** and notices that a row for class "Cat" sums to values slightly above 1.0 due to floating-point rounding in `cm_norm`. The colorbar shows values up to 1.002. What is the cleanest fix?

- [ ] A) `ax.imshow(cm_norm, vmin=0, vmax=1)` — locks the colorbar to [0, 1], clipping the 1.002 values without changing the data
- [ ] B) `cm_norm = cm_norm.clip(0, 1)` — clips all values to [0, 1] in the array before `imshow`
- [ ] C) Recompute `cm_norm` using `cm / cm.sum(axis=1, keepdims=True)` with higher precision dtype
- [ ] D) All of A, B, and C are valid approaches with different tradeoffs

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) All of A, B, and C are valid approaches with different tradeoffs

**Explanation:**

- **A is correct (as part of D):** `vmin=0, vmax=1` is the simplest visual fix. It locks the colorbar range to [0, 1] and clips the 1.002 values to the colormap maximum color — identical to 1.0. The underlying data is unchanged (still has 1.002) but the display is correct. Best for: when only the visualization needs fixing.
- **B is correct (as part of D):** `cm_norm.clip(0, 1)` creates a new array with all values bounded to [0, 1]. The 1.002 becomes exactly 1.0. The corrected array is then used in `imshow`. Best for: when downstream computations also use `cm_norm` and need clean values.
- **C is correct (as part of D):** Using `float64` (default for `cm.astype(float)`) already provides maximum precision. However, ensuring proper integer-to-float conversion before division minimizes accumulated floating-point error. Best for: preventing the issue at the source rather than fixing it after.
- **D is correct:** All three approaches are legitimate. In practice, both A and B are commonly used together: `vmin=0, vmax=1` for the visual display AND `.clip(0, 1)` for data cleanliness.

</details>

---

### Question 94

A developer visualizes **Grad-CAM heatmaps** overlaid on the original image:

```python
from PIL import Image
import matplotlib.cm as cm

orig_img = np.array(Image.open("cat.jpg").resize((224, 224)))  # (224, 224, 3), uint8
gradcam  = compute_gradcam(model, orig_img)                    # (14, 14), float32, [0,1]

# Upsample GradCAM to match image size
gradcam_resized = np.array(Image.fromarray((gradcam * 255).astype(np.uint8)).resize((224, 224)))
gradcam_norm = gradcam_resized / 255.0

# Apply colormap to GradCAM
heatmap_rgb = cm.hot(gradcam_norm)[:, :, :3]  # Drop alpha → (224, 224, 3)

# Overlay: blend original + heatmap
overlay = 0.5 * orig_img / 255.0 + 0.5 * heatmap_rgb

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(orig_img)
axes[1].imshow(gradcam_norm, cmap='hot', vmin=0, vmax=1)
axes[2].imshow(overlay)
```

What does `cm.hot(gradcam_norm)[:, :, :3]` do?

- [ ] A) Applies the `hot` colormap and drops the alpha channel — converting the 2D float array to an RGB image by mapping scalar values through the colormap
- [ ] B) Crops the top 3 rows from the `hot` colormap array
- [ ] C) Converts the gradcam array to a 3-channel grayscale image
- [ ] D) Applies normalization to the first 3 channels of the gradcam array

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Applies the `hot` colormap and drops the alpha channel — converting the 2D float array to an RGB image by mapping scalar values through the colormap

**Explanation:**

- **A is correct:** `cm.hot(gradcam_norm)` calls the `hot` colormap as a function, accepting a 2D float array [0, 1] and returning a `(224, 224, 4)` RGBA array — each scalar mapped to an RGBA color tuple. `[:, :, :3]` slices off the alpha channel (index 3), leaving only the `(224, 224, 3)` RGB channels. This produces a color heatmap image where low activation areas appear black/dark red and high activation areas appear bright yellow/white (the `hot` colormap progression). This RGB array can then be blended with the original image.
- **B is incorrect:** `[:, :, :3]` slices the last (channel) axis — not the row axis. It selects channels 0, 1, 2 (R, G, B) and drops channel 3 (A). It does not crop rows.
- **C is incorrect:** `cm.hot(array)` produces a **colorized** RGB image, not grayscale. The hot colormap maps values to colors (black → red → yellow → white) — full color, not luminance-only grayscale.
- **D is incorrect:** `cm.hot(gradcam_norm)` operates on the entire 2D input array holistically — it does not "normalize channels." The gradcam is already normalized [0, 1] before this call. The colormap application is a scalar-to-RGBA mapping, not a normalization operation.

</details>

---

### Question 95

A developer displays **confusion matrix** rows in the wrong order — classes appear as `[Dog, Bird, Cat]` instead of the intended `[Cat, Dog, Bird]`. The confusion matrix array `cm` is already computed correctly in alphabetical order `[Bird, Cat, Dog]`. What is the correct fix using only Matplotlib (no re-computing `cm`)?

- [ ] A) `ax.imshow(cm[::-1, ::-1])` — reverse both rows and columns
- [ ] B) Reorder the array: `order = [1, 2, 0]; ax.imshow(cm[order][:, order])` and update tick labels accordingly
- [ ] C) `ax.invert_yaxis()` — reverting y-axis order fixes class ordering
- [ ] D) `ax.imshow(cm.T)` — transposing fixes the class ordering

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Reorder the array: `order = [1, 2, 0]; ax.imshow(cm[order][:, order])` and update tick labels accordingly

**Explanation:**

- **A is incorrect:** `cm[::-1, ::-1]` reverses both rows and columns — producing `[Dog, Cat, Bird]` (reverse alphabetical), not the desired `[Cat, Dog, Bird]` order. Reversing both axes doesn't produce an arbitrary reordering.
- **B is correct:** To display classes in order `[Cat, Dog, Bird]` when `cm` is indexed as `[Bird=0, Cat=1, Dog=2]`: the desired order index is `[1, 2, 0]` (Cat first, Dog second, Bird third). `cm[order][:, order]` reorders BOTH rows and columns simultaneously — essential for a confusion matrix where rows are "true classes" and columns are "predicted classes." If only rows are reordered, the column labels would mismatch. Tick labels must also be reordered: `ax.set_xticklabels(['Cat', 'Dog', 'Bird'])`.
- **C is incorrect:** `ax.invert_yaxis()` vertically flips the display — changing `[Bird, Cat, Dog]` to `[Dog, Cat, Bird]` (reversed alphabetical). This does not produce the desired `[Cat, Dog, Bird]` ordering.
- **D is incorrect:** `cm.T` transposes the matrix — swapping true-class rows with predicted-class columns. This is semantically wrong: transposing a confusion matrix creates a precision matrix (column-normalized) from a recall matrix (row-normalized). It does not fix class ordering.

</details>

---

### Question 96

A developer creates a **hyperparameter search heatmap** for learning rate × batch size:

```python
lr_values    = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
batch_sizes  = [16, 32, 64, 128, 256]
val_accuracy = np.random.rand(5, 5)  # rows=lr, cols=batch

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(val_accuracy, cmap='RdYlGn', vmin=0.5, vmax=1.0,
               interpolation='nearest', aspect='auto')

ax.set_xticks(range(5))
ax.set_xticklabels(batch_sizes)
ax.set_yticks(range(5))
ax.set_yticklabels([f"{lr:.0e}" for lr in lr_values])
ax.set_xlabel("Batch Size"); ax.set_ylabel("Learning Rate")
ax.set_title("Validation Accuracy: LR × Batch Size Grid Search")
fig.colorbar(im, ax=ax, label="Val Accuracy", shrink=0.85)
```

The `vmin=0.5` setting means cells below 50% accuracy appear as pure red. What does this design choice communicate?

- [ ] A) All models below 50% accuracy are considered failures — clipping draws visual attention to the meaningful performance range [50%, 100%]
- [ ] B) `vmin=0.5` causes a normalization error for accuracy values
- [ ] C) The colormap cannot display values below 0.5 — they become invisible
- [ ] D) `vmin=0.5` forces the center of `RdYlGn` to appear at 75% accuracy

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) All models below 50% accuracy are considered failures — clipping draws visual attention to the meaningful performance range [50%, 100%]

**Explanation:**

- **A is correct:** For binary classification, random guessing achieves ~50% accuracy. Setting `vmin=0.5` is a deliberate design choice: any hyperparameter combination yielding < 50% is treated as a "failure" (all appear identical — pure red). The colormap's full range is dedicated to the meaningful performance space [50%, 100%], allowing subtle differences in the 70–90% range to be visually distinguishable. This is a standard practice for accuracy heatmaps where below-baseline performance is less important than comparing above-baseline results.
- **B is incorrect:** `vmin=0.5` is a valid normalization bound. No error occurs — values below 0.5 are simply clipped to the red extreme of `RdYlGn`. The normalization formula `(val - 0.5) / (1.0 - 0.5)` is well-defined.
- **C is incorrect:** Values below `vmin` are NOT invisible — they appear as the colormap's minimum color (pure red in `RdYlGn`). Only RGBA arrays with alpha=0 produce invisible pixels.
- **D is incorrect:** The center of `RdYlGn` (yellow/neutral) appears at `(vmin + vmax) / 2 = (0.5 + 1.0) / 2 = 0.75` — 75% accuracy. This IS actually correct (yellow at 75%), but the statement says this is wrong about `vmin=0.5` making it so. The center-at-75% is a consequence, not a problem.

</details>

---

### Question 97

A developer visualizes **attention weights** from a transformer model for a sequence-to-sequence task. The attention matrix shape is `(1, 8, 12, 12)` — batch=1, 8 heads, 12 src tokens × 12 tgt tokens. They display head 3:

```python
attn = model(src, tgt)  # (1, 8, 12, 12)
head3 = attn[0, 3].detach().numpy()  # (12, 12)

ax.imshow(head3, cmap='Blues', vmin=0, vmax=head3.max(),
          interpolation='nearest', aspect='equal')
ax.set_xlabel("Target Token Position")
ax.set_ylabel("Source Token Position")
```

Why is `vmax=head3.max()` used instead of `vmax=1.0` for attention weights?

- [ ] A) Attention weights don't sum to 1.0 — they can exceed 1.0 in transformers
- [ ] B) Attention weights are softmax-normalized per source position (each row sums to 1.0) — `head3.max()` may be less than 1.0, using it as `vmax` maximizes the colormap contrast to show which target position receives the most attention per source token
- [ ] C) `vmax=1.0` would cause an error for attention matrices
- [ ] D) `head3.max()` automatically computes the global maximum across all 8 heads

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Attention weights are softmax-normalized per source position (each row sums to 1.0) — `head3.max()` may be less than 1.0, using it as `vmax` maximizes the colormap contrast to show which target position receives the most attention per source token

**Explanation:**

- **A is incorrect:** Standard transformer attention weights ARE softmax-normalized — each row (source position) sums to exactly 1.0. Individual weights can range from 0.0 to 1.0 but cannot exceed 1.0.
- **B is correct:** For a 12-token sequence, uniform attention would give each cell weight ≈ 1/12 ≈ 0.083. Even focused attention rarely produces values at exactly 1.0 (the maximum). If `vmax=1.0`, most cells would appear very light (all values << 1.0), washing out the contrast. Using `vmax=head3.max()` (e.g., 0.4 for a moderately focused head) maximizes the colormap range to `[0, 0.4]` — making the attention pattern maximally visible. The tradeoff: `vmax=head3.max()` is good for single-head visualization but wrong for **cross-head comparison** (each head gets different scaling).
- **C is incorrect:** `vmax=1.0` is perfectly valid — it simply spreads all attention values across [0, 1.0], producing a washed-out (light-colored) visualization when most weights are small fractions.
- **D is incorrect:** `head3.max()` is called on the `(12, 12)` array for head 3 only — it computes the maximum within that single head, not across all 8 heads. Cross-head global maximum would require `attn[0].max()`.

</details>

---

### Question 98

A developer creates a **production-grade confusion matrix** function for a multi-label classification report:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, class_names, save_path,
                           normalize=True, figsize=(10, 9)):
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = '.1%'
        vmax = 1.0
        cmap = 'Blues'
    else:
        cm_display = cm
        fmt = 'd'
        vmax = cm.max()
        cmap = 'YlOrRd'

    n = len(class_names)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, cmap=cmap, vmin=0, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    fig.colorbar(im, ax=ax, shrink=0.85,
                 label='Recall' if normalize else 'Count')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Class"); ax.set_ylabel("True Class")

    for i in range(n):
        for j in range(n):
            val = cm_display[i, j]
            thresh = vmax / 2
            color = 'white' if val > thresh else 'black'
            label = format(val, fmt) if fmt == 'd' else f"{val:{fmt}}"
            ax.text(j, i, label, ha='center', va='center',
                    color=color, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
```

Which line contains a subtle bug for the case `normalize=False, fmt='d'`?

- [ ] A) `label = format(val, fmt) if fmt == 'd' else f"{val:{fmt}}"` — `format(val, 'd')` fails for float arrays
- [ ] B) `thresh = vmax / 2` — for raw counts, threshold should be 0
- [ ] C) `vmax = cm.max()` — should be `cm.sum()` for count normalization
- [ ] D) `cmap = 'YlOrRd'` — wrong colormap for count-based confusion matrices

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `label = format(val, fmt) if fmt == 'd' else f"{val:{fmt}}"` — `format(val, 'd')` fails for float arrays

**Explanation:**

- **A is correct:** When `normalize=False`, `cm_display = cm` (the raw integer counts), but `cm_display[i, j]` returns a **NumPy scalar** which may be of type `int64` or `float64` depending on the original array dtype. The format spec `'d'` (integer format) **fails** with a `ValueError` for float values (e.g., `format(np.float64(45.0), 'd')` raises an error). If `cm` was created as `float` at any point, this will crash. Fix: use `int(val)` before formatting — `label = str(int(val))` — or convert `cm` to integers at the top: `cm = cm.astype(int)`.
- **B is incorrect:** `thresh = vmax / 2` for raw counts correctly places the text contrast threshold at the midpoint of the color range [0, vmax]. This is valid — cells with counts above half-maximum get white text (on dark background), cells below get black text.
- **C is incorrect:** `vmax = cm.max()` correctly sets the colormap maximum to the largest count in the matrix (typically on-diagonal correct predictions). `cm.sum()` would be the total number of samples — using it as vmax would compress all values into a small fraction of the colormap.
- **D is incorrect:** `YlOrRd` (yellow → orange → red) is a reasonable sequential colormap for count-based matrices where higher counts = more intense color. It is not wrong — `Blues` is also commonly used. Neither is "wrong" for confusion matrices.

</details>

---

### Question 99

A developer creates an **activation map dashboard** for a CNN and uses `plt.subplots_adjust()` to control spacing:

```python
fig, axes = plt.subplots(4, 8, figsize=(18, 10))
# Display 32 feature maps
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(acts[idx], cmap='RdBu', vmin=-acts_max, vmax=acts_max,
              interpolation='nearest', aspect='equal')
    ax.axis('off')

plt.suptitle("Layer 3 Activations", fontsize=14, y=0.98)
plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                    wspace=0.05, hspace=0.05)
plt.savefig("activations.png", dpi=180, bbox_inches='tight')
plt.close()
```

Why is `cmap='RdBu'` with symmetric `vmin=-acts_max, vmax=acts_max` used for this activation map?

- [ ] A) CNN activations are always in [-1, 1] — ReLU is not used
- [ ] B) For layers using activations that allow negative values (e.g., after batch normalization, or with Tanh/PReLU), a diverging colormap with symmetric range reveals the bipolar activation structure — blue = inhibited/negative, red = excited/positive, white = near-zero
- [ ] C) `RdBu` is the only colormap compatible with negative float values
- [ ] D) Symmetric vmin/vmax is only needed for the first layer — deeper layers are always non-negative

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) For layers using activations that allow negative values (e.g., after batch normalization, or with Tanh/PReLU), a diverging colormap with symmetric range reveals the bipolar activation structure — blue = inhibited/negative, red = excited/positive, white = near-zero

**Explanation:**

- **A is incorrect:** ReLU activations ARE non-negative (ReLU clips negatives to 0). But batch normalization, which follows conv layers in modern architectures, shifts activations to have zero mean — producing negative values even after ReLU. Tanh and PReLU produce negative activations by definition.
- **B is correct:** After batch normalization, activations are zero-centered with both positive and negative values. `cmap='RdBu'` with `vmin=-acts_max, vmax=acts_max` creates a semantically meaningful visualization: **red** = strongly positive activation (feature detected), **white** = near-zero (feature absent/neutral), **blue** = strongly negative (feature suppressed). This bipolar view reveals which neurons are excited (detecting their preferred pattern) vs. suppressed (detecting the anti-pattern) for each spatial location. Symmetric `vmin/vmax` ensures zero always maps to white — the colormap center.
- **C is incorrect:** All Matplotlib colormaps handle negative float values. Any colormap can display negative values when `vmin` is set to a negative number. `RdBu` has no special capability for negative values beyond other colormaps.
- **D is incorrect:** Deeper layers can also have negative activations — especially after batch normalization (which is applied throughout the network in modern architectures like ResNet, VGG, etc.). The claim that only early layers need symmetric vmin/vmax is incorrect.

</details>

---

### Question 100

A developer builds a **complete production heatmap dashboard** as the final capstone. Identify which statements about the following code are TRUE:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(99)
n_classes = 5
class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Dog']

# Confusion matrix
cm = np.array([[48, 1, 0, 1, 0],
               [0, 50, 0, 0, 0],
               [2, 0, 44, 2, 2],
               [1, 0, 3, 44, 2],
               [0, 0, 1, 2, 47]])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
acts_max = np.abs(cm_norm).max()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Model Evaluation Dashboard", fontsize=14, fontweight='bold')

# Panel 1: Raw count
im1 = axes[0].imshow(cm, cmap='YlOrRd', vmin=0, vmax=cm.max(),
                      interpolation='nearest', aspect='equal')
for i in range(n_classes):
    for j in range(n_classes):
        c = 'white' if cm[i,j] > cm.max()/2 else 'black'
        axes[0].text(j, i, str(cm[i,j]), ha='center', va='center',
                     color=c, fontsize=10)
axes[0].set_title("Confusion Matrix (Counts)")
fig.colorbar(im1, ax=axes[0], shrink=0.8, label='Count')

# Panel 2: Normalized recall
im2 = axes[1].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1,
                      interpolation='nearest', aspect='equal')
for i in range(n_classes):
    for j in range(n_classes):
        c = 'white' if cm_norm[i,j] > 0.5 else 'black'
        axes[1].text(j, i, f"{cm_norm[i,j]:.1%}", ha='center', va='center',
                     color=c, fontsize=9)
axes[1].set_title("Normalized Confusion Matrix (Recall)")
fig.colorbar(im2, ax=axes[1], shrink=0.8, label='Recall')

for ax in axes:
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("eval_dashboard.png", dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close(fig)
```

Which statements are TRUE? (Select all that apply)

- [ ] A) `ax.text(j, i, ...)` — correct: x=j (column), y=i (row) — text is placed at the center of each cell
- [ ] B) `cm.sum(axis=1, keepdims=True)` produces a `(5, 1)` array enabling correct row-wise broadcasting division
- [ ] C) `plt.tight_layout(rect=[0, 0, 1, 0.95])` reserves the top 5% of figure height for the `suptitle`
- [ ] D) `matplotlib.use('Agg')` must be called AFTER `import matplotlib.pyplot` to take effect

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A, B, C

**Explanation:**

- **A is correct — TRUE:** `ax.text(x, y, ...)` uses data coordinates where x=column and y=row. `ax.text(j, i, ...)` with outer loop `i` (rows) and inner loop `j` (columns) correctly places text at column `j`, row `i` — the center of cell `[i, j]`. This is the standard, correct coordinate ordering for `imshow` cell annotations.
- **B is correct — TRUE:** `cm.sum(axis=1)` would give shape `(5,)`. With `keepdims=True`, the result is `(5, 1)`. Dividing `(5, 5) / (5, 1)` broadcasts correctly: each row of `cm` is divided by its corresponding row sum — producing per-class recall (row normalization). This is the essential `keepdims` use case for confusion matrix normalization.
- **C is correct — TRUE:** `plt.tight_layout(rect=[left, bottom, right, top])` constrains subplot layout to the specified rectangle in normalized figure coordinates. `rect=[0, 0, 1, 0.95]` restricts subplots to y∈[0, 0.95], leaving the top 5% (y∈[0.95, 1.0]) free for `fig.suptitle(...)`. Without this, tight_layout and suptitle overlap.
- **D is incorrect — FALSE:** `matplotlib.use('Agg')` MUST be called **BEFORE** `import matplotlib.pyplot` to take effect. Once pyplot is imported, the backend is initialized and locked. Calling `use()` after pyplot import raises a warning and may have no effect. The correct order is: `import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt`.

</details>
