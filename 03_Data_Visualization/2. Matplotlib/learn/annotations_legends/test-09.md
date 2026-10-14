# Matplotlib Annotations & Legends — Test 09: Advanced Legend Customization

---

### Question 81

A machine learning engineer plots training and validation curves and then tries to create two separate legends:

```python
fig, ax = plt.subplots()
l1, = ax.plot(epochs, train_loss, color="blue", label="Train Loss")
l2, = ax.plot(epochs, val_loss, color="orange", label="Val Loss")
l3 = ax.axhline(y=0.1, color="green", linestyle="--", label="Target")

legend1 = ax.legend(handles=[l1, l2], loc="upper right")
legend2 = ax.legend(handles=[l3], loc="lower right", title="Reference")
```

What bug exists in this code and what is the symptom?

- [ ] A) `ax.axhline` cannot be included in a legend — a `TypeError` is raised
- [ ] B) The second `ax.legend()` call silently removes `legend1` from the axes — only `legend2` is visible
- [ ] C) Both legends are visible but they overlap at the right edge of the axes
- [ ] D) `handles=[l1, l2]` and `handles=[l3]` conflict, raising a `DuplicateArtistError`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The second `ax.legend()` call silently removes `legend1` from the axes — only `legend2` is visible

**Explanation:**

- **A is incorrect:** `ax.axhline` returns a `Line2D` object that is fully compatible with the `handles` argument of `ax.legend()`. No `TypeError` is raised.
- **B is correct:** This is the classic "two-legend bug." In Matplotlib, each `ax.legend()` call replaces the previous legend on the axes. When the second `ax.legend()` is called, it creates a new `Legend` object and registers it as the axes' current legend, **automatically removing `legend1`** from the axes. The fix is to call `ax.add_artist(legend1)` after creating the first legend, before creating the second. `add_artist` re-adds the first legend as a persistent artist that won't be replaced by subsequent `ax.legend()` calls.
- **C is incorrect:** The two legends do not coexist — only the second one remains. There is no overlap scenario because the first legend is replaced, not supplemented.
- **D is incorrect:** No `DuplicateArtistError` exists in Matplotlib. The `handles` argument is independent between two `ax.legend()` calls, and no conflict detection is performed.

</details>

---

### Question 82

A researcher correctly implements two legends on a single axes using `add_artist`:

```python
line_artists = [l1, l2, l3]
line_labels = ["Train", "Val", "Test"]

ref_artists = [thresh_line, baseline]
ref_labels = ["Threshold", "Baseline"]

legend1 = ax.legend(handles=line_artists, labels=line_labels,
                    loc="upper right")
ax.add_artist(legend1)
legend2 = ax.legend(handles=ref_artists, labels=ref_labels,
                    loc="lower left", title="References")
```

Why is `ax.add_artist(legend1)` called before creating `legend2`?

- [ ] A) `add_artist` is required to set the legend's `zorder` higher than the default
- [ ] B) After `ax.legend()` creates `legend1`, it is not yet added to the axes' artist collection; `add_artist` explicitly registers it so the subsequent `ax.legend()` call for `legend2` does not destroy it
- [ ] C) `add_artist` makes `legend1` interactive (clickable) in the GUI backend
- [ ] D) `add_artist` converts the legend from a `Legend` object to a `Patch`, enabling better styling

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) After `ax.legend()` creates `legend1`, it is not yet added to the axes' artist collection; `add_artist` explicitly registers it so the subsequent `ax.legend()` call for `legend2` does not destroy it

**Explanation:**

- **A is incorrect:** `add_artist` has no effect on `zorder`. `zorder` is set via the `zorder` keyword argument on the legend or through `legend1.set_zorder(N)`.
- **B is correct:** `ax.legend()` returns a `Legend` object and registers it as the axes' `_legend` attribute. When a second `ax.legend()` is called, Matplotlib removes the existing `_legend` and replaces it with the new one. However, `ax.add_artist(legend1)` registers `legend1` into the axes' `_artists` list — a separate collection from `_legend`. Artists in `_artists` are not removed by subsequent `ax.legend()` calls. By calling `add_artist` before the second `ax.legend()`, `legend1` is preserved in the `_artists` collection even when `_legend` is replaced by `legend2`.
- **C is incorrect:** `add_artist` does not change the interactivity or event handling behavior of the legend. It simply adds the object to the axes' list of persistent artists.
- **D is incorrect:** `add_artist` does not convert the artist type. A `Legend` object remains a `Legend` object after being added via `add_artist`.

</details>

---

### Question 83

A data analyst customizes the legend frame:

```python
leg = ax.legend(loc="upper left",
                frameon=True,
                facecolor="#F8F4E8",
                edgecolor="#8B7355",
                framealpha=0.95,
                shadow=True,
                fancybox=True)
```

What does `fancybox=True` change about the legend's appearance?

- [ ] A) Adds a drop shadow on the right and bottom sides of the legend box
- [ ] B) Replaces the square corners of the legend box with rounded corners
- [ ] C) Adds gradient fill to the legend background
- [ ] D) Makes the legend box interactive — clicking it toggles visibility

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Replaces the square corners of the legend box with rounded corners

**Explanation:**

- **A is incorrect:** The drop shadow is added by `shadow=True`, not `fancybox=True`. Both are present in this code and work independently: `shadow=True` adds a shadow, `fancybox=True` rounds the corners.
- **B is correct:** `fancybox=True` changes the `Legend`'s background patch from a sharp-cornered rectangle to a rounded rectangle (using `FancyBboxPatch` with `boxstyle="round,pad=0.4"` or similar). The result is a legend box with softly rounded corners — a subtle aesthetic touch common in publication and presentation graphics.
- **C is incorrect:** Matplotlib's standard legend does not support gradient fills for the frame background. `facecolor` sets a uniform solid colour; gradient fills would require custom patch classes.
- **D is incorrect:** `fancybox=True` is purely a visual parameter. Matplotlib's standard legend boxes are not interactive click targets by default — no built-in click-to-hide functionality is added by `fancybox`.

</details>

---

### Question 84

A machine learning engineer uses `markerscale=2.5` in the legend:

```python
ax.scatter(X_test, y_test, s=10, label="Test Samples", color="teal")
ax.legend(markerscale=2.5)
```

The scatter marker size `s=10` is very small on the main plot. What does `markerscale=2.5` do in the legend?

- [ ] A) Increases the scatter plot marker size to s=25 on the main plot
- [ ] B) Scales the legend handle marker to 2.5× the size it would normally appear in the legend, making small markers more visible in the legend without affecting the main plot
- [ ] C) Adds 2.5 points of padding around the legend marker handle
- [ ] D) Sets the marker transparency to 1/2.5 = 0.4 in the legend

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Scales the legend handle marker to 2.5× the size it would normally appear in the legend, making small markers more visible in the legend without affecting the main plot

**Explanation:**

- **A is incorrect:** `markerscale` only affects the legend's handle markers. It has zero effect on the main scatter plot's `s` parameter. The scatter markers on the main plot remain at `s=10`.
- **B is correct:** `markerscale` is a multiplicative scaling factor applied to legend handle markers. The legend normally renders a miniature version of the actual marker used in the plot. With `markerscale=2.5`, the legend marker is drawn at 2.5× its standard legend size, making it easier to see and identify in the legend even when the actual plot markers are very small (e.g., `s=10` in a dense scatter). This is purely a legend rendering parameter.
- **C is incorrect:** Padding within the legend is controlled by `handletextpad` (space between handle and text) and `labelspacing` (space between rows). `markerscale` is a size multiplier, not a padding parameter.
- **D is incorrect:** `markerscale` is a size scaling factor, not an opacity modifier. Transparency would be controlled by `alpha`.

</details>

---

### Question 85

A researcher wants to add a custom proxy artist to the legend for a filled region created with `fill_between`, which doesn't have a built-in legend handle. Which code correctly creates a proxy patch?

- [ ] A) `from matplotlib.patches import Patch; proxy = Patch(facecolor="blue", alpha=0.3, label="Confidence Band")`
- [ ] B) `proxy = ax.fill_between(x, y1, y2, label="Confidence Band")` and then `ax.legend()`
- [ ] C) `proxy = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.3)` and `ax.legend(handles=[proxy])`
- [ ] D) Both A and B are valid approaches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Both A and B are valid approaches

**Explanation:**

- **A is partially correct:** `matplotlib.patches.Patch(facecolor="blue", alpha=0.3, label="Confidence Band")` creates a standalone Patch artist intended purely for legend representation. It must then be passed to `ax.legend(handles=[proxy])`. This is the "proxy artist" approach — creating a representative symbol just for the legend without adding it to the axes plot.
- **B is partially correct:** `ax.fill_between(x, y1, y2, label="Confidence Band")` directly labels the `PolyCollection` artist returned by `fill_between`. When `ax.legend()` is called, this labeled artist is automatically included in the legend with a representative colour swatch. This is the simpler, more direct approach.
- **C is incorrect while close:** `plt.Rectangle` is a valid patch but requires (x, y, width, height) where (0,0), 1, 1 represents a unit square starting at the origin in data space. While a `Rectangle` can technically be passed as a proxy handle, `matplotlib.patches.Patch` is the idiomatic proxy for legend-only colour swatches, as it doesn't imply specific x, y positioning.
- **D is correct:** Both approach A (proxy Patch) and approach B (label argument on `fill_between`) are valid and produce correct legend entries for the shaded region. Approach B is simpler; approach A provides more control over the legend symbol's appearance when the default `fill_between` handle doesn't match the desired aesthetic.

</details>

---

### Question 86

A data scientist wants to reorder legend entries so that the best-performing model appears first. She has:

```python
handles, labels = ax.get_legend_handles_labels()
# Current order: ["Model C", "Model A", "Model B"]
# Desired order: ["Model A", "Model B", "Model C"]
```

Which code produces the desired reordering?

- [ ] A) `ax.legend(handles=sorted(handles), labels=sorted(labels))`
- [ ] B) `order = [1, 2, 0]; ax.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order])`
- [ ] C) `ax.legend(handles=handles[::-1], labels=labels[::-1])`
- [ ] D) `ax.reorder_legend(["Model A", "Model B", "Model C"])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `order = [1, 2, 0]; ax.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order])`

**Explanation:**

- **A is incorrect:** `sorted(handles)` would sort `Line2D` objects, which are not naturally sortable by label. `sorted(labels)` would sort alphabetically, which might not always match the desired order and would misalign handles and labels.
- **B is correct:** The current order is [C=0, A=1, B=2]. The desired order [A, B, C] corresponds to indices [1, 2, 0]. Using `order = [1, 2, 0]` and list comprehensions to reindex both `handles` and `labels` produces the correct reordering. This pattern — manually specifying the desired index order — is the standard technique for reordering legend entries in Matplotlib.
- **C is incorrect:** `handles[::-1]` and `labels[::-1]` reverse the list, producing order [B=2, A=1, C=0] — which is "Model B, Model A, Model C". This is neither the original nor the desired order.
- **D is incorrect:** `ax.reorder_legend()` is not a valid Matplotlib method. Reordering is done manually through index manipulation of `handles` and `labels`.

</details>

---

### Question 87

A developer uses `handlelength=0` in the legend:

```python
ax.legend(handlelength=0, handletextpad=0.5)
```

What visual effect does `handlelength=0` produce?

- [ ] A) The legend handles (line/marker swatches) are removed — only text labels appear
- [ ] B) The legend text labels are removed — only colour swatches appear
- [ ] C) The legend box disappears entirely
- [ ] D) An error: `handlelength` cannot be set to 0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The legend handles (line/marker swatches) are removed — only text labels appear

**Explanation:**

- **A is correct:** `handlelength` controls the horizontal length of the line sample drawn in the legend handle (the coloured line or marker swatch before the label text). Setting `handlelength=0` collapses the handle's horizontal extent to zero, effectively removing the visible line sample. Only the text labels remain. This creates a text-only legend — useful for categorical annotations where the association is clear from context without needing explicit line samples.
- **B is incorrect:** `handlelength=0` removes the handle (colour swatch/line sample), not the text labels. Text label visibility is separate and would require setting `ax.legend()` to have empty label strings, or filtering entries.
- **C is incorrect:** Setting `handlelength=0` does not hide the entire legend box. The legend box remains with only its text content.
- **D is incorrect:** `handlelength=0` is a valid value. Matplotlib accepts any non-negative float for this parameter.

</details>

---

### Question 88

A researcher creates a figure and notices that the legend title is displayed without bold formatting, even though she expects it to be bold by default. Which approach correctly makes the legend title bold?

- [ ] A) `ax.legend(title="Performance Metrics", title_fontweight="bold")`
- [ ] B) `leg = ax.legend(title="Performance Metrics"); leg.get_title().set_fontweight("bold")`
- [ ] C) `ax.legend(title="**Performance Metrics**")` — using markdown bold syntax
- [ ] D) `ax.legend(title="Performance Metrics", title_fontstyle="bold")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `leg = ax.legend(title="Performance Metrics"); leg.get_title().set_fontweight("bold")`

**Explanation:**

- **A is incorrect:** `title_fontweight=` is not a documented keyword argument for `ax.legend()` in standard Matplotlib releases. The legend API's title font properties are limited in what can be passed directly through `ax.legend()` kwargs.
- **B is correct:** `ax.legend()` returns a `Legend` object. `leg.get_title()` retrieves the `Text` artist used for the legend title. Calling `set_fontweight("bold")` on that `Text` artist sets the title to bold. This post-creation approach is the standard method for modifying title font properties beyond what `ax.legend()` directly supports.
- **C is incorrect:** Matplotlib text does not process markdown formatting. `"**Performance Metrics**"` would be rendered literally as the string with double asterisks — `**Performance Metrics**`.
- **D is incorrect:** `title_fontstyle=` is also not a documented `ax.legend()` keyword. `fontstyle` controls italic vs. normal (not bold). Even if it were accepted, `title_fontstyle="bold"` would use the wrong property name (`style` vs. `weight`).

</details>

---

### Question 89

A data scientist wants to filter out two of five labeled artists from the legend without relabeling them. Her plot produces handles `[h1, h2, h3, h4, h5]` and labels `["A", "B", "C", "D", "E"]`. She wants only `["A", "C", "E"]` in the legend. Which code achieves this without changing the plot calls?

- [ ] A) Call `ax.legend(exclude=["B", "D"])` to filter by label
- [ ] B) Get all handles and labels, filter by index, and pass to `ax.legend`: `ax.legend(handles=[h1, h3, h5], labels=["A", "C", "E"])`
- [ ] C) Rename the labels: `ax.plot(..., label="_B")` and `ax.plot(..., label="_D")` using underscore prefix
- [ ] D) Both B and C are valid approaches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Both B and C are valid approaches

**Explanation:**

- **A is incorrect:** `exclude=` is not a valid keyword argument for `ax.legend()`. There is no built-in exclusion filter in the `ax.legend()` API.
- **B is partially correct:** Explicitly passing `handles=[h1, h3, h5]` and `labels=["A", "C", "E"]` to `ax.legend()` produces a legend with exactly the three desired entries. `h2` and `h4` are excluded. However, this requires knowing the handles in advance or using `ax.get_legend_handles_labels()` and then filtering by index.
- **C is partially correct:** Using underscore-prefixed labels like `label="_B"` or `label="_nolegend_"` makes those artists invisible to the auto-legend collection in `ax.legend()`. When `ax.legend()` is called without explicit handles, `h2` (label `"_B"`) and `h4` (label `"_D"`) are automatically excluded, leaving only "A", "C", "E". This requires modifying the `ax.plot()` calls but no post-hoc filtering.
- **D is correct:** Both B (post-hoc explicit handle filtering) and C (underscore-label exclusion at plot time) are valid and correct approaches to filtering legend entries. B is more flexible for arbitrary filtering after the fact; C is cleaner for cases where the exclusion is known at plot time.

</details>

---

### Question 90

A researcher adjusts multiple legend spacing parameters:

```python
ax.legend(labelspacing=0.2,
          handletextpad=0.4,
          columnspacing=1.5,
          borderpad=0.6)
```

Which parameter controls the vertical space between individual legend entries?

- [ ] A) `handletextpad=0.4` — the gap between the colour swatch and the text label
- [ ] B) `labelspacing=0.2` — the vertical spacing between rows of legend entries
- [ ] C) `columnspacing=1.5` — the space between columns (and thus rows)
- [ ] D) `borderpad=0.6` — the internal padding inside the legend box boundary

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `labelspacing=0.2` — the vertical spacing between rows of legend entries

**Explanation:**

- **A is incorrect:** `handletextpad` controls the horizontal gap between the legend handle (colour swatch/line) and the associated text label. It is a horizontal spacing parameter, not a vertical row separator.
- **B is correct:** `labelspacing` specifies the vertical space between consecutive legend entries (rows), measured in font-size units. A value of 0.2 (smaller than the default ~0.5) produces tighter, more compact row spacing between individual legend labels. Increasing it to 1.0 or more would create more spread-out, airy legend entries.
- **C is incorrect:** `columnspacing` controls the horizontal space between adjacent columns in a multi-column legend. It does not affect vertical row separation.
- **D is incorrect:** `borderpad` controls the padding between the legend box boundary and its internal content (handles and labels). It affects the margins around all content as a whole, not the spacing between individual entries.

</details>
