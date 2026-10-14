# Matplotlib Pie Chart Objective Test — Set 08

---

### 71. Question

What is the primary architectural purpose of a **nested (multi-ring) concentric donut chart** in data visualization?

- [ ] **A.** To plot time-series lines on a circular radar grid
- [ ] **B.** To visualize hierarchical part-to-whole relationships, where an outer ring displays macro parent categories and an inner concentric ring displays detailed sub-category breakdowns
- [ ] **C.** To calculate Pearson correlation coefficients between two continuous features
- [ ] **D.** To replace a 3D scatter plot

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Radar/spider charts plot multivariate profiles, not nested donut rings.
- **Option B (Correct):** Nested concentric donut charts are designed specifically to display two-level (or multi-level) hierarchical data. The outer annular ring represents high-level parent segments (e.g., Continents), while the inner concentric ring subdivides each parent segment into child sub-categories (e.g., Countries within each Continent), preserving angular alignment between parent and children.
- **Option C (Incorrect):** Correlation is visualized via scatter plots or heatmaps.
- **Option D (Incorrect):** 3D scatter plots display spatial data; nested rings represent discrete categorical hierarchies.

</details>

---

### 72. Question

An engineer designs a two-ring concentric donut chart using the following parameters:
- Outer ring: `radius=1.0`, `wedgeprops=dict(width=0.3)`
- Inner ring: `radius=0.7`, `wedgeprops=dict(width=0.3)`

What is the exact radius of the **empty central hole** remaining at the center of the inner ring?

- [ ] **A.** `0.4`
- [ ] **B.** `0.1`
- [ ] **C.** `0.7`
- [ ] **D.** `0.3`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):**
  - The outer ring has outer radius $1.0$ and width $0.3$, spanning from $r = 0.7$ to $r = 1.0$.
  - The inner ring begins precisely at the inner boundary of the outer ring ($r = 0.7$) with width $0.3$, spanning from $r = 0.4$ to $r = 0.7$.
  - The inner radius of the inner ring (the central hole) is calculated as:
    $$r_{\text{hole}} = r_{\text{inner\_outer}} - \text{width} = 0.7 - 0.3 = \mathbf{0.4}$$
- **Option B (Incorrect):** $0.1$ would require subtracting $0.6$ from $0.7$.
- **Option C (Incorrect):** $0.7$ is the outer boundary of the inner ring, not the hole.
- **Option D (Incorrect):** $0.3$ is the thickness of each individual ring.

</details>

---

### 73. Question

When building a nested two-ring donut chart, why is it mandatory that **both** `ax.pie()` calls specify the exact same `startangle` and `counterclock` parameters?

- [ ] **A.** Because Matplotlib will throw a `ConcentricAngleException` if they differ.
- [ ] **B.** Because different start angles cause the inner sub-category slices to rotate out of alignment with their respective outer parent categories, breaking hierarchical visual correspondence.
- [ ] **C.** Because the outer ring will become invisible if the angles do not match.
- [ ] **D.** Because it forces the colors to blend into grayscale.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not check or raise exceptions for differing angles across independent `pie()` calls.
- **Option B (Correct):** For a nested hierarchy to be meaningful, the radial dividing lines of the child sub-categories on the inner ring must align geometrically within the angular boundaries of their parent wedge on the outer ring. If the two calls use different `startangle` or `counterclock` settings, the inner slices will rotate independently, misaligning children from their true parents and producing a misleading graphic.
- **Option C (Incorrect):** Wedges render independently regardless of rotational alignment.
- **Option D (Incorrect):** Rotational angles have zero impact on palette colors.

</details>

---

### 74. Question

How are concentric multi-ring donut charts constructed programmatically in Matplotlib?

- [ ] **A.** By calling `plt.nested_pie()`
- [ ] **B.** By creating a 3D projection axes
- [ ] **C.** By setting `rings=2` inside a single `plt.pie()` call
- [ ] **D.** By executing two successive `ax.pie()` calls on the same Axes object with different `radius` and `width` values without clearing the canvas

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** There is no `nested_pie()` function in Matplotlib.
- **Option B (Incorrect):** 3D projections are not used for 2D concentric rings.
- **Option C (Incorrect):** `plt.pie()` does not have a `rings` argument; it accepts a 1D sequence of data.
- **Option D (Correct):** Concentric rings are constructed by leveraging Matplotlib's layered artist architecture. Calling `ax.pie(outer_data, radius=1.0, wedgeprops=dict(width=0.3))` draws the outer ring. Immediately calling `ax.pie(inner_data, radius=0.7, wedgeprops=dict(width=0.3))` on the same `ax` overlays the inner ring perfectly nested inside, producing a seamless multi-tier visualization.

</details>

---

### 75. Question

What is the recommended color palette design pattern for effectively communicating hierarchy in a nested donut chart?

- [ ] **A.** Use a primary hue (e.g., dark blue) for the parent category on the outer ring, and varying tints/shades of that same hue (e.g., medium blue, light blue) for its child sub-categories on the inner ring.
- [ ] **B.** Assign random rainbow colors to all inner and outer slices.
- [ ] **C.** Make the outer ring completely black and the inner ring completely white.
- [ ] **D.** Alternate red and green on every adjacent wedge across both rings.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Color should reinforce categorical hierarchy. When Parent Category A is shaded dark blue and Parent Category B is shaded dark orange, their respective child sub-categories on the inner ring should use coordinated lighter tints of blue and orange. This allows viewers to instantly associate each inner slice with its parent macro-group without reading text labels.
- **Option B (Incorrect):** Random colors sever the visual grouping between parent and child segments.
- **Option C (Incorrect):** Monochrome black and white obscures multiple sub-categories.
- **Option D (Incorrect):** Red-green alternation creates visual noise and causes severe accessibility problems for colorblind viewers.

</details>

---

### 76. Question

Why is using the `explode` parameter generally avoided in nested multi-ring donut charts?

- [ ] **A.** Exploding a slice in one ring displaces it radially, breaking the geometric boundary continuity with the adjacent ring and causing overlapping or disjointed wedge intersections.
- [ ] **B.** Exploding slices causes Matplotlib to delete the inner ring.
- [ ] **C.** Matplotlib raises an `ExplodeCollisionError` if `explode` is used with `wedgeprops`.
- [ ] **D.** Exploded slices cannot be rendered in color.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Concentric rings rely on shared circular contours and shared radial boundaries. If an outer wedge is exploded outward, it detaches from its corresponding inner child wedges, destroying the visual link between parent and child. Conversely, if an inner wedge is exploded, it pushes outward and physically collides with the solid band of the outer ring.
- **Option B (Incorrect):** Matplotlib does not delete artists.
- **Option C (Incorrect):** No exception is raised, but the visual result is broken and distorted.
- **Option D (Incorrect):** Exploded slices retain their assigned fill colors.

</details>

---

### 77. Question

To avoid overlapping text in a two-ring nested donut chart, what is the best labeling strategy?

- [ ] **A.** Print full category names and percentages on both the inner and outer rings simultaneously
- [ ] **B.** Display parent labels outside the outer ring (e.g., `labeldistance=1.15`), and display only percentage values inside the inner ring slices (or use a dedicated legend for sub-categories)
- [ ] **C.** Suppress all text and print nothing
- [ ] **D.** Force all labels into the central hole

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Slices in concentric rings have limited radial and circumferential space; printing full labels on both rings causes catastrophic text collisions.
- **Option B (Correct):** Spatial separation is key: placing parent labels outside the perimeter of the outer ring (`labeldistance=1.15` or greater) keeps the outer margin clear. The inner ring can then display clean numerical percentages inside its wedges via `autopct` or rely on an external legend, avoiding text collisions between tiers.
- **Option C (Incorrect):** Omitting all labels makes the graphic uninterpretable.
- **Option D (Incorrect):** Cramming dozens of labels into the central hole creates an unreadable knot of text.

</details>

---

### 78. Question

A biomedical researcher builds a three-ring nested donut chart with the following specifications:
- Outer ring: `radius=1.0`, `wedgeprops=dict(width=0.2)`
- Middle ring: `radius=0.8`, `wedgeprops=dict(width=0.2)`
- Inner ring: `radius=0.6`, `wedgeprops=dict(width=0.2)`

What is the exact radius of the **empty central hole**?

- [ ] **A.** `0.2`
- [ ] **B.** `0.6`
- [ ] **C.** `0.0`
- [ ] **D.** `0.4`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** $0.2$ is the width of each individual annular band.
- **Option B (Incorrect):** $0.6$ is the outer radius of the inner ring.
- **Option C (Incorrect):** The inner hole is not zero.
- **Option D (Correct):**
  - Outer ring spans $r \in [0.8, 1.0]$
  - Middle ring spans $r \in [0.6, 0.8]$
  - Inner ring spans $r \in [r_{\text{inner\_outer}} - \text{width}, r_{\text{inner\_outer}}] = [0.6 - 0.2, 0.6] = [0.4, 0.6]$
  The central hole radius is $0.6 - 0.2 = \mathbf{0.4}$.

</details>

---

### 79. Question

A parent category on the outer ring occupies an angular sweep of exactly $120^\circ$. It has two child sub-categories on the inner ring with observation counts in a $3:1$ ratio.

What are the respective angular sweeps of the two child wedges on the inner ring?

- [ ] **A.** $90^\circ$ and $30^\circ$
- [ ] **B.** $60^\circ$ and $60^\circ$
- [ ] **C.** $100^\circ$ and $20^\circ$
- [ ] **D.** $80^\circ$ and $40^\circ$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The two children partition the parent's total angular span of $120^\circ$ according to their $3:1$ ratio (a total of $3 + 1 = 4$ parts):
  $$\text{Child 1} = \frac{3}{4} \times 120^\circ = \mathbf{90^\circ}$$
  $$\text{Child 2} = \frac{1}{4} \times 120^\circ = \mathbf{30^\circ}$$
  Sum: $90^\circ + 30^\circ = 120^\circ$.
- **Option B (Incorrect):** $60^\circ$ and $60^\circ$ represents a $1:1$ ratio.
- **Option C (Incorrect):** $100^\circ$ and $20^\circ$ represents a $5:1$ ratio.
- **Option D (Incorrect):** $80^\circ$ and $40^\circ$ represents a $2:1$ ratio.

</details>

---

### 80. Question

Consider the following statements regarding nested (multi-ring) donut charts in Matplotlib:

1. Nested donut charts are created by calling `ax.pie()` multiple times on the same Axes with descending `radius` arguments.
2. The inner ring's outer radius should equal the inner boundary of the outer ring ($r_{\text{inner\_ring}} = r_{\text{outer\_ring}} - \text{width}_{\text{outer}}$) to eliminate gaps between bands.
3. Matching `startangle` and `counterclock` across all ring calls is essential for proper angular alignment between parent and child wedges.
4. Using coordinated shades of a base hue establishes clear visual grouping between parent categories and their child sub-categories.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Statement 1 is correct:** Successive calls overlay concentric rings on the shared Axes.
- **Statement 2 is correct:** Matching boundaries ensures adjacent rings touch smoothly without awkward white gaps.
- **Statement 3 is correct:** Rotational synchronization keeps child wedges nested directly within their parent's angular sector.
- **Statement 4 is correct:** Tonal color families provide immediate pre-attentive hierarchical cues.
- **Conclusion:** All four statements are correct, making **Option D** the correct choice.

</details>

---
