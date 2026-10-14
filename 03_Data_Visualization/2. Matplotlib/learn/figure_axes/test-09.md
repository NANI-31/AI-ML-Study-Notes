# Matplotlib Figure & Axes Objective Test — Set 09
# Topic: Memory Management & Object Lifecycle (Questions 81–90)

---

### Question 81

A developer writes a batch evaluation script that generates 1,000 model evaluation figures:

```python
for i, model in enumerate(models):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(model.history["loss"])
    ax.set_title(f"Model {i} Loss")
    plt.savefig(f"results/model_{i}.png", dpi=150)
    # No close call!
```

After running 200 iterations, the script crashes with `MemoryError`. What is the root cause?

- [ ] A) `plt.savefig()` loads all previously saved files back into memory
- [ ] B) Each `plt.subplots()` call creates a `Figure` object that is registered in Matplotlib's figure manager — without `plt.close(fig)`, all 200 figures accumulate in RAM simultaneously
- [ ] C) `dpi=150` is too high for batch processing and exhausts memory
- [ ] D) `ax.plot()` accumulates all 200 datasets in a shared internal buffer

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Each `plt.subplots()` call creates a `Figure` object registered in Matplotlib's figure manager — without `plt.close(fig)`, all 200 figures accumulate in RAM simultaneously

**Explanation:**

- **A is incorrect:** `plt.savefig()` writes to disk and does not re-load previous files into memory.
- **B is correct:** Every call to `plt.subplots()` (or `plt.figure()`) creates a `Figure` object that is registered in `matplotlib._pylab_helpers.Gcf` (the figure manager). Without `plt.close(fig)`, the figure manager holds a reference to every figure — preventing garbage collection. Each figure at `figsize=(8,6)` with `dpi=150` holds an in-memory raster buffer of 8×150 × 6×150 × 4 bytes RGBA = 1200×900×4 ≈ **4.3 MB per figure**. After 200 iterations: 200 × 4.3 MB ≈ **860 MB** of figure memory — causing a `MemoryError`. The fix is to add `plt.close(fig)` immediately after each `savefig()`.
- **C is incorrect:** `dpi=150` is a moderate resolution. The issue is accumulation of figures, not per-figure DPI. Even at `dpi=72`, the leak would still occur.
- **D is incorrect:** `ax.plot()` stores data in the `Line2D` artist attached to the axes — not in a shared cross-figure buffer. Each figure's data is isolated.

</details>

---

### Question 82

A developer fixes the batch loop from the previous question and compares two approaches:

```python
# Approach A:
plt.close(fig)

# Approach B:
plt.close("all")
```

In the context of a single-figure batch loop, which is preferred and why?

- [ ] A) Approach B (`plt.close("all")`) is always preferred — it guarantees all memory is freed
- [ ] B) Approach A (`plt.close(fig)`) is preferred in a loop — it closes only the current iteration's figure, avoiding accidentally closing figures created elsewhere (e.g., in a parent function or interactive session)
- [ ] C) Both are equivalent in a single-figure loop
- [ ] D) Neither works — figures in a loop require `del fig` for memory release

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Approach A (`plt.close(fig)`) is preferred in a loop — it closes only the current iteration's figure

**Explanation:**

- **A is incorrect:** `plt.close("all")` is aggressive — it closes **every open figure** in the Matplotlib figure manager, including figures that may have been created in other contexts. In a library or complex script with multiple figures open, this could accidentally close unrelated figures.
- **B is correct:** `plt.close(fig)` is the **targeted, safe approach** — it closes only the specific `Figure` object `fig` from the current loop iteration. If the script has other open figures (an interactive inspection figure, a progress figure, etc.), they are unaffected. In isolated batch loops where only one figure exists at a time, `plt.close(fig)` and `plt.close("all")` produce identical results — but `plt.close(fig)` is safer general practice.
- **C is incorrect:** While functionally equivalent in an isolated single-figure loop, they have different semantics. `plt.close("all")` closes ALL figures; `plt.close(fig)` closes ONE. They are not equivalent in contexts with multiple figures.
- **D is incorrect:** `del fig` removes the Python variable `fig` but does NOT guarantee memory release if the figure manager still holds a reference. `plt.close(fig)` deregisters the figure from the manager first, THEN `del fig` (or scope exit) allows garbage collection.

</details>

---

### Question 83

A developer uses `fig.clf()` to reuse a figure object across multiple plots in a GUI application:

```python
fig = plt.figure(figsize=(8, 5))

for i in range(10):
    fig.clf()                          # Clear figure
    ax = fig.add_subplot(1, 1, 1)     # Add fresh axes
    ax.plot(data[i], label=f"Run {i}")
    ax.legend()
    fig.canvas.draw()                  # Redraw
```

What does `fig.clf()` do, and how does this differ from creating a new figure each iteration?

- [ ] A) `fig.clf()` closes the figure window and reopens it; creating a new figure always keeps the old window open
- [ ] B) `fig.clf()` removes all axes, artists, and text from the figure object while keeping the figure registered in the manager and its window open; creating a new figure each iteration would accumulate 10 figures in memory
- [ ] C) `fig.clf()` resets the DPI and figsize to defaults; creating a new figure preserves these settings
- [ ] D) Both are identical — `fig.clf()` internally creates a new figure object

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `fig.clf()` removes all axes, artists, and text from the figure object while keeping the figure registered in the manager and its window open

**Explanation:**

- **A is incorrect:** `fig.clf()` does NOT close or reopen windows. The figure window (if open in a GUI backend) remains open and connected.
- **B is correct:** `fig.clf()` ("clear figure") calls `Figure.clear()` internally, which: removes all `Axes` from `fig.axes`, removes all figure-level artists (suptitle, text, patches), resets the figure's children list. The `Figure` object itself remains alive, registered in the manager, and its GUI window stays open. This is the correct pattern for GUI applications that need to update the same window repeatedly — one figure window stays open and its content is refreshed. Using `plt.figure()` each iteration would create 10 separate figure objects (10 windows in interactive mode, 10 objects in memory).
- **C is incorrect:** `fig.clf()` does NOT reset `figsize` or DPI. These properties belong to the `Figure` object itself and persist through `clf()`.
- **D is incorrect:** `fig.clf()` clears the SAME figure object — it does not create a new one. `id(fig)` remains the same before and after `clf()`.

</details>

---

### Question 84

A developer uses `ax.cla()` instead of `fig.clf()` in an animation update function:

```python
def update(frame):
    ax.cla()                          # Clear axes only
    ax.plot(x, data[frame], color="steelblue")
    ax.set_title(f"Frame {frame}")
    ax.set_ylim(-1, 1)
```

What is the key difference between `ax.cla()` and `fig.clf()`?

- [ ] A) `ax.cla()` clears only the axes' plotted data but retains tick labels; `fig.clf()` removes tick labels too
- [ ] B) `ax.cla()` clears only the specific `Axes` object (removing lines, patches, text) while leaving other axes in the figure untouched; `fig.clf()` clears ALL axes and figure-level content
- [ ] C) `ax.cla()` is faster than `fig.clf()` because it doesn't touch the canvas
- [ ] D) `fig.clf()` removes only figure-level text; `ax.cla()` removes only data lines

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.cla()` clears only the specific `Axes` object while leaving other axes in the figure untouched; `fig.clf()` clears ALL axes and figure-level content

**Explanation:**

- **A is incorrect:** `ax.cla()` clears the ENTIRE axes content — lines, patches, collections, text, tick labels, axis labels, title. It resets the axes to a blank state (though the axes OBJECT remains in the figure).
- **B is correct:** `ax.cla()` ("clear axes") resets one specific `Axes` object to blank — removing all its plotted content, title, labels, and tick customizations while leaving the axes object itself in place. Other `Axes` in the same figure are unaffected. `fig.clf()` ("clear figure") removes ALL axes and all figure-level artists (suptitle, figure-level text). In a single-axes animation, `ax.cla()` is the targeted choice — the axes object remains valid and doesn't need to be re-created with `add_subplot()`.
- **C is incorrect:** The canvas is touched by `fig.canvas.draw()` after either clear operation. The clearing operations themselves don't directly interact with the canvas renderer.
- **D is incorrect:** Both clear more than the described limited scope. `ax.cla()` clears the entire axes; `fig.clf()` clears all axes AND figure-level content.

</details>

---

### Question 85

A developer creates figures inside a function and is concerned about memory leaks:

```python
def generate_report(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data)
    fig.savefig("report.png")
    plt.close(fig)
    return "report.png"
```

After `plt.close(fig)`, is the figure object `fig` immediately garbage collected?

- [ ] A) Yes — `plt.close(fig)` immediately frees the memory
- [ ] B) `plt.close(fig)` deregisters the figure from Matplotlib's manager, removing the manager's reference. The figure is garbage collected when ALL references to it are released — including the local variable `fig` going out of scope when the function returns
- [ ] C) The figure is never garbage collected — it persists until the Python process ends
- [ ] D) `plt.close(fig)` only works in interactive backends — in script mode, `del fig` is required

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.close(fig)` deregisters the figure from Matplotlib's manager, removing the manager's reference. The figure is garbage collected when ALL references to it are released

**Explanation:**

- **A is incorrect:** `plt.close(fig)` alone does not immediately free memory — Python's garbage collector (reference counting + cyclic GC) handles the actual deallocation. `plt.close(fig)` removes the manager's reference, which is often the last strong external reference.
- **B is correct:** Python uses **reference counting** for memory management. `plt.close(fig)` removes the `Gcf` figure manager's reference to the figure. After the function returns, the local variable `fig` goes out of scope, dropping the last reference. At that point (with CPython's reference counting), the figure and all its artists are immediately deallocated. In this pattern: `plt.close(fig)` + function scope exit = complete memory release. This is the correct, production-safe pattern.
- **C is incorrect:** Python's garbage collector does reclaim figure memory once all references are dropped. Figures do not persist until process end unless references are kept alive (e.g., stored in a global list).
- **D is incorrect:** `plt.close(fig)` works in all backends — interactive (TkAgg, Qt5Agg) and non-interactive (Agg). In non-interactive backends, there is no GUI window to close, but the figure is still deregistered from the manager.

</details>

---

### Question 86

A developer writes a context manager for automatic figure cleanup:

```python
from contextlib import contextmanager

@contextmanager
def managed_figure(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)

# Usage:
with managed_figure(figsize=(8, 6)) as fig:
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    fig.savefig("output.png")
```

What is the benefit of this pattern?

- [ ] A) It automatically adjusts DPI when the context exits
- [ ] B) It guarantees `plt.close(fig)` is called even if an exception occurs during plotting — preventing memory leaks in error scenarios
- [ ] C) It locks the figure so other threads cannot modify it during the `with` block
- [ ] D) It automatically calls `tight_layout()` when the `with` block exits

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It guarantees `plt.close(fig)` is called even if an exception occurs during plotting — preventing memory leaks in error scenarios

**Explanation:**

- **A is incorrect:** DPI is not affected by context manager entry/exit. There is no automatic DPI adjustment in this pattern.
- **B is correct:** The `try/finally` block ensures that `plt.close(fig)` executes in the `finally` clause **regardless of whether an exception occurs** inside the `with` block. If `ax.plot()` or `fig.savefig()` raises an exception, normal code flow skips any code after the error — meaning a bare `plt.close(fig)` at the end of the `with` block would never execute. The `finally` clause always runs, ensuring no figure memory leak even in exception scenarios. This is the production-safe pattern for batch figure generation.
- **C is incorrect:** Matplotlib figures are not thread-safe and the context manager does not add locking. Thread safety requires external synchronization (`threading.Lock`), not context managers.
- **D is incorrect:** `tight_layout()` is not called automatically by this context manager. The `finally` block only calls `plt.close(fig)`.

</details>

---

### Question 87

A developer checks how many open figures exist using Matplotlib's API:

```python
import matplotlib.pyplot as plt

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

count = plt.get_fignums()
print(count)
print(len(count))
```

What does `plt.get_fignums()` return?

- [ ] A) An integer: the total count of open figures (returns `3`)
- [ ] B) A list of figure numbers (integer IDs) for all currently open figures: `[1, 2, 3]`
- [ ] C) A list of `Figure` objects: `[<Figure>, <Figure>, <Figure>]`
- [ ] D) A dictionary mapping figure numbers to Figure objects

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A list of figure numbers (integer IDs) for all currently open figures: `[1, 2, 3]`

**Explanation:**

- **A is incorrect:** `plt.get_fignums()` returns a LIST, not an integer. `len(plt.get_fignums())` gives the count.
- **B is correct:** `plt.get_fignums()` returns a **list of integer figure numbers** for all figures currently registered in Matplotlib's figure manager. Each figure is assigned an integer ID at creation (1, 2, 3, ...). The print output would be `[1, 2, 3]` and `len([1, 2, 3])` = `3`. To retrieve the actual `Figure` objects: `plt.figure(num)` or `[plt.figure(n) for n in plt.get_fignums()]`.
- **C is incorrect:** `plt.get_fignums()` returns integer IDs, not `Figure` objects. To get `Figure` objects from IDs, use `plt.figure(num)` or `matplotlib._pylab_helpers.Gcf.get_all_fig_managers()`.
- **D is incorrect:** The return type is a list, not a dictionary. Converting to a dict would require: `{n: plt.figure(n) for n in plt.get_fignums()}`.

</details>

---

### Question 88

A developer runs the following code and observes unexpected behavior:

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
plt.show()
# In a non-interactive backend (e.g., Agg in a script):
ax.set_title("Updated Title")
fig.savefig("after_show.png")
```

In a non-interactive backend (Agg, running as a script), what happens after `plt.show()`?

- [ ] A) The figure is destroyed and `ax.set_title()` raises an error
- [ ] B) `plt.show()` in non-interactive mode (Agg backend) renders and may block briefly, but the figure object remains alive; subsequent `ax.set_title()` and `fig.savefig()` work correctly
- [ ] C) The figure is cleared automatically by `plt.show()` in script mode
- [ ] D) `plt.show()` converts the figure to a static image — further modifications are ignored

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.show()` in non-interactive mode renders and may block briefly, but the figure object remains alive; subsequent operations work correctly

**Explanation:**

- **A is incorrect:** In non-interactive backends (like Agg used in scripts), `plt.show()` does not destroy figure objects. The figure and axes remain valid Python objects.
- **B is correct:** In a non-interactive backend (Agg), `plt.show()` may be a no-op or may flush the renderer — the figure object (`fig`) and axes (`ax`) remain fully alive Python objects. Subsequent `ax.set_title("Updated Title")` modifies the axes normally, and `fig.savefig("after_show.png")` saves the updated figure (with the new title) to disk. This behavior is backend-dependent: in interactive GUI backends (TkAgg, Qt5Agg), `plt.show()` may block until the window is closed.
- **C is incorrect:** `plt.show()` does NOT clear the figure. Clearing requires explicit `fig.clf()` or `ax.cla()`.
- **D is incorrect:** `plt.show()` does not convert the figure to a static snapshot. The `Figure` Python object remains mutable — all artist attributes can still be modified and re-saved.

</details>

---

### Question 89

A developer runs a batch script overnight generating 5,000 heatmap figures. Each figure uses `imshow()` with a colorbar:

```python
for i, matrix in enumerate(matrices):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis")
    plt.colorbar(im, ax=ax)
    fig.savefig(f"heatmap_{i:04d}.png", dpi=150)
    plt.close(fig)
```

A colleague adds `plt.close(fig)` at the end. Is this sufficient to prevent a memory leak?

- [ ] A) No — `plt.colorbar()` creates a separate figure that must be closed with `plt.close("all")`
- [ ] B) Yes — `plt.close(fig)` closes the figure and its associated colorbar axes (since the colorbar is added to the same figure via `ax=ax`); all memory is properly released
- [ ] C) No — `im` (the `AxesImage` from `imshow`) holds a reference to the matrix data, requiring `del im` before `plt.close(fig)`
- [ ] D) No — `colorbar` accumulates in a global list that requires manual clearing

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Yes — `plt.close(fig)` closes the figure including its colorbar axes

**Explanation:**

- **A is incorrect:** `plt.colorbar(im, ax=ax)` adds a colorbar axes to the SAME figure (`fig`) — it does not create a separate independent figure. `plt.close(fig)` closes the entire figure including the colorbar's axes object.
- **B is correct:** `plt.colorbar(im, ax=ax)` creates an additional `Axes` within the same `fig` object to display the colorbar. All axes within a figure (including colorbar axes) are owned by that figure. `plt.close(fig)` deregisters the entire figure from the manager — including all its axes (both the imshow axes and the colorbar axes). After `plt.close(fig)` and the loop variable `fig` is rebound in the next iteration, the reference count drops to zero and Python's GC reclaims the memory.
- **C is incorrect:** `del im` is not required. The `AxesImage` object `im` is referenced by `ax.images` (the axes' image list). When the axes is cleared via `plt.close(fig)`, the axes' reference to `im` is dropped, allowing GC. The matrix data (`matrix`) is a separate NumPy array and its lifecycle is managed by the `matrices` list, independent of the figure.
- **D is incorrect:** Matplotlib's colorbar does not accumulate in any persistent global list that would require separate clearing. The colorbar's memory is tied to its parent figure.

</details>

---

### Question 90

A developer observes that after running a long batch job, `plt.get_fignums()` still returns a large list even though `plt.close(fig)` was called in the loop:

```python
fig_list = []
for i in range(100):
    fig, ax = plt.subplots()
    ax.plot(data[i])
    fig_list.append(fig)       # Bug line!
    plt.close(fig)

print(len(plt.get_fignums()))  # Prints 0 (correct)
print(len(fig_list))           # Prints 100
```

Is there a memory leak? Why or why not?

- [ ] A) No memory leak — `plt.close(fig)` removed all figures from the manager; `fig_list` just holds lightweight references
- [ ] B) Yes — `fig_list.append(fig)` holds strong Python references to all 100 `Figure` objects; even though `plt.close(fig)` deregisters them from Matplotlib's manager, the figures stay in memory because `fig_list` prevents garbage collection
- [ ] C) No — Python's garbage collector automatically recycles figures appended to a list
- [ ] D) Yes — `plt.get_fignums()` returns 0 incorrectly; figures stay registered even after `plt.close()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Yes — `fig_list.append(fig)` holds strong Python references; figures stay in memory despite `plt.close(fig)`

**Explanation:**

- **A is incorrect:** `fig_list` holds FULL `Figure` objects — each with all their axes, artists, and raster buffers. These are not "lightweight references" — they hold the complete figure state in memory.
- **B is correct:** `plt.close(fig)` removes the figure from Matplotlib's `Gcf` manager (so `plt.get_fignums()` correctly shows 0). BUT the `fig` variables are also stored in `fig_list` — a Python list that holds strong references. Python's reference-counting GC cannot collect an object as long as ANY reference to it exists. With 100 entries in `fig_list`, all 100 `Figure` objects remain in RAM with their full artist trees and raster buffers. At `figsize=(6.4, 4.8)` and `dpi=100`: each figure ≈ 640×480×4 bytes ≈ 1.2 MB × 100 = **~120 MB**. The fix: either don't append to `fig_list`, or `del fig_list` when done, or use `fig_list.append(weakref.ref(fig))`.
- **C is incorrect:** Python's GC cannot collect objects with active strong references. `fig_list` holds strong references that prevent collection.
- **D is incorrect:** `plt.get_fignums()` correctly returns 0 after `plt.close(fig)` for all 100 figures. The manager no longer tracks them. But the memory leak exists in `fig_list`, which is independent of the manager.

</details>
