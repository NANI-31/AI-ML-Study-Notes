# Matplotlib Coding Interview Practice — Level 4: Coding-Round Challenge

This set contains 20 advanced, interview-grade coding-round challenges designed for senior ML/AI engineers and quantitative data scientists. These problems feature multi-step requirements, messy or corrupt data, debugging broken visualization code, architectural layouts using `GridSpec`, publication-grade aesthetics, and deep causal reasoning.

---

### Problem 61: Debugging & Refactoring: The "Ghost Overfitting" Scale Bug

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  # An engineer submitted the following training log for PR review
  epochs = np.arange(1, 26)
  # Binary cross entropy loss (per sample average)
  train_bce = 0.8 * np.exp(-0.18 * epochs) + 0.05
  # Auxiliary regularization penalty (L2 weight decay sum across millions of parameters)
  val_reg_loss = 120.0 * np.exp(-0.02 * epochs) + 15.0

  # BROKEN CODE SUBMITTED IN PR:
  # plt.plot(epochs, train_bce, label="Train Loss")
  # plt.plot(epochs, val_reg_loss, label="Validation Loss")
  # plt.legend()
  # plt.show()
  ```
- **Task:** The engineer claims that "Train loss is a flat line along zero, and validation loss is astronomical, indicating the model failed completely." Identify the fatal visualization flaw in their code, refactor it into an accurate, dual-axis diagnostic chart using Matplotlib's Object-Oriented API (`fig, ax = plt.subplots()`), and determine whether the model is actually training successfully.
- **Expected Output:** A properly scaled figure where primary loss (left axis, linear scale $[0.0, 1.0]$) and regularization loss (right axis, logarithmic or appropriately scaled $[10, 150]$) are independently visible without flattening either metric. Synchronized unified legend box, color-coordinated axis spines and ticks, and an annotation clarifying the magnitude discrepancy.
- **Constraints:**
  - Fix the visualization without modifying the raw numeric values in the dataset.
  - Apply `ax.twinx()` and ensure both handles and labels appear inside a single consolidated legend box.
- **What You Need to Determine:**
  1. Why did the naive `plt.plot()` completely flatten the `train_bce` curve into an apparent zero line?
  2. Is the auxiliary regularization loss actually exploding or is it steadily decaying over time?
  3. What engineering recommendation should you provide to the ML team regarding how the two loss components are logged or weighted?

---

### Problem 62: Messy Telemetry Cleaning & Latency Jitter Percentile Bands

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  # Raw distributed model server telemetry log with real-world artifacts
  n_samples = 600
  timestamps = pd.date_range("2026-03-31 10:00:00", periods=n_samples, freq="10s")

  raw_latency = np.random.normal(45, 8, n_samples)
  # Introduce artifacts:
  raw_latency[::45] = -999.0          # Missing/failed probe sentinel values
  raw_latency[120:135] = np.nan       # Network outage dropping packets
  raw_latency[250:265] += 180.0       # Garbage collection cold-start spike
  raw_latency[400:410] = raw_latency[400:410] * 3.5

  df_telemetry = pd.DataFrame({"timestamp": timestamps, "latency_ms": raw_latency})
  # Introduce duplicated timestamps (common distributed logging race condition)
  df_telemetry = pd.concat([df_telemetry, df_telemetry.iloc[50:60]]).sort_values("timestamp").reset_index(drop=True)
  ```
- **Task:** Clean this messy production telemetry log and visualize service latency over time. Your pipeline must: (1) deduplicate timestamps, (2) filter out negative sentinel values and NaNs, (3) resample data into 1-minute rolling windows computing the median (p50), 95th percentile (p95), and 99th percentile (p99), and (4) plot the p50 line with a shaded envelope between p50 and p95, and a distinct line for p99.
- **Expected Output:** A time-series chart with dates formatted cleanly along the x-axis using `mdates.DateFormatter("%H:%M")`. A solid blue line for median latency, a semi-transparent cyan band covering $[p50, p95]$, and a dashed red line tracking extreme p99 tail latency. A horizontal SLA limit line at $100\text{ ms}$.
- **Constraints:**
  - Handle data cleaning purely in code prior to plotting.
  - Automatically format overlapping x-axis dates using `fig.autofmt_xdate()`.
- **What You Need to Determine:**
  1. How many total invalid or corrupt rows were dropped during the cleaning phase?
  2. Between what exact time intervals did the server breach the $100\text{ ms}$ SLA at the 99th percentile?
  3. Why is the rolling median (p50) virtually immune to the short-lived garbage collection spike while p99 spikes dramatically?

---

### Problem 63: LLM Pareto Frontier: Latency vs. Throughput vs. Perplexity

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import pandas as pd

  # Benchmark data for 8 candidate generative models
  models = [
      {"name": "Titan-70B",   "latency_p90": 240, "throughput_tps": 45,  "perplexity": 4.1, "params_b": 70, "open_weights": False},
      {"name": "Nova-32B",    "latency_p90": 130, "throughput_tps": 95,  "perplexity": 5.4, "params_b": 32, "open_weights": True},
      {"name": "Pulse-14B",   "latency_p90": 85,  "throughput_tps": 160, "perplexity": 6.8, "params_b": 14, "open_weights": True},
      {"name": "Swift-7B",    "latency_p90": 42,  "throughput_tps": 290, "perplexity": 8.9, "params_b": 7,  "open_weights": True},
      {"name": "Micro-3B",    "latency_p90": 22,  "throughput_tps": 520, "perplexity": 12.4,"params_b": 3,  "open_weights": True},
      {"name": "Legacy-50B",  "latency_p90": 290, "throughput_tps": 30,  "perplexity": 6.2, "params_b": 50, "open_weights": False},
      {"name": "Bloat-14B",   "latency_p90": 110, "throughput_tps": 110, "perplexity": 7.9, "params_b": 14, "open_weights": False},
      {"name": "Nano-1.5B",   "latency_p90": 15,  "throughput_tps": 680, "perplexity": 16.5,"params_b": 1.5,"open_weights": True}
  ]
  df_llm = pd.DataFrame(models)
  ```
- **Task:** Create an executive model selection chart evaluating Latency (x-axis, lower is better) vs. Perplexity (y-axis, lower is better). Set bubble size proportional to parameter count (`params_b`), color by whether the model is open weights or proprietary, and calculate and draw the non-dominated Pareto Frontier curve connecting the optimal trade-off models.
- **Expected Output:** A 2D bubble scatter plot with model names annotated next to each point. The Pareto frontier is rendered as a stepped or convex line connecting the non-dominated set. Sub-optimal models dominated by the frontier (e.g., `Legacy-50B`, `Bloat-14B`) are visually distinct.
- **Constraints:**
  - A model $A$ dominates model $B$ if $\text{latency}_A \le \text{latency}_B$ and $\text{perplexity}_A \le \text{perplexity}_B$ with at least one strict inequality. Compute the Pareto non-dominated set algorithmically.
  - Scale bubble area proportionally to parameters using $s = \text{params\_b} \times 20 + 50$.
- **What You Need to Determine:**
  1. Which specific models belong to the optimal Pareto frontier?
  2. Which model is strictly dominated by `Nova-32B` across both latency and perplexity?
  3. If your production application has a hard latency budget of $100\text{ ms}$, which model on the Pareto frontier delivers the highest output quality (lowest perplexity)?

---

### Problem 64: Real-Time Anomaly Detection with Rolling Volatility Bands

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(123)
  n = 400
  time_steps = np.arange(n)
  signal = 50.0 + 0.05 * time_steps + np.sin(time_steps / 15.0) * 8.0 + np.random.normal(0, 1.8, n)

  # Inject flash anomalies
  signal[85] += 22.0    # Transient spike
  signal[190:196] -= 18.0 # Sustained dip
  signal[310] += 35.0   # Extreme burst

  df_signal = pd.DataFrame({"step": time_steps, "value": signal})
  ```
- **Task:** Implement a dynamic Bollinger-style anomaly detector using a rolling window of 25 steps. Compute the rolling mean $\mu_t$ and rolling standard deviation $\sigma_t$. Define the dynamic decision envelope as $[\mu_t - 2.5\sigma_t, \mu_t + 2.5\sigma_t]$. Plot the raw signal, the rolling baseline, the shaded confidence corridor, and highlight every detected outlier with an animated red halo marker.
- **Expected Output:** A line chart showing the raw signal in dark charcoal and rolling mean in navy. A shaded light-gray band represents the dynamic $2.5\sigma$ acceptance corridor. Data points outside the band are rendered as enlarged red scatter markers (`edgecolor='red', facecolor='none', s=80, lw=2`).
- **Constraints:**
  - Fill or handle rolling window burn-in NaNs cleanly (`min_periods=1`).
  - Use boolean indexing to filter and plot only the outlier points.
- **What You Need to Determine:**
  1. Exactly how many individual time steps violated the $2.5\sigma$ threshold envelope?
  2. Did the sustained dip (steps 190–195) cause the rolling envelope itself to expand, and how did that affect subsequent anomaly classification?
  3. What is the fundamental disadvantage of using a symmetric Gaussian $k\sigma$ threshold on skewed or heavy-tailed operational metrics?

---

### Problem 65: Uncovering Simpson's Paradox in Customer Churn vs. Engagement

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  tiers = ["Starter", "Professional", "Enterprise"]

  data = []
  # Starter: Low usage, High churn probability
  u_start = np.random.normal(15, 4, 100)
  c_start = 0.85 - 0.015 * u_start + np.random.normal(0, 0.04, 100)

  # Pro: Mid usage, Mid churn probability
  u_pro = np.random.normal(45, 6, 100)
  c_pro = 0.60 - 0.008 * u_pro + np.random.normal(0, 0.03, 100)

  # Enterprise: High usage, Low churn probability
  u_ent = np.random.normal(85, 8, 100)
  c_ent = 0.35 - 0.003 * u_ent + np.random.normal(0, 0.02, 100)

  df_paradox = pd.DataFrame({
      "usage_hours": np.concatenate([u_start, u_pro, u_ent]),
      "churn_prob": np.concatenate([c_start, c_pro, c_ent]),
      "tier": ["Starter"] * 100 + ["Professional"] * 100 + ["Enterprise"] * 100
  })
  ```
- **Task:** An executive looks at an aggregated regression line and claims "Higher usage increases churn!" when in reality Simpson's Paradox is occurring due to customer tier confounding. Build a $1 \times 2$ comparative figure: Left panel showing the aggregate scatter plot with a single naive global regression line; Right panel showing the data split by `tier` with distinct colors and three separate subgroup regression lines.
- **Expected Output:** Two side-by-side plots with identical axis limits ($x \in [0, 110]$, $y \in [0.0, 1.0]$). Left plot shows uniform gray points with a positive global trend line. Right plot shows three distinct clusters (Starter in blue, Pro in green, Enterprise in purple) with three negative downward-sloping regression lines.
- **Constraints:**
  - Compute linear regression slopes and intercepts using `np.polyfit(x, y, 1)`.
  - Display the numeric slope value ($m$) on each respective trendline.
- **What You Need to Determine:**
  1. What is the mathematical sign of the slope in the aggregated plot vs. the subgroup plots?
  2. Why does the naive global model invert the true underlying relationship between usage and churn?
  3. If an automated ML feature engineering pipeline computes a single global Pearson correlation coefficient, what misleading signal will it feed to the model?

---

### Problem 66: Spaghetti Chart Deconstruction into Small Multiples (Trellis Plot)

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(88)
  days = np.arange(1, 61)
  services = [f"Microservice_{chr(65+i)}" for i in range(12)]  # 12 microservices

  matrix = []
  for s in services:
      base = np.random.uniform(20, 80)
      trend = np.random.uniform(-0.4, 0.6)
      drift = base + trend * days + np.cumsum(np.random.normal(0, 1.2, len(days)))
      matrix.append(drift)

  df_services = pd.DataFrame(np.array(matrix).T, index=days, columns=services)
  ```
- **Task:** When plotting 12 distinct time-series on a single plot canvas, the result is an unreadable "spaghetti chart." Deconstruct this data into a $3 \times 4$ faceted "Small Multiples" grid. In each subplot, plot all 11 other microservices as subtle background reference lines (`color='lightgray', alpha=0.4, lw=0.8`), and highlight the current microservice with a bold colored line (`color='darkcyan', lw=2.0`). Overlay a dashed red horizontal threshold line at $y = 80$.
- **Expected Output:** A structured $3 \times 4$ grid (`plt.subplots(3, 4, sharex=True, sharey=True, figsize=(14, 10))`). Each individual subplot clearly identifies its respective service name in the corner. Shared y-axes enable instant visual comparison across all 12 services.
- **Constraints:**
  - All subplots must strictly share the exact same y-axis scale (`sharey=True`) so visual comparisons are statistically honest.
  - Remove redundant inner tick labels across internal subplot rows and columns.
- **What You Need to Determine:**
  1. Which microservice is the only one to exhibit a sustained upward trajectory breaching the 80 threshold?
  2. Which microservice suffered the sharpest sudden degradation in the final 10 days?
  3. Why is the small multiples technique vastly superior to a 12-line single legend chart for executive dashboards?

---

### Problem 67: Dimensionality Reduction Perplexity Diagnostics: t-SNE Parameter Sensitivity

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  # Simulated 2D projections of 3 concentric high-dimensional manifold rings
  # under three different perplexity settings: Perplexity=2, 30, and 100
  np.random.seed(42)
  n_pts = 150

  def generate_tsne_sim(perplexity_mode):
      if perplexity_mode == "low_perp_2":  # Under-smoothed, artificial local micro-clusters
          x = np.random.normal(0, 5, n_pts)
          y = np.random.normal(0, 5, n_pts)
      elif perplexity_mode == "optimal_30": # Clean separation of the 3 manifold groups
          angles = np.linspace(0, 2*np.pi, n_pts // 3)
          r1, r2, r3 = 5, 15, 25
          x = np.concatenate([r1*np.cos(angles), r2*np.cos(angles), r3*np.cos(angles)]) + np.random.normal(0, 0.8, n_pts)
          y = np.concatenate([r1*np.sin(angles), r2*np.sin(angles), r3*np.sin(angles)]) + np.random.normal(0, 0.8, n_pts)
      else:  # Over-smoothed, squashed global cloud
          x = np.random.uniform(-10, 10, n_pts)
          y = 0.1 * x + np.random.normal(0, 2, n_pts)
      return x, y

  labels = ["Cluster_A"] * 50 + ["Cluster_B"] * 50 + ["Cluster_C"] * 50
  ```
- **Task:** A common trap in unsupervised ML interviews is misinterpreting t-SNE artifacts. Build a $1 \times 3$ subplot row illustrating the visual manifold produced by: (1) `Perplexity = 2` (Micro-cluster hallucination), (2) `Perplexity = 30` (Optimal manifold separation), and (3) `Perplexity = 100` (Global squashing). Color points consistently by ground truth cluster labels.
- **Expected Output:** Three side-by-side scatter plots with subtitles noting the perplexity and diagnostic takeaway. Equal aspect ratio applied to all three panels. Single unified legend placed at the top or bottom of the entire figure canvas.
- **Constraints:**
  - Use `fig.legend()` to render one consolidated legend for all subplots rather than repeating legends.
  - Hide axis tick marks and numeric labels (`ax.set_xticks([])`, `ax.set_yticks([])`) to reflect the scale-free nature of t-SNE.
- **What You Need to Determine:**
  1. Why does a very low perplexity ($Perp = 2$) create false clusters that do not actually exist in the underlying data?
  2. Why are distances between separated clusters in t-SNE plots generally meaningless without knowing perplexity and initialization?
  3. In which perplexity regime are the true ground-truth cluster labels actually cleanly linearly separable?

---

### Problem 68: Publication-Grade Inset Zoom (`inset_axes`): Investigating Loss Anomalies

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(17)
  epochs = np.arange(1, 101)
  # Standard smooth convergence curve
  base_loss = 2.0 * np.exp(-0.05 * epochs) + 0.15 + np.random.normal(0, 0.008, 100)

  # Transient gradient explosion anomaly between epochs 42 and 48 caused by bad batch
  anomaly = np.zeros(100)
  anomaly[41:48] = [0.12, 0.45, 1.35, 0.95, 0.38, 0.15, 0.05]
  val_loss = base_loss + anomaly
  ```
- **Task:** During a 100-epoch training run, an unhandled numerical overflow in epoch 44 produced a sharp spike that recovered quickly. Create a publication-grade full-trajectory plot ($epochs \in [1, 100]$), and use `mpl_toolkits.axes_grid1.inset_locator.inset_axes` to construct a zoomed inset window focusing specifically on the anomaly region ($epochs \in [40, 50]$). Connect the zoomed inset to the parent axis with visual connecting lines (`mark_inset`).
- **Expected Output:** Main plot showing the complete 100-epoch training trajectory. An inset subplot positioned in the upper right quadrant ($width=35\%, height=35\%$ of parent) showing the detailed individual batch loss oscillations between epochs 40 and 50 with scatter markers and grid lines. Two connecting lines linking the inset box to the highlighted rectangle on the main plot.
- **Constraints:**
  - Use `inset_axes(ax, width="35%", height="35%", loc="upper right")` and `mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")`.
  - Mark the exact peak anomaly coordinate with a prominent annotation text box inside the inset.
- **What You Need to Determine:**
  1. At what exact epoch did the gradient anomaly reach its maximum peak value?
  2. How many epochs did it take for the optimizer to recover to baseline pre-spike loss levels?
  3. Why is an inset zoom superior to simply truncating or clipping the y-axis?

---

### Problem 69: Financial Machine Learning: Eliminating Weekend Gaps in Trading Data

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  # 3 weeks of business day stock trading closing prices (excludes Saturdays and Sundays)
  business_dates = pd.bdate_range("2026-03-02", periods=15)
  np.random.seed(9)
  prices = 180.0 + np.cumsum(np.random.normal(0.5, 2.5, len(business_dates)))
  df_stock = pd.DataFrame({"date": business_dates, "close": prices})
  ```
- **Task:** When plotting financial time-series with `datetime64` objects directly on a continuous temporal x-axis, Matplotlib inserts artificial horizontal gaps for non-trading weekends (Saturdays and Sundays), distorting technical moving averages. Write an index-based visualization that plots the price series without weekend flatline gaps, while formatting the x-axis tick labels to show the actual calendar date strings (`"Mon Mar 02"`, etc.) formatted every 2nd business day.
- **Expected Output:** A continuous, unbroken line chart without weekend plateau artifacts. The x-axis uses integer indices $[0, 1, \dots, 14]$ for point placement, but the visible tick labels display cleanly formatted date strings aligned at a $45^\circ$ angle.
- **Constraints:**
  - Do not let Matplotlib treat the x-axis as a continuous Gregorian calendar timeline. Use integer positions and map tick labels using `ax.set_xticks(indices)` and `ax.set_xticklabels(formatted_date_strings)`.
  - Calculate and overlay a 5-day Simple Moving Average (SMA).
- **What You Need to Determine:**
  1. What optical illusion occurs when financial time series containing non-trading holidays are plotted on a continuous datetime scale?
  2. How does using index-based x-coordinates resolve the weekend gap problem?
  3. Between which two business days did the largest single-session price change occur?

---

### Problem 70: Feature Correlation Drift: Tracking Macro Concept Shifts Across Quarters

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  features = ["Income", "Debt", "Credit_Lines", "Inquiries", "Savings"]

  # Correlation matrix during stable macroeconomic regime (Q1)
  corr_q1 = np.array([
      [ 1.00, -0.42,  0.35, -0.15,  0.62],
      [-0.42,  1.00,  0.55,  0.48, -0.68],
      [ 0.35,  0.55,  1.00,  0.32, -0.22],
      [-0.15,  0.48,  0.32,  1.00, -0.38],
      [ 0.62, -0.68, -0.22, -0.38,  1.00]
  ])

  # Correlation matrix during liquidity crunch regime (Q4)
  corr_q4 = np.array([
      [ 1.00, -0.10,  0.12,  0.25,  0.41],
      [-0.10,  1.00,  0.78,  0.82, -0.85],
      [ 0.12,  0.78,  1.00,  0.65, -0.45],
      [ 0.25,  0.82,  0.65,  1.00, -0.62],
      [ 0.41, -0.85, -0.45, -0.62,  1.00]
  ])
  ```
- **Task:** High-stakes credit risk models degrade when the correlation structure between input features drifts over time. Construct a $1 \times 3$ heatmap figure: Left panel showing Q1 Correlation; Middle panel showing Q4 Correlation; Right panel showing the Difference Matrix ($\Delta = \text{Corr}_{Q4} - \text{Corr}_{Q1}$). The Difference matrix must use a symmetric diverging colormap (`coolwarm` or `seismic`) centered strictly at $0.0$ with its own dedicated colorbar.
- **Expected Output:** Three synchronized heatmaps with feature names labeled on the ticks. Panels 1 and 2 share a standard correlation color scale $[-1.0, 1.0]$. Panel 3 uses a diverging color scale $[-0.5, 0.5]$ with zero rendered as neutral white. Numeric difference values annotated in each cell.
- **Constraints:**
  - Enforce `vmin=-0.5, vmax=0.5` on the delta heatmap using `matplotlib.colors.Normalize` or `TwoSlopeNorm(vcenter=0.0)`.
  - Ensure colorbars have explicit labels (`"Pearson r"` and `"Delta r"`).
- **What You Need to Determine:**
  1. Which specific pair of features experienced the largest positive correlation drift between Q1 and Q4?
  2. Did the relationship between `Debt` and `Inquiries` strengthen or weaken during the liquidity crunch?
  3. Why does multivariable correlation drift degrade tree-based and linear models even if single-feature marginal distributions appear stable?

---

### Problem 71: Debugging Memory Leaks in Matplotlib Automated Batch Generation

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np

  # A pipeline generates 200 model inference diagnostic plots per day
  # A junior engineer wrote this loop which causes an Out-Of-Memory (OOM) crash in production:
  
  # BROKEN PIPELINE SNIPPET:
  # for i in range(200):
  #     fig, ax = plt.subplots(figsize=(6, 4))
  #     data = np.random.normal(0, 1, 1000)
  #     ax.hist(data, bins=30)
  #     ax.set_title(f"Batch {i}")
  #     # Engineer forgot plt.close() or used global state
  #     # fig.savefig(f"batch_{i}.png")
  ```
- **Task:** Explain why the stateful `pyplot` interface retains every created figure in memory inside `_pylab_helpers.Gcf`, leading to continuous memory leaks in server environments. Write a robust, memory-safe generation function that benchmarks and verifies memory release, processes 20 sample iterations using `io.BytesIO` (in-memory buffer) without writing to disk, and explicitly closes figure handles.
- **Expected Output:** A working, leak-free loop using explicit `plt.close(fig)` or the `Figure` constructor without `pyplot` state pollution. A line chart plotting loop iteration index vs. memory usage or execution time per frame proving flat, constant memory consumption.
- **Constraints:**
  - Must explicitly call `plt.close(fig)` or use `from matplotlib.figure import Figure` with `from matplotlib.backends.backend_agg import FigureCanvasAgg`.
  - Simulate in-memory image generation using `import io; buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)`.
- **What You Need to Determine:**
  1. Why does calling `plt.clf()` or `plt.cla()` fail to free the figure from memory compared to `plt.close(fig)`?
  2. What is the fundamental difference between the stateful `pyplot` procedural API and the stateless pure object-oriented Matplotlib backend API?
  3. In a multi-threaded web service (e.g. FastAPI worker serving dynamic plots), why is `matplotlib.pyplot` unsafe to invoke concurrently?

---

### Problem 72: Silhouette Analysis for Determining Optimal Clustering Count ($k$)

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np

  # Pre-computed silhouette sample coefficients for k=2, 3, and 4 clusters
  # Sorted within each cluster group
  np.random.seed(42)
  
  # For k=3 (the true natural grouping):
  cluster_labels_k3 = np.array([0]*60 + [1]*50 + [2]*70)
  # High positive silhouette scores across all clusters
  sil_k3 = np.concatenate([
      np.sort(np.random.uniform(0.55, 0.85, 60)),
      np.sort(np.random.uniform(0.48, 0.78, 50)),
      np.sort(np.random.uniform(0.60, 0.90, 70))
  ])
  mean_sil_k3 = np.mean(sil_k3)

  # For k=2 (under-clustered):
  cluster_labels_k2 = np.array([0]*110 + [1]*70)
  sil_k2 = np.concatenate([
      np.sort(np.random.uniform(0.20, 0.70, 110)),
      np.sort(np.random.uniform(0.40, 0.75, 70))
  ])
  mean_sil_k2 = np.mean(sil_k2)
  ```
- **Task:** Build a classic Scikit-Learn style Silhouette Analysis visualization for $k=3$ vs. $k=2$. For each cluster, plot horizontal knife-blade polygons (`ax.fill_betweenx`) representing individual sample silhouette coefficients. Overlay a vertical red dashed line indicating the average silhouette score across the entire dataset.
- **Expected Output:** A $1 \times 2$ figure. The y-axis represents samples grouped by cluster with vertical gaps between clusters. The x-axis represents silhouette coefficient values ranging from $-0.1$ to $1.0$. Cluster silhouettes are sorted in ascending order within each group.
- **Constraints:**
  - Implement using `ax.fill_betweenx()` with y coordinates incrementing per sample.
  - Draw the average silhouette score as a vertical dashed line with an annotation.
- **What You Need to Determine:**
  1. How does the presence of negative silhouette scores in a cluster indicate misclassified or overlapping samples?
  2. Why does $k=3$ represent a statistically superior clustering solution compared to $k=2$ based on silhouette thickness and mean score?
  3. What cluster geometry causes the silhouette metric to fail (e.g. concentric circles or arbitrary densities)?

---

### Problem 73: Multi-Horizon Probabilistic Forecasting Fan Chart

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  history_steps = np.arange(1, 31)
  history_values = 100.0 + np.cumsum(np.random.normal(0.8, 2.0, 30))

  forecast_steps = np.arange(30, 51)
  median_forecast = history_values[-1] + 1.2 * (forecast_steps - 30)

  # Uncertainty expands with forecast horizon (heteroscedastic expansion)
  std_dev = 1.8 * np.sqrt(forecast_steps - 30)
  q95_upper = median_forecast + 1.96 * std_dev
  q95_lower = median_forecast - 1.96 * std_dev
  q80_upper = median_forecast + 1.28 * std_dev
  q80_lower = median_forecast - 1.28 * std_dev
  q50_upper = median_forecast + 0.67 * std_dev
  q50_lower = median_forecast - 0.67 * std_dev
  ```
- **Task:** In quantitative finance and weather forecasting, point forecasts are insufficient without uncertainty intervals. Construct a "Fan Chart" combining historical observed values with future probabilistic confidence bands. Shade three nested uncertainty corridors ($50\%$, $80\%$, and $95\%$) with progressive opacity fills (`alpha=0.4`, `0.25`, `0.15`), and draw a vertical boundary line separating historical facts from future predictions.
- **Expected Output:** A continuous timeline plot ($steps \in [1, 50]$). Historical data plotted as a solid black curve with dots. Future median forecast plotted as a dashed dark blue line. A multi-shaded blue fan expands into the future. A vertical dashed line at $step = 30$ with text `"Forecast Origin"`.
- **Constraints:**
  - Ensure the forecast bands connect seamlessly to the last historical point at $step = 30$ without visual gaps.
  - Create a structured legend clearly documenting each probability interval ($50\%$, $80\%$, $95\%$).
- **What You Need to Determine:**
  1. Why does the uncertainty envelope grow proportional to $\sqrt{t}$ in a random walk forecasting process?
  2. At step 50, what is the total width of the $95\%$ prediction interval in numeric units?
  3. If an executive demands a single deterministic forecast number, how does this chart visually refute that requirement?

---

### Problem 74: Multi-Modal Audio & Speech Timeline Alignment: Waveform, Spectrogram & Tokens

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np

  # Simulated 2-second audio clip at 16kHz
  sr = 16000
  duration = 2.0
  t = np.linspace(0, duration, int(sr * duration))
  # Audio waveform with phoneme bursts
  waveform = np.sin(2 * np.pi * 220 * t) * np.exp(-t) + 0.3 * np.sin(2 * np.pi * 880 * t) * (t > 0.8) * (t < 1.4)
  waveform += np.random.normal(0, 0.05, len(t))

  # Synthetic mel-spectrogram (64 mel bins x 80 time frames)
  spectrogram = np.random.uniform(-80, -20, (64, 80))
  spectrogram[15:25, 30:55] += 35.0  # Simulated formant energy burst

  # Token timestamp alignments
  tokens = [
      {"text": "Hello", "start": 0.15, "end": 0.65},
      {"text": "World", "start": 0.85, "end": 1.45},
      {"text": "[EOS]", "start": 1.50, "end": 1.70}
  ]
  ```
- **Task:** Build a synchronized $3 \times 1$ multi-modal speech analysis dashboard using `GridSpec`. Subplot 1: Audio raw waveform amplitude over time; Subplot 2: Mel-Spectrogram heatmap (`imshow`) with a decibel (dB) colorbar; Subplot 3: Alignment timeline showing transcribed word tokens as shaded horizontal span blocks (`axvspan` or `barh`) with centered text labels.
- **Expected Output:** Three vertically stacked subplots sharing the exact same time x-axis ($[0.0, 2.0]$ seconds). The top panel shows high-resolution oscillations. The middle panel shows frequency energy concentrations. The bottom panel shows discrete colored blocks containing text strings aligned with the phonetic bursts.
- **Constraints:**
  - Synchronize all x-axes so zooming or panning locks the temporal alignment across all three modalities.
  - Spectrogram extent must be mapped precisely to the time domain: `extent=[0, duration, 0, 8000]`.
- **What You Need to Determine:**
  1. Does the formant energy burst in the spectrogram align temporally with the token `"World"`?
  2. What is the duration of the silent gap between the words `"Hello"` and `"World"`?
  3. Why is shared-axis temporal alignment crucial when debugging ASR (Automatic Speech Recognition) forced-alignment models?

---

### Problem 75: Regression Influence Diagnostics: Cook's Distance and Leverage Outliers

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(33)
  n = 80
  # Independent variable
  x = np.random.uniform(10, 50, n)
  y = 2.5 * x + 15.0 + np.random.normal(0, 8.0, n)

  # Inject high-leverage influential outliers:
  x[0] = 75.0; y[0] = 40.0   # Extreme leverage + massive negative residual (severe influential point)
  x[1] = 78.0; y[1] = 210.0  # Extreme leverage, but aligns with slope (good leverage)
  x[2] = 25.0; y[2] = 180.0  # Low leverage, but huge residual (vertical outlier)

  # Pre-computed leverage (hat values h_ii) and Cook's distances
  hat_values = (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2) + (1.0 / n)
  residuals = y - (np.poly1d(np.polyfit(x, y, 1))(x))
  mse = np.sum(residuals**2) / (n - 2)
  cooks_d = (residuals**2 / (2 * mse)) * (hat_values / (1 - hat_values)**2)
  ```
- **Task:** Create a two-panel regression diagnostic figure: Left panel showing the scatter plot of $x$ vs. $y$ with the fitted regression line and bubble sizes scaled proportionally to Cook's Distance; Right panel showing the classic "Residuals vs. Leverage" diagnostic plot with Cook's distance threshold boundary contours ($D = 0.5$ and $D = 1.0$).
- **Expected Output:** Subplot 1: Data points with fitted OLS line. Highly influential points have visibly enlarged bubble markers. Subplot 2: Leverage ($h_{ii}$) on x-axis and standardized residuals on y-axis. Dashed hyperbolic contours indicating Cook's distance critical thresholds. Influential outlier points annotated with their row indices.
- **Constraints:**
  - Cook's distance critical threshold line ($4/n$) drawn as a reference threshold.
  - Scale marker sizes cleanly: `s = 30 + 600 * (cooks_d / np.max(cooks_d))`.
- **What You Need to Determine:**
  1. Which specific point (index 0, 1, or 2) possesses the single highest Cook's distance and exerts the most severe distortion on the regression slope?
  2. What distinguishes point 1 (high leverage, low residual) from point 0 (high leverage, high residual) in terms of model parameter impact?
  3. If point 0 were removed, would the estimated slope parameter increase or decrease?

---

### Problem 76: Cost-Sensitive Learning & Profit Curve vs. Classification Threshold

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  thresholds = np.linspace(0.01, 0.99, 100)
  n_total = 10000
  n_pos = 500  # Fraudulent transactions
  n_neg = 9500 # Legitimate transactions

  # True Positive Rate and False Positive Rate across thresholds
  tpr = 1.0 / (1.0 + np.exp(8 * (thresholds - 0.4)))
  fpr = 1.0 / (1.0 + np.exp(10 * (thresholds - 0.15)))

  # Business Cost Matrix (in USD):
  # True Positive: Caught fraud -> Save $500 fraud loss - $5 review fee = +$495 profit
  # False Positive: False alarm -> Friction / manual review cost = -$25
  # False Negative: Missed fraud -> Undetected fraud chargeback = -$500
  # True Negative: Correctly cleared transaction = $0
  profit = (tpr * n_pos * 495) + (fpr * n_neg * (-25)) + ((1 - tpr) * n_pos * (-500))
  ```
- **Task:** In production ML, selecting a probability threshold of $0.5$ is almost always sub-optimal under asymmetric business costs. Construct a dual-panel financial decision chart: Left panel showing TPR and FPR curves against threshold; Right panel showing Total Net Profit (in thousands of dollars) vs. Threshold. Highlight the Maximum Profit point with a vertical marker and financial payoff annotation, and compare it to the default $0.5$ threshold profit.
- **Expected Output:** A $1 \times 2$ figure. Right plot displays the inverted U-shaped Expected Profit curve. A gold star marks the global profit maximum. A shaded vertical ribbon marks the "Profitable Threshold Operating Range" ($Profit > 0$). Text callout indicates: `"Optimal Threshold = X.XX -> Max Profit = $YYk"`.
- **Constraints:**
  - Calculate maximum profit coordinate dynamically using `np.argmax(profit)`.
  - Format profit y-axis tick labels cleanly in thousands (`"$150k"`, `"$200k"`).
- **What You Need to Determine:**
  1. What is the optimal decision threshold that maximizes net business profit?
  2. How much net revenue does the business lose if the engineering team naively deploys the default $0.5$ threshold instead of the cost-optimal threshold?
  3. Why does the high cost of False Negatives (\$500) pull the optimal threshold to the left or right of $0.5$?

---

### Problem 77: Diagnosing Codebook Collapse in Discrete Generative Models (VQ-VAE)

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  # VQ-VAE model with codebook of 128 discrete embedding vectors
  codebook_size = 128
  indices = np.arange(codebook_size)

  # Checkpoint A (Step 500): Healthy uniform codebook utilization
  np.random.seed(42)
  usage_step500 = np.random.poisson(lam=75, size=codebook_size)

  # Checkpoint B (Step 10000): Severe Codebook Collapse
  # Only ~15 codes are heavily used, rest are dead/starved
  usage_step10000 = np.zeros(codebook_size)
  active_codes = [4, 12, 19, 27, 44, 58, 63, 77, 89, 94, 105, 112]
  usage_step10000[active_codes] = np.random.uniform(500, 1800, len(active_codes))
  usage_step10000 += np.random.poisson(lam=1.5, size=codebook_size)  # Background noise
  ```
- **Task:** Generative models with discrete latent variables (VQ-VAE, VQ-GAN) frequently suffer from "Codebook Collapse," where a small fraction of latent vectors dominate while the rest receive zero gradient updates. Create a $2 \times 1$ vertically stacked diagnostic figure comparing codebook index usage histograms between Step 500 and Step 10000. Calculate and display the Codebook Perplexity metric: $\mathcal{P} = \exp\left(-\sum p_i \log p_i\right)$ for both checkpoints.
- **Expected Output:** Two subplots sharing the x-axis ($[0, 127]$ codebook indices). Subplot 1 shows healthy, relatively uniform bar distributions across all 128 codes. Subplot 2 shows extreme sparsity with isolated tall needles and mostly dead codes. Large annotated metric card displaying the collapse in Perplexity from $\approx 125$ down to $\approx 14$.
- **Constraints:**
  - Compute normalized probabilities $p_i = \text{usage} / \sum \text{usage}$ and calculate perplexity mathematically using `np.exp(-np.sum(p * np.log(p + 1e-12)))`.
  - Highlight dead codes (usage $< 5$) with red tick marks along the baseline.
- **What You Need to Determine:**
  1. What percentage of the codebook vectors are virtually dead (unutilized) at Step 10000?
  2. How does the calculated Perplexity metric capture codebook utilization in a single summary statistic?
  3. What algorithmic regularizations (e.g. EMA codebook updates, commitment loss, codebook reset) prevent this failure mode?

---

### Problem 78: Multi-Task Learning: Dynamic Weight Averaging and Loss Balancing

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  epochs = np.arange(1, 41)
  # Multi-task autonomous driving network: 3 competing heads
  # Task 1: Semantic Segmentation (Cross-Entropy Loss ~ 1.5 -> 0.4)
  loss_seg = 1.5 * np.exp(-0.08 * epochs) + 0.35 + np.random.normal(0, 0.02, 40)
  # Task 2: Depth Estimation (Scale-Invariant MSE ~ 45.0 -> 12.0)
  loss_depth = 45.0 * np.exp(-0.05 * epochs) + 11.0 + np.random.normal(0, 0.5, 40)
  # Task 3: 3D Bounding Box Regression (Smooth L1 ~ 8.0 -> 1.5)
  loss_bbox = 8.0 * np.exp(-0.07 * epochs) + 1.2 + np.random.normal(0, 0.1, 40)

  # Dynamic loss weights assigned by GradNorm or Uncertainty Weighting
  w_seg = 0.8 + 0.02 * epochs
  w_depth = 0.025 * np.ones(40) - 0.0003 * epochs
  w_bbox = 0.15 + 0.005 * np.sin(epochs / 3.0)
  ```
- **Task:** Training multi-task networks with static equal weights causes gradient starvation when task losses operate on disparate numerical scales. Build a $1 \times 2$ dashboard: Left panel showing the three raw loss curves on a logarithmic y-scale; Right panel showing the dynamically evolving task weights ($w_{\text{seg}}, w_{\text{depth}}, w_{\text{bbox}}$) on a secondary twin axis or stacked area chart showing relative percentage contribution of each task to the total gradient.
- **Expected Output:** Two synchronized subplots. Left plot has log y-scale showing how Depth loss ($> 10$) dwarfs Segmentation loss ($< 1$) in raw magnitude. Right plot displays a $100\%$ stacked area chart (`ax.stackplot`) illustrating how dynamic weighting balances the gradient contributions over time.
- **Constraints:**
  - Left plot must employ a logarithmic y-axis (`ax.set_yscale('log')`).
  - Right plot must normalize weights such that $w_1 + w_2 + w_3 = 1.0$ at every epoch.
- **What You Need to Determine:**
  1. If static unweighted loss summation were used ($\mathcal{L} = \mathcal{L}_{\text{seg}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{bbox}}$), which task's gradients would overwhelm the shared backbone weights?
  2. Over the course of training, does the dynamic weighting policy prioritize segmentation or depth estimation more heavily?
  3. Why is normalized gradient balancing essential for preventing negative transfer between competing multi-task objectives?

---

### Problem 79: Algorithmic Fairness & Disparate Impact Audit (Equalized Odds)

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import pandas as pd

  demographics = ["Group_A (Majority)", "Group_B (Minority)", "Group_C (Protected)", "Group_D (Protected)"]
  
  audit_data = {
      "Group": demographics,
      "TPR": [0.84, 0.62, 0.79, 0.58],        # True Positive Rate (Opportunity)
      "FPR": [0.08, 0.19, 0.09, 0.24],        # False Positive Rate (Unfair Burden)
      "Selection_Rate": [0.38, 0.22, 0.35, 0.18], # Demographic Parity ratio
      "Sample_Size": [12000, 3500, 2800, 1100]
  }
  df_fairness = pd.DataFrame(audit_data)
  ```
- **Task:** An AI compliance committee requires an audit visualization for a credit scoring model under the Equalized Odds fairness criterion ($|\text{TPR}_g - \text{TPR}_{base}| \le 0.05$ and $|\text{FPR}_g - \text{FPR}_{base}| \le 0.05$). Construct an Equalized Odds scatter chart plotting False Positive Rate (x-axis) vs. True Positive Rate (y-axis). Draw a crosshair tolerance corridor centered at the majority reference group, and mark non-compliant demographic groups violating fair lending standards.
- **Expected Output:** A 2D fairness scatter plot with coordinate axes $[0.0, 0.3]$ for FPR and $[0.5, 1.0]$ for TPR. Reference group marked with a prominent diamond marker. A light green shaded rectangle represents the $\pm 5\%$ Equalized Odds compliance zone around the reference group. Protected groups falling outside the zone are labeled with red warning callouts.
- **Constraints:**
  - Shade the fairness tolerance envelope using `patches.Rectangle` centered at $(FPR_{\text{majority}}, TPR_{\text{majority}})$.
  - Annotate each scatter point with its demographic group name and sample size.
- **What You Need to Determine:**
  1. Which demographic group suffers from both a substantially lower True Positive Rate and a substantially higher False Positive Rate?
  2. Does Group C meet the $\pm 5\%$ Equalized Odds compliance standard relative to the majority group?
  3. What real-world harms occur when a group suffers from high FPR (unjust punishment/denial) combined with low TPR (missed opportunity)?

---

### Problem 80: Executive ML Architecture: Publication-Grade 9-Grid Model Evaluation Dashboard

- **Difficulty:** Level 4 — Coding-Round Challenge
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  # 1. Training dynamics
  epochs = np.arange(1, 21)
  t_loss = 1.8 * np.exp(-0.15 * epochs) + 0.1
  v_loss = 1.8 * np.exp(-0.13 * epochs) + 0.18 + 0.005 * epochs
  
  # 2. ROC & PR
  fpr = np.linspace(0, 1, 50); tpr = 1 - (1 - fpr)**4
  rec = np.linspace(0, 1, 50); prec = 1.0 / (1.0 + np.exp(7 * (rec - 0.7)))
  
  # 3. Calibration
  conf = np.linspace(0.1, 0.9, 8); acc = conf + np.random.normal(0, 0.04, 8)
  
  # 4. Confusion Matrix (3x3)
  cm = np.array([[850, 45, 12], [38, 720, 52], [15, 48, 680]])
  
  # 5. Top 5 Features
  feats = ["Payment_History", "Credit_Age", "Utilization", "Debt_Ratio", "Inquiries"]
  importances = [0.38, 0.24, 0.18, 0.12, 0.08]
  ```
- **Task:** High-level ML engineering interviews often conclude with an architectural dashboard challenge. Using Matplotlib's `GridSpec` layout engine (`fig.add_gridspec(3, 3)`), assemble a production-grade model evaluation dashboard incorporating:
  1. Loss Learning Curves (spanning row 0, cols 0-1)
  2. Calibration Reliability curve (spanning row 0, col 2)
  3. ROC Curve with AUC badge (row 1, col 0)
  4. Precision-Recall Curve (row 1, col 1)
  5. Feature Importance Horizontal Bars (row 1, col 2)
  6. Confusion Matrix Heatmap (row 2, cols 0-1)
  7. Key Model Performance KPIs executive summary text box card (row 2, col 2)
- **Expected Output:** A unified, executive-ready dashboard figure (`figsize=(16, 14)`). Consistent dark or tailored aesthetic, synchronized titles, zero overlapping labels, aligned grid lines, professional typography, and a master figure super-title (`fig.suptitle("Production Model V3.2 — Full Evaluation Audit", fontsize=18, fontweight='bold')`).
- **Constraints:**
  - Build exclusively using `matplotlib.gridspec.GridSpec`. Do not use `plt.subplot()`.
  - Format the KPI card in row 2, col 2 using `ax.text()` with a clean bounding box (`bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#ced4da')`), hiding the axis spines.
- **What You Need to Determine:**
  1. What are the key architectural advantages of using `GridSpec` over `plt.subplots()` for complex multi-scale dashboards?
  2. Based on the consolidated dashboard, is Model V3.2 ready for production deployment according to standard ML governance criteria?
  3. Which specific pane in the dashboard immediately signals whether the model suffers from overfitting?\n