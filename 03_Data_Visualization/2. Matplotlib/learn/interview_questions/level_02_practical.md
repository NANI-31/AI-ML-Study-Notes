# Matplotlib Coding Interview Practice — Level 2: Practical

This set contains 20 intermediate, practical coding problems focused on multi-variable exploration, subplot coordination, statistical comparisons, and real-world data visualization decisions in Data Science and ML engineering.

---

### Problem 21: Dual-Metric Dashboard — Loss and Learning Rate Synchronization

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  epochs = list(range(1, 21))
  train_loss = [2.8, 2.1, 1.6, 1.25, 0.98, 0.78, 0.62, 0.51, 0.43, 0.37, 0.33, 0.30, 0.28, 0.26, 0.25, 0.24, 0.23, 0.23, 0.22, 0.22]
  learning_rates = [1e-2 * (0.85 ** (e // 3)) for e in epochs]  # Step decay
  ```
- **Task:** An ML engineer wants to inspect training loss convergence alongside the stepped decay of the learning rate schedule over 20 epochs. Plot both metrics on a single chart using a dual y-axis (`ax.twinx()`), and generate a single unified legend.
- **Expected Output:** A dual-axis plot where the primary (left) axis displays training loss with a solid blue line, the secondary (right) axis displays learning rate on a logarithmic scale with a dashed amber line, with color-coded y-axis ticks and a single consolidated legend box.
- **Constraints:**
  - Consolidate legend entries from both axes into a single legend container using `get_legend_handles_labels()`.
  - Disable gridlines on the secondary axis (`ax2.grid(False)`) to prevent grid clash.
- **What You Need to Determine:**
  1. How does each step reduction in the learning rate correlate with the slope (rate of decrease) of the training loss curve?
  2. Why would plotting both metrics on a single standard linear y-axis be visually unviable?
  3. At which epoch does the learning rate undergo its first step decay?

---

### Problem 22: $2 \times 2$ Multi-Feature Distribution Matrix

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  n = 500
  df = pd.DataFrame({
      "gaussian_feat": np.random.normal(loc=50, scale=10, size=n),
      "skewed_feat": np.random.exponential(scale=15, size=n),
      "bimodal_feat": np.concatenate([np.random.normal(25, 4, n//2), np.random.normal(75, 6, n//2)]),
      "uniform_feat": np.random.uniform(low=0, high=100, size=n),
  })
  ```
- **Task:** Construct a $2 \times 2$ grid of subplots inspecting the probability density distribution of each of the 4 features, iterating through axes programmatically.
- **Expected Output:** A $2 \times 2$ figure where each panel contains a histogram with a Kernel Density Estimation (KDE) or density curve overlay (`density=True`), clear individual titles, distinct theme colors, and formatted axis labels.
- **Constraints:**
  - Iterate through the axes array cleanly using `axes.flat` rather than writing 4 repetitive hardcoded code blocks.
  - Set a shared figure super-title (`fig.suptitle()`) and apply `plt.tight_layout()`.
- **What You Need to Determine:**
  1. Which feature would require a log or Box-Cox transformation prior to linear regression modeling?
  2. What does the bimodal feature indicate about underlying latent subpopulations in the data?
  3. Which feature displays approximately constant density across its entire support domain?

---

### Problem 23: Multi-Architecture Benchmark Across 4 Evaluation Datasets

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  datasets = ["ImageNet-1k", "CIFAR-100", "SVHN", "Pascal-VOC"]
  models = {
      "ResNet-50": [76.2, 81.4, 95.2, 78.6],
      "EfficientNet-B4": [82.9, 87.1, 97.4, 84.3],
      "ViT-Base": [84.5, 89.2, 98.1, 86.7],
  }
  ```
- **Task:** Create a grouped bar chart comparing the top-1 test accuracy of the 3 model architectures across the 4 evaluation benchmark datasets.
- **Expected Output:** A grouped bar chart where each of the 4 dataset clusters contains 3 side-by-side vertical bars (one per model architecture), with distinct colors, a clear legend, and accuracy values labeled above each bar.
- **Constraints:**
  - Mathematically calculate bar offsets using `x_indices = np.arange(len(datasets))` and `width = 0.25`.
  - Center the x-axis tick labels directly beneath each 3-bar cluster.
  - Set the y-axis limits between $60\%$ and $105\%$ to highlight relative performance gaps without wasting vertical space.
- **What You Need to Determine:**
  1. Which model architecture achieves consistent superiority across all four benchmark datasets?
  2. On which dataset is the performance spread among the three models the narrowest?
  3. What is the average accuracy gain achieved by ViT-Base over ResNet-50 across these benchmarks?

---

### Problem 24: Detecting and Isolating Extreme Typographical Outliers

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import pandas as pd

  # Living area (sq ft) vs Sales Price (USD) with human data-entry typos
  sqft = [850, 1100, 1250, 1400, 1600, 1850, 2100, 2400, 2750, 3100, 3500, 125000, 1800]
  price = [220000, 275000, 310000, 350000, 395000, 440000, 510000, 580000, 650000, 720000, 810000, 320000, 42000000]
  df = pd.DataFrame({"sqft": sqft, "price": price})
  ```
- **Task:** Create a side-by-side $1 \times 2$ subplot comparison:
  - Panel 1: Raw scatter plot demonstrating how the two extreme data-entry typos ($125,000\text{ sq ft}$ and $\$42,000,000$) compress the remaining legitimate data into an indistinguishable clump at the origin.
  - Panel 2: Cleaned scatter plot where points exceeding $3\sigma$ or reasonable domain thresholds ($> 5,000\text{ sq ft}$ or $>\$2,000,000$) have been filtered out, showing the true underlying linear relationship.
- **Expected Output:** Two subplots side-by-side: Panel 1 titled `"Raw Data (Outliers Present)"` and Panel 2 titled `"Filtered Domain Data (Cleaned)"`, with currency and area axis formatting.
- **Constraints:**
  - Do not manually delete rows by hardcoded indices; write filtering logic based on threshold filtering.
- **What You Need to Determine:**
  1. What was the exact nature of the two typographical errors in the raw dataset?
  2. How does the presence of extreme outliers distort the scale of linear axes in Matplotlib?
  3. What is the approximate slope (price per square foot) visible in the cleaned dataset?

---

### Problem 25: Stacked Area vs. 100% Component Traffic Evolution

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import pandas as pd

  months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
  traffic = {
      "Organic Search": [120, 135, 150, 160, 175, 190, 210, 225, 240, 260, 290, 320],
      "Paid Search": [80, 95, 110, 105, 90, 85, 90, 100, 115, 130, 160, 180],
      "Social Media": [45, 52, 60, 75, 90, 110, 135, 150, 170, 195, 230, 270],
      "Direct": [60, 62, 65, 68, 70, 74, 78, 82, 85, 90, 98, 105],
  }
  df = pd.DataFrame(traffic, index=months)
  ```
- **Task:** Visualize the expansion of total inbound web traffic across 12 months using a Stacked Area chart (`ax.stackplot()`), and evaluate channel growth.
- **Expected Output:** A stacked area plot where each acquisition channel is shaded with a distinct, harmonious color palette, accompanied by a clear legend in the upper left, labeled axes (`"Month"` and `"Total Monthly Visits (Thousands)"`), and gridlines.
- **Constraints:**
  - Use `ax.stackplot()` with category labels and pass a custom color palette.
  - Set x-limits tightly to span from `"Jan"` to `"Dec"` without edge gaps.
- **What You Need to Determine:**
  1. Which traffic acquisition channel demonstrated the highest percentage growth between January and December?
  2. In which month did total combined traffic first surpass $500,000$ visits?
  3. What is a key cognitive limitation of stacked area charts when trying to read the exact trend of an interior series like "Paid Search"?

---

### Problem 26: 4-Variable Bubble Chart of Quantized LLM Models

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import pandas as pd

  models_data = {
      "model": ["TinyLlama-1B", "Mistral-7B-Q4", "Mistral-7B-FP16", "Llama-3-8B-Q4", "Llama-3-8B-FP16", "Llama-3-70B-Q4"],
      "latency_ms": [18, 42, 85, 48, 98, 310],          # Lower is better (X-axis)
      "mmlu_score": [36.2, 58.4, 60.1, 64.8, 66.4, 79.5],  # Higher is better (Y-axis)
      "param_billions": [1.1, 7.2, 7.2, 8.0, 8.0, 70.0],  # Bubble Size
      "vram_gb": [1.5, 4.8, 14.5, 5.5, 16.2, 42.0],       # Bubble Color
  }
  df = pd.DataFrame(models_data)
  ```
- **Task:** Construct a 4-dimensional bubble chart evaluating candidate LLMs for production deployment:
  - X-axis: Inference Latency (ms)
  - Y-axis: MMLU Benchmark Score
  - Marker Size: Model Parameter Count (billions)
  - Marker Color: VRAM Footprint (GB) with an associated colorbar
- **Expected Output:** A scatter bubble chart with scaled markers, a colorbar indicating VRAM footprint, model name text annotations placed beside each bubble, and reference dashed lines indicating target latency ($< 100\text{ms}$) and target accuracy ($> 60\%$).
- **Constraints:**
  - Scale marker size proportionally so bubbles are distinguishable without excessive overlap (`s = df['param_billions'] * 30`).
  - Use `ax.annotate()` to print each model's name adjacent to its marker.
- **What You Need to Determine:**
  1. Which model represents the optimal "sweet spot" of sub-100ms latency, $>60\%$ MMLU accuracy, and $<6\text{GB}$ VRAM footprint?
  2. How does 4-bit quantization (Q4) affect latency and VRAM relative to FP16 for the same parameter size?
  3. Why is Llama-3-70B-Q4 an outlier in latency, and what hardware constraints does it impose?

---

### Problem 27: Visualizing Missing Value Topology Across Features

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  n_samples = 100
  data = {
      "patient_id": range(1000, 1100),
      "age": [np.nan if np.random.rand() < 0.05 else int(x) for x in np.random.normal(50, 12, n_samples)],
      "systolic_bp": [np.nan if np.random.rand() < 0.15 else int(x) for x in np.random.normal(120, 15, n_samples)],
      "cholesterol": [np.nan if np.random.rand() < 0.35 else int(x) for x in np.random.normal(200, 30, n_samples)],
      "bmi": [np.nan if np.random.rand() < 0.10 else round(x, 1) for x in np.random.normal(27, 4, n_samples)],
      "smoking_status": [np.nan if np.random.rand() < 0.25 else "Smoker" for _ in range(n_samples)],
  }
  df = pd.DataFrame(data)
  ```
- **Task:** Create a missingness matrix heatmap (`ax.imshow()`) where missing entries (`NaN`) are rendered as dark bars and valid observations are rendered as light bars, revealing whether missingness is random or concentrated in specific columns.
- **Expected Output:** A binary heatmap matrix where rows represent individual patient samples ($0$ to $99$) and columns represent dataset features, with column names rotated $45^\circ$ along the horizontal axis and an annotation stating the column with the highest missingness percentage.
- **Constraints:**
  - Compute a boolean matrix using `df.isnull()` and map it via `ax.imshow(..., aspect='auto', cmap='binary')`.
  - Prevent x-tick labels from overlapping.
- **What You Need to Determine:**
  1. Which feature contains the highest proportion of missing values?
  2. Does missingness appear uniformly distributed across patients, or are there specific patient rows with multi-feature omissions?
  3. Would listwise row deletion (dropping all rows with any `NaN`) be acceptable here, or would it discard too much data?

---

### Problem 28: Bug Fix — Resolving 2D Subplot Indexing Errors

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np

  np.random.seed(42)
  data_streams = [np.random.randn(100).cumsum() for _ in range(6)]
  titles = [f"Sensor Node {i+1}" for i in range(6)]
  ```
- **Task:** An engineer wrote the following snippet to plot 6 sensor streams in a $2 \times 3$ grid, but the script crashes with `TypeError: 'Axes' object is not subscriptable` or `IndexError`:
  ```python
  # Faulty code:
  fig, axes = plt.subplots(2, 3)
  for i in range(6):
      axes[i].plot(data_streams[i])  # Crash here!
      axes[i].set_title(titles[i])
  plt.show()
  ```
  Explain why `axes[i]` fails on a 2D subplot grid, and rewrite the code using the canonical flattening pattern.
- **Expected Output:** A cleanly rendered $2 \times 3$ subplot figure containing all 6 sensor time-series streams, each with its designated title, unified y-axis limits, and a common x-axis label.
- **Constraints:**
  - Use `axes.flat` or `axes.ravel()` to iterate sequentially through the 2D array of axes.
  - Apply `plt.tight_layout()` to prevent vertical collisions between subplot titles and tick labels.
- **What You Need to Determine:**
  1. What is the shape of the `axes` NumPy array returned by `plt.subplots(2, 3)`?
  2. Why did indexing with a single integer `axes[i]` fail on the 2D array?
  3. Which sensor node demonstrated the largest positive cumulative drift over the 100-step window?

---

### Problem 29: Overlapping Distribution Comparison — Before vs. After Treatment

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np

  np.random.seed(123)
  # Diastolic blood pressure (mmHg) before and after medication
  bp_before = np.random.normal(loc=95, scale=8, size=300)
  bp_after = np.random.normal(loc=82, scale=7, size=300)
  ```
- **Task:** Compare the two clinical distributions on a single shared horizontal axis using overlapping histograms with transparency and statistical vertical markers.
- **Expected Output:** An overlapping dual-histogram plot where `bp_before` is shaded in coral red (`alpha=0.5`) and `bp_after` in sky blue (`alpha=0.5`), with shared bin edges, dashed vertical lines at the respective group means, and a legend.
- **Constraints:**
  - Define unified bin edges using `np.linspace(min_val, max_val, 30)` to ensure identical bar intervals for both distributions.
  - Set `density=True` so the vertical axis represents probability density rather than raw frequency count.
- **What You Need to Determine:**
  1. What is the absolute difference in mean blood pressure between the before and after cohorts?
  2. Did the medication reduce the variance (spread) of blood pressure, or primarily shift the central tendency?
  3. Approximately what percentage of patients in the "after" group have blood pressure below the "before" group mean?

---

### Problem 30: Feature Selection — Evaluating 4 Continuous Predictors vs. Target

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  n = 200
  target = np.linspace(10, 100, n)
  df = pd.DataFrame({
      "Feature_1": target * 0.8 + np.random.normal(0, 10, n),       # Strong positive linear
      "Feature_2": -0.5 * target + np.random.normal(0, 15, n),      # Moderate negative linear
      "Feature_3": np.sin(target / 10) * 30 + 50 + np.random.normal(0, 5, n), # Non-linear sinusoidal
      "Feature_4": np.random.normal(50, 20, n),                     # Zero correlation noise
      "Target": target,
  })
  ```
- **Task:** Create a $2 \times 2$ scatter subplot layout plotting each of the 4 features against `Target`. Compute and annotate the Pearson correlation coefficient ($r$) on each panel.
- **Expected Output:** Four scatter subplots in a $2 \times 2$ configuration, each plotting `Feature_k` on the x-axis and `Target` on the y-axis, with an annotated text box in the upper corner displaying `r = X.XX`.
- **Constraints:**
  - Compute Pearson correlation dynamically using `scipy.stats.pearsonr` or `np.corrcoef()`.
  - Maintain consistent marker size (`s=15`) and color palette across all subplots.
- **What You Need to Determine:**
  1. Which feature has the highest linear predictive utility for a standard linear regression model?
  2. Why does `Feature_3` exhibit a low Pearson $r$ despite having a strong, deterministic relationship with `Target`?
  3. Which feature should be immediately pruned during linear feature selection due to near-zero predictive signal?

---

### Problem 31: Time Series Decomposition — Trend, Seasonality, and Residuals

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  days = pd.date_range("2025-01-01", periods=180, freq="D")
  t = np.arange(180)
  trend = 100 + 0.5 * t
  seasonality = 15 * np.sin(2 * np.pi * t / 7)  # Weekly cycle
  noise = np.random.normal(0, 3, 180)
  observed = trend + seasonality + noise
  ```
- **Task:** Plot the classical additive time-series decomposition across 3 vertically stacked subplots with shared horizontal date rulers (`sharex=True`):
  - Panel 1: Observed raw series alongside the extracted Trend curve.
  - Panel 2: Extracted Seasonal oscillation cycle.
  - Panel 3: Random Noise residuals.
- **Expected Output:** A $3 \times 1$ subplot figure with connected date rulers, no redundant interior date tick labels, subtle horizontal zero reference lines on seasonal and residual panels, and individual panel titles.
- **Constraints:**
  - Use `fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))`.
  - Call `fig.autofmt_xdate()` so only the bottom panel displays rotated date labels.
- **What You Need to Determine:**
  1. What is the period of the seasonal oscillation (in days)?
  2. What is the average daily growth rate of the trend component?
  3. Do the residuals appear homoscedastic (stationary variance around zero), or is there evidence of structural drift?

---

### Problem 32: K-Fold Cross-Validation Performance Stability

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  folds = [f"Fold {i+1}" for i in range(5)]
  models = {
      "Logistic Regression": [0.78, 0.79, 0.77, 0.78, 0.78],
      "Random Forest": [0.85, 0.88, 0.82, 0.89, 0.81],
      "Gradient Boosting": [0.88, 0.89, 0.87, 0.90, 0.88],
  }
  ```
- **Task:** Create a visualization that compares not only the mean cross-validation accuracy of each model, but also the stability/variance across folds.
- **Expected Output:** A point plot or bar chart displaying the mean accuracy for each model with vertical error bars representing $\pm 1$ standard deviation, annotated with exact mean and std values (`"0.88 ± 0.01"`).
- **Constraints:**
  - Compute means and standard deviations dynamically using NumPy.
  - Set y-limits from $0.70$ to $0.95$ to make variance comparisons obvious.
- **What You Need to Determine:**
  1. Which model achieves the highest mean accuracy across the 5 folds?
  2. Which model displays the highest variance (lowest stability) across folds?
  3. In a high-reliability production environment, why might a team prefer a model with slightly lower mean accuracy if it has significantly lower variance?

---

### Problem 33: Multi-Collinearity Pairwise Scatter Matrix ($3 \times 3$)

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  n = 150
  x1 = np.random.normal(50, 10, n)
  x2 = 2.5 * x1 + np.random.normal(0, 5, n)      # Severe multi-collinearity!
  x3 = np.random.uniform(10, 40, n)             # Independent
  df = pd.DataFrame({"Engine_Size": x1, "Curb_Weight": x2, "Highway_MPG": x3})
  ```
- **Task:** Construct a custom $3 \times 3$ pairwise scatter plot matrix:
  - Diagonal panels ($i = j$): Histograms of the respective single variable.
  - Off-diagonal panels ($i \neq j$): Scatter plots of variable pairs.
- **Expected Output:** A cohesive $3 \times 3$ matrix of subplots with labeled outer perimeter axes, no redundant interior axis labels, and tight subplot spacing.
- **Constraints:**
  - Do not use Seaborn's `pairplot()`; construct the grid natively using Matplotlib `plt.subplots(3, 3)`.
- **What You Need to Determine:**
  1. Which pair of features exhibits severe multi-collinearity that could destabilize regression coefficient estimates?
  2. What visual clue on the off-diagonal scatter plots immediately signals strong linear dependency?
  3. What dimensionality reduction or feature engineering step should be applied before feeding these features to a linear model?

---

### Problem 34: Identifying Correlation Direction and Strength Across 4 Pairs

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np

  np.random.seed(42)
  n = 100
  x = np.linspace(0, 50, n)
  pairs = {
      "Pair 1": (x, x * 0.9 + np.random.normal(0, 3, n)),
      "Pair 2": (x, -0.8 * x + np.random.normal(0, 8, n)),
      "Pair 3": (x, np.random.normal(25, 10, n)),
      "Pair 4": (x, (x - 25)**2 / 10 + np.random.normal(0, 4, n)),
  }
  ```
- **Task:** Plot each of the 4 pairs in a $2 \times 2$ grid. Determine the direction, strength, and linearity of correlation for each pair.
- **Expected Output:** Four scatter subplots with trendlines or curve sketches where appropriate, labeled axes, and individual titles stating the qualitative relationship.
- **Constraints:**
  - Add horizontal and vertical zero reference lines where appropriate.
- **What You Need to Determine:**
  1. Which pair exhibits strong positive linear correlation?
  2. Which pair exhibits moderate negative linear correlation?
  3. Which pair exhibits zero correlation?
  4. Which pair exhibits strong non-linear (quadratic) dependence despite having a near-zero Pearson linear correlation?

---

### Problem 35: Executive Presentation Redesign — Replacing a 12-Slice 3D Pie Chart

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  departments = ["Engineering", "Sales", "Marketing", "Customer Support", "Operations", 
                 "Product", "Legal", "Finance", "HR", "Design", "Security", "Compliance"]
  budget_millions = [42.5, 38.0, 24.5, 18.2, 14.0, 12.5, 8.4, 7.8, 6.2, 5.5, 4.8, 3.1]
  ```
- **Task:** A junior analyst proposed a 12-slice pie chart for an executive board meeting. Redesign this visualization into an executive-ready horizontal bar chart sorted in descending order, with direct dollar value callouts and cumulative percentage share.
- **Expected Output:** A clean horizontal bar chart where the largest budget sits at the top (`ax.invert_yaxis()`), bars are colored in a muted corporate navy hue, and exact spending values (`"$42.5M"`) are annotated beside each bar.
- **Constraints:**
  - Format the x-axis with dollar currency units.
  - Highlight the top two departments ("Engineering" and "Sales") with a distinct accent color.
- **What You Need to Determine:**
  1. Why is a 12-slice pie chart ineffective for executive decision-making compared to a sorted horizontal bar chart?
  2. What combined percentage of the total budget is consumed by the top 3 departments alone?
  3. How does placing labels directly beside bars improve cognitive processing speed compared to a separate color legend?

---

### Problem 36: Detecting Data Leakage & Covariate Shift (Train vs. Test)

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  # Production data drift scenario
  train_feature = np.random.normal(loc=100, scale=15, size=1000)
  test_feature = np.random.normal(loc=125, scale=22, size=500)  # Significant drift!
  ```
- **Task:** Construct a diagnostic plot to detect whether covariate shift (distribution drift) exists between the model training dataset and the unseen test evaluation dataset.
- **Expected Output:** An overlapping probability density plot (KDE or normalized histograms with `density=True`), with distinct colors (blue for Train, red for Test), semi-transparent fills, vertical dashed lines at the respective means, and a legend.
- **Constraints:**
  - Enforce common binning across the global min and max of both sets.
  - Annotate the difference between the two distribution means in the upper plot margin.
- **What You Need to Determine:**
  1. What is the magnitude of the mean shift between the training set and the test set?
  2. How would an ML model trained strictly on `train_feature` likely perform when evaluated on `test_feature`?
  3. What remediation strategies (e.g. re-weighting, adversarial validation, domain adaptation) should be considered based on this visual diagnostic?

---

### Problem 37: Principal Component Analysis (PCA) Scree & Cumulative Variance Plot

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np

  explained_variance_ratio = [0.38, 0.22, 0.14, 0.09, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01]
  components = list(range(1, 11))
  cumulative_variance = np.cumsum(explained_variance_ratio)
  ```
- **Task:** Create a dual-representation PCA Scree plot showing both individual explained variance per component (as vertical bars) and cumulative explained variance (as an overlaid step/line curve), with a horizontal threshold line at $90\%$ explained variance.
- **Expected Output:** A combined bar and line chart on a single axis (or dual axis), with bars representing individual variance ratio, an overlaid step/line plot for cumulative variance, a red horizontal dashed line at $0.90$ ($90\%$), and an annotation marking the minimum number of components needed to retain $90\%$ information.
- **Constraints:**
  - Set y-limits from $0.0$ to $1.05$ with percentage tick formatting.
  - Ensure integer ticks along the horizontal axis ($1$ through $10$).
- **What You Need to Determine:**
  1. What is the minimum number of principal components required to preserve at least $80\%$ of total dataset variance?
  2. What is the minimum number required to capture at least $90\%$ variance?
  3. At which component does the "elbow" occur where additional components yield negligible explanatory value?

---

### Problem 38: Customer Lifetime Value (CLV) Across Demographic Segments

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  segments = ["Gen-Z", "Millennials", "Gen-X", "Boomers"]
  clv_data = {
      "Gen-Z": np.random.exponential(scale=350, size=200) + 50,
      "Millennials": np.random.normal(loc=1200, scale=300, size=200),
      "Gen-X": np.random.normal(loc=1650, scale=400, size=200),
      "Boomers": np.random.normal(loc=950, scale=250, size=200),
  }
  ```
- **Task:** Create a comparative Box Plot visualizing the median, interquartile range (IQR), and outlier profiles of Customer Lifetime Value across the 4 generational segments.
- **Expected Output:** A stylized box plot with custom box fills, mean markers (`showmeans=True`), clean currency y-axis ticks (`"$1,000"`), and clear categorical labels.
- **Constraints:**
  - Color each box patch with a distinct soft fill color using `patch_artist=True`.
  - Format the y-axis with thousands-separating commas.
- **What You Need to Determine:**
  1. Which demographic cohort possesses the highest median Customer Lifetime Value?
  2. Which cohort displays extreme right-skewed positive outlier behavior?
  3. Between Millennials and Gen-X, which group displays higher variance (IQR spread)?

---

### Problem 39: Bug Fix — Eliminating Subplot Title and Label Collisions

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import numpy as np

  t = np.linspace(0, 10, 100)
  signals = [np.sin(t * (i + 1)) for i in range(4)]
  titles = [f"Fourier Harmonic Mode {i+1} (Frequency = {i+1} Hz)" for i in range(4)]
  ```
- **Task:** A student created a $4 \times 1$ stacked subplot figure, but the x-axis tick labels of upper subplots overlap directly onto the title text of the subplots beneath them.
  ```python
  # Faulty layout code:
  fig, axes = plt.subplots(4, 1, figsize=(8, 6))
  for i in range(4):
      axes[i].plot(t, signals[i])
      axes[i].set_title(titles[i])
      axes[i].set_xlabel("Time (seconds)")
      axes[i].set_ylabel("Amplitude (V)")
  plt.savefig("signals_collided.png")
  ```
  Identify the root causes of visual crowding in this layout and fix it using proper axis sharing, margin adjustment, and label cleanup.
- **Expected Output:** A clean, uncrowded multi-panel graphic where `sharex=True` suppresses redundant interior x-axis labels, individual titles have adequate vertical padding (`pad=6`), and subplots have generous vertical spacing.
- **Constraints:**
  - Use `plt.subplots(4, 1, sharex=True)`.
  - Apply `plt.subplots_adjust(hspace=0.4)` or `fig.tight_layout()`.
- **What You Need to Determine:**
  1. Why did the xlabel `"Time (seconds)"` on row 1 collide with the title on row 2?
  2. How does `sharex=True` eliminate redundant text ink while preserving temporal synchronization?
  3. What is the frequency difference between Mode 1 and Mode 4?

---

### Problem 40: Multi-Quarter Economic Regime Shading

- **Difficulty:** Level 2 — Practical
- **Dataset:**
  ```python
  import pandas as pd
  import numpy as np

  dates = pd.date_range("2024-01-01", periods=24, freq="MS")  # 2 years of monthly data
  np.random.seed(42)
  inflation_rate = [3.1, 3.2, 3.4, 3.6, 3.9, 4.2, 4.5, 4.3, 4.1, 3.8, 3.5, 3.2,
                    3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.4, 2.3, 2.2, 2.1, 2.0]
  ```
- **Task:** Plot monthly inflation over the 24-month horizon. Highlight the "High Inflation Regime" (months where inflation exceeded $3.5\%$) with a shaded red background band (`ax.axvspan`), and the "Target Stable Regime" ($< 2.5\%$) with a light green shaded band.
- **Expected Output:** A continuous line chart with formatted month-year dates (`"%b %Y"`), distinct semi-transparent shaded bands for the two economic regimes, and a legend explaining the regime boundaries.
- **Constraints:**
  - Use `fig.autofmt_xdate(rotation=30, ha='right')` for date axis alignment.
  - Set the background span opacities to `alpha=0.15`.
- **What You Need to Determine:**
  1. In which calendar month did inflation peak, and what was the maximum value?
  2. In which month did the economy first transition into the "Target Stable Regime"?
  3. For how many consecutive months did inflation remain above the $3.5\%$ policy threshold?

---
