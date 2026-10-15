# Matplotlib Coding Interview Practice — Level 3: ML/AI-Oriented

This set contains 20 machine learning and artificial intelligence engineering problems. These questions simulate real-world ML model diagnostics, performance evaluation, error analysis, hyperparameter tuning, and latent representation visualization encountered in ML/AI engineering coding rounds.

---

### Problem 41: Diagnosing Overfitting and Generalization Gap via Learning Curves

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(101)
  epochs = np.arange(1, 31)

  # Simulated learning curves
  train_loss = 2.5 * np.exp(-0.15 * epochs) + 0.08 + np.random.normal(0, 0.015, len(epochs))
  val_loss = 2.5 * np.exp(-0.12 * epochs[:12]).tolist()
  # Validation loss starts diverging after epoch 12
  val_loss += [val_loss[-1] + 0.035 * (i ** 1.3) + np.random.normal(0, 0.02) for i in range(1, 19)]
  val_loss = np.array(val_loss)

  train_acc = 1.0 - 0.85 * np.exp(-0.14 * epochs) + np.random.normal(0, 0.01, len(epochs))
  val_acc = 1.0 - 0.85 * np.exp(-0.11 * epochs[:12]).tolist()
  val_acc += [val_acc[-1] - 0.012 * (i ** 1.1) + np.random.normal(0, 0.015) for i in range(1, 19)]
  val_acc = np.array(val_acc)

  history = pd.DataFrame({"epoch": epochs, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})
  ```
- **Task:** Create a $1 	imes 2$ diagnostic dashboard showing Loss on the left subplot and Accuracy on the right subplot. Highlight the optimal early-stopping epoch on both subplots with a vertical dashed line and a shaded region indicating the "Overfitting Zone".
- **Expected Output:** Two synchronized subplots sharing the x-axis ($1$ to $30$). The left subplot displays training vs. validation loss; the right displays training vs. validation accuracy. A red vertical dashed line marks the minimum validation loss, and an `axvspan` shades the post-minimum epochs in light red labeled "Overfitting Regime".
- **Constraints:**
  - Programmatically calculate the epoch where validation loss is minimized rather than hardcoding it.
  - Annotate the minimum validation loss point with an arrow pointing to the coordinate $(epoch^*, val\_loss^*)$.
  - Restrict accuracy y-axis to $[0.4, 1.05]$.
- **What You Need to Determine:**
  1. Exactly which epoch represents the optimal checkpoint before the model begins memorizing noise?
  2. What is the numerical generalization gap (difference between train accuracy and validation accuracy) at epoch 30 versus epoch 12?
  3. If an early stopping patience of 3 epochs were configured, at which epoch would training officially terminate?

---

### Problem 42: Precision-Recall Curve with Iso-$F_1$ Contours and Baseline Reference

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  # Simulated output from precision_recall_curve on an imbalanced fraud detection dataset
  recall = np.linspace(0.0, 1.0, 100)
  # Model precision degrades as recall increases
  precision = 1.0 / (1.0 + np.exp(6 * (recall - 0.65))) + np.random.normal(0, 0.01, 100)
  precision = np.clip(precision, 0.0, 1.0)
  precision = np.maximum.accumulate(precision[::-1])[::-1] # Enforce monotonic decreasing trend

  # Class balance: 350 fraud cases out of 10,000 transactions
  positive_prevalence = 350 / 10000
  ```
- **Task:** Plot the Precision-Recall curve for this fraud detection classifier. Overlay iso-$F_1$ score curves for $F_1 \in [0.2, 0.4, 0.6, 0.8]$ as dotted background contour lines, and add a horizontal dashed line representing the no-skill classifier baseline.
- **Expected Output:** A square figure ($8 \times 8$ inches) with Recall on the x-axis and Precision on the y-axis, both bounded in $[0.0, 1.05]$. Iso-$F_1$ contours labeled with their respective scores running across the plot. A bold navy curve for the model with the Area Under the PR Curve (PR-AUC) displayed in the legend.
- **Constraints:**
  - Calculate iso-$F_1$ contours using the harmonic mean formula rearranged for precision: $P = \frac{F_1 \cdot R}{2R - F_1}$.
  - Place contour text labels at $R \approx 0.85$ or using `ax.text()` along the curves.
  - Plot the horizontal no-skill baseline at $y = \text{positive\_prevalence}$.
- **What You Need to Determine:**
  1. What is the maximum $F_1$ score achievable by this model, and at approximately what recall value does it occur?
  2. Why is the no-skill baseline for a PR curve equal to the positive prevalence, unlike the constant $0.5$ diagonal baseline of an ROC curve?
  3. For a business requirement requiring at least $80\%$ precision, what is the maximum recall the system can provide?

---

### Problem 43: Regression Error Analysis — Predicted vs. Actual with Residual Envelopes

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  actual = np.random.uniform(100, 1000, 300)
  # Model has multiplicative heteroscedastic noise (larger error at higher prices)
  noise = np.random.normal(0, 0.12 * actual)
  predicted = actual + noise
  # Inject 5 severe prediction outliers (e.g. pipeline data formatting bugs)
  predicted[15] = 950
  actual[15] = 220
  predicted[88] = 200
  actual[88] = 780
  predicted[142] = 1100
  actual[142] = 400

  df = pd.DataFrame({"actual": actual, "predicted": predicted})
  ```
- **Task:** Construct a "Predicted vs. Actual" scatter plot for a property appraisal model. Overlay the ideal $y=x$ parity line, $\pm 15\%$ error tolerance margin bands, and highlight severe outliers ($|\text{predicted} - \text{actual}| > 200$) in bright red with index callout annotations.
- **Expected Output:** A scatter plot with equal aspect ratio (`ax.set_aspect('equal')`). The ideal line is dashed dark gray. A light blue shaded band covers $0.85x \le y \le 1.15x$. Normal predictions are plotted as semi-transparent dark circles, while outliers are flagged with distinct red markers and text labels showing their row indices.
- **Constraints:**
  - Use `ax.fill_between()` with the reference $x$ array to shade the $\pm 15\%$ tolerance corridor.
  - Enforce identical axis limits for both x and y ($[50, 1200]$) so the $y=x$ parity line aligns perfectly at a $45^\circ$ geometric angle.
- **What You Need to Determine:**
  1. Does the model suffer from systematic under-prediction or over-prediction at high property valuations ($> 800$)?
  2. What percentage of points fall inside the $\pm 15\%$ acceptable tolerance band?
  3. What common modeling phenomenon is evidenced by the increasing spread of points as `actual` increases?

---

### Problem 44: Normalized Multiclass Confusion Matrix with Counts and Percentages

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  classes = ["Benign", "Adware", "Spyware", "Ransomware", "Trojan"]
  # Raw confusion matrix counts (rows: True class, columns: Predicted class)
  cm_counts = np.array([
      [420,  15,   8,   2,   5],
      [ 22, 280,  45,   3,  10],
      [  5,  38, 210,  12,  25],
      [  1,   4,   6, 185,  14],
      [  8,  12,  30,  20, 310]
  ])
  ```
- **Task:** Build a normalized confusion matrix heatmap showing row-wise percentages (recall per class) as cell color intensity, while annotating each cell with both the normalized percentage and the raw integer count (e.g., `"78.5%\n(n=280)"`).
- **Expected Output:** A $5 \times 5$ heatmap matrix using `ax.imshow()` with an intuitive colormap (such as `Blues`). The x-axis represents "Predicted Label" and the y-axis represents "True Label". Ticks are explicitly labeled with class names. Colorbar indicates recall percentages $[0, 100\%]$. Text contrast automatically inverts (white text on dark cells, black text on light cells).
- **Constraints:**
  - Normalize rows such that each row sums to $1.0$ ($100\%$).
  - Dynamically set text color based on cell intensity threshold ($> 0.5 \to \text{white}$, $\le 0.5 \to \text{black}$).
  - Do not use Seaborn; implement entirely using native Matplotlib.
- **What You Need to Determine:**
  1. Which specific malware class experiences the highest rate of false negatives (misclassified as Benign)?
  2. Which pair of malware classes exhibits the strongest bidirectional confusion?
  3. What is the overall classification accuracy across all 5 classes combined?

---

### Problem 45: Multi-Model ROC Curve Comparison with Optimal Operating Point Detection

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  # Simulated FPR and TPR points for 3 competing models
  fpr_baseline = np.linspace(0, 1, 100)
  tpr_baseline = fpr_baseline ** 0.5  # Modest curvature (AUC ~ 0.67)

  fpr_rf = np.linspace(0, 1, 100)
  tpr_rf = 1 - (1 - fpr_rf) ** 3       # Stronger curvature (AUC ~ 0.85)

  fpr_xgb = np.linspace(0, 1, 100)
  tpr_xgb = 1 - (1 - fpr_xgb) ** 5      # Superior curvature (AUC ~ 0.93)
  ```
- **Task:** Plot the ROC curves for Baseline Logistic Regression, Random Forest, and XGBoost on a single axis. Compute the AUC for each model via the trapezoidal rule (`np.trapz`), display each AUC in the legend, and mark the optimal operating point (Youden's J statistic: $\max(TPR - FPR)$) on the best-performing model with a gold star marker and coordinate callout.
- **Expected Output:** A square ROC plot with False Positive Rate on the x-axis and True Positive Rate on the y-axis. A red dashed diagonal line indicates random chance ($AUC = 0.50$). Three curves with distinct colors and line styles. The maximum Youden point on XGBoost marked with a prominent star marker and text annotation giving $(FPR^*, TPR^*, J^*)$.
- **Constraints:**
  - Compute AUC programmatically using `np.trapz(tpr, fpr)`.
  - Both axes must range precisely from $0.0$ to $1.0$ with major gridlines at intervals of $0.1$.
- **What You Need to Determine:**
  1. Which model achieves the highest AUC, and what is its numeric value?
  2. At the optimal Youden's J threshold for XGBoost, what is the trade-off in False Positive Rate incurred to achieve the corresponding True Positive Rate?
  3. If a production SLA imposes a strict upper bound of $5\%$ False Positive Rate, which model yields the highest True Positive Rate within that budget?

---

### Problem 46: Permutation Feature Importance with Error Bars and Directional Color Coding

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import pandas as pd

  data = {
      "feature": ["credit_score", "debt_to_income", "annual_income", "num_delinquencies", 
                  "revolving_util", "loan_amount", "emp_length", "inquiries_6m", "home_ownership", "zip_code"],
      "mean_importance": [0.142, 0.125, 0.088, 0.065, 0.051, 0.038, 0.015, 0.008, -0.004, -0.018],
      "std_importance":  [0.018, 0.015, 0.012, 0.009, 0.008, 0.006, 0.005, 0.004,  0.003,  0.005]
  }
  df_importance = pd.DataFrame(data)
  ```
- **Task:** Create a clean, horizontal bar chart displaying permutation feature importance scores sorted in ascending order (most important feature at the top). Show standard deviation error bars, draw a vertical reference line at $0.0$, and color positive contributions in emerald green and negative contributions (features hurting validation score) in crimson red.
- **Expected Output:** A horizontal bar chart (`ax.barh`) where features are ranked from bottom to top. A dark gray dashed line at $x=0$. Error bars are capped with thin crossbars (`capsize=4`). Feature names are cleanly aligned on the y-axis without clipping.
- **Constraints:**
  - Sort the DataFrame programmatically before plotting so the bars are guaranteed to be monotonically ordered.
  - Use conditional list comprehension to assign bar colors based on whether `mean_importance >= 0`.
- **What You Need to Determine:**
  1. Which features appear to introduce noise or degrade generalization performance when permuted?
  2. What is the cumulative importance of the top two features relative to the total positive feature importance?
  3. Would dropping `zip_code` and `home_ownership` likely improve, degrade, or leave model cross-validation score unchanged?

---

### Problem 47: Residual Diagnostics — Heteroscedasticity Scatter and Residual Normality

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  np.random.seed(55)
  fitted_vals = np.linspace(10, 100, 250)
  # Residuals exhibiting slight non-linear curvature and heteroscedastic fan spread
  residuals = 0.05 * (fitted_vals - 50)**2 / 20 + np.random.normal(0, 0.08 * fitted_vals)
  ```
- **Task:** Build a 2-panel residual diagnostic figure. The left subplot must show Residuals vs. Fitted Values with a horizontal zero line and a smoothed trendline (polynomial or rolling mean). The right subplot must show a histogram of residuals with an overlaid Gaussian probability density function (mean and std matching residuals).
- **Expected Output:** A figure with two subplots ($1 \times 2$). Subplot 1: scatter plot of fitted values vs residuals with red dashed zero line and a blue trend curve highlighting non-linear pattern. Subplot 2: 25-bin histogram of residuals (density normalized) with a fitted parametric normal curve overlay.
- **Constraints:**
  - Overlay the normal curve using `scipy.stats.norm.pdf` or calculate manually via $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$.
  - Ensure the right histogram has `density=True`.
- **What You Need to Determine:**
  1. Does this regression model satisfy the Gauss-Markov assumptions of homoscedasticity (constant variance) and linearity?
  2. In which direction (positive or negative) are the residuals skewed, or are they heavy-tailed?
  3. What transformation on the target variable (e.g., $\log(y)$, $\sqrt{y}$) would you recommend based on the residual fan shape?

---

### Problem 48: Learning Rate Range Test Curve (Leslie Smith LR Finder)

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  np.random.seed(42)
  # Learning rate swept exponentially from 1e-6 to 1e-1
  lrs = np.logspace(-6, -1, 120)

  # Loss starts flat, decreases, reaches minimum, then explodes
  loss = (2.2 - 0.7 / (1 + np.exp(-1.5 * (np.log10(lrs) + 3.5))))
  loss[lrs > 1e-2] += 0.8 * (np.log10(lrs[lrs > 1e-2]) + 2)**3
  loss += np.random.normal(0, 0.015, len(lrs))
  # Smooth the noisy loss curve slightly
  loss_smooth = np.convolve(loss, np.ones(5)/5, mode='same')
  ```
- **Task:** Plot the learning rate finder curve with the learning rate on a logarithmic x-axis. Identify and visually annotate two critical coordinates: (1) the point of maximum negative gradient (steepest descent / highest learning progress), and (2) the minimum loss point before divergence.
- **Expected Output:** A semi-logarithmic plot (`ax.set_xscale('log')`) with learning rate on the x-axis and training loss on the y-axis. The smoothed curve plotted in royal blue. Two distinct vertical dotted lines and scatter points: one green marker at the steepest slope and one red marker at the loss nadir with informative annotation text boxes.
- **Constraints:**
  - Calculate numerical gradients using `np.gradient(loss_smooth, np.log10(lrs))`.
  - Ensure the y-axis limits trim exploding loss values above $2.5$ so the active descent region remains clearly visible.
- **What You Need to Determine:**
  1. According to the Leslie Smith 1cycle policy methodology, what base learning rate should be selected based on these two points?
  2. At approximately what learning rate order of magnitude does gradient explosion begin?
  3. Why is the steepest descent point preferred over the absolute minimum loss point when setting the peak learning rate?

---

### Problem 49: Latent Space Embedding Projection (t-SNE/UMAP) with Cluster Centroids

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  np.random.seed(42)
  # 3 latent clusters representing GenAI document embeddings (Finance, Tech, Medical)
  n_per_cluster = 120

  c1 = np.random.multivariate_normal([2.5, 3.0], [[0.6, 0.2], [0.2, 0.5]], n_per_cluster)
  c2 = np.random.multivariate_normal([-3.0, 1.5], [[0.8, -0.3], [-0.3, 0.7]], n_per_cluster)
  c3 = np.random.multivariate_normal([0.5, -3.5], [[0.5, 0.1], [0.1, 0.6]], n_per_cluster)

  df_embeddings = pd.DataFrame(
      np.vstack([c1, c2, c3]),
      columns=["dim_1", "dim_2"]
  )
  df_embeddings["topic"] = ["Finance"] * n_per_cluster + ["Technology"] * n_per_cluster + ["Healthcare"] * n_per_cluster
  ```
- **Task:** Create a scatter plot visualizing the 2D projected embedding space. Render each document cluster with distinct colors and translucent markers (`alpha=0.6`). Compute and plot the centroid of each topic cluster as a large, bold marker with a black outline, and annotate the centroid with the topic label.
- **Expected Output:** A clean scatter plot without distracting border spines (`ax.spines['top'].set_visible(False)`). Points grouped by topic with a curated palette. Centroids marked by prominent star or diamond markers (`s=180`) with high-contrast text labels. A well-formatted legend placed outside the plot canvas.
- **Constraints:**
  - Centroids must be calculated dynamically using `df_embeddings.groupby('topic').mean()`.
  - Do not overlap centroid text labels directly on top of the centroid marker; use an offset (`xytext=(8, 8)`).
- **What You Need to Determine:**
  1. Which two latent clusters have the shortest Euclidean distance between their respective centroids?
  2. Are there any visible border points that suggest semantic overlap between Healthcare and Finance documents?
  3. If a new incoming query embedding lands at coordinate $(0.0, 0.0)$, which topic cluster centroid is it closest to?

---

### Problem 50: Class Imbalance Distribution Before and After Resampling (SMOTE)

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import pandas as pd

  classes = ["Normal", "DDoS", "PortScan", "Botnet", "Infiltration"]
  raw_counts = [95000, 12000, 4500, 800, 120]
  # After hybrid SMOTE + undersampling
  resampled_counts = [25000, 25000, 20000, 15000, 10000]

  df_balance = pd.DataFrame({
      "Class": classes,
      "Before_Resampling": raw_counts,
      "After_Resampling": resampled_counts
  })
  ```
- **Task:** An AI security engineer needs to illustrate the effect of synthetic data generation and majority undersampling to stakeholders. Plot a side-by-side grouped bar chart comparing class frequencies before and after resampling. Because of the extreme skew in raw counts, format the y-axis with a logarithmic scale and display human-readable count labels on top of each bar (e.g., `"95k"`, `"120"`).
- **Expected Output:** A grouped bar chart with two bars per class ("Original" vs. "Resampled"). The y-axis uses `ax.set_yscale('log')`. Each bar has an explicit text label indicating its exact sample count formatted cleanly above the bar.
- **Constraints:**
  - Manually compute bar offsets ($x - \text{width}/2$ and $x + \text{width}/2$) with bar width $0.35$.
  - Use `ax.yaxis.set_major_formatter()` or standard scientific tick formatting to prevent awkward log tick exponents.
- **What You Need to Determine:**
  1. By what multiplicative factor was the "Infiltration" minority class expanded during resampling?
  2. By what percentage was the "Normal" traffic majority class reduced?
  3. Why is a linear scale visually incapable of representing this dataset effectively?

---

### Problem 51: Calibration Reliability Curve (ECE — Expected Calibration Error)

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  # 10 confidence bins from 0.0 to 1.0
  bin_centers = np.linspace(0.05, 0.95, 10)
  # Model predicted confidence vs actual empirical accuracy
  # Under-confident at low end, over-confident at high end
  empirical_acc = np.array([0.12, 0.18, 0.28, 0.35, 0.44, 0.52, 0.59, 0.65, 0.72, 0.79])
  sample_counts = np.array([50, 85, 120, 200, 310, 450, 380, 290, 180, 75])
  ```
- **Task:** Construct a two-panel Model Calibration diagnostic figure. The top panel must display the Reliability Curve (Mean Predicted Confidence vs. Fraction of Positives) along with the diagonal perfect-calibration dashed line. The bottom panel must show a confidence distribution histogram (binned sample counts) showing how predictions are distributed across the confidence range.
- **Expected Output:** A two-row vertically stacked plot (`figsize=(7, 8)`, with height ratios $2:1$). Top plot shows empirical accuracy connected by dots and lines, with shaded gaps between the empirical curve and the diagonal representing calibration error. Bottom plot shows bar chart of sample counts per bin.
- **Constraints:**
  - Both plots must share the x-axis (`sharex=True`).
  - Compute the weighted Expected Calibration Error: $\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|$ and display it in a text box inside the top plot.
- **What You Need to Determine:**
  1. Is this model systematically over-confident or under-confident for high-probability predictions ($> 0.7$)?
  2. What is the calculated numerical Expected Calibration Error (ECE) for this model?
  3. If this model predicts a cancer diagnosis with $90\%$ confidence, what is the actual observed probability according to this calibration curve?

---

### Problem 52: Hyperparameter Grid Search Heatmap with Best Candidate Callout

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
  max_depths = [3, 5, 7, 9, 12]

  # Validation F1-scores across the grid
  f1_matrix = np.array([
      [0.68, 0.72, 0.74, 0.71, 0.65],
      [0.73, 0.79, 0.82, 0.78, 0.70],
      [0.76, 0.84, 0.89, 0.83, 0.72],
      [0.74, 0.81, 0.85, 0.79, 0.69],
      [0.69, 0.75, 0.78, 0.72, 0.61]
  ])
  ```
- **Task:** Build a hyperparameter optimization matrix heatmap showing Validation $F_1$ Score across varying combinations of `learning_rate` (y-axis) and `max_depth` (x-axis). Annotate each cell with its $F_1$ score, highlight the optimal cell with a glowing or bold colored boundary rectangle, and place an annotated callout pointing to the best combination.
- **Expected Output:** A 2D heatmap using `ax.imshow()`. X-axis ticks labeled with max depth integers; y-axis ticks labeled with learning rates. Cell values formatted as `"{:.2f}"`. A `matplotlib.patches.Rectangle` drawn around the maximum score cell with an annotation arrow indicating the hyperparameter configuration.
- **Constraints:**
  - Identify the coordinate of the maximum $F_1$ score programmatically using `np.unravel_index(np.argmax(f1_matrix), f1_matrix.shape)`.
  - Draw the highlight bounding box using `patches.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor='gold', lw=3)`.
- **What You Need to Determine:**
  1. What is the optimal pair of hyperparameters and its corresponding validation $F_1$ score?
  2. What happens to the validation score when `max_depth` is increased beyond 7 at a learning rate of $0.01$?
  3. Is model performance more sensitive to changes in `max_depth` or changes in `learning_rate`?

---

### Problem 53: Cumulative Gains and Lift Charts for Model Targeting Efficiency

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  deciles = np.arange(10, 110, 10)  # Top 10%, 20%, ..., 100% of customers scored by model
  # Cumulative percentage of total responders captured
  cumulative_gains = np.array([38, 58, 72, 82, 89, 93, 96, 98, 99.5, 100.0])
  # Lift factor = cumulative_gains / deciles
  lift = cumulative_gains / deciles
  ```
- **Task:** In direct response targeting, marketing teams use Cumulative Gains and Lift charts to evaluate how much better a predictive propensity model is than random selection. Build a $1 \times 2$ figure: Left plot showing the Cumulative Gains curve against the diagonal random baseline; Right plot showing the Lift curve with a horizontal line at $\text{Lift} = 1.0$.
- **Expected Output:** Two side-by-side plots with deciles ($10\%$ to $100\%$) on both x-axes. Left plot has y-axis from $0$ to $105\%$, showing rapid ascent toward $100\%$. Right plot shows monotonically decreasing Lift from $\approx 3.8$ down to $1.0$. Shaded region between the gains curve and the diagonal baseline.
- **Constraints:**
  - Format the y-axis of the left plot with percentage strings (`PercentFormatter`).
  - Annotate the "Elbow Point" or top $20\%$ decile on both plots.
- **What You Need to Determine:**
  1. If marketing only has budget to contact the top $20\%$ of customer leads, what percentage of all potential responders will they capture using this model?
  2. What is the Lift factor at the top $10\%$ decile, and what does this number physically mean to an executive?
  3. Beyond which decile does the marginal gain per additional $10\%$ contacted drop below $5\%$?

---

### Problem 54: Normal Q-Q (Quantile-Quantile) Plot for Regression Error Assumption Testing

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np
  from scipy import stats

  np.random.seed(12)
  # Standardized residuals from an OLS model with heavy right tail (Student-t with df=3)
  residuals = stats.t.rvs(df=3, loc=0, scale=1, size=200)
  ```
- **Task:** Construct a theoretical normal Quantile-Quantile (Q-Q) plot from scratch using Matplotlib to assess whether regression error terms conform to a Gaussian distribution. Compute sample quantiles against theoretical normal distribution quantiles, plot them as scatter points, and overlay the $45^\circ$ reference line passing through the first and third quartiles.
- **Expected Output:** A square scatter plot with "Theoretical Quantiles" on the x-axis and "Sample Residual Quantiles" on the y-axis. A solid red reference line extending through the plot. Point deviations at extreme quantiles ($|x| > 2$) clearly visible.
- **Constraints:**
  - Do not use `statsmodels.api.qqplot` or `scipy.stats.probplot`. Compute theoretical quantiles directly using `scipy.stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)` and sorted sample residuals.
  - Draw the robust quartile fit line through $(Q_1^{\text{theory}}, Q_1^{\text{sample}})$ and $(Q_3^{\text{theory}}, Q_3^{\text{sample}})$.
- **What You Need to Determine:**
  1. Do the residuals conform to a normal distribution, or do they exhibit fat tails (leptokurtic distribution)?
  2. How do the extreme quantiles on both the positive and negative ends deviate from the straight theoretical line?
  3. What is the implication of this Q-Q plot pattern for the validity of p-values and confidence intervals derived from this regression model?

---

### Problem 55: Gradient Norm Tracking Across Deep Transformer Layers

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  layers = [f"L{i}" for i in range(1, 13)]  # 12 Transformer encoder layers
  # Average gradient L2 norm at different training checkpoints (Steps 100, 1000, 5000, 10000)
  grad_norms = {
      "Step 100":   [14.2, 11.5, 8.9, 6.2, 4.1, 2.8, 1.9, 1.2, 0.8, 0.5, 0.3, 0.15],
      "Step 1000":  [8.5,  7.2,  6.1, 5.0, 4.2, 3.8, 3.1, 2.5, 2.0, 1.8, 1.5, 1.2],
      "Step 5000":  [3.2,  3.1,  2.9, 3.0, 2.8, 2.7, 2.8, 2.6, 2.5, 2.7, 2.6, 2.5],
      "Step 10000": [0.8,  0.7,  0.8, 0.7, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.5, 0.5]
  }
  df_grads = pd.DataFrame(grad_norms, index=layers)
  ```
- **Task:** In deep neural networks, monitoring gradient norms across layers is critical to detect vanishing or exploding gradients. Plot layer-wise gradient norms across the four training steps as a multi-line chart. Apply a logarithmic y-scale, add distinct markers for each line, and highlight the vanishing gradient anomaly observed in early training (Step 100) using a shaded callout or arrow.
- **Expected Output:** A line chart with Transformer Layer (L1 to L12) on the x-axis and Gradient L2 Norm on a log-scaled y-axis. Step 100 shows a steep drop from 14.2 down to 0.15. An annotated region points out "Vanishing Gradient in Deep Layers prior to Warmup".
- **Constraints:**
  - Set y-axis to logarithmic scale (`ax.set_yscale('log')`).
  - Include major and minor grid lines to make order-of-magnitude comparisons legible.
- **What You Need to Determine:**
  1. Between Step 100 and Step 5000, how did the gradient flow across layers change in terms of stability?
  2. By what factor did the gradient norm decay from Layer 1 to Layer 12 during Step 100?
  3. What architectural modern techniques (e.g. Pre-LayerNorm, residual connections, learning rate warmup) prevent the Step 100 gradient degradation?

---

### Problem 56: Precision-Recall Trade-off vs. Decision Threshold Sweep

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  thresholds = np.linspace(0.05, 0.95, 50)
  # Precision monotonically rises with threshold, Recall monotonically falls
  precision = 0.3 + 0.68 / (1 + np.exp(-6 * (thresholds - 0.45)))
  recall = 0.98 - 0.75 / (1 + np.exp(-7 * (thresholds - 0.55)))
  f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

  df_thresh = pd.DataFrame({"threshold": thresholds, "precision": precision, "recall": recall, "f1": f1})
  ```
- **Task:** Product managers and engineers must pick a classification probability threshold based on business costs. Plot Precision, Recall, and $F_1$ score as three curves against decision threshold ($[0.0, 1.0]$). Locate and annotate the intersection point where Precision equals Recall, and the maximum $F_1$ threshold.
- **Expected Output:** A single plot with threshold on the x-axis and metric score ($[0, 1]$) on the y-axis. Precision in blue, Recall in orange, and $F_1$ in green. Two distinct vertical dotted lines marking: (1) the Precision-Recall equilibrium point, and (2) the optimal $F_1$ threshold.
- **Constraints:**
  - Find the intersection point numerically by minimizing $|\text{precision} - \text{recall}|$.
  - Shaded band marking the "High-Precision Regime" ($threshold \ge 0.75$).
- **What You Need to Determine:**
  1. What is the threshold value where precision and recall are balanced equally?
  2. Does the maximum $F_1$ threshold coincide exactly with the Precision-Recall intersection?
  3. If false positives cost \$10 and false negatives cost \$100, should the operating threshold be set to the left or right of $0.5$?

---

### Problem 57: LLM Benchmark Capability Spider / Radar Chart Comparison

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  categories = ["Reasoning (MMLU)", "Coding (HumanEval)", "Mathematics (GSM8K)", 
                "Agentic Tool Use", "Safety & Alignment", "Instruction Following"]
  # Scores out of 100
  model_gpt =   [88, 85, 92, 84, 95, 91]
  model_claude = [90, 89, 88, 86, 92, 94]
  model_oss =    [74, 76, 70, 68, 82, 80]
  ```
- **Task:** AI evaluation teams often use polar radar charts to benchmark Large Language Models across diverse evaluation axes. Construct an overlapping polar radar chart comparing the three LLMs across all 6 benchmark categories, with closed polygons, subtle alpha fills, and radial grid markings.
- **Expected Output:** A circular polar plot (`subplot_kw=dict(polar=True)`). The 6 categories are distributed evenly around $360^\circ$ ($2\pi$). Each model is represented by an enclosed line with semi-transparent fill (`alpha=0.15`). Radial grid rings run from $0$ to $100$ with step 20.
- **Constraints:**
  - Remember to append the first value to the end of each score array and angle array (`np.concatenate([values, [values[0]]])`) to close the polygon loop cleanly.
  - Set category labels at each angle using `ax.set_thetagrids(np.degrees(angles[:-1]), categories)`.
- **What You Need to Determine:**
  1. In which specific capability does Model Claude hold the clearest performance lead over Model GPT?
  2. Which benchmark represents the largest competitive deficit for the open-source model compared to proprietary models?
  3. Why are radar charts sometimes criticized in visualization literature despite their popularity in LLM benchmark reporting?

---

### Problem 58: Data Drift / Covariate Shift Detection (Training vs. Production Distributions)

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  np.random.seed(99)
  # Training baseline feature distribution (e.g. user session duration in minutes)
  baseline_data = np.random.gamma(shape=3.0, scale=2.5, size=1500)
  # Production data 6 months later: distribution has shifted (covariate shift)
  prod_data = np.random.gamma(shape=4.2, scale=3.1, size=1500)
  ```
- **Task:** An MLOps monitoring system flags potential feature drift on a key input variable. Plot the baseline training distribution and current production distribution as overlapping normalized step histograms with semi-transparent fills. Compute and display the Population Stability Index (PSI) or Kolmogorov-Smirnov statistic in an annotated badge, and draw vertical dashed lines marking the respective distribution means.
- **Expected Output:** A clean density distribution plot with identical bin edges for both distributions. Baseline shown in steel blue, production shown in vibrant amber. Vertical lines indicate the mean of each distribution. A stylized statistics card inside the plot canvas displays the two-sample KS-statistic and p-value.
- **Constraints:**
  - Calculate the KS-statistic using `scipy.stats.ks_2samp(baseline_data, prod_data)`.
  - Ensure bins are computed across the unified min and max of both arrays (`bins = np.linspace(min_val, max_val, 40)`).
- **What You Need to Determine:**
  1. Did the production feature distribution shift toward higher or lower session durations?
  2. Does the KS-test p-value indicate a statistically significant covariate shift at $\alpha = 0.01$?
  3. What corrective actions should the ML engineering team take if this input feature has high importance in the deployed model?

---

### Problem 59: Loss Function Penalty Comparison — MSE vs. MAE vs. Huber vs. Log-Cosh

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np

  # Prediction error: e = y_true - y_pred
  errors = np.linspace(-3.5, 3.5, 300)

  mse = errors ** 2
  mae = np.abs(errors)
  # Huber loss with delta = 1.0
  delta = 1.0
  huber = np.where(np.abs(errors) <= delta, 0.5 * errors**2, delta * (np.abs(errors) - 0.5 * delta))
  # Log-cosh loss
  log_cosh = np.log(np.cosh(errors))
  ```
- **Task:** To understand why certain loss functions are robust to outliers, create a comparative visualization of MSE, MAE, Huber Loss ($\delta = 1.0$), and Log-Cosh loss plotted against error $e \in [-3.5, 3.5]$. Highlight the transition point where Huber shifts from quadratic to linear penalty with vertical lines and annotations.
- **Expected Output:** A plot showing all 4 loss curves on a single axis. MSE rises steeply as a parabola; MAE forms a sharp V-shape; Huber transitions smoothly at $e = \pm 1.0$; Log-Cosh behaves like smooth MAE. Shaded region highlighting $|e| \le 1.0$ as the "Quadratic Region for Huber".
- **Constraints:**
  - Bounded y-limits between $0$ and $9$ to keep the scale focused on the comparative curvature.
  - Draw vertical lines at $e = -1.0$ and $e = 1.0$ indicating Huber transition boundaries.
- **What You Need to Determine:**
  1. For an outlier with an error of $e = 3.0$, how many times larger is the MSE penalty compared to the MAE penalty?
  2. Why does Huber loss provide smoother gradient optimization near zero than pure MAE?
  3. Why is Log-Cosh twice differentiable everywhere while Huber loss is only once differentiable at $\pm \delta$?

---

### Problem 60: Time Series Walk-Forward Cross-Validation Folds Schema

- **Difficulty:** Level 3 — ML/AI-Oriented
- **Dataset:**
  ```python
  import numpy as np
  import pandas as pd

  # 5 walk-forward rolling window CV splits for sequential financial data
  n_folds = 5
  total_periods = 12  # 12 quarters
  splits = [
      {"fold": 1, "train": [0, 1, 2, 3], "val": [4, 5]},
      {"fold": 2, "train": [0, 1, 2, 3, 4], "val": [5, 6]},
      {"fold": 3, "train": [0, 1, 2, 3, 4, 5], "val": [6, 7]},
      {"fold": 4, "train": [0, 1, 2, 3, 4, 5, 6], "val": [7, 8]},
      {"fold": 5, "train": [0, 1, 2, 3, 4, 5, 6, 7], "val": [8, 9]},
  ]
  ```
- **Task:** Standard k-fold cross-validation causes data leakage on time-series and sequential data. Build a schematic timeline visualization using horizontal bars (`ax.barh` or `broken_barh`) representing the 5 Walk-Forward (expanding window) CV folds. Clearly differentiate Training blocks, Validation blocks, and Unused Future Data blocks with distinct colors.
- **Expected Output:** A Gantt-style cross-validation diagram. The y-axis shows Folds 1 to 5. The x-axis shows quarters Q1 through Q12. Training periods are shaded in deep blue, validation periods in vibrant orange, and unused future periods in light hatch or dashed gray. A clear legend defines the block types.
- **Constraints:**
  - Implement using either `ax.barh()` with explicit start positions or `ax.broken_barh()`.
  - The x-axis ticks must be explicitly labeled with quarterly tokens (`["Q1", "Q2", ..., "Q12"]`).
- **What You Need to Determine:**
  1. Why would standard randomized 5-fold cross-validation produce overly optimistic (leaked) performance estimates on this data?
  2. Is this specific configuration an expanding training window or a sliding (fixed-size) training window?
  3. How many quarters ahead does each validation fold evaluate model performance?\n