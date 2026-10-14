import os

os.makedirs("basics/3. Mathplotlib/interview_questions", exist_ok=True)

content = """# Matplotlib Coding Interview Practice — Level 1: Fundamentals

This set contains 20 practical, interview-grade coding problems focused on foundational visualization tasks in Python, Data Science, and Machine Learning engineering.

---

### Problem 1: Study Hours vs. Exam Score Correlation

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  study_hours = [1.5, 2.0, 2.5, 3.2, 3.8, 4.0, 4.5, 5.1, 5.8, 6.0, 6.5, 7.2, 8.0, 8.5, 9.0]
  exam_scores = [52, 55, 58, 65, 63, 68, 72, 75, 78, 82, 85, 88, 91, 94, 98]
  ```
- **Task:** Construct a visualization that clearly portrays the individual observation pairs and reveals whether an empirical relationship exists between time invested and test performance. Add a trendline or line of best fit.
- **Expected Output:** A scatter plot with circular markers, labeled axes (`"Study Hours"` and `"Exam Score (0-100)"`), a descriptive title, subtle gridlines, and a linear trendline overlaid across the data span.
- **Constraints:**
  - Use Matplotlib's Object-Oriented interface (`fig, ax = plt.subplots()`).
  - Set the y-axis coordinate limits strictly between $40$ and $100$.
- **What You Need to Determine:**
  1. What type of correlation (positive, negative, or zero) is apparent between study hours and exam scores?
  2. Does the relationship appear linear, exponential, or logarithmic across this range?
  3. Are there any visible outlier students who scored unusually high or low relative to their study time?

---

### Problem 2: Temperature vs. Daily Ice Cream Revenue

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  import pandas as pd

  data = {
      "avg_temp_c": [14.2, 16.5, 18.1, 20.4, 22.8, 25.1, 27.3, 29.8, 31.5, 33.2, 35.0, 36.8],
      "daily_revenue": [420, 490, 560, 680, 810, 940, 1100, 1280, 1420, 1590, 1750, 1920],
  }
  df = pd.DataFrame(data)
  ```
- **Task:** Create a visualization that plots daily sales against ambient outdoor temperature, formatting the currency axis appropriately for an executive report.
- **Expected Output:** A scatter plot with temperature on the horizontal axis and revenue on the vertical axis, featuring colored markers with edge strokes, gridlines, axis labels with physical units ($^\circ\text{C}$ and $\$$), and a currency-formatted y-axis.
- **Constraints:**
  - Do not use a categorical bar chart; treat temperature as a continuous numerical feature.
  - Save the resulting figure as `"temp_vs_revenue.png"` with $300$ DPI and tight bounding box padding.
- **What You Need to Determine:**
  1. What is the observed directional rate of change in revenue per degree Celsius increase?
  2. Would a linear regression model be appropriate for modeling this relationship, or is there diminishing marginal return at extreme temperatures?
  3. Which variable should be considered the independent (predictor) variable and which the dependent (response) variable?

---

### Problem 3: Customer Age Distribution & Skewness

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  import numpy as np

  np.random.seed(42)
  # Simulated customer ages for a mobile fintech app
  customer_ages = np.concatenate([
      np.random.normal(loc=26, scale=4, size=150),  # Young adults
      np.random.normal(loc=38, scale=6, size=80),   # Mid-career
      np.random.normal(loc=55, scale=8, size=30),   # Older cohort
  ]).astype(int)
  customer_ages = np.clip(customer_ages, 18, 75)
  ```
- **Task:** Plot the distribution of customer ages to understand user demographics. Select an appropriate binning strategy, and overlay vertical lines indicating the sample mean and median.
- **Expected Output:** A frequency histogram with black bar borders, labeled axes, a prominent title, a dashed red vertical line at the mean, and a dotted green vertical line at the median with an informative legend.
- **Constraints:**
  - Test and select a bin count between $10$ and $25$ that captures the shape without excessive sparsity.
  - Ensure the histogram bars do not blend together into an undifferentiated block.
- **What You Need to Determine:**
  1. Is the age distribution symmetric, positively skewed (right-skewed), or negatively skewed (left-skewed)?
  2. How does the relationship between the mean and median reflect the skewness of the distribution?
  3. What is the primary target demographic for this application based on the mode of the distribution?

---

### Problem 4: Deep Learning Loss Curve Across Epochs

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  epochs = list(range(1, 13))
  training_loss = [2.45, 1.82, 1.34, 0.98, 0.72, 0.54, 0.42, 0.35, 0.31, 0.28, 0.27, 0.26]
  ```
- **Task:** Plot the loss convergence trajectory of a deep neural network over 12 epochs to determine whether training has reached a point of diminishing returns.
- **Expected Output:** A line plot with distinct circular markers at each epoch coordinate, integer tick marks on the horizontal axis ($1, 2, \dots, 12$), axis labels (`"Epoch"` and `"Training Loss (Cross-Entropy)"`), and an annotation pointing to the inflection plateau.
- **Constraints:**
  - Ensure the x-axis displays exact integers rather than fractional epoch floats ($1.5, 2.5$).
  - Add horizontal and vertical gridlines with subtle opacity (`alpha=0.4`).
- **What You Need to Determine:**
  1. Around which epoch does the model begin to plateau, where further training yields negligible loss reduction?
  2. If each epoch costs $\$50$ in GPU compute, at which epoch would you recommend triggering early stopping?
  3. Does the loss curve exhibit any instability or sudden divergence spikes?

---

### Problem 5: Regional Housing Price Medians

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  zones = ["Downtown", "North Suburbs", "West Valley", "East Coast Bay", "South Hills"]
  median_prices = [850000, 520000, 430000, 920000, 610000]  # in USD
  ```
- **Task:** Compare the median home valuation across the 5 municipal zones using an appropriate categorical plot.
- **Expected Output:** A vertical or horizontal bar chart with distinct colors or uniform professional palette, numeric value annotations rendered directly adjacent to or above each bar, and currency-formatted ticks.
- **Constraints:**
  - Ensure all zone names are fully legible without overlapping one another.
  - Sort the categories in ascending or descending order of price to maximize visual scannability.
- **What You Need to Determine:**
  1. Which zone represents the most expensive residential market, and what is the absolute dollar difference between the highest and lowest zones?
  2. Why is sorting categories by value superior to plotting in arbitrary or alphabetical order for decision-makers?
  3. Would a continuous line plot connecting these 5 points be statistically appropriate? Why or why not?

---

### Problem 6: SLA Uptime Monitoring with Threshold Boundary

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  days = [f"Day {i}" for i in range(1, 15)]
  uptime_pct = [99.95, 99.98, 99.92, 99.85, 99.99, 99.91, 99.70, 99.94, 99.96, 99.89, 99.97, 99.95, 99.99, 99.93]
  ```
- **Task:** Visualize daily service availability across a two-week period and evaluate whether the system complied with an enterprise Service Level Agreement (SLA) target of $99.90\%$.
- **Expected Output:** A line chart with circular data markers, a bold horizontal dashed red reference line at $99.90\%$, labeled axes, and an annotation highlighting the most severe SLA violation day.
- **Constraints:**
  - Set the y-axis coordinate limits tightly to encompass $[99.60\%, 100.00\%]$ so small fractional deviations are visible.
  - Include an informative legend identifying the operational metric and the SLA benchmark line.
- **What You Need to Determine:**
  1. On how many days did the infrastructure breach the contractual $99.90\%$ SLA threshold?
  2. Which specific day experienced the lowest availability, and what was the magnitude of the deficit?
  3. How does scaling the y-axis from $0\%$ to $100\%$ versus $99.6\%$ to $100\%$ affect a stakeholder's perception of stability?

---

### Problem 7: E-Commerce Product Category Market Share

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  categories = ["Apparel", "Electronics", "Home & Kitchen", "Books", "Health & Beauty", "Toys & Games", "Automotive"]
  revenue_share = [28.5, 24.2, 16.8, 12.1, 9.4, 5.8, 3.2]  # in percent
  ```
- **Task:** Create two alternative visualizations to present this categorical revenue breakdown to the executive board: a Donut chart and a Horizontal Bar chart.
- **Expected Output:** A $1 \times 2$ subplot figure with the Donut chart on the left (showing percentage labels and a central circular cutout) and a sorted Horizontal Bar chart on the right (showing explicit percentage values beside each bar).
- **Constraints:**
  - Set the donut hole radius to $0.6$ using a white central circle patch or `wedgeprops`.
  - Maintain identical categorical color mapping across both panels.
- **What You Need to Determine:**
  1. Which of the two visualizations makes it easier to accurately compare the difference between "Health & Beauty" ($9.4\%$) and "Toys & Games" ($5.8\%$)?
  2. Why do data visualization experts generally advise against pie/donut charts when the number of categories exceeds $5$?
  3. What visual cue in the horizontal bar chart facilitates immediate ordinal ranking?

---

### Problem 8: Marketing Campaign Daily Active Users (DAU)

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  import pandas as pd

  dates = pd.date_range("2026-03-01", periods=14, freq="D")
  dau = [12500, 12800, 13100, 13400, 15200, 18600, 17900, 13600, 13900, 14100, 14300, 16100, 19800, 18900]
  df = pd.DataFrame({"date": dates, "dau": dau})
  ```
- **Task:** Plot daily active user traffic across the 14-day window. Format the date axis cleanly and shade the weekend periods (Saturday and Sunday) to contextualize user engagement surges.
- **Expected Output:** A line chart with date-formatted tick labels (`"%b %d"`), slanted labels to prevent collision, and vertical shaded spans (`ax.axvspan`) highlighting both weekends (March 6–7 and March 13–14).
- **Constraints:**
  - Use `fig.autofmt_xdate(rotation=30, ha='right')` for date label formatting.
  - Set weekend span opacity to `alpha=0.15` using a neutral gray or light blue fill.
- **What You Need to Determine:**
  1. Does the app exhibit higher user engagement on weekdays or on weekends?
  2. What is the average percentage increase in DAU during weekends compared to weekday baselines?
  3. Is there an underlying upward trend across the campaign, independent of the weekly cyclicality?

---

### Problem 9: Bug Fix — Resolving Misaligned Rotated Tick Labels

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  models = ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor", "SupportVectorRegressor", "MultiLayerPerceptron"]
  rmse = [14.2, 9.8, 8.4, 11.6, 10.1]
  ```
- **Task:** An intern wrote the following code to plot model RMSE scores, but the x-axis labels are overlapping, centered under the wrong tick marks, and truncated off the bottom of the exported image:
  ```python
  # Buggy code provided by junior developer:
  plt.figure()
  plt.bar(models, rmse)
  plt.xticks(rotation=45)
  plt.title("Model RMSE Comparison")
  plt.savefig("rmse_buggy.png")
  ```
  Identify the technical defects in this code and rewrite it using modern Matplotlib Object-Oriented best practices.
- **Expected Output:** A clean, publication-ready bar chart where every model label is rotated $45^\circ$, right-aligned (`ha='right'`) directly beneath its respective bar, with zero label truncation in the saved file.
- **Constraints:**
  - Convert the stateful `plt` calls to the explicit `fig, ax = plt.subplots()` paradigm.
  - Apply `bbox_inches='tight'` or `plt.tight_layout()` during figure export.
- **What You Need to Determine:**
  1. Why does rotating tick labels without specifying `ha='right'` cause labels to visually associate with neighboring bars?
  2. Why was the bottom of the label text clipped off in `rmse_buggy.png`?
  3. Which model architecture achieved the best performance (lowest RMSE)?

---

### Problem 10: Advertising Spend vs. Conversion Rate with High-Leverage Outlier

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  # Ad spend in thousands of USD vs Lead Conversion Rate in %
  ad_spend = [1.2, 1.8, 2.4, 3.0, 3.5, 4.1, 4.8, 5.2, 5.9, 6.5, 7.1, 7.8, 14.5]
  conv_rate = [2.1, 2.5, 2.9, 3.4, 3.8, 4.2, 4.7, 5.0, 5.5, 5.9, 6.3, 6.8, 2.4]
  ```
- **Task:** Visualize this marketing relationship and annotate the anomalous data point to alert the growth marketing team.
- **Expected Output:** A scatter plot with an annotated arrow pointing to the coordinate $(14.5, 2.4)$ with text `"Anomaly: High Spend, Low Conversion"`, plus labeled axes and grid.
- **Constraints:**
  - Color the anomalous point red while keeping standard points blue.
  - Use `ax.annotate()` with explicit arrow styling (`arrowprops=dict(arrowstyle="->", color="black")`).
- **What You Need to Determine:**
  1. What linear relationship holds for ad spend between $\$1,000$ and $\$8,000$?
  2. What might have occurred operationally to cause the outlier at $\$14,500$ ad spend with only $2.4\%$ conversion?
  3. How would fitting an Ordinary Least Squares (OLS) linear regression model without removing this high-leverage point distort the regression slope?

---

### Problem 11: Employee Experience vs. Annual Compensation

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  years_exp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
  salary = [55000, 62000, 70000, 79000, 89000, 100000, 112000, 123000, 133000, 142000, 156000, 167000, 174000, 179000, 182000]
  ```
- **Task:** Determine whether salary progression scales linearly with years of experience or exhibits diminishing marginal returns over a 20-year career horizon.
- **Expected Output:** A combined scatter and line plot with currency-formatted y-ticks (displaying values as `"$60k"`, `"$100k"`, etc.), labeled axes, and gridlines.
- **Constraints:**
  - Format the y-axis tick labels using `FuncFormatter` or `StrMethodFormatter` to avoid rendering large numbers with trailing zeros.
- **What You Need to Determine:**
  1. During which career phase (years 1–7 vs. years 14–20) is salary growth steepest?
  2. Does a polynomial or logarithmic curve describe the trajectory better than a strictly straight line?
  3. What is the approximate annual salary increment during the first 5 years versus the final 5 years?

---

### Problem 12: Class Imbalance in Fraud Detection Dataset

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  classes = ["Legitimate Transaction", "Suspicious Activity", "Confirmed Fraud"]
  transaction_counts = [485200, 14300, 1200]
  ```
- **Task:** Create a visualization that communicates the severe class imbalance present in this financial fraud dataset for an ML model architecture review.
- **Expected Output:** A horizontal bar chart plotted on a logarithmic scale (`ax.set_xscale('log')`), with raw transaction count annotations placed beside each bar.
- **Constraints:**
  - Why is a linear scale bar chart misleading or unusable for this dataset? Verify by comparing linear vs. logarithmic scaling.
  - Set horizontal axis limits to cleanly accommodate the data from $10^2$ to $10^6$.
- **What You Need to Determine:**
  1. What is the exact percentage representation of "Confirmed Fraud" relative to the total transaction volume?
  2. What visual challenge occurs if you attempt to render this on a standard linear axis?
  3. Which evaluation metrics (e.g. Accuracy vs PR-AUC) should you recommend to the team based on this distribution?

---

### Problem 13: Memory Footprint Across 5 Microservices Over Time

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  import pandas as pd
  import numpy as np

  np.random.seed(101)
  hours = list(range(24))
  services = {
      "Auth-Service": 250 + np.random.normal(0, 10, 24).cumsum(),
      "Payment-Gateway": 400 + np.random.normal(0, 15, 24).cumsum(),
      "Recommendation-Engine": 800 + np.random.normal(5, 25, 24).cumsum(),  # Memory leak!
      "Inventory-DB": 500 + np.random.normal(0, 8, 24).cumsum(),
      "Notification-Service": 150 + np.random.normal(0, 5, 24).cumsum(),
  }
  ```
- **Task:** Plot hourly memory consumption (in MB) across all 5 microservices in a single chart to identify which service is exhibiting an abnormal memory leak.
- **Expected Output:** A multi-line plot with distinct colors, distinct linestyles (`'-'`, `'--'`, `':'`), a well-positioned legend outside the plotting area or in the upper left, and labeled axes (`"Hour of Day"` and `"RAM Usage (MB)"`).
- **Constraints:**
  - Place the legend outside the axes bounding box using `bbox_to_anchor=(1.02, 1)` so it does not collide with the lines.
  - Ensure the horizontal axis displays ticks at every 2 hours ($0, 2, 4, \dots, 22$).
- **What You Need to Determine:**
  1. Which microservice demonstrates an unbounded upward slope indicative of a memory leak?
  2. By hour 23, approximately how much RAM is the leaking service consuming compared to its baseline at hour 0?
  3. Which service displays the most stable, stationary memory profile throughout the day?

---

### Problem 14: Mobile App Star Rating Distribution

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  ratings = [1, 2, 3, 4, 5]
  review_counts = [1240, 430, 890, 3120, 8950]
  ```
- **Task:** Visualize the customer review rating profile of a newly launched mobile application. Compute and annotate the weighted average rating.
- **Expected Output:** A vertical bar chart with rating scores on the horizontal axis, colored bars, review count text placed on top of each bar, and a vertical dashed line marking the calculated weighted average rating.
- **Constraints:**
  - Calculate the weighted average mathematically: $\bar{x} = \frac{\sum (r \cdot c)}{\sum c}$.
  - Ensure y-axis starts at zero with a $15\%$ headroom margin above the maximum bar height.
- **What You Need to Determine:**
  1. What is the calculated weighted average star rating for this app?
  2. Does the rating profile represent a normal distribution, a bimodal distribution, or a J-shaped/polarized distribution?
  3. Why is the arithmetic mean of ratings alone insufficient to understand user sentiment without inspecting the full bar distribution?

---

### Problem 15: Code Prediction — Visualizing Marker Styling

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  x = [1, 2, 3, 4, 5]
  y1 = [10, 20, 15, 30, 25]
  y2 = [25, 18, 28, 22, 35]
  ```
- **Task:** Consider the following Matplotlib snippet:
  ```python
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(x, y1, 'ro--', linewidth=2, markersize=8, label="Pipeline A")
  ax.plot(x, y2, 'bs:', linewidth=1.5, markersize=6, label="Pipeline B")
  ax.set_xticks(x)
  ax.legend(loc="lower right")
  ```
  Before running the code, predict every visual property of both lines. Then execute the code and verify your predictions.
- **Expected Output:** An accurate technical breakdown followed by the rendered plot.
- **Constraints:**
  - Explain the shorthand string formatting syntax `'ro--'` and `'bs:'`.
- **What You Need to Determine:**
  1. What do the individual characters `'r'`, `'o'`, and `'--'` specify in `'ro--'`?
  2. What do `'b'`, `'s'`, and `':'` specify in `'bs:'`?
  3. At coordinate $x = 3$, which pipeline has the higher value, and what marker symbol sits at that position?

---

### Problem 16: GPU Thermal Throttling Monitoring

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  minutes = list(range(0, 65, 5))
  gpu_temps = [45, 52, 61, 68, 74, 79, 83, 86, 88, 89, 87, 85, 84]  # in deg C
  ```
- **Task:** Monitor GPU temperature during an intensive LLM fine-tuning job. Highlight the thermal throttling danger zone ($> 85^\circ\text{C}$) to assess cooling hardware adequacy.
- **Expected Output:** A line plot of temperature over time, with a horizontal red line at $85^\circ\text{C}$ labeled `"Thermal Throttling Limit"`, and the region above $85^\circ\text{C}$ shaded in translucent red using `ax.axhspan(85, 100, color='red', alpha=0.15)`.
- **Constraints:**
  - Set the y-axis coordinate limits between $40^\circ\text{C}$ and $95^\circ\text{C}$.
  - Label axes with descriptive units (`"Elapsed Training Time (Minutes)"` and `"Core Temperature (°C)"`).
- **What You Need to Determine:**
  1. At what elapsed minute did the GPU first cross into the thermal throttling danger zone?
  2. For approximately how many minutes did the GPU operate above safe thermal limits?
  3. What physical operational change likely occurred between minute 45 and minute 60 to cause temperature decline (e.g. throttling frequency reduction or fan ramp-up)?

---

### Problem 17: A/B Test Conversion Rates with Error Margins

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  variants = ["Control (A)", "Variant B (Green CTA)", "Variant C (Sticky CTA)"]
  conv_rates = [3.24, 3.85, 3.52]       # Mean conversion rate in %
  conf_intervals = [0.28, 0.31, 0.25]   # 95% confidence interval margin
  ```
- **Task:** Present A/B test experimental results to marketing stakeholders, incorporating uncertainty error bars to show whether observed differences are statistically meaningful.
- **Expected Output:** A bar chart with centered vertical error bars (`ax.bar(..., yerr=conf_intervals, capsize=5)`), clean percentage y-ticks, and explicit value annotations above each bar.
- **Constraints:**
  - Include horizontal error bar caps (`capsize=5`) to cleanly demarcate confidence intervals.
  - Set y-limits from $0$ to $5.0\%$.
- **What You Need to Determine:**
  1. Does the confidence interval of Variant B overlap with the confidence interval of Control A?
  2. Based on this visual overlap, can you definitively declare Variant B a statistically significant winner at the $95\%$ confidence level without further testing?
  3. Which variant produced the narrowest confidence interval, and what does that indicate about sample size or variance?

---

### Problem 18: Customer Churn Rate by Contract Duration

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  contract_types = ["Month-to-Month", "One-Year Contract", "Two-Year Contract"]
  churn_percentages = [42.7, 11.2, 2.8]
  retention_percentages = [57.3, 88.8, 97.2]
  ```
- **Task:** Create a stacked bar chart showing the breakdown of Churned vs. Retained customers across contract categories.
- **Expected Output:** A stacked vertical bar chart where the bottom bar represents Retained customers (in green/blue) and the top bar represents Churned customers (in coral/red), summing to $100\%$ for each contract type.
- **Constraints:**
  - Use the `bottom` parameter of `ax.bar()` to stack Churned counts on top of Retained counts.
  - Add percentage labels centered inside each segment.
- **What You Need to Determine:**
  1. What is the churn rate ratio between month-to-month customers and two-year contract customers?
  2. What strategic business recommendation would you make to product leadership based on this visualization?
  3. Why is a stacked bar chart preferred here over two separate side-by-side bar charts?

---

### Problem 19: Code Fix — Overwriting vs. Overlaying Lines

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  x = [1, 2, 3, 4, 5]
  model_v1_acc = [0.65, 0.72, 0.78, 0.81, 0.83]
  model_v2_acc = [0.68, 0.76, 0.84, 0.89, 0.92]
  ```
- **Task:** A data science student attempted to compare Model V1 and Model V2 accuracy across 5 epochs with this code:
  ```python
  # Faulty code:
  plt.plot(x, model_v1_acc, label="V1")
  plt.title("Model V1")
  plt.show()

  plt.plot(x, model_v2_acc, label="V2")
  plt.title("Model V2")
  plt.show()
  ```
  Explain why this code outputs two separate, isolated figures that are difficult to compare, and rewrite it so both curves appear on the same shared axes with distinct styling, a title, gridlines, and a legend.
- **Expected Output:** A single unified plot containing both curves, square markers on V1, triangle markers on V2, a legend in the lower right, and labeled axes.
- **Constraints:**
  - Use `fig, ax = plt.subplots()` to enforce proper figure management.
- **What You Need to Determine:**
  1. What does `plt.show()` do to the active figure canvas in the `pyplot` state machine?
  2. Why is overlaying both curves on a single axis vastly superior for comparative evaluation?
  3. In which epoch did Model V2 surpass Model V1's final maximum performance?

---

### Problem 20: Network Packet Latency — Linear vs. Logarithmic Histogram

- **Difficulty:** Level 1 — Fundamentals
- **Dataset:**
  ```python
  import numpy as np

  np.random.seed(42)
  # 95% normal traffic between 10-50ms, 5% extreme lag spikes up to 2000ms
  latency_ms = np.concatenate([
      np.random.normal(loc=25, scale=5, size=950),
      np.random.exponential(scale=300, size=50) + 100
  ])
  ```
- **Task:** Plot the latency distribution in a $1 \times 2$ subplot layout comparing a standard linear frequency histogram with a logarithmic frequency histogram (`ax.set_yscale('log')`).
- **Expected Output:** Panel 1 showing the linear histogram where tail latency spikes are completely invisible; Panel 2 showing the log-frequency histogram where the long-tail latency events spanning $500\text{--}2000\text{ms}$ are clearly visible.
- **Constraints:**
  - Use identical bin edges across both subplots (e.g. `bins=50`).
  - Label axes with appropriate units (`"Latency (ms)"` and `"Frequency / Count"`).
- **What You Need to Determine:**
  1. Why are long-tail extreme events ($>500\text{ms}$) completely invisible to human observers in the standard linear histogram?
  2. How does setting `ax.set_yscale('log')` change the visual representation of small counts?
  3. For network reliability and SLA compliance, why is inspecting tail latency distributions crucial in production systems?

---
"""

with open("basics/3. Mathplotlib/interview_questions/level_01_fundamentals.md", "w", encoding="utf-8") as f:
    f.write(content.strip() + "\n")
print("Wrote level_01_fundamentals.md")
