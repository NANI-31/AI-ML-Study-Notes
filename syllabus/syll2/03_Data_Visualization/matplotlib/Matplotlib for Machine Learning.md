# Matplotlib for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction to Matplotlib for Machine Learning

### 1.1 Why Visualization Matters in ML
- Understanding data before modeling
- Diagnosing model performance visually
- Communicating results to stakeholders
- Detecting data quality issues
- Tracking training progress

### 1.2 Role of Matplotlib in the ML Workflow
- Exploratory Data Analysis (EDA)
- Feature distribution visualization
- Correlation analysis visualization
- Model evaluation visualization
- Hyperparameter tuning analysis
- Comparing model performance

### 1.3 Installing and Setup
- Installing Matplotlib
- Jupyter Notebook / Google Colab setup
- Import conventions
  - `import matplotlib.pyplot as plt`
  - `import numpy as np`
  - `import pandas as pd`
- Inline display — `%matplotlib inline`

---

## 2. Matplotlib Fundamentals for ML

### 2.1 Figure and Axes Setup
- `fig, ax = plt.subplots(figsize=(10, 6))`
- Setting DPI for high-quality output
- Object-oriented API for ML dashboards
- `plt.tight_layout()` for clean spacing

### 2.2 Saving ML Figures
- `plt.savefig('filename.png', dpi=150, bbox_inches='tight')`
- Saving to PDF for reports
- Saving to SVG for presentations
- Using `transparent=True` for overlay figures

### 2.3 Styles for ML Plots
- `plt.style.use('ggplot')` — clean ML style
- `plt.style.use('seaborn')` — statistical style
- `plt.style.use('fivethirtyeight')` — report style
- `plt.style.use('dark_background')` — dark mode

---

## 3. Exploratory Data Analysis (EDA) Visualizations

### 3.1 Feature Distribution Plots
- Histogram for each feature — `plt.hist()`
- KDE (Kernel Density Estimation) overlay
- Multiple feature histograms in subplots
- Setting bin count by Sturges rule or Scott rule
- `density=True` — normalized histograms

### 3.2 Box Plots for Feature Analysis
- `plt.boxplot(data)` — identifying outliers
- Side-by-side box plots per feature
- Box plots grouped by class label
- Identifying skewness visually
- Flier point analysis

### 3.3 Scatter Plots for Feature Relationships
- `plt.scatter(x, y)` — feature vs feature
- Color-coding by class label — `c=labels`
- Size-coding by a feature — `s=values`
- Alpha for dense data — `alpha=0.4`
- Regression line overlay

### 3.4 Correlation Heatmap
- Computing correlation matrix — `df.corr()`
- `plt.imshow(corr_matrix, cmap='coolwarm')`
- Adding text annotations with correlation values
- `plt.colorbar()` for color scale
- Masking upper triangle for clarity

### 3.5 Pair Plots (Manual)
- Creating a grid of scatter plots
- Diagonal histograms
- Off-diagonal scatter plots colored by class
- Using `plt.subplots()` for pair plot grid

### 3.6 Class Distribution Bar Chart
- `plt.bar(classes, counts)` — class frequency
- `plt.barh()` — horizontal for many classes
- Color-coding by class
- Annotating count on each bar
- Pie chart for class proportion

---

## 4. Feature Engineering Visualizations

### 4.1 Before and After Transformation
- Histogram before and after log transformation
- Side-by-side subplots
- Skewness reduction visualization
- Box plot before and after scaling

### 4.2 Binning Visualization
- `plt.hist()` with bin edges shown
- Annotating bin ranges
- Distribution before and after binning

### 4.3 Encoding Visualization
- Bar chart of category frequencies
- One-hot encoded feature heatmap
- Label encoded distribution comparison

### 4.4 Feature Importance Visualization
- Horizontal bar chart — `plt.barh(features, importances)`
- Sorted by importance
- Color-coding by importance value
- Error bars for importance uncertainty

---

## 5. Data Preprocessing Visualizations

### 5.1 Missing Value Visualization
- Bar chart of missing counts per feature
- Heatmap of missing value patterns — `plt.imshow(df.isnull())`
- Missing percentage annotations on bars
- Comparing before and after imputation

### 5.2 Outlier Visualization
- Box plots for outlier detection
- Scatter plots with outlier highlighting
- Z-score threshold lines on distributions
- IQR boundary lines on box plots

### 5.3 Scaling Comparison
- Histogram before and after Min-Max normalization
- Histogram before and after Standardization
- Side-by-side subplots for comparison
- Violin plots before and after scaling

---

## 6. Model Evaluation Visualizations

### 6.1 Confusion Matrix
- `plt.imshow(cm, cmap='Blues')` — heatmap
- Annotating each cell with count or percentage
- Color intensity proportional to count
- Adding axis labels (Predicted, Actual)
- Normalized and unnormalized versions

### 6.2 ROC Curve
- Plotting TPR vs FPR — `plt.plot(fpr, tpr)`
- AUC annotation on plot
- Diagonal reference line (random classifier)
- Multi-class ROC — one-vs-rest curves
- Comparing multiple models on one plot

### 6.3 Precision-Recall Curve
- `plt.plot(recall, precision)` — PR curve
- AUC-PR annotation
- No-skill baseline line
- Multi-model comparison

### 6.4 Calibration Curve
- Fraction of positives vs mean predicted probability
- Perfect calibration diagonal line
- Comparing calibration of different models

### 6.5 Residual Plot (Regression)
- `plt.scatter(y_pred, residuals)` — residuals vs predictions
- Horizontal zero line
- Detecting heteroscedasticity
- Residual histogram distribution

### 6.6 Actual vs Predicted Plot (Regression)
- `plt.scatter(y_actual, y_predicted)` — scatter
- Perfect prediction diagonal line
- R² annotation
- Color-coding by error magnitude

### 6.7 Learning Curve
- Training score vs training set size
- Validation score vs training set size
- Gap between train and val (bias-variance)
- Shaded confidence intervals

---

## 7. Training and Optimization Visualizations

### 7.1 Loss Curve
- `plt.plot(epochs, train_loss)` — training loss
- `plt.plot(epochs, val_loss)` — validation loss
- Identifying overfitting and underfitting
- Annotating best epoch

### 7.2 Accuracy Curve
- Training accuracy vs epoch
- Validation accuracy vs epoch
- Convergence visualization

### 7.3 Gradient Descent Visualization
- Loss surface as a contour plot — `plt.contourf()`
- Gradient descent path on the surface
- Learning rate comparison paths
- Convergence speed visualization

### 7.4 Learning Rate Analysis
- Loss vs learning rate plot
- Identifying optimal learning rate range
- Cyclic learning rate schedule visualization

---

## 8. Feature Selection Visualizations

### 8.1 Correlation with Target
- Bar chart of feature-target correlations
- Sorted horizontal bar chart
- Color-coding positive and negative correlations

### 8.2 Variance Analysis
- Bar chart of feature variances
- Identifying low-variance features
- Threshold line for variance threshold selection

### 8.3 Mutual Information
- Bar chart of mutual information scores
- Sorted by importance
- Comparison between MI and correlation scores

---

## 9. Hyperparameter Tuning Visualizations

### 9.1 Validation Curve
- `plt.plot(param_values, train_scores)` — training score
- `plt.plot(param_values, val_scores)` — validation score
- Identifying underfitting and overfitting range
- Shaded ± std bands

### 9.2 Grid Search Heatmap
- 2D heatmap of hyperparameter combinations
- `plt.imshow(scores_matrix)` — grid search results
- Annotating best combination
- `plt.colorbar()` for score scale

### 9.3 Bias-Variance Tradeoff
- Complexity vs training error
- Complexity vs validation error
- Sweet spot visualization

---

## 10. Classification Visualizations

### 10.1 Decision Boundary
- Meshgrid creation — `np.meshgrid()`
- `plt.contourf(xx, yy, predictions)` — boundary regions
- Scatter of training points over boundary
- Multi-class decision boundaries

### 10.2 Class Probability Visualization
- Scatter with color as predicted probability
- Colormap for probability gradient
- Calibration curve comparison

### 10.3 Feature Space Visualization
- 2D projection of feature space
- Class-coded scatter plot
- Misclassified points highlighted

---

## 11. Regression Visualizations

### 11.1 Regression Line
- Scatter of data points
- `plt.plot(x, y_pred)` — regression line overlay
- Confidence interval shading
- Polynomial regression curves

### 11.2 Residual Analysis
- Residuals vs predicted values scatter
- Histogram of residuals
- Q-Q plot of residuals
- Residuals vs feature scatter

### 11.3 Feature Contribution
- Partial dependence plot
- Feature vs target scatter per feature
- Grid of feature-target relationships

---

## 12. Clustering Visualizations

### 12.1 Cluster Scatter Plot
- `plt.scatter(x, y, c=cluster_labels)` — colored clusters
- Cluster center markers
- Voronoi regions overlay

### 12.2 Elbow Method Plot
- `plt.plot(k_values, inertia)` — inertia vs k
- Identifying elbow point
- Annotating optimal k

### 12.3 Silhouette Score Plot
- Bar chart per cluster
- Average silhouette score line
- Identifying poor clustering

### 12.4 Dendrogram (Hierarchical Clustering)
- `scipy.cluster.hierarchy.dendrogram()`
- Threshold cut line
- Coloring by cluster

---

## 13. Dimensionality Reduction Visualizations

### 13.1 PCA Visualization
- 2D scatter after PCA — `plt.scatter(pc1, pc2)`
- Color by class or cluster
- Explained variance bar chart
- Cumulative explained variance line plot

### 13.2 t-SNE Visualization
- 2D scatter of t-SNE components
- Color by class label
- Perplexity comparison subplots

### 13.3 UMAP Visualization
- 2D UMAP scatter
- Color by class label
- Cluster boundary visualization

### 13.4 Scree Plot
- Explained variance ratio per component
- `plt.bar(components, variance_ratio)`
- Cumulative variance line
- Kaiser criterion line at 1.0

---

## 14. Multi-Model Comparison Visualizations

### 14.1 Performance Comparison Bar Chart
- Model name vs accuracy/F1/AUC
- `plt.bar(model_names, scores)`
- Sorted by performance
- Error bars for cross-validation std

### 14.2 ROC Curve Comparison
- Multiple ROC curves on one plot
- Each curve labeled with AUC
- Legend showing model names

### 14.3 Box Plot for Cross-Validation
- `plt.boxplot([model_scores])` per model
- Side-by-side comparison
- Median and spread visualization

---

## 15. Statistical Visualizations for ML

### 15.1 Error Bars
- `plt.errorbar(x, y, yerr=std)` — mean ± std
- Confidence intervals
- Multiple series with error bars

### 15.2 Violin Plots for Distributions
- `plt.violinplot(data)` — distribution shape
- Side-by-side comparison per class
- Median and IQR overlay

### 15.3 Cumulative Distribution Function (CDF)
- `plt.plot(sorted_data, cumulative_probs)`
- Comparing CDFs of two features
- 95th percentile line annotation

---

## 16. Advanced ML Visualization Techniques

### 16.1 Subplots Dashboard
- `plt.subplots(nrows, ncols)` — multi-panel dashboard
- Sharing axes for aligned plots
- Mixed plot types in one figure
- Super title — `fig.suptitle()`

### 16.2 Annotation and Highlighting
- Annotating best model point
- Highlighting overfitting region with `fill_between()`
- Marking train-val split point
- Annotating threshold lines

### 16.3 Inset Zoomed Plots
- `ax.inset_axes()` — zoomed region
- Connecting lines to zoomed area
- Highlighting important region in main plot

### 16.4 Custom Colormaps for ML
- Diverging colormaps for correlation (positive/negative)
- Sequential colormaps for continuous predictions
- Discrete colormaps for class labels

---

## 17. Matplotlib with Scikit-learn

### 17.1 Plotting Scikit-learn Results
- Plotting confusion matrix from `sklearn.metrics`
- `ConfusionMatrixDisplay` visualization
- ROC curve from `sklearn.metrics.roc_curve`
- PR curve from `sklearn.metrics.precision_recall_curve`
- Learning curve from `sklearn.model_selection.learning_curve`
- Validation curve from `sklearn.model_selection.validation_curve`

### 17.2 Decision Tree Visualization
- `sklearn.tree.plot_tree()` — tree structure
- `max_depth` for readability
- Feature names and class names annotation

### 17.3 Feature Importance Plot
- `plt.barh(feature_names, importances)` — sorted bar chart
- Random Forest and XGBoost importance plots

---

## 18. Matplotlib Best Practices for ML

### 18.1 Effective ML Visualization
- Always label axes with feature names and units
- Use consistent color schemes across figures
- Include the metric value (e.g., AUC) in the title
- Use `tight_layout()` for clean multi-panel figures
- Set appropriate DPI for publication quality

### 18.2 Reproducibility
- Fix random seeds before generating plots
- Save figures programmatically after creation
- Version figures with descriptive filenames

### 18.3 Common ML Visualization Mistakes
- Using too many colors without a legend
- Overly cluttered subplots
- Missing axis labels
- Not normalizing confusion matrix for imbalanced data
- Wrong scale (linear vs log) for loss curves

---

## 19. Matplotlib Interview Topics for ML

### 19.1 Frequently Asked Questions
- How to plot a confusion matrix?
- How to plot training and validation loss?
- How to visualize a decision boundary?
- How to create a feature importance bar chart?
- How to plot ROC and PR curves?

### 19.2 ML-Oriented Coding Problems
- Plot a confusion matrix with annotations
- Plot learning curves for a classifier
- Visualize PCA components in 2D
- Create a hyperparameter grid heatmap
- Plot ROC curves for three models on one figure

---

## 20. Real-World ML Visualization Projects

### 20.1 Beginner Projects
- EDA dashboard for a classification dataset
- Feature distribution grid of Iris dataset
- Confusion matrix for digit classification

### 20.2 Intermediate Projects
- Complete model evaluation report dashboard
- Training and validation curve monitoring
- Dimensionality reduction (PCA + t-SNE) comparison

### 20.3 Advanced Projects
- Interactive hyperparameter tuning visualization
- Multi-model performance comparison dashboard
- End-to-end ML experiment reporting pipeline

---

## 21. Matplotlib Roadmap for ML Engineers

### 21.1 Beginner Level
- Basic line, bar, scatter, histogram
- Adding labels, titles, legends
- Subplots for multi-panel EDA

### 21.2 Intermediate Level
- Confusion matrix heatmap
- ROC and PR curves
- Feature importance bar charts
- Learning and validation curves

### 21.3 Advanced Level
- Decision boundary visualization
- Hyperparameter grid heatmaps
- Dimensionality reduction plots
- Full model evaluation dashboard
