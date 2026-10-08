# Seaborn for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction to Seaborn for Machine Learning

### 1.1 Why Seaborn is Valuable in ML
- Faster and cleaner statistical plots than raw Matplotlib
- Built-in statistical estimation (mean, CI, regression)
- Native Pandas DataFrame support — no manual array conversion
- Ideal for EDA, feature analysis, and model evaluation
- Beautiful defaults that make results presentable

### 1.2 Role of Seaborn in the ML Workflow
- Exploratory Data Analysis (EDA)
- Feature distribution and relationship analysis
- Class imbalance visualization
- Correlation and feature selection analysis
- Model performance and evaluation visualization
- Hyperparameter tuning result visualization

### 1.3 Installing and Setup
- Installing Seaborn
- Jupyter Notebook / Google Colab setup
- Import conventions
  - `import seaborn as sns`
  - `import matplotlib.pyplot as plt`
  - `import pandas as pd`
  - `import numpy as np`
- Setting theme for ML notebooks
  - `sns.set_theme(style='whitegrid', context='notebook')`

---

## 2. Seaborn Fundamentals for ML

### 2.1 Working with ML Datasets
- `sns.load_dataset('titanic')` — classification dataset
- `sns.load_dataset('iris')` — multi-class classification
- `sns.load_dataset('tips')` — regression dataset
- `sns.load_dataset('penguins')` — clustering dataset
- Loading custom CSV datasets with Pandas and passing to Seaborn
- Long-format vs wide-format preference

### 2.2 Themes and Context for ML Reports
- `sns.set_style('whitegrid')` — clean ML notebook style
- `sns.set_context('talk')` — for presentation of results
- `sns.set_context('paper')` — for publication figures
- `sns.set_palette('colorblind')` — accessible palettes

### 2.3 Figure and Axes Integration
- `fig, ax = plt.subplots(figsize=(10, 6))`
- Passing `ax=ax` to Seaborn axes-level functions
- Using `plt.tight_layout()` after Seaborn plots
- Saving ML figures — `plt.savefig('output.png', dpi=150, bbox_inches='tight')`

---

## 3. Exploratory Data Analysis (EDA) for ML

### 3.1 Feature Distribution Analysis
- `sns.histplot(data=df, x='feature', kde=True)` — distribution + KDE
- `bins` — appropriate bin count
- `hue='target'` — distribution overlaid by class
- `multiple='stack'` — stacked distributions
- `stat='density'` — normalized for comparison
- Per-feature distribution grid using `FacetGrid`

### 3.2 KDE Plots for Feature Overlap
- `sns.kdeplot(data=df, x='feature', hue='target')` — class KDE overlay
- `fill=True` — filled KDE curves
- `bw_adjust` — smoothing control
- Identifying separability between classes
- Multi-feature KDE comparison subplots

### 3.3 ECDF for ML Analysis
- `sns.ecdfplot(data=df, x='feature', hue='target')` — cumulative distribution
- Comparing class cumulative distributions
- Identifying distribution shift across classes

### 3.4 Pair Plot for Feature Relationships
- `sns.pairplot(data=df, hue='target')` — full scatter matrix
- `vars=['f1', 'f2', 'f3']` — selected features only
- `diag_kind='kde'` — KDE on diagonal
- `kind='reg'` — regression lines in scatter panels
- Identifying linearly separable features
- Identifying correlated features

### 3.5 Joint Plot for Two Features
- `sns.jointplot(data=df, x='f1', y='f2', kind='scatter', hue='target')`
- `kind='kde'` — 2D density plot
- `kind='hex'` — hexbin for dense data
- `kind='reg'` — regression line
- Marginal distributions for feature understanding

### 3.6 Class Distribution Analysis
- `sns.countplot(data=df, x='target')` — class frequency
- `stat='proportion'` — class proportions
- `hue` — nested class breakdown
- `order=df['target'].value_counts().index` — sorted order
- Annotating count values on bars

---

## 4. Correlation and Feature Selection Visualizations

### 4.1 Correlation Heatmap
- `sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')` — correlation matrix
- `vmin=-1, vmax=1, center=0` — proper diverging scale
- `mask=np.triu(np.ones_like(corr))` — lower triangle only
- `linewidths=0.5` — cell borders
- `square=True` — equal cells
- Identifying multicollinearity for feature selection

### 4.2 Feature-Target Correlation
- Computing correlation of each feature with target
- `df.corr()['target'].sort_values()` — sorted correlations
- `sns.barplot(x=corr_values, y=feature_names)` — horizontal bar chart
- Color-coding positive and negative correlations

### 4.3 Cluster Map for Feature Grouping
- `sns.clustermap(df.corr(), cmap='coolwarm', figsize=(12, 12))` — clustered heatmap
- `z_score=1` — normalize per column
- Identifying groups of correlated features
- Feature redundancy detection

### 4.4 Pair Plot with Regression for Feature Analysis
- `sns.pairplot(data=df, kind='reg', hue='target')` — regression in each cell
- Identifying linear vs non-linear feature relationships

---

## 5. Outlier and Data Quality Visualization

### 5.1 Box Plot for Outlier Detection
- `sns.boxplot(data=df, x='target', y='feature')` — grouped box plots
- Identifying outliers as flier points
- `whis=1.5` — standard IQR whiskers
- `fliersize=3` — small outlier markers
- Per-feature box plots in subplots

### 5.2 Violin Plot for Distribution Shape
- `sns.violinplot(data=df, x='target', y='feature')` — violin by class
- `inner='box'` — box inside violin
- `split=True` — half violin for two classes
- Comparing class-wise distributions

### 5.3 Strip and Swarm Plots
- `sns.stripplot(data=df, x='target', y='feature', jitter=True)` — all points
- `sns.swarmplot(data=df, x='target', y='feature')` — non-overlapping
- Overlaying on box or violin — `ax=ax`
- Identifying sparse vs dense regions

---

## 6. Feature Engineering Visualizations

### 6.1 Before vs After Transformation
- KDE plots before and after log transformation
- `sns.kdeplot(data=df_before, x='feature')` vs `sns.kdeplot(data=df_after, x='feature')`
- Side-by-side `plt.subplots(1, 2)` comparison
- Skewness reduction visualization

### 6.2 Binning Visualization
- `sns.histplot(data=df, x='feature')` before binning
- `sns.countplot(data=df, x='feature_binned')` after binning
- Bin edge annotation

### 6.3 Encoding Visualization
- `sns.heatmap(pd.get_dummies(df['col']))` — one-hot encoding matrix
- `sns.countplot(data=df, x='label_encoded_col')` — label encoding distribution

### 6.4 Feature Interaction Visualization
- `sns.scatterplot(data=df, x='f1', y='f2', hue='target', size='f3')` — 4D scatter
- `sns.jointplot(data=df, x='f1*f2', y='target', kind='reg')` — interaction effect

---

## 7. Class Imbalance Visualization

### 7.1 Class Frequency
- `sns.countplot(data=df, x='target', stat='proportion')` — proportional count
- Annotating percentages on bars
- Horizontal bar chart for many classes — `y='target'`

### 7.2 Distribution by Class
- `sns.kdeplot(data=df, x='feature', hue='target', fill=True)` — overlapping distributions
- `multiple='stack'` — stacked distributions
- `sns.boxplot(data=df, x='target', y='feature')` — class-wise spread

### 7.3 Before vs After Balancing
- `sns.countplot` before SMOTE / oversampling
- `sns.countplot` after balancing
- Side-by-side subplots comparison

---

## 8. Model Evaluation Visualizations

### 8.1 Confusion Matrix Heatmap
- `sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')` — count matrix
- `sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')` — normalized
- `xticklabels=class_names`, `yticklabels=class_names` — label axes
- `linewidths=0.5` — cell borders
- `cbar=False` — removing colorbar for clarity

### 8.2 ROC Curve
- `sns.lineplot(x=fpr, y=tpr)` — ROC on Seaborn line plot
- AUC annotation with `plt.text()`
- Diagonal reference with `plt.plot([0,1],[0,1])`
- Multiple model comparison on one plot with `hue`

### 8.3 Precision-Recall Curve
- `sns.lineplot(x=recall, y=precision)` — PR curve
- No-skill line annotation

### 8.4 Feature Importance
- `sns.barplot(x=importances, y=feature_names, orient='h')` — horizontal bar
- Sorted by importance — `order=sorted_features`
- Color-coded by importance rank

### 8.5 Residual Plot (Regression)
- `sns.residplot(data=df, x='predicted', y='actual')` — residuals
- `lowess=True` — smoothed trend line
- Zero line reference — `plt.axhline(0)`

### 8.6 Actual vs Predicted
- `sns.scatterplot(data=df, x='actual', y='predicted', alpha=0.4)`
- `sns.regplot(data=df, x='actual', y='predicted')` — with regression line
- `plt.plot([min, max], [min, max])` — perfect prediction line

---

## 9. Training Monitoring with Seaborn

### 9.1 Loss and Metric Curves
- `sns.lineplot(data=history_df, x='epoch', y='loss')` — training loss
- `hue='split'` — train vs validation in same plot
- `errorbar='sd'` — standard deviation bands across runs

### 9.2 Multiple Run Comparison
- `sns.lineplot(data=runs_df, x='epoch', y='val_loss', hue='run_id')` — all runs
- `sns.lineplot(data=runs_df, x='epoch', y='val_accuracy', estimator='mean', errorbar='sd')` — mean + std

### 9.3 Metric Distribution Across Folds
- `sns.boxplot(data=cv_df, x='model', y='f1_score')` — CV score distribution
- `sns.violinplot(data=cv_df, x='model', y='accuracy')` — distribution shape

---

## 10. Hyperparameter Tuning Visualization

### 10.1 Validation Curve
- `sns.lineplot(x=param_values, y=train_scores)` — training scores
- `sns.lineplot(x=param_values, y=val_scores)` — validation scores
- Shaded ± std region using `fill_between`
- Annotating optimal parameter value

### 10.2 Grid Search Heatmap
- `sns.heatmap(scores_pivot, annot=True, fmt='.3f', cmap='YlGnBu')` — 2D grid
- `scores_pivot = pd.pivot_table(results_df, values='mean_test_score', index='param1', columns='param2')`
- Identifying the best combination

### 10.3 Hyperparameter Sensitivity
- `sns.lineplot(data=results_df, x='param', y='val_score')` — param vs score
- `hue='model'` — comparison across models
- `errorbar='sd'` — variability across CV folds

---

## 11. Dimensionality Reduction Visualization

### 11.1 PCA Scatter Plot
- `sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='target')` — PCA components
- Ellipse overlay for class regions
- Explained variance annotation in axis labels

### 11.2 t-SNE Scatter Plot
- `sns.scatterplot(data=tsne_df, x='tsne1', y='tsne2', hue='target', palette='tab10')` — t-SNE
- `style='target'` — different markers per class
- Perplexity comparison with `FacetGrid`

### 11.3 UMAP Scatter Plot
- `sns.scatterplot(data=umap_df, x='umap1', y='umap2', hue='target')` — UMAP
- Alpha for dense plots — `alpha=0.5`

### 11.4 Explained Variance Scree Plot
- `sns.barplot(x=components, y=variance_ratio)` — bar chart
- `sns.lineplot(x=components, y=cumulative_variance)` — cumulative line
- 95% threshold reference line — `plt.axhline(0.95)`

---

## 12. Regression Analysis Visualization

### 12.1 Regression Plot with Confidence Interval
- `sns.regplot(data=df, x='feature', y='target', ci=95)` — regression + CI
- `order=2` — polynomial regression
- `lowess=True` — non-parametric smooth

### 12.2 Multiple Feature Regression Grid
- `FacetGrid` with `regplot` per feature
- `sns.lmplot(data=df, x='feature', y='target', col='group')` — faceted regression

### 12.3 Regression Metrics Visualization
- `sns.heatmap(metrics_df, annot=True)` — R², MAE, RMSE per model
- `sns.barplot(data=metrics_df, x='model', y='RMSE')` — model comparison

---

## 13. Clustering Visualization

### 13.1 Cluster Scatter Plot
- `sns.scatterplot(data=df, x='f1', y='f2', hue='cluster_label', palette='tab10')` — colored clusters
- Cluster center markers using `plt.scatter()`

### 13.2 Cluster Distribution
- `sns.boxplot(data=df, x='cluster', y='feature')` — per-cluster distribution
- `sns.violinplot(data=df, x='cluster', y='feature')` — distribution shape per cluster

### 13.3 Cluster Heatmap
- `sns.heatmap(cluster_means, annot=True, cmap='YlOrRd')` — cluster profile heatmap
- `cluster_means = df.groupby('cluster').mean()`

### 13.4 Elbow and Silhouette Plot
- `sns.lineplot(x=k_values, y=inertia)` — elbow curve
- `sns.lineplot(x=k_values, y=silhouette_scores)` — silhouette curve
- Annotating optimal k

---

## 14. Multi-Model Comparison

### 14.1 Performance Bar Chart
- `sns.barplot(data=results_df, x='model', y='accuracy', errorbar='sd')` — mean + CI
- `order` — sorted by performance
- `hue='dataset'` — multiple datasets comparison

### 14.2 Score Distribution Box Plot
- `sns.boxplot(data=cv_results_df, x='model', y='f1_score')` — CV fold spread
- `sns.swarmplot(data=cv_results_df, x='model', y='f1_score', color='black', size=3)` — overlay points

### 14.3 ROC Comparison
- Multiple ROC curves on one plot using `sns.lineplot` with `hue='model'`

---

## 15. FacetGrid for ML Analysis

### 15.1 Feature Distribution per Class
- `g = sns.FacetGrid(data=df, col='feature', hue='target')`
- `g.map(sns.kdeplot, 'value')` — KDE per feature facet
- Per-feature distribution in one figure

### 15.2 Model Comparison per Metric
- `g = sns.FacetGrid(data=results_df, col='metric')`
- `g.map(sns.barplot, 'model', 'score')` — per-metric bar chart

### 15.3 Hyperparameter Grid
- `g = sns.FacetGrid(data=results_df, col='param1', row='param2')`
- Validation score scatter per param combination

---

## 16. Advanced ML Visualization Techniques

### 16.1 Annotating Key Results
- Adding accuracy annotations on confusion matrix cells
- Marking best epoch on loss curve with `plt.axvline()`
- Annotating AUC on ROC curve with `plt.text()`
- Annotating optimal k on elbow plot

### 16.2 Overlaying Multiple Plot Types
- Box plot with swarm plot overlay — `ax=ax` parameter
- KDE with rug plot overlay
- Regression plot with scatter colored by category

### 16.3 Color-Coding Results
- Sequential palette for continuous prediction confidence
- Diverging palette for signed errors (overpredict vs underpredict)
- Discrete palette for class labels

---

## 17. Seaborn Best Practices for ML

### 17.1 Effective ML Visualization
- Use `hue='target'` to compare distributions across classes
- Use `FacetGrid` for per-feature analysis grids
- Use `catplot` for grouped categorical analysis
- Annotate key metrics directly on plots

### 17.2 Reproducibility
- Fix palette and theme at notebook start
- Save figures programmatically after each analysis step
- Use consistent color mapping for class labels throughout

### 17.3 Common ML Mistakes
- Mixing training and test data in EDA visualizations
- Not sorting confusion matrix classes correctly
- Using too many hue levels without palette adjustment
- Forgetting to remove colorbar for binary confusion matrix

---

## 18. Seaborn Interview Topics for ML

### 18.1 Frequently Asked Questions
- How to create a correlation heatmap with Seaborn?
- How to visualize class distribution for a classification problem?
- How to use `pairplot` for feature selection?
- How to show model performance across CV folds?
- How to visualize feature importance?

### 18.2 ML-Oriented Coding Problems
- Build a full EDA dashboard for a classification dataset
- Create a confusion matrix heatmap with annotations
- Plot class-wise KDE distribution for all features
- Visualize hyperparameter grid search results as a heatmap
- Compare cross-validation scores of multiple models with box plots

---

## 19. Real-World ML Visualization Projects with Seaborn

### 19.1 Beginner Projects
- Iris EDA — pair plot, box plot, heatmap
- Titanic EDA — countplot, violin, heatmap
- House price correlation analysis

### 19.2 Intermediate Projects
- Complete ML pipeline EDA dashboard
- Customer churn feature analysis report
- Feature selection visualization using correlation and importance

### 19.3 Advanced Projects
- Multi-model evaluation dashboard
- Hyperparameter tuning visualization report
- Full dimensionality reduction and clustering analysis

---

## 20. Seaborn Roadmap for ML Engineers

### 20.1 Beginner Level
- `histplot`, `kdeplot`, `countplot` for EDA
- `heatmap` for correlation matrix
- `pairplot` for feature relationships

### 20.2 Intermediate Level
- `boxplot`, `violinplot`, `stripplot` for outlier and distribution analysis
- `regplot` and `lmplot` for regression analysis
- `catplot` and `FacetGrid` for multi-panel EDA
- Feature importance and confusion matrix visualization

### 20.3 Advanced Level
- Multi-model comparison dashboards
- Hyperparameter grid heatmaps
- Dimensionality reduction scatter plots
- Full pipeline EDA report generation
