# Plotly for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction to Plotly for Machine Learning

### 1.1 Why Interactive Visualization Matters in ML
- Exploring high-dimensional datasets interactively
- Zooming into decision boundaries and cluster regions
- Hovering over data points to inspect feature values
- Comparing model outputs interactively
- Sharing results as standalone HTML files with stakeholders

### 1.2 Role of Plotly in the ML Workflow
- Interactive EDA — zoom, filter, hover
- Feature relationship exploration
- Model evaluation dashboards
- Hyperparameter tuning result exploration
- Dimensionality reduction visualization
- Communicating ML results to non-technical audiences

### 1.3 Installing and Setup
- Installing Plotly
- Installing Kaleido for static export
- Jupyter / Google Colab setup
- Import conventions
  - `import plotly.express as px`
  - `import plotly.graph_objects as go`
  - `from plotly.subplots import make_subplots`
  - `import pandas as pd`
  - `import numpy as np`

---

## 2. Interactive EDA for Machine Learning

### 2.1 Feature Distribution
- `px.histogram(df, x='feature', color='target', barmode='overlay', marginal='box')` — interactive histogram
- `nbins` — control bin count
- `histnorm='probability'` — normalized distribution
- Hover to inspect bin counts
- `marginal='rug'` — rug plot overlay

### 2.2 Scatter Plot for Feature Relationships
- `px.scatter(df, x='f1', y='f2', color='target', hover_data=df.columns)` — interactive scatter
- `size='f3'` — bubble chart dimension
- `symbol='target'` — marker shape by class
- `trendline='ols'` — regression overlay
- Hover shows all feature values
- Zoom into dense regions

### 2.3 Pair Plot (Scatter Matrix)
- `px.scatter_matrix(df, dimensions=['f1','f2','f3','f4'], color='target')` — interactive pair plot
- Clicking diagonal to filter
- Brush selection across panels

### 2.4 Correlation Heatmap
- `px.imshow(df.corr(), text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1)` — interactive heatmap
- Hover to read exact correlation value
- Zoom into specific feature blocks
- `aspect='auto'` for wide feature sets

### 2.5 Box and Violin for Feature Analysis
- `px.box(df, x='target', y='feature', points='all')` — box + all points
- `px.violin(df, x='target', y='feature', box=True, points='all')` — violin + box + points
- Hover over individual outlier points
- Toggle categories via legend

### 2.6 Class Distribution
- `px.bar(class_counts, x='class', y='count', text_auto=True, color='class')` — class frequency
- `px.pie(df_counts, values='count', names='class')` — class proportion pie
- `hole=0.4` — donut chart

### 2.7 Density Plots
- `px.density_contour(df, x='f1', y='f2', color='target')` — 2D density
- `px.density_heatmap(df, x='f1', y='f2', marginal_x='histogram', marginal_y='histogram')` — density heatmap
- Comparing class-wise density regions

---

## 3. Feature Engineering Visualizations

### 3.1 Before and After Transformation
- `px.histogram(df, x='feature', title='Before Log Transform')`
- `px.histogram(df, x='log_feature', title='After Log Transform')`
- Side-by-side with `make_subplots(cols=2)`
- Hover to inspect bin range and count

### 3.2 Feature Importance Bar Chart
- `px.bar(fi_df, x='importance', y='feature', orientation='h', color='importance', color_continuous_scale='Blues')` — feature importance
- `text_auto=True` — show values
- Sort by importance — `fi_df.sort_values('importance')`
- Interactive hover showing importance score

### 3.3 Correlation with Target
- `px.bar(corr_df, x='feature', y='correlation', color='correlation', color_continuous_scale='RdBu', color_continuous_midpoint=0)` — target correlation
- Positive and negative correlation color-coded

### 3.4 Binning Visualization
- `px.histogram(df, x='feature', nbins=20)` — before binning
- `px.bar(binned_df, x='bin_label', y='count')` — after binning
- Hover to inspect bin boundaries

---

## 4. Outlier Detection Visualization

### 4.1 Interactive Box Plot
- `px.box(df, y='feature', points='outliers')` — outliers only
- `px.box(df, y='feature', points='all')` — all points visible
- Hover over outlier to inspect feature values and row ID

### 4.2 Z-score Scatter
- `px.scatter(df, x=df.index, y='feature', color='is_outlier', hover_data=df.columns)` — outlier highlight
- Color outliers vs inliers
- Interactive zoom on outlier region

### 4.3 Multi-feature Outlier Grid
- `make_subplots(rows, cols)` with `go.Box()` per feature
- Interactive pan and zoom across all box plots

---

## 5. Model Evaluation — Interactive Visualizations

### 5.1 Confusion Matrix
- `px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels={'x':'Predicted','y':'Actual'})` — interactive
- Hover to read exact count and percentage
- Normalized and unnormalized toggle via buttons

### 5.2 ROC Curve
- `px.line(roc_df, x='fpr', y='tpr', color='model', hover_data=['threshold'])` — multi-model ROC
- AUC annotation in legend
- `fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1)` — diagonal reference
- Hover on curve to read FPR, TPR, and threshold

### 5.3 Precision-Recall Curve
- `px.line(pr_df, x='recall', y='precision', color='model')` — PR curve
- AUC-PR annotation
- Hover to read precision, recall, and threshold value

### 5.4 Feature Importance (Interactive)
- `px.bar(fi_df, x='feature', y='importance', color='importance', text_auto='.3f')` — sorted bar
- Click to highlight feature
- Hover to read exact importance

### 5.5 Actual vs Predicted (Regression)
- `px.scatter(df, x='actual', y='predicted', color='error_magnitude', hover_data=['sample_id'])` — scatter
- `fig.add_shape(type='line')` — perfect prediction diagonal
- Hover to inspect individual sample error

### 5.6 Residual Plot
- `px.scatter(df, x='predicted', y='residual', color='residual', color_continuous_scale='RdBu', color_continuous_midpoint=0)` — residuals
- `fig.add_hline(y=0)` — zero reference line
- Hover to inspect individual residuals

### 5.7 Classification Report Heatmap
- `px.imshow(report_df.T, text_auto='.2f', color_continuous_scale='YlGnBu')` — per-class metrics
- Hover to read precision, recall, F1 per class

---

## 6. Learning and Validation Curves

### 6.1 Training Curve
- `px.line(history_df, x='epoch', y=['train_loss','val_loss'])` — loss curves
- Interactive hover showing exact loss per epoch
- Zoom into convergence region
- `log_y=True` — log scale for wide-range losses

### 6.2 Accuracy Curve
- `px.line(history_df, x='epoch', y=['train_accuracy','val_accuracy'])` — accuracy curves
- Toggle train vs val with legend click

### 6.3 Multi-Run Comparison
- `px.line(all_runs_df, x='epoch', y='val_loss', color='run_id')` — all runs
- Click legend to isolate runs
- Hover to see exact values per run and epoch

### 6.4 Validation Curve
- `px.line(vc_df, x='param_value', y=['train_score','val_score'], log_x=True)` — param vs score
- Zoom into optimal param range
- Hover to read exact score

---

## 7. Hyperparameter Tuning Visualization

### 7.1 Grid Search Heatmap
- `px.imshow(pivot_scores, text_auto='.3f', color_continuous_scale='YlGnBu')` — 2D grid
- Hover to read hyperparameter combination and score
- Zoom into best region

### 7.2 Parallel Coordinates Plot
- `px.parallel_coordinates(results_df, color='val_accuracy', dimensions=hp_columns + ['val_accuracy'])` — parallel coords
- Drag axes to reorder
- Brush each axis to filter hyperparameter ranges
- Color by validation accuracy

### 7.3 Parallel Categories Plot
- `px.parallel_categories(results_df, dimensions=cat_hp_cols, color='val_f1', color_continuous_scale='Viridis')` — categorical hyperparameters
- Tracing paths through hyperparameter combinations

### 7.4 3D Hyperparameter Surface
- `px.scatter_3d(results_df, x='lr', y='dropout', z='val_accuracy', color='val_accuracy', size='epochs')` — 3D scatter
- Hover to read full hyperparameter configuration

---

## 8. Dimensionality Reduction Visualization

### 8.1 PCA Interactive Scatter
- `px.scatter(pca_df, x='PC1', y='PC2', color='target', hover_data=['original_id'], symbol='target')` — 2D PCA
- Zoom into specific class clusters
- Hover to inspect original sample features
- `px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='target')` — 3D PCA

### 8.2 Explained Variance Plot
- `px.bar(var_df, x='component', y='variance_ratio', text_auto='.2%')` — explained variance
- `px.line(var_df, x='component', y='cumulative_variance')` — cumulative overlay
- `fig.add_hline(y=0.95)` — 95% threshold line

### 8.3 t-SNE Interactive Scatter
- `px.scatter(tsne_df, x='tsne1', y='tsne2', color='target', hover_data=df.columns)` — t-SNE
- Hover to read sample features in original space
- Zoom into cluster regions
- `px.scatter_3d(tsne_df, x='tsne1', y='tsne2', z='tsne3', color='target')` — 3D t-SNE

### 8.4 UMAP Interactive Scatter
- `px.scatter(umap_df, x='umap1', y='umap2', color='target', size='confidence')` — UMAP
- Interactive cluster exploration

---

## 9. Classification Visualizations

### 9.1 Decision Boundary
- `px.contour` or `go.Contour(x=xx, y=yy, z=probs)` — decision boundary
- `go.Scatter(x=X_test[:,0], y=X_test[:,1], mode='markers', marker_color=y_test)` — test points overlay
- Hover over regions to read predicted probability

### 9.2 Probability Distribution per Class
- `px.histogram(pred_df, x='confidence', color='correct', barmode='overlay')` — confidence by correctness
- `px.violin(pred_df, x='class', y='confidence', box=True)` — per-class confidence

### 9.3 Misclassification Explorer
- `px.scatter(error_df, x='f1', y='f2', color='true_label', symbol='predicted_label', hover_data=df.columns)` — error scatter
- Interactive filtering by label

---

## 10. Clustering Visualization

### 10.1 Interactive Cluster Scatter
- `px.scatter(df, x='f1', y='f2', color='cluster', hover_data=df.columns, symbol='cluster')` — cluster scatter
- Zoom into individual clusters
- `px.scatter_3d(df, x='f1', y='f2', z='f3', color='cluster')` — 3D clusters

### 10.2 Elbow Method
- `px.line(elbow_df, x='k', y='inertia', markers=True)` — elbow curve
- `px.line(elbow_df, x='k', y='silhouette', markers=True)` — silhouette score
- Hover to read exact inertia and k
- Annotating optimal k with `fig.add_vline()`

### 10.3 Cluster Profile Heatmap
- `px.imshow(cluster_means.T, text_auto='.2f', color_continuous_scale='RdBu')` — cluster feature means
- Hover to read feature value per cluster

### 10.4 Cluster Distribution
- `px.box(df, x='cluster', y='feature', color='cluster')` — per-cluster distribution
- Interactive comparison across clusters

---

## 11. Regression Visualization

### 11.1 Interactive Regression Line
- `px.scatter(df, x='feature', y='target', trendline='ols', hover_data=df.columns)` — scatter + regression
- `trendline='lowess'` — non-parametric fit
- Hover to inspect residual

### 11.2 Multi-Feature Partial Dependence
- `px.line(pd_df, x='feature_value', y='partial_dependence', facet_col='feature')` — partial dependence
- Interactive per-feature PD plots

### 11.3 Prediction Interval
- `go.Scatter(x=x, y=upper, fill=None)` — upper bound
- `go.Scatter(x=x, y=lower, fill='tonexty')` — lower bound + fill
- `go.Scatter(x=x, y=y_pred)` — prediction line overlay
- Hover to read prediction and interval

---

## 12. Multi-Model Comparison

### 12.1 Performance Bar Chart
- `px.bar(results_df, x='model', y='accuracy', error_y='std', color='model', text_auto='.3f')` — model comparison
- Interactive hover showing all metrics

### 12.2 ROC Curve Comparison
- `px.line(roc_all_df, x='fpr', y='tpr', color='model')` — all models
- Click legend to show/hide models

### 12.3 Parallel Coordinates for Model Comparison
- `px.parallel_coordinates(results_df, color='val_f1', dimensions=['lr','depth','dropout','val_acc','val_f1'])` — full view
- Brush axes to filter model configurations

### 12.4 Radar Chart (Spider Plot)
- `go.Scatterpolar(r=metrics, theta=metric_names, fill='toself', name=model_name)` — per model
- Multiple models on one radar chart
- Interactive model selection

---

## 13. Animated ML Visualizations

### 13.1 Animated Training Progress
- `px.scatter(df, x='f1', y='f2', animation_frame='epoch', color='cluster_assignment')` — cluster evolution
- Decision boundary evolution over training epochs

### 13.2 Animated Gradient Descent
- `px.scatter(gd_df, x='w1', y='w2', animation_frame='step', color='loss')` — weight trajectory
- Contour background for loss surface

### 13.3 Animated PCA/t-SNE
- `px.scatter(tsne_df, x='tsne1', y='tsne2', animation_frame='epoch', color='label')` — representation evolution
- Watching cluster formation over training

---

## 14. Interactive Dashboards with Subplots

### 14.1 EDA Dashboard
- `make_subplots(rows=2, cols=3)` — 6-panel EDA
  - Histogram per feature class
  - Correlation heatmap
  - Class distribution bar
  - Box plot per feature
  - Pair scatter
  - Target distribution

### 14.2 Model Evaluation Dashboard
- `make_subplots(rows=2, cols=2)` — 4-panel
  - Confusion matrix
  - ROC curve
  - PR curve
  - Feature importance

### 14.3 Training Monitoring Dashboard
- `make_subplots(rows=2, cols=2)` — training metrics
  - Loss curve
  - Accuracy curve
  - Learning rate schedule
  - Gradient norm per epoch

---

## 15. Plotly with Scikit-learn

### 15.1 Visualizing Scikit-learn Results
- Storing `cross_val_score` results in DataFrame → `px.box`
- Grid search results in DataFrame → `px.imshow` heatmap
- Feature importance from Random Forest → `px.bar`
- Decision tree visualization — `px.treemap`

### 15.2 Pipeline Visualization
- Visualizing preprocessing steps
- Feature names before and after transformation

---

## 16. Exporting ML Visualizations

### 16.1 Interactive HTML for Reports
- `fig.write_html('eda_report.html')` — shareable interactive report
- `include_plotlyjs='cdn'` — smaller file size
- Embedding multiple figures in one HTML

### 16.2 Static Images for Papers
- `fig.write_image('confusion_matrix.png', scale=2)` — high resolution
- `fig.write_image('roc_curve.pdf')` — vector PDF

---

## 17. Plotly Best Practices for ML

### 17.1 Effective ML Visualization
- Use `hover_data=df.columns` for full sample inspection
- Use `color='target'` consistently across EDA plots
- Use `px.parallel_coordinates` for hyperparameter exploration
- Use `trendline='ols'` for quick regression overlay

### 17.2 Performance for Large Datasets
- `go.Scattergl()` — WebGL for >10k points
- Downsampling before plotting — `df.sample(5000)`
- `px.density_heatmap` instead of scatter for very large data

### 17.3 Common Mistakes
- Not setting `color_continuous_midpoint=0` for diverging correlation heatmaps
- Forgetting `barmode='overlay'` for overlapping histograms
- Not installing `kaleido` for PNG export
- Not using `log_y=True` for exponentially decaying losses

---

## 18. Plotly Interview Topics for ML

### 18.1 Frequently Asked Questions
- How is Plotly different from Matplotlib for ML?
- How to create an interactive confusion matrix?
- How to visualize hyperparameter search results interactively?
- How to create animated training progress?
- How to use parallel coordinates for model comparison?

### 18.2 ML-Oriented Coding Problems
- Build an interactive EDA dashboard with subplots
- Create an animated scatter showing cluster evolution
- Visualize PCA in 3D with Plotly
- Build a multi-model ROC comparison chart
- Create a hyperparameter grid heatmap with hover

---

## 19. Real-World ML Projects with Plotly

### 19.1 Beginner Projects
- Interactive Iris EDA scatter matrix
- Titanic class distribution and survival pie chart
- Feature importance bar chart for Random Forest

### 19.2 Intermediate Projects
- Interactive model evaluation dashboard (confusion matrix + ROC + PR)
- Hyperparameter grid search heatmap explorer
- PCA and t-SNE 2D/3D scatter for classification dataset

### 19.3 Advanced Projects
- Full interactive ML experiment comparison dashboard
- Animated gradient descent visualization on loss surface
- Parallel coordinates hyperparameter explorer with Dash

---

## 20. Plotly Roadmap for ML Engineers

### 20.1 Beginner Level
- `px.scatter`, `px.histogram`, `px.bar`, `px.box` for EDA
- `px.imshow` for correlation heatmap and confusion matrix
- `px.scatter_matrix` for pair plot

### 20.2 Intermediate Level
- `px.line` for training curves with multi-line
- `px.scatter_3d` for dimensionality reduction
- `px.parallel_coordinates` for hyperparameter analysis
- `make_subplots` for multi-panel dashboards

### 20.3 Advanced Level
- Animated scatter for cluster and training evolution
- Fully interactive model evaluation dashboard
- `go.Contour` for decision boundaries
- Dash-based ML experiment explorer
