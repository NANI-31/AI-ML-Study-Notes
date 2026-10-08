Yes. If you're going to give the AI **only one EDA section at a time**, use this as your reusable **master Exploratory Data Analysis prompt**. Each time, just replace `[EDA SECTION]`.

I am learning Exploratory Data Analysis (EDA) as part of my AI/ML and Generative AI learning journey.

I will give you **one EDA roadmap section at a time**. Your job is to teach me that section thoroughly, from fundamentals to advanced understanding.

Assume that I know basic Python, NumPy, Pandas, and basic visualization (Matplotlib, Seaborn), but I want to develop a **strong practical and conceptual understanding** of EDA as a systematic process for understanding data before building ML models.

## EDA Section

[EDA SECTION]

---

# How I want you to teach me

For every topic and subtopic in the section, follow this learning structure:

### 1. What is it?

- Give a clear definition.
- Explain the problem it solves.
- Explain why this EDA step is important in the AI/ML workflow.

### 2. Intuition

- Explain the concept in very simple language.
- Use a real-world analogy when useful.
- Explain what questions this EDA step is trying to answer about the data.
- Then gradually move toward the technical explanation.

### 3. Prerequisites

- Mention any Python/Pandas/Statistics/Visualization concepts I need to understand first.
- If necessary, briefly teach the prerequisite before continuing.

### 4. Methodology and Core Concepts

- Describe the systematic approach or workflow for this EDA step.
- Show the important code/functions used.
- Explain what each step of the analysis tells you.
- Clearly state what you are looking for in the results.

### 5. Examples

Give examples progressively:

**Beginner → Intermediate → Advanced**

For every important example:

- Show the code.
- Explain it line by line.
- Show the expected output (table, statistic, or plot description).
- Explain what the output tells you about the data.
- Explain what action you would take based on the finding.

Use realistic datasets (Titanic, Iris, California Housing, or a simple synthetic dataset) so I can understand what EDA findings look like in practice.

### 6. Statistical Foundations

When relevant, explain the statistical concept behind the EDA step.

For example:

- What does a high skewness value mean?
- What does a correlation coefficient of 0.9 actually tell you?
- What is the difference between mean and median, and when does it matter?
- What is an outlier, and how is the IQR method derived?
- What does a Q-Q plot tell you about a distribution?
- What does the p-value mean in a statistical test used during EDA?

Do not assume I have a deep statistics background, but explain enough for me to interpret EDA results confidently.

### 7. AI/ML Connection

For every important concept, explain:

- How this EDA finding directly impacts ML model performance.
- What preprocessing or feature engineering step it motivates.
- Give a realistic AI/ML example.

Use examples involving things such as:

- How class imbalance discovered in EDA leads to using SMOTE or class weights
- How skewed features motivate log transformation before training
- How high correlation between features motivates dropping one to reduce multicollinearity
- How missing data patterns motivate imputation strategies
- How outliers motivate using robust scalers or outlier removal

Only use an example when it is genuinely relevant.

### 8. Tools and Libraries

For each EDA step, mention the best Python tools to use:

- Pandas (`.describe()`, `.info()`, `.value_counts()`, `.corr()`, etc.)
- Matplotlib (histograms, scatter plots, box plots)
- Seaborn (pairplot, heatmap, boxplot, violin plot, etc.)
- Plotly (interactive exploration)
- `pandas-profiling` / `ydata-profiling` (automated EDA)
- `missingno` (missing data visualization)
- SciPy (statistical tests)

Explain when to use each tool and why.

### 9. Common Mistakes

Show common mistakes beginners/intermediate learners make during EDA.

For each mistake:

**Wrong approach → Why it is wrong → Correct approach**

Also mention important pitfalls such as:

- Drawing causal conclusions from correlation
- Ignoring data leakage during EDA
- Normalizing before splitting train/test
- Not distinguishing between missing at random vs. not at random
- Reporting statistics without checking their assumptions

### 10. Checklist / Systematic Approach

Provide a practical EDA checklist for this section. For example:

- [ ] Check data shape
- [ ] Check dtypes
- [ ] Identify missing values
- [ ] Analyze distributions

This checklist should be something I can reuse on any real dataset.

### 11. Best Practices

Give practical rules I should follow when performing EDA for real AI/ML projects.

### 12. Interview Perspective

Give me important interview questions related to the section.

Include:

- Conceptual questions
- Interpretation questions (e.g., "What would you do if you found X in EDA?")
- Coding questions
- Statistical reasoning questions
- AI/ML pipeline-related EDA questions

Provide the answers and explanations after the questions.

### 13. Practice

Give me exercises at three levels:

**Beginner**

- 2–3 exercises

**Intermediate**

- 2–3 exercises

**Advanced**

- 2–3 exercises

Do NOT immediately provide solutions.

Let me solve them first.

When I submit my answers:

- Check them.
- Identify mistakes.
- Explain why they are wrong.
- Show the correct solution.
- Suggest a better approach if one exists.

### 14. Mini Project

If the section contains concepts that can be combined into a practical task, give me a small EDA mini-project using those concepts.

Give me:

- Problem statement
- Dataset to use (built-in or simple synthetic)
- Required EDA questions to answer
- Expected findings format
- Constraints
- Hints

Do not provide the complete solution until I attempt it.

---

# Important Teaching Rules

1. Do not just give definitions. Teach the reasoning behind the EDA step.

2. Prefer practical, data-driven examples over unnecessary theory.

3. Start simple and gradually increase complexity.

4. Always connect EDA findings to actionable next steps in the ML pipeline.

5. When statistical concepts are involved, explain them in plain language before showing code.

6. When plots are described, explain what patterns to look for and what they mean.

7. When performance is relevant, explain how to scale EDA to large datasets.

8. Do not introduce unrelated EDA topics.

9. If an advanced concept depends on another concept, explain the dependency first.

10. Do not assume that I understand statistics deeply.

11. Use tables and simple text diagrams when they make the concept easier to understand.

12. Distinguish clearly between:

- Univariate analysis (single variable)
- Bivariate analysis (two variables)
- Multivariate analysis (many variables)

13. Focus on understanding rather than memorization.

14. Relate EDA findings to ML model building only where the connection is meaningful.

15. Do not teach the entire EDA process at once.

16. Teach ONLY the section I provide.

17. At the end, give me a concise **Revision Sheet** containing:

- Key EDA questions answered in this section
- Important functions and methods used
- Important rules
- Common mistakes
- Statistical concepts summary
- Checklist summary

18. After teaching the section, STOP and wait for me. Do not automatically move to another EDA section.

---

# Teaching Goal

My goal is not simply to learn EDA tools.

I want to understand:

**What → Why → How → Statistical Meaning → ML Impact → Actionable Insights**

so that I can confidently explore any dataset, identify issues, and make informed preprocessing and modeling decisions.

Now teach me the following EDA section:

[EDA SECTION]

**How to use it:** paste the prompt once, then replace `[EDA SECTION]` with something like `3. Univariate Analysis` or a smaller subsection such as `3.1 Analyzing Numerical Feature Distributions`. This keeps each lesson focused and prevents the AI from overwhelming you.
