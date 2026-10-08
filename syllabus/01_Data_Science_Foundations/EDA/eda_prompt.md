I have a structured EDA learning roadmap organized as: Section > Topics > Subtopics.

I will provide you with ONE SECTION and all its topics/subtopics at a time.

**Your task**: Create a comprehensive `.md` file — a complete EDA interview-preparation module
for that section, specifically tailored for a fresher / entry-level **Data Analyst, Data Scientist, Business Analyst, ML Engineer** technical interview.

Do NOT teach the entire EDA library outside the topics I provide.
Stay focused on the given section, but include prerequisite Python, Pandas, NumPy, and basic statistics concepts when necessary.

---

### INPUT

```text
SECTION:
<Section name>

TOPICS:
<topic 1>
<topic 2>
...
```

Analyze every topic and generate the sections below in full detail.

---

### 1. SECTION OVERVIEW

Provide a concise, practical overview:
- **What this section is about**: Core themes and techniques covered.
- **Why these EDA concepts matter**: Their role in industry and why hiring managers evaluate them.
- **Where they are used in real projects**: Production use cases and workflows.
- **What a fresher should be able to do**: Concrete competencies after completing this section.

---

### 2. TOPIC-WISE INTERVIEW CONCEPTS

For EVERY topic provided, include:
- **Concept**: Precise mechanism without sacrificing rigor.
- **Why it matters**: What signal the interviewer seeks.
- **Technical / Algorithmic Intuition**: Design principle or computational mechanism.
- **Syntax / EDA API**: Exact Python/EDA functions, classes, standard import paths.
- **Minimal Working Example**: Small, realistic, runnable Python snippet.
- **Expected Output / Interpretation**: What the result looks like.
- **Important Parameters & Options**: Key parameters a fresher must understand.
- **Common Interview Confusion**: Related APIs or behaviors frequently conflated.

---

### 3. FRESHER INTERVIEW QUESTIONS

#### CRITICAL REQUIREMENT FOR ALL ANSWERS:
> Every answer must be 3-5 substantive sentences covering:
> 1. **Direct Answer & Definition**
> 2. **Technical & Computational Reasoning**
> 3. **Practical Application / Syntax**
> 4. **Nuances, Defaults & Pitfalls**

#### A. Basic Questions (20 Questions — In-Depth, Multi-Sentence Answers)
Generate **exactly 20** Basic / Fresher Questions.
Format: **Question** | **Answer** (comprehensive, multi-sentence, interview-grade)

#### B. Intermediate Questions (20 Questions — Detailed Conceptual Explanations)
Generate **exactly 20** Intermediate Questions.
Format: **Question** | **Expected Answer** | **Explanation & Deep-Dive**

#### C. Code-Based Questions (10 Questions)
Generate **exactly 10** EDA code snippets.
For each: snippet + expected answer + line-by-line technical explanation.

#### D. Output-Prediction Questions
Short snippets where the candidate predicts output, shapes, types, or behavior.
Provide code + detailed breakdown.

#### E. Debugging Questions (10 Questions)
Generate **exactly 10** debugging questions.
Cover issues including:
- describe() missing numeric summary due to object column dtype
- Pearson correlation on non-normal data without checking assumptions
- Missing % computed without dividing by len(df)
- Histogram bins too low hiding multimodal distributions
- Log transform applied to features with zero or negative values
- Outlier removal causing class imbalance before modeling
- corr() on boolean columns without converting to int
- CSV read with wrong encoding causing garbled string columns

Format: **Problem** | **Incorrect Code** | **Expected Correction** | **Explanation**

#### F. Scenario-Based Questions
- *Dataset with 40 features for customer churn. Walk through EDA process.*
- *20% of a key feature is missing. Drop, impute, or flag rows?*
- *Target variable is 95% class 0. How does this affect EDA and modeling?*
- *Two features have Pearson correlation of 0.97. What do you do?*
- *Training distributions very different from test set. What does this mean?*

Provide complete, structured candidate responses.

---

### 4. MOST IMPORTANT INTERVIEW QUESTIONS

- **Must Know**: 5–8 non-negotiable core questions — full interview-grade answers required.
- **Good to Know**: 5–8 questions distinguishing average from strong candidates.
- **Bonus**: 3–5 advanced questions showing exceptional depth.

---

### 5. COMMON BEGINNER MISTAKES (10 MISTAKES)

Generate **exactly 10** realistic mistakes.
Format: **Mistake** | **Why it happens** | **Correct approach** | **Example** (WRONG vs RIGHT code)

---

### 6. EDA API CONFUSIONS

Clarify frequently conflated pairs:
- df.describe() vs df.info() vs df.dtypes
- boxplot vs violinplot for distribution analysis
- pearsonr vs spearmanr vs kendalltau correlation
- dropna() vs fillna() strategy trade-offs
- value_counts() vs groupby().count()
- Outlier detection: IQR vs Z-score vs Isolation Forest
- heatmap (correlation matrix) vs pairplot (pairwise distributions)
- Skewness vs Kurtosis interpretation
- duplicated() vs drop_duplicates()
- Univariate vs Bivariate vs Multivariate analysis

Format: **Concept 1** | **Concept 2** | **Key Difference** | **When to Use Which** | **Code Comparison**

---

### 7. INTERVIEW TRAPS (10 TRAPS)

Generate **exactly 10** interview traps:
- *"If a feature has 30% missing values must you always drop it?"*
- *"Pearson correlation of 0.95 means one feature causes the other?"*
- *"No NaN values — can you still have missing data?"*
- *"How can mean and median differ significantly and what does that tell you?"*
- *"Is a histogram always sufficient for understanding a distribution?"*

Format: **Trap Question** | **Naive Answer** | **Trap Unveiled** | **Winning Response**

---

### 8. REAL-WORLD EDA QUESTIONS

Connect concepts to production systems and real-world challenges.
Only generate questions directly related to the provided section topics.

---

### 9. CODING TASKS (MINI & NORMAL)

#### A. Mini Coding Tasks (10 Tasks)
Exactly 10 quick tasks (1–5 lines of core logic) for live coding rounds.
Format: **Task** | **Requirements** | **Expected Approach** | **Reference Solution**

#### B. Normal Coding Tasks (10 Tasks with Complete Reference Solutions)
Exactly 10 production-grade end-to-end exercises.
Format: **Task** | **Requirements** | **Expected Approach** | **Reference Solution** (complete, runnable, PEP 8)

---

### 10. "EXPLAIN THIS CODE" QUESTIONS

3–5 representative EDA code snippets. For each:
- Present the code
- Ask the candidate to explain each critical line, intermediate transformations, design rationale, failure modes
- Provide the **Ideal Interview Answer**

---

### 11. "FIX THIS CODE" QUESTIONS

3–5 realistic, buggy EDA code samples.
Format: **Problem Statement** | **Broken Code** | **Expected Behavior** | **Corrected Code** | **Root Cause**

---

### 12. RAPID-FIRE QUESTIONS

15–30 short Q&As.
Format: **Q** | **A** (1–2 sentences, authoritative)

---

### 13. FLASHCARD SECTION

15–20 flashcards for active recall.
Format: **Front** (concept/question/formula) | **Back** (concise explanation with key caveat)

---

### 14. EDA INTERVIEW CHEAT SHEET

High-density tabular summary for last-minute review:
- Core Methods: describe(), info(), value_counts(), corr(), isnull().sum(), duplicated()
- Visualization: histogram, boxplot, violinplot, scatter, heatmap, pairplot
- Statistics: mean, median, std, IQR, skewness, kurtosis, Pearson r, Spearman rho
- Outlier Rules: Z-score > 3; IQR below Q1-1.5*IQR or above Q3+1.5*IQR
- Anti-Patterns: Dropping missing without analysis | ignoring distribution shape | correlation as causation

---

### 15. FRESHER ANSWER PATTERNS

For 3–5 high-frequency questions:
**Question** | **Weak Answer** | **Critique** | **Stronger Fresher Answer**

---

### 16. QUESTIONS THAT TEST UNDERSTANDING

Perturbation-style questions where the interviewer changes one variable.
Candidate must reason cause-and-effect from first principles.

---

### 17. PROJECT-BASED QUESTIONS

5–8 questions for a candidate with an end-to-end project using:
`Python`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-Learn`

Test technical decision-making, validation strategy, and trade-off articulation.

---

### 18. CROSS-CONCEPT QUESTIONS

Connect the section to adjacent fundamentals:
- **Pandas**: All EDA operations built on DataFrame/Series.
- **NumPy**: Statistical computations underlie describe(), percentile, std.
- **Matplotlib/Seaborn**: Every EDA visualization is created here.
- **Statistics**: EDA applies descriptive statistics and distribution theory.
- **Machine Learning**: EDA directly informs feature selection and preprocessing assumptions.

---

### 19. WHAT A FRESHER SHOULD MEMORIZE

- Core EDA methods: describe(), info(), value_counts(), corr(), isnull().sum()
- Statistical terms: mean, median, mode, variance, std, IQR, skewness, kurtosis
- Visualization types: histogram, boxplot, scatter, heatmap, pairplot, bar chart
- Outlier thresholds: Z-score > 3; IQR rule Q1-1.5*IQR, Q3+1.5*IQR
- Imports: import pandas as pd; import seaborn as sns; import matplotlib.pyplot as plt

---

### 20. WHAT A FRESHER SHOULD UNDERSTAND

- Why log-transforming right-skewed features can improve linear model performance
- Why correlation does not imply causation and how confounders appear
- How to distinguish MCAR, MAR, MNAR missing patterns
- Why visualizing distributions is essential even when summary stats look normal
- How target variable leakage can be detected in EDA before modeling

---

### 21. WHAT A FRESHER SHOULD PRACTICE

- Compute complete univariate statistics for all columns in a real dataset
- Generate a correlation heatmap and identify multicollinear feature pairs
- Detect outliers with both IQR and Z-score and compare results
- Analyze missing value patterns and propose imputation strategies
- Produce a full EDA report comparing two groups (churned vs retained)

---

### 22. INTERVIEW READINESS CHECKLIST

- [ ] I can compute descriptive statistics for numeric and categorical features.
- [ ] I can identify missing patterns and propose appropriate handling strategies.
- [ ] I can detect outliers using IQR and Z-score and justify decisions.
- [ ] I can produce and interpret a correlation heatmap and pairplot.
- [ ] I can explain skewness and kurtosis with real examples.
- [ ] I can communicate EDA insights with visualizations and written summaries.

---

### 23. FINAL MOCK INTERVIEW

Complete 30-minute mock: **5 Basic + 5 Conceptual + 5 Code-Based + 3 Debugging + 2 Scenario**

**Structure Rule**:
1. Present the entire Mock Interview **without answers** first.
2. Follow with the **Answer Key & Scoring Rubric**.

---

### 24. FINAL REVISION SUMMARY

2-minute pre-interview summary:
- **Core Concepts**: Non-negotiable fundamentals (bullet points)
- **Essential APIs & Modules**: Quick reference table
- **Key Algorithmic Differences**: Core distinctions summary
- **Top 5 Traps to Dodge**: Critical reminders
- **Must-Know Code Snippet**: The single most comprehensive template
- **Final Interview Pro-Tip**: Mindset and communication advice

---

### 25. IMPORTANT RULES & GUARDRAILS

1. **No Hallucinated APIs**: Use only authentic **Pandas (2.0+), Matplotlib, Seaborn, SciPy** and Python 3.10+ APIs.
2. **No Single-Line Answers**: Section 3 answers must be multi-sentence, technically rigorous.
3. **Exact Counts**: 20 Basic, 20 Intermediate, 10 Code-Based, 10 Debugging, 10 Traps,
   10 Mini Tasks, 10 Normal Tasks, 10 Mistakes.
4. **Valid Python Code**: All snippets syntactically valid, self-contained, PEP 8 compliant.
5. **Practical Over Esoteric**: Focus on real tech interview questions for freshers.

---

### 26. FINAL OUTPUT STRUCTURE

```markdown
# Section: <Section Name>

## 1. Section Overview
## 2. Topic-Wise Interview Concepts
### Topic 1: <Topic Name>
...
## 3. Fresher Interview Questions
### A. Basic Questions (20)
### B. Intermediate Questions (20)
### C. Code-Based Questions (10)
### D. Output-Prediction Questions
### E. Debugging Questions (10)
### F. Scenario-Based Questions
## 4. Top Questions to Definitely Prepare
### Must Know
### Good to Know
### Bonus
## 5. Mistakes to Avoid (10 Mistakes)
## 6. EDA API Confusions
## 7. Interview Traps (10 Traps)
## 8. Real-World EDA Questions
## 9. Coding Tasks
### A. Mini Coding Tasks (10)
### B. Normal Coding Tasks (10 with Reference Solutions)
## 10. Explain This Code
## 11. Fix This Code
## 12. Rapid-Fire Questions
## 13. Flashcards
## 14. EDA Interview Cheat Sheet
## 15. Fresher Answer Patterns
## 16. Questions That Test Understanding
## 17. Project-Based Questions
## 18. Cross-Concept Questions
## 19. What a Fresher Should Memorize
## 20. What a Fresher Should Understand
## 21. What a Fresher Should Practice
## 22. Interview Readiness Checklist
## 23. Final Mock Interview
### Mock Interview Questions
### Answer Key & Scoring Rubric
## 24. Final Revision Summary
```

---

### INPUT SECTION

```text
SECTION:
[PASTE SECTION NAME HERE]

TOPICS:
[PASTE ALL TOPICS/SUBTOPICS HERE]
```

Generate the complete interview-preparation module according to all instructions above.
