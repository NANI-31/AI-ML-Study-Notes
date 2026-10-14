Generate 100 New Objective Questions from the Reference Markdown File

Use the attached/reference Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference file:

[test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/ML/regression/Linear/test.md)

Task

Analyze the entire reference .md file carefully and generate 100 NEW multiple-choice objective questions based on the concepts covered in it.

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create:

test-01.md
test-02.md
test-03.md
test-04.md
test-05.md
test-06.md
test-07.md
test-08.md
test-09.md
test-10.md

Each file must contain exactly 10 questions, for a total of 100 questions.

1. Use the Reference File Correctly

First, thoroughly analyze the supplied test.md.

Identify:

- Topics and subtopics covered (Simple linear regression, multiple linear regression, slope, intercept, from-scratch gradient descent, learning rate behavior, loss functions, residual errors, evaluation metrics [MAE, MSE, RMSE], Pearson correlation r, train/test splitting, scikit-learn LinearRegression, Python zip() and list comprehensions)
- Important definitions (Slope m, intercept c, residuals, learning rate, gradient, MAE, MSE, RMSE, independent vs dependent variables, hyperplane)
- Mathematical concepts and formulas:
  - Simple linear model: $y = mx + c$
  - Multi-variable linear model: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k$
  - Residual / error: $e_i = \hat{y}_i - y_i$ (or $y_i - \hat{y}_i$)
  - Squared loss vs. absolute loss and differentiability
  - Gradient computation: $\text{gradient} = \frac{1}{n} \sum (\hat{y}_i - y_i) x_i$
  - Gradient descent parameter update: $m_{\text{new}} = m_{\text{old}} - \alpha \times \text{gradient}$
  - Evaluation metrics: $\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$, $\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$, $\text{RMSE} = \sqrt{\text{MSE}}$
  - Pearson correlation coefficient $r \in [-1, 1]$
- Python/scikit-learn behavior:
  - `LinearRegression()`, `.fit()`, `.predict()`, `.coef_`, `.intercept_`
  - `train_test_split(X, y, test_size=..., random_state=...)`
  - Number of coefficients equals number of features; exactly one intercept
  - Built-in Python functions: `zip()`, list comprehensions
- Common mistakes and misconceptions:
  - Confusing slope $m$ with intercept $c$
  - Believing each variable gets its own intercept in multiple regression
  - Assuming RMSE is always smaller than MSE (it is larger when MSE < 1)
  - Thinking gradient descent adds the gradient rather than subtracting it
  - Misinterpreting wild oscillations as a need for more steps instead of lowering the learning rate
  - Thinking negative correlation means a weak relationship
  - Confusing correlation with causality or using it as a strict variable exclusion rule
- Practical ML scenarios (housing prices, engine performance, delivery times, salary estimation based on experience/education)
- Difficulty level and pedagogical progression
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new situations, datasets, numerical values, code patterns, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference file.

Avoid:

- Copying the same question with different numbers
- Reversing the answer choices
- Changing only feature names (e.g., replacing `SquareFeet` with `LivingArea` without altering the core problem)
- Rephrasing an existing question without changing what it tests
- Reusing the exact same scenario repeatedly
- Asking the same concept in essentially the same way

Instead, create questions that require the learner to apply, interpret, compare, calculate, debug, or reason about the concepts.

For example, if the reference covers gradient descent updates, don't generate 10 questions that merely ask whether learning rate scales the update.

Instead, vary the question types:

- Manually calculate a model prediction $y = mx + c$ with positive and negative slopes/intercepts
- Compute MAE, MSE, and RMSE step-by-step for a 3-point or 4-point toy dataset
- Trace a single gradient descent update step from given errors and inputs
- Diagnose what happens when the learning rate is too large (divergence/oscillation) vs too small (slow convergence)
- Explain why squared loss is preferred for gradient descent over absolute loss (smooth differentiability everywhere)
- Distinguish between a line in 2D space and a hyperplane in multi-dimensional space
- Determine the exact shape of `model.coef_` and `model.intercept_` for multiple features
- Interpret Pearson correlation matrices: distinguish between magnitude (strength) and sign (direction)
- Trace Python logic using `zip()` and list comprehensions in custom regression loops
- Analyze how individual large outliers disproportionately inflate MSE and RMSE compared to MAE

3. Maintain the Same Question Format

Follow the interactive structure and style of the study notes repository.

Each question should contain:

- A realistic scenario or context
- A clearly stated question
- Four answer choices: A, B, C, D (using blank `- [ ]` checkboxes)
- One or more correct answers
- An HTML `<details>` and `<summary>` collapsible block containing:
  - An explicit `**Correct Answer:** **...**` declaration
  - A clear `#### Explanations:` breakdown explaining every single option individually

Use this exact structure:

### N. Question

[Realistic scenario and clearly phrased question]

- [ ] **A.** [Option text]
- [ ] **B.** [Option text]
- [ ] **C.** [Option text]
- [ ] **D.** [Option text]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Explain clearly and specifically why this option is wrong, identifying the common misconception.
- **Option B (Correct):** Explain clearly why this option is correct, providing mathematical or conceptual reasoning.
- **Option C (Incorrect):** Explain clearly why this option is wrong.
- **Option D (Incorrect):** Explain clearly why this option is wrong.

</details>

---

If multiple answers are correct (multi-select), list all correct letters in the answer key (e.g., `**Correct Answer:** **B, D**`) and explain every option individually.

4. Question Difficulty

Keep the difficulty aligned with the reference material, but gradually increase the reasoning and depth required across the 10 files:

- Sets 01–03 (Questions 1–30): Core definitions of linear regression, slope and intercept meanings, line vs hyperplane geometry, basic correlation concepts, intuitive understanding of errors.
- Sets 04–07 (Questions 31–70): Numerical evaluations (MAE, MSE, RMSE step-by-step arithmetic), from-scratch gradient descent mechanics, loss function properties (smooth differentiability of squared loss), Python `zip()` and list comprehension workflows.
- Sets 08–10 (Questions 71–100): Advanced multi-variable linear regression, scikit-learn API nuances (`.coef_`, `.intercept_`, train/test splits), learning rate divergence and stability, outlier sensitivity, and debugging broken training loops.

Use a healthy mixture of:

- Conceptual questions
- Numerical / calculation questions
- Code execution and debugging questions
- Diagnostic & scenario-based questions
- Metric comparison questions (MAE vs MSE vs RMSE)
- Edge-case questions (zero slope, negative intercept, perfect correlation $r = \pm 1$, MSE < 1 vs MSE > 1)
- "Which statements are correct?" questions (multi-select)

Avoid making the entire set trivial recall-based questions. The learner should frequently need to evaluate formulas, perform mental or pencil-and-paper arithmetic, or trace code updates.

5. Coverage

Across all 100 questions, cover linear regression concepts comprehensively:

- Simple Linear Regression ($y = mx + c$):
  - Meaning of slope ($m$): change in $y$ per unit change in $x$
  - Meaning of intercept ($c$): value of $y$ when $x = 0$
  - Calculating predicted $\hat{y}$ given $x$, $m$, and $c$
  - Solving for $x$ given a target $\hat{y}$
  - Geometric representation as a 2D line
- Multiple Linear Regression:
  - Equation: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k$
  - Exactly one intercept $\beta_0$ regardless of the number of features $k$
  - Exactly $k$ coefficients, each representing the partial rate of change holding other variables constant
  - Geometric representation: 2 features = plane in 3D; $k$ features = hyperplane in $(k+1)$D space
  - Independent variables ($X$) vs dependent variable ($y$)
- Optimization & Gradient Descent:
  - Cost / loss function: Mean Squared Error as a smooth, convex parabola/bowl with a single global minimum
  - Why squared loss is chosen over absolute loss: smooth differentiability everywhere (derivative exists at every error value; absolute loss has a non-differentiable cusp/kink at 0)
  - Gradient calculation: average product of error and feature value, $\frac{1}{n}\sum (\hat{y}_i - y_i) x_i$
  - Direction of gradient: negative gradient indicates predictions are smaller than actual values, requiring an increase in $m$
  - Update formula: $m = m - \alpha \times \text{gradient}$ (subtracting the scaled gradient)
  - Learning rate ($\alpha$):
    - Too large: causes overshoot, wild oscillations, and divergent blow-up
    - Too small: produces painfully slow, steady convergence
    - Hyperparameter nature: tuned empirically, not solved analytically
- Regression Evaluation Metrics:
  - Residuals: difference between actual and predicted values ($y_i - \hat{y}_i$)
  - Mean Absolute Error (MAE): linear penalty, same units as target, robust to large outliers
  - Mean Squared Error (MSE): quadratic penalty, squared units, highly sensitive to large outliers
  - Root Mean Squared Error (RMSE): square root of MSE, returned to target units, penalizes outliers more severely than MAE
  - Relationship between MSE and RMSE: RMSE < MSE only when MSE > 1; if MSE < 1, RMSE > MSE
- Exploratory Data Analysis & Correlation:
  - Pearson correlation coefficient ($r$): bounds $[-1, 1]$
  - Magnitude $|r|$ measures strength; sign (+ or -) indicates direction
  - Strong positive ($r \to +1$), strong negative ($r \to -1$), zero correlation ($r \approx 0$)
  - Early exploratory filtering vs hard exclusion rules
  - Correlation between predictors (introductory multicollinearity) vs correlation with target
  - Correlation does not equal causation
- Scikit-Learn Implementation:
  - `from sklearn.linear_model import LinearRegression`
  - `from sklearn.model_selection import train_test_split`
  - Proper train/test split setup (preventing data leakage)
  - Inspecting model parameters: `model.coef_` (1D array of weights) and `model.intercept_` (single scalar float)
  - Methods: `model.fit(X_train, y_train)` and `model.predict(X_test)`
  - What `model.score(X_test, y_test)` returns ($R^2$, not MSE or MAE)
- Python Mechanics in From-Scratch ML:
  - `zip(a, b)`: positional element-wise pairing into tuples `[(a0, b0), (a1, b1), ...]`
  - List comprehensions for vectorized-style operations: `[m * x for x in x_values]`
  - Accumulating gradients across batches: `sum(e * x for e, x in zip(errors, x_values)) / len(x_values)`

Only include concepts that are supported by or reasonably derived from the reference material. Do not introduce unrelated ML topics (such as classification, sigmoid, or tree algorithms) merely to fill question count.

6. Numerical Questions

Include meaningful numerical questions across the sets.

For example:

- Calculate the predicted price $y$ given $m = 0.08$, $c = 15$, and $x = 1200$
- Given a 3-point dataset (e.g., actuals $[10, 20, 30]$ and predictions $[12, 18, 33]$), calculate:
  - The residuals $[e_1, e_2, e_3]$
  - The Mean Absolute Error (MAE)
  - The Mean Squared Error (MSE)
  - The Root Mean Squared Error (RMSE)
- Trace one step of gradient descent:
  - Given starting $m = 0$, learning rate $\alpha = 0.05$, and dataset $x = [1, 2]$, $y = [3, 5]$
  - Compute predictions $[0, 0]$
  - Compute errors $[-3, -5]$
  - Compute gradient: $((-3 \times 1) + (-5 \times 2)) / 2 = -13 / 2 = -6.5$
  - Compute updated $m = 0 - 0.05 \times (-6.5) = +0.325$
- Given multiple regression coefficients $c = 5$, $\beta_1 = 2$, $\beta_2 = -3$, calculate $y$ for $x_1 = 4, x_2 = 1$

Ensure calculations use small, integer-friendly numbers so learners can compute or verify them easily without needing an external calculator.

7. Code-Based Questions

Include realistic Python and scikit-learn code snippets.

Use code snippets such as:

```python
x = [2, 4, 6]
y = [5, 9, 13]
pairs = list(zip(x, y))
```

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[["SquareFeet", "Bedrooms"]]
y = df["PriceLakhs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print(len(model.coef_))
print(model.intercept_)
```

```python
# From-scratch training step
predictions = [m * x + c for x in x_values]
errors = [p - y for p, y in zip(predictions, y_values)]
grad_m = sum(e * x for e, x in zip(errors, x_values)) / len(x_values)
grad_c = sum(errors) / len(errors)

m = m - lr * grad_m
c = c - lr * grad_c
```

Questions should ask the learner to determine:

- What specific output or value a code block prints
- Which line will throw an exception or produce a bug
- What the shape or length of `model.coef_` is for a given input dataframe
- What `list(zip(a, b))` returns for mismatched or matching lengths
- Why an update rule fails when written as `m = m + lr * grad` instead of subtraction
- Which metric `model.score(X_test, y_test)` evaluates

8. Correct Answer Distribution

Distribute correct answers reasonably across A, B, C, and D.

- Aim for an approximately balanced distribution across all 100 questions (roughly 25% each for A, B, C, D).
- For multi-select questions, vary the combinations (e.g., A & C, B & D, A & B, etc.).
- Avoid predictable patterns (e.g., avoid consecutive identical answers or alternating cycles).

9. Avoid Ambiguity

Every question must have an unambiguous, mathematically and technically defensible answer.

Avoid:

- Subjective questions without clear mathematical grounds
- Discrepancies in error convention ($y - \hat{y}$ vs $\hat{y} - y$) without defining the formula in the question
- Assuming RMSE is unconditionally smaller than MSE
- Stating that correlation proves causation
- Confusing 1D slope with partial regression coefficients in multiple regression

10. Explanations

Explanations are critical to learning.

For every single answer choice (A, B, C, D):

- Explain clearly why it is correct or incorrect.
- Never write shallow dismissals such as "Option A is wrong."
- Directly address the conceptual error or common trap behind every incorrect choice.
- Keep explanations clear, rigorous, and educational.

11. File Organization

Create exactly 10 Markdown files.

Each file must contain exactly 10 questions.

Use this continuous numbering across the files:

test-01.md → Questions 1–10
test-02.md → Questions 11–20
test-03.md → Questions 21–30
test-04.md → Questions 31–40
test-05.md → Questions 41–50
test-06.md → Questions 51–60
test-07.md → Questions 61–70
test-08.md → Questions 71–80
test-09.md → Questions 81–90
test-10.md → Questions 91–100

Do not restart numbering at 1 in each file.

12. File Titles

At the beginning of each file, include a clear header matching the set:

# Linear Regression Objective Test — Set 01

Followed immediately by the 10 questions belonging to that set.

13. Quality-Control Requirements

Before completing the generation, perform a comprehensive validation against these checkpoints:

- [ ] Exactly 100 questions total.
- [ ] Exactly 10 `.md` files created.
- [ ] Exactly 10 questions per file.
- [ ] Continuous question numbering from 1 to 100.
- [ ] Every question has exactly four choices: A, B, C, D with blank checkboxes `- [ ]`.
- [ ] Every question has at least one correct answer declared in `**Correct Answer:**`.
- [ ] Every question has an HTML `<details>` and `<summary>` dropdown block.
- [ ] Every option (A, B, C, D) has an explicit explanation inside `#### Explanations:`.
- [ ] Correct answers are reasonably balanced across A/B/C/D.
- [ ] All questions are original and not verbatim copies of the reference.
- [ ] Mathematical calculations (predictions, MAE, MSE, RMSE, gradient updates) are verified and accurate.
- [ ] Scikit-learn and Python code snippets are syntax-valid and technically correct.
- [ ] No contradictory explanations or ambiguous choices exist.
- [ ] Markdown and HTML formatting are valid and clean throughout.

Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive, conceptually rigorous questions that test deep understanding and application of Linear Regression fundamentals, optimization, evaluation, and implementation.
