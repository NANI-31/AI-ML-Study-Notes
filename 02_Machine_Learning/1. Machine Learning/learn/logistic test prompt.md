Generate 100 New Objective Questions from the Reference Markdown File

Use the attached/reference Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference file:

[test.md](file;file:///d%3A/Data/GitHub/AI-ML-Study-Notes/ML/regression/Logistic/test.md)

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

Topics and subtopics covered
Important definitions
Mathematical concepts and formulas
Python/scikit-learn behavior
Common mistakes and misconceptions
Practical ML scenarios
Relationships between concepts
Difficulty level
Question structure and wording style
Explanation style
Correct-answer format

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new situations, examples, values, wording, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference file.

Avoid:

Copying the same question with different numbers
Reversing the answer choices
Changing only names or variable values
Rephrasing an existing question without changing what it tests
Reusing the exact same scenario repeatedly
Asking the same concept in essentially the same way

Instead, create questions that require the learner to apply, interpret, compare, calculate, debug, or reason about the concepts.

For example, if the reference covers sigmoid behavior, don't generate 10 questions that simply ask whether sigmoid values are between 0 and 1.

Instead, vary the question types:

Calculate a sigmoid probability
Infer the sign of z from a probability
Determine how changing a feature affects probability
Compare two linear scores
Identify a threshold boundary
Diagnose an incorrect implementation
Interpret model output
Analyze an edge case
Compare logistic and linear regression behavior
Reason about classification decisions 3. Maintain the Same Question Format

Follow the structure and style of the reference file.

Each question should contain:

A realistic scenario or context
A clearly stated question
Four answer choices: A, B, C, D
One or more correct answers
A clear explanation for every option

Use this structure:

### N. Question

[Question scenario and question]

- [ ] **A.** ...
- [x] **B.** ...
- [ ] **C.** ...
- [ ] **D.** ...

---

### Explanation

- **Option A (Incorrect):** Explain clearly why it is wrong.
- **Option B (Correct):** Explain clearly why it is correct.
- **Option C (Incorrect):** Explain clearly why it is wrong.
- **Option D (Incorrect):** Explain clearly why it is wrong.

If multiple answers are correct, mark each correct answer with [x] and explain every option individually.

4. Question Difficulty

Keep the difficulty approximately aligned with the reference material, but gradually increase the reasoning required across the 10 files.

Use a mixture of:

Conceptual questions
Numerical questions
Code/output questions
Debugging questions
Interpretation questions
Scenario-based questions
Comparison questions
Edge-case questions
"What happens?" questions
"Which statement is correct?" questions
"Which statements are correct?" questions

Avoid making the entire set trivial recall-based questions.

The learner should sometimes need to reason through the problem before selecting an answer.

5. Coverage

Across all 100 questions, cover the concepts from the reference file comprehensively.

Do not over-focus on one concept.

Ensure that important concepts are revisited from different perspectives, while avoiding duplicate questions.

Where appropriate, include concepts involving:

Logistic regression
Linear score / decision function
Sigmoid function
Probability interpretation
Classification thresholds
Positive and negative coefficients
Feature effects
LogisticRegression
LinearRegression
predict()
predict_proba()
score()
Classification vs regression
Categorical encoding
One-hot encoding
Ordinal encoding
train_test_split
stratify
Class imbalance
Python variable naming/case sensitivity
Other concepts explicitly present in the reference file

Only include concepts that are supported by or reasonably derived from the reference material.

Do not introduce unrelated ML topics merely to increase the question count.

6. Numerical Questions

Include meaningful numerical questions where appropriate.

For example:

Calculate or approximate sigmoid probabilities
Determine whether a probability corresponds to a positive or negative score
Apply a classification threshold
Compare probabilities generated by different z values
Determine the effect of changing a feature
Interpret coefficient changes

Use calculations that are reasonable for the expected learner level.

If an exact calculation would normally require a calculator, provide values that allow reasonable interpretation or clearly state that an approximation is expected.

7. Code-Based Questions

Include realistic Python and scikit-learn questions where appropriate.

Use code snippets such as:

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

Questions may ask the learner to determine:

What the code returns
Why an error occurs
What a particular output means
Which line is incorrect
Which modification fixes the issue
How two model classes behave differently

Do not create code questions that depend on libraries or APIs not covered in the reference file.

8. Correct Answer Distribution

Distribute correct answers across A, B, C, and D.

Do not make the correct answer disproportionately A or B.

Across the 100 questions, aim for an approximately balanced distribution.

For example, roughly:

A: 25%
B: 25%
C: 25%
D: 25%

For multi-select questions, vary which combinations are correct.

Do not create predictable answer patterns.

9. Avoid Ambiguity

Every question must have a clearly defensible answer based on the concepts in the reference material.

Avoid:

Trick questions based on wording
Multiple answers that could reasonably be considered correct
Missing assumptions
Technically incorrect scikit-learn behavior
Ambiguous thresholds
Unclear terminology
Questions requiring knowledge outside the reference

When discussing library behavior, use the actual behavior of the relevant scikit-learn API.

10. Explanations

Explanations are important.

For every answer choice, explain why it is correct or incorrect.

Do not write explanations such as:

"A is incorrect."

Instead, explain the underlying reasoning.

The explanations should help the learner understand the concept and the misconception behind the wrong answer.

Keep explanations concise but technically accurate.

11. File Organization

Create exactly 10 Markdown files.

Each file must contain exactly 10 questions.

Use this numbering:

test-01.md

Questions 1–10

test-02.md

Questions 11–20

test-03.md

Questions 21–30

test-04.md

Questions 31–40

test-05.md

Questions 41–50

test-06.md

Questions 51–60

test-07.md

Questions 61–70

test-08.md

Questions 71–80

test-09.md

Questions 81–90

test-10.md

Questions 91–100

Do not restart numbering at 1 in each file.

12. File Titles

At the beginning of each file, include a simple title:

# Logistic Regression Objective Test — Set 01

Then include the 10 questions belonging to that file.

Use:

Set 01 → Questions 1–10
Set 02 → Questions 11–20
Set 03 → Questions 21–30
Set 04 → Questions 31–40
Set 05 → Questions 41–50
Set 06 → Questions 51–60
Set 07 → Questions 61–70
Set 08 → Questions 71–80
Set 09 → Questions 81–90
Set 10 → Questions 91–100

13. Quality-Control Requirements

Before finishing, perform a final validation of all 100 questions.

Verify that:

There are exactly 100 questions.
There are exactly 10 .md files.
Every file contains exactly 10 questions.
Question numbering is continuous from 1 to 100.
Every question has exactly four choices: A, B, C, D.
Every question has at least one correct answer.
Every option has an explanation.
Correct answers are reasonably balanced across A/B/C/D.
Questions are genuinely new rather than copied from the reference.
No concept is tested repeatedly in essentially identical ways.
The questions remain within the scope of the reference material.
Python/scikit-learn behavior is technically accurate.
Mathematical explanations are correct.
There are no contradictory answers or explanations.
Markdown formatting is valid and consistent.
The overall style remains consistent with the reference file.
Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive questions that genuinely test understanding of the concepts in the reference file.

The reference .md file defines the scope and style.

Generate new questions that test the learner's ability to understand and apply those concepts, rather than memorizing the original questions.
