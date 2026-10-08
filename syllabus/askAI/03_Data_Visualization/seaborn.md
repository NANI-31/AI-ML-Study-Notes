Yes. If you're going to give the AI **only one Seaborn section at a time**, use this as your reusable **master Seaborn prompt**. Each time, just replace `[SEABORN SECTION]`.

I am learning Seaborn as part of my AI/ML and Generative AI learning journey.

I will give you **one Seaborn roadmap section at a time**. Your job is to teach me that section thoroughly, from fundamentals to advanced understanding.

Assume that I know basic Python, NumPy, Pandas, and basic Matplotlib, but I want to develop a **strong practical and conceptual understanding** of Seaborn for statistical data visualization in AI/ML.

## Seaborn Section

[SEABORN SECTION]

---

# How I want you to teach me

For every topic and subtopic in the section, follow this learning structure:

### 1. What is it?

- Give a clear definition.
- Explain the problem it solves.
- Explain why Seaborn provides this feature on top of Matplotlib.

### 2. Intuition

- Explain the concept in very simple language.
- Use a real-world analogy when useful.
- Explain when to prefer Seaborn over raw Matplotlib.
- Then gradually move toward the technical explanation.

### 3. Prerequisites

- Mention any Python/NumPy/Pandas/Matplotlib/Seaborn concepts I need to understand first.
- If necessary, briefly teach the prerequisite before continuing.

### 4. Syntax and Core Concepts

- Show the important syntax.
- Explain important parameters, attributes, and functions.
- Explain what each part of the code does.
- Distinguish between Figure-level functions (e.g., `sns.relplot`, `sns.displot`) and Axes-level functions (e.g., `sns.scatterplot`, `sns.histplot`) wherever relevant.

### 5. Examples

Give examples progressively:

**Beginner → Intermediate → Advanced**

For every important example:

- Show the code.
- Explain it line by line.
- Describe the expected visual output.
- Explain why the plot looks the way it does.

Use small, clean datasets (or built-in Seaborn datasets like `tips`, `titanic`, `iris`, `penguins`) so I can manually understand what is being visualized.

### 6. Seaborn Internals

When relevant, explain what Seaborn is doing internally.

For example:

- How Seaborn wraps Matplotlib's Figure and Axes
- The FacetGrid and PairGrid architecture
- How Figure-level functions return a Grid object, not an Axes
- How Seaborn handles Pandas DataFrames and long-form vs. wide-form data
- The role of `hue`, `col`, `row` semantics
- How themes and styles override rcParams

Do not make the internal explanation unnecessarily complicated, but give enough detail for me to understand how Seaborn works under the hood.

### 7. AI/ML Connection

For every important concept, explain:

- Where it is used in AI/ML.
- Why it is useful.
- Give a realistic AI/ML example.

Use examples involving things such as:

- Visualizing feature distributions
- Correlation heatmaps for feature selection
- Pairplots for EDA
- Violin plots for class distribution analysis
- Visualizing model residuals
- Comparing performance across groups (e.g., hue by model type)
- Visualizing confusion matrices as heatmaps

Only use an example when it is genuinely relevant.

### 8. Performance

Explain performance considerations when relevant:

- Rendering large datasets efficiently
- Using `rasterized=True` for dense scatter plots
- Sampling data before plotting
- FacetGrid vs. manual subplots for many panels
- Memory usage when creating complex grid plots

Show inefficient and efficient approaches when useful.

### 9. Common Mistakes

Show common mistakes beginners/intermediate learners make.

For each mistake:

**Wrong approach → Why it is wrong → Correct approach**

Also mention important edge cases such as:

- Mixing Figure-level and Axes-level calls on the same grid
- Not passing the correct data format (long-form vs. wide-form)
- Overriding Seaborn styles accidentally
- Misusing `hue` with continuous vs. categorical variables

### 10. Comparison

When multiple approaches exist, compare them in a table.

For example:

| Approach | Advantages | Disadvantages | When to use |
| -------- | ---------- | ------------- | ----------- |

### 11. Best Practices

Give practical rules I should follow when writing Seaborn code for real AI/ML projects.

### 12. Interview Perspective

Give me important interview questions related to the section.

Include:

- Conceptual questions
- Output-based questions
- Debugging questions
- Coding questions
- AI/ML-related Seaborn visualization questions

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

If the section contains concepts that can be combined into a practical task, give me a small Seaborn mini-project using those concepts.

The project should be relevant to AI/ML wherever possible.

Give me:

- Problem statement
- Requirements
- Input (built-in dataset or generation instructions)
- Expected visual output description
- Constraints
- Hints

Do not provide the complete solution until I attempt it.

---

# Important Teaching Rules

1. Do not just give definitions. Teach the reasoning behind the concepts.

2. Prefer practical examples over unnecessary theory.

3. Start simple and gradually increase complexity.

4. Always distinguish between **Figure-level** and **Axes-level** functions when both are relevant.

5. When explaining FacetGrid or PairGrid, explicitly show the grid layout with a text diagram.

6. When statistical summaries are shown in plots (e.g., confidence intervals, KDE), explain what the statistic means.

7. When performance is relevant, explain **why** one approach is faster or more memory-efficient.

8. Do not introduce unrelated Seaborn topics.

9. If an advanced concept depends on another concept, explain the dependency first.

10. Do not assume that I understand Seaborn or Matplotlib internals.

11. Use tables and simple text diagrams when they make the concept easier to understand.

12. Distinguish clearly between:

- Axes-level function behavior
- Figure-level function behavior
- What the underlying Matplotlib object looks like

13. Focus on understanding rather than memorization.

14. Relate concepts to AI/ML only where the connection is meaningful.

15. Do not teach the entire Seaborn roadmap at once.

16. Teach ONLY the section I provide.

17. At the end, give me a concise **Revision Sheet** containing:

- Key concepts
- Important functions and parameters
- Figure-level vs. Axes-level quick reference
- Common mistakes
- Performance tips
- Long-form vs. wide-form data reminder

18. After teaching the section, STOP and wait for me. Do not automatically move to another Seaborn section.

---

# Teaching Goal

My goal is not simply to learn Seaborn syntax.

I want to understand:

**What → Why → How → Internals → Performance → AI/ML Usage**

so that I can confidently use Seaborn to explore data statistically, identify patterns, and communicate AI/ML insights clearly.

Now teach me the following Seaborn section:

[SEABORN SECTION]

**How to use it:** paste the prompt once, then replace `[SEABORN SECTION]` with something like `4. Categorical Plots` or a smaller subsection such as `4.2 Box Plots and Violin Plots`. This keeps each lesson focused and prevents the AI from overwhelming you.
