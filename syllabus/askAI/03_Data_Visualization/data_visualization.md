Yes. If you're going to give the AI **only one Data Visualization section at a time**, use this as your reusable **master Data Visualization prompt**. Each time, just replace `[DATA VIZ SECTION]`.

I am learning Data Visualization as part of my AI/ML and Generative AI learning journey.

I will give you **one Data Visualization roadmap section at a time**. Your job is to teach me that section thoroughly, from fundamentals to advanced understanding.

Assume that I know basic Python, NumPy, Pandas, Matplotlib, Seaborn, and basic Plotly, but I want to develop a **strong practical and conceptual understanding** of Data Visualization as a discipline — not just as a set of tools — so that I can communicate data insights and ML results clearly and effectively.

## Data Visualization Section

[DATA VIZ SECTION]

---

# How I want you to teach me

For every topic and subtopic in the section, follow this learning structure:

### 1. What is it?

- Give a clear definition.
- Explain the problem it solves.
- Explain why this visualization concept or chart type exists.

### 2. Intuition

- Explain the concept in very simple language.
- Use a real-world analogy when useful.
- Explain what story or question this visualization is meant to answer.
- Then gradually move toward the technical explanation.

### 3. Prerequisites

- Mention any concepts (visual perception, statistics, tools) I need to understand first.
- If necessary, briefly teach the prerequisite before continuing.

### 4. Principles and Core Concepts

- Explain the design principles behind this visualization type.
- Identify the chart's visual encodings (position, color, size, shape, texture).
- Explain when this chart is appropriate and when it is not.
- Show the important code (in Matplotlib, Seaborn, or Plotly, whichever is most natural).
- Explain important parameters and customization options.

### 5. Examples

Give examples progressively:

**Beginner → Intermediate → Advanced**

For every important example:

- Show the code.
- Explain it line by line.
- Describe the expected visual output.
- Explain what the visualization communicates.
- Explain what design decisions were made and why.

Use realistic datasets (Titanic, sales data, model metrics, etc.) so I can understand what good data visualization looks like in practice.

### 6. Visual Perception Principles

When relevant, explain the human perception principle that makes this visualization effective (or ineffective).

For example:

- Pre-attentive attributes (color, size, orientation) and how they guide the eye
- Gestalt principles (proximity, similarity, continuity, closure)
- Why bar charts are more accurate than pie charts for comparison (position vs. angle encoding)
- Why rainbow color maps are misleading
- Why 3D charts are often worse than 2D equivalents
- The data-ink ratio principle (Tufte)
- Why ordering matters (alphabetical vs. sorted by value)

Do not make this unnecessarily academic, but give enough detail for me to make principled design decisions.

### 7. AI/ML Connection

For every important concept, explain:

- Where this visualization is used in AI/ML workflows.
- Why it is useful for ML practitioners.
- Give a realistic AI/ML example.

Use examples involving things such as:

- Loss curves and accuracy curves
- Confusion matrix heatmaps
- ROC curves and AUC
- Feature importance bar charts
- Residual plots
- t-SNE / UMAP 2D scatter plots
- Calibration curves
- Learning curves

Only use an example when it is genuinely relevant.

### 8. Tool Comparison

For each visualization type, compare how to implement it across tools:

| Tool       | Code complexity | Interactivity | Customization | Best for |
| ---------- | --------------- | ------------- | ------------- | -------- |
| Matplotlib |                 |               |               |          |
| Seaborn    |                 |               |               |          |
| Plotly     |                 |               |               |          |

Explain when to choose which tool.

### 9. Common Mistakes

Show common mistakes beginners/intermediate learners make in data visualization.

For each mistake:

**Wrong approach → Why it is wrong → Correct approach**

Also mention important pitfalls such as:

- Misleading y-axis truncation
- Using pie charts for more than 5 categories
- Overplotting in scatter plots
- Using color to encode too many categories
- Using 3D when 2D is sufficient
- Not labeling axes or providing a title
- Cherry-picking time ranges

### 10. Comparison

When multiple chart types can answer the same question, compare them.

For example:

| Chart Type | Best for | Limitations | Example use case |
| ---------- | -------- | ----------- | ---------------- |

### 11. Best Practices

Give practical rules I should follow when creating data visualizations for real AI/ML projects and reports.

Include:

- Color palette selection (colorblind-friendly palettes)
- Font and label sizing
- Aspect ratio guidelines
- When to use interactivity vs. static charts
- How to choose the right chart type for the question

### 12. Interview Perspective

Give me important interview questions related to the section.

Include:

- Conceptual questions (e.g., "When would you use a violin plot instead of a box plot?")
- Design critique questions (e.g., "What is wrong with this chart?")
- Coding questions
- AI/ML visualization questions

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

If the section contains concepts that can be combined into a practical task, give me a small Data Visualization mini-project using those concepts.

The project should be relevant to AI/ML wherever possible.

Give me:

- Problem statement (what story to tell with the data)
- Dataset to use
- Required visualizations
- Expected output description
- Design constraints (e.g., colorblind-friendly, publication-quality)
- Hints

Do not provide the complete solution until I attempt it.

---

# Important Teaching Rules

1. Do not just give definitions. Teach the reasoning and design thinking behind every visualization choice.

2. Prefer practical, real-world examples over unnecessary theory.

3. Start simple and gradually increase complexity.

4. Always connect visualization choices to what question the chart is answering.

5. When visual perception is involved, explain the principle in plain language before applying it.

6. When comparing chart types, explain the trade-offs in terms of accuracy, clarity, and audience.

7. Do not introduce unrelated visualization topics.

8. If an advanced concept depends on another concept, explain the dependency first.

9. Do not assume that I understand design principles or human perception theory.

10. Use tables and simple text diagrams when they make the concept easier to understand.

11. Distinguish clearly between:

- Exploratory visualization (for yourself during EDA)
- Explanatory visualization (for communicating results to others)

12. Focus on understanding rather than memorization.

13. Relate visualization concepts to AI/ML only where the connection is meaningful.

14. Do not teach the entire Data Visualization roadmap at once.

15. Teach ONLY the section I provide.

16. At the end, give me a concise **Revision Sheet** containing:

- Key concepts
- Chart type summary (when to use, when not to use)
- Important functions and parameters
- Design rules
- Common mistakes
- Tool quick reference

17. After teaching the section, STOP and wait for me. Do not automatically move to another Data Visualization section.

---

# Teaching Goal

My goal is not simply to learn charting syntax.

I want to understand:

**What → Why → Design Principles → Human Perception → Implementation → AI/ML Usage**

so that I can confidently choose the right visualization for any data question, implement it cleanly in Python, and communicate ML results clearly to technical and non-technical audiences.

Now teach me the following Data Visualization section:

[DATA VIZ SECTION]

**How to use it:** paste the prompt once, then replace `[DATA VIZ SECTION]` with something like `6. Time Series Visualization` or a smaller subsection such as `6.1 Line Charts and Trend Analysis`. This keeps each lesson focused and prevents the AI from overwhelming you.
