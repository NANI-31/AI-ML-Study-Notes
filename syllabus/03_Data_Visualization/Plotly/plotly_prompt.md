I have a structured Plotly learning roadmap organized as: Section > Topics > Subtopics.

I will provide you with ONE SECTION and all its topics/subtopics at a time.

**Your task**: Create a comprehensive `.md` file — a complete Plotly interview-preparation module
for that section, specifically tailored for a fresher / entry-level **Data Analyst, Data Scientist, Data Engineer, Business Analyst** technical interview.

Do NOT teach the entire Plotly library outside the topics I provide.
Stay focused on the given section, but include prerequisite Python, Pandas, NumPy, and basic data visualization concepts concepts when necessary.

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
- **Why these Plotly concepts matter**: Their role in industry and why hiring managers evaluate them.
- **Where they are used in real projects**: Production use cases and workflows.
- **What a fresher should be able to do**: Concrete competencies after completing this section.

---

### 2. TOPIC-WISE INTERVIEW CONCEPTS

For EVERY topic provided, include:
- **Concept**: Precise mechanism without sacrificing rigor.
- **Why it matters**: What signal the interviewer seeks.
- **Technical / Algorithmic Intuition**: Design principle or computational mechanism.
- **Syntax / Plotly API**: Exact Python/Plotly functions, classes, standard import paths.
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
Generate **exactly 10** Plotly code snippets.
For each: snippet + expected answer + line-by-line technical explanation.

#### D. Output-Prediction Questions
Short snippets where the candidate predicts output, shapes, types, or behavior.
Provide code + detailed breakdown.

#### E. Debugging Questions (10 Questions)
Generate **exactly 10** debugging questions.
Cover issues including:
- fig.show() not rendering in Jupyter due to missing plotly extension
- Animation not working because animation_frame column has non-sequential values
- Hover text not showing because hover_data column has special characters
- Subplot layout misaligned due to incorrect row/col in make_subplots
- Color scale not applying because column is string instead of numeric
- Legend duplicating traces due to missing showlegend=False on secondary traces
- Map chart not rendering due to missing Mapbox token or wrong location format
- write_image() failing due to missing kaleido package

Format: **Problem** | **Incorrect Code** | **Expected Correction** | **Explanation**

#### F. Scenario-Based Questions
- *Build an interactive scatter plot where users filter by category.*
- *Dataset has 1 million points. How do you handle rendering performance?*
- *Create an animated bar chart showing metric changes over 5 years.*
- *Stakeholder needs an interactive HTML report with multiple linked charts.*
- *Need to visualize geographic sales data on a choropleth map.*

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

### 6. PLOTLY API CONFUSIONS

Clarify frequently conflated pairs:
- plotly.express vs plotly.graph_objects — high-level vs low-level API
- px.scatter() vs go.Scatter() for the same chart type
- fig.show() vs fig.write_html() vs fig.write_image()
- Plotly Express color vs symbol vs size encoding parameters
- update_layout() vs update_traces() for modifying figures
- add_trace() vs make_subplots() for multi-panel layouts
- Dash vs Plotly standalone — interactive app vs single chart
- animation_frame vs animation_group in animated charts
- Scatter vs Scattergl (WebGL) for large datasets
- facet_col vs facet_row vs subplot for multi-panel layouts

Format: **Concept 1** | **Concept 2** | **Key Difference** | **When to Use Which** | **Code Comparison**

---

### 7. INTERVIEW TRAPS (10 TRAPS)

Generate **exactly 10** interview traps:
- *"Does px.bar() automatically stack or group bars? How do you control this?"*
- *"Why does fig.show() not display in a non-Jupyter Python script?"*
- *"Can you mix plotly.express traces with plotly.graph_objects in the same figure?"*
- *"What happens with Scattergl and hover data on a million-point dataset?"*
- *"Why does px.line() produce a broken line when x-axis is not sorted?"*

Format: **Trap Question** | **Naive Answer** | **Trap Unveiled** | **Winning Response**

---

### 8. REAL-WORLD PLOTLY QUESTIONS

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

3–5 representative Plotly code snippets. For each:
- Present the code
- Ask the candidate to explain each critical line, intermediate transformations, design rationale, failure modes
- Provide the **Ideal Interview Answer**

---

### 11. "FIX THIS CODE" QUESTIONS

3–5 realistic, buggy Plotly code samples.
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

### 14. PLOTLY INTERVIEW CHEAT SHEET

High-density tabular summary for last-minute review:
- px functions: px.scatter, px.line, px.bar, px.histogram, px.box, px.choropleth
- Modification: fig.update_layout(), fig.update_traces(), fig.add_trace()
- Multi-panel: from plotly.subplots import make_subplots
- Export: fig.write_html('out.html'); fig.write_image('out.png', width=1200, height=800)
- Anti-Patterns: Scattergl without checking hover support | no kaleido for image export

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
`Python`, `Plotly`, `Pandas`, `NumPy`, `Dash`, `Scikit-Learn`

Test technical decision-making, validation strategy, and trade-off articulation.

---

### 18. CROSS-CONCEPT QUESTIONS

Connect the section to adjacent fundamentals:
- **Pandas**: Plotly Express natively accepts Pandas DataFrames with column-name params
- **Matplotlib/Seaborn**: Plotly replaces static visualization when interactivity needed
- **Dash**: Plotly figures are the core component in Dash web applications
- **EDA**: Plotly enables interactive EDA with zoom, pan, hover, and filter
- **Deployment**: Plotly HTML exports and Dash apps enable shareable interactive reports

---

### 19. WHAT A FRESHER SHOULD MEMORIZE

- Core px functions: px.scatter, px.line, px.bar, px.histogram, px.box, px.choropleth
- Encoding params: x, y, color, size, symbol, facet_col, animation_frame
- Layout modification: fig.update_layout(title=..., xaxis_title=...)
- Export: fig.write_html('chart.html'); fig.write_image('chart.png') (needs kaleido)
- Imports: import plotly.express as px; import plotly.graph_objects as go

---

### 20. WHAT A FRESHER SHOULD UNDERSTAND

- Why Plotly Express is preferred for quick charts vs graph_objects for full control
- How WebGL (Scattergl) enables rendering millions of points that SVG cannot
- How animation_frame creates time-based animations cycling through unique values
- Why choropleth maps require specific location codes (ISO-3166, FIPS)
- How Dash extends Plotly with callbacks for fully interactive web apps

---

### 21. WHAT A FRESHER SHOULD PRACTICE

- Multi-panel interactive dashboard with scatter, line, bar using make_subplots
- Animated scatter plot showing population trends over decades
- Choropleth map of country-level sales with custom color scale
- Figure with Plotly Express trace and custom graph_objects annotation
- Export complete interactive HTML report with multiple Plotly charts

---

### 22. INTERVIEW READINESS CHECKLIST

- [ ] I can create interactive charts with Plotly Express.
- [ ] I can use color, size, symbol, and facet_col to encode multiple dimensions.
- [ ] I can customize figures with update_layout and update_traces.
- [ ] I can create animated charts using animation_frame.
- [ ] I can export Plotly figures as HTML and PNG files.
- [ ] I can explain when to use Plotly Express vs Graph Objects.

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

1. **No Hallucinated APIs**: Use only authentic **Plotly (5.0+)** and Python 3.10+ APIs.
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
## 6. Plotly API Confusions
## 7. Interview Traps (10 Traps)
## 8. Real-World Plotly Questions
## 9. Coding Tasks
### A. Mini Coding Tasks (10)
### B. Normal Coding Tasks (10 with Reference Solutions)
## 10. Explain This Code
## 11. Fix This Code
## 12. Rapid-Fire Questions
## 13. Flashcards
## 14. Plotly Interview Cheat Sheet
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
